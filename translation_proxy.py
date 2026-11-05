"""translation_proxy.py - separate translation layer.

client <-> [this proxy :8765] <-> [ASR bridge :8766]

* The ASR bridge only TRANSCRIBES (source language). It never translates.
* This proxy takes each final subtitle the bridge produces, translates it with
  IndicTrans2 (PyTorch 1B fp16 on GPU, with automatic ONNX int8 fallback) in
  its own worker, and sends the translated subtitle to the client.
* Everything else (audio, status, meters, settings) is relayed untouched.
"""
import asyncio
from collections import OrderedDict
import gc
import json
import logging
import os
import sys
import threading
import time

try:
    from websockets.asyncio.server import serve
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed
except ImportError:  # lets the logic be imported/tested without websockets
    serve = connect = None

    class ConnectionClosed(Exception):
        pass

LISTEN_PORT = int(os.environ.get("PROXY_PORT", "8765"))
UPSTREAM = os.environ.get("UPSTREAM_URL", "ws://127.0.0.1:8766")
FORCE_TGT = os.environ.get("FORCE_TGT") or None        # e.g. "kn" to test English -> Kannada
MERGE_WAIT = float(os.environ.get("MERGE_WAIT", "0"))  # >0: join finals arriving within N s
MAX_MERGE_CHARS = int(os.environ.get("MAX_MERGE_CHARS", "160"))
SUPPORTED = ("en", "hi", "kn")
INDIC_ASR = ("hi", "kn")
TARGET_KEYS = ("tgt_lang", "target_lang", "subtitle_lang", "subtitle_language",
               "subtitleLanguage", "targetLanguage")
SUB_TYPES = ("final", "partial", "interim", "subtitle", "caption")
SENT_END = (".", "?", "!", "\u0964", "\u2026")

# Backend & Cache Configuration
TRANSLATOR_BACKEND = os.environ.get("TRANSLATOR_BACKEND", "torch1b").lower()
SHOW_SOURCE_PARTIALS = os.environ.get("SHOW_SOURCE_PARTIALS", "0").lower() in ("1", "true", "yes")
MAX_RESIDENT_MODELS = int(os.environ.get("MAX_RESIDENT_MODELS", "2"))

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tproxy")

# ---------------------------------------------------------------- translator
_tr_lock = threading.Lock()
_model_cache = OrderedDict()  # (backend_tag, repo_id) -> model instance (max 2 resident)
_text_cache = OrderedDict()   # (src, tgt, text) -> translated_text (max 512 entries)
_pivot_logged = False


def _repo_for_onnx(src, tgt):
    if tgt == "en":
        return "hari31416/indictrans2-indic-en-dist-200M-ONNX-int8"
    if src == "en":
        return "hari31416/indictrans2-en-indic-dist-200M-ONNX-int8"
    return "hari31416/indictrans2-indic-indic-dist-320M-ONNX-int8"


def _repo_for_gpu(src, tgt):
    if tgt == "en":
        return "ai4bharat/indictrans2-indic-en-1B"
    if src == "en":
        return "ai4bharat/indictrans2-en-indic-1B"
    return "ai4bharat/indictrans2-indic-indic-1B"


def _evict_oldest_model_if_needed():
    """Caller must hold _tr_lock."""
    while len(_model_cache) > MAX_RESIDENT_MODELS:
        old_key, old_model = _model_cache.popitem(last=False)
        del old_model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log.info(f"Evicted model {old_key} from resident cache (max={MAX_RESIDENT_MODELS})")


def _get_model(src, tgt, target_backend=None):
    """Caller must hold _tr_lock."""
    global _pivot_logged
    chosen_backend = target_backend or TRANSLATOR_BACKEND

    # If GPU requested, verify CUDA availability first
    if chosen_backend == "torch1b":
        try:
            import torch
            if not torch.cuda.is_available():
                log.warning("CUDA unavailable; falling back to ONNX backend")
                chosen_backend = "onnx"
        except Exception as e:
            log.warning(f"PyTorch check failed ({e}); falling back to ONNX backend")
            chosen_backend = "onnx"

    if chosen_backend == "torch1b":
        repo = _repo_for_gpu(src, tgt)
        cache_key = ("torch1b", repo)
        if cache_key in _model_cache:
            _model_cache.move_to_end(cache_key)
            return _model_cache[cache_key], "torch1b"
        try:
            from translator_gpu import IndicTrans2GPU
            model = IndicTrans2GPU(repo)
            _model_cache[cache_key] = model
            _evict_oldest_model_if_needed()
            return model, "torch1b"
        except Exception as exc:
            log.warning(f"GPU model load failed for {repo}: {exc}. Falling back to ONNX.")
            chosen_backend = "onnx"

    # ONNX Fallback / Default
    repo = _repo_for_onnx(src, tgt)
    cache_key = ("onnx", repo)
    if cache_key in _model_cache:
        _model_cache.move_to_end(cache_key)
        return _model_cache[cache_key], "onnx"

    from translator_live import IndicTransONNXModel
    model = IndicTransONNXModel(repo, intra_threads=2)
    _model_cache[cache_key] = model
    _evict_oldest_model_if_needed()
    return model, "onnx"


def _check_text_cache(src, tgt, text):
    with _tr_lock:
        key = (src, tgt, text)
        if key in _text_cache:
            _text_cache.move_to_end(key)
            return _text_cache[key]
    return None


def _put_text_cache(src, tgt, text, translated):
    with _tr_lock:
        key = (src, tgt, text)
        _text_cache[key] = translated
        if len(_text_cache) > 512:
            _text_cache.popitem(last=False)


def translate_with_meta(text, src, tgt):
    """Translate text and return (translated_text, backend_name, cache_hit). Never raises."""
    global _pivot_logged
    if not text or not text.strip():
        return "", "none", False

    cached = _check_text_cache(src, tgt, text)
    if cached is not None:
        return cached, "cache", True

    with _tr_lock:
        backend_used = TRANSLATOR_BACKEND
        res = None
        try:
            if TRANSLATOR_BACKEND == "torch1b":
                # Handle potential indic-to-indic pivoting if indic-indic-1B fails
                if src in ("hi", "kn") and tgt in ("hi", "kn"):
                    try:
                        model, backend_used = _get_model(src, tgt, target_backend="torch1b")
                        res = model.translate(text, src_iso=src, tgt_iso=tgt)
                    except Exception as e_ind:
                        if not _pivot_logged:
                            log.warning(f"Direct indic-indic-1B failed ({e_ind}); pivoting hi<->en<->kn via 1B models.")
                            _pivot_logged = True
                        m_to_en, _ = _get_model(src, "en", target_backend="torch1b")
                        mid_text = m_to_en.translate(text, src_iso=src, tgt_iso="en")
                        m_from_en, _ = _get_model("en", tgt, target_backend="torch1b")
                        res = m_from_en.translate(mid_text, src_iso="en", tgt_iso=tgt)
                        backend_used = "torch1b_pivot"
                else:
                    model, backend_used = _get_model(src, tgt, target_backend="torch1b")
                    res = model.translate(text, src_iso=src, tgt_iso=tgt)
            else:
                model, backend_used = _get_model(src, tgt, target_backend="onnx")
                res = model.translate(text, src_iso=src, tgt_iso=tgt)
        except Exception as exc:
            log.warning(f"Translator ({backend_used}) error for sentence '{text[:40]}...': {exc}. Falling back to ONNX.")
            try:
                model, _ = _get_model(src, tgt, target_backend="onnx")
                res = model.translate(text, src_iso=src, tgt_iso=tgt)
                backend_used = "onnx_fallback"
            except Exception as onnx_exc:
                log.error(f"ONNX fallback also failed: {onnx_exc}", exc_info=True)
                return text, "failed", False

    out = (res or "").strip() or text
    _put_text_cache(src, tgt, text, out)
    return out, backend_used, False


def translate(text, src, tgt):
    """Blocking translate function. Kept for benchmarks and backwards-compatible callers."""
    out, _, _ = translate_with_meta(text, src, tgt)
    return out


def preload(src, tgt):
    def _go():
        try:
            with _tr_lock:
                _get_model(src, tgt)
            log.info(f"translator ready {src}->{tgt}")
        except Exception as e:
            log.error(f"translator preload failed {src}->{tgt}: {e}", exc_info=True)
    threading.Thread(target=_go, daemon=True).start()

# ------------------------------------------------------------------- helpers
def norm_lang(v):
    v = str(v or "").strip().lower()
    for pref, code in (("kn", "kn"), ("kan", "kn"), ("hi", "hi"), ("en", "en")):
        if v.startswith(pref):
            return code
    return None


def rewrite_handshake(raw, state):
    """If client asks for a translated subtitle, make the ASR bridge transcribe only."""
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(data, dict) or data.get("type") != "HANDSHAKE":
        return raw
    log.info(f"client HANDSHAKE {raw[:300]}")
    cfg = data.get("asrConfig")
    state["src"] = state["tgt"] = None
    if not isinstance(cfg, dict):
        return raw
    cfg = dict(cfg)

    raw_asr_lang = cfg.get("asr_language")
    if not raw_asr_lang or str(raw_asr_lang).lower() in ("auto", "none"):
        log.info("translation layer OFF: asr_language is unset or 'auto' (explicit source language required for translation router)")
        return raw

    src = norm_lang(raw_asr_lang)
    tgt = None
    for k in TARGET_KEYS:
        v = cfg.get(k) or data.get(k)
        if v:
            tgt = norm_lang(v)
            break
    if not tgt and cfg.get("asr_task") == "translate":
        tgt = "en"
    if FORCE_TGT:
        tgt = norm_lang(FORCE_TGT)
    if tgt and src and src != tgt and src in SUPPORTED and tgt in SUPPORTED:
        cfg["asr_task"] = "transcribe"
        if src in INDIC_ASR:
            cfg.update(model_name="indic", indic_lang=src, asr_language=src)
        state["src"], state["tgt"] = src, tgt
        log.info(f"translation layer ON {src}->{tgt}; ASR forced to transcribe: {cfg}")
        preload(src, tgt)
        data["asrConfig"] = cfg
        return json.dumps(data, ensure_ascii=False)
    log.info(f"translation layer OFF (src={src} tgt={tgt}); passing config through")
    return raw


def parse_subtitle(raw):
    """Return the dict if this text frame is a subtitle message, else None."""
    try:
        d = json.loads(raw)
    except Exception:
        return None
    if isinstance(d, dict) and isinstance(d.get("text"), str) and (
            d.get("type") in SUB_TYPES or "is_final" in d):
        return d
    return None


def is_final(d):
    if "is_final" in d:
        return d["is_final"] is True
    return d.get("type") == "final"

# ------------------------------------------------------------------ relaying
_seen_shapes = set()


async def client_to_up(client, up, state):
    async for raw in client:
        if isinstance(raw, str):
            raw = rewrite_handshake(raw, state)
        await up.send(raw)


async def up_to_client(client, up, state, q):
    async for raw in up:
        if isinstance(raw, str):
            d = parse_subtitle(raw)
            try:
                shape = json.loads(raw)
                key = tuple(sorted(shape.keys())) if isinstance(shape, dict) else type(shape).__name__
            except Exception:
                key = "non-json"
            if key not in _seen_shapes:
                _seen_shapes.add(key)
                log.info(f"new upstream message shape {key}: {raw[:200]}")
            if d is not None and state["tgt"]:
                if is_final(d):
                    await q.put(d)       # translated in order by the worker
                    continue
                elif SHOW_SOURCE_PARTIALS:
                    # Forward non-final subtitle unchanged immediately (language = src)
                    partial_msg = dict(d, language=state["src"])
                    await client.send(json.dumps(partial_msg, ensure_ascii=False))
                    continue
                else:
                    continue             # non-final source-language text is dropped
        await client.send(raw)


async def translate_worker(client, state, q):
    loop = asyncio.get_running_loop()
    while True:
        batch = [await q.get()]
        if MERGE_WAIT > 0:
            while sum(len(b["text"]) for b in batch) < MAX_MERGE_CHARS and \
                    not batch[-1]["text"].rstrip().endswith(SENT_END):
                try:
                    batch.append(await asyncio.wait_for(q.get(), MERGE_WAIT))
                except asyncio.TimeoutError:
                    break
        src_text = " ".join(b["text"].strip() for b in batch if b["text"].strip())
        out = src_text
        backend_used = "none"
        cache_hit = False
        if src_text and not src_text.startswith("Server model load failed"):
            t0 = time.perf_counter()
            try:
                out, backend_used, cache_hit = await loop.run_in_executor(
                    None, translate_with_meta, src_text, state["src"], state["tgt"]
                )
            except Exception as e:
                log.error(f"translate failed, showing source text: {e}", exc_info=True)
                out = src_text
                backend_used = "error"
                cache_hit = False

            try:
                import torch
                vram_mb = round(torch.cuda.memory_allocated() / (1024.0 * 1024.0), 1) if torch.cuda.is_available() else 0.0
            except Exception:
                vram_mb = 0.0

            log.info("SUB " + json.dumps({
                "dir": f"{state['src']}->{state['tgt']}",
                "src": src_text,
                "out": out,
                "sec": round(time.perf_counter() - t0, 2),
                "merged": len(batch),
                "backlog": q.qsize(),
                "backend": backend_used,
                "vram_mb": vram_mb,
                "cache_hit": cache_hit,
            }, ensure_ascii=False))

        msg = dict(batch[-1], text=out, source_text=src_text, language=state["tgt"])
        await client.send(json.dumps(msg, ensure_ascii=False))


async def handler(client):
    state = {"src": None, "tgt": None}
    q = asyncio.Queue()
    log.info("client connected")
    tasks = []
    try:
        async with connect(UPSTREAM, max_size=None, compression=None) as up:
            tasks = [asyncio.create_task(client_to_up(client, up, state)),
                     asyncio.create_task(up_to_client(client, up, state, q)),
                     asyncio.create_task(translate_worker(client, state, q))]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except ConnectionClosed:
        pass
    except Exception as e:
        log.error(f"handler error: {e}", exc_info=True)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("client disconnected")


async def main():
    async with serve(handler, "0.0.0.0", LISTEN_PORT, max_size=None):
        log.info(f"translation proxy listening :{LISTEN_PORT} -> {UPSTREAM} "
                 f"(FORCE_TGT={FORCE_TGT}, MERGE_WAIT={MERGE_WAIT}, "
                 f"BACKEND={TRANSLATOR_BACKEND}, MAX_RESIDENT={MAX_RESIDENT_MODELS})")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

