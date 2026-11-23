"""WebSocket ASR bridge for remote subtitle clients.

client <-> [translation_proxy :8765] <-> [this bridge :8766]

Each WebSocket client gets its OWN pipeline instance.  Subtitles are sent only
to the client that supplied the audio, and the server never opens a local
WASAPI loopback device.  This makes the bridge safe for a shared team server:
one client's language choice or disconnect cannot affect another client.

Messages:
  in:  HANDSHAKE / START_CAPTURE / STOP_CAPTURE / UPDATE_SETTINGS / CLEAR_HISTORY
  out: connected / status / flush_ack / final / partial subtitles

Usage:
  python backend/app_bridge.py [--host 0.0.0.0] [--port 8766] [--model lazy]
"""
import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import time
from typing import Any, Dict, Optional, Tuple

import websockets
from websockets.exceptions import ConnectionClosed

try:
    from websockets.asyncio.server import serve
except ImportError:  # websockets < 12
    from websockets.server import serve

# Add project root and backend directory to the import path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BACKEND = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.settings import load_settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8766))

# Fallback model used when a client handshake omits model_name.  "lazy" means
# do not preload anything; the first HANDSHAKE decides which model to load.
DEFAULT_MODEL: str = "lazy"


def config_key(cfg: Dict[str, Any]) -> Tuple[Optional[str], ...]:
    """Identity of an ASR configuration; used to avoid rebuilding the pipeline."""
    return (
        str(cfg.get("model_name") or DEFAULT_MODEL or "base"),
        cfg.get("indic_lang") or None,
        cfg.get("asr_language") or None,
        cfg.get("asr_task") or None,
        cfg.get("tgt_lang") or None,
    )


class BridgeOverlay:
    """Overlay that forwards captions to exactly one client session."""

    def __init__(self, session: "ClientSession"):
        self.session = session

    def push_live_text(self, text: str, is_final: bool = True, source_text: Optional[str] = None):
        payload: Dict[str, Any] = {
            "type": "final" if is_final else "partial",
            "text": text,
            "language": getattr(self.session.pipeline, "last_detected_language", None)
            or self.session.asr_language
            or "en",
            "is_final": is_final,
        }
        if source_text:
            payload["source_text"] = source_text
        self.session.send_threadsafe(payload)


class ClientSession:
    """All state belonging to a single connected client."""

    def __init__(self, ws, loop: asyncio.AbstractEventLoop, default_model: str = "lazy"):
        self.ws = ws
        self.loop = loop
        self.default_model = default_model
        self.pipeline = None  # type: ignore[assignment]
        self.config: Optional[Tuple[Optional[str], ...]] = None
        self.sample_rate = 48000
        self.num_channels = 2
        self.asr_language: Optional[str] = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def send(self, obj: Dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            await self.ws.send(json.dumps(obj, ensure_ascii=False))
        except Exception as exc:
            logging.debug(f"[Bridge] send failed: {exc}")

    def send_threadsafe(self, obj: Dict[str, Any]) -> None:
        """Called from the pipeline worker thread; never blocks that thread."""
        if self._closed or self.loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.send(obj), self.loop)
        except RuntimeError:
            pass

    async def apply_config(self, cfg: Dict[str, Any]) -> None:
        """(Re)build this client's pipeline to match its handshake config."""
        key = config_key(cfg)
        async with self._lock:
            if self.pipeline is not None and self.config == key:
                logging.info(f"[Bridge] ASR config unchanged {key}; keeping pipeline")
                return
            self.asr_language = cfg.get("asr_language") or cfg.get("indic_lang")
            logging.info(f"[Bridge] Switching ASR config -> {key}")
            self.pipeline = await self.loop.run_in_executor(
                None, apply_asr_config, cfg, self
            )
            self.config = key
            self._set_format()
            logging.info("[Bridge] ASR config applied for client")

    def _set_format(self) -> None:
        if self.pipeline and hasattr(self.pipeline, "preprocessor"):
            try:
                self.pipeline.preprocessor.set_format(self.sample_rate, self.num_channels)
            except Exception as exc:
                logging.warning(f"[Bridge] preprocessor.set_format failed: {exc}")

    def feed_audio(self, raw: bytes) -> None:
        """Queue PCM audio received from this client; drop the oldest on overflow."""
        pipeline = self.pipeline
        if pipeline is None:
            return
        self._set_format()
        try:
            pipeline.audio_queue.put_nowait((raw, False, time.monotonic()))
        except queue.Full:
            try:
                pipeline.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                pipeline.audio_queue.put_nowait((raw, False, time.monotonic()))
            except queue.Full:
                pass

    async def close(self) -> None:
        self._closed = True
        pipeline = self.pipeline
        self.pipeline = None
        if pipeline is not None:
            try:
                await self.loop.run_in_executor(None, pipeline.stop)
            except Exception as exc:
                logging.warning(f"[Bridge] pipeline stop failed: {exc}")


def apply_asr_config(cfg: Dict[str, Any], session: ClientSession):
    """Build and start a pipeline for one session.  Runs in a worker thread.

    Kept at module level (rather than only as a method) so the Colab notebook
    can verify that dynamic ASR configuration is built into the bridge.
    """
    import main_app

    active_indic = cfg.get("indic_lang") or None
    active_model = str(cfg.get("model_name") or session.default_model or "base")
    if active_model.lower() in ("lazy", "none"):
        active_model = "base"

    if active_indic:
        try:
            from indic_backend import load_indic_model

            load_indic_model()
        except Exception as exc:
            logging.warning(
                f"[Bridge] IndicConformer unavailable ({exc}); falling back to Whisper"
            )
            active_indic = None
            if active_model in ("indic", "lazy", "auto"):
                active_model = "small"

    pipeline = main_app.DesktopLiveSubtitlePipeline(
        overlay=BridgeOverlay(session),
        model_name=active_model,
        indic_lang=active_indic,
        asr_language=cfg.get("asr_language") or active_indic,
        asr_task=cfg.get("asr_task") or "transcribe",
        tgt_lang=cfg.get("tgt_lang"),
        capture_locally=False,
    )
    try:
        pipeline.start()
    except Exception as exc:
        # Model loading can fail; keep the session alive so an error can be
        # reported to this client rather than taking down the whole server.
        logging.error(f"[Bridge] pipeline start failed: {exc}", exc_info=True)
    return pipeline


async def handle_client(ws, *args, **kwargs) -> None:
    loop = asyncio.get_running_loop()
    session = ClientSession(ws, loop, default_model=DEFAULT_MODEL)
    remote = getattr(ws, "remote_address", "?")
    logging.info(f"[Bridge] Client connected {remote}")

    try:
        await session.send(
            {
                "type": "connected",
                "message": f"Bridge ready (model={DEFAULT_MODEL})",
            }
        )
        async for raw in ws:
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue

                msg_type = data.get("type")
                if msg_type == "HANDSHAKE":
                    logging.info(f"[Bridge] HANDSHAKE from {remote}: {data}")
                    session.sample_rate = int(data.get("sampleRate") or 48000)
                    session.num_channels = int(data.get("numChannels") or 2)
                    cfg = data.get("asrConfig")
                    if isinstance(cfg, dict):
                        try:
                            await session.apply_config(cfg)
                        except Exception as exc:
                            logging.error(f"[Bridge] apply_asr_config failed: {exc}", exc_info=True)
                            await session.send(
                                {"type": "final", "text": f"Server model load failed: {exc}", "is_final": True}
                            )
                elif msg_type in ("START_CAPTURE", "STOP_CAPTURE"):
                    active = msg_type == "START_CAPTURE"
                    if session.pipeline:
                        try:
                            await loop.run_in_executor(None, session.pipeline.set_capturing, active)
                        except Exception as exc:
                            logging.warning(f"[Bridge] set_capturing failed: {exc}")
                    await session.send({"type": "status", "capturing": active})
                elif msg_type == "UPDATE_SETTINGS":
                    model = (data.get("settings") or {}).get("model")
                    if model and session.pipeline:
                        try:
                            await loop.run_in_executor(None, session.pipeline.set_model, model)
                            await session.send(
                                {
                                    "type": "status",
                                    "model": model,
                                    "task": getattr(session.pipeline, "asr_task", None),
                                }
                            )
                        except Exception as exc:
                            logging.warning(f"[Bridge] set_model failed: {exc}")
                elif msg_type == "CLEAR_HISTORY":
                    await session.send({"type": "flush_ack"})
                # Unknown control messages are intentionally ignored.
            elif isinstance(raw, bytes):
                session.feed_audio(raw)
    except ConnectionClosed:
        pass
    except Exception as exc:
        logging.error(f"[Bridge] client handler error: {exc}", exc_info=True)
    finally:
        await session.close()
        logging.info(f"[Bridge] Client disconnected {remote}")


async def main_async(args) -> None:
    global DEFAULT_MODEL
    DEFAULT_MODEL = args.model or "lazy"
    logging.info(
        f"[Bridge] Serving on ws://{args.host}:{args.port} "
        "(per-client pipelines, no local audio capture)"
    )
    async with serve(handle_client, args.host, args.port, max_size=None):
        await asyncio.Future()


def create_pipeline(model_name: str, overlay, capture_locally: bool = False):
    """Convenience wrapper used by scripts/tests that build a bridge pipeline."""
    import main_app

    return main_app.DesktopLiveSubtitlePipeline(
        overlay=overlay, model_name=model_name, capture_locally=capture_locally
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared ASR WebSocket bridge")
    parser.add_argument("--host", default=HOST, help=f"Host to bind (default {HOST})")
    parser.add_argument(
        "--model",
        default=None,
        help="Fallback ASR model when a handshake omits it (default: lazy/base)",
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"WebSocket port (default {PORT})"
    )
    parser.add_argument("--list-devices", action="store_true", help="List WASAPI devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        from app.loopback_capture import list_loopback_devices

        list_loopback_devices()
        return

    if not args.model:
        args.model = str(load_settings().get("model", "lazy"))
    args.port = int(args.port)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logging.info("[Bridge] Shutting down...")


if __name__ == "__main__":
    main()
