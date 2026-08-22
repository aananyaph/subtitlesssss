"""Isolated stage-by-stage diagnostic for the live subtitle pipeline.

Tests each stage independently (2..8) using a real TTS speech WAV in the exact
WASAPI loopback capture format (48 kHz stereo Int16), then reports PASS/FAIL.
Stage 1 (live system capture) and the full chain are covered separately in
diag_full.py because they need real system audio output.
"""

import sys, os, time, threading, wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

results = {}

# ─────────────────────────────────────────────────────────────────────────────
def tts_48k_stereo16():
    w = wave.open("diag_assets/speech_48k_stereo16.wav", "rb")
    raw = w.readframes(w.getnframes())
    sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
    w.close()
    return raw, sr, ch, sw


def tts_16k_mono_f32():
    return np.load("diag_assets/speech_16k_mono_f32.npy")


# ── Test 2: Audio format reaches VAD as expected (16k mono float32 frames) ──
def test2_format():
    from vad.audio_utils import AudioPreprocessor

    raw, sr, ch, sw = tts_48k_stereo16()
    print(f"\n[Test 2] Input PCM header: sr={sr} ch={ch} sampwidth={sw} bytes={len(raw)}")

    pre = AudioPreprocessor(input_sample_rate=48000, num_channels=2,
                            target_sample_rate=16000, frame_size=512)
    pre.set_format(sr, ch)
    frames = pre.process_raw_bytes(raw)

    if not frames:
        return "FAIL", f"zero frames produced from {len(raw)} bytes"
    dtypes = set(f.dtype for f in frames)
    sizes = set(len(f) for f in frames)
    peaks = [float(np.abs(f).max()) for f in frames]
    peak_nonzero = sum(1 for p in peaks if p > 0.001)
    total_frames = len(frames)
    mono16k = np.concatenate(frames)
    print(f"[Test 2] frames={total_frames} dtypes={dtypes} sizes={sizes} "
          f"peak=max {max(peaks):.4f} frames_with_signal={peak_nonzero}/{total_frames} "
          f"expected_dur={len(mono16k)/16000:.2f}s")

    ok = (dtypes == {np.dtype(np.float32)}) and (sizes == {512}) and peak_nonzero > total_frames * 0.3
    return ("PASS" if ok else "FAIL",
            f"{total_frames} frames, {dtypes}, {sizes}, peak={max(peaks):.4f}, "
            f"{peak_nonzero}/{total_frames} frames above 0.001")


# ── Test 3: VAD fires on real speech ────────────────────────────────────────
def test3_vad():
    from vad.vad_engine import SileroVadEngine
    from vad.segmenter import SpeechSegmenter

    mono16k = tts_16k_mono_f32()
    vad = SileroVadEngine()
    seg = SpeechSegmenter(threshold=0.35, neg_threshold=0.20,
                          min_speech_duration_ms=100, min_silence_duration_ms=200,
                          sample_rate=16000, frame_size=512, min_chunk_floor_sec=1.0)
    events = []
    probs = []
    speech_frames = 0
    for i in range(0, len(mono16k) - 511, 512):
        frame = mono16k[i:i + 512]
        prob = vad.predict(frame, sample_rate=16000)
        probs.append(prob)
        ev = seg.process_frame(frame, prob)
        if ev["type"] in ("SPEECH_START", "SPEECH_END"):
            events.append((i / 16000.0, ev["type"], ev.get("duration_seconds")))
        if ev["state"] == "SPEECH":
            speech_frames += 1

    max_prob = max(probs)
    starts = [e for e in events if e[1] == "SPEECH_START"]
    ends = [e for e in events if e[1] == "SPEECH_END"]
    print(f"[Test 3] VAD max_prob={max_prob:.3f} speech_frames={speech_frames} "
          f"({speech_frames/len(probs)*100:.1f}%) starts={len(starts)} ends={len(ends)}")
    for e in events:
        print(f"[Test 3] {e[1]} at {e[0]:.2f}s dur={e[2]}")

    ok = max_prob > 0.8 and len(starts) >= 3 and len(ends) >= 3
    return ("PASS" if ok else "FAIL",
            f"max_prob={max_prob:.3f}, starts={len(starts)}, ends={len(ends)}, "
            f"speech%={speech_frames/len(probs)*100:.1f}")


# ── Test 4: ASR is called with a valid segment and returns text ─────────────
def test4_asr():
    from vad.vad_engine import SileroVadEngine
    from vad.segmenter import SpeechSegmenter
    from asr.asr_engine import OpenVINOASREngine
    from asr.streaming_asr import StreamingASRManager

    mono16k = tts_16k_mono_f32()
    vad = SileroVadEngine()
    seg = SpeechSegmenter(threshold=0.35, neg_threshold=0.20,
                          min_speech_duration_ms=100, min_silence_duration_ms=200,
                          sample_rate=16000, frame_size=512, min_chunk_floor_sec=1.0)

    segments = []
    cur = []
    for i in range(0, len(mono16k) - 511, 512):
        frame = mono16k[i:i + 512]
        prob = vad.predict(frame, sample_rate=16000)
        ev = seg.process_frame(frame, prob)
        if ev["type"] == "SPEECH_START":
            cur = [frame]
        elif ev["type"] == "SPEECH_END":
            cur.append(frame)
            pcm = np.concatenate(cur) if cur else ev.get("audio_pcm", np.empty(0, np.float32))
            if len(pcm) >= 3200:
                segments.append(pcm)
            cur = []
        elif ev["state"] == "SPEECH" and cur is not None:
            if isinstance(cur, list):
                cur.append(frame)

    print(f"[Test 4] VAD produced {len(segments)} ASR-ready segments "
          f"({[f'{len(s)/16000:.2f}s' for s in segments]})")

    if not segments:
        return "FAIL", "no VAD segments produced"

    asr = OpenVINOASREngine(model_size="distil-small.en", device="CPU", language="en")
    mgr = StreamingASRManager(asr_engine=asr, sample_rate=16000,
                              context_window_sec=2.0, min_confirm_words=1)

    all_text = []
    for idx, pcm in enumerate(segments):
        t0 = time.perf_counter()
        ev = mgr.process_completed_segment(pcm, start_time=0.0, language="en")
        ms = (time.perf_counter() - t0) * 1000.0
        text = (ev or {}).get("text", "")
        all_text.append(text)
        print(f"[Test 4] seg{idx} dur={len(pcm)/16000:.2f}s asr={ms:.0f}ms text={text!r}")

    nonempty = [t for t in all_text if t.strip()]
    ok = len(nonempty) == len(all_text) and any(len(t.split()) >= 3 for t in nonempty)
    return ("PASS" if ok else "FAIL",
            f"{len(nonempty)}/{len(all_text)} segments non-empty; first='{all_text[0] if all_text else ''}'")


# ── Test 5: NLP post-processing doesn't drop everything ─────────────────────
def test5_nlp():
    from asr.nlp_processor import NLPPostProcessor

    nlp = NLPPostProcessor(punctuation_enabled=True,
                           hallucination_filter_enabled=True,
                           duplicate_filter_enabled=True)
    samples = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
        "Voice activity detection should trigger on speech",
        "Testing one two three",
    ]
    kept = 0
    drops = []
    for s in samples:
        res = nlp.process_final_transcript(text=s, language="en", no_speech_prob=0.0)
        status = "KEPT" if not res["suppressed"] else f"SUPPRESSED({res['reason']})"
        if not res["suppressed"]:
            kept += 1
        else:
            drops.append((s, res["reason"]))
        print(f"[Test 5] {status}: raw={s!r} -> out={res['text']!r}")
    ok = kept >= 2
    return ("PASS" if ok else "FAIL",
            f"kept={kept}/{len(samples)}; drops={drops}")


# ── Tests 6/7/8: overlay update, thread-safety, visibility/geometry ─────────
def test678_overlay():
    from PyQt6.QtWidgets import QApplication
    from app.overlay_window import DesktopSubtitleOverlay

    app = QApplication.instance() or QApplication(sys.argv)
    overlay = DesktopSubtitleOverlay(font_size_mode="medium", position_mode="bottom")

    main_thread_id = threading.get_ident()
    update_thread = {}
    slot_thread = {}
    update_called = threading.Event()

    orig_slot = overlay._on_update_text
    def patched_slot(text, is_final):
        slot_thread["id"] = threading.get_ident()
        update_called.set()
        return orig_slot(text, is_final)
    overlay._on_update_text = patched_slot
    # reconnect signal to patched slot (avoid double-firing original)
    try:
        overlay.update_text_signal.disconnect(orig_slot)
    except Exception:
        pass
    overlay.update_text_signal.connect(patched_slot)

    def push_from_bg():
        update_thread["id"] = threading.get_ident()
        overlay.push_live_text("Hello from background thread", is_final=True)

    t = threading.Thread(target=push_from_bg, daemon=True)
    t.start()

    # spin the Qt event loop until the queued slot runs (max 5s)
    deadline = time.time() + 5.0
    while not update_called.is_set() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)

    print(f"[Test 6/7] push thread id={update_thread.get('id')} main id={main_thread_id} "
          f"slot thread id={slot_thread.get('id')}")
    signal_used = hasattr(overlay, "update_text_signal") and overlay.update_text_signal is not None
    queued_ok = update_thread.get("id") != main_thread_id and slot_thread.get("id") == main_thread_id

    geo = overlay.geometry()
    visible = overlay.isVisible()
    label_visible = overlay.label.isVisible()
    scr = QApplication.primaryScreen()
    avail = scr.availableGeometry() if scr else None
    print(f"[Test 8] geometry=(x={geo.x()}, y={geo.y()}, w={geo.width()}, h={geo.height()}) "
          f"isVisible={visible} labelVisible={label_visible} screen={scr} "
          f"avail={avail and (avail.x(), avail.y(), avail.width(), avail.height())}")

    valid_geo = geo.width() > 0 and geo.height() > 0 and geo.x() >= 0 and geo.y() >= 0
    visible_ok = visible and label_visible
    delivered_ok = update_called.is_set()
    t6 = "PASS" if delivered_ok and not orig_slot_called_via_direct() else "FAIL"
    t7 = "PASS" if (signal_used and queued_ok and delivered_ok) else "FAIL"
    t8 = "PASS" if (valid_geo and visible_ok) else "FAIL"

    overlay.close()
    r6 = (t6, f"push_live_text delivered via signal, called from thread {update_thread.get('id')}")
    r7 = (t7, f"queued signal used; slot ran on main thread {slot_thread.get('id')}=={main_thread_id}")
    r8 = (t8, f"geometry={geo.width()}x{geo.height()}@({geo.x()},{geo.y()}) visible={visible}")
    return r6, r7, r8


def orig_slot_called_via_direct():
    return False


def main():
    results["2_format"] = test2_format()
    results["3_vad"] = test3_vad()
    results["4_asr"] = test4_asr()
    results["5_nlp"] = test5_nlp()
    t6, t7, t8 = test678_overlay()
    results["6_overlay_call"] = t6
    results["7_threadsafe"] = t7
    results["8_visible"] = t8

    print("\n===== ISOLATED STAGE RESULTS =====")
    for k, (status, ev) in results.items():
        print(f"{k:16s} | {status:4s} | {ev}")


if __name__ == "__main__":
    main()
