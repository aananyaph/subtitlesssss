"""Single-Process Desktop Live Subtitle Application (Module 12).

Orchestrates:
    WASAPI Loopback Capture Thread (pyaudiowpatch)
        │
        ▼ (raw Int16 PCM 48kHz stereo)
    AudioPreprocessor (16kHz mono Float32, 512-sample VAD frames)
        │
        ▼
    Silero VAD Engine & SpeechSegmenter (Natural Room Tone Floor)
        │
        ▼ (Speech segment PCM)
    OpenVINO ASR Engine (distil-whisper small.en int8)
        │
        ▼
    NLP Post-Processing Layer (Punctuation, Safety & Duplicate Filter)
        │
        ▼ (Direct Python function call / Qt Signal)
    Desktop Subtitle Overlay Window (Frameless, Transparent, Click-Through)
    Optional Local LLM Post-Correction (feature-gated, on-demand)
"""

import argparse
import logging
import os
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from vad.audio_utils import AudioPreprocessor
from vad.vad_engine import SileroVadEngine
from vad.segmenter import SpeechSegmenter
from asr.asr_engine import BaseASREngine, OpenVINOASREngine, FasterWhisperEngine
from asr.streaming_asr import StreamingASRManager
from asr.nlp_processor import NLPPostProcessor
from app.loopback_capture import WASAPILoopbackCapture, HAS_PYAUDIO
from app.settings import load_settings, save_settings
from app.settings_window import SettingsWindow
from app.tray_icon import SystemTrayController
from app.overlay_window import DesktopSubtitleOverlay
from app.remote_client import RemoteLiveSubtitlePipeline

# ── Qt availability check ─────────────────────────────────────────────────────
try:
    from PyQt6.QtWidgets import QApplication
    HAS_QT = True
except ImportError:
    try:
        from PySide6.QtWidgets import QApplication
        HAS_QT = True
    except ImportError:
        HAS_QT = False


# ── Feature flag ───────────────────────────────────────────────────────────────
ASYNC_LLM_CORRECTION_ENABLED = os.environ.get("ASYNC_LLM_CORRECTION_ENABLED", "").lower() in {"1", "true", "yes"}

# ── Global timeout for staleness safety net (overridden in main() from settings) ──
STALE_AUDIO_TIMEOUT_SEC = 15.0
try:
    _s = load_settings()
    STALE_AUDIO_TIMEOUT_SEC = float(_s.get("stale_audio_timeout_sec", 15.0))
except Exception:
    pass

# ── LLMCorrector — ported from backend/asr/llm_corrector.py (Module 9) ──────
# Provides correct(text, language) and validate_output(original, generated) with
# 5 rejection rules: empty, conversational prefixes, markdown, excessive length,
# identical text. Feature-flag gated; if model path is missing or llama-cpp-python
# is unavailable, correct() silently returns None and the initial caption stands.

class LLMCorrector:
    """Local LLM Post-Correction Engine using llama-cpp-python (if available)."""

    def __init__(
        self,
        model_path: str = "",
        enabled: bool = False,
        n_ctx: int = 512,
        n_threads: int = 2,
    ):
        self.enabled = enabled
        self.model_path = model_path or os.environ.get("LLM_MODEL_PATH", "")
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llm = None
        self.is_available = False

        if self.enabled:
            # Lazy load — attempt GGUF model import; safe fallback if missing
            try:
                from llama_cpp import Llama  # type: ignore
                if os.path.exists(self.model_path):
                    logging.info(
                        f"[LLM] Loading GGUF model from '{self.model_path}' "
                        f"(threads={self.n_threads}, ctx={self.n_ctx})..."
                    )
                    self.llm = Llama(
                        model_path=self.model_path,
                        n_ctx=self.n_ctx,
                        n_threads=self.n_threads,
                        verbose=False,
                    )
                    self.is_available = True
                    logging.info("[LLM] GGUF model loaded successfully.")
                else:
                    logging.warning(
                        f"[LLM] Disabled: Model file not found at '{self.model_path}'."
                    )
            except ImportError:
                logging.warning("[LLM] Disabled: 'llama-cpp-python' package is not installed.")
            except Exception as exc:
                logging.error(f"[LLM] Disabled: Failed to load model: {exc}", exc_info=True)
        else:
            logging.info("[LLM] Async LLM correction disabled by feature flag.")

    def correct(self, text: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Runs LLM post-correction on original text.

        Returns:
            Dict containing {"corrected_text": str, "inference_ms": float}
            or None if correction was skipped, failed, or rejected by validation.
        """
        if not self.is_available or not self.llm or not text or not text.strip():
            return None

        clean_text = text.strip()

        # Construct short, deterministic prompt
        prompt = (
            "You are correcting an automatic speech recognition subtitle.\n\n"
            "Fix only:\n"
            "- obvious spelling errors\n"
            "- obvious ASR word errors\n"
            "- grammar errors\n"
            "- punctuation\n\n"
            "Preserve the exact meaning.\n"
            "Do not add information.\n"
            "Do not summarize.\n"
            "Do not explain.\n"
            "Return ONLY the corrected subtitle.\n\n"
            f"Subtitle: {clean_text}\n"
            "Corrected:"
        )

        t0 = time.perf_counter()
        try:
            res = self.llm(
                prompt,
                max_tokens=64,
                stop=["\n", "Corrected:", "Explanation:", "Note:"],
                temperature=0.1,
                top_p=0.9,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            raw_out = res["choices"][0]["text"].strip()
            validated_text = self.validate_output(clean_text, raw_out)

            if validated_text is None:
                logging.debug(f"[LLM Validator] Rejected output for '{clean_text}': raw='{raw_out}'")
                return None

            return {
                "corrected_text": validated_text,
                "inference_ms": elapsed_ms,
            }

        except Exception as exc:
            logging.warning(f"[LLM Worker] Inference error for '{clean_text}': {exc}", exc_info=True)
            return None

    @staticmethod
    def validate_output(original: str, generated: str) -> Optional[str]:
        """Validates LLM output string to prevent invalid/hallucinated overlays.

        Rejection rules:
            1. Empty output.
            2. Contains explanations/conversational intros ("Here is", "Sure", "Note:").
            3. Contains markdown code blocks or backticks.
            4. Excessive length (> 2x original length + 20 chars).
            5. If generated is identical to original, no update needed (returns None).
        """
        if not generated:
            return None

        # Clean surrounding quotes
        candidate = generated.strip()
        if (candidate.startswith('"') and candidate.endswith('"')) or (
            candidate.startswith("'") and candidate.endswith("'")
        ):
            candidate = candidate[1:-1].strip()

        if not candidate:
            return None

        # Rejection Rule 2: Conversational or explanatory prefixes
        lower = candidate.lower()
        forbidden_prefixes = (
            "here is",
            "here's",
            "sure",
            "note:",
            "explanation:",
            "corrected:",
            "subtitle:",
            "the corrected",
        )
        if any(lower.startswith(prefix) for prefix in forbidden_prefixes):
            return None

        # Rejection Rule 3: Markdown formatting
        if "`" in candidate or "```" in candidate or "##" in candidate:
            return None

        # Rejection Rule 4: Length check (prevent added details/summaries)
        if len(candidate) > (len(original) * 2 + 20):
            return None

        # If generated is identical to original, no update needed
        if candidate == original:
            return None

        return candidate
# ── End LLMCorrector port ─────────────────────────────────────────────────────


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def _resolve_asr_task_language(model_name: str):
    """Helper: multilingual models (no .en suffix) use task='translate' + auto-detect;
    English-only models use task='transcribe' + language='en'."""
    if model_name.endswith(".en"):
        return "en", "transcribe"
    else:
        # base, small, medium, large, turbo — multilingual -> translate to English
        return None, "translate"



class DesktopLiveSubtitlePipeline:
    """Single-process live subtitle engine linking WASAPI audio loopback to Qt desktop overlay."""

    def __init__(
        self,
        overlay: Optional["DesktopSubtitleOverlay"] = None,
        model_name: str = "base",
        min_chunk_floor_sec: float = 1.0,
        device_index: Optional[int] = None,
        indic_lang: Optional[str] = None,
        asr_language: Optional[str] = None,
        asr_task: Optional[str] = None,
        meter_callback: Optional[Callable[[float, float, str], None]] = None,
        subtitle_callback: Optional[Callable[[str, bool], None]] = None,
    ):
        self.overlay = overlay
        self.model_name = model_name
        self.min_chunk_floor_sec = min_chunk_floor_sec
        self.device_index = device_index
        self.indic_lang = indic_lang
        self.meter_callback = meter_callback
        self.subtitle_callback = subtitle_callback
        self._seg_counter = 0

        self.is_capturing = True
        self._shutdown_event = threading.Event()

        # ── Pipeline Components ───────────────────────────────────────────────
        self.preprocessor = AudioPreprocessor(
            input_sample_rate=48000,
            num_channels=2,
            target_sample_rate=16000,
            frame_size=512,
        )
        self.vad_engine = SileroVadEngine()
        self.segmenter = SpeechSegmenter(
            threshold=0.35,
            neg_threshold=0.20,
            min_speech_duration_ms=100,
            min_silence_duration_ms=200,
            sample_rate=16000,
            frame_size=512,
            min_chunk_floor_sec=min_chunk_floor_sec,
            max_chunk_duration_sec=2.5,
        )

        if not self.indic_lang:
            default_lang, default_task = _resolve_asr_task_language(model_name)
            self.asr_language = asr_language if asr_language is not None else default_lang
            self.asr_task = asr_task if asr_task is not None else default_task
            logging.info(f"[Pipeline] Loading ASR model '{model_name}' (task={self.asr_task} lang={self.asr_language!r}, Intel CPU optimized)...")
            self.asr_engine = OpenVINOASREngine(
                model_size=model_name,
                device="CPU",
                language=self.asr_language if self.asr_language is not None else "en" if self.asr_task == "transcribe" else None,
            )
            self.streaming_asr = StreamingASRManager(
                asr_engine=self.asr_engine,
                sample_rate=16000,
                context_window_sec=2.0,
                min_confirm_words=1,
            )
            self.nlp_processor = NLPPostProcessor(
                punctuation_enabled=True,
                hallucination_filter_enabled=True,
                duplicate_filter_enabled=True,
            )
            # For translate speed: cache last detected language to use as hint after first auto-detect
            self.last_detected_language = None

            # ── Step 3: Optional LLM post-correction (feature-gated, on-demand) ───────
            self.llm_corrector = LLMCorrector(
                model_path=os.environ.get("LLM_MODEL_PATH", ""),
                enabled=ASYNC_LLM_CORRECTION_ENABLED,
            )
        else:
            logging.info(f"[Pipeline] Indic mode enabled for language '{self.indic_lang}' (core_engine router).")
            self.asr_engine = None
            self.streaming_asr = None
            self.nlp_processor = None
            self.llm_corrector = None
            self.last_detected_language = None

        # Dynamically scale staleness safety ceiling to model inference duration on Intel CPU
        if self.indic_lang:
            self.stale_timeout_sec = 60.0
        elif model_name == "small":
            self.stale_timeout_sec = 18.0
        elif model_name == "base":
            self.stale_timeout_sec = 10.0
        else:
            self.stale_timeout_sec = max(6.0, STALE_AUDIO_TIMEOUT_SEC)

        self.audio_queue = queue.Queue(maxsize=300)
        self._format_logged = False
        self._stale_chunks_dropped = 0
        self.pipeline_thread = threading.Thread(
            target=self._pipeline_worker, name="PipelineWorker", daemon=True
        )
        self.loopback_capture: Optional[WASAPILoopbackCapture] = None

    def start(self):
        """Starts loopback capture thread and pipeline processing worker thread."""
        self._shutdown_event.clear()
        self.pipeline_thread.start()

        def on_raw_pcm(raw_bytes: bytes, sample_rate: int, num_channels: int, is_float32: bool = False):
            if not self.is_capturing:
                return
            self.preprocessor.set_format(sample_rate, num_channels)
            if not self._format_logged:
                self._format_logged = True
                logging.info(
                    f"[FORMAT] Capture->Preprocessor: {len(raw_bytes)} bytes | "
                    f"sr={sample_rate}Hz ch={num_channels} float32={is_float32}"
                )
            try:
                self.audio_queue.put_nowait((raw_bytes, is_float32, time.monotonic()))
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass

        self.loopback_capture = WASAPILoopbackCapture(
            audio_callback=on_raw_pcm, device_index=self.device_index
        )
        self.loopback_capture.start()
        logging.info("[Pipeline] Single-process PC Application Pipeline Started.")

    def stop(self):
        """Stops audio capture and pipeline worker."""
        logging.info("[Pipeline] Stopping PC Application Pipeline...")
        self._shutdown_event.set()
        if self.loopback_capture:
            self.loopback_capture.stop()

    def set_capturing(self, active: bool):
        """Toggles audio capture on/off from system tray menu."""
        self.is_capturing = active
        logging.info(f"[Pipeline] Audio capture state: {'ACTIVE' if active else 'PAUSED'}")
        if not active and self.overlay:
            self.overlay.push_live_text("")

    def set_model(self, model_name: str):
        """Dynamically reloads ASR model size ('base' multilingual / 'distil-small.en' / 'tiny.en')."""
        if model_name == self.model_name:
            return
        logging.info(f"[Pipeline] Switching ASR model to '{model_name}'...")
        self.model_name = model_name
        if not self.indic_lang:
            self.asr_language, self.asr_task = _resolve_asr_task_language(model_name)
            self.asr_engine = OpenVINOASREngine(
                model_size=model_name,
                device="CPU",
                language=self.asr_language if self.asr_language is not None else "en" if self.asr_task == "transcribe" else None,
            )
            if self.streaming_asr:
                self.streaming_asr.asr_engine = self.asr_engine
            self.last_detected_language = None
            logging.info(f"[Pipeline] Model switched to '{model_name}' (task={self.asr_task} lang={self.asr_language!r}) successfully.")
        else:
            logging.info(f"[Pipeline] Model name set to '{model_name}' (Indic mode active; using IndicConformer for '{self.indic_lang}').")

    def _pipeline_worker(self):
        """Single-process pipeline thread: audio_queue -> VAD -> ASR -> NLP -> Overlay."""
        frame_idx = 0
        speech_frames: list[np.ndarray] = []
        pre_roll_buffer: list[np.ndarray] = []
        pre_roll_max_frames = 6
        speech_start_ts = 0.0

        while not self._shutdown_event.is_set():
            try:
                raw_item = self.audio_queue.get(timeout=0.05)
                if isinstance(raw_item, tuple):
                    raw_bytes = raw_item[0]
                    is_float32 = raw_item[1] if len(raw_item) > 1 else False
                    enqueued_at = raw_item[2] if len(raw_item) > 2 else None
                else:
                    raw_bytes, is_float32, enqueued_at = raw_item, False, None
            except queue.Empty:
                continue

            # ── Staleness safety net ──────────────────────────────────────────
            if enqueued_at is not None and (
                time.monotonic() - enqueued_at
            ) > self.stale_timeout_sec:
                self._stale_chunks_dropped += 1
                if self._stale_chunks_dropped % 10 == 1:
                    logging.warning(
                        f"[STALE] Dropping audio chunk #{self._stale_chunks_dropped} "
                        f"(queued {(time.monotonic() - enqueued_at):.1f}s ago, ASR falling behind live audio)"
                    )
                # A stale chunk arriving mid-speech would leave the segmenter
                # stuck in SPEECH state forever; force-flush it.
                if self.segmenter.state.value != "NON_SPEECH" or speech_frames:
                    self.segmenter.reset()
                    speech_frames.clear()
                    if self.streaming_asr:
                        self.streaming_asr.reset()
                continue

            try:
                frames = self.preprocessor.process_raw_bytes(raw_bytes, is_float32=is_float32)
                for frame in frames:
                    frame_idx += 1
                    current_ts = round(frame_idx * 0.032, 3)

                    t_vad = time.perf_counter()
                    prob = self.vad_engine.predict(frame, sample_rate=16000)
                    event = self.segmenter.process_frame(frame, prob)
                    ev_type = event["type"]

                    peak = float(np.abs(frame).max()) if len(frame) > 0 else 0.0
                    if frame_idx % 31 == 0:  # ~1s meter
                        logging.info(f"[Audio Meter] t={current_ts:.2f}s | Peak: {peak:.4f} ({peak:.1%}) | VAD Prob: {prob:.3f} | State: {event['state']}")

                    if self.meter_callback and frame_idx % 2 == 0:
                        try:
                            self.meter_callback(peak, prob, event["state"])
                        except Exception:
                            pass

                    if ev_type == "SPEECH_START":
                        speech_frames = list(pre_roll_buffer) + [frame]
                        speech_start_ts = current_ts
                        pre_roll_buffer.clear()
                        logging.info(f"[VAD] >>> SPEECH START at t={current_ts:.2f}s (prob={prob:.3f}) <<<")

                    elif ev_type in ("SPEECH_CONTINUE", "NO_CHANGE") and event["state"] == "SPEECH":
                        speech_frames.append(frame)

                    elif ev_type == "SPEECH_END":
                        end_pcm = (
                            np.concatenate(speech_frames)
                            if speech_frames
                            else event.get("audio_pcm", np.empty(0, dtype=np.float32))
                        )
                        duration_s = len(end_pcm) / 16000.0
                        logging.info(f"[VAD] <<< SPEECH END at t={current_ts:.2f}s | duration={duration_s:.2f}s <<<")

                        if len(end_pcm) >= 3200:  # >= 200ms
                            self._process_completed_segment(end_pcm, speech_start_ts)

                        speech_frames.clear()

                    else:
                        pre_roll_buffer.append(frame)
                        if len(pre_roll_buffer) > pre_roll_max_frames:
                            pre_roll_buffer.pop(0)
            except Exception:
                logging.error("[PipelineWorker] Exception in pipeline worker:", exc_info=True)

    def _process_completed_segment(self, end_pcm: np.ndarray, speech_start_ts: float):
        """Transcribes a completed VAD speech segment and flushes the buffer to prevent leftover contamination."""
        if self.indic_lang:
            self._seg_counter += 1
            seg_num = self._seg_counter
            audio_dur_s = len(end_pcm) / 16000.0
            t0 = time.perf_counter()
            try:
                from core_engine import transcribe
                audio_float32 = end_pcm.astype(np.float32, copy=False)
                result = transcribe(audio_float32, language=self.indic_lang)
                asr_time_s = time.perf_counter() - t0
                rtf = asr_time_s / audio_dur_s if audio_dur_s > 0 else 0.0
                text = " ".join(s["text"] for s in result.get("segments", [])).strip()
                char_count = len(text)
                print(
                    f"[indic] seg {seg_num} | audio {audio_dur_s:.1f}s | asr {asr_time_s:.1f}s | RTF {rtf:.2f} | chars {char_count}",
                    flush=True,
                )
                if asr_time_s > audio_dur_s:
                    print(
                        f"[indic] seg {seg_num} | FALLING BEHIND (asr {asr_time_s:.1f}s > audio {audio_dur_s:.1f}s)",
                        flush=True,
                    )
                if text and self.overlay:
                    self.overlay.push_live_text(text, is_final=True)
                if text and self.subtitle_callback:
                    try:
                        self.subtitle_callback(text, True)
                    except Exception:
                        pass
            except Exception as e:
                asr_time_s = time.perf_counter() - t0
                logging.error(f"[indic] seg {seg_num} transcription failed: {e}")
            return

        t_asr = time.perf_counter()
        # For translate, use cached language hint after first detection to cut ~50% latency (auto-detect is expensive)
        if self.asr_task == "translate" and self.last_detected_language:
            effective_lang = self.last_detected_language
        else:
            effective_lang = self.asr_language  # None = auto-detect
        ev = self.streaming_asr.process_completed_segment(
            segment_audio=end_pcm, start_time=speech_start_ts, language=effective_lang, task=self.asr_task
        )
        asr_ms = (time.perf_counter() - t_asr) * 1000.0
        # Cache detected language for next segment speedup
        if ev and ev.get("language"):
            self.last_detected_language = ev.get("language")

        if not ev or not ev.get("text"):
            return

        raw_text = ev["text"].strip()
        is_final = ev.get("type") == "final"

        # NLP Post-Processing
        t_nlp = time.perf_counter()
        nlp_res = self.nlp_processor.process_final_transcript(
            text=raw_text, language=effective_lang or "en", no_speech_prob=0.0
        )
        nlp_ms = (time.perf_counter() - t_nlp) * 1000.0

        if nlp_res["suppressed"]:
            logging.info(f"[NLP] Suppressed segment (reason: '{nlp_res['reason']}')")
            return

        final_text = nlp_res["text"]
        detected = ev.get("language", "?")
        logging.info(
            f"[SUBTITLE] Final Output: \"{final_text}\" "
            f"(ASR: {asr_ms:.1f}ms, NLP: {nlp_ms:.1f}ms, lang: {detected}, task: {self.asr_task})"
        )

        # Push initial caption to Qt Desktop Overlay immediately (unchanged pipeline)
        if self.overlay:
            self.overlay.push_live_text(final_text, is_final=is_final)
        if self.subtitle_callback:
            try:
                self.subtitle_callback(final_text, is_final)
            except Exception:
                pass

        # ── Step 3: Schedule on-demand LLM post-correction (non-blocking) ────────
        if ASYNC_LLM_CORRECTION_ENABLED and self.llm_corrector:
            # Delay correction so the initial caption is displayed first, then corrected smoothly
            # after a short stabilization window (2s). Uses threading.Timer so ASR thread is not blocked.
            correction_timer = threading.Timer(
                2.0,
                self._apply_llm_correction,
                args=(final_text,),
            )
            correction_timer.daemon = True
            correction_timer.start()

    def _apply_llm_correction(self, raw_text: str):
        """Run LLM correction on the given text and push corrected caption if valid."""
        try:
            res = self.llm_corrector.correct(raw_text)
            if res and res.get("corrected_text"):
                corrected = res["corrected_text"]
                logging.info(
                    f"[LLM Correction] \"{raw_text}\" -> \"{corrected}\" "
                    f"(LLM: {res.get('inference_ms', 0)}ms)"
                )
                if self.overlay:
                    self.overlay.push_live_text(corrected, is_final=True)
            # If None returned (model unavailable, validation rejection, or no change),
            # the initial caption remains — no error, no jitter.
        except Exception as exc:
            logging.warning(f"[LLM Correction] Inference error: {exc}", exc_info=True)
# ── End Step 3 LLM correction scheduling ──────────────────────────────────────

    # Legacy per-segment handler (kept for reference; replaced by _transcribe_streaming_window)
    def _process_asr_segment(self, audio_pcm: np.ndarray, start_ts: float, duration_s: float):
        """Per-segment ASR transcription — DEPRECATED in favor of batched _transcribe_streaming_window."""
        logging.warning("[Pipeline] _process_asr_segment called (deprecated path)")
        self.streaming_asr.feed_audio(audio_pcm)
        self._transcribe_streaming_window(start_ts)


def main():
    parser = argparse.ArgumentParser(description="Single-Process PC Application Live Subtitles")
    parser.add_argument(
        "-i",
        "--device",
        type=int,
        default=None,
        help="WASAPI Loopback Device Index (e.g. 17 for Realtek Speaker). Default: Auto-detect",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available WASAPI audio loopback devices and exit",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="ASR Model size ('base' multilingual translate, 'distil-small.en' or 'tiny.en'). Default: from settings",
    )
    parser.add_argument(
        "--min-chunk-floor",
        type=float,
        default=None,
        help="Minimum chunk floor duration in seconds. Default: from settings",
    )
    parser.add_argument(
        "--font-size",
        default=None,
        choices=["small", "medium", "large"],
        help="Subtitle font size. Default: from settings",
    )
    parser.add_argument(
        "--position",
        default=None,
        choices=["bottom", "middle", "top"],
        help="Subtitle screen position. Default: from settings",
    )
    parser.add_argument(
        "--indic-lang",
        choices=["hi", "kn"],
        default=None,
        help="Opt-in Indic transcription language ('hi' or 'kn'). Default: not set",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Target spoken language (optional).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run headless / tray-only without opening the Control Center GUI.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Automatically start live subtitles immediately on launch.",
    )
    parser.add_argument(
        "--mode",
        choices=["shared", "local"],
        default=None,
        help="Inference mode: 'shared' (connect to remote Docker server) or 'local' (run in-process / local engine).",
    )
    parser.add_argument(
        "--remote-url",
        default=None,
        help="WebSocket URL of shared ML server (e.g. ws://shared-ml-server:8765).",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="Shortcut to enable shared remote inference mode.",
    )
    args = parser.parse_args()

    if args.list_devices:
        from app.loopback_capture import list_loopback_devices
        list_loopback_devices()
        sys.exit(0)

    indic_lang = args.indic_lang
    if indic_lang:
        print("Loading Indic model...", flush=True)
        t_load = time.time()
        from indic_backend import load_indic_model
        load_indic_model()
        elapsed = time.time() - t_load
        print(f"Indic model ready ({elapsed:.1f} s)", flush=True)

    # Load persisted settings; explicit CLI flags override them.
    global STALE_AUDIO_TIMEOUT_SEC
    settings = load_settings()
    if args.shared:
        settings["asr_mode"] = "shared"
    elif args.mode:
        settings["asr_mode"] = args.mode
    if args.remote_url:
        settings["remote_asr_url"] = args.remote_url

    model_from_cli = args.model is not None
    model_name = args.model if model_from_cli else str(settings.get("model", "distil-small.en"))
    min_chunk_floor = args.min_chunk_floor if args.min_chunk_floor is not None \
        else float(settings.get("min_chunk_floor_sec", 1.0))
    font_size = args.font_size if args.font_size is not None \
        else str(settings.get("font_size", "medium"))
    position = args.position if args.position is not None \
        else str(settings.get("position", "bottom"))
    STALE_AUDIO_TIMEOUT_SEC = float(settings.get("stale_audio_timeout_sec", 15.0))

    if not HAS_QT:
        logging.error("PyQt6 or PySide6 is required. Install via: pip install PyQt6")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Instantiate Desktop Overlay Window
    overlay = DesktopSubtitleOverlay(
        font_size_mode=font_size, position_mode=position
    )

    if indic_lang:
        orig_apply_style = overlay.apply_style
        def apply_indic_style():
            orig_apply_style()
            style = overlay.label.styleSheet()
            overlay.label.setStyleSheet(
                style.replace("'Segoe UI', Arial, sans-serif", "'Nirmala UI', sans-serif")
            )
        overlay.apply_style = apply_indic_style
        overlay.apply_style()

    # Enumerate devices for UI
    devices = []
    try:
        from app.loopback_capture import get_loopback_devices
        devices = get_loopback_devices()
    except Exception as e:
        logging.warning(f"Could not enumerate loopback devices: {e}")

    # Instantiate Studio Control Panel GUI
    from app.control_panel import ControlPanelWindow
    control_panel = ControlPanelWindow(current_settings=settings, device_list=devices)

    active_pipeline = None

    def start_pipeline_with_config(config: dict):
        nonlocal active_pipeline
        if active_pipeline:
            active_pipeline.stop()
            active_pipeline = None

        req_mode = str(config.get("asr_mode", settings.get("asr_mode", "shared")))
        req_remote_url = str(config.get("remote_asr_url", settings.get("remote_asr_url", "ws://localhost:8765")))
        req_model = str(config.get("model_name", model_name))
        req_indic = config.get("indic_lang")
        req_asr_lang = config.get("asr_language")
        req_asr_task = config.get("asr_task")
        req_dev_idx = config.get("device_index")
        req_font_size = str(config.get("font_size", font_size))
        req_pos = str(config.get("position", position))

        overlay.set_font_size(req_font_size)
        overlay.set_position(req_pos)

        if req_mode == "shared":
            logging.info(f"[ControlCenter] Launching Remote ASR Client -> {req_remote_url}")
            active_pipeline = RemoteLiveSubtitlePipeline(
                server_url=req_remote_url,
                overlay=overlay,
                device_index=req_dev_idx,
                meter_callback=lambda peak, prob, state: control_panel.meter_signal.emit(peak, prob, state),
                subtitle_callback=lambda text, is_final: control_panel.subtitle_signal.emit(text, is_final),
            )
            active_pipeline.start()
            logging.info(f"[ControlCenter] Remote pipeline streaming to {req_remote_url}.")
            return

        if req_indic:
            try:
                from indic_backend import load_indic_model
                load_indic_model()
            except Exception as e:
                logging.error(f"Indic model load failed: {e}")

        active_pipeline = DesktopLiveSubtitlePipeline(
            overlay=overlay,
            model_name=req_model,
            min_chunk_floor_sec=min_chunk_floor,
            device_index=req_dev_idx,
            indic_lang=req_indic,
            asr_language=req_asr_lang,
            asr_task=req_asr_task,
            meter_callback=lambda peak, prob, state: control_panel.meter_signal.emit(peak, prob, state),
            subtitle_callback=lambda text, is_final: control_panel.subtitle_signal.emit(text, is_final),
        )
        active_pipeline.start()
        logging.info(
            f"[ControlCenter] Local pipeline started (model='{req_model}', indic='{req_indic}', "
            f"lang='{req_asr_lang}', task='{req_asr_task}')."
        )

    def stop_active_pipeline():
        nonlocal active_pipeline
        if active_pipeline:
            active_pipeline.stop()
            active_pipeline = None
            logging.info("[ControlCenter] Pipeline stopped.")

    control_panel.start_requested.connect(start_pipeline_with_config)
    control_panel.stop_requested.connect(stop_active_pipeline)
    control_panel.pause_toggled.connect(
        lambda active: active_pipeline.set_capturing(active) if active_pipeline else None
    )
    control_panel.overlay_preview_requested.connect(lambda size, pos: (
        overlay.set_font_size(size),
        overlay.set_position(pos),
        overlay.push_live_text("✨ Live Subtitle Studio: Subtitle overlay preview!", is_final=True)
    ))

    def open_control_center():
        control_panel.show()
        control_panel.raise_()
        control_panel.activateWindow()

    # System Tray Controller
    tray = SystemTrayController(
        on_toggle_capture=lambda active: active_pipeline.set_capturing(active) if active_pipeline else None,
        on_change_model=lambda m: active_pipeline.set_model(m) if active_pipeline else None,
        on_change_font_size=overlay.set_font_size,
        on_change_position=overlay.set_position,
        on_open_settings=open_control_center,
        on_quit=lambda: _quit_app(app, lambda: active_pipeline),
    )

    logging.info("=" * 70)
    logging.info("Live Subtitle Studio Application Ready")
    logging.info("Control Center Frontend Active")
    logging.info("=" * 70)

    if not args.no_gui:
        control_panel.show()
        control_panel.raise_()
        control_panel.activateWindow()

    # If --start flag or headless CLI mode was passed, launch pipeline immediately
    if args.start or (args.no_gui and (model_from_cli or indic_lang)):
        control_panel._on_start_toggle_clicked()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        _quit_app(app, lambda: active_pipeline)


def _quit_app(app, get_pipeline_fn=None):
    logging.info("[App] Exiting PC Application...")
    if get_pipeline_fn:
        p = get_pipeline_fn()
        if p:
            p.stop()
    app.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
