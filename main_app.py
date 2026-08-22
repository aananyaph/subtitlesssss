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
from asr.asr_engine import OpenVINOASREngine, FasterWhisperEngine
from asr.streaming_asr import StreamingASRManager
from asr.nlp_processor import NLPPostProcessor
from app.loopback_capture import WASAPILoopbackCapture, HAS_PYAUDIO
from app.settings import load_settings, save_settings
from app.settings_window import SettingsWindow

# Staleness safety net: audio chunks that sit in the queue longer than this
# (because the ASR worker is busy) are dropped instead of transcribed late.
STALE_AUDIO_TIMEOUT_SEC = 2.0

try:
    from PyQt6.QtWidgets import QApplication
    from app.overlay_window import DesktopSubtitleOverlay
    from app.tray_icon import SystemTrayController
    HAS_QT = True
except ImportError:
    try:
        from PySide6.QtWidgets import QApplication
        from app.overlay_window import DesktopSubtitleOverlay
        from app.tray_icon import SystemTrayController
        HAS_QT = True
    except ImportError:
        HAS_QT = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class DesktopLiveSubtitlePipeline:
    """Single-process live subtitle engine linking WASAPI audio loopback to Qt desktop overlay."""

    def __init__(
        self,
        overlay: Optional["DesktopSubtitleOverlay"] = None,
        model_name: str = "distil-small.en",
        min_chunk_floor_sec: float = 1.0,
        device_index: Optional[int] = None,
    ):
        self.overlay = overlay
        self.model_name = model_name
        self.min_chunk_floor_sec = min_chunk_floor_sec
        self.device_index = device_index

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
        )

        logging.info(f"[Pipeline] Loading ASR model '{model_name}' (Intel CPU optimized)...")
        self.asr_engine = OpenVINOASREngine(
            model_size=model_name,
            device="CPU",
            language="en",
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
        """Dynamically reloads ASR model size ('distil-small.en' / 'tiny.en')."""
        if model_name == self.model_name:
            return
        logging.info(f"[Pipeline] Switching ASR model to '{model_name}'...")
        self.model_name = model_name
        self.asr_engine = OpenVINOASREngine(model_size=model_name, device="CPU", language="en")
        self.streaming_asr.asr_engine = self.asr_engine
        logging.info(f"[Pipeline] Model switched to '{model_name}' successfully.")

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
            ) > STALE_AUDIO_TIMEOUT_SEC:
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
        t_asr = time.perf_counter()
        ev = self.streaming_asr.process_completed_segment(
            segment_audio=end_pcm, start_time=speech_start_ts, language="en"
        )
        asr_ms = (time.perf_counter() - t_asr) * 1000.0

        if not ev or not ev.get("text"):
            return

        raw_text = ev["text"].strip()
        is_final = ev.get("type") == "final"

        # NLP Post-Processing
        t_nlp = time.perf_counter()
        nlp_res = self.nlp_processor.process_final_transcript(
            text=raw_text, language="en", no_speech_prob=0.0
        )
        nlp_ms = (time.perf_counter() - t_nlp) * 1000.0

        if nlp_res["suppressed"]:
            logging.info(f"[NLP] Suppressed segment (reason: '{nlp_res['reason']}')")
            return

        final_text = nlp_res["text"]
        logging.info(
            f"[SUBTITLE] Final Output: \"{final_text}\" "
            f"(ASR: {asr_ms:.1f}ms, NLP: {nlp_ms:.1f}ms)"
        )

        # Push to Qt Desktop Overlay
        if self.overlay:
            self.overlay.push_live_text(final_text, is_final=is_final)

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
        help="ASR Model size ('distil-small.en' or 'tiny.en'). Default: from settings",
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
    args = parser.parse_args()

    if args.list_devices:
        from app.loopback_capture import list_loopback_devices
        list_loopback_devices()
        sys.exit(0)

    # Load persisted settings; explicit CLI flags override them.
    global STALE_AUDIO_TIMEOUT_SEC
    settings = load_settings()
    model_from_cli = args.model is not None
    model_name = args.model if model_from_cli else str(settings.get("model", "distil-small.en"))
    min_chunk_floor = args.min_chunk_floor if args.min_chunk_floor is not None \
        else float(settings.get("min_chunk_floor_sec", 1.0))
    font_size = args.font_size if args.font_size is not None \
        else str(settings.get("font_size", "medium"))
    position = args.position if args.position is not None \
        else str(settings.get("position", "bottom"))
    STALE_AUDIO_TIMEOUT_SEC = float(settings.get("stale_audio_timeout_sec", 2.0))

    if not HAS_QT:
        logging.error("PyQt6 or PySide6 is required. Install via: pip install PyQt6")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Instantiate Desktop Overlay Window
    overlay = DesktopSubtitleOverlay(
        font_size_mode=font_size, position_mode=position
    )

    # Instantiate Pipeline Orchestrator
    pipeline = DesktopLiveSubtitlePipeline(
        overlay=overlay,
        model_name=model_name,
        min_chunk_floor_sec=min_chunk_floor,
        device_index=args.device,
    )

    # Settings window (persists to JSON; display options apply immediately)
    def apply_settings(saved: dict):
        save_settings(saved)
        try:
            overlay.set_font_size(str(saved["font_size"]))
        except Exception:
            pass
        try:
            overlay.set_position(str(saved["position"]))
        except Exception:
            pass
        try:
            pipeline.set_model(str(saved["model"]))
        except Exception:
            pass

    settings_window = SettingsWindow(settings, on_saved=None if model_from_cli else apply_settings)

    def open_settings():
        settings_window.show()
        settings_window.raise_()
        settings_window.activateWindow()

    # Instantiate System Tray Controller
    tray = SystemTrayController(
        on_toggle_capture=pipeline.set_capturing,
        on_change_model=pipeline.set_model,
        on_change_font_size=overlay.set_font_size,
        on_change_position=overlay.set_position,
        on_open_settings=open_settings,
        on_quit=lambda: _quit_app(app, pipeline),
    )

    pipeline.start()

    logging.info("=" * 70)
    logging.info("Single-Process PC Subtitle Application Ready")
    logging.info("Capturing System Audio via WASAPI Loopback")
    logging.info("Subtitles rendering on Desktop Overlay Window")
    logging.info("=" * 70)

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        _quit_app(app, pipeline)


def _quit_app(app, pipeline):
    logging.info("[App] Exiting PC Application...")
    pipeline.stop()
    app.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
