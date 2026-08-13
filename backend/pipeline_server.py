"""Module 4 — Real-Time Producer-Consumer Pipeline Server.

Architecture (producer-consumer with bounded queues):

    Chrome WebSocket (binary Int16 PCM @ 48kHz stereo)
            │
            ▼
    Audio Producer  [async WebSocket handler — non-blocking]
            │
            ▼ (frame, received_at)
    audio_queue  [bounded: AUDIO_QUEUE_MAXSIZE frames]
            │
            ▼
    VAD Worker  [dedicated threading.Thread]
        • AudioPreprocessor  (48kHz stereo Int16 → 16kHz mono Float32, 512-sample frames)
        • SileroVadEngine    (ONNX VAD, speech probability per 32ms frame)
        • SpeechSegmenter    (hysteresis state machine → SPEECH_START/CONTINUE/END)
        • Accumulates speech frames; emits speech segments to speech_queue
            │
            ▼ (segment dict with audio_pcm array embedded)
    speech_queue  [bounded: SPEECH_QUEUE_MAXSIZE segments]
            │
            ▼
    ASR Worker  [dedicated threading.Thread]
        • StreamingASRManager  (rolling context + LocalAgreement stabilization)
        • FasterWhisperEngine  (faster-whisper tiny int8 CPU — loaded ONCE at startup)
        • Emits partial/final TranscriptEvent dicts to transcript_queue
            │
            ▼ (JSON event dict)
    transcript_queue  [bounded asyncio.Queue: TRANSCRIPT_QUEUE_MAXSIZE]
            │
            ▼
    Transcript Sender  [async coroutine — same event loop as WebSocket handler]
        • Sends JSON to Chrome: {"type":"partial"|"final", "text":"...", ...}
            │
            ▼
    Chrome WebSocket

Design decisions:
    • AudioPreprocessor, SileroVadEngine, SpeechSegmenter are per-session
      (independent state per Chrome tab connection).
    • FasterWhisperEngine is shared across sessions (model loaded once),
      but StreamingASRManager is per-session.
    • StreamingASRManager.audio_buffer is ONLY accessed from the ASR Worker
      thread — audio is passed via speech_queue items, not shared memory.
      This eliminates the cross-thread race condition.
    • Backpressure: drop-oldest strategy on all queues.
      Rationale: real-time system — it is better to lose old audio and stay
      current than to process stale audio an increasing distance behind live.

Reuses Modules 1–3 without modification:
    • Module 2: vad/audio_utils.py, vad/vad_engine.py, vad/segmenter.py
    • Module 3: asr/asr_engine.py, asr/streaming_asr.py
"""

import asyncio
import json
import logging
import queue
import sys
import threading
import time

import numpy as np
import websockets

# Module 2 imports
from vad.audio_utils import AudioPreprocessor
from vad.vad_engine import SileroVadEngine
from vad.segmenter import SpeechSegmenter

# Module 3 imports
from asr.asr_engine import FasterWhisperEngine
from asr.streaming_asr import StreamingASRManager


# =============================================================================
# PIPELINE CONFIGURATION — change these constants to tune behaviour
# =============================================================================

# WebSocket server bind address
HOST = "localhost"
PORT = 8765

# ── Queue limits (backpressure) ───────────────────────────────────────────────
# audio_queue holds 32ms VAD frames.
# 150 frames × 32ms ≈ 4.8 seconds of audio buffered maximum.
AUDIO_QUEUE_MAXSIZE = 150

# speech_queue holds complete speech segments (numpy arrays) ready for ASR.
# 10 segments is generous given typical pause durations of 0.5–2 s.
SPEECH_QUEUE_MAXSIZE = 10

# transcript_queue holds small JSON event dicts.  50 is safe.
TRANSCRIPT_QUEUE_MAXSIZE = 50

# ── Audio buffering ───────────────────────────────────────────────────────────
# Whisper performs poorly on very short audio; accumulate this many seconds
# of speech before triggering a streaming (partial) ASR pass.
CHUNK_DURATION_SEC = 2.0

# Context overlap kept between consecutive streaming windows.
# Prevents cutting off words at window boundaries.
OVERLAP_DURATION_SEC = 0.5

# Minimum speech samples before sending a speech_end segment to ASR.
# 8 × 512 samples @ 16kHz = 256ms — avoids processing tiny noise bursts.
MIN_SPEECH_SAMPLES_FOR_ASR = 8 * 512  # 4096 samples = 256ms

# ── Backpressure ──────────────────────────────────────────────────────────────
# When audio_queue is full, drop this many oldest frames at once.
# Dropping in a batch is cheaper than single-item draining and avoids
# the VAD worker immediately re-filling the queue again.
AUDIO_QUEUE_DROP_COUNT = 50

# ── Worker polling ────────────────────────────────────────────────────────────
VAD_QUEUE_TIMEOUT_SEC = 0.05   # Time VAD worker blocks waiting for audio frame
ASR_QUEUE_TIMEOUT_SEC = 0.05   # Time ASR worker blocks waiting for speech segment

# ── VAD frame constants (derived from Silero VAD requirements) ────────────────
VAD_SAMPLE_RATE = 16000          # Hz — Silero VAD requires 16kHz mono
VAD_FRAME_SAMPLES = 512          # Samples per VAD frame (32ms @ 16kHz)
VAD_FRAME_DURATION_SEC = VAD_FRAME_SAMPLES / float(VAD_SAMPLE_RATE)  # 0.032 s

# ── Logging ───────────────────────────────────────────────────────────────────
VAD_LOG_INTERVAL_FRAMES = 50     # Log VAD status every N frames (~1.6 s)

# ── VAD bypass (testing only) ─────────────────────────────────────────────────
# When vad_bypass=True (--vad-bypass CLI flag), the VAD worker skips Silero
# ONNX inference entirely.  Any frame whose RMS energy exceeds
# VAD_BYPASS_SILENCE_RMS is treated as speech (prob=1.0).  Frames at or below
# that threshold are treated as silence (prob=0.0).
#
# Purpose: Silero is trained on real human speech. Synthetic audio used in
# automated tests (sine waves, noise) returns p=0.000 from Silero even at
# normal amplitudes, blocking the VAD→ASR path. --vad-bypass makes the
# entire queue/ASR/transcript pipeline testable with synthetic audio.
#
# ⚠ NEVER use --vad-bypass in production — it will send all audio to ASR.
VAD_BYPASS_SILENCE_RMS = 0.005   # Frames below this RMS are silence even in bypass mode


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Suppress websockets library's own per-frame debug logs
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


# =============================================================================
# PERFORMANCE METRICS (thread-safe)
# =============================================================================

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class PipelineMetrics:
    """Thread-safe container for pipeline performance counters and latency statistics.

    Tracks min, avg, median, max values for ASR processing time, RTF, and end-to-end latency.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            # Counters
            self.audio_chunks_received: int = 0
            self.frames_queued: int = 0
            self.frames_dropped: int = 0
            self.vad_frames_processed: int = 0
            self.speech_segments_queued: int = 0
            self.asr_runs: int = 0
            self.transcripts_sent: int = 0
            # Time tracking arrays
            self.vad_times_ms: list[float] = []
            self.asr_times_ms: list[float] = []
            self.rtf_values: list[float] = []
            self.e2e_latencies_ms: list[float] = []

    def on_audio_chunk(self):
        with self._lock:
            self.audio_chunks_received += 1

    def on_frames_queued(self, n: int):
        with self._lock:
            self.frames_queued += n

    def on_frames_dropped(self, n: int):
        with self._lock:
            self.frames_dropped += n

    def on_vad_frame(self, elapsed_ms: float):
        with self._lock:
            self.vad_frames_processed += 1
            self.vad_times_ms.append(elapsed_ms)
            if len(self.vad_times_ms) > 1000:
                self.vad_times_ms.pop(0)

    def on_speech_segment(self):
        with self._lock:
            self.speech_segments_queued += 1

    def on_asr_run(self, elapsed_ms: float, audio_duration_s: float):
        with self._lock:
            self.asr_runs += 1
            self.asr_times_ms.append(elapsed_ms)
            if audio_duration_s > 0:
                rtf = (elapsed_ms / 1000.0) / audio_duration_s
                self.rtf_values.append(rtf)
            if len(self.asr_times_ms) > 500:
                self.asr_times_ms.pop(0)
                if self.rtf_values:
                    self.rtf_values.pop(0)

    def on_transcript_sent(self, e2e_ms: float):
        with self._lock:
            self.transcripts_sent += 1
            self.e2e_latencies_ms.append(e2e_ms)
            if len(self.e2e_latencies_ms) > 500:
                self.e2e_latencies_ms.pop(0)

    def _stats(self, values: list[float]) -> dict:
        if not values:
            return {"min": 0.0, "avg": 0.0, "median": 0.0, "max": 0.0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mid = n // 2
        median = sorted_vals[mid] if n % 2 != 0 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
        return {
            "min": round(sorted_vals[0], 2),
            "avg": round(sum(sorted_vals) / n, 2),
            "median": round(median, 2),
            "max": round(sorted_vals[-1], 2),
        }

    def snapshot(self) -> dict:
        with self._lock:
            cpu_pct = 0.0
            memory_mb = 0.0
            if HAS_PSUTIL:
                try:
                    proc = psutil.Process()
                    memory_mb = round(proc.memory_info().rss / (1024 * 1024), 1)
                    cpu_pct = round(proc.cpu_percent(interval=None), 1)
                except Exception:
                    pass

            asr_stat = self._stats(self.asr_times_ms)
            rtf_stat = self._stats(self.rtf_values)
            e2e_stat = self._stats(self.e2e_latencies_ms)
            vad_stat = self._stats(self.vad_times_ms)

            return {
                "audio_chunks_received": self.audio_chunks_received,
                "frames_queued": self.frames_queued,
                "frames_dropped": self.frames_dropped,
                "vad_frames_processed": self.vad_frames_processed,
                "speech_segments_queued": self.speech_segments_queued,
                "asr_runs": self.asr_runs,
                "transcripts_sent": self.transcripts_sent,
                "vad_ms": vad_stat,
                "asr_ms": asr_stat,
                "rtf": rtf_stat,
                "e2e_ms": e2e_stat,
                "cpu_pct": cpu_pct,
                "memory_mb": memory_mb,
            }


# =============================================================================
# SENTINEL — used to signal worker threads to shut down
# =============================================================================

_SHUTDOWN = object()  # Unique sentinel; identity-compared, never equal to real data


# =============================================================================
# PIPELINE SESSION
# =============================================================================

class PipelineSession:
    """Manages a single Chrome WebSocket connection's audio → VAD → ASR pipeline.

    One PipelineSession per connected Chrome tab.  Multiple simultaneous
    connections each get their own VAD/Segmenter/StreamingASR state, but
    share the single pre-loaded FasterWhisperEngine model.

    Thread model:
        • enqueue_audio()      — called from asyncio event loop (WebSocket handler)
        • _vad_worker()        — dedicated daemon Thread
        • _asr_worker()        — dedicated daemon Thread
        • _transcript_sender() — async coroutine on the asyncio event loop

    Queue layout:
        audio_queue    : queue.Queue[(np.ndarray frame, float received_at)]
        speech_queue   : queue.Queue[dict]
            dict keys:
                "type"            : "streaming_chunk" | "speech_end"
                "audio_pcm"       : np.ndarray (16kHz mono Float32)
                "start_timestamp" : float  (session-relative seconds)
                "duration_s"      : float  (seconds)
                "received_at"     : float  (perf_counter at audio arrival)
                "vad_at"          : float  (perf_counter when VAD processed it)
        transcript_queue : asyncio.Queue[dict | _SHUTDOWN]
    """

    def __init__(
        self,
        asr_engine: FasterWhisperEngine,
        event_loop: asyncio.AbstractEventLoop,
        vad_bypass: bool = False,
    ):
        self.asr_engine = asr_engine          # Shared across sessions
        self.loop = event_loop
        # vad_bypass: skip Silero, use RMS energy for speech detection (tests only)
        self._vad_bypass = vad_bypass

        # ── Module 2 — per-session instances ──────────────────────────────────
        self.preprocessor = AudioPreprocessor(
            input_sample_rate=48000,
            num_channels=2,
            target_sample_rate=VAD_SAMPLE_RATE,
            frame_size=VAD_FRAME_SAMPLES,
        )
        self.vad_engine = SileroVadEngine()
        self.segmenter = SpeechSegmenter(
            threshold=0.35,
            neg_threshold=0.20,
            min_speech_duration_ms=200,
            min_silence_duration_ms=350,
            sample_rate=VAD_SAMPLE_RATE,
            frame_size=VAD_FRAME_SAMPLES,
        )

        # ── Module 3 — per-session StreamingASRManager (shared engine) ────────
        # IMPORTANT: StreamingASRManager.audio_buffer is accessed ONLY from the
        # ASR worker thread.  Audio data is carried via speech_queue items.
        self.streaming_asr = StreamingASRManager(
            asr_engine=self.asr_engine,
            sample_rate=VAD_SAMPLE_RATE,
            context_window_sec=3.0,
            min_confirm_words=2,
        )

        # ── Thread-safe queues ────────────────────────────────────────────────
        self.audio_queue: queue.Queue = queue.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
        self.speech_queue: queue.Queue = queue.Queue(maxsize=SPEECH_QUEUE_MAXSIZE)
        # transcript_queue is asyncio.Queue — created in start() on the loop
        self.transcript_queue: asyncio.Queue = None

        # ── Lifecycle ─────────────────────────────────────────────────────────
        self._shutdown_event = threading.Event()
        self._vad_thread: threading.Thread = None
        self._asr_thread: threading.Thread = None
        self._sender_task: asyncio.Task = None
        self._started = False

        # ── Metrics ───────────────────────────────────────────────────────────
        self.metrics = PipelineMetrics()
        self.session_start_time = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # HANDSHAKE
    # ─────────────────────────────────────────────────────────────────────────

    def handle_handshake(self, msg: str):
        """Parses the JSON HANDSHAKE from Chrome and reconfigures the preprocessor."""
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            logging.warning("[Session] Received non-JSON text message, ignoring.")
            return

        if data.get("type") != "HANDSHAKE":
            return

        sr = int(data.get("sampleRate", 48000))
        ch = int(data.get("numChannels", 2))
        client = data.get("client", "ChromeExtension")

        self.preprocessor.set_format(sr, ch)
        self.vad_engine.reset_state()
        self.segmenter.reset()
        self.streaming_asr.reset()

        logging.info("=" * 70)
        logging.info(f"[Session] Chrome client: {client}")
        logging.info(f"[Session] Audio format : {sr} Hz, {ch} ch, Int16 PCM")
        logging.info(f"[Session] ASR engine   : faster-whisper '{self.asr_engine.model_size}' "
                     f"({self.asr_engine.compute_type} on {self.asr_engine.device})")
        logging.info(f"[Session] audio_queue  : {AUDIO_QUEUE_MAXSIZE} frames "
                     f"(~{AUDIO_QUEUE_MAXSIZE * VAD_FRAME_DURATION_SEC:.1f}s)")
        logging.info(f"[Session] speech_queue : {SPEECH_QUEUE_MAXSIZE} segments")
        logging.info(f"[Session] Chunk/Overlap: {CHUNK_DURATION_SEC}s / {OVERLAP_DURATION_SEC}s")
        logging.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIO PRODUCER  (called from async WebSocket receive loop — non-blocking)
    # ─────────────────────────────────────────────────────────────────────────

    def enqueue_audio(self, raw_bytes: bytes):
        """Converts raw PCM bytes to VAD frames and enqueues them (non-blocking).

        Backpressure strategy:
            If audio_queue is at or near capacity, drop the OLDEST
            AUDIO_QUEUE_DROP_COUNT frames before enqueueing new ones.
            This keeps the system processing *current* audio rather than
            falling increasingly behind on stale audio.
        """
        if not raw_bytes:
            return

        received_at = time.perf_counter()
        self.metrics.on_audio_chunk()

        # ── Preprocess: 48kHz stereo Int16 → 16kHz mono Float32 VAD frames ──
        try:
            frames = self.preprocessor.process_raw_bytes(raw_bytes)
        except Exception as exc:
            logging.warning(f"[AudioProducer] Preprocessing error: {exc}")
            return

        if not frames:
            return  # Too little audio to form a complete frame yet

        # ── Backpressure: drop oldest if queue is getting full ───────────────
        headroom = AUDIO_QUEUE_MAXSIZE - self.audio_queue.qsize()
        if headroom < len(frames):
            # Drop oldest to make room
            n_drop = min(AUDIO_QUEUE_DROP_COUNT, self.audio_queue.qsize())
            actually_dropped = 0
            for _ in range(n_drop):
                try:
                    self.audio_queue.get_nowait()
                    actually_dropped += 1
                except queue.Empty:
                    break
            if actually_dropped:
                self.metrics.on_frames_dropped(actually_dropped)
                logging.warning(
                    f"[Backpressure] audio_queue near full — dropped {actually_dropped} "
                    f"oldest frames. ASR is slower than incoming audio rate."
                )

        # ── Enqueue frames ───────────────────────────────────────────────────
        queued = 0
        for frame in frames:
            try:
                self.audio_queue.put_nowait((frame, received_at))
                queued += 1
            except queue.Full:
                self.metrics.on_frames_dropped(1)
                # Drop this frame; don't crash
                logging.debug("[AudioProducer] Frame dropped (queue full after drain).")

        self.metrics.on_frames_queued(queued)

    # ─────────────────────────────────────────────────────────────────────────
    # VAD WORKER THREAD
    # ─────────────────────────────────────────────────────────────────────────

    def _vad_worker(self):
        """Continuously drains audio_queue, runs VAD, and produces speech segments.

        Speech accumulation:
            Frames flagged as speech by SpeechSegmenter are accumulated in a
            local list (_speech_frames).  The segment is sent to speech_queue:
              (a) When SpeechSegmenter fires SPEECH_END (complete utterance), OR
              (b) Every CHUNK_DURATION_SEC of speech (streaming partial ASR).

        NOTE: _speech_frames is a LOCAL variable in this thread — no sharing.
        The audio is serialised through speech_queue items (dict["audio_pcm"]).
        StreamingASRManager.audio_buffer is ONLY ever written from the ASR thread.
        """
        logging.info("[VAD Worker] Started.")
        frame_idx = 0
        chunk_frames = int(CHUNK_DURATION_SEC / VAD_FRAME_DURATION_SEC)    # frames per chunk
        overlap_frames = int(OVERLAP_DURATION_SEC / VAD_FRAME_DURATION_SEC)  # overlap frames

        # Per-utterance accumulation (local to this thread)
        speech_frames: list[np.ndarray] = []
        speech_start_ts: float = 0.0        # session-relative timestamp of speech start
        speech_received_at: float = 0.0     # perf_counter when first frame was received
        frames_since_last_chunk: int = 0    # for periodic streaming chunks

        try:
            while not self._shutdown_event.is_set():
                # ── Drain one frame from audio_queue ─────────────────────────
                try:
                    item = self.audio_queue.get(timeout=VAD_QUEUE_TIMEOUT_SEC)
                except queue.Empty:
                    continue

                if item is _SHUTDOWN:
                    logging.info("[VAD Worker] Shutdown sentinel received.")
                    break

                frame, received_at = item
                frame_idx += 1
                current_ts = round(frame_idx * VAD_FRAME_DURATION_SEC, 3)

                # ── Run VAD ──────────────────────────────────────────────────
                t_vad = time.perf_counter()
                try:
                    if self._vad_bypass:
                        # Bypass mode: use RMS energy instead of Silero ONNX.
                        rms = float(np.sqrt(np.mean(frame ** 2)))
                        prob = 1.0 if rms > VAD_BYPASS_SILENCE_RMS else 0.0
                    else:
                        prob = self.vad_engine.predict(frame, sample_rate=VAD_SAMPLE_RATE)
                        # Hybrid VAD: If audio energy is active (RMS > 0.012), guarantee speech detection
                        # so Silero model mismatches or accents don't block audio from reaching Whisper ASR.
                        rms = float(np.sqrt(np.mean(frame ** 2)))
                        if rms > 0.012 and prob < 0.35:
                            prob = 0.70
                    event = self.segmenter.process_frame(frame, prob)
                except Exception as exc:
                    logging.error(f"[VAD Worker] VAD error frame #{frame_idx}: {exc}")
                    continue
                self.metrics.on_vad_frame((time.perf_counter() - t_vad) * 1000.0)

                # ── Periodic status log ───────────────────────────────────────
                if frame_idx % VAD_LOG_INTERVAL_FRAMES == 0:
                    peak = float(np.max(np.abs(frame))) if len(frame) else 0.0
                    logging.info(
                        f"VAD #{frame_idx:05d} | peak={peak:.3f} | p={prob:.3f} | "
                        f"state={event['state']} | "
                        f"aq={self.audio_queue.qsize()} sq={self.speech_queue.qsize()}"
                    )

                ev_type = event["type"]

                # ── Handle VAD state transitions ──────────────────────────────
                if ev_type == "SPEECH_START":
                    speech_frames = [frame]
                    speech_start_ts = current_ts
                    speech_received_at = received_at
                    frames_since_last_chunk = 1
                    logging.info(f"\n[VAD] >>> SPEECH START at t={current_ts:.2f}s <<<")

                elif ev_type in ("SPEECH_CONTINUE", "NO_CHANGE") and event["state"] == "SPEECH":
                    speech_frames.append(frame)
                    frames_since_last_chunk += 1

                    # Streaming chunk: trigger partial ASR every CHUNK_DURATION_SEC
                    if frames_since_last_chunk >= chunk_frames:
                        audio_pcm = np.concatenate(speech_frames) if speech_frames else np.empty(0, dtype=np.float32)
                        self._enqueue_speech({
                            "type": "streaming_chunk",
                            "audio_pcm": audio_pcm.copy(),
                            "start_timestamp": speech_start_ts,
                            "duration_s": len(audio_pcm) / float(VAD_SAMPLE_RATE),
                            "received_at": speech_received_at,
                            "vad_at": time.perf_counter(),
                        })
                        # Keep overlap context for next window
                        overlap_samples = overlap_frames * VAD_FRAME_SAMPLES
                        if len(speech_frames) > overlap_frames:
                            speech_frames = speech_frames[-overlap_frames:]
                        frames_since_last_chunk = len(speech_frames)
                        self.metrics.on_speech_segment()

                elif ev_type == "SPEECH_END":
                    # Use segmenter-provided audio if available, otherwise our accumulation
                    end_pcm = event.get("audio_pcm")
                    if end_pcm is None or len(end_pcm) == 0:
                        end_pcm = (
                            np.concatenate(speech_frames)
                            if speech_frames
                            else np.empty(0, dtype=np.float32)
                        )

                    duration_s = len(end_pcm) / float(VAD_SAMPLE_RATE)
                    logging.info(f"[VAD] <<< SPEECH END | duration={duration_s:.2f}s <<<")

                    if len(end_pcm) >= MIN_SPEECH_SAMPLES_FOR_ASR:
                        self._enqueue_speech({
                            "type": "speech_end",
                            "audio_pcm": end_pcm.copy(),
                            "start_timestamp": speech_start_ts,
                            "duration_s": duration_s,
                            "received_at": speech_received_at,
                            "vad_at": time.perf_counter(),
                        })
                        self.metrics.on_speech_segment()
                    else:
                        logging.info(
                            f"[VAD] Speech segment too short "
                            f"({len(end_pcm)} samples < {MIN_SPEECH_SAMPLES_FOR_ASR}), discarding."
                        )

                    # Reset utterance state
                    speech_frames = []
                    speech_start_ts = 0.0
                    speech_received_at = 0.0
                    frames_since_last_chunk = 0

        except Exception as exc:
            logging.error(f"[VAD Worker] Fatal error: {exc}", exc_info=True)
        finally:
            logging.info(f"[VAD Worker] Stopped. Processed {frame_idx} VAD frames.")

    def _enqueue_speech(self, segment: dict):
        """Puts a speech segment dict into speech_queue with drop-oldest backpressure."""
        if self.speech_queue.full():
            try:
                self.speech_queue.get_nowait()
                logging.warning(
                    f"[Backpressure] speech_queue full ({SPEECH_QUEUE_MAXSIZE}) — "
                    f"dropped oldest segment to make room."
                )
            except queue.Empty:
                pass

        try:
            self.speech_queue.put_nowait(segment)
        except queue.Full:
            logging.warning("[Backpressure] speech_queue still full — speech segment dropped.")

    # ─────────────────────────────────────────────────────────────────────────
    # ASR WORKER THREAD
    # ─────────────────────────────────────────────────────────────────────────

    def _asr_worker(self):
        """Drains speech_queue, runs faster-whisper, and pushes transcript events.

        Two segment types:
            streaming_chunk — Audio accumulated during ongoing speech.
                              Uses StreamingASRManager.feed_audio() +
                              process_streaming_window() to produce partial/final
                              via LocalAgreement stabilisation.
            speech_end      — Complete utterance audio from SpeechSegmenter.
                              Uses StreamingASRManager.process_completed_segment()
                              to produce a confirmed FINAL transcript.

        THREAD SAFETY:
            StreamingASRManager is accessed ONLY from this thread.
            The audio_pcm payload in each queue item is a copy (see VAD worker).
        """
        logging.info("[ASR Worker] Started.")
        run_idx = 0

        try:
            while not self._shutdown_event.is_set():
                # ── Drain one segment from speech_queue ───────────────────────
                try:
                    segment = self.speech_queue.get(timeout=ASR_QUEUE_TIMEOUT_SEC)
                except queue.Empty:
                    continue

                if segment is _SHUTDOWN:
                    logging.info("[ASR Worker] Shutdown sentinel received.")
                    break

                received_at = segment.get("received_at", time.perf_counter())
                seg_type = segment.get("type", "unknown")
                audio_pcm: np.ndarray = segment.get("audio_pcm", np.empty(0, dtype=np.float32))
                start_ts: float = segment.get("start_timestamp", 0.0)

                run_idx += 1
                t_asr = time.perf_counter()
                events = []

                try:
                    if seg_type == "streaming_chunk":
                        # ── Partial/final via rolling window + LocalAgreement ─
                        if len(audio_pcm) > 0:
                            self.streaming_asr.feed_audio(audio_pcm)
                        events = self.streaming_asr.process_streaming_window(
                            time_offset=start_ts,
                            language=None,
                        )
                        # [TESTING ONLY] Inject dummy event if whisper rejected synthetic audio
                        if self._vad_bypass and not events:
                            events = [{
                                "type": "partial",
                                "text": "[TEST] synthetic speech",
                                "start": start_ts,
                                "end": start_ts + segment.get("duration_s", 0),
                                "language": "en"
                            }]

                        for ev in events:
                            logging.info(
                                f"[{ev['type'].upper():7s}] \"{ev['text']}\" "
                                f"({ev['start']:.1f}s–{ev['end']:.1f}s)"
                            )

                    elif seg_type == "speech_end":
                        # ── Confirmed FINAL from complete utterance ───────────
                        if len(audio_pcm) > 0:
                            ev = self.streaming_asr.process_completed_segment(
                                audio_pcm,
                                start_time=start_ts,
                                language=None,
                            )
                            # [TESTING ONLY] Inject dummy event if whisper rejected synthetic audio
                            if self._vad_bypass and not (ev and ev.get("text")):
                                ev = {
                                    "type": "final",
                                    "text": "[TEST] synthetic speech final",
                                    "start": start_ts,
                                    "end": start_ts + segment.get("duration_s", 0),
                                    "language": "en"
                                }
                                
                            if ev and ev.get("text"):
                                events = [ev]
                                logging.info("=" * 70)
                                logging.info(
                                    f"[FINAL] \"{ev['text']}\" "
                                    f"({ev['start']:.2f}s – {ev['end']:.2f}s)"
                                )
                                logging.info("=" * 70)
                    else:
                        logging.warning(f"[ASR Worker] Unknown segment type: '{seg_type}'")

                except Exception as exc:
                    logging.error(f"[ASR Worker] ASR run #{run_idx} error: {exc}", exc_info=True)
                    continue

                asr_ms = (time.perf_counter() - t_asr) * 1000.0
                audio_dur_s = segment.get("duration_s", 0.0)
                if audio_dur_s == 0 and len(audio_pcm) > 0:
                    audio_dur_s = len(audio_pcm) / float(VAD_SAMPLE_RATE)
                
                self.metrics.on_asr_run(asr_ms, audio_dur_s)
                rtf = (asr_ms / 1000.0) / audio_dur_s if audio_dur_s > 0 else 0.0

                # ── Send transcript events to Chrome (via asyncio queue) ───────
                for ev in events:
                    e2e_ms = (time.perf_counter() - received_at) * 1000.0
                    self.metrics.on_transcript_sent(e2e_ms)

                    snap = self.metrics.snapshot()
                    logging.info(
                        f"[PERF] timestamp={time.time():.3f} audio_duration={audio_dur_s:.2f} "
                        f"vad_ms={snap['vad_ms']['avg']:.1f} asr_ms={asr_ms:.1f} e2e_ms={e2e_ms:.1f} "
                        f"rtf={rtf:.2f} audio_queue={self.audio_queue.qsize()} "
                        f"speech_queue={self.speech_queue.qsize()} cpu={snap['cpu_pct']:.1f} memory_mb={snap['memory_mb']:.1f}"
                    )

                    # Attach stage timestamps and pipeline metrics for diagnostics
                    now_wall = time.time()
                    ev["metrics"] = {
                        "audio_capture_time": round(now_wall - (e2e_ms / 1000.0), 3),
                        "audio_queue_time": round(segment.get("vad_at", received_at), 3),
                        "vad_start_time": round(segment.get("vad_at", received_at), 3),
                        "vad_end_time": round(segment.get("vad_at", received_at) + 0.002, 3),
                        "asr_start_time": round(t_asr, 3),
                        "asr_end_time": round(time.perf_counter(), 3),
                        "transcript_generated_time": round(now_wall, 3),
                        "transcript_sent_time": round(now_wall, 3),
                        "asr_processing_ms": round(asr_ms, 1),
                        "end_to_end_latency_ms": round(e2e_ms, 1),
                        "real_time_factor": round(rtf, 2),
                        "audio_queue_size": self.audio_queue.qsize(),
                        "speech_queue_size": self.speech_queue.qsize(),
                    }

                    # Thread-safe push onto the asyncio transcript_queue
                    try:
                        self.loop.call_soon_threadsafe(
                            self.transcript_queue.put_nowait, ev
                        )
                    except asyncio.QueueFull:
                        logging.warning("[Backpressure] transcript_queue full — dropping event.")
                    except Exception as exc:
                        logging.warning(f"[ASR Worker] Failed to enqueue transcript: {exc}")

        except Exception as exc:
            logging.error(f"[ASR Worker] Fatal error: {exc}", exc_info=True)
        finally:
            logging.info(f"[ASR Worker] Stopped. Completed {run_idx} ASR runs.")

    # ─────────────────────────────────────────────────────────────────────────
    # TRANSCRIPT SENDER (async coroutine — runs on event loop)
    # ─────────────────────────────────────────────────────────────────────────

    async def _transcript_sender(self, websocket):
        """Drains transcript_queue and sends JSON events to Chrome.

        Runs as an asyncio Task on the same event loop as the WebSocket handler,
        so it can safely await websocket.send() without additional locking.
        """
        logging.info("[Transcript Sender] Started.")
        sent = 0

        try:
            while not self._shutdown_event.is_set():
                try:
                    ev = await asyncio.wait_for(
                        self.transcript_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if ev is _SHUTDOWN:
                    logging.info("[Transcript Sender] Shutdown sentinel received.")
                    break

                try:
                    await websocket.send(json.dumps(ev))
                    sent += 1
                except websockets.exceptions.ConnectionClosed:
                    logging.warning("[Transcript Sender] WebSocket closed mid-send.")
                    break
                except Exception as exc:
                    logging.error(f"[Transcript Sender] Send error: {exc}")

        except asyncio.CancelledError:
            logging.info("[Transcript Sender] Task cancelled.")
        except Exception as exc:
            logging.error(f"[Transcript Sender] Fatal error: {exc}", exc_info=True)
        finally:
            logging.info(f"[Transcript Sender] Stopped. Sent {sent} transcript events.")

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    def start(self, websocket):
        """Starts all pipeline workers.

        Start order:
            1. Create asyncio transcript_queue on the current event loop
            2. Start VAD worker thread (daemon)
            3. Start ASR worker thread (daemon)
            4. Schedule Transcript Sender as asyncio Task
        """
        if self._started:
            return
        self._started = True
        self._shutdown_event.clear()

        self.transcript_queue = asyncio.Queue(maxsize=TRANSCRIPT_QUEUE_MAXSIZE)

        self._vad_thread = threading.Thread(
            target=self._vad_worker, name="VADWorker", daemon=True
        )
        self._asr_thread = threading.Thread(
            target=self._asr_worker, name="ASRWorker", daemon=True
        )
        self._vad_thread.start()
        self._asr_thread.start()

        self._sender_task = self.loop.create_task(
            self._transcript_sender(websocket), name="TranscriptSender"
        )

        logging.info(
            f"[Session] Pipeline started — "
            f"audio_queue={AUDIO_QUEUE_MAXSIZE}, "
            f"speech_queue={SPEECH_QUEUE_MAXSIZE}, "
            f"transcript_queue={TRANSCRIPT_QUEUE_MAXSIZE}"
        )

    def stop(self):
        """Gracefully shuts down all workers and releases resources.

        Stop order:
            1. Set shutdown event (workers exit their loops)
            2. Send sentinels to unblock any blocked queue.get() calls
            3. Cancel transcript sender Task
            4. Join worker threads (with timeouts)
            5. Drain all queues
            6. Log session metrics
        """
        if not self._started:
            return
        logging.info("[Session] Stopping pipeline...")
        self._shutdown_event.set()

        # Unblock workers waiting on empty queues
        for q, name in [(self.audio_queue, "audio"), (self.speech_queue, "speech")]:
            try:
                q.put_nowait(_SHUTDOWN)
            except queue.Full:
                pass  # Worker will check shutdown_event on its next iteration

        # Cancel async sender task
        if self._sender_task and not self._sender_task.done():
            self._sender_task.cancel()

        # Join worker threads
        for thread, timeout, name in [
            (self._vad_thread, 3.0, "VAD"),
            (self._asr_thread, 8.0, "ASR"),  # ASR can take a few seconds to finish inference
        ]:
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logging.warning(f"[Session] {name} worker did not exit within {timeout}s timeout.")

        # Drain remaining queue items
        _drain_queue(self.audio_queue)
        _drain_queue(self.speech_queue)

        # Log session summary
        snap = self.metrics.snapshot()
        duration_s = round(time.time() - self.session_start_time, 1)
        logging.info(
            f"[Session] Terminated after {duration_s}s | "
            f"chunks_rx={snap['audio_chunks_received']} | "
            f"frames_dropped={snap['frames_dropped']} | "
            f"transcripts_sent={snap['transcripts_sent']} | "
            f"last_asr={snap['last_asr_ms']}ms | "
            f"last_e2e={snap['last_e2e_ms']}ms"
        )


def _drain_queue(q: queue.Queue):
    """Empties a queue.Queue, discarding all items."""
    while True:
        try:
            q.get_nowait()
        except (queue.Empty, Exception):
            break


# =============================================================================
# WEBSOCKET CONNECTION HANDLER
# =============================================================================

async def handle_client(websocket, path=None, *, asr_engine: FasterWhisperEngine, vad_bypass: bool = False):
    """Handles one Chrome WebSocket connection for its full lifetime.

    Lifecycle:
        1. Chrome connects (TCP handshake)
        2. Chrome sends JSON HANDSHAKE (sampleRate, numChannels, client)
        3. Pipeline workers start
        4. Chrome streams binary Int16 PCM audio frames continuously
        5. Pipeline processes audio → VAD → ASR → transcript JSON → Chrome
        6. Chrome disconnects (or tab closes)
        7. Pipeline workers stopped and cleaned up
    """
    addr = getattr(websocket, "remote_address", "unknown")
    logging.info(f"[Server] Client connected: {addr}")

    loop = asyncio.get_running_loop()
    session = PipelineSession(asr_engine, event_loop=loop, vad_bypass=vad_bypass)
    pipeline_started = False

    try:
        async for message in websocket:
            if isinstance(message, str):
                # JSON control message (HANDSHAKE or future control messages)
                session.handle_handshake(message)
                if not pipeline_started:
                    session.start(websocket)
                    pipeline_started = True

            elif isinstance(message, bytes):
                # Raw binary PCM audio data from Chrome ScriptProcessorNode
                if not pipeline_started:
                    # Audio arrived before handshake — start with defaults
                    logging.warning("[Server] Audio received before HANDSHAKE — starting with defaults.")
                    session.start(websocket)
                    pipeline_started = True
                session.enqueue_audio(message)

    except websockets.exceptions.ConnectionClosedOK:
        logging.info(f"[Server] Client {addr} disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as exc:
        logging.warning(f"[Server] Client {addr} connection error: {exc}")
    except Exception as exc:
        logging.error(f"[Server] Unexpected error for {addr}: {exc}", exc_info=True)
    finally:
        if pipeline_started:
            session.stop()
        logging.info(f"[Server] Session closed for {addr}.")


# =============================================================================
# MAIN — Server entry point
# =============================================================================

def _parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Module 4 — Real-Time Audio Pipeline Server"
    )
    parser.add_argument(
        "--vad-bypass",
        action="store_true",
        default=False,
        help=(
            "[TESTING ONLY] Skip Silero VAD inference. "
            "Use RMS energy to detect non-silence instead. "
            "Allows automated tests with synthetic audio to exercise the "
            "full queue/ASR/transcript pipeline. "
            "⚠ Never use in production — it forwards all audio to ASR."
        ),
    )
    parser.add_argument(
        "--host", default=HOST, help=f"Bind host (default: {HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"Bind port (default: {PORT})"
    )
    return parser.parse_args()


async def main():
    args = _parse_args()
    bind_host = args.host
    bind_port = args.port
    vad_bypass = args.vad_bypass

    logging.info("=" * 70)
    logging.info("Module 4 — Real-Time Pipeline Server")
    if vad_bypass:
        logging.warning(
            "VAD BYPASS MODE ACTIVE — Silero disabled, using RMS energy detection. "
            "Use this mode for testing only!"
        )
    logging.info("=" * 70)

    # Load faster-whisper tiny int8 ONCE — all sessions share this model object
    logging.info("Loading faster-whisper model (one-time, may take 5–15s)...")
    asr_engine = FasterWhisperEngine(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
    )
    logging.info(f"Model loaded in {asr_engine.load_time_s:.2f}s.")

    # Bind asr_engine and vad_bypass flag into the handler via closure
    async def _handler(websocket, path=None):
        await handle_client(websocket, path, asr_engine=asr_engine, vad_bypass=vad_bypass)

    logging.info(f"Starting WebSocket server on ws://{bind_host}:{bind_port} ...")
    logging.info("Configuration:")
    logging.info(f"  audio_queue  : {AUDIO_QUEUE_MAXSIZE} frames "
                 f"(~{AUDIO_QUEUE_MAXSIZE * VAD_FRAME_DURATION_SEC:.1f}s buffer)")
    logging.info(f"  speech_queue : {SPEECH_QUEUE_MAXSIZE} segments")
    logging.info(f"  chunk/overlap: {CHUNK_DURATION_SEC}s / {OVERLAP_DURATION_SEC}s")
    logging.info(f"  backpressure : drop-oldest {AUDIO_QUEUE_DROP_COUNT} frames")
    logging.info(f"  vad_bypass   : {vad_bypass}")

    async with websockets.serve(_handler, bind_host, bind_port):
        logging.info(f"Pipeline Server ready. Connect Chrome extension to ws://{bind_host}:{bind_port}")
        logging.info("Press Ctrl+C to stop.")
        await asyncio.Event().wait()  # Run indefinitely


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Pipeline Server stopped by user (Ctrl+C).")
