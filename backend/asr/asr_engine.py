"""Automatic Speech Recognition (ASR) Engine Architecture.

Provides pluggable BaseASREngine interface and FasterWhisperEngine implementation
using faster-whisper (tiny int8 CPU) loaded ONCE at startup.
"""
from abc import ABC, abstractmethod
import time
import numpy as np


class BaseASREngine(ABC):
    """Abstract Base Class for ASR Engine implementations (Pluggable Architecture)."""

    @abstractmethod
    def transcribe(self, audio_16k_float32: np.ndarray, language: str = "en") -> dict:
        """Transcribe 1D Float32 mono audio array at 16000 Hz.
        
        Returns:
            dict containing 'text', 'segments', 'language', and 'processing_time_s'.
        """
        pass


class FasterWhisperEngine(BaseASREngine):
    """faster-whisper ASR Engine using CTranslate2 C++ backend on CPU (int8 quantized)."""

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en"
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.default_language = language
        self.model = None
        self.load_time_s = 0.0

        self._load_model()

    def _load_model(self):
        """Loads the faster-whisper model ONCE upon initialization."""
        print(f"[ASR Engine] Loading faster-whisper model '{self.model_size}' ({self.compute_type} on {self.device})...")
        t0 = time.perf_counter()

        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self.load_time_s = time.perf_counter() - t0
            print(f"[ASR Engine] Model '{self.model_size}' loaded successfully in {self.load_time_s:.2f} seconds.")
        except Exception as e:
            print(f"[ASR Engine ERROR] Failed to load faster-whisper model '{self.model_size}': {e}")
            raise RuntimeError(f"ASR Model initialization failed: {e}")

    def transcribe(self, audio_16k_float32: np.ndarray, language: str = None) -> dict:
        """Transcribes 1D Float32 16kHz mono audio array using faster-whisper.
        
        Settings chosen specifically for low-latency CPU streaming:
          - beam_size=1: Fast greedy decoding
          - vad_filter=False: Module 2 Silero VAD already handles VAD
          - condition_on_previous_text=False: Prevents hallucination loops on context windows
        """
        if self.model is None:
            raise RuntimeError("ASR Model is not initialized.")

        if audio_16k_float32 is None or len(audio_16k_float32) == 0:
            return {"text": "", "segments": [], "language": language or self.default_language, "processing_time_s": 0.0}

        target_lang = language or self.default_language or "en"
        t0 = time.perf_counter()

        try:
            segments_generator, info = self.model.transcribe(
                audio_16k_float32,
                language=target_lang,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                word_timestamps=False,
                initial_prompt="English transcript.",
                no_speech_threshold=0.6
            )

            segments = []
            text_parts = []

            for seg in segments_generator:
                cleaned_text = seg.text.strip()
                if cleaned_text:
                    segments.append({
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": cleaned_text,
                        "probability": round(seg.avg_logprob, 3)
                    })
                    text_parts.append(cleaned_text)

            full_text = " ".join(text_parts).strip()
            elapsed = time.perf_counter() - t0

            return {
                "text": full_text,
                "segments": segments,
                "language": info.language if hasattr(info, "language") else target_lang,
                "processing_time_s": elapsed
            }

        except Exception as e:
            print(f"[ASR Engine ERROR] Error during transcription: {e}")
            return {
                "text": "",
                "segments": [],
                "language": target_lang,
                "processing_time_s": time.perf_counter() - t0,
                "error": str(e)
            }
