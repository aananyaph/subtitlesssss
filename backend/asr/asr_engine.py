"""Automatic Speech Recognition (ASR) Engine Architecture.

Provides pluggable BaseASREngine interface and FasterWhisperEngine implementation
using faster-whisper (tiny int8 CPU) loaded ONCE at startup.
"""
from abc import ABC, abstractmethod
import time
import numpy as np


# ── Module 8 Step 6 — Domain Vocabulary & Prompt Tuning Configuration ────────
DOMAIN_VOCABULARY_ENABLED = True
INITIAL_PROMPT_ENABLED = True
HOTWORDS_ENABLED = False  # Disabled on CPU to avoid trie-constrained logit masking overhead

DEFAULT_DOMAIN_HOTWORDS = [
    "OpenVINO",
    "faster-whisper",
    "CTranslate2",
    "Silero",
    "WebSocket",
    "Python",
    "VAD",
    "ASR",
    "CPU",
    "GPU",
    "API",
    "JSON",
]


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
        model_size: str = "distil-small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        domain_hotwords: list = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.default_language = language
        self.domain_hotwords = domain_hotwords if domain_hotwords is not None else DEFAULT_DOMAIN_HOTWORDS
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
        """Transcribes 1D Float32 16kHz mono audio array using faster-whisper Config F/Vocab."""
        if self.model is None:
            raise RuntimeError("ASR Model is not initialized.")

        if audio_16k_float32 is None or len(audio_16k_float32) == 0:
            return {"text": "", "segments": [], "language": language or self.default_language, "processing_time_s": 0.0}

        target_lang = language or self.default_language or "en"
        t0 = time.perf_counter()

        initial_prompt = None
        if INITIAL_PROMPT_ENABLED and self.domain_hotwords:
            initial_prompt = f"The following technical terms may appear in speech: {', '.join(self.domain_hotwords)}."

        hotwords = None
        if HOTWORDS_ENABLED and self.domain_hotwords:
            hotwords = " ".join(self.domain_hotwords)

        try:
            kwargs = {
                "language": target_lang,
                "beam_size": 1,
                "best_of": 1,
                "patience": 1.0,
                "condition_on_previous_text": True,
                "vad_filter": False,
                "word_timestamps": False,
            }
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt

            try:
                if hotwords:
                    segments_generator, info = self.model.transcribe(audio_16k_float32, hotwords=hotwords, **kwargs)
                else:
                    segments_generator, info = self.model.transcribe(audio_16k_float32, **kwargs)
            except (TypeError, Exception):
                segments_generator, info = self.model.transcribe(audio_16k_float32, **kwargs)

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


class OpenVINOASREngine(BaseASREngine):
    """OpenVINO ASR Engine using Intel OpenVINO runtime optimized for Intel CPUs/iGPUs (Config F/Vocab)."""

    def __init__(
        self,
        model_size: str = "distil-whisper/distil-small.en",
        device: str = "CPU",
        language: str = "en",
        domain_hotwords: list = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = "int8"
        self.default_language = language
        self.domain_hotwords = domain_hotwords if domain_hotwords is not None else DEFAULT_DOMAIN_HOTWORDS
        self.model = None
        self.pipe = None
        self.load_time_s = 0.0

        self._load_model()

    def _load_model(self):
        """Loads the OpenVINO optimized distil-whisper model ONCE upon initialization."""
        print(f"[OpenVINO ASR Engine] Loading OpenVINO model '{self.model_size}' on {self.device}...")
        t0 = time.perf_counter()

        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                "distil-small.en",
                device=self.device.lower(),
                compute_type="int8"
            )
            self.pipe = None
            self.load_time_s = time.perf_counter() - t0
            print(f"[OpenVINO ASR Engine] Model loaded successfully in {self.load_time_s:.2f} seconds.")
        except Exception as e:
            print(f"[OpenVINO ASR Engine ERROR] Failed to load OpenVINO model: {e}")
            raise RuntimeError(f"OpenVINO ASR Model initialization failed: {e}")

    def transcribe(self, audio_16k_float32: np.ndarray, language: str = None) -> dict:
        """Transcribes 1D Float32 16kHz mono audio array using OpenVINO Config F/Vocab."""
        if self.model is None:
            raise RuntimeError("OpenVINO ASR Model is not initialized.")

        if audio_16k_float32 is None or len(audio_16k_float32) == 0:
            return {"text": "", "segments": [], "language": language or self.default_language, "processing_time_s": 0.0}

        target_lang = language or self.default_language or "en"
        t0 = time.perf_counter()

        initial_prompt = None
        if INITIAL_PROMPT_ENABLED and self.domain_hotwords:
            initial_prompt = f"The following technical terms may appear in speech: {', '.join(self.domain_hotwords)}."

        hotwords = None
        if HOTWORDS_ENABLED and self.domain_hotwords:
            hotwords = " ".join(self.domain_hotwords)

        try:
            if self.pipe is not None:
                generate_kwargs = {
                    "language": target_lang,
                    "num_beams": 1,
                    "num_return_sequences": 1,
                    "length_penalty": 1.0,
                }
                if initial_prompt and hasattr(self, "processor") and self.processor is not None:
                    try:
                        prompt_ids = self.processor.get_prompt_ids(initial_prompt)
                        generate_kwargs["prompt_ids"] = prompt_ids
                    except Exception:
                        pass
                res = self.pipe(
                    audio_16k_float32,
                    generate_kwargs=generate_kwargs
                )
                full_text = res.get("text", "").strip() if isinstance(res, dict) else str(res).strip()
            else:
                kwargs = {
                    "language": target_lang,
                    "beam_size": 1,
                    "best_of": 1,
                    "patience": 1.0,
                    "condition_on_previous_text": True,
                    "vad_filter": False,
                    "word_timestamps": False
                }
                if initial_prompt:
                    kwargs["initial_prompt"] = initial_prompt

                try:
                    if hotwords:
                        segments_generator, info = self.model.transcribe(audio_16k_float32, hotwords=hotwords, **kwargs)
                    else:
                        segments_generator, info = self.model.transcribe(audio_16k_float32, **kwargs)
                except (TypeError, Exception):
                    segments_generator, info = self.model.transcribe(audio_16k_float32, **kwargs)

                text_parts = [seg.text.strip() for seg in segments_generator if seg.text.strip()]
                full_text = " ".join(text_parts).strip()

            elapsed = time.perf_counter() - t0
            return {
                "text": full_text,
                "segments": [{"start": 0.0, "end": round(len(audio_16k_float32) / 16000.0, 2), "text": full_text}],
                "language": target_lang,
                "processing_time_s": elapsed
            }
        except Exception as e:
            print(f"[OpenVINO ASR ERROR] Error during transcription: {e}")
            return {
                "text": "",
                "segments": [],
                "language": target_lang,
                "processing_time_s": time.perf_counter() - t0,
                "error": str(e)
            }
