"""Indic ASR backend: AI4Bharat IndicConformer-600M-Multi.

Single source of truth for ALL Indic code. core_engine.py imports this lazily,
so Whisper-only users never need torch/transformers.

Proven inference call on this project's machine:
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    text = model(wav, "kn", "ctc")   # wav: float32 tensor (1, N), 16 kHz mono

The model returns one text string per call and no timestamps. Timestamps come
from VAD: speech regions (max ~15 s each) are transcribed one by one, and the
region boundaries become segment start/end.

Output contract (same as core_engine):
    {"language": "kn", "segments": [{"start": 0.0, "end": 3.2, "text": "..."}]}
"""

import wave
from pathlib import Path

import numpy as np

__all__ = ["load_indic_model", "transcribe_indic"]

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
SAMPLE_RATE = 16000
MAX_CHUNK_SECONDS = 15.0
MIN_CHUNK_SECONDS = 0.3

_MODEL = None


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def load_indic_model():
    """Load (or return cached) IndicConformer model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "Indic backend needs: pip install transformers torch torchaudio "
            "onnxruntime onnx"
        ) from e

    try:
        _MODEL = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    except OSError as e:
        text = str(e)
        if "gated" in text.lower() or "401" in text or "403" in text:
            raise RuntimeError(
                f"Model {MODEL_ID} is gated. Accept the terms at "
                f"https://huggingface.co/{MODEL_ID} and set HF_TOKEN "
                "(or run once with network access after logging in)."
            ) from e
        raise RuntimeError(f"Could not load {MODEL_ID}: {e}") from e
    return _MODEL


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
def _load_audio_16k_mono(source):
    """Return float32 numpy array, 16 kHz mono. Accepts wav path or ndarray.

    Non-wav files (mp3, mp4, ...): decoded via faster-whisper's ffmpeg-free
    decoder (PyAV), which already ships with faster-whisper.
    """
    if isinstance(source, np.ndarray):
        return source.astype(np.float32, copy=False).reshape(-1)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                sr = w.getframerate()
                channels = w.getnchannels()
                width = w.getsampwidth()
                raw = w.readframes(w.getnframes())
        except wave.Error:
            width = None  # unusual wav encoding: fall through to decoder
        else:
            if width == 2:
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                data = data.reshape(-1, channels).mean(axis=1)
                if sr != SAMPLE_RATE:
                    data = _resample(data, sr)
                return data.astype(np.float32, copy=False)

    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=SAMPLE_RATE)


def _resample(data, orig_sr):
    import torch
    import torchaudio

    wav = torch.from_numpy(data).unsqueeze(0)
    wav = torchaudio.functional.resample(wav, orig_sr, SAMPLE_RATE)
    return wav.squeeze(0).numpy()


# --------------------------------------------------------------------------
# Chunking (numpy only, testable without torch)
# --------------------------------------------------------------------------
def _vad_chunks(audio):
    """Return list of (start_sample, end_sample) speech regions, each <= 15 s."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        min_silence_duration_ms=500,
        max_speech_duration_s=MAX_CHUNK_SECONDS,
        speech_pad_ms=200,
    )
    try:
        regions = get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
    except Exception as e:
        raise RuntimeError(f"VAD failed: {e}") from e

    chunks = []
    for r in regions:
        start, end = int(r["start"]), int(r["end"])
        if (end - start) / SAMPLE_RATE >= MIN_CHUNK_SECONDS:
            chunks.append((start, end))
    return chunks


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------
def transcribe_indic(source, language="kn", decoding="ctc", progress_callback=None):
    """Transcribe with IndicConformer. Returns the standard segments dict."""
    if decoding not in ("ctc", "rnnt"):
        raise ValueError("decoding must be 'ctc' or 'rnnt'")

    audio = _load_audio_16k_mono(source)
    if audio.size == 0:
        return {"language": language, "segments": []}

    chunks = _vad_chunks(audio)
    if not chunks:
        return {"language": language, "segments": []}

    import torch

    model = load_indic_model()
    segments = []
    total_chunks = len(chunks)
    for idx, (start, end) in enumerate(chunks, 1):
        wav = torch.from_numpy(audio[start:end]).unsqueeze(0)
        try:
            with torch.no_grad():
                text = model(wav, language, decoding)
        except Exception as e:
            raise RuntimeError(
                f"Indic model failed on chunk {start / SAMPLE_RATE:.1f}s-"
                f"{end / SAMPLE_RATE:.1f}s: {e}"
            ) from e

        text = (text if isinstance(text, str) else str(text)).strip()
        if text:
            segments.append(
                {
                    "start": round(start / SAMPLE_RATE, 3),
                    "end": round(end / SAMPLE_RATE, 3),
                    "text": text,
                }
            )
        if progress_callback is not None:
            progress_callback(float(idx) / float(total_chunks))

    return {"language": language, "segments": segments}