"""Module 1 - Core Transcription Engine.

Input: audio file path or 16 kHz mono float32 numpy array.
Output contract (all downstream modules depend on it):
    {"language": "hi", "segments": [{"start": 0.0, "end": 3.2, "text": "..."}]}

Backends:
    whisper : faster-whisper (default for most languages)
    indic   : AI4Bharat IndicConformer (Kannada etc.), lives in indic_backend.py

No formatting, no UI, no progress bars. Errors are raised, never swallowed.
"""

from pathlib import Path

try:
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
except ImportError as e:
    raise ImportError(
        "faster-whisper is not installed. Install with: pip install faster-whisper"
    ) from e

__all__ = [
    "INDIC_LANGS",
    "load_model",
    "transcribe",
    "detect_language",
]

_VALID_SIZES = {"tiny", "base", "small", "medium", "large"}
_VALID_DEVICES = {"cpu", "cuda"}
_VALID_BACKENDS = {"auto", "whisper", "indic"}

# Languages routed to the Indic backend when backend="auto".
INDIC_LANGS = {"kn", "hi"}

_SAMPLE_RATE = 16000
_DETECT_SECONDS = 30

_MODEL_CACHE = {}
_INDIC_BACKEND = None


# --------------------------------------------------------------------------
# Indic backend (lazy: Whisper-only users never import torch/transformers)
# --------------------------------------------------------------------------
def _get_indic_backend():
    global _INDIC_BACKEND
    if _INDIC_BACKEND is not None:
        return _INDIC_BACKEND
    import importlib
    import sys

    # Ensure indic_backend is imported with its full namespace
    if "indic_backend" not in sys.modules:
        import indic_backend  # noqa: F401  # ensure module-level imports run
    else:
        indic_backend = sys.modules["indic_backend"]

    # Reload to ensure fresh state (especially numpy namespace)
    indic_backend = importlib.reload(indic_backend)

    fn = indic_backend.transcribe_indic
    load_fn = indic_backend.load_indic_model

    # Cache the function (not the module, to avoid reload issues)
    _INDIC_BACKEND = fn
    return _INDIC_BACKEND


# --------------------------------------------------------------------------
# Whisper model
# --------------------------------------------------------------------------
def _check_size_device(size, device):
    if size not in _VALID_SIZES:
        raise ValueError(
            f"Invalid model size {size!r}. Choose one of: {sorted(_VALID_SIZES)}"
        )
    if device not in _VALID_DEVICES:
        raise ValueError(
            f"Invalid device {device!r}. Choose one of: {sorted(_VALID_DEVICES)}"
        )


def load_model(size="small", device="cpu", download_root=None):
    """Load (or return cached) multilingual faster-whisper model."""
    _check_size_device(size, device)
    key = (size, device, download_root)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    compute_type = "int8" if device == "cpu" else "float16"
    try:
        model = WhisperModel(
            size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load Whisper model {size!r} on {device!r}: {e}"
        ) from e

    _MODEL_CACHE[key] = model
    return model


def _prepare_source(source):
    """Return something faster-whisper accepts: str path or float32 ndarray."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return str(path)
    # numpy array (live streaming path): pass through unchanged
    return source


def _audio_array(source):
    """Decode a path to a 16 kHz mono float32 array; pass arrays through."""
    if isinstance(source, str):
        return decode_audio(source, sampling_rate=_SAMPLE_RATE)
    return source


# --------------------------------------------------------------------------
# Language detection
# --------------------------------------------------------------------------
def detect_language(audio_sample, model_size="tiny", device="cpu"):
    """Detect spoken language from the first ~30 s of audio.

    Returns (language_code, confidence). audio_sample: path or numpy array.
    """
    source = _prepare_source(audio_sample)
    audio = _audio_array(source)
    audio = audio[: _SAMPLE_RATE * _DETECT_SECONDS]
    model = load_model(model_size, device)
    try:
        lang, prob, _ = model.detect_language(audio)
    except Exception as e:
        raise RuntimeError(f"Language detection failed: {e}") from e
    return lang, float(prob)


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------
def _transcribe_whisper(
    source, language, model_size, device, beam_size, vad_filter, progress_callback=None
):
    model = load_model(model_size, device)
    try:
        segments, info = model.transcribe(
            source,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        total_duration = float(info.duration) if info.duration else 0.0
        seg_list = []
        for s in segments:
            if s.text.strip():
                seg_list.append(
                    {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
                )
            if progress_callback is not None and total_duration > 0:
                progress_callback(min(1.0, float(s.end) / total_duration))
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}") from e
    return {"language": info.language, "segments": seg_list}


def _transcribe_indic(source, language, progress_callback=None):
    try:
        fn = _get_indic_backend()
        return fn(source, language=language, progress_callback=progress_callback)
    except ImportError as e:
        raise RuntimeError(
            "Indic backend needs torch, transformers and onnxruntime. "
            f"Install them first. Detail: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Indic transcription failed: {e}") from e


def transcribe(
    source,
    language=None,
    model_size="small",
    device="cpu",
    beam_size=5,
    vad_filter=True,
    backend="auto",
    progress_callback=None,
):
    """Transcribe audio. Returns {"language": str, "segments": [...]}.

    source   : file path (str/Path) or 16 kHz mono float32 numpy array
    language : None = auto-detect, else ISO code ("en", "hi", "kn")
    backend  : "auto" routes by language, or force "whisper" / "indic"
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend {backend!r}. Choose one of: {sorted(_VALID_BACKENDS)}"
        )
    _check_size_device(model_size, device)
    source = _prepare_source(source)

    if backend == "indic":
        if language is None:
            raise ValueError("backend='indic' needs an explicit language, e.g. 'kn'.")
        return _transcribe_indic(source, language, progress_callback=progress_callback)

    if backend == "whisper":
        return _transcribe_whisper(
            source,
            language,
            model_size,
            device,
            beam_size,
            vad_filter,
            progress_callback=progress_callback,
        )

    # backend == "auto"
    if language is None:
        language, _conf = detect_language(source, model_size=model_size, device=device)
    if language in INDIC_LANGS:
        return _transcribe_indic(source, language, progress_callback=progress_callback)
    return _transcribe_whisper(
        source,
        language,
        model_size,
        device,
        beam_size,
        vad_filter,
        progress_callback=progress_callback,
    )