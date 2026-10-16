"""Module 3 — Batch Pipeline.

Handles single-file end-to-end subtitle generation:
File in -> audio extraction (ffmpeg) -> ASR transcription (core_engine) -> subtitle formatting (subtitle_formatter).
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional

import core_engine
import subtitle_formatter

__all__ = [
    "ALLOWED_INPUT_EXTENSIONS",
    "ALLOWED_OUTPUT_FORMATS",
    "get_duration",
    "extract_audio",
    "process_file",
]

ALLOWED_INPUT_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
}

ALLOWED_OUTPUT_FORMATS = {"srt", "vtt", "txt"}


def _check_tools() -> None:
    """Ensure ffmpeg and ffprobe are available on PATH."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg executable not found on PATH. Please install ffmpeg.")
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe executable not found on PATH. Please install ffmpeg/ffprobe.")


def get_duration(path: str) -> float:
    """Return media duration in seconds via ffprobe."""
    _check_tools()
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0 or not res.stdout.strip():
        stderr_lines = [line.strip() for line in res.stderr.splitlines() if line.strip()]
        last_line = stderr_lines[-1] if stderr_lines else "Unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed to get duration: {last_line}")

    try:
        return float(res.stdout.strip())
    except ValueError as e:
        raise RuntimeError(f"Could not parse duration '{res.stdout.strip()}' as float: {e}") from e


def _has_audio_stream(input_path: str) -> bool:
    """Check if file contains at least one audio stream via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        return False
    return bool(res.stdout.strip())


def extract_audio(input_path: str, wav_path: str) -> None:
    """Extract audio from input media to 16 kHz mono PCM wav via ffmpeg.

    Used for every input (audio or video) to guarantee downstream gets uniform format.
    """
    _check_tools()
    in_p = Path(input_path)
    if not in_p.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not _has_audio_stream(str(in_p)):
        raise RuntimeError(f"File has no audio stream: {input_path}")

    out_p = Path(wav_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_p),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_p),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        stderr_lines = [line.strip() for line in res.stderr.splitlines() if line.strip()]
        last_line = stderr_lines[-1] if stderr_lines else "Unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg audio extraction failed: {last_line}")

    if not out_p.is_file() or out_p.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed to produce output audio file: {wav_path}")


def process_file(
    input_path: str,
    output_path: Optional[str] = None,
    output_format: str = "srt",
    model_size: str = "small",
    language: Optional[str] = None,
    device: str = "cpu",
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict:
    """Process a single audio or video file end-to-end to generate subtitles.

    Steps:
      1. Validate input path, extension, and output format.
      2. Extract 16 kHz mono PCM wav into a guaranteed temporary directory.
      3. Transcribe with core_engine.transcribe.
      4. Format and save subtitles via subtitle_formatter.save_subtitle.
      5. Report 1.0 progress callback on completion.

    Returns:
      {"output_path": str, "language": str, "segment_count": int, "duration": float}
    """
    _check_tools()

    # Step 1: Validate input existence
    in_file = Path(input_path)
    if not in_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Validate input extension
    ext = in_file.suffix.lower()
    if ext not in ALLOWED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed extensions: {sorted(ALLOWED_INPUT_EXTENSIONS)}"
        )

    # Validate output format
    fmt = output_format.lower().lstrip(".") if output_format else ""
    if fmt not in ALLOWED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format '{output_format}'. "
            f"Allowed formats: {sorted(ALLOWED_OUTPUT_FORMATS)}"
        )

    # Determine output path
    if output_path is None:
        target_output = str(in_file.with_suffix(f".{fmt}"))
    else:
        target_output = str(output_path)

    # Get media duration
    duration = get_duration(str(in_file))

    # Step 2: Extract to temporary wav with guaranteed cleanup
    tmp_dir = tempfile.TemporaryDirectory()
    try:
        temp_wav = Path(tmp_dir.name) / "extracted_16k_mono.wav"
        extract_audio(str(in_file), str(temp_wav))

        # Step 3: Transcribe via core_engine
        transcription = core_engine.transcribe(
            str(temp_wav),
            language=language,
            model_size=model_size,
            device=device,
            progress_callback=progress_callback,
        )

        # Step 4: Save subtitles via subtitle_formatter
        segments = transcription.get("segments", [])
        saved_output_path = subtitle_formatter.save_subtitle(
            segments,
            target_output,
            format=fmt,
        )
    finally:
        try:
            tmp_dir.cleanup()
        except Exception:
            pass

    # Forward 1.0 at completion
    if progress_callback is not None:
        progress_callback(1.0)

    return {
        "output_path": saved_output_path,
        "language": transcription.get("language", language or ""),
        "segment_count": len(segments),
        "duration": duration,
    }
