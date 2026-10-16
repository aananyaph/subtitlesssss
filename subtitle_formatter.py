"""Module 2 — Subtitle Formatter.

Pure data-to-text conversion. Knows nothing about Whisper, audio, or GUI.
Takes Module 1's output segments and produces SRT / VTT / TXT.

Input contract (from Module 1):
{
  "language": "hi",
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "..."},
    ...
  ]
}
"""

from pathlib import Path
from typing import List, Dict

# — Tunables matching spec: max 42 chars/line, 2 lines, ~7s duration —
MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CHARS_PER_SEGMENT = MAX_CHARS_PER_LINE * MAX_LINES  # 84
MAX_DURATION_SEC = 7.0


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds (float) to SRT timestamp HH:MM:SS,mmm.

    Handles hours >=1 correctly. Clamps negative to 0.
    """
    if seconds is None:
        raise ValueError("seconds_to_srt_time() received None")
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"seconds must be int/float, got {type(seconds).__name__}")
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(float(seconds) * 1000))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    ms = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    """Convert seconds (float) to VTT timestamp HH:MM:SS.mmm.

    Identical math to SRT but with '.' separator and WEBVTT requires same hours handling.
    """
    if seconds is None:
        raise ValueError("seconds_to_vtt_time() received None")
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"seconds must be int/float, got {type(seconds).__name__}")
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(float(seconds) * 1000))
    hours = total_ms // 3_600_000
    remainder = total_ms % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    ms = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _wrap_lines(text: str, max_chars: int = MAX_CHARS_PER_LINE, max_lines: int = MAX_LINES) -> str:
    """Wrap text to max_chars per line, max_lines lines, returning \\n-joined string."""
    text = text.strip()
    if len(text) <= max_chars or max_lines == 1:
        return text

    words = text.split()
    lines: List[str] = []
    current = ""

    for w in words:
        # If single word exceeds max_chars, hard-break it (rare for natural language)
        if len(w) > max_chars:
            if current:
                lines.append(current)
                current = ""
            # chunk long word
            for i in range(0, len(w), max_chars):
                lines.append(w[i : i + max_chars])
                if len(lines) >= max_lines:
                    break
            continue

        candidate = f"{current} {w}".strip() if current else w
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = w
            if len(lines) >= max_lines - 1:
                # Last line: append remainder as-is (will exceed slightly but keeps 2-line limit)
                # Collect remaining words into last line
                remaining = " ".join(words[words.index(w) + 1 :])
                if remaining:
                    current = f"{current} {remaining}".strip()
                break

    if current:
        lines.append(current)

    # Enforce max_lines: merge overflow into last line
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1 :])]

    return "\n".join(lines)


def _split_text_by_punct_and_length(text: str, max_len: int = MAX_CHARS_PER_SEGMENT) -> List[str]:
    """Split long text into chunks <= max_len, preferring punctuation boundaries."""
    text = text.strip()
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    remaining = text

    while len(remaining) > max_len:
        window = remaining[:max_len]

        # Prefer punctuation splits in the latter half of the window
        split_pos = -1
        # Ordered preference: sentence boundaries first
        for punct in [". ", "? ", "! ", "; ", ": ", ", ", " "]:
            idx = window.rfind(punct)
            if idx != -1 and idx >= max_len * 0.4:
                # Include the punctuation char, exclude trailing space
                if punct.strip() in {".", "?", "!", ";", ":", ","}:
                    split_pos = idx + 1  # keep punctuation
                else:
                    split_pos = idx  # space — split before it
                break

        if split_pos == -1 or split_pos == 0:
            # Fallback: last space
            idx = window.rfind(" ")
            split_pos = idx if idx != -1 else max_len

        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].strip()
        if not remaining:
            break

    if remaining:
        chunks.append(remaining)

    return chunks


def split_long_segment(segment: Dict) -> List[Dict]:
    """Split a single Whisper segment if it exceeds duration or char limits.

    Returns list of segments with proportionally split timestamps.
    Whisper sometimes emits 10+ second chunks; display needs ~7s and 84 chars max.
    """
    if not isinstance(segment, dict):
        raise TypeError(f"segment must be dict, got {type(segment).__name__}")
    if "start" not in segment or "end" not in segment or "text" not in segment:
        raise ValueError(f"segment missing required keys 'start'/'end'/'text': {segment}")

    start = float(segment["start"])
    end = float(segment["end"])
    text = str(segment["text"]).strip()

    if end < start:
        raise ValueError(f"segment end {end} < start {start}")

    duration = end - start

    needs_split = (
        duration > MAX_DURATION_SEC
        or len(text) > MAX_CHARS_PER_SEGMENT
        or len(_wrap_lines(text).split("\n")) > MAX_LINES
    )

    if not needs_split or not text:
        return [{"start": start, "end": end, "text": text}]

    text_chunks = _split_text_by_punct_and_length(text, MAX_CHARS_PER_SEGMENT)

    # If no effective split (e.g. single long word), return as-is wrapped
    if len(text_chunks) <= 1:
        return [{"start": start, "end": end, "text": text}]

    # Allocate duration proportionally by character length (simple, readable)
    total_chars = sum(len(c) for c in text_chunks)
    result: List[Dict] = []
    cursor = start

    for i, chunk in enumerate(text_chunks):
        # Last chunk snaps to original end to avoid floating drift
        if i == len(text_chunks) - 1:
            chunk_end = end
        else:
            ratio = len(chunk) / total_chars if total_chars else 1 / len(text_chunks)
            chunk_dur = duration * ratio
            # Clamp to at least 1.0s for readability
            chunk_dur = max(1.0, chunk_dur)
            chunk_end = cursor + chunk_dur
            # Prevent overflow
            remaining_min = (len(text_chunks) - i - 1) * 1.0
            if chunk_end > end - remaining_min:
                chunk_end = end - remaining_min

        result.append({"start": round(cursor, 3), "end": round(chunk_end, 3), "text": chunk})
        cursor = chunk_end

    return result


def _normalize_segments(segments: List[Dict]) -> List[Dict]:
    """Expand all segments via split_long_segment and validate."""
    if segments is None:
        raise ValueError("segments must not be None")
    if not isinstance(segments, list):
        raise TypeError(f"segments must be list, got {type(segments).__name__}")

    normalized: List[Dict] = []
    for seg in segments:
        normalized.extend(split_long_segment(seg))
    return normalized


def format_srt(segments: List[Dict]) -> str:
    """Format segments to SRT string.

    Uses HH:MM:SS,mmm --> HH:MM:SS,mmm, index starting at 1.
    Long segments are auto-split; text is wrapped to 42 chars/line, max 2 lines.
    """
    normalized = _normalize_segments(segments)
    if not normalized:
        return ""

    lines: List[str] = []
    for idx, seg in enumerate(normalized, start=1):
        start_s = seconds_to_srt_time(float(seg["start"]))
        end_s = seconds_to_srt_time(float(seg["end"]))
        text = _wrap_lines(str(seg["text"]).strip())
        lines.append(f"{idx}\n{start_s} --> {end_s}\n{text}\n")

    return "\n".join(lines).strip() + "\n"


def format_vtt(segments: List[Dict]) -> str:
    """Format segments to WebVTT string.

    Header WEBVTT, timestamps HH:MM:SS.mmm --> HH:MM:SS.mmm.
    """
    normalized = _normalize_segments(segments)
    if not normalized:
        return "WEBVTT\n"

    lines: List[str] = ["WEBVTT", ""]
    for seg in normalized:
        start_s = seconds_to_vtt_time(float(seg["start"]))
        end_s = seconds_to_vtt_time(float(seg["end"]))
        text = _wrap_lines(str(seg["text"]).strip())
        lines.append(f"{start_s} --> {end_s}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def format_txt(segments: List[Dict]) -> str:
    """Format segments to plain text — no timestamps, just readable text."""
    normalized = _normalize_segments(segments)
    if not normalized:
        return ""
    # Join each segment's text on new line (preserves sentence boundaries)
    texts = [str(seg["text"]).strip() for seg in normalized if str(seg["text"]).strip()]
    return "\n".join(texts) + "\n" if texts else ""


def save_subtitle(segments: List[Dict], output_path: str, format: str = None) -> str:
    """Save segments to file, dispatching by format or file extension.

    Args:
        segments: List of {"start": float, "end": float, "text": str}.
        output_path: Destination file path (e.g. "out.srt").
        format: Optional explicit format: "srt" | "vtt" | "txt" (with or without dot).
                If None, inferred from output_path extension.

    Returns:
        The output_path string written.

    Raises:
        ValueError for unknown format or invalid inputs.
    """
    if not output_path or not str(output_path).strip():
        raise ValueError("output_path must be a non-empty string")
    if segments is None:
        raise ValueError("segments must not be None")

    # Resolve format
    fmt = format.lower().lstrip(".") if format else None
    if fmt is None:
        ext = Path(output_path).suffix.lower().lstrip(".")
        fmt = ext

    if fmt not in {"srt", "vtt", "txt"}:
        raise ValueError(
            f"Unknown subtitle format '{format or Path(output_path).suffix}'. "
            f"Expected 'srt', 'vtt', or 'txt'."
        )

    if fmt == "srt":
        content = format_srt(segments)
    elif fmt == "vtt":
        content = format_vtt(segments)
    else:
        content = format_txt(segments)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return str(out)
