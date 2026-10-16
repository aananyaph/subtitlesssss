"""Standalone test for Module 2 — Subtitle Formatter.

Hardcoded fake segments -> calls format_srt/vtt/txt -> prints to terminal
-> saves .srt/.vtt/.txt to test_output/. No argparse, no GUI.
"""

from pathlib import Path
from subtitle_formatter import format_srt, format_vtt, format_txt, save_subtitle

# Hardcoded fake segments list covering:
# 1. Normal-length segment
# 2. One segment >7s to test splitting (duration + char length)
# 3. One timestamp crossing 1hr boundary
FAKE_SEGMENTS = [
    {"start": 0.0, "end": 3.2, "text": "Hello world, this is a test of the subtitle formatter."},
    {
        "start": 3.2,
        "end": 14.5,
        "text": (
            "This second segment is intentionally very long so it will exceed the forty-two "
            "characters per line and seven seconds duration rule and must be split correctly "
            "into multiple readable subtitle cues with proper timing."
        ),
    },
    {"start": 3661.0, "end": 3665.5, "text": "Hour boundary check — this tests HH:MM:SS handling beyond one hour."},
]

print("=== SRT ===")
srt_text = format_srt(FAKE_SEGMENTS)
print(srt_text)

print("=== VTT ===")
vtt_text = format_vtt(FAKE_SEGMENTS)
print(vtt_text)

print("=== TXT ===")
txt_text = format_txt(FAKE_SEGMENTS)
print(txt_text)

out_dir = Path("test_output")
out_dir.mkdir(parents=True, exist_ok=True)

srt_path = save_subtitle(FAKE_SEGMENTS, str(out_dir / "test_output.srt"), format="srt")
vtt_path = save_subtitle(FAKE_SEGMENTS, str(out_dir / "test_output.vtt"), format="vtt")
txt_path = save_subtitle(FAKE_SEGMENTS, str(out_dir / "test_output.txt"), format="txt")

print(f"Saved: {srt_path}")
print(f"Saved: {vtt_path}")
print(f"Saved: {txt_path}")
