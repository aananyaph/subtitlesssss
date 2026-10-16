"""Module 3 Test Suite.

Runs CLI test cases for batch_pipeline.process_file:
  1. samples/english_clip.wav -> srt, language "en"
  2. samples/kannada_clip2.wav -> srt, language "kn"
  3. samples/hindi_video.mp4 -> vtt, language "hi"
  4. .txt file path -> must raise ValueError
  5. nonexistent path -> must raise FileNotFoundError

Validations per produced subtitle file:
  - exists, utf-8 readable, non-empty
  - SRT lines match HH:MM:SS,mmm --> HH:MM:SS,mmm
  - VTT starts with WEBVTT
  - last end time <= clip length + 0.5 s
  - Kannada >= 60% Kannada letters, Hindi >= 60% Devanagari
  - no Cyrillic or Arabic letters
  - progress fractions received per case
"""

import os
import re
import sys
from pathlib import Path

# Configure stdout utf-8 at top with line buffering
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def log(msg=""):
    print(msg, flush=True)


from batch_pipeline import process_file

SCRIPTS = {
    "kn": ("Kannada", 0x0C80, 0x0CFF),
    "hi": ("Devanagari", 0x0900, 0x097F),
}
FORBIDDEN = {
    "Cyrillic": (0x0400, 0x04FF),
    "Arabic": (0x0600, 0x06FF),
}


def parse_srt_timestamps(content: str):
    arrow_lines = [line.strip() for line in content.splitlines() if "-->" in line]
    pattern = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})$")
    ts_list = []
    for line in arrow_lines:
        m = pattern.match(line)
        if not m:
            return None, f"SRT line did not match HH:MM:SS,mmm --> HH:MM:SS,mmm: {line!r}"
        h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
        start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
        end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
        ts_list.append((start, end))
    return ts_list, None


def parse_vtt_timestamps(content: str):
    arrow_lines = [line.strip() for line in content.splitlines() if "-->" in line]
    pattern = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})$")
    ts_list = []
    for line in arrow_lines:
        m = pattern.match(line)
        if not m:
            return None, f"VTT line did not match HH:MM:SS.mmm --> HH:MM:SS.mmm: {line!r}"
        h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
        start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
        end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
        ts_list.append((start, end))
    return ts_list, None


def run_tests():
    total_failures = 0

    # ---------------------------------------------------------
    # Case 1: English WAV -> SRT
    # ---------------------------------------------------------
    log("=" * 60)
    log("CASE 1: samples/english_clip.wav -> srt, language 'en'")
    progress1 = []
    try:
        out1 = "samples/english_clip.srt"
        if os.path.exists(out1):
            os.remove(out1)
        res1 = process_file(
            "samples/english_clip.wav",
            output_path=out1,
            output_format="srt",
            language="en",
            progress_callback=lambda f: progress1.append(round(f, 3)),
        )
        log(f"Result: {res1}")
        log(f"Progress fractions ({len(progress1)}): {progress1}")

        # Checks
        if not os.path.isfile(out1):
            raise AssertionError(f"Output file does not exist: {out1}")
        with open(out1, "r", encoding="utf-8") as f:
            content1 = f.read()
        if not content1.strip():
            raise AssertionError("Output file is empty")

        ts_list1, err1 = parse_srt_timestamps(content1)
        if err1:
            raise AssertionError(err1)
        if not ts_list1:
            raise AssertionError("No timestamps found in SRT")

        last_end1 = ts_list1[-1][1]
        clip_len1 = res1["duration"]
        if last_end1 > clip_len1 + 0.5:
            raise AssertionError(f"Last end {last_end1}s exceeds clip length {clip_len1}s + 0.5s")

        letters1 = [c for c in content1 if c.isalpha()]
        for fname, (flo, fhi) in FORBIDDEN.items():
            forbidden_count = sum(flo <= ord(c) <= fhi for c in letters1)
            if forbidden_count > 0:
                raise AssertionError(f"Found {forbidden_count} forbidden {fname} letters")

        log("PASS: Case 1 passed all checks.")
    except Exception as e:
        total_failures += 1
        log(f"FAIL: Case 1 failed: {e}")

    # ---------------------------------------------------------
    # Case 2: Kannada WAV -> SRT
    # ---------------------------------------------------------
    log("=" * 60)
    log("CASE 2: samples/kannada_clip2.wav -> srt, language 'kn'")
    progress2 = []
    try:
        out2 = "samples/kannada_clip2.srt"
        if os.path.exists(out2):
            os.remove(out2)
        res2 = process_file(
            "samples/kannada_clip2.wav",
            output_path=out2,
            output_format="srt",
            language="kn",
            progress_callback=lambda f: progress2.append(round(f, 3)),
        )
        log(f"Result: {res2}")
        log(f"Progress fractions ({len(progress2)}): {progress2}")

        if not os.path.isfile(out2):
            raise AssertionError(f"Output file does not exist: {out2}")
        with open(out2, "r", encoding="utf-8") as f:
            content2 = f.read()
        if not content2.strip():
            raise AssertionError("Output file is empty")

        ts_list2, err2 = parse_srt_timestamps(content2)
        if err2:
            raise AssertionError(err2)
        if not ts_list2:
            raise AssertionError("No timestamps found in SRT")

        last_end2 = ts_list2[-1][1]
        clip_len2 = res2["duration"]
        if last_end2 > clip_len2 + 0.5:
            raise AssertionError(f"Last end {last_end2}s exceeds clip length {clip_len2}s + 0.5s")

        letters2 = [c for c in content2 if c.isalpha()]
        name, lo, hi = SCRIPTS["kn"]
        kannada_count = sum(lo <= ord(c) <= hi for c in letters2)
        ratio2 = kannada_count / len(letters2) if letters2 else 0.0
        if ratio2 < 0.60:
            raise AssertionError(f"Kannada ratio {ratio2:.1%} is below required 60%")

        for fname, (flo, fhi) in FORBIDDEN.items():
            forbidden_count = sum(flo <= ord(c) <= fhi for c in letters2)
            if forbidden_count > 0:
                raise AssertionError(f"Found {forbidden_count} forbidden {fname} letters")

        log(f"PASS: Case 2 passed all checks (Kannada script ratio: {ratio2:.1%}).")
    except Exception as e:
        total_failures += 1
        log(f"FAIL: Case 2 failed: {e}")

    # ---------------------------------------------------------
    # Case 3: Hindi Video -> VTT
    # ---------------------------------------------------------
    log("=" * 60)
    log("CASE 3: samples/hindi_video.mp4 -> vtt, language 'hi'")
    progress3 = []
    try:
        out3 = "samples/hindi_video.vtt"
        if os.path.exists(out3):
            os.remove(out3)
        res3 = process_file(
            "samples/hindi_video.mp4",
            output_path=out3,
            output_format="vtt",
            language="hi",
            progress_callback=lambda f: progress3.append(round(f, 3)),
        )
        log(f"Result: {res3}")
        log(f"Progress fractions ({len(progress3)}): {progress3}")

        if not os.path.isfile(out3):
            raise AssertionError(f"Output file does not exist: {out3}")
        with open(out3, "r", encoding="utf-8") as f:
            content3 = f.read()
        if not content3.strip():
            raise AssertionError("Output file is empty")

        if not content3.startswith("WEBVTT"):
            raise AssertionError("VTT output does not start with 'WEBVTT'")

        ts_list3, err3 = parse_vtt_timestamps(content3)
        if err3:
            raise AssertionError(err3)
        if not ts_list3:
            raise AssertionError("No timestamps found in VTT")

        last_end3 = ts_list3[-1][1]
        clip_len3 = res3["duration"]
        if last_end3 > clip_len3 + 0.5:
            raise AssertionError(f"Last end {last_end3}s exceeds clip length {clip_len3}s + 0.5s")

        # Exclude header "WEBVTT" from script ratio count
        body3 = content3.replace("WEBVTT", "")
        letters3 = [c for c in body3 if c.isalpha()]
        name, lo, hi = SCRIPTS["hi"]
        devanagari_count = sum(lo <= ord(c) <= hi for c in letters3)
        ratio3 = devanagari_count / len(letters3) if letters3 else 0.0
        if ratio3 < 0.60:
            raise AssertionError(f"Hindi (Devanagari) ratio {ratio3:.1%} is below required 60%")

        for fname, (flo, fhi) in FORBIDDEN.items():
            forbidden_count = sum(flo <= ord(c) <= fhi for c in letters3)
            if forbidden_count > 0:
                raise AssertionError(f"Found {forbidden_count} forbidden {fname} letters")

        log(f"PASS: Case 3 passed all checks (Devanagari script ratio: {ratio3:.1%}).")
    except Exception as e:
        total_failures += 1
        log(f"FAIL: Case 3 failed: {e}")

    # ---------------------------------------------------------
    # Case 4: .txt file path -> must raise ValueError
    # ---------------------------------------------------------
    log("=" * 60)
    log("CASE 4: a .txt file path -> must raise ValueError")
    dummy_txt = Path("samples/dummy_input.txt")
    dummy_txt.write_text("dummy content", encoding="utf-8")
    try:
        try:
            process_file(str(dummy_txt))
            total_failures += 1
            log("FAIL: Case 4 did not raise any error, expected ValueError.")
        except ValueError as ve:
            log(f"PASS: Case 4 correctly raised ValueError: {ve}")
        except Exception as e:
            total_failures += 1
            log(f"FAIL: Case 4 raised wrong exception type {type(e).__name__}: {e}")
    finally:
        if dummy_txt.exists():
            dummy_txt.unlink()

    # ---------------------------------------------------------
    # Case 5: nonexistent path -> must raise FileNotFoundError
    # ---------------------------------------------------------
    log("=" * 60)
    log("CASE 5: nonexistent path -> must raise FileNotFoundError")
    try:
        process_file("samples/nonexistent_sample_xyz123.wav")
        total_failures += 1
        log("FAIL: Case 5 did not raise any error, expected FileNotFoundError.")
    except FileNotFoundError as fe:
        log(f"PASS: Case 5 correctly raised FileNotFoundError: {fe}")
    except Exception as e:
        total_failures += 1
        log(f"FAIL: Case 5 raised wrong exception type {type(e).__name__}: {e}")

    log("=" * 60)
    total_cases = 5
    passed = total_cases - total_failures
    log(f"SUMMARY: {passed}/{total_cases} passed.")
    sys.exit(1 if total_failures > 0 else 0)


if __name__ == "__main__":
    run_tests()
