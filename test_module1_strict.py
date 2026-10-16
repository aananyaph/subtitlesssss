"""Strict Module 1 test: fails on garbage, not just on wrong dict shape.

Per clip it checks:
  1. Output contract: language, segments with float start < end, in order.
  2. Last segment ends within clip length + 0.5 s (catches hallucinated timestamps).
  3. At least 60% of letters belong to the expected script.
  4. No Cyrillic or Arabic-script letters unless that script is expected.

Run:
  $env:PYTHONUTF8="1"; $env:HF_HUB_OFFLINE="1"; python test_module1_strict.py
"""

import sys
import time
import wave

SCRIPTS = {
    "kn": ("Kannada", 0x0C80, 0x0CFF),
    "hi": ("Devanagari", 0x0900, 0x097F),
    "en": ("Latin", 0x0041, 0x024F),
}
FORBIDDEN = {
    "Cyrillic": (0x0400, 0x04FF),
    "Arabic": (0x0600, 0x06FF),
}

CASES = [
    # (clip, language, extra kwargs)
    ("samples/kannada_clip2.wav", "kn", {}),
    ("samples/hindi_clip.wav", "hi", {}),
    ("samples/english_clip.wav", "en", {"backend": "whisper", "model_size": "base"}),
]

MIN_SCRIPT_RATIO = 0.60
DURATION_SLACK = 0.5


def clip_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def in_range(ch, lo, hi):
    return lo <= ord(ch) <= hi


def check_result(result, duration, lang):
    """Return a list of failure strings (empty list = pass)."""
    problems = []
    name, lo, hi = SCRIPTS[lang]

    if not isinstance(result, dict) or "segments" not in result or "language" not in result:
        return ["result is not {'language', 'segments'}"]
    segs = result["segments"]
    if not segs:
        return ["no segments returned"]

    prev_end = 0.0
    for i, s in enumerate(segs):
        for key in ("start", "end", "text"):
            if key not in s:
                problems.append(f"segment {i} missing '{key}'")
        if "start" in s and "end" in s:
            if not s["start"] < s["end"]:
                problems.append(f"segment {i}: start >= end")
            if s["start"] < prev_end - 1e-6:
                problems.append(f"segment {i}: overlaps previous")
            prev_end = max(prev_end, s["end"])

    if segs[-1]["end"] > duration + DURATION_SLACK:
        problems.append(
            f"last end {segs[-1]['end']:.2f}s exceeds clip length {duration:.2f}s"
        )

    text = " ".join(s.get("text", "") for s in segs)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        problems.append("no letters in output")
        return problems

    ratio = sum(in_range(c, lo, hi) for c in letters) / len(letters)
    if ratio < MIN_SCRIPT_RATIO:
        problems.append(f"only {ratio:.0%} {name} letters (need {MIN_SCRIPT_RATIO:.0%})")

    for fname, (flo, fhi) in FORBIDDEN.items():
        n = sum(in_range(c, flo, fhi) for c in letters)
        if n:
            problems.append(f"{n} {fname} letters in output")

    return problems


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from core_engine import transcribe

    failed = 0
    for clip, lang, kwargs in CASES:
        print("=" * 60)
        print(f"{clip}  lang={lang}  {kwargs}")
        duration = clip_duration(clip)
        t0 = time.time()
        result = transcribe(clip, language=lang, **kwargs)
        elapsed = time.time() - t0
        print(f"time {elapsed:.1f}s | clip {duration:.1f}s | segments {len(result['segments'])}")
        for s in result["segments"]:
            print(f"  [{s['start']:.1f}-{s['end']:.1f}] {s['text']}")

        problems = check_result(result, duration, lang)
        if problems:
            failed += 1
            print("FAIL:")
            for p in problems:
                print("  -", p)
        else:
            print("PASS")

    print("=" * 60)
    print(f"{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
