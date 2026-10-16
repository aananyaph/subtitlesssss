"""Make Subs — Command-line interface and entrypoint for batch subtitle generation.

Wraps batch_pipeline.process_file for convenient file-to-subtitle processing.
"""

import sys
from pathlib import Path
from batch_pipeline import (
    ALLOWED_INPUT_EXTENSIONS,
    ALLOWED_OUTPUT_FORMATS,
    extract_audio,
    get_duration,
    process_file,
)

__all__ = [
    "ALLOWED_INPUT_EXTENSIONS",
    "ALLOWED_OUTPUT_FORMATS",
    "get_duration",
    "extract_audio",
    "process_file",
    "main",
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python make_subs.py <input_file> [output_path] [format] [language] [model_size] [device]")
        print("Example: python make_subs.py video.mp4 output.srt srt en small cpu")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    output_format = sys.argv[3] if len(sys.argv) > 3 else "srt"
    language = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "auto" else None
    model_size = sys.argv[5] if len(sys.argv) > 5 else "small"
    device = sys.argv[6] if len(sys.argv) > 6 else "cpu"

    def progress_callback(fraction: float):
        percent = int(fraction * 100)
        sys.stdout.write(f"\rProgress: {percent}% [{fraction:.2f}]")
        sys.stdout.flush()

    print(f"Processing: {input_path}")
    result = process_file(
        input_path=input_path,
        output_path=output_path,
        output_format=output_format,
        model_size=model_size,
        language=language,
        device=device,
        progress_callback=progress_callback,
    )
    print(f"\nDone! Output saved to: {result['output_path']}")
    print(f"Language: {result['language']} | Segments: {result['segment_count']} | Duration: {result['duration']:.2f}s")


if __name__ == "__main__":
    main()
