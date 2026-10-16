"""Standalone CLI test for Module 1 — Core Transcription Engine.

Hardcoded file path -> calls transcribe() -> prints returned dict.
No GUI, no argument parsing — proves core_engine.py works in isolation.
"""

from core_engine import transcribe

# Hardcoded test file — change this path to test other files
AUDIO_FILE = "diag_assets/speech_48k_stereo16.wav"

result = transcribe(
    AUDIO_FILE,
    language=None,        # auto-detect
    model_size="small",
    device="cpu",
    beam_size=5,
    vad_filter=True,
)

# Print the returned segments dict to terminal (exact structure)
print(result)
