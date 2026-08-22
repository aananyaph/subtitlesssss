import os
import time
import wave
import numpy as np
from faster_whisper import WhisperModel

# 1. Load project TTS speech fixture (~24.75s)
wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_assets", "speech_48k_stereo16.wav")

if os.path.exists(wav_path):
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        frames = w.readframes(w.getnframes())
        audio_int16 = np.frombuffer(frames, dtype=np.int16)
        if nch == 2:
            audio_int16 = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        audio = audio_float[::3] if sr == 48000 else audio_float
else:
    audio = np.zeros(16000 * 25, dtype=np.float32)

print(f"Loaded audio clip: {len(audio)/16000.0:.2f}s ({len(audio)} samples)")
print("Initializing WhisperModel('distil-small.en', device='cpu', compute_type='int8', cpu_threads=4)...")
model = WhisperModel("distil-small.en", device="cpu", compute_type="int8", cpu_threads=4)

test_lengths_sec = [0.5, 1, 2, 3, 5]

print("\n--- Short Segment Latency Benchmark ---")
for dur in test_lengths_sec:
    clip = audio[:int(dur * 16000)]
    t0 = time.perf_counter()
    segments, info = model.transcribe(clip, beam_size=1, best_of=1)
    text = " ".join(s.text for s in segments)
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"{dur:3.1f}s clip: {elapsed:6.1f}ms | text: {text}")
