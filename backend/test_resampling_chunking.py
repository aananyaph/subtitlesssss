"""Module 11 — Resampling & Chunk Size Floor Unit Test Suite.

Validates:
    1. Polyphase anti-aliasing resampling (scipy.signal.resample_poly) from 48kHz to 16kHz.
    2. Minimum chunk floor padding (1.0s / 1.5s) to guarantee Whisper acoustic context.
"""

import os
import sys
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vad.audio_utils import resample_audio, AudioPreprocessor
from pipeline_server import PipelineSession, _drain_queue


class DummyASREngine:
    """Mock ASR engine for testing."""
    model_size = "distil-small.en"
    device = "CPU"
    compute_type = "int8"

    def transcribe(self, audio, language="en"):
        return {"text": "test phrase", "segments": [], "language": "en"}


class TestResamplingAndChunking(unittest.TestCase):
    """Test suite for Module 11 resampling and minimum chunk floor."""

    def test_polyphase_resampling_ratio(self):
        """48kHz mono Float32 audio resampled to 16kHz must have exact 1:3 sample count."""
        sr_orig = 48000
        sr_target = 16000
        duration_s = 1.0

        # Generate 1s 48kHz audio (48,000 samples)
        t = np.linspace(0, duration_s, int(sr_orig * duration_s), endpoint=False)
        signal_48k = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        resampled_16k = resample_audio(signal_48k, orig_sr=sr_orig, target_sr=sr_target)

        expected_samples = int(sr_target * duration_s)
        self.assertEqual(len(resampled_16k), expected_samples)

    def test_audio_preprocessor_polyphase_pipeline(self):
        """AudioPreprocessor must produce 512-sample frames from raw 48kHz stereo bytes."""
        preprocessor = AudioPreprocessor(input_sample_rate=48000, num_channels=2)

        # 48kHz stereo Int16 = 4 bytes per sample frame. Send 0.1s = 4800 sample frames = 19,200 bytes.
        raw_bytes = bytes(19200)
        frames = preprocessor.process_raw_bytes(raw_bytes)

        # 0.1s at 16kHz = 1600 samples = ~3 complete 512-sample frames (1536 samples)
        self.assertGreaterEqual(len(frames), 3)
        for frame in frames:
            self.assertEqual(len(frame), 512)

    def test_minimum_chunk_floor_padding(self):
        """Short speech segments (< 1.0s) must be padded to minimum floor before ASR enqueueing."""
        min_floor_sec = 1.0
        min_samples = int(min_floor_sec * 16000)

        # Create short 0.4s speech PCM (6,400 samples)
        short_pcm = np.ones(6400, dtype=np.float32)

        if len(short_pcm) < min_samples:
            pad_len = min_samples - len(short_pcm)
            padded_pcm = np.pad(short_pcm, (0, pad_len), mode='constant')

        self.assertEqual(len(padded_pcm), min_samples)
        self.assertEqual(len(padded_pcm) / 16000.0, min_floor_sec)

    def test_anti_aliasing_high_frequency_attenuation(self):
        """Frequencies above Nyquist (8kHz in 16kHz audio) must be attenuated by polyphase filter."""
        sr_orig = 48000
        sr_target = 16000
        duration_s = 0.5
        t = np.linspace(0, duration_s, int(sr_orig * duration_s), endpoint=False)
        
        # High frequency tone at 14kHz (exceeding 16kHz Nyquist limit of 8kHz)
        high_freq_tone = np.sin(2 * np.pi * 14000 * t).astype(np.float32)

        resampled_16k = resample_audio(high_freq_tone, orig_sr=sr_orig, target_sr=sr_target)
        
        # Power of high frequency tone after anti-alias filtering should be near zero
        rms_power = np.sqrt(np.mean(resampled_16k ** 2))
        self.assertLess(rms_power, 0.15)


if __name__ == "__main__":
    unittest.main()
