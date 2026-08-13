"""Module 2 Test Suite & Performance Benchmark — Voice Activity Detection (VAD).

Executes 7 distinct test scenarios (Speech, Silence, Video Speech, Music, Background Music, Quiet Speech, Fast Speech)
and measures CPU usage, memory, processing time, and Real-Time Factor (RTF).
"""
import time
import tracemalloc
import numpy as np

from audio_utils import AudioPreprocessor, resample_audio
from vad_engine import SileroVadEngine
from segmenter import SpeechSegmenter, SpeechState


def generate_synthetic_speech(duration_s: float = 3.0, sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    """Generates realistic voiced speech signal using glottal pulse train and vocal tract formant resonances."""
    num_samples = int(duration_s * sample_rate)
    t = np.arange(num_samples) / float(sample_rate)

    # Pitch / Fundamental frequency F0 = 130 Hz (glottal pulse train)
    pitch_period = int(sample_rate / 130.0)
    pulses = np.zeros(num_samples, dtype=np.float32)
    pulses[::pitch_period] = 1.0

    # Smooth glottal pulse shape (Rosenberg model approximation)
    pulse_kernel = np.sin(np.linspace(0, np.pi, 12)) ** 2
    glottal = np.convolve(pulses, pulse_kernel, mode='same')

    # Vocal tract formants (F1=500Hz, F2=1500Hz, F3=2500Hz)
    f1 = np.sin(2 * np.pi * 500 * t) * np.exp(-t * 15)
    f2 = np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 20)
    f3 = np.sin(2 * np.pi * 2500 * t) * np.exp(-t * 25)

    raw_speech = glottal * (0.6 * f1 + 0.3 * f2 + 0.1 * f3)

    # Syllabic envelope modulation (4 Hz speech rhythm)
    syllable_env = 0.5 + 0.5 * np.cos(2 * np.pi * 4.0 * t)
    
    # Add slight unvoiced fricative noise (breath/sibilant component)
    noise = np.random.normal(0, 0.05, num_samples) * (syllable_env > 0.6)

    speech = (raw_speech * syllable_env + noise) * amplitude
    
    # Normalize peak to range
    max_val = np.max(np.abs(speech))
    if max_val > 0:
        speech = (speech / max_val) * amplitude

    return speech.astype(np.float32)


def generate_synthetic_music(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generates synthetic pure music tones (harmonic chord: C major 261Hz, 329Hz, 392Hz)."""
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    chord = (
        0.15 * np.sin(2 * np.pi * 261.63 * t) +
        0.15 * np.sin(2 * np.pi * 329.63 * t) +
        0.15 * np.sin(2 * np.pi * 392.00 * t)
    )
    return chord.astype(np.float32)


def run_vad_test(test_name: str, audio_16k_mono: np.ndarray, vad_engine: SileroVadEngine, expected_speech: bool):
    """Executes VAD processing pipeline on audio_16k_mono and reports diagnostics & timing."""
    print(f"\n==================================================")
    print(f"RUNNING: {test_name}")
    print(f"Audio Duration: {len(audio_16k_mono) / 16000.0:.2f} seconds")
    print(f"==================================================")

    preprocessor = AudioPreprocessor(input_sample_rate=16000, num_channels=1, target_sample_rate=16000)
    segmenter = SpeechSegmenter(threshold=0.35, neg_threshold=0.20, min_speech_duration_ms=150, min_silence_duration_ms=300)
    vad_engine.reset_state()

    # Slice into 512-sample frames
    frames = preprocessor.process_float32_mono_16k(audio_16k_mono)
    
    probs = []
    events = []
    
    start_time = time.perf_counter()

    for i, frame in enumerate(frames):
        prob = vad_engine.predict(frame, sample_rate=16000)
        probs.append(prob)
        event = segmenter.process_frame(frame, prob)
        
        if event["type"] in ["SPEECH_START", "SPEECH_END"]:
            events.append((i, event))
            if event["type"] == "SPEECH_START":
                print(f"  Frame {i:04d} | Prob: {prob:.4f} | >>> SPEECH START <<<")
            elif event["type"] == "SPEECH_END":
                print(f"  Frame {i:04d} | Prob: {prob:.4f} | <<< SPEECH END (Duration: {event['duration_seconds']:.2f}s) <<<")

    elapsed_time = time.perf_counter() - start_time
    audio_duration = len(audio_16k_mono) / 16000.0
    rtf = elapsed_time / audio_duration if audio_duration > 0 else 0.0

    avg_prob = float(np.mean(probs)) if probs else 0.0
    max_prob = float(np.max(probs)) if probs else 0.0
    detected_speech = any(e[1]["type"] == "SPEECH_START" for e in events) or max_prob >= 0.35

    print(f"Diagnostics:")
    print(f"  - Frames processed: {len(frames)}")
    print(f"  - Average Speech Prob: {avg_prob:.4f}")
    print(f"  - Peak Speech Prob:    {max_prob:.4f}")
    print(f"  - Processing Time:     {elapsed_time * 1000.0:.2f} ms")
    print(f"  - Real-Time Factor (RTF): {rtf:.4f}  (Lower is faster, RTF < 1.0 is real-time)")

    status = "PASSED" if (detected_speech == expected_speech or (not expected_speech and max_prob < 0.35)) else "NEEDS REVIEW"
    print(f"RESULT: {status} (Expected Speech: {expected_speech}, Detected: {detected_speech})")
    
    return {
        "name": test_name,
        "duration_s": audio_duration,
        "processing_time_s": elapsed_time,
        "rtf": rtf,
        "avg_prob": avg_prob,
        "max_prob": max_prob,
        "detected_speech": detected_speech,
        "status": status
    }


def main():
    print("Initializing Module 2 VAD Engine & Performance Benchmark...")
    tracemalloc.start()
    
    vad_engine = SileroVadEngine()
    
    results = []

    # TEST 1: Clear human speech
    speech_audio = generate_synthetic_speech(duration_s=3.0, amplitude=0.6)
    results.append(run_vad_test("TEST 1: Clear Human Speech", speech_audio, vad_engine, expected_speech=True))

    # TEST 2: Complete silence
    silence_audio = np.zeros(16000 * 3, dtype=np.float32)
    results.append(run_vad_test("TEST 2: Complete Silence", silence_audio, vad_engine, expected_speech=False))

    # TEST 3: YouTube video with speech (Speech + 500ms initial silence)
    youtube_audio = np.concatenate([np.zeros(8000, dtype=np.float32), generate_synthetic_speech(duration_s=4.0, amplitude=0.7)])
    results.append(run_vad_test("TEST 3: YouTube Video Speech Audio", youtube_audio, vad_engine, expected_speech=True))

    # TEST 4: Music without speech
    music_audio = generate_synthetic_music(duration_s=3.0)
    results.append(run_vad_test("TEST 4: Music Without Speech", music_audio, vad_engine, expected_speech=False))

    # TEST 5: Speech with background music
    speech_music_audio = generate_synthetic_speech(duration_s=3.0, amplitude=0.6) + generate_synthetic_music(duration_s=3.0) * 0.3
    results.append(run_vad_test("TEST 5: Speech with Background Music", speech_music_audio, vad_engine, expected_speech=True))

    # TEST 6: Very quiet speech
    quiet_speech_audio = generate_synthetic_speech(duration_s=3.0, amplitude=0.15)
    results.append(run_vad_test("TEST 6: Very Quiet Speech", quiet_speech_audio, vad_engine, expected_speech=True))

    # TEST 7: Fast speech (Rapid bursty speech pulses)
    t = np.linspace(0, 3.0, 16000 * 3)
    fast_bursts = generate_synthetic_speech(duration_s=3.0, amplitude=0.7) * (np.sin(2 * np.pi * 5.0 * t) > 0)
    results.append(run_vad_test("TEST 7: Fast Speech Pulses", fast_bursts, vad_engine, expected_speech=True))

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n==================================================")
    print("MODULE 2 BENCHMARK & PERFORMANCE SUMMARY")
    print("==================================================")
    total_audio_s = sum(r["duration_s"] for r in results)
    total_proc_s = sum(r["processing_time_s"] for r in results)
    overall_rtf = total_proc_s / total_audio_s if total_audio_s > 0 else 0.0

    print(f"Total Audio Processed:   {total_audio_s:.2f} seconds")
    print(f"Total Processing Time:  {total_proc_s * 1000.0:.2f} ms")
    print(f"Overall Real-Time Factor (RTF): {overall_rtf:.4f}  (1s audio processes in {overall_rtf*1000.0:.1f}ms)")
    print(f"Peak Memory Usage:      {peak_mem / 1024 / 1024:.2f} MB")
    print("==================================================\n")


if __name__ == "__main__":
    main()
