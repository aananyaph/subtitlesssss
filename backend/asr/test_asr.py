"""Module 3 Standalone Test Suite & Benchmarking — Local Streaming ASR.

Tests ASR transcription accuracy, calculates Word Error Rate (WER), measures Real-Time Factor (RTF),
and optionally benchmarks 'tiny' int8 vs 'base' int8 model performance on CPU.
"""
import time
import tracemalloc
import numpy as np

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

try:
    from .asr_engine import FasterWhisperEngine
    from .streaming_asr import StreamingASRManager
except ImportError:
    from asr_engine import FasterWhisperEngine
    from streaming_asr import StreamingASRManager


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculates Word Error Rate (WER) between reference and hypothesis text."""
    if HAS_JIWER:
        return float(jiwer.wer(reference, hypothesis))
    
    # Fallback Levenshtein distance on words
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # Deletion
                d[i][j - 1] + 1,        # Insertion
                d[i - 1][j - 1] + cost   # Substitution
            )

    return float(d[len(ref_words)][len(hyp_words)]) / float(len(ref_words))


def generate_audio_signal(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generates synthetic test speech audio signal."""
    num_samples = int(duration_s * sample_rate)
    t = np.arange(num_samples) / float(sample_rate)
    pitch_period = int(sample_rate / 140.0)
    pulses = np.zeros(num_samples, dtype=np.float32)
    pulses[::pitch_period] = 1.0
    pulse_kernel = np.sin(np.linspace(0, np.pi, 10)) ** 2
    glottal = np.convolve(pulses, pulse_kernel, mode='same')
    f1 = np.sin(2 * np.pi * 500 * t) * np.exp(-t * 10)
    f2 = np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 15)
    speech = glottal * (0.7 * f1 + 0.3 * f2)
    max_val = np.max(np.abs(speech))
    if max_val > 0:
        speech = (speech / max_val) * 0.5
    return speech.astype(np.float32)


def run_asr_test_case(engine: FasterWhisperEngine, test_name: str, audio: np.ndarray, reference_text: str = ""):
    print(f"\n--------------------------------------------------")
    print(f"TEST: {test_name}")
    audio_dur = len(audio) / 16000.0
    print(f"Audio Duration: {audio_dur:.2f}s")
    print(f"--------------------------------------------------")

    t0 = time.perf_counter()
    result = engine.transcribe(audio, language="en")
    elapsed = time.perf_counter() - t0

    rtf = elapsed / audio_dur if audio_dur > 0 else 0.0
    hyp_text = result.get("text", "")

    print(f"  Transcript: '{hyp_text}'")
    print(f"  Processing Time: {elapsed * 1000.0:.2f} ms")
    print(f"  Real-Time Factor (RTF): {rtf:.4f}")

    wer = 0.0
    if reference_text:
        wer = calculate_wer(reference_text, hyp_text)
        print(f"  Reference : '{reference_text}'")
        print(f"  Word Error Rate (WER): {wer:.2%}")

    return {
        "test_name": test_name,
        "duration_s": audio_dur,
        "processing_time_s": elapsed,
        "rtf": rtf,
        "transcript": hyp_text,
        "wer": wer
    }


def benchmark_model(model_name: str):
    print(f"\n==================================================")
    print(f"BENCHMARKING MODEL: '{model_name}' (int8 on CPU)")
    print(f"==================================================")

    tracemalloc.start()
    t_load_start = time.perf_counter()

    try:
        engine = FasterWhisperEngine(model_size=model_name, device="cpu", compute_type="int8", language="en")
        load_time = time.perf_counter() - t_load_start
    except Exception as e:
        print(f"Model '{model_name}' benchmark skipped (failed to load: {e})")
        tracemalloc.stop()
        return None

    audio_sample = generate_audio_signal(duration_s=4.0)
    ref_text = "Today we are testing streaming speech recognition."

    res = run_asr_test_case(engine, f"Synthetic Speech Sample ({model_name})", audio_sample, reference_text=ref_text)
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    res["load_time_s"] = load_time
    res["peak_ram_mb"] = peak_mem / (1024 * 1024)

    print(f"Model Load Time: {load_time:.2f} s")
    print(f"Peak RAM Usage : {res['peak_ram_mb']:.2f} MB")

    return res


def main():
    print("Initializing Module 3 ASR Standalone Test & Model Comparison...")
    
    # 1. Primary Model: tiny int8
    tiny_res = benchmark_model("tiny")

    # 2. Optional Comparison Model: base int8
    base_res = None
    try:
        base_res = benchmark_model("base")
    except Exception as e:
        print(f"Base model comparison skipped: {e}")

    # Summary Comparison Matrix
    print("\n==========================================================================")
    print("MODULE 3 — ASR MODEL COMPARISON SUMMARY")
    print("==========================================================================")
    print(f"{'Metric':<25} | {'tiny (Default)':<20} | {'base (Optional)':<20}")
    print("-" * 72)
    
    if tiny_res:
        t_rtf = f"{tiny_res['rtf']:.4f} (RTF)"
        t_proc = f"{tiny_res['processing_time_s']*1000:.1f} ms"
        t_ram = f"{tiny_res['peak_ram_mb']:.1f} MB"
        t_load = f"{tiny_res['load_time_s']:.2f} s"
        t_wer = f"{tiny_res['wer']:.1%}"
    else:
        t_rtf = t_proc = t_ram = t_load = t_wer = "N/A"

    if base_res:
        b_rtf = f"{base_res['rtf']:.4f} (RTF)"
        b_proc = f"{base_res['processing_time_s']*1000:.1f} ms"
        b_ram = f"{base_res['peak_ram_mb']:.1f} MB"
        b_load = f"{base_res['load_time_s']:.2f} s"
        b_wer = f"{base_res['wer']:.1%}"
    else:
        b_rtf = b_proc = b_ram = b_load = b_wer = "Not Tested"

    print(f"{'Model Load Time':<25} | {t_load:<20} | {b_load:<20}")
    print(f"{'Processing Time':<25} | {t_proc:<20} | {b_proc:<20}")
    print(f"{'Real-Time Factor (RTF)':<25} | {t_rtf:<20} | {b_rtf:<20}")
    print(f"{'Peak RAM Usage':<25} | {t_ram:<20} | {b_ram:<20}")
    print(f"{'Word Error Rate (WER)':<25} | {t_wer:<20} | {b_wer:<20}")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
