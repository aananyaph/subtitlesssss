"""Module 1 - Capture system audio (WASAPI loopback) to WAV.

Non-blocking callback capture with a hard watchdog (os._exit) so it never
hangs when the audio engine is idle (known pyaudiowpatch pitfall).

Usage:
    python -u capture.py                     # 10s of whatever is playing -> capture.wav
    python -u capture.py -d 5 -o out.wav
    python -u capture.py --list              # list loopback devices and exit
    python -u capture.py --selftest          # play 1kHz tone + capture it back via loopback
"""
import argparse
import os
import sys
import threading
import time
from collections import deque

import pyaudiowpatch as pyaudio

import numpy as np
import soundfile as sf

CHUNK = 1024
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2  # bytes per int16 sample


def log(msg):
    print(msg, flush=True)


def find_loopback_device(p):
    """Return the default WASAPI loopback device info dict."""
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        log("ERROR: WASAPI host API not available.")
        sys.exit(1)

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("hostApi") == wasapi_info["index"] and info.get("isLoopbackDevice"):
            return info

    log("ERROR: No WASAPI loopback device found.")
    sys.exit(1)


def list_loopback_devices(p):
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        log("ERROR: WASAPI host API not available.")
        return
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("hostApi") == wasapi_info["index"] and info.get("isLoopbackDevice"):
            log(f"[{i}] {info.get('name')} | {info.get('maxInputChannels')}ch | {info.get('defaultSampleRate')} Hz")


def watchdog(limit_s):
    """Hard-kill the process if a capture does not finish in time."""
    time.sleep(limit_s)
    log(f"\nERROR: watchdog fired after {limit_s}s. Loopback likely blocked "
        f"(is the audio engine idle?). Aborting.")
    os._exit(2)


def wait_for_capture(frames, rate, duration_s, timeout_s):
    """Sleep until the callback has captured enough audio, or watchdog kills us."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        with frames["lock"]:
            elapsed = len(frames["buf"]) * CHUNK / rate
        if elapsed >= duration_s:
            return True
        time.sleep(0.05)
    return False


def capture(duration_s, out_path, device_index=None):
    p = pyaudio.PyAudio()
    device = p.get_device_info_by_index(device_index) if device_index is not None else find_loopback_device(p)

    rate = int(device.get("defaultSampleRate"))
    channels = int(device.get("maxInputChannels")) or 2

    log(f"Capturing {duration_s}s from: {device['name']}")
    log(f"  rate={rate} Hz, channels={channels}")

    state = {"buf": deque(), "peak": 0.0, "lock": threading.Lock()}

    def callback(in_data, frame_count, time_info, status_flags):
        if in_data is not None:
            audio = np.frombuffer(in_data, dtype=np.int16)
            with state["lock"]:
                state["buf"].append(in_data)
                state["peak"] = max(state["peak"], float(np.abs(audio).max()))
                meter = int((state["peak"] / 32768.0) * 50)
                sys.stdout.write(f"\r  peak={state['peak'] / 32768.0:6.1%}  {'#' * meter}")
                sys.stdout.flush()
        return (None, pyaudio.paContinue)

    # Watchdog must be armed BEFORE any open() call (open can block on idle engine).
    threading.Thread(target=watchdog, args=(duration_s + 20,), daemon=True).start()

    stream = p.open(
        format=FORMAT,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device["index"],
        stream_callback=callback,
    )

    stream.start_stream()
    ok = wait_for_capture(state, rate, duration_s, timeout_s=duration_s + 20)
    stream.stop_stream()
    stream.close()
    p.terminate()
    log("")

    with state["lock"]:
        raw = b"".join(state["buf"])
        peak = state["peak"]

    if not ok or not raw:
        log("ERROR: No audio captured.")
        sys.exit(1)

    audio = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
    sf.write(out_path, audio, rate, subtype="PCM_16")
    log(f"Saved {audio.shape[0] / rate:.1f}s -> {out_path}")
    log(f"Peak level: {peak / 32768.0:.1%}")

    if peak == 0:
        log("WARNING: pure silence captured. Play some audio and retry.")
    else:
        log("OK: captured real audio (not silence).")


def play_tone_thread(out_stream, tone_float32_bytes):
    out_stream.write(tone_float32_bytes)
    try:
        out_stream.stop_stream()
    except OSError:
        pass
    out_stream.close()


def selftest(out_path="selftest.wav", duration_s=4.0, tone_hz=1000.0):
    """Play a 1kHz tone on the default output, capture loopback, then verify the tone came back."""
    p = pyaudio.PyAudio()
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    out_dev = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    loop_dev = find_loopback_device(p)

    rate = 48000
    channels = 2

    # --- build tone: 1 kHz sine, with fade in/out to avoid clicks ---
    n = int(rate * duration_s)
    t = np.arange(n) / rate
    fade = int(0.05 * rate)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    tone = (0.2 * np.sin(2 * np.pi * tone_hz * t) * env).astype(np.float32)
    stereo = np.column_stack([tone, tone])
    tone_float32_bytes = stereo.tobytes()  # stream is paFloat32: 4 bytes/sample

    state = {"buf": deque(), "lock": threading.Lock()}

    def capture_cb(in_data, frame_count, time_info, status_flags):
        if in_data:
            with state["lock"]:
                state["buf"].append(in_data)
        return (None, pyaudio.paContinue)

    # Arm hard watchdog before opening any stream (open() can block on idle engine).
    threading.Thread(target=watchdog, args=(duration_s + 25,), daemon=True).start()

    # 1) Start playback FIRST so the audio engine is rendering (unsticks loopback open).
    log(f"Opening output on '{out_dev['name']}'...")
    out_stream = p.open(
        format=pyaudio.paFloat32,
        channels=channels,
        rate=rate,
        output=True,
        output_device_index=out_dev["index"],
    )
    threading.Thread(target=play_tone_thread, args=(out_stream, tone_float32_bytes), daemon=True).start()
    time.sleep(0.5)  # let the render engine spin up

    # 2) Now open loopback capture while the engine is active.
    log(f"Playing {tone_hz:.0f} Hz tone on '{out_dev['name']}' for {duration_s}s...")
    log(f"Opening loopback input on '{loop_dev['name']}'...")
    in_stream = p.open(
        format=FORMAT,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=loop_dev["index"],
        frames_per_buffer=CHUNK,
        stream_callback=capture_cb,
    )
    log("Loopback opened OK.")
    in_stream.start_stream()
    ok = wait_for_capture(state, rate, duration_s, timeout_s=duration_s + 25)
    in_stream.stop_stream()
    in_stream.close()
    p.terminate()
    log("")

    with state["lock"]:
        raw = b"".join(state["buf"])

    if not ok or not raw:
        log("ERROR: No audio captured during selftest.")
        sys.exit(1)

    audio = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
    sf.write(out_path, audio, rate, subtype="PCM_16")

    # --- verify: RMS high enough + dominant FFT peak near 1 kHz ---
    left = audio[:, 0].astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(left ** 2)))
    spectrum = np.abs(np.fft.rfft(left))
    freqs = np.fft.rfftfreq(len(left), 1 / rate)
    peak_freq = float(freqs[np.argmax(spectrum)])
    tone_band = spectrum[np.abs(freqs - tone_hz) < 50].max()

    log(f"Saved -> {out_path} ({audio.shape[0] / rate:.1f}s)")
    log(f"  RMS level      : {rms:.1%}")
    log(f"  dominant freq  : {peak_freq:.0f} Hz  (expected ~{tone_hz:.0f} Hz)")
    log(f"  tone-band peak : {tone_band:.0f}")

    ok_rms = rms > 0.01
    ok_freq = abs(peak_freq - tone_hz) < 100 or tone_band > 100
    if ok_rms and ok_freq:
        log("SELFTEST PASS: loopback captured the tone that was played.")
        sys.exit(0)
    log("SELFTEST FAIL: captured audio does not match the played tone.")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Capture system audio (WASAPI loopback) to WAV.")
    ap.add_argument("-d", "--duration", type=float, default=10.0)
    ap.add_argument("-o", "--output", default="capture.wav")
    ap.add_argument("-i", "--device", type=int, default=None, help="loopback device index")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="play tone + capture loopback to verify")
    args = ap.parse_args()

    p = pyaudio.PyAudio()
    if args.list:
        list_loopback_devices(p)
        p.terminate()
        return
    if args.selftest:
        p.terminate()
        selftest(out_path=args.output, duration_s=args.duration)
        return

    device_index = args.device
    if device_index is None:
        device_index = find_loopback_device(p)["index"]
    p.terminate()

    capture(args.duration, args.output, device_index=device_index)


if __name__ == "__main__":
    main()
