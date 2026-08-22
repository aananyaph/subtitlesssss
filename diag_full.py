"""Full end-to-end diagnostic runner.

Stage 1: standalone WASAPI loopback capture while real speech plays through the
         system default output device. Verifies real audio + correct device.
Full chain: the real DesktopLiveSubtitlePipeline + Qt overlay running on real
         system audio; verifies VAD events -> ASR text -> NLP -> overlay render.
"""

import sys, os, time, threading, subprocess, queue, logging

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

WAV = os.path.join(ROOT, "diag_assets", "speech_48k_stereo16.wav")
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def play_wav(path, loops=1):
    """Plays a WAV through the system default output device via SoundPlayer."""
    ps = (
        f"$p = New-Object System.Media.SoundPlayer('{path}'); "
        f"$p.Load(); 1..{loops} | ForEach-Object {{ $p.PlaySync() }}"
    )
    return subprocess.Popen([POWERSHELL, "-NoProfile", "-Command", ps])


# ── Stage 1: standalone capture ──────────────────────────────────────────────
def stage1_capture():
    from app.loopback_capture import WASAPILoopbackCapture
    import pyaudiowpatch as pyaudio

    print("\n===== STAGE 1: WASAPI LOOPBACK CAPTURE (standalone) =====")
    p = pyaudio.PyAudio()
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    print(f"Default output device: index={wasapi['defaultOutputDevice']} name={default_out['name']}")
    print("--- All WASAPI loopback devices ---")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("hostApi") == wasapi["index"] and info.get("isLoopbackDevice"):
            print(f"  [{i}] {info['name']} | {info['maxInputChannels']}ch | {info['defaultSampleRate']} Hz")
    default_name = default_out.get("name", "").lower()
    default_loop_idx = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("hostApi") == wasapi["index"] and info.get("isLoopbackDevice") \
                and default_name in info.get("name", "").lower():
            default_loop_idx = i
    print(f"Loopback index matching default output '{default_out['name']}': {default_loop_idx}")
    p.terminate()

    peaks = []
    formats = set()
    first_data = {"got": False, "sr": None, "ch": None, "bytes": None}
    t0 = time.time()

    def cb(raw, sr, ch, is_f32):
        peaks.append((time.time(), float(abs(int.from_bytes(raw[:2], "little", signed=True)) / 32768.0)))
        if not first_data["got"]:
            first_data.update(got=True, sr=sr, ch=ch, bytes=len(raw))
        formats.add((sr, ch, is_f32))

    cap = WASAPILoopbackCapture(audio_callback=cb, device_index=None)
    cap.start()

    player = play_wav(WAV, loops=1)
    print(f"Playing '{WAV}' through default output device for ~26s...")
    while time.time() - t0 < 30:
        now = time.time()
        window = [pk for ts, pk in peaks if now - 1.2 < ts <= now]
        if window:
            print(f"[Meter] t={now-t0:5.1f}s peak={max(window):.4f} ({max(window):.1%})")
        time.sleep(1.0)
    player.terminate()
    cap.stop()

    nonzero = [pk for _, pk in peaks if pk > 0.001]
    print(f"Stage 1 summary: chunks={len(peaks)} nonzero_peak_chunks={len(nonzero)} "
          f"formats={formats} first_data={first_data}")
    peak_ok = len(nonzero) > 100
    dev_ok = default_loop_idx is not None
    fmt_ok = (48000, 2) in {(s, c) for s, c, _ in formats}
    return ("PASS" if (peak_ok and dev_ok) else "FAIL",
            f"nonzero_chunks={len(nonzero)}/{len(peaks)}, "
            f"default_output_loopback_idx={default_loop_idx}, formats={formats}")


# ── Full chain: real pipeline + overlay ──────────────────────────────────────
def full_chain(duration_s=75):
    from PyQt6.QtWidgets import QApplication
    from app.overlay_window import DesktopSubtitleOverlay
    from main_app import DesktopLiveSubtitlePipeline

    print("\n===== FULL CHAIN: capture -> VAD -> ASR -> NLP -> overlay =====")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    overlay = DesktopSubtitleOverlay(font_size_mode="medium", position_mode="bottom")
    pipeline = DesktopLiveSubtitlePipeline(overlay=overlay, model_name="distil-small.en",
                                           min_chunk_floor_sec=1.0, device_index=None)
    pipeline.start()

    player = play_wav(WAV, loops=2)
    print(f"Playing speech through speakers for {duration_s}s...")

    rendered = []
    for _ in range(duration_s):
        time.sleep(1.0)
        txt = overlay.label.text()
        if txt and txt not in rendered:
            rendered.append(txt)
        app.processEvents()

    player.terminate()
    pipeline.stop()

    print(f"\nFull-chain summary: unique subtitle lines rendered={len(rendered)}")
    for r in rendered:
        print(f"  SUBTITLE: {r!r}")
    print(f"final label text: {overlay.label.text()!r}")
    ok = len(rendered) >= 2
    return ("PASS" if ok else "FAIL", f"unique rendered subtitles={len(rendered)}")


def main():
    s1, s1ev = stage1_capture()
    fc, fcev = full_chain()
    print("\n===== FULL-CHAIN DIAGNOSTIC RESULTS =====")
    print(f"Stage1_capture | {s1:4s} | {s1ev}")
    print(f"Full_chain     | {fc:4s} | {fcev}")


if __name__ == "__main__":
    main()
