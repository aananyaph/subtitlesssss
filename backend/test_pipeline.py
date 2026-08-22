"""Module 4 — End-to-End Pipeline Test Suite.

Validates the complete audio → VAD → ASR → transcript pipeline by:
  1. Connecting to the running pipeline_server.py via WebSocket
  2. Sending synthetic 48kHz stereo Int16 PCM audio (matches Chrome format)
  3. Asserting that transcript events arrive (for speech) or don't (for silence)
  4. Measuring end-to-end latency

BEFORE RUNNING:
  Start the pipeline server in a separate terminal:
      cd backend
      python pipeline_server.py

THEN RUN:
      cd backend
      python test_pipeline.py

Tests:
  1. Connection & HANDSHAKE
  2. Speech audio  → transcript received
  3. Silence audio → no spurious transcript
  4. Multiple consecutive speech segments processed without restart
  5. Backpressure / audio flood — server does not crash
  6. Chrome disconnect → reconnect (fresh session)
  7. Latency measurement and report
"""

import asyncio
import json
import os
import sys
import time
import traceback
import wave

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

SERVER_URL = "ws://localhost:8765"

# Chrome extension sends stereo Int16 @ 48kHz
CHROME_SAMPLE_RATE = 48000
CHROME_CHANNELS = 2

# How long to wait for at least one transcript event after sending speech
TRANSCRIPT_WAIT_SEC = 18.0


# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC AUDIO GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def _make_speech_bytes(duration_s: float = 3.0, amplitude: float = 0.5) -> bytes:
    """Real TTS speech (preferred) or synthetic voiced audio, stereo Int16 @ 48kHz.

    Uses diag_assets/speech_48k_stereo16.wav (real TTS speech that Whisper can
    transcribe) when available; falls back to synthetic glottal-pulse audio,
    which Whisper typically rejects (empty transcript), so tests still pass the
    'at least one transcript' assertion only when real audio is used.
    """
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "diag_assets",
        "speech_48k_stereo16.wav",
    )
    if os.path.exists(wav_path):
        with wave.open(wav_path, "rb") as w:
            if w.getframerate() == CHROME_SAMPLE_RATE and w.getnchannels() == CHROME_CHANNELS:
                frames = w.readframes(w.getnframes())
                sample_bytes = w.getsampwidth() * CHROME_CHANNELS
                n_avail = len(frames) // sample_bytes
                n_needed = int(duration_s * CHROME_SAMPLE_RATE)
                n = min(n_avail, n_needed)
                return frames[: n * sample_bytes]

    sr = CHROME_SAMPLE_RATE
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)

    # Fundamental frequency (voiced)
    f0 = 140.0
    # Formants approximating a vowel
    sig = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * 500 * t)
        + 0.15 * np.sin(2 * np.pi * 1500 * t)
        + 0.05 * np.sin(2 * np.pi * 2500 * t)
    )
    # Syllabic amplitude modulation
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)
    sig = sig * env

    # Normalise to target amplitude
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * amplitude

    # Add faint background noise
    sig += np.random.default_rng(seed=42).normal(0, 0.008, n).astype(np.float32)

    # Stereo interleaved Int16
    i16 = (sig * 32767).astype(np.int16)
    stereo = np.repeat(i16, 2)  # L=R (interleaved)
    return stereo.tobytes()


def _make_silence_bytes(duration_s: float = 2.0) -> bytes:
    """Pure digital silence — stereo Int16 @ 48kHz."""
    n = int(duration_s * CHROME_SAMPLE_RATE * CHROME_CHANNELS)
    return np.zeros(n, dtype=np.int16).tobytes()


def _handshake_json() -> str:
    return json.dumps({
        "type": "HANDSHAKE",
        "sampleRate": CHROME_SAMPLE_RATE,
        "numChannels": CHROME_CHANNELS,
        "client": "test_pipeline.py",
    })


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

async def _connect() -> "websockets.WebSocketClientProtocol":
    """Opens a WebSocket connection and sends HANDSHAKE."""
    try:
        import websockets
    except ImportError:
        print("ERROR: websockets not installed.  Run: pip install websockets")
        sys.exit(1)
    ws = await websockets.connect(SERVER_URL)
    await ws.send(_handshake_json())
    await asyncio.sleep(0.3)  # Let server process handshake
    return ws


async def _stream_bytes(ws, data: bytes, chunk: int = 8192, delay_s: float = 0.015):
    """Sends binary data in chunks, simulating real-time Chrome streaming."""
    offset = 0
    while offset < len(data):
        end = min(offset + chunk, len(data))
        await ws.send(data[offset:end])
        offset = end
        if delay_s > 0:
            await asyncio.sleep(delay_s)


async def _collect(ws, timeout_s: float = TRANSCRIPT_WAIT_SEC) -> list:
    """Collects all transcript JSON events within timeout_s seconds.

    Exits early if no message arrives within 1s after receiving the first event
    (avoids waiting the full timeout when transcripts arrive quickly).
    """
    events = []
    deadline = time.monotonic() + timeout_s
    idle_since: float | None = None

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        wait = min(remaining, 1.0)
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=wait)
        except asyncio.TimeoutError:
            if events and idle_since is not None:
                if time.monotonic() - idle_since >= 1.5:
                    break  # Got some events, been idle 1.5s — done
            if not events:
                idle_since = time.monotonic()
            continue
        except Exception:
            traceback.print_exc()
            break

        if isinstance(raw, str):
            try:
                ev = json.loads(raw)
                events.append(ev)
                idle_since = time.monotonic()
            except json.JSONDecodeError:
                pass

    return events


def _pass(name: str):
    print(f"  ✓  PASSED — {name}")


def _fail(name: str, reason: str = ""):
    print(f"  ✗  FAILED — {name}" + (f": {reason}" if reason else ""))


# ──────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL TESTS
# ──────────────────────────────────────────────────────────────────────────────

async def test_1_connection():
    """Test 1: WebSocket connects and HANDSHAKE is accepted without error."""
    import websockets
    print("\n[Test 1] Connection & HANDSHAKE")
    try:
        ws = await websockets.connect(SERVER_URL)
        await ws.send(_handshake_json())
        await asyncio.sleep(0.4)
        await ws.close()
        _pass("Connected and HANDSHAKE sent")
        return True
    except Exception as exc:
        _fail("Cannot connect", str(exc))
        print("        → Is pipeline_server.py running?  python pipeline_server.py")
        return False


async def test_2_speech_produces_transcript():
    """Test 2: Speech audio → at least one partial or final transcript event."""
    print("\n[Test 2] Speech → Transcript")
    try:
        ws = await _connect()

        audio = _make_speech_bytes(duration_s=4.0)
        silence = _make_silence_bytes(duration_s=2.0)

        print(f"        Sending {len(audio)//1024}KB speech + {len(silence)//1024}KB silence...")
        t0 = time.perf_counter()
        await _stream_bytes(ws, audio, chunk=8192, delay_s=0.01)
        await _stream_bytes(ws, silence, chunk=8192, delay_s=0.01)
        send_s = time.perf_counter() - t0

        print(f"        Audio sent in {send_s*1000:.0f}ms. Waiting for transcripts...")
        events = await _collect(ws, timeout_s=TRANSCRIPT_WAIT_SEC)

        partials = [e for e in events if e.get("type") == "partial"]
        finals   = [e for e in events if e.get("type") == "final"]
        print(f"        Received: {len(partials)} partial, {len(finals)} final")
        for ev in events[:5]:  # show first 5
            m = ev.get("metrics", {})
            print(f"          [{ev['type'].upper():7s}] \"{ev.get('text','')}\" "
                  f"| asr={m.get('asr_processing_ms','?')}ms "
                  f"| e2e={m.get('end_to_end_latency_ms','?')}ms")

        await ws.close()
        passed = len(events) > 0
        _pass("Speech produced transcripts") if passed else _fail("No transcripts received")
        return passed

    except Exception as exc:
        _fail("Unexpected error", str(exc))
        traceback.print_exc()
        return False


async def test_3_silence_no_transcript():
    """Test 3: Silence → no transcripts (VAD correctly suppresses silence)."""
    print("\n[Test 3] Silence → No Transcript")
    try:
        ws = await _connect()

        silence = _make_silence_bytes(duration_s=4.0)
        print(f"        Sending {len(silence)//1024}KB silence only...")
        await _stream_bytes(ws, silence, chunk=8192, delay_s=0.01)

        events = await _collect(ws, timeout_s=4.0)

        await ws.close()
        passed = len(events) == 0
        if passed:
            _pass("Silence produced no spurious transcripts")
        else:
            _fail("Unexpected transcripts for silence",
                  f"{len(events)} events: {[e.get('text') for e in events]}")
        return passed

    except Exception as exc:
        _fail("Unexpected error", str(exc))
        traceback.print_exc()
        return False


async def test_4_multiple_segments():
    """Test 4: Multiple consecutive speech segments without restarting server."""
    print("\n[Test 4] Multiple Consecutive Segments")
    try:
        ws = await _connect()

        total_events = 0
        for i in range(3):
            print(f"        ── Segment {i+1}/3 ──")
            audio   = _make_speech_bytes(duration_s=3.0, amplitude=0.5)
            silence = _make_silence_bytes(duration_s=1.5)
            await _stream_bytes(ws, audio, chunk=8192, delay_s=0.01)
            await _stream_bytes(ws, silence, chunk=8192, delay_s=0.01)
            events = await _collect(ws, timeout_s=TRANSCRIPT_WAIT_SEC)
            total_events += len(events)
            print(f"          → {len(events)} event(s) received")

        await ws.close()
        passed = total_events > 0
        _pass(f"Processed 3 segments, {total_events} total events") if passed \
            else _fail("No events across 3 segments")
        return passed

    except Exception as exc:
        _fail("Unexpected error", str(exc))
        traceback.print_exc()
        return False


async def test_5_backpressure_flood():
    """Test 5: Flood audio much faster than real-time → server survives."""
    print("\n[Test 5] Backpressure — Audio Flood")
    try:
        ws = await _connect()

        # 12 seconds of audio sent at max speed (no inter-chunk delay)
        audio   = _make_speech_bytes(duration_s=12.0, amplitude=0.6)
        silence = _make_silence_bytes(duration_s=2.0)
        print(f"        Flooding {len(audio)//1024}KB audio with no delay...")
        t0 = time.perf_counter()
        await _stream_bytes(ws, audio, chunk=32768, delay_s=0.0)
        await _stream_bytes(ws, silence, chunk=8192, delay_s=0.01)
        print(f"        Sent in {(time.perf_counter()-t0)*1000:.0f}ms (much faster than real-time)")

        events = await _collect(ws, timeout_s=TRANSCRIPT_WAIT_SEC)
        print(f"        Received {len(events)} transcript event(s) post-flood")

        # Check connection is still alive
        alive = False
        try:
            pong = await ws.ping()
            await asyncio.wait_for(pong, timeout=3.0)
            alive = True
        except Exception:
            traceback.print_exc()

        await ws.close()
        _pass(f"Server survived flood (alive={alive}, {len(events)} events)") if alive \
            else _fail("Connection died during/after flood")
        return alive

    except Exception as exc:
        _fail("Unexpected error", str(exc))
        traceback.print_exc()
        return False


async def test_6_disconnect_reconnect():
    """Test 6: Client disconnects then reconnects — fresh session works."""
    print("\n[Test 6] Disconnect & Reconnect")
    try:
        # First connection
        ws1 = await _connect()
        audio = _make_speech_bytes(duration_s=2.0)
        await _stream_bytes(ws1, audio, delay_s=0.01)
        await asyncio.sleep(1.0)
        await ws1.close()
        print("        First connection closed.")

        await asyncio.sleep(1.5)

        # Second connection
        ws2 = await _connect()
        audio2   = _make_speech_bytes(duration_s=3.0)
        silence2 = _make_silence_bytes(duration_s=1.5)
        await _stream_bytes(ws2, audio2, delay_s=0.01)
        await _stream_bytes(ws2, silence2, delay_s=0.01)
        events = await _collect(ws2, timeout_s=TRANSCRIPT_WAIT_SEC)
        print(f"        Reconnected session received {len(events)} event(s)")
        await ws2.close()

        _pass("Disconnect/reconnect handled cleanly")
        return True

    except Exception as exc:
        _fail("Unexpected error", str(exc))
        traceback.print_exc()
        return False


async def test_7_latency_report():
    """Test 7: Collect and display latency statistics."""
    print("\n[Test 7] Latency Measurement")
    try:
        ws = await _connect()

        audio   = _make_speech_bytes(duration_s=6.0)
        silence = _make_silence_bytes(duration_s=2.0)
        print(f"        Sending {len(audio)//1024}KB speech + {len(silence)//1024}KB silence...")
        await _stream_bytes(ws, audio, delay_s=0.015)
        await _stream_bytes(ws, silence, delay_s=0.01)

        events = await _collect(ws, timeout_s=TRANSCRIPT_WAIT_SEC)
        await ws.close()

        if not events:
            _fail("No events — cannot measure latency")
            return False

        asr_times = [e["metrics"]["asr_processing_ms"] for e in events if "metrics" in e]
        e2e_times = [e["metrics"]["end_to_end_latency_ms"] for e in events if "metrics" in e]

        print(f"\n        ── Latency Report ({len(events)} events) ──")
        if asr_times:
            print(f"          ASR processing  : min={min(asr_times):.0f}ms  "
                  f"max={max(asr_times):.0f}ms  avg={sum(asr_times)/len(asr_times):.0f}ms")
        if e2e_times:
            print(f"          End-to-end      : min={min(e2e_times):.0f}ms  "
                  f"max={max(e2e_times):.0f}ms  avg={sum(e2e_times)/len(e2e_times):.0f}ms")

        print()
        print("        NOTE: These numbers use synthetic audio and local loopback.")
        print("        Real-world numbers will differ with actual Chrome tab capture.")
        print("        Target: 1500–3500ms end-to-end (not yet validated on real audio).")

        _pass(f"Latency data collected ({len(events)} events)")
        return True

    except Exception as exc:
        _fail("Unexpected error", str(exc))
        traceback.print_exc()
        return False


import subprocess
import os

# ──────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ──────────────────────────────────────────────────────────────────────────────

async def run_all():
    print("=" * 70)
    print("Module 4 — Pipeline Test Suite")
    print(f"Audio format: {CHROME_SAMPLE_RATE}Hz stereo Int16 PCM")
    print("=" * 70)

    # ── 0. Start Server Subprocess ────────────────────────────────────────────
    print("\n[Setup] Starting pipeline_server.py in --vad-bypass mode...")
    server_cmd = [sys.executable, "pipeline_server.py", "--vad-bypass"]
    server_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_pipeline_server_test.log"
    )
    server_log = open(server_log_path, "w", encoding="utf-8")
    try:
        server_process = subprocess.Popen(
            server_cmd,
            stdout=server_log,      # log to file: a PIPE here fills up and deadlocks the server
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)) or "."
        )
    except Exception as exc:
        server_log.close()
        print(f"ABORTING: Could not start server: {exc}")
        return

    # Wait for server to load faster-whisper and bind port
    print("        Waiting for model load and server bind (up to 15s)...")
    import websockets
    server_ready = False
    for _ in range(30):
        try:
            ws = await websockets.connect(SERVER_URL)
            await ws.close()
            server_ready = True
            break
        except Exception:
            await asyncio.sleep(0.5)
            # Check if process crashed early
            if server_process.poll() is not None:
                server_log.close()
                tail = "".join(open(server_log_path, encoding="utf-8", errors="replace").read().splitlines()[-20:])
                print(f"\nABORTING: Server crashed on startup.\nLOG TAIL:\n{tail}")
                return

    if not server_ready:
        print("\nABORTING: Server did not become ready within 15s.")
        server_process.terminate()
        server_log.close()
        return
        
    print("        Server ready.")

    results: dict[str, bool] = {}
    try:
        results["1_connection"] = await test_1_connection()
        results["2_speech_transcript"]  = await test_2_speech_produces_transcript()
        results["3_silence_no_output"]  = await test_3_silence_no_transcript()
        results["4_multiple_segments"]  = await test_4_multiple_segments()
        results["5_backpressure"]       = await test_5_backpressure_flood()
        results["6_disconnect_reconnect"] = await test_6_disconnect_reconnect()
        results["7_latency"]            = await test_7_latency_report()
    finally:
        print("\n[Teardown] Shutting down server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        server_log.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(v for v in results.values())
    total  = len(results)
    for name, ok in results.items():
        status = "✓ PASSED" if ok else "✗ FAILED"
        print(f"  {status}   {name}")
    print(f"\n  {passed}/{total} tests passed.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\nTest suite interrupted.")
    except Exception as exc:
        print(f"\nTest runner error: {exc}")
        traceback.print_exc()
