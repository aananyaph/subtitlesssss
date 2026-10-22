"""WebSocket Bridge for Electron Frontend ↔ Python Pipeline.

Listens on ws://localhost:8765 (same port as legacy vad_server for compatibility).
- Receives Electron control messages: START_CAPTURE / STOP_CAPTURE / UPDATE_SETTINGS / HANDSHAKE
- Broadcasts live subtitles to all connected Electron/Chrome clients:
  {type: "final"|"partial", text, language, start, end, processing_time_s}
  {type: "AUDIO_DIAGNOSTICS", diagnostics: {...}}
- Runs DesktopLiveSubtitlePipeline (base multilingual translate) and forwards overlay pushes.

Usage:
  python backend/app_bridge.py [--model base] [--port 8765]
  Then start Electron: cd frontend && npm start

This keeps VAD, punct, overlay exactly as main_app.py; only adds WS broadcasting.
"""
import argparse
import asyncio
import json
import logging
import sys
import os
import threading
import time
from typing import Set

import websockets

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.settings import load_settings

# Import pipeline after path setup
from vad.audio_utils import AudioPreprocessor
from vad.vad_engine import SileroVadEngine
from vad.segmenter import SpeechSegmenter
from asr.asr_engine import OpenVINOASREngine
from asr.streaming_asr import StreamingASRManager
from asr.nlp_processor import NLPPostProcessor
from app.loopback_capture import WASAPILoopbackCapture

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8765))

connected: Set[websockets.WebSocketServerProtocol] = set()
event_loop: asyncio.AbstractEventLoop = None
pipeline = None  # type: ignore

# ---------- Broadcast helpers ----------
async def broadcast_json(obj):
    if not connected:
        return
    msg = json.dumps(obj, ensure_ascii=False)
    dead = set()
    for ws in list(connected):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    for d in dead:
        connected.discard(d)

def broadcast_json_threadsafe(obj):
    if event_loop and not event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(broadcast_json(obj), event_loop)

# ---------- Bridge Overlay (forwards to Electron) ----------
class BridgeOverlay:
    """Fake overlay that broadcasts to WS clients instead of Qt window."""
    def push_live_text(self, text, is_final=True):
        logging.info(f"[BridgeOverlay] push {'FINAL' if is_final else 'PARTIAL'}: {text!r}")
        broadcast_json_threadsafe({
            "type": "final" if is_final else "partial",
            "text": text,
            "language": getattr(pipeline, 'last_detected_language', None) or "en",
            "is_final": is_final,
        })

# ---------- WebSocket handler ----------
async def handle_client(ws):
    global pipeline
    logging.info(f"[Bridge] Client connected {ws.remote_address}")
    connected.add(ws)
    # Send handshake ack
    try:
        await ws.send(json.dumps({"type": "connected", "message": f"Bridge ready model={getattr(pipeline, 'model_name', '?')} task={getattr(pipeline, 'asr_task', '?')}"}))
    except Exception:
        pass
    try:
        async for raw in ws:
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                t = data.get("type")
                if t == "HANDSHAKE":
                    logging.info(f"[Bridge] HANDSHAKE {data}")
                    if pipeline and hasattr(pipeline, 'preprocessor'):
                        sr = data.get("sampleRate", 48000)
                        ch = data.get("numChannels", 2)
                        try:
                            pipeline.preprocessor.set_format(sr, ch)
                        except Exception:
                            pass
                elif t in ("START_CAPTURE", "STOP_CAPTURE"):
                    active = (t == "START_CAPTURE")
                    if pipeline:
                        pipeline.set_capturing(active)
                    await broadcast_json({"type": "status", "capturing": active})
                elif t == "UPDATE_SETTINGS":
                    settings = data.get("settings", {})
                    model = settings.get("model")
                    if model and pipeline:
                        try:
                            pipeline.set_model(model)
                            await broadcast_json({"type": "status", "model": model, "task": getattr(pipeline, 'asr_task', None)})
                        except Exception as e:
                            logging.warning(f"[Bridge] set_model failed {e}")
                elif t == "CLEAR_HISTORY":
                    await broadcast_json({"type": "flush_ack"})
                # Ignore other types
            elif isinstance(raw, bytes):
                # Optional: if Electron ever sends PCM directly, feed to pipeline
                if pipeline and hasattr(pipeline, 'audio_queue'):
                    # Same path as WASAPI: push raw bytes
                    try:
                        pipeline.preprocessor.set_format(48000, 2)
                    except Exception:
                        pass
                    try:
                        pipeline.audio_queue.put_nowait((raw, False, time.monotonic()))
                    except Exception:
                        pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected.discard(ws)
        logging.info(f"[Bridge] Client disconnected {ws.remote_address}")

# ---------- Pipeline wrapper ----------
def create_pipeline(model_name: str, overlay):
    # Reuse DesktopLiveSubtitlePipeline but without Qt
    import main_app
    p = main_app.DesktopLiveSubtitlePipeline(overlay=overlay, model_name=model_name)
    return p

async def main_async(args):
    global event_loop, pipeline
    event_loop = asyncio.get_running_loop()

    overlay = BridgeOverlay()
    pipeline = create_pipeline(args.model, overlay)
    # Also broadcast diagnostics periodically (meter)
    # Start pipeline worker (non-Qt path) — it starts WASAPI thread
    try:
        pipeline.start()
        logging.info("[Bridge] Pipeline started (WASAPI loopback active if device present)")
    except Exception as e:
        logging.warning(f"[Bridge] Pipeline start failed (no audio device?) {e}. Continuing WS-only mode.")

    # Start WS server
    logging.info(f"[Bridge] Serving on ws://{args.host}:{args.port} — Electron connect there")
    async with websockets.serve(handle_client, args.host, args.port, max_size=None):
        # Also broadcast a periodic heartbeat
        while True:
            await asyncio.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Electron Bridge — Live Subtitle WS server")
    parser.add_argument("--host", default=HOST, help="Host to bind (default 0.0.0.0 or HOST env)")
    parser.add_argument("--model", default=None, help="ASR model (base, distil-small.en, tiny.en). Default from settings")
    parser.add_argument("--port", type=int, default=PORT, help="WebSocket port (default 8765)")
    parser.add_argument("--list-devices", action="store_true", help="List WASAPI devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        from app.loopback_capture import list_loopback_devices
        list_loopback_devices()
        sys.exit(0)

    if not args.model:
        s = load_settings()
        args.model = str(s.get("model", "base"))
    args.port = int(args.port)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logging.info("[Bridge] Shutting down...")
        if pipeline:
            try:
                pipeline.stop()
            except Exception:
                pass

if __name__ == "__main__":
    main()
