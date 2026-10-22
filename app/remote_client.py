"""Remote ASR Client Pipeline for Live Subtitle PC App.

Enables teammates on separate laptops to capture local Windows WASAPI loopback audio
and stream PCM chunks to a shared remote Docker ML inference server over WebSocket.
This avoids downloading multi-GB model weights and eliminates heavy CPU inference on laptops.
"""

import asyncio
import json
import logging
import os
import queue
import threading
import time
from typing import Callable, Optional

from app.loopback_capture import WASAPILoopbackCapture

try:
    import websockets
    import websockets.client
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


class RemoteLiveSubtitlePipeline:
    """Streams local Windows loopback audio to a shared ASR inference server."""

    def __init__(
        self,
        server_url: str = "ws://localhost:8765",
        overlay=None,
        device_index: Optional[int] = None,
        meter_callback: Optional[Callable[[float, float, str], None]] = None,
        subtitle_callback: Optional[Callable[[str, bool], None]] = None,
    ):
        self.server_url = server_url.strip() or "ws://localhost:8765"
        self.overlay = overlay
        self.device_index = device_index
        self.meter_callback = meter_callback
        self.subtitle_callback = subtitle_callback

        self.is_capturing = True
        self._shutdown_event = threading.Event()
        self.audio_queue = queue.Queue(maxsize=150)
        self.loopback_capture: Optional[WASAPILoopbackCapture] = None
        self._worker_thread: Optional[threading.Thread] = None
        self.is_connected = False

    def start(self):
        """Starts WASAPI audio capture and the remote WebSocket streaming worker."""
        if not HAS_WEBSOCKETS:
            err = "websockets package is required for remote ASR. Install with: pip install websockets"
            logging.error(f"[RemotePipeline] {err}")
            if self.overlay:
                self.overlay.push_live_text(f"⚠️ {err}", is_final=True)
            return

        logging.info(f"[RemotePipeline] Starting remote client -> {self.server_url}")
        self._shutdown_event.clear()

        # Start WASAPI loopback capture on host laptop
        self.loopback_capture = WASAPILoopbackCapture(
            audio_queue=self.audio_queue,
            device_index=self.device_index,
        )
        self.loopback_capture.start()

        # Start WebSocket streaming background thread
        self._worker_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="RemoteASRWorker",
        )
        self._worker_thread.start()

    def stop(self):
        """Stops audio capture and closes the remote connection."""
        logging.info("[RemotePipeline] Stopping remote pipeline...")
        self._shutdown_event.set()
        if self.loopback_capture:
            self.loopback_capture.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self.is_connected = False

    def set_capturing(self, active: bool):
        """Toggles audio capture on/off."""
        self.is_capturing = active
        logging.info(f"[RemotePipeline] Audio capture state: {'ACTIVE' if active else 'PAUSED'}")
        if not active and self.overlay:
            self.overlay.push_live_text("")

    def set_model(self, model_name: str):
        """Requests remote server to switch model via WebSocket control message."""
        logging.info(f"[RemotePipeline] Requesting remote model switch: {model_name}")

    def _run_event_loop(self):
        """Runs the asyncio loop for WebSocket connection and streaming."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_client_loop())
        finally:
            loop.close()

    async def _ws_client_loop(self):
        """Main WebSocket connection loop with automatic reconnect."""
        while not self._shutdown_event.is_set():
            try:
                logging.info(f"[RemotePipeline] Connecting to shared ASR server {self.server_url}...")
                async with websockets.connect(self.server_url, ping_interval=20, ping_timeout=10) as ws:
                    self.is_connected = True
                    logging.info(f"[RemotePipeline] Connected to shared ASR server successfully.")
                    if self.overlay:
                        self.overlay.push_live_text(f" Connected to shared ASR: {self.server_url}", is_final=True)

                    # Send handshake with audio specs
                    handshake = json.dumps({
                        "type": "HANDSHAKE",
                        "sampleRate": 48000,
                        "numChannels": 2,
                    })
                    await ws.send(handshake)

                    # Run sender and receiver concurrently
                    send_task = asyncio.create_task(self._send_audio_loop(ws))
                    recv_task = asyncio.create_task(self._recv_subtitles_loop(ws))

                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()

            except Exception as e:
                self.is_connected = False
                if not self._shutdown_event.is_set():
                    logging.warning(f"[RemotePipeline] Connection to {self.server_url} lost/failed: {e}. Retrying in 3s...")
                    if self.overlay:
                        self.overlay.push_live_text(f"⚠️ Reconnecting to ASR server: {self.server_url}...", is_final=True)
                    await asyncio.sleep(3.0)

    async def _send_audio_loop(self, ws):
        """Streams audio chunks from queue to WebSocket server."""
        while not self._shutdown_event.is_set():
            try:
                item = self.audio_queue.get_nowait()
                if isinstance(item, tuple):
                    raw_bytes = item[0]
                else:
                    raw_bytes = item

                if self.is_capturing and raw_bytes:
                    await ws.send(raw_bytes)
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                logging.warning(f"[RemotePipeline] Error sending audio chunk: {e}")
                break

    async def _recv_subtitles_loop(self, ws):
        """Receives live subtitles from server and forwards to overlay/callbacks."""
        async for message in ws:
            if self._shutdown_event.is_set():
                break
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                if msg_type in ("final", "partial"):
                    text = data.get("text", "")
                    is_final = (msg_type == "final")
                    if text:
                        if self.overlay:
                            self.overlay.push_live_text(text, is_final=is_final)
                        if self.subtitle_callback:
                            self.subtitle_callback(text, is_final)
                elif msg_type == "AUDIO_DIAGNOSTICS" and self.meter_callback:
                    diag = data.get("diagnostics", {})
                    self.meter_callback(
                        diag.get("peak", 0.0),
                        diag.get("vad_prob", 0.0),
                        diag.get("state", "IDLE"),
                    )
            except Exception as e:
                logging.debug(f"[RemotePipeline] Error parsing message: {e}")
