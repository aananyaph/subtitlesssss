"""Asynchronous WebSocket Server for Module 2 Voice Activity Detection (VAD).

Binds to ws://localhost:8765 to receive live audio stream from Chrome Extension (Module 1).
Processes incoming PCM audio through AudioPreprocessor -> SileroVadEngine -> SpeechSegmenter.
"""
import asyncio
import json
import logging
import sys
import websockets
import numpy as np

from audio_utils import AudioPreprocessor
from vad_engine import SileroVadEngine
from segmenter import SpeechSegmenter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

HOST = "localhost"
PORT = 8765


class VadServerSession:
    """Manages state for an active WebSocket audio capture session."""

    def __init__(self, vad_engine: SileroVadEngine):
        self.vad_engine = vad_engine
        self.preprocessor = AudioPreprocessor(input_sample_rate=48000, num_channels=2, target_sample_rate=16000)
        self.segmenter = SpeechSegmenter(
            threshold=0.50,
            neg_threshold=0.35,
            min_speech_duration_ms=250,
            min_silence_duration_ms=400
        )
        self.frame_count = 0
        self.log_throttle = 50  # Print frame probability every 50 frames (~1.6s) to prevent console spam

    def handle_handshake(self, message_str: str):
        try:
            data = json.loads(message_str)
            if data.get("type") == "HANDSHAKE":
                sample_rate = data.get("sampleRate", 48000)
                num_channels = data.get("numChannels", 2)
                self.preprocessor.set_format(sample_rate, num_channels)
                self.vad_engine.reset_state()
                self.segmenter.reset()
                self.frame_count = 0
                logging.info(f"[VAD Server] Handshake received from client: {data.get('client', 'Unknown')}")
                logging.info(f"[VAD Server] Audio Config: {sample_rate} Hz, {num_channels} channels")
        except Exception as e:
            logging.warning(f"[VAD Server] Could not parse text message: {e}", exc_info=True)

    def handle_pcm_data(self, raw_pcm_bytes: bytes):
        frames = self.preprocessor.process_raw_bytes(raw_pcm_bytes)
        
        for frame in frames:
            self.frame_count += 1
            prob = self.vad_engine.predict(frame, sample_rate=16000)
            event = self.segmenter.process_frame(frame, prob)

            # Throttled debug frame output
            if self.frame_count % self.log_throttle == 0:
                logging.info(
                    f"VAD: Frame #{self.frame_count:05d} | Probability: {prob:.4f} | State: {event['state']}"
                )

            # High-visibility event logs for state transitions
            if event["type"] == "SPEECH_START":
                logging.info("=" * 60)
                logging.info(f"VAD EVENT: >>> SPEECH START <<< (Frame #{self.frame_count}, Prob: {prob:.2f})")
                logging.info("=" * 60)

            elif event["type"] == "SPEECH_END":
                duration_s = event["duration_seconds"]
                logging.info("=" * 60)
                logging.info(
                    f"VAD EVENT: <<< SPEECH END <<< (Duration: {duration_s:.2f}s, Speech PCM Samples: {len(event['audio_pcm'])})"
                )
                logging.info("=" * 60)
                # NOTE: In Module 3, event['audio_pcm'] will be dispatched to local ASR (Whisper / faster-whisper)


async def handle_client(websocket, path=None):
    logging.info(f"[VAD Server] Client connected from {websocket.remote_address}")
    
    # Instantiate VAD engine for session
    vad_engine = SileroVadEngine()
    session = VadServerSession(vad_engine)

    try:
        async for message in websocket:
            if isinstance(message, str):
                session.handle_handshake(message)
            elif isinstance(message, bytes):
                session.handle_pcm_data(message)
    except websockets.exceptions.ConnectionClosedOK:
        logging.info("[VAD Server] Client disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.warn(f"[VAD Server] Client connection error: {e}")
    except Exception as e:
        logging.error(f"[VAD Server] Unexpected session error: {e}", exc_info=True)
    finally:
        logging.info("[VAD Server] Session terminated.")


async def main():
    logging.info(f"Starting Module 2 Local VAD Server on ws://{HOST}:{PORT}...")
    async with websockets.serve(handle_client, HOST, PORT):
        logging.info(f"Server ready. Open Chrome Extension and click 'Start Capture' with WebSocket enabled.")
        await asyncio.Event().wait()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("VAD Server stopped by user.")
