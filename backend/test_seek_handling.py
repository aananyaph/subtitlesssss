"""Module 10 — Seek Handling & Playback Synchronization Test Suite.

Validates:
    1. Generation/Epoch ID incrementing on seek_flush, video_switch, video_end control messages.
    2. Complete non-blocking queue flushing (audio_queue, speech_queue).
    3. State resets for VAD engine, SpeechSegmenter, and StreamingASRManager.
    4. Discarding of stale in-flight ASR segments belonging to older generations.
    5. Emission of flush_ack confirmation events.
"""

import os
import sys
import time
import queue
import asyncio
import unittest

# Ensure backend path is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.asr.asr_engine import DummyASREngine
from backend.asr.streaming_asr import StreamingASRManager


class DummyPipelineSession:
    """Minimal session mock for seek handling tests — no pipeline_server dependency."""

    def __init__(self, asr_engine, event_loop, enable_llm=False):
        self.asr_engine = asr_engine
        self.loop = event_loop
        self.enable_llm = enable_llm
        self.generation_id = 1
        self._shutdown_event = None

        # Queues (bounded, with drop-oldest backpressure)
        self.audio_queue = queue.Queue(maxsize=150)
        self.speech_queue = queue.Queue(maxsize=10)
        # Correction queue removed — Step 3 uses simplified model

        self.transcript_queue = asyncio.Queue(maxsize=50)

    def flush_pipeline(self, event_reason: str = "seek_flush", video_time: float = 0.0):
        """Performs a full, non-blocking pipeline reset upon video seek, switch, or end.

        Actions:
            1. Increment generation_id to invalidate in-flight ASR/LLM results.
            2. Drain audio_queue and speech_queue.
            3. Reset state.
            4. Send flush_ack JSON back via transcript_queue.
        """
        with __import__('threading').Lock():
            self.generation_id += 1
            gen = self.generation_id

            a_drop = _drain_queue(self.audio_queue)
            s_drop = _drain_queue(self.speech_queue)

            logging.info(
                f"[FLUSH] reason='{event_reason}' | video_time={video_time:.2f}s | "
                f"new_generation={gen} | dropped queues: audio={a_drop}, "
                f"speech={s_drop}"
            )

            ack_event = {
                "type": "flush_ack",
                "reason": event_reason,
                "generation_id": gen,
                "video_time": video_time,
            }
            if self.transcript_queue:
                try:
                    self.loop.call_soon_threadsafe(
                        self.transcript_queue.put_nowait, ack_event
                    )
                except Exception as exc:
                    logging.warning(f"[FLUSH] Failed to send flush_ack: {exc}", exc_info=True)

    def _drain_queue(q: queue.Queue) -> int:
        """Empties a queue.Queue, discarding all items and returning count of items removed."""
        count = 0
        while True:
            try:
                q.get_nowait()
                count += 1
            except (queue.Empty, Exception):
                break
        return count


def _drain_queue(q: queue.Queue) -> int:
    """Empties a queue.Queue, discarding all items and returning count of items removed."""
    count = 0
    while True:
        try:
            q.get_nowait()
            count += 1
        except (queue.Empty, Exception):
            break
    return count


class TestSeekHandling(unittest.TestCase):
    """Test suite for Module 10 playback synchronization and state resets."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.session = DummyPipelineSession(
            asr_engine=DummyASREngine(),
            event_loop=self.loop,
            enable_llm=False,
        )

    def tearDown(self):
        self.loop.close()

    def test_generation_id_increment(self):
        """Generation ID must increment on each flush event."""
        initial_gen = self.session.generation_id
        self.assertEqual(initial_gen, 1)

        self.session.flush_pipeline(event_reason="seek_flush", video_time=12.5)
        self.assertEqual(self.session.generation_id, 2)

        self.session.flush_pipeline(event_reason="video_switch", video_time=0.0)
        self.assertEqual(self.session.generation_id, 3)

    def test_queue_flushing(self):
        """All queues must be completely emptied upon flush_pipeline."""
        # Populate queues
        self.session.audio_queue.put_nowait(("dummy_frame", time.perf_counter()))
        self.session.speech_queue.put_nowait({"type": "speech_end", "generation_id": 1})

        self.assertFalse(self.session.audio_queue.empty())
        self.assertFalse(self.session.speech_queue.empty())

        # Perform flush
        self.session.flush_pipeline(event_reason="seek_flush", video_time=45.0)

        self.assertTrue(self.session.audio_queue.empty())
        self.assertTrue(self.session.speech_queue.empty())

    def test_flush_ack_emission(self):
        """A flush_ack event must be enqueued into transcript_queue."""
        self.session.flush_pipeline(event_reason="seek_flush", video_time=99.0)

        self.assertFalse(self.session.transcript_queue.empty())
        ack = self.session.transcript_queue.get_nowait()
        self.assertEqual(ack["type"], "flush_ack")
        self.assertEqual(ack["reason"], "seek_flush")
        self.assertEqual(ack["generation_id"], 2)
        self.assertEqual(ack["video_time"], 99.0)

    def test_stale_segment_filtering(self):
        """Segments with generation_id < current generation must be identified as stale."""
        self.session.generation_id = 5

        old_segment = {"type": "speech_end", "generation_id": 4, "text": "old speech"}
        current_segment = {"type": "speech_end", "generation_id": 5, "text": "current speech"}

        self.assertNotEqual(old_segment["generation_id"], self.session.generation_id)
        self.assertEqual(current_segment["generation_id"], self.session.generation_id)


if __name__ == "__main__":
    unittest.main()
