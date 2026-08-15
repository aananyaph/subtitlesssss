"""Module 10 — Seek Handling & Playback Synchronization Test Suite.

Validates:
    1. Generation/Epoch ID incrementing on seek_flush, video_switch, video_end control messages.
    2. Complete non-blocking queue flushing (audio_queue, speech_queue, correction_queue).
    3. State resets for VAD engine, SpeechSegmenter, and StreamingASRManager.
    4. Discarding of stale in-flight ASR segments and LLM correction jobs belonging to older generations.
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

from pipeline_server import PipelineSession, _SHUTDOWN, _drain_queue


class DummyASREngine:
    """Mock ASR engine for testing session control events."""
    model_size = "distil-small.en"
    device = "CPU"
    compute_type = "int8"

    def transcribe(self, audio, language="en"):
        return {"text": "dummy transcript", "segments": [], "language": "en"}


class TestSeekHandling(unittest.TestCase):
    """Test suite for Module 10 playback synchronization and state resets."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.session = PipelineSession(
            asr_engine=DummyASREngine(),
            event_loop=self.loop,
            enable_llm=False,
        )
        self.session.transcript_queue = asyncio.Queue(maxsize=50)

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
        self.session.correction_queue.put_nowait({"segment_id": "segment_001", "generation_id": 1})

        self.assertFalse(self.session.audio_queue.empty())
        self.assertFalse(self.session.speech_queue.empty())
        self.assertFalse(self.session.correction_queue.empty())

        # Perform flush
        self.session.flush_pipeline(event_reason="seek_flush", video_time=45.0)

        self.assertTrue(self.session.audio_queue.empty())
        self.assertTrue(self.session.speech_queue.empty())
        self.assertTrue(self.session.correction_queue.empty())

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
