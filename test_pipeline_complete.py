"""Regression tests for pipeline completeness fixes.

Covers:
  * subtitle formatter: no duplicated/truncated words, monotonic timestamps,
    valid empty WebVTT;
  * shared bridge: per-client caption routing, isolated pipelines, no local
    WASAPI capture on the server;
  * small broken-code fixes (NLP method indentation, ASR logging import);
  * dependency/notebook sanity for the final Colab workflow.

All tests are offline and use mocks where heavy models would be needed.
"""

import json
import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import app_bridge  # noqa: E402
import main_app  # noqa: E402
import subtitle_formatter as sf  # noqa: E402
from asr.nlp_processor import SileroPunctuationEngine  # noqa: E402


class FakeSession:
    def __init__(self, language="en"):
        self.sent = []
        self.pipeline = None
        self.asr_language = language

    def send_threadsafe(self, obj):
        self.sent.append(obj)


class TestSubtitleFormatter(unittest.TestCase):
    def test_repeated_words_are_not_duplicated(self):
        text = "hello world " * 10
        wrapped = sf._wrap_lines(text)
        self.assertEqual(wrapped.split().count("hello"), 10)
        self.assertEqual(wrapped.split().count("world"), 10)
        self.assertEqual(len(wrapped.splitlines()), sf.MAX_LINES)

    def test_long_word_is_chunked_not_truncated(self):
        word = "Supercalifragilisticexpialidocious" * 4
        wrapped = sf._wrap_lines(word)
        self.assertEqual("".join(wrapped.split()), word)

    def test_split_timestamps_are_monotonic_and_in_range(self):
        segment = {"start": 5.0, "end": 6.0, "text": " ".join(["word"] * 60)}
        pieces = sf.split_long_segment(segment)
        self.assertGreater(len(pieces), 1)
        for i, piece in enumerate(pieces):
            self.assertGreaterEqual(piece["start"], segment["start"])
            self.assertLessEqual(piece["end"], segment["end"])
            self.assertGreater(piece["end"], piece["start"])
            if i:
                self.assertGreaterEqual(piece["start"], pieces[i - 1]["end"])

    def test_empty_vtt_has_blank_line(self):
        self.assertEqual(sf.format_vtt([]), "WEBVTT\n\n")


class TestSharedBridgeIsolation(unittest.TestCase):
    def test_overlay_targets_only_its_session(self):
        s1, s2 = FakeSession("hi"), FakeSession("kn")
        app_bridge.BridgeOverlay(s1).push_live_text("नमस्ते", is_final=True)
        self.assertEqual(len(s1.sent), 1)
        self.assertEqual(s1.sent[0]["language"], "hi")
        self.assertEqual(s2.sent, [])

    def test_pipeline_is_built_without_local_capture_and_with_target(self):
        session = FakeSession()
        cfg = {
            "model_name": "distil-small.en",
            "asr_language": "en",
            "asr_task": "transcribe",
            "tgt_lang": "kn",
        }
        with mock.patch.object(main_app, "DesktopLiveSubtitlePipeline") as factory:
            app_bridge.apply_asr_config(cfg, session)
        kwargs = factory.call_args.kwargs
        self.assertFalse(kwargs["capture_locally"])
        self.assertEqual(kwargs["tgt_lang"], "kn")
        self.assertEqual(kwargs["asr_language"], "en")
        factory.return_value.start.assert_called_once()

    def test_remote_mode_does_not_open_wasapi_device(self):
        pipeline = main_app.DesktopLiveSubtitlePipeline.__new__(
            main_app.DesktopLiveSubtitlePipeline
        )
        pipeline.capture_locally = False
        pipeline.device_index = None
        pipeline._shutdown_event = threading.Event()
        pipeline.pipeline_thread = mock.Mock()
        pipeline.preprocessor = mock.Mock()
        pipeline.audio_queue = mock.Mock()
        pipeline.is_capturing = True
        pipeline._format_logged = False
        with mock.patch.object(main_app, "WASAPILoopbackCapture") as capture:
            pipeline.start()
        capture.assert_not_called()
        self.assertIsNone(pipeline.loopback_capture)
        pipeline.pipeline_thread.start.assert_called_once()

    def test_no_global_pipeline_or_broadcast_in_bridge(self):
        source = (ROOT / "backend" / "app_bridge.py").read_text(encoding="utf-8")
        self.assertNotIn("broadcast_json", source)
        self.assertNotIn("global pipeline", source)
        self.assertIn("per-client", source)


class TestSmallFixes(unittest.TestCase):
    def test_restore_punctuation_is_a_method(self):
        engine = SileroPunctuationEngine.__new__(SileroPunctuationEngine)
        engine.model = None
        engine.apply_te = None
        engine.proper_noun_map = {"python": "Python"}
        self.assertTrue(hasattr(engine, "restore_punctuation"))
        self.assertEqual(engine.restore_punctuation("hello world"), "Hello world.")

    def test_requirements_cover_translation_stack(self):
        root = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        backend = (ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8").lower()
        for name in ("torch", "onnxruntime", "huggingface_hub", "tokenizers",
                     "indictranstoolkit", "indic-nlp-library", "psutil"):
            self.assertIn(name.lower(), root, f"{name} missing from requirements.txt")
            self.assertIn(name.lower(), backend, f"{name} missing from backend/requirements.txt")

    def test_final_notebook_does_not_reset_checked_out_code(self):
        nb = json.loads((ROOT / "Me1.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(nb["cells"]), 10)
        sources = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        self.assertNotIn("git reset --hard", sources)
        self.assertIn("backend/app_bridge.py", sources)
        self.assertIn("'--port', '8766'", sources)
        self.assertIn("'PROXY_PORT'] = '8765'", sources)


if __name__ == "__main__":
    unittest.main()
