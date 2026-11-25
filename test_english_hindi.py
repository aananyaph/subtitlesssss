"""Regression tests for English audio -> Hindi/Kannada subtitle translation.

Run with:  python -m unittest -v test_english_hindi
No model downloads or network calls: the heavy translator is mocked.
"""

import json
import unittest
from unittest import mock

import numpy as np

import batch_pipeline
import main_app
import subtitle_translation
import translation_proxy
from app.control_panel import resolve_model_strategy
from app.remote_client import RemoteLiveSubtitlePipeline


class FakeStreamingASR:
    def process_completed_segment(self, segment_audio, start_time, language, task):
        return {
            "type": "final",
            "text": "Real confidence is calm.",
            "language": "en",
        }


class FakeNLP:
    def process_final_transcript(self, text, language, no_speech_prob=0.0):
        return {"suppressed": False, "text": text, "reason": ""}


class FakeOverlay:
    def __init__(self):
        self.pushed = []

    def push_live_text(self, text, is_final=False):
        self.pushed.append((text, is_final))


class TestEnglishToHindi(unittest.TestCase):
    def test_01_control_panel_config(self):
        cfg = resolve_model_strategy("en", "hi", "auto")
        self.assertEqual(cfg["asr_language"], "en", "ASR must use the spoken language")
        self.assertEqual(cfg["tgt_lang"], "hi")
        self.assertEqual(cfg["asr_task"], "transcribe")
        self.assertNotIn("indic", cfg["model_name"])

    def test_02_handshake_keeps_target(self):
        cfg = resolve_model_strategy("en", "hi", "auto")
        client = RemoteLiveSubtitlePipeline(asr_config=dict(cfg), tgt_lang=cfg["tgt_lang"])
        payload = json.loads(client._build_handshake_payload())
        self.assertEqual(payload["asrConfig"]["asr_language"], "en")
        self.assertEqual(payload["asrConfig"]["tgt_lang"], "hi")

    def test_03_proxy_enables_translation(self):
        cfg = resolve_model_strategy("en", "hi", "auto")
        client = RemoteLiveSubtitlePipeline(asr_config=dict(cfg), tgt_lang=cfg["tgt_lang"])
        handshake = client._build_handshake_payload()
        state = {"src": None, "tgt": None}
        with mock.patch.object(translation_proxy, "preload"):
            rewritten = json.loads(translation_proxy.rewrite_handshake(handshake, state))
        self.assertEqual(rewritten["asrConfig"]["asr_task"], "transcribe")
        self.assertEqual(state["src"], "en")
        self.assertEqual(state["tgt"], "hi")

    def test_04_same_language_needs_no_translation(self):
        self.assertFalse(subtitle_translation.needs_translation("en", "en"))
        self.assertFalse(subtitle_translation.needs_translation("hi", None))
        self.assertTrue(subtitle_translation.needs_translation("en", "hi"))

        with mock.patch.object(subtitle_translation, "_proxy_translate") as translate:
            self.assertEqual(
                subtitle_translation.translate_text("Hello", "en", "hi"),
                translate.return_value,
            )
            translate.assert_called_once_with("Hello", "en", "hi")

    def test_05_local_pipeline_translates_before_overlay(self):
        pipeline = main_app.DesktopLiveSubtitlePipeline.__new__(
            main_app.DesktopLiveSubtitlePipeline
        )
        overlay = FakeOverlay()
        pipeline.overlay = overlay
        pipeline.subtitle_callback = None
        pipeline.indic_lang = None
        pipeline.asr_language = "en"
        pipeline.asr_task = "transcribe"
        pipeline.tgt_lang = "hi"
        pipeline.last_detected_language = None
        pipeline.streaming_asr = FakeStreamingASR()
        pipeline.nlp_processor = FakeNLP()
        pipeline.llm_corrector = None
        pipeline._seg_counter = 0

        with mock.patch.object(main_app, "translate_text", return_value="वास्तविक आत्मविश्वास शांत है।") as tr:
            pipeline._process_completed_segment(np.zeros(16000, dtype=np.float32), 0.0)

        tr.assert_called_once()
        self.assertEqual(overlay.pushed[0][0], "वास्तविक आत्मविश्वास शांत है।")

    def test_06_batch_translates_segments(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        with mock.patch.object(batch_pipeline.core_engine, "transcribe") as transcribe, \
             mock.patch.object(batch_pipeline, "get_duration", return_value=1.0), \
             mock.patch.object(batch_pipeline, "extract_audio"), \
             mock.patch.object(batch_pipeline, "translate_segments") as translate, \
             mock.patch.object(batch_pipeline.subtitle_formatter, "save_subtitle", return_value="out.srt") as save:
            transcribe.return_value = {"language": "en", "segments": segments}
            result = batch_pipeline.process_file(
                "samples/english_clip.wav",
                output_path="out.srt",
                output_format="srt",
                language="en",
                target_language="hi",
            )

        translate.assert_called_once_with(segments, "en", "hi")
        self.assertEqual(result["output_path"], "out.srt")
        save.assert_called_once()

    def test_07_control_panel_config_en_kn(self):
        cfg = resolve_model_strategy("en", "kn", "auto")
        self.assertEqual(cfg["asr_language"], "en", "ASR must use the spoken language")
        self.assertEqual(cfg["tgt_lang"], "kn")
        self.assertEqual(cfg["asr_task"], "transcribe")
        self.assertNotIn("indic", cfg["model_name"])

    def test_08_handshake_en_kn(self):
        cfg = resolve_model_strategy("en", "kn", "auto")
        client = RemoteLiveSubtitlePipeline(asr_config=dict(cfg), tgt_lang=cfg["tgt_lang"])
        payload = json.loads(client._build_handshake_payload())
        self.assertEqual(payload["asrConfig"]["asr_language"], "en")
        self.assertEqual(payload["asrConfig"]["tgt_lang"], "kn")

    def test_09_proxy_enables_translation_en_kn(self):
        cfg = resolve_model_strategy("en", "kn", "auto")
        client = RemoteLiveSubtitlePipeline(asr_config=dict(cfg), tgt_lang=cfg["tgt_lang"])
        state = {"src": None, "tgt": None}
        with mock.patch.object(translation_proxy, "preload"):
            rewritten = json.loads(
                translation_proxy.rewrite_handshake(client._build_handshake_payload(), state)
            )
        self.assertEqual(rewritten["asrConfig"]["asr_task"], "transcribe")
        self.assertEqual(state["src"], "en")
        self.assertEqual(state["tgt"], "kn")

    def test_10_local_pipeline_translates_kn(self):
        pipeline = main_app.DesktopLiveSubtitlePipeline.__new__(
            main_app.DesktopLiveSubtitlePipeline
        )
        overlay = FakeOverlay()
        pipeline.overlay = overlay
        pipeline.subtitle_callback = None
        pipeline.indic_lang = None
        pipeline.asr_language = "en"
        pipeline.asr_task = "transcribe"
        pipeline.tgt_lang = "kn"
        pipeline.last_detected_language = None
        pipeline.streaming_asr = FakeStreamingASR()
        pipeline.nlp_processor = FakeNLP()
        pipeline.llm_corrector = None
        pipeline._seg_counter = 0

        with mock.patch.object(main_app, "translate_text", return_value="ನಿಜವಾದ ಆತ್ಮವಿಶ್ವಾಸ ಶಾಂತವಾಗಿದೆ.") as tr:
            pipeline._process_completed_segment(np.zeros(16000, dtype=np.float32), 0.0)

        tr.assert_called_once()
        self.assertEqual(overlay.pushed[0][0], "ನಿಜವಾದ ಆತ್ಮವಿಶ್ವಾಸ ಶಾಂತವಾಗಿದೆ.")

    def test_11_hindi_to_english_handshake_and_proxy(self):
        cfg = resolve_model_strategy("hi", "en", "auto")
        self.assertEqual(cfg["asr_language"], "hi")
        self.assertEqual(cfg["tgt_lang"], "en")
        self.assertEqual(cfg["indic_lang"], "hi")

        client = RemoteLiveSubtitlePipeline(asr_config=dict(cfg), tgt_lang=cfg["tgt_lang"])
        state = {"src": None, "tgt": None}
        with mock.patch.object(translation_proxy, "preload"):
            rewritten = json.loads(
                translation_proxy.rewrite_handshake(client._build_handshake_payload(), state)
            )
        self.assertEqual(rewritten["asrConfig"]["asr_task"], "transcribe")
        self.assertEqual(rewritten["asrConfig"]["model_name"], "indic")
        self.assertEqual(state["src"], "hi")
        self.assertEqual(state["tgt"], "en")

    def test_12_hindi_to_kannada_handshake_and_proxy(self):
        cfg = resolve_model_strategy("hi", "kn", "auto")
        self.assertEqual(cfg["asr_language"], "hi")
        self.assertEqual(cfg["tgt_lang"], "kn")
        self.assertEqual(cfg["indic_lang"], "hi")

        client = RemoteLiveSubtitlePipeline(asr_config=dict(cfg), tgt_lang=cfg["tgt_lang"])
        state = {"src": None, "tgt": None}
        with mock.patch.object(translation_proxy, "preload"):
            rewritten = json.loads(
                translation_proxy.rewrite_handshake(client._build_handshake_payload(), state)
            )
        self.assertEqual(rewritten["asrConfig"]["asr_task"], "transcribe")
        self.assertEqual(rewritten["asrConfig"]["model_name"], "indic")
        self.assertEqual(state["src"], "hi")
        self.assertEqual(state["tgt"], "kn")


if __name__ == "__main__":
    unittest.main()
