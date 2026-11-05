"""test_proxy_mocked.py - Unit tests for translation_proxy.py with mocked translators (no torch/CUDA required)."""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure subtitlesssss directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import translation_proxy as tp


class TestTranslationProxy(unittest.TestCase):

    def setUp(self):
        with tp._tr_lock:
            tp._model_cache.clear()
            tp._text_cache.clear()
            tp._pivot_logged = False
        tp.TRANSLATOR_BACKEND = "torch1b"

    def test_norm_lang(self):
        self.assertEqual(tp.norm_lang("kn"), "kn")
        self.assertEqual(tp.norm_lang("kan"), "kn")
        self.assertEqual(tp.norm_lang("kannada"), "kn")
        self.assertEqual(tp.norm_lang("hi"), "hi")
        self.assertEqual(tp.norm_lang("hindi"), "hi")
        self.assertEqual(tp.norm_lang("en"), "en")
        self.assertEqual(tp.norm_lang("english"), "en")
        self.assertIsNone(tp.norm_lang("auto"))
        self.assertIsNone(tp.norm_lang(""))
        self.assertIsNone(tp.norm_lang(None))

    @patch("translation_proxy.preload")
    def test_rewrite_handshake_auto_disabled(self, mock_preload):
        state = {"src": None, "tgt": None}
        raw = json.dumps({"type": "HANDSHAKE", "asrConfig": {"asr_language": "auto", "tgt_lang": "en"}})
        res = tp.rewrite_handshake(raw, state)
        self.assertIsNone(state["src"])
        self.assertIsNone(state["tgt"])
        self.assertEqual(raw, res)
        mock_preload.assert_not_called()

    @patch("translation_proxy.preload")
    def test_rewrite_handshake_kn_to_en(self, mock_preload):
        state = {"src": None, "tgt": None}
        raw = json.dumps({
            "type": "HANDSHAKE",
            "asrConfig": {
                "asr_language": "kn",
                "tgt_lang": "en",
            }
        })
        res = tp.rewrite_handshake(raw, state)
        data = json.loads(res)
        cfg = data["asrConfig"]
        self.assertEqual(state["src"], "kn")
        self.assertEqual(state["tgt"], "en")
        self.assertEqual(cfg["asr_task"], "transcribe")
        self.assertEqual(cfg["model_name"], "indic")
        self.assertEqual(cfg["indic_lang"], "kn")
        mock_preload.assert_called_once_with("kn", "en")

    @patch("translation_proxy.preload")
    def test_rewrite_handshake_en_to_hi(self, mock_preload):
        state = {"src": None, "tgt": None}
        raw = json.dumps({
            "type": "HANDSHAKE",
            "asrConfig": {
                "asr_language": "en",
                "tgt_lang": "hi",
            }
        })
        res = tp.rewrite_handshake(raw, state)
        data = json.loads(res)
        cfg = data["asrConfig"]
        self.assertEqual(state["src"], "en")
        self.assertEqual(state["tgt"], "hi")
        self.assertEqual(cfg["asr_task"], "transcribe")
        mock_preload.assert_called_once_with("en", "hi")

    def test_parse_subtitle_and_is_final(self):
        # final
        final_msg = json.dumps({"type": "final", "text": "Hello world", "is_final": True})
        parsed = tp.parse_subtitle(final_msg)
        self.assertIsNotNone(parsed)
        self.assertTrue(tp.is_final(parsed))

        # partial
        partial_msg = json.dumps({"type": "partial", "text": "Hello", "is_final": False})
        parsed_p = tp.parse_subtitle(partial_msg)
        self.assertIsNotNone(parsed_p)
        self.assertFalse(tp.is_final(parsed_p))

        # invalid / non-subtitle
        status_msg = json.dumps({"type": "status", "data": "ready"})
        self.assertIsNone(tp.parse_subtitle(status_msg))

    def test_text_lru_cache(self):
        tp._put_text_cache("en", "hi", "Hello", "नमस्ते")
        self.assertEqual(tp._check_text_cache("en", "hi", "Hello"), "नमस्ते")
        self.assertIsNone(tp._check_text_cache("en", "hi", "Goodbye"))

        # Test cache hit returns backend='cache' and cache_hit=True
        out, backend, hit = tp.translate_with_meta("Hello", "en", "hi")
        self.assertEqual(out, "नमस्ते")
        self.assertEqual(backend, "cache")
        self.assertTrue(hit)

    def test_gpu_translate_and_fallback_to_onnx(self):
        # Mock GPU model and ONNX model
        mock_gpu_model = MagicMock()
        mock_gpu_model.translate.return_value = "GPU Translated Text"

        mock_onnx_model = MagicMock()
        mock_onnx_model.translate.return_value = "ONNX Fallback Text"

        with tp._tr_lock:
            tp._model_cache[("torch1b", "ai4bharat/indictrans2-en-indic-1B")] = mock_gpu_model
            tp._model_cache[("onnx", "hari31416/indictrans2-en-indic-dist-200M-ONNX-int8")] = mock_onnx_model

        # 1. Successful GPU translation
        with patch("torch.cuda.is_available", return_value=True):
            out, backend, hit = tp.translate_with_meta("Good morning", "en", "hi")
            self.assertEqual(out, "GPU Translated Text")
            self.assertEqual(backend, "torch1b")
            self.assertFalse(hit)

        # 2. GPU translation fails -> falls back to ONNX
        mock_gpu_model.translate.side_effect = RuntimeError("CUDA OOM simulated")
        with patch("torch.cuda.is_available", return_value=True):
            out2, backend2, hit2 = tp.translate_with_meta("A new sentence", "en", "hi")
            self.assertEqual(out2, "ONNX Fallback Text")
            self.assertEqual(backend2, "onnx_fallback")
            self.assertFalse(hit2)

    def test_model_cache_lru_eviction(self):
        m1 = MagicMock()
        m2 = MagicMock()
        m3 = MagicMock()

        with tp._tr_lock:
            tp._model_cache[("torch1b", "repo1")] = m1
            tp._model_cache[("torch1b", "repo2")] = m2
            tp._evict_oldest_model_if_needed()
            self.assertEqual(len(tp._model_cache), 2)

            # Access repo1 to make repo2 the oldest
            tp._model_cache.move_to_end(("torch1b", "repo1"))

            # Add third model
            tp._model_cache[("torch1b", "repo3")] = m3
            tp._evict_oldest_model_if_needed()

            self.assertEqual(len(tp._model_cache), 2)
            self.assertNotIn(("torch1b", "repo2"), tp._model_cache)
            self.assertIn(("torch1b", "repo1"), tp._model_cache)
            self.assertIn(("torch1b", "repo3"), tp._model_cache)


if __name__ == "__main__":
    unittest.main()
