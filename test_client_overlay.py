"""test_client_overlay.py - Unit tests for client overlay, bilingual formatting, and remote client message handling."""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure subtitlesssss directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.overlay_window import wrap_text, DesktopSubtitleOverlay, QT_BINDING
if QT_BINDING:
    from PyQt6.QtWidgets import QApplication
else:
    QApplication = None

from app.remote_client import RemoteLiveSubtitlePipeline


class TestClientOverlayAndRemotePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if QT_BINDING:
            cls.qapp = QApplication.instance() or QApplication(sys.argv)
        else:
            cls.qapp = None

    def test_wrap_text_under_limit(self):
        short_text = "Hello world this is short"
        wrapped = wrap_text(short_text, max_chars=42, max_lines=2)
        self.assertEqual(wrapped, short_text)

    def test_wrap_text_wrapping_and_line_limit(self):
        long_text = "This is a very long sentence that definitely exceeds forty-two characters and should be wrapped into multiple lines properly."
        wrapped = wrap_text(long_text, max_chars=42, max_lines=2)
        lines = wrapped.split("\n")
        self.assertLessEqual(len(lines), 2)
        for line in lines:
            self.assertLessEqual(len(line), 42)

    def test_overlay_bilingual_rendering(self):
        overlay = DesktopSubtitleOverlay(show_source=True)
        overlay.label = MagicMock()

        # 1. Message with source_text and translated text
        overlay._on_update_text_with_source(
            text="Hello, how are you?",
            is_final=True,
            source_text="नमस्ते, आप कैसे हैं?",
        )
        overlay.label.setText.assert_called_once()
        html_rendered = overlay.label.setText.call_args[0][0]
        self.assertIn("नमस्ते, आप कैसे हैं?", html_rendered)
        self.assertIn("Hello, how are you?", html_rendered)
        self.assertIn("font-size: 18px", html_rendered)  # small source
        self.assertIn("font-size: 28px", html_rendered)  # large text

        # 2. Message without source_text (or empty)
        overlay.label.reset_mock()
        overlay._on_update_text_with_source(
            text="Hello world",
            is_final=True,
            source_text="",
        )
        html_rendered2 = overlay.label.setText.call_args[0][0]
        self.assertIn("Hello world", html_rendered2)
        self.assertNotIn("font-size: 18px", html_rendered2)

        # 3. Message with show_source=False
        overlay.set_show_source(False)
        overlay.label.reset_mock()
        overlay._on_update_text_with_source(
            text="Hello again",
            is_final=True,
            source_text="नमस्ते",
        )
        html_rendered3 = overlay.label.setText.call_args[0][0]
        self.assertIn("Hello again", html_rendered3)
        self.assertNotIn("नमस्ते", html_rendered3)

    def test_remote_client_handshake_payload(self):
        # 1. Different languages: asr_lang=kn, tgt_lang=en -> sends tgt_lang
        client_kn_en = RemoteLiveSubtitlePipeline(
            asr_config={"asr_language": "kn", "tgt_lang": "en"}
        )
        payload1 = json.loads(client_kn_en._build_handshake_payload())
        cfg1 = payload1["asrConfig"]
        self.assertEqual(cfg1["asr_language"], "kn")
        self.assertEqual(cfg1["tgt_lang"], "en")

        # 2. Equal languages: asr_lang=en, tgt_lang=en -> does NOT send tgt_lang (proxy bypasses)
        client_en_en = RemoteLiveSubtitlePipeline(
            asr_config={"asr_language": "en", "tgt_lang": "en"}
        )
        payload2 = json.loads(client_en_en._build_handshake_payload())
        cfg2 = payload2["asrConfig"]
        self.assertEqual(cfg2["asr_language"], "en")
        self.assertNotIn("tgt_lang", cfg2)

        # 3. Translation to Hindi: asr_lang=en, tgt_lang=hi -> sends tgt_lang: hi
        client_en_hi = RemoteLiveSubtitlePipeline(
            asr_config={"asr_language": "en", "tgt_lang": "hi"}
        )
        payload3 = json.loads(client_en_hi._build_handshake_payload())
        cfg3 = payload3["asrConfig"]
        self.assertEqual(cfg3["asr_language"], "en")
        self.assertEqual(cfg3["tgt_lang"], "hi")

    def test_remote_client_recv_messages(self):
        mock_overlay = MagicMock()
        mock_sub_cb = MagicMock()

        client = RemoteLiveSubtitlePipeline(
            overlay=mock_overlay,
            subtitle_callback=mock_sub_cb,
        )

        import asyncio

        async def simulate_messages():
            messages = [
                # A. Final message with source_text
                json.dumps({
                    "type": "final",
                    "text": "Hello, how are you?",
                    "source_text": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",
                    "language": "en",
                    "is_final": True
                }),
                # B. Partial message without source_text (SHOW_SOURCE_PARTIALS)
                json.dumps({
                    "type": "partial",
                    "text": "ನಮಸ್ಕಾರ",
                    "language": "kn",
                    "is_final": False
                }),
                # C. Final message without source_text (e.g. from upstream bridge)
                json.dumps({
                    "type": "final",
                    "text": "Good morning.",
                    "language": "en",
                    "is_final": True
                }),
                # D. Malformed / unexpected message shapes (must be tolerated without crashing)
                json.dumps({"unknown_key": "some_value"}),
                "not a json string at all",
                json.dumps({"type": "AUDIO_DIAGNOSTICS", "diagnostics": {"peak": 0.5, "vad_prob": 0.8, "state": "SPEECH"}})
            ]

            class FakeWebSocket:
                def __aiter__(self):
                    self._iter = iter(messages)
                    return self
                async def __anext__(self):
                    try:
                        return next(self._iter)
                    except StopIteration:
                        raise StopAsyncIteration

            await client._recv_subtitles_loop(FakeWebSocket())

        asyncio.run(simulate_messages())

        # Verify overlay received the calls
        self.assertGreaterEqual(mock_overlay.push_live_text.call_count, 3)

        # Call A: with source_text
        call_a = mock_overlay.push_live_text.call_args_list[0]
        self.assertEqual(call_a[0][0], "Hello, how are you?")
        self.assertTrue(call_a[1]["is_final"])
        self.assertEqual(call_a[1]["source_text"], "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?")

        # Call B: partial
        call_b = mock_overlay.push_live_text.call_args_list[1]
        self.assertEqual(call_b[0][0], "ನಮಸ್ಕಾರ")
        self.assertFalse(call_b[1]["is_final"])
        self.assertIsNone(call_b[1]["source_text"])

        # Call C: final without source_text
        call_c = mock_overlay.push_live_text.call_args_list[2]
        self.assertEqual(call_c[0][0], "Good morning.")
        self.assertTrue(call_c[1]["is_final"])
        self.assertIsNone(call_c[1]["source_text"])


if __name__ == "__main__":
    unittest.main()
