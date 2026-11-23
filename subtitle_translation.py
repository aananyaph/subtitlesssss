"""Shared IndicTrans2 translation helper.

Used by both the local live pipeline and the offline batch pipeline so that a
translated subtitle is produced in every mode, not only through the WebSocket
proxy.  The heavy work is delegated to ``translation_proxy.translate`` which
already provides GPU -> ONNX fallback, model caching and a text cache.
"""

from typing import Dict, List, Optional

# Reuse the proxy's model cache and fallback logic instead of loading a second
# copy of IndicTrans2 in this process.
from translation_proxy import translate as _proxy_translate

SUPPORTED_LANGS = ("en", "hi", "kn")


def needs_translation(src_lang: Optional[str], tgt_lang: Optional[str]) -> bool:
    """True only when a real cross-language translation is requested."""
    if not src_lang or not tgt_lang:
        return False
    src = str(src_lang).strip().lower()
    tgt = str(tgt_lang).strip().lower()
    return src in SUPPORTED_LANGS and tgt in SUPPORTED_LANGS and src != tgt


def translate_text(text: str, src_lang: Optional[str], tgt_lang: Optional[str]) -> str:
    """Translate one subtitle string; return it unchanged when no translation is needed."""
    if not text or not text.strip() or not needs_translation(src_lang, tgt_lang):
        return text
    return _proxy_translate(text, str(src_lang).strip().lower(), str(tgt_lang).strip().lower())


def translate_segments(
    segments: List[Dict], src_lang: Optional[str], tgt_lang: Optional[str]
) -> List[Dict]:
    """Translate the ``text`` field of each segment in place and return the list."""
    if not needs_translation(src_lang, tgt_lang):
        return segments
    for segment in segments:
        if isinstance(segment, dict) and "text" in segment:
            segment["text"] = translate_text(segment["text"], src_lang, tgt_lang)
    return segments
