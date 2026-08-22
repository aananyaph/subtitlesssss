"""Module 9 — Async Local LLM Subtitle Post-Correction Layer.

Provides LLMCorrector using llama-cpp-python for safe, asynchronous local GGUF
subtitle post-correction without blocking the primary ASR pipeline.
"""

import os
import sys
import time
import logging
import re
from typing import Optional, Dict, Any

# Suppress debug noise from llama_cpp if needed
logging.getLogger("llama_cpp").setLevel(logging.WARNING)


class LLMCorrector:
    """Async Local LLM Post-Correction Engine using llama-cpp-python.

    Features:
        • One-time CPU model loading.
        • Safe fallback if llama-cpp-python or GGUF model is missing.
        • Short deterministic system prompt for zero-elaboration correction.
        • Output validation to reject hallucinated, conversational, or bad output.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        enabled: bool = False,
        n_ctx: int = 512,
        n_threads: int = 2,
    ):
        self.enabled = enabled
        self.model_path = model_path or os.environ.get("LLM_MODEL_PATH", "")
        self.n_ctx = n_ctx
        self.n_threads = n_threads

        self.llm = None
        self.is_available = False

        if self.enabled:
            self._load_model()
        else:
            logging.info("[LLM] Async LLM correction disabled by feature flag.")

    def _load_model(self):
        """Loads GGUF model once via llama-cpp-python."""
        if not self.model_path:
            logging.warning("[LLM] Disabled: No LLM_MODEL_PATH specified.")
            self.is_available = False
            return

        if not os.path.exists(self.model_path):
            logging.warning(f"[LLM] Disabled: Model file not found at '{self.model_path}'.")
            self.is_available = False
            return

        try:
            from llama_cpp import Llama  # Lazy import
            t0 = time.perf_counter()
            logging.info(f"[LLM] Loading GGUF model from '{self.model_path}' (threads={self.n_threads}, ctx={self.n_ctx})...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            elapsed = round(time.perf_counter() - t0, 2)
            self.is_available = True
            logging.info(f"[LLM] GGUF model loaded successfully in {elapsed}s.")
        except ImportError:
            logging.warning("[LLM] Disabled: 'llama-cpp-python' package is not installed.")
            self.is_available = False
        except Exception as exc:
            logging.error(f"[LLM] Disabled: Failed to load model '{self.model_path}': {exc}", exc_info=True)
            self.is_available = False

    def correct(self, text: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Runs LLM post-correction on original text.

        Returns:
            Dict containing {"corrected_text": str, "inference_ms": float}
            or None if correction was skipped, failed, or rejected by validation.
        """
        if not self.is_available or not self.llm or not text or not text.strip():
            return None

        clean_text = text.strip()

        # Construct short, deterministic prompt
        prompt = (
            "You are correcting an automatic speech recognition subtitle.\n\n"
            "Fix only:\n"
            "- obvious spelling errors\n"
            "- obvious ASR word errors\n"
            "- grammar errors\n"
            "- punctuation\n\n"
            "Preserve the exact meaning.\n"
            "Do not add information.\n"
            "Do not summarize.\n"
            "Do not explain.\n"
            "Return ONLY the corrected subtitle.\n\n"
            f"Subtitle: {clean_text}\n"
            "Corrected:"
        )

        t0 = time.perf_counter()
        try:
            res = self.llm(
                prompt,
                max_tokens=64,
                stop=["\n", "Subtitle:", "Explanation:", "Note:"],
                temperature=0.1,
                top_p=0.9,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            raw_out = res["choices"][0]["text"].strip()
            validated_text = self.validate_output(clean_text, raw_out)

            if validated_text is None:
                logging.debug(f"[LLM Validator] Rejected output for '{clean_text}': raw='{raw_out}'")
                return None

            return {
                "corrected_text": validated_text,
                "inference_ms": elapsed_ms,
            }

        except Exception as exc:
            logging.warning(f"[LLM Worker] Inference error for '{clean_text}': {exc}", exc_info=True)
            return None

    def validate_output(self, original: str, generated: str) -> Optional[str]:
        """Validates LLM output string to prevent invalid/hallucinated overlays.

        Rejection rules:
            1. Empty output.
            2. Contains explanations/conversational intros ("Here is", "Sure", "Note:").
            3. Contains markdown code blocks or backticks.
            4. Excessive length (> 2x original length + 15 chars).
            5. Substantially unchanged text (returns original clean string or None if identical).
        """
        if not generated:
            return None

        # Clean surrounding quotes
        candidate = generated.strip()
        if (candidate.startswith('"') and candidate.endswith('"')) or (
            candidate.startswith("'") and candidate.endswith("'")
        ):
            candidate = candidate[1:-1].strip()

        if not candidate:
            return None

        # Rejection Rule 2: Conversational or explanatory prefixes
        lower = candidate.lower()
        forbidden_prefixes = (
            "here is",
            "here's",
            "sure",
            "note:",
            "explanation:",
            "corrected:",
            "subtitle:",
            "the corrected",
        )
        if any(lower.startswith(prefix) for prefix in forbidden_prefixes):
            return None

        # Rejection Rule 3: Markdown formatting
        if "`" in candidate or "```" in candidate or "##" in candidate:
            return None

        # Rejection Rule 4: Length check (prevent added details/summaries)
        if len(candidate) > (len(original) * 2 + 20):
            return None

        # If generated is identical to original, no update needed
        if candidate == original:
            return None

        return candidate
