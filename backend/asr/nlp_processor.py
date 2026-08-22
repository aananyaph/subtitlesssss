"""NLP Post-Processing Quality Layer for Final Transcript Segments.

Applies:
  1. Hallucination / Safety filtering (no_speech_prob & non-ASCII ratio for English).
  2. Exact and near-duplicate segment filtering.
  3. Local Silero/Rule-based Punctuation and Capitalization Restoration.

Executes ONLY on FINAL transcript segments. Partial transcripts bypass this layer to preserve fast streaming updates.
"""
import re
import time
import traceback
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple


class SileroPunctuationEngine:
    """Local CPU Punctuation and Capitalization Restoration Engine."""

    def __init__(self):
        self.model = None
        self._load_model()

        # Dictionary of common technical proper nouns for clean capitalization
        self.proper_noun_map = {
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "python": "Python",
            "chrome": "Chrome",
            "youtube": "YouTube",
            "netflix": "Netflix",
            "github": "GitHub",
            "html": "HTML",
            "css": "CSS",
            "api": "API",
            "json": "JSON",
            "vad": "VAD",
            "asr": "ASR",
            "cpu": "CPU",
            "gpu": "GPU",
            "ram": "RAM",
            "openvino": "OpenVINO",
        }

    def _load_model(self):
        """Attempts loading Silero Punctuation ONNX/Torch model with safe fallback."""
        try:
            import torch
            model, example_texts, languages, punct, apply_te = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_te',
                trust_repo=True
            )
            self.model = model
            self.apply_te = apply_te
            self.punct = punct
            print("[NLP Engine] Loaded Silero Punctuation & Capitalization model successfully.")
        except Exception as e:
            print(f"[NLP Engine] Silero model PyTorch hub loading note ({e}) — using fast local ONNX/rule-based engine.")
            self.model = None
            self.apply_te = None

def restore_punctuation(self, text: str) -> str:
        """Restores sentence capitalization, proper nouns, and boundary punctuation using Silero text enhancement model."""
        if not text or not text.strip():
            return ""

        raw_text = text.strip()

        # If PyTorch Silero Punctuation model is loaded and apply_te is available
        if self.model is not None and self.apply_te is not None:
            try:
                punctuated = self.apply_te(raw_text, self.punct)
                return punctuated
            except Exception as exc:
                print(f"[NLP] Silero model error ({exc}), using rule-based fallback")
                # Fall through to rule-based engine

        # Fast rule-based Punctuation & Capitalization Engine
        words = raw_text.split()
        if not words:
            return raw_text

        # 1. Capitalize proper nouns
        cleaned_words = []
        for w in words:
            clean_word = w.lower().strip(",.!?")
            if clean_word in self.proper_noun_map:
                mapped = self.proper_noun_map[clean_word]
                # Preserve existing trailing punctuation
                if w[-1] in ",.!?":
                    mapped += w[-1]
                cleaned_words.append(mapped)
            else:
                cleaned_words.append(w)

        # 2. Capitalize first letter of sentence
        if cleaned_words:
            cleaned_words[0] = cleaned_words[0].capitalize()

        result = " ".join(cleaned_words)

        # 3. Restore question mark if starts with interrogative word
        first_word_lower = words[0].lower().strip(",.!?")
        interrogatives = {"what", "why", "how", "when", "where", "who", "which", "is", "are", "can", "could", "would", "do", "does", "did"}

        if result[-1] not in ".!?":
            if first_word_lower in interrogatives:
                result += "?"
            else:
                result += "."

        return result


class NLPPostProcessor:
    """Post-processing pipeline for FINAL transcript segments."""

    def __init__(
        self,
        punctuation_enabled: bool = True,
        hallucination_filter_enabled: bool = True,
        duplicate_filter_enabled: bool = True,
        no_speech_prob_threshold: float = 0.60,
        non_ascii_threshold: float = 0.50,
        duplicate_similarity_threshold: float = 0.90,
    ):
        self.punctuation_enabled = punctuation_enabled
        self.hallucination_filter_enabled = hallucination_filter_enabled
        self.duplicate_filter_enabled = duplicate_filter_enabled

        self.no_speech_prob_threshold = no_speech_prob_threshold
        self.non_ascii_threshold = non_ascii_threshold
        self.duplicate_similarity_threshold = duplicate_similarity_threshold

        self.punctuation_engine = SileroPunctuationEngine() if punctuation_enabled else None
        self.last_final_normalized: Optional[str] = None

    def _normalize(self, text: str) -> str:
        """Normalizes text for duplicate comparisons (lowercase, strip punctuation/spaces)."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def process_final_transcript(
        self,
        text: str,
        language: str = "en",
        no_speech_prob: float = 0.0
    ) -> Dict:
        """Processes a FINAL transcript segment through hallucination, duplicate, and punctuation filters.
        
        Returns:
            dict containing:
              - 'suppressed': bool
              - 'reason': str or None
              - 'text': str (cleaned/punctuated)
              - 'nlp_latency_ms': float
        """
        t0 = time.perf_counter()

        if not text or not text.strip():
            return {
                "suppressed": True,
                "reason": "empty_text",
                "text": "",
                "nlp_latency_ms": (time.perf_counter() - t0) * 1000.0,
            }

        raw_text = text.strip()

        # ── 1. Hallucination Filter (no_speech_prob check) ────────────────────
        if self.hallucination_filter_enabled and no_speech_prob >= self.no_speech_prob_threshold:
            print(f"[FILTER] Suppressed segment (no_speech_prob={no_speech_prob:.2f} >= {self.no_speech_prob_threshold})")
            return {
                "suppressed": True,
                "reason": "no_speech_prob",
                "text": raw_text,
                "nlp_latency_ms": (time.perf_counter() - t0) * 1000.0,
            }

        # ── 2. English Non-ASCII Safety Check ────────────────────────────────
        if self.hallucination_filter_enabled and language == "en":
            non_ascii_count = sum(1 for char in raw_text if ord(char) > 127)
            non_ascii_ratio = non_ascii_count / float(max(1, len(raw_text)))
            if non_ascii_ratio > self.non_ascii_threshold:
                print(f"[FILTER] Suppressed non-ASCII hallucination (ratio={non_ascii_ratio:.2f} > {self.non_ascii_threshold})")
                return {
                    "suppressed": True,
                    "reason": "non_ascii_ratio",
                    "text": raw_text,
                    "nlp_latency_ms": (time.perf_counter() - t0) * 1000.0,
                }

        # ── 3. Duplicate Segment Filter ───────────────────────────────────────
        normalized = self._normalize(raw_text)
        if self.duplicate_filter_enabled and normalized and self.last_final_normalized:
            # Exact duplicate check
            if normalized == self.last_final_normalized:
                print(f"[FILTER] Suppressed exact duplicate segment: '{raw_text}'")
                return {
                    "suppressed": True,
                    "reason": "exact_duplicate",
                    "text": raw_text,
                    "nlp_latency_ms": (time.perf_counter() - t0) * 1000.0,
                }

            # Near-duplicate similarity check
            ratio = SequenceMatcher(None, normalized, self.last_final_normalized).ratio()
            if ratio >= self.duplicate_similarity_threshold:
                print(f"[FILTER] Suppressed near-duplicate segment (similarity={ratio:.2f} >= {self.duplicate_similarity_threshold}): '{raw_text}'")
                return {
                    "suppressed": True,
                    "reason": "near_duplicate",
                    "text": raw_text,
                    "nlp_latency_ms": (time.perf_counter() - t0) * 1000.0,
                }

        # Update last final normalized segment
        self.last_final_normalized = normalized

        # ── 4. Punctuation & Capitalization Restoration ──────────────────────
        punctuated_text = raw_text
        if self.punctuation_enabled and self.punctuation_engine:
            punctuated_text = self.punctuation_engine.restore_punctuation(raw_text)
            print(f"[NLP] raw='{raw_text}' -> punctuated='{punctuated_text}'")

        nlp_latency = (time.perf_counter() - t0) * 1000.0
        print(f"[NLP] Final transcript processed in {nlp_latency:.2f}ms.")

        return {
            "suppressed": False,
            "reason": None,
            "text": punctuated_text,
            "nlp_latency_ms": nlp_latency,
        }
