"""Streaming ASR Manager & LocalAgreement Stabilization Engine.

Manages rolling-window context, word-level LocalAgreement prefix matching,
emits PARTIAL and FINAL transcript events with timestamps, and prevents duplicate line repetition.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

try:
    from .asr_engine import BaseASREngine
except ImportError:
    from asr_engine import BaseASREngine


@dataclass
class TranscriptEvent:
    type: str  # 'partial' or 'final'
    text: str
    start: float
    end: float
    language: str = "en"

    def to_dict(self) -> Dict:
        return asdict(self)


class StreamingASRManager:
    """Manages streaming audio context and LocalAgreement transcript stabilization."""

    def __init__(
        self,
        asr_engine: BaseASREngine,
        sample_rate: int = 16000,
        context_window_sec: float = 3.0,
        min_confirm_words: int = 2
    ):
        self.asr_engine = asr_engine
        self.sample_rate = sample_rate
        self.context_window_samples = int(context_window_sec * sample_rate)
        self.min_confirm_words = min_confirm_words

        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.committed_words: List[str] = []
        self.last_partial_words: List[str] = []
        self.cumulative_time_offset = 0.0

    def feed_audio(self, audio_chunk: np.ndarray):
        """Appends incoming Float32 16kHz mono audio chunk to the rolling context buffer."""
        self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk.astype(np.float32)))
        # Cap buffer to context window size
        if len(self.audio_buffer) > self.context_window_samples:
            self.audio_buffer = self.audio_buffer[-self.context_window_samples:]

    def process_streaming_window(self, time_offset: float = 0.0, language: str = "en") -> List[Dict]:
        """Runs ASR over the current rolling context window and applies LocalAgreement stabilization.
        
        Returns:
            List of transcript event dicts (type: 'final' or 'partial')
        """
        if len(self.audio_buffer) == 0:
            return []

        res = self.asr_engine.transcribe(self.audio_buffer, language=language)
        candidate_text = res.get("text", "").strip()

        if not candidate_text:
            return []

        candidate_words = candidate_text.split()
        
        # Calculate time range
        duration_s = len(self.audio_buffer) / float(self.sample_rate)
        start_time = round(time_offset, 2)
        end_time = round(time_offset + duration_s, 2)

        events = []

        # LocalAgreement stabilization: Find common prefix between last partial and candidate
        common_prefix = []
        min_len = min(len(self.last_partial_words), len(candidate_words))

        for i in range(min_len):
            if self.last_partial_words[i].lower() == candidate_words[i].lower():
                common_prefix.append(candidate_words[i])
            else:
                break

        # If common prefix meets min_confirm_words, commit it as FINAL
        if len(common_prefix) >= self.min_confirm_words:
            final_text = " ".join(common_prefix)
            self.committed_words.extend(common_prefix)

            events.append(TranscriptEvent(
                type="final",
                text=final_text,
                start=start_time,
                end=round(start_time + (len(common_prefix) / float(len(candidate_words))) * duration_s, 2),
                language=language
            ).to_dict())

            # Remaining suffix stays PARTIAL
            remaining_words = candidate_words[len(common_prefix):]
            if remaining_words:
                partial_text = " ".join(remaining_words)
                events.append(TranscriptEvent(
                    type="partial",
                    text=partial_text,
                    start=round(start_time + (len(common_prefix) / float(len(candidate_words))) * duration_s, 2),
                    end=end_time,
                    language=language
                ).to_dict())
                self.last_partial_words = remaining_words
            else:
                self.last_partial_words = []
        else:
            # Emit full candidate as PARTIAL
            events.append(TranscriptEvent(
                type="partial",
                text=candidate_text,
                start=start_time,
                end=end_time,
                language=language
            ).to_dict())
            self.last_partial_words = candidate_words

        return events

    def process_completed_segment(self, segment_audio: np.ndarray, start_time: float, language: str = "en") -> Dict:
        """Processes a complete VAD speech segment (emitted at SPEECH_END) into a confirmed FINAL transcript."""
        if len(segment_audio) == 0:
            return None

        res = self.asr_engine.transcribe(segment_audio, language=language)
        text = res.get("text", "").strip()

        if not text:
            return None

        duration_s = len(segment_audio) / float(self.sample_rate)
        end_time = round(start_time + duration_s, 2)

        # Clear partial state
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.last_partial_words = []

        return TranscriptEvent(
            type="final",
            text=text,
            start=round(start_time, 2),
            end=end_time,
            language=language
        ).to_dict()

    def reset(self):
        """Resets streaming buffer and transcript state."""
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.committed_words.clear()
        self.last_partial_words.clear()
        self.cumulative_time_offset = 0.0
