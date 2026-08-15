"""Speech Temporal Hysteresis Segmenter.

Converts frame-by-frame VAD probabilities into continuous SPEECH START and SPEECH END events
using dual-threshold hysteresis and minimum duration smoothing to avoid speech fragmentation.
"""
from enum import Enum
import numpy as np


class SpeechState(Enum):
    NON_SPEECH = "NON_SPEECH"
    SPEECH = "SPEECH"


class SpeechSegmenter:
    """Stateful Speech Segmenter with hysteresis dual-thresholding and temporal smoothing."""

    def __init__(
        self,
        threshold: float = 0.50,
        neg_threshold: float = 0.35,
        min_speech_duration_ms: float = 250.0,
        min_silence_duration_ms: float = 400.0,
        sample_rate: int = 16000,
        frame_size: int = 512,
        min_chunk_floor_sec: float = 1.0,
    ):
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.min_chunk_floor_sec = min_chunk_floor_sec

        self.ms_per_frame = (frame_size / float(sample_rate)) * 1000.0  # ~32ms per 512 samples @ 16kHz
        self.min_speech_frames = int(np.ceil(min_speech_duration_ms / self.ms_per_frame))
        self.min_silence_frames = int(np.ceil(min_silence_duration_ms / self.ms_per_frame))

        self.state = SpeechState.NON_SPEECH
        self.speech_counter = 0
        self.silence_counter = 0

        self.current_segment_frames = []
        self.total_frames_processed = 0
        self.total_speech_duration_s = 0.0

    def process_frame(self, frame_audio: np.ndarray, prob: float) -> dict:
        """Processes a single 512-sample (32ms) frame with its VAD probability score.
        
        Returns:
            dict containing event status ('type': 'NO_CHANGE' | 'SPEECH_START' | 'SPEECH_CONTINUE' | 'SPEECH_END', details)
        """
        self.total_frames_processed += 1
        is_speech_frame = prob >= self.threshold
        is_silence_frame = prob < self.neg_threshold

        event = {"type": "NO_CHANGE", "state": self.state.value, "probability": prob}

        if self.state == SpeechState.NON_SPEECH:
            if is_speech_frame:
                self.speech_counter += 1
                self.current_segment_frames.append(frame_audio)

                if self.speech_counter >= self.min_speech_frames:
                    # Transition NON_SPEECH -> SPEECH
                    self.state = SpeechState.SPEECH
                    self.silence_counter = 0
                    event = {
                        "type": "SPEECH_START",
                        "state": self.state.value,
                        "probability": prob,
                        "frame_index": self.total_frames_processed
                    }
            else:
                self.speech_counter = 0
                self.current_segment_frames.clear()

        elif self.state == SpeechState.SPEECH:
            self.current_segment_frames.append(frame_audio)

            if is_silence_frame:
                self.silence_counter += 1
                if self.silence_counter >= self.min_silence_frames:
                    # Transition SPEECH -> NON_SPEECH
                    self.state = SpeechState.NON_SPEECH
                    
                    # Trim trailing silence frames
                    speech_frames_only = self.current_segment_frames[:-self.min_silence_frames]
                    if not speech_frames_only:
                        speech_frames_only = self.current_segment_frames

                    segment_pcm = np.concatenate(speech_frames_only) if speech_frames_only else np.empty(0, dtype=np.float32)
                    duration_s = len(segment_pcm) / float(self.sample_rate)
                    self.total_speech_duration_s += duration_s

                    event = {
                        "type": "SPEECH_END",
                        "state": self.state.value,
                        "probability": prob,
                        "duration_seconds": duration_s,
                        "audio_pcm": segment_pcm,
                        "frame_count": len(speech_frames_only)
                    }

                    self.speech_counter = 0
                    self.silence_counter = 0
                    self.current_segment_frames.clear()
            else:
                self.silence_counter = 0
                event = {
                    "type": "SPEECH_CONTINUE",
                    "state": self.state.value,
                    "probability": prob
                }

        return event

    def reset(self):
        """Resets segmenter state."""
        self.state = SpeechState.NON_SPEECH
        self.speech_counter = 0
        self.silence_counter = 0
        self.current_segment_frames.clear()
