"""Audio Preprocessing Utilities for Voice Activity Detection (VAD).

Handles format conversions, stereo-to-mono downmixing, sample rate resampling,
and frame slicing required for Silero VAD (16kHz mono Float32).
"""
import numpy as np

try:
    from scipy.signal import resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def int16_to_float32(pcm_int16: np.ndarray) -> np.ndarray:
    """Converts Int16 PCM array [-32768, 32767] to normalized Float32 array [-1.0, 1.0]."""
    if pcm_int16.dtype != np.int16:
        pcm_int16 = np.frombuffer(pcm_int16, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32768.0


def stereo_to_mono(interleaved_stereo: np.ndarray) -> np.ndarray:
    """Converts interleaved 2-channel stereo Float32 array to 1-channel mono Float32 array.
    
    Calculation: (left_channel + right_channel) / 2.0
    """
    if len(interleaved_stereo) % 2 != 0:
        # Truncate incomplete trailing sample if any
        interleaved_stereo = interleaved_stereo[:len(interleaved_stereo) - (len(interleaved_stereo) % 2)]
    
    left = interleaved_stereo[0::2]
    right = interleaved_stereo[1::2]
    return (left + right) * 0.5


def resample_audio(audio_mono: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resamples mono Float32 audio array from orig_sr to target_sr (default 16000 Hz).
    Uses scipy.signal.resample_poly for anti-aliased polyphase downsampling.
    """
    if orig_sr == target_sr or len(audio_mono) == 0:
        return audio_mono.astype(np.float32)

    if HAS_SCIPY:
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        resampled = resample_poly(audio_mono, up, down)
        return resampled.astype(np.float32)
    else:
        # Fallback to integer decimation or interpolation if scipy is unavailable
        if orig_sr % target_sr == 0:
            factor = orig_sr // target_sr
            return audio_mono[::factor].astype(np.float32)
        duration = len(audio_mono) / float(orig_sr)
        num_target_samples = int(np.round(duration * target_sr))
        orig_indices = np.linspace(0, len(audio_mono) - 1, num=len(audio_mono))
        target_indices = np.linspace(0, len(audio_mono) - 1, num=num_target_samples)
        resampled = np.interp(target_indices, orig_indices, audio_mono)
        return resampled.astype(np.float32)



class AudioPreprocessor:
    """Stream audio buffer that ingests arbitrary PCM packets (e.g. 48kHz stereo Int16),
    preprocesses them to 16kHz mono Float32, and emits fixed 512-sample (32ms) frames
    expected by Silero VAD.
    """

    def __init__(self, input_sample_rate: int = 48000, num_channels: int = 2, target_sample_rate: int = 16000, frame_size: int = 512):
        self.input_sample_rate = input_sample_rate
        self.num_channels = num_channels
        self.target_sample_rate = target_sample_rate
        self.frame_size = frame_size  # 512 samples @ 16kHz = 32 ms frame
        self._buffer = np.empty(0, dtype=np.float32)

    def set_format(self, sample_rate: int, num_channels: int):
        """Dynamically updates input format parameters (e.g. from WebSocket metadata handshake)."""
        self.input_sample_rate = sample_rate
        self.num_channels = num_channels

    def process_raw_bytes(self, raw_bytes: bytes) -> list[np.ndarray]:
        """Ingests raw binary Int16 PCM bytes from WebSocket, converts, resamples,
        and returns a list of 512-sample (32ms @ 16kHz) Float32 numpy arrays.
        """
        int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
        return self.process_int16(int16_arr)

    def process_int16(self, int16_arr: np.ndarray) -> list[np.ndarray]:
        """Ingests Int16 PCM numpy array."""
        float32_arr = int16_to_float32(int16_arr)
        
        if self.num_channels == 2:
            mono_arr = stereo_to_mono(float32_arr)
        else:
            mono_arr = float32_arr
            
        resampled_16k = resample_audio(mono_arr, self.input_sample_rate, self.target_sample_rate)
        
        # Append resampled audio to internal ring buffer
        self._buffer = np.concatenate((self._buffer, resampled_16k))
        
        # Extract all complete 512-sample frames
        frames = []
        while len(self._buffer) >= self.frame_size:
            frame = self._buffer[:self.frame_size]
            self._buffer = self._buffer[self.frame_size:]
            frames.append(frame)
            
        return frames

    def process_float32_mono_16k(self, float32_mono_16k: np.ndarray) -> list[np.ndarray]:
        """Ingests already normalized 16kHz mono Float32 audio directly."""
        self._buffer = np.concatenate((self._buffer, float32_mono_16k.astype(np.float32)))
        frames = []
        while len(self._buffer) >= self.frame_size:
            frame = self._buffer[:self.frame_size]
            self._buffer = self._buffer[self.frame_size:]
            frames.append(frame)
        return frames

    def reset(self):
        """Clears internal ring buffer."""
        self._buffer = np.empty(0, dtype=np.float32)
