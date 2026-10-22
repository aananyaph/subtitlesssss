"""WASAPI Loopback Audio Capture Engine for Windows.

Uses pyaudiowpatch to capture system-wide audio output (YouTube, Netflix, VLC, etc.)
via non-blocking WASAPI loopback and feeds raw PCM bytes into the VAD/ASR pipeline.
"""

import logging
import os
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False


class WASAPILoopbackCapture(threading.Thread):
    """Threaded WASAPI Loopback Audio Capture engine."""

    def __init__(
        self,
        audio_callback: Callable[[bytes, int, int], None],
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        super().__init__(name="WASAPILoopbackThread", daemon=True)
        self.audio_callback = audio_callback
        self.chunk_size = chunk_size
        self.device_index = device_index
        self._stop_event = threading.Event()

        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.stream = None
        self.sample_rate = 48000
        self.num_channels = 2

    def _find_loopback_device(self, p: pyaudio.PyAudio) -> dict:
        """Find default WASAPI loopback device info dict."""
        if self.device_index is not None:
            try:
                info = p.get_device_info_by_index(self.device_index)
                logging.info(f"[Loopback] Using specified device index {self.device_index}: '{info.get('name')}'")
                return info
            except Exception as e:
                logging.warning(
                    f"[Loopback] Specified device index {self.device_index} is invalid or unavailable ({e}). "
                    "Falling back to auto-detecting active WASAPI loopback device."
                )

        # 1. Use pyaudiowpatch's get_default_wasapi_loopback() for active default output
        if hasattr(p, "get_default_wasapi_loopback"):
            try:
                device = p.get_default_wasapi_loopback()
                if device:
                    return device
            except Exception as e:
                logging.warning(f"[Loopback] get_default_wasapi_loopback failed: {e}")

        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            raise RuntimeError("WASAPI host API not available on this system.")

        # 2. Fallback: match name of defaultOutputDevice with isLoopbackDevice
        try:
            default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            default_name = default_out.get("name", "").lower()
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if (
                    info.get("hostApi") == wasapi_info["index"]
                    and info.get("isLoopbackDevice")
                    and default_name in info.get("name", "").lower()
                ):
                    return info
        except Exception as e:
            logging.warning(f"[Loopback] name-match fallback failed: {e}")

        # 3. Fallback: search all loopback devices
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                if info.get("hostApi") == wasapi_info["index"] and info.get("isLoopbackDevice"):
                    return info
            except Exception:
                continue

        raise RuntimeError("No WASAPI loopback device found. Ensure audio is playing.")

    def run(self):
        if not HAS_PYAUDIO:
            logging.error("[Loopback] pyaudiowpatch is not installed.")
            return

        logging.info("[Loopback] Initializing WASAPI loopback audio capture...")
        self.pyaudio_instance = pyaudio.PyAudio()
        self._first_chunk_logged = False
        self._read_err_logged = False

        try:
            device = self._find_loopback_device(self.pyaudio_instance)
            self.sample_rate = int(device.get("defaultSampleRate", 48000))
            self.num_channels = int(device.get("maxInputChannels", 2)) or 2
            dev_idx = device.get("index")

            logging.info(
                f"[Loopback] Captured Device: '{device.get('name')}' (Index #{dev_idx}) | "
                f"Rate: {self.sample_rate} Hz | Channels: {self.num_channels}"
            )

            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.num_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=self.chunk_size,
            )

            self.stream.start_stream()
            logging.info("[Loopback] Audio stream active. Polling WASAPI audio buffer...")

            while not self._stop_event.is_set():
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    if data:
                        if not self._first_chunk_logged:
                            self._first_chunk_logged = True
                            logging.info(f"[Loopback] Audio engine active! Captured {len(data)} PCM bytes from WASAPI.")
                        self.audio_callback(data, self.sample_rate, self.num_channels, False)
                except Exception as exc:
                    if not self._read_err_logged:
                        self._read_err_logged = True
                        logging.error(f"[Loopback] stream.read failed: {exc}", exc_info=True)
                    time.sleep(0.01)

        except Exception as exc:
            logging.error(f"[Loopback] Capture error: {exc}", exc_info=True)
        finally:
            self._cleanup()

    def stop(self):
        """Stops the audio capture stream."""
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)

    def _cleanup(self):
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as exc:
                logging.error(f"[Loopback] stream cleanup failed: {exc}", exc_info=True)
            self.stream = None

        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception as exc:
                logging.error(f"[Loopback] PyAudio terminate failed: {exc}", exc_info=True)
            self.pyaudio_instance = None
        logging.info("[Loopback] Capture stream closed.")


def list_loopback_devices():
    """List all available audio devices and WASAPI loopback devices."""
    if not HAS_PYAUDIO:
        print("[Loopback] pyaudiowpatch is not installed.")
        return
    p = pyaudio.PyAudio()
    print("\n=== WASAPI Audio Devices ===")
    try:
        default_lb = p.get_default_wasapi_loopback()
        print(f"Default Loopback Device: Index #{default_lb.get('index')} - {default_lb.get('name')}")
    except Exception as e:
        print(f"Default Loopback Device: Not found ({e})")
    print("-" * 60)
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            is_lb = info.get("isLoopbackDevice", False)
            lb_tag = " [WASAPI Loopback]" if is_lb else ""
            print(f"Index {i:2d}: {info.get('name')}{lb_tag}")
        except Exception:
            pass
    print("=" * 60 + "\n")
    p.terminate()


def get_loopback_devices():
    """Returns a list of dicts: [{'index': int, 'name': str, 'is_loopback': bool, 'is_default': bool}]."""
    devices = []
    if not HAS_PYAUDIO:
        return devices
    try:
        p = pyaudio.PyAudio()
        default_idx = None
        try:
            default_lb = p.get_default_wasapi_loopback()
            if default_lb:
                default_idx = default_lb.get("index")
        except Exception:
            pass

        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                is_lb = bool(info.get("isLoopbackDevice", False))
                name = info.get("name", f"Device #{i}")
                devices.append({
                    "index": i,
                    "name": name,
                    "is_loopback": is_lb,
                    "is_default": (i == default_idx),
                })
            except Exception:
                pass
        p.terminate()
    except Exception as e:
        logging.error(f"[Loopback] Error enumerating devices: {e}")
    return devices
