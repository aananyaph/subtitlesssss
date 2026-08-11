# Live Subtitle — Free, Local-Only Subtitle System

A local-only, free, open-source live subtitle pipeline for browser videos (e.g. YouTube, Netflix) designed to run on CPU-only hardware (e.g., Intel UHD 620 integrated graphics).

---

## 🎯 Overall Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                 Module 1 — Chrome Extension                 │
│              (Active Tab Audio Egress @ 48kHz)              │
└──────────────────────────────┬──────────────────────────────┘
                               │ Binary Int16 PCM WebSocket
                               ▼ (ws://localhost:8765)
┌─────────────────────────────────────────────────────────────┐
│              Module 2 — Voice Activity Detection            │
│                                                             │
│ 1. Audio Preprocessor (Int16 → Float32, Stereo → Mono, 16k)  │
│ 2. Silero VAD ONNX Engine (Local CPU Inference)             │
│ 3. Hysteresis Speech Segmenter (START / END Events)          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                    [Segmented Speech Chunks]
                               │
                               ▼ (Future Module 3)
                     [Local ASR / Whisper]
```

---

## 📁 Repository Structure

```text
live-subtitle/
├── live-subtitle-extension/       # Module 1: Chrome Extension (Manifest V3)
│   ├── manifest.json
│   ├── popup/
│   │   ├── popup.html
│   │   ├── popup.css
│   │   └── popup.js
│   ├── background/
│   │   └── service-worker.js
│   ├── offscreen/
│   │   ├── offscreen.html
│   │   └── offscreen.js
│   └── audio/
│       └── audio-processor.js
│
├── backend/                       # Module 2: Python VAD Backend
│   ├── vad/
│   │   ├── __init__.py
│   │   ├── silero_vad.onnx        # Local 2MB Silero VAD ONNX model
│   │   ├── vad_engine.py          # Silero VAD ONNX inference engine
│   │   ├── audio_utils.py         # Stereo-to-mono, downsampling to 16kHz, preprocessor
│   │   ├── segmenter.py           # Temporal hysteresis speech state machine
│   │   ├── vad_server.py          # Asynchronous WebSocket server (ws://localhost:8765)
│   │   └── test_vad.py            # Comprehensive 7-scenario test suite & benchmark
│   └── requirements.txt
│
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Python Environment Setup (Module 2)

```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

---

### 2. Run Module 2 Automated Tests & Benchmark

To execute the 7 automated audio test scenarios (Speech, Silence, Video Speech, Music, Background Music, Quiet Speech, Fast Speech) and measure CPU/RAM performance:

```bash
python vad/test_vad.py
```

**Expected Benchmark Results**:
* **Real-Time Factor (RTF)**: `~0.01 - 0.05` (1 second of audio processes in `< 5 ms`).
* **Memory Footprint**: `< 50 MB` peak RAM.
* **Accuracy**: Speech detected cleanly; silence and pure music rejected.

---

### 3. Run Live VAD WebSocket Server

To start the local WebSocket server that receives audio live from the Chrome Extension:

```bash
python vad/vad_server.py
```

The server binds to `ws://localhost:8765`.

---

### 4. Connect Chrome Extension (Module 1)

1. Open Google Chrome and go to `chrome://extensions`.
2. Enable **Developer mode** (top right).
3. Click **Load unpacked** and select the `live-subtitle-extension` folder.
4. Open a YouTube video (e.g. news or speech clip).
5. Open the extension popup, enable the **Stream to Local Python (ws://localhost:8765)** switch, and click **Start Capture**.
6. Observe the terminal console running `vad_server.py` — you will see live frame diagnostics and clean event logs:
   ```text
   VAD EVENT: >>> SPEECH START <<< (Frame #120, Prob: 0.94)
   VAD EVENT: <<< SPEECH END <<< (Duration: 2.43s, Speech PCM Samples: 38912)
   ```

---

## 🔬 Technical Deep-Dive — Module 2 (VAD)

### 1. Why Silero VAD?
Silero VAD is an enterprise-grade pre-trained neural Voice Activity Detector packaged in a ~2 MB ONNX model. Running it via `onnxruntime` CPU execution provider eliminates the need for PyTorch dependencies (~200MB+), reducing memory usage to under 50 MB and inference time to `< 0.5 ms` per 32ms frame on Intel UHD 620 CPU-only hardware.

### 2. Audio Preprocessing Pipeline

```text
Incoming Stereo Int16 PCM (48000 Hz)
      ↓ int16_to_float32
Float32 Array [-1.0, 1.0]
      ↓ stereo_to_mono ((L + R) / 2)
Mono Float32 Array (48000 Hz)
      ↓ resample_audio (downsampled 3:1)
Mono Float32 Array (16000 Hz)
      ↓ AudioPreprocessor ring buffer
512-Sample Float32 Frames (32ms @ 16kHz)
      ↓
Silero VAD ONNX Engine
```

### 3. Speech Segmentation & Hysteresis Parameters
To prevent fragmented start/stop events during speech pauses or brief noises, `SpeechSegmenter` uses dual-threshold hysteresis:
* `threshold = 0.50`: Probability required to trigger speech candidacy.
* `neg_threshold = 0.35`: Lower threshold required to consider speech drop.
* `min_speech_duration_ms = 250`: Minimum continuous speech duration required to trigger `SPEECH_START` (filters out brief noise clicks).
* `min_silence_duration_ms = 400`: Minimum silence duration required to trigger `SPEECH_END` (prevents word splitting during pauses).
