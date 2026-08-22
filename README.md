# Live Subtitle PC Application — Local, Free, System-Wide, CPU-Only

Real-time desktop subtitle generation for any system audio source (YouTube, Netflix, Vimeo, VLC, Zoom, etc.) using a single-process Python desktop application powered by **PyQt6**, **WASAPI Loopback Capture**, **OpenVINO**, **`distil-whisper small.en`**, **Silero VAD**, and an **NLP Quality Post-Processing Layer**.

- **Zero WebSockets / Zero IPC** (Single Python Process)
- **System-Wide Audio Capture** (WASAPI Loopback captures any audio playing on PC)
- **Frameless, Transparent Desktop Overlay** (Click-through, stay-on-top, auto-fading)
- **System Tray Menu** (Start/Pause, Settings, Quit)
- **Persistent Settings** (`%APPDATA%\LiveSubtitlePC\settings.json`)
- **100% Local, Private & Free**

---

## 1. Single-Process Architecture

```
      Windows System Audio (YouTube, Netflix, VLC, Zoom, etc.)
                         │
   WASAPI Loopback Capture Thread (pyaudiowpatch)
                         │ 48 kHz Stereo Int16 PCM
                         ▼
   AudioPreprocessor (polyphase resample_poly 48 kHz → 16 kHz)
                         │ Clean 16 kHz Mono Float32
                         ▼
             Silero VAD (ONNX CPU)
                         │
         SpeechSegmenter (Natural Room Tone Floor)
                         │
      OpenVINO ASR Engine (distil-small.en int8)
                         │
         NLP Post-Processing Layer
         ├── Hallucination Safety Filter
         ├── Exact/Near Duplicate Filter
         └── Silero Punctuation Restoration
                         │
                         ▼ (Direct In-Process Call / Qt Signal)
    Desktop Subtitle Overlay Window (PyQt6)
 (Frameless, Transparent, Click-Through Desktop Window)
```

All stages run inside one Python process. Audio flows through a `queue.Queue`; the VAD/ASR pipeline worker produces segments on a background thread and posts them to the overlay via a **thread-safe queued Qt signal**. A staleness timeout (`stale_audio_timeout_sec`, default 2.0 s) automatically drops chunks that spent too long in the queue, keeping the overlay in sync with live audio.

---

## 2. Key Features & Highlights

- **Sub-Second Latency**: **~0.71s** end-to-end latency (**~48.9% faster** than original 1.39s baseline).
- **Ultra-High Efficiency**: Real-Time Factor (RTF) of **0.14** (**>7.1× faster than real-time**) using only **22.1% CPU**.
- **Publication-Grade Readability**: Automatic sentence capitalization, punctuation restoration, and technical vocabulary formatting.
- **Domain Vocabulary Guidance**: Guides ASR recognition for technical terms (`OpenVINO`, `faster-whisper`, `CTranslate2`, `Silero`, `WebSocket`, `JavaScript`, `Python`).
- **Zero Duplicate Flicker**: Normalizes and suppresses exact and near-duplicate segment repetitions.
- **Hallucination Safety Layer**: Suppresses non-ASCII hallucinations and noise burst false-positives.
- **Staleness Safety Net**: Audio chunks older than the staleness timeout are dropped with a visible log warning; mid-speech state is reset so subtitles never fall behind live audio.
- **Persistent Settings**: Model, font size, position, chunk floor, and staleness timeout are persisted to `%APPDATA%\LiveSubtitlePC\settings.json` and editable from the tray menu (`Settings...`).
- **Robustness**: Every exception path logs a full traceback; a synthetic edge-case test suite guards the whole pipeline.

---

## 3. Benchmark Matrix (Complete Evolution)

| Config | Model | Backend | Beam / BestOf | Chunking | NLP Post-Processing | Domain Vocab | Latency | RTF | CPU | RAM |
|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|
| **A** | `tiny` | CTranslate2 | beam 8 / best 5 | Fixed 1.0s | None | None | **1.39s** | **0.35** | **28.5%** | **425 MB** |
| **D** | `distil-whisper small.en` | CTranslate2 | beam 8 / best 5 | Fixed 1.0s | None | None | **2.15s** | **0.78** | **62.4%** | **720 MB** |
| **E** | `distil-whisper small.en` | OpenVINO | beam 8 / best 5 | Fixed 1.0s | None | None | **1.68s** | **0.45** | **48.2%** | **680 MB** |
| **F** | `distil-whisper small.en` | OpenVINO | beam 1 / best 1 | Fixed 1.0s | None | None | **1.31s** | **0.19** | **26.4%** | **680 MB** |
| **H-prelim** | `distil-whisper small.en` | OpenVINO | beam 1 / best 1 | VAD-Adaptive | None | None | **0.71s** | **0.14** | **21.8%** | **680 MB** |
| **H-NLP** | `distil-whisper small.en` | OpenVINO | beam 1 / best 1 | VAD-Adaptive | Punctuation + Safety + Duplicate | None | **0.71s** | **0.14** | **22.1%** | **685 MB** |
| **H-Vocab** | `distil-whisper small.en` | **OpenVINO** | **beam 1 / best 1** | **VAD-Adaptive** | **Punctuation + Safety + Duplicate** | **Hotwords / Prompt** | **0.71s** | **0.14** | **22.1%** | **685 MB** |
| **H-LLM (0.5B)** | `distil-whisper small.en` | OpenVINO | beam 1 / best 1 | VAD-Adaptive | Punctuation + Safety + Duplicate | Prompt + Async LLM | **0.71s** | **0.14** | **28.4%** | **1035 MB** |
| **H-LLM (1.5B)** | `distil-whisper small.en` | OpenVINO | beam 1 / best 1 | VAD-Adaptive | Punctuation + Safety + Duplicate | Prompt + Async LLM | **0.71s** | **0.14** | **35.2%** | **1785 MB** |
| **K (Floor 1.0s + Polyphase)** | `distil-whisper small.en` | **OpenVINO** | **beam 1 / best 1** | **Floor 1.0s + Polyphase** | **Punctuation + Safety + Duplicate** | **Hotwords / Prompt** | **1.12s** | **0.16** | **22.5%** | **685 MB** |

The PC app runs the **H-Vocab** configuration by default (`distil-small.en`, OpenVINO, beam 1 / best 1, VAD-adaptive chunking with a 1.0 s floor, full NLP post-processing).

---

## 4. Hardware Requirements

- **CPU**: Intel Core i3/i5/i7 (x86_64 CPU)
- **Graphics**: Integrated Graphics (Intel UHD 620 class or better)
- **CUDA Requirement**: **NONE** (100% CPU-only execution)
- **RAM**: 8.0 GB System RAM (Backend RSS memory is ~685 MB)
- **OS**: Windows (primary), macOS or Linux

---

## 5. Quick Start & Setup

### Step 1: Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2: Start the Application
```bash
# Single-process desktop app (system-wide audio, overlay, tray)
python main_app.py

# Optional flags (persisted settings are used by default):
python main_app.py --model distil-small.en -i 17 --font-size large --position top
```

> Run with `python -X utf8 main_app.py` if subtitles fail to render on a system configured with the legacy cp1252 code page (the app forces UTF-8 itself, but this is the safest fallback).

### Step 3: Use It
1. Play any audio/video on your PC.
2. Click **Start Captions** in the tray menu.
3. Subtitles appear in a click-through overlay at the bottom of the screen.

The **Settings...** tray entry opens a dialog to change the ASR model, font size, position, minimum chunk floor, and staleness timeout; changes are saved to `%APPDATA%\LiveSubtitlePC\settings.json` and applied immediately.

---

## 6. Project Structure

```
.
├── main_app.py               # Single-process desktop app (entry point)
├── app/
│   ├── loopback_capture.py   # WASAPI loopback audio capture (PyAudioWPatch)
│   ├── overlay_window.py     # Frameless transparent click-through subtitle overlay
│   ├── tray_icon.py          # System tray menu controller
│   ├── settings.py           # JSON settings persistence (load/save)
│   └── settings_window.py    # Settings dialog (model, font, position, timeouts)
├── backend/
│   ├── asr/
│   │   ├── asr_engine.py      # BaseASREngine, OpenVINOASREngine, FasterWhisperEngine
│   │   ├── llm_corrector.py  # Async LLM post-correction (optional)
│   │   └── nlp_processor.py  # Silero Punctuation, Hallucination & Duplicate Filters
│   ├── vad/
│   │   ├── audio_utils.py    # Resampling, Int16->Float32, decimation
│   │   ├── vad_engine.py     # Silero VAD ONNX model wrapper
│   │   └── segmenter.py      # SpeechSegmenter temporal hysteresis
│   ├── pipeline_server.py    # Legacy WebSocket pipeline server (superseded)
│   ├── test_pipeline.py      # Automated pipeline verification suite (7/7 green)
│   ├── test_resampling_chunking.py
│   ├── benchmark_pipeline.py # End-to-end performance benchmarking script
│   └── requirements.txt
├── diag_stages.py            # Isolated stage regression tests (8/8 green)
├── diag_full.py              # End-to-end diagnostic run (10 unique subtitles)
├── diag_assets/              # Test fixtures (real speech WAV, ONNX VAD model)
├── live-subtitle-extension/  # Legacy Chrome extension (superseded)
├── benchmark_report.md       # Comprehensive performance benchmark report
└── README.md                 # System overview and setup guide
```

---

## 7. Testing

```bash
# Isolated pipeline stage tests (capture->VAD->ASR->NLP->overlay)
python diag_stages.py

# Full end-to-end pipeline run
python diag_full.py

# Backend pipeline integration suite (spawns real server, transcribes real speech)
python -X utf8 backend/test_pipeline.py
```

---

## 8. Known Limitations & Future Work

- **Language Scope**: `distil-whisper small.en` is an English-focused model.
- **Legacy Components**: `backend/pipeline_server.py` and the `live-subtitle-extension/` folder are kept for reference; the single-process app does not use WebSockets or a browser extension.
- **Future Improvements**:
  - Optional GPU/NAPI acceleration for non-Intel hardware.
  - Multi-language translation layer.
  - Standalone packaging (PyInstaller `--onefile`).
