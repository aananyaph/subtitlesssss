# Live Subtitle Overlay — Local, Free, CPU-Only

Real-time subtitle generation for browser video (YouTube, Netflix, Vimeo, etc.) using a fully local Python backend powered by **OpenVINO**, **`distil-whisper small.en`**, **Silero VAD**, and an **NLP Quality Post-Processing Layer**. 

- **Zero Cloud APIs**
- **Zero Paid Subscriptions**
- **100% Private & Local Inference**
- **Tested on Intel CPU / UHD 620 Integrated Graphics**

---

## 1. Complete Final Architecture

```
                       Chrome Browser (Active Video Tab)
                                     │
                        chrome.tabCapture (Audio API)
                                     │  48kHz Stereo Int16 PCM
                                     ▼
                      Offscreen Audio Engine (offscreen.js)
                                     │  WebSocket ws://localhost:8765
                                     ▼
                Python Pipeline Server (pipeline_server.py)
                                     │
                               Audio Queue (bounded 150)
                                     │
                                     ▼
                         Silero VAD (ONNX Runtime CPU)
                                     │
                          VAD-Adaptive Chunker
                                     │
                                     ├────────────────────────┐
                                     │                        │ (initial_prompt bias)
                                     ▼                        ▼
                        OpenVINO ASR Engine (CPU) ◄── Domain Vocabulary Tuning
                      distil-whisper small.en (int8)
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ▼                                       ▼
        PARTIAL Transcripts                     FINAL Transcripts
                 │                                       │
        (Fast Path: Instant)               NLP Post-Processing Layer
                 │                          ├── Hallucination Safety Filter
                 │                          ├── Exact/Near Duplicate Filter
                 │                          └── Silero Punctuation Restoration
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │  JSON Transcript Event
                                     ▼
                      Service Worker (service-worker.js)
                                     │  tabs.sendMessage
                                     ▼
                    Subtitle Renderer (content/overlay.js)
                     (Floating Subtitle DOM Overlay)
```

---

## 2. Key Features & Highlights

- **Sub-Second Latency**: **~0.71s** end-to-end latency (**~48.9% faster** than original 1.39s baseline).
- **Ultra-High Efficiency**: Real-Time Factor (RTF) of **0.14** (**>7.1× faster than real-time**) using only **22.1% CPU**.
- **Publication-Grade Readability**: Automatic sentence capitalization, punctuation restoration, and technical vocabulary formatting.
- **Domain Vocabulary Guidance**: Guides ASR recognition for technical terms (`OpenVINO`, `faster-whisper`, `CTranslate2`, `Silero`, `WebSocket`, `JavaScript`, `Python`).
- **Zero Duplicate Flicker**: Normalizes and suppresses exact and near-duplicate segment repetitions.
- **Hallucination Safety Layer**: Suppresses non-ASCII hallucinations and noise burst false-positives.
- **Playback Synchronization (Module 10)**: Detects video seeks, video switching, pause, and end events. Immediately clears stale overlays and flushes backend queues using generation epoch IDs to prevent old subtitles from rendering after seeking.
- **Dynamic User Controls**: Custom font size (`Small`, `Medium`, `Large`), position (`Bottom`, `Middle`, `Top`), and persistent storage (`chrome.storage.sync`).

---

## 3. Benchmark Matrix (Complete Evolution)

| Config | Model | Backend | Beam / BestOf | Chunking | NLP Post-Processing | Domain Vocab | Latency | RTF | CPU | RAM |
|---|---|---|---:|---|---|---|---:|---:|---:|---:|
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

---

## 4. Hardware Requirements

- **CPU**: Intel Core i3/i5/i7 (x86_64 CPU)
- **Graphics**: Integrated Graphics (Intel UHD 620 class or better)
- **CUDA Requirement**: **NONE** (100% CPU-only execution)
- **RAM**: 8.0 GB System RAM (Backend RSS memory is ~685 MB)
- **OS**: Windows, macOS, or Linux

---

## 5. Quick Start & Setup

### Step 1: Install Dependencies & Start Python Server
```bash
# Navigate to backend directory
cd backend

# Install Python requirements
pip install -r requirements.txt

# Start the pipeline server
python pipeline_server.py
```

### Step 2: Load Extension in Chrome
1. Open Chrome and go to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in upper right).
3. Click **Load unpacked** and select the `live-subtitle-extension/` directory.

### Step 3: Start Captions
1. Open any video webpage (YouTube, Netflix, Vimeo).
2. Click the **Live Subtitle** extension icon in Chrome toolbar.
3. Click **Start Captions**. Live subtitles will immediately appear over the video!

---

## 6. Project Structure

```
.
├── backend/
│   ├── asr/
│   │   ├── asr_engine.py      # BaseASREngine, OpenVINOASREngine, FasterWhisperEngine
│   │   ├── streaming_asr.py  # StreamingASRManager & LocalAgreement stabilization
│   │   └── nlp_processor.py  # Silero Punctuation, Hallucination & Duplicate Filters
│   ├── vad/
│   │   ├── audio_utils.py    # Resampling, Int16->Float32, decimation
│   │   ├── vad_engine.py     # Silero VAD ONNX model wrapper
│   │   └── segmenter.py      # SpeechSegmenter temporal hysteresis
│   ├── pipeline_server.py    # Real-time WebSocket pipeline server & thread workers
│   ├── test_pipeline.py      # Automated pipeline verification suite
│   └── benchmark_pipeline.py # End-to-end performance benchmarking script
├── live-subtitle-extension/
│   ├── manifest.json         # Manifest V3 configuration & permissions
│   ├── popup/                # Extension popup UI & user settings controller
│   ├── background/           # Service Worker message router
│   ├── offscreen/            # Offscreen document audio processing engine
│   └── content/              # Webpage video overlay renderer
├── benchmark_report.md       # Comprehensive performance benchmark report
└── README.md                 # System overview and setup guide
```

---

## 7. Known Limitations & Future Work

- **Language Scope**: `distil-whisper small.en` is an English-focused model.
- **Top-Level Video Elements**: Optimized for HTML5 `<video>` elements (YouTube, Netflix, Vimeo).
- **Future Improvements**:
  - Optional GPU/NAPI acceleration for non-Intel hardware.
  - Multi-language translation layer.
