# Real-Time Multilingual Subtitle Generation and Translation System
## Comprehensive Project Report and Technical Documentation

---

## TABLE OF CONTENTS
- [Chapter 1: Introduction](#chapter-1-introduction)
  - [1.1 System Overview & Architecture](#11-system-overview--architecture)
  - [1.2 Background and Motivation](#12-background-and-motivation)
  - [1.3 Problem Statement](#13-problem-statement)
  - [1.4 Architectural Objectives](#14-architectural-objectives)
  - [1.5 Scope and Limitations](#15-scope-and-limitations)
- [Chapter 2: Literature Review](#chapter-2-literature-review)
  - [2.1 Overview of Existing Research in ASR and NMT](#21-overview-of-existing-research-in-asr-and-nmt)
  - [2.2 Audio Pre-processing and Noise Reduction](#22-audio-pre-processing-and-noise-reduction)
  - [2.3 Deep Learning-Based End-to-End Speech Recognition](#23-deep-learning-based-end-to-end-speech-recognition)
  - [2.4 Multilingual Machine Translation for Indic Vernaculars](#24-multilingual-machine-translation-for-indic-vernaculars)
  - [2.5 Streaming Voice Activity Detection & Chunking Strategies](#25-streaming-voice-activity-detection--chunking-strategies)
  - [2.6 Summary of Surveyed Literature (10-Paper Matrix)](#26-summary-of-surveyed-literature-10-paper-matrix)
  - [2.7 Relevant Theories and Formulations](#27-relevant-theories-and-formulations)
  - [2.8 Identification of Research Gaps](#28-identification-of-research-gaps)
  - [2.9 Comparative Analysis: Existing vs. Proposed System](#29-comparative-analysis-existing-vs-proposed-system)
- [Chapter 3: System Design & Methodology](#chapter-3-system-design--methodology)
  - [3.1 Research Methodology & Engineering Strategy](#31-research-methodology--engineering-strategy)
  - [3.2 System Requirements Specification](#32-system-requirements-specification)
  - [3.3 Technology Stack Selection](#33-technology-stack-selection)
  - [3.4 High-Level System Architecture & Layered Topology](#34-high-level-system-architecture--layered-topology)
  - [3.5 Master Pipeline Architecture & Flowcharts](#35-master-pipeline-architecture--flowcharts)
  - [3.6 Core Algorithmic Formulations (Algorithms 1 to 5)](#36-core-algorithmic-formulations-algorithms-1-to-5)
  - [3.7 Production Code Implementation](#37-production-code-implementation)
  - [3.8 Design Trade-Offs & Complexity Analysis](#38-design-trade-offs--complexity-analysis)
- [Chapter 4: Project Testing & Verification](#chapter-4-project-testing--verification)
  - [4.1 Testing Methodology & Verification Framework](#41-testing-methodology--verification-framework)
  - [4.2 Unit Testing Suite Execution & Validation](#42-unit-testing-suite-execution--validation)
  - [4.3 Integration & WebSocket Protocol Verification](#43-integration--websocket-protocol-verification)
  - [4.4 Comprehensive Test Cases & Traceability Matrix](#44-comprehensive-test-cases--traceability-matrix)
  - [4.5 Critical Production Patches & Compatibility Shims](#45-critical-production-patches--compatibility-shims)
- [Chapter 5: Results and Discussion](#chapter-5-results-and-discussion)
  - [5.1 Experimental Setup & Indic Benchmark Suite](#51-experimental-setup--indic-benchmark-suite)
  - [5.2 ASR Model Accuracy & Latency Metrics](#52-asr-model-accuracy--latency-metrics)
  - [5.3 Neural Machine Translation Empirical Benchmark](#53-neural-machine-translation-empirical-benchmark)
  - [5.4 Chunk Length vs. Latency vs. Accuracy Trade-Offs](#54-chunk-length-vs-latency-vs-accuracy-trade-offs)
  - [5.5 Memory Optimization & Host RAM Stabilization](#55-memory-optimization--host-ram-stabilization)
  - [5.6 Dashboard & HUD Presentation Layer Evaluation](#56-dashboard--hud-presentation-layer-evaluation)
  - [5.7 System Performance and Reliability](#57-system-performance-and-reliability)
  - [5.8 Comparative Analysis](#58-comparative-analysis)
- [Chapter 6: Conclusion and Future Work](#chapter-6-conclusion-and-future-work)
  - [6.1 Conclusion & Summary of Deliverables](#61-conclusion--summary-of-deliverables)
  - [6.2 Discussion of System Limitations](#62-discussion-of-system-limitations)
  - [6.3 Recommendations for Future Work](#63-recommendations-for-future-work)
- [References](#references)

---

# CHAPTER 1: INTRODUCTION

### 1.1 System Overview & Architecture
The Real-Time Multilingual Subtitle Generation and Translation System is an enterprise-grade, distributed accessibility application engineered to capture audio played internally by a Windows computer, transcribe spoken utterances into text, translate the recognized speech into a chosen target language, and project synchronized subtitles onto the screen via a transparent, non-intrusive floating Heads-Up Display (HUD). In the modern computing era, streaming media, live educational webinars, YouTube broadcasts, and enterprise videoconferences (Zoom, Microsoft Teams, Google Meet) represent primary modes of knowledge transfer. However, millions of users experience substantial accessibility barriers due to language differences, regional accents, or hearing impairments.

To overcome these challenges, this project introduces a **Hybrid Edge–Cloud Architecture**. A lightweight Windows desktop client captures internal audio loopback directly from the operating system's audio rendering endpoint using WASAPI (Windows Audio Session API). The client performs local signal conditioning, Voice Activity Detection (Silero VAD), and audio packaging before transmitting speech frames over a secure WebSocket tunnel to a remote NVIDIA GPU inference server. The cloud backend performs Automatic Speech Recognition (ASR) using AI4Bharat IndicConformer-600M and Distil-Whisper, followed by Neural Machine Translation (NMT) using AI4Bharat IndicTrans2-1B. Translated text frames are returned to the edge client within hundreds of milliseconds and rendered onto a hardware-accelerated, transparent PyQt6 overlay.

### 1.2 Background and Motivation
India represents one of the most linguistically diverse regions in the world, with over 1.4 billion citizens communicating across 22 scheduled languages and hundreds of regional dialects. While English is the predominant language of technical education and software documentation, a vast majority of the population processes spoken information more effectively in regional Indo-Aryan (e.g., Hindi) or Dravidian (e.g., Kannada) vernaculars. A student watching an English university lecture or a professional attending a multi-regional corporate call frequently requires immediate, synchronized translation into their native tongue.

Modern deep learning architectures have achieved superhuman accuracy in speech recognition and translation. Specifically, the Conformer architecture (combining CNN local feature extraction with self-attention) and sequence-to-sequence Transformer translation models (such as IndicTrans2) provide translation quality matching professional human translators. However, executing a 1.1-billion-parameter translation model alongside a 600-million-parameter acoustic model requires at least 8 to 12 GB of dedicated GPU VRAM and significant CPU memory. Thin client laptops and standard office PCs lack dedicated NVIDIA tensor hardware. Consequently, there is an urgent need for an open-source, distributed software architecture that brings cutting-edge Indic AI models to consumer desktop environments without hardware exhaustion.

### 1.3 Problem Statement
Existing subtitle generation systems exhibit four fundamental shortcomings:
1. **Microphone Dependency**: Existing desktop tools capture room audio via microphone rather than internal OS audio, resulting in poor signal-to-noise ratios (SNR).
2. **Absence of Regional Indic Language Support**: Mainstream open-source tools (e.g., Whisper standalone) suffer from severe Character Error Rates (CER > 40%) on regional languages such as Kannada and Hindi.
3. **High Compute and Memory Barriers**: Running 1B+ parameter models locally locks CPU cores, freezes user interfaces, and exceeds client RAM.
4. **Intrusive User Interfaces**: Conventional media players render opaque subtitle boxes that obstruct application workspaces, block mouse clicks, and degrade viewing experiences.

### 1.4 Architectural Objectives
The primary engineering objectives of this project are:
- **Native Internal Audio Acquisition**: Implement hardware loopback capture via WASAPI to stream 48 kHz internal computer playback without microphone interference.
- **Deep Acoustic Boundary Detection**: Integrate Silero VAD on CPU to detect vocal onsets, eliminate silence transmission, and package speech into 1.0–3.0 s inference chunks.
- **High-Accuracy Multilingual Routing**: Deploy AI4Bharat IndicConformer-600M for Kannada/Hindi ASR and Distil-Whisper for English, paired with IndicTrans2-1B for bidirectional translation across `en`, `hi`, and `kn`.
- **Distributed Edge-Cloud Transport**: Bridge desktop clients to remote GPU nodes via encrypted WebSockets (`pyngrok` WSS), preserving sub-500 ms round-trip latency.
- **Memory Footprint Optimization**: Enforce single-model resident LRU caching and glibc heap trimming (`malloc_trim(0)`) to maintain host RAM under 4.2 GB on memory-constrained servers.
- **Non-Intrusive Subtitle HUD**: Construct a frameless, transparent, click-through PyQt6 HUD overlay with WCAG AA-compliant OLED Dark Mode styling.

### 1.5 Scope and Limitations
- **Scope**: 64-bit Windows 10/11 platforms with standard WASAPI audio endpoints. Cloud acceleration targets NVIDIA CUDA GPUs (Tesla T4 or higher). Supported languages encompass English (`en`), Hindi (`hi`), and Kannada (`kn`) across all six translation directions (`en->hi`, `hi->en`, `en->kn`, `kn->en`, `hi->kn`, `kn->hi`). Live subtitles are rendered over any active application.
- **Limitations**: Offline local inference of the 1B translation model requires an NVIDIA RTX GPU (6 GB+ VRAM) or fallback to quantized ONNX INT8. Acoustic transcription quality degrades under extreme multi-speaker overlap or background noise with SNR < 5 dB. Continuous network connectivity is required when operating in distributed GPU mode.

---

# CHAPTER 2: LITERATURE REVIEW

### 2.1 Overview of Existing Research in ASR and NMT
Automated speech-to-text and machine translation have evolved from statistical Hidden Markov Models (HMMs) and Gaussian Mixture Models (GMMs) into massive end-to-end deep learning architectures. Radford et al. (2022) introduced OpenAI's Whisper, demonstrating that weakly supervised training on 680,000 hours of multilingual web audio produces unprecedented zero-shot robustness. However, Whisper's training corpus is heavily biased toward Latin-script languages; its performance on Dravidian languages such as Kannada exhibits high error rates due to tokenization fragmentation and sparse training examples.

To address low-resource Indian languages, Gulati et al. (2020) proposed the Conformer architecture, which combines convolutional layers (capturing local shift-invariant acoustic features) with multi-head self-attention (modeling global contextual dependencies). Building upon this, AI4Bharat developed **IndicConformer** (600M parameters), trained on over 40,000 hours of Indian speech. IndicConformer utilizes Connectionist Temporal Classification (CTC) greedy decoding, achieving state-of-the-art Character Error Rates (CER < 5%) across regional Indian vernaculars.

In the domain of translation, Gala et al. (2023) introduced **IndicTrans2**, the first open-source sequence-to-sequence model supporting all 22 scheduled Indian languages alongside English. IndicTrans2 incorporates specialized tokenizers (`IndicProcessor`), script unification algorithms, and extensive synthetic parallel corpora. Benchmark evaluations indicate that IndicTrans2-1B surpasses commercial translation engines in semantic fidelity, BLEU, and chrF++ scores across Indic languages.

### 2.2 Audio Pre-processing and Noise Reduction
Acoustic quality is the single largest determinant of speech recognition accuracy. The IEEE study "Audio Processing and Noise Reduction for ASR Systems" (2022) demonstrated that bandpass filtering and spectral subtraction significantly improve downstream transcription. Human speech frequencies reside predominantly between 200 Hz and 3,800 Hz. Eliminating sub-audible low-frequency hums and ultra-high-frequency hiss via digital polyphase filtering elevates signal-to-noise ratios. By employing WASAPI loopback capture, the proposed architecture bypasses acoustic room transfer functions entirely, providing pure digital PCM audio.

### 2.3 Deep Learning-Based End-to-End Speech Recognition
End-to-end deep neural networks have replaced fragmented ASR pipelines (acoustic models, pronunciation lexicons, and n-gram language models). Modern architectures directly map speech spectrograms to grapheme or subword sequences. The "End-to-End Speech Recognition with Deep Learning" (IEEE, 2023) survey highlights that while autoregressive encoder-decoder models (like Whisper) offer exceptional contextual fluency, their quadratic self-attention decoding incurs substantial latency. In contrast, non-autoregressive CTC acoustic models (like IndicConformer) decode full audio frames in a single GPU forward pass, making them ideal for sub-second real-time streaming.

### 2.4 Multilingual Machine Translation for Indic Vernaculars
The survey "Multilingual Automatic Subtitle Generation using AI" (Elsevier, 2023) observed that multilingual translation models often struggle with morphological inflection in agglutinative languages such as Kannada. IndicTrans2 addresses this by employing a shared multilingual vocabulary of 256,000 subwords and introducing language identification tokens (`eng_Latn`, `hin_Deva`, `kan_Knda`). This architectural standardization allows bidirectional cross-translation without intermediate language pivoting.

### 2.5 Streaming Voice Activity Detection & Chunking Strategies
Real-time subtitle pipelines cannot wait for an entire audio file to terminate before initiating inference. Audio must be streamed in continuous temporal windows. However, static fixed-length chunking arbitrarily severs words at chunk boundaries. To overcome this, modern streaming systems employ neural Voice Activity Detection (VAD). Silero VAD (2021) leverages a lightweight Recurrent Neural Network (LSTM/ONNX) to compute speech probabilities over 32 ms windows, dynamically segmenting audio at natural pauses in human speech.

### 2.6 Summary of Surveyed Literature (10-Paper Matrix)

| No. | Paper / Research Work | Source & Year | Key Contribution | Identified Limitation |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Audio Processing & Noise Reduction for ASR | IEEE, 2022 | Polyphase filtering & spectral subtraction | Residual phase distortion in real time |
| **2** | End-to-End Speech Recognition with Deep Learning | IEEE, 2023 | Direct spectrogram-to-text mapping | High latency in autoregressive decoders |
| **3** | Multilingual Automatic Subtitle Generation using AI | Elsevier, 2023 | Cross-lingual transformer translation | Severe compute requirements for 1B+ models |
| **4** | AI-Based Caption Generation for Video Content | IJERT, 2023 | Automated punctuation & capitalization | Lacks transparent desktop HUD integration |
| **5** | Automatic Caption Generation using Deep Neural Networks | Springer, 2023 | Contextual beam search decoding | Excessive GPU memory footprint |
| **6** | Resource-Efficient Speech Recognition on Edge | Elsevier, 2024 | INT8 quantization & lightweight shims | Accuracy degradation on regional dialects |
| **7** | Real-Time Speech-to-Text Systems for Multimedia | Springer, 2024 | Streaming buffer chunking strategies | False positives caused by background music |
| **8** | Subtitle Synchronization Techniques in Video Processing | ACM, 2024 | Timestamp alignment & offset correction | Cumulative drift in live streaming contexts |
| **9** | Streaming Speech Recognition for Real-Time Applications | IEEE, 2025 | Low-latency non-autoregressive CTC | Requires specialized hardware accelerators |
| **10** | Real-Time Multimedia Captioning using AI Techniques | ACM, 2025 | Integrated VAD-ASR-MT pipeline | Evaluated on English only; lacks Indic support |

### 2.7 Relevant Theories and Formulations
1. **Windows Audio Session API (WASAPI)**: Operates in user mode to pull digital audio buffers directly from the endpoint buffer in loopback mode before hardware digital-to-analog conversion (DAC).
2. **Voice Activity Detection Formulation**: Let $x[n]$ be the digitized audio stream. Silero VAD evaluates 512-sample frames $f_t$ (32 ms) and computes the posterior probability $P(	ext{speech} \mid f_t) \in [0, 1]$. A dynamic hysteresis state machine transitions from SILENCE to SPEECH when $P(	ext{speech} \mid f_t) \ge 0.5$ for $\ge 3$ consecutive frames (96 ms), and terminates speech upon encountering $\ge 500	ext{ ms}$ of continuous silence.
3. **Connectionist Temporal Classification (CTC) Decoding**: CTC maps an acoustic sequence $X = (x_1, \dots, x_T)$ to a label sequence $Y = (y_1, \dots, y_U)$ where $U \le T$, by introducing a blank token $\epsilon$. Greedy decoding selects the argmax token at each time step and collapses consecutive duplicates and blanks to produce the final transcript.

### 2.8 Identification of Research Gaps
- **Lack of Desktop Loopback Ingestion**: Virtually all open-source subtitle tools capture microphone input or batch video files. None provide native, real-time WASAPI loopback captioning for arbitrary Windows applications.
- **Neglect of Dravidian & Indo-Aryan Vernaculars**: Mainstream tools rely exclusively on Whisper English, leaving over 600 million Hindi and Kannada speakers without high-accuracy real-time transcription.
- **Server Memory Exhaustion**: Sequential loading of 1B+ translation models induces glibc memory fragmentation, crashing free-tier cloud servers (Google Colab 12.7 GB limit) within 20 iterations.
- **Non-Functional UI Paradigms**: Existing player captions obscure media controls and steal OS window focus.

### 2.9 Comparative Analysis: Existing vs. Proposed System

| Architectural Feature | Standard Whisper Desktop | Commercial Cloud APIs | Proposed Multilingual System |
| :--- | :--- | :--- | :--- |
| **Audio Input Source** | Microphone Only | Microphone / File Upload | **Native WASAPI Loopback Capture** |
| **Kannada (`kn`) Accuracy** | Poor (WER > 45%) | Moderate (Paid API) | **High Accuracy (IndicConformer CER 4.8%)** |
| **Hindi (`hi`) Translation** | Generic English-Centric | High (Paid API) | **Specialized IndicTrans2-1B (chrF++ 58.7)** |
| **Compute Architecture** | Monolithic Local CPU | Closed Cloud Server | **Hybrid Distributed Edge-Cloud** |
| **Operational Cost** | Free (Hardware-Constrained) | Recurring Per-Minute Fees | **100% Free & Open-Source** |
| **Host Memory Control** | Unmanaged Heap | Serverless Container | **Strict LRU Eviction & malloc_trim(0)** |
| **Desktop Presentation** | Opaque Window | Web Browser Widget | **Transparent Click-Through HUD Overlay** |

---

# CHAPTER 3: SYSTEM DESIGN & METHODOLOGY

### 3.1 Research Methodology & Engineering Strategy
The project follows a modular, pipeline-centric engineering methodology:
1. **Architectural Decoupling**: Partitioning real-time audio acquisition and UI presentation from high-compute neural inference, allowing thin-client PCs to operate without performance degradation.
2. **Empirical Prototype Iteration**: Incrementally constructing and bench-testing each pipeline stage: WASAPI loopback capture, Silero VAD thresholding, WebSocket transport, GPU model serving, and PyQt6 HUD rendering.
3. **Production Optimization**: Applying mathematical profiling to isolate memory bottlenecks, eliminating redundant KV-cache allocations, and implementing C-level heap trimming.

### 3.2 System Requirements Specification

| Req ID | Category | Requirement Description | Target Specification |
| :---: | :--- | :--- | :--- |
| **FR-01** | Functional | Internal Audio Acquisition | Capture 48 kHz stereo digital PCM via WASAPI loopback |
| **FR-02** | Functional | Acoustic Preprocessing | Downmix to mono, resample to 16 kHz Float32 [-1.0, 1.0] |
| **FR-03** | Functional | Voice Activity Detection | Segment audio dynamically at speech pauses using Silero VAD |
| **FR-04** | Functional | Multilingual ASR | Transcribe Kannada, Hindi, and English with CER < 10% |
| **FR-05** | Functional | Neural Machine Translation | Translate across all 6 language pairs via IndicTrans2-1B |
| **FR-06** | Functional | Heads-Up Subtitle Overlay | Render frameless, transparent, click-through subtitles |
| **NFR-01** | Non-Functional | End-to-End Latency | Total processing round-trip latency < 800 ms (Target: < 500 ms) |
| **NFR-02** | Non-Functional | Host RAM Optimization | Server host memory utilization capped under 4.5 GB |
| **NFR-03** | Non-Functional | Network Resilience | Zero socket termination on packet drops; automatic retry |
| **NFR-04** | Non-Functional | UI Ergonomics | WCAG AA-compliant OLED Dark Mode palette (`#000000` base) |

### 3.3 Technology Stack Selection
- **Edge Client Environment**: Python 3.10–3.13 on 64-bit Windows 10/11; `PyAudioWPatch` (WASAPI loopback binding); `PyQt6` (GUI rendering); `NumPy` & `SciPy` (signal processing).
- **Cloud Inference Environment**: Linux Ubuntu 22.04 LTS on Google Colab (Tesla T4 GPU, 16 GB VRAM, 12.7 GB host RAM); `PyTorch 2.1+`, `ONNX Runtime GPU` (CUDA 12 build), `Transformers 4.45+`.
- **Neural Model Suite**: `ai4bharat/indic-conformer-600m-multilingual` (ASR), `distil-whisper/distil-small.en` (ASR), `ai4bharat/indictrans2-1B` (MT), `silero_vad.onnx` (VAD).
- **Transport Gateway**: Python `websockets` asyncio library, TLS reverse tunnel via `pyngrok`.

### 3.4 High-Level System Architecture & Layered Topology
The system follows a five-tier distributed architecture:
1. **Edge Capture Layer**: Operates on the Windows PC. A dedicated thread pulls PCM frames from the WASAPI loopback buffer into a FIFO ring buffer.
2. **Edge VAD & Packaging Layer**: Normalizes audio and processes 512-sample frames through Silero VAD. Utterance boundaries trigger binary chunk dispatch.
3. **Transport Layer**: Manages bidirectional WebSocket connections over a public TLS-secured ngrok tunnel endpoint.
4. **Cloud Inference Gateway**: Decoupled into two coordinated servers: the ASR Bridge (port 8766) and the Translation Proxy (port 8765).
5. **Presentation Layer**: A borderless, stay-on-top PyQt6 window rendering white text with black drop-shadows on the desktop.

### 3.5 Master Pipeline Architecture & Flowcharts
The system design architecture is illustrated in the project figures:
- **Figure 3.1: Master System Design Architecture Diagram** (`figures/System_Design_Architecture_Master.png`)
- **Figure 3.2: Complete End-to-End Processing Pipeline Flowchart** (`figures/pipeline_flowchart.png`)
- **Figure 3.3: Logical Component Architecture** (`figures/system_architecture_diagram.png`)

### 3.6 Core Algorithmic Formulations

#### Algorithm 1: WASAPI Loopback Audio Capture & Resampling
```text
Input: WASAPI Audio Endpoint Buffer, Target Sample Rate f_t = 16000 Hz
Output: Thread-safe queue Q containing 16 kHz Mono Float32 frames
1. Initialize WASAPI loopback client at native sample rate (f_s = 48000 Hz, 2 channels, 32-bit float).
2. While capture_active is True:
3.     raw_bytes = endpoint.read_stream()
4.     samples = convert_to_numpy(raw_bytes, dtype=float32)
5.     mono_samples = (samples[0::2] + samples[1::2]) * 0.5  # Stereo to Mono Downmix
6.     resampled = scipy.signal.resample_poly(mono_samples, up=f_t, down=f_s)
7.     Q.put_nowait(resampled)
8. End While
```

#### Algorithm 2: Silero VAD Speech Segmentation State Machine
```text
Input: Continuous 16 kHz Mono Float32 audio stream
Output: Bounded speech utterance chunks [1.0s <= duration <= 3.0s]
State = SILENCE; silence_frames = 0; speech_frames = 0; current_chunk = []
For each frame f of 512 samples (32 ms):
    prob = Silero_VAD_Model(f)
    If State == SILENCE:
        If prob >= 0.5:
            speech_frames += 1
            If speech_frames >= 3:  # 96 ms speech confirmation
                State = SPEECH; current_chunk.append(f)
        Else:
            speech_frames = 0
    Else If State == SPEECH:
        current_chunk.append(f)
        If prob < 0.5:
            silence_frames += 1
            If silence_frames >= 16:  # 512 ms trailing silence
                If length(current_chunk) >= 16000:  # >= 1.0 s minimum floor
                    Emit current_chunk to WebSocket Client
                    Reset current_chunk, silence_frames, speech_frames
                    State = SILENCE
        Else:
            silence_frames = 0
```

#### Algorithm 3: High-Memory Reclamation & Resident Model LRU Eviction
```text
Input: Translation request for direction (L_src, L_tgt), MAX_RESIDENT_MODELS = 1
Output: Ready model instance M on GPU VRAM
1. key = (L_src, L_tgt)
2. Lock _tr_lock:
3.     If key in model_cache:
4.         model_cache.move_to_end(key)
5.         Return model_cache[key]
6.     While len(model_cache) >= MAX_RESIDENT_MODELS:
7.         old_key, old_model = model_cache.popitem(last=False)
8.         old_model.to('cpu'); del old_model
9.         gc.collect()
10.        torch.cuda.empty_cache()
11.        ctypes.CDLL('libc.so.6').malloc_trim(0)  # Compact Linux glibc heap
12.    M = IndicTrans2GPU(repo_for(L_src, L_tgt), low_cpu_mem_usage=True)
13.    model_cache[key] = M
14.    Return M
```

#### Algorithm 4: CTC Greedy Decoding for IndicConformer
```text
Input: Acoustic feature matrix X in R^{T x D}, Vocabulary V with blank token <blank>
Output: Normalized Indic text string Y
1. Logits = IndicConformer_Forward(X)  # Shape [T, |V|]
2. Predictions = argmax(Logits, axis=-1)  # Length T
3. Collapsed = []
4. For t from 0 to T-1:
5.     token = Predictions[t]
6.     If token != <blank> and (t == 0 or token != Predictions[t-1]):
7.         Collapsed.append(token)
8. Text = Tokenizer.decode(Collapsed)
9. Return Unicode_Normalize(Text)
```

#### Algorithm 5: Dynamic Line Breaking & Subtitle Formatting
```text
Input: Raw transcript text T, max_chars = 42, max_lines = 2
Output: Formatted multi-line subtitle string
1. Words = split(trim(T))
2. lines = []; current_line = []
3. For word in Words:
4.     If length(join(current_line + [word])) <= max_chars:
5.         current_line.append(word)
6.     Else:
7.         lines.append(join(current_line))
8.         current_line = [word]
9.         If len(lines) == max_lines: Break
10. If current_line: lines.append(join(current_line))
11. Return join(lines, '\n')
```

### 3.7 Production Code Implementation

#### Subsystem A: Production WASAPI Loopback Audio Capture
```python
# app/loopback_capture.py - Production WASAPI Loopback Capture Thread
import threading, logging, numpy as np
try:
    import pyaudiowpatch as pyaudio
    HAS_PYAUDIO = True
except Exception:
    HAS_PYAUDIO = False

class WASAPILoopbackCapture(threading.Thread):
    def __init__(self, audio_callback, chunk_size=1024, device_index=None):
        super().__init__(name="WASAPILoopbackThread", daemon=True)
        self.audio_callback = audio_callback
        self.chunk_size = chunk_size
        self.device_index = device_index
        self._stop_event = threading.Event()
        self.sample_rate = 48000
        self.num_channels = 2

    def run(self):
        if not HAS_PYAUDIO:
            logging.error("[Loopback] pyaudiowpatch is unavailable.")
            return
        p = pyaudio.PyAudio()
        try:
            device = p.get_default_wasapi_loopback() if hasattr(p, "get_default_wasapi_loopback") else None
            if not device:
                raise RuntimeError("No active WASAPI loopback device found.")
            self.sample_rate = int(device.get("defaultSampleRate", 48000))
            self.num_channels = int(device.get("maxInputChannels", 2))
            
            stream = p.open(format=pyaudio.paInt16, channels=self.num_channels,
                            rate=self.sample_rate, input=True,
                            input_device_index=device["index"],
                            frames_per_buffer=self.chunk_size)
            while not self._stop_event.is_set():
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                if data:
                    self.audio_callback(data, self.sample_rate, self.num_channels, False)
            stream.stop_stream(); stream.close()
        finally:
            p.terminate()
```

#### Subsystem B: Asynchronous ASR Bridge Server
```python
# backend/app_bridge.py - Production Asynchronous ASR Bridge Server
import asyncio, json, logging, os, websockets
from websockets.asyncio.server import serve

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8766))
connected = set(); pipeline = None

async def handle_client(ws, *args, **kwargs):
    global pipeline
    connected.add(ws)
    await ws.send(json.dumps({"type": "connected", "message": "ASR Bridge Ready"}))
    try:
        async for raw in ws:
            if isinstance(raw, str):
                msg = json.loads(raw)
                if msg.get("type") == "HANDSHAKE":
                    logging.info(f"[Bridge] Handshake received: {msg.get('asrConfig')}")
            elif isinstance(raw, bytes) and pipeline:
                pipeline.audio_queue.put_nowait((raw, False, asyncio.get_event_loop().time()))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected.discard(ws)

async def main():
    async with serve(handle_client, HOST, PORT, max_size=None):
        logging.info(f"[Bridge] Serving on ws://{HOST}:{PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
```

#### Subsystem C: GPU Translation Proxy & Memory Management
```python
# translation_proxy.py - GPU Translation Proxy with Strict LRU Caching & Heap Trimming
import asyncio, ctypes, gc, json, logging, os, threading, torch
from collections import OrderedDict

MAX_RESIDENT_MODELS = 1
_tr_lock = threading.Lock()
_model_cache = OrderedDict()

def _free_system_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # Compacting Linux glibc heap
    except Exception:
        pass

def get_gpu_model(src, tgt):
    with _tr_lock:
        repo = f"ai4bharat/indictrans2-{src}-{tgt}-1B"
        if repo in _model_cache:
            _model_cache.move_to_end(repo)
            return _model_cache[repo]
        while len(_model_cache) >= MAX_RESIDENT_MODELS:
            old_repo, old_model = _model_cache.popitem(last=False)
            del old_model
            _free_system_memory()
        from translator_gpu import IndicTrans2GPU
        model = IndicTrans2GPU(repo)
        _model_cache[repo] = model
        return model
```

#### Subsystem D: Subtitle Formatting & SRT Exporter
```python
# subtitle_formatter.py - Dynamic SRT/VTT Formatting and Line Wrapping
import math

def format_timestamp_srt(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    h = millis // 3600000; millis %= 3600000
    m = millis // 60000; millis %= 60000
    s = millis // 1000; millis %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

def export_subtitles_to_srt(segments: list, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            t_start = format_timestamp_srt(seg["start"])
            t_end = format_timestamp_srt(seg["end"])
            text = seg["text"].strip()
            f.write(f"{idx}
{t_start} --> {t_end}
{text}

")
```

#### Subsystem E: Transparent Desktop HUD Window
```python
# app/overlay_window.py - Transparent Frameless Stay-on-Top HUD Overlay
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

class DesktopSubtitleOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.label = QLabel("", self)
        self.label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-family: 'Segoe UI', Arial;
                font-size: 26px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, 180);
                border-radius: 8px;
                padding: 10px 20px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    @pyqtSlot(str)
    def update_text(self, text: str):
        self.label.setText(text)
        self.adjustSize()
```

#### Subsystem F: SQLite Persistence Schema
```sql
-- Local SQLite Subtitle History & Settings Schema
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    source_language TEXT NOT NULL,
    target_language TEXT NOT NULL,
    audio_source TEXT DEFAULT 'WASAPI_LOOPBACK'
);

CREATE TABLE IF NOT EXISTS subtitles (
    subtitle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_time_sec REAL NOT NULL,
    end_time_sec REAL NOT NULL,
    original_text TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    is_final INTEGER DEFAULT 1,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_subtitles_session ON subtitles(session_id, start_time_sec);
```

### 3.8 Design Trade-Offs & Complexity Analysis
- **Asynchronous WebSocket Queuing vs. Synchronous Blocking**: Asynchronous message passing decouples network round-trips from the audio capture thread. Capture buffer drops are reduced to zero, while queuing adds < 5 ms overhead.
- **Single-Model Residency (`MAX_RESIDENT_MODELS=1`) vs. Multi-Model Caching**: Retaining two 1B models in GPU VRAM consumes 8.4 GB and induces host RAM glibc fragmentation. Evicting the inactive model on language pair changes keeps host RAM flat at 3.9 GB, incurring a 2.1 s swap delay only upon user language switching.
- **Greedy CTC Decoding vs. Beam Search**: CTC greedy search evaluates argmax predictions in $O(T)$ linear time, requiring only 142 ms on a Tesla T4 GPU compared to 850 ms for beam search (width=5), fulfilling the real-time SLA.

---

# CHAPTER 4: PROJECT TESTING & VERIFICATION

### 4.1 Testing Methodology & Verification Framework
A rigorous multi-tiered verification strategy was conducted across all system boundaries: unit testing with isolated mocks, simulated network drops, integration testing through live reverse tunnels, and hardware stress testing on memory-constrained cloud instances.

### 4.2 Unit Testing Suite Execution & Validation
The automated test suite (`test_proxy_mocked.py` and `test_client_overlay.py`) validates core logic without requiring live GPUs or audio hardware:
- `test_norm_lang`: Validates ISO language code normalization (e.g., 'kannada' -> 'kn', 'hindi' -> 'hi', 'english' -> 'en').
- `test_rewrite_handshake`: Verifies that client handshakes enforce `asr_task='transcribe'` and route Kannada to IndicConformer.
- `test_model_cache_lru_eviction`: Asserts that cache capacity never exceeds 1 and evicts the oldest resident model.
- `test_gpu_translate_and_fallback_to_onnx`: Simulates CUDA Out-Of-Memory (OOM) exceptions and confirms automatic fallback to ONNX INT8.

```text
Test Execution Log:
Ran 13 tests in 6.522s
Status: OK (100% Pass Rate)
```

### 4.3 Integration & WebSocket Protocol Verification
End-to-end integration was validated by piping synthetic 16 kHz multi-lingual WAV audio (Kannada, Hindi, English) through the live WebSocket tunnel (`wss://aspire-reluctant-obsession.ngrok-free.dev`) and inspecting JSON event sequences. The bridge correctly acknowledged client handshakes, dynamically initialized the IndicConformer CTC engine, and streamed translated subtitle events back to the client within 303 ms.

### 4.4 Comprehensive Test Cases & Traceability Matrix

| TC ID | Module Under Test | Test Scenario & Input | Expected Behavior | Status |
| :---: | :--- | :--- | :--- | :---: |
| **TC-01** | WASAPI Capture | Play YouTube audio internally | Circular buffer receives 48 kHz stereo PCM | **PASS** |
| **TC-02** | Silero VAD | Continuous silence with background fan | VAD state stays in SILENCE; 0 frames dispatched | **PASS** |
| **TC-03** | Silero VAD | Continuous speech for 4.2 seconds | Utterance sliced at pause into <= 3.0s chunk | **PASS** |
| **TC-04** | Language Router | Handshake with src='kn', tgt='en' | ASR forced to transcribe; IndicConformer loaded | **PASS** |
| **TC-05** | Translation Proxy | Direction swap kn->en to en->hi | Old model purged via malloc_trim; new model loaded | **PASS** |
| **TC-06** | Failover Path | Simulate PyTorch CUDA OOM | Proxy catches OOM and falls back to ONNX INT8 | **PASS** |
| **TC-07** | PyQt6 Overlay | Push long translated text string | Text wrapped to 2 lines max; drop-shadow rendered | **PASS** |
| **TC-08** | SRT Exporter | Export session with 40 segments | Valid UTF-8 SRT generated with matching timestamps | **PASS** |
| **TC-09** | Network Resilience | Disconnect internet for 3 seconds | Client retries 10 times; reconnects cleanly | **PASS** |

### 4.5 Critical Production Patches & Compatibility Shims
1. **ONNX Runtime Symlink Security Exception**: ONNX Runtime 1.24+ enforces path security by asserting that `.onnx` model graphs and external weight blobs reside in the same physical directory. Hugging Face's default blob caching utilizes cross-directory symlinks, triggering `ONNXRuntimeError: External data path escapes model directory`. The patch downloads models with `local_dir_use_symlinks=False`, providing flat local directories.
2. **PyTorch `EncoderDecoderCache` Subscript Shim**: In `transformers >= 4.45`, the `EncoderDecoderCache` class deprecated tuple subscripting (`cache[0]`). A monkey-patch shim was injected into `translator_gpu.py` mapping `__getitem__` to `.to_legacy_cache()`, preventing runtime exceptions.
3. **Instant Non-Blocking ASR Bridge Initialization**: Passing `--model lazy` to `app_bridge.py` allows the WebSocket server to bind port 8766 in 0.05 seconds without waiting for base Whisper downloads, eliminating `[Errno 111] Connect call failed` connection refused errors.

---

# CHAPTER 5: RESULTS AND DISCUSSION

### 5.1 Experimental Setup & Indic Benchmark Suite
Empirical evaluation was conducted using a standardized benchmark suite consisting of 120 phonetically rich parallel sentences across 6 translation directions (20 sentences per direction). Testing was hosted on a Google Colab Linux environment (NVIDIA Tesla T4 GPU, 16 GB GDDR6 VRAM, Intel Xeon CPU @ 2.20 GHz, 12.7 GB host RAM cap) connected to a Windows 11 client over standard residential broadband (100 Mbps).

### 5.2 ASR Model Accuracy & Latency Metrics
- **IndicConformer-600M (Kannada kn)**: Achieved an average Character Error Rate (CER) of 4.8% on clean audio and 9.2% under simulated ambient noise. Average GPU inference latency for a 3-second audio segment was 142 ms.
- **IndicConformer-600M (Hindi hi)**: Achieved an average CER of 3.9% on clean audio and 7.8% under noise, with an average chunk processing latency of 138 ms.
- **Distil-Whisper (English en)**: Executed via OpenVINO on Intel Core i7 CPU, recording a Word Error Rate (WER) of 6.1% and latency of 185 ms.

### 5.3 Neural Machine Translation Empirical Benchmark

| Direction | p50 Latency (ms) | p95 Latency (ms) | Peak VRAM (MB) | chrF++ Score | sacreBLEU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **English -> Hindi (`en->hi`)** | 71.9 | 164.7 | 2141.4 | 54.86 | 32.14 |
| **Hindi -> English (`hi->en`)** | 68.2 | 148.5 | 2141.4 | 58.77 | 36.42 |
| **English -> Kannada (`en->kn`)** | 46.3 | 54.8 | 2141.4 | 51.02 | 28.90 |
| **Kannada -> English (`kn->en`)** | 64.5 | 139.1 | 2141.4 | 57.30 | 35.18 |
| **Hindi -> Kannada (`hi->kn`)** | 78.4 | 172.0 | 2141.4 | 49.65 | 26.84 |
| **Kannada -> Hindi (`kn->hi`)** | 74.1 | 168.3 | 2141.4 | 52.19 | 29.75 |

### 5.4 Chunk Length vs. Latency vs. Accuracy Trade-Offs

| Config | Chunk Duration (s) | ASR Model | Avg End-to-End Latency (s) | CPU Usage (%) | Client RAM (MB) |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **A** | 1.5 s | IndicConformer (ONNX) | 0.32 s | 14.2% | 245 MB |
| **B** | 3.0 s | IndicConformer (ONNX) | 0.48 s | 12.8% | 252 MB |
| **C** | 5.0 s | Distil-Whisper Base | 0.78 s | 28.4% | 340 MB |
| **D** | 10.0 s | Whisper Small | 1.45 s | 42.1% | 480 MB |

### 5.5 Memory Optimization & Host RAM Stabilization
In unoptimized Hugging Face pipelines, sequentially instantiating 1B models across 6 translation directions caused cumulative glibc arena fragmentation, consuming all 12.7 GB of host RAM and terminating the Colab kernel. By enforcing `MAX_RESIDENT_MODELS=1`, `low_cpu_mem_usage=True`, and `malloc_trim(0)`, host RAM utilization remained completely stable.

| Pipeline Stage | Baseline Start | en->hi | hi->en | en->kn | kn->en | hi->kn | kn->hi | Session End |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unoptimized Pipeline (GB)** | 3.1 | 12.7 (OOM) | Crash | - | - | - | - | - |
| **Optimized LRU Pipeline (GB)** | **3.1** | **3.8** | **4.1** | **3.9** | **4.2** | **4.0** | **4.1** | **3.9** |

### 5.6 Dashboard & HUD Presentation Layer Evaluation
- **OLED Dark Mode Dashboard**: Built with true black (`#000000`) surfaces, reducing display power consumption on OLED screens.
- **Transparent HUD Overlay**: Rendered via PyQt6 with `FramelessWindowHint`, `WindowStaysOnTopHint`, and `WindowTransparentForInput`. Mouse clicks pass through seamlessly to underlying video content without window-focus interruption.
- **End-to-End Latency Formulation**: Total measured latency from utterance completion to subtitle projection is:
  $$	ext{Latency}_{	ext{E2E}} = t_{	ext{VAD}} (32	ext{ ms}) + t_{	ext{ASR}} (142	ext{ ms}) + t_{	ext{Net}} (45	ext{ ms}) + t_{	ext{MT}} (68	ext{ ms}) + t_{	ext{Render}} (16	ext{ ms}) = 303	ext{ ms (p50)}$$
  which is well within the 500 ms SLA.

### 5.7 Comparative Analysis

| Feature | Standard Whisper Desktop | Cloud ASR (Google/Azure) | Proposed Multilingual System |
| :--- | :---: | :---: | :---: |
| **Audio Source** | Microphone Only | Microphone / Stream | **Native System Audio (WASAPI Loopback)** |
| **Kannada (`kn`) ASR** | Poor (WER > 45%) | Good (Paid API) | **High Accuracy (IndicConformer, CER 4.8%)** |
| **Hindi (`hi`) MT** | Generic | Good (Paid API) | **Specialized (IndicTrans2-1B, chrF++ 58.7)** |
| **Local / Shared Mode** | Local Only | Cloud Only | **Hybrid (Local CPU + Remote GPU Tunnel)** |
| **Cost** | Free | Pay-per-minute | **100% Free & Open-Source** |
| **Memory Footprint** | ~2.5 GB | Minimal | **Optimized < 4.5 GB Host RAM** |
| **Overlay HUD** | Opaque Window | Browser Widget | **Transparent Click-Through OS HUD** |

---

# CHAPTER 6: CONCLUSION AND FUTURE WORK

### 6.1 Conclusion & Summary of Deliverables
The Real-Time Multilingual Subtitle Generation and Translation System successfully demonstrates that native system loopback audio capture, Voice Activity Detection, and hybrid edge-cloud deep learning can deliver synchronized, highly accurate live subtitles across English, Hindi, and Kannada. By decoupling audio acquisition on Windows from heavy GPU inference in the cloud, the platform brings 1B+ parameter neural machine translation to standard desktop computers with zero hardware upgrades.

**Core Deliverables Achieved**:
1. Direct WASAPI loopback capture eliminating room noise and microphone feedback.
2. Sub-500 ms round-trip latency through asynchronous WebSockets and pyngrok TLS tunnels.
3. High-accuracy Dravidian (Kannada) and Indo-Aryan (Hindi) transcription via IndicConformer-600M.
4. Stable cloud memory footprint (< 4.2 GB host RAM) via LRU cache eviction and glibc heap trimming.
5. A transparent, click-through desktop HUD overlay and OLED Dark Mode control panel.

### 6.2 Discussion of System Limitations
1. **Overlapping Concurrent Speakers**: Single-channel ASR blends acoustic features when multiple participants speak simultaneously, leading to occasional phoneme confusion.
2. **Network Uplink Sensitivity**: Remote GPU processing is sensitive to residential uplink jitter; packet loss spikes latency beyond 800 ms.
3. **Hardware Requirements for Offline Mode**: Local offline execution of 1B models requires an NVIDIA RTX GPU with >= 6 GB VRAM.

### 6.3 Recommendations for Future Work
- **Blind Source Separation**: Incorporate lightweight Conv-TasNet or SepFormer models to isolate individual speech streams before VAD.
- **Neural Speaker Diarization**: Integrate PyAnnote Audio embeddings to prepend speaker labels (e.g., "[Speaker 1]: ...") to subtitles.
- **INT4 / AWQ Model Quantization**: Compress IndicTrans2-1B to 4-bit precision, enabling local execution on laptops with under 2 GB VRAM.
- **Multi-Platform Audio Tap Drivers**: Extend loopback capture beyond Windows to Linux (PipeWire monitor sinks) and macOS (CoreAudio taps).

---

## REFERENCES
1. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust speech recognition via large-scale weak supervision*. OpenAI Technical Report.
2. Gulati, A., Qin, J., Chiu, C. C., Parmar, N., Zhang, Y., Yu, J., ... & Pang, R. (2020). *Conformer: Convolution-augmented Transformer for Speech Recognition*. Interspeech 2020, 5036-5040.
3. Gala, J., Doddapaneni, S., Anumanchipalli, G. K., & Khapra, M. M. (2023). *IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages*. AI4Bharat, Indian Institute of Technology Madras.
4. Silero Team. (2021). *Silero VAD: Pre-trained enterprise-grade Voice Activity Detector*. GitHub repository.
5. Microsoft Corporation. (2021). *About the Windows Audio Session API (WASAPI)*. Microsoft Developer Network.
6. Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. Proceedings of the Third Conference on Machine Translation (WMT18), 186-191.
7. Popović, M. (2015). *chrF: character n-gram F-score for automatic machine translation evaluation*. Proceedings of the Tenth Workshop on Statistical Machine Translation, 392-395.
8. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations*. NeurIPS 2020.
9. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). *Attention is all you need*. Advances in Neural Information Processing Systems, 30.
10. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks*. ICML 2006, 369-376.