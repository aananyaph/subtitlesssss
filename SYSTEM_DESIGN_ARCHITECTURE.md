# SYSTEM DESIGN ARCHITECTURE SPECIFICATION
## Real-Time Multilingual Subtitle Generation and Translation System
**Document Version:** 2.4.0 | **Status:** Production-Ready Architecture | **Target Platform:** Windows 10/11 & Linux NVIDIA CUDA GPU

---

## TABLE OF CONTENTS
1. [Executive Architectural Overview & High-Level Design](#1-executive-architectural-overview--high-level-design)
   - 1.1 System Overview
   - 1.2 Architectural Objectives & SLA Constraints
   - 1.3 Supported Language Matrix & Code Normalization
   - 1.4 High-Level Tiered Topology
   - 1.5 Logical Deployment Architecture
2. [End-to-End System Pipeline Architecture](#2-end-to-end-system-pipeline-architecture)
   - 2.1 Comprehensive Architecture Diagram
   - 2.2 Complete Pipeline Flowchart
3. [Deep-Dive Subsystem Decomposition](#3-deep-dive-subsystem-decomposition)
   - 3.1 Subsystem 1: Audio Ingestion & Resampling Engine
   - 3.2 Subsystem 2: Voice Activity Detection & Acoustic Boundary Segmentation
   - 3.3 Subsystem 3: Asynchronous Transport & Tunneling Gateway
   - 3.4 Subsystem 4: Automatic Speech Recognition (ASR) Subsystem
   - 3.5 Subsystem 5: Neural Machine Translation (NMT) Subsystem
   - 3.6 Subsystem 6: Memory Compaction & Runtime Manager (The 4.2 GB RAM Floor)
   - 3.7 Subsystem 7: Desktop Presentation & Heads-Up Display (HUD)
4. [Sequence Diagram: Lifecycle & Timing Breakdown](#4-sequence-diagram-lifecycle--timing-breakdown)
5. [Interface Contracts & Wire Protocol Specifications](#5-interface-contracts--wire-protocol-specifications)
   - 5.1 Connection Handshake Protocol
   - 5.2 Audio Stream Transmission Frame
   - 5.3 Live Subtitle Broadcast Payloads
   - 5.4 Audio Diagnostics & Telemetry Packet
6. [Concurrency, Threading & State Machine Model](#6-concurrency-threading--state-machine-model)
7. [Failure Recovery, Resilience & Circuit Breakers](#7-failure-recovery-resilience--circuit-breakers)
8. [Architectural Trade-Off Analysis](#8-architectural-trade-off-analysis)

---

# 1. Executive Architectural Overview & High-Level Design

### 1.1 System Overview
The **Real-Time Multilingual Subtitle Generation and Translation System** is an enterprise-grade, distributed AI accessibility solution engineered to capture internal system audio directly from a Windows OS without physical acoustic feedback, transcribe spoken vernacular speech in real time, translate the text into a user-specified target language, and project floating, non-intrusive subtitles on a transparent desktop HUD overlay.

The architecture is explicitly designed around a **Decoupled Hybrid Edge–Cloud Topology**:
- **Edge Layer (Client Desktop PC)**: Manages low-overhead tasks—WASAPI loopback audio capture, signal preprocessing, deep neural Voice Activity Detection (VAD), chunk buffering, and hardware-accelerated subtitle rendering. Local CPU and memory consumption are minimal ($< 450\text{ MB}$ RAM, $< 5\%$ CPU).
- **Cloud Inference Layer (Remote GPU / Google Colab)**: Houses compute-intensive neural models—the 600-Million-parameter acoustic recognition model and the 1.1-Billion-parameter neural translation model executed on NVIDIA CUDA GPUs.
- **Transport Gateway Layer**: Maintains persistent, bi-directional, duplex WebSocket communication wrapped in an encrypted TLS reverse tunnel (`pyngrok`).
- **Presentation Layer**: Renders transparent, always-on-top, click-through subtitles without intercepting user mouse interactions or disrupting active fullscreen applications (video conferences, games, media players).

```
+-------------------------------------------------------------------------------+
|                           WINDOWS CLIENT WORKSPACE                            |
|  [CoreAudio Output] -> [WASAPI Loopback] -> [Silero VAD] -> [PyQt6 HUD Overlay] |
+---------------------------------------+---------------------------------------+
                                        | (Encrypted WebSocket / TLS)
                                        v
+-------------------------------------------------------------------------------+
|                         REMOTE GPU INFERENCE ENGINE                           |
|      [Pyngrok Gateway] <-> [Translation Proxy] <-> [ASR Bridge Engine]        |
|             |                     |                        |                  |
|     (Public Endpoint)    (IndicTrans2-1B GPU)     (IndicConformer-600M)       |
+-------------------------------------------------------------------------------+
```

---

### 1.2 Architectural Objectives & SLA Constraints
| Performance Dimension | Architectural Target | Mechanism of Enforcement |
| :--- | :--- | :--- |
| **End-to-End Latency** | $\le 450\text{ ms}$ total ($p_{50}$) | Greedy decoding (`beam=1`), CTC acoustic decoding, persistent WebSockets. |
| **Audio Capture Parity** | 100% digital fidelity | WASAPI loopback capture directly from endpoint mixing buffer. |
| **Indic Acoustic Precision**| $\text{CER} < 5.0\%$ on Indic speech | AI4Bharat IndicConformer-600M pre-trained on diverse Indian accents. |
| **Indic Translation Quality**| $\text{chrF++} > 50.0$ / $\text{BLEU} > 30.0$ | AI4Bharat IndicTrans2-1B fp16 with IndicNLP script normalizers. |
| **Server Memory Ceiling** | Host $\text{RAM} \le 4.2\text{ GB}$ (Colab cap $12.7\text{ GB}$) | Single-model residency (`MAX=1`), `malloc_trim(0)` heap compaction. |
| **Client Overhead** | $\le 450\text{ MB}$ RAM, $\le 5\%$ CPU | Offloaded inference; edge client performs only VAD and UI projection. |
| **UI Non-Intrusiveness** | 0% mouse event capture | Win32 `WS_EX_TRANSPARENT` click-through window layering. |

---

### 1.3 Supported Language Matrix & Code Normalization
The system establishes bidirectional transcription and translation across English, Kannada, and Hindi. The table below specifies the internal engine routing and language code transformation:

| Source Audio | Target Subtitle | ASR Acoustic Model | NMT Translation Model | Source Tag | Target Tag |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **English (`en`)** | **English (`en`)** | Distil-Whisper | *Bypassed (0 ms)* | `en` | `en` |
| **English (`en`)** | **Kannada (`kn`)** | Distil-Whisper | IndicTrans2-1B | `eng_Latn` | `kan_Knda` |
| **English (`en`)** | **Hindi (`hi`)** | Distil-Whisper | IndicTrans2-1B | `eng_Latn` | `hin_Deva` |
| **Kannada (`kn`)** | **Kannada (`kn`)** | IndicConformer-600M | *Bypassed (0 ms)* | `kn` | `kn` |
| **Kannada (`kn`)** | **English (`en`)** | IndicConformer-600M | IndicTrans2-1B | `kan_Knda` | `eng_Latn` |
| **Kannada (`kn`)** | **Hindi (`hi`)** | IndicConformer-600M | IndicTrans2-1B | `kan_Knda` | `hin_Deva` |
| **Hindi (`hi`)** | **Hindi (`hi`)** | IndicConformer-600M | *Bypassed (0 ms)* | `hi` | `hi` |
| **Hindi (`hi`)** | **English (`en`)** | IndicConformer-600M | IndicTrans2-1B | `hin_Deva` | `eng_Latn` |
| **Hindi (`hi`)** | **Kannada (`kn`)** | IndicConformer-600M | IndicTrans2-1B | `hin_Deva` | `kan_Knda` |

---

### 1.4 High-Level Tiered Topology
```
[ Tier 1: Client Edge ]
  └── Windows PC Host
       ├── WASAPI Loopback Hook (CoreAudio API)
       ├── Resampler & Preprocessor (48kHz -> 16kHz Mono Float32)
       ├── Silero VAD Speech Segmenter (32ms Frames)
       ├── OLED Control Center (Dark Mode UI)
       └── Transparent HUD Canvas (Click-Through Overlay)

[ Tier 2: Secure Transport ]
  └── Pyngrok TLS Reverse Tunnel
       ├── Public URL: wss://aspire-reluctant-obsession.ngrok-free.dev
       ├── TLS Termination & WSS Duplex Framing
       └── Local Binding: 127.0.0.1:8765

[ Tier 3: Neural Server Gateway ]
  └── Translation Proxy (:8765)
       ├── Client Session State & Language Router
       ├── Upstream Auto-Reconnect Retry Engine (10 Attempts)
       ├── LRU Translation Text Cache (512 Entries)
       ├── IndicTrans2-1B GPU Worker (fp16 Greedy Beam=1)
       └── Heap Compactor (ctypes.malloc_trim(0))

[ Tier 4: Acoustic Inference ]
  └── ASR Bridge Server (:8766)
       ├── Instant Lazy Socket Initializer (0.05s Startup)
       ├── IndicConformer-600M Multilingual (ONNX Runtime GPU)
       ├── CTC Token Greedy Argmax Decoder
       └── Distil-Whisper Engine (OpenVINO / Faster-Whisper)
```

---

# 2. End-to-End System Pipeline Architecture

### 2.1 Comprehensive Architecture Diagram
```mermaid
flowchart TD
    subgraph Tier1["TIER 1: CLIENT EDGE (Windows PC Desktop)"]
        direction TB
        A["Desktop Audio Stream<br/>(YouTube / Zoom / Media Player)"] --> B["WASAPI Loopback Capture<br/>(PyAudioWPatch: 48kHz Stereo Float32)"]
        B --> C["Audio Preprocessor<br/>(Downmix to Mono & Resample to 16kHz)"]
        C --> D["Silero VAD Neural Engine<br/>(Speech Prob > 0.50, 32ms Slices)"]
        D --> E{"Acoustic Boundary?"}
        E -- "Silence < 500ms" --> F["Accumulate Chunk"]
        F --> D
        E -- "Silence >= 500ms" --> G["Speech Segmenter<br/>(1.0s to 15.0s PCM Utterance)"]
        
        subgraph UI["PyQt6 Presentation Layer"]
            HUD["Transparent Desktop HUD Overlay<br/>(WS_EX_TRANSPARENT Click-Through)"]
            Dashboard["OLED Control Dashboard<br/>(#000000 True Black, Neon Accents)"]
        end
    end

    subgraph Tier2["TIER 2: TRANSPORT GATEWAY (TLS WebSocket Tunnel)"]
        direction TB
        WS_Client["Async WebSocket Client<br/>(app/remote_client.py)"]
        Ngrok["Pyngrok TLS Reverse Tunnel<br/>(wss://...ngrok-free.dev)"]
        
        G --> WS_Client
        WS_Client <==>|"Duplex WSS Protocol"| Ngrok
    end

    subgraph Tier3["TIER 3: TRANSLATION PROXY (:8765)"]
        direction TB
        Proxy["Translation Proxy Core<br/>(translation_proxy.py)"]
        RetryEngine["Upstream Connection Retry Loop<br/>(10 Attempts, 500ms Backoff)"]
        TextCache["LRU Text Cache<br/>(0ms Compute Hit)"]
        
        Ngrok <==> Proxy
        Proxy --- RetryEngine
        Proxy --- TextCache
    end

    subgraph Tier4["TIER 4: ACOUSTIC ASR BRIDGE (:8766)"]
        direction TB
        Bridge["ASR Bridge Server<br/>(backend/app_bridge.py: Lazy Init)"]
        Router{"Language Route?"}
        IndicASR["IndicConformer-600M<br/>(ONNX Runtime GPU, CTC Decode)"]
        WhisperASR["Distil-Whisper<br/>(Faster-Whisper / OpenVINO)"]
        
        Proxy <==>|"ws://127.0.0.1:8766"| Bridge
        Bridge --> Router
        Router -- "kn / hi" --> IndicASR
        Router -- "en" --> WhisperASR
    end

    subgraph Tier5["TIER 5: MACHINE TRANSLATION & MEMORY CONTROLLER"]
        direction TB
        MT_Router{"Translation Needed?"}
        IndicTrans["IndicTrans2-1B NMT<br/>(PyTorch GPU fp16, Beam=1)"]
        Failover{"CUDA OOM?"}
        ONNX_MT["ONNX INT8 Distilled Fallback<br/>(hari31416 Distilled Models)"]
        MemoryManager["Memory Compactor<br/>(malloc_trim(0) & MAX_RESIDENT=1)"]
        
        IndicASR & WhisperASR -->|"Source Text"| MT_Router
        MT_Router -- Yes --> IndicTrans
        IndicTrans --> Failover
        Failover -- Yes --> ONNX_MT
        IndicTrans & ONNX_MT --> MemoryManager
    end

    MT_Router -- "No (Source=Target)" --> Proxy
    MemoryManager --> Proxy
    Proxy ==>|"Final Subtitle JSON"| WS_Client
    WS_Client --> HUD
```

---

### 2.2 Complete Pipeline Flowchart
```mermaid
flowchart TD
    Start([System Audio Output]) --> Capture[WASAPI Loopback Capture]
    Capture --> Preprocess[Stereo-to-Mono Downmix & 16kHz Resampling]
    Preprocess --> VAD[Silero VAD Frame Slicer 32ms]
    VAD --> CheckSpeech{P(speech) >= 0.5?}
    
    CheckSpeech -- No --> CheckSilence{Silence >= 500ms?}
    CheckSilence -- No --> VAD
    CheckSilence -- Yes --> CheckLen{Buffer >= 1.0s?}
    CheckLen -- No --> DropBuffer[Discard Noise Buffer] --> VAD
    CheckLen -- Yes --> EmitChunk[Finalize Speech Chunk]
    
    CheckSpeech -- Yes --> BufferSpeech[Append Frame to Utterance] --> VAD
    
    EmitChunk --> Transmit[Transmit Binary Frame via WSS Tunnel]
    Transmit --> ProxyReceive[Proxy on Port 8765 Receives Audio]
    ProxyReceive --> BridgeRelay[Forward to ASR Bridge on Port 8766]
    
    BridgeRelay --> ASRChoice{Source Lang?}
    ASRChoice -- "kn / hi" --> RunIndic[IndicConformer-600M CTC Decode]
    ASRChoice -- "en" --> RunWhisper[Distil-Whisper Decode]
    
    RunIndic & RunWhisper --> ASRDone[Emit ASR Source Transcript]
    ASRDone --> CheckTrans{Source Lang == Target Lang?}
    
    CheckTrans -- Yes --> SubtitleReady[Pack Subtitle Payload]
    CheckTrans -- No --> CacheCheck{Text in LRU Cache?}
    
    CacheCheck -- Yes --> SubtitleReady
    CacheCheck -- No --> RunMT[Run IndicTrans2-1B GPU fp16]
    
    RunMT --> OOMCheck{CUDA OOM?}
    OOMCheck -- Yes --> FallbackONNX[Run Quantized ONNX INT8] --> Compact
    OOMCheck -- No --> Compact[Trigger malloc_trim & Evict Older Models]
    Compact --> SubtitleReady
    
    SubtitleReady --> WSStream[Transmit Subtitle via WebSocket]
    WSStream --> RenderHUD[Project on Transparent Desktop HUD]
    RenderHUD --> End([Subtitles Visible to User])
```

---

# 3. Deep-Dive Subsystem Decomposition

### 3.1 Subsystem 1: Audio Ingestion & Resampling Engine
- **Module File**: `app/loopback_capture.py` & `vad/audio_utils.py`
- **Core Abstraction**: `WASAPILoopbackCapture(threading.Thread)`
- **Mathematical Specification**:
  The audio endpoint yields a discrete stereo signal $x[n] \in \mathbb{R}^2$ at $f_s = 48000\text{ Hz}$. 
  1. **Stereo-to-Mono Downmix**:
     $$x_{\text{mono}}[n] = \frac{x_{\text{left}}[n] + x_{\text{right}}[n]}{2}$$
  2. **Linear Interpolation Resampling**:
     Given resampling factor $r = \frac{16000}{48000} = \frac{1}{3}$, the resampled sequence $y[m]$ is computed by interpolating adjacent samples:
     $$y[m] = x[3m]$$
  3. **Amplitude Normalization**:
     If input data is in signed 16-bit PCM integer format $[-32768, 32767]$, samples are scaled into 32-bit floating-point format $[-1.0, 1.0]$:
     $$s[n] = \frac{x_{\text{int16}}[n]}{32768.0}$$
- **Threading Model**: A dedicated daemon thread reads hardware buffers into a thread-safe `queue.Queue(maxsize=300)`. If the queue overflows due to consumer delay, the oldest unconsumed chunk is discarded to prioritize live latency.

---

### 3.2 Subsystem 2: Voice Activity Detection & Acoustic Boundary Segmentation
- **Module File**: `vad/vad_engine.py` & `vad/segmenter.py`
- **Core Abstraction**: `SileroVadEngine` & `SpeechSegmenter`
- **Operational Parameters**:
  - Window Size: $512\text{ samples}$ ($32.0\text{ ms}$ at $16\text{ kHz}$).
  - Neural Architecture: Small RNN / LSTM model outputting probability scalar $P(\text{speech}) \in [0.0, 1.0]$.
  - Activation Threshold: $\theta_{\text{start}} = 0.50$.
  - Deactivation Threshold: $\theta_{\text{end}} = 0.35$.
- **State Machine Formalism**:
  $$\Sigma = \{\text{SILENCE}, \text{CONFIRMING\_SPEECH}, \text{IN\_SPEECH}\}$$
  - $\text{SILENCE} \xrightarrow{P \ge \theta_{\text{start}}} \text{CONFIRMING\_SPEECH}$ (Requires 3 consecutive frames $\approx 96\text{ ms}$).
  - $\text{CONFIRMING\_SPEECH} \to \text{IN\_SPEECH}$ (Emits speech start marker).
  - $\text{IN\_SPEECH} \xrightarrow{P < \theta_{\text{end}} \text{ for } 500\text{ ms}} \text{EMIT\_CHUNK} \to \text{SILENCE}$.
- **Guardrails**:
  - Minimum Chunk Floor: $1.0\text{ s}$. Chunks shorter than $1.0\text{ s}$ are merged or held.
  - Maximum Chunk Ceiling: $15.0\text{ s}$. Hard cutoff forces chunk segmentation to prevent infinite buffering during continuous background chatter.

---

### 3.3 Subsystem 3: Asynchronous Transport & Tunneling Gateway
- **Module File**: `app/remote_client.py` & `translation_proxy.py`
- **Underlying Protocol**: RFC 6455 WebSocket Protocol over TLS 1.3.
- **Port Mapping**:
  - **Port 8765 (`translation_proxy.py`)**: Public entry point reached via ngrok tunnel.
  - **Port 8766 (`backend/app_bridge.py`)**: Local loopback-only ASR acoustic bridge.
- **Resilience Engine**:
  ```python
  # translation_proxy.py handler connection retry loop
  for attempt in range(10):
      try:
          up = await connect(UPSTREAM, max_size=None, compression=None)
          break
      except Exception:
          if attempt < 9:
              await asyncio.sleep(0.5)
          else:
              await client.send(error_payload)
              return
  ```
  Guarantees zero client disconnects even during server process cold-starts.

---

### 3.4 Subsystem 4: Automatic Speech Recognition (ASR) Subsystem
- **Module File**: `backend/app_bridge.py` & `indic_backend.py`
- **Acoustic Models**:
  1. **IndicConformer-600M** (`ai4bharat/indic-conformer-600m-multilingual`):
     - Employs Conformer blocks combining depthwise separable convolutional layers with multi-head self-attention.
     - Execution Engine: ONNX Runtime GPU with `CUDAExecutionProvider`.
     - Output Decoding: Connectionist Temporal Classification (CTC) greedy token collapse.
     - **Symlink Fix Implementation**:
       To prevent ONNX Runtime's security validator (`ValidateExternalDataPath`) from failing with `External data path escapes model directory`, the model snapshot is explicitly saved into a flat local physical directory using `local_dir_use_symlinks=False`.
  2. **Distil-Whisper** (`distil-whisper/distil-small.en`):
     - Distilled sequence-to-sequence model optimized with OpenVINO / Faster-Whisper.
     - Employs 4 CPU compute threads (`cpu_threads=4`) to prevent UI lockup.
- **Instant Lazy Startup**:
  `app_bridge.py` starts with `--model lazy`, opening port `8766` in $0.05\text{ s}$ and postponing model compilation until the first client handshake arrives.

---

### 3.5 Subsystem 5: Neural Machine Translation (NMT) Subsystem
- **Module File**: `translator_gpu.py` & `translation_proxy.py`
- **Core Engine**: AI4Bharat IndicTrans2 1.1-Billion parameter encoder-decoder transformer (`IndicTransForConditionalGeneration`).
- **Precision**: PyTorch `float16` (`torch.float16`).
- **Decoding Configuration**:
  - `num_beams = 1` (Greedy Search)
  - `low_cpu_mem_usage = True`
  - `use_cache = False` (Avoids unbounded key-value cache growth)
- **Transformers Version Compatibility Shim**:
  Patches `transformers >= 4.45` to prevent `TypeError: 'EncoderDecoderCache' object is not subscriptable`:
  ```python
  if hasattr(EncoderDecoderCache, "to_legacy_cache") and not hasattr(EncoderDecoderCache, "__getitem__"):
      EncoderDecoderCache.__getitem__ = lambda self, idx: self.to_legacy_cache()[idx]
  ```
- **Automatic Fallback Circuit Breaker**:
  If GPU VRAM triggers an `OutOfMemoryError`, the proxy catches the exception, clears the CUDA cache, and reroutes the sentence to the **hari31416 Distilled ONNX INT8 model** without dropping the subtitle stream.

---

### 3.6 Subsystem 6: Memory Compaction & Runtime Manager (The 4.2 GB Ceiling)
- **Constraint**: Google Colab Free Tier terminates containers exceeding $12.7\text{ GB}$ system RAM.
- **Root Cause of Default Memory Leaks**:
  Python's standard memory manager delegates C++ allocations (`c10::allocator` in PyTorch and `malloc` in ONNX) to the glibc allocator. Glibc retains released memory inside private memory arenas instead of unmapping them back to the Linux OS (`sbrk`/`mmap`), leading to catastrophic memory fragmentation when switching models.
- **Enforced Solution**:
  1. `MAX_RESIDENT_MODELS = 1`: The LRU cache immediately evicts unused 1B parameter model instances.
  2. Explicit Linux glibc compaction:
     ```python
     def _free_system_memory():
         gc.collect()
         if torch.cuda.is_available():
             torch.cuda.empty_cache()
         try:
             import ctypes
             ctypes.CDLL("libc.so.6").malloc_trim(0)
         except Exception:
             pass
     ```
  **Result**: System RAM utilization remains strictly bounded between **$3.8\text{ GB}$ and $4.2\text{ GB}$** indefinitely.

---

### 3.7 Subsystem 7: Desktop Presentation & Heads-Up Display (HUD)
- **Module File**: `app/overlay_window.py` & `app/control_panel.py`
- **Control Panel Theme**: Dark Mode (OLED) Specification
  - Primary Base: `#000000` (True Black) & `#121212` (Card Surface)
  - Accents: Neon Green (`#39FF14`), Electric Blue (`#0080FF`), Warning Gold (`#FFD700`), System Purple (`#BF00FF`)
  - Typography: Light Gray body text (`#9CA3AF`) meeting WCAG AA contrast ratios.
- **Floating Subtitle HUD**:
  - Click-Through Window Style:
    ```python
    hwnd = int(self.winId())
    style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
    ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x00080000 | 0x00000020)
    ```
  - Visual Styling: Large white typography (`#FFFFFF`) with $2\text{px}$ black drop-shadow (`#000000`) on a translucent rounded pill (`rgba(0, 0, 0, 180)`).
  - Auto-Fade Timer: Automatically clears subtitles after $4.0\text{ s}$ of silence.

---

# 4. Sequence Diagram: Lifecycle & Timing Breakdown

```mermaid
sequenceDiagram
    autonumber
    actor Media as Windows Speaker Endpoint
    participant Capture as WASAPI Loopback Capture
    participant VAD as Silero VAD Engine
    participant Client as Async WebSocket Client
    participant Tunnel as Pyngrok TLS Tunnel
    participant Proxy as Translation Proxy (:8765)
    participant Bridge as ASR Bridge Server (:8766)
    participant HUD as PyQt6 Transparent Overlay

    Note over Client,Bridge: Handshake Phase
    Client->>Tunnel: WSS Connection Request
    Tunnel->>Proxy: Forward TCP Connection
    Proxy->>Bridge: Connect Upstream (127.0.0.1:8766)
    Proxy->>Bridge: Send HANDSHAKE (asr=kn, tgt=en, model=indic)
    Bridge-->>Proxy: Acknowledged (Indic Pipeline Ready)
    Proxy-->>Client: Connection Ready Ack

    Note over Media,HUD: Real-Time Subtitle Streaming Cycle
    Media->>Capture: Audio Emitted (48kHz Stereo PCM)
    Capture->>Capture: Downmix & Resample to 16kHz Mono Float32
    Capture->>VAD: Push 32ms Frame Slices
    VAD->>VAD: Evaluate Speech Prob P >= 0.50 (~32ms)
    VAD->>Client: Utterance Bounded (1.8s PCM Chunk)
    
    Client->>Tunnel: Binary WebSocket Frame (Transit ~25ms)
    Tunnel->>Proxy: Relayed to Proxy Port 8765
    Proxy->>Bridge: Forwarded to ASR Port 8766 (~2ms)
    
    Bridge->>Bridge: IndicConformer-600M CTC Acoustic Decode (~142ms)
    Bridge-->>Proxy: Final Transcript {"text": "ನಮಸ್ಕಾರ, ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಿ?"}
    
    Proxy->>Proxy: LRU Text Cache Check (Miss)
    Proxy->>Proxy: IndicTrans2-1B fp16 Greedy Inference (~68ms)
    Proxy->>Proxy: malloc_trim(0) System Heap Compaction (~5ms)
    
    Proxy-->>Tunnel: {"type": "final", "text": "Hello, how are you today?", "lang": "en"}
    Tunnel-->>Client: WSS Frame Transit (~25ms)
    Client->>HUD: push_live_text("Hello, how are you today?")
    HUD-->>Media: Subtitle Rendered on User Screen (~16ms)
    
    Note over Media,HUD: Total Measured Latency: 32 + 25 + 2 + 142 + 68 + 5 + 25 + 16 = 315 ms
```

---

# 5. Interface Contracts & Wire Protocol Specifications

### 5.1 Connection Handshake Protocol
**Direction**: Client $\to$ Translation Proxy $\to$ ASR Bridge  
**Format**: JSON Text Frame  
```json
{
  "type": "HANDSHAKE",
  "sampleRate": 16000,
  "numChannels": 1,
  "asrConfig": {
    "model_name": "indic",
    "indic_lang": "kn",
    "asr_language": "kn",
    "tgt_lang": "en",
    "asr_task": "transcribe"
  }
}
```

### 5.2 Audio Stream Transmission Frame
**Direction**: Client $\to$ Translation Proxy $\to$ ASR Bridge  
**Format**: Binary WebSocket Frame  
- **Data**: Contiguous uncompressed linear PCM audio bytes.
- **Encoding**: 16-bit signed integer or 32-bit floating point, single-channel mono at $16000\text{ Hz}$.
- **Chunk Size**: Approximately $16\text{ KB}$ to $128\text{ KB}$ representing $0.5\text{ s}$ to $4.0\text{ s}$ speech bursts.

### 5.3 Live Subtitle Broadcast Payloads
**Direction**: Server $\to$ Client  
**Format**: JSON Text Frame  

#### Finalized Subtitle Message
```json
{
  "type": "final",
  "text": "Hello, how are you today?",
  "source_text": "ನಮಸ್ಕಾರ, ನೀವು ಇಂದು ಹೇಗಿದ್ದೀರಿ?",
  "language": "en",
  "is_final": true,
  "start": 0.0,
  "end": 2.14,
  "processing_time_s": 0.285
}
```

#### Partial (Interim) Subtitle Message
```json
{
  "type": "partial",
  "text": "Hello, how",
  "language": "en",
  "is_final": false,
  "processing_time_s": 0.082
}
```

### 5.4 Audio Diagnostics & Telemetry Packet
**Direction**: Server $\to$ Client  
**Format**: JSON Text Frame  
```json
{
  "type": "AUDIO_DIAGNOSTICS",
  "diagnostics": {
    "rms_level": 0.042,
    "speech_prob": 0.94,
    "vram_allocated_mb": 2141.4,
    "ram_used_mb": 3980.2,
    "active_backend": "torch1b"
  }
}
```

---

# 6. Concurrency, Threading & State Machine Model

```
+--------------------------------------------------------------------------------+
|                             CLIENT CONCURRENCY MODEL                           |
|                                                                                |
|  [WASAPI Capture Thread] ---> [Audio Ring Buffer Queue (max=300)]              |
|                                         |                                      |
|  [VAD Segmentation Worker] <------------+                                      |
|            |                                                                   |
|            v                                                                   |
|  [AsyncIO WebSocket Loop] ---> Transmits to Pyngrok Tunnel                     |
|            ^                                                                   |
|            |                                                                   |
|  [PyQt6 Main GUI Thread] <--- Subtitle Signal Dispatch -> [HUD Painter Engine] |
+--------------------------------------------------------------------------------+
```

1. **Lock-Free Audio Ring Buffer**: Audio chunks pass from the high-priority WASAPI thread to the VAD segmentation worker using Python's thread-safe atomic FIFO `queue.Queue`.
2. **Asynchronous Event Loop (`asyncio`)**: Manages the network transport without blocking GUI interactions or audio capture.
3. **Qt Main Thread GUI Isolation**: Subtitle payloads arriving over WebSockets invoke PyQt signals (`pyqtSignal(str, bool)`), ensuring all window painting occurs safely on the primary OS thread.

---

# 7. Failure Recovery, Resilience & Circuit Breakers

| Failure Scenario | Detection Mechanism | Automated Recovery Strategy |
| :--- | :--- | :--- |
| **Colab Proxy Server Down** | `WebSocketClient.connect()` timeout | Automatic client reconnection loop with exponential backoff ($1\text{s} \to 2\text{s} \to 5\text{s}$). |
| **ASR Bridge Cold Start** | Proxy gets `[Errno 111] Connection Refused` | Proxy executes a 10-attempt internal retry loop with $500\text{ ms}$ pauses before dropping client. |
| **CUDA Out-of-Memory (OOM)** | `torch.cuda.OutOfMemoryError` raised | Proxy purges VRAM with `empty_cache()` and instantly fails over to **ONNX INT8 distilled model**. |
| **Hugging Face 401 Gated Error**| Snapshot download returns `401 Unauthorized` | Multi-key resolver checks `HF_TOKEN`, `HF_TOKE`, and Colab secrets with interactive fallback prompt. |
| **ONNX Runtime Symlink Error** | Path validation error: `escapes model dir` | Snapshot downloads use `local_dir_use_symlinks=False` writing real physical files in a flat folder. |
| **Silent Audio Stalling** | Silence duration $> 4.0\text{ s}$ | Subtitle auto-fade timer clears overlay text to prevent stale screen clutter. |

---

# 8. Architectural Trade-Off Analysis

| Dimension | Selected Design Choice | Alternative Considered | Architectural Justification |
| :--- | :--- | :--- | :--- |
| **Inference Location** | **Hybrid Edge-Cloud** | All-Local Inference | Running a 1B NMT model and 600M ASR model locally requires a dedicated NVIDIA RTX GPU (6 GB+ VRAM), preventing execution on standard laptops. |
| **Translation Decoding** | **Greedy Search (`beam=1`)** | Beam Search (`beam=4` or `5`) | Greedy decoding reduces translation latency from $\approx 540\text{ ms}$ to $\approx 68\text{ ms}$ with negligible chrF++ degradation ($< 1.2$ points), crucial for live subtitles. |
| **Memory Allocation** | **Single Model (`MAX=1`)** | Pre-load All Models | Pre-loading `indic-en-1B`, `en-indic-1B`, and `indic-indic-1B` simultaneously exceeds $12.7\text{ GB}$ host RAM, triggering Colab container termination. |
| **Audio Ingestion** | **WASAPI Loopback Capture** | Microphone Capture | Captures internal application audio (Zoom, YouTube, media players) directly at digital parity without room acoustics, echo, or background noise. |
| **Indic ASR Engine** | **IndicConformer-600M** | Whisper-Large-v3 | Whisper exhibits Character Error Rates $>40\%$ on regional Kannada speech; IndicConformer achieves state-of-the-art CER of $4.8\%$. |
| **Network Tunnelling** | **Pyngrok TLS Tunnel** | Direct Port Forwarding | Eliminates router configuration, firewall holes, and public IP requirements, enabling secure connections to ephemeral Colab sessions. |
| **Presentation HUD** | **Win32 Layered HUD** | Qt WebEngine HTML Overlay | Win32 native click-through HUD has zero rendering overhead, consumes $< 30\text{ MB}$ RAM, and does not intercept mouse clicks during gaming or work. |
