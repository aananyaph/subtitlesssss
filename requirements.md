# System Dependencies and Library Requirements

This document outlines all libraries and dependencies required to run the **Real-Time Multilingual Subtitle Generation and Translation System** across both the **Client (Desktop PC)** and the **Server (Google Colab / GPU Node)**.

---

## 1. Quick Installation Commands

### A. Client Desktop (Windows PC)
Run in PowerShell within your project root:
```powershell
pip install -r requirements-client.txt
```
*Or manual install:*
```powershell
pip install PyQt6 pyaudiowpatch websockets numpy scipy
```

### B. Remote Inference Server (Google Colab / Linux GPU)
Run within your notebook or terminal:
```bash
pip install -r requirements-server.txt
```
*Or manual install:*
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install faster-whisper websockets pyngrok sacremoses sacrebleu sentencepiece indic-nlp-library jiwer scipy transformers
pip install --no-deps git+https://github.com/AI4Bharat/IndicTransToolkit.git
```

---

## 2. Comprehensive Library Breakdown by Subsystem

### A. Client Desktop & User Interface
| Library | Version | Role in Architecture |
| :--- | :--- | :--- |
| **`PyQt6`** | `>= 6.5.0` | Powers the OLED Dark Mode Control Panel and the frameless transparent desktop HUD subtitle overlay with click-through support (`WS_EX_TRANSPARENT`). |
| **`pyaudiowpatch`** | `>= 0.2.12` | Provides non-blocking Windows Audio Session API (WASAPI) loopback audio capture directly from output speaker endpoints without physical microphones. |
| **`websockets`** | `>= 12.0` | Asynchronous WebSocket client connecting the desktop client to the remote GPU translation proxy. |
| **`numpy`** | `>= 1.24.0` | High-performance array operations for PCM audio downmixing, stereo-to-mono normalization, and buffer slicing. |
| **`scipy`** | `>= 1.10.0` | High-fidelity audio resampling ($48\text{ kHz} \to 16\text{ kHz}$) and signal filtering. |

---

### B. Voice Activity Detection (VAD) & Segmentation
| Library | Version | Role in Architecture |
| :--- | :--- | :--- |
| **`silero-vad` / `onnx`** | `>= 4.0` | Deep neural Voice Activity Detection evaluating speech probability across $32\text{ ms}$ windows to segment continuous speech. |
| **`faster-whisper`** | `>= 0.10.0` | High-speed CTranslate2 Whisper implementation used for local fallback transcription and Silero VAD integration. |

---

### C. ASR (Speech Recognition) & Indic Audio Engines
| Library | Version | Role in Architecture |
| :--- | :--- | :--- |
| **`transformers`** | `>= 4.45.0` | Hugging Face model loading interface for `AutoModel` acoustic pipelines and sequence generation. |
| **`onnxruntime-gpu`** | `>= 1.18.0` | GPU-accelerated inference engine for the **AI4Bharat IndicConformer-600M** multilingual speech model. |
| **`torchaudio`** | `>= 2.1.0` | Audio tensor transformations, spectrogram loading, and phonetic signal normalization. |
| **`huggingface_hub`** | `>= 0.24.0` | Model repository authentication, gated model snapshot management, and local directory caching. |

---

### D. Neural Machine Translation (NMT) & Indic NLP
| Library | Version | Role in Architecture |
| :--- | :--- | :--- |
| **`torch`** | `>= 2.1.0` | Core deep learning framework executing **IndicTrans2-1B** sequence-to-sequence translation in fp16 precision. |
| **`IndicTransToolkit`** | `Latest (git)` | AI4Bharat official preprocessing and postprocessing toolkit for 22 scheduled Indian languages and English. |
| **`indic-nlp-library`** | `>= 0.92` | Indic script normalization, phonetic tokenization, and Unicode transliteration. |
| **`sentencepiece`** | `>= 0.1.99` | Subword tokenization and vocabulary encoding for multilingual translation models. |
| **`sacremoses`** | `>= 0.0.53` | Moses tokenization and detokenization routines for English and Romanized scripts. |

---

### E. Networking & Reverse Tunneling
| Library | Version | Role in Architecture |
| :--- | :--- | :--- |
| **`pyngrok`** | `>= 7.1.0` | Manages secure public reverse TLS tunnels bridging the remote Google Colab GPU instance to the local PC desktop app. |
| **`asyncio`** | `Built-in` | Asynchronous I/O event loops coordinating upstream bridge communication and translation worker dispatch. |

---

### F. Performance Evaluation & Benchmarking
| Library | Version | Role in Architecture |
| :--- | :--- | :--- |
| **`sacrebleu`** | `>= 2.3.0` | Computes standardized corpus-level and sentence-level BLEU and chrF++ translation quality metrics. |
| **`jiwer`** | `>= 3.0.0` | Measures Word Error Rate (WER) and Character Error Rate (CER) for ASR evaluation. |
| **`python-docx`** | `>= 1.1.0` | Automated compilation of technical documentation and academic project reports into Microsoft Word format. |

---

## 3. Platform Compatibility Matrix

| Feature | Windows Client (Local) | Colab / Linux Server (Remote) |
| :--- | :---: | :---: |
| **Operating System** | Windows 10 / 11 | Ubuntu Linux 22.04 LTS |
| **Python Version** | Python 3.10 – 3.12 | Python 3.10 – 3.13 |
| **Hardware Required** | Dual-Core CPU, 4 GB RAM | NVIDIA T4 / A100 GPU, 12 GB+ VRAM |
| **Host RAM Ceiling** | $\le 450\text{ MB}$ | $\le 4.2\text{ GB}$ (Compacted via `malloc_trim`) |
