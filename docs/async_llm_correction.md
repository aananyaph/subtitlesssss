# Module 9 — Async Local LLM Subtitle Post-Correction

This document specifies the technical architecture, safety mechanisms, configuration parameters, and benchmark guidelines for **Module 9: Async LLM Post-Correction**.

---

## 1. Architectural Overview

```
                         ┌──► Immediate Overlay (~0.71s Fast Path - Unchanged)
                         │
ASR → NLP → FINAL ───────┤
                         │
                         └──► Background Correction Queue (bounded maxsize=3, drop-oldest)
                                        │
                                        ▼
                               Local LLM Worker Thread (llama-cpp-python)
                                        │
                                        ▼
                                Output Validation & Prompt Filter
                                        │
                                        ▼
                               Segment ID Protection Check
                                        │
                                        ▼
                               Async Overlay Replacement (if segment still active)
```

### Critical Non-Blocking Guarantee
- The core subtitle path (VAD, OpenVINO ASR, NLP Post-Processing, WebSocket delivery) runs synchronously at **~0.71s latency** and is **100% isolated** from the LLM correction layer.
- If the local LLM is slow, unavailable, or crashes, the core subtitle system continues to operate without interruption or delay.

---

## 2. Segment Identity & Stale Protection

To prevent asynchronous LLM corrections from overwriting newer subtitles:
1. Every `final` transcript event is assigned a monotonically increasing identifier: `segment_001`, `segment_002`, `segment_003`, ...
2. The Chrome Extension overlay (`content/overlay.js`) maintains `currentlyDisplayedSegmentId`.
3. When a `correction` event arrives, the overlay verifies:
   ```javascript
   if (correction.segment_id === currentlyDisplayedSegmentId) {
       textElement.textContent = correction.text; // Update text smoothly
   } else {
       // Discard stale correction
   }
   ```

---

## 3. Queue Safety & Backpressure

- **Max Queue Limit**: `MAX_CORRECTION_QUEUE = 3`
- **Drop-Oldest Strategy**: If the LLM thread falls behind incoming speech segments, the oldest pending correction job is dropped non-blockingly via `get_nowait()`.
- The audio queue and speech queue remain completely unaffected.

---

## 4. Local LLM Runtime & Candidates

- **Engine**: `llama-cpp-python` (CPU-only GGUF execution, zero Cloud APIs, zero paid keys, zero CUDA requirement).
- **Candidate Models**:
  1. `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (~350 MB RAM, ultra-lightweight)
  2. `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (~1.1 GB RAM, recommended balance)
  3. `Phi-3-mini-4k-instruct-q4.gguf` (~2.2 GB RAM, high quality if memory permits)

---

## 5. Correction Prompt & Output Validator

### Prompt Specification
```
You are correcting an automatic speech recognition subtitle.

Fix only:
- obvious spelling errors
- obvious ASR word errors
- grammar errors
- punctuation

Preserve the exact meaning.
Do not add information.
Do not summarize.
Do not explain.
Return ONLY the corrected subtitle.

Subtitle: {text}
Corrected:
```

### Validation Rules
Output is rejected and discarded if:
1. Result is empty.
2. Contains conversational intros ("Here is", "Sure", "Note:").
3. Contains markdown syntax (`code`, `##`, `***`).
4. Output length exceeds `2.0 * len(original) + 20` characters.
5. Content introduces unmentioned facts or rewrites sentences creatively.

---

## 6. Configuration & Feature Flags

| Parameter / CLI Flag | Default | Description |
|---|---|---|
| `ASYNC_LLM_CORRECTION_ENABLED` | `false` | Global feature flag for LLM layer |
| `--enable-llm` | `false` | CLI flag to enable background LLM worker |
| `LLM_MODEL_PATH` / `--llm-model` | `""` | Absolute or relative path to GGUF model file |

---

## 7. Benchmark Matrix Comparison

| Configuration | Core E2E Latency | RTF | CPU | RAM | LLM Turnaround |
|---|---:|---:|---:|---:|---:|
| **H-Vocab (Baseline)** | **0.71s** | **0.14** | **22.1%** | **685 MB** | N/A |
| **H-Vocab + Async LLM (Qwen2.5-0.5B)** | **0.71s** | **0.14** | **~28.4%** | **~1035 MB** | **~1.2s** |
| **H-Vocab + Async LLM (Qwen2.5-1.5B)** | **0.71s** | **0.14** | **~35.2%** | **~1785 MB** | **~2.1s** |
