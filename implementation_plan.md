# Module 9 Step 1 Implementation Plan — Async LLM Post-Correction

This plan details the technical architecture and implementation for **Module 9 Step 1: Async LLM Post-Correction**.

---

## 1. Architecture Overview

```
                      Fast Path (Instant - Sub-second ~0.71s)
   ASR → NLP Post-Processor → FINAL Event ───► Chrome Overlay (Initial Display)
                                │
                                ▼ (if ASYNC_LLM_CORRECTION_ENABLED=true)
                        Correction Queue (bounded maxsize=3, drop-oldest)
                                │
                                ▼
                       LLM Worker Thread (dedicated Python daemon thread)
                                │
                                ▼
                       llama-cpp-python GGUF Model (loaded ONCE at startup)
                                │
                                ▼
                     Output Validation & Prompt Filtering
                                │
                                ▼ (if valid & segment_id matches current visible text)
                        Corrected Event ───► Chrome Overlay Text Replacement
```

---

## 2. Key Technical Guarantees

1. **Non-Blocking Real-Time Critical Path**:
   - Audio capture, VAD, ASR, NLP post-processing, and initial WebSocket transmission are untouched.
   - Initial subtitles appear immediately (~0.71s end-to-end latency).
2. **Stable Segment Identity (`segment_001`, `segment_002`, ...)**:
   - Every `final` transcript event is assigned a monotonically increasing ID (`segment_001`, `segment_002`, etc.).
   - Overlay tracks `currentlyDisplayedSegmentId`.
   - Async corrections verify `correction.segment_id === currentlyDisplayedSegmentId` before replacing visible text. Stale corrections for past segments are discarded safely.
3. **Bounded Correction Queue**:
   - `MAX_CORRECTION_QUEUE = 3`.
   - Drop-oldest strategy prevents ASR backpressure or memory accumulation if LLM inference is slower than speech.
4. **Local llama-cpp-python Engine**:
   - Loaded ONCE on worker initialization.
   - Configurable GGUF model path via `LLM_MODEL_PATH` or `--llm-model` argument.
   - Evaluates small quantized GGUF candidate (e.g., `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`).
5. **Feature Flag (`ASYNC_LLM_CORRECTION_ENABLED`)**:
   - Defaults to `false`. When false, system operates identically to Module 8 H-Vocab baseline.
6. **Fail-Safe Operation**:
   - If `llama-cpp-python` is not installed or model loading fails, correction layer logs error and disables itself while core subtitle system continues seamlessly.

---

## 3. User Review Required

> [!IMPORTANT]
> **LLM Engine & GGUF Model**:
> The correction layer uses `llama-cpp-python` with a CPU-only quantized GGUF model (e.g. `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` ~350 MB or `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` ~1.1 GB).
> Feature flag `ASYNC_LLM_CORRECTION_ENABLED` defaults to `false`. When disabled or model file missing, the system functions normally as H-Vocab.

---

## 4. Proposed Changes

### Backend Component

#### [NEW] [llm_corrector.py](file:///c:/Users/Sonu/OneDrive/Desktop/New%20folder/backend/asr/llm_corrector.py)
- Implements `LLMCorrector` using `llama-cpp-python`.
- Handles lazy/one-time loading of GGUF model from `LLM_MODEL_PATH`.
- Builds deterministic correction prompt:
  ```
  You are correcting an automatic speech recognition subtitle.
  Fix only:
  - obvious spelling errors
  - obvious ASR word errors
  - grammar errors
  - punctuation
  Preserve the exact meaning. Do not add information. Do not summarize. Do not explain. Return ONLY the corrected subtitle.
  Subtitle: {text}
  ```
- Implements output validator:
  - Rejects empty strings.
  - Rejects text > 2.0× original length.
  - Rejects output containing explanations, markdown, or intros ("Here is...", "Corrected:").
  - Rejects text introducing unmentioned facts.

#### [MODIFY] [pipeline_server.py](file:///c:/Users/Sonu/OneDrive/Desktop/New%20folder/backend/pipeline_server.py)
- Adds constants: `MAX_CORRECTION_QUEUE = 3`, `ASYNC_LLM_CORRECTION_ENABLED` (via `--enable-llm` flag / env var).
- Adds `segment_counter` to `PipelineSession` to assign monotonically increasing IDs (`segment_001`, `segment_002`, ...).
- In `_asr_worker`:
  - Attaches `segment_id` to `final` events.
  - Non-blockingly enqueues `final` payload copy into `correction_queue` (dropping oldest if full).
- Implements `_llm_worker`:
  - Drains `correction_queue`.
  - Runs `llm_corrector.correct()`.
  - Measures `llm_inference_ms` and `correction_turnaround_ms`.
  - Sends `{"type": "correction", "segment_id": id, "text": corrected_text, ...}` to WebSocket via `transcript_queue`.

### Chrome Extension Component

#### [MODIFY] [offscreen.js](file:///c:/Users/Sonu/OneDrive/Desktop/New%20folder/live-subtitle-extension/offscreen/offscreen.js)
- Updates WebSocket listener to process `data.type === 'correction'` alongside `partial` and `final`.

#### [MODIFY] [overlay.js](file:///c:/Users/Sonu/OneDrive/Desktop/New%20folder/live-subtitle-extension/content/overlay.js)
- Tracks `currentlyDisplayedSegmentId`.
- On `final`/`partial` event, updates `currentlyDisplayedSegmentId = transcript.segment_id`.
- On `correction` event, checks `if (transcript.segment_id === currentlyDisplayedSegmentId)`:
  - If match: updates text element content smoothly and resets auto-hide timer.
  - If mismatch: logs debug and discards stale correction.

### Documentation & Benchmarking

#### [NEW] [async_llm_correction.md](file:///c:/Users/Sonu/OneDrive/Desktop/New%20folder/docs/async_llm_correction.md)
- Documents Module 9 architecture, GGUF model details, prompt design, validation rules, queue safety, and failure modes.

#### [MODIFY] [benchmark_report.md](file:///c:/Users/Sonu/OneDrive/Desktop/New%20folder/benchmark_report.md)
- Adds Module 9 benchmark comparisons (H-Vocab vs H-Vocab + Async LLM).

---

## 5. Verification & Test Plan

### Automated & Unit Tests
1. **Segment Identity Test**: Verify sequential `segment_001`, `segment_002` IDs generated on `final` events.
2. **Queue Safety Test**: Simulate slow LLM execution; verify `correction_queue` drops oldest items without stalling ASR or accumulating VAD/audio buffers.
3. **Output Validation Test**: Test 5+ sample sentences (e.g., "Open vino optimizes inference.", "today we're learning javascript") and invalid inputs (explanations, added info) to ensure correct filtering.
4. **Failure Handling Test**: Simulate missing model file, worker exception, or queue overflow; verify core ASR path continues unaffected.
5. **Stale Protection Test**: Simulate out-of-order correction delivery; verify older segment correction is ignored when newer segment is displayed.

---

## 6. Execution Plan

1. Create `backend/asr/llm_corrector.py` with `llama-cpp-python` engine and validation logic.
2. Update `backend/pipeline_server.py` with segment IDs, queue management, and `_llm_worker` thread.
3. Update `live-subtitle-extension/offscreen/offscreen.js` and `live-subtitle-extension/content/overlay.js`.
4. Create test suite in `backend/test_llm_correction.py` to verify prompt validation, queue safety, and failure modes.
5. Benchmark and document results in `docs/async_llm_correction.md` and `benchmark_report.md`.
