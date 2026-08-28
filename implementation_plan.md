# Module 9 Step 3 Implementation Plan — LLM Post-Correction in main_app.py

This plan details the implementation for **Module 9 Step 3: LLM Post-Correction integrated directly in main_app.py**.

---

## 1. Overview

Step 3 adds local LLM post-correction to the main desktop application (`main_app.py`). The correction logic is embedded inline rather than running in a separate background thread with queue infrastructure. This simplifies the architecture while retaining the core validation benefits.

**Key design choices:**
- `LLMCorrector` class (with `correct()` and `validate_output()`) is ported from the previous `backend/asr/llm_corrector.py` — deleted as part of the old architecture cleanup.
- Correction is triggered on-demand (e.g., user-initiated, timer-based, or after initial caption stabilization), not as a persistent daemon thread.
- The initial subtitle pipeline (ASR → NLP → WebSocket) remains **completely untouched** — no background threads, no bounded queues, no backpressure concerns.
- Feature flag `ASYNC_LLM_CORRECTION_ENABLED` defaults to `false`; when enabled, correction runs after the initial caption is displayed, replacing text smoothly without delaying startup.

---

## 2. Technical Guarantees

1. **Initial subtitle pipeline untouched** — ASR → NLP → WebSocket latency unchanged (~0.71s end-to-end). No background thread introduces jitter or blocking.
2. **`OpenVINO`, `distil-whisper`, `RTF`, `ASR`, `VAD`, `CPU`, `GPU` all staying untouched** — Step 3 adds correction as a post-processing overlay, not a pipeline modifier.
3. **Feature flag gated** — `ASYNC_LLM_CORRECTION_ENABLED` defaults to `false`. When disabled, the system operates identically to the Module 8 H-Vocab baseline.
4. **Fail-safe operation** — If the GGUF model fails to load or `llama-cpp-python` is unavailable, correction silently skips; core subtitle system continues seamlessly.
5. **Technical-term protection preserved** — `validate_output()` rejection rules (empty, conversational prefixes, markdown, excessive length) remain in effect. The `proper_noun_map` capitalization from the Silero NLP engine continues to guard OpenVINO, faster-whisper, CTranslate2, etc.

---

## 3. User Stories

- **As a user**, I want subtitles to appear immediately without waiting for LLM correction.
- **As a user**, I want obviously wrong ASR words (e.g., "openvino" → "OpenVINO") to be corrected after the initial display.
- **As a user**, I want the correction to be safe — no hallucinated content, no markdown, no conversational intros.
- **As a user**, when correction is disabled or the model is missing, nothing breaks — subtitles work as before.

---

## 4. Proposed Changes

### main_app.py

#### Port `LLMCorrector` class

Include the `LLMCorrector` class with two methods:

- `correct(text, language="en")` — runs LLM inference, validates output via `validate_output()`, returns `{"corrected_text": str, "inference_ms": float}` or `None`.
- `validate_output(original, generated)` — 5 rejection rules:
  - Empty output → `None`
  - Conversational prefixes ("Here is", "Sure", "Note:") → `None`
  - Markdown (` ``` `, backticks, `##`) → `None`
  - Excessive length (>2x original + 20 chars) → `None`
  - Identical to original → `None` (returns `None`, no update needed)

- Add `ASYNC_LLM_CORRECTION_ENABLED = False` feature flag.
- Add `LLM_MODEL_PATH` environment variable support (optional; if not set or model missing, correction silently disables).

#### Correction trigger flow

1. Initial ASR caption is captured and sent to Chrome **immediately** (no blocking).
2. After a short stabilization delay (e.g., 1–2s), or on user-trigger (menu command keyboard shortcut), call `llm_corrector.corrected_text(text)`.
3. If a corrected text is returned, send a correction event to Chrome overlay to replace the initial caption.
4. If `None` is returned (model unavailable, validation rejection, or no change needed), no event is sent; the initial caption remains.

#### Example code structure in main_app.py

```python
class main_app:
    def __init__(self):
        self.llm_corrector = LLMCorrector(enabled=ASYNC_LLM_CORRECTION_ENABLED,
                                          model_path=os.environ.get("LLM_MODEL_PATH", ""))
        self.llm_enabled = ASYNC_LLM_CORRECTION_ENABLED

    def on_caption_ready(self, raw_text):
        """Called when ASR produces a new final caption."""
        # Immediately dispatch the raw caption (unchanged pipeline)
        self.send_to_chrome(raw_text)

        # Schedule correction after stabilization (non-blocking, no delay to initial display)
        threading.Timer(2.0, self._correct_and_replace, [raw_text]).start()

    def _correct_and_replace(self, raw_text):
        if not self.llm_enabled:
            return
        res = self.llm_corrector.correct(raw_text)
        if res and res.get("corrected_text"):
            self.send_to_chrome(res["corrected_text"], is_correction=True)
        # If None, initial caption stays — no error, no jitter
```

---

## 5. Verification & Test Plan

### Automated tests (to be added to main_app.py test suite)

1. **Initial caption latency** — Verify raw caption appears at ~0.71s end-to-end with no correction thread blocking.
2. **Correction on-demand** — After the stabilization timer fires, verify `correct()` returns validated text for test sentences (e.g., "openvino optimizes inference" → "OpenVINO optimizes inference.").
3. **Validation rejection** — Verify `validate_output()` rejects: conversational intros, markdown, excessive length, empty output.
4. **Feature flag disabled** — With `ASYNC_LLM_CORRECTION_ENABLED=false`, confirm no correction timer is scheduled and the pipeline runs identically to Module 8.
5. **Model missing** — With no `LLM_MODEL_PATH` set, confirm `correct()` returns `None` and no errors propagate to the ASR pipeline.

### Manual validation

- Run the OpenVINO technical-term protection test (from the earlier single-consistent test) to verify `OpenVINO` stays capitalized through the new correction flow.
- Test with "openvino" in middle of sentence (proper_noun_map capitalization), at sentence start (capitalization edge case), and with conversational prefixes (LLM validation rejection).

---

## 6. Execution Plan

1. **Port `LLMCorrector`** — Copy `correct()` and `validate_output()` from the deleted `backend/asr/llm_corrector.py` into `main_app.py`. Remove any dependency on `backend/asr/` packages. Add feature flag and model-path config.
2. **Add correction trigger** — Implement the stabilization-timer + on-demand correction flow in `main_app.py` as described above. Do NOT add a background daemon thread or `correction_queue`.
3. **Update main_app.py entry point** — Ensure `ASYNC_LLM_CORRECTION_ENABLED` env var / flag is respected at startup.
4. **Run verification tests** — Execute the test plan above. Confirm OpenVINO protection, RTF/ASR/VAD/CPU/GPU untouched, feature flag gating.
5. **Benchmark** — Measure that initial caption latency is unchanged with the feature on vs off.
6. **Delete old artifacts** — `backend/pipeline_server.py`, `backend/asr/llm_corrector.py`, `backend/test_llm_correction.py`, `docs/async_llm_correction.md` are already removed (see git history). No background LLM worker architecture in this Step.

---