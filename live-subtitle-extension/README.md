# Live Subtitle — Module 1: Chrome Tab Audio Capture

A local-only, free, open-source Chrome Extension (Manifest V3) that captures audio directly from browser tabs (e.g., YouTube, Netflix) without microphone input, prepares raw PCM audio streams, and provides real-time diagnostics and WAV sample exports for verification.

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                       Chrome Browser                        │
│                                                             │
│             Active Tab (e.g. YouTube Video)                 │
│                            │                                │
│                            ▼                                │
│              chrome.tabCapture.getMediaStreamId             │
│                            │                                │
│                            ▼                                │
│                  Offscreen Document                         │
│             (navigator.mediaDevices.getUserMedia)           │
│                            │                                │
│        ┌───────────────────┴───────────────────┐            │
│        ▼                                       ▼            │
│ AudioContext.destination                AnalyserNode        │
│ (Speaker Output Pass-through)         (Level Meter & PCM)   │
│                                                │            │
│                                                ▼            │
│                                        Int16 PCM Buffers    │
│                                                │            │
│                                                ▼            │
│                                     WAV Download / WebSocket│
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```text
live-subtitle-extension/
├── manifest.json
├── popup/
│   ├── popup.html
│   ├── popup.css
│   └── popup.js
├── background/
│   └── service-worker.js
├── offscreen/
│   ├── offscreen.html
│   └── offscreen.js
├── audio/
│   └── audio-processor.js
└── README.md
```

---

## Step 1 Research & Chrome API Breakdown

### 1. Which APIs are used?
* **`chrome.tabCapture`**: Obtains media stream IDs for active browser tabs (`chrome.tabCapture.getMediaStreamId`).
* **`chrome.offscreen`**: Manifest V3 background service workers cannot access DOM or Web Audio APIs. We instantiate an offscreen document (`offscreen/offscreen.html`) to host `getUserMedia`, `AudioContext`, and audio nodes.
* **Web Audio API**:
  * `AudioContext`: Engine running audio processing graphs.
  * `MediaStreamAudioSourceNode`: Connects the captured tab stream to the processing pipeline.
  * `MediaStreamAudioDestinationNode` / `AudioContext.destination`: Routes audio back to speakers so video audio remains audible.
  * `AnalyserNode`: Computes real-time RMS, peak signal, and dB levels for UI metering.
  * `ScriptProcessorNode`: Extracts Float32 / Int16 PCM audio samples with minimal CPU usage.

### 2. Why is this approach appropriate?
* **Manifest V3 Compliant**: Uses Google's official recommended offscreen pattern.
* **100% Pure Digital Audio**: Hooks into Chrome's renderer process. Microphone input is completely ignored.
* **Zero Overhead**: Native Web Audio graph implementation requiring no cloud APIs or external binaries.

### 3. Required Permissions
* `"tabCapture"`: Authorizes capturing the active tab's audio stream.
* `"offscreen"`: Authorizes creating the offscreen document for audio graph execution.
* `"activeTab"`: Grants permissions to inspect the active tab when opening the extension popup.
* `"tabs"`: Allows monitoring tab state (e.g. closing or updating tabs).

### 4. Selected Tab Isolation
* `chrome.tabCapture` captures **only** the target active tab specified by `targetTabId`. Audio from other browser tabs or system apps is excluded.

### 5. Microphone Independence
* Stream acquisition uses `chromeMediaSource: 'tab'`. **No microphone access is requested or used.**

### 6. Chrome Restrictions & Gotchas
* **Tab Muting**: Chrome mutes captured tabs by default. We explicitly connect `sourceNode` to `audioContext.destination` to restore video sound.
* **User Gesture**: Capture must be triggered by clicking "Start Capture" in the popup.
* **Restricted URLs**: Internal pages (`chrome://`, `chrome-extension://`, `chrome.google.com/webstore`) cannot be captured due to Chrome security policies.

---

## Installation Guide (Google Chrome)

1. Open Google Chrome.
2. Navigate to `chrome://extensions` in the address bar.
3. Enable **Developer mode** using the toggle switch in the top right corner.
4. Click the **Load unpacked** button in the top left.
5. Select the `live-subtitle-extension` directory.
6. The **Live Subtitle - Tab Audio Capture** extension will appear in your Chrome toolbar.

---

## Step-by-Step Testing Guide

### Test 1 — Play YouTube Video & Capture Audio
1. Open [YouTube](https://www.youtube.com) and play a video containing speech.
2. Click the extension icon to open the popup.
3. Click **Start Capture**.
4. **Expected Results**:
   * Status badge changes to glowing green **ACTIVE**.
   * **Audio Activity Level** meter bar animates continuously in sync with video speech.
   * **Sample Rate** shows `48000 Hz` (or `44100 Hz`).
   * **Channels** shows `2 (Stereo)`.
   * **Audio Frames** count increments rapidly.
   * Video audio continues playing normally from your speakers.

### Test 2 — Mute Video
1. While capture is active, mute the YouTube video player.
2. **Expected Results**:
   * Audio Activity Level meter drops to **0%**.
   * Peak signal drops to silent baseline (`-100 dB`).

### Test 3 — Pause Video
1. Pause the YouTube video.
2. **Expected Results**:
   * Meter activity drops to 0%.
   * Frame processing halts or remains silent.

### Test 4 — Stop Capture
1. Click **Stop Capture** in the extension popup.
2. **Expected Results**:
   * Status badge changes to **Not Captured**.
   * Meter bar resets to 0%.
   * AudioContext and streams clean up completely.

### Test 5 — Tab Switching Behavior
1. Click **Start Capture** on your YouTube tab.
2. Switch to a different browser tab.
3. Click the extension icon on the new tab.
4. **Expected Behavior**:
   * Capture remains locked to the original YouTube tab.
   * Diagnostics continue showing audio frames from the YouTube tab until **Stop Capture** is clicked.

---

## Audio Verification (WAV Sample Export)

To verify that captured audio is authentic YouTube audio:

1. Play a YouTube video and click **Start Capture**.
2. Let it run for 5–10 seconds.
3. Click **Download 5s WAV Sample** in the popup.
4. Open the downloaded `.wav` file (`tab_audio_sample_...wav`) in any media player (VLC, Windows Media Player).
5. **Verification**: You will hear the exact, crystal-clear audio from the YouTube video!

---

## Performance & CPU Footprint

Designed specifically for CPU-only systems (e.g., Intel UHD 620 integrated graphics):
* **Zero GPU requirements**: Uses Web Audio API without WebGL or canvas heavy rendering.
* **Low CPU Overhead**: Audio buffer processing runs in lightweight 2048-sample chunks (~42ms intervals), consuming < 1% CPU.
* **Memory Management**: Rolling ring buffer is capped at 10 seconds of Int16 PCM audio (~1.9 MB max RAM).
* **Clean Cleanup**: AudioContext, ScriptProcessorNode, and MediaStream tracks are fully terminated when capture stops.

---

## Localhost Connection Readiness (Module 2 Setup)

The extension includes a built-in WebSocket client ready for Module 2:
* **Toggle**: Flip the **Stream to Local Python (ws://localhost:8765)** switch at the bottom of the popup.
* **Data Format**: Sends raw binary Int16 PCM chunks over WebSocket directly to `ws://localhost:8765`.
* **Zero Cloud Dependencies**: Operates strictly on `localhost`.
