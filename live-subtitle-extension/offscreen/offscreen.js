import {
  calculateAudioLevels,
  float32ToInt16,
  encodeWavBlob,
  resampleBuffer
} from '../audio/audio-processor.js';

/**
 * Offscreen Document Audio Engine
 * Handles stream acquisition, pass-through audio routing, level analysis,
 * ring buffering for WAV exports, and WebSocket readiness for Module 2.
 */

let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let analyserNode = null;
let processorNode = null;
let websocketClient = null;

// Ring buffer storing up to 10 seconds of captured PCM audio (Int16)
let pcmBufferChunks = [];
let totalPcmSamples = 0;
const MAX_RING_BUFFER_SECONDS = 10;

let isCapturing = false;
let activeTabId = null;
let activeTabTitle = '';
let totalFramesProcessed = 0;
let lastDiagnosticsUpdate = 0;

// Localhost WebSocket settings for Module 2 readiness
let wsUrl = 'ws://localhost:8765';
let wsEnabled = true;

console.log('[Offscreen Engine] Offscreen document initialized.');

// Listen for commands from service worker or popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case 'START_OFFSCREEN_CAPTURE':
      handleStartCapture(message.streamId, message.tabId, message.tabTitle)
        .then(() => sendResponse({ success: true }))
        .catch((err) => sendResponse({ success: false, error: err.message }));
      return true; // Async response

    case 'STOP_OFFSCREEN_CAPTURE':
      handleStopCapture();
      sendResponse({ success: true });
      break;

    case 'EXPORT_WAV_SAMPLE':
      exportWavDataUrl()
        .then((dataUrl) => sendResponse({ success: true, dataUrl }))
        .catch((err) => sendResponse({ success: false, error: err.message }));
      return true; // Async response

    case 'SET_WEBSOCKET_CONFIG':
      wsEnabled = message.enabled;
      if (message.url) wsUrl = message.url;
      if (wsEnabled) {
        initWebSocket();
      } else if (websocketClient) {
        websocketClient.close();
        websocketClient = null;
      }
      sendResponse({ success: true, wsEnabled, wsUrl });
      break;

    case 'GET_OFFSCREEN_STATUS':
      sendResponse({
        isCapturing,
        activeTabId,
        activeTabTitle,
        sampleRate: audioContext ? audioContext.sampleRate : 0,
        totalFramesProcessed,
        wsConnected: websocketClient && websocketClient.readyState === WebSocket.OPEN
      });
      break;

    case 'SEND_CONTROL_MESSAGE':
      if (websocketClient && websocketClient.readyState === WebSocket.OPEN && message.control) {
        websocketClient.send(JSON.stringify(message.control));
      }
      sendResponse({ success: true });
      break;


    default:
      break;
  }
});

/**
 * Starts audio capture using the provided tab MediaStream ID.
 */
async function handleStartCapture(streamId, tabId, tabTitle) {
  if (isCapturing) {
    handleStopCapture();
  }

  console.log(`[Offscreen Engine] Initiating tab capture for Tab ID ${tabId} (${tabTitle})`);
  activeTabId = tabId;
  activeTabTitle = tabTitle || 'Active Tab';

  try {
    // 1. Obtain MediaStream from tab using chromeMediaSourceId
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId
        }
      },
      video: false
    });

    if (!mediaStream || mediaStream.getAudioTracks().length === 0) {
      throw new Error('Failed to acquire audio track from tab media stream.');
    }

    const audioTrack = mediaStream.getAudioTracks()[0];
    console.log(`[Offscreen Engine] Tab audio track acquired: ${audioTrack.label}`);

    // Track ending listener (e.g., tab closed or user stopped sharing)
    audioTrack.onended = () => {
      console.warn('[Offscreen Engine] Audio track ended unexpectedly.');
      handleStopCapture();
      chrome.runtime.sendMessage({ type: 'CAPTURE_ENDED_UNEXPECTEDLY', tabId: activeTabId });
    };

    // 2. Initialize AudioContext & Nodes
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }
    sourceNode = audioContext.createMediaStreamSource(mediaStream);

    // CRITICAL: Route captured audio to speakers so the user can still hear the video!
    sourceNode.connect(audioContext.destination);

    // 3. Create AnalyserNode for real-time visual metering
    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 512;
    sourceNode.connect(analyserNode);

    // 4. Create ScriptProcessorNode for PCM buffer extraction (Buffer size 2048 samples = ~42ms chunk)
    const bufferSize = 2048;
    processorNode = audioContext.createScriptProcessor(bufferSize, 2, 2);
    sourceNode.connect(processorNode);
    // Connect processor to destination to drive audio processing loop in browser
    processorNode.connect(audioContext.destination);

    // Reset ring buffer and stats
    pcmBufferChunks = [];
    totalPcmSamples = 0;
    totalFramesProcessed = 0;
    isCapturing = true;

    // 5. Audio Process Loop Callback
    processorNode.onaudioprocess = (audioProcessingEvent) => {
      if (!isCapturing) return;

      if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume();
      }

      const inputBuffer = audioProcessingEvent.inputBuffer;
      const numChannels = inputBuffer.numberOfChannels;
      const length = inputBuffer.length;
      const sampleRate = inputBuffer.sampleRate;

      // Extract left and right channels
      const leftChannel = inputBuffer.getChannelData(0);
      const rightChannel = numChannels > 1 ? inputBuffer.getChannelData(1) : leftChannel;

      // Create stereo-interleaved Float32 array for audio processing & level metering
      const interleavedFloat32 = new Float32Array(length * numChannels);
      for (let i = 0; i < length; i++) {
        interleavedFloat32[i * numChannels] = leftChannel[i];
        if (numChannels > 1) {
          interleavedFloat32[i * numChannels + 1] = rightChannel[i];
        }
      }

      // Calculate audio levels (RMS, Peak, dB, Percent)
      const levels = calculateAudioLevels(interleavedFloat32);

      // Convert Float32 to Int16 PCM
      const int16PCM = float32ToInt16(interleavedFloat32);
      
      // Store in ring buffer for WAV sample download verification
      appendPcmToRingBuffer(int16PCM, sampleRate, numChannels);

      totalFramesProcessed += length;

      // Optional: Send live PCM chunk via WebSocket if connected (Module 2 preparation)
      if (websocketClient && websocketClient.readyState === WebSocket.OPEN) {
        // Send raw binary Int16 PCM
        websocketClient.send(int16PCM.buffer);
      }

      // Throttle runtime diagnostics updates to ~10 updates per second (every 100ms)
      const now = Date.now();
      if (now - lastDiagnosticsUpdate > 100) {
        lastDiagnosticsUpdate = now;
        chrome.runtime.sendMessage({
          type: 'AUDIO_DIAGNOSTICS',
          diagnostics: {
            isCapturing: true,
            sampleRate,
            numChannels,
            totalFramesProcessed,
            peakLevel: levels.peak,
            rmsLevel: levels.rms,
            db: levels.db,
            percent: levels.percent,
            tabTitle: activeTabTitle,
            wsConnected: websocketClient && websocketClient.readyState === WebSocket.OPEN
          }
        }).catch(() => {
          // Extension popup might be closed; ignore runtime channel errors silently
        });
      }
    };

    if (wsEnabled) {
      initWebSocket();
    }

    console.log('[Offscreen Engine] Tab audio capture running smoothly.');
  } catch (err) {
    console.error('[Offscreen Engine] Error starting tab audio capture:', err);
    handleStopCapture();
    throw err;
  }
}

/**
 * Appends Int16 PCM data to rolling ring buffer, maintaining up to 10 seconds of max audio.
 */
function appendPcmToRingBuffer(int16Chunk, sampleRate, numChannels) {
  const maxSamples = MAX_RING_BUFFER_SECONDS * sampleRate * numChannels;
  pcmBufferChunks.push(int16Chunk);
  totalPcmSamples += int16Chunk.length;

  while (totalPcmSamples > maxSamples && pcmBufferChunks.length > 0) {
    const shiftChunk = pcmBufferChunks.shift();
    totalPcmSamples -= shiftChunk.length;
  }
}

/**
 * Encodes the stored ring buffer into a downloadable WAV Data URL.
 */
async function exportWavDataUrl() {
  if (pcmBufferChunks.length === 0) {
    throw new Error('No audio frames captured yet. Please play audio and retry.');
  }

  const sampleRate = audioContext ? audioContext.sampleRate : 48000;
  const numChannels = 2;

  // Flatten ring buffer chunks into single Int16Array
  const combinedBuffer = new Int16Array(totalPcmSamples);
  let offset = 0;
  for (const chunk of pcmBufferChunks) {
    combinedBuffer.set(chunk, offset);
    offset += chunk.length;
  }

  // Encode to WAV Blob
  const wavBlob = encodeWavBlob(combinedBuffer, sampleRate, numChannels);

  // Convert Blob to Data URL
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Failed to create WAV Data URL'));
    reader.readAsDataURL(wavBlob);
  });
}

/**
 * Initializes WebSocket connection to localhost Python server (Module 2 readiness).
 */
function initWebSocket() {
  if (websocketClient && (websocketClient.readyState === WebSocket.OPEN || websocketClient.readyState === WebSocket.CONNECTING)) {
    return;
  }

  if (websocketClient) {
    try { websocketClient.close(); } catch (e) {}
    websocketClient = null;
  }

  try {
    console.log(`[Offscreen Engine] Connecting WebSocket to ${wsUrl}...`);
    websocketClient = new WebSocket(wsUrl);

    websocketClient.onopen = () => {
      console.log('[Offscreen Engine] WebSocket connected successfully.');
      // Send initial metadata handshake
      const handshake = {
        type: 'HANDSHAKE',
        sampleRate: audioContext ? audioContext.sampleRate : 48000,
        numChannels: 2,
        client: 'ChromeExtension_Module1'
      };
      websocketClient.send(JSON.stringify(handshake));
    };

    websocketClient.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'partial' || data.type === 'final' || data.type === 'correction' || data.type === 'flush_ack') {
          // Forward transcript or flush ack to service worker to route to active tab
          chrome.runtime.sendMessage({
            type: 'TRANSCRIPT_EVENT',
            transcript: data,
            tabId: activeTabId
          }).catch(() => {}); // Ignore error if service worker isn't listening yet
        }
      } catch (err) {
        console.warn('[Offscreen Engine] Failed to parse WebSocket message:', err);
      }
    };


    websocketClient.onerror = (err) => {
      console.warn('[Offscreen Engine] WebSocket connection error (check if backend pipeline_server.py is running).');
    };

    websocketClient.onclose = () => {
      console.log('[Offscreen Engine] WebSocket connection closed.');
    };
  } catch (e) {
    console.warn('[Offscreen Engine] Failed to initialize WebSocket:', e);
  }
}

/**
 * Stops capture, cleans up tracks, and closes AudioContext.
 */
function handleStopCapture() {
  console.log('[Offscreen Engine] Stopping tab audio capture...');
  isCapturing = false;

  if (processorNode) {
    processorNode.onaudioprocess = null;
    try { processorNode.disconnect(); } catch (e) {}
    processorNode = null;
  }

  if (analyserNode) {
    try { analyserNode.disconnect(); } catch (e) {}
    analyserNode = null;
  }

  if (sourceNode) {
    try { sourceNode.disconnect(); } catch (e) {}
    sourceNode = null;
  }

  if (audioContext && audioContext.state !== 'closed') {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  if (websocketClient) {
    try { websocketClient.close(); } catch (e) {}
    websocketClient = null;
  }

  activeTabId = null;
  activeTabTitle = '';

  // Notify runtime that capture stopped
  chrome.runtime.sendMessage({
    type: 'AUDIO_DIAGNOSTICS',
    diagnostics: {
      isCapturing: false,
      sampleRate: 0,
      numChannels: 0,
      totalFramesProcessed: 0,
      peakLevel: 0,
      rmsLevel: 0,
      db: -100,
      percent: 0,
      tabTitle: '',
      wsConnected: false
    }
  }).catch(() => {});

  console.log('[Offscreen Engine] Audio capture cleanup complete.');
}
