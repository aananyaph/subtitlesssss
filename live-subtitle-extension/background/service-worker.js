/**
 * Background Service Worker (Manifest V3)
 * Manages active tab verification, chrome.tabCapture stream IDs,
 * offscreen document lifecycle, and tab state change listeners.
 */

let currentCaptureState = {
  isCapturing: false,
  tabId: null,
  tabTitle: '',
  lastError: null
};

const OFFSCREEN_DOCUMENT_PATH = 'offscreen/offscreen.html';

console.log('[Service Worker] Extension service worker started.');

// Handle extension message routing
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.type) {
    case 'START_CAPTURE':
      handleStartCapture()
        .then((res) => sendResponse({ success: true, ...res }))
        .catch((err) => {
          currentCaptureState.lastError = err.message;
          sendResponse({ success: false, error: err.message });
        });
      return true; // Async response

    case 'STOP_CAPTURE':
      handleStopCapture()
        .then(() => sendResponse({ success: true }))
        .catch((err) => sendResponse({ success: false, error: err.message }));
      return true;

    case 'GET_STATUS':
      sendResponse({ success: true, state: currentCaptureState });
      break;

    case 'CAPTURE_ENDED_UNEXPECTEDLY':
      currentCaptureState.isCapturing = false;
      currentCaptureState.tabId = null;
      currentCaptureState.tabTitle = '';
      currentCaptureState.lastError = 'Audio stream ended unexpectedly (tab was closed or reloaded).';
      break;

    default:
      break;
  }
});

/**
 * Validates target tab, generates tab capture stream ID, ensures offscreen document exists,
 * and passes stream ID to offscreen script.
 */
async function handleStartCapture() {
  // 1. Query active tab in current window
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tabs || tabs.length === 0) {
    throw new Error('No active tab found in the current window.');
  }

  const activeTab = tabs[0];

  // 2. Validate tab URL to prevent chrome:// privileges errors
  if (!activeTab.url || isRestrictedUrl(activeTab.url)) {
    throw new Error(`Cannot capture audio from restricted tab "${activeTab.url || 'Internal page'}". Please switch to a video site like YouTube or Netflix.`);
  }

  console.log(`[Service Worker] Requesting stream ID for tab ID ${activeTab.id}: ${activeTab.title}`);

  // 3. Obtain MediaStream ID from chrome.tabCapture
  const streamId = await new Promise((resolve, reject) => {
    chrome.tabCapture.getMediaStreamId({ targetTabId: activeTab.id }, (id) => {
      if (chrome.runtime.lastError) {
        return reject(new Error(chrome.runtime.lastError.message));
      }
      if (!id) {
        return reject(new Error('Failed to obtain media stream ID for the active tab.'));
      }
      resolve(id);
    });
  });

  // 4. Ensure Offscreen Document exists
  await ensureOffscreenDocument();

  // 5. Send stream ID to Offscreen document to initialize AudioContext & getUserMedia
  const response = await chrome.runtime.sendMessage({
    type: 'START_OFFSCREEN_CAPTURE',
    streamId,
    tabId: activeTab.id,
    tabTitle: activeTab.title
  });

  if (!response || !response.success) {
    throw new Error(response?.error || 'Offscreen audio pipeline failed to start capture.');
  }

  currentCaptureState.isCapturing = true;
  currentCaptureState.tabId = activeTab.id;
  currentCaptureState.tabTitle = activeTab.title;
  currentCaptureState.lastError = null;

  return {
    tabId: activeTab.id,
    tabTitle: activeTab.title
  };
}

/**
 * Sends stop command to offscreen document and cleans up.
 */
async function handleStopCapture() {
  if (await hasOffscreenDocument()) {
    try {
      await chrome.runtime.sendMessage({ type: 'STOP_OFFSCREEN_CAPTURE' });
    } catch (e) {
      console.warn('[Service Worker] Offscreen document already closed or unresponsive:', e);
    }
    await closeOffscreenDocument();
  }

  currentCaptureState.isCapturing = false;
  currentCaptureState.tabId = null;
  currentCaptureState.tabTitle = '';
  currentCaptureState.lastError = null;
}

/**
 * Checks if target URL is restricted by Chrome security policies.
 */
function isRestrictedUrl(url) {
  return (
    url.startsWith('chrome://') ||
    url.startsWith('chrome-extension://') ||
    url.startsWith('edge://') ||
    url.startsWith('about:') ||
    url.startsWith('chrome-search://') ||
    url.startsWith('https://chrome.google.com/webstore')
  );
}

/**
 * Ensures an offscreen document is open.
 */
async function ensureOffscreenDocument() {
  if (await hasOffscreenDocument()) {
    return;
  }

  console.log('[Service Worker] Creating offscreen document...');
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_DOCUMENT_PATH,
    reasons: ['USER_MEDIA'],
    justification: 'Capture active browser tab audio for local real-time subtitle generation'
  });
}

/**
 * Closes the offscreen document if currently open.
 */
async function closeOffscreenDocument() {
  if (await hasOffscreenDocument()) {
    console.log('[Service Worker] Closing offscreen document...');
    await chrome.offscreen.closeDocument();
  }
}

/**
 * Checks whether an offscreen document currently exists.
 */
async function hasOffscreenDocument() {
  if ('hasDocument' in chrome.offscreen) {
    return await chrome.offscreen.hasDocument();
  }
  // Fallback for older Chrome versions
  const matchedClients = await clients.matchAll();
  return matchedClients.some((c) => c.url.includes(OFFSCREEN_DOCUMENT_PATH));
}

// Cleanup if captured tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  if (currentCaptureState.isCapturing && currentCaptureState.tabId === tabId) {
    console.log(`[Service Worker] Captured tab ${tabId} was closed. Stopping audio capture.`);
    handleStopCapture();
  }
});
