/**
 * Module 7 — Application Features and User Controls (Popup Controller)
 * Manages Start/Stop caption buttons, status badges, persistent settings storage
 * (chrome.storage.sync), and real-time settings relay to content script and Python backend.
 */

const DEFAULT_SETTINGS = {
  fontSize: 'medium',
  position: 'bottom',
  language: 'auto',
  model: 'tiny'
};

document.addEventListener('DOMContentLoaded', async () => {
  const statusBadge = document.getElementById('statusBadge');
  const statusText = document.getElementById('statusText');
  const tabTitleText = document.getElementById('tabTitleText');
  
  const errorBanner = document.getElementById('errorBanner');
  const errorMessage = document.getElementById('errorMessage');

  const btnStart = document.getElementById('btnStart');
  const btnStop = document.getElementById('btnStop');
  const btnExportWav = document.getElementById('btnExportWav');
  const toggleWs = document.getElementById('toggleWs');

  const selectSize = document.getElementById('selectSize');
  const selectPosition = document.getElementById('selectPosition');
  const selectLanguage = document.getElementById('selectLanguage');
  const selectModel = document.getElementById('selectModel');

  const meterFill = document.getElementById('meterFill');
  const meterPercent = document.getElementById('meterPercent');

  let currentSettings = { ...DEFAULT_SETTINGS };

  // 1. Query active tab title
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs && tabs.length > 0) {
      tabTitleText.textContent = tabs[0].title || 'Active Tab';
    }
  } catch (e) {
    tabTitleText.textContent = 'Active Tab';
  }

  // 2. Load persistent settings from chrome.storage.sync
  if (chrome.storage && chrome.storage.sync) {
    chrome.storage.sync.get(DEFAULT_SETTINGS, (items) => {
      if (items) {
        currentSettings = { ...DEFAULT_SETTINGS, ...items };
        applySettingsToUi(currentSettings);
      }
    });
  }

  function applySettingsToUi(settings) {
    if (settings.fontSize) selectSize.value = settings.fontSize;
    if (settings.position) selectPosition.value = settings.position;
    if (settings.language) selectLanguage.value = settings.language;
    if (settings.model) selectModel.value = settings.model;
  }

  function saveSettings(newPartial) {
    currentSettings = { ...currentSettings, ...newPartial };
    if (chrome.storage && chrome.storage.sync) {
      chrome.storage.sync.set(currentSettings);
    }
    // Relay updated settings to content script for real-time DOM update
    chrome.runtime.sendMessage({
      type: 'UPDATE_SETTINGS',
      settings: currentSettings
    }).catch(() => {});
  }

  // 3. Fetch initial capture status from background service worker
  chrome.runtime.sendMessage({ type: 'GET_STATUS' }, (response) => {
    if (chrome.runtime.lastError) {
      updateStatusBadge('DISCONNECTED');
      showError('Local subtitle service is unavailable.');
      return;
    }

    if (response && response.success && response.state) {
      const state = response.state;
      if (state.isCapturing) {
        updateStatusBadge('LISTENING');
        updateUiState(true, state.tabTitle, state.lastError);
      } else {
        updateStatusBadge('CONNECTED');
        updateUiState(false, '', state.lastError);
      }
    }
  });

  // 4. Listen for live audio diagnostics & status updates
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'AUDIO_DIAGNOSTICS' && message.diagnostics) {
      const diag = message.diagnostics;
      renderAudioDiagnostics(diag);
    }
  });

  // --- Settings Event Listeners ---
  selectSize.addEventListener('change', () => saveSettings({ fontSize: selectSize.value }));
  selectPosition.addEventListener('change', () => saveSettings({ position: selectPosition.value }));
  selectLanguage.addEventListener('change', () => saveSettings({ language: selectLanguage.value }));
  selectModel.addEventListener('change', () => saveSettings({ model: selectModel.value }));

  // --- Start / Stop Action Listeners ---
  btnStart.addEventListener('click', () => {
    clearError();
    updateStatusBadge('CONNECTING');
    btnStart.disabled = true;

    // Ensure WebSocket streaming is enabled
    chrome.runtime.sendMessage({
      type: 'SET_WEBSOCKET_CONFIG',
      enabled: toggleWs.checked,
      url: 'ws://localhost:8765'
    });

    chrome.runtime.sendMessage({ type: 'START_CAPTURE' }, (response) => {
      if (chrome.runtime.lastError) {
        showError('Unable to connect to local service.');
        updateStatusBadge('ERROR');
        updateUiState(false);
        return;
      }

      if (!response || !response.success) {
        const err = response?.error || 'Unable to capture browser audio.';
        showError(err);
        updateStatusBadge('ERROR');
        updateUiState(false);
        return;
      }

      updateStatusBadge('LISTENING');
      updateUiState(true, response.tabTitle);
    });
  });

  btnStop.addEventListener('click', () => {
    btnStop.disabled = true;
    chrome.runtime.sendMessage({ type: 'STOP_CAPTURE' }, () => {
      updateStatusBadge('CONNECTED');
      updateUiState(false);
    });
  });

  btnExportWav.addEventListener('click', () => {
    btnExportWav.disabled = true;
    chrome.runtime.sendMessage({ type: 'EXPORT_WAV_SAMPLE' }, (response) => {
      btnExportWav.disabled = false;
      if (chrome.runtime.lastError || !response || !response.success || !response.dataUrl) {
        showError('No audio frames captured yet to export WAV.');
        return;
      }

      const link = document.createElement('a');
      link.href = response.dataUrl;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      link.download = `tab_audio_sample_${timestamp}.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  });

  toggleWs.addEventListener('change', () => {
    chrome.runtime.sendMessage({
      type: 'SET_WEBSOCKET_CONFIG',
      enabled: toggleWs.checked,
      url: 'ws://localhost:8765'
    });
  });

  // --- Helper UI Functions ---

  function updateStatusBadge(state) {
    statusBadge.className = 'status-badge';
    switch (state) {
      case 'LISTENING':
      case 'PROCESSING':
        statusBadge.classList.add('status-active');
        statusText.textContent = 'Listening';
        break;
      case 'CONNECTING':
        statusBadge.classList.add('status-connecting');
        statusText.textContent = 'Connecting...';
        break;
      case 'CONNECTED':
        statusBadge.classList.add('status-inactive');
        statusText.textContent = 'Connected';
        break;
      case 'ERROR':
        statusBadge.classList.add('status-error');
        statusText.textContent = 'Error';
        break;
      case 'DISCONNECTED':
      default:
        statusBadge.classList.add('status-inactive');
        statusText.textContent = 'Disconnected';
        break;
    }
  }

  function updateUiState(isCapturing, tabTitle = '', errorMsg = null) {
    if (errorMsg) showError(errorMsg);

    if (tabTitle) tabTitleText.textContent = tabTitle;

    if (isCapturing) {
      btnStart.disabled = true;
      btnStop.disabled = false;
      btnExportWav.disabled = false;
    } else {
      btnStart.disabled = false;
      btnStop.disabled = true;
      resetMeter();
    }
  }

  function renderAudioDiagnostics(diag) {
    if (!diag.isCapturing) return;

    updateStatusBadge(diag.wsConnected ? 'LISTENING' : 'CONNECTING');
    btnStart.disabled = true;
    btnStop.disabled = false;
    btnExportWav.disabled = false;

    const percent = Math.min(100, Math.max(0, diag.percent || 0));
    meterFill.style.width = percent + '%';
    meterPercent.textContent = percent + '%';

    if (diag.tabTitle) tabTitleText.textContent = diag.tabTitle;
  }

  function resetMeter() {
    meterFill.style.width = '0%';
    meterPercent.textContent = '0%';
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove('hidden');
  }

  function clearError() {
    errorBanner.classList.add('hidden');
    errorMessage.textContent = '';
  }
});
