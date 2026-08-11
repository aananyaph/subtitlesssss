/**
 * Popup Controller
 * Connects UI interactions to background service worker & offscreen audio engine,
 * renders real-time audio diagnostics, handles WAV sample exports & error banners.
 */

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

  const meterFill = document.getElementById('meterFill');
  const meterPercent = document.getElementById('meterPercent');

  const diagSampleRate = document.getElementById('diagSampleRate');
  const diagChannels = document.getElementById('diagChannels');
  const diagFrames = document.getElementById('diagFrames');
  const diagPeak = document.getElementById('diagPeak');

  // Query active tab title to display in header
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs && tabs.length > 0) {
      tabTitleText.textContent = tabs[0].title || 'Active Tab';
    }
  } catch (e) {
    tabTitleText.textContent = 'Active Tab';
  }

  // Fetch initial capture status from background worker
  chrome.runtime.sendMessage({ type: 'GET_STATUS' }, (response) => {
    if (chrome.runtime.lastError) {
      showError('Failed to communicate with service worker.');
      return;
    }

    if (response && response.success && response.state) {
      const state = response.state;
      updateUiState(state.isCapturing, state.tabTitle, state.lastError);
    }
  });

  // Listen for live audio diagnostics & status updates
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'AUDIO_DIAGNOSTICS' && message.diagnostics) {
      const diag = message.diagnostics;
      renderAudioDiagnostics(diag);
    }
  });

  // --- Button Action Listeners ---

  // Start Capture Button
  btnStart.addEventListener('click', () => {
    clearError();
    btnStart.disabled = true;
    btnStart.textContent = 'Starting...';

    chrome.runtime.sendMessage({ type: 'START_CAPTURE' }, (response) => {
      btnStart.textContent = 'Start Capture';
      
      if (chrome.runtime.lastError) {
        showError(chrome.runtime.lastError.message);
        updateUiState(false);
        return;
      }

      if (!response || !response.success) {
        showError(response?.error || 'Failed to start tab audio capture.');
        updateUiState(false);
        return;
      }

      updateUiState(true, response.tabTitle);
    });
  });

  // Stop Capture Button
  btnStop.addEventListener('click', () => {
    btnStop.disabled = true;
    btnStop.textContent = 'Stopping...';

    chrome.runtime.sendMessage({ type: 'STOP_CAPTURE' }, (response) => {
      btnStop.textContent = 'Stop Capture';
      updateUiState(false);
    });
  });

  // Export 5s WAV Sample Button
  btnExportWav.addEventListener('click', () => {
    btnExportWav.disabled = true;
    btnExportWav.textContent = 'Preparing WAV...';

    chrome.runtime.sendMessage({ type: 'EXPORT_WAV_SAMPLE' }, (response) => {
      btnExportWav.textContent = 'Download 5s WAV Sample';
      btnExportWav.disabled = false;

      if (chrome.runtime.lastError) {
        showError('WAV export error: ' + chrome.runtime.lastError.message);
        return;
      }

      if (!response || !response.success || !response.dataUrl) {
        showError(response?.error || 'No audio frames captured yet to export WAV.');
        return;
      }

      // Download WAV file
      const link = document.createElement('a');
      link.href = response.dataUrl;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      link.download = `tab_audio_sample_${timestamp}.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  });

  // WebSocket Toggle Switch
  toggleWs.addEventListener('change', () => {
    chrome.runtime.sendMessage({
      type: 'SET_WEBSOCKET_CONFIG',
      enabled: toggleWs.checked,
      url: 'ws://localhost:8765'
    });
  });

  // --- Helper Rendering Functions ---

  function updateUiState(isCapturing, tabTitle = '', errorMsg = null) {
    if (errorMsg) {
      showError(errorMsg);
    }

    if (tabTitle) {
      tabTitleText.textContent = tabTitle;
    }

    if (isCapturing) {
      statusBadge.className = 'status-badge status-active';
      statusText.textContent = 'ACTIVE';
      btnStart.disabled = true;
      btnStop.disabled = false;
      btnExportWav.disabled = false;
    } else {
      statusBadge.className = 'status-badge status-inactive';
      statusText.textContent = 'Not Captured';
      btnStart.disabled = false;
      btnStop.disabled = true;
      resetMeterAndDiagnostics();
    }
  }

  function renderAudioDiagnostics(diag) {
    if (!diag.isCapturing) return;

    // Update active tab badge if available
    statusBadge.className = 'status-badge status-active';
    statusText.textContent = 'ACTIVE';
    btnStart.disabled = true;
    btnStop.disabled = false;
    btnExportWav.disabled = false;

    // Meter Bar (Percent: 0 to 100)
    const percent = Math.min(100, Math.max(0, diag.percent || 0));
    meterFill.style.width = percent + '%';
    meterPercent.textContent = percent + '%';

    // Diagnostics readout
    diagSampleRate.textContent = diag.sampleRate ? `${diag.sampleRate} Hz` : '-- Hz';
    diagChannels.textContent = diag.numChannels ? (diag.numChannels === 2 ? '2 (Stereo)' : '1 (Mono)') : '--';
    diagFrames.textContent = (diag.totalFramesProcessed || 0).toLocaleString();
    diagPeak.textContent = `${percent}% (${(diag.db || -100).toFixed(1)} dB)`;

    if (diag.tabTitle) {
      tabTitleText.textContent = diag.tabTitle;
    }
  }

  function resetMeterAndDiagnostics() {
    meterFill.style.width = '0%';
    meterPercent.textContent = '0%';
    diagSampleRate.textContent = '-- Hz';
    diagChannels.textContent = '--';
    diagFrames.textContent = '0';
    diagPeak.textContent = '0.0 %';
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
