/**
 * Module 5 - Subtitle Overlay
 * Injected into all pages. Detects HTML5 video and overlays real-time transcripts.
 */

if (!window.__LIVE_SUBTITLE_OVERLAY_INJECTED__) {
  window.__LIVE_SUBTITLE_OVERLAY_INJECTED__ = true;

  let currentVideo = null;
  let wrapper = null;
  let textElement = null;
  let subtitleTimeout = null;
  let rafId = null;
  let lastRect = {};

  console.log('[SubtitleOverlay] Content script loaded.');

  // ── DOM Setup ───────────────────────────────────────────────────────────────

  function createOverlay() {
    if (wrapper) return;
    
    wrapper = document.createElement('div');
    wrapper.id = 'live-subtitle-wrapper';
    
    textElement = document.createElement('div');
    textElement.id = 'live-subtitle-text';
    textElement.className = 'hidden';
    
    wrapper.appendChild(textElement);
    console.log('[SubtitleOverlay] Overlay created.');
  }

  // ── Positioning & Lifecycle ─────────────────────────────────────────────────

  function attachOverlay(video) {
    createOverlay();
    
    const targetParent = (video && video.parentElement) || document.fullscreenElement || document.body;
    
    if (wrapper.parentElement !== targetParent) {
      targetParent.appendChild(wrapper);
      lastRect = {}; // Force re-apply styles
    }
  }

  function updateWrapperRect() {
    if (!currentVideo || !wrapper) return;

    if (document.fullscreenElement) {
      // In fullscreen, let the wrapper fill the fullscreen element
      if (lastRect.fullscreen !== true) {
        wrapper.style.position = 'absolute';
        wrapper.style.top = '0px';
        wrapper.style.left = '0px';
        wrapper.style.width = '100%';
        wrapper.style.height = '100%';
        lastRect = { fullscreen: true };
      }
    } else {
      // Not fullscreen: float securely over the video's exact bounding rect
      const rect = currentVideo.getBoundingClientRect();
      if (
        rect.top !== lastRect.top ||
        rect.left !== lastRect.left ||
        rect.width !== lastRect.width ||
        rect.height !== lastRect.height
      ) {
        wrapper.style.position = 'fixed';
        wrapper.style.top = rect.top + 'px';
        wrapper.style.left = rect.left + 'px';
        wrapper.style.width = rect.width + 'px';
        wrapper.style.height = rect.height + 'px';
        lastRect = {
          top: rect.top,
          left: rect.left,
          width: rect.width,
          height: rect.height,
          fullscreen: false
        };
      }
    }

    // Continue tracking position
    rafId = requestAnimationFrame(updateWrapperRect);
  }

  // Handle fullscreen toggle explicitly as a layout change hint
  document.addEventListener('fullscreenchange', () => {
    console.log('[SubtitleOverlay] Fullscreen changed.');
    if (currentVideo) {
      attachOverlay(currentVideo);
    }
  });

  // ── Module 10 — Playback Synchronization & Seek Detection ──────────────────
  let currentGenerationId = 1;
  let currentlyDisplayedSegmentId = null;
  let stalenessTimer = null;
  let lastVideoTime = 0;
  const SEEK_JUMP_THRESHOLD_SEC = 1.5;
  const OVERLAY_STALE_TIMEOUT_SEC = 4.0;

  function bindVideoEvents(video) {
    if (!video || video.__LIVE_SUBTITLE_EVENTS_BOUND__) return;
    video.__LIVE_SUBTITLE_EVENTS_BOUND__ = true;
    lastVideoTime = video.currentTime;

    video.addEventListener('seeking', () => {
      console.log(`[SEEK] Video seeking to ${video.currentTime.toFixed(2)}s`);
      triggerSeekFlush('seek_flush');
    });

    video.addEventListener('timeupdate', () => {
      const curTime = video.currentTime;
      const diff = Math.abs(curTime - lastVideoTime);
      lastVideoTime = curTime;
      if (diff > SEEK_JUMP_THRESHOLD_SEC) {
        console.log(`[SEEK] Timeupdate jump detected: diff=${diff.toFixed(2)}s (${curTime.toFixed(2)}s)`);
        triggerSeekFlush('seek_flush');
      }
    });

    video.addEventListener('pause', () => {
      console.log('[VIDEO] Video paused');
      if (stalenessTimer) clearTimeout(stalenessTimer);
    });

    video.addEventListener('playing', () => {
      console.log('[VIDEO] Video playing');
      resetStalenessTimer();
    });

    video.addEventListener('ended', () => {
      console.log('[VIDEO] Video ended');
      triggerSeekFlush('video_end');
      hideOverlay('video_end');
    });

    video.addEventListener('emptied', () => {
      console.log('[VIDEO] Video source changed');
      triggerSeekFlush('video_switch');
      hideOverlay('video_switch');
    });
  }

  function triggerSeekFlush(reason) {
    currentGenerationId++;
    hideOverlay(reason);

    const videoTime = currentVideo ? currentVideo.currentTime : 0;
    if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.sendMessage) {
      chrome.runtime.sendMessage({
        type: 'VIDEO_CONTROL_EVENT',
        event: reason,
        video_time: videoTime,
        generation_id: currentGenerationId
      }).catch(() => {});
    }
  }

  function hideOverlay(reason) {
    if (textElement) {
      textElement.className = 'hidden';
    }
    if (subtitleTimeout) {
      clearTimeout(subtitleTimeout);
      subtitleTimeout = null;
    }
    if (stalenessTimer) {
      clearTimeout(stalenessTimer);
      stalenessTimer = null;
    }
    console.log(`[OVERLAY] Cleared overlay (reason='${reason}')`);
  }

  function resetStalenessTimer() {
    if (stalenessTimer) clearTimeout(stalenessTimer);
    if (!currentVideo || currentVideo.paused || currentVideo.ended) return;

    stalenessTimer = setTimeout(() => {
      if (currentVideo && !currentVideo.paused && !currentVideo.ended) {
        hideOverlay('stale_timeout');
      }
    }, OVERLAY_STALE_TIMEOUT_SEC * 1000);
  }

  // ── Video Detection Loop ───────────────────────────────────────────────────

  function findActiveVideo() {
    const videos = Array.from(document.querySelectorAll('video')).filter(v => 
      v.offsetWidth > 100 && v.offsetHeight > 100 // Ignore tiny tracking pixels
    );
    if (videos.length === 0) return null;
    // Prefer playing video, or fallback to the first visible one
    return videos.find(v => !v.paused) || videos[0];
  }

  setInterval(() => {
    const video = findActiveVideo();
    if (video && video !== currentVideo) {
      console.log('[SubtitleOverlay] Active video detected/changed.');
      currentVideo = video;
      
      bindVideoEvents(video);
      attachOverlay(video);
      
      // Start tracking loop if not already running
      if (rafId) cancelAnimationFrame(rafId);
      updateWrapperRect();
    } else if (video) {
      bindVideoEvents(video);
    }
  }, 1000);

  let currentSettings = { fontSize: 'medium', position: 'bottom' };

  function applySettings(settings) {
    if (!settings) return;
    currentSettings = { ...currentSettings, ...settings };

    createOverlay();

    if (wrapper) {
      if (currentSettings.position === 'top') {
        wrapper.style.justifyContent = 'flex-start';
        wrapper.style.paddingTop = '8%';
        wrapper.style.paddingBottom = '0%';
      } else if (currentSettings.position === 'middle') {
        wrapper.style.justifyContent = 'center';
        wrapper.style.paddingTop = '0%';
        wrapper.style.paddingBottom = '0%';
      } else {
        wrapper.style.justifyContent = 'flex-end';
        wrapper.style.paddingBottom = '8%';
        wrapper.style.paddingTop = '0%';
      }
    }

    if (textElement) {
      if (currentSettings.fontSize === 'small') {
        textElement.style.fontSize = '20px';
        textElement.style.padding = '6px 14px';
      } else if (currentSettings.fontSize === 'large') {
        textElement.style.fontSize = '36px';
        textElement.style.padding = '14px 28px';
      } else {
        textElement.style.fontSize = '28px';
        textElement.style.padding = '10px 20px';
      }
    }
  }

  // Fetch initial settings from chrome.storage if available
  if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.sync) {
    chrome.storage.sync.get(['fontSize', 'position'], (items) => {
      if (items) applySettings(items);
    });
  }

  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'RENDER_TRANSCRIPT') {
      renderTranscript(message.transcript);
    } else if (message.type === 'UPDATE_SETTINGS') {
      applySettings(message.settings);
    }
  });

  function renderTranscript(transcript) {
    if (!currentVideo) {
      currentVideo = findActiveVideo();
    }
    if (!currentVideo) {
      console.log('[SubtitleOverlay] Transcript received but no active <video> element found on page.');
      return;
    }

    if (transcript.type === 'flush_ack') {
      console.log(`[FLUSH] Received backend flush_ack for generation ${transcript.generation_id}`);
      if (transcript.generation_id) currentGenerationId = transcript.generation_id;
      return;
    }

    // Module 10: Generation Guard & Fresh Session Resync
    if (transcript.generation_id) {
      if (transcript.generation_id > currentGenerationId) {
        currentGenerationId = transcript.generation_id;
      } else if (transcript.generation_id < currentGenerationId) {
        // If backend started a fresh session (gen 1), resync frontend generation ID
        if (transcript.generation_id === 1) {
          console.log('[SubtitleOverlay] Resynced generation ID to fresh backend session (gen: 1)');
          currentGenerationId = 1;
        } else {
          console.log(`[STALE] Discarded transcript from past generation ${transcript.generation_id} (active: ${currentGenerationId})`);
          return;
        }
      }
    }

    createOverlay();
    attachOverlay(currentVideo);

    // Module 9: Stale Correction Protection
    if (transcript.type === 'correction') {
      if (!transcript.segment_id || transcript.segment_id !== currentlyDisplayedSegmentId) {
        console.log(`[SubtitleOverlay] Discarded stale LLM correction for ${transcript.segment_id} (active: ${currentlyDisplayedSegmentId})`);
        return;
      }
      console.log(`[SubtitleOverlay] Applied LLM correction for ${transcript.segment_id}: "${transcript.text}"`);
      textElement.textContent = transcript.text;

      if (subtitleTimeout) {
        clearTimeout(subtitleTimeout);
      }
      subtitleTimeout = setTimeout(() => {
        textElement.className = 'hidden';
        console.log('[SubtitleOverlay] Subtitle faded out after correction.');
      }, 4000);
      resetStalenessTimer();
      return;
    }

    // Handle partial and final events
    if (transcript.segment_id) {
      currentlyDisplayedSegmentId = transcript.segment_id;
    }
    
    const overlayReceivedTime = Date.now();

    // Update text
    textElement.textContent = transcript.text;
    
    // Show it if hidden
    if (textElement.className === 'hidden') {
      textElement.className = 'visible';
    }
    
    const overlayRenderedTime = Date.now();
    const renderDurationMs = overlayRenderedTime - overlayReceivedTime;

    if (transcript.metrics && transcript.metrics.audio_capture_time) {
      const audioToOverlaySec = ((overlayRenderedTime / 1000.0) - transcript.metrics.audio_capture_time).toFixed(2);
      const audioToTranscriptSec = (transcript.metrics.end_to_end_latency_ms / 1000.0).toFixed(2);
      console.log(`[PERF] Audio -> Transcript: ${audioToTranscriptSec}s | Audio -> Overlay: ${audioToOverlaySec}s (Render: ${renderDurationMs}ms)`);
    }

    console.log(`[SubtitleOverlay] ${transcript.type === 'final' ? 'Final' : 'Partial'} [gen:${transcript.generation_id || 1} seg:${transcript.segment_id || 'stream'}] received: "${transcript.text}"`);

    // Reset staleness safety timer on fresh update
    resetStalenessTimer();

    // Clear any existing cleanup timer
    if (subtitleTimeout) {
      clearTimeout(subtitleTimeout);
      subtitleTimeout = null;
    }

    // If this is a final transcript, it should fade out after a few seconds of silence
    if (transcript.type === 'final') {
      subtitleTimeout = setTimeout(() => {
        textElement.className = 'hidden';
        console.log('[SubtitleOverlay] Stale subtitle faded out.');
      }, 4000); // 4 second timeout before hiding
    }
  }
}


