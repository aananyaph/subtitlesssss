/**
 * Audio Processing Utilities for Tab Audio Capture
 * Handles signal metrics, PCM conversions, WAV blob encoding, and resampling.
 */

/**
 * Calculates RMS, peak level, and dB value from Float32 PCM sample array.
 * @param {Float32Array} float32Array 
 * @returns {{ rms: number, peak: number, db: number, percent: number }}
 */
export function calculateAudioLevels(float32Array) {
  if (!float32Array || float32Array.length === 0) {
    return { rms: 0, peak: 0, db: -100, percent: 0 };
  }

  let sumSquares = 0;
  let peak = 0;

  for (let i = 0; i < float32Array.length; i++) {
    const absVal = Math.abs(float32Array[i]);
    if (absVal > peak) {
      peak = absVal;
    }
    sumSquares += absVal * absVal;
  }

  const rms = Math.sqrt(sumSquares / float32Array.length);
  // Prevent log10(0) infinity
  const db = rms > 0.00001 ? 20 * Math.log10(rms) : -100;
  // Convert peak level (0.0 to 1.0) to user-friendly percentage (0 to 100)
  const percent = Math.min(100, Math.round(peak * 100));

  return { rms, peak, db, percent };
}

/**
 * Converts Float32Array audio samples [-1.0, 1.0] to Int16Array [-32768, 32767].
 * Clamps values outside [-1.0, 1.0] to prevent audio clipping distortion.
 * @param {Float32Array} float32Array 
 * @returns {Int16Array}
 */
export function float32ToInt16(float32Array) {
  const int16Array = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16Array;
}

/**
 * Encodes an Int16Array of PCM samples into a downloadable RIFF WAV Blob.
 * @param {Int16Array} int16Samples 
 * @param {number} sampleRate 
 * @param {number} numChannels 
 * @returns {Blob}
 */
export function encodeWavBlob(int16Samples, sampleRate, numChannels = 2) {
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = int16Samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  /* RIFF chunk descriptor */
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  /* fmt sub-chunk */
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true);  // AudioFormat (1 for PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // BitsPerSample

  /* data sub-chunk */
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write PCM audio samples
  let offset = 44;
  for (let i = 0; i < int16Samples.length; i++, offset += 2) {
    view.setInt16(offset, int16Samples[i], true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Helper to write ASCII strings to DataView.
 */
function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

/**
 * Downsamples Float32Array PCM audio from inputSampleRate to targetSampleRate (e.g., 48000Hz -> 16000Hz).
 * Lightweight linear interpolation suited for low CPU footprint (Intel UHD 620).
 * @param {Float32Array} buffer 
 * @param {number} inputSampleRate 
 * @param {number} targetSampleRate 
 * @returns {Float32Array}
 */
export function resampleBuffer(buffer, inputSampleRate, targetSampleRate) {
  if (inputSampleRate === targetSampleRate) {
    return buffer;
  }
  const ratio = inputSampleRate / targetSampleRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const originPos = i * ratio;
    const index1 = Math.floor(originPos);
    const index2 = Math.min(index1 + 1, buffer.length - 1);
    const weight = originPos - index1;

    result[i] = buffer[index1] * (1 - weight) + buffer[index2] * weight;
  }

  return result;
}
