"""Silero VAD ONNX Engine Wrapper.

Executes local Voice Activity Detection using ONNX Runtime on CPU.
Auto-downloads official silero_vad.onnx model if not locally cached.
"""
import os
import urllib.request
import numpy as np
import onnxruntime as ort

MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")


class SileroVadEngine:
    """Silero VAD ONNX Runtime CPU Inference Engine."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._ensure_model_exists()

        # Suppress ONNX runtime verbose logging and configure single-threaded CPU execution for minimal footprint
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1

        print(f"[SileroVAD] Loading ONNX model from: {self.model_path}")
        self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])

        # Dynamically inspect input and output specifications
        self.input_specs = {i.name: i for i in self.session.get_inputs()}
        self.output_specs = {o.name: o for o in self.session.get_outputs()}

        self.reset_state()

    def _ensure_model_exists(self):
        """Downloads official silero_vad.onnx model if file does not exist locally."""
        if not os.path.exists(self.model_path):
            print(f"[SileroVAD] Local model not found. Downloading from {MODEL_URL}...")
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                print(f"[SileroVAD] Model downloaded successfully to {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download Silero VAD ONNX model: {e}")

    def reset_state(self):
        """Resets recurrent hidden state for new audio stream session."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def predict(self, frame_float32: np.ndarray, sample_rate: int = 16000) -> float:
        """Executes VAD inference on a 1D Float32 frame (e.g. 512 samples @ 16kHz).
        
        Returns:
            probability: float value between 0.0 (non-speech/silence) and 1.0 (speech).
        """
        if frame_float32.ndim == 1:
            tensor_in = np.expand_dims(frame_float32, axis=0)  # Shape [1, num_samples]
        else:
            tensor_in = frame_float32

        # Build dynamic ONNX inputs dict based on model input names
        inputs = {}
        if "input" in self.input_specs:
            inputs["input"] = tensor_in.astype(np.float32)
        elif "x" in self.input_specs:
            inputs["x"] = tensor_in.astype(np.float32)

        if "sr" in self.input_specs:
            inputs["sr"] = np.array(sample_rate, dtype=np.int64)

        if "state" in self.input_specs:
            inputs["state"] = self._state
        elif "h" in self.input_specs and "c" in self.input_specs:
            inputs["h"] = self._state[0:1]
            inputs["c"] = self._state[1:2]

        # Run ONNX session
        ort_outs = self.session.run(None, inputs)

        # Extract speech probability and update recurrent state
        if len(ort_outs) == 2:
            out, new_state = ort_outs
            self._state = new_state
            prob = float(out[0][0]) if out.ndim > 1 else float(out[0])
        else:
            out = ort_outs[0]
            prob = float(out[0][0]) if out.ndim > 1 else float(out[0])

        return max(0.0, min(1.0, prob))
