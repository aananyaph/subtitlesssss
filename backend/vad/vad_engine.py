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

        print(f"[SileroVAD Specs] Inputs : {[(i.name, i.shape, i.type) for i in self.session.get_inputs()]}")
        print(f"[SileroVAD Specs] Outputs: {[(o.name, o.shape, o.type) for o in self.session.get_outputs()]}")

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
        """Resets recurrent hidden state and rolling window context for a new audio stream session."""
        # 64-sample rolling context required by the v5 model's 576-sample input window
        self._context_64 = np.zeros(64, dtype=np.float32)

        if "h" in self.input_specs and "c" in self.input_specs:
            # Shape for h and c in Silero VAD v4: [1, 1, 64] or [2, 1, 64]
            h_shape = self.input_specs["h"].shape
            c_shape = self.input_specs["c"].shape
            h_dim = [dim if isinstance(dim, int) else 1 for dim in h_shape]
            c_dim = [dim if isinstance(dim, int) else 1 for dim in c_shape]
            self._h = np.zeros(h_dim, dtype=np.float32)
            self._c = np.zeros(c_dim, dtype=np.float32)
            self._state = None
        else:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._h = None
            self._c = None

    def predict(self, frame_float32: np.ndarray, sample_rate: int = 16000) -> float:
        """Executes VAD inference on a 1D Float32 frame (512 samples @ 16kHz).

        The bundled silero_vad.onnx is a v5 model whose 16kHz input contract is a
        576-sample window: 64 samples of rolling context followed by 512 new
        samples. Feeding a bare 512-sample frame makes the model collapse to a
        flat ~0.0006 output (silent detection failure), so the rolling context is
        maintained here and prepended to every frame.

        Returns:
            probability: float value between 0.0 (non-speech/silence) and 1.0 (speech).
        """
        if frame_float32.ndim == 1:
            frame_1d = frame_float32
        else:
            frame_1d = np.ravel(frame_float32)

        if len(frame_1d) >= 512:
            frame512 = frame_1d[-512:]
        else:
            frame512 = np.pad(frame_1d, (0, 512 - len(frame_1d)), mode="constant")

        # Build the 576-sample input window (64 rolling context + 512 new samples)
        window576 = np.concatenate([self._context_64, frame512.astype(np.float32)])
        self._context_64 = window576[-64:].astype(np.float32)

        tensor_in = np.expand_dims(window576, axis=0)  # Shape [1, 576]

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
