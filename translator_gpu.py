"""translator_gpu.py - PyTorch fp16 IndicTrans2 1B GPU translation layer.

Compatible drop-in replacement for IndicTransONNXModel with identical surface:
  .repo_id: str
  .translate(text: str, src_iso: str, tgt_iso: str) -> str
"""
import logging
import math
import os
import sys

logger = logging.getLogger("translator_gpu")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(_sh)

# Standard FLORES-200 language code mapping
LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "eng_Latn": "eng_Latn",
    "hin_Deva": "hin_Deva",
    "kan_Knda": "kan_Knda",
}

# Verified Hugging Face 1B Checkpoints
REPO_EN_INDIC = "ai4bharat/indictrans2-en-indic-1B"
REPO_INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"
REPO_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-1B"


def get_gpu_repo(src: str, tgt: str) -> str:
    """Return default IndicTrans2 1B checkpoint repo ID for direction."""
    if tgt == "en":
        return REPO_INDIC_EN
    if src == "en":
        return REPO_EN_INDIC
    return REPO_INDIC_INDIC


class IndicTrans2GPU:
    """GPU-accelerated PyTorch fp16 IndicTrans2 1B translation model."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; IndicTrans2GPU requires an active CUDA GPU.")

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError as e:
            raise ImportError(
                "IndicTransToolkit is required for IndicTrans2GPU. Install with: pip install indictranstoolkit"
            ) from e

        hf_token = os.environ.get("HF_TOKEN") or None
        logger.info(f"[IndicTrans2GPU] Loading model and tokenizer for {repo_id}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                trust_remote_code=True,
                token=hf_token,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                repo_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=hf_token,
            ).to("cuda").eval()

            self.ip = IndicProcessor(inference=True)

            vram_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            logger.info(f"[IndicTrans2GPU] Loaded {repo_id} | VRAM allocated: {vram_mb:.1f} MB")

            # Warm up with one dummy sentence after load
            self._warm_up()
        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            logger.error(f"[IndicTrans2GPU] CUDA OOM while loading {repo_id}: {oom}")
            raise RuntimeError(f"CUDA OOM loading {repo_id}: {oom}") from oom
        except Exception as exc:
            logger.error(f"[IndicTrans2GPU] Failed loading {repo_id}: {exc}", exc_info=True)
            raise

    def _warm_up(self):
        """Warm up model and CUDA kernels with a single sentence."""
        try:
            if "en-indic" in self.repo_id:
                self.translate("Hello.", "en", "hi")
            elif "indic-en" in self.repo_id:
                self.translate("नमस्ते", "hi", "en")
            else:
                self.translate("नमस्ते", "hi", "kn")
            logger.info(f"[IndicTrans2GPU] Warm-up complete for {self.repo_id}")
        except Exception as e:
            logger.warning(f"[IndicTrans2GPU] Warm-up warning for {self.repo_id}: {e}")

    def translate(self, text: str, src_iso: str, tgt_iso: str) -> str:
        """Translate text from src_iso to tgt_iso using IndicTrans2 1B model on CUDA."""
        if not text or not text.strip():
            return ""

        src_lang = LANG_MAP.get(src_iso, src_iso)
        tgt_lang = LANG_MAP.get(tgt_iso, tgt_iso)
        if src_lang == tgt_lang:
            return text

        import torch

        tr_beams = int(os.environ.get("TR_BEAMS", "2"))
        tr_max_ratio = float(os.environ.get("TR_MAX_RATIO", "1.5"))

        try:
            batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to("cuda")

            src_tokens = inputs.input_ids.shape[1]
            max_new_tokens = math.ceil(tr_max_ratio * src_tokens) + 8

            gen_kwargs = {
                "use_cache": True,
                "max_new_tokens": max_new_tokens,
                "num_beams": tr_beams,
            }
            if tr_beams == 1:
                gen_kwargs["do_sample"] = False

            with torch.inference_mode():
                generated_tokens = self.model.generate(**inputs, **gen_kwargs)

            outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translations = self.ip.postprocess_batch(outputs, lang=tgt_lang)
            return translations[0].strip() if translations else ""

        except torch.cuda.OutOfMemoryError as oom:
            torch.cuda.empty_cache()
            logger.error(f"[IndicTrans2GPU] CUDA OOM during translate(): {oom}")
            raise RuntimeError(f"CUDA OOM during translate: {oom}") from oom

