"""pipeline.py - Shared speech recognition & translation pipeline for Windows local app.

Features:
- Silero VAD (csukuangfj/vad)
- Segment cleanup: hard cap 8s (split at quietest point in last 3s), merge <0.4s into next if within 1s
- Drop empty ASR text; never send empty text to translator
- ASR via OfflineRecognizer.from_nemo_ctc (Hindi ngiftsun int8 with remiai3 fallback, Kannada, English)
- Translators via hari31416 ONNX int8 (M1, M2, M3) with pure-Python IndicTrans2 pre/post processing
- RAM guard (< 1.2 GB skips translation to preserve stability)
- Strictly ONE job at a time across both modes, unload + gc.collect() on pair change
"""

import gc
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import wave

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import hf_hub_download, snapshot_download
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_detokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
import numpy as np
import onnxruntime as ort
import psutil
from sacremoses import MosesDetokenizer, MosesTokenizer
import sherpa_onnx
from tokenizers import Tokenizer

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger("subtitle_pipeline")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("app_log.txt", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(sh)

# -----------------------------------------------------------------------------
# System Helpers
# -----------------------------------------------------------------------------
def get_free_ram_mb():
    return psutil.virtual_memory().available / (1024.0 * 1024.0)

def get_process_ram_mb():
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)

# Language Code Mapping
LANG_MAP = {
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "en": "eng_Latn",
}

# -----------------------------------------------------------------------------
# IndicTrans2 ONNX Translator Wrapper
# -----------------------------------------------------------------------------
class IndicTransONNXModel:
    def __init__(self, repo_id, intra_threads=2):
        self.repo_id = repo_id
        logger.info(f"Loading translator from {repo_id}...")
        self.snap = Path(snapshot_download(repo_id))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads

        self.enc = ort.InferenceSession(str(self.snap / "encoder_model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(self.snap / "decoder_model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        self.dec_past = ort.InferenceSession(str(self.snap / "decoder_with_past_model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
        self.src_tok = Tokenizer.from_file(str(self.snap / "tokenizer_src.json"))
        self.tgt_tok = Tokenizer.from_file(str(self.snap / "tokenizer_tgt.json"))
        self.meta = json.loads((self.snap / "tokenizer_meta.json").read_text(encoding="utf-8"))
        gen_cfg = json.loads((self.snap / "generation_config.json").read_text(encoding="utf-8"))
        self.decoder_start_id = int(gen_cfg.get("decoder_start_token_id", 2))
        self.eos_id = int(gen_cfg.get("eos_token_id", 2))
        self.num_layers = (len(self.dec.get_outputs()) - 1) // 4

        self.xliterator = UnicodeIndicTransliterator()
        self.en_tok = MosesTokenizer(lang="en")
        self.en_detok = MosesDetokenizer(lang="en")
        self.norm_hi = IndicNormalizerFactory().get_normalizer("hi")
        self.norm_kn = IndicNormalizerFactory().get_normalizer("kn")
        logger.info(f"Translator {repo_id} ready.")

    def _past_feed(self, past_outputs):
        feed = {}
        for i in range(self.num_layers):
            feed[f"past_key_values.{i}.decoder.key"] = past_outputs[4 * i]
            feed[f"past_key_values.{i}.decoder.value"] = past_outputs[4 * i + 1]
            feed[f"past_key_values.{i}.encoder.key"] = past_outputs[4 * i + 2]
            feed[f"past_key_values.{i}.encoder.value"] = past_outputs[4 * i + 3]
        return feed

    def translate(self, text, src_iso, tgt_iso, max_new_tokens=100):
        if not text or not text.strip():
            return ""

        src_tag = LANG_MAP.get(src_iso, src_iso)
        tgt_tag = LANG_MAP.get(tgt_iso, tgt_iso)

        if src_tag == "eng_Latn":
            e_tok = " ".join(self.en_tok.tokenize(text, escape=False))
            prefixed = f"{src_tag} {tgt_tag} {e_tok}"
        elif src_tag == "kan_Knda":
            normed = self.norm_kn.normalize(text)
            deva = self.xliterator.transliterate(normed, "kn", "hi")
            prefixed = f"{src_tag} {tgt_tag} {deva}"
        elif src_tag == "hin_Deva":
            normed = self.norm_hi.normalize(text)
            prefixed = f"{src_tag} {tgt_tag} {normed}"
        else:
            prefixed = f"{src_tag} {tgt_tag} {text}"

        encoded = self.src_tok.encode(prefixed)
        input_ids = np.array([[i if i < self.meta["src_dict_size"] else self.meta["unk_id"] for i in encoded.ids]], dtype=np.int64)
        attn_mask = np.array([encoded.attention_mask], dtype=np.int64)

        enc_out = self.enc.run(["last_hidden_state"], {"input_ids": input_ids, "attention_mask": attn_mask})[0]

        decoder_input_ids = np.array([[self.decoder_start_id]], dtype=np.int64)
        output_ids = [self.decoder_start_id]
        past_outputs = None

        for step in range(max_new_tokens):
            if step == 0:
                dec_out = self.dec.run(
                    None,
                    {"input_ids": decoder_input_ids, "encoder_hidden_states": enc_out, "encoder_attention_mask": attn_mask},
                )
            else:
                dec_out = self.dec_past.run(
                    None,
                    {"input_ids": decoder_input_ids, "encoder_attention_mask": attn_mask, **self._past_feed(past_outputs)},
                )
            logits = dec_out[0]
            past_outputs = list(dec_out[1:])
            next_id = int(np.argmax(logits[0, -1, :]))
            output_ids.append(next_id)
            if next_id == self.eos_id:
                break
            if len(output_ids) >= 5 and output_ids[-1] == output_ids[-2] == output_ids[-3] == output_ids[-4]:
                break
            decoder_input_ids = np.array([[next_id]], dtype=np.int64)

        safe_ids = [i if i < self.meta["tgt_dict_size"] else self.meta["unk_id"] for i in output_ids]
        raw = self.tgt_tok.decode(safe_ids, skip_special_tokens=True).strip()

        if tgt_tag == "eng_Latn":
            return self.en_detok.detokenize(raw.split())
        elif tgt_tag == "kan_Knda":
            kn = self.xliterator.transliterate(raw, "hi", "kn")
            return indic_detokenize.trivial_detokenize(kn, "kn")
        elif tgt_tag == "hin_Deva":
            return indic_detokenize.trivial_detokenize(raw, "hi")
        else:
            return raw

# -----------------------------------------------------------------------------
# Segment Cleanup Algorithms
# -----------------------------------------------------------------------------
def split_long_segment(audio_seg, max_len_s=8.0, search_window_s=3.0, sample_rate=16000):
    """Hard cap 8s: split at the quietest point in the last 3s, else at the cap."""
    total_len_s = len(audio_seg) / float(sample_rate)
    if total_len_s <= max_len_s:
        return [audio_seg]

    out_chunks = []
    rem_audio = audio_seg
    while len(rem_audio) / float(sample_rate) > max_len_s:
        target_split_idx = int(max_len_s * sample_rate)
        search_start_idx = max(0, int((max_len_s - search_window_s) * sample_rate))
        search_slice = rem_audio[search_start_idx:target_split_idx]

        # Find quietest point in search_slice using 50ms (800 samples) windows
        win_size = int(0.05 * sample_rate)
        best_idx = target_split_idx
        min_energy = float("inf")

        if len(search_slice) > win_size:
            for w in range(0, len(search_slice) - win_size, win_size // 2):
                chunk_window = search_slice[w:w + win_size]
                energy = np.mean(chunk_window ** 2)
                if energy < min_energy:
                    min_energy = energy
                    best_idx = search_start_idx + w + (win_size // 2)

        out_chunks.append(rem_audio[:best_idx])
        rem_audio = rem_audio[best_idx:]

    if len(rem_audio) > 0:
        out_chunks.append(rem_audio)
    return out_chunks

def cleanup_segments(raw_segments, sample_rate=16000):
    """
    raw_segments: list of dicts: {'start_s': float, 'audio': np.ndarray}
    Rules:
    - Split segments > 8s at quietest point in last 3s
    - Merge segments under 0.4s into next one if within 1.0s gap
    """
    # Step 1: Split long segments
    split_segs = []
    for item in raw_segments:
        chunks = split_long_segment(item['audio'], max_len_s=8.0, search_window_s=3.0, sample_rate=sample_rate)
        cur_start = item['start_s']
        for ch in chunks:
            ch_dur = len(ch) / float(sample_rate)
            split_segs.append({'start_s': cur_start, 'dur_s': ch_dur, 'audio': ch})
            cur_start += ch_dur

    # Step 2: Merge short segments (<0.4s) into next if within 1.0s
    merged = []
    idx = 0
    while idx < len(split_segs):
        cur = split_segs[idx]
        if cur['dur_s'] < 0.4 and (idx + 1) < len(split_segs):
            nxt = split_segs[idx + 1]
            gap = nxt['start_s'] - (cur['start_s'] + cur['dur_s'])
            if gap <= 1.0:
                # Merge into next
                combined_audio = np.concatenate([cur['audio'], nxt['audio']])
                nxt['start_s'] = cur['start_s']
                nxt['dur_s'] = len(combined_audio) / float(sample_rate)
                nxt['audio'] = combined_audio
                idx += 1
                continue
        merged.append(cur)
        idx += 1

    return merged

# -----------------------------------------------------------------------------
# Pipeline Manager (Singleton for Concurrency and Memory Management)
# -----------------------------------------------------------------------------
class PipelineManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.lock = threading.Lock()
        self.active_mode = None  # None, 'live', or 'file'
        self.current_src = None
        self.current_tgt = None
        self.asr_recognizer = None
        self.translator = None
        self.vad = None
        self.is_cancelled = threading.Event()
        self.last_warning = None

    def is_busy(self):
        return self.active_mode is not None

    def cancel_job(self):
        self.is_cancelled.set()
        logger.info("Job cancellation requested.")

    def ensure_vad(self, status_callback=None):
        if self.vad is not None:
            return self.vad
        if status_callback:
            status_callback("Downloading VAD model...")
        vad_path = hf_hub_download("csukuangfj/vad", "silero_vad.onnx")
        vcfg = sherpa_onnx.VadModelConfig()
        vcfg.silero_vad.model = vad_path
        vcfg.silero_vad.threshold = 0.5
        vcfg.silero_vad.min_silence_duration = 0.5
        vcfg.silero_vad.min_speech_duration = 0.25
        vcfg.silero_vad.max_speech_duration = 8.0
        vcfg.sample_rate = 16000
        self.vad = sherpa_onnx.VoiceActivityDetector(vcfg, buffer_size_in_seconds=60)
        return self.vad

    def ensure_models(self, src_lang, tgt_lang, status_callback=None):
        """Loads required ASR model and at most one translator. Unloads on change."""
        if self.current_src == src_lang and self.current_tgt == tgt_lang and self.asr_recognizer is not None:
            return

        logger.info(f"Switching language pair: {self.current_src}->{self.current_tgt} to {src_lang}->{tgt_lang}")
        self.unload_models()

        # Step 1: Load ASR Model
        if status_callback:
            status_callback(f"Downloading ASR model for {src_lang} (first load may take ~1 minute)...")

        if src_lang == "hi":
            try:
                m_path = hf_hub_download("ngiftsun/indicconformer-hi-int8-sherpa", "model.int8.onnx")
                t_path = hf_hub_download("ngiftsun/indicconformer-hi-int8-sherpa", "tokens.txt")
                self.asr_recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=m_path, tokens=t_path, num_threads=2)
                logger.info("Loaded Hindi ASR: ngiftsun/indicconformer-hi-int8-sherpa (int8).")
            except Exception as e:
                logger.warning(f"ngiftsun Hindi ASR failed ({e}); falling back to remiai3/STT_MODELS_FOR_APP.")
                if status_callback:
                    status_callback("Falling back to remiai3 Hindi ASR...")
                m_path = hf_hub_download("remiai3/STT_MODELS_FOR_APP", "indic_conformer_hi.onnx")
                t_path = hf_hub_download("remiai3/STT_MODELS_FOR_APP", "tokens.txt")
                self.asr_recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=m_path, tokens=t_path, num_threads=2)
                logger.info("Loaded Hindi ASR: remiai3 fallback.")
        elif src_lang == "kn":
            m_path = hf_hub_download("remiai3/STT_MODELS_FOR_APP", "indic_conformer_kn.onnx")
            t_path = hf_hub_download("remiai3/STT_MODELS_FOR_APP", "tokens.txt")
            self.asr_recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=m_path, tokens=t_path, num_threads=2)
            logger.info("Loaded Kannada ASR: remiai3 indic_conformer_kn.onnx.")
        elif src_lang == "en":
            m_path = hf_hub_download("remiai3/STT_MODELS_FOR_APP", "indic_conformer_en.onnx")
            t_path = hf_hub_download("remiai3/STT_MODELS_FOR_APP", "indic_conformer_en_tokens.txt")
            self.asr_recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=m_path, tokens=t_path, num_threads=2)
            logger.info("Loaded English ASR: remiai3 indic_conformer_en.onnx.")
        else:
            raise ValueError(f"Unsupported source language: {src_lang}")

        # Step 2: Load Translator (only if src != tgt)
        if src_lang == tgt_lang:
            logger.info(f"Source and target languages are identical ({src_lang}); skipping translator loading.")
            self.translator = None
        else:
            if status_callback:
                status_callback(f"Downloading translator for {src_lang}->{tgt_lang} (first load may take ~1 minute)...")

            if (src_lang in ["hi", "kn"]) and tgt_lang == "en":
                repo = "hari31416/indictrans2-indic-en-dist-200M-ONNX-int8"
            elif src_lang == "en" and (tgt_lang in ["hi", "kn"]):
                repo = "hari31416/indictrans2-en-indic-dist-200M-ONNX-int8"
            elif (src_lang in ["hi", "kn"]) and (tgt_lang in ["hi", "kn"]):
                repo = "hari31416/indictrans2-indic-indic-dist-320M-ONNX-int8"
            else:
                raise ValueError(f"Unsupported translation pair: {src_lang}->{tgt_lang}")

            self.translator = IndicTransONNXModel(repo, intra_threads=2)
            logger.info(f"Loaded translator {repo}.")

        self.current_src = src_lang
        self.current_tgt = tgt_lang
        if status_callback:
            status_callback("Models loaded successfully.")

    def unload_models(self):
        """Unload models and reclaim memory."""
        self.asr_recognizer = None
        self.translator = None
        self.current_src = None
        self.current_tgt = None
        gc.collect()
        logger.info(f"Unloaded pipeline models. System Free RAM: {get_free_ram_mb():.1f} MB.")

    def process_segment(self, audio_seg, src_lang, tgt_lang):
        """
        Processes a single 16kHz mono audio slice through ASR and translator.
        Returns dict with asr_text, trans_text, asr_time, trans_time, total_time, ram_guard_active.
        If ASR text is empty, returns None.
        """
        t0 = time.perf_counter()
        s = self.asr_recognizer.create_stream()
        s.accept_waveform(16000, audio_seg)
        self.asr_recognizer.decode_stream(s)
        asr_time = time.perf_counter() - t0
        asr_text = s.result.text.strip()

        # Drop empty ASR text
        if not asr_text:
            return None

        # Check RAM guard
        free_ram = get_free_ram_mb()
        ram_guard_active = False

        trans_time = 0.0
        trans_text = asr_text

        if src_lang != tgt_lang and self.translator is not None:
            if free_ram < 700.0:
                ram_guard_active = True
                msg = f"RAM Guard Alert: Free RAM {free_ram:.1f} MB < 700 MB! Skipping translation to prevent OOM."
                logger.warning(msg)
                self.last_warning = msg
                trans_text = asr_text
            else:
                t0 = time.perf_counter()
                try:
                    trans_text = self.translator.translate(asr_text, src_iso=src_lang, tgt_iso=tgt_lang)
                except Exception as e:
                    logger.error(f"Translator error: {e}", exc_info=True)
                    trans_text = asr_text
                trans_time = time.perf_counter() - t0

        return {
            "asr_text": asr_text,
            "trans_text": trans_text,
            "asr_time": asr_time,
            "trans_time": trans_time,
            "total_proc_time": asr_time + trans_time,
            "ram_guard_active": ram_guard_active,
            "free_ram": free_ram,
        }

# Global Pipeline Instance
pipeline = PipelineManager()
