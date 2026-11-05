import json, logging
from pathlib import Path
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_detokenize
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from sacremoses import MosesDetokenizer, MosesTokenizer
from tokenizers import Tokenizer

logger = logging.getLogger("translator_live")

LANG_MAP = {
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "en": "eng_Latn",
}
class IndicTransONNXModel:
    def __init__(self, repo_id, intra_threads=2):
        self.repo_id = repo_id
        logger.info(f"Loading translator from {repo_id}...")
        self.snap = Path(snapshot_download(repo_id))
        import os, shutil  # _symlink_fix
        for _dp, _, _fns in os.walk(self.snap):
            for _fn in _fns:
                _fp = os.path.join(_dp, _fn)
                if os.path.islink(_fp):
                    _t = os.path.realpath(_fp)
                    os.remove(_fp)
                    shutil.copy2(_t, _fp)
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
