

# ==== audio backend helper (auto-added) ====
def _torch_ge_29():
    try:
        import torch
        v = torch.__version__.split("+")[0]
        major, minor = map(int, v.split(".")[:2])
        return (major, minor) >= (2, 9)
    except Exception:
        return False

def load_audio_segment(audio_path, offset=0, num_frames=None):
    if not _torch_ge_29():
        import torchaudio
        return torchaudio.load(
            audio_path,
            frame_offset=offset,
            num_frames=num_frames
        )

    import soundfile as sf
    import torch

    wav, sr = sf.read(audio_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    start = offset or 0
    end = None if num_frames is None else start + num_frames
    wav = wav[start:end]

    wav = torch.from_numpy(wav).unsqueeze(0)
    return wav, sr

"""Convert a T5Gemma-TTS checkpoint (.pth) to HF safetensors with trust_remote_code.

Usage:
  python z_scripts_new/export_t5gemma_voice_hf.py \
      --ckpt pretrained.pth \
      --out t5gemma_voice_hf \
      --base_repo google/t5gemma-2b-2b-ul2

Notes:
- The script copies the local modeling/config files under hf_export/ into the
  output directory and saves weights as safetensors (sharded if needed).
- tie_word_embeddings / tie_input_output_embeddings are forced to False when
  prune_text_modules>=1 to avoid HF tying errors (lm_head is pruned).
"""

import argparse
import os
import pathlib
import shutil
import sys

import torch
from transformers.models.t5gemma import T5GemmaConfig

# allow running from repo root
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from hf_export.configuration_t5gemma_voice import T5GemmaVoiceConfig
from hf_export.modeling_t5gemma_voice import (
    T5GemmaVoiceForConditionalGeneration,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="bundle_step64000.pth")
    p.add_argument("--out", default="t5gemma_voice_hf")
    p.add_argument("--base_repo", default=None, help="HF repo id or local dir for base T5Gemma; if None use ckpt args")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ns = ckpt["args"]

    base_repo = args.base_repo or getattr(ns, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2")
    base_cfg = T5GemmaConfig.from_pretrained(base_repo)
    # disable tie to avoid lm_head issues
    base_cfg.tie_word_embeddings = False
    base_cfg.tie_input_output_embeddings = False
    base_cfg.tie_encoder_decoder = False
    if hasattr(base_cfg, "encoder"):
        base_cfg.encoder.tie_word_embeddings = False
        base_cfg.encoder.tie_input_output_embeddings = False
        base_cfg.encoder.tie_encoder_decoder = False
    if hasattr(base_cfg, "decoder"):
        base_cfg.decoder.tie_word_embeddings = False
        base_cfg.decoder.tie_input_output_embeddings = False
        base_cfg.decoder.tie_encoder_decoder = False

    prune = getattr(ns, "prune_text_modules", 0)
    drop_lm_head = prune >= 1 or getattr(ns, "drop_lm_head", 0)
    drop_dec_embed = prune >= 2
    if drop_lm_head or drop_dec_embed:
        base_cfg.tie_word_embeddings = False
        base_cfg.tie_input_output_embeddings = False

    # derive sep tokens / vocab
    raw_vocab_size = getattr(ns, "audio_vocab_size", 65536)
    try:
        audio_vocab_size = int(raw_vocab_size)
        primary_vocab = audio_vocab_size
    except Exception:
        # allow list/tuple for multi-codebook configs
        audio_vocab_size = list(raw_vocab_size)
        primary_vocab = int(audio_vocab_size[0])
    y_sep_default = primary_vocab + 4
    x_sep_default = int(getattr(ns, "x_sep_token", 255999))

    voice_cfg = T5GemmaVoiceConfig(
        t5gemma_model_name=base_repo,
        t5_config_dict=base_cfg.to_dict(),
        attn_implementation=getattr(ns, "attn_implementation", "eager"),
        precision=getattr(ns, "precision", "float32"),
        prune_text_modules=prune,
        use_pm_rope=getattr(ns, "use_pm_rope", 1),
        tie_word_embeddings=False,
        tie_input_output_embeddings=False,
        n_codebooks=getattr(ns, "n_codebooks", 1),
        audio_vocab_size=audio_vocab_size,
        n_special=getattr(ns, "n_special", 5),
        empty_token=getattr(ns, "empty_token", primary_vocab),
        eog=getattr(ns, "eog", primary_vocab + 1),
        eos=getattr(ns, "eos", primary_vocab + 3),
        audio_pad_token=getattr(ns, "audio_pad_token", primary_vocab + 2),
        audio_mask_token=getattr(ns, "audio_mask_token", 1024),
        y_sep_token=getattr(ns, "y_sep_token", y_sep_default),
        x_sep_token=x_sep_default,
        special_first=getattr(ns, "special_first", 0),
        encodec_sr=getattr(ns, "encodec_sr", 50.0),
        progress_scale=getattr(ns, "progress_scale", 2000.0),
        progress_lookahead_secs=getattr(ns, "progress_lookahead_secs", 2.0),
        extra_cutoff=getattr(ns, "extra_cutoff", 5.0),
        text_guard_frames_per_token=getattr(ns, "text_guard_frames_per_token", 0),
        add_eos_to_text=getattr(ns, "add_eos_to_text", 0),
        add_bos_to_text=getattr(ns, "add_bos_to_text", 0),
        parallel_pattern=getattr(ns, "parallel_pattern", 0),
        audio_max_length=getattr(ns, "audio_max_length", 40.0),
        audio_tokenizer=getattr(ns, "audio_tokenizer", "xcodec2"),
        xcodec2_model_name=getattr(ns, "xcodec2_model_name", None),
        codec_audio_sr=getattr(ns, "codec_audio_sr", None),
        text_tokenizer_name=getattr(ns, "text_tokenizer_name", None),
    )

    model = T5GemmaVoiceForConditionalGeneration(voice_cfg)
    model.to(torch.bfloat16)

    state = ckpt["model"].copy()
    if prune >= 1:
        for k in list(state.keys()):
            if k.startswith("backbone.lm_head."):
                state.pop(k)
    if prune >= 2:
        for k in list(state.keys()):
            if k.startswith("backbone.model.decoder.embed_tokens."):
                state.pop(k)

    model.load_state_dict(state, strict=False)

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    # save model (safetensors, sharded if needed)
    model.save_pretrained(outdir, safe_serialization=True)
    voice_cfg.save_pretrained(outdir)

    # copy custom code for trust_remote_code
    shutil.copy("hf_export/modeling_t5gemma_voice.py", os.path.join(outdir, "modeling_t5gemma_voice.py"))
    shutil.copy("hf_export/configuration_t5gemma_voice.py", os.path.join(outdir, "configuration_t5gemma_voice.py"))

    print(f"Exported HF model to {outdir}")


if __name__ == "__main__":
    main()