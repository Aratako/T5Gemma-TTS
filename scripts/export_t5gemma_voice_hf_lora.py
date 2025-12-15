

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

"""
Export a LoRA-finetuned T5Gemma-TTS checkpoint to a Hugging Face format.

Features:
- Merges LoRA adapters into the base weights for a single standalone HF model.
- Optionally saves the LoRA adapter weights separately (PEFT format).
- Copies custom config/modeling files for trust_remote_code usage.

Usage:
  python z_scripts_new/export_t5gemma_voice_hf_lora.py \
    --ckpt lora.pth \
    --out t5gemma_voice_hf_lora_merged \
    --base_repo google/t5gemma-2b-2b-ul2 \
    --save_adapter_dir adapters/lora
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

from config import apply_repo_defaults
from models.t5gemma import T5GemmaVoiceModel
from hf_export.configuration_t5gemma_voice import T5GemmaVoiceConfig
from hf_export.modeling_t5gemma_voice import (
    T5GemmaVoiceForConditionalGeneration,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="LoRA fine-tuned bundle*.pth")
    p.add_argument("--out", default="t5gemma_voice_hf_lora_merged", help="HF output dir")
    p.add_argument("--base_repo", default=None, help="HF repo id or local dir for base T5Gemma (default: ckpt args.t5gemma_model_name)")
    p.add_argument("--save_adapter_dir", default=None, help="Optional directory to save LoRA adapter weights (PEFT)")
    return p.parse_args()


def _dtype_from_precision(precision: str):
    precision = str(precision).lower()
    if precision in ("float16", "fp16", "half"):
        return torch.float16
    if precision in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ns = ckpt["args"]
    # ensure LoRA flag on
    setattr(ns, "use_lora", 1)
    ns = apply_repo_defaults(ns)

    target_dtype = _dtype_from_precision(getattr(ns, "precision", "float32"))

    # build model and load weights (LoRA-enabled)
    model = T5GemmaVoiceModel(ns)
    model.carefully_load_state_dict(ckpt["model"])

    # optionally save adapter before merging
    if args.save_adapter_dir:
        os.makedirs(args.save_adapter_dir, exist_ok=True)
        model.backbone.save_pretrained(args.save_adapter_dir, safe_serialization=True)
        print(f"[Info] Saved LoRA adapter to {args.save_adapter_dir}")

    # merge LoRA into base weights
    model.backbone = model.backbone.merge_and_unload()

    # build HF config
    base_repo = args.base_repo or getattr(ns, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2")
    base_cfg = T5GemmaConfig.from_pretrained(base_repo)
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

    primary_vocab = int(getattr(ns, "audio_vocab_size", 65536))
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
        audio_vocab_size=primary_vocab,
        n_special=getattr(ns, "n_special", 5),
        empty_token=getattr(ns, "empty_token", primary_vocab),
        eog=getattr(ns, "eog", primary_vocab + 1),
        eos=getattr(ns, "eos", primary_vocab + 3),
        audio_pad_token=getattr(ns, "audio_pad_token", primary_vocab + 2),
        audio_mask_token=getattr(ns, "audio_mask_token", 1024),
        y_sep_token=getattr(ns, "y_sep_token", primary_vocab + 4),
        x_sep_token=getattr(ns, "x_sep_token", 255999),
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

    hf_model = T5GemmaVoiceForConditionalGeneration(voice_cfg)
    hf_model.to(dtype=target_dtype)

    state = model.state_dict()  # LoRA merged
    if drop_lm_head:
        for k in list(state.keys()):
            if k.startswith("backbone.lm_head."):
                state.pop(k)
    if drop_dec_embed:
        for k in list(state.keys()):
            if k.startswith("backbone.model.decoder.embed_tokens."):
                state.pop(k)

    hf_model.load_state_dict(state, strict=False)

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    hf_model.save_pretrained(outdir, safe_serialization=True)
    voice_cfg.save_pretrained(outdir)
    shutil.copy("hf_export/modeling_t5gemma_voice.py", os.path.join(outdir, "modeling_t5gemma_voice.py"))
    shutil.copy("hf_export/configuration_t5gemma_voice.py", os.path.join(outdir, "configuration_t5gemma_voice.py"))

    print(f"[Done] Exported merged LoRA HF model to {outdir}")


if __name__ == "__main__":
    main()