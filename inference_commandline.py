import os
import torch
import numpy as np
import random
import fire
from argparse import Namespace

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

from models.t5gemma import T5GemmaVoiceModel
from inference_tts_utils import (
    inference_one_sample,
    normalize_text_with_lang,
    get_audio_info,
    get_sample_rate,
    save_audio,
    transcribe_audio,
)

############################################################
# Utility Functions
############################################################

def seed_everything(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

############################################################
# Main Inference Function
############################################################

def run_inference(
    reference_speech=None,
    target_text="I cannot believe that the same model can also do text to speech synthesis too! And you know what? this audio is 8 seconds long.",
    # Model
    model_name="bundle_step64000_infer",  # T5Gemma bundle filename without .pth
    model_root=".",
    # Additional optional
    reference_text=None,  # if None => run whisper on reference_speech
    target_duration=None, # if None => estimate from reference_speech and target_text
    # Default hyperparameters from snippet
    codec_audio_sr=16000, # do not change
    codec_sr=50, # do not change
    top_k=30, # try 10, 20, 30, 40
    top_p=0.9, # do not change
    min_p=0, # default: disabled
    temperature=0.8,
    silence_tokens=None, # do not change it
    multi_trial=None, # do not change it
    repeat_prompt=0, # increase this manually when you need stronger prompt repetition
    stop_repetition=3, # will not use it
    sample_batch_size=1, # do not change
    # Others
    seed=1,
    output_dir="./generated_tts",
    # Some snippet-based defaults
    cut_off_sec=100, # do not adjust this, we always use the entire reference speech. If you wish to change, also make sure to change the reference_transcript, so that it's only the trasnscript of the speech remained
    dump_tokens=False,
    lang=None,
):
    """
    Inference script using Fire.

    Example:
        python inference_commandline.py \
            --reference_speech "./demo/5895_34622_000026_000002.wav" \
            --target_text "I cannot believe ... this audio is 10 seconds long." \
            --reference_text "(optional) text to use as prefix" \
            --target_duration (optional float) 
    """

    # Seed everything
    seed_everything(seed)

    # Load model bundle and args
    torch.serialization.add_safe_globals([Namespace])
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    ckpt_fn = os.path.join(model_root, model_name + ".pth")
    if not os.path.exists(ckpt_fn):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_fn}. Please point model_root/model_name to a T5Gemma bundle.")
    bundle = torch.load(ckpt_fn, map_location=device, weights_only=True)
    args = bundle["args"]
    # Text-only mode
    if AutoTokenizer is None:
        raise ImportError("transformers is required for text tokenization. Please install it before running inference.")
    tokenizer_name = getattr(args, "text_tokenizer_name", None) or getattr(args, "t5gemma_model_name", "google/t5gemma-b-b-ul2")
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_arch = getattr(args, "model_arch", "t5gemma")
    if model_arch != "t5gemma":
        raise ValueError(f"VoiceStar support has been removed. Expected model_arch 't5gemma', got {model_arch}")
    model = T5GemmaVoiceModel(args)

    # Ensure checkpoint tensors and live model share the requested precision.
    precision_opt = str(getattr(args, "precision", "float32")).lower()
    _dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    target_dtype = _dtype_map.get(precision_opt, torch.float32)

    state_dict = bundle["model"]
    if target_dtype != torch.float32:
        # Cast floating tensors only; keep ints (e.g., embeddings indices) as-is.
        state_dict = {
            k: (v.to(target_dtype) if torch.is_floating_point(v) else v)
            for k, v in state_dict.items()
        }

    strict_load = not getattr(args, "use_lora", 0)
    model.load_state_dict(state_dict, strict=strict_load)
    model.to(device=device, dtype=target_dtype)
    model.eval()
    del bundle
    torch.cuda.empty_cache()

    # If reference_text not provided, use whisper large-v3-turbo
    no_reference_audio = str(reference_speech).lower() in {"none", "", "null"} or reference_speech is None
    has_reference_text = not (
        reference_text is None or str(reference_text).strip().lower() in {"", "none", "null"}
    )

    # Disallow providing reference_text without accompanying reference_speech
    if no_reference_audio and has_reference_text:
        raise ValueError(
            "reference_text was provided but reference_speech is missing. "
            "Please supply a reference_speech or omit reference_text."
        )

    if no_reference_audio:
        prefix_transcript = ""
    elif not has_reference_text:
        print("[Info] No reference_text provided, transcribing reference_speech with Whisper.")
        prefix_transcript = transcribe_audio(reference_speech, device)
        print(f"[Info] Whisper transcribed text: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    # Language + normalization (Japanese only)
    lang = None if lang in {None, "", "none", "null"} else str(lang)
    target_text, lang_code = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang_code)

    # If target_duration not provided, estimate from phonemes + optional reference audio
    if target_duration is None:
        target_generation_length = estimate_duration(
            target_text=target_text,
            reference_speech=None if no_reference_audio else reference_speech,
            reference_transcript=None if no_reference_audio else prefix_transcript,
            target_lang=lang_code,
            reference_lang=lang_code,
        )
        print(
            f"[Info] target_duration not provided, estimated as {target_generation_length:.2f} seconds. "
            "If not desired, please provide a target_duration."
        )
    else:
        target_generation_length = float(target_duration)

    # xcodec2 is the only supported backend
    signature = None
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(args, "xcodec2_model_name", None),
    )

    if silence_tokens is None:
        # default from snippet
        silence_tokens = []

    if multi_trial is None:
        # default from snippet
        multi_trial = []

    codec_audio_sr = getattr(args, "codec_audio_sr", codec_audio_sr)
    codec_sr = getattr(args, "encodec_sr", codec_sr)
    codec_audio_sr = audio_tokenizer.sample_rate

    # We can compute prompt_end_frame if we want, from snippet
    if not no_reference_audio:
        info = get_audio_info(reference_speech)
        prompt_end_frame = int(cut_off_sec * get_sample_rate(info))
    else:
        prompt_end_frame = 0

    # Prepare tokenizers

    # decode_config from snippet
    decode_config = {
        'top_k': top_k,
        'top_p': top_p,
        'min_p': min_p,
        'temperature': temperature,
        'stop_repetition': stop_repetition,
        'codec_audio_sr': codec_audio_sr,
        'codec_sr': codec_sr,
        'silence_tokens': silence_tokens,
        'sample_batch_size': sample_batch_size
    }

    # Run inference
    print("[Info] Running TTS inference...")
    inference_kwargs = dict(
        model=model,
        model_args=args,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        audio_fn=None if no_reference_audio else reference_speech,
        target_text=target_text,
        lang=lang_code,
        device=device,
        decode_config=decode_config,
        prompt_end_frame=prompt_end_frame,
        target_generation_length=target_generation_length,
        prefix_transcript=prefix_transcript,
        multi_trial=multi_trial,
        repeat_prompt=repeat_prompt,
        return_frames=dump_tokens,
    )

    if dump_tokens:
        concated_audio, gen_audio, concat_frames, gen_frames = inference_one_sample(**inference_kwargs)
    else:
        concated_audio, gen_audio = inference_one_sample(**inference_kwargs)

    # The model returns a list of waveforms, pick the first
    concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()

    # Save the audio (just the generated portion, as the snippet does)
    os.makedirs(output_dir, exist_ok=True)
    out_filename = "generated.wav"
    out_path = os.path.join(output_dir, out_filename)
    save_audio(out_path, gen_audio, codec_audio_sr)
    max_abs = torch.max(gen_audio.abs()).item()
    rms = torch.sqrt((gen_audio ** 2).mean()).item()
    print(f"[Info] Generated audio stats -> max_abs: {max_abs:.6f}, rms: {rms:.6f}")

    if dump_tokens:
        np.save(os.path.join(output_dir, "generated_frames.npy"), gen_frames.squeeze(0).cpu().numpy())
        np.save(os.path.join(output_dir, "concat_frames.npy"), concat_frames.squeeze(0).cpu().numpy())
        print(f"[Info] Saved token arrays to {output_dir}")

    print(f"[Success] Generated audio saved to {out_path}")


def main():
    fire.Fire(run_inference)

if __name__ == "__main__":
    main()
