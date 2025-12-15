

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

"""TTS inference script for HF safetensors.

Thin wrapper to load a T5GemmaVoiceForConditionalGeneration checkpoint saved in
Hugging Face format without touching the original inference_commandline.py.
"""

import os
import random

import fire
import numpy as np
import torch
import torchaudio
import whisper

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import inference_one_sample, normalize_text_with_lang

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


def seed_everything(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_inference(
    reference_speech=None,
    target_text="こんにちは、私はAIです。これは音声合成のテストです。",
    model_dir="./t5gemma_voice_hf",
    reference_text=None,
    target_duration=None,
    codec_audio_sr=16000,
    codec_sr=50,
    top_k=30,
    top_p=0.9,
    min_p=0,  # default: disabled
    temperature=0.8,
    silence_tokens=None,
    multi_trial=None,
    repeat_prompt=0,
    stop_repetition=3,
    sample_batch_size=1,
    seed=1,
    output_dir="./generated_tts",
    cut_off_sec=100,
    dump_tokens=False,
    lang=None,
):
    seed_everything(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if AutoModelForSeq2SeqLM is None:
        raise ImportError("transformers is not installed")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    cfg = model.config

    if AutoTokenizer is None:
        raise ImportError("transformers is not installed")
    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Audio tokenizer (supports xcodec2 only)
    signature = None
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(cfg, "xcodec2_model_name", None),
    )
    codec_audio_sr = getattr(cfg, "codec_audio_sr", codec_audio_sr)
    codec_sr = getattr(cfg, "encodec_sr", codec_sr)
    # Align the actual sample rate with the tokenizer (e.g., 44.1 kHz)
    codec_audio_sr = audio_tokenizer.sample_rate

    if silence_tokens is None:
        silence_tokens = []
    if isinstance(silence_tokens, str):
        silence_tokens = eval(silence_tokens)

    multi_trial = multi_trial or []

    no_reference_audio = str(reference_speech).lower() in {"none", "", "null"} or reference_speech is None
    has_reference_text = not (
        reference_text is None or str(reference_text).strip().lower() in {"", "none", "null"}
    )

    if no_reference_audio and has_reference_text:
        raise ValueError(
            "reference_text was provided but reference_speech is missing. "
            "Please supply a reference_speech or omit reference_text."
        )

    if no_reference_audio:
        prefix_transcript = ""
    elif not has_reference_text:
        wh_model = whisper.load_model("large-v3-turbo")
        result = wh_model.transcribe(reference_speech)
        prefix_transcript = result["text"]
        print(f"[Info] Whisper transcribed text: {prefix_transcript}")
    else:
        prefix_transcript = reference_text

    # Language + normalization (Japanese only)
    lang = None if lang in {None, "", "none", "null"} else str(lang)
    target_text, lang_code = normalize_text_with_lang(target_text, lang)
    if prefix_transcript:
        prefix_transcript, _ = normalize_text_with_lang(prefix_transcript, lang_code)

    if target_duration is None:
        target_generation_length = estimate_duration(
            target_text=target_text,
            reference_speech=None if no_reference_audio else reference_speech,
            reference_transcript=None if no_reference_audio else prefix_transcript,
            target_lang=lang_code,
            reference_lang=lang_code,
        )
        print(f"[Info] target_duration not provided, estimated as {target_generation_length:.2f} seconds.")
    else:
        target_generation_length = float(target_duration)

    if not no_reference_audio:
        if _torch_ge_29():
            import soundfile as sf
            info = sf.info(reference_speech)
            sr = info.samplerate
        else:
            import torchaudio
            info = torchaudio.info(reference_speech)
            sr = info.sample_rate
        prompt_end_frame = int(cut_off_sec * sr)
    else:
        prompt_end_frame = 0

    decode_config = {
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "temperature": temperature,
        "stop_repetition": stop_repetition,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "silence_tokens": silence_tokens,
        "sample_batch_size": sample_batch_size,
    }

    res = inference_one_sample(
        model=model,
        model_args=cfg,
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
        concat_audio, gen_audio, concat_frames, gen_frames = res
    else:
        concat_audio, gen_audio = res

    concat_audio, gen_audio = concat_audio[0].cpu(), gen_audio[0].cpu()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated.wav")
    if _torch_ge_29():
        import soundfile as sf
        sf.write(out_path,
                 gen_audio.squeeze().detach().cpu().numpy(),
                 codec_audio_sr)
    else:
        import torchaudio
        torchaudio.save(out_path, gen_audio, codec_audio_sr)

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