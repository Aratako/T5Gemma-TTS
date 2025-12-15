

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
Gradio demo for HF-format T5GemmaVoice checkpoints.

Usage:
    python inference_gradio.py --model_dir ./t5gemma_voice_hf --port 7860
"""

import argparse
import os
import random
from functools import lru_cache
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torchaudio
import whisper

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import inference_one_sample, normalize_text_with_lang

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: Optional[int]) -> int:
    """
    Seed all RNGs. If seed is None, draw a fresh random seed and return it.
    """
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
        print(f"[Info] No seed provided; using random seed {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


# ---------------------------------------------------------------------------
# Model / tokenizer loaders (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_whisper_model():
    """Cache Whisper model to avoid reloading on every inference."""
    print("[Info] Loading Whisper model (large-v3-turbo)...")
    return whisper.load_model("large-v3-turbo")


@lru_cache(maxsize=1)
def _load_resources(model_dir: str, xcodec2_model_name: Optional[str], xcodec2_sample_rate: Optional[int], use_torch_compile: bool):
    """Load model and tokenizers from HF-format directory or repo."""
    if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
        raise ImportError("Please install transformers before running the demo.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    cfg = model.config

    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    xcodec2_model_name = xcodec2_model_name or getattr(cfg, "xcodec2_model_name", None)

    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=xcodec2_model_name,
        sample_rate=xcodec2_sample_rate,
    )

    # Apply torch.compile for faster inference (2nd run onwards)
    if use_torch_compile and torch.cuda.is_available():
        print("[Info] Applying torch.compile to model and codec (this may take a minute on first inference)...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            audio_tokenizer.codec = torch.compile(audio_tokenizer.codec, mode="reduce-overhead")
            print("[Info] torch.compile applied successfully.")
        except Exception as e:
            print(f"[Warning] torch.compile failed, falling back to eager mode: {e}")

    codec_audio_sr = getattr(cfg, "codec_audio_sr", audio_tokenizer.sample_rate)
    if xcodec2_sample_rate is not None:
        codec_audio_sr = xcodec2_sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)

    return {
        "model": model,
        "cfg": cfg,
        "text_tokenizer": text_tokenizer,
        "audio_tokenizer": audio_tokenizer,
        "device": device,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    reference_speech: Optional[str],
    reference_text: Optional[str],
    target_text: str,
    target_duration: Optional[float],
    top_k: int,
    top_p: float,
    min_p: float,
    temperature: float,
    seed: Optional[int],
    resources: dict,
    cut_off_sec: int = 100,
    lang: Optional[str] = None,
) -> Tuple[int, np.ndarray]:
    """
    Run TTS and return a (sample_rate, waveform) tuple for Gradio playback.
    """
    used_seed = seed_everything(None if seed is None else int(seed))

    model = resources["model"]
    cfg = resources["cfg"]
    text_tokenizer = resources["text_tokenizer"]
    audio_tokenizer = resources["audio_tokenizer"]
    device = resources["device"]
    codec_audio_sr = resources["codec_audio_sr"]
    codec_sr = resources["codec_sr"]

    silence_tokens = []
    repeat_prompt = 0  # fixed; UI removed
    stop_repetition = 3  # keep sensible default from CLI HF script
    sample_batch_size = 1

    no_reference_audio = reference_speech is None or str(reference_speech).strip().lower() in {"", "none", "null"}
    has_reference_text = reference_text not in (None, "", "none", "null")

    if no_reference_audio and has_reference_text:
        raise ValueError("reference_text was provided but reference_speech is missing.")

    if no_reference_audio:
        prefix_transcript = ""
    elif not has_reference_text:
        print("[Info] No reference text; transcribing reference speech with Whisper (large-v3-turbo).")
        wh_model = _get_whisper_model()
        result = wh_model.transcribe(reference_speech)
        prefix_transcript = result["text"]
        print(f"[Info] Whisper transcription: {prefix_transcript}")
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
        "top_k": int(top_k),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "temperature": float(temperature),
        "stop_repetition": stop_repetition,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "silence_tokens": silence_tokens,
        "sample_batch_size": sample_batch_size,
    }

    concat_audio, gen_audio = inference_one_sample(
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
        multi_trial=[],
        repeat_prompt=repeat_prompt,
        return_frames=False,
    )

    # Take generated audio, move to CPU, convert to numpy for Gradio
    gen_audio = gen_audio[0].detach().cpu()
    if gen_audio.ndim == 2 and gen_audio.shape[0] == 1:
        gen_audio = gen_audio.squeeze(0)
    waveform = gen_audio.numpy()

    max_abs = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(waveform**2)))
    print(f"[Info] Generated audio stats -> max_abs: {max_abs:.6f}, rms: {rms:.6f}")
    print(f"[Info] Seed used for this run: {used_seed}")

    return codec_audio_sr, waveform


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo(resources, server_port: int, share: bool):
    description = (
        "Reference speech is optional. If provided without reference text, Whisper (large-v3-turbo) "
        "will auto-transcribe and use it as the prompt text."
    )

    with gr.Blocks() as demo:
        gr.Markdown("## T5Gemma-TTS (HF)")
        gr.Markdown(description)

        with gr.Row():
            reference_speech_input = gr.Audio(
                label="Reference Speech (optional)",
                type="filepath",
            )
            reference_text_box = gr.Textbox(
                label="Reference Text (optional, leave blank to auto-transcribe)",
                lines=2,
            )

        target_text_box = gr.Textbox(
            label="Target Text",
            value="こんにちは、私はAIです。これは音声合成のテストです。",
            lines=3,
        )

        with gr.Row():
            target_duration_box = gr.Textbox(
                label="Target Duration (seconds, optional)",
                value="",
                placeholder="Leave blank for auto estimate",
            )
            seed_box = gr.Textbox(
                label="Random Seed (optional)",
                value="",
                placeholder="Leave blank to use a random seed each run",
            )

        with gr.Row():
            top_k_box = gr.Slider(label="top_k", minimum=0, maximum=100, step=1, value=30)
            top_p_box = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, step=0.05, value=0.9)
            min_p_box = gr.Slider(label="min_p (0 = disabled)", minimum=0.0, maximum=1.0, step=0.05, value=0.0)
            temperature_box = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, step=0.05, value=0.8)

        generate_button = gr.Button("Generate")
        output_audio = gr.Audio(label="Generated Audio", type="numpy", interactive=False)

        def gradio_inference(
            reference_speech,
            reference_text,
            target_text,
            target_duration,
            top_k,
            top_p,
            min_p,
            temperature,
            seed,
        ):
            dur = float(target_duration) if str(target_duration).strip() not in {"", "None", "none"} else None
            seed_val = None
            if str(seed).strip() not in {"", "None", "none"}:
                seed_val = int(float(seed))
            sr, wav = run_inference(
                reference_speech=reference_speech,
                reference_text=reference_text,
                target_text=target_text,
                target_duration=dur,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
                seed=seed_val,
                resources=resources,
            )
            return (sr, wav)

        generate_button.click(
            fn=gradio_inference,
            inputs=[
                reference_speech_input,
                reference_text_box,
                target_text_box,
                target_duration_box,
                top_k_box,
                top_p_box,
                min_p_box,
                temperature_box,
                seed_box,
            ],
            outputs=[output_audio],
        )

    demo.launch(server_name="0.0.0.0", server_port=server_port, share=share, debug=True)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gradio demo for HF T5GemmaVoice")
    parser.add_argument("--model_dir", type=str, default="./t5gemma_voice_hf", help="HF model directory or repo id")
    parser.add_argument("--xcodec2_model_name", type=str, default=None, help="Override xcodec2 model name from config")
    parser.add_argument("--xcodec2_sample_rate", type=int, default=None, help="Override xcodec2 sample rate from config")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile (faster startup, slower inference)")
    args = parser.parse_args()

    resources = _load_resources(args.model_dir, args.xcodec2_model_name, args.xcodec2_sample_rate, use_torch_compile=not args.no_compile)
    build_demo(resources=resources, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()