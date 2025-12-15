

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
Configuration for inference-only T5GemmaVoice model.

Kept intentionally minimal: only fields that affect inference-time shapes
or sampling behaviour are retained.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from transformers.configuration_utils import PretrainedConfig


class T5GemmaVoiceConfig(PretrainedConfig):
    model_type = "t5gemma_voice"
    is_encoder_decoder = True

    def __init__(
        self,
        # backbone
        t5gemma_model_name: str = "google/t5gemma-2b-2b-ul2",
        t5_config_dict: Optional[Dict[str, Any]] = None,
        attn_implementation: str = "eager",
        precision: str = "float32",
        prune_text_modules: int = 0,
        use_pm_rope: int = 1,
        tie_word_embeddings: Optional[bool] = None,
        tie_input_output_embeddings: Optional[bool] = None,
        n_codebooks: int = 1,
        audio_vocab_size: Union[int, List[int]] = 65536,
        n_special: int = 5,
        empty_token: int = 65536,
        eog: int = 65537,
        eos: int = 65539,
        audio_pad_token: int = 65538,
        audio_mask_token: int = 1024,
        y_sep_token: int = 65540,
        x_sep_token: int = 255999,
        special_first: int = 0,
        encodec_sr: float = 50.0,
        progress_scale: float = 2000.0,
        progress_lookahead_secs: float = 2.0,
        extra_cutoff: float = 5.0,
        text_guard_frames_per_token: int = 0,
        add_eos_to_text: int = 0,
        add_bos_to_text: int = 0,
        parallel_pattern: int = 0,
        audio_max_length: float = 40.0,
        audio_tokenizer: str = "xcodec2",
        xcodec2_model_name: Optional[str] = None,
        codec_audio_sr: Optional[float] = None,
        text_tokenizer_name: Optional[str] = None,
        # misc
        **kwargs,
    ) -> None:
        kwargs = dict(kwargs)
        # avoid duplicate values when loading from config.json that already stores these ids
        for _key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            kwargs.pop(_key, None)

        super().__init__(
            bos_token_id=empty_token,
            eos_token_id=eos,
            pad_token_id=audio_pad_token,
            **kwargs,
        )

        # store backbone config for offline instantiation
        self.t5_config_dict = t5_config_dict
        self.t5gemma_model_name = t5gemma_model_name
        self.attn_implementation = attn_implementation
        self.precision = precision
        self.prune_text_modules = prune_text_modules
        self.use_pm_rope = use_pm_rope
        self.tie_word_embeddings = tie_word_embeddings
        self.tie_input_output_embeddings = tie_input_output_embeddings

        self.text_input_type = "text"
        self.n_codebooks = n_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.n_special = n_special
        self.empty_token = empty_token
        self.eog = eog
        self.eos = eos
        self.audio_pad_token = audio_pad_token
        self.audio_mask_token = audio_mask_token
        self.y_sep_token = y_sep_token
        self.x_sep_token = x_sep_token
        self.special_first = special_first
        self.encodec_sr = encodec_sr
        self.progress_scale = progress_scale
        self.progress_lookahead_secs = progress_lookahead_secs
        self.extra_cutoff = extra_cutoff
        self.text_guard_frames_per_token = text_guard_frames_per_token
        self.add_eos_to_text = add_eos_to_text
        self.add_bos_to_text = add_bos_to_text
        self.parallel_pattern = parallel_pattern
        self.audio_max_length = audio_max_length
        self.audio_tokenizer = audio_tokenizer
        self.xcodec2_model_name = xcodec2_model_name
        self.codec_audio_sr = codec_audio_sr
        self.text_tokenizer_name = text_tokenizer_name

        # tell Auto* which files to load when trust_remote_code=True
        self.auto_map = {
            "AutoConfig": "configuration_t5gemma_voice.T5GemmaVoiceConfig",
            "AutoModelForSeq2SeqLM": "modeling_t5gemma_voice.T5GemmaVoiceForConditionalGeneration",
        }

    @property
    def audio_vocab_sizes(self) -> List[int]:
        """Utility to normalize audio_vocab_size to list form."""
        if isinstance(self.audio_vocab_size, list):
            return list(self.audio_vocab_size)
        return [int(self.audio_vocab_size)] * int(self.n_codebooks)


__all__ = ["T5GemmaVoiceConfig"]