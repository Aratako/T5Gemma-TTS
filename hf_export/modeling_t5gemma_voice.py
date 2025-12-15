

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
Hugging Face compatible wrapper of the T5Gemma-TTS model.

This is largely a drop-in copy of `models/t5gemma.py`, but inherits
`PreTrainedModel` so that it can be loaded via `AutoModelForSeq2SeqLM` with
`trust_remote_code=True`. Only the inference-oriented pieces are kept intact.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.t5gemma.modeling_t5gemma import (
    ALL_ATTENTION_FUNCTIONS,
    EncoderDecoderCache,
    T5GemmaCrossAttention,
    T5GemmaDecoderLayer,
    T5GemmaRotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)

try:
    from .configuration_t5gemma_voice import T5GemmaVoiceConfig
except ImportError:  # when executed inside the repo package
    from hf_export.configuration_t5gemma_voice import T5GemmaVoiceConfig

logger = logging.getLogger(__name__)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Return Bool mask [B, T] where True indicates padding."""
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    return expanded_lengths >= lengths.unsqueeze(-1)


def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
    min_p=0.0,
    filter_value=-float("Inf"),
    min_tokens_to_keep=1,
):
    min_p_enabled = 0.0 < min_p < 1.0
    if min_p_enabled:
        probs = F.softmax(logits, dim=-1)
        indices_to_remove = probs < min_p
        if torch.all(indices_to_remove.sum(-1) < logits.size(-1)):
            logits = logits.masked_fill(indices_to_remove, filter_value)
            top_k = 0
            top_p = 1.0

    if isinstance(top_k, int) and top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value
    elif isinstance(top_k, list):
        assert len(top_k) == logits.size(
            0
        ), f"top_k list length ({len(top_k)}) must match logits.size(0) ({logits.size(0)})"
        for i in range(logits.size(0)):
            k_i = top_k[i]
            if k_i > 0:
                k_i = min(max(k_i, min_tokens_to_keep), logits.size(-1))
                row_threshold = torch.topk(logits[i], k_i, dim=-1)[0][-1]
                indices_to_remove_i = logits[i] < row_threshold
                logits[i, indices_to_remove_i] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, min_p=0.0, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, min_p=min_p)
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


class PMCrossAttention(T5GemmaCrossAttention):
    """T5Gemma cross-attention augmented with Progress-Monitoring RoPE."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.decoder_rotary_emb = T5GemmaRotaryEmbedding(config=config)
        self.encoder_rotary_emb = T5GemmaRotaryEmbedding(config=config)

    @staticmethod
    def _apply_rotary_with_progress(
        projected_states: torch.Tensor,
        base_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        rotary_module: T5GemmaRotaryEmbedding,
    ) -> torch.Tensor:
        if position_ids is None:
            return projected_states
        cos, sin = rotary_module(base_states, position_ids)
        cos = cos.unsqueeze(1).to(
            dtype=projected_states.dtype, device=projected_states.device
        )
        sin = sin.unsqueeze(1).to(
            dtype=projected_states.dtype, device=projected_states.device
        )
        return (projected_states * cos) + (rotate_half(projected_states) * sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        pm_decoder_position_ids: Optional[torch.Tensor] = None,
        pm_encoder_position_ids: Optional[torch.Tensor] = None,
        **kwargs: FlashAttentionKwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if encoder_hidden_states is None:
            raise ValueError("Encoder hidden state is required for cross attention.")

        pm_decoder_position_ids = kwargs.pop(
            "pm_decoder_position_ids", pm_decoder_position_ids
        )
        pm_encoder_position_ids = kwargs.pop(
            "pm_encoder_position_ids", pm_encoder_position_ids
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if pm_decoder_position_ids is not None:
            query_states = self._apply_rotary_with_progress(
                query_states,
                hidden_states,
                pm_decoder_position_ids,
                self.decoder_rotary_emb,
            )

        if past_key_values is not None:
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            curr_past_key_values = past_key_values.cross_attention_cache

        if past_key_values is None or not is_updated:
            encoder_input_shape = encoder_hidden_states.shape[:-1]
            encoder_hidden_shape = (*encoder_input_shape, -1, self.head_dim)
            key_states = (
                self.k_proj(encoder_hidden_states)
                .view(encoder_hidden_shape)
                .transpose(1, 2)
            )
            if pm_encoder_position_ids is not None:
                key_states = self._apply_rotary_with_progress(
                    key_states,
                    encoder_hidden_states,
                    pm_encoder_position_ids,
                    self.encoder_rotary_emb,
                )
            value_states = (
                self.v_proj(encoder_hidden_states)
                .view(encoder_hidden_shape)
                .transpose(1, 2)
            )

            if past_key_values is not None:
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx
                )
                past_key_values.is_updated[self.layer_idx] = True
        else:
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=None,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PMDecoderLayer(T5GemmaDecoderLayer):
    """Decoder layer variant with PM-RoPE cross-attention built in."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.cross_attn = PMCrossAttention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        pm_decoder_position_ids: Optional[torch.Tensor] = None,
        pm_encoder_position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        pm_decoder_position_ids = kwargs.pop(
            "pm_decoder_position_ids", pm_decoder_position_ids
        )
        pm_encoder_position_ids = kwargs.pop(
            "pm_encoder_position_ids", pm_encoder_position_ids
        )

        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=(
                past_key_values.self_attention_cache
                if past_key_values is not None
                else None
            ),
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_cross_attn_layernorm(hidden_states)
        hidden_states, _ = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            pm_decoder_position_ids=pm_decoder_position_ids,
            pm_encoder_position_ids=pm_encoder_position_ids,
            **kwargs,
        )
        hidden_states = self.post_cross_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)
        return hidden_states


def _make_args_from_config(config: T5GemmaVoiceConfig):
    """Namespace-like shim; keeps attribute access identical to training code."""

    class _Obj:
        pass

    o = _Obj()
    for k, v in config.to_dict().items():
        setattr(o, k, v)
    return o


class T5GemmaVoiceForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = T5GemmaVoiceConfig
    base_model_prefix = "backbone"
    _keys_to_ignore_on_save = ["encoder_module", "decoder_module"]

    def __init__(self, config: T5GemmaVoiceConfig):
        super().__init__(config)
        # keep compatibility with original code that expects self.args
        self.args = _make_args_from_config(config)
        if getattr(self.args, "n_codebooks", 1) != 1:
            logging.info("Resetting n_codebooks to 1 for XCodec2 backend.")
            self.args.n_codebooks = 1

        logging.info(f"Loading T5Gemma backbone: {self.args.t5gemma_model_name}")
        precision = getattr(self.args, "precision", "float32")
        if precision == "float16":
            dtype = torch.float16
        elif precision == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # instantiate backbone from config to avoid weight download during load
        if config.t5_config_dict is not None:
            from transformers.models.t5gemma import T5GemmaConfig

            base_cfg = T5GemmaConfig(**config.t5_config_dict)
            base_cfg._attn_implementation = getattr(
                self.args, "attn_implementation", "eager"
            )
            # Force bf16/specified dtype initialization and disable all tying.
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
            self.backbone = AutoModelForSeq2SeqLM.from_config(
                base_cfg, torch_dtype=dtype
            )
        else:
            self.backbone = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.t5gemma_model_name,
                attn_implementation=getattr(self.args, "attn_implementation", "eager"),
                torch_dtype=dtype,
            )

        prune_text_modules = getattr(self.args, "prune_text_modules", 0)
        drop_lm_head = prune_text_modules >= 1
        drop_decoder_embed = prune_text_modules >= 2

        if drop_lm_head and hasattr(self.backbone, "lm_head"):
            del self.backbone.lm_head
            self.backbone.lm_head = nn.Identity()
            if hasattr(self.backbone.config, "tie_word_embeddings"):
                self.backbone.config.tie_word_embeddings = False
            logging.info("lm_head removed (prune_text_modules=%d)", prune_text_modules)

        if drop_decoder_embed:
            decoder = getattr(
                self.backbone, "model", getattr(self.backbone, "decoder", None)
            )
            decoder = getattr(decoder, "decoder", decoder)
            if decoder is not None and hasattr(decoder, "embed_tokens"):
                del decoder.embed_tokens
                decoder.embed_tokens = nn.Identity()
                if hasattr(self.backbone.config, "tie_word_embeddings"):
                    self.backbone.config.tie_word_embeddings = False
                logging.info(
                    "decoder.embed_tokens removed (prune_text_modules=%d)",
                    prune_text_modules,
                )

        # This wrapper is inference-only, so keep cache enabled.
        self.backbone.config.use_cache = True

        if hasattr(self.backbone, "model"):
            self.encoder_module = self.backbone.model.encoder
            self.decoder_module = self.backbone.model.decoder
        else:
            self.encoder_module = getattr(self.backbone, "encoder", None)
            self.decoder_module = getattr(self.backbone, "decoder", None)
        if self.encoder_module is None or self.decoder_module is None:
            raise AttributeError(
                "Failed to locate encoder/decoder modules on T5Gemma backbone."
            )

        config_hidden_size = getattr(self.backbone.config, "d_model", None)
        if config_hidden_size is None:
            config_hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if config_hidden_size is None:
            enc = getattr(self.backbone.config, "encoder", None)
            if enc is not None:
                config_hidden_size = getattr(enc, "hidden_size", None)
        if config_hidden_size is None:
            raise AttributeError("T5Gemma config does not expose d_model/hidden_size.")

        self.hidden_size = config_hidden_size
        self.args.audio_embedding_dim = getattr(
            self.args, "audio_embedding_dim", self.hidden_size
        )

        self._enable_pm_rope_cross_attention()

        self.text_input_type = "text"  # fixed
        self.text_embedding = None
        self.text_dropout = nn.Identity()

        if isinstance(self.args.audio_vocab_size, list):
            audio_vocab_sizes = [
                size + self.args.n_special for size in self.args.audio_vocab_size
            ]
        else:
            audio_vocab_sizes = [
                self.args.audio_vocab_size + self.args.n_special
            ] * self.args.n_codebooks
        self.n_audio_tokens = audio_vocab_sizes

        self.audio_embedding = nn.ModuleList(
            [
                nn.Embedding(audio_vocab_sizes[k], self.hidden_size)
                for k in range(self.args.n_codebooks)
            ]
        )
        self.audio_dropout = nn.Dropout(0.0)

        self.predict_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hidden_size, audio_vocab_sizes[k]),
                )
                for k in range(self.args.n_codebooks)
            ]
        )
        self.progress_scale = getattr(self.args, "progress_scale", 2000.0)

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError("Output embeddings are pruned in this model.")

    # avoid transformers default tying logic (lm_head is removed)
    def tie_weights(self):
        return

    def get_encoder(self):
        return self.encoder_module

    def get_decoder(self):
        return self.decoder_module

    def state_dict(self, *args, **kwargs):  # pragma: no cover - save hook
        sd = super().state_dict(*args, **kwargs)
        drop_keys = [
            k
            for k in sd
            if k.startswith("encoder_module.") or k.startswith("decoder_module.")
        ]
        for k in drop_keys:
            sd.pop(k)
        return sd

    def _progress_positions_single(self, length: int, device) -> torch.Tensor:
        if length <= 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        if length == 1:
            return torch.zeros(1, device=device, dtype=torch.float32)
        base = torch.arange(length, device=device, dtype=torch.float32)
        return base / (length - 1) * self.progress_scale

    def _build_position_ids(
        self, lengths: torch.Tensor, max_len: int, device
    ) -> torch.Tensor:
        # Vectorized implementation: avoid Python loop over batch dimension.
        # Ensure lengths is on the correct device to prevent device mismatch.
        lengths = lengths.to(device=device)
        pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]  # [1, T]

        # Clamp denominator to avoid division by zero for length <= 1.
        # For length 0 or 1, result will be masked to zero anyway.
        denom = (lengths.clamp(min=2).to(torch.float32) - 1.0)[:, None]  # [B, 1]
        position_ids = pos / denom * self.progress_scale  # [B, T]

        # Mask out positions beyond each sequence's length.
        mask = pos < lengths[:, None]  # [B, T] (bool)
        return position_ids.masked_fill(~mask, 0.0)

    def _enable_pm_rope_cross_attention(self) -> None:
        if getattr(self, "_pm_rope_enabled", False):
            return
        if not getattr(self.args, "use_pm_rope", 1):
            logging.info("PM-RoPE cross-attention disabled by config.")
            return
        decoder_layers = getattr(self.decoder_module, "layers", None)
        if decoder_layers is None:
            logging.warning(
                "Decoder module does not expose layers attribute; skipping PM-RoPE injection."
            )
            return

        new_layers = nn.ModuleList()
        for layer in decoder_layers:
            pm_layer = PMDecoderLayer(layer.config, layer.layer_idx)
            pm_layer.load_state_dict(layer.state_dict(), strict=False)
            pm_layer.gradient_checkpointing = getattr(
                layer, "gradient_checkpointing", False
            )
            if hasattr(layer, "_gradient_checkpointing_func"):
                pm_layer._gradient_checkpointing_func = (
                    layer._gradient_checkpointing_func
                )
            new_layers.append(pm_layer)
        self.decoder_module.layers = new_layers
        self._pm_rope_enabled = True
        logging.info(
            "PM-RoPE cross-attention enabled for %d decoder layers.", len(new_layers)
        )

    # Generation-style inference (identical to original implementation)
    @torch.inference_mode()
    def inference_tts(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        tgt_y_lens: torch.Tensor,
        top_k: Union[int, List[int]] = -100,
        top_p: float = 1.0,
        min_p: float = 0.0,
        temperature: float = 1.0,
        stop_repetition: int = 3,
        silence_tokens: List[int] = None,
        multi_trial: List[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.args, "n_codebooks", 1) != 1:
            raise ValueError("XCodec2 inference expects n_codebooks=1.")

        self.backbone.config.use_cache = True
        if multi_trial:
            logging.warning("multi_trial is unsupported and will be ignored.")
        silence_tokens = silence_tokens or []

        device = x.device
        eog_inference = (
            self.args.eos if getattr(self.args, "eos", -1) > 0 else self.args.eog
        )
        batch_size = x.shape[0]
        assert batch_size == 1, "Current implementation only supports batch size 1."

        x_padding_mask = make_pad_mask(x_lens).to(device)
        encoder_attention_mask = (~x_padding_mask).long()
        if getattr(self.args, "use_pm_rope", 1):
            encoder_position_ids = self._build_position_ids(x_lens, x.shape[1], device)
        else:
            encoder_position_ids = None
        if self.text_input_type == "text":
            encoder_outputs = self.encoder_module(
                input_ids=x,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
            )
        else:
            x_embeds = self.text_dropout(self.text_embedding(x))
            encoder_outputs = self.encoder_module(
                inputs_embeds=x_embeds,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
            )
        memory = encoder_outputs.last_hidden_state

        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2, 1).contiguous()  # [B, 1, T]
        y_len = y.shape[-1]
        prompt_frames = kwargs.get("prompt_frames", y_len)
        target_total = None
        cutoff_limit = None
        if tgt_y_lens is not None:
            target_total = int(tgt_y_lens[0].item())
            extra_cutoff = getattr(self.args, "extra_cutoff", 5.0)
            codec_sr = int(getattr(self.args, "encodec_sr", 50))
            cutoff_limit = target_total + int(codec_sr * extra_cutoff)

        bos = torch.full(
            (batch_size, 1, 1),
            self.args.empty_token,
            dtype=torch.long,
            device=device,
        )
        cated_y = torch.cat([bos, y], dim=2)

        new_y_len_value = cated_y.shape[-1]
        new_y_lens = torch.full(
            (batch_size,), new_y_len_value, dtype=torch.long, device=device
        )
        embedded_y = self.audio_embedding[0](cated_y[:, 0])
        embedded_y = self.audio_dropout(embedded_y)

        y_padding_mask = torch.full(
            (batch_size, embedded_y.shape[1]), False, device=device
        )
        current_length = embedded_y.shape[1]
        prompt_offset = prompt_frames + 1  # +BOS

        decoder_attention_mask = (~y_padding_mask).long()

        if target_total is not None:
            est_total = int(target_total) + 1
        elif cutoff_limit is not None:
            est_total = int(cutoff_limit)
        else:
            lookahead = getattr(self.args, "progress_lookahead_secs", 2.0)
            est_total = int(current_length + int(self.args.encodec_sr) * lookahead)
        est_total = max(est_total, current_length)

        # Pre-allocate attention mask buffer to avoid per-step tensor creation.
        # Use est_total + safety margin to avoid reallocation.
        max_gen_length = est_total + int(getattr(self.args, "encodec_sr", 50) * 10)
        full_dec_attention_mask = torch.ones(
            (batch_size, max_gen_length), dtype=torch.long, device=device
        )

        cur_len = embedded_y.shape[1]
        pm_kwargs = {}
        decoder_position_ids_full = None
        if getattr(self.args, "use_pm_rope", 1):
            base = torch.arange(cur_len, device=device, dtype=torch.float32).unsqueeze(
                0
            )
            decoder_position_ids_full = (
                base / max(1, est_total - 1) * self.progress_scale
            )
            pm_kwargs["position_ids"] = decoder_position_ids_full
            pm_kwargs["pm_decoder_position_ids"] = decoder_position_ids_full
            pm_kwargs["pm_encoder_position_ids"] = encoder_position_ids
        else:
            pm_kwargs["position_ids"] = None

        decoder_outputs = self.decoder_module(
            inputs_embeds=embedded_y,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            **pm_kwargs,
        )
        last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
        past_key_values = decoder_outputs.past_key_values

        generated_tokens: List[torch.Tensor] = []
        cur_num_gen = 0
        prev_token = -1
        consec_silence_count = 0
        silence_set = set(silence_tokens)

        def sample_helper(
            logits,
            top_k,
            top_p,
            min_p,
            temperature,
            prev_token,
            consec_silence_count,
            stop_repetition,
            silence_tokens,
            cur_num_gen,
            current_length,
            target_total,
            prompt_offset,
        ):
            effective_length = max(0, current_length - prompt_offset)
            logits_adjust = logits
            if effective_length == 0:
                logits_adjust[eog_inference] = -1e9

            if isinstance(top_k, list):
                kk = top_k[min(len(top_k) - 1, cur_num_gen)]
            else:
                kk = top_k

            if cur_num_gen <= self.args.encodec_sr // 5:
                logits_adjust[eog_inference] = -10000.0

            if (
                stop_repetition > 0
                and prev_token in silence_tokens
                and consec_silence_count > stop_repetition
            ):
                if logits_adjust[prev_token] < 0:
                    logits_adjust[prev_token] = logits_adjust[prev_token] * (
                        consec_silence_count - (stop_repetition - 1)
                    )
                else:
                    logits_adjust[prev_token] = logits_adjust[prev_token] / (
                        consec_silence_count - (stop_repetition - 1)
                    )

            token = topk_sampling(
                logits_adjust,
                top_k=kk,
                top_p=top_p,
                min_p=min_p,
                temperature=temperature,
            )
            token_id = int(token.item())

            should_force_stop = (
                token_id == eog_inference or int(torch.argmax(logits).item()) == eog_inference
            )

            text_mode = getattr(self.args, "text_input_type", "text") == "text"
            frames_per_token_cap = getattr(self.args, "text_guard_frames_per_token", 0)
            first_input_len = int(x_lens[0].item())
            if not text_mode:
                token_budget = first_input_len * max(
                    1, int(self.args.encodec_sr) // 4
                )
                should_force_stop = should_force_stop or (
                    effective_length > token_budget
                )
            elif frames_per_token_cap > 0:
                token_budget = max(1, first_input_len) * frames_per_token_cap
                should_force_stop = should_force_stop or (
                    effective_length > token_budget
                )

            time_budget_exceeded = target_total is not None and cur_num_gen > (
                target_total
                - prompt_offset
                + int(self.args.encodec_sr) * getattr(self.args, "extra_cutoff", 5)
            )
            if should_force_stop or time_budget_exceeded:
                token_id = eog_inference

            if token_id in silence_set and token_id == prev_token:
                consec_silence_count += 1
            else:
                consec_silence_count = 0
            prev_token = token_id
            return token_id, prev_token, consec_silence_count

        while True:
            logits = self.predict_layer[0](last_hidden).squeeze(0).squeeze(0)
            token_id, prev_token, consec_silence_count = sample_helper(
                logits,
                top_k,
                top_p,
                min_p,
                temperature,
                prev_token,
                consec_silence_count,
                stop_repetition,
                silence_tokens,
                cur_num_gen,
                current_length,
                target_total,
                prompt_offset,
            )

            token_tensor = torch.tensor([[token_id]], device=device, dtype=torch.long)
            generated_tokens.append(token_tensor.squeeze(0))
            cur_num_gen += 1
            current_length += 1

            if token_id == eog_inference:
                break

            samples_emb = self.audio_embedding[0](token_tensor)
            samples_emb = self.audio_dropout(samples_emb)

            if getattr(self.args, "use_pm_rope", 1):
                new_pos_value = (
                    float(current_length - 1)
                    / max(1, est_total - 1)
                    * self.progress_scale
                )
                new_pos_value = min(new_pos_value, self.progress_scale)
                # Avoid torch.cat: create single-token position tensor directly
                pos_1 = torch.tensor(
                    [[new_pos_value]], device=device, dtype=torch.float32
                )
                pm_kwargs = {
                    "position_ids": pos_1,
                    "pm_decoder_position_ids": pos_1,
                    "pm_encoder_position_ids": encoder_position_ids,
                }
            else:
                pm_kwargs = {"position_ids": None}

            # With KV cache, decoder only needs mask length covering all seen tokens.
            # Reuse pre-allocated buffer slice instead of creating new tensor each step.
            decoder_outputs = self.decoder_module(
                inputs_embeds=samples_emb,
                attention_mask=full_dec_attention_mask[:, :current_length],
                encoder_hidden_states=memory,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                **pm_kwargs,
            )
            past_key_values = decoder_outputs.past_key_values
            last_hidden = decoder_outputs.last_hidden_state

        if generated_tokens:
            generated_tensor = torch.stack(generated_tokens, dim=1)  # [1, T_gen]
        else:
            generated_tensor = torch.zeros((1, 0), device=device, dtype=torch.long)

        expected_y_len = y_len + generated_tensor.shape[1]
        res = torch.cat([y[0], generated_tensor], dim=1).unsqueeze(0)
        assert res.shape == torch.Size((1, 1, expected_y_len))

        if self.args.special_first:
            res = res - int(self.args.n_special)
            generated_tensor = generated_tensor - int(self.args.n_special)
        return res, generated_tensor.unsqueeze(0)