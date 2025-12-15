

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

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
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

from .utils import make_pad_mask, topk_sampling


class PMCrossAttention(T5GemmaCrossAttention):
    """T5Gemma cross-attention augmented with Progress-Monitoring RoPE."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        # Independent rotary embeddings for decoder queries and encoder keys.
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
        # Broadcast cos/sin to match [B, num_heads, seq, head_dim]
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

        # Protect downstream flash attention wrappers from unexpected kwargs.
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
        # Replace the existing cross-attention with a PM-enabled version.
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


def _require_transformers():
    try:
        from transformers import AutoModelForSeq2SeqLM
    except ImportError as exc:
        raise ImportError(
            "transformers library not found. Run `pip install transformers`."
        ) from exc
    return AutoModelForSeq2SeqLM


def _require_peft():
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required when use_lora=1. Run `pip install peft`."
        ) from exc
    return LoraConfig, get_peft_model


class T5GemmaVoiceModel(nn.Module):
    """
    T5Gemma-based audio token generation model that mirrors the VoiceStar training/eval API.
    Interfaces for returns and loss computation are aligned so the existing Trainer can be reused as-is.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        if getattr(self.args, "n_codebooks", 1) != 1:
            logging.info("Resetting n_codebooks to 1 for XCodec2 backend.")
            self.args.n_codebooks = 1
        AutoModelForSeq2SeqLM = _require_transformers()

        logging.info(f"Loading T5Gemma backbone: {self.args.t5gemma_model_name}")
        precision = getattr(self.args, "precision", "float32")
        if precision == "float16":
            dtype = torch.float16
        elif precision == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        self.backbone = AutoModelForSeq2SeqLM.from_pretrained(
            self.args.t5gemma_model_name,
            attn_implementation=getattr(self.args, "attn_implementation", "eager"),
            torch_dtype=dtype,
        )
        prune_text_modules = getattr(self.args, "prune_text_modules", None)
        # backward compatibility: allow old --drop_lm_head
        legacy_drop_lm_head = getattr(self.args, "drop_lm_head", 0)
        if prune_text_modules is None:
            prune_text_modules = 1 if legacy_drop_lm_head else 0
        drop_lm_head = prune_text_modules >= 1
        drop_decoder_embed = prune_text_modules >= 2

        if drop_lm_head and hasattr(self.backbone, "lm_head"):
            # lm_head is unused for audio tasks; removing it saves roughly 200M parameters.
            del self.backbone.lm_head
            self.backbone.lm_head = nn.Identity()
            if hasattr(self.backbone.config, "tie_word_embeddings"):
                self.backbone.config.tie_word_embeddings = False
            logging.info("lm_head removed (prune_text_modules=%d)", prune_text_modules)

        if drop_decoder_embed:
            decoder = getattr(self.backbone, "model", getattr(self.backbone, "decoder", None))
            decoder = getattr(decoder, "decoder", decoder)
            if decoder is not None and hasattr(decoder, "embed_tokens"):
                del decoder.embed_tokens
                decoder.embed_tokens = nn.Identity()
                if hasattr(self.backbone.config, "tie_word_embeddings"):
                    self.backbone.config.tie_word_embeddings = False
                logging.info("decoder.embed_tokens removed (prune_text_modules=%d)", prune_text_modules)
        self._gc_enabled = bool(getattr(self.args, "t5_gradient_checkpointing", 0))
        if self._gc_enabled:
            # use_reentrant=False avoids the double-ready issue in DDP.
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Disable cache while using gradient checkpointing to save memory.
            self.backbone.config.use_cache = False
        else:
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
        config = self.backbone.config
        hidden_size = getattr(config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "encoder", None)
            if hidden_size is not None:
                hidden_size = getattr(hidden_size, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("T5Gemma config does not expose d_model/hidden_size.")
        self.hidden_size = hidden_size
        self.args.audio_embedding_dim = getattr(
            self.args, "audio_embedding_dim", self.hidden_size
        )

        self._enable_pm_rope_cross_attention()

        self._enable_lora()

        if getattr(self.args, "freeze_t5gemma", 0):
            if getattr(self.args, "use_lora", 0):
                logging.warning(
                    "freeze_t5gemma is ignored when use_lora=1 because LoRA freezes the base model automatically."
                )
            else:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                logging.info("Backbone parameters frozen (freeze_t5gemma=1)")

        self.text_input_type = getattr(self.args, "text_input_type", "text")
        if self.text_input_type == "text":
            self.text_embedding = None
        else:
            text_vocab = self.args.text_vocab_size + 1
            self.text_embedding = nn.Embedding(text_vocab, self.hidden_size)
        self.text_dropout = nn.Dropout(
            getattr(self.args, "text_embedding_dropout", 0.0)
        )

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
        self.audio_dropout = nn.Dropout(
            getattr(self.args, "audio_embedding_dropout", 0.0)
        )

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

        # Keep metric lightweight; compute top-k accuracy on the fly to avoid large one-hot tensors.
        self.topk_eval = 10

        class_weight = torch.ones(audio_vocab_sizes[0])
        if getattr(self.args, "eog_weight", 1.0) != 1.0:
            class_weight[self.args.eog] = self.args.eog_weight
        self.register_buffer("class_weight", class_weight, persistent=False)
        self.progress_scale = getattr(self.args, "progress_scale", 2000.0)

        # Ensure only LoRA adapters are trainable when use_lora=1 (freeze late-created heads/embeddings too)
        self._freeze_to_lora_only()

    def carefully_load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        When drop_lm_head=1, ignore lm_head weights in the checkpoint.
        Also tolerate checkpoints that lack lm_head so loading does not fail.
        """
        try:
            target_dtype = next(self.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32

        def cast_fp(t: torch.Tensor):
            return t.to(dtype=target_dtype) if torch.is_floating_point(t) else t

        prune = getattr(self.args, "prune_text_modules", 0)
        drop_lm_head = prune >= 1 or getattr(self.args, "drop_lm_head", 0)
        drop_dec_embed = prune >= 2

        has_lm_head = any(k.startswith("backbone.lm_head.") for k in state_dict)
        has_dec_embed = any(k.startswith("backbone.model.decoder.embed_tokens.") for k in state_dict)

        removed_keys = []
        if drop_lm_head:
            removed_keys += [k for k in list(state_dict.keys()) if k.startswith("backbone.lm_head.")]
        if drop_dec_embed:
            removed_keys += [k for k in list(state_dict.keys()) if k.startswith("backbone.model.decoder.embed_tokens.")]
        for k in removed_keys:
            state_dict.pop(k)

        if removed_keys:
            strict = False
            logging.info("Dropped %d text-related keys while loading (prune_text_modules=%d)", len(removed_keys), prune)
        elif (not has_lm_head and not drop_lm_head) or (not has_dec_embed and not drop_dec_embed and prune >= 2):
            # checkpoint missing something we expect; be tolerant
            strict = False
            logging.warning("Checkpoint missing text modules; loading with strict=False")
        else:
            strict = True

        if getattr(self.args, "use_lora", 0):
            # If loading a vanilla checkpoint into a LoRA-wrapped model:
            #  - map backbone.* -> backbone.base_model.model.*
            #  - map target weights to *.base_layer.* and drop the original keys to avoid "unexpected" entries.
            targets = getattr(self.args, "lora_target_modules", None)
            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(",") if t.strip()]
            targets = targets or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            remapped = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith("backbone."):
                    new_key = "backbone.base_model.model." + k[len("backbone.") :]
                # rewrite target module weights to base_layer.*
                for tgt in targets:
                    suffix = f".{tgt}.weight"
                    if new_key.endswith(suffix) and "lora_" not in new_key and ".base_layer." not in new_key:
                        new_key = new_key.replace(f".{tgt}.weight", f".{tgt}.base_layer.weight")
                        break
                remapped[new_key] = cast_fp(v)
            # fill missing LoRA adapter params with existing initialized values to silence missing-key logs
            current_sd = self.state_dict()
            for k, v in current_sd.items():
                if "lora_" in k and k not in remapped:
                    remapped[k] = v
            state_dict = remapped
            strict = False
        else:
            # cast non-LoRA load path as well
            state_dict = {k: cast_fp(v) for k, v in state_dict.items()}

        result = self.load_state_dict(state_dict, strict=strict)
        missing = getattr(result, "missing_keys", [])
        unexpected = getattr(result, "unexpected_keys", [])
        if missing:
            logging.info("Missing keys: %s", missing)
        if unexpected:
            logging.info("Unexpected keys: %s", unexpected)

        # keep precision consistent with args.precision (e.g., bf16)
        try:
            target_dtype = next(self.parameters()).dtype
            self.to(dtype=target_dtype)
        except StopIteration:
            pass
        return result

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
            # Carry over preserved gradient-checkpointing settings and hooks.
            pm_layer.gradient_checkpointing = getattr(
                layer, "gradient_checkpointing", False
            )
            if hasattr(layer, "_gradient_checkpointing_func"):
                pm_layer._gradient_checkpointing_func = layer._gradient_checkpointing_func
            new_layers.append(pm_layer)
        self.decoder_module.layers = new_layers
        self._pm_rope_enabled = True
        logging.info(
            "PM-RoPE cross-attention enabled for %d decoder layers.", len(new_layers)
        )

    def _freeze_to_lora_only(self) -> None:
        if not getattr(self.args, "use_lora", 0):
            return
        for _, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def _enable_lora(self) -> None:
        if getattr(self, "_lora_enabled", False):
            return
        if not getattr(self.args, "use_lora", 0):
            return
        LoraConfig, get_peft_model = _require_peft()
        targets = getattr(self.args, "lora_target_modules", None)
        if isinstance(targets, str):
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        if not targets:
            targets = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        lora_config = LoraConfig(
            r=getattr(self.args, "lora_r", 16),
            lora_alpha=getattr(self.args, "lora_alpha", 32),
            lora_dropout=getattr(self.args, "lora_dropout", 0.05),
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=targets,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        # Freeze all params, then re-enable LoRA adapter weights only
        for _, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        self._lora_enabled = True
        try:
            self.backbone.print_trainable_parameters()
        except Exception:
            pass
        logging.info(
            "LoRA enabled: targets=%s, r=%d, alpha=%d, dropout=%.3f",
            targets,
            lora_config.r,
            lora_config.lora_alpha,
            lora_config.lora_dropout,
        )

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

    def _prepare_decoder_inputs(self, y: torch.Tensor, y_lens: torch.Tensor):
        """
        Single-codebook decoder input prep for XCodec2.
        Returns
            targets (List[Tensor]): target arrays shaped [B][1, T]
            decoder_inputs: Tensor shape [B, T', hidden]
            new_y_lens: Tensor shape [B]
        """
        if getattr(self.args, "n_codebooks", 1) != 1:
            raise ValueError("XCodec2 path only supports n_codebooks=1.")

        eos_token = self.args.eos if getattr(self.args, "eos", -1) > 0 else self.args.eog
        targets: List[torch.Tensor] = []
        decoder_inputs_list: List[torch.Tensor] = []
        bos_token = torch.full_like(y[0][:, :1], self.args.empty_token)

        for i, item in enumerate(y):
            with_eos = torch.cat(
                [item[:, : y_lens[i]], torch.full_like(item[:, :1], eos_token)],
                dim=-1,
            )  # [1, T+1]
            targets.append(with_eos)
            # Shift right by one: BOS then y tokens (except final EOS).
            decoder_inputs_list.append(torch.cat([bos_token, with_eos[:, :-1]], dim=-1))

        new_y_lens = y_lens + 1  # BOS added

        # [T_i+1, 1] -> pad -> [T_max, B, 1] -> [B, T_max, 1] after permute
        cated_y = torch.nn.utils.rnn.pad_sequence(
            [item.transpose(1, 0) for item in decoder_inputs_list],
            batch_first=False,
            padding_value=self.args.audio_pad_token,
        ).permute(2, 0, 1)

        embedded_y = self.audio_embedding[0](cated_y[0]).transpose(1, 0)
        embedded_y = self.audio_dropout(embedded_y)
        y_padding_mask = make_pad_mask(new_y_lens).to(y.device)

        return targets, embedded_y, new_y_lens, y_padding_mask

    def forward(self, batch: Dict[str, torch.Tensor]):
        x, x_lens, y, y_lens = batch["x"], batch["x_lens"], batch["y"], batch["y_lens"]
        if len(x) == 0:
            return None

        x = x[:, : x_lens.max()]
        y = y[..., : y_lens.max()]

        x_padding_mask = make_pad_mask(x_lens).to(x.device)
        encoder_attention_mask = (~x_padding_mask).long()
        if getattr(self.args, "use_pm_rope", 1):
            encoder_position_ids = self._build_position_ids(x_lens, x.shape[1], x.device)
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

        targets, decoder_inputs, new_y_lens, y_padding_mask = (
            self._prepare_decoder_inputs(y, y_lens)
        )
        # padding mask (1 = keep)
        base_padding_mask = (~y_padding_mask).long()  # [B, T]
        # causal mask: upper triangular masked with -inf
        seq_len = decoder_inputs.shape[1]
        causal = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=decoder_inputs.device,
            dtype=decoder_inputs.dtype,
        )
        causal = torch.triu(causal, diagonal=1)  # [T, T]
        causal = causal.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B, T, T]
        # incorporate padding on keys: mask columns where padding=True
        if y_padding_mask.any():
            key_mask = y_padding_mask[:, None, :].to(decoder_inputs.dtype) * -1e9  # [B,1,T]
            causal = causal + key_mask
        decoder_attention_mask = causal[:, None, :, :]  # [B,1,T,T]
        pm_kwargs = {}
        if getattr(self.args, "use_pm_rope", 1):
            decoder_position_ids = self._build_position_ids(
                new_y_lens, decoder_inputs.shape[1], decoder_inputs.device
            )
            pm_kwargs["position_ids"] = decoder_position_ids
            pm_kwargs["pm_decoder_position_ids"] = decoder_position_ids
            pm_kwargs["pm_encoder_position_ids"] = encoder_position_ids
        else:
            # Standard T5Gemma uses standard position_ids if not provided
            pm_kwargs["position_ids"] = None

        decoder_outputs = self.decoder_module(
            inputs_embeds=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,  # training: disable cache to save memory
            **pm_kwargs,
        )
        decoder_hidden = decoder_outputs.last_hidden_state

        logits = torch.stack(
            [
                self.predict_layer[i](decoder_hidden)
                for i in range(self.args.n_codebooks)
            ],
            dim=1,
        )  # [B, K, T, V]
        logits_use = [logit[:, : new_y_lens[i]] for i, logit in enumerate(logits)]

        logits_final = []
        if self.args.n_codebooks == 1:
            logits_final = logits_use
        else:
            for i, logit in enumerate(logits_use):
                logit_copy = logit.clone()
                for ii in range(self.args.n_codebooks):
                    logit_copy[ii] = torch.roll(logit_copy[ii], shifts=-(ii), dims=0)
                logit = logit_copy[:, : -self.args.n_codebooks]
                logits_final.append(logit)

        if getattr(self.args, "no_loss_on_prefix", 0):
            if "y_sep_token_position" not in batch:
                raise KeyError(
                    "When no_loss_on_prefix=1, the batch must include y_sep_token_position."
                )
            logit_trimmed = []
            target_trimmed = []
            sep_positions = batch["y_sep_token_position"]
            for jj, (logit, target) in enumerate(zip(logits_final, targets)):
                prefix_len = (
                    int(sep_positions[jj].item())
                    if sep_positions.ndim > 0
                    else int(sep_positions.item())
                )
                logit_trimmed.append(logit[:, prefix_len:])
                target_trimmed.append(target[:, prefix_len:])
            logits_final = logit_trimmed
            targets = target_trimmed

        logits = torch.cat(logits_final, dim=1)  # [K, total_len, V]
        targets = torch.cat(targets, dim=1)  # [K, total_len]

        losses = []
        ntokens = []
        top10acc = []  # store correct-count per codebook to keep semantics with trainer
        for logit, target in zip(logits, targets):
            losses.append(
                F.cross_entropy(
                    logit,
                    target,
                    reduction="mean",
                    weight=(
                        self.class_weight.data if self.args.eog_weight != 1 else None
                    ),
                    ignore_index=(
                        self.args.y_sep_token
                        if getattr(self.args, "y_sep_token", None) is not None
                        else -100
                    ),
                )
            )

            # Manual top-k accuracy without torchmetrics one-hot expansion (saves a few GB).
            with torch.no_grad():
                k_val = min(self.topk_eval, logit.shape[-1])
                topk_idx = logit.topk(k_val, dim=-1).indices  # [T, k]
                correct = (topk_idx == target.unsqueeze(-1)).any(dim=-1)
                top10acc.append(correct.sum())

            ntokens.append(target.numel())

        total_tokens = sum(ntokens)
        if getattr(self.args, "codebook_weight", None) is not None:
            codebook_weight = (
                eval(self.args.codebook_weight)
                if isinstance(self.args.codebook_weight, str)
                else self.args.codebook_weight
            )
        else:
            codebook_weight = [1.0] * self.args.n_codebooks

        loss = sum(l * nt * cw for l, nt, cw in zip(losses, ntokens, codebook_weight))
        perplexity_by_codebook = [torch.exp(l).detach() for l in losses]

        # top10acc stores correct-counts already; summing keeps trainer semantics.
        top10acc_weighted = top10acc
        top10acc_total = sum(top10acc_weighted)
        effective_ntokens = torch.tensor(total_tokens).to(logits.device)

        return {
            "loss": loss,
            "perplexity_by_codebook": perplexity_by_codebook,
            "top10acc": top10acc_total,
            "top10acc_by_codebook": top10acc_weighted,
            "effective_ntoken": effective_ntokens,
        }

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

        # Enable KV cache automatically during inference for speed.
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

        # Prepend BOS (=empty_token) for causal shift consistency with training.
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
            est_total = int(target_total) + 1  # account for BOS we added
        elif cutoff_limit is not None:  # Fallback just in case
            est_total = int(cutoff_limit)
        else:
            # Additional fallback: keep a slightly overestimated length
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
            base = torch.arange(cur_len, device=device, dtype=torch.float32).unsqueeze(0)
            decoder_position_ids_full = base / max(1, est_total - 1) * self.progress_scale
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
                    float(current_length - 1) / max(1, est_total - 1) * self.progress_scale
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

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load weights while skipping legacy `accuracy_metrics.*` entries.

        Older checkpoints stored torchmetrics state that no longer exists after
        switching to manual top-k accuracy. We drop those keys here to stay
        backward compatible while keeping the default strict loading behavior
        for all other parameters.
        """
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("accuracy_metrics")}
        return super().load_state_dict(filtered, strict=strict)