

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

import torch
import torch.nn.functional as F


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Build padding mask for variable-length sequences.
    Returns bool tensor with True at pad positions.
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded = seq_range.unsqueeze(0).expand(n, max_len)
    return expanded >= lengths.unsqueeze(-1)


def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
    min_p=0.0,
    filter_value=-float("Inf"),
    min_tokens_to_keep=1,
):
    """
    Apply top-k / nucleus filtering with optional minimum probability cutoff.
    Adapted from Hugging Face's sampling helpers.

    Notes
    -----
    * min_p <= 0.0 : disabled (no masking, preserves caller's top_k/top_p)
    * 0 < min_p < 1.0 : mask tokens whose probability is below min_p; only if
      at least one token survives. When active, top-k / nucleus are bypassed to
      avoid double-masking.
    """
    min_p_enabled = 0.0 < min_p < 1.0
    if min_p_enabled:
        probs = F.softmax(logits, dim=-1)
        indices_to_remove = probs < min_p
        # Skip if everything would be removed
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
        assert len(top_k) == logits.size(0)
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

        # Map sorted mask back to original indices and apply
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, min_p=0.0, temperature=1.0):
    """
    Temperature-scaled sampling with top-k / nucleus filtering.
    """
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, min_p=min_p)
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token