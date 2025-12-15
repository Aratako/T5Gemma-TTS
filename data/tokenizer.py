

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

from typing import Any, Optional

import torch
import torchaudio

try:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from xcodec2.configuration_bigcodec import BigCodecConfig
    from xcodec2.modeling_xcodec2 import XCodec2Model

    _HAS_XCODEC2 = True
except ImportError:
    _HAS_XCODEC2 = False


class AudioTokenizer:
    """Audio tokenizer wrapper (XCodec2 only)."""

    def __init__(
        self,
        device: Any = None,
        signature: Optional[str] = None,
        backend: str = "xcodec2",
        model_name: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        if backend != "xcodec2":
            raise ValueError(f"Only xcodec2 backend is supported now (got {backend}).")

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda")
        self._device = device
        self.backend = backend
        self.signature = signature
        self.model_name = model_name

        if not _HAS_XCODEC2:
            raise ImportError(
                "xcodec2 module not found. Install with `pip install xcodec2` or add the bundled version to PYTHONPATH."
            )
        model_id = model_name or signature or "NandemoGHS/Anime-XCodec2-44.1kHz-v2"
        ckpt_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        ckpt = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                ckpt[k.replace(".beta", ".bias")] = f.get_tensor(k)
            codec_config = BigCodecConfig.from_pretrained(model_id)
            self.codec = XCodec2Model.from_pretrained(
                None, config=codec_config, state_dict=ckpt
            )
            self.codec.eval()
            self.codec.to(device)
        self.sample_rate = sample_rate or None
        if self.sample_rate is None:
            self.sample_rate = int(getattr(self.codec.config, "sample_rate", 44100))
        encode_sr = getattr(self.codec.config, "encoder_sample_rate", None)
        if encode_sr is None:
            fe = getattr(self.codec, "feature_extractor", None)
            encode_sr = getattr(fe, "sampling_rate", None) if fe is not None else None
        self.encode_sample_rate = int(encode_sr) if encode_sr is not None else 16000
        self.channels = 1

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        wav = wav.to(self.device)
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        codes = self.codec.encode_code(input_waveform=wav, sample_rate=self.encode_sample_rate)
        if codes.ndim == 2:
            codes = codes.unsqueeze(-1)
        codes = codes.permute(0, 2, 1).contiguous()
        return codes.to(dtype=torch.long)

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        codes = frames
        if codes.ndim == 2:
            codes = codes.unsqueeze(1)
        codes = codes.long().to(self.device)
        recon = self.codec.decode_code(codes)
        return recon

def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str, offset = -1, num_frames=-1):
    # Load and pre-process the audio waveform
    if offset != -1 and num_frames!=-1:
        wav, sr = torchaudio.load(audio_path, frame_offset=offset, num_frames=num_frames)
    else:
        wav, sr = torchaudio.load(audio_path)
    target_sr = getattr(tokenizer, "encode_sample_rate", tokenizer.sample_rate)
    if sr != target_sr:
        # ensure float32 for torchaudio resample
        wav = wav.to(dtype=torch.float32)
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    if wav.shape[0] == 2:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0)
    # Extract discrete codes from mimi
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames