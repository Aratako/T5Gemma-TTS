from typing import Any, Optional

import torch
import torchaudio

# PyTorch 2.9+ compatibility helper
def _torch_ge_29() -> bool:
    """Return True if PyTorch version is >= 2.9."""
    try:
        v = torch.__version__.split("+")[0]
        major, minor = map(int, v.split(".")[:2])
        return (major, minor) >= (2, 9)
    except Exception:
        return False


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
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
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

def _load_audio(audio_path: str, offset: int = -1, num_frames: int = -1):
    """
    Load audio file using appropriate backend.
    For PyTorch >= 2.9 without TorchCodec, uses soundfile as fallback.
    """
    if _torch_ge_29():
        # Use soundfile to avoid TorchCodec dependency in PyTorch 2.9+
        import soundfile as sf
        info = sf.info(audio_path)
        sr = info.samplerate

        if offset != -1 and num_frames != -1:
            wav, _ = sf.read(audio_path, start=offset, stop=offset + num_frames, dtype="float32")
        else:
            wav, _ = sf.read(audio_path, dtype="float32")

        # Convert to torch tensor with shape [channels, samples]
        wav = torch.from_numpy(wav)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2:
            # soundfile returns [samples, channels], transpose to [channels, samples]
            wav = wav.T
        return wav, sr
    else:
        # Use torchaudio for PyTorch < 2.9
        if offset != -1 and num_frames != -1:
            wav, sr = torchaudio.load(audio_path, frame_offset=offset, num_frames=num_frames)
        else:
            wav, sr = torchaudio.load(audio_path)
        return wav, sr


def tokenize_audio(tokenizer: AudioTokenizer, audio_path: str, offset = -1, num_frames=-1):
    # Load and pre-process the audio waveform
    wav, sr = _load_audio(audio_path, offset, num_frames)
    target_sr = getattr(tokenizer, "encode_sample_rate", tokenizer.sample_rate)
    if sr != target_sr:
        # Ensure float32 for torchaudio resample (avoids Float vs Double mismatch in PyTorch 2.9+)
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
