

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
Prepare the English portion of ``amphion/Emilia-Dataset`` for the T5Gemma-TTS
training using XCodec2 acoustic tokens.

Filters:
  - keep only the requested language codes (default: ``en``)
  - drop samples whose IDs are in the original EN block‑list
  - drop samples containing disallowed substrings (e.g., Arabic letters mixed in)
  - drop samples with excessive short-pattern repetition
  - trim leading spaces; no phoneme conversion

Output structure:
    <output_root>/
        text/<utt_id>.txt
        xcodec2_1cb/<utt_id>.txt
        manifest_final/<split>.txt
        neighbors/<utt_id>.txt

Usage (example):
    python scripts/prepare_emilia_en.py \\
        --output-dir datasets/emilia-yodas-en \\
        --data-files '{\"train\": \"Emilia-YODAS/**/*.tar\"}' \\
        --encoder-devices auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import queue
import random
import re
import sys
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset as hf_load_dataset
from tqdm import tqdm

# Project root (two levels up from this script)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.tokenizer import AudioTokenizer

LOGGER = logging.getLogger("prepare_emilia_en")

# Block-list copied from the original Emilia EN blocklist file
DEFAULT_EN_BLOCKLIST = {
    "EN_B00013_S00913",
    "EN_B00042_S00120",
    "EN_B00055_S04111",
    "EN_B00061_S00693",
    "EN_B00061_S01494",
    "EN_B00061_S03375",
    "EN_B00059_S00092",
    "EN_B00111_S04300",
    "EN_B00100_S03759",
    "EN_B00087_S03811",
    "EN_B00059_S00950",
    "EN_B00089_S00946",
    "EN_B00078_S05127",
    "EN_B00070_S04089",
    "EN_B00074_S09659",
    "EN_B00061_S06983",
    "EN_B00061_S07060",
    "EN_B00059_S08397",
    "EN_B00082_S06192",
    "EN_B00091_S01238",
    "EN_B00089_S07349",
    "EN_B00070_S04343",
    "EN_B00061_S02400",
    "EN_B00076_S01262",
    "EN_B00068_S06467",
    "EN_B00076_S02943",
    "EN_B00064_S05954",
    "EN_B00061_S05386",
    "EN_B00066_S06544",
    "EN_B00076_S06944",
    "EN_B00072_S08620",
    "EN_B00076_S07135",
    "EN_B00076_S09127",
    "EN_B00065_S00497",
    "EN_B00059_S06227",
    "EN_B00063_S02859",
    "EN_B00075_S01547",
    "EN_B00061_S08286",
    "EN_B00079_S02901",
    "EN_B00092_S03643",
    "EN_B00096_S08653",
    "EN_B00063_S04297",
    "EN_B00063_S04614",
    "EN_B00079_S04698",
    "EN_B00104_S01666",
    "EN_B00061_S09504",
    "EN_B00061_S09694",
    "EN_B00065_S05444",
    "EN_B00063_S06860",
    "EN_B00065_S05725",
    "EN_B00069_S07628",
    "EN_B00083_S03875",
    "EN_B00071_S07665",
    "EN_B00071_S07665",
    "EN_B00062_S04187",
    "EN_B00065_S09873",
    "EN_B00065_S09922",
    "EN_B00084_S02463",
    "EN_B00067_S05066",
    "EN_B00106_S08060",
    "EN_B00073_S06399",
    "EN_B00073_S09236",
    "EN_B00087_S00432",
    "EN_B00085_S05618",
    "EN_B00064_S01262",
    "EN_B00072_S01739",
    "EN_B00059_S03913",
    "EN_B00069_S04036",
    "EN_B00067_S05623",
    "EN_B00060_S05389",
    "EN_B00060_S07290",
    "EN_B00062_S08995",
}

_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_\-./]")


@dataclass
class ManifestEntry:
    utt_id: str
    token_len: int


def ensure_neighbor_dir(base: Path, folder: str, overwrite: bool) -> Path:
    neighbor_dir = base / folder
    if overwrite and neighbor_dir.exists():
        for file in neighbor_dir.glob("*.txt"):
            try:
                file.unlink()
            except OSError:
                LOGGER.warning("Failed to remove %s", file, exc_info=True)
    neighbor_dir.mkdir(parents=True, exist_ok=True)
    return neighbor_dir


def load_manifest_map(manifest_path: Path) -> Dict[str, ManifestEntry]:
    """Loads manifest into a lookup dictionary."""
    manifest_map: Dict[str, ManifestEntry] = {}
    if not manifest_path.exists():
        LOGGER.warning("Manifest file not found: %s; skipping.", manifest_path)
        return manifest_map
    with manifest_path.open("r", encoding="utf-8") as mf:
        for line in mf:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            try:
                token_len = int(float(parts[1]))
            except ValueError:
                LOGGER.warning(
                    "Unexpected token length in %s: %s", manifest_path, parts[1]
                )
                continue
            if utt_id in manifest_map:
                LOGGER.warning(
                    "Duplicate utt_id '%s' found in manifest %s",
                    utt_id,
                    manifest_path,
                )
            manifest_map[utt_id] = ManifestEntry(utt_id=utt_id, token_len=token_len)
    return manifest_map


def sanitize_utt_id(raw_id: str) -> str:
    sanitized = _ID_SAFE_RE.sub("_", str(raw_id).strip())
    sanitized = sanitized.strip("._/")
    return sanitized or "utt"


def parse_data_files_arg(value: Optional[str]):
    if value is None:
        return None
    # First, try direct JSON.
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        original_exc = exc
        candidate = value.strip()

        # Common CLI quoting: `{\\\"train\\\": \\\"path\\\"}` (escaped quotes left in)
        if '\\"' in candidate:
            try:
                return json.loads(candidate.replace('\\"', '"'))
            except json.JSONDecodeError:
                pass

        # Another common case: single‑quoted JSON: {'train': 'path'}
        if candidate.startswith(("{", "[")) and "'" in candidate:
            try:
                return json.loads(candidate.replace("'", '"'))
            except json.JSONDecodeError:
                pass

        # If it still looks like JSON, surface the original error.
        if candidate.startswith("{") or candidate.startswith("["):
            raise original_exc

        # Fallback: treat comma-separated list or bare string.
        if "," in candidate:
            return [v.strip() for v in candidate.split(",") if v.strip()]
        return candidate


def load_audio_tensor(audio_field, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Convert dataset audio field into mono tensor at target sample rate."""
    if audio_field is None:
        raise ValueError("Audio field is None.")

    if isinstance(audio_field, dict):
        array = audio_field.get("array", None)
        sr = audio_field.get("sampling_rate", None)
        path = audio_field.get("path", None)
        if array is None or sr is None:
            if path is None:
                raise ValueError("Audio dict missing array/path information.")
            array, sr = sf.read(path, always_2d=False)
    elif isinstance(audio_field, str):
        array, sr = sf.read(audio_field, always_2d=False)
    else:
        raise ValueError(f"Unsupported audio field type: {type(audio_field)}")

    if isinstance(array, list):
        array = np.array(array)

    audio_np = np.asarray(array, dtype=np.float32)
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    elif audio_np.ndim != 1:
        raise ValueError(f"Unexpected audio ndim={audio_np.ndim}")

    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    waveform = waveform * 0.99
    return waveform, sr


def ensure_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        "text": base / "text",
        "codes": base / "xcodec2_1cb",
        "manifest": base / "manifest_final",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_outputs(
    dirs: Dict[str, Path],
    split: str,
    utt_id: str,
    tokens: torch.Tensor,
    text: str,
    overwrite: bool,
) -> int:
    """Write text, tokens, and manifest entry; returns token length."""
    shard_id = hashlib.md5(utt_id.encode("utf-8")).hexdigest()[:2]
    text_parent_dir = dirs["text"] / shard_id
    codes_parent_dir = dirs["codes"] / shard_id
    text_parent_dir.mkdir(exist_ok=True)
    codes_parent_dir.mkdir(exist_ok=True)

    text_path = text_parent_dir / f"{utt_id}.txt"
    codes_path = codes_parent_dir / f"{utt_id}.txt"

    if not overwrite and (text_path.exists() or codes_path.exists()):
        raise FileExistsError(
            f"Destination files already exist for {utt_id}; rerun with --overwrite."
        )

    text_path.write_text(str(text).strip() + "\n", encoding="utf-8")

    tokens_np = tokens.cpu().numpy()
    if tokens_np.ndim == 1:
        tokens_np = tokens_np[None, :]
    elif tokens_np.ndim == 2 and tokens_np.shape[0] > tokens_np.shape[1]:
        tokens_np = tokens_np.T
    lines = [" ".join(str(int(tok)) for tok in row.tolist()) for row in tokens_np]
    codes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_entry = f"{shard_id}/{utt_id}\t{tokens_np.shape[-1]}\n"
    manifest_path = dirs["manifest"] / f"{split}.txt"
    with manifest_path.open("a", encoding="utf-8") as mf:
        mf.write(manifest_entry)

    return tokens_np.shape[-1]


def normalize(text: str) -> str:
    """Lightweight English text normalization: trim + collapse whitespace."""
    return " ".join(str(text).lstrip().split())


def is_allowed() -> bool:
    """English text filter: always allow (language checks handled elsewhere)."""
    return True


@dataclass
class EncodeJob:
    job_id: int
    idx: int
    dest_split: str
    utt_id: str
    text: str
    allow_overwrite: bool
    forced: bool
    waveform: torch.Tensor


def repetition_found(text: str, length: int = 4, tolerance: int = 10) -> bool:
    """Detect over-repetition of short patterns."""
    from collections import defaultdict

    if length <= 0:
        return False
    counts = defaultdict(int)
    for i in range(max(0, len(text) - length + 1)):
        counts[text[i : i + length]] += 1
    return any(cnt > tolerance for cnt in counts.values())


@dataclass
class SampleRecord:
    utt_id: str
    speaker: str
    duration_sec: float
    split: str


def get_field(example: Dict, key: str, meta_col: Optional[str]) -> Optional[object]:
    """Fetch key from top-level or from nested metadata column (dict)."""
    if key in example:
        return example.get(key)
    if meta_col and meta_col in example and isinstance(example.get(meta_col), dict):
        return example[meta_col].get(key)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare amphion/Emilia-Dataset (English) and build neighbors."
    )

    # Shared with prepare_textmode_dataset.py
    parser.add_argument("--dataset-name", default="amphion/Emilia-Dataset")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--data-files",
        default=None,
        help="JSON/dict/list/CSV of files passed through to datasets.load_dataset.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-column", default="mp3")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="_id")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--valid-ratio", type=float, default=0.0)
    parser.add_argument("--valid-split-name", default="valid")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--codec-sample-rate", type=int, default=16000)
    parser.add_argument(
        "--tokenizer-model",
        default="NandemoGHS/Anime-XCodec2-44.1kHz-v2",
        help="XCodec2 checkpoint for acoustic tokenisation.",
    )
    parser.add_argument(
        "--encoder-devices",
        default=None,
        help="Comma-separated CUDA indices or 'auto'.",
    )
    parser.add_argument("--encoder-queue-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf-num-proc",
        type=int,
        default=1,
        help="Workers for HF map/filter (ignored in streaming mode).",
    )

    # Neighbor parameters
    parser.add_argument("--speaker-column", default="speaker")
    parser.add_argument("--duration-column", default="duration")
    parser.add_argument("--neighbor-folder", default="neighbors")
    parser.add_argument("--group-by", choices=("speaker",), default="speaker")
    parser.add_argument(
        "--distance-metric", choices=("duration_diff", "zero"), default="duration_diff"
    )
    parser.add_argument("--max-neighbors-per-utt", type=int, default=None)
    parser.add_argument(
        "--encodec-sr",
        type=float,
        default=50.0,
        help="Tokens/sec fallback when duration column is missing.",
    )
    parser.add_argument(
        "--neighbors-only",
        action="store_true",
        help="Skip dataset preparation and only rebuild neighbors.",
    )

    # English-specific filters
    parser.add_argument(
        "--language-column",
        default="language",
        help="Column containing language code (e.g., 'en').",
    )
    parser.add_argument(
        "--allowed-languages",
        default="en",
        help="Comma-separated language codes to keep. Empty string keeps all.",
    )
    parser.add_argument(
        "--no-default-blocklist",
        action="store_true",
        help="Disable the built-in EN block-list.",
    )
    parser.add_argument(
        "--extra-blocklist",
        nargs="*",
        default=None,
        help="Additional utterance IDs to drop (matched against id-column).",
    )
    parser.add_argument(
        "--bad-substrings",
        default="ا,い,て",
        help="Comma-separated substrings that cause a sample to be dropped.",
    )
    parser.add_argument(
        "--repetition-length",
        type=int,
        default=4,
        help="Pattern length for repetition filter (<=0 disables).",
    )
    parser.add_argument(
        "--repetition-tolerance",
        type=int,
        default=10,
        help="Maximum allowed repeats of the same pattern.",
    )
    parser.add_argument(
        "--metadata-column",
        default="json",
        help="If set, pull id/text/etc from this nested dict column when missing at top level.",
    )

    return parser.parse_args()


def build_filter_fn(args: argparse.Namespace) -> Callable[[Dict], bool]:
    allowed_langs: Set[str] = {
        lang.strip().lower()
        for lang in (args.allowed_languages or "").split(",")
        if lang.strip()
    }
    bad_substrings: List[str] = [
        s for s in (args.bad_substrings or "").split(",") if s
    ]

    blocklist: Set[str] = set()
    if not args.no_default_blocklist:
        blocklist.update(DEFAULT_EN_BLOCKLIST)
    if args.extra_blocklist:
        blocklist.update(args.extra_blocklist)

    stats = {"seen": 0, "kept": 0, "dropped": 0}

    def _passes(example: Dict) -> bool:
        stats["seen"] += 1
        # Language filter
        lang_val = get_field(example, args.language_column, args.metadata_column)
        if allowed_langs and lang_val is not None:
            lang_val = str(lang_val).lower().strip()
            if lang_val not in allowed_langs:
                stats["dropped"] += 1
                return False

        # ID blocklist
        if args.id_column:
            raw_id = get_field(example, args.id_column, args.metadata_column)
            if raw_id is not None:
                raw_id = str(raw_id).strip()
                if raw_id in blocklist:
                    stats["dropped"] += 1
                    return False

        # Text checks
        raw_text = get_field(example, args.text_column, args.metadata_column)
        if raw_text is None:
            stats["dropped"] += 1
            return False
        text = str(raw_text)
        if not text.strip():
            stats["dropped"] += 1
            return False
        text = text.lstrip()
        if bad_substrings and any(bad in text for bad in bad_substrings):
            stats["dropped"] += 1
            return False
        if repetition_found(
            text, length=args.repetition_length, tolerance=args.repetition_tolerance
        ):
            stats["dropped"] += 1
            return False
        stats["kept"] += 1
        return True

    _passes.stats = stats
    return _passes


def expand_metadata_columns(ds, args: argparse.Namespace):
    """If metadata_column exists, copy needed fields to top-level."""
    meta_col = args.metadata_column
    if not meta_col or meta_col not in ds.column_names:
        return ds

    target_keys = {
        args.id_column,
        args.text_column,
        args.audio_column,
        args.speaker_column,
        args.duration_column,
        args.language_column,
    }
    target_keys = {k for k in target_keys if k}

    def _extract(example):
        meta = example.get(meta_col, {})
        if not isinstance(meta, dict):
            return example
        for k in target_keys:
            if k not in example and k in meta:
                example[k] = meta[k]
        return example

    num_proc = None if args.streaming else max(1, args.hf_num_proc or 1)
    return ds.map(_extract, remove_columns=None, num_proc=num_proc)


def resolve_encoder_devices(
    device_spec: Optional[str],
) -> Optional[List[torch.device]]:
    if device_spec is None:
        return None

    device_spec = device_spec.strip()
    if not device_spec:
        return None

    if device_spec.lower() == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [
                torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())
            ]
        return [torch.device("cpu")]

    tokens = [token.strip() for token in device_spec.split(",") if token.strip()]
    if not tokens:
        return None

    devices: List[torch.device] = []
    for token in tokens:
        lower = token.lower()
        if lower == "cpu":
            devices.append(torch.device("cpu"))
            continue

        if lower.startswith("cuda"):
            if ":" in token:
                _, suffix = token.split(":", 1)
            else:
                suffix = token[4:] if len(token) > 4 else "0"
        elif token.isdigit():
            suffix = token
        else:
            raise ValueError(f"Unrecognized device specifier '{token}'.")

        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA device '{token}' requested but CUDA is not available on this host."
            )
        try:
            index = int(suffix)
        except ValueError as exc:  # noqa: BLE001
            raise ValueError(f"Invalid CUDA device index in '{token}'.") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} out of range (available count={torch.cuda.device_count()})."
            )
        devices.append(torch.device(f"cuda:{index}"))

    unique_devices: List[torch.device] = []
    seen = set()
    for dev in devices:
        if dev in seen:
            continue
        unique_devices.append(dev)
        seen.add(dev)

    return unique_devices or None


# Guard model construction so Hugging Face / torch meta init is not raced by
# multiple threads loading the tokenizer at the same time.
_TOKENIZER_INIT_LOCK = threading.Lock()


class EncoderWorker(threading.Thread):
    def __init__(
        self,
        worker_id: int,
        device: torch.device,
        tokenizer_kwargs: Dict[str, Any],
        encode_sample_rate: Optional[int],
        task_queue: "queue.Queue[Optional[EncodeJob]]",
        result_queue: "queue.Queue[Tuple[EncodeJob, Optional[torch.Tensor], Optional[BaseException]]]",
    ) -> None:
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.device = device
        self.tokenizer_kwargs = dict(tokenizer_kwargs)
        self.encode_sample_rate = encode_sample_rate
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self) -> None:
        torch.set_grad_enabled(False)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        # serialize tokenizer init to avoid meta device errors
        with _TOKENIZER_INIT_LOCK:
            tokenizer = AudioTokenizer(
                device=self.device, **self.tokenizer_kwargs
            )

        default_encode_sr = getattr(
            tokenizer, "encode_sample_rate", tokenizer.sample_rate
        )
        if (
            self.encode_sample_rate is not None
            and self.encode_sample_rate != default_encode_sr
        ):
            tokenizer.encode_sample_rate = self.encode_sample_rate
            tokenizer.sample_rate = self.encode_sample_rate

        while True:
            job = self.task_queue.get()
            if job is None:
                self.task_queue.task_done()
                break

            try:
                waveform = job.waveform.to(tokenizer.device, non_blocking=True)
                with torch.no_grad():
                    codes = tokenizer.encode(waveform)
                if codes.ndim >= 3:
                    codes = codes.squeeze(0)
                job.waveform = None
                self.result_queue.put((job, codes.cpu(), None))
            except Exception as exc:  # noqa: BLE001
                job.waveform = None
                LOGGER.warning(
                    "Encoder worker %d failed for utt_id=%s: %s",
                    self.worker_id,
                    getattr(job, "utt_id", "unknown"),
                    exc,
                    exc_info=True,
                )
                self.result_queue.put((job, None, exc))
            finally:
                self.task_queue.task_done()


class EncoderPool:
    def __init__(
        self,
        devices: List[torch.device],
        tokenizer_kwargs: Dict[str, Any],
        encode_sample_rate: Optional[int],
        max_queue_size: int,
    ) -> None:
        queue_size = max_queue_size if max_queue_size and max_queue_size > 0 else 0
        self._task_queue: "queue.Queue[Optional[EncodeJob]]" = queue.Queue(queue_size)
        self._result_queue: "queue.Queue[Tuple[EncodeJob, Optional[torch.Tensor], Optional[BaseException]]]" = queue.Queue()
        self._workers = [
            EncoderWorker(
                worker_id=idx,
                device=device,
                tokenizer_kwargs=tokenizer_kwargs,
                encode_sample_rate=encode_sample_rate,
                task_queue=self._task_queue,
                result_queue=self._result_queue,
            )
            for idx, device in enumerate(devices)
        ]
        for worker in self._workers:
            worker.start()

    def submit(self, job: EncodeJob) -> None:
        self._task_queue.put(job)

    def get_result(
        self, block: bool = False, timeout: Optional[float] = None
    ) -> Optional[Tuple[EncodeJob, Optional[torch.Tensor], Optional[BaseException]]]:
        try:
            return self._result_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def join(self) -> None:
        self._task_queue.join()

    def shutdown(self) -> None:
        for _ in self._workers:
            self._task_queue.put(None)
        for worker in self._workers:
            worker.join()


def prepare_dataset(
    args: argparse.Namespace, prefilter_fn: Optional[Callable[[Dict], bool]] = None
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    LOGGER.info("Preparing dataset with args: %s", args)

    data_files = parse_data_files_arg(args.data_files)
    dataset = hf_load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        data_files=data_files,
        split=args.split,
        streaming=args.streaming,
    )
    dataset = expand_metadata_columns(dataset, args)

    if (
        not args.streaming
        and args.id_column
        and args.id_column not in dataset.column_names
    ):
        raise KeyError(
            f"id column '{args.id_column}' not found in dataset columns: {dataset.column_names}"
        )

    # Optional EN-specific prefilter
    if prefilter_fn is not None:
        try:
            orig_len = len(dataset) if not args.streaming else None
        except Exception:
            orig_len = None
        num_proc = None if args.streaming else max(1, args.hf_num_proc or 1)
        dataset = dataset.filter(
            prefilter_fn, desc="prefilter(en)", num_proc=num_proc
        )
        try:
            filt_len = len(dataset) if not args.streaming else None
        except Exception:
            filt_len = None

        dropped = None
        if orig_len is not None and filt_len is not None:
            dropped = orig_len - filt_len
        elif hasattr(prefilter_fn, "stats"):
            st = getattr(prefilter_fn, "stats")
            dropped = st.get("dropped")
            if filt_len is None:
                filt_len = st.get("kept")
        LOGGER.info(
            "Prefilter stats: original=%s kept=%s dropped=%s",
            orig_len,
            filt_len,
            dropped,
        )

    output_root = Path(args.output_dir).absolute()
    dirs = ensure_dirs(output_root)

    devices = resolve_encoder_devices(args.encoder_devices)
    if not devices:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            devices = [torch.device("cuda:0")]
        else:
            devices = [torch.device("cpu")]

    device_types = {dev.type for dev in devices}
    if len(device_types) > 1:
        raise ValueError(
            f"Mixing encoder device types ({device_types}) is not supported."
        )

    multi_device_mode = len(devices) > 1 and next(iter(device_types)) == "cuda"
    if len(devices) > 1 and not multi_device_mode:
        LOGGER.warning(
            "Multi-device mode currently only supports CUDA devices; falling back to single-device encoding on %s.",
            devices[0],
        )
        devices = [devices[0]]
        multi_device_mode = False

    tokenizer_kwargs = {
        "backend": "xcodec2",
        "model_name": args.tokenizer_model,
    }

    encode_sr: Optional[int] = None
    encoder_pool: Optional[EncoderPool] = None
    tokenizer: Optional[AudioTokenizer] = None

    if multi_device_mode:
        LOGGER.info(
            "Multi-device encoder activated across %d GPUs: %s",
            len(devices),
            ", ".join(str(dev) for dev in devices),
        )
        probe_tokenizer = AudioTokenizer(
            device=torch.device("cpu"), **tokenizer_kwargs
        )
        default_encode_sr = getattr(
            probe_tokenizer, "encode_sample_rate", probe_tokenizer.sample_rate
        )
        encode_sr = args.codec_sample_rate or default_encode_sr
        if args.codec_sample_rate and args.codec_sample_rate != default_encode_sr:
            LOGGER.info(
                "Overriding codec input sample rate: tokenizer expects %d, using %d per argument.",
                default_encode_sr,
                args.codec_sample_rate,
            )
        if encode_sr != default_encode_sr:
            probe_tokenizer.encode_sample_rate = encode_sr
            probe_tokenizer.sample_rate = encode_sr
        del probe_tokenizer
        queue_size = args.encoder_queue_size if multi_device_mode else 0
        encoder_pool = EncoderPool(
            devices=devices,
            tokenizer_kwargs=tokenizer_kwargs,
            encode_sample_rate=encode_sr,
            max_queue_size=queue_size,
        )
    else:
        device = devices[0]
        tokenizer = AudioTokenizer(device=device, **tokenizer_kwargs)
        default_encode_sr = getattr(
            tokenizer, "encode_sample_rate", tokenizer.sample_rate
        )
        encode_sr = args.codec_sample_rate or default_encode_sr
        if args.codec_sample_rate and args.codec_sample_rate != default_encode_sr:
            LOGGER.info(
                "Overriding codec input sample rate: tokenizer expects %d, using %d per argument.",
                default_encode_sr,
                args.codec_sample_rate,
            )
            tokenizer.encode_sample_rate = encode_sr
            tokenizer.sample_rate = encode_sr
        LOGGER.info("Encoder device: %s", device)

    iterator: Iterable = dataset
    processed = 0
    skipped = 0

    try:
        dataset_len = len(dataset) if not args.streaming else None
        if dataset_len is not None and args.max_samples is not None:
            dataset_len = min(dataset_len, args.max_samples)
    except (TypeError, NotImplementedError):
        dataset_len = None

    progress_bar = tqdm(
        total=dataset_len,
        unit="sample",
        dynamic_ncols=True,
        desc=f"{args.split}",
    )

    rng = random.Random(args.seed)
    dataset_label = sanitize_utt_id(args.dataset_name.replace("/", "_"))

    target_splits = [args.split]
    if args.valid_ratio and args.valid_ratio > 0:
        target_splits.append(args.valid_split_name)

    manifest_states = {}

    def init_manifest_state(split_name: str):
        manifest_path = dirs["manifest"] / f"{split_name}.txt"
        existing_ids = set()
        ordered_ids = []
        next_index = 0
        if args.overwrite and manifest_path.exists():
            manifest_path.unlink()
        elif manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as mf:
                for line in mf:
                    parts = line.strip().split("\t")
                    if parts:
                        existing_id = parts[0]
                        existing_ids.add(existing_id)
                        ordered_ids.append(existing_id)
            prefix = f"{dataset_label}_{split_name}_"
            numerical_suffixes = []
            for item in existing_ids:
                if item.startswith(prefix):
                    suffix = item[len(prefix) :]
                    if suffix.isdigit():
                        numerical_suffixes.append(int(suffix))
            if numerical_suffixes:
                next_index = max(numerical_suffixes) + 1
        manifest_states[split_name] = {
            "manifest_path": manifest_path,
            "existing_ids": existing_ids,
            "next_index": next_index,
            "resume_queue": deque(ordered_ids),
        }

    for split_name in target_splits:
        init_manifest_state(split_name)

    seen_ids = set().union(
        *(state["existing_ids"] for state in manifest_states.values())
    )
    split_counts = {split_name: 0 for split_name in target_splits}
    resume_skipped = {split_name: 0 for split_name in target_splits}
    resume_reencoded = {split_name: 0 for split_name in target_splits}
    pending_jobs = 0

    def drain_results(block: bool = False) -> None:
        nonlocal processed, skipped, pending_jobs
        if encoder_pool is None:
            return

        timeout = 0.1 if block else 0.0
        while True:
            result = encoder_pool.get_result(block=block, timeout=timeout)
            if result is None:
                break

            job, codes, error = result
            pending_jobs = max(0, pending_jobs - 1)

            if error is not None or codes is None or codes.numel() == 0:
                skipped += 1
                if error is not None:
                    LOGGER.warning(
                        "Skipping sample idx=%d (utt_id=%s) due to encoder error.",
                        job.idx,
                        job.utt_id,
                    )
                else:
                    LOGGER.warning(
                        "Skipping sample idx=%d (utt_id=%s) because encoder returned empty codes.",
                        job.idx,
                        job.utt_id,
                    )
                continue

            try:
                token_len = write_outputs(
                    dirs=dirs,
                    split=job.dest_split,
                    utt_id=job.utt_id,
                    tokens=codes,
                    text=job.text,
                    overwrite=job.allow_overwrite,
                )
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                LOGGER.warning(
                    "Failed to write outputs for idx=%d, utt_id=%s: %s",
                    job.idx,
                    job.utt_id,
                    exc,
                    exc_info=True,
                )
                continue

            split_counts[job.dest_split] += 1
            processed += 1
            if job.forced:
                resume_reencoded[job.dest_split] += 1
            if processed % args.log_every == 0:
                LOGGER.info(
                    "Processed %d samples (skipped %d) – last utt_id=%s len=%d tokens",
                    processed,
                    skipped,
                    job.utt_id,
                    token_len,
                )

    def process_example(current_idx: int, example) -> None:
        nonlocal skipped, processed, pending_jobs
        try:
            if args.valid_ratio and args.valid_ratio > 0:
                assign_valid = rng.random() < args.valid_ratio
                dest_split = args.valid_split_name if assign_valid else args.split
            else:
                dest_split = args.split

            state = manifest_states[dest_split]
            resume_queue = state["resume_queue"]
            forced_utt_id = None
            allow_overwrite = args.overwrite

            if resume_queue:
                forced_utt_id = resume_queue.popleft()
                text_path = dirs["text"] / f"{forced_utt_id}.txt"
                codes_path = dirs["codes"] / f"{forced_utt_id}.txt"
                files_exist = text_path.exists() and codes_path.exists()
                if not args.overwrite and files_exist:
                    resume_skipped[dest_split] += 1
                    return
                allow_overwrite = True

            base_prefix = f"{dataset_label}_{dest_split}"

            if forced_utt_id is not None:
                utt_id = forced_utt_id
            else:
                if args.id_column and example.get(args.id_column) is not None:
                    raw_id = str(example[args.id_column])
                    utt_id = sanitize_utt_id(raw_id)
                    base_id = utt_id
                    suffix = 1
                    while not utt_id or utt_id in seen_ids:
                        if utt_id in state["existing_ids"] and not args.overwrite:
                            LOGGER.debug(
                                "Skipping existing sample with utt_id=%s in split=%s",
                                utt_id,
                                dest_split,
                            )
                            resume_skipped[dest_split] += 1
                            utt_id = None
                            break
                        utt_id = sanitize_utt_id(f"{base_id}_{suffix}")
                        suffix += 1
                    if not utt_id:
                        return
                else:
                    while True:
                        utt_id = f"{base_prefix}_{state['next_index']:08d}"
                        state["next_index"] += 1
                        if utt_id not in seen_ids:
                            break

            seen_ids.add(utt_id)
            state["existing_ids"].add(utt_id)

            text_value = example.get(args.text_column, "")
            if text_value is None or not str(text_value).strip():
                skipped += 1
                LOGGER.debug(
                    "Skipping idx=%s, utt_id=%s due to missing raw text",
                    current_idx,
                    utt_id,
                )
                return
            text = normalize(str(text_value))
            if not text.strip():
                skipped += 1
                LOGGER.debug(
                    "Skipping idx=%s, utt_id=%s due to empty normalized text",
                    current_idx,
                    utt_id,
                )
                return
            if not is_allowed():
                skipped += 1
                LOGGER.debug(
                    "Skipping idx=%s, utt_id=%s due to invalid characters after normalization",
                    current_idx,
                    utt_id,
                )
                return

            waveform, sr = load_audio_tensor(example.get(args.audio_column), encode_sr)
            if waveform.numel() == 0:
                skipped += 1
                return

            duration_sec = waveform.shape[-1] / sr
            MAX_DURATION_SEC = 30.0
            MIN_DURATION_SEC = 0.1
            if duration_sec > MAX_DURATION_SEC or duration_sec < MIN_DURATION_SEC:
                LOGGER.info(
                    "Skipping idx%s, utt_id=%s due to duration %.2f sec (min=%.2f, max=%.2f)",
                    current_idx,
                    utt_id,
                    duration_sec,
                    MIN_DURATION_SEC,
                    MAX_DURATION_SEC,
                )
                skipped += 1
                return

            if encoder_pool is not None:
                job = EncodeJob(
                    job_id=current_idx,
                    idx=current_idx,
                    dest_split=dest_split,
                    utt_id=utt_id,
                    text=text,
                    allow_overwrite=allow_overwrite,
                    forced=forced_utt_id is not None,
                    waveform=waveform,
                )
                encoder_pool.submit(job)
                pending_jobs += 1
            else:
                waveform = waveform.to(tokenizer.device)
                codes = tokenizer.encode(waveform)
                codes = codes.squeeze(0)
                if codes.numel() == 0:
                    skipped += 1
                    return

                token_len = write_outputs(
                    dirs=dirs,
                    split=dest_split,
                    utt_id=utt_id,
                    tokens=codes.cpu(),
                    text=text,
                    overwrite=allow_overwrite,
                )
                split_counts[dest_split] += 1
                processed += 1
                if forced_utt_id is not None:
                    resume_reencoded[dest_split] += 1

                if processed % args.log_every == 0:
                    LOGGER.info(
                        "Processed %d samples (skipped %d) – last utt_id=%s len=%d tokens",
                        processed,
                        skipped,
                        utt_id,
                        token_len,
                    )
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            LOGGER.warning(
                "Skipping sample idx=%d due to error: %s",
                current_idx,
                exc,
                exc_info=True,
            )

    if not args.streaming:
        total_examples = dataset_len if dataset_len is not None else len(dataset)

        for idx in range(total_examples):
            if encoder_pool is not None:
                drain_results(block=False)

            if (
                args.max_samples is not None
                and (processed + pending_jobs) >= args.max_samples
            ):
                break

            progress_bar.update(1)
            try:
                example = dataset[idx]
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                LOGGER.warning(
                    "Skipping sample idx=%d due to decode error: %s",
                    idx,
                    exc,
                    exc_info=True,
                )
                continue

            process_example(idx, example)
    else:
        data_iter = iter(iterator)
        idx = 0
        while True:
            try:
                example = next(data_iter)
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                LOGGER.warning(
                    "Skipping sample during dataset iteration due to decode error: %s",
                    exc,
                    exc_info=True,
                )
                continue

            if encoder_pool is not None:
                drain_results(block=False)

            if (
                args.max_samples is not None
                and (processed + pending_jobs) >= args.max_samples
            ):
                break

            progress_bar.update(1)
            process_example(idx, example)
            idx += 1
    if encoder_pool is not None:
        while pending_jobs > 0:
            drain_results(block=False)
        encoder_pool.join()
        encoder_pool.shutdown()

    progress_bar.close()

    for split_name in target_splits:
        LOGGER.info(
            "Split %s → newly added=%d, resumed skipped=%d, resumed reencoded=%d (total manifest entries=%d, file=%s)",
            split_name,
            split_counts[split_name],
            resume_skipped[split_name],
            resume_reencoded[split_name],
            len(manifest_states[split_name]["existing_ids"]),
            manifest_states[split_name]["manifest_path"],
        )
    LOGGER.info("Overall processed=%d, skipped=%d", processed, skipped)


def derive_group_key(record: SampleRecord, mode: str) -> str:
    if mode == "speaker":
        return record.speaker
    raise ValueError(f"Unsupported group mode: {mode}")


def generate_neighbors(
    args: argparse.Namespace, filter_fn: Callable[[Dict], bool]
) -> None:
    output_root = Path(args.output_dir).absolute()
    manifest_root = output_root / "manifest_final"
    if not manifest_root.exists():
        raise FileNotFoundError(
            f"Manifest directory not found: {manifest_root}. Run without --neighbors-only first."
        )

    target_splits = [args.split]
    if args.valid_ratio and args.valid_ratio > 0:
        target_splits.append(args.valid_split_name)

    manifest_maps: Dict[str, Dict[str, object]] = {}
    total_manifest_entries = 0
    for split_name in target_splits:
        manifest_path = manifest_root / f"{split_name}.txt"
        manifest_maps[split_name] = load_manifest_map(manifest_path)
        total_manifest_entries += len(manifest_maps[split_name])
    if total_manifest_entries == 0:
        LOGGER.error("No manifest entries loaded; aborting neighbor generation.")
        return

    data_files = parse_data_files_arg(args.data_files)
    dataset = hf_load_dataset(
        args.dataset_name,
        name=args.dataset_config,
        data_files=data_files,
        split=args.split,
        streaming=args.streaming,
    )
    dataset = expand_metadata_columns(dataset, args)

    # Drop heavy audio column *before* filtering to avoid decoding audio during filter.
    if not args.streaming and args.audio_column in dataset.column_names:
        try:
            dataset = dataset.remove_columns([args.audio_column])
        except Exception:
            LOGGER.warning("Could not remove audio column before filter", exc_info=True)

    dataset = dataset.filter(filter_fn)

    per_split_records: Dict[str, List[SampleRecord]] = {
        split: [] for split in target_splits
    }
    unmatched_manifest_maps = {split: manifest_maps[split].copy() for split in target_splits}
    seen_ids = set()
    dataset_label = sanitize_utt_id(args.dataset_name.replace("/", "_"))
    stats = {"processed_hits": 0, "manifest_misses": 0, "skipped_empty_text": 0}
    iterator: Iterable = dataset if args.streaming else tqdm(dataset, unit="sample", desc="replaying")

    for idx, example in enumerate(iterator):
        if args.max_samples is not None and stats["processed_hits"] >= args.max_samples:
            break

        text_val = example.get(args.text_column, "")
        if text_val is None or not str(text_val).strip():
            stats["skipped_empty_text"] += 1
            continue

        raw_id = get_field(example, args.id_column, args.metadata_column)
        base_id = (
            sanitize_utt_id(str(raw_id))
            if raw_id is not None
            else sanitize_utt_id(f"{dataset_label}_{args.split}_{idx:08d}")
        )
        utt_id = base_id
        suffix = 1
        while not utt_id or utt_id in seen_ids:
            utt_id = sanitize_utt_id(f"{base_id}_{suffix}")
            suffix += 1
        seen_ids.add(utt_id)

        # Manifests are stored with a shard prefix (first two hex of md5).
        # Recreate that key so we can look up the corresponding manifest entry.
        shard_id = hashlib.md5(utt_id.encode("utf-8")).hexdigest()[:2]
        manifest_key = f"{shard_id}/{utt_id}"

        dest_split = None
        manifest_entry = None
        for split_name in target_splits:
            # Prefer the sharded key but fall back to the raw key for
            # compatibility with any pre-existing manifests.
            candidate = manifest_maps[split_name].get(manifest_key)
            if candidate is None:
                candidate = manifest_maps[split_name].get(utt_id)
            if candidate is not None:
                dest_split = split_name
                manifest_entry = candidate
                break

        if manifest_entry is None or dest_split is None:
            stats["manifest_misses"] += 1
            continue

        unmatched_manifest_maps[dest_split].pop(manifest_entry.utt_id, None)
        stats["processed_hits"] += 1

        speaker_raw = get_field(example, args.speaker_column, args.metadata_column) or ""
        speaker = str(speaker_raw or "").strip()
        duration_val = get_field(example, args.duration_column, args.metadata_column)
        if duration_val is None:
            duration_sec = float(manifest_entry.token_len) / float(args.encodec_sr)
        else:
            try:
                duration_sec = float(duration_val)
            except (TypeError, ValueError):
                duration_sec = float(manifest_entry.token_len) / float(args.encodec_sr)

        per_split_records[dest_split].append(
            SampleRecord(
                utt_id=manifest_entry.utt_id,
                speaker=speaker,
                duration_sec=duration_sec,
                split=dest_split,
            )
        )

    LOGGER.info(
        "Replay complete. Hits=%d, skipped_empty_text=%d, manifest_misses=%d",
        stats["processed_hits"],
        stats["skipped_empty_text"],
        stats["manifest_misses"],
    )
    for split_name, remaining in unmatched_manifest_maps.items():
        if remaining:
            LOGGER.warning(
                "Unmatched manifest entries for split=%s: %d (first 5) %s",
                split_name,
                len(remaining),
                list(remaining.keys())[:5],
            )

    neighbor_dir = ensure_neighbor_dir(output_root, args.neighbor_folder, args.overwrite)
    total_written = 0
    total_empty = 0
    rng2 = random.Random(args.seed)

    for split_name, records in per_split_records.items():
        if not records:
            continue

        groups: Dict[str, List[SampleRecord]] = defaultdict(list)
        for record in records:
            groups[derive_group_key(record, args.group_by)].append(record)
        for group_records in groups.values():
            group_records.sort(key=lambda r: r.utt_id)

        for record in tqdm(records, desc=f"writing neighbors {split_name}"):
            group_key = derive_group_key(record, args.group_by)
            neighbors = [n for n in groups[group_key] if n.utt_id != record.utt_id]
            neighbor_path = neighbor_dir / f"{record.utt_id}.txt"
            neighbor_path.parent.mkdir(parents=True, exist_ok=True)
            if not neighbors:
                neighbor_path.touch(exist_ok=True)
                total_empty += 1
                continue

            if args.distance_metric == "duration_diff":
                neighbors.sort(key=lambda n: abs(n.duration_sec - record.duration_sec))

            if args.max_neighbors_per_utt is not None and len(neighbors) > args.max_neighbors_per_utt:
                limit = args.max_neighbors_per_utt
                stride = len(neighbors) / float(limit)
                sampled: List[SampleRecord] = []
                for i in range(limit):
                    pos = int(rng2.uniform(i * stride, (i + 1) * stride))
                    pos = max(0, min(len(neighbors) - 1, pos))
                    sampled.append(neighbors[pos])
                neighbors = sampled

            with neighbor_path.open("w", encoding="utf-8") as nf:
                for neighbor in neighbors:
                    distance_val = (
                        abs(neighbor.duration_sec - record.duration_sec)
                        if args.distance_metric == "duration_diff"
                        else 0.0
                    )
                    nf.write(
                        f"{neighbor.utt_id}.txt\t{distance_val:.3f}\t{neighbor.duration_sec:.3f}\n"
                    )
            total_written += 1

    LOGGER.info(
        "Neighbor generation complete. Non-empty=%d, empty=%d, output_dir=%s",
        total_written,
        total_empty,
        neighbor_dir,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    filter_fn = build_filter_fn(args)

    if not args.neighbors_only:
        prepare_dataset(args, prefilter_fn=filter_fn)
        # Log post-pass filter stats (first stage)
        st = getattr(filter_fn, "stats", {})
        LOGGER.info(
            "Prefilter total(seen=%s, kept=%s, dropped=%s)",
            st.get("seen"),
            st.get("kept"),
            st.get("dropped"),
        )
    generate_neighbors(args, filter_fn)


if __name__ == "__main__":
    main()