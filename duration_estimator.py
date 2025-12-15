

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

import os
import re
from typing import Optional, Tuple

import torchaudio

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # deterministic prediction
except ImportError:
    DetectorFactory = None
    LangDetectException = Exception
    detect = None

try:
    import nltk
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    from g2p_en import G2p
except ImportError:
    G2p = None

try:
    import pyopenjtalk
except ImportError:
    pyopenjtalk = None

try:
    from pypinyin import Style, lazy_pinyin
except ImportError:
    Style = None
    lazy_pinyin = None


# Seconds per phoneme defaults and clamps per language
SPP_DEFAULT = {"en": 0.085, "ja": 0.10, "zh": 0.27, "other": 0.11}
SPP_MINMAX = {
    "en": (0.06, 0.12),
    "ja": (0.07, 0.15),
    "zh": (0.18, 0.36),
    "other": (0.07, 0.18),
}
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 120.0

_g2p_en = None


def _safe_detect_language(text: str) -> str:
    """Return coarse language code: en / ja / zh / other."""
    text = text.strip()
    if not text:
        return "other"

    # Quick character heuristics as a fallback
    def _heuristic() -> Optional[str]:
        has_hira_kata = re.search(r"[\u3040-\u30ff]", text)
        has_cjk = re.search(r"[\u4e00-\u9fff]", text)
        if has_hira_kata:
            return "ja"
        if has_cjk:
            return "zh"
        return None

    # langdetect when available
    if detect is not None:
        try:
            lang = detect(text)
            if lang.startswith("ja"):
                return "ja"
            if lang.startswith("zh") or lang in {"yue"}:
                return "zh"
            if lang.startswith("en"):
                return "en"
        except LangDetectException:
            pass

    guess = _heuristic()
    if guess:
        return guess
    return "en"  # fallback to English pacing


def _phoneme_count_en(text: str) -> int:
    global _g2p_en
    if G2p is None:
        return len(text)
    if _g2p_en is None:
        _g2p_en = G2p()
    phonemes = _g2p_en(text)
    return len([p for p in phonemes if p and p not in {" ", "<pad>", "<s>", "</s>", "<unk>"}])


def _phoneme_count_ja(text: str) -> int:
    if pyopenjtalk is None:
        return len(text)
    ph = pyopenjtalk.g2p(text)
    return len([p for p in ph.split(" ") if p and p not in {"pau", "sil"}])


def _phoneme_count_zh(text: str) -> int:
    if lazy_pinyin is None or Style is None:
        return len(text)
    syllables = lazy_pinyin(text, style=Style.NORMAL, neutral_tone_with_five=True)
    return len([s for s in syllables if s and re.search(r"[a-zA-Z]", s)])


def _phoneme_count(text: str, lang: str) -> int:
    if lang == "en":
        return _phoneme_count_en(text)
    if lang == "ja":
        return _phoneme_count_ja(text)
    if lang == "zh":
        return _phoneme_count_zh(text)
    # other languages: rough fallback on characters
    return max(len(text), 1)


def _punctuation_bonus_sec(text: str) -> float:
    """
    Add small pauses for punctuation/ellipsis.
    - Major stops inside a sentence (.,!?。！？) add more than minor stops.
    - Trailing sentence-final punctuation is ignored to avoid double counting.
    - Ellipsis (“…” or “...”) and long dash (“—”/“--”) add extra pause.
    """
    t = text.strip()
    major_chars = ".!?。！？"
    minor_chars = "、，,;；:"

    major = len(re.findall(r"[.!?。！？]", t))
    minor = len(re.findall(r"[、，,;；:]", t))

    # Don’t count sentence-final major punctuation
    if t and t[-1] in major_chars:
        major = max(0, major - 1)

    # Ellipsis and dash pauses
    ellipsis = len(re.findall(r"(…|\.\.\.)", t))
    dash_pause = len(re.findall(r"(—|--)", t))

    major_bonus = major * 0.40
    minor_bonus = minor * 0.20
    ellipsis_bonus = ellipsis * 1.0
    dash_bonus = dash_pause * 0.12

    return min(10.0, major_bonus + minor_bonus + ellipsis_bonus + dash_bonus)


def _clamp(val: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, val))


def detect_language(text: str) -> str:
    """Expose the internal detector for reuse in inference pipelines."""
    return _safe_detect_language(text)


def _canonicalize_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    lang = lang.lower()
    if lang.startswith("ja"):
        return "ja"
    if lang.startswith("zh") or lang in {"yue"}:
        return "zh"
    if lang.startswith("en"):
        return "en"
    return lang


def estimate_duration(
    target_text: str,
    reference_speech: Optional[str] = None,
    reference_transcript: Optional[str] = None,
    target_lang: Optional[str] = None,
    reference_lang: Optional[str] = None,
) -> float:
    """
    Estimate target duration (seconds) using phoneme-aware pacing.

    - If reference audio + its transcript are available, derive seconds-per-phoneme
      from them and apply to target text.
    - Otherwise, fall back to language-specific default seconds-per-phoneme.
    """
    target_text = target_text or ""
    ref_has_audio = reference_speech and os.path.isfile(reference_speech)

    # Language for target text (optional override to avoid repeated detection)
    tgt_lang = _canonicalize_lang(target_lang) or (_safe_detect_language(target_text) if target_text else "en")
    tgt_phonemes = max(_phoneme_count(target_text, tgt_lang), 1)

    spp = SPP_DEFAULT.get(tgt_lang, SPP_DEFAULT["other"])

    if ref_has_audio:
        try:
            info = torchaudio.info(reference_speech)
            audio_duration = info.num_frames / info.sample_rate
        except Exception:
            audio_duration = None

        if audio_duration and audio_duration > 0:
            # Prefer the provided transcript; if absent, reuse target text
            ref_text = reference_transcript or target_text
            ref_lang = _canonicalize_lang(reference_lang) or _safe_detect_language(ref_text)
            ref_phonemes = max(_phoneme_count(ref_text, ref_lang), 1)
            spp = audio_duration / ref_phonemes
            spp = _clamp(spp, SPP_MINMAX.get(ref_lang, SPP_MINMAX["other"]))

    # Compute duration
    if ref_has_audio:
        punct_bonus = _punctuation_bonus_sec(target_text) * 0.3
    else:
        punct_bonus = _punctuation_bonus_sec(target_text)
    duration = tgt_phonemes * spp + punct_bonus
    duration = max(MIN_DURATION_SEC, min(duration, MAX_DURATION_SEC))
    return duration