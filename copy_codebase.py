

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
import shutil
import fnmatch

def parse_gitignore(gitignore_path):
    """Parse a .gitignore file and return a list of patterns."""
    patterns = []
    with open(gitignore_path, "r") as f:
        for line in f:
            # Ignore comments and blank lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle wildcards and directory separators
            patterns.append(line)
    return patterns

def file_matches_patterns(relative_path, patterns):
    """Check if a file matches any of the patterns in .gitignore."""
    relative_posix = relative_path.replace(os.sep, "/")
    for pattern in patterns:
        pattern_posix = pattern.replace("\\", "/")
        if pattern_posix.endswith("/"):
            if relative_posix.startswith(pattern_posix.rstrip("/")):
                return True
            continue
        if fnmatch.fnmatch(relative_posix, pattern_posix):
            return True
    return False

def copy_codebase(src, dst, max_size_mb=5, gitignore_path=None):
    """ Copy files from src to dst, skipping files larger than max_size_mb and matching .gitignore patterns. """
    if gitignore_path and os.path.exists(gitignore_path):
        patterns = parse_gitignore(gitignore_path)
    else:
        patterns = []
    print("patterns to ignore: ", patterns)
    os.makedirs(dst, exist_ok=True)
    dst_abs = os.path.abspath(dst)
    for root, dirs, files in os.walk(src):
        abs_root = os.path.abspath(root)
        if abs_root.startswith(dst_abs):
            continue
        # Prevent descending into destination folder and ignored directories.
        dirs[:] = [
            d for d in dirs
            if not os.path.abspath(os.path.join(root, d)).startswith(dst_abs)
        ]
        dirs[:] = [
            d for d in dirs
            if not file_matches_patterns(os.path.relpath(os.path.join(root, d), src), patterns)
        ]
        for file in files:
            file_path = os.path.join(root, file)
            abs_file_path = os.path.abspath(file_path)
            if abs_file_path.startswith(dst_abs):
                continue
            relative_path = os.path.relpath(file_path, src)
            if relative_path.startswith("runs" + os.sep) or relative_path == "runs":
                continue
            dst_path = os.path.join(dst, relative_path)
            # ignore .git because of permission issues
            if "/.git/" in file_path:
                continue

            # Check .gitignore patterns
            if file_matches_patterns(relative_path, patterns):
                # print(f"Skipping {file_path} because it matches a pattern in .gitignore")
                continue

            # Check file size
            if os.path.getsize(file_path) > max_size_mb * 1024 * 1024:
                print(f"Skipping {file_path} because it's larger than {max_size_mb}MB")
                continue


            # Make sure the destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(file_path, dst_path)