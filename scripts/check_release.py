"""
Purpose: Validate the Git-visible StackDiff public release workspace for
oversized files, private artifacts, local absolute paths, and non-public
checkpoint references. The script honors .gitignore by checking tracked files
and unignored untracked files only, then prints release issues and exits with a
non-zero status when cleanup is required.
Related files: .gitignore, README.md, models/checkpoints/README.md, and
configs/**/*.yml.
CLI usage: python scripts/check_release.py (run from the repository root; no
arguments are required).
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 20 * 1024 * 1024
LOCAL_PATH_MARKER = "/" + "app/moire"
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".tar",
    ".gz",
    ".zip",
    ".docx",
    ".pptx",
}
FORBIDDEN_DIRS = {
    "__pycache__",
    ".venv",
    "venv",
    "trash",
    "tmp",
    "DCGAN-tensorflow",
    "pgdgan",
}
ALLOWED_CHECKPOINT_PATHS = {
    "models/checkpoints/ReS2/ema_0.9999_200000.pt",
    "models/checkpoints/MoS2/ema_0.9999_200000.pt",
    "models/checkpoints/MoTe2/ema_0.9999_200000.pt",
    "models/checkpoints/1T-TaS2/ema_0.9999_200000.pt",
    "models/checkpoints/1H-TaS2/ema_0.9999_200000.pt",
    "models/checkpoints/CrBr3/ema_0.9999_200000.pt",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard", "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        files: list[Path] = []
        for raw in result.stdout.split(b"\0"):
            if raw:
                path = ROOT / raw.decode("utf-8")
                if path.is_file():
                    files.append(path)
        return files
    except Exception:
        files = []
        for path in ROOT.rglob("*"):
            if ".git" in path.parts:
                continue
            if path.is_file():
                files.append(path)
        return files


def is_public_runtime_config(path: Path) -> bool:
    rel_path = rel(path)
    return rel_path.startswith("configs/") and rel_path.endswith((".yml", ".yaml"))


def main() -> int:
    problems: list[str] = []

    for path in iter_files():
        for part in path.relative_to(ROOT).parts[:-1]:
            if part in FORBIDDEN_DIRS:
                problems.append(f"forbidden directory: {rel(path.parent)}")
                break
        name = path.name
        suffixes = {s.lower() for s in path.suffixes}
        if path.stat().st_size > MAX_BYTES:
            problems.append(f"large file >20MB: {rel(path)}")
        if suffixes & FORBIDDEN_SUFFIXES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden artifact: {rel(path)}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if LOCAL_PATH_MARKER in text:
            problems.append(f"local absolute path reference: {rel(path)}")
        if path.suffix in {".yml", ".yaml"} and is_public_runtime_config(path) and "models/checkpoints/" in text:
            for allowed in ALLOWED_CHECKPOINT_PATHS:
                text = text.replace(allowed, "")
            if "models/checkpoints/" in text:
                problems.append(f"non-public checkpoint path in config: {rel(path)}")

    if problems:
        print("Release check failed:")
        for item in problems:
            print(f"- {item}")
        return 1

    print("Release check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
