"""
【作用概述】检查 StackDiff 待发布工作区是否误包含大文件、模型权重、压缩包、办公文档、baseline 目录或本机绝对路径；输出问题列表并以非零状态码提示发布前需要处理。
【关联说明】文件/模块：.gitignore（发布忽略规则）；configs/*.yml（权重路径检查）；docs/download_weights.md（公开权重路径约定）。
【命令行用法】python scripts/check_release.py（参数：无需参数；在仓库根目录运行）
"""

from __future__ import annotations

import sys
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
    "models/checkpoints/TaS2/ema_0.9999_200000.pt",
    "models/checkpoints/WS2/ema_0.9999_150000.pt",
    "models/checkpoints/CrI3/ema_0.9999_200000.pt",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file():
            files.append(path)
    return files


def is_public_runtime_config(path: Path) -> bool:
    rel_path = rel(path)
    return rel_path.startswith("configs/separate/") or rel_path.startswith("configs/sample/")


def main() -> int:
    problems: list[str] = []

    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            problems.append(f"forbidden directory: {rel(path)}")

    for path in iter_files():
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
