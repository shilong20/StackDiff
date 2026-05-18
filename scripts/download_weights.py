"""
【作用概述】按材料名下载公开发布的推荐 EMA 权重到 configs 使用的默认路径；当前 manifest 中的 URL/SHA256 可在权重上传到 Hugging Face Hub 或 Zenodo 后补齐。
【关联说明】文件/模块：docs/download_weights.md（权重说明）；configs/separate/*.yml 与 configs/sample/*.yml（默认权重路径）；models/checkpoints/README.md（本地放置约定）。
【命令行用法】python scripts/download_weights.py --material ReS2（参数：--material=材料名或 all；--force=覆盖已有文件）
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "ReS2": {
        "path": "models/checkpoints/ReS2_online/ema_0.9999_200000.pt",
        "url": "",
        "sha256": "",
    },
    "MoS2": {
        "path": "models/checkpoints/MoS2/ema_0.9999_200000.pt",
        "url": "",
        "sha256": "",
    },
    "MoTe2": {
        "path": "models/checkpoints/MoTe2/ema_0.9999_200000.pt",
        "url": "",
        "sha256": "",
    },
    "TaS2": {
        "path": "models/checkpoints/TaS2/ema_0.9999_200000.pt",
        "url": "",
        "sha256": "",
    },
    "WS2": {
        "path": "models/checkpoints/WS2_online/ema_0.9999_150000.pt",
        "url": "",
        "sha256": "",
    },
    "CrI3": {
        "path": "models/checkpoints/CrI3/ema_0.9999_200000.pt",
        "url": "",
        "sha256": "",
    },
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def selected_materials(name: str) -> list[str]:
    if name == "all":
        return list(WEIGHTS)
    if name not in WEIGHTS:
        raise SystemExit(f"Unknown material: {name}. Choices: {', '.join(WEIGHTS)} or all")
    return [name]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download released StackDiff checkpoints.")
    parser.add_argument("--material", default="all", help="Material name or all.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing checkpoint files.")
    args = parser.parse_args()

    missing_urls: list[str] = []
    for material in selected_materials(str(args.material)):
        info = WEIGHTS[material]
        target = ROOT / info["path"]
        url = str(info["url"])
        expected_sha = str(info["sha256"])

        if not url:
            missing_urls.append(material)
            continue

        if target.exists() and not args.force:
            print(f"{material}: exists, skip {target.relative_to(ROOT)}")
        else:
            print(f"{material}: downloading to {target.relative_to(ROOT)}")
            download(url, target)

        if expected_sha:
            actual_sha = sha256sum(target)
            if actual_sha != expected_sha:
                raise SystemExit(f"{material}: SHA256 mismatch: expected {expected_sha}, got {actual_sha}")
            print(f"{material}: SHA256 ok")

    if missing_urls:
        print("The following materials do not have download URLs yet:")
        for material in missing_urls:
            print(f"- {material}: {WEIGHTS[material]['path']}")
        print("Fill scripts/download_weights.py after uploading checkpoints.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
