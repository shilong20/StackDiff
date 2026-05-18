#!/usr/bin/env python3
"""
【作用概述】批量缩放图片，使短边不超过给定长度（默认 1024）；默认仅缩小不放大，可选择输出到镜像目录或原地覆盖。
【关联说明】文件/模块：无强耦合；可用于预处理任意数据目录（含 generate_sample 或实验图像）。
【命令行用法】python tools/resize_short_edge.py --root <dir_or_file> --target-short 1024（参数：--recursive=递归；--out-dir=输出目录；--allow-upscale=允许放大）。
"""

import argparse
import concurrent.futures
import sys
from pathlib import Path
from typing import Iterable, Tuple, Optional

from PIL import Image


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_SUFFIXES:
            yield root
        return

    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                yield p
    else:
        for p in root.glob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                yield p


def compute_new_size(width: int, height: int, target_short: int, allow_upscale: bool) -> Tuple[int, int]:
    short_edge = min(width, height)
    if short_edge == 0:
        return width, height
    if not allow_upscale and short_edge <= target_short:
        return width, height
    scale = target_short / short_edge
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_w, new_h


def save_image(img: Image.Image, dst: Path, src_format: str, force_format: Optional[str] = None) -> None:
    save_kwargs = {}
    fmt = force_format or src_format
    # Pillow needs explicit format sometimes when saving to non-matching suffixes
    if fmt is None:
        fmt = Image.registered_extensions().get(dst.suffix.lower(), None)

    # Reasonable defaults per format to keep size smaller
    if (force_format or dst.suffix.lower()) in {"JPEG", ".jpg", ".jpeg"}:
        save_kwargs.update({"quality": 90, "optimize": True})
        fmt = "JPEG"
    elif (force_format or dst.suffix.lower()) in {"PNG", ".png"}:
        save_kwargs.update({"optimize": True, "compress_level": 6})
        fmt = "PNG"
    elif (force_format or dst.suffix.lower()) in {"TIFF", ".tif", ".tiff"}:
        # Use deflate compression to reduce size while staying lossless
        save_kwargs.update({"compression": "tiff_deflate"})
        fmt = "TIFF"

    img.save(dst, format=fmt, **save_kwargs)


def process_one(path: Path, out_dir: Path, target_short: int, allow_upscale: bool, dry_run: bool, convert_to: Optional[str], delete_source: bool) -> Tuple[Path, str]:
    rel = path
    if out_dir is not None:
        try:
            rel = path.relative_to(common_root)
        except Exception:
            rel = path.name
    dst = path if out_dir is None else (out_dir / rel)

    # if converting format, adjust destination suffix accordingly
    if convert_to:
        suffix_map = {"png": ".png", "jpeg": ".jpg", "tiff": ".tif", "bmp": ".bmp"}
        new_suffix = suffix_map.get(convert_to.lower())
        if new_suffix:
            dst = dst.with_suffix(new_suffix)
    if out_dir is not None:
        dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(path) as im:
            width, height = im.size
            new_w, new_h = compute_new_size(width, height, target_short, allow_upscale)
            if (new_w, new_h) == (width, height):
                # No resize needed. If conversion or out path differs, still write.
                need_convert = bool(convert_to)
                dest_differs = dst.resolve() != path.resolve()
                if not need_convert and not dest_differs:
                    return path, f"skip ({width}x{height})"

                action_prefix = "would " if dry_run else ""
                if dry_run:
                    conv_note = f" to {convert_to.upper()}" if convert_to else ""
                    move_note = " (to new location)" if dest_differs and not need_convert else ""
                    return path, f"{action_prefix}convert {width}x{height}{conv_note}{move_note}"

                force_format = convert_to.upper() if convert_to else None
                save_image(im, dst, im.format, force_format)
                if delete_source and dest_differs:
                    try:
                        path.unlink()
                    except Exception:
                        pass
                return path, f"ok {width}x{height}{(' to ' + convert_to.upper()) if convert_to else ''}"

            # Use high-quality downscale
            resample = getattr(Image, "Resampling", Image).LANCZOS
            resized = im.resize((new_w, new_h), resample=resample)

            action_prefix = "would " if dry_run else ""

            if dry_run:
                # report convert target if any
                conv_note = f", to {convert_to.upper()}" if convert_to else ""
                return path, f"{action_prefix}resize {width}x{height} -> {new_w}x{new_h}{conv_note}"

            # choose forced format if converting
            force_format = None
            if convert_to:
                force_format = convert_to.upper()
            save_image(resized, dst, im.format, force_format)

            # delete source if requested and destination path differs
            if delete_source and dst.resolve() != path.resolve():
                try:
                    path.unlink()
                except Exception:
                    pass

            return path, f"ok {width}x{height} -> {new_w}x{new_h}{(' to ' + (convert_to.upper())) if convert_to else ''}"
    except Exception as e:
        return path, f"error: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Resize images by short edge")
    parser.add_argument("--root", type=str, required=True, help="Root file or directory to process")
    parser.add_argument("--short-edge", type=int, default=1024, help="Target short edge length")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--allow-upscale", action="store_true", help="Allow upscaling when short edge < target")
    parser.add_argument("--out-dir", type=str, default=None, help="Optional output directory; overwrite in-place if omitted")
    parser.add_argument("--to-format", type=str, choices=["png", "jpeg", "tiff", "bmp"], default=None, help="Convert images to target format (e.g., png)")
    parser.add_argument("--delete-source", action="store_true", help="Delete original file after successful convert when extension changes")
    parser.add_argument("--dry-run", action="store_true", help="List actions without writing files")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    global common_root  # used for relative path resolution
    common_root = Path(args.root).resolve()

    root_path = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    images = list(iter_images(root_path, recursive=True if root_path.is_dir() else False or args.recursive))
    if not images:
        print(f"No images found under {root_path}")
        return 0

    total = len(images)
    print(f"Found {total} images. Target short edge: {args.short_edge}. Out: {out_dir or 'in-place'}")

    processed = 0
    skipped = 0
    failed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(
                process_one,
                p,
                out_dir,
                args.short_edge,
                args.allow_upscale,
                args.dry_run,
                args.to_format,
                args.delete_source,
            )
            for p in images
        ]
        for fut in concurrent.futures.as_completed(futures):
            path, status = fut.result()
            if status.startswith("ok") or status.startswith("would resize"):
                processed += 1
            elif status.startswith("skip"):
                skipped += 1
            else:
                failed += 1
            print(f"{path}: {status}")

    print(f"Done. processed={processed}, skipped={skipped}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
