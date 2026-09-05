#!/usr/bin/env python3
"""Audit R9 SAM tracks directly against Waymo RGB frames."""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


DEFAULT_WAYMO_ROOT = Path(
    "/inspire/hdd/global_user/guoluosong-253108120129/"
    "aaai/long_dggt/data/waymo/processed"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/hdd/global_user/guoluosong-253108120129/"
    "yx-ufo/data/r9_sam_tracks"
)
CAMERAS = ("1", "0", "2")
INT64_INFO = np.iinfo(np.int64)


def numeric_key(value):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo-root", type=Path, default=DEFAULT_WAYMO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--no-write-manifest", action="store_true")
    parser.add_argument("--require-full", action="store_true")
    return parser.parse_args()


def collect_indices(image_dir, camera):
    pattern = re.compile(rf"^(\d+)_{re.escape(camera)}\.jpg$")
    indices = []
    for path in image_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def int64_conversion_error(mask):
    """Return why converting this mask to int64 would change its ID values."""
    if np.issubdtype(mask.dtype, np.signedinteger):
        if mask.dtype.itemsize <= np.dtype(np.int64).itemsize:
            return None
    elif np.issubdtype(mask.dtype, np.unsignedinteger):
        if mask.dtype.itemsize < np.dtype(np.int64).itemsize:
            return None
        if mask.size == 0 or int(mask.max()) <= INT64_INFO.max:
            return None
    elif np.issubdtype(mask.dtype, np.floating):
        if mask.size == 0:
            return None
        finite = np.isfinite(mask)
        if not np.all(finite):
            return "contains non-finite values"
        if np.any(mask != np.floor(mask)):
            return "contains fractional instance IDs"
        minimum = float(mask.min())
        maximum = float(mask.max())
        # float64(INT64_MAX) rounds up to 2**63, so use the previous
        # representable float as the largest value accepted here.
        float_int64_max = np.nextafter(np.float64(INT64_INFO.max), -np.inf)
        if minimum >= INT64_INFO.min and maximum <= float_int64_max:
            return None
    return f"values cannot be represented safely as int64 (dtype={mask.dtype})"


def audit_pair(scene, camera, indices, output_root):
    pair_dir = output_root / scene / camera
    mask_dir = pair_dir / "mask_data"
    marker = pair_dir / ".r9_sam2_done.json"
    expected_names = {f"mask_{frame_idx:06d}.npy" for frame_idx in indices}
    actual_paths = list(mask_dir.glob("mask_*.npy")) if mask_dir.is_dir() else []
    actual_names = {path.name for path in actual_paths}
    errors = []

    if not indices:
        errors.append("no RGB frames for camera")

    if not marker.is_file():
        errors.append("missing done marker")
    else:
        try:
            record = json.loads(marker.read_text())
            if record.get("num_frames") != len(indices):
                errors.append("done marker num_frames mismatch")
            if record.get("mask_count") != len(indices):
                errors.append("done marker mask_count mismatch")
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid done marker: {error}")

    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        errors.append(f"missing masks={len(missing)} first={missing[:3]}")
    if extra:
        errors.append(f"extra masks={len(extra)} first={extra[:3]}")

    for frame_idx in indices:
        path = mask_dir / f"mask_{frame_idx:06d}.npy"
        if not path.is_file():
            continue
        try:
            mask = np.load(path, mmap_mode="r", allow_pickle=False)
            if mask.ndim != 2:
                errors.append(f"{path.name}: ndim={mask.ndim}")
                continue
            if mask.size == 0:
                errors.append(f"{path.name}: empty mask")
                continue
            conversion_error = int64_conversion_error(mask)
            if conversion_error:
                errors.append(f"{path.name}: {conversion_error}")
                continue
            if np.any(mask < 0):
                errors.append(f"{path.name}: negative instance ID")
                continue
            if not np.any(mask == 0):
                errors.append(f"{path.name}: background 0 missing")
        except Exception as error:
            errors.append(f"{path.name}: load failed: {error!r}")

    return {
        "scene": scene,
        "camera": camera,
        "rgb_frame_count": len(indices),
        "mask_frame_count": len(actual_paths),
        "complete": not errors,
        "errors": errors,
    }


def main():
    args = parse_args()
    training_root = args.waymo_root / "training"
    if not training_root.is_dir():
        raise FileNotFoundError(training_root)
    if not args.output_root.is_dir():
        raise FileNotFoundError(args.output_root)

    all_scenes = sorted(
        [path.name for path in training_root.iterdir() if path.is_dir()],
        key=numeric_key,
    )
    scenes = [str(scene) for scene in args.scene] if args.scene else all_scenes
    started = time.time()
    pair_results = []
    total_expected_masks = 0
    total_actual_masks = 0

    for scene in scenes:
        image_dir = training_root / scene / "images"
        if not image_dir.is_dir():
            for camera in CAMERAS:
                pair_results.append({
                    "scene": scene,
                    "camera": camera,
                    "rgb_frame_count": 0,
                    "mask_frame_count": 0,
                    "complete": False,
                    "errors": [f"missing image directory: {image_dir}"],
                })
            continue
        for camera in CAMERAS:
            indices = collect_indices(image_dir, camera)
            result = audit_pair(scene, camera, indices, args.output_root)
            pair_results.append(result)
            total_expected_masks += result["rgb_frame_count"]
            total_actual_masks += result["mask_frame_count"]

    failed = [result for result in pair_results if not result["complete"]]
    complete_pairs = len(pair_results) - len(failed)
    manifest = {
        "waymo_root": str(args.waymo_root),
        "output_root": str(args.output_root),
        "total_scenes": len(scenes),
        "expected_pairs": len(pair_results),
        "complete_pairs": complete_pairs,
        "incomplete_pairs": len(failed),
        "total_expected_masks": total_expected_masks,
        "total_actual_masks": total_actual_masks,
        "failed_pairs": len(failed),
        "failures": failed,
        "elapsed_seconds": time.time() - started,
    }
    if not args.no_write_manifest:
        path = args.output_root / "preprocess_manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)

    for key in (
        "total_scenes",
        "expected_pairs",
        "complete_pairs",
        "incomplete_pairs",
        "total_expected_masks",
        "total_actual_masks",
        "failed_pairs",
    ):
        print(f"{key}: {manifest[key]}")
    for failure in failed[:20]:
        print(
            f"FAIL scene={failure['scene']} camera={failure['camera']} "
            f"errors={failure['errors'][:5]}"
        )

    if args.require_full:
        full_ok = (
            len(all_scenes) == 798
            and len(scenes) == 798
            and len(pair_results) == 2394
            and complete_pairs == 2394
            and not failed
        )
        if not full_ok:
            raise SystemExit(2)
    elif failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
