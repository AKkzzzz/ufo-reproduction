#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


DEFAULT_WAYMO_ROOT = Path(
    "/inspire/hdd/global_user/guoluosong-253108120129/"
    "aaai/long_dggt/data/waymo/processed"
)

DEFAULT_GSAM_ROOT = Path(
    "/inspire/hdd3/project/intelligent-driving-agent/"
    "public/workspace/yx/Grounded-SAM-2"
)

DEFAULT_GSAM_PYTHON = Path(
    "/inspire/hdd3/project/intelligent-driving-agent/"
    "public/workspace/yx/.venv-groundedsam2/bin/python"
)

DEFAULT_OUTPUT_ROOT = Path("data/r9_sam_tracks")

# UFO 3-camera protocol.
DEFAULT_CAMERAS = ["1", "0", "2"]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--waymo-root", type=Path, default=DEFAULT_WAYMO_ROOT)
    p.add_argument("--gsam-root", type=Path, default=DEFAULT_GSAM_ROOT)
    p.add_argument("--gsam-python", type=Path, default=DEFAULT_GSAM_PYTHON)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    p.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Process only selected scene. Repeatable.",
    )
    p.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Process only selected camera. Repeatable.",
    )

    p.add_argument("--limit-scenes", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-debug", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


def numeric_scene_key(name):
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def collect_frames(scene_dir: Path, camera: str):
    image_dir = scene_dir / "images"

    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    regex = re.compile(rf"^(\d+)_{re.escape(camera)}\.jpg$")

    frames = []

    for p in image_dir.iterdir():
        m = regex.match(p.name)
        if m:
            frames.append((int(m.group(1)), p))

    frames.sort(key=lambda x: x[0])

    if not frames:
        raise RuntimeError(
            f"No camera {camera} frames found in {image_dir}"
        )

    return frames


def stage_frames(frames, stage_dir: Path):
    """
    Grounded-SAM2 expects a directory containing sequential JPEGs.

    We intentionally rename:
        Waymo: 000_0.jpg
        Stage: 000000.jpg

    so Grounded-SAM2 naturally emits:
        mask_000000.npy

    which exactly matches R9 loader contract.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, src in frames:
        dst = stage_dir / f"{frame_idx:06d}.jpg"

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        os.symlink(src.resolve(), dst)


def expected_masks(frames, output_dir: Path):
    mask_dir = output_dir / "mask_data"

    return [
        mask_dir / f"mask_{frame_idx:06d}.npy"
        for frame_idx, _ in frames
    ]


def validate_output(frames, output_dir: Path, verbose=True):
    expected = expected_masks(frames, output_dir)
    missing = [p for p in expected if not p.exists()]

    if missing:
        if verbose:
            print(
                f"[INVALID] {output_dir}: "
                f"{len(missing)}/{len(expected)} masks missing"
            )
            for p in missing[:5]:
                print("  missing:", p)
        return False, None

    # Inspect representative masks.
    sample_indices = sorted(set([
        0,
        len(expected) // 2,
        len(expected) - 1,
    ]))

    stats = []

    for i in sample_indices:
        p = expected[i]
        x = np.load(p)

        if x.ndim != 2:
            raise RuntimeError(
                f"Expected 2D track mask, got {x.shape}: {p}"
            )

        u = np.unique(x)

        stats.append({
            "file": p.name,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "min": int(x.min()) if x.size else 0,
            "max": int(x.max()) if x.size else 0,
            "unique_count": int(len(u)),
            "first_ids": [int(v) for v in u[:20]],
        })

    return True, stats


def done_path(output_dir: Path):
    return output_dir / ".r9_sam2_done.json"


def already_complete(frames, output_dir: Path):
    marker = done_path(output_dir)

    if not marker.exists():
        return False

    ok, _ = validate_output(frames, output_dir, verbose=False)
    return ok


def clear_partial_output(output_dir: Path):
    if not output_dir.exists():
        return

    for name in [
        "mask_data",
        "json_data",
        "result",
        "result_reverse",
    ]:
        p = output_dir / name
        if p.exists():
            shutil.rmtree(p)

    for p in output_dir.glob("*.mp4"):
        p.unlink()

    marker = done_path(output_dir)
    if marker.exists():
        marker.unlink()


def run_one(
    *,
    scene,
    camera,
    frames,
    output_dir,
    args,
):
    if already_complete(frames, output_dir) and not args.overwrite:
        print(
            f"[SKIP] scene={scene} camera={camera} "
            f"already complete"
        )
        return "skip"

    if args.overwrite or output_dir.exists():
        clear_partial_output(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            f"[DRY] scene={scene} camera={camera} "
            f"frames={len(frames)} -> {output_dir}"
        )
        return "dry"

    stage_parent = Path(
        tempfile.mkdtemp(prefix=f"r9sam_{scene}_{camera}_")
    )
    stage_dir = stage_parent / "frames"

    try:
        stage_frames(frames, stage_dir)

        output_video = output_dir / "tracking_debug.mp4"

        env = os.environ.copy()
        env["GSAM_VIDEO_DIR"] = str(stage_dir)
        env["GSAM_OUTPUT_DIR"] = str(output_dir.resolve())
        env["GSAM_OUTPUT_VIDEO"] = str(output_video.resolve())

        # Prevent accidental package leakage where possible.
        env["PYTHONNOUSERSITE"] = "1"

        script = args.gsam_root / "grounded_sam2_tracking_ufo.py"

        cmd = [
            str(args.gsam_python),
            str(script),
        ]

        print()
        print("=" * 80)
        print(
            f"[RUN] scene={scene} camera={camera} "
            f"frames={len(frames)}"
        )
        print("input :", stage_dir)
        print("output:", output_dir)
        print("python:", args.gsam_python)
        print("=" * 80)

        t0 = time.time()

        proc = subprocess.run(
            cmd,
            cwd=args.gsam_root,
            env=env,
        )

        elapsed = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"Grounded-SAM2 failed: "
                f"scene={scene} camera={camera} "
                f"returncode={proc.returncode}"
            )

        ok, stats = validate_output(
            frames,
            output_dir,
            verbose=True,
        )

        if not ok:
            raise RuntimeError(
                f"R9 mask contract failed: "
                f"scene={scene} camera={camera}"
            )

        record = {
            "scene": str(scene),
            "camera": str(camera),
            "num_frames": len(frames),
            "first_frame": int(frames[0][0]),
            "last_frame": int(frames[-1][0]),
            "elapsed_seconds": elapsed,
            "mask_samples": stats,
        }

        done_path(output_dir).write_text(
            json.dumps(record, indent=2) + "\n"
        )

        if not args.keep_debug:
            for name in [
                "json_data",
                "result",
                "result_reverse",
            ]:
                p = output_dir / name
                if p.exists():
                    shutil.rmtree(p)

            if output_video.exists():
                output_video.unlink()

        print(
            f"[PASS] scene={scene} camera={camera}: "
            f"{len(frames)} masks in {elapsed:.1f}s"
        )

        for stat in stats:
            print(
                " ",
                stat["file"],
                "shape=", stat["shape"],
                "unique=", stat["unique_count"],
                "ids=", stat["first_ids"][:10],
            )

        return "pass"

    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)


def main():
    args = parse_args()

    training_root = args.waymo_root / "training"

    assert training_root.is_dir(), training_root
    assert args.gsam_root.is_dir(), args.gsam_root
    assert args.gsam_python.exists(), args.gsam_python
    assert (
        args.gsam_root / "grounded_sam2_tracking_ufo.py"
    ).exists()

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.scene:
        scenes = [str(x) for x in args.scene]
    else:
        scenes = sorted(
            [
                p.name
                for p in training_root.iterdir()
                if p.is_dir()
            ],
            key=numeric_scene_key,
        )

    if args.limit_scenes is not None:
        scenes = scenes[:args.limit_scenes]

    cameras = args.camera or DEFAULT_CAMERAS

    print("Waymo root :", args.waymo_root)
    print("Output root:", args.output_root.resolve())
    print("Scenes     :", len(scenes))
    print("Cameras    :", cameras)

    summary = {
        "pass": 0,
        "skip": 0,
        "dry": 0,
        "fail": 0,
    }

    failures = []

    for scene in scenes:
        scene_dir = training_root / scene

        for camera in cameras:
            try:
                frames = collect_frames(
                    scene_dir,
                    camera,
                )

                out = (
                    args.output_root
                    / scene
                    / str(camera)
                )

                status = run_one(
                    scene=scene,
                    camera=camera,
                    frames=frames,
                    output_dir=out,
                    args=args,
                )

                summary[status] += 1

            except Exception as e:
                summary["fail"] += 1
                failures.append({
                    "scene": scene,
                    "camera": camera,
                    "error": repr(e),
                })

                print(
                    f"[FAIL] scene={scene} "
                    f"camera={camera}: {e}",
                    file=sys.stderr,
                )

                # Fail fast for smoke / explicitly requested scenes.
                if args.scene:
                    raise

    manifest = {
        "waymo_root": str(args.waymo_root),
        "output_root": str(args.output_root.resolve()),
        "cameras": cameras,
        "num_scenes": len(scenes),
        "summary": summary,
        "failures": failures,
    }

    manifest_path = (
        args.output_root / "preprocess_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print()
    print("=" * 80)
    print("PREPROCESS SUMMARY")
    print(json.dumps(summary, indent=2))
    print("manifest:", manifest_path)
    print("=" * 80)

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
