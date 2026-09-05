#!/usr/bin/env python3
"""Exercise the production R9 SAM mask loader on deterministic samples."""

import argparse
import random
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ufo.dataset.sam_tracks import load_sam_track_mask


CAMERAS = (1, 0, 2)
DEFAULT_WAYMO_ROOT = Path(
    "/inspire/hdd/global_user/guoluosong-253108120129/"
    "aaai/long_dggt/data/waymo/processed"
)


def numeric_key(value):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def frame_indices(image_dir, camera):
    pattern = re.compile(rf"^(\d+)_{camera}\.jpg$")
    return sorted(
        int(match.group(1))
        for path in image_dir.iterdir()
        if (match := pattern.match(path.name))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo-root", type=Path, default=DEFAULT_WAYMO_ROOT)
    parser.add_argument("--sam-root", type=Path, default=Path("data/r9_sam_tracks"))
    parser.add_argument("--scene", action="append")
    parser.add_argument("--num-scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    training_root = args.waymo_root / "training"
    complete_scenes = sorted(
        {
            path.parent.parent.name
            for path in args.sam_root.glob("*/*/.r9_sam2_done.json")
        },
        key=numeric_key,
    )
    scenes = [str(value) for value in args.scene] if args.scene else complete_scenes
    if not args.scene and len(scenes) > args.num_scenes:
        scenes = sorted(
            random.Random(args.seed).sample(scenes, args.num_scenes),
            key=numeric_key,
        )
    if not scenes:
        raise RuntimeError("No completed scenes available for contract test")

    checked = 0
    for scene in scenes:
        namespaces = []
        for slot, camera in enumerate(CAMERAS):
            indices = frame_indices(training_root / scene / "images", camera)
            if not indices:
                raise RuntimeError(f"No RGB frames: scene={scene} camera={camera}")
            foreground = set()
            for frame_idx in sorted({indices[0], indices[len(indices) // 2], indices[-1]}):
                mask = load_sam_track_mask(
                    args.sam_root,
                    scene,
                    camera,
                    frame_idx,
                    target_size=(160, 240),
                    camera_slot=slot,
                )
                assert mask.dtype == torch.int64
                assert tuple(mask.shape) == (160, 240)
                unique = torch.unique(mask)
                assert int(unique[0]) == 0
                foreground.update(int(value) for value in unique[unique > 0].tolist())
            lower = (slot + 1) * 10000
            upper = (slot + 2) * 10000
            if foreground and not all(lower < value < upper for value in foreground):
                raise RuntimeError(
                    f"Offset namespace violation: scene={scene} camera={camera}"
                )
            namespaces.append(foreground)
            checked += 3
        for left in range(len(namespaces)):
            for right in range(left + 1, len(namespaces)):
                if not namespaces[left].isdisjoint(namespaces[right]):
                    raise RuntimeError(f"Camera namespaces collide in scene {scene}")
        print(f"PASS scene={scene} cameras=1,0,2 target_size=160x240")
    print(f"R9_LOADER_CONTRACT_PASS scenes={len(scenes)} masks_checked={checked}")


if __name__ == "__main__":
    main()
