#!/usr/bin/env python3
"""Validate the Full Waymo dynamic-mixture sampling pool against annotations."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("offline_assets/data_contract/dynamic_rich_pool.json"),
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=Path("data/UFO_paper/scene_list/waymo_train.txt"),
    )
    parser.add_argument("--timespan", type=float, default=0.5)
    parser.add_argument("--num-target-chunks", type=int, default=4)
    args = parser.parse_args()

    payload = json.loads(args.pool.read_text())
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("dynamic rich pool must contain a non-empty samples list")

    annotation_paths = [
        Path(line.strip())
        for line in args.train_list.read_text().splitlines()
        if line.strip()
    ]
    if len(annotation_paths) != 798:
        raise RuntimeError(f"expected 798 Waymo annotations, got {len(annotation_paths)}")

    annotations = []
    for scene_index, path in enumerate(annotation_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        annotation = json.loads(path.read_text())
        if int(annotation["scene_id"]) != scene_index:
            raise RuntimeError(
                f"train-list index {scene_index} maps to scene_id={annotation['scene_id']}"
            )
        annotations.append(annotation)

    for sample_index, sample in enumerate(samples):
        if "scene_index" not in sample or "start_frame" not in sample:
            raise RuntimeError(f"sample {sample_index} lacks scene_index/start_frame")
        scene_index = int(sample["scene_index"])
        start_frame = int(sample["start_frame"])
        if not 0 <= scene_index < len(annotations):
            raise RuntimeError(f"sample {sample_index}: scene_index={scene_index} out of range")
        if start_frame < 0:
            raise RuntimeError(f"sample {sample_index}: negative start_frame={start_frame}")

        annotation = annotations[scene_index]
        if sample.get("scene_id") not in (None, annotation["scene_name"]):
            raise RuntimeError(
                f"sample {sample_index}: scene_id does not match train-list index"
            )
        frames_per_chunk = int(args.timespan * float(annotation["fps"]))
        required_frames = args.num_target_chunks * frames_per_chunk
        # UFODataset resamples when start + required_frames >= num_timesteps,
        # so an explicit rich-pool start must remain strictly below that bound.
        if start_frame + required_frames >= int(annotation["num_timesteps"]):
            raise RuntimeError(
                f"sample {sample_index}: incomplete window scene={scene_index} "
                f"start={start_frame} required={required_frames} "
                f"num_timesteps={annotation['num_timesteps']}"
            )

    print(
        "DYNAMIC_RICH_POOL_PASS "
        f"scenes={len(annotations)} samples={len(samples)} "
        f"timespan={args.timespan} chunks={args.num_target_chunks}"
    )


if __name__ == "__main__":
    main()
