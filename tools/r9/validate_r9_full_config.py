#!/usr/bin/env python3
"""Fail closed when the R9 Full Waymo configuration drifts."""

import argparse
import json
from pathlib import Path


EXPECTED = {
    "dataset": "waymo",
    "load_sam_tracks": True,
    "sam_track_root": "data/r9_sam_tracks",
    "dynamic_renderer_mode": "sam_object_fusion",
    "sam_canonical_fusion_impl": "feature_fusion",
    "num_iterations": 100000,
    "batch_size": 8,
    "warmup_iters": 5000,
    "lr": 1e-4,
    "min_lr": 1e-6,
    "lr_sched": "cosine",
    "ckpt_every_n_iters": 5000,
    "pose_free_training": False,
    "pose_free_camera_only": False,
    "pose_override_mode": "none",
    "intrinsics_override_mode": "none",
    "sam_r5_min_track_pixels": 16,
    "sam_r5_association_max_distance": 4.0,
    "sam_r5_association_min_overlap": 1,
    "sam_r9_hidden_dim": 256,
    "sam_r9_voxel_size": 0.12,
    "sam_r9_min_voxel_support": 1,
    "sam_r9_max_mean_residual": 0.08,
    "sam_r9_max_log_scale_residual": 0.35,
    "sam_r9_max_quat_residual": 0.2,
    "sam_r9_max_color_residual": 0.25,
    "sam_r9_spatial_prior": 0.5,
    "sam_r9_temporal_prior": 0.25,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--train-list",
        type=Path,
        default=Path("data/UFO_paper/scene_list/waymo_train.txt"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    errors = []
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {config.get(key)!r}")
    if "train_scene_indices" in config:
        errors.append("train_scene_indices must be absent for Full Waymo")
    for key in ("pose_override_sequence_dir", "intrinsics_override_sequence_dir"):
        if config.get(key) is not None:
            errors.append(f"{key} must be null")
    if config.get("resume_from") is not None or config.get("load_from") is not None:
        errors.append("full training must start from scratch")
    if config.get("auto_resume") is not False:
        errors.append("auto_resume must be false")
    if int(config.get("keep_n_ckpts", 0)) < 20:
        errors.append("keep_n_ckpts must retain all 5k checkpoints through 100k")
    train_scenes = [line.strip() for line in args.train_list.read_text().splitlines() if line.strip()]
    if len(train_scenes) != 798:
        errors.append(f"Waymo train list must have 798 scenes, got {len(train_scenes)}")
    if errors:
        raise RuntimeError("Invalid R9 Full config:\n" + "\n".join(errors))
    print("R9_FULL_CONFIG_PASS scenes=798 iterations=100000 native_camera=true scratch=true")


if __name__ == "__main__":
    main()
