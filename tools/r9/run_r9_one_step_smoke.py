#!/usr/bin/env python3
"""Run one native main.py optimizer step on scene 621 for R9 wiring validation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
WAYMO_ROOT = Path(
    "/inspire/hdd/global_user/guoluosong-253108120129/"
    "aaai/long_dggt/data/waymo/processed"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=int, default=621)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/root/miniconda3/envs/dggt_data/bin/python"),
    )
    args = parser.parse_args()

    train_list = REPO_ROOT / "data/UFO_paper/scene_list/waymo_train.txt"
    annotations = [line for line in train_list.read_text().splitlines() if line]
    annotation = Path(annotations[args.scene_id])
    if not annotation.is_file():
        raise FileNotFoundError(annotation)

    output_root = REPO_ROOT / "outputs/r9_one_step_smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r9_scene621_data_") as directory:
        data_root = Path(directory)
        (data_root / "scene_list").mkdir()
        (data_root / "datasets").mkdir()
        (data_root / "scene_list/waymo_train.txt").write_text(str(annotation) + "\n")
        (data_root / "datasets/waymo").symlink_to(WAYMO_ROOT, target_is_directory=True)

        command = [
            str(args.python),
            "main.py",
            "--config",
            "configs/experiments/ufo_scene621_sam_object_detail_r9_scratch20k.json",
            "--data_root",
            str(data_root),
            "--sam_track_root",
            str(REPO_ROOT / "data/r9_sam_tracks"),
            "--batch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--num_workers",
            "0",
            "--num_iterations",
            "1",
            "--ckpt_every_n_iters",
            "1000000",
            "--project",
            "r9_one_step_smoke",
            "--exp_name",
            "scene621_native_sam_input",
            "--output_dir",
            str(REPO_ROOT / "outputs"),
            "--skip_initial_validation",
            "--skip_final_evaluation",
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "offline",
            }
        )
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)

    print("R9_ONE_STEP_FORWARD_BACKWARD_PASS scene_id=621 optimizer_steps=1")


if __name__ == "__main__":
    main()
