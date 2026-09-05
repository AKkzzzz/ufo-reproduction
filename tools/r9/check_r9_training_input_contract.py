#!/usr/bin/env python3
"""Exercise the real UFODataset -> collate -> R9 input preparation path."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main import get_args_parser
from ufo.dataset.data_utils import prepare_inputs_and_targets
from ufo.dataset.dataset import UFODataset
from ufo.utils.config import merge_config_and_args


def assert_track_tensor(track_ids, scene_id, chunk_index):
    if track_ids.dtype != torch.int64:
        raise RuntimeError(f"chunk {chunk_index}: dtype={track_ids.dtype}")
    if track_ids.ndim != 5 or tuple(track_ids.shape[:1] + track_ids.shape[2:3]) != (1, 3):
        raise RuntimeError(
            f"chunk {chunk_index}: expected [B,T,3,H,W], got {tuple(track_ids.shape)}"
        )
    if tuple(track_ids.shape[-2:]) != (160, 240):
        raise RuntimeError(f"chunk {chunk_index}: shape={tuple(track_ids.shape)}")

    namespaces = []
    for slot in range(3):
        camera_ids = track_ids[:, :, slot]
        if not torch.any(camera_ids == 0):
            raise RuntimeError(f"chunk {chunk_index} camera_slot={slot}: background 0 missing")
        foreground = torch.unique(camera_ids[camera_ids > 0])
        lower = (slot + 1) * 10000
        upper = (slot + 2) * 10000
        if foreground.numel() and not torch.all((foreground > lower) & (foreground < upper)):
            raise RuntimeError(
                f"chunk {chunk_index} camera_slot={slot}: namespace violation"
            )
        namespaces.append(set(int(value) for value in foreground.tolist()))
    for left in range(3):
        for right in range(left + 1, 3):
            if not namespaces[left].isdisjoint(namespaces[right]):
                raise RuntimeError(
                    f"scene={scene_id} chunk={chunk_index}: camera namespaces collide"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/h200/ufo_r9_waymo_full_global64.json"),
    )
    parser.add_argument("--scene-id", type=int, default=621)
    parser.add_argument("--start-frame", type=int, default=0)
    args_cli = parser.parse_args()

    model_parser = get_args_parser()
    args = merge_config_and_args(
        model_parser,
        config_path=str(args_cli.config),
        cli_args=["--config", str(args_cli.config), "--batch_size", "1"],
    )
    if not args.load_sam_tracks:
        raise RuntimeError("full config must enable load_sam_tracks")

    train_list = Path(args.data_root) / "scene_list/waymo_train.txt"
    dataset = UFODataset(
        data_root=args.data_root,
        annotation_txt_file_list=str(train_list),
        target_size=tuple(args.input_size),
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        subset_indices=[args_cli.scene_id],
        load_depth=args.load_depth,
        load_flow=args.load_flow and not args.disable_train_flow_loading,
        load_dynamic_mask=args.load_dynamic_mask,
        skip_sky_mask=args.skip_sky_mask,
        num_target_chunks=args.num_target_chunks,
        static=args.static,
        reverse=args.reverse,
        args=args,
    )
    annotation = dataset.annotations[0]
    if int(annotation["scene_id"]) != args_cli.scene_id:
        raise RuntimeError("train-list index and annotation scene_id differ")
    expected_sam_scene = f"{args_cli.scene_id:03d}"
    if not (Path(args.sam_track_root) / expected_sam_scene).is_dir():
        raise FileNotFoundError(Path(args.sam_track_root) / expected_sam_scene)

    np.random.seed(0)
    original_getitem = dataset.__getitem__

    class FixedStartDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return original_getitem((0, args_cli.start_frame, False))

    batch = next(iter(DataLoader(FixedStartDataset(), batch_size=1, num_workers=0)))
    prepared = prepare_inputs_and_targets(
        batch,
        device=torch.device("cpu"),
        timespan=args.timespan,
        from_list=True,
        args=args,
    )
    if len(prepared) != args.num_target_chunks:
        raise RuntimeError(
            f"expected {args.num_target_chunks} recurrent chunks, got {len(prepared)}"
        )
    for chunk_index, (input_dict, target_dict) in enumerate(prepared):
        if "sam_track_ids" not in input_dict:
            raise RuntimeError(f"chunk {chunk_index}: input sam_track_ids missing")
        assert_track_tensor(input_dict["sam_track_ids"], args_cli.scene_id, chunk_index)
        if "target_sam_track_ids" not in target_dict:
            raise RuntimeError(f"chunk {chunk_index}: target sam_track_ids missing")
        assert_track_tensor(
            target_dict["target_sam_track_ids"], args_cli.scene_id, chunk_index
        )

    first_shape = tuple(prepared[0][0]["sam_track_ids"].shape)
    print(
        "R9_TRAINING_INPUT_CONTRACT_PASS "
        f"scene_id={args_cli.scene_id} sam_scene={expected_sam_scene} "
        f"segment={annotation['scene_name']} chunks={len(prepared)} "
        f"shape={first_shape} dtype=torch.int64"
    )


if __name__ == "__main__":
    main()
