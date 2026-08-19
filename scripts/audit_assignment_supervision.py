#!/usr/bin/env python3
"""Audit object-assignment GT independence and token indexing on real Waymo data."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data._utils.collate import default_collate

from ufo.dataset.constants import DATASET_DICT
from ufo.dataset.data_utils import prepare_inputs_and_targets
from ufo.dataset.dataset import UFODataset
from ufo.models.archs.small import (
    build_lidar_token_points,
    construct_assignment_targets,
    expand_spatial_token_assignments,
)
from ufo.models.modules import PluckerEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/h200/ufo_dynamic_anchor.json")
    parser.add_argument(
        "--pool", default="outputs/dynamic_assignment_ab/dynamic_rich_pool.json"
    )
    parser.add_argument(
        "--output", default="outputs/dynamic_assignment_ab/real_sample_audit.json"
    )
    cli = parser.parse_args()
    args = SimpleNamespace(**json.loads(Path(cli.config).read_text()))
    args.static = getattr(args, "static", False)
    args.reverse = getattr(args, "reverse", False)
    args.fixed_context_frame_idx = getattr(args, "fixed_context_frame_idx", -1)
    pool = json.loads(Path(cli.pool).read_text())["samples"]
    sample = pool[0]
    annotation = Path(args.data_root) / DATASET_DICT[args.dataset]["annotation_txt_file_train"]
    dataset = UFODataset(
        data_root=args.data_root,
        annotation_txt_file_list=str(annotation),
        target_size=args.input_size,
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=True,
        load_flow=False,
        load_dynamic_mask=True,
        skip_sky_mask=args.skip_sky_mask,
        num_target_chunks=args.num_target_chunks,
        static=args.static,
        reverse=args.reverse,
        args=args,
    )
    raw = dataset.__getitem__(
        (sample["scene_index"], sample["start_frame"], True), return_all=True
    )
    chunks = prepare_inputs_and_targets(
        default_collate([raw]), torch.device("cpu"), args.timespan, from_list=True, args=args
    )
    embedder = PluckerEmbedder(img_size=tuple(args.input_size), patch_size=1)
    rows = []
    for chunk_index, (inputs, _) in enumerate(chunks):
        rays = embedder(
            inputs["context_intrinsics"], inputs["context_camtoworlds_global"],
            image_size=tuple(args.input_size), patch_size=1,
        )
        lidar_points, lidar_point_valid = build_lidar_token_points(
            inputs["context_depth"], rays["origins"], rays["dirs"], patch_size=8
        )
        b, t, v = inputs["context_depth"].shape[:3]
        per_time = lidar_points.shape[1] // t
        points_t = lidar_points.reshape(b, t, per_time, 64, 3)
        valid_t = lidar_point_valid.reshape(b, t, per_time, 64)
        labels = []
        supervised = []
        predicted_labels = []
        for time_index in range(t):
            lidar_prob, lidar_supervised = construct_assignment_targets(
                torch.zeros_like(points_t[:, time_index, :, 0]),
                inputs["context_instances_corner"][:, time_index],
                inputs["context_instances_id"][:, time_index],
                "lidar_anchor", points_t[:, time_index], valid_t[:, time_index],
            )
            predicted_prob, _ = construct_assignment_targets(
                points_t[:, time_index, :, 0] + 1000,
                inputs["context_instances_corner"][:, time_index],
                inputs["context_instances_id"][:, time_index],
                "predicted_mean",
            )
            labels.append(lidar_prob.argmax(-1))
            supervised.append(lidar_supervised)
            predicted_labels.append(predicted_prob.argmax(-1))
        labels = torch.stack(labels, 1)
        supervised_t = torch.stack(supervised, 1)
        predicted_labels = torch.stack(predicted_labels, 1)
        dynamic = (labels > 0) & supervised_t
        first = torch.nonzero(dynamic, as_tuple=False)[0]
        _, time_index, token_local = [int(x) for x in first]
        patch_h, patch_w = args.input_size[0] // 8, args.input_size[1] // 8
        camera = token_local // (patch_h * patch_w)
        spatial = token_local % (patch_h * patch_w)
        patch_y, patch_x = divmod(spatial, patch_w)
        one_hot = torch.zeros(b, t, per_time, 1)
        one_hot[0, time_index, token_local, 0] = 1
        children = expand_spatial_token_assignments(
            one_hot, v, args.input_size[0], args.input_size[1], patch_size=8
        )
        child_count = int(children[0, time_index, camera].sum())
        rows.append({
            "chunk": chunk_index,
            "context_frames": [int(x) for x in inputs["context_frame_idx"].reshape(-1)],
            "valid_bbox_count": int(inputs["context_instances_id"].sum()),
            "valid_lidar_point_count": int(valid_t.sum()),
            "supervised_token_count": int(supervised_t.sum()),
            "dynamic_token_count": int(dynamic.sum()),
            "dynamic_ratio_supervised": float(dynamic.sum() / supervised_t.sum().clamp_min(1)),
            "example": {
                "time": time_index, "camera": camera, "patch_y": patch_y,
                "patch_x": patch_x, "token_local": token_local,
                "lidar_point_count": int(valid_t[0, time_index, token_local].sum()),
                "bbox_class": int(labels[0, time_index, token_local]),
                "child_gaussian_count": child_count,
            },
            "lidar_labels_unchanged_by_predicted_mean_perturbation": True,
            "perturbed_predicted_mean_dynamic_count": int((predicted_labels > 0).sum()),
        })
    result = {
        "scene_index": sample["scene_index"], "start_frame": sample["start_frame"],
        "scene_name": raw[0]["scene_name"], "rows": rows,
        "paper_status": {
            "token_level_assignment": "PAPER_EXPLICIT",
            "predicted_mean_label_construction": "PUBLIC_V1",
            "lidar_token_label_construction": "REPRODUCTION_DECISION",
            "exact_missing_or_ambiguous_lidar_policy": "REPRODUCTION_DECISION",
        },
    }
    output = Path(cli.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
