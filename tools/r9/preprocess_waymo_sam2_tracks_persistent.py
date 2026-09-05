#!/usr/bin/env python3
"""Persistent single-GPU Grounded-SAM2 preprocessing for R9 Waymo tracks.

The tracking algorithm below is intentionally kept equivalent to
Grounded-SAM-2/grounded_sam2_tracking_ufo.py. Model construction is lifted out
of the per-video path, visualization is omitted, and staged frame positions are
mapped back to the original Waymo frame indices when masks are finalized.
"""

import argparse
import copy
import gc
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAYMO_ROOT = Path(
    "/inspire/hdd/global_user/guoluosong-253108120129/"
    "aaai/long_dggt/data/waymo/processed"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/r9_sam_tracks"
DEFAULT_GSAM_ROOT = REPO_ROOT / "third_party/Grounded-SAM-2"
DEFAULT_CHECKPOINT = DEFAULT_GSAM_ROOT / "checkpoints/sam2.1_hiera_large.pt"
DEFAULT_CAMERAS = ("1", "0", "2")
DEFAULT_TEXT = "car. truck. bus. motorcycle. bicycle. pedestrian."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo-root", type=Path, default=DEFAULT_WAYMO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gsam-root", type=Path, default=DEFAULT_GSAM_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--camera", action="append", choices=DEFAULT_CAMERAS)
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the deterministic shard assignment without loading models.",
    )
    return parser.parse_args()


def numeric_key(value):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def build_global_pairs(scene_names):
    scenes = sorted(scene_names, key=numeric_key)
    return [(scene, camera) for scene in scenes for camera in DEFAULT_CAMERAS]


def select_shard_pairs(global_pairs, num_shards, shard_index):
    if num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    return global_pairs[shard_index::num_shards]


def collect_frames(scene_dir, camera):
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    pattern = re.compile(rf"^(\d+)_{re.escape(camera)}\.jpg$")
    frames = []
    for path in image_dir.iterdir():
        match = pattern.match(path.name)
        if match:
            frames.append((int(match.group(1)), path))
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise RuntimeError(f"No camera {camera} frames in {image_dir}")
    indices = [frame_idx for frame_idx, _ in frames]
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Duplicate frame indices for camera {camera} in {image_dir}")
    return frames


def stage_frames(frames, stage_dir):
    stage_dir.mkdir(parents=True)
    for staged_idx, (_, source) in enumerate(frames):
        os.symlink(source.resolve(), stage_dir / f"{staged_idx:06d}.jpg")


def expected_mask_paths(frames, output_dir):
    return [
        output_dir / "mask_data" / f"mask_{frame_idx:06d}.npy"
        for frame_idx, _ in frames
    ]


def validate_masks(frames, output_dir):
    mask_dir = output_dir / "mask_data"
    expected = expected_mask_paths(frames, output_dir)
    actual = list(mask_dir.glob("mask_*.npy")) if mask_dir.is_dir() else []
    if len(actual) != len(frames) or any(not path.is_file() for path in expected):
        return False, []

    sample_positions = sorted({0, len(expected) // 2, len(expected) - 1})
    samples = []
    for position in sample_positions:
        path = expected[position]
        mask = np.load(path, allow_pickle=False)
        if mask.ndim != 2:
            raise RuntimeError(f"Mask must be 2D, got {mask.shape}: {path}")
        unique = np.unique(mask)
        if unique.size == 0 or unique[0] != 0:
            raise RuntimeError(f"Mask has no background ID 0: {path}")
        if np.issubdtype(mask.dtype, np.signedinteger) and int(mask.min()) < 0:
            raise RuntimeError(f"Mask has negative instance IDs: {path}")
        samples.append(
            {
                "file": path.name,
                "shape": list(mask.shape),
                "dtype": str(mask.dtype),
                "unique_ids": [int(value) for value in unique[:64]],
                "unique_count": int(unique.size),
            }
        )
    return True, samples


def completion_is_valid(frames, output_dir):
    marker = output_dir / ".r9_sam2_done.json"
    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    # The original one-video smoke marker predates mask_count. It is accepted
    # only after the same strict on-disk validation and upgraded by main().
    if record.get("mask_count", len(frames)) != len(frames) or record.get("num_frames") != len(frames):
        return False
    valid, _ = validate_masks(frames, output_dir)
    return valid


def clear_partial(output_dir):
    if not output_dir.exists():
        return
    for name in ("mask_data", "json_data", "result", "result_reverse"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
    for path in output_dir.glob("*.mp4"):
        path.unlink()
    for marker_name in (".r9_sam2_done.json", ".r9_sam2_done.json.tmp"):
        marker = output_dir / marker_name
        if marker.exists():
            marker.unlink()


def remove_debug_artifacts(output_dir):
    for name in ("json_data", "result", "result_reverse"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    for path in output_dir.glob("*.mp4"):
        path.unlink()


def write_completion_record(output_dir, scene, camera, frames, elapsed, samples):
    record = {
        "scene": scene,
        "camera": camera,
        "num_frames": len(frames),
        "first_frame": frames[0][0],
        "last_frame": frames[-1][0],
        "mask_count": len(frames),
        "elapsed_seconds": elapsed,
        "sample_masks": samples,
    }
    marker_tmp = output_dir / ".r9_sam2_done.json.tmp"
    marker_tmp.write_text(json.dumps(record, indent=2) + "\n")
    marker_tmp.replace(output_dir / ".r9_sam2_done.json")


class PersistentGroundedSAM2:
    def __init__(self, args):
        # Imports happen after adding the already-installed Grounded-SAM2 repo.
        sys.path.insert(0, str(args.gsam_root))
        import torch
        from PIL import Image
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

        self.torch = torch
        self.Image = Image
        self.MaskDictionaryModel = MaskDictionaryModel
        self.ObjectInfo = ObjectInfo
        self.args = args
        self.device = "cuda"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this preprocessing job")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Match the original script's process-wide CUDA bfloat16 autocast.
        self.autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.autocast.__enter__()

        print("Loading SAM2 video predictor once...", flush=True)
        self.video_predictor = build_sam2_video_predictor(
            args.model_cfg, str(args.checkpoint)
        )
        print("Loading SAM2 image predictor once...", flush=True)
        image_model = build_sam2(
            args.model_cfg, str(args.checkpoint), device=self.device
        )
        self.image_predictor = SAM2ImagePredictor(image_model)
        print("Loading GroundingDINO processor/model once...", flush=True)
        self.processor = AutoProcessor.from_pretrained(args.grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.grounding_model
        ).to(self.device)
        self.grounding_model.eval()
        print(
            f"Models ready; allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB",
            flush=True,
        )

    def process_video(self, frames, stage_dir, output_dir):
        torch = self.torch
        MaskDictionaryModel = self.MaskDictionaryModel
        ObjectInfo = self.ObjectInfo
        mask_dir = output_dir / "mask_data"
        json_dir = output_dir / "json_data"
        mask_dir.mkdir(parents=True)
        json_dir.mkdir(parents=True)
        frame_names = [f"{idx:06d}.jpg" for idx in range(len(frames))]
        inference_state = None

        try:
            inference_state = self.video_predictor.init_state(video_path=str(stage_dir))
            sam2_masks = MaskDictionaryModel()
            # The upstream dataclass defaults empty masks to 1080x1920. Waymo
            # frames here are 1280x1920, so bind empty output to this video.
            sam2_masks.mask_height = inference_state["video_height"]
            sam2_masks.mask_width = inference_state["video_width"]
            objects_count = 0
            frame_object_count = {}

            for start_frame_idx in range(0, len(frame_names), self.args.step):
                print(f"  detect/forward staged_frame={start_frame_idx}", flush=True)
                image = self.Image.open(stage_dir / frame_names[start_frame_idx]).convert("RGB")
                staged_base = f"{start_frame_idx:06d}"
                mask_dict = MaskDictionaryModel(
                    promote_type="mask", mask_name=f"mask_{staged_base}.npy"
                )
                mask_dict.mask_height = inference_state["video_height"]
                mask_dict.mask_width = inference_state["video_width"]
                inputs = self.processor(
                    images=image, text=self.args.text, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.grounding_model(**inputs)
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=0.25,
                    text_threshold=0.25,
                    target_sizes=[image.size[::-1]],
                )
                self.image_predictor.set_image(np.array(image))
                input_boxes = results[0]["boxes"]
                labels = results[0]["labels"]
                if input_boxes.shape[0] != 0:
                    masks, scores, logits = self.image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )
                    if masks.ndim == 2:
                        masks = masks[None]
                    elif masks.ndim == 4:
                        masks = masks.squeeze(1)
                    mask_dict.add_new_frame_annotation(
                        mask_list=torch.tensor(masks).to(self.device),
                        box_list=torch.tensor(input_boxes),
                        label_list=labels,
                    )
                else:
                    print(f"  no detection at staged_frame={start_frame_idx}", flush=True)
                    mask_dict = sam2_masks

                objects_count = mask_dict.update_masks(
                    tracking_annotation_dict=sam2_masks,
                    iou_threshold=0.8,
                    objects_count=objects_count,
                )
                frame_object_count[start_frame_idx] = objects_count

                if len(mask_dict.labels) == 0:
                    mask_dict.save_empty_mask_and_json(
                        str(mask_dir),
                        str(json_dir),
                        image_name_list=frame_names[
                            start_frame_idx : start_frame_idx + self.args.step
                        ],
                    )
                else:
                    self.video_predictor.reset_state(inference_state)
                    for object_id, object_info in mask_dict.labels.items():
                        self.video_predictor.add_new_mask(
                            inference_state,
                            start_frame_idx,
                            object_id,
                            object_info.mask,
                        )

                    video_segments = {}
                    for out_frame_idx, out_obj_ids, out_mask_logits in (
                        self.video_predictor.propagate_in_video(
                            inference_state,
                            max_frame_num_to_track=self.args.step,
                            start_frame_idx=start_frame_idx,
                        )
                    ):
                        frame_masks = MaskDictionaryModel()
                        for idx, out_obj_id in enumerate(out_obj_ids):
                            out_mask = out_mask_logits[idx] > 0.0
                            object_info = ObjectInfo(
                                instance_id=out_obj_id,
                                mask=out_mask[0],
                                class_name=mask_dict.get_target_class_name(out_obj_id),
                                logit=mask_dict.get_target_logit(out_obj_id),
                            )
                            object_info.update_box()
                            frame_masks.labels[out_obj_id] = object_info
                            frame_masks.mask_name = f"mask_{out_frame_idx:06d}.npy"
                            frame_masks.mask_height = out_mask.shape[-2]
                            frame_masks.mask_width = out_mask.shape[-1]
                        video_segments[out_frame_idx] = frame_masks
                        sam2_masks = copy.deepcopy(frame_masks)

                    for frame_masks in video_segments.values():
                        mask_img = torch.zeros(
                            frame_masks.mask_height, frame_masks.mask_width
                        )
                        for object_id, object_info in frame_masks.labels.items():
                            mask_img[object_info.mask == True] = object_id
                        np.save(
                            mask_dir / frame_masks.mask_name,
                            mask_img.numpy().astype(np.uint16),
                        )
                        frame_masks.to_json(
                            str(json_dir / frame_masks.mask_name.replace(".npy", ".json"))
                        )

                del image, inputs, outputs, results, mask_dict

            print("  reverse refinement", flush=True)
            start_object_id = 0
            object_info_dict = {}
            for frame_idx, current_object_count in frame_object_count.items():
                added_prompt_count = 0
                if frame_idx != 0:
                    self.video_predictor.reset_state(inference_state)
                    json_path = json_dir / f"mask_{frame_idx:06d}.json"
                    mask_path = mask_dir / f"mask_{frame_idx:06d}.npy"
                    json_data = MaskDictionaryModel().from_json(str(json_path))
                    mask_array = np.load(mask_path, allow_pickle=False)
                    for object_id in range(start_object_id + 1, current_object_count + 1):
                        object_mask = mask_array == object_id
                        if not object_mask.any() or object_id not in json_data.labels:
                            continue
                        object_info_dict[object_id] = json_data.labels[object_id]
                        self.video_predictor.add_new_mask(
                            inference_state, frame_idx, object_id, object_mask
                        )
                        added_prompt_count += 1
                start_object_id = current_object_count
                if added_prompt_count == 0:
                    continue

                for out_frame_idx, out_obj_ids, out_mask_logits in (
                    self.video_predictor.propagate_in_video(
                        inference_state,
                        max_frame_num_to_track=self.args.step * 2,
                        start_frame_idx=frame_idx,
                        reverse=True,
                    )
                ):
                    json_path = json_dir / f"mask_{out_frame_idx:06d}.json"
                    mask_path = mask_dir / f"mask_{out_frame_idx:06d}.npy"
                    json_data = MaskDictionaryModel().from_json(str(json_path))
                    mask_array = np.load(mask_path, allow_pickle=False)
                    for idx, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[idx] > 0.0).cpu()
                        if out_mask.sum() == 0:
                            continue
                        object_info = object_info_dict[out_obj_id]
                        object_info.mask = out_mask[0]
                        object_info.update_box()
                        json_data.labels[out_obj_id] = object_info
                        mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                        mask_array[object_info.mask] = out_obj_id
                    np.save(mask_path, mask_array)
                    json_data.to_json(str(json_path))

            # Staged names are positional; rename them to original Waymo indices.
            for staged_idx, (original_idx, _) in enumerate(frames):
                if staged_idx == original_idx:
                    continue
                staged_mask = mask_dir / f"mask_{staged_idx:06d}.npy"
                original_mask = mask_dir / f".mapped_mask_{original_idx:06d}.npy"
                staged_mask.rename(original_mask)
            for original_idx, _ in frames:
                mapped = mask_dir / f".mapped_mask_{original_idx:06d}.npy"
                if mapped.exists():
                    mapped.rename(mask_dir / f"mask_{original_idx:06d}.npy")
        finally:
            if inference_state is not None:
                self.video_predictor.reset_state(inference_state)
                inference_state.clear()
            self.image_predictor.reset_predictor()
            del inference_state
            gc.collect()
            torch.cuda.empty_cache()


def write_manifest(path, args, total_scenes, global_pair_count, assigned_pair_count, completed, failures, total_masks, elapsed):
    manifest = {
        "waymo_root": str(args.waymo_root),
        "output_root": str(args.output_root),
        "total_scenes": total_scenes,
        "global_expected_scene_camera_pairs": global_pair_count,
        "expected_scene_camera_pairs": assigned_pair_count,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "completed_pairs": completed,
        "failed_pairs": len(failures),
        "total_masks": total_masks,
        "elapsed_seconds": elapsed,
        "failures": failures,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def main():
    args = parse_args()
    training_root = args.waymo_root / "training"
    for required in (training_root, args.gsam_root, args.checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_scenes = sorted(
        [path.name for path in training_root.iterdir() if path.is_dir()], key=numeric_key
    )
    global_pairs = build_global_pairs(all_scenes)
    pairs = select_shard_pairs(global_pairs, args.num_shards, args.shard_index)

    selected_scenes = {str(value) for value in args.scene} if args.scene else None
    selected_cameras = set(args.camera) if args.camera else None
    if selected_scenes is not None:
        pairs = [pair for pair in pairs if pair[0] in selected_scenes]
    if selected_cameras is not None:
        pairs = [pair for pair in pairs if pair[1] in selected_cameras]
    if args.limit_scenes is not None:
        kept_scenes = []
        for scene, _ in pairs:
            if scene not in kept_scenes:
                kept_scenes.append(scene)
        kept = set(kept_scenes[: args.limit_scenes])
        pairs = [pair for pair in pairs if pair[0] in kept]
    if not pairs:
        raise RuntimeError("No scene/camera pairs selected")

    print(
        f"Waymo scenes={len(all_scenes)} global_pairs={len(global_pairs)} "
        f"shard={args.shard_index}/{args.num_shards} assigned_pairs={len(pairs)}",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps({
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "global_pair_count": len(global_pairs),
            "assigned_pair_count": len(pairs),
            "first_pairs": pairs[:5],
            "last_pairs": pairs[-5:],
        }, indent=2))
        return

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices and "," in visible_devices:
        raise RuntimeError(
            "Each persistent process must see exactly one CUDA device; "
            f"got CUDA_VISIBLE_DEVICES={visible_devices!r}"
        )
    runner = PersistentGroundedSAM2(args)
    failures = []
    completed = 0
    total_masks = 0
    run_start = time.time()
    durations = []
    if args.num_shards == 1:
        failure_path = args.output_root / "failures.jsonl"
        manifest_path = args.output_root / "preprocess_manifest.json"
    else:
        failure_path = args.output_root / f"shard_{args.shard_index}_failures.jsonl"
        manifest_path = args.output_root / f"shard_{args.shard_index}_manifest.json"
    # Each invocation owns its failure report; stale smoke failures must not be
    # mistaken for failures from a later full or resumed run.
    failure_path.write_text("")

    for pair_index, (scene, camera) in enumerate(pairs, 1):
        pair_start = time.time()
        frames = []
        output_dir = args.output_root / scene / camera
        try:
            frames = collect_frames(training_root / scene, camera)
            if completion_is_valid(frames, output_dir) and not args.overwrite:
                _, samples = validate_masks(frames, output_dir)
                previous = json.loads((output_dir / ".r9_sam2_done.json").read_text())
                write_completion_record(
                    output_dir,
                    scene,
                    camera,
                    frames,
                    float(previous.get("elapsed_seconds", 0.0)),
                    samples,
                )
                remove_debug_artifacts(output_dir)
                status = "SKIP"
            else:
                clear_partial(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                stage_parent = Path(tempfile.mkdtemp(prefix=f"r9sam_{scene}_{camera}_"))
                try:
                    stage_dir = stage_parent / "frames"
                    stage_frames(frames, stage_dir)
                    runner.process_video(frames, stage_dir, output_dir)
                finally:
                    shutil.rmtree(stage_parent, ignore_errors=True)

                valid, samples = validate_masks(frames, output_dir)
                if not valid:
                    raise RuntimeError("Incomplete mask output after tracking")
                elapsed = time.time() - pair_start
                write_completion_record(
                    output_dir, scene, camera, frames, elapsed, samples
                )
                remove_debug_artifacts(output_dir)
                status = "PASS"

            completed += 1
            total_masks += len(frames)
        except Exception as error:
            marker = output_dir / ".r9_sam2_done.json"
            if marker.exists():
                marker.unlink()
            failure = {
                "scene": scene,
                "camera": camera,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            with failure_path.open("a") as stream:
                stream.write(json.dumps(failure) + "\n")
            traceback.print_exc()
            status = "FAIL"

        pair_elapsed = time.time() - pair_start
        durations.append(pair_elapsed)
        total_elapsed = time.time() - run_start
        remaining = len(pairs) - pair_index
        eta_seconds = (sum(durations) / len(durations)) * remaining
        print(
            f"[{pair_index:04d}/{len(pairs):04d}] scene={scene} camera={camera} "
            f"frames={len(frames)} status={status} elapsed={pair_elapsed:.1f}s "
            f"total_elapsed={total_elapsed:.1f}s ETA={eta_seconds:.1f}s "
            f"gpu_alloc={runner.torch.cuda.memory_allocated() / 2**30:.2f}GiB "
            f"gpu_reserved={runner.torch.cuda.memory_reserved() / 2**30:.2f}GiB",
            flush=True,
        )
        write_manifest(
            manifest_path,
            args,
            len(all_scenes),
            len(global_pairs),
            len(pairs),
            completed,
            failures,
            total_masks,
            total_elapsed,
        )

    print(
        f"Finished: completed_pairs={completed} failed_pairs={len(failures)} "
        f"total_masks={total_masks}",
        flush=True,
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
