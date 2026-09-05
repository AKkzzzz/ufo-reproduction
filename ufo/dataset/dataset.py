# Copyright (C) 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import json
import logging
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import bisect
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import trange

from .constants import DATASET_DICT, DATASETS, MEAN, STD
from .data_utils import resize_depth, resize_flow, to_float_tensor, to_tensor
from ufo.paper_contract import split_context_supervision

logger = logging.getLogger("UFO")


def find_closest_idx(l_frame, frame_idx):
    if frame_idx in l_frame:
        list_frame_idx = l_frame.index(frame_idx)
    elif frame_idx > l_frame[-1]:
        list_frame_idx = len(l_frame) - 1
    elif frame_idx < l_frame[0]: 
        list_frame_idx = 0
    else:
        list_frame_idx = bisect.bisect(l_frame, frame_idx)

    return list_frame_idx

def get_box_corners(instance_to_world, box_size):
    """
    Get 8 corner points of a 3D box in world coordinates.
    
    Args:
        instance_to_world: 4x4 transformation matrix
        box_size: [3,] array with [width, height, depth]
    
    Returns:
        corners: [8, 3] array of corner points in world space
    """
    # Get half extents
    half_size = box_size / 2.0
    
    # Define 8 corners in local space (centered at origin)
    # Each row is [x, y, z] for one corner
    local_corners = np.array([
        [-half_size[0], -half_size[1], -half_size[2]],  # corner 0
        [ half_size[0], -half_size[1], -half_size[2]],  # corner 1
        [-half_size[0],  half_size[1], -half_size[2]],  # corner 2
        [ half_size[0],  half_size[1], -half_size[2]],  # corner 3
        [-half_size[0], -half_size[1],  half_size[2]],  # corner 4
        [ half_size[0], -half_size[1],  half_size[2]],  # corner 5
        [-half_size[0],  half_size[1],  half_size[2]],  # corner 6
        [ half_size[0],  half_size[1],  half_size[2]],  # corner 7
    ])
    
    # Convert to homogeneous coordinates (add column of ones)
    local_corners_homogeneous = np.hstack([local_corners, np.ones((8, 1))])
    
    # Transform to world space: (instance_to_world @ corners.T).T
    world_corners_homogeneous = (instance_to_world @ local_corners_homogeneous.T).T
    
    # Convert back from homogeneous coordinates (divide by w and drop it)
    world_corners = world_corners_homogeneous[:, :3] / world_corners_homogeneous[:, 3:]
    
    return world_corners

class UFODataset(Dataset):
    def __init__(
        self,
        data_root: str,
        annotation_txt_file_list: Union[str, List[str]],
        target_size: Tuple[int, int] = (160, 240),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[1, 3, 5, 6, 7] = 3,
        timespan: float = 2.0,  # 2.0 seconds
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        load_depth: bool = True,
        load_flow: bool = False,
        load_dynamic_mask: bool = False,
        load_ground_label: bool = False,
        return_context_as_target: bool = False,
        skip_sky_mask: bool = False,
        num_target_chunks: int = 1,
        static: bool = False,
        reverse: bool = False,
        args = None
    ):
        super().__init__()
        self.static = static
        self.reverse = reverse
        self.num_target_chunks = num_target_chunks
        self.args = args

        self.data_root = data_root
        self.target_size = target_size
        self.num_context_timesteps = num_context_timesteps
        self.num_target_timesteps = num_target_timesteps
        self.num_max_cams = num_max_cams
        self.timespan = timespan
        self.load_depth = load_depth
        self.load_flow = load_flow
        self.load_dynamic_mask = load_dynamic_mask
        self.load_ground_label = load_ground_label
        self.skip_sky_mask = skip_sky_mask
        if isinstance(annotation_txt_file_list, str):
            annotation_txt_file_list = [annotation_txt_file_list]
        scene_list = []
        for annotation_txt_file in annotation_txt_file_list:
            with open(annotation_txt_file, "r") as f:
                scene_list += f.readlines()
        annotation_paths = [line.strip() for line in scene_list]
        if subset_indices is not None:
            annotation_paths = [annotation_paths[i] for i in subset_indices]
        self.annotations = []
        for annotation_path in annotation_paths:
            with open(os.path.join(data_root, annotation_path), "r") as f:
                self.annotations.append(json.load(f))
        self._instance_cache = OrderedDict()
        self._instance_scene_manifest = None
        instance_manifest = getattr(args, "instance_scene_index_manifest", None) if args is not None else None
        if instance_manifest:
            with open(instance_manifest) as handle:
                self._instance_scene_manifest = json.load(handle)["scenes"]
        logger.info(f"Loaded {len(self.annotations)} annotations.")
        self.num_replicas = num_replicas
        if self.num_replicas > 1:
            self.annotations *= self.num_replicas
        self.equispaced = equispaced
        self.return_context_as_target = return_context_as_target
        self.img_transformation = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=Image.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def _with_instances(self, scene_json):
        if "instances_info" in scene_json and "frame_instances" in scene_json:
            return scene_json
        cache_key = scene_json["scene_name"]
        if cache_key in self._instance_cache:
            self._instance_cache.move_to_end(cache_key)
            instances_info, frame_instances = self._instance_cache[cache_key]
        else:
            if "instances_info_path" in scene_json:
                instances_info_path = scene_json["instances_info_path"]
                frame_instances_path = scene_json["frame_instances_path"]
            elif self._instance_scene_manifest is not None:
                relative_dir = self._instance_scene_manifest.get(scene_json["scene_name"])
                if relative_dir is None:
                    raise KeyError(
                        f"scene {scene_json['scene_name']} is absent from instance manifest"
                    )
                instance_dir = Path(self.data_root) / relative_dir
                instances_info_path = instance_dir / "instances_info.json"
                frame_instances_path = instance_dir / "frame_instances.json"
            else:
                first_image = next(iter(scene_json["relative_image_path"].values()))[0]
                scene_relative = Path(first_image).parent.parent
                instance_dir = Path(self.data_root) / "datasets" / scene_json["dataset"] / scene_relative / "instances"
                instances_info_path = instance_dir / "instances_info.json"
                frame_instances_path = instance_dir / "frame_instances.json"
            with open(instances_info_path) as handle:
                instances_info = json.load(handle)
            with open(frame_instances_path) as handle:
                frame_instances = json.load(handle)
            if self._instance_scene_manifest is not None and instances_info:
                distances = []
                ego_poses = scene_json.get("ego_pose", scene_json.get("ego_to_world", []))
                for instance in instances_info.values():
                    annotations = instance.get("frame_annotations", {})
                    for frame, pose in zip(
                        annotations.get("frame_idx", [])[:1],
                        annotations.get("obj_to_world", [])[:1],
                    ):
                        if frame < len(ego_poses):
                            distances.append(np.linalg.norm(
                                np.asarray(pose, dtype=np.float64)[:3, 3]
                                - np.asarray(ego_poses[frame], dtype=np.float64)[:3, 3]
                            ))
                if distances and min(distances) > 300.0:
                    raise ValueError(
                        "instance/scene alignment failed for "
                        f"{scene_json['scene_name']}: nearest object is {min(distances):.1f}m "
                        f"from ego; source={instances_info_path}"
                    )
            self._instance_cache[cache_key] = (instances_info, frame_instances)
            while len(self._instance_cache) > 2:
                self._instance_cache.popitem(last=False)
        scene_json = scene_json.copy()
        scene_json["instances_info"] = instances_info
        scene_json["frame_instances"] = frame_instances
        return scene_json

    def get_frame(
        self,
        scene_json: Dict[str, Any],
        frame_idx: int,
        source_frame_idx: int = -1,
        global_source_frame_idx: int = -1,
        available_instances = None
    ) -> Dict[str, Any]:
        """Retrieve a single frame from the dataset."""
        normalized_intrinsics = scene_json["normalized_intrinsics"]
        dataset_name = scene_json["dataset"]
        cam_to_world = scene_json["camera_to_world"]

        images, depths, sky_masks, flows = [], [], [], []
        camtoworlds, intrinsics = [], []
        dynamic_masks, ground_masks = [], []

        camtoworlds_global = []

        if source_frame_idx < 0:
            source_frame_idx = frame_idx
        if global_source_frame_idx < 0:
            global_source_frame_idx = source_frame_idx

        camera_list = DATASET_DICT[dataset_name]["camera_list"][self.num_max_cams]
        ref_camera_name = DATASET_DICT[dataset_name]["ref_camera"]

        c2w_real_cur_frame = np.array(cam_to_world[ref_camera_name][source_frame_idx])
        c2w_real_global = np.array(cam_to_world[ref_camera_name][global_source_frame_idx])

        world_to_canonical = np.linalg.inv(c2w_real_cur_frame)
        world_to_canonical_global = np.linalg.inv(c2w_real_global)

        for camera in camera_list:
            img_relative_path = scene_json["relative_image_path"][camera][frame_idx]
            if dataset_name in ["waymo", "nuscenes", "argoverse2", "argoverse"]:
                img_relative_path = img_relative_path.replace("images", f"images_4")
                img_relative_path = img_relative_path.replace("sweeps", f"sweeps_4")
                img_relative_path = img_relative_path.replace("samples", f"samples_4")

            # Get RGB
            img_path = os.path.join(self.data_root, "datasets", dataset_name, img_relative_path)
            img = Image.open(img_path).convert("RGB")
            img = self.img_transformation(img)
            images.append(img)

            # Get sky mask
            if dataset_name in ["waymo", "nuscenes", "argoverse2"]:
                # if dataset_name in ["waymo", "nuscenes"]:
                if dataset_name == "nuscenes":
                    sky_path = img_path.replace("samples", "samples_sky_mask")
                    sky_path = sky_path.replace("sweeps", "sweeps_sky_mask")
                elif dataset_name == "waymo":
                    sky_path = img_path.replace("images_4", "sky_masks")
                elif dataset_name == "argoverse2":
                    sky_path = img_path.replace("images_4", "sky_masks_512")
                sky_path = sky_path.replace("jpg", "png")
                if self.skip_sky_mask:
                    sky = torch.zeros(self.target_size[0], self.target_size[1]).float()
                else:
                    try:
                        sky = Image.open(sky_path).convert("L").resize(self.target_size[::-1])
                    except FileNotFoundError:
                        sky = Image.open(sky_path).convert("L").resize(self.target_size[::-1])
                sky = to_tensor(np.array(sky) > 0).float()
                sky_masks.append(sky)

            # Get dynamic mask, this is for dynamic region evaluation.
            if dataset_name in ["waymo"] and self.load_dynamic_mask:
                dynamic_path = img_path.replace("images_8", "dynamic_masks")
                dynamic_path = dynamic_path.replace("images_4", "dynamic_masks")
                dynamic_path = dynamic_path.replace("jpg", "png")
                dynamic_mask = Image.open(dynamic_path).convert("L").resize(self.target_size[::-1])
                dynamic_mask = to_tensor(np.array(dynamic_mask) > 0).float()
                dynamic_masks.append(dynamic_mask)

            # Get ground label, this is for flow evaluation, i.e., we use this to exclude the ground lidar points.
            if dataset_name in ["waymo"] and self.load_ground_label:
                ground_path = img_path.replace("images", "ground_label")
                ground_path = ground_path.replace("jpg", "png")
                ground = Image.open(ground_path).convert("L").resize(self.target_size[::-1])
                ground = to_tensor(np.array(ground) > 0).float()
                ground_masks.append(ground)

            camtoworld = (
                DATASETS[dataset_name]["canonical_to_flu"]
                @ world_to_canonical
                @ cam_to_world[camera][frame_idx]
                @ DATASETS[dataset_name]["opencv2dataset"]
            )

            camtoworld = to_tensor(camtoworld)
            camtoworlds.append(camtoworld)

            camtoworld_global = (
                DATASETS[dataset_name]["canonical_to_flu"]
                @ world_to_canonical_global
                @ cam_to_world[camera][frame_idx]
                @ DATASETS[dataset_name]['opencv2dataset']
            )
            camtoworld_global = to_tensor(camtoworld_global)
            camtoworlds_global.append(camtoworld_global)

            # intrinsics
            fx, fy, cx, cy = np.array(normalized_intrinsics[camera])
            fx = fx * self.target_size[1]
            fy = fy * self.target_size[0]
            cx = cx * self.target_size[1]
            cy = cy * self.target_size[0]
            intrinsics.append(
                torch.tensor(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ]
                ).float()
            )

            if self.load_depth or self.load_flow:
                if dataset_name == "waymo":
                    depth_path = img_path.replace("images", "depth_flows").replace("jpg", "npy")
                    depth_and_flow = np.load(
                        depth_path, mmap_mode=None if self.load_flow else "r"
                    )
                    if self.load_depth:
                        depth = depth_and_flow[..., 0]
                        depth = torch.tensor(depth).float()
                        depth = resize_depth(depth, self.target_size)
                        depths.append(depth)
                    if self.load_flow:
                        flow = depth_and_flow[..., 1:]
                        flow = torch.tensor(flow).float()
                        flow = resize_flow(flow, self.target_size)
                        # there must be a better way to do this: rotate the flow to the canonical view
                        flow = (
                            flow
                            @ torch.tensor(
                                (
                                    world_to_canonical
                                    @ cam_to_world[camera][frame_idx]
                                    @ np.linalg.inv(scene_json["camera_to_ego"][camera])
                                )
                            )
                            .float()[:3, :3]
                            .T
                        )
                        flows.append(flow)
                if dataset_name == "nuscenes":
                    depth_path = img_path.replace("samples", "samples_depth")
                    depth_path = depth_path.replace("sweeps", "sweeps_depth")
                    depth_path = depth_path.replace("jpg", "npy")
                    depth = np.load(depth_path)
                    depth = torch.tensor(depth).float()
                    depth = resize_depth(depth, self.target_size)
                    depths.append(depth)

                if dataset_name == "argoverse2":
                    depth_path = img_path.replace("images", "depths")
                    depth_path = depth_path.replace("jpg", "npy")
                    depth = np.load(depth_path)
                    depth = torch.tensor(depth).float()
                    depth = resize_depth(depth, self.target_size)
                    depths.append(depth)

        self.load_objects = True
        if self.load_objects:
            instances_info = scene_json['instances_info']
            frame_instances = scene_json['frame_instances']

            # cur_frame_instances = frame_instances[str(frame_idx)]

            num_instances = self.args.num_bbox
            # instances_pose_ts = torch.zeros(num_instances, 4, 4)
            instances_pose_ts = torch.eye(4).reshape(1, 4, 4).repeat(num_instances, 1, 1)
            instances_id_ts = torch.zeros(num_instances, ).type(torch.int64)
            instances_track_id_ts = torch.full((num_instances,), -1, dtype=torch.int64)
            instances_corner_ts = torch.zeros(num_instances, 8, 3)
            instances_corner_local_ts = torch.zeros(num_instances, 8, 3)


            for slot, inst_id in enumerate(available_instances[:num_instances]):
                # if inst_id in cur_frame_instances and instances_info[str(inst_id)]['class_name'] == 'Vehicle':
                # if frame_idx in instances_info[str(inst_id)]['frame_annotations']['frame_idx']:
                #     list_frame_idx = instances_info[str(inst_id)]['frame_annotations']['frame_idx'].index(frame_idx)
                # elif frame_idx > instances_info[str(inst_id)]['frame_annotations']['frame_idx'][-1]:
                #     list_frame_idx = len(instances_info[str(inst_id)]['frame_annotations']['frame_idx']) - 1
                # elif frame_idx < instances_info[str(inst_id)]['frame_annotations']['frame_idx'][0]: 
                #     list_frame_idx = 0
                # else:
                #     list_frame_idx = bisect.bisect(instances_info[str(inst_id)]['frame_annotations']['frame_idx'], inst_id)

                annotated_frames = instances_info[str(inst_id)]['frame_annotations']['frame_idx']
                if frame_idx not in annotated_frames:
                    continue
                list_frame_idx = annotated_frames.index(frame_idx)

                inst2world_real = np.array(instances_info[str(inst_id)]['frame_annotations']['obj_to_world'][list_frame_idx], dtype=np.float64)
                inst2world = (
                    DATASETS[dataset_name]["canonical_to_flu"]
                    @ world_to_canonical_global
                    @ inst2world_real
                )
                inst2local = (
                    DATASETS[dataset_name]["canonical_to_flu"]
                    @ world_to_canonical
                    @ inst2world_real
                )
                instances_pose_ts[slot] = torch.from_numpy(inst2world).type(torch.float32)

                inst_box_size = np.array(instances_info[str(inst_id)]['frame_annotations']['box_size'][list_frame_idx], dtype=np.float64)

                instances_corner_ts[slot] = torch.from_numpy(
                    get_box_corners(inst2world, inst_box_size)
                ).type(torch.float32)
                instances_corner_local_ts[slot] = torch.from_numpy(
                    get_box_corners(inst2local, inst_box_size)
                ).type(torch.float32)
                instances_id_ts[slot] = 1
                instances_track_id_ts[slot] = int(inst_id)
        else:
            instances_pose_ts = None
            instances_corner_ts = None
            instances_corner_local_ts = None
            instances_id_ts = None
            instances_track_id_ts = None

        frame_images = torch.stack(images)
        frame_depths = torch.stack(depths) if len(depths) > 0 else None
        frame_sky_masks = torch.stack(sky_masks) if len(sky_masks) > 0 else None
        frame_flows = torch.stack(flows) if len(flows) > 0 else None
        frame_dynamic_masks = torch.stack(dynamic_masks) if len(dynamic_masks) > 0 else None
        frame_camtoworlds = torch.stack(camtoworlds)
        frame_camtoworlds_global = torch.stack(camtoworlds_global)
        frame_intrinsics = torch.stack(intrinsics)
        ground_masks = torch.stack(ground_masks) if len(ground_masks) > 0 else None

        toglobal_translation = (frame_camtoworlds_global[..., :3, 3] - frame_camtoworlds[..., :3, 3]).mean(dim=0)
        data_dict = {
            "image": frame_images,
            "camtoworld": frame_camtoworlds,
            "camtoworld_global": frame_camtoworlds_global,
            "intrinsics": frame_intrinsics,
            "frame_idx": frame_idx,
            "depth": frame_depths,
            "sky_masks": frame_sky_masks,
            "flow": frame_flows,
            "dynamic_masks": frame_dynamic_masks,
            "ground_masks": ground_masks,
            "instances_corner": instances_corner_ts,
            "instances_corner_local": instances_corner_local_ts,
            "instances_id": instances_id_ts,
            "instances_track_id": instances_track_id_ts,
            "instances_pose": instances_pose_ts
        }

        return {k: v for k, v in data_dict.items() if v is not None}

    def __getitem__(
            self, index: int, context_frame_idx: int = -1, return_all=False
        ):
        if True:
            sampling_dynamic_rich = False
            if isinstance(index, (tuple, list)):
                index, sampled_start, sampling_dynamic_rich = index
                if int(sampled_start) >= 0:
                    context_frame_idx = int(sampled_start)
            N = self.num_target_chunks
            scene_json = self._with_instances(self.annotations[index % len(self.annotations)])
            scene_id = scene_json["scene_id"]
            num_timesteps = scene_json["num_timesteps"]
            fps = scene_json["fps"]
            num_max_future_frames = N * int(self.timespan * fps)  # total frames for N windows
            num_max_frames_per_window = int(self.timespan * fps)
            
            # Adjust if not enough frames in the scene
            if num_max_future_frames > num_timesteps:
                raise Exception("Not enough frames")
            
            time_in_seconds = scene_json["normalized_time"]
            
            # Store the initial starting frame (rename to avoid confusion)
            if context_frame_idx < 0:
                initial_start_frame = np.random.randint(0, max(1, num_timesteps - num_max_future_frames))
            else:
                initial_start_frame = context_frame_idx
                
            # Validate and adjust if necessary
            if initial_start_frame + num_max_future_frames >= num_timesteps:
                initial_start_frame = np.random.randint(0, max(1, num_timesteps - num_max_future_frames))
            
            value_list = []
            
            first_context_idx = None
            last_frame_idx = initial_start_frame + num_max_future_frames

            # get all possible instances
            frame_instances = scene_json['frame_instances']
            available_instances = []
            for i in range(initial_start_frame, last_frame_idx):
                available_instances.extend(frame_instances[str(i)])
            available_instances = list(set(available_instances))



            ### filter out bboxes with small amount of lidar points
            # remove instances with less than 20 points
            instances_info = scene_json['instances_info']
            filtered_instances = []
            for ind in available_instances:
                if instances_info[str(ind)]['num_lidar'] < 20:
                    continue

                # filter motion
                search_start = find_closest_idx(instances_info[str(ind)]['frame_annotations']['frame_idx'], initial_start_frame)
                search_end = find_closest_idx(instances_info[str(ind)]['frame_annotations']['frame_idx'], last_frame_idx)
                center_points = np.array(instances_info[str(ind)]['frame_annotations']['obj_to_world'])[search_start:search_end, :3, 3]
                if len(center_points) == 0 or (center_points.max(axis=0) - center_points.min(axis=0)).max() < 0.5: 
                    continue
                
                filtered_instances.append(ind)

            # sort in descending order (# of lidar points)
            filtered_instances.sort(key=lambda i : instances_info[str(i)]['num_lidar'], reverse=True)

            available_instances = filtered_instances[:self.args.num_bbox]

            for window in range(N):
                # Calculate window boundaries
                window_start = initial_start_frame + (window * num_max_frames_per_window)
                window_end = min(window_start + num_max_frames_per_window, num_timesteps)
                
                
                # Generate context frame indices for this window
                if getattr(self.args, "paper_frame_protocol", False):
                    frame_protocol = split_context_supervision(window_start, window_end)
                    window_context_indices = np.asarray(frame_protocol.context)
                elif self.equispaced:
                    window_context_indices = np.arange(
                        window_start,
                        window_end,
                        max(1, (window_end - window_start) // self.num_context_timesteps),
                    )[:self.num_context_timesteps]
                else:
                    num_available_frames = window_end - window_start
                    num_context = min(self.num_context_timesteps, num_available_frames)
                    window_context_indices = np.random.choice(
                        np.arange(window_start, window_end),
                        size=num_context,
                        replace=False,
                    )
                    window_context_indices = sorted(window_context_indices)

                if window == 0:
                    first_context_idx = window_context_indices[0]
                


                # target_window_start = window_start
                if True:
                    target_window_start = first_context_idx
                    target_window_end = last_frame_idx
                    global_frame = True
                    global_frame = False
                else:
                    raise Exception("Not Implemented")


                reverse = self.reverse
                # Generate target frame indices for this window
                if getattr(self.args, "paper_frame_protocol", False):
                    # The paper fixes the frame partition but does not disclose
                    # whether training renders these frames per chunk or from S_T.
                    target_frame_idx = np.asarray(frame_protocol.supervision)
                elif return_all:
                    # Return all frames in this window
                    if reverse:
                        target_frame_idx = np.arange(window_start, target_window_end)
                    else:
                        target_frame_idx = np.arange(target_window_start, window_end)
                else:
                    # Randomly sample "num_target_timesteps" frames from this window
                    num_available_frames = window_end - target_window_start
                    num_targets = min(self.num_target_timesteps, num_available_frames)
                    if reverse:


                        # Create indices
                        indices = np.arange(window_start, target_window_end)

                        decay = False

                        ### decay
                        if decay:
                            # Create exponential decay weights (highest at window_start, lowest at target_window_end)
                            decay_rate = 0.02  # Adjust this to control steepness of decay
                            weights = np.exp(-decay_rate * (indices - window_start))

                            # Normalize to get probabilities
                            probabilities = weights / weights.sum()

                            target_frame_idx = np.random.choice(
                                indices,
                                num_targets,
                                replace=False,
                                p=probabilities
                            )
                        else:
                            target_frame_idx = np.random.choice(
                                indices,
                                num_targets,
                                replace=False
                            )
                        
                        # print(target_frame_idx, window_start, target_window_end)
                    else:
                        target_frame_idx = np.random.choice(
                            np.arange(target_window_start, window_end),
                            num_targets,
                            replace=False,
                        )
                
                target_frame_idx = [min(idx, num_timesteps - 1) for idx in target_frame_idx]
                target_frame_idx = sorted(target_frame_idx)


                if global_frame:
                    start_context_idx = first_context_idx
                else:
                    start_context_idx = window_context_indices[0]
                
                # Get context frames
                context_dict_list = []
                for ctx_id in window_context_indices:
                    context_dict = self.get_frame(
                        scene_json=scene_json,
                        frame_idx=ctx_id,
                        source_frame_idx=window_context_indices[0],
                        global_source_frame_idx=last_frame_idx,
                        available_instances=available_instances
                    )
                    if self.static:
                        if context_dict['flow'].abs().mean() - 0 > 1e-5:
                            return self.__getitem__(index + 1, context_frame_idx, return_all)
                    context_dict["time"] = torch.tensor(
                        [time_in_seconds[ctx_id] - time_in_seconds[last_frame_idx]]
                        * self.num_max_cams
                    )
                    context_dict_list.append(context_dict)
                
                if self.return_context_as_target:
                    target_frame_idx = window_context_indices
                
                # Get target frames
                target_dict_list = []
                for target_id in target_frame_idx:
                    target_dict = self.get_frame(
                        scene_json=scene_json,
                        frame_idx=target_id,
                        source_frame_idx=window_context_indices[0],
                        global_source_frame_idx=last_frame_idx,
                        available_instances=available_instances
                    )
                    target_dict["time"] = torch.tensor(
                        [time_in_seconds[target_id] - time_in_seconds[last_frame_idx]]
                        * self.num_max_cams
                    )
                    target_dict_list.append(target_dict)
                
                # Collate the dictionaries
                context_dict = default_collate(context_dict_list)
                target_dict = default_collate(target_dict_list)
                
                # Concatenate tensors
                for k, v in context_dict.items():
                    if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                        context_dict[k] = torch.cat([d for d in v], dim=0)
                for k, v in target_dict.items():
                    if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                        target_dict[k] = torch.cat([d for d in v], dim=0)
                
                sample = {
                    "context": context_dict,
                    "target": target_dict,
                    "scene_id": scene_id,
                    "scene_name": scene_json["scene_name"],
                    "width": self.target_size[1],
                    "height": self.target_size[0],
                    "fps": fps,
                    "timespan": self.timespan,
                    "window_index": window,  # Add window index for tracking
                    "window_start_frame": window_start,  # Add start frame of this window
                    "global_frame": global_frame
                }
                sample["sample_start_frame"] = int(initial_start_frame)
                sample["sample_scene_index"] = int(index % len(self.annotations))
                sample["sampling_dynamic_rich"] = bool(sampling_dynamic_rich)
                value_list.append(to_float_tensor(sample))
            
            # get gs frames
            # gs_start_idx = initial_start_frame
            # gs_end_idx = initial_start_frame + num_max_future_frames
            # gs_indices = np.arange(
            #     gs_start_idx,
            #     gs_end_idx,
            #     max(1, (gs_end_idx - gs_start_idx) // self.num_context_timesteps),
            # )[:self.num_context_timesteps]
            return value_list  # This should be outside the loop
            
        # except Exception as e:
        #     logger.info(
        #         f"Error in scene_id: {scene_id}, context_frame_idx: {context_frame_idx}, scene_name: {scene_json['scene_name']}"
        #     )
        #     logger.info(e)
        #     try:
        #         return self.__getitem__(index + 1, 0, return_all)
        #     except Exception as e:
        #         logger.info(e)
        #         return self.__getitem__(index + 1)
    
    

class UFODatasetEval(UFODataset):
    def __init__(
        self,
        data_root: str,
        annotation_txt_file_list: Union[str, List[str]],
        target_size: Tuple[int, int] = (225, 400),
        num_context_timesteps: int = 4,
        num_target_timesteps: int = 4,
        num_max_cams: Literal[1, 3, 5, 6, 7] = 3,
        timespan: float = 2.0,  # how many seconds
        subset_indices: Optional[List[int]] = None,
        num_replicas: int = 1,
        equispaced: bool = True,
        load_depth: bool = True,
        load_flow: bool = False,
        load_dynamic_mask: bool = False,
        load_ground_label: bool = False,
        return_context_as_target: bool = False,
        skip_sky_mask: bool = False,
        scene_id_list: Optional[List[int]] = None,
        **kwargs
    ):
        super(UFODatasetEval, self).__init__(
            data_root=data_root,
            annotation_txt_file_list=annotation_txt_file_list,
            target_size=target_size,
            num_context_timesteps=num_context_timesteps,
            num_target_timesteps=num_target_timesteps,
            num_max_cams=num_max_cams,
            timespan=timespan,
            subset_indices=subset_indices,
            num_replicas=num_replicas,
            equispaced=equispaced,
            load_depth=load_depth,
            load_flow=load_flow,
            load_dynamic_mask=load_dynamic_mask,
            load_ground_label=load_ground_label,
            return_context_as_target=return_context_as_target,
            skip_sky_mask=skip_sky_mask,
            **kwargs
        )
        val_sample_list = []
        if scene_id_list is None:
            scene_id_list = list(range(len(self.annotations)))
        for scene_id in scene_id_list:
            for start_id in range(0, 200, 20):
                if scene_id == 63 and start_id == 0:
                    continue
                val_sample_list.append((scene_id, start_id))
        self.val_sample_list = val_sample_list

    def __len__(self) -> int:
        return len(self.val_sample_list)

    def __getitem__(self, index: int):
        return super(UFODatasetEval, self).__getitem__(
            self.val_sample_list[index][0],
            self.val_sample_list[index][1],
            return_all=True,
        )
