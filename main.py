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

import argparse
import copy
import contextlib
import datetime
import hashlib
import json
import logging
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from ufo.utils.misc import update_scene

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter  # Add TensorBoard import
from einops import rearrange, repeat
import torch.nn.functional as F
# UFO imports
import ufo.models as models
import ufo.utils.distributed as distributed
import ufo.utils.misc as misc
from ufo.utils.engine import evaluate, evaluate_flow, visualize
from ufo.dataset.constants import DATASET_DICT
from ufo.dataset.data_utils import prepare_inputs_and_targets
from ufo.dataset.samplers import DynamicMixtureSampler, InfiniteSampler, NoPaddingDistributedSampler
from ufo.dataset.dataset import UFODataset, UFODatasetEval
from ufo.utils.logging import MetricLogger, WandbLogger, setup_logging
from ufo.utils.losses import compute_loss
from ufo.utils.diagnostics import gaussian_metrics, parameter_grad_norms, reconstruction_metrics
from ufo.utils.lpips_loss import RGBLpipsLoss
from ufo.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from ufo.utils.misc import combine_dict_entries, project_boxes_to_image, convert_to_chunks
from ufo.utils.misc import compute_point_visibility, compute_visible_topk_indices_any_view, batched_index_gather, batched_index_update
from ufo.ar import val
from ufo.paper_contract import assert_paper_training_ready
from typing import Dict, Any, Optional

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True


# ============================================================================
# TENSORBOARD LOGGER
# ============================================================================

class TensorBoardLogger:
    def __init__(
        self,
        log_dir: str,
        comment: str = "",
        *,
        distributed: bool = True,     # only used to detect rank; no collectives are called
        purge_step: Optional[int] = None,
        flush_secs: int = 30,
        max_queue: int = 1000,
    ):
        """
        Rank-zero-only TensorBoard logger:
        - No all_reduce / cross-rank sync.
        - No barriers.
        - Only rank 0 creates a SummaryWriter and writes events.
        """
        self.step = 0

        # Figure out rank without synchronizing.
        dist_inited = distributed and dist.is_available() and dist.is_initialized()
        if dist_inited:
            try:
                rank = dist.get_rank()
            except Exception:
                rank = int(os.getenv("RANK", "0"))
        else:
            # fall back to env var or assume single-process
            rank = int(os.getenv("RANK", "0")) if distributed else 0

        self.rank = rank
        self.is_master = (self.rank == 0)

        # Only master prepares the log dir and writer
        self.writer = None
        if self.is_master:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir,
                comment=comment,
                purge_step=purge_step,
                flush_secs=flush_secs,
                max_queue=max_queue,
            )

    # ---------- public API ----------

    def set_step(self, step: Optional[int] = None):
        """Set the current step for logging."""
        if step is not None:
            self.step = int(step)
        else:
            self.step += 1

    def update(
        self,
        metrics_dict: Dict[str, Any],
        *,
        step: Optional[int] = None,
        **_: Any,  # ignore any old kwargs like reduce/sync
    ):
        """
        Log metrics to TensorBoard (rank 0 only).
        - Accepts nested dicts: {tag: value} or {group_tag: {sub: value, ...}}
        - Values can be int/float or torch.Tensor; tensors become local scalars.
        - No cross-rank reduction; values reflect rank 0 only.
        """
        if step is not None:
            self.set_step(step)

        if not (self.is_master and self.writer is not None):
            return  # no-op on non-master ranks

        flat_scalars, grouped_scalars = {}, {}

        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                sub = {}
                for sk, sv in value.items():
                    sval = self._to_scalar_local(sv)
                    if sval is not None:
                        sub[sk] = sval
                if sub:
                    grouped_scalars[key] = sub
            else:
                sval = self._to_scalar_local(value)
                if sval is not None:
                    flat_scalars[key] = sval

        for k, v in flat_scalars.items():
            self.writer.add_scalar(k, v, self.step)
        for k, sub in grouped_scalars.items():
            self.writer.add_scalars(k, sub, self.step)

    def flush(self):
        if self.is_master and self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.is_master and self.writer is not None:
            self.writer.close()

    # ---------- helpers ----------

    def log_image(self, tag: str, imgCHW: torch.Tensor, step: Optional[int] = None, every: int = 1000):
        """Log an image [C,H,W] in [0,1] from rank 0 only."""
        if step is not None:
            self.set_step(step)
        if self.is_master and self.writer is not None and (self.step % every == 0):
            self.writer.add_image(tag, imgCHW, self.step)

    def log_hist(self, tag: str, tensor: torch.Tensor, step: Optional[int] = None, every: int = 100):
        """Log a histogram from rank 0 only."""
        if step is not None:
            self.set_step(step)
        if self.is_master and self.writer is not None and (self.step % every == 0):
            self.writer.add_histogram(tag, tensor.detach().float().cpu().numpy(), self.step)

    def _to_scalar_local(self, v: Any) -> Optional[float]:
        """
        Convert int/float/tensor to a local float scalar on the current rank.
        No device moves beyond what's needed to read a scalar.
        """
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, torch.Tensor):
            t = v.detach()
            if t.numel() == 0:
                return None
            t = t.float()
            if t.numel() > 1:
                t = t.mean()
            # .item() will sync if t is on CUDA; that's fine on rank 0 only.
            return float(t.item())
        return None


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def get_args_parser():
    parser = argparse.ArgumentParser("UFO training")

    # =============== Configuration file ================= #
    parser.add_argument("--config", type=str, default=None,
                       help="Path to JSON config file (default: configs/default.json). CLI args override config values.")

    # =============== Model parameters ================= #
    parser.add_argument("--arch", default="small", type=str,
                       help="Architecture type (e.g. 'small'). Selects which model class is built; "
                            "see ufo/models/archs/ for available types.")
    parser.add_argument("--model", default="UFO-B/8", type=str,
                       help="Backbone size identifier within the chosen --arch (e.g. UFO-B/8, UFO-L/8).")
    parser.add_argument("--num_context_timesteps", default=4, type=int)
    parser.add_argument("--num_target_timesteps", default=4, type=int)
    parser.add_argument("--gs_dim", default=3, type=int, help="Number of gs dimensions")
    parser.add_argument("--use_sky_token", action="store_true")
    parser.add_argument("--use_affine_token", action="store_true")
    parser.add_argument("--use_latest_gsplat", action="store_true")
    parser.add_argument("--max_gaussian_scale", default=0.5, type=float,
                        help="Hard upper bound applied after exponential Gaussian scale activation.")
    parser.add_argument(
        "--decoder_type",
        type=str,
        choices=["dummy", "conv"],
        default="dummy",
    )
    parser.add_argument("--num_motion_tokens", default=16, type=int, help="Number of motion tokens")
    parser.add_argument("--filter_num", default=3600, type=int, help="Number of visible tokens to keep when filtering scene (k for top-k filtering)")

    # =============== Losses =============== #
    parser.add_argument("--enable_depth_loss", action="store_true")
    parser.add_argument("--depth_loss_coeff", type=float, default=1.0,
                        help="Depth-loss multiplier; 1.0 preserves the public implementation.")
    parser.add_argument("--depth_loss_normalization", choices=["target_max", "raw"],
                        default="target_max", help="Depth L1 units used by the objective.")

    # Option 1: push the sky depth to a fixed value
    parser.add_argument("--enable_sky_depth_loss", action="store_true")
    parser.add_argument("--sky_depth", type=float, default=300.0)
    # Option 2: make sky gaussians transparent and use a sky token to represent sky
    parser.add_argument("--enable_sky_opacity_loss", action="store_true")
    parser.add_argument("--sky_opacity_loss_coeff", type=float, default=0.1)

    # flow regularization loss
    parser.add_argument("--enable_flow_reg_loss", action="store_true")
    parser.add_argument("--flow_reg_coeff", type=float, default=0.005)

    # lifespan regularization loss (enabled by default)
    parser.add_argument("--enable_lifespan_reg_loss", type=bool, default=True,
                        help="Enable L1 regularization on lifespan to encourage persistent Gaussians")
    parser.add_argument("--lifespan_reg_coeff", type=float, default=1e-4,
                        help="Coefficient for lifespan L1 regularization loss")

    # perceptual loss
    parser.add_argument("--enable_perceptual_loss", action="store_true")
    parser.add_argument("--perceptual_weight", default=0.05, type=float, help="LPIPS weight")
    parser.add_argument("--perceptual_loss_start_iter", default=5000, type=int)

    # ============= Optimizer and LR parameters ============= #
    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=8e-4, help="base learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_sched", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_iters", type=int, default=5000, help="iters to warmup LR")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=3.0, help="Gradient clip")
    parser.add_argument("--disable_grad_checkpointing", action="store_true")
    parser.add_argument("--sequential_chunk_backward", action="store_true",
                        help="Backward each detached recurrent chunk immediately, then take one optimizer step.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Dataloader microbatches per optimizer step; independent of recurrent chunks.")
    parser.add_argument("--ddp_accumulation_no_sync", action="store_true",
                        help="Synchronize DDP gradients only on the final accumulation microbatch.")
    parser.add_argument("--ddp_smoke_assertions", action="store_true",
                        help="Assert distinct rank data and synchronized gradients/parameters once.")
    parser.add_argument("--detach_scene_between_chunks", action="store_true",
                        help="Detach all recurrent scene tensors between chunks (truncated recurrent gradients).")
    parser.add_argument("--allow_old_scene_grad", action="store_true",
                        help="Keep old latent scene tokens attached for recurrent BPTT.")
    parser.add_argument("--gaussian_decoder_layers", choices=["linear", "mlp2"], default="mlp2",
                        help="Official-v1 Linear or paper-described 2-layer Gaussian decoder.")
    parser.add_argument("--scene_token_input", choices=["official_rgb", "latent"], default="official_rgb",
                        help="Encode visible old tokens from stored RGB (v1) or recurrent latent state (paper path).")
    parser.add_argument("--attention_mode", choices=["official_v1_dense"], default="official_v1_dense",
                        help="Released v1 dense SDPA. The paper custom flex-attention mask is not public.")
    parser.add_argument("--enable_lifespan_renderer", action="store_true",
                        help="Apply temporal Gaussian opacity in the renderer.")
    parser.add_argument("--lifespan_parameterization", choices=["official_precision", "paper_beta"],
                        default="official_precision")
    parser.add_argument("--object_assignment_loss_coeff", type=float, default=0.01)
    parser.add_argument("--object_assignment_background_weight", type=float, default=0.1)
    parser.add_argument("--object_soft_target_temperature", type=float, default=0.1)
    parser.add_argument(
        "--object_assignment_gt_mode",
        choices=["predicted_mean", "lidar_anchor", "lidar_token"],
        default="lidar_token",
        help=(
            "Independent patch LiDAR-token supervision (default); lidar_anchor is "
            "a compatibility alias and predicted_mean is deprecated diagnostics only."
        ),
    )
    parser.add_argument("--training_sampling_mode", choices=["uniform", "dynamic_mixture"], default="uniform")
    parser.add_argument("--dynamic_rich_pool", type=str, default=None)
    parser.add_argument("--dynamic_sampling_ratio", type=float, default=0.0)
    parser.add_argument(
        "--instance_scene_index_manifest",
        type=str,
        default=None,
        help="Optional scene-name mapping for Waymo instances; prevents annotation/index mismatch.",
    )
    parser.add_argument("--recurrent_aux_tokens", action="store_true",
                        help="Carry updated sky/affine auxiliary tokens across recurrent chunks.")
    parser.add_argument("--legacy_sky_full_opacity_loss", action="store_true",
                        help="Keep the extra v1 full-image opacity-to-one loss when sky-depth loss is active.")
    parser.add_argument("--disable_legacy_time_offset", action="store_true",
                        help="Keep dataset-global normalized time instead of forcing every chunk context to -1.")
    parser.add_argument("--paper_affine_transform", action="store_true",
                        help="Use the paper A*c+b transform with identity initialization.")
    parser.add_argument("--paper_bbox_rotation", action="store_true",
                        help="Apply soft bbox-guided relative yaw to Gaussian quaternions.")
    parser.add_argument("--mask_invalid_bbox_tokens", action="store_true",
                        help="Exclude padded bbox tokens from soft assignment (v1 TODO fix).")
    parser.add_argument("--stable_bbox_delta_transform", action="store_true",
                        help="Blend bbox motion deltas instead of absolute means for BF16 stability.")
    parser.add_argument("--paper_frame_protocol", action="store_true",
                        help="Use every fifth frame as context and every other frame as supervision.")
    parser.add_argument("--paper_supervision_mode", choices=["unknown", "per_chunk", "final_scene"],
                        default="unknown", help="Undisclosed paper training render/loss timing.")
    parser.add_argument("--paper_forward_flow_impl", action="store_true",
                        help="Set only when an evidence-backed paper forward-flow implementation exists.")
    parser.add_argument("--allow_missing_paper_components", action="store_true",
                        help="Sprint-only override; logs missing flow/mask components without claiming contract equivalence.")

    parser.add_argument("--start_iteration", default=0, type=int, help="start iteration")
    parser.add_argument("--best_validation_psnr", default=-1.0, type=float,
                        help="Best held-out gate PSNR persisted in checkpoints.")
    parser.add_argument("--num_iterations", default=200_000, type=int, help="num of iterations")
    parser.add_argument("--resume_from", default=None, help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--load_from", type=str, default=None)

    # ============= Dataset parameters ============= #
    parser.add_argument("--data_root", default="./data", type=str, help="dataset path")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--input_size", default=(160, 240), type=int, nargs=2)
    parser.add_argument("--num_max_cameras", type=int, default=3)
    parser.add_argument("--timespan", type=float, default=2.0)
    parser.add_argument("--load_ground", action="store_true")
    parser.add_argument("--load_depth", action="store_true")
    parser.add_argument("--load_flow", action="store_true")
    parser.add_argument("--load_dynamic_mask", action="store_true",
                        help="Load preprocessed dynamic masks for dynamic-region diagnostics.")
    parser.add_argument("--dataset", default="waymo", type=str, choices=DATASET_DICT.keys())
    parser.add_argument("--subset_ratio", default=1.0, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--skip_sky_mask", action="store_true", help="skip sky mask loading")
    # ============= Logging ============= #
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--num_vis_samples", type=int, default=1)
    parser.add_argument("--log_every_n_iters", type=int, default=50)
    parser.add_argument("--vis_every_n_iters", type=int, default=5000)
    parser.add_argument("--ckpt_every_n_iters", type=int, default=5000)
    parser.add_argument("--eval_every_n_iters", type=int, default=50000)
    parser.add_argument(
        "--validation_steps", default="",
        help="Comma-separated optimizer steps for held-out validation.",
    )
    parser.add_argument("--total_elapsed_time", type=float, default=0.0, help="total time elapsed")
    parser.add_argument("--keep_n_ckpts", default=5, type=int)

    # ============= Miscellaneous ============= #
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--visualization_only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--skip_flow_evaluation", action="store_true")
    parser.add_argument("--skip_initial_validation", action="store_true")
    parser.add_argument("--skip_final_evaluation", action="store_true")

    # ============= WandB and TensorBoard ============= #
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--enable_tensorboard", action="store_true", 
                       help="Enable TensorBoard logging (default: True if wandb is not enabled)")
    parser.add_argument("--project", default="debug", type=str)
    parser.add_argument("--entity", default="YOUR_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--overwrite_wandb", action="store_true")


    parser.add_argument("--num_target_chunks", default=1, type=int)
    parser.add_argument("--num_window_chunks", default=3, type=int)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--num_mem_tokens", default=0, type=int)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--ar", action="store_true")
    parser.add_argument("--num_bbox", default=32, type=int)

    return parser


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def backup_python_files(backup_dir):
    """Backup all Python files to the backup directory."""
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)

    # Get the root directory (current working directory)
    root_dir = os.getcwd()

    # Find all Python files recursively
    for root, dirs, files in os.walk(root_dir):
        # Skip backup directory itself, hidden directories, data directory, and output
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != os.path.basename(backup_dir) and d != 'data' and d != 'output']

        for file in files:
            if file.endswith('.py'):
                src_path = os.path.join(root, file)
                # Create relative path for destination
                rel_path = os.path.relpath(src_path, root_dir)
                dst_path = os.path.join(backup_dir, rel_path)

                # Create destination directory if needed
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                # Copy the file
                shutil.copy2(src_path, dst_path)

    return backup_dir


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_run_manifest(args, world_size, train_annotation, val_annotation):
    def git(*command):
        return subprocess.check_output(["git", *command], text=True).strip()

    tracked_inputs = {
        "train_scene_list": train_annotation,
        "validation_scene_list": val_annotation,
        "instance_scene_manifest": args.instance_scene_index_manifest,
    }
    git_diff = subprocess.check_output(["git", "diff", "--binary"], text=False)
    manifest = {
        "command": [sys.executable, *sys.argv],
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "git_diff_stat": git("diff", "--stat"),
        "git_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
        "hostname": os.uname().nodename,
        "container_id": os.environ.get("HOSTNAME"),
        "runtime": {
            "python": sys.version,
            "pytorch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_capabilities": [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())],
        },
        "seed": args.seed,
        "resolved_config": vars(args),
        "training_scale": {
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch": world_size * args.batch_size * args.gradient_accumulation_steps,
            "sequences_per_optimizer_step": world_size * args.batch_size * args.gradient_accumulation_steps,
            "chunks_per_sequence": args.num_target_chunks,
            "optimizer_step_definition": "one AdamW update after gradient_accumulation_steps sequences",
        },
        "inputs": {},
    }
    for name, path in tracked_inputs.items():
        if path and os.path.exists(path):
            manifest["inputs"][name] = {"path": path, "sha256": _sha256(path)}
    with open(os.path.join(args.log_dir, "run_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    # ========================================================================
    # Setup: Distributed training, logging, directories
    # ========================================================================

    distributed.enable(overwrite=True)

    global logger
    args.exp_name = args.model.replace("/", "-") if args.exp_name is None else args.exp_name
    log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    video_dir = os.path.join(log_dir, "videos")
    backup_dir = os.path.join(log_dir, "backup")
    tensorboard_dir = os.path.join("tensorboard", args.exp_name)  # Add tensorboard directory
    args.log_dir, args.ckpt_dir, args.video_dir = log_dir, checkpoint_dir, video_dir

    device = torch.device(args.device)
    world_size, global_rank = distributed.get_world_size(), distributed.get_global_rank()
    seed = args.seed + global_rank
    misc.fix_random_seeds(seed)

    log_writer = None
    if global_rank == 0:
        [os.makedirs(d, exist_ok=True) for d in [log_dir, checkpoint_dir, video_dir, backup_dir]]

    if global_rank == 0 and args.enable_tensorboard:
        os.makedirs(tensorboard_dir, exist_ok=True)
        # Backup all Python files
        backup_python_files(backup_dir)
        
        if args.enable_wandb:
            # WandB logging
            run_id_path, run_id = os.path.join(log_dir, "wandb_run_id.txt"), None
            if os.path.exists(run_id_path) and not args.overwrite_wandb:
                with open(run_id_path, "r") as f:
                    run_id = f.readlines()[-1].strip()
            log_writer = WandbLogger(args=args, resume="must", id=run_id)
            if run_id is None:
                with open(run_id_path, "a") as f:
                    f.write(log_writer.run_id + "\n")
        elif args.enable_tensorboard:
            # TensorBoard logging (enabled by default if wandb is not enabled)
            log_writer = TensorBoardLogger(
                log_dir=tensorboard_dir, 
                comment=f"{args.project}_{args.exp_name}"
            )
            # Log hyperparameters
            if hasattr(log_writer, 'writer'):
                log_writer.writer.add_text('hyperparameters', 
                                            json.dumps(args.__dict__, indent=2), 0)

    # set up logging
    setup_logging(output=log_dir, level=logging.INFO)
    logger = logging.getLogger("UFO")
    logger.info(f"hostname: {os.uname().nodename}\n")
    logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info(f"Logging to {log_dir}")
    if global_rank == 0 and log_writer is not None:
        if isinstance(log_writer, TensorBoardLogger):
            logger.info(f"TensorBoard logging enabled. Run 'tensorboard --logdir {tensorboard_dir}' to view logs")
        elif hasattr(log_writer, 'run_id'):
            logger.info(f"WandB logging enabled with run_id: {log_writer.run_id}")
    logger.info(json.dumps(args.__dict__, indent=4, sort_keys=True))
    if global_rank == 0:
        # Save final merged configuration
        from ufo.utils.config import save_config, args_to_dict
        final_config = args_to_dict(args)
        save_config(final_config, os.path.join(log_dir, "config.json"))
        logger.info(f"Saved final configuration to {os.path.join(log_dir, 'config.json')}")

    # ========================================================================
    # Dataset initialization
    # ========================================================================

    dataset_meta = DATASET_DICT[args.dataset]
    train_annotation = dataset_meta["annotation_txt_file_train"]
    val_annotation = dataset_meta["annotation_txt_file_val"]
    if train_annotation is not None:
        if args.dataset == "nuscenes":
            train_annotation = f"data/dataset_scene_list/nuscenes_train.txt"
        else:
            train_annotation = f"{args.data_root}/{train_annotation}"
    if val_annotation is not None:
        if args.dataset == "nuscenes":
            val_annotation = f"data/dataset_scene_list/nuscenes_val.txt"
        else:
            val_annotation = f"{args.data_root}/{val_annotation}"
        if not os.path.exists(val_annotation):
            val_annotation = None

    dataset_train = UFODataset(
        data_root=args.data_root,
        annotation_txt_file_list=train_annotation,
        target_size=args.input_size,
        num_context_timesteps=args.num_context_timesteps,
        num_target_timesteps=args.num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
        load_dynamic_mask=args.load_dynamic_mask,
        skip_sky_mask=args.skip_sky_mask,
        num_target_chunks=args.num_target_chunks,
        static=args.static,
        reverse=args.reverse,
        args=args
    )
    if args.training_sampling_mode == "uniform":
        sampler_train = InfiniteSampler(sample_count=len(dataset_train), shuffle=True, seed=seed)
    else:
        if not args.dynamic_rich_pool:
            raise ValueError("dynamic sampling requires --dynamic_rich_pool")
        sampler_train = DynamicMixtureSampler(
            sample_count=len(dataset_train),
            rich_pool_path=args.dynamic_rich_pool,
            rich_ratio=args.dynamic_sampling_ratio,
            seed=seed,
        )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=True,
    )

    if val_annotation is not None:
        dataset_val = UFODataset(
            data_root=args.data_root,
            annotation_txt_file_list=val_annotation,
            target_size=args.input_size,
            num_context_timesteps=args.num_context_timesteps,
            num_target_timesteps=args.num_target_timesteps,
            timespan=args.timespan,
            num_max_cams=args.num_max_cameras,
            load_depth=args.load_depth,
            load_flow=args.load_flow,
            load_dynamic_mask=args.load_dynamic_mask,
            skip_sky_mask=args.skip_sky_mask,
            num_target_chunks=args.num_target_chunks,
            static=args.static,
            reverse=args.reverse,
            args=args
        )
        dataset_eval = UFODatasetEval(
            data_root=args.data_root,
            annotation_txt_file_list=val_annotation,
            target_size=args.input_size,
            num_context_timesteps=args.num_context_timesteps,
            num_target_timesteps=args.num_target_timesteps,
            timespan=args.timespan,
            num_max_cams=args.num_max_cameras,
            load_depth=args.load_depth,
            load_flow=args.load_flow,
            load_dynamic_mask=True,
            load_ground_label=args.load_ground,
            skip_sky_mask=args.skip_sky_mask,
            num_target_chunks=args.num_target_chunks,
            static=args.static,
            reverse=args.reverse,
            args=args
        )
        dataset_eval_flow = UFODatasetEval(
            data_root=args.data_root,
            annotation_txt_file_list=val_annotation,
            target_size=args.input_size,
            num_context_timesteps=args.num_context_timesteps,
            num_target_timesteps=args.num_target_timesteps,
            timespan=args.timespan,
            num_max_cams=args.num_max_cameras,
            load_depth=args.load_depth,
            load_flow=args.load_flow,
            load_dynamic_mask=False,
            load_ground_label=args.load_ground,
            return_context_as_target=True,
            skip_sky_mask=args.skip_sky_mask,
            num_target_chunks=args.num_target_chunks,
            static=args.static,
            reverse=args.reverse,
            args=args
        )
        sampler = NoPaddingDistributedSampler(
            dataset_eval,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
        )
        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            pin_memory=False,
            persistent_workers=True if args.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
        )
        data_loader_eval_flow = torch.utils.data.DataLoader(
            dataset_eval_flow,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            pin_memory=False,
            persistent_workers=True if args.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
        )
    else:
        dataset_val = None
        dataset_eval = None
        dataset_eval_flow = None
        data_loader_eval = None
        data_loader_eval_flow = None

    logger.info(f"Dataset: {args.dataset}, train: {train_annotation}, val: {val_annotation}")
    logger.info(f"Dataset contains {len(dataset_train):,} sequences using {train_annotation}.")

    # ========================================================================
    # Model initialization
    # ========================================================================

    if args.arch not in models.ARCHITECTURES:
        raise ValueError(
            f"Invalid arch: {args.arch!r}. "
            f"Available architectures: {sorted(models.ARCHITECTURES.keys())}"
        )
    arch_models = models.ARCHITECTURES[args.arch]
    if args.model not in arch_models:
        raise ValueError(
            f"Invalid model name {args.model!r} for arch={args.arch!r}. "
            f"Available models for this arch: {sorted(arch_models.keys())}"
        )
    model = arch_models[args.model](
        img_size=args.input_size,
        gs_dim=args.gs_dim,
        decoder_type=args.decoder_type,
        grad_checkpointing=not args.disable_grad_checkpointing,
        use_sky_token=args.use_sky_token,
        use_affine_token=args.use_affine_token,
        num_cams=args.num_max_cameras,
        num_motion_tokens=args.num_motion_tokens,
        use_latest_gsplat=args.use_latest_gsplat,
        max_scale=args.max_gaussian_scale,
        static=args.static,
        num_mem_tokens=args.num_mem_tokens,
        args=args
    )

    logger.info(f"Model = {str(model)}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    model.to(device)
    model_without_ddp = model

    if distributed.is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model_without_ddp = model.module
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    global_batch_size = args.batch_size * world_size * args.gradient_accumulation_steps
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * global_batch_size / 256
    logger.info(
        "Effective global batch size: %d (%d GPUs x %d/GPU x %d accumulation)",
        global_batch_size, world_size, args.batch_size, args.gradient_accumulation_steps,
    )
    logger.info(f"Base lr: {args.lr * 256 / global_batch_size:.2e}, Actual lr: {args.lr:.2e}")

    # ========================================================================
    # Optimizer and loss scaler
    # ========================================================================

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    logger.info(f"Optimizer = {optimizer}")
    logger.info(f"Loss Scaler = {loss_scaler}")

    # Load checkpoint or resume training
    logger.info(f"Original start Iteration: {args.start_iteration}")
    vis_slice_id = misc.load_model(args, model_without_ddp, optimizer, loss_scaler)
    logger.info(f"New start iteration {args.start_iteration}")
    sampler_advance = args.start_iteration * args.gradient_accumulation_steps * args.batch_size
    if hasattr(sampler_train, "set_advance"):
        sampler_train.set_advance(sampler_advance)
        logger.info("Advanced rank-local sampler stream by %d samples", sampler_advance)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {num_trainable_params / 1e6:.2f}M")
    logger.info(f"Training with {world_size} GPUs")

    data_iter_step = args.start_iteration
    if log_writer is not None:
        log_writer.set_step(data_iter_step)

    # ========================================================================
    # Evaluation and visualization
    # ========================================================================

    if args.evaluate:
        eval_result = evaluate(data_loader_eval, model_without_ddp, args)
        if log_writer is not None and eval_result is not None:
            eval_result = {f"eval/{k}": v for k, v in eval_result.items()}
            log_writer.update(eval_result)
        if args.dataset == "waymo" and not args.skip_flow_evaluation:
            if args.decoder_type != "conv":
                flow_eval_result = evaluate_flow(data_loader_eval_flow, model_without_ddp, args)
                if log_writer is not None and flow_eval_result is not None:
                    flow_eval_result = {f"eval/{k}": v for k, v in flow_eval_result.items()}
                    log_writer.update(flow_eval_result)
        logger.info("Evaluation done, exiting.")
        exit()

    valid_slice_id = copy.deepcopy(vis_slice_id)
    if dataset_val is not None and valid_slice_id >= len(dataset_val):
        valid_slice_id = 0

    if not args.skip_initial_validation:
        val(args, model_without_ddp, dataset_val, log_writer=log_writer, output_prefix=os.path.join(args.log_dir, "videos", f"{data_iter_step:07d}"))

    if args.visualization_only:
        logger.info("Visualization done, exiting.")
        exit()

    rgb_and_lpips_loss = RGBLpipsLoss(
        perceptual_weight=args.perceptual_weight,
        enable_perceptual_loss=args.enable_perceptual_loss,
    ).to(device)
    rgb_and_lpips_loss.set_perceptual_loss(False)
    # ========================================================================
    # Training loop
    # ========================================================================

    if args.allow_missing_paper_components:
        logger.warning(
            "MISSING_PAPER_COMPONENT: sprint override active; forward-flow and flex-mask gaps remain"
        )
    else:
        assert_paper_training_ready(args)

    logger.info(f"Starting training from iteration {args.start_iteration} to {args.num_iterations}")
    if distributed.is_main_process():
        save_run_manifest(args, world_size, train_annotation, val_annotation)
    metrics_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    start_time = time.time()
    termination = {"signal": None}
    ddp_smoke_state = {"data_checked": False, "gradient_checked": False, "parameter_checked": False}

    def representative_tensor_checksum(tensor):
        flat = tensor.detach().float().reshape(-1)
        sample = flat[: min(flat.numel(), 4096)]
        weights = torch.linspace(1.0, 2.0, sample.numel(), device=sample.device)
        return torch.stack((sample.sum(), (sample * weights).sum(), sample.square().sum()))

    def assert_all_ranks_close(value, label):
        gathered = [torch.empty_like(value) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, value)
        reference = gathered[0]
        for rank, candidate in enumerate(gathered[1:], start=1):
            if not torch.allclose(reference, candidate, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f"DDP smoke {label} mismatch between rank0 and rank{rank}")

    def batch_fingerprint(batch):
        first = batch[0] if isinstance(batch, (list, tuple)) else batch
        scene_names = first.get("scene_name", [])
        starts = first.get("sample_start_frame", [])
        if torch.is_tensor(starts):
            starts = starts.detach().cpu().tolist()
        return repr((scene_names, starts))

    def request_emergency_checkpoint(signum, _frame):
        termination["signal"] = int(signum)

    signal.signal(signal.SIGTERM, request_emergency_checkpoint)
    signal.signal(signal.SIGINT, request_emergency_checkpoint)

    def save_training_checkpoint(step, filename):
        local_rng_state = misc.capture_rng_state()
        rng_states = [None for _ in range(world_size)]
        if world_size > 1:
            torch.distributed.all_gather_object(rng_states, local_rng_state)
        else:
            rng_states[0] = local_rng_state
        if distributed.is_main_process():
            elapsed_t = time.time() - start_time + args.total_elapsed_time
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_scaler": loss_scaler.state_dict(),
                "latest_step": step,
                "vis_slice_id": vis_slice_id,
                "args": args,
                "total_elapsed_time": elapsed_t,
                "rng_states": rng_states,
                "sampler_advance_per_rank": (step + 1) * args.gradient_accumulation_steps * args.batch_size,
                "scheduler_state": {
                    "type": args.lr_sched,
                    "warmup_iters": args.warmup_iters,
                    "optimizer_step": step + 1,
                },
                "perceptual_loss_active": bool(
                    args.enable_perceptual_loss and step + 1 >= args.perceptual_loss_start_iter
                ),
                "best_validation_psnr": args.best_validation_psnr,
            }
            checkpoint_path = os.path.join(args.ckpt_dir, filename)
            torch.save(checkpoint, checkpoint_path)
            misc.cleanup_checkpoints(args.ckpt_dir, keep_num=args.keep_n_ckpts)
            logger.info("Saved checkpoint to %s", checkpoint_path)
        if world_size > 1:
            torch.distributed.barrier()
    num_tokens_printed = False
    micro_step = args.start_iteration * args.gradient_accumulation_steps
    validation_steps = {
        int(step) for step in args.validation_steps.split(",") if step.strip()
    }
    for data_dict in metric_logger.log_every(
        data_loader_train,
        print_freq=args.log_every_n_iters,
        header="Training",
        n_iterations=args.num_iterations * args.gradient_accumulation_steps,
        start_iteration=micro_step,
    ):
        if micro_step >= args.num_iterations * args.gradient_accumulation_steps:
            break
        accumulation_index = micro_step % args.gradient_accumulation_steps
        should_step = accumulation_index == args.gradient_accumulation_steps - 1
        if args.ddp_smoke_assertions and world_size > 1 and not ddp_smoke_state["data_checked"]:
            local_fingerprint = batch_fingerprint(data_dict)
            fingerprints = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(fingerprints, local_fingerprint)
            if len(set(fingerprints)) == 1:
                raise RuntimeError(f"DDP smoke ranks consumed identical data: {fingerprints[0]}")
            ddp_smoke_state["data_checked"] = True
            logger.info("DDP smoke distinct rank data PASS: %s", fingerprints)
        if accumulation_index == 0:
            optimizer.zero_grad()
            accumulation_loss_dict = {}
            accumulation_metric_dict = {}
            accumulation_scene_diagnostics = []
            misc.adjust_learning_rate(optimizer, data_iter_step, args)
        if log_writer is not None:
            log_writer.set_step(data_iter_step)
        if args.enable_perceptual_loss and data_iter_step >= args.perceptual_loss_start_iter:
            rgb_and_lpips_loss.set_perceptual_loss(True)

        model.train()
        accumulation_sync_context = (
            model.no_sync()
            if (
                args.ddp_accumulation_no_sync
                and world_size > 1
                and hasattr(model, "no_sync")
                and not should_step
            )
            else contextlib.nullcontext()
        )
        accumulation_sync_context.__enter__()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Forward pass and loss computation
            loss_total = 0
            inout_dicts = prepare_inputs_and_targets(data_dict, device, timespan=args.timespan, from_list=True, args=args)


            all_gs_features = {}
            loss_dict_accum = {}

            if args.reverse:
                range_inout = list(range(len(inout_dicts) - 1, -1, -1))
            else:
                range_inout = list(range(len(inout_dicts)))
            final_scene_supervision = args.paper_supervision_mode == "final_scene"
            final_scene_inputs = []
            final_scene_update_outputs = []

            # Autoregressive processing loop
            for chunk_position, i in enumerate(range_inout):
                input_dict, target_dict = inout_dicts[i]

                sync_now = should_step and chunk_position == len(range_inout) - 1
                sync_context = (
                    contextlib.nullcontext()
                    if (
                        world_size == 1
                        or not hasattr(model, "no_sync")
                        or sync_now
                        or not args.sequential_chunk_backward
                    )
                    else model.no_sync()
                )
                with sync_context:
                    pred_dict, all_gs_features = update_scene(
                        input_dict, model, scene=all_gs_features, export_ply=False,
                        profile=False, render=not final_scene_supervision,
                        filter_num=args.filter_num, log_dir=args.log_dir,
                        detach_old_scene=not args.allow_old_scene_grad,
                    )
                    if final_scene_supervision:
                        final_scene_inputs.append((input_dict, target_dict))
                        final_scene_update_outputs.append(pred_dict)
                    else:
                        loss_dict = compute_loss(pred_dict, target_dict, args, rgb_and_lpips_loss)
                        loss_dict.update({'class_loss': pred_dict['class_loss']})
                        if 'ray_loss' in pred_dict:
                            loss_dict.update({'ray_loss': pred_dict['ray_loss']})
                        nonfinite_losses = {
                            key: value.detach().float().cpu().tolist()
                            for key, value in loss_dict.items()
                            if torch.is_tensor(value) and not torch.isfinite(value).all()
                        }
                        if nonfinite_losses:
                            nonfinite_predictions = {
                                key: tuple(value.shape)
                                for key, value in pred_dict.items()
                                if torch.is_tensor(value) and not torch.isfinite(value).all()
                            }
                            logger.error(
                                "Non-finite chunk=%d context_frames=%s target_frames=%s "
                                "losses=%s predictions=%s",
                                i,
                                input_dict.get("frame_idx"),
                                target_dict.get("frame_idx"),
                                nonfinite_losses,
                                nonfinite_predictions,
                            )
                        loss_value = sum(loss for k, loss in loss_dict.items() if "loss" in k)
                        for key, value in loss_dict.items():
                            loss_dict_accum[key] = loss_dict_accum.get(key, 0.0) + value.detach()
                        if args.sequential_chunk_backward:
                            if not args.detach_scene_between_chunks:
                                raise ValueError("--sequential_chunk_backward requires --detach_scene_between_chunks")
                            loss_scaler.backward(loss_value / args.gradient_accumulation_steps)
                            loss_total += loss_value.detach()
                        else:
                            loss_total += loss_value

                if not final_scene_supervision:
                    with torch.no_grad():
                        chunk_metrics = reconstruction_metrics(pred_dict, target_dict)
                        chunk_metrics.update(gaussian_metrics(pred_dict, args.max_gaussian_scale))
                        for key, value in chunk_metrics.items():
                            accumulation_metric_dict[key] = accumulation_metric_dict.get(key, 0.0) + value
                    accumulation_scene_diagnostics.append(pred_dict.get("scene_diagnostics", {}))

                if args.detach_scene_between_chunks:
                    all_gs_features = misc.detach_tensors(all_gs_features)

            if final_scene_supervision:
                if args.sequential_chunk_backward:
                    raise ValueError("final_scene supervision requires one backward after all renders")
                for (render_source, target_dict), update_output in zip(
                    final_scene_inputs, final_scene_update_outputs
                ):
                    render_input = render_source.copy()
                    render_input.update(all_gs_features)
                    render_input = model(render_input, stage=2, motion=False)
                    pred_dict = model(render_input, stage=3)
                    loss_dict = compute_loss(pred_dict, target_dict, args, rgb_and_lpips_loss)
                    if 'class_loss' in update_output:
                        loss_dict['class_loss'] = update_output['class_loss']
                    if 'ray_loss' in update_output:
                        loss_dict['ray_loss'] = update_output['ray_loss']
                    loss_value = sum(loss for key, loss in loss_dict.items() if "loss" in key)
                    loss_total += loss_value
                    for key, value in loss_dict.items():
                        loss_dict_accum[key] = loss_dict_accum.get(key, 0.0) + value.detach()
                    with torch.no_grad():
                        chunk_metrics = reconstruction_metrics(pred_dict, target_dict)
                        chunk_metrics.update(gaussian_metrics(pred_dict, args.max_gaussian_scale))
                        for key, value in chunk_metrics.items():
                            accumulation_metric_dict[key] = accumulation_metric_dict.get(key, 0.0) + value

            loss_dict = {key: value / len(inout_dicts) for key, value in loss_dict_accum.items()}
            for key, value in loss_dict.items():
                accumulation_loss_dict[key] = accumulation_loss_dict.get(key, 0.0) + value

        if not math.isfinite(loss_total):
            logger.info("NaN detected")
            raise AssertionError

        if not args.sequential_chunk_backward:
            loss_scaler.backward(loss_total / args.gradient_accumulation_steps)
        accumulation_sync_context.__exit__(None, None, None)

        micro_step += 1
        if not should_step:
            continue

        if args.ddp_smoke_assertions and world_size > 1 and not ddp_smoke_state["gradient_checked"]:
            gradient = next((parameter.grad for parameter in model.parameters() if parameter.grad is not None), None)
            if gradient is None:
                raise RuntimeError("DDP smoke found no gradient before optimizer step")
            assert_all_ranks_close(representative_tensor_checksum(gradient), "gradient")
            ddp_smoke_state["gradient_checked"] = True
            logger.info("DDP smoke gradient all-reduce PASS")

        grad_norm = loss_scaler.step(
            optimizer, parameters=model.parameters(), clip_grad=args.grad_clip
        )
        grad_group_norms = parameter_grad_norms(model_without_ddp)
        optimizer.zero_grad()
        if args.ddp_smoke_assertions and world_size > 1 and not ddp_smoke_state["parameter_checked"]:
            parameter = next(model.parameters())
            assert_all_ranks_close(representative_tensor_checksum(parameter), "parameter")
            ddp_smoke_state["parameter_checked"] = True
            if distributed.is_main_process():
                status_path = os.path.join(args.log_dir, "ddp_smoke_status.json")
                with open(status_path, "w") as handle:
                    json.dump({
                        "distinct_rank_data": True,
                        "gradient_all_reduce": True,
                        "parameters_synchronized": True,
                        "single_logical_checkpoint_writer": "rank0",
                    }, handle, indent=2, sort_keys=True)
            logger.info("DDP smoke synchronized parameters PASS")
        torch.cuda.synchronize()
        loss_dict = {
            key: value / args.gradient_accumulation_steps
            for key, value in accumulation_loss_dict.items()
        }
        diagnostic_divisor = args.gradient_accumulation_steps * len(inout_dicts)
        training_diagnostics = {
            key: value / diagnostic_divisor for key, value in accumulation_metric_dict.items()
        }
        if world_size > 1:
            [torch.distributed.all_reduce(v) for v in loss_dict.values()]
        loss_dict_reduced = {k: v.item() / world_size for k, v in loss_dict.items()}
        total_loss_reduced = sum(loss for k, loss in loss_dict_reduced.items() if "loss" in k)
        lr = optimizer.param_groups[0]["lr"]
        psnr = -10 * np.log10(loss_dict_reduced["rgb_loss"])
        metric_logger.update(lr=lr, psnr=psnr, loss=total_loss_reduced, **loss_dict_reduced)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(peak_gpu_mb=torch.cuda.max_memory_allocated() / (1024 ** 2))
        metric_logger.update(**training_diagnostics, **grad_group_norms)

        if "num_tokens" in pred_dict and not num_tokens_printed:
            logger.info(f"num_tokens: {pred_dict['num_tokens']}")
            num_tokens_printed = True

        if log_writer is not None:
            log_writer.update(
                {
                    "train/psnr": psnr,
                    "train/loss": total_loss_reduced,
                    **{f"train/{k}": v for k, v in loss_dict_reduced.items()},
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    **{f"train/{k}": v for k, v in training_diagnostics.items()},
                    **{f"train/{k}": v for k, v in grad_group_norms.items()},
                }
            )
            for diagnostic in accumulation_scene_diagnostics[-len(inout_dicts):]:
                chunk = diagnostic.get("chunk")
                if chunk is not None:
                    log_writer.update({
                        f"scene/chunk{chunk}/{key}": value
                        for key, value in diagnostic.items()
                        if key != "chunk" and isinstance(value, (int, float))
                    })
            # Flush TensorBoard periodically
            if isinstance(log_writer, TensorBoardLogger) and data_iter_step % 100 == 0:
                log_writer.flush()

        if (data_iter_step + 1) % args.ckpt_every_n_iters == 0:
            save_training_checkpoint(data_iter_step, f"ckpt_{data_iter_step:06d}.pth")
            torch.cuda.empty_cache()

        termination_flag = torch.tensor(
            int(termination["signal"] is not None), device=device, dtype=torch.int32
        )
        if world_size > 1:
            torch.distributed.all_reduce(termination_flag, op=torch.distributed.ReduceOp.MAX)
        if termination_flag.item():
            save_training_checkpoint(data_iter_step, f"emergency_{data_iter_step:06d}.pth")
            logger.warning("Emergency checkpoint completed after termination request")
            return

        completed_step = data_iter_step + 1
        if (
            completed_step in validation_steps
            or (args.vis_every_n_iters > 0 and completed_step % args.vis_every_n_iters == 0)
        ):
            validation_metrics = val(
                args,
                model_without_ddp,
                dataset_val,
                log_writer=log_writer,
                output_prefix=os.path.join(args.log_dir, "videos", f"{data_iter_step:07d}"),
            )
            validation_psnr = float(validation_metrics["psnr"])
            if validation_psnr > args.best_validation_psnr:
                args.best_validation_psnr = validation_psnr
                save_training_checkpoint(data_iter_step, "best.pth")
                logger.info(
                    "New validation-best checkpoint: step=%d psnr=%.4f",
                    completed_step,
                    validation_psnr,
                )

        data_iter_step += 1

    metric_logger.synchronize_between_processes()

    # ========================================================================
    # Final evaluation
    # ========================================================================

    total_time = time.time() - start_time + args.total_elapsed_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    if args.skip_final_evaluation:
        logger.info("Skipping final evaluation by request.")
        return
    eval_result = evaluate(data_loader_eval, model_without_ddp, args)
    if log_writer is not None and eval_result is not None:
        log_writer.update({f"eval/{k}": v for k, v in eval_result.items()})
    if args.decoder_type != "conv" and args.dataset == "waymo":
        flow_eval_result = evaluate_flow(data_loader_eval_flow, model_without_ddp, args)
        if log_writer is not None and flow_eval_result is not None:
            log_writer.update({f"eval/{k}": v for k, v in flow_eval_result.items()})
    logger.info("Done!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    from ufo.utils.config import merge_config_and_args

    parser = get_args_parser()

    # Default config file path
    default_config = Path("config.json")

    # First parse to check if --config was specified
    temp_args = parser.parse_args()

    # Use specified config, or default if it exists
    config_path = temp_args.config if temp_args.config else (str(default_config) if default_config.exists() else None)

    # Merge config and args (CLI args take precedence)
    args = merge_config_and_args(parser, config_path=config_path)

    main(args)
