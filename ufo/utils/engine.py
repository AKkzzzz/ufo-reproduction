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

import datetime
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from einops import rearrange, repeat

import ufo.utils.distributed as distributed
from ufo.dataset.constants import MEAN, STD
from ufo.dataset.data_utils import prepare_inputs_and_targets
from ufo.utils.losses import compute_scene_flow_metrics
from ufo.visualization.video_maker import make_video
from ufo.utils.misc import combine_dict_entries, project_boxes_to_image
from ufo.utils.misc import compute_point_visibility, compute_visible_topk_indices_any_view, batched_index_gather, batched_index_update
from ufo.utils.misc import combine_dict_entries, project_boxes_to_image, convert_to_chunks
from ufo.utils.misc import update_scene
# Import depth evaluation functions
from reference_depth_eval import depth_evaluation

logger = logging.getLogger("UFO")


def evaluation_target_indices(num_timesteps, paper_frame_protocol):
    """Select supervision targets without re-filtering paper-protocol windows."""
    if paper_frame_protocol:
        return list(range(num_timesteps))
    return [idx for idx in range(num_timesteps) if idx % 5 != 0]


def resize_binary_mask_nearest(mask, output_size):
    """Resize a ``[..., H, W]`` mask without introducing fractional labels."""
    shape = mask.shape
    return F.interpolate(
        mask.reshape(-1, 1, *shape[-2:]).float(), size=output_size, mode="nearest"
    ).reshape(*shape[:-2], *output_size)


@torch.no_grad()
def visualize(args, model, dset_train, step, train_vis_id, device, dset_val=None, val_vis_id=None, log_writer=None):
    model.eval()
    global_rank = distributed.get_global_rank()
    split = "train"
    for vis_id, dataset, dsettype in zip([train_vis_id, val_vis_id], [dset_train, dset_val], ["train", "val"]):
        if vis_id is None or dataset is None:  # sometimes there is no validation set
            continue

        sample_id = global_rank * 80 + vis_id
        out_pth = f"{args.video_dir}/step{step}-rank{global_rank}-sample{sample_id}-{split}.mp4"

        logger.info(f"saving video to {out_pth}")
        make_video(
            args,
            dataset,
            model,
            device,
            output_filename=out_pth,
            scene_id=sample_id,
            skip_plot_gt_depth_and_flow=False,
            eval_metrics= dsettype == "val",
            log_writer=log_writer
        )

        logger.info(f"saved video to {out_pth}")
        split = "val"

    torch.cuda.empty_cache()
    return train_vis_id + 1, val_vis_id + 1 if val_vis_id is not None else None


@torch.no_grad()
def evaluate(dataloader, model, args, name_str=None):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    mean = torch.tensor(MEAN).to(device)
    std = torch.tensor(STD).to(device)

    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to {eval_result_dir}")
    # use yr-mo-dy-hr-min
    if name_str is None:
        name_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    def get_numpy(tensor):
        return tensor.squeeze().detach().cpu().numpy()

    # Initialize running sums and counts
    total_samples, total_dynamic_samples, total_valid_dynamic_depth_samples = 0, 0, 0
    total_dynamic_frame_count, total_dynamic_pixel_count = 0, 0
    total_psnr, total_ssim, total_depth_rmse = 0.0, 0.0, 0.0
    total_occupied_psnr, total_occupied_ssim = 0.0, 0.0
    total_dynamic_psnr, total_dynamic_ssim, total_dynamic_rmse = 0.0, 0.0, 0.0

    # Comprehensive depth metrics from reference_depth_eval.py
    total_depth_abs_rel, total_depth_sq_rel, total_depth_log_rmse = 0.0, 0.0, 0.0
    total_depth_delta1, total_depth_delta2, total_depth_delta3 = 0.0, 0.0, 0.0
    total_valid_depth_samples = 0

    # test_indices = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
    printed = False
    pbar = tqdm(dataloader, desc="Evaluating")
    for data_dict in pbar:
        inout_dicts = prepare_inputs_and_targets(data_dict, device, timespan=args.timespan, from_list=True, args=args)
        gs_state = None
        mis_state = None
        prev_gs_time, prev_gs_dirs, prev_gs_origins = None, None, None
        prev_gs_c2w_global = None
        all_gs_features = {}
        posterior_gs_state_means = None
        posterior_gs_features = None
        window_size = args.num_window_chunks

        if args.reverse:
            range_inout = range(len(inout_dicts) - 1, -1, -1)
        else:
            range_inout = range(len(inout_dicts))
        # Collect every recurrent chunk's supervision targets. Paper-protocol
        # targets are already separated from context by the dataset.
        all_pred_rgb = []
        all_pred_depth = []
        all_gt_rgb = []
        all_gt_depth = []
        all_gt_sky_mask = []
        all_gt_dynamic_mask = []

        for i in range_inout:
            input_dict, target_dict = inout_dicts[i]
            render_source, all_gs_features = update_scene(
                input_dict, model, scene=all_gs_features, export_ply=False,
                profile=True, filter_num=args.filter_num, log_dir=args.log_dir,
            )
            num_timesteps = target_dict['target_image'].shape[1]
            test_indices = evaluation_target_indices(
                num_timesteps, getattr(args, "paper_frame_protocol", False)
            )
            if not test_indices:
                logger.warning("No evaluation targets selected for recurrent chunk %d", i)
                continue

            render_batch_size = 5
            num_batches = (len(test_indices) + render_batch_size - 1) // render_batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * render_batch_size
                end_idx = min(start_idx + render_batch_size, len(test_indices))
                batch_test_indices = test_indices[start_idx:end_idx]

                batch_input_dict = {}
                for key in render_source:
                    if key.startswith('target_') and isinstance(render_source[key], torch.Tensor):
                        if render_source[key].dim() >= 2:
                            batch_input_dict[key] = render_source[key][:, batch_test_indices]
                        else:
                            batch_input_dict[key] = render_source[key]
                    else:
                        batch_input_dict[key] = render_source[key]

                batch_pred_dict = model(batch_input_dict, stage=3)
                batch_rendered_results = batch_pred_dict["render_results"]
                batch_pred_rgb = (
                    batch_rendered_results[batch_rendered_results["rgb_key"]] * std + mean
                ).detach()
                if batch_rendered_results["decoder_depth_key"] is None:
                    batch_pred_depth = batch_rendered_results[batch_rendered_results["depth_key"]]
                else:
                    batch_pred_depth = batch_rendered_results[
                        batch_rendered_results["decoder_depth_key"]
                    ]

                batch_gt_rgb = target_dict["target_image"][:, batch_test_indices]
                batch_gt_rgb = batch_gt_rgb.permute(0, 1, 2, 4, 5, 3) * std + mean
                batch_gt_depth = target_dict["target_depth"][:, batch_test_indices]
                batch_gt_sky_mask = target_dict["target_sky_masks"][:, batch_test_indices]
                batch_gt_dynamic_mask = target_dict.get("target_dynamic_masks")
                if batch_gt_dynamic_mask is not None:
                    batch_gt_dynamic_mask = batch_gt_dynamic_mask[:, batch_test_indices]
                    batch_gt_dynamic_mask = resize_binary_mask_nearest(
                        batch_gt_dynamic_mask, batch_gt_rgb.shape[-3:-1]
                    )

                all_pred_rgb.append(batch_pred_rgb)
                all_pred_depth.append(batch_pred_depth)
                all_gt_rgb.append(batch_gt_rgb)
                all_gt_depth.append(batch_gt_depth)
                all_gt_sky_mask.append(batch_gt_sky_mask)
                all_gt_dynamic_mask.append(batch_gt_dynamic_mask)
                torch.cuda.empty_cache()

        if not all_pred_rgb:
            logger.warning("Skipping sample with no selected supervision targets")
            continue

        # Concatenate all batches
        pred_rgb = torch.cat(all_pred_rgb, dim=1)
        pred_depth = torch.cat(all_pred_depth, dim=1)
        gt_rgb = torch.cat(all_gt_rgb, dim=1)
        gt_depth = torch.cat(all_gt_depth, dim=1)
        gt_sky_mask = torch.cat(all_gt_sky_mask, dim=1)

        if all_gt_dynamic_mask and all(mask is not None for mask in all_gt_dynamic_mask):
            gt_dynamic_mask = torch.cat(all_gt_dynamic_mask, dim=1)
        else:
            gt_dynamic_mask = None

            ####================================forward pass==============================================

        # evaluate on real target images:
        # b, t, v, c, h, w (already have the data from batched processing above)

        # Reshape for metric calculation
        height, width = gt_rgb.shape[-3], gt_rgb.shape[-2]
        # btv, h, w, c
        gt_rgb = gt_rgb.reshape(-1, height, width, 3)
        pred_rgb = pred_rgb.reshape(-1, height, width, 3)
        pred_rgb = torch.clamp(pred_rgb, 0, 1)

        gt_depth = gt_depth.view(-1, height, width)
        pred_depth = pred_depth.view(-1, height, width)
        gt_sky_mask = gt_sky_mask.view(-1, height, width)

        occupied_mask = (gt_sky_mask == 0).bool()
        if gt_dynamic_mask is not None:
            gt_dynamic_mask = gt_dynamic_mask.view(-1, height, width)
            dynamic_mask = gt_dynamic_mask.bool()
        else:
            dynamic_mask = torch.zeros_like(occupied_mask)
        valid_depth_mask = gt_depth > 0.0
        total_dynamic_frame_count += int(dynamic_mask.flatten(1).any(dim=1).sum())
        total_dynamic_pixel_count += int(dynamic_mask.sum())

        psnrs, ssim_scores, depth_rmses = [], [], []
        occupied_ssims, occupied_psnrs = [], []
        dynamic_ssims, dynamic_psnrs, dynamic_depth_rmses = [], [], []
        for i in range(len(gt_rgb)):
            ssim_score = ssim(
                get_numpy(pred_rgb[i]),
                get_numpy(gt_rgb[i]),
                data_range=1.0,
                channel_axis=-1,
            )
            ssim_scores.append(ssim_score)
            occupied_ssims.append(
                ssim(
                    get_numpy(pred_rgb[i]),
                    get_numpy(gt_rgb[i]),
                    data_range=1.0,
                    channel_axis=-1,
                    full=True,
                )[1][get_numpy(occupied_mask[i])].mean()
            )
            psnrs.append(
                -10
                * torch.log10(
                    F.mse_loss(
                        pred_rgb[i],
                        gt_rgb[i],
                    )
                ).item()
            )
            occupied_psnrs.append(
                -10
                * torch.log10(
                    F.mse_loss(
                        pred_rgb[i][occupied_mask[i]],
                        gt_rgb[i][occupied_mask[i]],
                    )
                ).item()
            )
            depth_rms = torch.sqrt(
                F.mse_loss(
                    pred_depth[i][valid_depth_mask[i]],
                    gt_depth[i][valid_depth_mask[i]],
                )
            ).item()
            depth_rmses.append(depth_rms)

            # Comprehensive depth metrics using depth_evaluation
            try:
                depth_metrics, _, _, _ = depth_evaluation(
                    pred_depth[i].unsqueeze(0),
                    gt_depth[i].unsqueeze(0),
                    max_depth=80,
                    custom_mask=None,
                    align_with_lstsq=False,
                    use_gpu=torch.cuda.is_available()
                )
                total_depth_abs_rel += depth_metrics["Abs Rel"]
                total_depth_sq_rel += depth_metrics["Sq Rel"]
                total_depth_log_rmse += depth_metrics["Log RMSE"]
                total_depth_delta1 += depth_metrics["δ < 1.25"]
                total_depth_delta2 += depth_metrics["δ < 1.25^2"]
                total_depth_delta3 += depth_metrics["δ < 1.25^3"]
                total_valid_depth_samples += 1
            except Exception as e:
                logger.warning(f"Depth evaluation failed for sample {i}: {e}")
            if dynamic_mask[i].sum() == 0:
                continue
            dynamic_ssims.append(
                ssim(
                    get_numpy(pred_rgb[i]),
                    get_numpy(gt_rgb[i]),
                    data_range=1.0,
                    channel_axis=-1,
                    full=True,
                )[1][get_numpy(dynamic_mask[i])].mean()
            )
            dynamic_psnrs.append(
                -10
                * torch.log10(
                    F.mse_loss(
                        pred_rgb[i][dynamic_mask[i]],
                        gt_rgb[i][dynamic_mask[i]],
                    )
                ).item()
            )

            total_dynamic_samples += 1
            _valid_depth_mask = dynamic_mask[i] & valid_depth_mask[i]
            if _valid_depth_mask.sum() == 0:
                continue
            dynamic_depth_rms = torch.sqrt(
                F.mse_loss(
                    pred_depth[i][dynamic_mask[i] & valid_depth_mask[i]],
                    gt_depth[i][dynamic_mask[i] & valid_depth_mask[i]],
                )
            ).item()
            dynamic_depth_rmses.append(dynamic_depth_rms)
            total_valid_dynamic_depth_samples += 1

        psnr_sum = np.sum(psnrs)
        ssim_sum = np.sum(ssim_scores)
        depth_rmse_sum = np.sum(depth_rmses)
        occupied_ssim_sum = np.sum(occupied_ssims)
        occupied_psnr_sum = np.sum(occupied_psnrs)
        dynamic_ssim_sum = np.sum(dynamic_ssims)
        dynamic_psnr_sum = np.sum(dynamic_psnrs)
        dynamic_depth_rmse_sum = np.sum(dynamic_depth_rmses)
        batch_size = len(gt_rgb)
        # Update running sums and counts
        total_psnr += psnr_sum
        total_ssim += ssim_sum
        total_depth_rmse += depth_rmse_sum
        total_occupied_psnr += occupied_psnr_sum
        total_occupied_ssim += occupied_ssim_sum
        total_dynamic_psnr += dynamic_psnr_sum
        total_dynamic_ssim += dynamic_ssim_sum
        total_dynamic_rmse += dynamic_depth_rmse_sum
        total_samples += batch_size
        def fmt(x):
            return f"{x:.4f}"

        pbar.set_postfix(
            psnr=fmt(psnr_sum / batch_size),
            ssim=fmt(ssim_sum / batch_size),
            depth_rmse=fmt(depth_rmse_sum / batch_size),
            avg_psnr=fmt(total_psnr / total_samples),
            avg_occupied_psnr=fmt(total_occupied_psnr / total_samples),
            total_occupied_ssim=fmt(total_occupied_ssim / total_samples),
            avg_depth_rmse=fmt(total_depth_rmse / total_samples),
            avg_dynamic_psnr=(
                fmt(total_dynamic_psnr / total_dynamic_samples)
                if total_dynamic_samples else "NO_VALID_DYNAMIC_PIXELS"
            ),
            avg_dynamic_depth_rmse=(
                fmt(total_dynamic_rmse / total_valid_dynamic_depth_samples)
                if total_valid_dynamic_depth_samples else "NO_VALID_DYNAMIC_DEPTH"
            ),
        )

    # Create tensors for sums and counts
    total_psnr_tensor = torch.tensor(total_psnr, device=device)
    total_ssim_tensor = torch.tensor(total_ssim, device=device)
    total_depth_rmse_tensor = torch.tensor(total_depth_rmse, device=device)
    total_occupied_psnr_tensor = torch.tensor(total_occupied_psnr, device=device)
    total_occupied_ssim_tensor = torch.tensor(total_occupied_ssim, device=device)
    total_dynamic_psnr_tensor = torch.tensor(total_dynamic_psnr, device=device)
    total_dynamic_ssim_tensor = torch.tensor(total_dynamic_ssim, device=device)
    total_dynamic_rmse_tensor = torch.tensor(total_dynamic_rmse, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    total_dynamic_samples_tensor = torch.tensor(total_dynamic_samples, device=device)
    total_valid_dynamic_depth_samples_tensor = torch.tensor(
        total_valid_dynamic_depth_samples, device=device
    )
    total_dynamic_frame_count_tensor = torch.tensor(total_dynamic_frame_count, device=device)
    total_dynamic_pixel_count_tensor = torch.tensor(total_dynamic_pixel_count, device=device)

    # Comprehensive depth metrics tensors
    total_depth_abs_rel_tensor = torch.tensor(total_depth_abs_rel, device=device)
    total_depth_sq_rel_tensor = torch.tensor(total_depth_sq_rel, device=device)
    total_depth_log_rmse_tensor = torch.tensor(total_depth_log_rmse, device=device)
    total_depth_delta1_tensor = torch.tensor(total_depth_delta1, device=device)
    total_depth_delta2_tensor = torch.tensor(total_depth_delta2, device=device)
    total_depth_delta3_tensor = torch.tensor(total_depth_delta3, device=device)
    total_valid_depth_samples_tensor = torch.tensor(total_valid_depth_samples, device=device)

    torch.cuda.synchronize()

    if distributed.is_enabled():
        # Aggregate sums across all processes
        torch.distributed.all_reduce(total_psnr_tensor)
        torch.distributed.all_reduce(total_ssim_tensor)
        torch.distributed.all_reduce(total_depth_rmse_tensor)
        torch.distributed.all_reduce(total_occupied_psnr_tensor)
        torch.distributed.all_reduce(total_occupied_ssim_tensor)
        torch.distributed.all_reduce(total_dynamic_psnr_tensor)
        torch.distributed.all_reduce(total_dynamic_ssim_tensor)
        torch.distributed.all_reduce(total_dynamic_rmse_tensor)
        torch.distributed.all_reduce(total_samples_tensor)
        torch.distributed.all_reduce(total_dynamic_samples_tensor)
        torch.distributed.all_reduce(total_valid_dynamic_depth_samples_tensor)
        torch.distributed.all_reduce(total_dynamic_frame_count_tensor)
        torch.distributed.all_reduce(total_dynamic_pixel_count_tensor)

        # Aggregate comprehensive depth metrics
        torch.distributed.all_reduce(total_depth_abs_rel_tensor)
        torch.distributed.all_reduce(total_depth_sq_rel_tensor)
        torch.distributed.all_reduce(total_depth_log_rmse_tensor)
        torch.distributed.all_reduce(total_depth_delta1_tensor)
        torch.distributed.all_reduce(total_depth_delta2_tensor)
        torch.distributed.all_reduce(total_depth_delta3_tensor)
        torch.distributed.all_reduce(total_valid_depth_samples_tensor)
    result = None
    if distributed.is_main_process():
        avg_psnr = total_psnr_tensor.item() / total_samples_tensor.item()
        avg_ssim = total_ssim_tensor.item() / total_samples_tensor.item()
        avg_depth_rmse = total_depth_rmse_tensor.item() / total_samples_tensor.item()
        avg_occupied_psnr = total_occupied_psnr_tensor.item() / total_samples_tensor.item()
        avg_occupied_ssim = total_occupied_ssim_tensor.item() / total_samples_tensor.item()
        dynamic_count = total_dynamic_samples_tensor.item()
        dynamic_depth_count = total_valid_dynamic_depth_samples_tensor.item()
        avg_dynamic_psnr = (
            total_dynamic_psnr_tensor.item() / dynamic_count if dynamic_count else None
        )
        avg_dynamic_ssim = (
            total_dynamic_ssim_tensor.item() / dynamic_count if dynamic_count else None
        )
        avg_dynamic_rmse = (
            total_dynamic_rmse_tensor.item() / dynamic_depth_count
            if dynamic_depth_count else None
        )
        dynamic_status = "OK" if dynamic_count else "NO_VALID_DYNAMIC_PIXELS"

        def _metric_text(value):
            return f"{value:.4f}" if value is not None else dynamic_status

        # Compute comprehensive depth metric averages
        def _safe_avg(tensor, count_tensor):
            count = count_tensor.item()
            return (tensor.item() / count) if count > 0 else float("nan")

        avg_depth_abs_rel = _safe_avg(total_depth_abs_rel_tensor, total_valid_depth_samples_tensor)
        avg_depth_sq_rel = _safe_avg(total_depth_sq_rel_tensor, total_valid_depth_samples_tensor)
        avg_depth_log_rmse = _safe_avg(total_depth_log_rmse_tensor, total_valid_depth_samples_tensor)
        avg_depth_delta1 = _safe_avg(total_depth_delta1_tensor, total_valid_depth_samples_tensor)
        avg_depth_delta2 = _safe_avg(total_depth_delta2_tensor, total_valid_depth_samples_tensor)
        avg_depth_delta3 = _safe_avg(total_depth_delta3_tensor, total_valid_depth_samples_tensor)
        with open(os.path.join(eval_result_dir, f"eval_{name_str}.txt"), "w") as f:
            # RGB metrics
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average Occupied PSNR: {avg_occupied_psnr:.4f}\n")
            f.write(f"Average Occupied SSIM: {avg_occupied_ssim:.4f}\n")
            f.write(f"Average Dynamic PSNR: {_metric_text(avg_dynamic_psnr)}\n")
            f.write(f"Average Dynamic SSIM: {_metric_text(avg_dynamic_ssim)}\n")
            f.write(f"Dynamic Status: {dynamic_status}\n")
            f.write(f"Dynamic Frame Count: {int(total_dynamic_frame_count_tensor.item())}\n")
            f.write(f"Dynamic Pixel Count: {int(total_dynamic_pixel_count_tensor.item())}\n")

            # Basic depth metrics
            f.write(f"Average Depth RMSE: {avg_depth_rmse:.4f}\n")
            f.write(f"Average Dynamic Depth RMSE: {_metric_text(avg_dynamic_rmse)}\n")

            # Comprehensive depth metrics
            f.write(f"\nComprehensive Depth Metrics:\n")
            f.write(f"Average Depth Abs Rel: {avg_depth_abs_rel:.4f}\n")
            f.write(f"Average Depth Sq Rel: {avg_depth_sq_rel:.4f}\n")
            f.write(f"Average Depth Log RMSE: {avg_depth_log_rmse:.4f}\n")
            f.write(f"Average Depth δ < 1.25: {avg_depth_delta1:.4f}\n")
            f.write(f"Average Depth δ < 1.25^2: {avg_depth_delta2:.4f}\n")
            f.write(f"Average Depth δ < 1.25^3: {avg_depth_delta3:.4f}\n")
        logger.info("Evaluation results saved.")
        logger.info(f"Evaluated on {total_samples_tensor.item()} samples.")
        logger.info(
            f"Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average Depth RMSE: {avg_depth_rmse:.4f}"
        )
        logger.info(
            f"Average Occupied PSNR: {avg_occupied_psnr:.4f}, Average Occupied SSIM: {avg_occupied_ssim:.4f}"
        )
        logger.info(
            "Average Dynamic PSNR: %s, Average Dynamic SSIM: %s, "
            "Average Dynamic Depth RMSE: %s, frames=%d, pixels=%d, status=%s",
            _metric_text(avg_dynamic_psnr), _metric_text(avg_dynamic_ssim),
            _metric_text(avg_dynamic_rmse), int(total_dynamic_frame_count_tensor.item()),
            int(total_dynamic_pixel_count_tensor.item()), dynamic_status,
        )
        logger.info(
            f"Comprehensive Depth Metrics - Abs Rel: {avg_depth_abs_rel:.4f}, Sq Rel: {avg_depth_sq_rel:.4f}, Log RMSE: {avg_depth_log_rmse:.4f}"
        )
        logger.info(
            f"Depth Accuracy - δ<1.25: {avg_depth_delta1:.4f}, δ<1.25^2: {avg_depth_delta2:.4f}, δ<1.25^3: {avg_depth_delta3:.4f}"
        )
        result = {
            # RGB metrics
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "occupied_psnr": avg_occupied_psnr,
            "occupied_ssim": avg_occupied_ssim,
            "dynamic_psnr": avg_dynamic_psnr,
            "dynamic_ssim": avg_dynamic_ssim,

            # Basic depth metrics
            "depth_rmse": avg_depth_rmse,
            "dynamic_depth_rmse": avg_dynamic_rmse,
            "dynamic_status": dynamic_status,
            "dynamic_frame_count": int(total_dynamic_frame_count_tensor.item()),
            "dynamic_pixel_count": int(total_dynamic_pixel_count_tensor.item()),

            # Comprehensive depth metrics from reference_depth_eval.py
            "depth_abs_rel": avg_depth_abs_rel,
            "depth_sq_rel": avg_depth_sq_rel,
            "depth_log_rmse": avg_depth_log_rmse,
            "depth_delta_1.25": avg_depth_delta1,
            "depth_delta_1.25^2": avg_depth_delta2,
            "depth_delta_1.25^3": avg_depth_delta3,
        }
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def evaluate_flow(dataloader, model, args, name_str=None):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)
    logger.info(f"Saving evaluation results to {eval_result_dir}")
    # use yr-mo-dy-hr-min
    if name_str is None:
        name_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    (
        total_flow_epes,
        total_flow_accs_strict,
        total_flow_accs_relax,
        total_flow_angles,
        total_flow_rmse,
        total_numb_flow_samples,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pbar = tqdm(dataloader, desc="Evaluating")
    for data_dict in pbar:
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device)
        pred_dict = model(input_dict)
        # evaluate on real target images:
        # b, t, v, c, h, w
        b, t, v, height, width = target_dict["target_depth"].shape
        gt_depth = target_dict["target_depth"].view(b * t, -1, height, width)
        num_imgs = gt_depth.shape[0]
        valid_depth_mask = gt_depth > 0.0
        rendered_results = pred_dict["render_results"]
        if args.load_ground:
            gt_ground_mask = target_dict["target_ground_masks"].view(b * t, -1, height, width)
            gt_ground_mask = gt_ground_mask.bool()
        num_valid_samples = 0
        eval_flow = (
            "rendered_flow" in rendered_results
            and args.decoder_type == "dummy"
            and args.load_flow
            and "target_flow" in target_dict
        )
        if eval_flow:
            gt_flow = target_dict["target_flow"].view(b * t, -1, height, width, 3)
            pred_flow = rendered_results["rendered_flow"].view(b * t, -1, height, width, 3)
            # pred_flow = gt_flow.clone() + torch.rand_like(gt_flow) * 0.05
            flow_epes, flow_accs_strict, flow_accs_relax, flow_angles = [], [], [], []
            flow_rmse = []

            for i in range(num_imgs):
                if torch.max(gt_flow.norm(dim=-1)) > 1.0:
                    if args.load_ground:
                        non_ground_gt_flow = gt_flow[i][~gt_ground_mask[i] & valid_depth_mask[i]]
                        non_ground_pred_flow = pred_flow[i][
                            ~gt_ground_mask[i] & valid_depth_mask[i]
                        ]
                    else:
                        non_ground_gt_flow = gt_flow[i][valid_depth_mask[i]]
                        non_ground_pred_flow = pred_flow[i][valid_depth_mask[i]]
                    flow_metrics = compute_scene_flow_metrics(
                        non_ground_pred_flow, non_ground_gt_flow
                    )
                    flow_epes.append(flow_metrics["EPE3D"])
                    flow_accs_strict.append(flow_metrics["acc3d_strict"] * 100)
                    flow_accs_relax.append(flow_metrics["acc3d_relax"] * 100)
                    flow_angles.append(flow_metrics["angle_error"])
                    flow_rmse.append(
                        torch.sqrt(
                            F.mse_loss(
                                pred_flow[i][valid_depth_mask[i]],
                                gt_flow[i][valid_depth_mask[i]],
                            )
                        ).item()
                    )
                    num_valid_samples += 1

            flow_epe_sum = np.sum(flow_epes)
            flow_acc_strict_sum = np.sum(flow_accs_strict)
            flow_acc_relax_sum = np.sum(flow_accs_relax)
            flow_angle_sum = np.sum(flow_angles)
            flow_rmse_sum = np.sum(flow_rmse)
            valid_flow_samples = num_valid_samples

            # Update running sums and counts
            total_flow_epes += flow_epe_sum
            total_flow_accs_strict += flow_acc_strict_sum
            total_flow_accs_relax += flow_acc_relax_sum
            total_flow_angles += flow_angle_sum
            total_flow_rmse += flow_rmse_sum
            total_numb_flow_samples += valid_flow_samples

        pbar.set_postfix(
            avg_flow_epe=total_flow_epes / total_numb_flow_samples,
            avg_flow_acc_relax=total_flow_accs_relax / total_numb_flow_samples,
            avg_flow_acc_strict=total_flow_accs_strict / total_numb_flow_samples,
            avg_flow_angle=total_flow_angles / total_numb_flow_samples,
            avg_flow_rmse=total_flow_rmse / total_numb_flow_samples,
        )

    # Create tensors for sums and counts
    result = None
    if eval_flow:
        total_flow_epes_tensor = torch.tensor(total_flow_epes, device=device)
        total_flow_accs_strict_tensor = torch.tensor(total_flow_accs_strict, device=device)
        total_flow_accs_relax_tensor = torch.tensor(total_flow_accs_relax, device=device)
        total_flow_angles_tensor = torch.tensor(total_flow_angles, device=device)
        total_flow_rmse_tensor = torch.tensor(total_flow_rmse, device=device)
        total_numb_flow_samples_tensor = torch.tensor(total_numb_flow_samples, device=device)

        torch.cuda.synchronize()

        if distributed.is_enabled():
            # Aggregate sums across all processes
            torch.distributed.all_reduce(total_flow_epes_tensor)
            torch.distributed.all_reduce(total_flow_accs_strict_tensor)
            torch.distributed.all_reduce(total_flow_accs_relax_tensor)
            torch.distributed.all_reduce(total_flow_angles_tensor)
            torch.distributed.all_reduce(total_flow_rmse_tensor)
            torch.distributed.all_reduce(total_numb_flow_samples_tensor)
        if distributed.is_main_process() and total_numb_flow_samples_tensor.item() > 0:
            avg_flow_epe = total_flow_epes_tensor.item() / total_numb_flow_samples_tensor.item()
            avg_flow_acc_strict = (
                total_flow_accs_strict_tensor.item() / total_numb_flow_samples_tensor.item()
            )
            avg_flow_acc_relax = (
                total_flow_accs_relax_tensor.item() / total_numb_flow_samples_tensor.item()
            )
            avg_flow_angle = total_flow_angles_tensor.item() / total_numb_flow_samples_tensor.item()
            avg_flow_rmse = total_flow_rmse_tensor.item() / total_numb_flow_samples_tensor.item()
            with open(os.path.join(eval_result_dir, f"eval_{name_str}_flow.txt"), "w") as f:
                f.write(f"Average Flow EPE: {avg_flow_epe:.4f}\n")
                f.write(f"Average Flow Acc Strict: {avg_flow_acc_strict:.4f}\n")
                f.write(f"Average Flow Acc Relax: {avg_flow_acc_relax:.4f}\n")
                f.write(f"Average Flow Angle: {avg_flow_angle:.4f}\n")
                f.write(f"Average Flow RMSE: {avg_flow_rmse:.4f}\n")
            logger.info("Evaluation results saved.")
            logger.info(f"Evaluated on {total_numb_flow_samples_tensor.item()} samples.")
            logger.info(
                f"Average Flow EPE: {avg_flow_epe:.4f}, Average Flow Acc Strict: {avg_flow_acc_strict:.4f}, Average Flow Acc Relax: {avg_flow_acc_relax:.4f}, Average Flow Angle: {avg_flow_angle:.4f}"
            )
            logger.info(f"Average Flow RMSE: {avg_flow_rmse:.4f}")
            result = {
                "flow_epe": avg_flow_epe,
                "flow_acc_strict": avg_flow_acc_strict,
                "flow_acc_relax": avg_flow_acc_relax,
                "flow_angle": avg_flow_angle,
                "flow_rmse": avg_flow_rmse,
            }
    torch.cuda.empty_cache()
    return result
