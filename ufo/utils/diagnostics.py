"""Training diagnostics for the clean reproduction run."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

from ufo.dataset.constants import MEAN, STD


def tensor_summary(tensor, prefix):
    values = tensor.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {f"{prefix}_{name}": float("nan") for name in ("mean", "median", "p10", "p90", "p99", "min", "max")}
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9, 0.99], device=values.device))
    return {
        f"{prefix}_mean": values.mean().item(),
        f"{prefix}_median": quantiles[1].item(),
        f"{prefix}_p10": quantiles[0].item(),
        f"{prefix}_p90": quantiles[2].item(),
        f"{prefix}_p99": quantiles[3].item(),
        f"{prefix}_min": values.min().item(),
        f"{prefix}_max": values.max().item(),
    }


def _ssim_map(pred, target):
    pred = rearrange(pred, "... h w c -> (...) c h w")
    target = rearrange(target, "... h w c -> (...) c h w")
    mu_x = F.avg_pool2d(pred, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(pred.square(), 11, stride=1, padding=5) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), 11, stride=1, padding=5) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, 11, stride=1, padding=5) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean(dim=1)


def _ssim(pred, target):
    return _ssim_map(pred, target).mean()


def reconstruction_metrics(output_dict, target_dict):
    render = output_dict["render_results"]
    device = render[render["rgb_key"]].device
    mean = torch.tensor(MEAN, device=device)
    std = torch.tensor(STD, device=device)
    pred = (render[render["rgb_key"]].float() * std + mean).clamp(0, 1)
    target = (
        rearrange(target_dict["target_image"], "b t v c h w -> b t v h w c").float() * std + mean
    ).clamp(0, 1)
    mse = F.mse_loss(pred, target)
    metrics = {
        "rgb_mse": mse.item(),
        "psnr": (-10 * torch.log10(mse.clamp_min(1e-12))).item(),
        "ssim": _ssim(pred, target).item(),
    }
    if "target_depth" in target_dict:
        pred_depth = render[render["depth_key"]].float().squeeze()
        target_depth = target_dict["target_depth"].float().squeeze()
        if pred_depth.shape != target_depth.shape:
            leading = pred_depth.shape[:-2]
            pred_depth = F.interpolate(
                pred_depth.reshape(-1, 1, *pred_depth.shape[-2:]),
                size=target_depth.shape[-2:], mode="bilinear", align_corners=False,
            ).reshape(*leading, *target_depth.shape[-2:])
        valid = target_depth > 0.01
        metrics["depth_rmse_metric"] = (
            torch.sqrt(F.mse_loss(pred_depth[valid], target_depth[valid])).item() if valid.any() else float("nan")
        )
    dynamic = target_dict.get("target_dynamic_masks")
    if dynamic is not None:
        dynamic = rearrange(dynamic, "... h w -> (...) h w").bool()
        pred_flat = rearrange(pred, "... h w c -> (...) h w c")
        target_flat = rearrange(target, "... h w c -> (...) h w c")
        if dynamic.any():
            dynamic_mse = F.mse_loss(pred_flat[dynamic], target_flat[dynamic])
            metrics["dynamic_psnr"] = (-10 * torch.log10(dynamic_mse.clamp_min(1e-12))).item()
            metrics["dynamic_ssim"] = _ssim_map(pred, target)[dynamic].mean().item()
            metrics["dynamic_pixel_ratio"] = dynamic.float().mean().item()
            if "target_depth" in target_dict:
                pred_depth_flat = pred_depth.reshape(-1, *pred_depth.shape[-2:])
                target_depth_flat = target_depth.reshape(-1, *target_depth.shape[-2:])
                dynamic_depth = dynamic & (target_depth_flat > 0.01)
                metrics["dynamic_depth_rmse"] = (
                    torch.sqrt(F.mse_loss(
                        pred_depth_flat[dynamic_depth], target_depth_flat[dynamic_depth]
                    )).item() if dynamic_depth.any() else float("nan")
                )
    return metrics


def lpips_metric(output_dict, target_dict, metric_model):
    render = output_dict["render_results"]
    device = render[render["rgb_key"]].device
    mean = torch.tensor(MEAN, device=device)
    std = torch.tensor(STD, device=device)
    pred = (render[render["rgb_key"]].float() * std + mean).clamp(0, 1)
    target = (
        rearrange(target_dict["target_image"], "b t v c h w -> b t v h w c").float() * std + mean
    ).clamp(0, 1)
    pred = rearrange(pred, "... h w c -> (...) c h w") * 2 - 1
    target = rearrange(target, "... h w c -> (...) c h w") * 2 - 1
    return metric_model(pred, target).mean().item()


def gaussian_metrics(output_dict, max_scale=None):
    gs = output_dict["gs_params"]
    result = {}
    depth = gs.get("depth")
    origins = output_dict.get("gs_origins")
    if depth is None and origins is not None and "means" in gs and origins.shape == gs["means"].shape:
        depth = (gs["means"] - origins).norm(dim=-1)
    if depth is not None:
        result.update(tensor_summary(depth, "depth"))
    if "opacities" in gs:
        result.update(tensor_summary(gs["opacities"], "opacity"))
    if "scales" in gs:
        result.update(tensor_summary(gs["scales"], "scale"))
        if max_scale is not None:
            scales = gs["scales"].detach().float()
            result["scale_saturation_ratio"] = torch.isclose(
                scales, torch.as_tensor(max_scale, device=scales.device), rtol=1e-5, atol=1e-6
            ).float().mean().item()
    if "lifespan" in gs:
        result.update(tensor_summary(gs["lifespan"], "beta"))
    if "quats" in gs:
        result["quaternion_norm_mean"] = gs["quats"].detach().float().norm(dim=-1).mean().item()
    if "means" in gs:
        result["gaussian_count"] = int(gs["means"].numel() // 3)
    result["flow_reg_available"] = float("forward_flow" in gs)
    weights = output_dict.get("bbox_weights")
    if weights is not None:
        result["background_assignment_ratio"] = weights[..., 0].detach().float().mean().item()
        result["dynamic_assignment_ratio"] = (1.0 - weights[..., 0].detach().float()).mean().item()
    affine = gs.get("affine")
    if affine is not None:
        affine = affine.detach().float()
        result["affine_abs_mean"] = affine.abs().mean().item()
        result["affine_abs_max"] = affine.abs().max().item()
        matrix, bias = affine[..., :3], affine[..., 3]
        identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        result["affine_A_identity_l2_mean"] = (matrix - identity).norm(dim=(-2, -1)).mean().item()
        result["affine_b_abs_mean"] = bias.abs().mean().item()
        result["affine_b_abs_max"] = bias.abs().max().item()
        for camera_idx in range(affine.shape[-3]):
            camera_matrix = matrix[..., camera_idx, :, :]
            camera_bias = bias[..., camera_idx, :]
            result[f"affine_camera{camera_idx}_A_identity_l2"] = (
                camera_matrix - identity
            ).norm(dim=(-2, -1)).mean().item()
            result[f"affine_camera{camera_idx}_b_abs_mean"] = camera_bias.abs().mean().item()
    for key in (
        "object_assignment_accuracy", "object_dynamic_assignment_accuracy",
        "object_dynamic_background_error_ratio", "object_dynamic_gt_ratio",
        "object_dynamic_gt_count", "object_supervised_token_ratio",
        "object_foreground_recall", "object_foreground_precision",
        "object_background_precision", "object_predicted_dynamic_ratio",
        "object_background_probability", "object_assignment_entropy", "bbox_valid_count",
        "bbox_pose_mean_translation", "bbox_pose_max_translation",
        "bbox_pose_mean_rotation_deg", "bbox_pose_max_rotation_deg",
        "bbox_motion_mean_displacement", "bbox_motion_max_displacement",
    ):
        if key in output_dict:
            result[key] = output_dict[key].detach().float().item()
    return result


def assignment_metrics(output_dict):
    result = {}
    for key in (
        "object_assignment_accuracy", "object_dynamic_assignment_accuracy",
        "object_dynamic_background_error_ratio", "object_dynamic_gt_ratio",
        "object_dynamic_gt_count", "object_supervised_token_ratio",
        "object_foreground_recall", "object_foreground_precision",
        "object_background_precision", "object_predicted_dynamic_ratio",
        "object_background_probability", "object_assignment_entropy", "bbox_valid_count",
        "bbox_pose_mean_translation", "bbox_pose_max_translation",
        "bbox_pose_mean_rotation_deg", "bbox_pose_max_rotation_deg",
        "bbox_motion_mean_displacement", "bbox_motion_max_displacement",
    ):
        if key in output_dict:
            result[key] = output_dict[key].detach().float().item()
    return result


def parameter_grad_norms(model):
    groups = {
        "transformer_grad_norm": ("blocks.", "patch_embed", "camera_embed", "time_embed"),
        "gs_head_grad_norm": ("gs_pred", "mem_gs_pred"),
        "bbox_head_grad_norm": ("bbox_embed", "bbox_query_head", "bbox_key_head"),
        "lifespan_head_grad_norm": ("gs_life_pred",),
        "affine_grad_norm": ("affine_linear", "affine_token"),
    }
    result = {}
    named = list(model.named_parameters())
    for key, patterns in groups.items():
        params = [p for name, p in named if p.grad is not None and any(pattern in name for pattern in patterns)]
        if params:
            result[key] = math.sqrt(sum(p.grad.detach().float().square().sum().item() for p in params))
        else:
            result[key] = 0.0
    return result
