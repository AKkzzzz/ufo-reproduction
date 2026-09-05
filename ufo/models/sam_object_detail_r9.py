import torch
import torch.nn as nn
import torch.nn.functional as F

from ufo.models.sam_object_motion_r5 import predict_sam_canonical_motion


def _segment_softmax(scores, inverse, num_clusters):
    scores32 = scores.float()
    max_per_cluster = torch.full(
        (num_clusters,), -float("inf"),
        device=scores.device, dtype=torch.float32,
    )
    max_per_cluster.scatter_reduce_(
        0, inverse, scores32, reduce="amax", include_self=True
    )
    exp_scores = torch.exp(scores32 - max_per_cluster[inverse])
    denom = torch.zeros(
        num_clusters, device=scores.device, dtype=torch.float32
    )
    denom.index_add_(0, inverse, exp_scores)
    return (exp_scores / denom[inverse].clamp_min(1e-8)).to(scores.dtype)


def _segment_weighted_sum(values, inverse, weights, num_clusters):
    out = torch.zeros(
        num_clusters, values.shape[-1],
        device=values.device, dtype=torch.float32,
    )
    out.index_add_(
        0, inverse, values.float() * weights.float()[:, None]
    )
    return out.to(values.dtype)


def _reference_representatives(inverse, temporal_distance, num_clusters):
    count = inverse.numel()
    order = torch.argsort(temporal_distance.detach(), stable=True)
    rank = torch.arange(count, device=inverse.device, dtype=torch.long)
    min_rank = torch.full(
        (num_clusters,), count,
        device=inverse.device, dtype=torch.long,
    )
    min_rank.scatter_reduce_(
        0, inverse[order], rank, reduce="amin", include_self=True
    )
    return order[min_rank]


class SAMObjectDetailFusionHead(nn.Module):
    """Feature-level canonical dynamic Gaussian fusion for R9."""

    def __init__(
        self,
        embed_dim,
        color_dim,
        hidden_dim=256,
        max_mean_residual=0.08,
        max_log_scale_residual=0.35,
        max_quat_residual=0.20,
        max_color_residual=0.25,
    ):
        super().__init__()
        self.max_mean_residual = float(max_mean_residual)
        self.max_log_scale_residual = float(max_log_scale_residual)
        self.max_quat_residual = float(max_quat_residual)
        self.max_color_residual = float(max_color_residual)

        self.feature_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.appearance_proj = nn.Sequential(
            nn.Linear(3 + color_dim + 1, 96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.geometry_proj = nn.Sequential(
            nn.Linear(4, 96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.evidence_proj = nn.Sequential(
            nn.Linear(hidden_dim + 96 + 96, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mean_head = nn.Linear(hidden_dim, 3)
        self.log_scale_head = nn.Linear(hidden_dim, 3)
        self.quat_head = nn.Linear(hidden_dim, 4)
        self.opacity_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Linear(hidden_dim, color_dim)

        for head in (
            self.mean_head,
            self.log_scale_head,
            self.quat_head,
            self.opacity_head,
            self.color_head,
        ):
            # Near-identity initialization, but not exactly zero:
            # upstream feature/confidence modules receive gradients from step 0.
            nn.init.normal_(head.weight, std=1e-4)
            nn.init.zeros_(head.bias)

        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

    def encode_evidence(
        self,
        patch_features,
        raw_rgb,
        gaussian_colors,
        gaussian_opacity,
        relative_xyz,
        relative_time,
        voxel_size,
    ):
        # Canonical geometry is intentionally computed in FP32 for
        # metric precision, while network activations may be BF16/FP16.
        # Make every MLP boundary explicit instead of relying on autocast.

        feature_input = patch_features.to(
            dtype=self.feature_proj[1].weight.dtype
        )
        feature = self.feature_proj(feature_input)

        appearance_input = torch.cat(
            [
                raw_rgb,
                gaussian_colors,
                gaussian_opacity[:, None],
            ],
            dim=-1,
        ).to(
            dtype=self.appearance_proj[0].weight.dtype
        )
        appearance = self.appearance_proj(appearance_input)

        geometry_input = torch.cat(
            [
                relative_xyz / max(float(voxel_size), 1e-6),
                relative_time[:, None],
            ],
            dim=-1,
        ).to(
            dtype=self.geometry_proj[0].weight.dtype
        )
        geometry = self.geometry_proj(geometry_input)

        evidence_input = torch.cat(
            [feature, appearance, geometry],
            dim=-1,
        ).to(
            dtype=self.evidence_proj[0].weight.dtype
        )

        return self.evidence_proj(evidence_input)

    def confidence(self, evidence):
        return self.confidence_head(evidence).squeeze(-1)

    def refine(
        self,
        fused_feature,
        anchor_mean,
        anchor_scale,
        anchor_quat,
        anchor_opacity,
        anchor_color,
    ):
        fused_feature = self.fusion_mlp(fused_feature)

        delta_mean = (
            self.max_mean_residual * torch.tanh(self.mean_head(fused_feature))
        )
        mean = anchor_mean + delta_mean

        log_scale_delta = (
            self.max_log_scale_residual
            * torch.tanh(self.log_scale_head(fused_feature))
        )
        scale = torch.exp(
            torch.log(anchor_scale.float().clamp_min(1e-6))
            + log_scale_delta.float()
        ).to(anchor_scale.dtype)

        quat_delta = (
            self.max_quat_residual * torch.tanh(self.quat_head(fused_feature))
        )
        quat = F.normalize(
            anchor_quat.float() + quat_delta.float(),
            dim=-1, eps=1e-6,
        ).to(anchor_quat.dtype)

        alpha = anchor_opacity.float().clamp(1e-4, 1.0 - 1e-4)
        opacity_logit = torch.log(alpha) - torch.log1p(-alpha)
        opacity = torch.sigmoid(
            opacity_logit
            + self.opacity_head(fused_feature).squeeze(-1).float()
        ).to(anchor_opacity.dtype)

        color_delta = (
            self.max_color_residual
            * torch.tanh(self.color_head(fused_feature))
        )
        color = anchor_color + color_delta

        return (
            mean, scale, quat, opacity, color,
            {
                "mean_residual_abs": delta_mean.detach().float().abs().mean(),
                "color_residual_abs": color_delta.detach().float().abs().mean(),
            },
        )


def fuse_canonical_detail_features(
    gs_state,
    gs_params,
    context_image,
    global_ids,
    velocity,
    gs_time,
    timespan,
    detail_head,
    patch_size=8,
    voxel_size=0.12,
    min_voxel_support=1,
    spatial_prior=0.50,
    temporal_prior=0.25,
):
    means = gs_params["means"]
    scales = gs_params["scales"]
    quats = gs_params["quats"]
    opacities = gs_params["opacities"]
    colors = gs_params["colors"]

    B, T, V, H, W, _ = means.shape
    C = gs_state.shape[-1]
    PH, PW = H // patch_size, W // patch_size

    expected_tokens = T * V * PH * PW
    if gs_state.shape != (B, expected_tokens, C):
        raise RuntimeError(
            f"R9 token mismatch: {tuple(gs_state.shape)} "
            f"vs {(B, expected_tokens, C)}"
        )
    if global_ids.shape != (B, T, V, H, W):
        raise RuntimeError("R9 global-id shape mismatch")
    if context_image.shape != (B, T, V, 3, H, W):
        raise RuntimeError(
            f"R9 context-image shape mismatch: "
            f"{tuple(context_image.shape)} "
            f"vs {(B, T, V, 3, H, W)}"
        )

    token_features = gs_state.reshape(B, T, V, PH, PW, C)
    rgb = context_image.permute(0, 1, 2, 4, 5, 3).contiguous()

    N = T * V * H * W
    means_flat = means.reshape(B, N, 3)
    scales_flat = scales.reshape(B, N, 3)
    quats_flat = quats.reshape(B, N, 4)
    opacity_flat = opacities.reshape(B, N)
    colors_flat = colors.reshape(B, N, colors.shape[-1])
    velocity_flat = velocity.reshape(B, N, 3)
    global_flat = global_ids.reshape(B, N)

    physical_time = gs_time.float() * float(timespan)
    source_time = (
        physical_time[..., None, None]
        .expand(B, T, V, H, W)
        .reshape(B, N)
        .clone()
    )

    refined_means = means_flat.clone()
    refined_scales = scales_flat.clone()
    refined_quats = quats_flat.clone()
    refined_opacity = opacity_flat.clone()
    refined_colors = colors_flat.clone()
    active_mask = global_flat == 0

    object_count = 0
    total_dynamic_points = 0
    total_voxels = 0
    support_stats = []
    entropy_stats = []
    temporal_span_stats = []
    mean_residual_stats = []
    color_residual_stats = []

    hw = H * W
    vhw = V * hw

    for batch_idx in range(B):
        object_ids = torch.unique(global_flat[batch_idx])
        object_ids = object_ids[object_ids > 0]

        for gid_tensor in object_ids:
            gid = int(gid_tensor.item())
            object_indices = (
                (global_flat[batch_idx] == gid)
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )
            if object_indices.numel() == 0:
                continue

            object_count += 1
            total_dynamic_points += int(object_indices.numel())

            t_idx = torch.div(
                object_indices, vhw, rounding_mode="floor"
            )
            rem = object_indices.remainder(vhw)
            v_idx = torch.div(rem, hw, rounding_mode="floor")
            pix = rem.remainder(hw)
            y_idx = torch.div(pix, W, rounding_mode="floor")
            x_idx = pix.remainder(W)
            py = torch.div(y_idx, patch_size, rounding_mode="floor")
            px = torch.div(x_idx, patch_size, rounding_mode="floor")

            patch_feat = token_features[
                batch_idx, t_idx, v_idx, py, px
            ]
            raw_rgb = rgb[
                batch_idx, t_idx, v_idx, y_idx, x_idx
            ].to(patch_feat.dtype)

            xyz = means_flat[batch_idx, object_indices]
            vv = velocity_flat[batch_idx, object_indices]
            tt = source_time[batch_idx, object_indices]

            object_velocity = vv.mean(dim=0)
            reference_time = tt.mean().detach()

            canonical_xyz = (
                xyz
                - object_velocity[None]
                * (tt[:, None] - reference_time)
            )

            voxel_coord = torch.floor(
                canonical_xyz.detach() / float(voxel_size)
            ).to(torch.int64)
            _, inverse = torch.unique(
                voxel_coord, dim=0, return_inverse=True
            )
            num_clusters = int(inverse.max().item()) + 1

            cluster_support = torch.zeros(
                num_clusters, device=means.device, dtype=torch.long
            )
            cluster_support.index_add_(
                0, inverse,
                torch.ones_like(inverse, dtype=torch.long)
            )
            valid_cluster = (
                cluster_support >= int(min_voxel_support)
            )
            if not valid_cluster.any():
                continue

            temporal_distance = (tt - reference_time).abs()
            representative_local = _reference_representatives(
                inverse, temporal_distance, num_clusters
            )
            anchor_xyz_all = canonical_xyz[representative_local]
            relative_xyz = (
                canonical_xyz - anchor_xyz_all[inverse]
            )
            relative_time = (
                (tt - reference_time)
                / max(float(timespan), 1e-6)
            ).to(patch_feat.dtype)

            evidence = detail_head.encode_evidence(
                patch_features=patch_feat,
                raw_rgb=raw_rgb,
                gaussian_colors=colors_flat[
                    batch_idx, object_indices
                ],
                gaussian_opacity=opacity_flat[
                    batch_idx, object_indices
                ],
                relative_xyz=relative_xyz,
                relative_time=relative_time,
                voxel_size=voxel_size,
            )

            learned_score = detail_head.confidence(evidence)
            score = (
                learned_score.float()
                - float(spatial_prior)
                * (
                    relative_xyz.float().norm(dim=-1)
                    / max(float(voxel_size), 1e-6)
                ).square()
                - float(temporal_prior)
                * relative_time.float().abs()
            ).to(learned_score.dtype)

            weights = _segment_softmax(
                score, inverse, num_clusters
            )
            fused_feature = _segment_weighted_sum(
                evidence, inverse, weights, num_clusters
            )

            representative_global = (
                object_indices[representative_local]
            )

            (
                cluster_mean,
                cluster_scale,
                cluster_quat,
                cluster_opacity,
                cluster_color,
                refine_diag,
            ) = detail_head.refine(
                fused_feature=fused_feature,
                anchor_mean=anchor_xyz_all,
                anchor_scale=scales_flat[
                    batch_idx, representative_global
                ],
                anchor_quat=quats_flat[
                    batch_idx, representative_global
                ],
                anchor_opacity=opacity_flat[
                    batch_idx, representative_global
                ],
                anchor_color=colors_flat[
                    batch_idx, representative_global
                ],
            )

            refined_opacity[batch_idx, object_indices] = 0.0
            valid_rep = representative_global[valid_cluster]

            # Geometry/canonical fusion may run in FP32 while the
            # dense UFO Gaussian tensors are BF16/FP16. Cast only at
            # the write-back boundary so metric computations retain
            # FP32 precision.
            refined_means[batch_idx, valid_rep] = (
                cluster_mean[valid_cluster]
                .to(dtype=refined_means.dtype)
            )
            refined_scales[batch_idx, valid_rep] = (
                cluster_scale[valid_cluster]
                .to(dtype=refined_scales.dtype)
            )
            refined_quats[batch_idx, valid_rep] = (
                cluster_quat[valid_cluster]
                .to(dtype=refined_quats.dtype)
            )
            refined_opacity[batch_idx, valid_rep] = (
                cluster_opacity[valid_cluster]
                .to(dtype=refined_opacity.dtype)
            )
            refined_colors[batch_idx, valid_rep] = (
                cluster_color[valid_cluster]
                .to(dtype=refined_colors.dtype)
            )
            active_mask[batch_idx, valid_rep] = True
            source_time[batch_idx, object_indices] = reference_time

            w32 = weights.float()
            entropy = -(w32 * torch.log(w32.clamp_min(1e-8)))
            entropy_per_cluster = torch.zeros(
                num_clusters, device=means.device, dtype=torch.float32
            )
            entropy_per_cluster.index_add_(
                0, inverse, entropy
            )

            total_voxels += int(valid_cluster.sum())
            support_stats.append(
                cluster_support[valid_cluster].float().mean()
            )
            entropy_stats.append(
                entropy_per_cluster[valid_cluster].mean()
            )
            temporal_span_stats.append(
                (tt.max() - tt.min()).detach().float()
            )
            mean_residual_stats.append(
                refine_diag["mean_residual_abs"]
            )
            color_residual_stats.append(
                refine_diag["color_residual_abs"]
            )

    def mean_or_zero(values):
        if not values:
            return means.new_zeros(())
        return torch.stack(
            [x.to(device=means.device) for x in values]
        ).mean()

    outputs = {
        "means": refined_means.reshape_as(means),
        "scales": refined_scales.reshape_as(scales),
        "quats": refined_quats.reshape_as(quats),
        "opacities": refined_opacity.reshape_as(opacities),
        "colors": refined_colors.reshape_as(colors),
        "source_time_seconds": source_time.reshape(B, T, V, H, W),
        "active_mask": active_mask.reshape(B, T, V, H, W),
    }
    diagnostics = {
        "r9_object_count": object_count,
        "r9_dynamic_input_points": total_dynamic_points,
        "r9_fused_voxel_count": total_voxels,
        "r9_fusion_ratio": (
            float(total_voxels) / max(total_dynamic_points, 1)
        ),
        "r9_voxel_support_mean": mean_or_zero(support_stats).detach(),
        "r9_fusion_entropy_mean": mean_or_zero(entropy_stats).detach(),
        "r9_temporal_span_mean": mean_or_zero(
            temporal_span_stats
        ).detach(),
        "r9_mean_residual_abs": mean_or_zero(
            mean_residual_stats
        ).detach(),
        "r9_color_residual_abs": mean_or_zero(
            color_residual_stats
        ).detach(),
    }
    return outputs, diagnostics


def predict_sam_detail_motion(
    gs_state,
    gs_params,
    context_image,
    local_track_ids,
    gs_time,
    timespan,
    motion_head,
    detail_head,
    patch_size=8,
    min_observations=2,
    min_track_pixels=16,
    association_max_distance=4.0,
    association_min_overlap=1,
    voxel_size=0.12,
    min_voxel_support=1,
    spatial_prior=0.50,
    temporal_prior=0.25,
):
    (
        velocity,
        global_ids,
        _,
        motion_diag,
    ) = predict_sam_canonical_motion(
        gs_state=gs_state,
        means=gs_params["means"],
        local_track_ids=local_track_ids,
        gs_time=gs_time,
        timespan=timespan,
        motion_head=motion_head,
        patch_size=patch_size,
        min_observations=min_observations,
        min_track_pixels=min_track_pixels,
        association_max_distance=association_max_distance,
        association_min_overlap=association_min_overlap,
    )

    fused_params, detail_diag = fuse_canonical_detail_features(
        gs_state=gs_state,
        gs_params=gs_params,
        context_image=context_image,
        global_ids=global_ids,
        velocity=velocity,
        gs_time=gs_time,
        timespan=timespan,
        detail_head=detail_head,
        patch_size=patch_size,
        voxel_size=voxel_size,
        min_voxel_support=min_voxel_support,
        spatial_prior=spatial_prior,
        temporal_prior=temporal_prior,
    )

    return (
        velocity,
        global_ids,
        fused_params,
        {**motion_diag, **detail_diag},
    )
