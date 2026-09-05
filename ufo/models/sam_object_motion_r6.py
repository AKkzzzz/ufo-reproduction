import torch
import torch.nn.functional as F

from ufo.models.sam_object_motion_r5 import (
    predict_sam_canonical_motion,
)


def _weighted_reduce(
    values,
    inverse,
    weights,
    num_clusters,
):
    """
    Mixed-precision-safe differentiable reduction.

    BF16/FP16 inputs are accumulated in FP32, then cast
    back to the original value dtype.
    """
    output_dtype = values.dtype

    values_fp32 = values.float()
    weights_fp32 = weights.float()

    if values.ndim == 1:
        out = torch.zeros(
            num_clusters,
            device=values.device,
            dtype=torch.float32,
        )

        out.index_add_(
            0,
            inverse,
            values_fp32 * weights_fp32,
        )

        denom = torch.zeros(
            num_clusters,
            device=values.device,
            dtype=torch.float32,
        )

        denom.index_add_(
            0,
            inverse,
            weights_fp32,
        )

        result = (
            out
            / denom.clamp_min(1e-6)
        )

    else:
        out = torch.zeros(
            num_clusters,
            values.shape[-1],
            device=values.device,
            dtype=torch.float32,
        )

        out.index_add_(
            0,
            inverse,
            values_fp32
            * weights_fp32[:, None],
        )

        denom = torch.zeros(
            num_clusters,
            device=values.device,
            dtype=torch.float32,
        )

        denom.index_add_(
            0,
            inverse,
            weights_fp32,
        )

        result = (
            out
            / denom[:, None]
            .clamp_min(1e-6)
        )

    return result.to(output_dtype)

def fuse_canonical_gaussians(
    gs_params,
    global_ids,
    velocity,
    gs_time,
    timespan,
    voxel_size=0.12,
    min_voxel_support=1,
):
    """
    R6 canonical fusion.

    For each global object:

        observations at t_i
              ↓
        x_i - v * (t_i - t_ref)
              ↓
        common canonical coordinate
              ↓
        metric voxel grouping
              ↓
        fuse xyz / color / scale /
        quaternion / opacity
              ↓
        one canonical Gaussian set

    Static Gaussians are untouched.

    Group assignment itself is non-differentiable
    (computed from detached xyz), while the actual
    fused attributes remain differentiable.
    """

    means = gs_params["means"]
    scales = gs_params["scales"]
    quats = gs_params["quats"]
    opacities = gs_params["opacities"]
    colors = gs_params["colors"]

    B, T, V, H, W, _ = means.shape
    N = T * V * H * W

    if global_ids.shape != (B, T, V, H, W):
        raise RuntimeError(
            "R6 global-id shape mismatch: "
            f"{global_ids.shape} vs "
            f"{means.shape}"
        )

    means_flat = means.reshape(B, N, 3)
    scales_flat = scales.reshape(B, N, 3)
    quats_flat = quats.reshape(B, N, 4)
    opacity_flat = opacities.reshape(B, N)
    colors_flat = colors.reshape(
        B,
        N,
        colors.shape[-1],
    )

    velocity_flat = velocity.reshape(
        B,
        N,
        3,
    )

    global_flat = global_ids.reshape(
        B,
        N,
    )

    physical_time = (
        gs_time.float()
        * float(timespan)
    )

    if physical_time.shape != (B, T, V):
        raise RuntimeError(
            "R6 gs_time shape mismatch: "
            f"{physical_time.shape}"
        )

    source_time = (
        physical_time[
            ...,
            None,
            None,
        ]
        .expand(B, T, V, H, W)
        .reshape(B, N)
        .clone()
    )

    # Outputs retain original dense tensor shape.
    # Dynamic non-representatives get opacity=0,
    # but gradients from their attributes still flow
    # through the fused representative.
    fused_means = means_flat.clone()
    fused_scales = scales_flat.clone()
    fused_quats = quats_flat.clone()
    fused_opacity = opacity_flat.clone()
    fused_colors = colors_flat.clone()

    active_mask = (
        global_flat == 0
    )

    total_dynamic_points = 0
    total_fused_voxels = 0
    support_values = []
    object_count = 0

    for batch_idx in range(B):

        object_ids = torch.unique(
            global_flat[batch_idx]
        )

        object_ids = object_ids[
            object_ids > 0
        ]

        for gid_tensor in object_ids:

            gid = int(
                gid_tensor.item()
            )

            object_mask = (
                global_flat[
                    batch_idx
                ]
                == gid
            )

            object_indices = (
                object_mask
                .nonzero(
                    as_tuple=False
                )
                .squeeze(-1)
            )

            if object_indices.numel() == 0:
                continue

            object_count += 1

            total_dynamic_points += int(
                object_indices.numel()
            )

            xyz = means_flat[
                batch_idx,
                object_indices,
            ]

            vv = velocity_flat[
                batch_idx,
                object_indices,
            ]

            tt = source_time[
                batch_idx,
                object_indices,
            ]

            # One shared velocity already exists
            # per global object. Mean keeps gradients
            # well behaved even if numeric copies differ.
            object_velocity = vv.mean(
                dim=0
            )

            # Mid-time canonical reference rather than
            # earliest frame: minimizes average warp length.
            reference_time = (
                tt.mean().detach()
            )

            canonical_xyz = (
                xyz
                - object_velocity[
                    None
                ]
                * (
                    tt[:, None]
                    - reference_time
                )
            )

            # ------------------------------------------------
            # Hard voxel assignment only controls grouping.
            # No gradient needed through the assignment.
            # ------------------------------------------------
            voxel_coord = torch.floor(
                canonical_xyz.detach()
                / float(voxel_size)
            ).to(torch.int64)

            _, inverse = torch.unique(
                voxel_coord,
                dim=0,
                return_inverse=True,
            )

            num_clusters = int(
                inverse.max().item()
            ) + 1

            local_arange = torch.arange(
                object_indices.numel(),
                device=means.device,
                dtype=torch.long,
            )

            representative_local = torch.full(
                (num_clusters,),
                object_indices.numel(),
                device=means.device,
                dtype=torch.long,
            )

            representative_local.scatter_reduce_(
                0,
                inverse,
                local_arange,
                reduce="amin",
                include_self=True,
            )

            cluster_support = torch.zeros(
                num_clusters,
                device=means.device,
                dtype=torch.long,
            )

            cluster_support.index_add_(
                0,
                inverse,
                torch.ones_like(
                    inverse,
                    dtype=torch.long,
                ),
            )

            valid_cluster = (
                cluster_support
                >= int(
                    min_voxel_support
                )
            )

            if not valid_cluster.any():
                continue

            # Opacity-based fusion confidence.
            weights = (
                opacity_flat[
                    batch_idx,
                    object_indices,
                ]
                .detach()
                .float()
                .clamp(
                    min=0.05,
                    max=1.0,
                )
                .to(means.dtype)
            )

            cluster_means = (
                _weighted_reduce(
                    canonical_xyz,
                    inverse,
                    weights,
                    num_clusters,
                )
            )

            # Use geometric mean for scale.
            cluster_log_scales = (
                _weighted_reduce(
                    torch.log(
                        scales_flat[
                            batch_idx,
                            object_indices,
                        ]
                        .clamp_min(1e-6)
                    ),
                    inverse,
                    weights,
                    num_clusters,
                )
            )

            cluster_scales = torch.exp(
                cluster_log_scales
            )

            cluster_colors = (
                _weighted_reduce(
                    colors_flat[
                        batch_idx,
                        object_indices,
                    ],
                    inverse,
                    weights,
                    num_clusters,
                )
            )

            cluster_opacity = (
                _weighted_reduce(
                    opacity_flat[
                        batch_idx,
                        object_indices,
                    ],
                    inverse,
                    weights,
                    num_clusters,
                )
            )

            # Quaternion hemisphere alignment
            # before weighted averaging.
            object_quats = quats_flat[
                batch_idx,
                object_indices,
            ]

            reference_quats = (
                object_quats[
                    representative_local
                ][inverse]
                .detach()
            )

            quat_sign = torch.where(
                (
                    object_quats
                    * reference_quats
                )
                .sum(
                    dim=-1,
                    keepdim=True,
                )
                < 0,
                -torch.ones(
                    1,
                    device=means.device,
                    dtype=means.dtype,
                ),
                torch.ones(
                    1,
                    device=means.device,
                    dtype=means.dtype,
                ),
            )

            aligned_quats = (
                object_quats
                * quat_sign
            )

            cluster_quats = (
                _weighted_reduce(
                    aligned_quats,
                    inverse,
                    weights,
                    num_clusters,
                )
            )

            cluster_quats = F.normalize(
                cluster_quats,
                dim=-1,
                eps=1e-6,
            )

            representative_global = (
                object_indices[
                    representative_local
                ]
            )

            representative_global = (
                representative_global[
                    valid_cluster
                ]
            )

            # Every old dynamic Gaussian is deactivated.
            fused_opacity[
                batch_idx,
                object_indices,
            ] = 0.0

            # Only fused canonical representatives survive.
            fused_means[
                batch_idx,
                representative_global,
            ] = cluster_means[
                valid_cluster
            ].to(
                dtype=fused_means.dtype
            )

            fused_scales[
                batch_idx,
                representative_global,
            ] = cluster_scales[
                valid_cluster
            ].to(
                dtype=fused_scales.dtype
            )

            fused_colors[
                batch_idx,
                representative_global,
            ] = cluster_colors[
                valid_cluster
            ].to(
                dtype=fused_colors.dtype
            )

            fused_opacity[
                batch_idx,
                representative_global,
            ] = cluster_opacity[
                valid_cluster
            ].to(
                dtype=fused_opacity.dtype
            )

            fused_quats[
                batch_idx,
                representative_global,
            ] = cluster_quats[
                valid_cluster
            ].to(
                dtype=fused_quats.dtype
            )

            active_mask[
                batch_idx,
                representative_global,
            ] = True

            # All canonical representatives share
            # the same reference time.
            source_time[
                batch_idx,
                object_indices,
            ] = reference_time

            valid_support = (
                cluster_support[
                    valid_cluster
                ]
            )

            total_fused_voxels += int(
                valid_cluster.sum()
            )

            support_values.append(
                valid_support.float().mean()
            )

    if support_values:
        mean_support = torch.stack(
            support_values
        ).mean()
    else:
        mean_support = means.new_zeros(())

    compression_ratio = (
        float(total_fused_voxels)
        / max(
            total_dynamic_points,
            1,
        )
    )

    outputs = {
        "means": fused_means.reshape_as(
            means
        ),
        "scales": fused_scales.reshape_as(
            scales
        ),
        "quats": fused_quats.reshape_as(
            quats
        ),
        "opacities": fused_opacity.reshape_as(
            opacities
        ),
        "colors": fused_colors.reshape_as(
            colors
        ),
        "source_time_seconds":
            source_time.reshape(
                B,
                T,
                V,
                H,
                W,
            ),
        "active_mask":
            active_mask.reshape(
                B,
                T,
                V,
                H,
                W,
            ),
    }

    diagnostics = {
        "r6_object_count":
            object_count,

        "r6_dynamic_input_points":
            total_dynamic_points,

        "r6_fused_voxel_count":
            total_fused_voxels,

        "r6_fusion_ratio":
            compression_ratio,

        "r6_voxel_support_mean":
            mean_support.detach(),
    }

    return outputs, diagnostics


def predict_sam_fused_motion(
    gs_state,
    gs_params,
    local_track_ids,
    gs_time,
    timespan,
    motion_head,
    patch_size=8,
    min_observations=2,
    min_track_pixels=16,
    association_max_distance=4.0,
    association_min_overlap=1,
    voxel_size=0.12,
    min_voxel_support=1,
):
    """
    Full R6:
      SAM local track
          ↓
      global object association
          ↓
      one object velocity
          ↓
      all observations warp to canonical time
          ↓
      voxel fusion
          ↓
      one canonical Gaussian object
    """

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
        association_max_distance=(
            association_max_distance
        ),
        association_min_overlap=(
            association_min_overlap
        ),
    )

    fused_params, fusion_diag = (
        fuse_canonical_gaussians(
            gs_params=gs_params,
            global_ids=global_ids,
            velocity=velocity,
            gs_time=gs_time,
            timespan=timespan,
            voxel_size=voxel_size,
            min_voxel_support=(
                min_voxel_support
            ),
        )
    )

    diagnostics = {
        **motion_diag,
        **fusion_diag,
    }

    return (
        velocity,
        global_ids,
        fused_params,
        diagnostics,
    )
