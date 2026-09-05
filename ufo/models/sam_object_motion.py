import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMObjectMotionHead(nn.Module):
    """
    One SAM track -> one shared world-space translation velocity.

    Input:
        pooled latent appearance/state
        temporal latent change
        track duration
        track spatial support

    Supervision:
        only downstream RGB rendering loss.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim=384,
        max_speed=20.0,
    ):
        super().__init__()

        self.max_speed = float(max_speed)

        input_dim = 2 * embed_dim + 2

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        # Start from zero motion.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, descriptor):
        raw = self.net(descriptor)

        # derivative around zero is ~1, while bounding extreme speed
        return self.max_speed * torch.tanh(
            raw / self.max_speed
        )


def predict_sam_object_velocity(
    gs_state,
    sam_track_ids,
    gs_time,
    timespan,
    motion_head,
    patch_size=8,
    min_observations=2,
    min_patch_support=0.25,
):
    """
    gs_state:
        [B, T*V*PH*PW, C]

    sam_track_ids:
        [B,T,V,H,W]

    Returns:
        dense_velocity [B,T,V,H,W,3]

    Important:
        velocity is predicted once per SAM track and then copied
        to every Gaussian belonging to that track.
    """

    if sam_track_ids is None:
        raise RuntimeError(
            "sam_object_motion requires sam_track_ids"
        )

    B, T, V, H, W = sam_track_ids.shape
    _, N, C = gs_state.shape

    PH = H // patch_size
    PW = W // patch_size

    expected = T * V * PH * PW

    if N != expected:
        raise RuntimeError(
            "SAM/scene-token mismatch: "
            f"state={gs_state.shape}, "
            f"sam={sam_track_ids.shape}, "
            f"expected_tokens={expected}"
        )

    if gs_time.shape != (B, T, V):
        raise RuntimeError(
            f"gs_time mismatch: {gs_time.shape} "
            f"expected {(B,T,V)}"
        )

    features = gs_state.reshape(
        B, T, V, PH, PW, C
    )

    physical_time = (
        gs_time.float() * float(timespan)
    )

    velocity = gs_state.new_zeros(
        (B, T, V, H, W, 3)
    )

    track_count = 0
    moving_candidates = 0
    skipped_count = 0
    speeds = []

    for b in range(B):

        track_ids = torch.unique(
            sam_track_ids[b]
        )
        track_ids = track_ids[
            track_ids > 0
        ]

        for track_id in track_ids:

            track_count += 1

            # ------------------------------------------
            # Pixel SAM mask -> patch coverage.
            #
            # Do NOT nearest-sample the track ID:
            # small vehicles would disappear.
            # ------------------------------------------
            pixel_mask = (
                sam_track_ids[b] == track_id
            ).float()

            patch_coverage = F.avg_pool2d(
                pixel_mask.reshape(
                    T * V, 1, H, W
                ),
                kernel_size=patch_size,
                stride=patch_size,
            ).reshape(
                T, V, PH, PW
            )

            obs_features = []
            obs_times = []
            obs_support = []

            for ti in range(T):

                coverage = patch_coverage[ti]

                support = coverage.sum()

                if float(support.detach()) < min_patch_support:
                    continue

                feat = features[b, ti]

                weighted_feature = (
                    feat * coverage[..., None]
                ).sum(
                    dim=(0, 1, 2)
                ) / support.clamp_min(1e-6)

                # Cameras are synchronized, but weighting by visible
                # support also works if one view is absent.
                view_support = coverage.sum(
                    dim=(1, 2)
                )

                obs_time = (
                    physical_time[b, ti]
                    * view_support
                ).sum() / view_support.sum().clamp_min(
                    1e-6
                )

                obs_features.append(
                    weighted_feature
                )
                obs_times.append(obs_time)
                obs_support.append(support)

            if len(obs_features) < min_observations:
                skipped_count += 1
                continue

            obs_features = torch.stack(
                obs_features
            )

            obs_times = torch.stack(
                obs_times
            ).float()

            obs_support = torch.stack(
                obs_support
            ).float()

            # ------------------------------------------
            # Track descriptor.
            # ------------------------------------------
            feat_mean = obs_features.mean(
                dim=0
            )

            t_mean = obs_times.mean()
            dt = obs_times - t_mean

            denom = (dt * dt).sum()

            if float(denom.detach()) > 1e-8:
                feat_slope = (
                    dt[:, None]
                    * (
                        obs_features
                        - feat_mean
                    )
                ).sum(dim=0) / denom
            else:
                feat_slope = torch.zeros_like(
                    feat_mean
                )

            duration = (
                obs_times.max()
                - obs_times.min()
            )

            support_feature = torch.log1p(
                obs_support.mean()
            ) / 5.0

            descriptor = torch.cat(
                [
                    feat_mean,
                    feat_slope,
                    duration.to(
                        feat_mean.dtype
                    ).reshape(1),
                    support_feature.to(
                        feat_mean.dtype
                    ).reshape(1),
                ]
            )

            object_velocity = motion_head(
                descriptor.unsqueeze(0)
            )[0]

            # ONE motion for the whole object.
            mask = (
                sam_track_ids[b]
                == track_id
            )

            velocity[b][mask] = (
                object_velocity.to(
                    velocity.dtype
                )
            )

            moving_candidates += 1
            speeds.append(
                object_velocity.float().norm()
            )

    if speeds:
        speeds = torch.stack(speeds)
        speed_mean = speeds.mean().detach()
        speed_max = speeds.max().detach()
    else:
        speed_mean = torch.zeros(
            (),
            device=gs_state.device,
        )
        speed_max = speed_mean

    diagnostics = {
        "sam_object_track_count":
            track_count,
        "sam_object_predicted_count":
            moving_candidates,
        "sam_object_skipped_count":
            skipped_count,
        "sam_object_speed_mean":
            speed_mean,
        "sam_object_speed_max":
            speed_max,
    }

    return velocity, diagnostics
