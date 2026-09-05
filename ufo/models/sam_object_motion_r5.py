import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMCanonicalObjectMotionHead(nn.Module):
    """
    One global object -> one shared world-space translation velocity.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim=384,
        max_speed=20.0,
    ):
        super().__init__()

        self.max_speed = float(max_speed)

        self.net = nn.Sequential(
            nn.LayerNorm(2 * embed_dim + 2),
            nn.Linear(2 * embed_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        # R4 checkpoint can overwrite these weights.
        # From scratch -> zero motion.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, descriptor):
        raw = self.net(descriptor)
        return self.max_speed * torch.tanh(
            raw / self.max_speed
        )


@torch.no_grad()
def build_global_track_ids(
    means,
    local_track_ids,
    min_track_pixels=16,
    association_max_distance=4.0,
    association_min_overlap=1,
):
    """
    Camera-local SAM tracks -> global object IDs.

    means:
        [B,T,V,H,W,3]

    local_track_ids:
        [B,T,V,H,W]

    Returns:
        global_ids:
            [B,T,V,H,W]

        canonical_keep:
            [B,T,V,H,W]
            For each camera-local track, only its earliest observation
            is kept as an active canonical Gaussian source.
    """

    B, T, V, H, W, _ = means.shape

    if local_track_ids.shape != (B, T, V, H, W):
        raise RuntimeError(
            f"track shape mismatch: means={means.shape}, "
            f"tracks={local_track_ids.shape}"
        )

    global_ids = torch.zeros_like(
        local_track_ids,
        dtype=torch.long,
    )

    canonical_keep = torch.zeros_like(
        local_track_ids,
        dtype=torch.bool,
    )

    total_local_tracks = 0
    total_global_objects = 0
    total_merges = 0

    # ============================================================
    # IMPORTANT:
    # batch index is ALWAYS named batch_idx.
    # track variables are ALWAYS track_a / track_b / track_id.
    # Never use bare `b`.
    # ============================================================

    for batch_idx in range(B):

        track_ids = torch.unique(
            local_track_ids[batch_idx]
        )
        track_ids = track_ids[track_ids > 0]

        total_local_tracks += int(track_ids.numel())

        track_info = {}

        # --------------------------------------------------------
        # Build temporal 3D observations for each camera-local ID.
        # --------------------------------------------------------
        for track_tensor in track_ids:

            track_id = int(track_tensor.item())

            locations = (
                local_track_ids[batch_idx] == track_id
            ).nonzero(as_tuple=False)

            if locations.numel() == 0:
                continue

            # shape of locations:
            # [N, 4] = [time, camera, y, x]
            camera_idx = int(
                locations[0, 1].item()
            )

            # Sanity: one offset SAM ID must belong to one camera.
            cameras = torch.unique(locations[:, 1])

            if cameras.numel() != 1:
                raise RuntimeError(
                    f"track {track_id} appears in multiple cameras: "
                    f"{cameras.tolist()}"
                )

            centers = {}
            valid_times = []

            for time_idx in range(T):

                mask = (
                    local_track_ids[
                        batch_idx,
                        time_idx,
                        camera_idx,
                    ]
                    == track_id
                )

                pixel_count = int(mask.sum())

                if pixel_count == 0:
                    continue

                valid_times.append(time_idx)

                if pixel_count < min_track_pixels:
                    continue

                xyz = means[
                    batch_idx,
                    time_idx,
                    camera_idx,
                ][mask].detach().float()

                finite = torch.isfinite(
                    xyz
                ).all(dim=-1)

                xyz = xyz[finite]

                if xyz.shape[0] < min_track_pixels:
                    continue

                centers[time_idx] = (
                    xyz.median(dim=0).values
                )

            track_info[track_id] = {
                "camera": camera_idx,
                "centers": centers,
            }

            # ----------------------------------------------------
            # Canonical temporal source:
            # only earliest observation of this local SAM track.
            # ----------------------------------------------------
            if valid_times:

                canonical_time = min(
                    valid_times
                )

                canonical_keep[
                    batch_idx,
                    canonical_time,
                    camera_idx,
                ] |= (
                    local_track_ids[
                        batch_idx,
                        canonical_time,
                        camera_idx,
                    ]
                    == track_id
                )

        valid_tracks = list(
            track_info.keys()
        )

        # ========================================================
        # Disjoint-set / union-find for cross-view association.
        # ========================================================

        parent = {
            track_id: track_id
            for track_id in valid_tracks
        }

        root_cameras = {
            track_id: {
                track_info[track_id]["camera"]
            }
            for track_id in valid_tracks
        }

        def find(track_id):

            root = track_id

            while parent[root] != root:
                root = parent[root]

            while parent[track_id] != track_id:
                next_id = parent[track_id]
                parent[track_id] = root
                track_id = next_id

            return root

        def union(track_a, track_b):

            root_a = find(track_a)
            root_b = find(track_b)

            if root_a == root_b:
                return False

            # One global object cannot contain two tracks
            # from the same camera.
            if (
                root_cameras[root_a]
                & root_cameras[root_b]
            ):
                return False

            parent[root_b] = root_a

            root_cameras[root_a] = (
                root_cameras[root_a]
                | root_cameras[root_b]
            )

            return True

        def track_distance(track_a, track_b):

            centers_a = track_info[
                track_a
            ]["centers"]

            centers_b = track_info[
                track_b
            ]["centers"]

            common_times = sorted(
                set(centers_a.keys())
                & set(centers_b.keys())
            )

            if (
                len(common_times)
                < association_min_overlap
            ):
                return float("inf")

            distances = torch.stack([
                torch.linalg.vector_norm(
                    centers_a[time_idx]
                    - centers_b[time_idx]
                )
                for time_idx in common_times
            ])

            return float(
                distances.median().item()
            )

        # ========================================================
        # Cross-camera mutual-nearest-neighbour association.
        # ========================================================

        for camera_a in range(V):

            tracks_a = [
                track_id
                for track_id in valid_tracks
                if (
                    track_info[
                        track_id
                    ]["camera"]
                    == camera_a
                    and len(
                        track_info[
                            track_id
                        ]["centers"]
                    ) > 0
                )
            ]

            if not tracks_a:
                continue

            for camera_b in range(
                camera_a + 1,
                V,
            ):

                tracks_b = [
                    track_id
                    for track_id in valid_tracks
                    if (
                        track_info[
                            track_id
                        ]["camera"]
                        == camera_b
                        and len(
                            track_info[
                                track_id
                            ]["centers"]
                        ) > 0
                    )
                ]

                if not tracks_b:
                    continue

                nearest_ab = {}

                for track_a in tracks_a:

                    candidates = [
                        (
                            track_distance(
                                track_a,
                                track_b,
                            ),
                            track_b,
                        )
                        for track_b
                        in tracks_b
                    ]

                    distance, best_track_b = min(
                        candidates,
                        key=lambda item: item[0],
                    )

                    nearest_ab[track_a] = (
                        best_track_b,
                        distance,
                    )

                nearest_ba = {}

                for track_b in tracks_b:

                    candidates = [
                        (
                            track_distance(
                                track_a,
                                track_b,
                            ),
                            track_a,
                        )
                        for track_a
                        in tracks_a
                    ]

                    distance, best_track_a = min(
                        candidates,
                        key=lambda item: item[0],
                    )

                    nearest_ba[track_b] = (
                        best_track_a,
                        distance,
                    )

                for (
                    track_a,
                    (
                        track_b,
                        distance,
                    ),
                ) in nearest_ab.items():

                    reverse_track_a, _ = (
                        nearest_ba[track_b]
                    )

                    if (
                        reverse_track_a
                        != track_a
                    ):
                        continue

                    if not math.isfinite(
                        distance
                    ):
                        continue

                    if (
                        distance
                        > association_max_distance
                    ):
                        continue

                    if union(
                        track_a,
                        track_b,
                    ):
                        total_merges += 1

        # ========================================================
        # Compact global IDs.
        # ========================================================

        roots = sorted({
            find(track_id)
            for track_id
            in valid_tracks
        })

        root_to_global_id = {
            root: index + 1
            for index, root
            in enumerate(roots)
        }

        total_global_objects += len(
            roots
        )

        for track_id in valid_tracks:

            global_id = (
                root_to_global_id[
                    find(track_id)
                ]
            )

            mask = (
                local_track_ids[
                    batch_idx
                ]
                == track_id
            )

            global_ids[
                batch_idx
            ][mask] = global_id

    diagnostics = {
        "r5_local_track_count":
            total_local_tracks,

        "r5_global_object_count":
            total_global_objects,

        "r5_cross_view_merge_count":
            total_merges,

        "r5_canonical_dynamic_ratio":
            canonical_keep
            .float()
            .mean()
            .detach(),
    }

    return (
        global_ids,
        canonical_keep,
        diagnostics,
    )

def predict_sam_canonical_motion(
    gs_state,
    means,
    local_track_ids,
    gs_time,
    timespan,
    motion_head,
    patch_size=8,
    min_observations=2,
    min_track_pixels=16,
    association_max_distance=4.0,
    association_min_overlap=1,
):
    """
    R5 pipeline:

      camera-local SAM tracks
                ↓
      cross-view 3D association
                ↓
          global object ID
                ↓
      pool ALL observations
                ↓
      one object velocity
                ↓
      canonical Gaussian source
    """

    if local_track_ids is None:
        raise RuntimeError(
            "R5 requires sam_track_ids"
        )

    (
        global_ids,
        canonical_keep,
        association_diag,
    ) = build_global_track_ids(
        means=means,
        local_track_ids=local_track_ids,
        min_track_pixels=min_track_pixels,
        association_max_distance=(
            association_max_distance
        ),
        association_min_overlap=(
            association_min_overlap
        ),
    )

    B, T, V, H, W = (
        global_ids.shape
    )

    _, N, C = gs_state.shape

    PH = H // patch_size
    PW = W // patch_size

    expected = T * V * PH * PW

    if N != expected:
        raise RuntimeError(
            "R5 token alignment mismatch: "
            f"state={gs_state.shape}, "
            f"track={global_ids.shape}, "
            f"expected={expected}"
        )

    features = gs_state.reshape(
        B, T, V, PH, PW, C
    )

    physical_time = (
        gs_time.float()
        * float(timespan)
    )

    velocity = gs_state.new_zeros(
        B, T, V, H, W, 3
    )

    predicted_count = 0
    skipped_count = 0
    speed_values = []

    for b in range(B):

        gids = torch.unique(
            global_ids[b]
        )

        gids = gids[gids > 0]

        for gid_tensor in gids:

            gid = int(
                gid_tensor.item()
            )

            pixel_mask = (
                global_ids[b] == gid
            ).float()

            # Pixel SAM mask -> token coverage.
            coverage = F.avg_pool2d(
                pixel_mask.reshape(
                    T * V,
                    1,
                    H,
                    W,
                ),
                kernel_size=patch_size,
                stride=patch_size,
            ).reshape(
                T,
                V,
                PH,
                PW,
            )

            observation_features = []
            observation_times = []
            observation_support = []

            for t in range(T):

                cov = coverage[t]

                support = cov.sum()

                if float(
                    support.detach()
                ) <= 0:
                    continue

                feat = features[b, t]

                pooled = (
                    feat
                    * cov[..., None]
                ).sum(
                    dim=(0, 1, 2)
                ) / support.clamp_min(
                    1e-6
                )

                view_support = cov.sum(
                    dim=(1, 2)
                )

                obs_time = (
                    physical_time[b, t]
                    * view_support
                ).sum() / (
                    view_support.sum()
                    .clamp_min(1e-6)
                )

                observation_features.append(
                    pooled
                )

                observation_times.append(
                    obs_time
                )

                observation_support.append(
                    support
                )

            if (
                len(observation_features)
                < min_observations
            ):
                skipped_count += 1
                continue

            obs_feat = torch.stack(
                observation_features
            )

            obs_time = torch.stack(
                observation_times
            ).float()

            obs_support = torch.stack(
                observation_support
            ).float()

            feat_mean = obs_feat.mean(
                dim=0
            )

            time_mean = obs_time.mean()

            dt = (
                obs_time - time_mean
            )

            denom = (
                dt.square().sum()
            )

            if float(
                denom.detach()
            ) > 1e-8:

                feat_slope = (
                    dt[:, None]
                    * (
                        obs_feat
                        - feat_mean
                    )
                ).sum(dim=0) / denom

            else:

                feat_slope = (
                    torch.zeros_like(
                        feat_mean
                    )
                )

            duration = (
                obs_time.max()
                - obs_time.min()
            )

            support_feature = (
                torch.log1p(
                    obs_support.mean()
                ) / 5.0
            )

            descriptor = torch.cat([
                feat_mean,
                feat_slope,

                duration.to(
                    feat_mean.dtype
                ).reshape(1),

                support_feature.to(
                    feat_mean.dtype
                ).reshape(1),
            ])

            object_velocity = (
                motion_head(
                    descriptor[None]
                )[0]
            )

            velocity[b][
                global_ids[b] == gid
            ] = object_velocity.to(
                velocity.dtype
            )

            predicted_count += 1

            speed_values.append(
                object_velocity
                .float()
                .norm()
            )

    if speed_values:

        speeds = torch.stack(
            speed_values
        )

        speed_mean = (
            speeds.mean().detach()
        )

        speed_max = (
            speeds.max().detach()
        )

    else:

        speed_mean = torch.zeros(
            (),
            device=gs_state.device,
        )

        speed_max = speed_mean

    diagnostics = {
        **association_diag,

        "r5_predicted_object_count":
            predicted_count,

        "r5_skipped_object_count":
            skipped_count,

        "r5_object_speed_mean":
            speed_mean,

        "r5_object_speed_max":
            speed_max,
    }

    return (
        velocity,
        global_ids,
        canonical_keep,
        diagnostics,
    )
