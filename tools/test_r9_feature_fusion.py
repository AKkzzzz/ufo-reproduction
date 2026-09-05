#!/usr/bin/env python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ufo.models.sam_object_detail_r9 import (
    SAMObjectDetailFusionHead,
    fuse_canonical_detail_features,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    B, T, V, H, W = 1, 2, 1, 8, 8
    patch, C = 4, 16
    PH, PW = H // patch, W // patch

    gs_state = torch.randn(
        B, T * V * PH * PW, C,
        device=device, dtype=dtype, requires_grad=True,
    )

    yy, xx = torch.meshgrid(
        torch.linspace(-0.3, 0.3, H, device=device),
        torch.linspace(-0.5, 0.5, W, device=device),
        indexing="ij",
    )
    base = torch.stack(
        [xx, yy, torch.full_like(xx, 10.0)], dim=-1
    )
    means = torch.stack(
        [
            base,
            base + torch.tensor(
                [0.5, 0.0, 0.0], device=device
            ),
        ],
        dim=0,
    )[None, :, None].to(dtype)

    scales = torch.full_like(means, 0.05)
    quats = torch.zeros(
        B, T, V, H, W, 4, device=device, dtype=dtype
    )
    quats[..., 0] = 1.0
    opacities = torch.full(
        (B, T, V, H, W), 0.5, device=device, dtype=dtype
    )
    colors = torch.randn(
        B, T, V, H, W, 3, device=device, dtype=dtype
    ) * 0.1
    velocity = torch.zeros_like(means)
    velocity[..., 0] = 1.0
    global_ids = torch.ones(
        B, T, V, H, W, device=device, dtype=torch.long
    )
    context_image = torch.randn(
        B, T, V, 3, H, W, device=device, dtype=dtype
    )
    gs_time = torch.tensor(
        [[[0.0], [1.0]]], device=device, dtype=torch.float32
    )

    head = SAMObjectDetailFusionHead(
        embed_dim=C, color_dim=3, hidden_dim=32
    ).to(device=device, dtype=dtype)

    outputs, diag = fuse_canonical_detail_features(
        gs_state=gs_state,
        gs_params={
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "colors": colors,
        },
        context_image=context_image,
        global_ids=global_ids,
        velocity=velocity,
        gs_time=gs_time,
        timespan=0.5,
        detail_head=head,
        patch_size=patch,
        voxel_size=0.10,
    )

    loss = (
        outputs["means"].float().square().mean()
        + outputs["colors"].float().square().mean()
        + outputs["opacities"].float().mean()
    )
    loss.backward()

    head_grad = sum(
        float(p.grad.float().norm())
        for p in head.parameters()
        if p.grad is not None
    )
    state_grad = float(gs_state.grad.float().norm())

    assert torch.isfinite(loss)
    assert head_grad > 0
    assert state_grad > 0
    assert float(diag["r9_fused_voxel_count"]) > 0

    print("R9 FEATURE FUSION BF16 FORWARD/BACKWARD PASS")
    print("voxels=", int(diag["r9_fused_voxel_count"]))
    print("fusion_ratio=", float(diag["r9_fusion_ratio"]))
    print("head_grad=", head_grad)
    print("state_grad=", state_grad)


if __name__ == "__main__":
    main()
