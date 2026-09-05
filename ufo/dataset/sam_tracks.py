from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def load_sam_track_mask(
    root,
    scene_name,
    camera,
    frame_idx,
    target_size,
    camera_slot=None,
):
    """
    Returns [H,W] int64 pseudo track IDs.

    0 = background.
    SAM IDs are camera-local, so optionally offset them to avoid
    accidental collisions before later cross-view 3D association.
    """
    path = (
        Path(root)
        / scene_name
        / str(camera)
        / "mask_data"
        / f"mask_{frame_idx:06d}.npy"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    mask = np.load(path)
    mask = torch.from_numpy(mask.astype(np.int64))

    H, W = target_size
    if tuple(mask.shape) != (H, W):
        mask = F.interpolate(
            mask[None, None].float(),
            size=(H, W),
            mode="nearest",
        )[0, 0].long()

    # Preserve 0 as background.
    if camera_slot is not None:
        fg = mask > 0
        mask[fg] += (camera_slot + 1) * 10000

    return mask
