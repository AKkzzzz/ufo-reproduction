import math
from types import SimpleNamespace

import torch

from ufo.models.vit import Mlp
from ufo.paper_contract import (
    GAUSSIANS_PER_TOKEN,
    expand_token_assignments,
    relative_se3,
    split_aux_tokens,
    split_context_supervision,
    transform_directions,
    transform_points,
    assert_paper_training_ready,
)
from ufo.utils.misc import compute_visible_topk_indices_any_view


def test_waymo_frame_protocol_partitions_every_fifth_frame():
    protocol = split_context_supervision(20, 40)
    assert protocol.context == (20, 25, 30, 35)
    assert protocol.supervision == tuple(i for i in range(20, 40) if i % 5 != 0)
    assert set(protocol.context).isdisjoint(protocol.supervision)
    assert sorted(protocol.context + protocol.supervision) == list(range(20, 40))


def test_full_se3_local_world_roundtrip_for_points_and_directions():
    angle = math.pi / 3
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    scene_from_local = torch.eye(4).unsqueeze(0)
    scene_from_local[0, :3, :3] = rotation
    scene_from_local[0, :3, 3] = torch.tensor([10.0, -4.0, 2.0])
    local_c2w = torch.eye(4).unsqueeze(0)
    global_c2w = scene_from_local @ local_c2w
    actual = relative_se3(global_c2w, local_c2w)
    points = torch.tensor([[[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]]])
    directions = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    world_points = transform_points(actual[:, None], points)
    world_dirs = transform_directions(actual[:, None], directions)
    inverse = torch.linalg.inv(actual)
    assert torch.allclose(transform_points(inverse[:, None], world_points), points, atol=1e-6)
    assert torch.allclose(transform_directions(inverse[:, None], world_dirs), directions, atol=1e-6)
    assert not torch.allclose(world_dirs, directions)


def test_visible_topk_uses_euclidean_camera_center_distance():
    # Point 0 has larger camera-Z but is closer in Euclidean distance than point 1.
    points = torch.tensor([[[0.0, 0.0, 3.0], [5.0, 0.0, 2.0]]])
    cameras = torch.eye(4).reshape(1, 1, 4, 4)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
    selected = compute_visible_topk_indices_any_view(
        points, cameras, intrinsics, H=10, W=10, filter_num=1, cell_size=None
    )
    assert selected.item() == 0


def test_scene_token_assignment_is_shared_by_64_gaussians():
    token_weights = torch.tensor([[[0.2, 0.8], [0.7, 0.3]]])
    expanded = expand_token_assignments(token_weights, GAUSSIANS_PER_TOKEN)
    assert expanded.shape == (1, 128, 2)
    assert torch.equal(expanded[:, :64], token_weights[:, :1].expand(-1, 64, -1))
    assert torch.equal(expanded[:, 64:], token_weights[:, 1:].expand(-1, 64, -1))


def test_aux_outputs_are_split_from_recurrent_aux_sequence():
    aux = torch.arange(7.0).reshape(1, 7, 1)
    sky, affine, motion = split_aux_tokens(aux, num_cameras=3, num_motion_tokens=3)
    assert sky.flatten().tolist() == [0.0]
    assert affine.flatten().tolist() == [1.0, 2.0, 3.0]
    assert motion.flatten().tolist() == [4.0, 5.0, 6.0]


def test_contract_mlp_has_two_linear_layers():
    module = Mlp(31, 768, 768)
    assert sum(isinstance(layer, torch.nn.Linear) for layer in module.modules()) == 2


def test_formal_training_is_blocked_while_paper_semantics_are_unknown():
    args = SimpleNamespace(
        paper_frame_protocol=True,
        paper_supervision_mode="unknown",
        enable_flow_reg_loss=True,
        paper_forward_flow_impl=False,
    )
    try:
        assert_paper_training_ready(args)
    except RuntimeError as error:
        assert "supervision" in str(error)
        assert "forward-flow" in str(error)
    else:
        raise AssertionError("unverified paper training must be blocked")
