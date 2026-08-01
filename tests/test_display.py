import numpy as np
import torch
import pytest

from nerf_scan.display import (
    compose_rgb,
    to_uint8_bgr,
    compute_depth_vis,
    normalize_depth_inverted,
    rgb_tensor_to_bgr_image,
    depth_tensor_to_inferno_image,
    build_display_frame,
)


def test_compose_rgb_fully_opaque_ignores_background():
    rgb_map = torch.tensor([[[1.0, 0.0, 0.0]]])
    acc_map = torch.tensor([[1.0]])
    composited = compose_rgb(rgb_map, acc_map, background=(0.5, 0.5, 0.5))
    assert torch.allclose(composited, rgb_map)


def test_compose_rgb_fully_empty_shows_background():
    rgb_map = torch.tensor([[[0.0, 0.0, 0.0]]])
    acc_map = torch.tensor([[0.0]])
    composited = compose_rgb(rgb_map, acc_map, background=(0.5, 0.2, 0.1))
    assert torch.allclose(composited, torch.tensor([[[0.5, 0.2, 0.1]]]))


def test_to_uint8_bgr_swaps_channels_and_clamps():
    composited = torch.tensor([[[2.0, 0.0, -1.0]]])
    img = to_uint8_bgr(composited)
    assert tuple(img[0, 0]) == (0, 0, 255)


def test_compute_depth_vis_pushes_empty_space_to_far():
    depth_map = torch.tensor([[3.0]])
    acc_map = torch.tensor([[0.0]])
    depth_vis = compute_depth_vis(depth_map, acc_map, far=6.0)
    assert depth_vis.item() == pytest.approx(6.0)


def test_normalize_depth_inverted_near_is_hot_far_is_cold():
    depth_vis = torch.tensor([2.0, 6.0])
    normalized = normalize_depth_inverted(depth_vis, near=2.0, far=6.0)
    assert normalized[0].item() == pytest.approx(1.0)
    assert normalized[1].item() == pytest.approx(0.0)


def test_rgb_tensor_to_bgr_image_shape_and_dtype():
    rgb_map = torch.rand(4, 4, 3)
    acc_map = torch.rand(4, 4)
    img = rgb_tensor_to_bgr_image(rgb_map, acc_map)
    assert img.shape == (4, 4, 3)
    assert img.dtype == np.uint8


def test_depth_tensor_to_inferno_image_shape_and_dtype():
    depth_map = torch.rand(4, 4) * 4 + 2
    acc_map = torch.rand(4, 4)
    img = depth_tensor_to_inferno_image(depth_map, acc_map, near=2.0, far=6.0)
    assert img.shape == (4, 4, 3)
    assert img.dtype == np.uint8


def test_build_display_frame_layout():
    H, W, scale = 4, 4, 2
    rgb_map = torch.rand(H, W, 3)
    depth_map = torch.rand(H, W) * 4 + 2
    acc_map = torch.rand(H, W)

    frame = build_display_frame(
        rgb_map, depth_map, acc_map, near=2.0, far=6.0,
        step=0, loss_val=0.1, device_type="cpu", scale=scale,
    )

    header_h, footer_h, gap = 36, 30, 6
    expected_w = W * scale * 2 + gap
    expected_h = header_h + H * scale + footer_h
    assert frame.shape == (expected_h, expected_w, 3)
    assert frame.dtype == np.uint8
