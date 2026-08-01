import numpy as np
import torch
import pytest

from nerf_scan.camera import pose_spherical, get_rays
from nerf_scan.display import (
    compose_rgb,
    to_uint8_bgr,
    normalize_depth_inverted,
    rgb_tensor_to_bgr_image,
    depth_tensor_to_inferno_image,
    build_display_frame,
)
from nerf_scan.rendering import (
    stratified_sample_t_vals,
    sample_points_along_rays,
    compute_deltas,
    compute_alpha,
    composite,
)


@pytest.fixture
def identity_c2w():
    return torch.eye(4, dtype=torch.float32)


def test_pose_spherical_front_on_axis():
    c2w = pose_spherical(theta_deg=0.0, radius=4.0, elevation_deg=0.0)
    expected = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 4.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert torch.allclose(c2w, expected, atol=1e-5)


def test_pose_spherical_eye_matches_spherical_coords():
    c2w = pose_spherical(theta_deg=90.0, radius=2.0, elevation_deg=0.0)
    eye = c2w[:3, 3]
    assert torch.allclose(eye, torch.tensor([2.0, 0.0, 0.0]), atol=1e-4)


def test_pose_spherical_rotation_is_orthonormal():
    c2w = pose_spherical(theta_deg=37.0, radius=3.0, elevation_deg=15.0)
    rot = c2w[:3, :3]
    should_be_identity = rot.T @ rot
    assert torch.allclose(should_be_identity, torch.eye(3), atol=1e-5)


def test_get_rays_shapes_and_origin(identity_c2w):
    H, W, focal = 2, 2, 1.0
    rays_o, rays_d = get_rays(H, W, focal, identity_c2w, torch.device("cpu"))
    assert rays_o.shape == (H, W, 3)
    assert rays_d.shape == (H, W, 3)
    assert torch.allclose(rays_o, torch.zeros(H, W, 3))


def test_get_rays_directions_hand_verified(identity_c2w):
    H, W, focal = 2, 2, 1.0
    _, rays_d = get_rays(H, W, focal, identity_c2w, torch.device("cpu"))
    expected = torch.tensor([
        [[-1.0, 1.0, -1.0], [0.0, 1.0, -1.0]],
        [[-1.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
    ])
    assert torch.allclose(rays_d, expected, atol=1e-6)


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


def test_stratified_sample_t_vals_no_perturb_matches_linspace():
    t_vals = stratified_sample_t_vals(
        near=2.0, far=6.0, n_samples=5, H=2, W=2, device=torch.device("cpu"), perturb=False
    )
    expected = torch.linspace(2.0, 6.0, 5)
    assert t_vals.shape == (2, 2, 5)
    for a in range(2):
        for b in range(2):
            assert torch.allclose(t_vals[a, b], expected)


def test_stratified_sample_t_vals_perturb_stays_in_bounds():
    torch.manual_seed(0)
    t_vals = stratified_sample_t_vals(
        near=2.0, far=6.0, n_samples=8, H=3, W=3, device=torch.device("cpu"), perturb=True
    )
    assert t_vals.shape == (3, 3, 8)
    assert t_vals.min() >= 2.0
    assert t_vals.max() <= 6.0
    assert torch.all(t_vals[..., 1:] >= t_vals[..., :-1])


def test_sample_points_along_rays():
    rays_o = torch.zeros(1, 1, 3)
    rays_d = torch.tensor([[[0.0, 0.0, -1.0]]])
    t_vals = torch.tensor([[[1.0, 2.0, 3.0]]])
    pts = sample_points_along_rays(rays_o, rays_d, t_vals)
    expected = torch.tensor([[[[0.0, 0.0, -1.0], [0.0, 0.0, -2.0], [0.0, 0.0, -3.0]]]])
    assert torch.allclose(pts, expected)


def test_compute_deltas():
    t_vals = torch.tensor([[[1.0, 2.0, 4.0]]])
    dists = compute_deltas(t_vals)
    assert torch.allclose(dists[..., 0], torch.tensor([1.0]))
    assert torch.allclose(dists[..., 1], torch.tensor([2.0]))
    assert dists[..., 2].item() == pytest.approx(1e10)


def test_compute_alpha_zero_density_is_transparent():
    sigma = torch.zeros(1, 1, 3)
    dists = torch.ones(1, 1, 3)
    alpha = compute_alpha(sigma, dists)
    assert torch.allclose(alpha, torch.zeros(1, 1, 3))


def test_compute_alpha_high_density_is_opaque():
    sigma = torch.full((1, 1, 1), 100.0)
    dists = torch.ones(1, 1, 1)
    alpha = compute_alpha(sigma, dists)
    assert alpha.item() == pytest.approx(1.0, abs=1e-6)


def test_composite_hand_verified():
    weights = torch.tensor([[[0.5, 0.5]]])
    rgb = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
    t_vals = torch.tensor([[[2.0, 3.0]]])
    rgb_map, depth_map, acc_map = composite(weights, rgb, t_vals)
    assert torch.allclose(rgb_map, torch.tensor([[[0.5, 0.5, 0.0]]]))
    assert torch.allclose(depth_map, torch.tensor([[2.5]]))
    assert torch.allclose(acc_map, torch.tensor([[1.0]]))


def test_ray_box_intersect_hits_head_on():
    from nerf_scan.scene import ray_box_intersect
    rays_o = torch.tensor([[0.0, 0.0, 3.0]])
    rays_d = torch.tensor([[0.0, 0.0, -1.0]])
    box_min = torch.tensor([-1.0, -1.0, -1.0])
    box_max = torch.tensor([1.0, 1.0, 1.0])
    t_near, t_far, hit = ray_box_intersect(rays_o, rays_d, box_min, box_max)
    assert hit.item() is True
    assert t_near.item() == pytest.approx(2.0, abs=1e-4)
    assert t_far.item() == pytest.approx(4.0, abs=1e-4)
