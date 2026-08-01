import torch
import pytest

from nerf_scan.rendering import (
    stratified_sample_t_vals,
    sample_points_along_rays,
    compute_deltas,
    compute_alpha,
    compute_weights,
    composite,
)


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


def test_compute_weights_hand_verified():
    alpha = torch.tensor([[[0.5, 1.0]]])
    weights = compute_weights(alpha)
    expected = torch.tensor([[[0.5, 0.5]]])
    assert torch.allclose(weights, expected, atol=1e-4)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_composite_hand_verified():
    weights = torch.tensor([[[0.5, 0.5]]])
    rgb = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
    t_vals = torch.tensor([[[2.0, 3.0]]])

    rgb_map, depth_map, acc_map = composite(weights, rgb, t_vals)

    assert torch.allclose(rgb_map, torch.tensor([[[0.5, 0.5, 0.0]]]))
    assert torch.allclose(depth_map, torch.tensor([[2.5]]))
    assert torch.allclose(acc_map, torch.tensor([[1.0]]))
