import torch
import pytest

from nerf_scan.display import compute_depth_vis
from nerf_scan.rendering import compute_weights
from nerf_scan.scene import ray_box_intersect, synthetic_cube_target, GREEN
from nerf_scan.encoding import PositionalEncoder


def test_compute_depth_vis_pushes_empty_space_to_far():
    depth_map = torch.tensor([[3.0]])
    acc_map = torch.tensor([[0.0]])
    depth_vis = compute_depth_vis(depth_map, acc_map, far=6.0)
    assert depth_vis.item() == pytest.approx(6.0)


def test_compute_depth_vis_preserves_solid_geometry():
    depth_map = torch.tensor([[3.5]])
    acc_map = torch.tensor([[1.0]])
    depth_vis = compute_depth_vis(depth_map, acc_map, far=6.0)
    assert depth_vis.item() == pytest.approx(3.5)


def test_compute_weights_hand_verified():
    alpha = torch.tensor([[[0.5, 1.0]]])
    weights = compute_weights(alpha)
    expected = torch.tensor([[[0.5, 0.5]]])
    assert torch.allclose(weights, expected, atol=1e-4)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_compute_weights_opaque_first_sample_blocks_rest():
    alpha = torch.tensor([[[1.0, 1.0, 1.0]]])
    weights = compute_weights(alpha)
    assert weights[0, 0, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert weights[0, 0, 1].item() == pytest.approx(0.0, abs=1e-4)
    assert weights[0, 0, 2].item() == pytest.approx(0.0, abs=1e-4)


def test_ray_box_intersect_misses_when_box_is_behind():
    rays_o = torch.tensor([[0.0, 0.0, 3.0]])
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])
    box_min = torch.tensor([-1.0, -1.0, -1.0])
    box_max = torch.tensor([1.0, 1.0, 1.0])
    _, _, hit = ray_box_intersect(rays_o, rays_d, box_min, box_max)
    assert hit.item() is False


def test_synthetic_cube_target_fills_hit_pixels_with_green():
    rays_o = torch.tensor([[[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]]])
    rays_d = torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]])
    target = synthetic_cube_target(rays_o, rays_d, half_size=1.0, device=torch.device("cpu"))
    assert target.shape == (1, 2, 3)
    assert torch.allclose(target[0, 0], torch.tensor(GREEN))
    assert torch.allclose(target[0, 1], torch.zeros(3))


def test_encoding_uses_exponential_frequency_spacing():
    encoder = PositionalEncoder(input_dims=3, num_freqs=6, include_input=False)
    freqs = encoder.freq_bands
    assert freqs.shape == (6,)
    assert freqs[0].item() == pytest.approx(1.0, abs=1e-5)
    assert freqs[-1].item() == pytest.approx(32.0, abs=1e-4)
    for i in range(len(freqs) - 1):
        ratio = freqs[i + 1].item() / freqs[i].item()
        assert ratio == pytest.approx(2.0, abs=1e-4), \
            f"Expected exponential spacing (ratio=2), got {ratio:.4f} at index {i}"
