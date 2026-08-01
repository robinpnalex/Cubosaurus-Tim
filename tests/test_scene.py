import torch
import pytest

from nerf_scan.scene import ray_box_intersect, synthetic_cube_target, GREEN


def test_ray_box_intersect_hits_head_on():
    rays_o = torch.tensor([[0.0, 0.0, 3.0]])
    rays_d = torch.tensor([[0.0, 0.0, -1.0]])
    box_min = torch.tensor([-1.0, -1.0, -1.0])
    box_max = torch.tensor([1.0, 1.0, 1.0])

    t_near, t_far, hit = ray_box_intersect(rays_o, rays_d, box_min, box_max)

    assert hit.item() is True
    assert t_near.item() == pytest.approx(2.0, abs=1e-4)
    assert t_far.item() == pytest.approx(4.0, abs=1e-4)


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
