import torch
import pytest

from nerf_scan.camera import pose_spherical, get_rays


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
