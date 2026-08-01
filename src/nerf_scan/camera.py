import numpy as np
import torch


def pose_spherical(theta_deg: float, radius: float = 4.0, elevation_deg: float = 20.0) -> torch.Tensor:
    theta = np.radians(theta_deg)
    phi = np.radians(elevation_deg)

    eye = np.array([
        radius * np.cos(phi) * np.sin(theta),
        radius * np.sin(phi),
        radius * np.cos(phi) * np.cos(theta),
    ], dtype=np.float32)

    target = np.zeros(3, dtype=np.float32)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return torch.from_numpy(c2w)


def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor, device: torch.device):
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i),
    ], dim=-1)

    rot = c2w[:3, :3].to(device)
    rays_d = torch.sum(dirs[..., None, :] * rot, dim=-1)
    rays_o = c2w[:3, 3].to(device).expand(rays_d.shape)
    return rays_o, rays_d
