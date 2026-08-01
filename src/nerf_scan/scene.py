import torch

GREEN = (0.15, 0.85, 0.25)


def ray_box_intersect(rays_o: torch.Tensor, rays_d: torch.Tensor,
                       box_min: torch.Tensor, box_max: torch.Tensor):
    inv_d = 1.0 / (rays_d + 1e-10)
    t0 = (box_min - rays_o) * inv_d
    t1 = (box_max - rays_o) * inv_d
    t_near = torch.minimum(t0, t1).amax(dim=-1)
    t_far = torch.maximum(t0, t1).amin(dim=-1)
    hit = (t_far > t_near)
    return t_near, t_far, hit


def synthetic_cube_target(rays_o: torch.Tensor, rays_d: torch.Tensor,
                           half_size: float = 1.1, device: torch.device = None) -> torch.Tensor:
    H, W, _ = rays_o.shape
    box_min = torch.tensor([-half_size] * 3, device=device)
    box_max = torch.tensor([half_size] * 3, device=device)

    _, _, hit = ray_box_intersect(rays_o, rays_d, box_min, box_max)

    target = torch.zeros(H, W, 3, device=device)
    green = torch.tensor(GREEN, device=device)
    target[hit] = green
    return target
