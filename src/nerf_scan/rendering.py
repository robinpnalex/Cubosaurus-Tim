import torch


def stratified_sample_t_vals(near, far, n_samples, H, W, device, perturb=True):
    t_vals = torch.linspace(near, far, n_samples, device=device)
    if perturb:
        mids = 0.5 * (t_vals[1:] + t_vals[:-1])
        lower = torch.cat([mids, t_vals[-1:]])
        upper = torch.cat([t_vals[:1], mids])
        t_rand = torch.rand(H, W, n_samples, device=device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = t_vals.view(1, 1, n_samples).expand(H, W, n_samples)
    return t_vals


def sample_points_along_rays(rays_o, rays_d, t_vals):
    return rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., None]


def compute_deltas(t_vals):
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    return dists


def compute_alpha(sigma, dists):
    return 1.0 - torch.exp(-sigma * dists)


def compute_weights(alpha):
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-1], dim=-1), dim=-1
    )[..., :-1]
    return alpha * transmittance


def composite(weights, rgb, t_vals):
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * t_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)
    return rgb_map, depth_map, acc_map


def render_rays(model, encoder, rays_o, rays_d, near, far, n_samples, device, perturb=True):
    H, W, _ = rays_d.shape

    t_vals = stratified_sample_t_vals(near, far, n_samples, H, W, device, perturb)
    pts = sample_points_along_rays(rays_o, rays_d, t_vals)

    encoded = encoder(pts.reshape(-1, 3))
    rgb_flat, sigma_flat = model(encoded)
    rgb = rgb_flat.view(H, W, 3, 3)
    sigma = sigma_flat.view(H, W, n_samples)

    dists = compute_deltas(t_vals)
    alpha = compute_alpha(sigma, dists)
    weights = compute_weights(alpha)

    rgb_map, depth_map, acc_map = composite(weights, rgb, t_vals)
    return rgb_map, depth_map, acc_map
