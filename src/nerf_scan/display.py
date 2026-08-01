import cv2
import numpy as np
import torch

BACKGROUND = (0.04, 0.04, 0.07)


def compose_rgb(rgb_map: torch.Tensor, acc_map: torch.Tensor,
                 background=BACKGROUND) -> torch.Tensor:
    bg = torch.tensor(background, device=rgb_map.device)
    return rgb_map + (1.0 - acc_map[..., None]) * bg


def to_uint8_bgr(composited: torch.Tensor) -> np.ndarray:
    img = composited.clamp(0.0, 1.0).detach().cpu().numpy()
    img = (img * 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def rgb_tensor_to_bgr_image(rgb_map: torch.Tensor, acc_map: torch.Tensor) -> np.ndarray:
    composited = compose_rgb(rgb_map, acc_map)
    return to_uint8_bgr(composited)


def compute_depth_vis(depth_map: torch.Tensor, acc_map: torch.Tensor, far: float) -> torch.Tensor:
    return depth_map


def normalize_depth_inverted(depth_vis: torch.Tensor, near: float, far: float) -> torch.Tensor:
    normalized = 1.0 - (depth_vis - near) / (far - near)
    return normalized.clamp(0.0, 1.0)


def depth_tensor_to_inferno_image(depth_map: torch.Tensor, acc_map: torch.Tensor,
                                   near: float, far: float) -> np.ndarray:
    depth_vis = compute_depth_vis(depth_map, acc_map, far)
    normalized = normalize_depth_inverted(depth_vis, near, far)
    depth_u8 = (normalized.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)


def build_display_frame(rgb_map, depth_map, acc_map, near, far, step, loss_val,
                         device_type: str = "cpu", scale: int = 4) -> np.ndarray:
    H, W = rgb_map.shape[0], rgb_map.shape[1]

    rgb_img = rgb_tensor_to_bgr_image(rgb_map, acc_map)
    depth_img = depth_tensor_to_inferno_image(depth_map, acc_map, near, far)

    rgb_img = cv2.resize(rgb_img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    depth_img = cv2.resize(depth_img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)

    header_h, footer_h, gap = 36, 30, 6
    canvas_w = W * scale * 2 + gap
    canvas_h = header_h + H * scale + footer_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[header_h:header_h + H * scale, 0:W * scale] = rgb_img
    canvas[header_h:header_h + H * scale, W * scale + gap:] = depth_img

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "3D Render", (10, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "CT Depth", (W * scale + gap + 10, 25), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    footer_text = f"Step {step:05d}   |   Loss {loss_val:.5f}   |   Device: {device_type}"
    cv2.putText(canvas, footer_text, (10, canvas_h - 10), font, 0.55, (0, 255, 140), 1, cv2.LINE_AA)

    return canvas
