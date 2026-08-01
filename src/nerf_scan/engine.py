import time

import cv2
import torch

from .config import (
    H, W, FOCAL, NEAR, FAR, N_SAMPLES, RADIUS, ELEVATION,
    ANGLE_STEP, DISPLAY_SCALE, LR, CUBE_HALF_SIZE,
)
from .device import get_device
from .encoding import PositionalEncoder
from .model import TinyNeRFModel
from .camera import pose_spherical, get_rays
from .rendering import render_rays
from .scene import synthetic_cube_target
from .display import build_display_frame

WINDOW_NAME = "NeRF-Scan  |  3D Render + CT Depth  |  press q to quit"


def run():
    torch.manual_seed(0)
    device = get_device()

    print(f"[NeRF-Scan] device = {device}")
    print("[NeRF-Scan] press 'q' or Esc in the window to quit")

    encoder = PositionalEncoder(input_dims=3, num_freqs=6, include_input=True).to(device)
    model = TinyNeRFModel(pos_enc_dim=encoder.out_dim, hidden=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    step = 0
    theta = 0.0
    t0 = time.time()

    try:
        while True:
            c2w = pose_spherical(theta, radius=RADIUS, elevation_deg=ELEVATION)
            rays_o, rays_d = get_rays(H, W, FOCAL, c2w, device)
            target = synthetic_cube_target(rays_o, rays_d, half_size=CUBE_HALF_SIZE, device=device)

            rgb_map, depth_map, acc_map = render_rays(
                model, encoder, rays_o, rays_d, NEAR, FAR, N_SAMPLES, device, perturb=True
            )

            loss = torch.mean((rgb_map - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            frame = build_display_frame(
                rgb_map, depth_map, acc_map, NEAR, FAR, step, loss.item(),
                device_type=device.type, scale=DISPLAY_SCALE,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            theta = (theta + ANGLE_STEP) % 360.0
            step += 1

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - t0
        fps = step / elapsed if elapsed > 0 else 0.0
        print(f"[NeRF-Scan] stopped after {step} steps ({fps:.1f} it/s)")
        cv2.destroyAllWindows()
