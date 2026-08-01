"""
NeRF CT-Scan Engine
====================
A single-file-turned-modular, from-scratch Neural Radiance Field (NeRF)
demo that trains live in a window, rendering two synchronized views of the
same 3D volume:

    LEFT  : the RGB volumetric render ("3D Render")
    RIGHT : the volumetric depth map, false-colored to look like a CT scan

Run:
    python main.py

Controls:
    q / Esc  -> quit
"""

from nerf_scan.engine import run

if __name__ == "__main__":
    run()
