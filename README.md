# Cubosaurus-Tim 🦖🧊

![Cubosaurus-Tim](cover.gif)

## The Story of Tim
Dino Tim was just a plastic T-Rex and the desk's lead debugging mascot, but tonight he was meant for digital immortality. Jasmine and Sparsh had spent hours trying to code Tim into a 3D space completely from scratch, using zero external files or datasets.

"The engine just can't handle his aura," Sparsh muttered, pointing at the screen. Instead of a fierce apex predator, their code had spat out a perfectly smooth, synthetic green cube generated purely in the machine's memory.

"His chaotic energy broke the system," Jasmine observed. "It compressed his essence into a hyper-cube."

"If we stop looking at it, the geometry might collapse," Sparsh warned.

To contain the anomaly, Jasmine rigged a virtual camera to endlessly orbit the object on its Y-axis. The monitor instantly split in two. On the left, the glowing green cube spun flawlessly in 3D space. On the right, a blazing, inferno-colored depth map tracked its exact hidden structure. 

Sparsh stared at the synchronized render, then down at the plastic dinosaur. "Well, Tim. You're a cube now."

---

## What is this project?
This is a from-scratch Neural Radiance Field (NeRF) that trains **live in a window**, rendering two synchronized views of the same 3D volume as it learns. 

You'll explore the codebase to see how rays are cast, how positional encodings handle spatial frequencies, how volumetric intersections work, and how a tiny MLP turns all that into a 3D rendering.

## Setup & Installation

We recommend using `uv` for lightning-fast package management.

### 1. Install uv
If you don't have it installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(On Windows, use `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)*

### 2. Run the project
```bash
uv sync --extra dev
uv run main.py
```
Press `q` or `Esc` in the window to quit.

## Project Structure
The core logic lives inside the `src/nerf_scan/` directory. You will find:
- **`engine.py`**: The main entry point tying the training loop and UI together.
- **`model.py`**, **`encoding.py`**, **`rendering.py`**: The Neural Radiance Field math, the Fourier-features, and the volume rendering integral.
- **`camera.py`**, **`scene.py`**, **`display.py`**: The virtual camera logic, the synthetic scene generation, and the live OpenCV UI.
- **`tests/`**: A comprehensive suite of unit tests validating the math behind the scenes. We encourage you to poke around to understand how everything connects!

---

## 🐛 The Bug Hunt Challenge

This repo runs as an ongoing bug-hunt challenge. A maintainer occasionally introduces a real, working bug into `src/nerf_scan/` and opens a GitHub Issue describing the *symptom* (not the cause). Your job is to track down the cause and fix it.

### How to participate:
1. **Pick an open issue** from the Issues tab. 
2. **Fork & clone**, then install dev dependencies:
   ```bash
   uv sync --extra dev
   ```
3. **Reproduce it.** Two ways:
   - Run `uv run pytest -v` — bugs in the math will usually show up as a failing unit test with a clear expected-vs-actual diff.
   - Run `uv run main.py` and watch the live window — a subtler visual bug may only be visible there.
4. **Fix the bug** in the smallest way that addresses the root cause. Avoid papering over it by changing test expectations — the tests encode the correct math/behavior; they should not need to change.
5. **Add a test** if the bug wasn't already caught by one.
6. **Open a PR** that references the issue number (e.g. `Fixes #7`) and briefly explains root cause + fix.

### Guidelines
- Keep PRs scoped to one issue at a time.
- Don't refactor unrelated code in a bug-fix PR.
- All tests must pass (`uv run pytest`) before a PR is merged.

### Background reading
- NeRF: Mildenhall et al., *"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"* (2020)
- Positional (Fourier feature) encoding for coordinate-based MLPs
- The "slab method" for ray/AABB intersection
