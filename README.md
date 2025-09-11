# Integral Render (Radiosity + Galerkin)

This project implements a simple indoor radiosity renderer for a box room with a cube and optional extra rectangular prisms. It assembles a dense form‑factor matrix using a center‑to‑center approximation, solves the radiosity system with a P0 (per‑patch constant) Galerkin discretization, and renders with a small CPU raycaster.

## Quick Start

- Run full pipeline with multiple light scenarios and images:
  - `python3 main.py`
  - Outputs: images in `outputs/`, and linear system / solution CSVs in `linear_system_csv/`.

- Export linear system from parameters in `setup.py` (CLI):
  - `python3 linear_system.py`
  - Outputs: `linear_system_csv/A.csv`, `linear_system_csv/b.csv` (canonical Ax=b) and legacy `M.csv`, `E.csv` (E may contain multiple columns if multiple scenarios).

- Solve linear system from CSV (CLI):
  - `python3 solve.py`
  - Inputs (preferred): `linear_system_csv/A.csv`, `linear_system_csv/b.csv`. Falls back to `M.csv`/`E.csv`.
  - Outputs: `linear_system_csv/x.csv` (radiosity), `linear_system_csv/L.csv` (radiance).

- Render directly from CSV solution (CLI):
  - `python3 render.py`
  - Finds in order (multi-column supported): `L.csv` → `x.csv` → legacy `L_multi.csv`/`B.csv`/`B_multi.csv`. Renders one image per column to `outputs/`.

All tunable parameters live in `setup.py`.

## Scripts

- `main.py`
  - Orchestrates the pipeline by calling the CLIs: export `A.csv`/`b.csv` (and legacy `M.csv`/`E.csv`), solve to `x.csv`/`L.csv`, then render images from CSV.

- `linear_system.py` (CLI)
  - Uses `setup.py` to build the scene and assemble the form‑factor matrix `F`.
  - Exports canonical `A.csv`/`b.csv`. Also writes legacy `M.csv`/`E.csv` (where `E.csv` may contain multiple columns if multiple scenarios are defined).

- `solve.py` (CLI)
  - Loads `A.csv`/`b.csv` (or legacy `M.csv`/`E.csv`), solves `Ax=b`.
  - Saves `x.csv` (radiosity) and `L.csv` (`L=x/π`). Supports multiple right‑hand sides (columns) in `b.csv`.

- `render.py` (CLI)
  - Loads the scene geometry from `setup.py` and renders images directly from CSV solutions in `linear_system_csv/` (prefers `L.csv`, supports multiple columns).

- `setup.py`
  - Central place to edit all parameters: room size, materials, cube/prisms, light scenarios, camera, render resolution, tone mapping.

## Parameters (setup.py)

- Scene
  - `BoxDimensions(W, D, H)`: room width, depth, height.
  - `Materials`: `rho_floor`, `rho_ceiling`, `rho_walls`, `rho_cube` (diffuse reflectance in [0,1]); `Le_light` (emitted radiance of lights).
  - `CubeParams(size, z0)`: main cube edge length and base height above floor.
  - `RectPrism(x0,x1,y0,y1,z0,z1)`: extra prisms (axis‑aligned) bounds.
  - `SceneConfig`: packs the above plus `light_size` and optional `light_positions` (unused when using scenarios), and `extra_prisms`.

- Lights
  - `light_size=(lx, ly)`: rectangular emitter size on ceiling.
  - `scenarios`: list of dicts per light setup:
    - `name`: scenario name; used in file suffixes and CSV headers.
    - `positions`: list of emitter centers `(x, y)` on ceiling plane.
    - `size`: optional override for `light_size`.
    - `Le`: optional override for `materials.Le_light`.

- Camera & Render
  - `cam_pos`, `cam_look`, `cam_up`: camera position, look‑at point, and world up.
  - `width`, `height`, `fov_y_deg`: image size and vertical FoV (degrees).
  - `exposure`, `brightness`: tone mapping controls. Exposure scales scene radiance before Reinhard; brightness is post‑map gain then clamped to [0,1].

## Outputs

- `outputs/*.png`: rendered images from `main.py` or `render.py`.
- `linear_system_csv/A.csv`, `linear_system_csv/b.csv`: canonical linear system (b may have multiple columns).
- `linear_system_csv/M.csv`, `linear_system_csv/E.csv`: legacy names for compatibility.
- `linear_system_csv/x.csv`, `linear_system_csv/L.csv`: solution radiosity and radiance (may be multi‑column if `b.csv` is multi‑column).

## How It Works (Principles)

- Radiosity with Lambertian surfaces (diffuse BRDF `f_r = ρ/π`). We solve for patch radiosity `B` (power per area) and derive radiance `L = B/π`.
- Galerkin P0 discretization: each surface patch uses a constant basis. The integral equation reduces to a dense linear system `(I - diag(ρ) F) B = E`, where `E = π·Le` on light patches.
- Form factors `Fij` via center‑to‑center approximation with cosine terms and inverse‑square falloff, then enforce reciprocity `Ai Fij = Aj Fji`.
- Visibility: simplified — no geometric occlusion term is computed; only cosine sign checks. This yields smooth energy coupling but may miss shadowing and self‑occlusion.
- Rendering: a CPU raycaster analytically intersects room planes, the main cube, and extra prisms. At the hit point, it maps to the patch grid of that face, looks up the per‑patch `L`, applies Reinhard tone mapping + gamma, and saves the image.

## Repository Structure

```
setup.py            # All parameters (scene, lights, camera, render)
main.py             # Orchestrates: export Ax=b → solve → render from CSV
scene.py            # Geometry, grid generation, masks; extra prisms support
linear_system.py    # Assemble F, reciprocity, export Ax=b (and legacy M/E) (CLI)
solve.py            # Solve Ax=b from CSV, export x/L (CLI)
render.py           # Raycast render from CSV solution (CLI)
requirements.txt    # numpy, matplotlib, tqdm
```

## Notes & Tips

- Performance: `F` is dense O(N²). Increase subdivisions cautiously. A progress bar (tqdm) is used when available.
- Multi‑scenario note: each scenario changes light mask and thus `ρ`, so `M` differs per scenario. `M.csv`/`E.csv` exported by the CLI correspond to the first scenario only.
- Extending lights: current masks target ceiling rectangles; you can add similar helpers for walls/floor if needed.

## License

MIT
