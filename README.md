# Integral Render (Radiosity + Galerkin)

## Overview

This project implements a simple indoor scene renderer based on Radiosity under a Lambertian assumption. Starting from the rendering equation, we use a Galerkin P0 discretization (piecewise constant per patch). The box room (floor, ceiling, four walls) and a cube are tessellated into rectangular patches. A dense form-factor matrix F is assembled with a center-to-center approximation and then corrected with reciprocity (Ai Fij = Aj Fji).

- Linear system: with diffuse reflectance rho, we solve (I - diag(rho) F) B = E, where E = pi * Le is the emission vector and B is the radiosity. The per-patch radiance is L = B / pi.
- Rendering: a simple CPU raycaster with an interior camera analytically intersects the room/cube planes, maps hit points to their patch grids, looks up constant radiance per patch, then applies Reinhard tone mapping and gamma. Exposure and brightness are user-adjustable.

## Run

```bash
python3 main.py
```

- Output images: `outputs/radiosity_box_galerkin_out_<scenario>.png`
- Light scenarios (positions/size/Le) are configured in `main.py` under `scenarios`.

## Control Image Brightness

`render.render_photo` exposes two parameters (wired in `main.py`):

- `exposure`: scales scene radiance before tone mapping (camera-like exposure)
- `brightness`: post–tone-map multiplier (clamped to [0,1])

Example in `main.py`:

```python
exposure = 10.0
brightness = 2.2
```

## Export Linear System to CSV

From `linear_system.py`:

- `build_system_matrix(F, rho)` returns `M = I - diag(rho) F`
- `export_system_to_csv(M, E, out_dir)` writes `M.csv` and `E.csv` with headers

### Multiple Light Scenarios and E-matrix CSV

- Define multiple light setups in `main.py` (list `scenarios`).
- The program solves and renders each scenario and saves all emission vectors side-by-side to `linear_system_csv/E_multi.csv` (rows = patches, columns = scenarios).

Git ignores generated CSV exports:

```
linear_system_csv/
```

Re-run `main.py` to regenerate.

## Repository Structure

```
scene.py           # Scene config, grid generation, Patch, build_scene()
linear_system.py   # Form-factor assembly, reciprocity, build M, CSV export
render.py          # Build E/rho, solve radiosity, build L_face, render image
main.py            # Example: build → assemble → multi-E CSV → solve → render series
outputs/           # Rendered images (ignored in Git)
linear_system_csv/ # CSV exports (M/E, plus E_multi.csv) (ignored in Git)
```

## Notes

- Form factors use a center-to-center approximation with reciprocity enforcement.
- Dense O(N^2) assembly; increase subdivisions with care.

## License

MIT
