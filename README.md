# Integral Render (Radiosity + Galerkin)

Radiosity (Lambertian) solver for a simple box scene with a cube, using a Galerkin P0 formulation. The pipeline is split into:

1) Scene construction (`scene.py`)
2) Linear system assembly (`linear_system.py`)
3) Solve and render (`render.py`)

## Requirements

- Python 3.9+
- numpy, matplotlib, tqdm (optional)
- macOS/Linux/Windows

Install dependencies (example):

```bash
ython3 -m pip install numpy matplotlib tqdm
```

## Run

```bash
python3 main.py
```

- Output image: `outputs/radiosity_box_galerkin_out.png`
- Light position is configured in `main.py` via `light_positions`.

## Control Image Brightness

`render.render_photo` exposes two parameters (already wired in `main.py`):

- `exposure`: scales scene radiance before tone mapping (like camera exposure)
- `brightness`: post–tone-map multiplier (clamped to [0,1])

Example in `main.py`:

```python
exposure = 10.0
brightness = 2.2
```

## Export Linear System to CSV

`linear_system.py` provides CSV export of the dense system:

- `build_system_matrix(F, rho)` returns `M = I - diag(rho) F`
- `export_system_to_csv(M, E, out_dir)` saves `M.csv` and `E.csv` with headers

This repository ignores the generated folder in Git:

```
linear_system_csv/
```

You can regenerate locally by running `main.py`.

## Repository Structure

```
scene.py           # Scene config, grid generation, Patch class, build_scene()
linear_system.py   # Form-factor assembly, reciprocity, build M, CSV export
render.py          # Build E/rho, solve radiosity, build L_face, render image
main.py            # Example: build → assemble → export CSV → solve → render
outputs/           # Rendered images (ignored in Git)
linear_system_csv/ # CSV exports of M and E (ignored in Git)
```

## Notes

- Form factors use center-to-center approximation and reciprocity enforcement.
- This is a dense O(N^2) assembly; increase subdivisions with care.

## License

MIT
