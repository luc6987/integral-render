# Integral Render (Radiosity + Galerkin)

## 简介（基于 ppt.tex 要点）

本项目实现了一个基于辐射度（Radiosity）的室内场景渲染示例。以渲染方程为出发点：
\(L_o = L_e + \int_{\Omega^+} f_r L_i (\omega_i\!\cdot\!n)\, d\omega_i\)，
在朗伯（Lambertian）近似下，采用 Galerkin P0（每个面片常值）离散方法，将场景划分为若干矩形面片（地面、天花板、墙面与一个立方体的六个面），构建面片之间的形状因子矩阵 \(F\)。

- 形状因子（Form Factors）：使用中心到中心近似（center-to-center）快速估计，随后施加互易性 \(A_i F_{ij} = A_j F_{ji}\) 做对称化修正。
- 线性系统：在朗伯反射率 \(\rho\) 下，建立 \((I - \mathrm{diag}(\rho) F)\,B = E\)，其中 \(E=\pi L_e\) 为自发光项、\(B\) 为面片的辐射度。解得 \(B\) 后，辐亮度 \(L = B/\pi\)。
- 渲染：用一个简单的 CPU 光线投射器（相机模型+与盒体/立方体的解析相交）在图像平面逐像素取样，查表各面片的常值辐亮度并做色调映射（Reinhard）与 gamma 矫正。README 中的 `exposure` 与 `brightness` 可直接调节图像明暗。

---

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
