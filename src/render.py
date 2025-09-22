import numpy as np  # type: ignore
from pathlib import Path
from typing import Dict, Tuple, Optional

from .scene import BuiltScene, BoxDimensions, build_scene
from setup import (
    scene_config,
    cam_pos,
    cam_look,
    cam_up,
    width,
    height,
    fov_y_deg,
    exposure,
    brightness,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-9


def build_emission_and_reflectance(
    scene: BuiltScene, Le_light: float
) -> Tuple[np.ndarray, np.ndarray]:
    N = scene.centers.shape[0]
    E = np.zeros(N)
    E[scene.is_light] = np.pi * Le_light
    rho = np.where(
        scene.is_light,
        0.0,
        scene.rho,
    )
    print(
        f"[Render] Built E/rho from scene.is_light: light_patches={int(scene.is_light.sum())}, Le={Le_light}"
    )
    return E, rho


def build_emission_and_reflectance_with_mask(
    scene: BuiltScene,
    Le_light: float,
    is_light_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build E and rho given an explicit light mask.

    - E[i] = pi * Le_light for light patches, else 0
    - rho[i] = 0 for light patches, else scene.rho[i]
    """
    N = scene.centers.shape[0]
    if is_light_mask.shape[0] != N:
        raise ValueError("is_light_mask length mismatch with scene patches")
    E = np.zeros(N)
    E[is_light_mask] = np.pi * Le_light
    rho = np.where(is_light_mask, 0.0, scene.rho)
    print(
        f"[Render] Built E/rho with custom mask: light_patches={int(is_light_mask.sum())}, Le={Le_light}"
    )
    return E, rho


def solve_radiosity(
    F: np.ndarray, rho: np.ndarray, E: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    print(f"[Render] Solving radiosity: N={F.shape[0]}")
    RF = (rho[:, None]) * F
    M = np.eye(F.shape[0]) - RF
    B = np.linalg.solve(M, E)
    L = B / np.pi
    print("[Render] Solve complete. Returning B and L.")
    return B, L


def build_L_face(scene: BuiltScene, L: np.ndarray) -> Dict[str, np.ndarray]:
    if scene.basis_type == "P1":
        return build_L_face_p1(scene, L)
    
    # P0 implementation (original)
    def collect_face_map(face_name: str, sub: Tuple[int, int]) -> np.ndarray:
        nx, ny = sub
        V = np.zeros((ny, nx))
        for idx, p in enumerate(scene.patches):
            if p.face == face_name:
                i, j = p.ij
                V[j, i] = L[idx]
        return V

    # Build maps for all faces registered in sub_by_face (handles prisms dynamically)
    L_face: Dict[str, np.ndarray] = {}
    for face_name, sub in scene.sub_by_face.items():
        L_face[face_name] = collect_face_map(face_name, sub)
    print(f"[Render] Built L_face for {len(L_face)} faces.")
    return L_face


def build_L_face_p1(scene: BuiltScene, L: np.ndarray) -> Dict[str, np.ndarray]:
    """Build per-face node-value grids for P1.

    For each face with subdivision (nx, ny), we return an array of shape (ny+1, nx+1)
    where each entry is the radiance value at that P1 node. Rendering will perform
    bilinear sampling at the hit uv.
    """
    def build_face_node_grid(face_name: str, sub: Tuple[int, int]) -> np.ndarray:
        nx, ny = sub
        if nx < 1 or ny < 1:
            return np.zeros((ny + 1, nx + 1))
        face_nodes = [node for node in (scene.nodes or []) if node.face == face_name]
        node_id_grid = -np.ones((ny + 1, nx + 1), dtype=np.int64)
        for node in face_nodes:
            ii, jj = node.ij
            if 0 <= ii <= nx and 0 <= jj <= ny:
                node_id_grid[jj, ii] = node.global_id
        if (node_id_grid < 0).any():
            return np.zeros((ny + 1, nx + 1))
        return L[node_id_grid]

    L_face: Dict[str, np.ndarray] = {}
    for face_name, sub in scene.sub_by_face.items():
        L_face[face_name] = build_face_node_grid(face_name, sub)
    print(f"[Render] Built P1 node grids for {len(L_face)} faces.")
    return L_face


def render_photo(
    scene: BuiltScene,
    box: BoxDimensions,
    L_face: Dict[str, np.ndarray],
    width: int = 800,
    height: int = 600,
    fov_y_deg: float = 40.0,
    cam_pos: np.ndarray = np.array([0, 0, 1.5]),
    cam_look: np.ndarray = np.array([2.5, 3.0, 0.0]),
    cam_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    exposure: float = 1.0,
    brightness: float = 1.0,
) -> np.ndarray:
    W, D, H = box.W, box.D, box.H
    cube_x0 = scene.cube_bounds["cube_x0"]
    cube_x1 = scene.cube_bounds["cube_x1"]
    cube_y0 = scene.cube_bounds["cube_y0"]
    cube_y1 = scene.cube_bounds["cube_y1"]
    cube_z0 = scene.cube_bounds["cube_z0"]
    cube_z1 = scene.cube_bounds["cube_z1"]

    print(
        f"[Render] Rendering image {width}x{height}, fov_y={fov_y_deg}, exposure={exposure}, brightness={brightness}"
    )
    aspect = width / height
    fov_y = np.deg2rad(fov_y_deg)
    forward = cam_look - cam_pos
    forward = forward / (np.linalg.norm(forward) + EPS)
    right = np.cross(forward, cam_up)
    right = right / (np.linalg.norm(right) + EPS)
    up = np.cross(right, forward)
    tan_half_fov = np.tan(fov_y / 2.0)

    img = np.zeros((height, width, 3), dtype=np.float32)

    def intersect_prism(ray_o: np.ndarray, ray_d: np.ndarray, name: str, bounds) -> Tuple[float, Optional[str], Optional[np.ndarray]]:
        bx0, bx1 = bounds.x0, bounds.x1
        by0, by1 = bounds.y0, bounds.y1
        bz0, bz1 = bounds.z0, bounds.z1
        tmin = np.inf
        hit_face = None
        hit_p = None
        # z planes
        if abs(ray_d[2]) > 1e-8:
            t = (bz0 - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = bx0 <= p[0] <= bx1 and by0 <= p[1] <= by1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, f"{name}_bottom", p
            t = (bz1 - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = bx0 <= p[0] <= bx1 and by0 <= p[1] <= by1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, f"{name}_top", p
        # x planes
        if abs(ray_d[0]) > 1e-8:
            t = (bx0 - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = by0 <= p[1] <= by1 and bz0 <= p[2] <= bz1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, f"{name}_x0", p
            t = (bx1 - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = by0 <= p[1] <= by1 and bz0 <= p[2] <= bz1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, f"{name}_x1", p
        # y planes
        if abs(ray_d[1]) > 1e-8:
            t = (by0 - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = bx0 <= p[0] <= bx1 and bz0 <= p[2] <= bz1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, f"{name}_y0", p
            t = (by1 - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = bx0 <= p[0] <= bx1 and bz0 <= p[2] <= bz1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, f"{name}_y1", p
        return tmin, hit_face, hit_p

    def intersect_scene(ray_o: np.ndarray, ray_d: np.ndarray):
        tmin = np.inf
        hit_face = None
        hit_p = None
        # Intersect all extra prisms
        for pb in scene.prisms:
            t, f, p = intersect_prism(ray_o, ray_d, pb.name, pb)
            if f is not None and t < tmin:
                tmin, hit_face, hit_p = t, f, p
        if abs(ray_d[2]) > 1e-8:
            t = (cube_z0 - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = cube_x0 <= p[0] <= cube_x1 and cube_y0 <= p[1] <= cube_y1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "cube_bottom", p
            t = (cube_z1 - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = cube_x0 <= p[0] <= cube_x1 and cube_y0 <= p[1] <= cube_y1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "cube_top", p
        if abs(ray_d[0]) > 1e-8:
            t = (cube_x0 - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = cube_y0 <= p[1] <= cube_y1 and cube_z0 <= p[2] <= cube_z1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "cube_x0", p
            t = (cube_x1 - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = cube_y0 <= p[1] <= cube_y1 and cube_z0 <= p[2] <= cube_z1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "cube_x1", p
        if abs(ray_d[1]) > 1e-8:
            t = (cube_y0 - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = cube_x0 <= p[0] <= cube_x1 and cube_z0 <= p[2] <= cube_z1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "cube_y0", p
            t = (cube_y1 - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = cube_x0 <= p[0] <= cube_x1 and cube_z0 <= p[2] <= cube_z1
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "cube_y1", p
        if abs(ray_d[2]) > 1e-8:
            t = (0.0 - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = 0.0 <= p[0] <= W and 0.0 <= p[1] <= D
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "floor", p
            t = (H - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = 0.0 <= p[0] <= W and 0.0 <= p[1] <= D
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "ceiling", p
        if abs(ray_d[0]) > 1e-8:
            t = (0.0 - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = 0.0 <= p[1] <= D and 0.0 <= p[2] <= H
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "wall_x0", p
            t = (W - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = 0.0 <= p[1] <= D and 0.0 <= p[2] <= H
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "wall_x1", p
        if abs(ray_d[1]) > 1e-8:
            t = (0.0 - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = 0.0 <= p[0] <= W and 0.0 <= p[2] <= H
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "wall_y0", p
            t = (D - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                ok = 0.0 <= p[0] <= W and 0.0 <= p[2] <= H
                if ok and t < tmin:
                    tmin, hit_face, hit_p = t, "wall_y1", p
        return tmin, hit_face, hit_p

    def face_uv(face: str, p: np.ndarray) -> Tuple[float, float, int, int]:
        if face == "floor":
            nx, ny = scene.sub_by_face["floor"]
            u, v = p[0] / W, p[1] / D
        elif face == "ceiling":
            nx, ny = scene.sub_by_face["ceiling"]
            u, v = p[0] / W, p[1] / D
        elif face == "wall_y0":
            nx, ny = scene.sub_by_face["wall_y0"]
            u, v = p[0] / W, p[2] / H
        elif face == "wall_y1":
            nx, ny = scene.sub_by_face["wall_y1"]
            u, v = p[0] / W, p[2] / H
        elif face == "wall_x0":
            nx, ny = scene.sub_by_face["wall_x0"]
            u, v = p[1] / D, p[2] / H
        elif face == "wall_x1":
            nx, ny = scene.sub_by_face["wall_x1"]
            u, v = p[1] / D, p[2] / H
        elif face in ("cube_top", "cube_bottom"):
            nx, ny = scene.sub_by_face["cube_top"]
            u = (p[0] - cube_x0) / (cube_x1 - cube_x0)
            v = (p[1] - cube_y0) / (cube_y1 - cube_y0)
        elif face in ("cube_x0", "cube_x1"):
            nx, ny = scene.sub_by_face["cube_x0"]
            u = (p[1] - cube_y0) / (cube_y1 - cube_y0)
            v = (p[2] - cube_z0) / (cube_z1 - cube_z0)
        elif face in ("cube_y0", "cube_y1"):
            nx, ny = scene.sub_by_face["cube_y0"]
            u = (p[0] - cube_x0) / (cube_x1 - cube_x0)
            v = (p[2] - cube_z0) / (cube_z1 - cube_z0)
        elif face in scene.prism_bounds_map:
            nx, ny = scene.sub_by_face[face]
            pb = scene.prism_bounds_map[face]
            suff = face.rsplit("_", 1)[-1]
            if suff in ("top", "bottom"):
                u = (p[0] - pb.x0) / (pb.x1 - pb.x0)
                v = (p[1] - pb.y0) / (pb.y1 - pb.y0)
            elif suff in ("x0", "x1"):
                u = (p[1] - pb.y0) / (pb.y1 - pb.y0)
                v = (p[2] - pb.z0) / (pb.z1 - pb.z0)
            elif suff in ("y0", "y1"):
                u = (p[0] - pb.x0) / (pb.x1 - pb.x0)
                v = (p[2] - pb.z0) / (pb.z1 - pb.z0)
            else:
                return (0.0, 0.0, 1, 1)
        else:
            return (0.0, 0.0, 1, 1)
        # Clamp uv to [0,1)
        u = max(0.0, min(1.0 - 1e-9, float(u)))
        v = max(0.0, min(1.0 - 1e-9, float(v)))
        return (u, v, nx, ny)

    def sample_L(face: str, hp: np.ndarray) -> float:
        u, v, nx, ny = face_uv(face, hp)
        if scene.basis_type == "P1":
            # Bilinear on node grid (ny+1, nx+1)
            grid = L_face[face]
            # Convert to node indices
            x = u * nx
            y = v * ny
            i0 = min(nx - 1, max(0, int(np.floor(x))))
            j0 = min(ny - 1, max(0, int(np.floor(y))))
            du = x - i0
            dv = y - j0
            L00 = grid[j0 + 0, i0 + 0]
            L10 = grid[j0 + 0, i0 + 1]
            L11 = grid[j0 + 1, i0 + 1]
            L01 = grid[j0 + 1, i0 + 0]
            return (1 - du) * (1 - dv) * L00 + du * (1 - dv) * L10 + du * dv * L11 + (1 - du) * dv * L01
        else:
            # P0: piecewise constant per patch
            i = min(nx - 1, max(0, int(u * nx)))
            j = min(ny - 1, max(0, int(v * ny)))
            return float(L_face[face][j, i])

    # Tone mapping reference
    Lmax = 0.0
    for k, arr in L_face.items():
        Lmax = max(Lmax, float(np.max(arr)))
    Lmax = (Lmax + 1e-8) * exposure

    try:
        from tqdm import tqdm  # type: ignore
        rows = tqdm(range(height), desc="Rendering", leave=False)
    except Exception:
        rows = range(height)
    for ypix in rows:
        py = 1.0 - 2.0 * (ypix + 0.5) / height
        for xpix in range(width):
            px = 2.0 * (xpix + 0.5) / width - 1.0
            dir_cam = (
                forward
                + px * tan_half_fov * aspect * right
                + py * tan_half_fov * up
            )
            dir_cam = dir_cam / (np.linalg.norm(dir_cam) + EPS)
            t, face, hp = intersect_scene(cam_pos, dir_cam)
            if face is None or not np.isfinite(t):
                continue
            Lhit = sample_L(face, hp) * exposure
            val = Lhit / (Lhit + Lmax)
            val = np.power(np.clip(val, 0.0, 1.0), 1.0 / 2.2)
            val = np.clip(val * brightness, 0.0, 1.0)
            img[ypix, xpix, :] = val
    return img


def save_image(img: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_path), np.clip(img, 0, 1))
    print(f"[Render] Saved image to {out_path}")


def _load_csv_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


"""Legacy multi-header parsing removed."""


def main_from_csv():
    """Render images by loading radiosity/radiance from CSV.

    Reads L.csv (single or multi-column), otherwise x.csv (converted to L).
    """
    csv_dir = PROJECT_ROOT / "linear_system_csv"
    out_dir = PROJECT_ROOT / "outputs"
    scene = build_scene(scene_config)

    # Files
    L_path = csv_dir / "L.csv"
    x_path = csv_dir / "x.csv"

    def render_columns(Lmat: np.ndarray):
        cols = 1 if Lmat.ndim == 1 else Lmat.shape[1]
        for k in range(cols):
            Lvec = Lmat if cols == 1 else Lmat[:, k]
            L_face = build_L_face(scene, Lvec)
            img = render_photo(
                scene=scene,
                box=scene_config.box,
                L_face=L_face,
                width=width,
                height=height,
                fov_y_deg=fov_y_deg,
                cam_pos=cam_pos,
                cam_look=cam_look,
                cam_up=cam_up,
                exposure=exposure,
                brightness=brightness,
            )
            name = f"col{k}"
            save_image(img, out_dir / f"render_from_csv_{name}.png")

    if L_path.exists():
        print(f"[Render:CLI] Loading L from {L_path}")
        Lm = _load_csv_matrix(L_path)
        render_columns(Lm)
        return
    if x_path.exists():
        print(f"[Render:CLI] Loading x from {x_path}")
        Xm = _load_csv_matrix(x_path)
        Lm = Xm / np.pi
        render_columns(Lm)
        return
    raise FileNotFoundError("No L.csv or x.csv found in linear_system_csv")


if __name__ == "__main__":
    main_from_csv()
