import numpy as np  # type: ignore
from pathlib import Path
from typing import Dict, Tuple, Optional

from scene import BuiltScene, BoxDimensions

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
    return E, rho


def solve_radiosity(
    F: np.ndarray, rho: np.ndarray, E: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    RF = (rho[:, None]) * F
    M = np.eye(F.shape[0]) - RF
    B = np.linalg.solve(M, E)
    L = B / np.pi
    return B, L


def build_L_face(scene: BuiltScene, L: np.ndarray) -> Dict[str, np.ndarray]:
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

    def face_ij(face: str, p: np.ndarray) -> Tuple[int, int]:
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
            # suffix: last token after '_'
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
                return (0, 0)
        else:
            return (0, 0)
        i = min(nx - 1, max(0, int(u * nx)))
        j = min(ny - 1, max(0, int(v * ny)))
        return (i, j)

    # Compute exposure-adjusted Lmax for tone mapping
    Lmax = max([np.max(L_face[k]) for k in L_face.keys()]) + 1e-8
    Lmax *= exposure

    for ypix in range(height):
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
            i, j = face_ij(face, hp)
            Lhit = L_face[face][j, i] * exposure
            val = Lhit / (Lhit + Lmax)
            val = np.power(np.clip(val, 0.0, 1.0), 1.0 / 2.2)
            val = np.clip(val * brightness, 0.0, 1.0)
            img[ypix, xpix, :] = val
    return img


def save_image(img: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_path), np.clip(img, 0, 1))
