# Radiosity + Galerkin (P0) for a single box room, then render a photo from an interior camera.
# Dependencies: numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import time
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# -------------------------
# Parameters
# -------------------------
W, D, H = 5.0, 5.0, 3.0  # Room dimensions x, y, z
rho_floor = 0.3
rho_ceiling = 0.0
rho_walls = 0.0
Le_light = 10.0  # Augmentation de l'intensité lumineuse pour améliorer la luminosité globale
light_size = (1.0, 1.0)  # Light panel size on ceiling (centered)


# Subdivisions (nx, ny) per face (increase for finer result; O(N^2) assembly)
rhodiv = 10 # the density factor for the patches

# the
sub_floor = (rhodiv*int(W), rhodiv*int(D))
# the subdivisions for ceiling
sub_ceiling = (int(W), int(D))
#the subdivisions for wall x=0
sub_wall_x0 = (1, 1)
#the subdivisions for wall x=W
sub_wall_x1 = (1, 1)
#the subdivisions for wall y=0
sub_wall_y0 = (1, 1)
#the subdivisions for wall y=D
sub_wall_y1 = (1, 1)
# Cube parameters (1 m^3) centered on floor,
# slight z offset to avoid coplanar issues
cube_size = 1.0
rho_cube = 0.999
cube_z0 = 1e-3
cube_center = np.array([W / 2.0, D / 2.0, cube_z0 + cube_size / 2.0])
sub_cube = (rhodiv, rhodiv)
# Precompute cube bounds
cube_x0 = cube_center[0] - cube_size / 2.0
cube_x1 = cube_center[0] + cube_size / 2.0
cube_y0 = cube_center[1] - cube_size / 2.0
cube_y1 = cube_center[1] + cube_size / 2.0
cube_z1 = cube_z0 + cube_size

ROW_NORMALIZE_TARGET = 0.999
EPS = 1e-9

# Camera parameters for rendering
cam_pos = np.array([0, 0, H/2.0])
cam_look = np.array([W, D, H/3.0])-[0, 0, H/2.0]  # look at center, slightly down
# -------------------------
# Geometry (patch list)
# -------------------------
class Patch:
    __slots__ = ("center", "normal", "area", "rho", "is_light", "face", "ij")

    def __init__(self, center, normal, area, rho, is_light, face, ij):
        self.center = center
        self.normal = normal
        self.area = area
        self.rho = rho
        self.is_light = is_light
        self.face = face
        self.ij = ij


def make_grid_on_plane(origin, ux, uy, nx, ny, rho, normal, face,
                       light_mask=None):
    patches = []
    dx = 1.0 / nx
    dy = 1.0 / ny
    cell_area = np.linalg.norm(np.cross(ux * dx, uy * dy))
    n = normal / (np.linalg.norm(normal) + EPS)
    print(f"  - 生成面 {face} 网格 {nx}x{ny}")
    for i in tqdm(range(nx), desc=f"{face} 行", leave=False):
        for j in range(ny):
            c = origin + ux * ((i + 0.5) * dx) + uy * ((j + 0.5) * dy)
            is_light = light_mask(i, j) if light_mask is not None else False
            patches.append(Patch(c, n, cell_area, rho, is_light, face, (i, j)))
    return patches


print("[1/7] 构建几何和网格...")
t_geom0 = time.perf_counter()
patches = []

# Floor z=0, normal +Z
origin_floor = np.array([0.0, 0.0, 0.0])
ux_floor = np.array([W, 0.0, 0.0])
uy_floor = np.array([0.0, D, 0.0])
patches += make_grid_on_plane(
    origin_floor,
    ux_floor,
    uy_floor,
    *sub_floor,
    rho_floor,
    np.array([0, 0, 1]),
    face="floor",
)

# Ceiling z=H, normal -Z; center light rectangle
origin_ceil = np.array([0.0, 0.0, H])
ux_ceil = np.array([W, 0.0, 0.0])
uy_ceil = np.array([0.0, D, 0.0])


def mk_multi_ceiling_light_mask(sub, W, D, light_size, light_pos_list):
    """Retourne True si la cellule (i,j) appartient à AU MOINS un rectangle lumineux.
    Fusion de toutes les sources plafond en un seul masque."""
    nx, ny = sub
    lx, ly = light_size
    rects = []
    for cx, cy in light_pos_list:
        rects.append((cx - lx / 2.0, cx + lx / 2.0, cy - ly / 2.0, cy + ly / 2.0))

    def mask(i, j):
        u = (i + 0.5) / nx
        v = (j + 0.5) / ny
        x = u * W
        y = v * D
        for x0, x1, y0, y1 in rects:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False
    return mask

# Génération unique du plafond avec toutes les sources
light_mask_fn = mk_multi_ceiling_light_mask(sub_ceiling, W, D, light_size, light_pos)
patches += make_grid_on_plane(
    origin_ceil,
    ux_ceil,
    uy_ceil,
    *sub_ceiling,
    rho_ceiling,
    np.array([0, 0, -1]),
    face="ceiling",
    light_mask=light_mask_fn,
)

# Walls
# y=0 wall, normal +Y
origin_y0 = np.array([0.0, 0.0, 0.0])
ux_y0 = np.array([W, 0.0, 0.0])
uy_y0 = np.array([0.0, 0.0, H])
patches += make_grid_on_plane(
    origin_y0,
    ux_y0,
    uy_y0,
    *sub_wall_y0,
    rho_walls,
    np.array([0, 1, 0]),
    face="wall_y0",
)

# y=D wall, normal -Y
origin_y1 = np.array([0.0, D, 0.0])
ux_y1 = np.array([W, 0.0, 0.0])
uy_y1 = np.array([0.0, 0.0, H])
patches += make_grid_on_plane(
    origin_y1,
    ux_y1,
    uy_y1,
    *sub_wall_y1,
    rho_walls,
    np.array([0, -1, 0]),
    face="wall_y1",
)

# x=0 wall, normal +X
origin_x0 = np.array([0.0, 0.0, 0.0])
ux_x0 = np.array([0.0, D, 0.0])
uy_x0 = np.array([0.0, 0.0, H])
patches += make_grid_on_plane(
    origin_x0,
    ux_x0,
    uy_x0,
    *sub_wall_x0,
    rho_walls,
    np.array([1, 0, 0]),
    face="wall_x0",
)

# x=W wall, normal -X
origin_x1 = np.array([W, 0.0, 0.0])
ux_x1 = np.array([0.0, D, 0.0])
uy_x1 = np.array([0.0, 0.0, H])
patches += make_grid_on_plane(
    origin_x1,
    ux_x1,
    uy_x1,
    *sub_wall_x1,
    rho_walls,
    np.array([-1, 0, 0]),
    face="wall_x1",
)

# Cube faces (six faces)
# Top (z=cube_z1), normal +Z
origin_ctop = np.array([cube_x0, cube_y0, cube_z1])
ux_ctop = np.array([cube_x1 - cube_x0, 0.0, 0.0])
uy_ctop = np.array([0.0, cube_y1 - cube_y0, 0.0])
patches += make_grid_on_plane(
    origin_ctop,
    ux_ctop,
    uy_ctop,
    *sub_cube,
    rho_cube,
    np.array([0, 0, 1]),
    face="cube_top",
)
# Bottom (z=cube_z0), normal -Z
origin_cbot = np.array([cube_x0, cube_y0, cube_z0])
ux_cbot = np.array([cube_x1 - cube_x0, 0.0, 0.0])
uy_cbot = np.array([0.0, cube_y1 - cube_y0, 0.0])
patches += make_grid_on_plane(
    origin_cbot,
    ux_cbot,
    uy_cbot,
    *sub_cube,
    rho_cube,
    np.array([0, 0, -1]),
    face="cube_bottom",
)
# x=cube_x0, normal -X
origin_cx0 = np.array([cube_x0, cube_y0, cube_z0])
ux_cx0 = np.array([0.0, cube_y1 - cube_y0, 0.0])
uy_cx0 = np.array([0.0, 0.0, cube_z1 - cube_z0])
patches += make_grid_on_plane(
    origin_cx0,
    ux_cx0,
    uy_cx0,
    *sub_cube,
    rho_cube,
    np.array([-1, 0, 0]),
    face="cube_x0",
)
# x=cube_x1, normal +X
origin_cx1 = np.array([cube_x1, cube_y0, cube_z0])
ux_cx1 = np.array([0.0, cube_y1 - cube_y0, 0.0])
uy_cx1 = np.array([0.0, 0.0, cube_z1 - cube_z0])
patches += make_grid_on_plane(
    origin_cx1,
    ux_cx1,
    uy_cx1,
    *sub_cube,
    rho_cube,
    np.array([1, 0, 0]),
    face="cube_x1",
)
# y=cube_y0, normal -Y
origin_cy0 = np.array([cube_x0, cube_y0, cube_z0])
ux_cy0 = np.array([cube_x1 - cube_x0, 0.0, 0.0])
uy_cy0 = np.array([0.0, 0.0, cube_z1 - cube_z0])
patches += make_grid_on_plane(
    origin_cy0,
    ux_cy0,
    uy_cy0,
    *sub_cube,
    rho_cube,
    np.array([0, -1, 0]),
    face="cube_y0",
)
# y=cube_y1, normal +Y
origin_cy1 = np.array([cube_x0, cube_y1, cube_z0])
ux_cy1 = np.array([cube_x1 - cube_x0, 0.0, 0.0])
uy_cy1 = np.array([0.0, 0.0, cube_z1 - cube_z0])
patches += make_grid_on_plane(
    origin_cy1,
    ux_cy1,
    uy_cy1,
    *sub_cube,
    rho_cube,
    np.array([0, 1, 0]),
    face="cube_y1",
)

t_geom1 = time.perf_counter()
print(f"  几何构建完成，用时 {t_geom1 - t_geom0:.2f}s")

N = len(patches)
centers = np.array([p.center for p in patches])
normals = np.array([p.normal for p in patches])
areas = np.array([p.area for p in patches])
rho = np.array([p.rho for p in patches])
is_light = np.array([p.is_light for p in patches])
print(f"  patch 总数: {N}")

# Emission term: E = π * Le_light sur les patches lumineux
E = np.zeros(N)
E[is_light] = np.pi * Le_light
# Les patches émetteurs ne réfléchissent pas
rho = np.where(is_light, 0.0, rho)
print(f"  发光 patch 数量: {np.count_nonzero(is_light)}")

# -------------------------
# Form-factor matrix F (center-to-center approx), reciprocity fix,
# row normalization
# -------------------------
print("[2/7] 组装形状因子矩阵 F ... (O(N^2))")
t_F0 = time.perf_counter()
F = np.zeros((N, N), dtype=np.float64)

for i in tqdm(range(N), desc="组装F"):
    ci = centers[i]
    ni = normals[i]
    v = centers - ci  # vectors to all j
    r2 = np.sum(v * v, axis=1) + EPS
    r = np.sqrt(r2)
    wi = v / r[:, None]  # directions i->j
    cos_i = wi @ ni  # dot(wi, ni)
    cos_j = -np.einsum("ij,ij->i", wi, normals)  # dot(-wi, nj)
    vis = (cos_i > 0) & (cos_j > 0)
    vis[i] = False
    contrib = np.zeros(N)
    contrib[vis] = (cos_i[vis] * cos_j[vis]) / (np.pi * r2[vis]) * areas[vis]
    F[i, :] = contrib

t_F1 = time.perf_counter()
print(f"  F 组装完成，用时 {t_F1 - t_F0:.2f}s")
    contrib[vis] = (cos_i[vis] * cos_j[vis]) / (np.pi * r2[vis]) * areas[vis]
    F[i, :] = contrib

t_F1 = time.perf_counter()
print(f"  F 组装完成，用时 {t_F1 - t_F0:.2f}s")

print("[3/7] 应用互易性 Ai F_ij = Aj F_ji ...")
t_R0 = time.perf_counter()
A = areas
for i in tqdm(range(N), desc="互易性"):
    for j in range(i + 1, N):
        Tij = 0.5 * (A[i] * F[i, j] + A[j] * F[j, i])
        F[i, j] = Tij / (A[i] + EPS)
        F[j, i] = Tij / (A[j] + EPS)
np.fill_diagonal(F, 0.0)
t_R1 = time.perf_counter()
print(f"  互易性完成，用时 {t_R1 - t_R0:.2f}s")

print("[5/7] 求解线性系统 (I - diag(rho) F) B = E ...")
t_S0 = time.perf_counter()
RF = (rho[:, None]) * F
M = np.eye(N) - RF
print(f"  维度: {N}  稀疏度: {np.count_nonzero(F) / (N * N):.6f}")
# Résolution directe
B = np.linalg.solve(M, E)
L = B / np.pi

t_S1 = time.perf_counter()
print(f"  求解完成，用时 {t_S1 - t_S0:.2f}s")

# -------------------------
# Prepare per-face L maps for fast lookup during rendering
# -------------------------
print("[6/7] 准备每个面的辐亮度贴图 L_face ...")
t_P0 = time.perf_counter()


def collect_face_map(face_name, sub):
    nx, ny = sub
    V = np.zeros((ny, nx))
    for idx, p in enumerate(patches):
        if p.face == face_name:
            i, j = p.ij
            V[j, i] = L[idx]
    return V


L_face = {
    "floor": collect_face_map("floor", sub_floor),
    "ceiling": collect_face_map("ceiling", sub_ceiling),
    "wall_x0": collect_face_map("wall_x0", sub_wall_x0),
    "wall_x1": collect_face_map("wall_x1", sub_wall_x1),
    "wall_y0": collect_face_map("wall_y0", sub_wall_y0),
    "wall_y1": collect_face_map("wall_y1", sub_wall_y1),
    # Cube faces
    "cube_top": collect_face_map("cube_top", sub_cube),
    "cube_bottom": collect_face_map("cube_bottom", sub_cube),
    "cube_x0": collect_face_map("cube_x0", sub_cube),
    "cube_x1": collect_face_map("cube_x1", sub_cube),
    "cube_y0": collect_face_map("cube_y0", sub_cube),
    "cube_y1": collect_face_map("cube_y1", sub_cube),
}

t_P1 = time.perf_counter()
print(f"  L_face 准备完成，用时 {t_P1 - t_P0:.2f}s")

# Store grid sizes for mapping world point -> (i,j)
sub_by_face = {
    "floor": sub_floor,
    "ceiling": sub_ceiling,
    "wall_x0": sub_wall_x0,
    "wall_x1": sub_wall_x1,
    "wall_y0": sub_wall_y0,
    "wall_y1": sub_wall_y1,
    # Cube
    "cube_top": sub_cube,
    "cube_bottom": sub_cube,
    "cube_x0": sub_cube,
    "cube_x1": sub_cube,
    "cube_y0": sub_cube,
    "cube_y1": sub_cube,
}


# -------------------------
# Simple CPU raycaster to "render a photo" from inside the box
# -------------------------

def render_photo(
    width=800,
    height=600,
    fov_y_deg=60.0,
    cam_pos=np.array([0, 0, 1.5]),
    cam_look=np.array([2.5, 3.0, 0]),
    cam_up=np.array([0.0, 0.0, 1.0]),
):
    print("[7/7] 渲染图像...")
    t_Rend0 = time.perf_counter()
    aspect = width / height
    fov_y = np.deg2rad(fov_y_deg)
    # Camera basis
    forward = cam_look - cam_pos
    forward = forward / (np.linalg.norm(forward) + EPS)
    right = np.cross(forward, cam_up)
    right = right / (np.linalg.norm(right) + EPS)
    up = np.cross(right, forward)

    # Precompute scale
    tan_half_fov = np.tan(fov_y / 2.0)
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Utility: intersect ray with planes of the box and cube,
    # return nearest hit (t, face, point)
    def intersect_box(ray_o, ray_d):
        tmin = np.inf
        hit_face = None
        hit_p = None

        # First, intersect cube faces (prioritize cube to avoid
        # floor/cube coplanar ambiguity)
        if abs(ray_d[2]) > 1e-8:
            # cube bottom z=cube_z0
            t = (cube_z0 - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if cube_x0 <= p[0] <= cube_x1 and cube_y0 <= p[1] <= cube_y1:
                    if t < tmin:
                        tmin = t
                        hit_face = "cube_bottom"
                        hit_p = p
            # cube top z=cube_z1
            t = (cube_z1 - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if cube_x0 <= p[0] <= cube_x1 and cube_y0 <= p[1] <= cube_y1:
                    if t < tmin:
                        tmin = t
                        hit_face = "cube_top"
                        hit_p = p
        if abs(ray_d[0]) > 1e-8:
            # cube x=cube_x0
            t = (cube_x0 - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if cube_y0 <= p[1] <= cube_y1 and cube_z0 <= p[2] <= cube_z1:
                    if t < tmin:
                        tmin = t
                        hit_face = "cube_x0"
                        hit_p = p
            # cube x=cube_x1
            t = (cube_x1 - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if cube_y0 <= p[1] <= cube_y1 and cube_z0 <= p[2] <= cube_z1:
                    if t < tmin:
                        tmin = t
                        hit_face = "cube_x1"
                        hit_p = p
        if abs(ray_d[1]) > 1e-8:
            # cube y=cube_y0
            t = (cube_y0 - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if cube_x0 <= p[0] <= cube_x1 and cube_z0 <= p[2] <= cube_z1:
                    if t < tmin:
                        tmin = t
                        hit_face = "cube_y0"
                        hit_p = p
            # cube y=cube_y1
            t = (cube_y1 - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if cube_x0 <= p[0] <= cube_x1 and cube_z0 <= p[2] <= cube_z1:
                    if t < tmin:
                        tmin = t
                        hit_face = "cube_y1"
                        hit_p = p

        # Then, intersect room faces
        # z=0 (floor), normal +Z
        if abs(ray_d[2]) > 1e-8:
            t = (0.0 - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if 0.0 <= p[0] <= W and 0.0 <= p[1] <= D:
                    if t < tmin:
                        tmin = t
                        hit_face = "floor"
                        hit_p = p

        # z=H (ceiling), normal -Z
        if abs(ray_d[2]) > 1e-8:
            t = (H - ray_o[2]) / ray_d[2]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if 0.0 <= p[0] <= W and 0.0 <= p[1] <= D:
                    if t < tmin:
                        tmin = t
                        hit_face = "ceiling"
                        hit_p = p

        # x=0 (wall_x0), normal +X
        if abs(ray_d[0]) > 1e-8:
            t = (0.0 - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if 0.0 <= p[1] <= D and 0.0 <= p[2] <= H:
                    if t < tmin:
                        tmin = t
                        hit_face = "wall_x0"
                        hit_p = p

        # x=W (wall_x1), normal -X
        if abs(ray_d[0]) > 1e-8:
            t = (W - ray_o[0]) / ray_d[0]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if 0.0 <= p[1] <= D and 0.0 <= p[2] <= H:
                    if t < tmin:
                        tmin = t
                        hit_face = "wall_x1"
                        hit_p = p

        # y=0 (wall_y0), normal +Y
        if abs(ray_d[1]) > 1e-8:
            t = (0.0 - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if 0.0 <= p[0] <= W and 0.0 <= p[2] <= H:
                    if t < tmin:
                        tmin = t
                        hit_face = "wall_y0"
                        hit_p = p

        # y=D (wall_y1), normal -Y
        if abs(ray_d[1]) > 1e-8:
            t = (D - ray_o[1]) / ray_d[1]
            if t > 1e-6:
                p = ray_o + t * ray_d
                if 0.0 <= p[0] <= W and 0.0 <= p[2] <= H:
                    if t < tmin:
                        tmin = t
                        hit_face = "wall_y1"
                        hit_p = p

        return tmin, hit_face, hit_p

    # Map world hit point to face grid index
    def face_ij(face, p):
        if face == "floor":
            nx, ny = sub_floor
            x, y = p[0], p[1]
            u, v = x / W, y / D
        elif face == "ceiling":
            nx, ny = sub_ceiling
            x, y = p[0], p[1]
            u, v = x / W, y / D
        elif face == "wall_y0":
            nx, ny = sub_wall_y0
            x, z = p[0], p[2]
            u, v = x / W, z / H
        elif face == "wall_y1":
            nx, ny = sub_wall_y1
            x, z = p[0], p[2]
            u, v = x / W, z / H
        elif face == "wall_x0":
            nx, ny = sub_wall_x0
            y, z = p[1], p[2]
            u, v = y / D, z / H
        elif face == "wall_x1":
            nx, ny = sub_wall_x1
            y, z = p[1], p[2]
            u, v = y / D, z / H
        elif face == "cube_top":
            nx, ny = sub_cube
            x, y = p[0], p[1]
            u = (x - cube_x0) / (cube_x1 - cube_x0)
            v = (y - cube_y0) / (cube_y1 - cube_y0)
        elif face == "cube_bottom":
            nx, ny = sub_cube
            x, y = p[0], p[1]
            u = (x - cube_x0) / (cube_x1 - cube_x0)
            v = (y - cube_y0) / (cube_y1 - cube_y0)
        elif face == "cube_x0":
            nx, ny = sub_cube
            y, z = p[1], p[2]
            u = (y - cube_y0) / (cube_y1 - cube_y0)
            v = (z - cube_z0) / (cube_z1 - cube_z0)
        elif face == "cube_x1":
            nx, ny = sub_cube
            y, z = p[1], p[2]
            u = (y - cube_y0) / (cube_y1 - cube_y0)
            v = (z - cube_z0) / (cube_z1 - cube_z0)
        elif face == "cube_y0":
            nx, ny = sub_cube
            x, z = p[0], p[2]
            u = (x - cube_x0) / (cube_x1 - cube_x0)
            v = (z - cube_z0) / (cube_z1 - cube_z0)
        elif face == "cube_y1":
            nx, ny = sub_cube
            x, z = p[0], p[2]
            u = (x - cube_x0) / (cube_x1 - cube_x0)
            v = (z - cube_z0) / (cube_z1 - cube_z0)
        else:
            return None
        # Clamp to valid cell
        i = min(nx - 1, max(0, int(u * nx)))
        j = min(ny - 1, max(0, int(v * ny)))
        return (i, j)

    # Max radiance for exposure
    Lmax = max([np.max(L_face[k]) for k in L_face.keys()])
    print(f"  Lmax = {Lmax:.6f}")

    for ypix in tqdm(range(height), desc="渲染扫描行"):
        # NDC y in [-1,1], top->+1
        py = 1.0 - 2.0 * (ypix + 0.5) / height
        for xpix in range(width):
            px = 2.0 * (xpix + 0.5) / width - 1.0
            # Direction in camera space
            dir_cam = (
                forward
                + px * tan_half_fov * aspect * right
                + py * tan_half_fov * up
            )
            dir_cam = dir_cam / (np.linalg.norm(dir_cam) + EPS)
            # Intersect
            t, face, hp = intersect_box(cam_pos, dir_cam)
            if face is None or not np.isfinite(t):
                continue
            ij = face_ij(face, hp)
            if ij is None:
                continue
            i, j = ij
            Lhit = L_face[face][j, i]  # Lambertian: constant radiance
            # Simple tonemapping: exposure to map Lmax to ~1.0, then gamma
            # 曝光调整：允许更高的亮度范围

            # 如果需要，可以禁用色调映射以保持原始亮度
            # val = Lhit * exposure  # 直接应用曝光，而不是色调映射
            val = Lhit / (Lhit + Lmax) * 50  # Reinhard tone map
            # Reinhard tone map (grayscale here)

            # gamma to sRGB-ish
            val = np.power(np.clip(val, 0.0, 1.0), 1.0 / 2.2)
            img[ypix, xpix, :] = val

    t_Rend1 = time.perf_counter()
    print(f"  渲染完成，用时 {t_Rend1 - t_Rend0:.2f}s")
    return img


img = render_photo(
    width=800,
    height=600,
    fov_y_deg=40.0,
    cam_pos=cam_pos,
    cam_look=cam_look,
    #cam_up= (look x z)xlook
    cam_up=np.cross(np.cross(cam_look, np.array([0, 0, 1])), cam_look)
    )

# Save and show (robust local path)
out_dir = Path(__file__).parent / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "radiosity_box_galerkin_out.png"
plt.imsave(str(out_path), np.clip(img, 0, 1))
print(f"已保存图像: {out_path}")
out_path

# =========================
# Ajout : rendu batch pour plusieurs configurations de light_pos
# =========================

def _make_ceiling_light_mask_factory(light_pos_list, light_size, W, D, sub_ceiling):
    """Retourne une fonction mask(i,j) qui est True si la cellule (i,j) appartient
    à AU MOINS un rectangle lumineux défini par son centre dans light_pos_list."""
    nx, ny = sub_ceiling
    lx, ly = light_size
    # Pré-calcul des bornes en coordonnées monde pour chaque source
    rects = []
    for cx, cy in light_pos_list:
        x0 = cx - lx / 2.0
        x1 = cx + lx / 2.0
        y0 = cy - ly / 2.0
        y1 = cy + ly / 2.0
        rects.append((x0, x1, y0, y1))

    def mask(i, j):
        # centre de la case (i,j) sur le plafond
        u = (i + 0.5) / nx
        v = (j + 0.5) / ny
        x = u * W
        y = v * D
        for (x0, x1, y0, y1) in rects:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    return mask


def compute_image_for_light_pos(light_pos_list, output_path, width=800, height=600, fov_y_deg=40.0,
                                cam_pos=np.array([0, 0, H/2.0]),
                                cam_look=np.array([W, D, H/3.0]) - np.array([0, 0, H/2.0]),
                                cam_up=None,
                                light_size=(1.0, 1.0), Le_light=10.0):
    """Construit toute la scène pour une configuration de sources lumineuses (centres sur le plafond)
    et enregistre une image. Retourne le chemin du fichier écrit."""
    if cam_up is None:
        cam_up = np.cross(np.cross(cam_look, np.array([0, 0, 1])), cam_look)

    # ---- Construction géométrie locale ----
    local_patches = []
    # Réutilisation de la classe Patch et de make_grid_on_plane existants
    # Sol :
    origin_floor = np.array([0.0, 0.0, 0.0])
    ux_floor = np.array([W, 0.0, 0.0])
    uy_floor = np.array([0.0, D, 0.0])
    local_patches += make_grid_on_plane(origin_floor, ux_floor, uy_floor, *sub_floor, rho_floor,
                                        np.array([0, 0, 1]), face="floor")

    # Plafond :
    origin_ceil = np.array([0.0, 0.0, H])
    ux_ceil = np.array([W, 0.0, 0.0])
    uy_ceil = np.array([0.0, D, 0.0])
    light_mask = _make_ceiling_light_mask_factory(light_pos_list, light_size, W, D, sub_ceiling)
    local_patches += make_grid_on_plane(origin_ceil, ux_ceil, uy_ceil, *sub_ceiling, rho_ceiling,
                                        np.array([0, 0, -1]), face="ceiling", light_mask=light_mask)

    # Murs (mêmes subdivisions que global)
    origin_y0 = np.array([0.0, 0.0, 0.0])
    ux_y0 = np.array([W, 0.0, 0.0])
    uy_y0 = np.array([0.0, 0.0, H])
    local_patches += make_grid_on_plane(origin_y0, ux_y0, uy_y0, *sub_wall_y0, rho_walls, np.array([0, 1, 0]), face="wall_y0")

    origin_y1 = np.array([0.0, D, 0.0])
    ux_y1 = np.array([W, 0.0, 0.0])
    uy_y1 = np.array([0.0, 0.0, H])
    local_patches += make_grid_on_plane(origin_y1, ux_y1, uy_y1, *sub_wall_y1, rho_walls, np.array([0, -1, 0]), face="wall_y1")

    origin_x0 = np.array([0.0, 0.0, 0.0])
    ux_x0 = np.array([0.0, D, 0.0])
    uy_x0 = np.array([0.0, 0.0, H])
    local_patches += make_grid_on_plane(origin_x0, ux_x0, uy_x0, *sub_wall_x0, rho_walls, np.array([1, 0, 0]), face="wall_x0")

    origin_x1 = np.array([W, 0.0, 0.0])
    ux_x1 = np.array([0.0, D, 0.0])
    uy_x1 = np.array([0.0, 0.0, H])
    local_patches += make_grid_on_plane(origin_x1, ux_x1, uy_x1, *sub_wall_x1, rho_walls, np.array([-1, 0, 0]), face="wall_x1")

    # Cube
    origin_ctop = np.array([cube_x0, cube_y0, cube_z1])
    ux_ctop = np.array([cube_x1 - cube_x0, 0.0, 0.0])
    uy_ctop = np.array([0.0, cube_y1 - cube_y0, 0.0])
    local_patches += make_grid_on_plane(origin_ctop, ux_ctop, uy_ctop, *sub_cube, rho_cube, np.array([0, 0, 1]), face="cube_top")

    origin_cbot = np.array([cube_x0, cube_y0, cube_z0])
    ux_cbot = np.array([cube_x1 - cube_x0, 0.0, 0.0])
    uy_cbot = np.array([0.0, cube_y1 - cube_y0, 0.0])
    local_patches += make_grid_on_plane(origin_cbot, ux_cbot, uy_cbot, *sub_cube, rho_cube, np.array([0, 0, -1]), face="cube_bottom")

    origin_cx0 = np.array([cube_x0, cube_y0, cube_z0])
    ux_cx0 = np.array([0.0, cube_y1 - cube_y0, 0.0])
    uy_cx0 = np.array([0.0, 0.0, cube_z1 - cube_z0])
    local_patches += make_grid_on_plane(origin_cx0, ux_cx0, uy_cx0, *sub_cube, rho_cube, np.array([-1, 0, 0]), face="cube_x0")

    origin_cx1 = np.array([cube_x1, cube_y0, cube_z0])
    ux_cx1 = np.array([0.0, cube_y1 - cube_y0, 0.0])
    uy_cx1 = np.array([0.0, 0.0, cube_z1 - cube_z0])
    local_patches += make_grid_on_plane(origin_cx1, ux_cx1, uy_cx1, *sub_cube, rho_cube, np.array([1, 0, 0]), face="cube_x1")

    origin_cy0 = np.array([cube_x0, cube_y0, cube_z0])
    ux_cy0 = np.array([cube_x1 - cube_x0, 0.0, 0.0])
    uy_cy0 = np.array([0.0, 0.0, cube_z1 - cube_z0])
    local_patches += make_grid_on_plane(origin_cy0, ux_cy0, uy_cy0, *sub_cube, rho_cube, np.array([0, -1, 0]), face="cube_y0")

    origin_cy1 = np.array([cube_x0, cube_y1, cube_z0])
    ux_cy1 = np.array([cube_x1 - cube_x0, 0.0, 0.0])
    uy_cy1 = np.array([0.0, 0.0, cube_z1 - cube_z0])
    local_patches += make_grid_on_plane(origin_cy1, ux_cy1, uy_cy1, *sub_cube, rho_cube, np.array([0, 1, 0]), face="cube_y1")

    Nloc = len(local_patches)
    centers = np.array([p.center for p in local_patches])
    normals = np.array([p.normal for p in local_patches])
    areas = np.array([p.area for p in local_patches])
    rho_arr = np.array([p.rho for p in local_patches])
    is_light_arr = np.array([p.is_light for p in local_patches])

    # Emission
    E = np.zeros(Nloc)
    E[is_light_arr] = np.pi * Le_light  # radiosité émise = π * Le
    rho_arr = np.where(is_light_arr, 0.0, rho_arr)

    # Form factors (centre-à-centre)
    F = np.zeros((Nloc, Nloc), dtype=np.float64)
    for i in range(Nloc):
        ci = centers[i]
        ni = normals[i]
        v = centers - ci
        r2 = np.sum(v * v, axis=1) + EPS
        r = np.sqrt(r2)
        wi = v / r[:, None]
        cos_i = wi @ ni
        cos_j = -np.einsum("ij,ij->i", wi, normals)
        vis = (cos_i > 0) & (cos_j > 0)
        vis[i] = False
        contrib = np.zeros(Nloc)
        contrib[vis] = (cos_i[vis] * cos_j[vis]) / (np.pi * r2[vis]) * areas[vis]
        F[i, :] = contrib

    # Réciprocité
    for i in range(Nloc):
        for j in range(i + 1, Nloc):
            Tij = 0.5 * (areas[i] * F[i, j] + areas[j] * F[j, i])
            F[i, j] = Tij / (areas[i] + EPS)
            F[j, i] = Tij / (areas[j] + EPS)
    np.fill_diagonal(F, 0.0)

    # Résolution (I - rho F) B = E
    M = np.eye(Nloc) - (rho_arr[:, None] * F)
    B = np.linalg.solve(M, E)
    L = B / np.pi  # radiance lambertienne constante

    # Construction des cartes L_face
    def collect_face_map(face_name, sub):
        nx, ny = sub
        V = np.zeros((ny, nx))
        for idx, p in enumerate(local_patches):
            if p.face == face_name:
                i, j = p.ij
                V[j, i] = L[idx]
        return V

    L_face_local = {
        "floor": collect_face_map("floor", sub_floor),
        "ceiling": collect_face_map("ceiling", sub_ceiling),
        "wall_x0": collect_face_map("wall_x0", sub_wall_x0),
        "wall_x1": collect_face_map("wall_x1", sub_wall_x1),
        "wall_y0": collect_face_map("wall_y0", sub_wall_y0),
        "wall_y1": collect_face_map("wall_y1", sub_wall_y1),
        "cube_top": collect_face_map("cube_top", sub_cube),
        "cube_bottom": collect_face_map("cube_bottom", sub_cube),
        "cube_x0": collect_face_map("cube_x0", sub_cube),
        "cube_x1": collect_face_map("cube_x1", sub_cube),
        "cube_y0": collect_face_map("cube_y0", sub_cube),
        "cube_y1": collect_face_map("cube_y1", sub_cube),
    }

    # Rendu (réutilise render_photo mais version locale simplifiée)
    def render_local(L_face_dict):
        aspect = width / height
        fov_y = np.deg2rad(fov_y_deg)
        forward = cam_look - cam_pos
        forward = forward / (np.linalg.norm(forward) + EPS)
        right = np.cross(forward, cam_up)
        right = right / (np.linalg.norm(right) + EPS)
        up_v = np.cross(right, forward)
        tan_half_fov = np.tan(fov_y / 2.0)
        img = np.zeros((height, width, 3), dtype=np.float32)
        Lmax = max(np.max(v) for v in L_face_dict.values()) + 1e-8

        # fonctions locales de géométrie pour mapping (simples, mêmes que global)
        def face_ij(face, p):
            if face == "floor":
                nx, ny = sub_floor; u, v = p[0] / W, p[1] / D
            elif face == "ceiling":
                nx, ny = sub_ceiling; u, v = p[0] / W, p[1] / D
            elif face == "wall_y0":
                nx, ny = sub_wall_y0; u, v = p[0] / W, p[2] / H
            elif face == "wall_y1":
                nx, ny = sub_wall_y1; u, v = p[0] / W, p[2] / H
            elif face == "wall_x0":
                nx, ny = sub_wall_x0; u, v = p[1] / D, p[2] / H
            elif face == "wall_x1":
                nx, ny = sub_wall_x1; u, v = p[1] / D, p[2] / H
            elif face.startswith("cube_"):
                nx, ny = sub_cube
                if face in ("cube_top", "cube_bottom"):
                    u = (p[0] - cube_x0) / (cube_x1 - cube_x0)
                    v = (p[1] - cube_y0) / (cube_y1 - cube_y0)
                elif face in ("cube_x0", "cube_x1"):
                    u = (p[1] - cube_y0) / (cube_y1 - cube_y0)
                    v = (p[2] - cube_z0) / (cube_z1 - cube_z0)
                else:  # cube_y0, cube_y1
                    u = (p[0] - cube_x0) / (cube_x1 - cube_x0)
                    v = (p[2] - cube_z0) / (cube_z1 - cube_z0)
            else:
                return None
            i = min(nx - 1, max(0, int(u * nx)))
            j = min(ny - 1, max(0, int(v * ny)))
            return (i, j)

        def intersect_scene(ray_o, ray_d):
            tmin = np.inf; hit_face = None; hit_p = None
            # cube
            if abs(ray_d[2]) > 1e-8:
                for z_val, face_name in ((cube_z0, "cube_bottom"), (cube_z1, "cube_top")):
                    t = (z_val - ray_o[2]) / ray_d[2]
                    if t > 1e-6:
                        p = ray_o + t * ray_d
                        if cube_x0 <= p[0] <= cube_x1 and cube_y0 <= p[1] <= cube_y1 and t < tmin:
                            tmin, hit_face, hit_p = t, face_name, p
            if abs(ray_d[0]) > 1e-8:
                for x_val, face_name in ((cube_x0, "cube_x0"), (cube_x1, "cube_x1")):
                    t = (x_val - ray_o[0]) / ray_d[0]
                    if t > 1e-6:
                        p = ray_o + t * ray_d
                        if cube_y0 <= p[1] <= cube_y1 and cube_z0 <= p[2] <= cube_z1 and t < tmin:
                            tmin, hit_face, hit_p = t, face_name, p
            if abs(ray_d[1]) > 1e-8:
                for y_val, face_name in ((cube_y0, "cube_y0"), (cube_y1, "cube_y1")):
                    t = (y_val - ray_o[1]) / ray_d[1]
                    if t > 1e-6:
                        p = ray_o + t * ray_d
                        if cube_x0 <= p[0] <= cube_x1 and cube_z0 <= p[2] <= cube_z1 and t < tmin:
                            tmin, hit_face, hit_p = t, face_name, p
            # box
            if abs(ray_d[2]) > 1e-8:
                for z_val, face_name in ((0.0, "floor"), (H, "ceiling")):
                    t = (z_val - ray_o[2]) / ray_d[2]
                    if t > 1e-6:
                        p = ray_o + t * ray_d
                        if 0.0 <= p[0] <= W and 0.0 <= p[1] <= D and t < tmin:
                            tmin, hit_face, hit_p = t, face_name, p
            if abs(ray_d[0]) > 1e-8:
                for x_val, face_name in ((0.0, "wall_x0"), (W, "wall_x1")):
                    t = (x_val - ray_o[0]) / ray_d[0]
                    if t > 1e-6:
                        p = ray_o + t * ray_d
                        if 0.0 <= p[1] <= D and 0.0 <= p[2] <= H and t < tmin:
                            tmin, hit_face, hit_p = t, face_name, p
            if abs(ray_d[1]) > 1e-8:
                for y_val, face_name in ((0.0, "wall_y0"), (D, "wall_y1")):
                    t = (y_val - ray_o[1]) / ray_d[1]
                    if t > 1e-6:
                        p = ray_o + t * ray_d
                        if 0.0 <= p[0] <= W and 0.0 <= p[2] <= H and t < tmin:
                            tmin, hit_face, hit_p = t, face_name, p
            return tmin, hit_face, hit_p

        for ypix in range(height):
            py = 1.0 - 2.0 * (ypix + 0.5) / height
            for xpix in range(width):
                px = 2.0 * (xpix + 0.5) / width - 1.0
                dir_cam = forward + px * tan_half_fov * aspect * right + py * tan_half_fov * up_v
                dir_cam = dir_cam / (np.linalg.norm(dir_cam) + EPS)
                t, face, hp = intersect_scene(cam_pos, dir_cam)
                if face is None or not np.isfinite(t):
                    continue
                ij = face_ij(face, hp)
                if ij is None:
                    continue
                i, j = ij
                Lhit = L_face_dict[face][j, i]
                val = Lhit / (Lhit + Lmax)  # Reinhard simple
                val = np.power(np.clip(val, 0.0, 1.0), 1.0 / 2.2)
                img[ypix, xpix, :] = val
        return img

    img = render_local(L_face_local)
    from pathlib import Path as _P
    _out_dir = _P(__file__).parent / "outputs"
    _out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as _plt
    _plt.imsave(str(output_path), np.clip(img, 0, 1))
    return output_path


def batch_render_light_positions(list_of_light_pos_lists, base_name="batch_light"):
    """Prend une liste de configurations (chaque élément = liste de centres [x,y])
    et génère une image pour chacune."""
    from pathlib import Path as _P
    results = []
    out_dir = _P(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, lp in enumerate(list_of_light_pos_lists):
        fname = f"{base_name}_{idx}.png"
        path = out_dir / fname
        print(f"[Batch] Rendu configuration {idx} : {lp} -> {path}")
        compute_image_for_light_pos(lp, path)
        results.append(path)
    return results

# Exemple d'utilisation (décommentez pour lancer un batch) :
# scenarios = [
#     [[2.5, 2.5]],  # une seule source au centre
#     [[1.0, 1.0], [4.0, 4.0]],  # deux sources diagonales
#     [[0.5, 0.5], [2.5, 2.5], [4.5, 0.5]]  # trois sources
# ]
# batch_render_light_positions(scenarios, base_name="multi_light_test")
