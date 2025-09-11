import numpy as np  # type: ignore
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

EPS = 1e-9


@dataclass
class Subdivisions:
    floor: Tuple[int, int]
    ceiling: Tuple[int, int]
    wall_x0: Tuple[int, int]
    wall_x1: Tuple[int, int]
    wall_y0: Tuple[int, int]
    wall_y1: Tuple[int, int]
    cube: Tuple[int, int]


@dataclass
class Materials:
    rho_floor: float
    rho_ceiling: float
    rho_walls: float
    rho_cube: float
    Le_light: float


@dataclass
class BoxDimensions:
    W: float
    D: float
    H: float


@dataclass
class CubeParams:
    size: float
    z0: float


@dataclass
class SceneConfig:
    box: BoxDimensions
    materials: Materials
    light_size: Tuple[float, float] = (1.0, 1.0)
    light_positions: Optional[List[Tuple[float, float]]] = None
    subdivisions: Optional[Subdivisions] = None
    cube: CubeParams = field(
        default_factory=lambda: CubeParams(size=1.0, z0=1e-3)
    )


class Patch:
    __slots__ = (
        "center",
        "normal",
        "area",
        "rho",
        "is_light",
        "face",
        "ij",
    )

    def __init__(
        self,
        center: np.ndarray,
        normal: np.ndarray,
        area: float,
        rho: float,
        is_light: bool,
        face: str,
        ij: Tuple[int, int],
    ) -> None:
        self.center = center
        self.normal = normal
        self.area = area
        self.rho = rho
        self.is_light = is_light
        self.face = face
        self.ij = ij


def _make_grid_on_plane(
    origin: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    nx: int,
    ny: int,
    rho: float,
    normal: np.ndarray,
    face: str,
    light_mask: Optional[Callable[[int, int], bool]] = None,
) -> List[Patch]:
    patches: List[Patch] = []
    dx = 1.0 / nx
    dy = 1.0 / ny
    cell_area = float(np.linalg.norm(np.cross(ux * dx, uy * dy)))
    n = normal / (np.linalg.norm(normal) + EPS)
    for i in range(nx):
        for j in range(ny):
            c = origin + ux * ((i + 0.5) * dx) + uy * ((j + 0.5) * dy)
            is_light = light_mask(i, j) if light_mask is not None else False
            patches.append(
                Patch(c, n, cell_area, rho, is_light, face, (i, j))
            )
    return patches


def _mk_multi_ceiling_light_mask(
    sub: Tuple[int, int],
    W: float,
    D: float,
    light_size: Tuple[float, float],
    light_pos_list: Sequence[Tuple[float, float]],
) -> Callable[[int, int], bool]:
    nx, ny = sub
    lx, ly = light_size
    rects = []
    for cx, cy in light_pos_list:
        rects.append(
            (cx - lx / 2.0, cx + lx / 2.0, cy - ly / 2.0, cy + ly / 2.0)
        )

    def mask(i: int, j: int) -> bool:
        u = (i + 0.5) / nx
        v = (j + 0.5) / ny
        x = u * W
        y = v * D
        for x0, x1, y0, y1 in rects:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    return mask


@dataclass
class BuiltScene:
    patches: List[Patch]
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    rho: np.ndarray
    is_light: np.ndarray
    sub_by_face: Dict[str, Tuple[int, int]]
    cube_bounds: Dict[str, float]


def build_scene(config: SceneConfig) -> BuiltScene:
    W, D, H = config.box.W, config.box.D, config.box.H
    rho_floor = config.materials.rho_floor
    rho_ceiling = config.materials.rho_ceiling
    rho_walls = config.materials.rho_walls
    rho_cube = config.materials.rho_cube

    if config.subdivisions is None:
        rhodiv = 10
        sub_floor = (rhodiv * int(W), rhodiv * int(D))
        sub_ceiling = (int(W), int(D))
        sub_wall_x0 = (1, 1)
        sub_wall_x1 = (1, 1)
        sub_wall_y0 = (1, 1)
        sub_wall_y1 = (1, 1)
        sub_cube = (rhodiv, rhodiv)
    else:
        s = config.subdivisions
        sub_floor = s.floor
        sub_ceiling = s.ceiling
        sub_wall_x0 = s.wall_x0
        sub_wall_x1 = s.wall_x1
        sub_wall_y0 = s.wall_y0
        sub_wall_y1 = s.wall_y1
        sub_cube = s.cube

    cube_size = config.cube.size
    cube_z0 = config.cube.z0
    cube_center = np.array(
        [W / 2.0, D / 2.0, cube_z0 + cube_size / 2.0]
    )
    cube_x0 = float(cube_center[0] - cube_size / 2.0)
    cube_x1 = float(cube_center[0] + cube_size / 2.0)
    cube_y0 = float(cube_center[1] - cube_size / 2.0)
    cube_y1 = float(cube_center[1] + cube_size / 2.0)
    cube_z1 = float(cube_z0 + cube_size)

    patches: List[Patch] = []

    # Floor
    origin_floor = np.array([0.0, 0.0, 0.0])
    ux_floor = np.array([W, 0.0, 0.0])
    uy_floor = np.array([0.0, D, 0.0])
    patches += _make_grid_on_plane(
        origin_floor,
        ux_floor,
        uy_floor,
        *sub_floor,
        rho_floor,
        np.array([0, 0, 1]),
        face="floor",
    )

    # Ceiling
    origin_ceil = np.array([0.0, 0.0, H])
    ux_ceil = np.array([W, 0.0, 0.0])
    uy_ceil = np.array([0.0, D, 0.0])
    light_mask = None
    if config.light_positions and len(config.light_positions) > 0:
        light_mask = _mk_multi_ceiling_light_mask(
            sub_ceiling, W, D, config.light_size, config.light_positions
        )
    patches += _make_grid_on_plane(
        origin_ceil,
        ux_ceil,
        uy_ceil,
        *sub_ceiling,
        rho_ceiling,
        np.array([0, 0, -1]),
        face="ceiling",
        light_mask=light_mask,
    )

    # Walls
    origin_y0 = np.array([0.0, 0.0, 0.0])
    ux_y0 = np.array([W, 0.0, 0.0])
    uy_y0 = np.array([0.0, 0.0, H])
    patches += _make_grid_on_plane(
        origin_y0,
        ux_y0,
        uy_y0,
        *sub_wall_y0,
        rho_walls,
        np.array([0, 1, 0]),
        face="wall_y0",
    )

    origin_y1 = np.array([0.0, D, 0.0])
    ux_y1 = np.array([W, 0.0, 0.0])
    uy_y1 = np.array([0.0, 0.0, H])
    patches += _make_grid_on_plane(
        origin_y1,
        ux_y1,
        uy_y1,
        *sub_wall_y1,
        rho_walls,
        np.array([0, -1, 0]),
        face="wall_y1",
    )

    origin_x0 = np.array([0.0, 0.0, 0.0])
    ux_x0 = np.array([0.0, D, 0.0])
    uy_x0 = np.array([0.0, 0.0, H])
    patches += _make_grid_on_plane(
        origin_x0,
        ux_x0,
        uy_x0,
        *sub_wall_x0,
        rho_walls,
        np.array([1, 0, 0]),
        face="wall_x0",
    )

    origin_x1 = np.array([W, 0.0, 0.0])
    ux_x1 = np.array([0.0, D, 0.0])
    uy_x1 = np.array([0.0, 0.0, H])
    patches += _make_grid_on_plane(
        origin_x1,
        ux_x1,
        uy_x1,
        *sub_wall_x1,
        rho_walls,
        np.array([-1, 0, 0]),
        face="wall_x1",
    )

    # Cube
    origin_ctop = np.array([cube_x0, cube_y0, cube_z1])
    ux_ctop = np.array([cube_x1 - cube_x0, 0.0, 0.0])
    uy_ctop = np.array([0.0, cube_y1 - cube_y0, 0.0])
    patches += _make_grid_on_plane(
        origin_ctop,
        ux_ctop,
        uy_ctop,
        *sub_cube,
        rho_cube,
        np.array([0, 0, 1]),
        face="cube_top",
    )

    origin_cbot = np.array([cube_x0, cube_y0, cube_z0])
    ux_cbot = np.array([cube_x1 - cube_x0, 0.0, 0.0])
    uy_cbot = np.array([0.0, cube_y1 - cube_y0, 0.0])
    patches += _make_grid_on_plane(
        origin_cbot,
        ux_cbot,
        uy_cbot,
        *sub_cube,
        rho_cube,
        np.array([0, 0, -1]),
        face="cube_bottom",
    )

    origin_cx0 = np.array([cube_x0, cube_y0, cube_z0])
    ux_cx0 = np.array([0.0, cube_y1 - cube_y0, 0.0])
    uy_cx0 = np.array([0.0, 0.0, cube_z1 - cube_z0])
    patches += _make_grid_on_plane(
        origin_cx0,
        ux_cx0,
        uy_cx0,
        *sub_cube,
        rho_cube,
        np.array([-1, 0, 0]),
        face="cube_x0",
    )

    origin_cx1 = np.array([cube_x1, cube_y0, cube_z0])
    ux_cx1 = np.array([0.0, cube_y1 - cube_y0, 0.0])
    uy_cx1 = np.array([0.0, 0.0, cube_z1 - cube_z0])
    patches += _make_grid_on_plane(
        origin_cx1,
        ux_cx1,
        uy_cx1,
        *sub_cube,
        rho_cube,
        np.array([1, 0, 0]),
        face="cube_x1",
    )

    origin_cy0 = np.array([cube_x0, cube_y0, cube_z0])
    ux_cy0 = np.array([cube_x1 - cube_x0, 0.0, 0.0])
    uy_cy0 = np.array([0.0, 0.0, cube_z1 - cube_z0])
    patches += _make_grid_on_plane(
        origin_cy0,
        ux_cy0,
        uy_cy0,
        *sub_cube,
        rho_cube,
        np.array([0, -1, 0]),
        face="cube_y0",
    )

    origin_cy1 = np.array([cube_x0, cube_y1, cube_z0])
    ux_cy1 = np.array([cube_x1 - cube_x0, 0.0, 0.0])
    uy_cy1 = np.array([0.0, 0.0, cube_z1 - cube_z0])
    patches += _make_grid_on_plane(
        origin_cy1,
        ux_cy1,
        uy_cy1,
        *sub_cube,
        rho_cube,
        np.array([0, 1, 0]),
        face="cube_y1",
    )

    centers = np.array([p.center for p in patches])
    normals = np.array([p.normal for p in patches])
    areas = np.array([p.area for p in patches])
    rho_arr = np.array([p.rho for p in patches])
    is_light_arr = np.array([p.is_light for p in patches])

    sub_by_face = {
        "floor": sub_floor,
        "ceiling": sub_ceiling,
        "wall_x0": sub_wall_x0,
        "wall_x1": sub_wall_x1,
        "wall_y0": sub_wall_y0,
        "wall_y1": sub_wall_y1,
        "cube_top": sub_cube,
        "cube_bottom": sub_cube,
        "cube_x0": sub_cube,
        "cube_x1": sub_cube,
        "cube_y0": sub_cube,
        "cube_y1": sub_cube,
    }

    cube_bounds = {
        "cube_x0": cube_x0,
        "cube_x1": cube_x1,
        "cube_y0": cube_y0,
        "cube_y1": cube_y1,
        "cube_z0": cube_z0,
        "cube_z1": cube_z1,
    }

    return BuiltScene(
        patches=patches,
        centers=centers,
        normals=normals,
        areas=areas,
        rho=rho_arr,
        is_light=is_light_arr,
        sub_by_face=sub_by_face,
        cube_bounds=cube_bounds,
    )
