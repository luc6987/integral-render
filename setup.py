import numpy as np  # type: ignore

from src.scene import (
    Materials,
    BoxDimensions,
    SceneConfig,
    RectPrism,
    CubeParams,
    Subdivisions,
)

# ---- Room dimensions ----
W, D, H = 5.0, 5.0, 3.0

# ---- Materials ----
materials = Materials(
    rho_floor=0.3,
    rho_ceiling=0.0,
    rho_walls=0.0,
    rho_cube=0.5,
    Le_light=10.0,
)

# ---- Basis function type ----
# Options: "P0" (constant per patch) or "P1" (Q1 bilinear per element)
basis_type = "P1"  # Change to "P1" for Q1 bilinear basis

# ---- Visibility options ----
# If True, skip computing geometric occlusion and use V(x,y)=1 (except diagonal)
skip_visibility = False

# ---- Subdivisions ----
rhodiv = 7  # density factor for patches
subdivisions = Subdivisions(
    floor=(rhodiv * int(W), rhodiv * int(D)),
    ceiling=(int(W), int(D)),
    wall_x0=(1, 1),
    wall_x1=(1, 1),
    wall_y0=(1, 1),
    wall_y1=(1, 1),
    cube=(rhodiv, rhodiv),
)

# ---- Light defaults ----
light_size = (1.0, 1.0)

z_0 = 0.0

# ---- Composition-aware extra prisms (same subdivision as cube) ----
prisms = [
    # P0: foreground slender (near camera, left), tall for depth
    RectPrism(x0=0.6, x1=1.1, y0=1.4, y1=2.0, z0=z_0, z1=0.2),
    # P1: mid-right broad low block (secondary mass)
    RectPrism(x0=3.4, x1=4.2, y0=1.8, y1=2.5, z0=z_0, z1=0.6),
    # P2: back-right slim tall (counterweight, far depth cue)
    RectPrism(x0=4.1, x1=4.6, y0=4.0, y1=4.6, z0=z_0, z1=1.8),
    # P3: back-left small medium height (light anchor)
    RectPrism(x0=0.6, x1=1.1, y0=3.8, y1=4.4, z0=z_0, z1=0.9),
]

# ---- Main cube ----
cube = CubeParams(size=1.0, z0=z_0)

# ---- Scene configuration (no fixed lights; scenarios below will define masks) ----
scene_config = SceneConfig(
    box=BoxDimensions(W=W, D=D, H=H),
    materials=materials,
    light_size=light_size,
    light_positions=None,
    extra_prisms=prisms,
    cube=cube,
    subdivisions=subdivisions,
    basis_type=basis_type,
)

# ---- Light scenarios ----
scenarios = [
    {
        "name": "corner_left_up",
        "positions": [(0.5, D - 0.5)],
        "size": light_size,
        "Le": materials.Le_light,
    },
]

# ---- Camera and render settings ----
cam_pos = np.array([0.0, 0.0, 1.5])
cam_look = np.array([5.0, 5.0, 1.0])
cam_up = np.array([0.0, 0.0, 1.0])

width = 960
height = 720
fov_y_deg = 55.0
exposure = 5.0
brightness = 0.6
# Percentile for tone mapping white point (e.g., 99.0 means 99th percentile)
tone_white_percentile = 99.0

