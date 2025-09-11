import numpy as np  # type: ignore

from src.scene import (
    Materials,
    BoxDimensions,
    SceneConfig,
    RectPrism,
    CubeParams,
)

# ---- Room dimensions ----
W, D, H = 5.0, 5.0, 3.0

# ---- Materials ----
materials = Materials(
    rho_floor=0.3,
    rho_ceiling=0.0,
    rho_walls=0.0,
    rho_cube=0.999,
    Le_light=10.0,
)

# ---- Light defaults ----
light_size = (1.0, 1.0)

# ---- Composition-aware extra prisms (same subdivision as cube) ----
prisms = [
    # P0: foreground slender (near camera, left), tall for depth
    RectPrism(x0=0.5, x1=1.0, y0=0.6, y1=1.1, z0=1e-3, z1=1.4),
    # P1: mid-right broad low block (secondary mass)
    RectPrism(x0=3.4, x1=4.2, y0=1.8, y1=2.5, z0=1e-3, z1=0.6),
    # P2: back-right slim tall (counterweight, far depth cue)
    RectPrism(x0=4.1, x1=4.6, y0=4.0, y1=4.6, z0=1e-3, z1=1.8),
    # P3: back-left small medium height (light anchor)
    RectPrism(x0=0.6, x1=1.1, y0=3.8, y1=4.4, z0=1e-3, z1=0.9),
]

# ---- Main cube ----
cube = CubeParams(size=1.6, z0=1e-3)

# ---- Scene configuration (no fixed lights; scenarios below will define masks) ----
scene_config = SceneConfig(
    box=BoxDimensions(W=W, D=D, H=H),
    materials=materials,
    light_size=light_size,
    light_positions=None,
    extra_prisms=prisms,
    cube=cube,
)

# ---- Light scenarios ----
scenarios = [
    {
        "name": "corner_left_up",
        "positions": [(0.5, D - 0.5)],
        "size": light_size,
        "Le": materials.Le_light,
    },
    {
        "name": "center",
        "positions": [(W / 2.0, D / 2.0)],
        "size": light_size,
        "Le": materials.Le_light,
    },
    {
        "name": "two_lights",
        "positions": [(0.8, 0.8), (W - 0.8, D - 0.8)],
        "size": (0.8, 0.8),
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
exposure = 10.0
brightness = 2.2

