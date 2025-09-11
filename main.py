import numpy as np  # type: ignore
from pathlib import Path

from scene import (
    SceneConfig,
    BoxDimensions,
    Materials,
    build_scene,
    compute_ceiling_light_mask,
    RectPrism,
    CubeParams,
)
from linear_system import (
    build_linear_system,
    build_system_matrix,
    export_system_to_csv,
    export_multiple_E_to_csv,
)
from render import (
    build_emission_and_reflectance,
    build_emission_and_reflectance_with_mask,
    solve_radiosity,
    build_L_face,
    render_photo,
    save_image,
)


def main():
    W, D, H = 5.0, 5.0, 3.0
    materials = Materials(
        rho_floor=0.3,
        rho_ceiling=0.0,
        rho_walls=0.0,
        rho_cube=0.999,
        Le_light=10.0,
    )

    # Base light size (per scenario can override)
    light_size = (1.0, 1.0)

    # Add four rectangular prisms (composition-aware), same subdivision as cube
    # Guiding ideas: big–medium–small hierarchy, unequal spacing/heights, overlap, no tangents
    prisms = [
        # P0: foreground slender (near camera, left), tall for depth
        RectPrism(x0=0.5, x1=1.0, y0=0.6, y1=1.1, z0=1e-3, z1=1.4),
        # P1: mid-right broad low block (secondary mass, avoids central cube)
        RectPrism(x0=3.4, x1=4.2, y0=1.8, y1=2.5, z0=1e-3, z1=0.6),
        # P2: back-right slim tall (counterweight, far depth cue)
        RectPrism(x0=4.1, x1=4.6, y0=4.0, y1=4.6, z0=1e-3, z1=1.8),
        # P3: back-left small medium height (light anchor, avoids tangents)
        RectPrism(x0=0.6, x1=1.1, y0=3.8, y1=4.4, z0=1e-3, z1=0.9),
    ]

    config = SceneConfig(
        box=BoxDimensions(W=W, D=D, H=H),
        materials=materials,
        # Scene built without fixed lights; scenarios specify masks dynamically
        light_size=light_size,
        light_positions=None,
        extra_prisms=prisms,
        cube=CubeParams(size=1.6, z0=1e-3),  # larger main cube for hierarchy
    )

    print("[Main] Start building scene...")
    scene = build_scene(config)
    print("[Main] Scene built.")

    print("[Main] Building linear system...")
    F, _ = build_linear_system(scene)
    print("[Main] Linear system built (F, E0)")

    # Define a series of light scenarios
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

    cam_pos = np.array([0, 0, H / 2.0])
    cam_look = np.array([W, D, H / 3.0]) - np.array([0, 0, H / 2.0])
    cam_up = np.cross(np.cross(cam_look, np.array([0, 0, 1])), cam_look)

    exposure = 10.0
    brightness = 2.2

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all E vectors for a combined CSV
    all_E = []
    all_names = []

    print(f"[Main] Running {len(scenarios)} light scenarios...")
    for sc in scenarios:
        name = sc["name"]
        print(f"[Main] Scenario: {name}")
        positions = sc["positions"]
        size = sc.get("size", light_size)
        Le = sc.get("Le", materials.Le_light)

        is_light_mask = compute_ceiling_light_mask(
            scene, light_positions=positions, light_size=size
        )
        E, rho = build_emission_and_reflectance_with_mask(
            scene, Le_light=Le, is_light_mask=is_light_mask
        )
        all_E.append(E)
        all_names.append(name)

        # Optionally export a single-system CSV for the first scenario
        if len(all_E) == 1:
            M = build_system_matrix(F, rho)
            csv_dir = Path(__file__).parent / "linear_system_csv"
            export_system_to_csv(M, E, csv_dir)

        # Solve and render image for this scenario
        _, L = solve_radiosity(F, rho, E)
        L_face = build_L_face(scene, L)
        img = render_photo(
            scene=scene,
            box=config.box,
            L_face=L_face,
            width=800,
            height=600,
            fov_y_deg=40.0,
            cam_pos=cam_pos,
            cam_look=cam_look,
            cam_up=cam_up,
            exposure=exposure,
            brightness=brightness,
        )
        out_path = out_dir / f"radiosity_box_galerkin_out_{name}.png"
        save_image(img, out_path)

    # Export all E vectors into one CSV with columns as scenarios
    multi_csv = Path(__file__).parent / "linear_system_csv" / "E_multi.csv"
    export_multiple_E_to_csv(all_E, all_names, multi_csv)
    print(f"[Main] Saved multi-E CSV: {multi_csv}")


if __name__ == "__main__":
    main()
