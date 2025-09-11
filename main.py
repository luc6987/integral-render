import numpy as np  # type: ignore
from pathlib import Path

from scene import SceneConfig, BoxDimensions, Materials, build_scene
from linear_system import build_linear_system, build_system_matrix, export_system_to_csv
from render import (
    build_emission_and_reflectance,
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

    light_size = (1.0, 1.0)
    # Move light to room's left-up corner (assume world origin at floor (0,0))
    light_positions = [(0.5, D - 0.5)]

    config = SceneConfig(
        box=BoxDimensions(W=W, D=D, H=H),
        materials=materials,
        light_size=light_size,
        light_positions=light_positions,
    )

    scene = build_scene(config)

    F, _ = build_linear_system(scene)

    E, rho = build_emission_and_reflectance(scene, Le_light=materials.Le_light)
    M = build_system_matrix(F, rho)

    # Export linear system to CSV in a new folder
    csv_dir = Path(__file__).parent / "linear_system_csv"
    export_system_to_csv(M, E, csv_dir)

    # Solve and render
    _, L = solve_radiosity(F, rho, E)

    L_face = build_L_face(scene, L)

    cam_pos = np.array([0, 0, H / 2.0])
    cam_look = np.array([W, D, H / 3.0]) - np.array([0, 0, H / 2.0])
    cam_up = np.cross(np.cross(cam_look, np.array([0, 0, 1])), cam_look)

    # Adjust these to control image brightness directly
    exposure = 10.0  # scales scene radiance before tone mapping
    brightness = 2.2  # post-tone-map multiplier (clamped)

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

    out_dir = Path(__file__).parent / "outputs"
    out_path = out_dir / "radiosity_box_galerkin_out.png"
    save_image(img, out_path)
    print(f"Saved image: {out_path}")


if __name__ == "__main__":
    main()
