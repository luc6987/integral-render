from pathlib import Path
import numpy as np  # type: ignore

from src.linear_system import (
    main_export_linear_system,
    build_linear_system,
    _build_M_E_for_scenario,
)
from src.render import (
    main_from_csv,
    render_photo,
    build_L_face,
    save_image,
)
from src.scene import build_scene
from setup import (
    scene_config,
    scenarios,
    cam_pos,
    cam_look,
    cam_up,
    width,
    height,
    fov_y_deg,
    exposure,
    brightness,
    save_intermediate_csv,
    hide_walls_ceiling,
)


def main():
    csv_dir = Path(__file__).parent / "linear_system_csv"
    if save_intermediate_csv:
        # Original CSV-based pipeline
        print("[Main] Exporting A and b (from setup.scenarios)...")
        main_export_linear_system()
        from src.solve import solve_from_csv  # lazy import to keep deps minimal
        print("[Main] Solving Ax=b from CSV ...")
        solve_from_csv(csv_dir)
        print("[Main] Rendering from CSV solution ...")
        main_from_csv()
        return

    # In-memory pipeline: build → solve → render, only saving final images
    print("[Main] Running in-memory pipeline (no CSV export/save)...")
    scene = build_scene(scene_config)
    F0, _ = build_linear_system(scene)
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, sc in enumerate(scenarios):
        A, b = _build_M_E_for_scenario(
            scene,
            F0,
            positions=sc.get("positions", []),
            size=sc.get("size", scene_config.light_size),
            Le=sc.get("Le", scene_config.materials.Le_light),
        )
        # Solve in memory
        x = np.linalg.solve(A, b)
        L = x / np.pi
        
        # Compute light mask for rendering
        from src.scene import compute_ceiling_light_mask
        light_mask = compute_ceiling_light_mask(
            scene, 
            light_positions=sc.get("positions", []), 
            light_size=sc.get("size", scene_config.light_size)
        )
        
        # Render
        L_face = build_L_face(scene, L, light_mask, hide_walls_ceiling)
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
        name = sc.get("name", f"sc{k}")
        save_image(img, out_dir / f"render_{name}.png")


if __name__ == "__main__":
    main()
