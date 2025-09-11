import numpy as np  # type: ignore
from pathlib import Path
from typing import Dict, Tuple, List

from scene import BuiltScene
from scene import build_scene, compute_ceiling_light_mask
from setup import scene_config, scenarios, light_size, materials
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # fallback if tqdm not available


def assemble_form_factor_matrix(
    centers: np.ndarray, normals: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    N = centers.shape[0]
    EPS = 1e-9
    F = np.zeros((N, N), dtype=np.float64)
    print(f"[LinearSystem] Assembling form factor matrix F of size {N}x{N}...")
    rng = range(N)
    if tqdm is not None:
        rng = tqdm(rng, desc="Assembling F", leave=False)
    for i in rng:
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
        contrib = np.zeros(N)
        contrib[vis] = (
            (cos_i[vis] * cos_j[vis]) / (np.pi * r2[vis]) * areas[vis]
        )
        F[i, :] = contrib
    return F


def enforce_reciprocity(F: np.ndarray, areas: np.ndarray) -> None:
    N = F.shape[0]
    EPS = 1e-9
    print(f"[LinearSystem] Enforcing reciprocity on F (N={N})...")
    for i in range(N):
        for j in range(i + 1, N):
            Tij = 0.5 * (areas[i] * F[i, j] + areas[j] * F[j, i])
            F[i, j] = Tij / (areas[i] + EPS)
            F[j, i] = Tij / (areas[j] + EPS)
    np.fill_diagonal(F, 0.0)
    print("[LinearSystem] Reciprocity enforcement complete.")


def build_linear_system(scene: BuiltScene) -> Tuple[np.ndarray, np.ndarray]:
    print("[LinearSystem] Building linear system (F, E)...")
    F = assemble_form_factor_matrix(
        scene.centers,
        scene.normals,
        scene.areas,
    )
    enforce_reciprocity(F, scene.areas)
    E = np.zeros(scene.centers.shape[0])
    print(
        f"[LinearSystem] Built F (shape={F.shape}), initialized E (len={E.shape[0]})."
    )
    return F, E


def save_system(F: np.ndarray, E: np.ndarray, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    F_path = out_dir / "F.npy"
    E_path = out_dir / "E.npy"
    np.save(F_path, F)
    np.save(E_path, E)
    print(f"[LinearSystem] Saved F to {F_path}")
    print(f"[LinearSystem] Saved E to {E_path}")
    return {
        "F": F_path,
        "E": E_path,
    }


def load_system(in_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    F = np.load(in_dir / "F.npy")
    E = np.load(in_dir / "E.npy")
    return F, E


def build_system_matrix(F: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Return M = I - diag(rho) F for (I - rho F) B = E."""
    RF = (rho[:, None]) * F
    M = np.eye(F.shape[0]) - RF
    return M


def export_system_to_csv(M: np.ndarray, E: np.ndarray, out_dir: Path) -> Dict[str, Path]:
    """Export dense M and E as CSV files into out_dir, with headers.

    If E has multiple columns, they are saved in one E.csv (columns=scenarios).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    M_path = out_dir / "M.csv"
    e_path = out_dir / "E.csv"
    mh = f"M = I - diag(rho) F, shape {M.shape[0]}x{M.shape[1]}"
    eh = (
        f"E (emission), length {E.shape[0]}"
        if E.ndim == 1
        else f"E (emission), shape {E.shape[0]}x{E.shape[1]} (columns=scenarios)"
    )
    np.savetxt(M_path, M, delimiter=",", header=mh, comments="")
    np.savetxt(e_path, E, delimiter=",", header=eh, comments="")
    print(f"[LinearSystem] Exported M.csv to {M_path}")
    print(f"[LinearSystem] Exported E.csv to {e_path}")
    return {"M": M_path, "E": e_path}


def export_Ab_to_csv(A: np.ndarray, b: np.ndarray, out_dir: Path) -> Dict[str, Path]:
    """Export in canonical Ax=b naming.

    If b has multiple columns, save them in a single b.csv file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    A_path = out_dir / "A.csv"
    b_path = out_dir / "b.csv"
    ah = f"A, shape {A.shape[0]}x{A.shape[1]}"
    bh = (
        f"b, length {b.shape[0]}"
        if b.ndim == 1
        else f"b, shape {b.shape[0]}x{b.shape[1]} (columns=scenarios)"
    )
    np.savetxt(A_path, A, delimiter=",", header=ah, comments="")
    np.savetxt(b_path, b, delimiter=",", header=bh, comments="")
    print(f"[LinearSystem] Exported A.csv and b.csv to {out_dir}")
    return {"A": A_path, "b": b_path}


def export_multiple_E_to_csv(
    E_list: "list[np.ndarray]",
    names: "list[str]",
    out_path: Path,
) -> Path:
    """Export multiple E vectors side-by-side into a single CSV file.

    Rows correspond to patches, columns to scenarios.
    Header encodes number of patches and scenario names.
    """
    if len(E_list) == 0:
        raise ValueError("E_list is empty")
    N = E_list[0].shape[0]
    for idx, E in enumerate(E_list):
        if E.shape[0] != N:
            raise ValueError(f"E_list[{idx}] length mismatch: {E.shape[0]} != {N}")
    if len(names) != len(E_list):
        raise ValueError("names and E_list lengths differ")
    # Stack as N x K (columns are scenarios)
    K = len(E_list)
    M = np.zeros((N, K), dtype=float)
    for k, E in enumerate(E_list):
        M[:, k] = E
    header = (
        f"E_multi: patches={N}, scenarios={K}; columns=|" + "|".join(names) + "|"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, M, delimiter=",", header=header, comments="")
    print(
        f"[LinearSystem] Exported multi-scenario E to {out_path} (N={N}, K={K})."
    )
    return out_path


def _save_csv_vector(vec: np.ndarray, out_path: Path, header: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, vec, delimiter=",", header=header, comments="")
    return out_path


def _build_M_E_for_scenario(scene: BuiltScene, F: np.ndarray, *, positions, size, Le) -> Tuple[np.ndarray, np.ndarray]:
    # Build E and rho from light mask, then M = I - diag(rho) F
    is_light_mask = compute_ceiling_light_mask(scene, light_positions=positions, light_size=size)
    E = np.zeros(scene.centers.shape[0])
    E[is_light_mask] = np.pi * Le
    rho = np.where(is_light_mask, 0.0, scene.rho)
    M = build_system_matrix(F, rho)
    return M, E


def main_export_linear_system() -> None:
    # Build scene and F
    out_dir = Path(__file__).parent / "linear_system_csv"
    print("[LinearSystem:CLI] Building scene and assembling F...")
    scene = build_scene(scene_config)
    F, _ = build_linear_system(scene)
    # First scenario defines A (M). b may be multi-column.
    if not scenarios:
        raise RuntimeError("setup.scenarios is empty; define at least one scenario")
    sc0 = scenarios[0]
    M0, E0 = _build_M_E_for_scenario(
        scene,
        F,
        positions=sc0["positions"],
        size=sc0.get("size", light_size),
        Le=sc0.get("Le", materials.Le_light),
    )
    # Build b by stacking all scenarios' E (columns) if >1
    b_cols: List[np.ndarray] = []
    for sc in scenarios:
        _, Ei = _build_M_E_for_scenario(
            scene,
            F,
            positions=sc["positions"],
            size=sc.get("size", light_size),
            Le=sc.get("Le", materials.Le_light),
        )
        b_cols.append(Ei)
    b = E0 if len(b_cols) == 1 else np.stack(b_cols, axis=1)
    # Export with both namings for convenience
    export_Ab_to_csv(M0, b, out_dir)
    export_system_to_csv(M0, b, out_dir)
    print(f"[LinearSystem:CLI] Export complete in {out_dir}")


if __name__ == "__main__":
    main_export_linear_system()
