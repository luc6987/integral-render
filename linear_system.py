import numpy as np  # type: ignore
from pathlib import Path
from typing import Dict, Tuple

from scene import BuiltScene


def assemble_form_factor_matrix(
    centers: np.ndarray, normals: np.ndarray, areas: np.ndarray
) -> np.ndarray:
    N = centers.shape[0]
    EPS = 1e-9
    F = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
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
    for i in range(N):
        for j in range(i + 1, N):
            Tij = 0.5 * (areas[i] * F[i, j] + areas[j] * F[j, i])
            F[i, j] = Tij / (areas[i] + EPS)
            F[j, i] = Tij / (areas[j] + EPS)
    np.fill_diagonal(F, 0.0)


def build_linear_system(scene: BuiltScene) -> Tuple[np.ndarray, np.ndarray]:
    F = assemble_form_factor_matrix(
        scene.centers,
        scene.normals,
        scene.areas,
    )
    enforce_reciprocity(F, scene.areas)
    E = np.zeros(scene.centers.shape[0])
    return F, E


def save_system(F: np.ndarray, E: np.ndarray, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    F_path = out_dir / "F.npy"
    E_path = out_dir / "E.npy"
    np.save(F_path, F)
    np.save(E_path, E)
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
    """Export dense M and E as CSV files into out_dir, with headers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    M_path = out_dir / "M.csv"
    e_path = out_dir / "E.csv"
    mh = f"M = I - diag(rho) F, shape {M.shape[0]}x{M.shape[1]}"
    eh = f"E (emission), length {E.shape[0]}"
    np.savetxt(M_path, M, delimiter=",", header=mh, comments="")
    np.savetxt(e_path, E, delimiter=",", header=eh, comments="")
    return {"M": M_path, "E": e_path}
