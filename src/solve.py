import numpy as np  # type: ignore
from pathlib import Path


def _load_csv_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _save_csv_vector(vec: np.ndarray, out_path: Path, header: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, vec, delimiter=",", header=header, comments="")
    return out_path


def solve_from_csv(csv_dir: Path) -> dict:
    """Solve Ax=b from CSV in csv_dir; supports single or multiple right-hand sides.

    - Uses `A.csv` and `b.csv` only.
    - Saves `x.csv` (radiosity) and `L.csv` (radiance = x/pi).
    """
    A_path = csv_dir / "A.csv"
    B_path = csv_dir / "b.csv"
    if not A_path.exists() or not B_path.exists():
        raise FileNotFoundError("A.csv or b.csv not found in " + str(csv_dir))
    print(f"[Solve] Loading A from {A_path}")
    print(f"[Solve] Loading b from {B_path}")
    A = _load_csv_matrix(A_path)
    b = _load_csv_matrix(B_path)

    # Ensure b is 2D: (N,) -> (N,1)
    if b.ndim == 1:
        b = b[:, None]
    print(f"[Solve] Solving {A.shape[0]}x{A.shape[1]} with {b.shape[1]} RHS...")
    x = np.linalg.solve(A, b)
    L = x / np.pi
    # Save possibly multi-column CSVs
    x_hdr = (
        f"x (radiosity), length {x.shape[0]}"
        if x.shape[1] == 1
        else f"x (radiosity), shape {x.shape[0]}x{x.shape[1]} (columns)"
    )
    L_hdr = (
        f"L (radiance), length {L.shape[0]}"
        if L.shape[1] == 1
        else f"L (radiance), shape {L.shape[0]}x{L.shape[1]} (columns)"
    )
    x_path = _save_csv_vector(x, csv_dir / "x.csv", header=x_hdr)
    L_path = _save_csv_vector(L, csv_dir / "L.csv", header=L_hdr)
    print(f"[Solve] Saved x to {x_path}")
    print(f"[Solve] Saved L to {L_path}")
    return {"x": x_path, "L": L_path}


def main():
    # Use project root's linear_system_csv by default
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    csv_dir = PROJECT_ROOT / "linear_system_csv"
    solve_from_csv(csv_dir)


if __name__ == "__main__":
    main()
