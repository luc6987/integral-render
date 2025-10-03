import os
import numpy as np  # type: ignore
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .scene import BuiltScene, P1Element
from .scene import build_scene, compute_ceiling_light_mask
from .visibility import get_or_build_visibility, is_segment_blocked
from setup import scene_config, scenarios, light_size, materials, skip_visibility
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # fallback if tqdm not available

# --- Optional GPU backend (CuPy) ---
try:  # pragma: no cover - optional dependency
    import cupy as cp  # type: ignore

    try:
        _GPU_DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()  # type: ignore[attr-defined]
    except Exception:
        _GPU_DEVICE_COUNT = 0
    CUPY_AVAILABLE = _GPU_DEVICE_COUNT > 0
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    CUPY_AVAILABLE = False

def _gpu_enabled_by_env() -> bool:
    """Return True if GPU should be used when available.

    Environment variable INTEGRAL_GPU can force behavior:
      - '0' → disable
      - '1' → enable (if CuPy and a device are available)
    When unset, defaults to enabling if CuPy is available.
    """
    flag = os.getenv("INTEGRAL_GPU")
    if flag is None:
        return CUPY_AVAILABLE
    return flag.strip() != "0" and CUPY_AVAILABLE


def assemble_form_factor_matrix(
    centers: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    V: Optional[np.ndarray] = None,
    *,
    use_gpu: Optional[bool] = None,
    gpu_tile: Optional[int] = None,
) -> np.ndarray:
    """Assemble form-factor matrix for P0 with optional GPU acceleration.

    - When CuPy + CUDA are available and `use_gpu` is True (default auto),
      computation runs on GPU in tiles of rows (size `gpu_tile`).
    - Falls back to the original NumPy implementation otherwise.
    """
    N = centers.shape[0]
    EPS = 1e-9

    if use_gpu is None:
        use_gpu = _gpu_enabled_by_env()

    if use_gpu:
        if not CUPY_AVAILABLE:
            print("[LinearSystem][GPU] CuPy/CUDA not available; falling back to CPU.")
        else:
            try:
                tsize = int(os.getenv("INTEGRAL_GPU_TILE", "256")) if gpu_tile is None else int(gpu_tile)
            except Exception:
                tsize = 256
            tsize = max(32, tsize)
            print(
                f"[LinearSystem][GPU] Assembling F on GPU (N={N}), tile={tsize} rows..."
            )
            try:
                # Move inputs to GPU
                c_centers = cp.asarray(centers, dtype=cp.float64)
                c_normals = cp.asarray(normals, dtype=cp.float64)
                c_areas = cp.asarray(areas, dtype=cp.float64)
                c_V = None if V is None else cp.asarray(V, dtype=cp.float64)
                c_F = cp.zeros((N, N), dtype=cp.float64)
                pi = float(np.pi)

                for i0 in range(0, N, tsize):
                    i1 = min(N, i0 + tsize)
                    rows = i1 - i0
                    ci = c_centers[i0:i1]  # (rows, 3)
                    ni = c_normals[i0:i1]  # (rows, 3)
                    # Broadcast against all targets j
                    v = c_centers[None, :, :] - ci[:, None, :]  # (rows, N, 3)
                    r2 = cp.sum(v * v, axis=2) + EPS
                    r = cp.sqrt(r2)
                    wi = v / r[:, :, None]
                    cos_i = cp.sum(wi * ni[:, None, :], axis=2)  # (rows, N)
                    cos_j = -cp.sum(wi * c_normals[None, :, :], axis=2)  # (rows, N)
                    base = (cos_i * cos_j) / (pi * r2) * c_areas[None, :]
                    vis_mask = (cos_i > 0.0) & (cos_j > 0.0)
                    base = cp.where(vis_mask, base, 0.0)
                    if c_V is not None:
                        base = base * c_V[i0:i1, :]
                    # Zero the diagonal entries for these rows
                    rr = cp.arange(rows)
                    cc = cp.arange(i0, i1)
                    base[rr, cc] = 0.0
                    c_F[i0:i1, :] = base

                F_np = cp.asnumpy(c_F)
                return F_np
            except Exception as e:  # pragma: no cover
                print(f"[LinearSystem][GPU] GPU path failed: {e}; falling back to CPU.")

    # ---- CPU fallback (original implementation) ----
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
        base = (cos_i * cos_j) / (np.pi * r2) * areas
        if V is not None:
            base = base * V[i, :]
        contrib[vis] = base[vis]
        F[i, :] = contrib
    return F


def enforce_reciprocity(F: np.ndarray, areas: np.ndarray) -> None:
    """Enforce Ai Fij = Aj Fji (vectorized; supports NumPy and CuPy arrays)."""
    EPS = 1e-9
    # Choose backend based on F type (np or cp)
    is_gpu = CUPY_AVAILABLE and hasattr(cp, "ndarray") and isinstance(F, cp.ndarray)
    xp = cp if is_gpu else np
    a = areas if is_gpu else areas
    if is_gpu:
        a = cp.asarray(areas)
    N = F.shape[0]
    print(f"[LinearSystem] Enforcing reciprocity on F (N={N})...")
    iu = xp.triu_indices(N, k=1)
    i, j = iu[0], iu[1]
    Ai = a[i]
    Aj = a[j]
    Fij = F[i, j]
    Fji = F[j, i]
    Tij = 0.5 * (Ai * Fij + Aj * Fji)
    F[i, j] = Tij / (Ai + EPS)
    F[j, i] = Tij / (Aj + EPS)
    # Zero diagonal
    if is_gpu:
        F[xp.arange(N), xp.arange(N)] = 0.0
    else:
        np.fill_diagonal(F, 0.0)
    print("[LinearSystem] Reciprocity enforcement complete.")


def build_linear_system(scene: BuiltScene) -> Tuple[np.ndarray, np.ndarray]:
    """Build linear system for either P0 or P1 basis functions."""
    if scene.basis_type == "P1":
        return build_linear_system_p1(scene)
    
    # P0 implementation (original)
    print("[LinearSystem] Building P0 linear system (F, E)...")
    # Visibility matrix shared across bases
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    cache_dir = PROJECT_ROOT / "linear_system_csv"
    Vmat, vtag, vpath = get_or_build_visibility(
        scene, cache_dir, skip=skip_visibility, basis_type=scene.basis_type
    )
    F = assemble_form_factor_matrix(
        scene.centers,
        scene.normals,
        scene.areas,
        V=Vmat,
    )
    enforce_reciprocity(F, scene.areas)
    E = np.zeros(scene.centers.shape[0])
    print(
        f"[LinearSystem] Built F (shape={F.shape}), initialized E (len={E.shape[0]})."
    )
    return F, E


def q1_basis_functions(xi: float, eta: float) -> np.ndarray:
    """Q1 bilinear basis functions on reference square [0,1]^2."""
    return np.array([
        (1 - xi) * (1 - eta),  # N1
        xi * (1 - eta),        # N2
        xi * eta,              # N3
        (1 - xi) * eta,        # N4
    ])


def q1_basis_gradients(xi: float, eta: float) -> np.ndarray:
    """Gradients of Q1 basis functions on [0,1]^2 for a given (xi, eta)."""
    return np.array([
        [-(1.0 - eta), -(1.0 - xi)],  # grad N1
        [1.0 - eta, -xi],             # grad N2
        [eta, xi],                    # grad N3
        [-eta, 1.0 - xi],             # grad N4
    ])


def compute_element_mass_matrix(element: P1Element) -> np.ndarray:
    """Compute 4x4 mass matrix for a Q1 element."""
    # For Q1 elements on uniform grid, the mass matrix is:
    # M = (h^2/36) * [[4,2,2,1], [2,4,1,2], [2,1,4,2], [1,2,2,4]]
    h = np.sqrt(element.area)  # Assuming square elements
    return (h * h / 36.0) * np.array([
        [4, 2, 2, 1],
        [2, 4, 1, 2],
        [2, 1, 4, 2],
        [1, 2, 2, 4]
    ])


def compute_element_albedo_matrix(element: P1Element) -> np.ndarray:
    """Compute 4x4 albedo matrix for a Q1 element."""
    # R = rho * M for constant rho per element
    rho = element.nodes[0].rho  # Assume constant rho within element
    M = compute_element_mass_matrix(element)
    return rho * M


def gauss_quadrature_2d(n_points: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """2D Gauss quadrature points and weights on [0,1]^2."""
    if n_points == 1:
        points = np.array([[0.5, 0.5]])
        weights = np.array([1.0])
    elif n_points == 2:
        # 2x2 Gauss points
        g1d = 1.0 / np.sqrt(3.0)
        points = np.array([
            [0.5 - g1d/2, 0.5 - g1d/2],
            [0.5 + g1d/2, 0.5 - g1d/2],
            [0.5 + g1d/2, 0.5 + g1d/2],
            [0.5 - g1d/2, 0.5 + g1d/2],
        ])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        # Higher order - use 3x3 for now
        g1d = np.sqrt(0.6)
        points_list = []
        weights_list = []
        for i in [-g1d, 0, g1d]:
            for j in [-g1d, 0, g1d]:
                points_list.append([0.5 + i/2, 0.5 + j/2])
                weights_list.append(5.0/81.0)
        points = np.array(points_list)
        weights = np.array(weights_list)
    
    return points, weights

def _compute_single_element_pair_kernel(args):
    """Helper function for parallel computation of element pair kernels."""
    element_i, element_j, n_quad = args
    return compute_geometric_kernel_element_pair(element_i, element_j, n_quad)

def compute_geometric_kernel_element_pair(
    element_i: P1Element, 
    element_j: P1Element,
    n_quad: int = 2,
    scene: Optional[BuiltScene] = None,
) -> np.ndarray:
    """Compute 4x4 geometric kernel matrix between two Q1 elements."""
    K = np.zeros((4, 4))
    EPS = 1e-12

    # Get quadrature points
    quad_points, quad_weights = gauss_quadrature_2d(n_quad)

    # Get element geometry
    nodes_i = element_i.nodes
    nodes_j = element_j.nodes
    pos_i = np.array([node.position for node in nodes_i])
    pos_j = np.array([node.position for node in nodes_j])
    n_i = np.array(nodes_i[0].normal)
    n_j_const = np.array(nodes_j[0].normal)

    # Precompute mapping info for quadrature points on each element
    quad_info_i = []
    for (xi_q, eta_q), w_q in zip(quad_points, quad_weights):
        N_i = q1_basis_functions(xi_q, eta_q)
        grad_i = q1_basis_gradients(xi_q, eta_q)
        x_q = N_i @ pos_i
        dpos_dxi = grad_i[:, 0] @ pos_i
        dpos_deta = grad_i[:, 1] @ pos_i
        J_i = np.linalg.norm(np.cross(dpos_dxi, dpos_deta))
        quad_info_i.append((x_q, N_i, w_q, J_i))

    quad_info_j = []
    for (xi_p, eta_p), w_p in zip(quad_points, quad_weights):
        N_j = q1_basis_functions(xi_p, eta_p)
        grad_j = q1_basis_gradients(xi_p, eta_p)
        y_p = N_j @ pos_j
        dpos_dxi = grad_j[:, 0] @ pos_j
        dpos_deta = grad_j[:, 1] @ pos_j
        J_j = np.linalg.norm(np.cross(dpos_dxi, dpos_deta))
        quad_info_j.append((y_p, N_j, w_p, J_j))

    # Double integration over both elements
    for x_q, N_i, w_q, J_i in quad_info_i:
        if J_i <= EPS:
            continue
        for y_p, N_j, w_p, J_j in quad_info_j:
            if J_j <= EPS:
                continue
            n_j = n_j_const

            # Compute geometric kernel G(x,y)
            r_vec = y_p - x_q
            r2 = np.dot(r_vec, r_vec) + EPS
            r = np.sqrt(r2)

            # Cosine terms (clamped to hemisphere)
            cos_i = max(0.0, float(np.dot(r_vec, n_i) / r))
            cos_j = max(0.0, float(-np.dot(r_vec, n_j) / r))

            # Binary visibility via geometry test (independent of cosine)
            Vflag = 1.0
            if scene is not None and is_segment_blocked(scene, x_q, y_p):
                Vflag = 0.0

            # Geometric kernel
            G = (cos_i * cos_j) / (np.pi * r2)

            # Accumulate into kernel matrix
            weight = w_q * w_p * J_i * J_j * Vflag
            if weight == 0.0:
                continue
            for a in range(4):
                for b in range(4):
                    K[a, b] += N_i[a] * G * N_j[b] * weight

    return K


def assemble_p1_mass_matrix(scene: BuiltScene) -> np.ndarray:
    """Assemble global mass matrix for P1 system."""
    n_nodes = len(scene.nodes)
    M = np.zeros((n_nodes, n_nodes))
    
    print(f"[LinearSystem] Assembling P1 mass matrix ({n_nodes}x{n_nodes})...")
    
    for element in scene.elements:
        M_elem = compute_element_mass_matrix(element)
        # Assemble into global matrix
        for i, node_i in enumerate(element.nodes):
            for j, node_j in enumerate(element.nodes):
                M[node_i.global_id, node_j.global_id] += M_elem[i, j]
    
    return M


def assemble_p1_albedo_matrix(scene: BuiltScene) -> np.ndarray:
    """Assemble global albedo matrix for P1 system."""
    n_nodes = len(scene.nodes)
    R = np.zeros((n_nodes, n_nodes))
    
    print(f"[LinearSystem] Assembling P1 albedo matrix ({n_nodes}x{n_nodes})...")
    
    for element in scene.elements:
        R_elem = compute_element_albedo_matrix(element)
        # Assemble into global matrix
        for i, node_i in enumerate(element.nodes):
            for j, node_j in enumerate(element.nodes):
                R[node_i.global_id, node_j.global_id] += R_elem[i, j]
    
    return R


def assemble_p1_geometric_kernel(scene: BuiltScene, n_workers: int = None) -> np.ndarray:
    """Assemble global geometric kernel matrix for P1 system with parallel computation."""
    n_nodes = len(scene.nodes)
    F = np.zeros((n_nodes, n_nodes))
    
    print(f"[LinearSystem] Assembling P1 geometric kernel ({n_nodes}x{n_nodes})...")
    
    # Prepare visibility matrix (shared for basis types)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    cache_dir = PROJECT_ROOT / "linear_system_csv"
    Vmat, vtag, vpath = get_or_build_visibility(
        scene, cache_dir, skip=skip_visibility, basis_type=scene.basis_type
    )

    # Helper to accumulate a local kernel block into the global matrix with visibility weight
    def accumulate(elem_i: P1Element, elem_j: P1Element, K_elem: np.ndarray) -> None:
        for a, node_a in enumerate(elem_i.nodes):
            idx_a = node_a.global_id
            for b, node_b in enumerate(elem_j.nodes):
                idx_b = node_b.global_id
                vis = Vmat[idx_a, idx_b]
                if vis == 0.0:
                    continue
                F[idx_a, idx_b] += K_elem[a, b] * vis

    # Prepare element pairs for parallel computation
    element_pairs = []
    for i, elem_i in enumerate(scene.elements):
        for j, elem_j in enumerate(scene.elements):
            element_pairs.append((elem_i, elem_j, 2))  # n_quad=2
    
    total_pairs = len(element_pairs)
    print(f"[LinearSystem] Computing {total_pairs} element pairs with parallel processing...")
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Limit to 8 to avoid memory issues
    
    # For small problems, use sequential processing
    if total_pairs < 1000:
        print(f"[LinearSystem] Small problem size ({total_pairs} pairs), using sequential processing...")
        for i, (elem_i, elem_j, n_quad) in enumerate(element_pairs):
            if tqdm is not None and i % 100 == 0:
                tqdm.write(f"Processing element pair {i}/{total_pairs}")
            K_elem = compute_geometric_kernel_element_pair(elem_i, elem_j, n_quad, scene=scene)
            accumulate(elem_i, elem_j, K_elem)
    else:
        print(f"[LinearSystem] Using {n_workers} parallel workers...")
        
        # Process element pairs in parallel
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                future_to_pair = {
                    executor.submit(_compute_single_element_pair_kernel, pair): pair 
                    for pair in element_pairs
                }
                
                # Process completed tasks with progress bar
                if tqdm is not None:
                    futures = tqdm(as_completed(future_to_pair), 
                                  total=total_pairs, 
                                  desc="P1 kernel assembly", 
                                  leave=False)
                else:
                    futures = as_completed(future_to_pair)
                
                for future in futures:
                    try:
                        K_elem = future.result()
                        pair = future_to_pair[future]
                        elem_i, elem_j, _ = pair
                        accumulate(elem_i, elem_j, K_elem)
                    except Exception as e:
                        print(f"[LinearSystem] Error computing element pair: {e}")
                        continue
        except Exception as e:
            print(f"[LinearSystem] Parallel processing failed: {e}")
            print("[LinearSystem] Falling back to sequential processing...")
            
            # Fallback to sequential processing
            for i, (elem_i, elem_j, n_quad) in enumerate(element_pairs):
                if tqdm is not None and i % 100 == 0:
                    tqdm.write(f"Processing element pair {i}/{total_pairs}")
                K_elem = compute_geometric_kernel_element_pair(elem_i, elem_j, n_quad, scene=scene)
                accumulate(elem_i, elem_j, K_elem)
    
    print(f"[LinearSystem] Kernel assembly complete.")
    return F


def assemble_p1_mass_lumped(scene: BuiltScene) -> np.ndarray:
    """Row-sum lumping of the P1 mass matrix.

    Returns m (shape [n_nodes]), where M_lumped = diag(m).
    """
    n_nodes = len(scene.nodes)
    m = np.zeros(n_nodes, dtype=np.float64)
    for element in scene.elements:
        M_elem = compute_element_mass_matrix(element)
        # add row sums of element mass into node diagonals
        row_sums = np.sum(M_elem, axis=1)
        for i_local, node_i in enumerate(element.nodes):
            m[node_i.global_id] += row_sums[i_local]
    return m


def _build_p1_lumped_system(scene: BuiltScene) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build lumped P1 components: diag(m), R_lumped = diag(rho*m), and F.

    Returns (m, rdiag, F) so that A = diag(m) - diag(rdiag) @ F.
    """
    # Lumped mass diag
    m = assemble_p1_mass_lumped(scene)
    # Lumped reflectance diag: rho per node times lumped mass
    rho_node = np.array([node.rho for node in scene.nodes], dtype=np.float64)
    rdiag = rho_node * m
    # Geometric kernel unchanged
    F = assemble_p1_geometric_kernel(scene)
    return m, rdiag, F


def _build_p1_lumped_rhs(scene: BuiltScene, *, positions, size, Le) -> np.ndarray:
    """Build lumped RHS: f = diag(m) @ E_node, where E_node at light nodes = pi*Le."""
    m = assemble_p1_mass_lumped(scene)
    mask = _compute_p1_light_mask(scene, positions, size)
    E_node = np.zeros(len(scene.nodes), dtype=np.float64)
    E_node[mask] = np.pi * Le
    return m * E_node


def build_linear_system_p1(scene: BuiltScene, n_workers: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Build P1 linear system with mass-lumping: (diag(m) - diag(rho*m) F) b = diag(m) E.

    This localizes emission on non-reflective ceilings and reduces artificial diffusion.
    """
    print("[LinearSystem] Building P1 linear system (lumped mass)...")
    # Lumped components
    m = assemble_p1_mass_lumped(scene)
    rho_node = np.array([node.rho for node in scene.nodes], dtype=np.float64)
    rdiag = rho_node * m
    F = assemble_p1_geometric_kernel(scene, n_workers)
    # System matrix A = diag(m) - diag(rdiag) @ F
    A = np.diag(m) - (rdiag[:, None] * F)
    # RHS: f = diag(m) @ E_node (built later per scenario in export func),
    # but for completeness we keep zero here; actual b's built in export.
    f = np.zeros(len(scene.nodes), dtype=np.float64)
    print(f"[LinearSystem] Built P1 (lumped) system: A={A.shape}, f={f.shape}")
    return A, f


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


# Legacy CSV export removed; use export_Ab_to_csv only.


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


# Legacy multi-E export removed; stack columns into b.csv via export_Ab_to_csv.


def _save_csv_vector(vec: np.ndarray, out_path: Path, header: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, vec, delimiter=",", header=header, comments="")
    return out_path


def _compute_p1_light_mask(scene: BuiltScene, light_positions, light_size):
    """Compute light mask for P1 nodes."""
    if not light_positions:
        return np.zeros(len(scene.nodes), dtype=bool)
    
    lx, ly = light_size
    rects = [
        (cx - lx / 2.0, cx + lx / 2.0, cy - ly / 2.0, cy + ly / 2.0)
        for (cx, cy) in light_positions
    ]
    
    is_light = np.zeros(len(scene.nodes), dtype=bool)
    count = 0
    for idx, node in enumerate(scene.nodes):
        if node.face != "ceiling":
            continue
        x, y = float(node.position[0]), float(node.position[1])
        for x0, x1, y0, y1 in rects:
            if x0 <= x <= x1 and y0 <= y <= y1:
                is_light[idx] = True
                count += 1
                break
    print(f"[LinearSystem] Computed P1 light mask: {count} light nodes.")
    return is_light


def _integrate_p1_rhs_for_lights(scene: BuiltScene, light_positions, light_size, Le: float) -> np.ndarray:
    """Assemble P1 RHS f_i = ∫ phi_i(x) E(x) dA, where E=pi*Le on ceiling light rectangles.

    We integrate over P1 elements on the ceiling using tensor-product Gauss quadrature.
    """
    n_nodes = len(scene.nodes)
    f = np.zeros(n_nodes, dtype=np.float64)
    if not light_positions or Le <= 0.0:
        return f
    lx, ly = light_size
    rects = [(cx - lx / 2.0, cx + lx / 2.0, cy - ly / 2.0, cy + ly / 2.0) for (cx, cy) in light_positions]

    # Use modest quadrature (2x2) which matches element mass matrix degree here
    quad_points, quad_weights = gauss_quadrature_2d(2)

    for elem in scene.elements or []:
        if elem.face != "ceiling":
            continue
        nodes = elem.nodes
        h = np.sqrt(elem.area)
        J = h * h  # uniform square assumption consistent with assembly

        # Map reference to world position
        def map_pos(xi, eta):
            pos = np.zeros(3)
            Ni = q1_basis_functions(xi, eta)
            for k in range(4):
                pos += Ni[k] * nodes[k].position
            return pos

        for q, (xi_q, eta_q) in enumerate(quad_points):
            wq = quad_weights[q]
            pos = map_pos(xi_q, eta_q)
            xw, yw = float(pos[0]), float(pos[1])
            inside = False
            for x0, x1, y0, y1 in rects:
                if x0 <= xw <= x1 and y0 <= yw <= y1:
                    inside = True
                    break
            if not inside:
                continue
            Ni = q1_basis_functions(xi_q, eta_q)
            for a in range(4):
                f[nodes[a].global_id] += (np.pi * Le) * Ni[a] * wq * J
    return f


def _build_M_E_for_scenario(scene: BuiltScene, F: np.ndarray, *, positions, size, Le) -> Tuple[np.ndarray, np.ndarray]:
    if scene.basis_type == "P1":
        # F is A (already lumped) from build_linear_system_p1
        A = F
        # Lumped RHS per scenario
        f = _build_p1_lumped_rhs(scene, positions=positions, size=size, Le=Le)
        return A, f
    
    # P0 implementation (original)
    # Build E and rho from light mask, then M = I - diag(rho) F
    is_light_mask = compute_ceiling_light_mask(scene, light_positions=positions, light_size=size)
    E = np.zeros(scene.centers.shape[0])
    E[is_light_mask] = np.pi * Le
    rho = np.where(is_light_mask, 0.0, scene.rho)
    M = build_system_matrix(F, rho)
    return M, E


def main_export_linear_system() -> None:
    # Build scene and F
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    out_dir = PROJECT_ROOT / "linear_system_csv"
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
    # Export canonical Ax=b only
    export_Ab_to_csv(M0, b, out_dir)
    print(f"[LinearSystem:CLI] Export complete in {out_dir}")


if __name__ == "__main__":
    main_export_linear_system()
