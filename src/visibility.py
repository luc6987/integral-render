import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np  # type: ignore

try:
    # Relative import when used inside package
    from .scene import BuiltScene
except Exception:  # pragma: no cover
    # Fallback for running this file directly
    from src.scene import BuiltScene


def _hash_scene_geometry(scene: BuiltScene, basis_type: str) -> str:
    """Return a short hash fingerprint for visibility cache based on geometry.

    Uses centers, normals, prisms bounds, basis, element counts when P1.
    """
    h = hashlib.sha256()
    h.update(basis_type.encode("utf-8"))
    h.update(np.asarray(scene.centers, dtype=np.float64).tobytes())
    h.update(np.asarray(scene.normals, dtype=np.float64).tobytes())
    # Include prisms bounds and cube bounds to capture occluders
    for k in sorted(scene.cube_bounds.keys()):
        h.update(str(k).encode("utf-8"))
        h.update(str(scene.cube_bounds[k]).encode("utf-8"))
    for pb_name, pb in sorted(scene.prism_bounds_map.items()):
        h.update(pb_name.encode("utf-8"))
        for fld in ("x0", "x1", "y0", "y1", "z0", "z1"):
            h.update(str(getattr(pb, fld)).encode("utf-8"))
    if scene.basis_type == "P1":
        # Include number of nodes/elements
        n_nodes = 0 if scene.nodes is None else len(scene.nodes)
        n_elems = 0 if scene.elements is None else len(scene.elements)
        h.update(
            f"n_nodes={n_nodes},n_elems={n_elems}".encode("utf-8")
        )
    return h.hexdigest()[:16]


def _ray_intersects_aabb(
    p0: np.ndarray,
    p1: np.ndarray,
    bounds: Tuple[float, float, float, float, float, float],
) -> bool:
    """Check if segment p0->p1 intersects axis-aligned box defined by
    bounds=(x0,x1,y0,y1,z0,z1).

    Uses slab method for a line segment.
    """
    x0, x1, y0, y1, z0, z1 = bounds
    d = p1 - p0
    tmin = 0.0
    tmax = 1.0
    for a, bmin, bmax in ((0, x0, x1), (1, y0, y1), (2, z0, z1)):
        if abs(d[a]) < 1e-12:
            if p0[a] < bmin or p0[a] > bmax:
                return False
        else:
            inv = 1.0 / d[a]
            t1 = (bmin - p0[a]) * inv
            t2 = (bmax - p0[a]) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def _segment_blocked_by_any_prism(
    p0: np.ndarray, p1: np.ndarray, scene: BuiltScene
) -> bool:
    """Return True if the segment p0->p1 intersects any occluding prism or cube.

    Endpoints are excluded by shrinking segment slightly.
    """
    eps = 1e-8
    p0s = (1.0 - eps) * p0 + eps * p1
    p1s = (1.0 - eps) * p1 + eps * p0

    # Room six faces as a hollow box do not occlude; only internal solids
    # (cube + prisms) occlude
    # Cube bounds
    cb = scene.cube_bounds
    cube_bounds = (
        cb["cube_x0"],
        cb["cube_x1"],
        cb["cube_y0"],
        cb["cube_y1"],
        cb["cube_z0"],
        cb["cube_z1"],
    )
    if _ray_intersects_aabb(p0s, p1s, cube_bounds):
        return True
    # Extra prisms
    for pb in scene.prisms:
        b = (pb.x0, pb.x1, pb.y0, pb.y1, pb.z0, pb.z1)
        if _ray_intersects_aabb(p0s, p1s, b):
            return True
    return False


def is_segment_blocked(scene: BuiltScene, p0: np.ndarray, p1: np.ndarray) -> bool:
    """Public helper: return True if segment p0->p1 is blocked by cube/prisms."""
    return _segment_blocked_by_any_prism(p0, p1, scene)


def compute_visibility(
    scene: BuiltScene, *, use_cosine_hemi: bool = True, skip: bool = False
) -> np.ndarray:
    """Compute binary visibility V (NxN) between centers (P0) or nodes (P1).

    - If skip is True: returns all-ones except diagonal zeros.
    - Uses cosine-hemisphere check (dot>0 on both) when use_cosine_hemi.
    - Blocks if segment between points intersects cube or any extra prisms.
    """
    N = scene.centers.shape[0]
    V = np.ones((N, N), dtype=np.float64)
    np.fill_diagonal(V, 0.0)
    if skip:
        return V
    # Cosine hemisphere gating
    normals = scene.normals
    centers = scene.centers
    for i in range(N):
        pi = centers[i]
        ni = normals[i]
        # vector to all j
        v = centers - pi
        r = np.linalg.norm(v, axis=1) + 1e-12
        wi = v / r[:, None]
        cos_i = wi @ ni
        cos_j = -np.einsum("ij,ij->i", wi, normals)
        mask = (cos_i > 0) & (cos_j > 0)
        mask[i] = False
        for j in np.where(mask)[0]:
            if _segment_blocked_by_any_prism(pi, centers[j], scene):
                V[i, j] = 0.0
            else:
                V[i, j] = 1.0
        # others already default to 0 or 1 depending on mask; set non-mask to 0
        for j in np.where(~mask)[0]:
            V[i, j] = 0.0
    return V


def save_visibility(V: np.ndarray, out_dir: Path, *, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"V_{tag}.npy"
    np.save(path, V)
    # Per-tag meta
    meta = {"tag": tag, "shape": V.shape}
    with open(out_dir / f"visibility_{tag}.json", "w") as f:
        json.dump(meta, f)
    # Aggregate stats for quick inspection
    total = int(V.size)
    blocked = int((V == 0.0).sum() - V.shape[0])  # exclude diagonal zeros
    visible = total - blocked - V.shape[0]
    stats = {
        "tag": tag,
        "nodes": int(V.shape[0]),
        "pairs_total": total,
        "pairs_visible": visible,
        "pairs_blocked": blocked,
        "visible_ratio": float(visible / max(1, (total - V.shape[0]))),
    }
    with open(out_dir / "visibility_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return path


def load_visibility(out_dir: Path, *, tag: str) -> Optional[np.ndarray]:
    path = out_dir / f"V_{tag}.npy"
    if path.exists():
        try:
            return np.load(path)
        except Exception:
            return None
    return None


def get_or_build_visibility(
    scene: BuiltScene, cache_dir: Path, *, skip: bool, basis_type: str
) -> Tuple[np.ndarray, str, Path]:
    """Load V from cache or compute and save it. Returns (V, tag, path)."""
    tag = _hash_scene_geometry(scene, basis_type) + ("_skip" if skip else "")
    V = load_visibility(cache_dir, tag=tag)
    if V is not None:
        return V, tag, cache_dir / f"V_{tag}.npy"
    V = compute_visibility(scene, skip=skip)
    path = save_visibility(V, cache_dir, tag=tag)
    return V, tag, path

