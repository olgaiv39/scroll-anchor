"""Surface-normal estimation for tifxyz grids."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .tifxyz import Surface


def compute_normals(surface: Surface, consistent_sign: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate unit normals and a validity mask in world coordinates."""
    pts = surface.points().astype(np.float64)
    valid = surface.valid

    p = pts.copy()
    p[~valid] = np.nan

    t_col = np.gradient(p, axis=1)
    t_row = np.gradient(p, axis=0)

    n = np.cross(t_col, t_row)
    norm = np.linalg.norm(n, axis=-1)
    normal_valid = valid & np.isfinite(norm) & (norm > 1e-8)

    normals = np.zeros_like(n, dtype=np.float32)
    safe = norm.copy()
    safe[~normal_valid] = 1.0
    normals = (n / safe[..., None]).astype(np.float32)
    normals[~normal_valid] = 0.0

    if consistent_sign and normal_valid.any():
        mean_n = normals[normal_valid].mean(axis=0)
        mn = np.linalg.norm(mean_n)
        if mn > 1e-8:
            mean_n = mean_n / mn
            flip = (normals @ mean_n) < 0
            normals[flip] *= -1.0
            normals[~normal_valid] = 0.0

    return normals, normal_valid
