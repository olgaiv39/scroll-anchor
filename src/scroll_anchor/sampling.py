"""Normal-profile sampling for CT volumes."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import SamplingConfig
from .logging_setup import get_logger
from .volume import VolumeROI

log = get_logger(__name__)


def offset_axis(cfg: SamplingConfig) -> np.ndarray:
    """Return signed offsets along a normal ray."""
    n = int(np.floor(cfg.radius / cfg.step))
    return np.arange(-n, n + 1, dtype=np.float32) * cfg.step


def sample_profiles(
    points_xyz: np.ndarray,
    normals: np.ndarray,
    volume: VolumeROI,
    cfg: SamplingConfig,
    chunk_rows: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample chunked intensity profiles along per-vertex normals."""
    H, W, _ = points_xyz.shape
    offsets = offset_axis(cfg)
    T = offsets.shape[0]
    profiles = np.zeros((H, W, T), dtype=np.float32)

    for r0 in range(0, H, chunk_rows):
        r1 = min(H, r0 + chunk_rows)
        p = points_xyz[r0:r1]
        n = normals[r0:r1]
        rays = p[:, :, None, :] + offsets[None, None, :, None] * n[:, :, None, :]
        vals = volume.sample_world(rays, order=cfg.order, cval=cfg.cval)
        profiles[r0:r1] = vals.astype(np.float32)
        log.debug("sampled rows %d:%d (%d x %d x %d)", r0, r1, r1 - r0, W, T)

    return profiles, offsets
