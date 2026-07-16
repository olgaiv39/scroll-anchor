"""Bounded CT volume access and interpolation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class VolumeROI:
    """A bounded ``[z, y, x]`` volume with a world-space origin."""

    data: np.ndarray
    origin: Tuple[int, int, int] = (0, 0, 0)

    @classmethod
    def from_array(cls, data: np.ndarray, origin: Tuple[int, int, int] = (0, 0, 0)) -> "VolumeROI":
        if data.ndim != 3:
            raise ValueError(f"volume must be 3D [z,y,x], got shape {data.shape}")
        return cls(data=np.ascontiguousarray(data), origin=tuple(int(o) for o in origin))

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    def sample_world(
        self,
        pts_xyz: np.ndarray,
        order: int = 1,
        cval: float = 0.0,
    ) -> np.ndarray:
        """Sample world points in ``(X, Y, Z)`` order."""
        pts = np.asarray(pts_xyz, dtype=np.float64)
        if pts.shape[-1] != 3:
            raise ValueError("pts_xyz last dim must be 3 (X, Y, Z)")
        lead = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        z0, y0, x0 = self.origin
        coords = np.empty((3, flat.shape[0]), dtype=np.float64)
        coords[0] = flat[:, 2] - z0
        coords[1] = flat[:, 1] - y0
        coords[2] = flat[:, 0] - x0
        vals = map_coordinates(
            self.data, coords, order=order, mode="constant", cval=cval, prefilter=(order > 1)
        )
        return vals.reshape(lead).astype(np.float32)


def load_zarr_roi(
    array,
    bbox_xyz: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    margin: int = 0,
) -> VolumeROI:
    """Load a bounded ROI from a chunked ``[z, y, x]`` array."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox_xyz
    Dz, Dy, Dx = array.shape
    z0 = max(0, int(np.floor(zmin)) - margin)
    y0 = max(0, int(np.floor(ymin)) - margin)
    x0 = max(0, int(np.floor(xmin)) - margin)
    z1 = min(Dz, int(np.ceil(zmax)) + 1 + margin)
    y1 = min(Dy, int(np.ceil(ymax)) + 1 + margin)
    x1 = min(Dx, int(np.ceil(xmax)) + 1 + margin)
    block = np.asarray(array[z0:z1, y0:y1, x0:x1])
    return VolumeROI.from_array(block, origin=(z0, y0, x0))


def open_zarr(path: str):
    """Open a local or remote zarr array."""
    try:
        import zarr  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "zarr is required for remote/zarr access; install with the 'remote' extra"
        ) from exc
    import zarr

    return zarr.open(path, mode="r")
