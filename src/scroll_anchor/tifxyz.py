"""Dependency-light tifxyz surface I/O."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile


@dataclass
class Surface:
    """Quad-grid world coordinates and validity mask."""

    x: np.ndarray  # (H, W) float32, world X
    y: np.ndarray  # (H, W) float32, world Y
    z: np.ndarray  # (H, W) float32, world Z
    valid: np.ndarray  # (H, W) bool
    scale: Tuple[float, float] = (1.0, 1.0)  # (sx, sy) grid spacing in surface units
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.x.shape  # type: ignore[return-value]

    def points(self) -> np.ndarray:
        """Return stacked world coordinates in ``(X, Y, Z)`` order."""
        return np.stack([self.x, self.y, self.z], axis=-1).astype(np.float32)

    def copy(self) -> "Surface":
        return Surface(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            valid=self.valid.copy(),
            scale=self.scale,
            meta=dict(self.meta),
        )


def _read_coord(path: str) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected single-channel 2D TIFF, got shape {arr.shape}")
    return arr.astype(np.float32)


def read_tifxyz(directory: str, use_mask: bool = True) -> Surface:
    """Read a tifxyz directory and apply its validity rules."""
    meta_path = os.path.join(directory, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"missing meta.json in {directory}")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    x = _read_coord(os.path.join(directory, "x.tif"))
    y = _read_coord(os.path.join(directory, "y.tif"))
    z = _read_coord(os.path.join(directory, "z.tif"))
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x/y/z.tif shapes differ")

    valid = z > 0

    mask_path = os.path.join(directory, "mask.tif")
    if use_mask and os.path.isfile(mask_path):
        mask = tifffile.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape == x.shape:
            valid &= mask >= 255
        else:
            sy, sx = mask.shape[0] / x.shape[0], mask.shape[1] / x.shape[1]
            if sy.is_integer() and sx.is_integer() and sy >= 1 and sx >= 1:
                sub = mask[:: int(sy), :: int(sx)][: x.shape[0], : x.shape[1]]
                valid &= sub >= 255

    scale = tuple(meta.get("scale", [1.0, 1.0]))  # type: ignore[assignment]
    return Surface(x=x, y=y, z=z, valid=valid, scale=(float(scale[0]), float(scale[1])), meta=meta)


def write_tifxyz(
    directory: str,
    surface: Surface,
    extra_channels: Optional[Dict[str, np.ndarray]] = None,
    overwrite: bool = False,
) -> None:
    """Write a tifxyz directory with optional per-vertex channels."""
    if os.path.exists(directory) and not overwrite and os.listdir(directory):
        raise FileExistsError(f"{directory} exists and is not empty (use overwrite=True)")
    os.makedirs(directory, exist_ok=True)

    x = surface.x.astype(np.float32).copy()
    y = surface.y.astype(np.float32).copy()
    z = surface.z.astype(np.float32).copy()
    invalid = ~surface.valid
    x[invalid] = -1.0
    y[invalid] = -1.0
    z[invalid] = -1.0

    tifffile.imwrite(os.path.join(directory, "x.tif"), x)
    tifffile.imwrite(os.path.join(directory, "y.tif"), y)
    tifffile.imwrite(os.path.join(directory, "z.tif"), z)
    mask = np.where(surface.valid, np.uint8(255), np.uint8(0))
    tifffile.imwrite(os.path.join(directory, "mask.tif"), mask)

    for name, arr in (extra_channels or {}).items():
        tifffile.imwrite(os.path.join(directory, f"{name}.tif"), np.asarray(arr))

    meta = dict(surface.meta)
    meta["format"] = "tifxyz"
    meta["scale"] = [float(surface.scale[0]), float(surface.scale[1])]
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def list_extra_channels(directory: str) -> List[str]:
    """List optional per-vertex channel names."""
    core = {"x", "y", "z", "mask", "generations"}
    out = []
    for fn in sorted(os.listdir(directory)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in (".tif", ".tiff") and stem not in core:
            out.append(stem)
    return out
