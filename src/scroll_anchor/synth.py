"""Synthetic multi-sheet scenes and controlled corruptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .normals import compute_normals
from .tifxyz import Surface
from .volume import VolumeROI

CLEAN, DRIFT, SWITCH, HOLE, AMBIGUOUS = 0, 1, 2, 3, 4


@dataclass
class SheetModel:
    """Curved Gaussian sheets in a ``[z, y, x]`` volume."""

    H: int
    W: int
    Dz: int
    z_base: float
    spacing: float
    n_sheets: int
    sigma: float
    curv_amp: float
    k0: int

    def center(self, x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Return the Z coordinate of sheet ``k``."""
        curv = self.curv_amp * np.sin(2 * np.pi * x / max(self.W, 1))
        return self.z_base + k * self.spacing + curv

    def sheet_id_at(self, pts_xyz: np.ndarray) -> np.ndarray:
        """Return the nearest sheet index for world points."""
        x = pts_xyz[..., 0]
        y = pts_xyz[..., 1]
        z = pts_xyz[..., 2]
        curv = self.curv_amp * np.sin(2 * np.pi * x / max(self.W, 1))
        k = np.round((z - self.z_base - curv) / self.spacing)
        return k.astype(np.int64)

    def render(self, amplitude: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
        """Render the synthetic CT volume."""
        zz = np.arange(self.Dz)[:, None, None]
        yy = np.arange(self.H)[None, :, None]
        xx = np.arange(self.W)[None, None, :]
        vol = np.zeros((self.Dz, self.H, self.W), dtype=np.float32)
        for k in range(self.n_sheets):
            c = self.center(xx, yy, k)
            vol += amplitude[None, :, :] * np.exp(-((zz - c) ** 2) / (2 * self.sigma ** 2))
        if noise > 0:
            vol += rng.normal(0.0, noise, size=vol.shape).astype(np.float32)
        return vol


@dataclass
class SyntheticScene:
    volume: VolumeROI
    clean: Surface
    corrupt: Surface
    sheet_model: SheetModel
    gt: Dict[str, np.ndarray]


def _clean_surface(model: SheetModel) -> Surface:
    rows = np.arange(model.H)
    cols = np.arange(model.W)
    xx, yy = np.meshgrid(cols, rows)
    x = xx.astype(np.float32)
    y = yy.astype(np.float32)
    z = model.center(x, y, model.k0).astype(np.float32)
    valid = np.ones((model.H, model.W), dtype=bool)
    return Surface(x=x, y=y, z=z, valid=valid, scale=(1.0, 1.0), meta={"type": "seg"})


def make_scene(
    H: int = 80,
    W: int = 80,
    spacing: float = 10.0,
    sigma: float = 1.6,
    curv_amp: float = 2.0,
    noise: float = 0.03,
    amplitude: float = 1.0,
    seed: int = 0,
    drift_delta: float = 3.0,
    ambiguous_contrast: float = 0.18,
    ambiguous_drift: float = 4.0,
) -> SyntheticScene:
    """Build a scene with localized, labeled surface corruptions."""
    rng = np.random.default_rng(seed)
    z_base = 15.0
    n_sheets = 5
    k0 = 2
    Dz = int(z_base + (n_sheets - 1) * spacing + 6 * sigma + 5)
    model = SheetModel(
        H=H, W=W, Dz=Dz, z_base=z_base, spacing=spacing, n_sheets=n_sheets,
        sigma=sigma, curv_amp=curv_amp, k0=k0,
    )

    clean = _clean_surface(model)
    normals, _ = compute_normals(clean)

    ctype = np.zeros((H, W), dtype=np.int64)
    inj = np.zeros((H, W), dtype=np.float32)

    def block(r0, r1, c0, c1):
        return (slice(r0, r1), slice(c0, c1))

    q = W // 5
    drift_zone = block(H // 6, H // 6 + H // 4, q, q + q)
    switch_zone = block(H // 2, H // 2 + H // 4, 2 * q, 3 * q)
    ambig_zone = block(H // 6, H // 6 + H // 4, 3 * q, 4 * q)
    hole_zone = block(H // 2 + H // 8, H // 2 + H // 8 + H // 8, q // 2, q // 2 + q // 2)

    ctype[drift_zone] = DRIFT
    inj[drift_zone] = drift_delta
    ctype[switch_zone] = SWITCH
    inj[switch_zone] = spacing
    ctype[ambig_zone] = AMBIGUOUS
    inj[ambig_zone] = ambiguous_drift
    ctype[hole_zone] = HOLE

    corrupt = clean.copy()
    pts = clean.points()
    disp = inj[..., None] * normals
    new_pts = pts + disp
    corrupt.x = new_pts[..., 0].astype(np.float32)
    corrupt.y = new_pts[..., 1].astype(np.float32)
    corrupt.z = new_pts[..., 2].astype(np.float32)
    corrupt.valid = clean.valid.copy()
    corrupt.valid[ctype == HOLE] = False

    amp = np.full((H, W), amplitude, dtype=np.float32)
    amp[ctype == AMBIGUOUS] = amplitude * ambiguous_contrast
    volume_arr = model.render(amp, noise=noise, rng=rng)
    volume = VolumeROI.from_array(volume_arr, origin=(0, 0, 0))

    true_sheet = np.full((H, W), k0, dtype=np.int64)
    gt = {"corruption_type": ctype, "injected_offset": inj, "true_sheet": true_sheet}
    return SyntheticScene(volume=volume, clean=clean, corrupt=corrupt, sheet_model=model, gt=gt)
