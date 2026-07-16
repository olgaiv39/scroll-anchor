"""Optional NRRD ingestion for the real-cube benchmark.

Resolves stored NRRD axes to ScrollAnchor's internal ``[z, y, x]`` order and
validates CT/mask alignment. Never transposes silently: ambiguous metadata is a
hard error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _require_nrrd():
    try:
        import nrrd  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pynrrd is required for NRRD ingestion; install with the 'benchmark' extra"
        ) from exc
    import nrrd

    return nrrd


@dataclass
class NrrdVolume:
    """A cube resolved to internal ``[z, y, x]`` order with recorded geometry."""

    data: np.ndarray
    spacing: Tuple[float, float, float]   # (z, y, x) world units per voxel
    origin: Tuple[float, float, float]    # scroll-world (z, y, x) of voxel [0,0,0]
    axis_perm: Tuple[int, int, int]       # stored array axis -> internal axis
    header: Dict[str, object]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]


def header_summary(header: Dict[str, object]) -> Dict[str, object]:
    """JSON-serialisable view of an NRRD header (arrays -> lists)."""
    out: Dict[str, object] = {}
    for k, v in header.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _resolve_perm(header: Dict[str, object]) -> Tuple[int, int, int]:
    """Map each stored array axis to an internal axis (0=z, 1=y, 2=x).

    The volumetric-instance-label cubes store world components in ``(z, y, x)``
    order (folder name ``z_y_x``; ``space origin`` ordered to match). We therefore
    treat stored world component *i* as internal axis *i* and only need to undo any
    axis permutation encoded in ``space directions``.
    """
    sd = header.get("space directions")
    if sd is None:
        raise ValueError("NRRD header lacks 'space directions'; axis order is ambiguous")
    sd = np.asarray(sd, dtype=float)
    if sd.shape != (3, 3):
        raise ValueError(f"expected 3x3 'space directions', got {sd.shape}")

    perm = []
    for a in range(3):
        row = np.abs(sd[a])
        order = np.argsort(row)[::-1]
        dominant, second = row[order[0]], row[order[1]]
        if dominant <= 0 or second > 0.1 * dominant:
            raise ValueError(
                "non-axis-aligned 'space directions'; refusing to transpose ambiguous data"
            )
        perm.append(int(order[0]))
    if sorted(perm) != [0, 1, 2]:
        raise ValueError(f"'space directions' axes are not a permutation: {perm}")
    return tuple(perm)  # type: ignore[return-value]


def read_nrrd(path: str) -> NrrdVolume:
    """Read one NRRD file and return it in internal ``[z, y, x]`` order."""
    nrrd = _require_nrrd()
    data, header = nrrd.read(path)
    data = np.ascontiguousarray(data)
    perm = _resolve_perm(header)

    # order[i] = the stored axis that carries internal axis i.
    order = [perm.index(i) for i in range(3)]
    data_internal = np.ascontiguousarray(np.transpose(data, axes=order))

    sd = np.abs(np.asarray(header["space directions"], dtype=float))
    stored_spacing = [float(np.linalg.norm(sd[a])) for a in range(3)]
    origin_raw = header.get("space origin")
    stored_origin = (
        [float(v) for v in np.asarray(origin_raw, dtype=float)]
        if origin_raw is not None
        else [0.0, 0.0, 0.0]
    )
    # Internal axis i carries world component i (stored order is z, y, x).
    spacing = tuple(stored_spacing[order[i]] for i in range(3))
    origin = tuple(stored_origin[i] for i in range(3))

    return NrrdVolume(
        data=data_internal,
        spacing=spacing,  # type: ignore[arg-type]
        origin=origin,  # type: ignore[arg-type]
        axis_perm=perm,
        header=header,
    )


def load_cube(volume_path: str, mask_path: str) -> Tuple[NrrdVolume, NrrdVolume]:
    """Load CT + instance mask and verify they share the same voxel grid."""
    vol = read_nrrd(volume_path)
    mask = read_nrrd(mask_path)
    if vol.shape != mask.shape:
        raise ValueError(f"CT/mask shape mismatch: {vol.shape} vs {mask.shape}")
    if vol.axis_perm != mask.axis_perm:
        raise ValueError(f"CT/mask axis order mismatch: {vol.axis_perm} vs {mask.axis_perm}")
    if not np.allclose(vol.spacing, mask.spacing, atol=1e-6):
        raise ValueError(f"CT/mask spacing mismatch: {vol.spacing} vs {mask.spacing}")
    if not np.allclose(vol.origin, mask.origin, atol=1e-3):
        raise ValueError(f"CT/mask origin mismatch: {vol.origin} vs {mask.origin}")
    return vol, mask
