"""Real Scroll-1 cube benchmark: medial-surface extraction + controlled corruptions.

Uses real CT intensities and real volumetric sheet-instance geometry. Corruptions
(drift, neighbouring-sheet switch) are injected deliberately; this is *not*
validation on naturally occurring annotation errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .normals import compute_normals
from .tifxyz import Surface
from .volume import VolumeROI


# --------------------------------------------------------------------------- #
# Instance inspection and ROI selection
# --------------------------------------------------------------------------- #
def instance_report(mask: np.ndarray, min_voxels: int = 2000) -> Dict[str, object]:
    """Summarise nonzero instances and their pairwise adjacency."""
    ids, counts = np.unique(mask[mask > 0], return_counts=True)
    instances = []
    for idv, cnt in zip(ids.tolist(), counts.tolist()):
        if cnt < min_voxels:
            continue
        zz, yy, xx = np.where(mask == idv)
        _, n_cc = ndi.label(mask == idv)
        instances.append(
            {
                "id": int(idv),
                "voxels": int(cnt),
                "bbox_zyx": [int(zz.min()), int(zz.max()), int(yy.min()),
                             int(yy.max()), int(xx.min()), int(xx.max())],
                "n_components": int(n_cc),
                "touches_border": bool(
                    zz.min() == 0 or yy.min() == 0 or xx.min() == 0
                    or zz.max() == mask.shape[0] - 1 or yy.max() == mask.shape[1] - 1
                    or xx.max() == mask.shape[2] - 1
                ),
            }
        )
    instances.sort(key=lambda d: d["voxels"], reverse=True)

    pairs: Dict[Tuple[int, int], int] = {}
    keep_ids = {d["id"] for d in instances}
    for d in instances:
        idv = d["id"]
        m = mask == idv
        shell = ndi.binary_dilation(m, iterations=3) & ~m
        nb, cnt = np.unique(mask[shell], return_counts=True)
        for n, c in zip(nb.tolist(), cnt.tolist()):
            if n > 0 and n != idv and n in keep_ids:
                key = (min(idv, n), max(idv, n))
                pairs[key] = pairs.get(key, 0) + int(c)
    pair_list = [{"pair": [a, b], "contact_voxels": c} for (a, b), c in pairs.items()]
    pair_list.sort(key=lambda d: d["contact_voxels"], reverse=True)
    return {"instances": instances, "adjacent_pairs": pair_list}


@dataclass
class RoiSelection:
    source_id: int
    target_id: int
    roi_origin: Tuple[int, int, int]     # (z0, y0, x0) in cube indices
    roi_size: int
    source_voxels: int
    target_voxels: int
    sep_median: float
    sep_p10: float


def select_pair_and_roi(
    mask: np.ndarray,
    roi_size: int = 96,
    dilation: int = 3,
    min_switch_area: int = 200,
) -> RoiSelection:
    """Pick a source/target pair and a compact ROI.

    Selection maximises the *reliable switch-constructible area*: grid vertices
    where both sheets have an unambiguous single-run medial surface and the source
    lies in its largest connected patch. This favours a locally flat, well-supported
    region rather than merely close or large instances.
    """
    report = instance_report(mask)
    best: Optional[RoiSelection] = None
    best_area = -1
    for pd in report["adjacent_pairs"]:
        a, b = pd["pair"]
        ca, cb = int((mask == a).sum()), int((mask == b).sum())
        S, T = (a, b) if ca >= cb else (b, a)
        sel = _roi_for_pair(mask, S, T, roi_size, dilation)
        if sel is None:
            continue
        z0, y0, x0 = sel.roi_origin
        sl = (slice(z0, z0 + roi_size), slice(y0, y0 + roi_size), slice(x0, x0 + roi_size))
        ms = extract_medial_surface(mask[sl] == S, sel.roi_origin)
        _, valid_t = _medial_along(mask[sl] == T, ms.proj_axis)
        area = int((ms.valid & valid_t).sum())
        if area < min_switch_area:
            continue
        if area > best_area:
            best_area, best = area, sel
    if best is None:
        raise RuntimeError("no valid neighbouring pair supports the switch benchmark")
    return best


def _roi_for_pair(mask, S, T, roi_size, dilation) -> Optional[RoiSelection]:
    ms = mask == S
    mt = mask == T
    contact = ndi.binary_dilation(ms, iterations=dilation) & mt
    if not contact.any():
        return None
    cz, cy, cx = (int(round(v)) for v in np.array(np.where(contact)).mean(1))

    def clip(c, dim):
        return int(min(max(c - roi_size // 2, 0), dim - roi_size))

    z0 = clip(cz, mask.shape[0])
    y0 = clip(cy, mask.shape[1])
    x0 = clip(cx, mask.shape[2])
    sl = (slice(z0, z0 + roi_size), slice(y0, y0 + roi_size), slice(x0, x0 + roi_size))
    sub_s = ms[sl]
    sub_t = mt[sl]
    if not sub_s.any() or not sub_t.any():
        return None
    edt = ndi.distance_transform_edt(~sub_s)
    sep = edt[sub_t]
    return RoiSelection(
        source_id=int(S), target_id=int(T), roi_origin=(z0, y0, x0), roi_size=int(roi_size),
        source_voxels=int(sub_s.sum()), target_voxels=int(sub_t.sum()),
        sep_median=float(np.median(sep)), sep_p10=float(np.percentile(sep, 10)),
    )


# --------------------------------------------------------------------------- #
# Medial-surface extraction
# --------------------------------------------------------------------------- #
def _medial_along(instance_roi: np.ndarray, proj_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (depth, single_run) for a projection axis.

    A grid column is valid only if the labelled sheet is a single contiguous run
    along ``proj_axis`` (unambiguous medial crossing); the medial depth is the run's
    centroid index.
    """
    mm = np.moveaxis(instance_roi, proj_axis, 0)  # (depth, row, col)
    any_col = mm.any(0)
    starts = (np.diff(mm.astype(np.int8), axis=0) == 1).sum(0) + mm[0].astype(int)
    single = (starts == 1) & any_col
    idx = np.arange(mm.shape[0])[:, None, None]
    depth = (mm * idx).sum(0) / np.clip(mm.sum(0), 1, None)
    return depth.astype(np.float32), single


def _largest_component(valid: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(valid)
    if n == 0:
        return valid
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return lab == keep


@dataclass
class MedialSurface:
    depth: np.ndarray            # (H, W) medial index along proj_axis (ROI-local)
    valid: np.ndarray            # (H, W) bool, reliable single-run + largest CC
    proj_axis: int
    roi_origin: Tuple[int, int, int]

    def grid_axes(self) -> Tuple[int, int]:
        return tuple(a for a in range(3) if a != self.proj_axis)  # type: ignore[return-value]


def extract_medial_surface(
    instance_roi: np.ndarray,
    roi_origin: Tuple[int, int, int],
    proj_axis: Optional[int] = None,
) -> MedialSurface:
    """Extract a single-valued medial surface; choose projection axis from geometry."""
    if proj_axis is None:
        best_axis, best_valid, best_size = 0, None, -1
        for p in range(3):
            _, single = _medial_along(instance_roi, p)
            cc = _largest_component(single)
            if cc.sum() > best_size:
                best_axis, best_valid, best_size = p, cc, int(cc.sum())
        proj_axis = best_axis
        depth, _ = _medial_along(instance_roi, proj_axis)
        valid = best_valid
    else:
        depth, single = _medial_along(instance_roi, proj_axis)
        valid = _largest_component(single)
    return MedialSurface(depth=depth, valid=valid, proj_axis=proj_axis, roi_origin=roi_origin)


def medial_depth_on_grid(instance_roi: np.ndarray, proj_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Medial depth + single-run validity for a fixed projection axis (no CC filter)."""
    return _medial_along(instance_roi, proj_axis)


def _grid_to_cube_coords(depth, proj_axis, roi_origin):
    """Map (row, col, depth) grid to cube-index (X, Y, Z) arrays.

    Internal axes are (0=z, 1=y, 2=x); world coords equal cube indices so the ROI
    subvolume can be sampled directly with ``VolumeROI(origin=roi_origin)``.
    """
    H, W = depth.shape
    rem = [a for a in range(3) if a != proj_axis]  # ascending
    rows = np.broadcast_to(np.arange(H)[:, None], (H, W)).astype(np.float32)
    cols = np.broadcast_to(np.arange(W)[None, :], (H, W)).astype(np.float32)
    coord = [None, None, None]
    coord[proj_axis] = roi_origin[proj_axis] + depth
    coord[rem[0]] = roi_origin[rem[0]] + rows
    coord[rem[1]] = roi_origin[rem[1]] + cols
    Z, Y, X = coord[0], coord[1], coord[2]
    return X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32)


def medial_to_surface(ms: MedialSurface) -> Surface:
    """Build a tifxyz Surface (cube-index world frame) from a medial surface."""
    x, y, z = _grid_to_cube_coords(ms.depth, ms.proj_axis, ms.roi_origin)
    valid = ms.valid.copy()
    meta = {
        "type": "seg",
        "source": "real_cube_medial",
        "proj_axis": int(ms.proj_axis),
        "roi_origin_zyx": [int(v) for v in ms.roi_origin],
    }
    return Surface(x=x, y=y, z=z, valid=valid, scale=(1.0, 1.0), meta=meta)


def roi_volume(ct_roi: np.ndarray, roi_origin: Tuple[int, int, int]) -> VolumeROI:
    """Wrap a cropped CT ROI so its cube-index frame matches the surface."""
    return VolumeROI.from_array(ct_roi.astype(np.float32), origin=tuple(int(v) for v in roi_origin))


# --------------------------------------------------------------------------- #
# Controlled corruptions
# --------------------------------------------------------------------------- #
@dataclass
class Corruption:
    surface: Surface
    region: np.ndarray                 # (H, W) bool corrupted patch
    injected_offset: np.ndarray        # (H, W) signed voxels (drift); 0 elsewhere
    info: Dict[str, object] = field(default_factory=dict)


def _compact_patch(valid: np.ndarray, center_frac, half, rng=None) -> np.ndarray:
    """A square patch of valid vertices around a fractional grid centre."""
    H, W = valid.shape
    cr = int(center_frac[0] * H)
    cc = int(center_frac[1] * W)
    patch = np.zeros_like(valid)
    patch[max(0, cr - half): cr + half, max(0, cc - half): cc + half] = True
    return patch & valid


def make_drift(surface: Surface, offset: float, half: int = 8,
               center_frac=(0.35, 0.35)) -> Corruption:
    """Move a compact patch along real local normals by a known signed offset."""
    normals, nvalid = compute_normals(surface)
    region = _compact_patch(surface.valid & nvalid, center_frac, half)
    inj = np.zeros(surface.shape, dtype=np.float32)
    inj[region] = offset
    corrupt = surface.copy()
    pts = corrupt.points() + inj[..., None] * normals
    corrupt.x, corrupt.y, corrupt.z = (pts[..., 0].astype(np.float32),
                                       pts[..., 1].astype(np.float32),
                                       pts[..., 2].astype(np.float32))
    disp = np.abs(inj)
    info = {
        "type": "drift",
        "offset_voxels": float(offset),
        "n_vertices": int(region.sum()),
        "drift_magnitude_mean": float(disp[region].mean()) if region.any() else 0.0,
    }
    return Corruption(surface=corrupt, region=region, injected_offset=inj, info=info)


def make_switch(
    surface: Surface,
    target_ms: MedialSurface,
    mask_roi: np.ndarray,
    target_id: int,
    source_id: int,
    half: int = 8,
    center_frac=(0.5, 0.5),
) -> Corruption:
    """Replace a contiguous patch with the neighbour instance's medial surface.

    Only vertices where both sheets have a valid medial surface and the target
    coordinate genuinely lands on the target instance are switched.
    """
    tx, ty, tz = _grid_to_cube_coords(target_ms.depth, target_ms.proj_axis, target_ms.roi_origin)
    overlap = surface.valid & target_ms.valid
    region = _compact_patch(overlap, center_frac, half)

    z0, y0, x0 = target_ms.roi_origin
    corrupt = surface.copy()
    verified = np.zeros(surface.shape, dtype=bool)
    disp = np.zeros(surface.shape, dtype=np.float32)
    rows, cols = np.where(region)
    for r, c in zip(rows.tolist(), cols.tolist()):
        zi = int(round(tz[r, c] - z0)); yi = int(round(ty[r, c] - y0)); xi = int(round(tx[r, c] - x0))
        if not (0 <= zi < mask_roi.shape[0] and 0 <= yi < mask_roi.shape[1]
                and 0 <= xi < mask_roi.shape[2]):
            continue
        if mask_roi[zi, yi, xi] != target_id:
            continue  # reject correspondences that miss the target sheet
        d = float(np.hypot(np.hypot(tx[r, c] - corrupt.x[r, c], ty[r, c] - corrupt.y[r, c]),
                           tz[r, c] - corrupt.z[r, c]))
        corrupt.x[r, c], corrupt.y[r, c], corrupt.z[r, c] = tx[r, c], ty[r, c], tz[r, c]
        verified[r, c] = True
        disp[r, c] = d

    info = {
        "type": "switch",
        "source_id": int(source_id),
        "target_id": int(target_id),
        "n_vertices": int(verified.sum()),
        "n_rejected": int(region.sum() - verified.sum()),
        "true_displacement_mean": float(disp[verified].mean()) if verified.any() else 0.0,
    }
    return Corruption(surface=corrupt, region=verified, injected_offset=disp, info=info)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _prf(pred, true, mask):
    pred = pred & mask
    true = true & mask
    tp = int(np.sum(pred & true)); fp = int(np.sum(pred & ~true)); fn = int(np.sum(~pred & true))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def sample_instance_at(surface: Surface, mask_vol: VolumeROI) -> np.ndarray:
    """Nearest-neighbour instance id under each surface vertex."""
    ids = mask_vol.sample_world(surface.points(), order=0, cval=0.0)
    return np.rint(ids).astype(np.int64)


def clean_metrics(diag, valid) -> Dict[str, object]:
    conf = diag.confidence[valid]
    return {
        "review_fraction": float(diag.review[valid].mean()),
        "switch_positive_fraction": float((diag.switch_score[valid] >= 0.5).mean()),
        "drift_flagged_fraction": float((diag.drift_score[valid] > 0).mean()),
        "confidence_mean": float(conf.mean()),
        "confidence_p10": float(np.percentile(conf, 10)),
        "clean_stability": float((valid & ~diag.review)[valid].mean()),
        "estimated_spacing": float(diag.estimated_spacing),
    }


def drift_metrics(diag, region, inj, valid, drift_min) -> Dict[str, object]:
    pred_switch = diag.switch_score >= 0.5
    pred = (diag.drift_score >= drift_min) & ~pred_switch
    prf = _prf(pred, region, valid)
    dmask = region & valid & np.isfinite(diag.chosen_offset)
    if dmask.any():
        mae = float(np.mean(np.abs(diag.chosen_offset[dmask] + inj[dmask])))
        sign = float(np.mean(np.sign(diag.chosen_offset[dmask]) == np.sign(-inj[dmask])))
    else:
        mae, sign = float("nan"), float("nan")
    outside = valid & ~region
    fpr = float(np.mean(pred[outside])) if outside.any() else 0.0
    return {**prf, "displacement_mae": mae, "sign_accuracy": sign, "fp_rate_outside": fpr}


def switch_metrics(diag, region, valid, on_wrong_instance) -> Dict[str, object]:
    pred = diag.switch_score >= 0.5
    prf = _prf(pred, region, valid)
    review_recall = float(diag.review[region & valid].mean()) if (region & valid).any() else 1.0
    outside = valid & ~region
    fpr = float(np.mean(pred[outside])) if outside.any() else 0.0
    accepted = valid & ~diag.review
    acc_wrong = accepted & on_wrong_instance
    harmful = float(acc_wrong.sum() / max(1, accepted.sum()))
    return {
        **prf,
        "review_recall": review_recall,
        "fp_rate_outside": fpr,
        "harmful_acceptance_rate": harmful,
        "accepted_on_wrong_instance": int(acc_wrong.sum()),
        "accepted_total": int(accepted.sum()),
    }
