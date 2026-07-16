"""Drift and sheet-switch diagnostics from normal CT profiles"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import find_peaks

from .config import DiagnosticsConfig

@dataclass
class Diagnostics:
    valid: np.ndarray            # (H, W) bool
    chosen_offset: np.ndarray    # (H, W) signed voxels to chosen sheet peak
    drift_score: np.ndarray      # (H, W) |chosen_offset| (0 below drift_min)
    switch_score: np.ndarray     # (H, W) in [0, 1]
    geom_offset: np.ndarray      # (H, W) signed normal residual vs smoothed grid
    margin: np.ndarray           # (H, W) distance-weighted best-vs-second peak
    evidence: np.ndarray         # (H, W) normalized chosen-peak height
    contrast: np.ndarray         # (H, W) profile dynamic range / global median
    confidence: np.ndarray       # (H, W) in [0, 1]
    review: np.ndarray           # (H, W) bool
    correction_offset: np.ndarray  # (H, W) proposed move (nan if none)
    estimated_spacing: float


def _grid_normal_residual(points_xyz, normals, valid, window):
    """Return signed normal residuals from a local mean surface"""
    H, W, _ = points_xyz.shape
    w = max(3, int(window) | 1)
    m = valid.astype(np.float32)
    sm = np.zeros_like(points_xyz)
    denom = uniform_filter(m, size=w, mode="nearest")
    denom = np.clip(denom, 1e-6, None)
    for c in range(3):
        ch = points_xyz[..., c] * m
        sm[..., c] = uniform_filter(ch, size=w, mode="nearest") / denom
    resid = points_xyz - sm
    geom = np.sum(resid * normals, axis=-1)
    geom[~valid] = 0.0
    return geom.astype(np.float32)


def _robust_residual_magnitude(points_xyz, valid, window):
    """Return distance from a large-window median surface"""
    w = max(5, int(window) | 1)
    ref = np.empty_like(points_xyz)
    for c in range(3):
        ref[..., c] = median_filter(points_xyz[..., c], size=w, mode="nearest")
    mag = np.linalg.norm(points_xyz - ref, axis=-1)
    mag[~valid] = 0.0
    return mag.astype(np.float32)


def _hysteresis(raw: np.ndarray, high: float, low: float) -> np.ndarray:
    """Keep weak connected regions containing at least one strong vertex"""
    from scipy.ndimage import label as cc_label

    weak = raw >= low
    strong = raw >= high
    labels, n = cc_label(weak)
    if n == 0:
        return np.zeros_like(raw)
    keep = np.zeros(n + 1, dtype=bool)
    strong_labels = np.unique(labels[strong])
    keep[strong_labels] = True
    keep[0] = False
    return keep[labels].astype(np.float32)


def _estimate_spacing(all_peak_offsets, cfg: DiagnosticsConfig) -> float:
    if cfg.sheet_spacing is not None:
        return float(cfg.sheet_spacing)
    diffs = []
    for offs in all_peak_offsets:
        if offs.size >= 2:
            diffs.extend(np.diff(np.sort(offs)).tolist())
    if diffs:
        d = float(np.median(diffs))
        if d > 1e-3:
            return d
    return 8.0


def compute_diagnostics(
    profiles: np.ndarray,
    offsets: np.ndarray,
    points_xyz: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    cfg: DiagnosticsConfig,
    correction=None,
) -> Diagnostics:
    """Compute per-vertex drift, switch, confidence, and correction signals"""
    H, W, T = profiles.shape
    step = float(offsets[1] - offsets[0]) if T > 1 else 1.0
    min_dist = max(1, int(round(cfg.peak_min_separation / step)))

    geom_offset = _grid_normal_residual(points_xyz, normals, valid, cfg.smooth_window)
    switch_mag = _robust_residual_magnitude(points_xyz, valid, cfg.switch_smooth_window)

    pmin = profiles.min(axis=2)
    pmax = profiles.max(axis=2)
    prange = pmax - pmin
    med_range = float(np.median(prange[valid])) if valid.any() else 1.0
    med_range = med_range if med_range > 1e-6 else 1.0

    chosen_offset = np.full((H, W), np.nan, dtype=np.float32)
    evidence = np.zeros((H, W), dtype=np.float32)
    margin = np.ones((H, W), dtype=np.float32)
    peak_offsets_grid = np.empty((H, W), dtype=object)

    # Detect profile peaks and estimate inter-sheet spacing
    for i in range(H):
        for j in range(W):
            if not valid[i, j]:
                peak_offsets_grid[i, j] = np.empty(0)
                continue
            rng_ij = prange[i, j]
            if rng_ij < 1e-6:
                peak_offsets_grid[i, j] = np.empty(0)
                continue
            norm_prof = (profiles[i, j] - pmin[i, j]) / rng_ij
            peaks, props = find_peaks(
                norm_prof, prominence=cfg.peak_min_prominence_frac, distance=min_dist
            )
            if peaks.size == 0:
                # Fall back to the global maximum of the profile
                peaks = np.array([int(np.argmax(norm_prof))])
            offs = offsets[peaks]
            heights = norm_prof[peaks]
            peak_offsets_grid[i, j] = (offs, heights)

    spacing = _estimate_spacing(
        [po[0] if isinstance(po, tuple) else po for po in peak_offsets_grid.ravel()], cfg
    )
    tau = max(spacing, 1e-3)

    # Prefer strong peaks near the current surface
    for i in range(H):
        for j in range(W):
            entry = peak_offsets_grid[i, j]
            if not isinstance(entry, tuple) or entry[0].size == 0:
                continue
            offs, heights = entry
            weighted = heights * np.exp(-np.abs(offs) / tau)
            order = np.argsort(weighted)[::-1]
            c = order[0]
            chosen_offset[i, j] = offs[c]
            evidence[i, j] = heights[c]
            if order.size >= 2:
                w1 = weighted[order[0]]
                w2 = weighted[order[1]]
                margin[i, j] = float((w1 - w2) / w1) if w1 > 1e-6 else 0.0
            else:
                margin[i, j] = 1.0

    # A switch is a spacing-scale geometric jump with strong sheet evidence
    switch_ratio = switch_mag / max(spacing, 1e-6)
    on_a_sheet = evidence >= 0.4
    switch_raw = np.where(valid & on_a_sheet, switch_ratio, 0.0).astype(np.float32)
    switch_score = _hysteresis(switch_raw, high=cfg.switch_frac_of_spacing, low=0.35)

    contrast = np.clip(prange / med_range, 0.0, 1.0).astype(np.float32)
    drift_score = np.where(
        np.isfinite(chosen_offset) & (np.abs(chosen_offset) >= cfg.drift_min),
        np.abs(chosen_offset),
        0.0,
    ).astype(np.float32)

    margin_conf = np.clip(margin / max(cfg.margin_soft, 1e-6), 0.0, 1.0)
    geom_conf = 1.0 - np.clip(np.abs(geom_offset) / spacing, 0.0, 1.0)
    confidence = (contrast * margin_conf * geom_conf * np.clip(evidence, 0.0, 1.0)).astype(np.float32)
    confidence[~valid] = 0.0

    return _finalize(
        valid, chosen_offset, drift_score, switch_score, geom_offset, margin,
        evidence, contrast, confidence, spacing, cfg, correction,
    )


def _finalize(
    valid, chosen_offset, drift_score, switch_score, geom_offset, margin,
    evidence, contrast, confidence, spacing, cfg, correction,
) -> Diagnostics:
    H, W = valid.shape
    big_drift = drift_score >= 0.35 * spacing
    review = valid & (
        (switch_score >= 0.5)
        | (confidence < 0.5)
        | big_drift
    )

    correction_offset = np.full((H, W), np.nan, dtype=np.float32)
    if correction is not None and getattr(correction, "enabled", False):
        gate = (
            valid
            & (confidence >= correction.min_confidence)
            & (margin >= correction.require_margin)
            & (switch_score < 0.5)
            & np.isfinite(chosen_offset)
            & (np.abs(chosen_offset) <= correction.max_offset)
            & (drift_score > 0.0)
        )
        correction_offset[gate] = chosen_offset[gate]

    return Diagnostics(
        valid=valid,
        chosen_offset=chosen_offset,
        drift_score=drift_score,
        switch_score=switch_score,
        geom_offset=geom_offset,
        margin=margin,
        evidence=evidence,
        contrast=contrast,
        confidence=confidence,
        review=review,
        correction_offset=correction_offset,
        estimated_spacing=float(spacing),
    )
