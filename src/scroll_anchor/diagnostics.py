"""Drift and sheet-switch diagnostics from normal CT profiles"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter, uniform_filter
from scipy.signal import find_peaks

from .config import DiagnosticsConfig, ReviewConfig


class ProfileSelectionState(IntEnum):
    """How a profile reference was obtained, independent of review policy."""

    NOT_EVALUATED = 0
    LOCAL_PEAK_SINGLE = 1
    LOCAL_PEAK_MULTIPLE = 2
    GLOBAL_MAX_FALLBACK = 3
    PROFILE_UNUSABLE = 4


@dataclass
class Diagnostics:
    valid: np.ndarray            # (H, W) bool
    chosen_offset: np.ndarray    # (H, W) signed voxels to chosen sheet peak
    profile_selection_state: np.ndarray  # (H, W) uint8 ProfileSelectionState
    drift_score: np.ndarray      # (H, W) |chosen_offset| (0 below drift_min)
    switch_score: np.ndarray     # (H, W) in [0, 1]
    geom_offset: np.ndarray      # (H, W) signed normal residual vs smoothed grid
    margin: np.ndarray           # (H, W) distance-weighted best-vs-second peak
    evidence: np.ndarray         # (H, W) normalized chosen-peak height
    contrast: np.ndarray         # (H, W) profile dynamic range / global median
    confidence: np.ndarray       # (H, W) in [0, 1]
    review: np.ndarray           # (H, W) bool
    review_low_confidence: np.ndarray  # (H, W) bool
    review_switch: np.ndarray    # (H, W) bool
    review_drift: np.ndarray     # (H, W) bool; populated only for legacy opt-in
    correction_offset: np.ndarray  # (H, W) proposed move (nan if none)
    estimated_spacing: float


def review_cause_masks(
    valid: np.ndarray,
    confidence: np.ndarray,
    switch_score: np.ndarray,
    drift_score: np.ndarray,
    estimated_spacing: float,
    review_cfg: ReviewConfig,
):
    """Return the actionable review causes for the configured policy."""
    low_confidence = valid & (confidence < review_cfg.confidence_review_below)
    switch = valid & (switch_score >= 0.5)
    drift = np.zeros_like(valid, dtype=bool)
    if review_cfg.include_drift_in_review:
        drift = valid & (drift_score >= 0.35 * estimated_spacing)
    return low_confidence, switch, drift


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
    H, W = valid.shape
    # The requested background scale must fit the grid to define a reference.
    if not valid.any() or w > H or w > W:
        return np.zeros((H, W), dtype=np.float32)

    # Make the reference independent of arbitrary values stored outside the surface.
    completed = points_xyz.copy()
    nearest = distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    completed[~valid] = points_xyz[tuple(nearest[:, ~valid])]

    ref = np.empty_like(points_xyz)
    for c in range(3):
        ref[..., c] = median_filter(completed[..., c], size=w, mode="nearest")
    mag = np.linalg.norm(completed - ref, axis=-1)
    mag[~valid] = 0.0
    return mag.astype(np.float32)


def _hysteresis(
    raw: np.ndarray, high: float, low: float, strong_seed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Keep weak connected regions containing at least one strong seed."""
    from scipy.ndimage import label as cc_label

    weak = raw >= low
    strong = raw >= high if strong_seed is None else (weak & strong_seed)
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


def _center_supported_segments(sample_support: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Keep the supported component connected to the correspondence offset zero."""
    support = np.asarray(sample_support, dtype=bool)
    if support.ndim != 3:
        raise ValueError("sample_support must have shape (H, W, T)")
    keep = np.zeros_like(support)
    center = int(np.argmin(np.abs(offsets)))
    for i in range(support.shape[0]):
        for j in range(support.shape[1]):
            row = support[i, j]
            if not row[center]:
                continue
            lo = center
            while lo > 0 and row[lo - 1]:
                lo -= 1
            hi = center + 1
            while hi < row.size and row[hi]:
                hi += 1
            keep[i, j, lo:hi] = True
    return keep


def compute_diagnostics(
    profiles: np.ndarray,
    offsets: np.ndarray,
    points_xyz: np.ndarray,
    normals: np.ndarray,
    valid: np.ndarray,
    cfg: DiagnosticsConfig,
    correction=None,
    review_cfg: Optional[ReviewConfig] = None,
    sample_support: Optional[np.ndarray] = None,
) -> Diagnostics:
    """Compute per-vertex drift, switch, confidence, and correction signals"""
    H, W, T = profiles.shape
    step = float(offsets[1] - offsets[0]) if T > 1 else 1.0
    min_dist = max(1, int(round(cfg.peak_min_separation / step)))

    geom_offset = _grid_normal_residual(points_xyz, normals, valid, cfg.smooth_window)
    switch_mag = _robust_residual_magnitude(points_xyz, valid, cfg.switch_smooth_window)

    support = None
    if sample_support is None:
        pmin = profiles.min(axis=2)
        pmax = profiles.max(axis=2)
    else:
        if sample_support.shape != profiles.shape:
            raise ValueError("sample_support shape must match profiles")
        support = _center_supported_segments(sample_support, offsets)
        pmin = np.min(np.where(support, profiles, np.inf), axis=2)
        pmax = np.max(np.where(support, profiles, -np.inf), axis=2)
    prange = pmax - pmin
    usable_ranges = prange[valid & np.isfinite(prange)]
    med_range = float(np.median(usable_ranges)) if usable_ranges.size else 1.0
    med_range = med_range if med_range > 1e-6 else 1.0

    chosen_offset = np.full((H, W), np.nan, dtype=np.float32)
    profile_selection_state = np.full(
        (H, W), ProfileSelectionState.NOT_EVALUATED, dtype=np.uint8
    )
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
            sample_indices = (
                np.flatnonzero(support[i, j]) if support is not None else np.arange(T)
            )
            if sample_indices.size == 0 or not np.isfinite(rng_ij) or rng_ij < 1e-6:
                profile_selection_state[i, j] = ProfileSelectionState.PROFILE_UNUSABLE
                peak_offsets_grid[i, j] = np.empty(0)
                continue
            norm_prof = (profiles[i, j, sample_indices] - pmin[i, j]) / rng_ij
            peaks, _ = find_peaks(
                norm_prof, prominence=cfg.peak_min_prominence_frac, distance=min_dist
            )
            if peaks.size == 0:
                # Fall back to the global maximum of the profile
                profile_selection_state[i, j] = ProfileSelectionState.GLOBAL_MAX_FALLBACK
                peaks = np.array([int(np.argmax(norm_prof))])
            elif peaks.size == 1:
                profile_selection_state[i, j] = ProfileSelectionState.LOCAL_PEAK_SINGLE
            else:
                profile_selection_state[i, j] = ProfileSelectionState.LOCAL_PEAK_MULTIPLE
            offs = offsets[sample_indices[peaks]]
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

    # A switch is a spacing-scale geometric jump with strong sheet evidence.
    # Fallback is a deterministic reference, not affirmative correspondence
    # evidence: it can bridge a retained weak component but cannot seed one.
    switch_ratio = switch_mag / max(spacing, 1e-6)
    on_a_sheet = evidence >= 0.4
    switch_raw = np.where(valid & on_a_sheet, switch_ratio, 0.0).astype(np.float32)
    accepted_local_peak = np.isin(
        profile_selection_state,
        [ProfileSelectionState.LOCAL_PEAK_SINGLE, ProfileSelectionState.LOCAL_PEAK_MULTIPLE],
    )
    strong_seed = (
        valid & on_a_sheet & accepted_local_peak
        & (switch_ratio >= cfg.switch_frac_of_spacing)
    )
    switch_score = _hysteresis(
        switch_raw, high=cfg.switch_frac_of_spacing, low=0.35, strong_seed=strong_seed,
    )

    contrast = np.clip(prange / med_range, 0.0, 1.0).astype(np.float32)
    drift_score = np.where(
        np.isfinite(chosen_offset) & (np.abs(chosen_offset) >= cfg.drift_min),
        np.abs(chosen_offset),
        0.0,
    ).astype(np.float32)

    margin_conf = np.clip(margin / max(cfg.margin_soft, 1e-6), 0.0, 1.0)
    # Confidence describes local CT-profile quality.  Geometric residual is
    # retained as a diagnostic, but is not a sheet-specific CT-confidence
    # factor on irregular real surface geometry.
    confidence = (contrast * margin_conf * np.clip(evidence, 0.0, 1.0)).astype(np.float32)
    confidence[~valid] = 0.0

    return _finalize(
        valid, chosen_offset, profile_selection_state, drift_score, switch_score,
        geom_offset, margin, evidence, contrast, confidence, spacing, cfg,
        correction, review_cfg,
    )


def _finalize(
    valid, chosen_offset, profile_selection_state, drift_score, switch_score,
    geom_offset, margin, evidence, contrast, confidence, spacing, cfg, correction,
    review_cfg,
) -> Diagnostics:
    H, W = valid.shape
    # Drift is exploratory information, not an actionable default review
    # trigger.  ``report.apply_review`` can opt into the legacy policy via
    # ReviewConfig.include_drift_in_review without changing any diagnostics.
    review_cfg = review_cfg or ReviewConfig()
    review_low_confidence, review_switch, review_drift = review_cause_masks(
        valid, confidence, switch_score, drift_score, spacing, review_cfg,
    )
    review = review_low_confidence | review_switch | review_drift

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
        profile_selection_state=profile_selection_state,
        drift_score=drift_score,
        switch_score=switch_score,
        geom_offset=geom_offset,
        margin=margin,
        evidence=evidence,
        contrast=contrast,
        confidence=confidence,
        review=review,
        review_low_confidence=review_low_confidence,
        review_switch=review_switch,
        review_drift=review_drift,
        correction_offset=correction_offset,
        estimated_spacing=float(spacing),
    )
