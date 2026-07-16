"""Machine-readable reports and per-vertex output channels"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
from scipy.ndimage import label as cc_label

from .config import ReviewConfig, RunConfig
from .diagnostics import Diagnostics
from .tifxyz import Surface, write_tifxyz


def apply_review(diag: Diagnostics, cfg: ReviewConfig) -> np.ndarray:
    """(Re)compute the review mask from confidence, switch and drift signals"""
    spacing = diag.estimated_spacing
    big_drift = diag.drift_score >= 0.35 * spacing
    review = diag.valid & (
        (diag.switch_score >= 0.5)
        | (diag.confidence < cfg.confidence_review_below)
        | big_drift
    )
    diag.review = review
    return review


def build_review_regions(diag: Diagnostics, cfg: ReviewConfig) -> List[Dict[str, object]]:
    """Cluster the review mask into prioritised regions"""
    labels, n = cc_label(diag.review)
    regions: List[Dict[str, object]] = []
    for lab in range(1, n + 1):
        m = labels == lab
        size = int(m.sum())
        if size < cfg.min_region_vertices:
            continue
        rows, cols = np.nonzero(m)
        mean_conf = float(np.mean(diag.confidence[m]))
        has_switch = bool(np.any(diag.switch_score[m] >= 0.5))
        mean_drift = float(np.mean(diag.drift_score[m]))
        if has_switch:
            reason = "sheet_switch"
            base = 3.0
        elif mean_drift >= 0.35 * diag.estimated_spacing:
            reason = "drift"
            base = 2.0
        else:
            reason = "low_confidence"
            base = 1.0
        priority = base + (1.0 - mean_conf) + np.log1p(size) * 0.1
        regions.append(
            {
                "id": lab,
                "reason": reason,
                "priority": round(float(priority), 4),
                "size_vertices": size,
                "bbox_rowcol": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
                "centroid_rowcol": [float(rows.mean()), float(cols.mean())],
                "mean_confidence": round(mean_conf, 4),
                "mean_drift_voxels": round(mean_drift, 4),
                "has_switch": has_switch,
            }
        )
    regions.sort(key=lambda r: r["priority"], reverse=True)
    return regions[: cfg.max_regions]


def _summary(diag: Diagnostics) -> Dict[str, object]:
    v = diag.valid
    nvalid = int(v.sum())

    def frac(mask):
        return float(mask[v].mean()) if nvalid else 0.0

    return {
        "n_vertices": int(v.size),
        "n_valid": nvalid,
        "estimated_sheet_spacing_voxels": round(diag.estimated_spacing, 4),
        "frac_review": frac(diag.review),
        "frac_switch": frac(diag.switch_score >= 0.5),
        "frac_drift_flagged": frac(diag.drift_score > 0),
        "mean_confidence": float(np.mean(diag.confidence[v])) if nvalid else 0.0,
        "n_correction_proposals": int(np.sum(np.isfinite(diag.correction_offset))),
    }


def write_reports(
    outdir: str,
    surface: Surface,
    diag: Diagnostics,
    config: RunConfig,
    regions: List[Dict[str, object]],
    write_channels: bool = True,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    diagnostics = {
        "format": "scroll-anchor.diagnostics/v0",
        "config": config.to_dict(),
        "summary": _summary(diag),
    }
    with open(os.path.join(outdir, "diagnostics.json"), "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)

    with open(os.path.join(outdir, "review_regions.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_regions": len(regions), "regions": regions}, fh, indent=2)

    arr_dir = os.path.join(outdir, "arrays")
    os.makedirs(arr_dir, exist_ok=True)
    for name, arr in {
        "confidence": diag.confidence,
        "drift_score": diag.drift_score,
        "switch_score": diag.switch_score,
        "chosen_offset": diag.chosen_offset,
        "geom_offset": diag.geom_offset,
        "review": diag.review.astype(np.uint8),
        "correction_offset": diag.correction_offset,
    }.items():
        np.save(os.path.join(arr_dir, f"{name}.npy"), arr)

    if write_channels:
        seg_dir = os.path.join(outdir, "surface")
        extras = {
            "sa_confidence": diag.confidence.astype(np.float32),
            "sa_drift": diag.drift_score.astype(np.float32),
            "sa_switch": diag.switch_score.astype(np.float32),
            "sa_review": diag.review.astype(np.uint8),
        }
        write_tifxyz(seg_dir, surface, extra_channels=extras, overwrite=True)
