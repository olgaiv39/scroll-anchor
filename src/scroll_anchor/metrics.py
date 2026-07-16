"""Metrics for the synthetic corruption benchmark"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from .diagnostics import Diagnostics
from .synth import CLEAN, DRIFT, SWITCH, SheetModel


def _prf(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    pred = pred & mask
    true = true & mask
    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _final_sheet(points_xyz, offsets_along_normal, normals, model: SheetModel) -> np.ndarray:
    pts = points_xyz + offsets_along_normal[..., None] * normals
    return model.sheet_id_at(pts)


@dataclass
class BenchmarkResult:
    switch_detection: Dict[str, float]
    drift_detection: Dict[str, float]
    drift_displacement_mae: float
    harmful_rate_label_as_is: float
    harmful_rate_naive_snap: float
    harmful_rate_scroll_anchor: float
    accepted_frac_scroll_anchor: float
    clean_stability: float
    review_frac: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate(
    diag: Diagnostics,
    gt: Dict[str, np.ndarray],
    model: SheetModel,
    corrupt_points: np.ndarray,
    normals: np.ndarray,
    profiles: np.ndarray,
    offsets: np.ndarray,
    drift_min: float,
) -> BenchmarkResult:
    ctype = gt["corruption_type"]
    inj = gt["injected_offset"]
    true_sheet = gt["true_sheet"]
    valid = diag.valid

    pred_switch = diag.switch_score >= 0.5
    true_switch = ctype == SWITCH
    switch_det = _prf(pred_switch, true_switch, valid)

    pred_drift = (diag.drift_score >= drift_min) & (~pred_switch)
    true_drift = ctype == DRIFT
    drift_det = _prf(pred_drift, true_drift, valid)

    dmask = (ctype == DRIFT) & valid & np.isfinite(diag.chosen_offset)
    if np.any(dmask):
        disp_err = np.abs(diag.chosen_offset[dmask] + inj[dmask])
        drift_mae = float(np.mean(disp_err))
    else:
        drift_mae = float("nan")

    zero_off = np.zeros_like(inj)

    def harmful_rate(accepted: np.ndarray, move: np.ndarray) -> float:
        acc = accepted & valid
        if not np.any(acc):
            return 0.0
        final_sheet = _final_sheet(corrupt_points, move, normals, model)
        wrong = final_sheet != true_sheet
        return float(np.sum(acc & wrong) / np.sum(acc))

    h_asis = harmful_rate(valid, zero_off)

    strongest = offsets[np.argmax(profiles, axis=2)].astype(np.float32)
    strongest[~valid] = 0.0
    h_naive = harmful_rate(valid, strongest)

    accepted_sa = valid & ~diag.review
    move_sa = np.where(np.isfinite(diag.correction_offset), diag.correction_offset, 0.0).astype(
        np.float32
    )
    h_sa = harmful_rate(accepted_sa, move_sa)

    accepted_frac = float(np.sum(accepted_sa) / max(1, np.sum(valid)))
    clean_mask = (ctype == CLEAN) & valid
    clean_stability = (
        float(np.sum(accepted_sa & clean_mask) / max(1, np.sum(clean_mask)))
        if np.any(clean_mask)
        else 1.0
    )
    review_frac = float(np.sum(diag.review & valid) / max(1, np.sum(valid)))

    return BenchmarkResult(
        switch_detection=switch_det,
        drift_detection=drift_det,
        drift_displacement_mae=drift_mae,
        harmful_rate_label_as_is=h_asis,
        harmful_rate_naive_snap=h_naive,
        harmful_rate_scroll_anchor=h_sa,
        accepted_frac_scroll_anchor=accepted_frac,
        clean_stability=clean_stability,
        review_frac=review_frac,
    )
