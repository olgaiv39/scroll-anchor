#!/usr/bin/env python
"""External evaluation of the frozen ScrollAnchor baseline on PHerc1667 candidate-07.

Read-only measurement script. It does not import, alter, tune or re-implement any
detector decision: it reads the emitted arrays from an existing `scroll-anchor
analyze` run and compares them with the package ground truth.

Three domains are reported separately because they are not interchangeable:

  A. surface-valid       - the 100 tifxyz-valid vertices, with an explicit
                           per-vertex evaluation state.
  B. detector-evaluable  - the vertices the detector actually scored. This is the
                           only domain where a confusion matrix is a statement
                           about detector decisions.
  C. literal mask        - a conventional confusion matrix over all 100 vertices,
                           reading the emitted review mask literally. Emitted
                           zeros outside detector coverage are NOT detector
                           negatives, so this domain is coverage-confounded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, label
from scipy.stats import pearsonr, spearmanr

# Detector-side helpers, imported read-only to re-derive the validity mask
# independently of the emitted arrays.
from scroll_anchor.normals import compute_normals
from scroll_anchor.tifxyz import read_tifxyz

REVIEW_CONDITION_SPACING_FRAC = 0.35  # big_drift = drift_score >= 0.35 * spacing
SWITCH_THRESHOLD = 0.5
CONFIDENCE_REVIEW_BELOW = 0.5


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _py(o):
    """Recursively convert numpy scalars/arrays to plain Python for JSON."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _git(repo, *args):
    try:
        out = subprocess.run(
            ["git", "-C", repo] + list(args),
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _dist_stats(values, name):
    """Interpretable univariate summary; no derived or decorative scores."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"name": name, "count": 0}
    return {
        "name": name,
        "count": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def _corr(a, b):
    """Pearson and Spearman, guarded against degenerate (constant) inputs."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = {"n": int(a.size)}
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        out["pearson_r"] = None
        out["spearman_rho"] = None
        out["degenerate"] = True
        return out
    out["pearson_r"] = float(pearsonr(a, b)[0])
    out["spearman_rho"] = float(spearmanr(a, b)[0])
    out["degenerate"] = False
    return out


def _confusion(pred, truth):
    """Confusion counts plus the standard overlap measures."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if (prec is not None and rec is not None) else None)
    union = tp + fp + fn
    iou = tp / union if union else None
    dice = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": int(pred.size),
        "precision": prec, "recall": rec, "f1": f1,
        "iou": iou, "dice": dice,
    }


# --------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------
def evaluate(package, reproduced, outdir, repo, package_zip=None):
    os.makedirs(outdir, exist_ok=True)
    report = {}

    # ---------------- provenance ----------------
    report["repository_provenance"] = {
        "repo_path_note": "recorded relative; absolute paths intentionally omitted",
        "branch": _git(repo, "branch", "--show-current"),
        "head_sha": _git(repo, "rev-parse", "HEAD"),
        "last_commit": _git(repo, "log", "-1", "--oneline"),
        "status_short": _git(repo, "status", "--short"),
        "working_tree_clean": (_git(repo, "status", "--short") == ""),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }

    pkg_prov = {
        "logical_filename": "scrollanchor-package-pherc1667-candidate07.zip",
        "source": "Alan Thompson / Discord altommo",
        "received_date": "2026-08-03",
        "expected_size_bytes": 14732072,
        "expected_sha256": "2460120eba5be9c0dde08827ffe8c5f196f14e1eb6db0fe60645751bd56871a6",
        "extraction_location": "data/external/pherc1667-candidate07/ (gitignored)",
    }
    if package_zip and os.path.isfile(package_zip):
        pkg_prov["observed_size_bytes"] = os.path.getsize(package_zip)
        pkg_prov["observed_sha256"] = _sha256(package_zip)
        pkg_prov["size_matches"] = pkg_prov["observed_size_bytes"] == pkg_prov["expected_size_bytes"]
        pkg_prov["sha256_matches"] = pkg_prov["observed_sha256"] == pkg_prov["expected_sha256"]
    report["package_provenance"] = pkg_prov

    # ---------------- inputs ----------------
    surf = read_tifxyz(os.path.join(package, "surface"))
    gens = tifffile.imread(os.path.join(package, "surface", "generations.tif"))
    ct = np.load(os.path.join(package, "ct_roi.npy"), mmap_mode="r")

    dist = np.load(os.path.join(package, "ground_truth", "distance_to_reference.npy"))
    signed = np.load(os.path.join(package, "ground_truth", "signed_normal_offset.npy"))
    gt = tifffile.imread(os.path.join(package, "ground_truth", "failure_sector_mask.tif")) > 0

    surface_valid = surf.valid
    pts = surf.points()
    H, W = surface_valid.shape

    expected_block = np.zeros_like(surface_valid)
    expected_block[2:12, 2:12] = True

    Dz, Dy, Dx = ct.shape
    xv, yv, zv = surf.x[surface_valid], surf.y[surface_valid], surf.z[surface_valid]

    gt_labels, gt_ncomp = label(gt)
    gt_rows, gt_cols = np.nonzero(gt)

    validation = {
        "files_present": {
            n: os.path.exists(os.path.join(package, n))
            for n in [
                "ct_roi.npy", "README.md",
                "surface/x.tif", "surface/y.tif", "surface/z.tif",
                "surface/generations.tif", "surface/meta.json",
                "ground_truth/distance_to_reference.npy",
                "ground_truth/signed_normal_offset.npy",
                "ground_truth/failure_sector_mask.tif",
                "ground_truth/failure_sector.json",
                "scrollanchor_firstpass/diagnostics.json",
                "scrollanchor_firstpass/review_regions.json",
            ]
        },
        "surface_grid_shape": list(surface_valid.shape),
        "xyz_shapes_agree": bool(surf.x.shape == surf.y.shape == surf.z.shape == gens.shape),
        "dtypes": {
            "x": str(surf.x.dtype), "y": str(surf.y.dtype), "z": str(surf.z.dtype),
            "generations": str(gens.dtype), "ct": str(ct.dtype),
            "distance_to_reference": str(dist.dtype),
            "signed_normal_offset": str(signed.dtype),
        },
        "mask_tif_present": os.path.isfile(os.path.join(package, "surface", "mask.tif")),
        "validity_rule": "valid = z > 0 (no mask.tif supplied)",
        "sentinel_value_observed": float(np.unique(surf.z[surf.z <= 0])[0]) if (surf.z <= 0).any() else None,
        "all_coords_finite": bool(np.isfinite(pts).all()),
        "surface_valid_count": int(surface_valid.sum()),
        "surface_valid_equals_rows2_11_cols2_11": bool((surface_valid == expected_block).all()),
        "ct_shape": list(ct.shape),
        "ct_axis_contract": "ct indexed [z,y,x]; surface point (X,Y,Z) samples index (Z,Y,X); origin zero",
        "coords_in_ct_bounds": bool(
            (xv >= 0).all() and (xv <= Dx - 1).all()
            and (yv >= 0).all() and (yv <= Dy - 1).all()
            and (zv >= 0).all() and (zv <= Dz - 1).all()
        ),
        "coord_ranges": {
            "X": [float(xv.min()), float(xv.max())],
            "Y": [float(yv.min()), float(yv.max())],
            "Z": [float(zv.min()), float(zv.max())],
        },
        "gt_distance_finite_mask_equals_surface_valid": bool((np.isfinite(dist) == surface_valid).all()),
        "gt_signed_finite_mask_equals_surface_valid": bool((np.isfinite(signed) == surface_valid).all()),
        "gt_mask_subset_of_surface_valid": bool(((gt & ~surface_valid).sum() == 0)),
        "gt_mask_equals_distance_gt_4_on_surface_valid": bool(
            (gt[surface_valid] == (dist[surface_valid] > 4)).all()
        ),
        "gt_positive_count": int(gt.sum()),
        "gt_connected_components_4conn": int(gt_ncomp),
        "gt_component_sizes": [int((gt_labels == i).sum()) for i in range(1, gt_ncomp + 1)],
        "gt_bbox_rowcol_full_grid": [
            [int(gt_rows.min()), int(gt_cols.min())],
            [int(gt_rows.max()), int(gt_cols.max())],
        ],
        "gt_bbox_rowcol_block_local": [
            [int(gt_rows.min()) - 2, int(gt_cols.min()) - 2],
            [int(gt_rows.max()) - 2, int(gt_cols.max()) - 2],
        ],
        "gt_bbox_matches_package_json_block_local": bool(
            [int(gt_rows.min()) - 2, int(gt_cols.min()) - 2] == [4, 0]
            and [int(gt_rows.max()) - 2, int(gt_cols.max()) - 2] == [9, 9]
        ),
        "gt_max_distance": float(dist[gt].max()),
    }
    report["input_validation"] = validation

    # ---------------- detector-evaluable mask ----------------
    normals, normal_valid = compute_normals(surf)
    evaluable = surface_valid & normal_valid
    excluded = surface_valid & ~evaluable

    eroded = binary_erosion(surface_valid, structure=np.ones((3, 3), bool), border_value=0)
    inner_block = np.zeros_like(surface_valid)
    inner_block[3:11, 3:11] = True

    # Cross-check against the detector's own emitted mask: chosen_offset is NaN
    # exactly where the detector did not score a vertex.
    arr_dir = os.path.join(reproduced, "arrays")
    review = np.load(os.path.join(arr_dir, "review.npy")).astype(bool)
    confidence = np.load(os.path.join(arr_dir, "confidence.npy"))
    drift = np.load(os.path.join(arr_dir, "drift_score.npy"))
    switch = np.load(os.path.join(arr_dir, "switch_score.npy"))
    chosen = np.load(os.path.join(arr_dir, "chosen_offset.npy"))
    geom = np.load(os.path.join(arr_dir, "geom_offset.npy"))

    emitted_scored = np.isfinite(chosen)

    diag_json = json.load(open(os.path.join(reproduced, "diagnostics.json")))
    spacing = float(diag_json["summary"]["estimated_sheet_spacing_voxels"])

    # ---------------- reproduction vs Alan's first pass ----------------
    fp = os.path.join(package, "scrollanchor_firstpass")
    fp_diag = json.load(open(os.path.join(fp, "diagnostics.json")))
    fp_reg = json.load(open(os.path.join(fp, "review_regions.json")))
    rp_reg = json.load(open(os.path.join(reproduced, "review_regions.json")))

    array_cmp = {}
    all_identical = True
    for fn in sorted(os.listdir(os.path.join(fp, "arrays"))):
        a = np.load(os.path.join(fp, "arrays", fn))
        b = np.load(os.path.join(arr_dir, fn))
        same_shape_dtype = (a.shape == b.shape) and (a.dtype == b.dtype)
        if a.dtype.kind == "f":
            equal = bool(np.array_equal(a, b, equal_nan=True))
            fin = np.isfinite(a) & np.isfinite(b)
            mad = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
            nan_same = bool(np.array_equal(np.isnan(a), np.isnan(b)))
        else:
            equal = bool(np.array_equal(a, b))
            mad = float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))) if a.size else 0.0
            nan_same = True
        bytes_same = a.tobytes() == b.tobytes()
        array_cmp[fn] = {
            "shape": list(a.shape), "dtype": str(a.dtype),
            "shape_dtype_match": same_shape_dtype,
            "equal_nan_aware": equal,
            "nan_pattern_identical": nan_same,
            "bytes_identical": bytes_same,
            "max_abs_difference_finite": mad,
            "tolerance": "exact equality required; NaN treated as equal where both are NaN",
        }
        all_identical &= same_shape_dtype and equal and nan_same and bytes_same

    report["reproduction_comparison"] = {
        "reference": "scrollanchor_firstpass/ supplied by Alan Thompson",
        "command": (
            "scroll-anchor analyze --surface <pkg>/surface --volume <pkg>/ct_roi.npy "
            "--config configs/default.yaml --output <pkg>/reproduced-default --no-channels"
        ),
        "resolved_config_identical": bool(fp_diag["config"] == diag_json["config"]),
        "diagnostics_format_identical": bool(fp_diag["format"] == diag_json["format"]),
        "diagnostics_summary_identical": bool(fp_diag["summary"] == diag_json["summary"]),
        "summary_first_pass": fp_diag["summary"],
        "summary_reproduced": diag_json["summary"],
        "review_region_count_first_pass": fp_reg["n_regions"],
        "review_region_count_reproduced": rp_reg["n_regions"],
        "review_regions_identical": bool(fp_reg == rp_reg),
        "review_region_fields_reproduced": rp_reg["regions"],
        "per_array_comparison": array_cmp,
        "all_supplied_arrays_bytewise_identical": bool(all_identical),
        "verdict": (
            "EXACT" if all_identical else "MATERIAL DIFFERENCE - investigate before use"
        ),
    }

    report["resolved_frozen_configuration"] = diag_json["config"]
    report["detector_evaluable_derivation"] = {
        "definition": "surface.valid & normal_valid, exactly as pipeline.analyze_surface computes it",
        "normal_valid_mechanism": (
            "normals.compute_normals sets invalid vertices to NaN before np.gradient. "
            "np.gradient's central/one-sided stencils therefore propagate NaN to every "
            "vertex whose 3x3 neighbourhood touches an invalid vertex, so the valid "
            "block is eroded by exactly one vertex on each side."
        ),
        "count": int(evaluable.sum()),
        "equals_one_vertex_3x3_erosion_of_surface_valid": bool((evaluable == eroded).all()),
        "equals_rows3_10_cols3_10_inner_8x8": bool((evaluable == inner_block).all()),
        "agrees_with_emitted_finite_chosen_offset": bool((evaluable == emitted_scored).all()),
        "excluded_ring_count": int(excluded.sum()),
        "reported_n_valid_in_diagnostics": int(diag_json["summary"]["n_valid"]),
        "estimated_sheet_spacing_voxels": spacing,
    }

    # ---------------- generation distribution ----------------
    def gen_dist(mask):
        vals, counts = np.unique(gens[mask], return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    report["generation_distribution"] = {
        "surface_valid": gen_dist(surface_valid),
        "detector_evaluable": gen_dist(evaluable),
        "gt_positive": gen_dist(gt),
        "excluded_from_detector": gen_dist(excluded),
        "gt_positive_and_excluded": gen_dist(gt & excluded),
    }

    # ---------------- evaluation state ----------------
    state = np.empty((H, W), dtype=object)
    state[:] = ""
    state[evaluable & review] = "reviewed"
    state[evaluable & ~review] = "evaluated_not_reviewed"
    state[excluded] = "not_evaluated"

    # ---------------- Domain A: surface-valid coverage ----------------
    n_surface = int(surface_valid.sum())
    clean = surface_valid & ~gt
    report["domain_A_surface_valid"] = {
        "description": "All 100 tifxyz-valid vertices, described by coverage and evaluation state.",
        "n_surface_valid": n_surface,
        "detector_coverage": int(evaluable.sum()) / n_surface,
        "n_detector_evaluable": int(evaluable.sum()),
        "n_not_evaluated": int(excluded.sum()),
        "not_evaluated_fraction": int(excluded.sum()) / n_surface,
        "reviewed_fraction_of_surface_valid": int((review & surface_valid).sum()) / n_surface,
        "gt_positive": {
            "total": int(gt.sum()),
            "coverage": int((gt & evaluable).sum()) / int(gt.sum()),
            "reviewed": int((gt & review).sum()),
            "evaluated_not_reviewed": int((gt & evaluable & ~review).sum()),
            "not_evaluated": int((gt & excluded).sum()),
        },
        "clean": {
            "total": int(clean.sum()),
            "coverage": int((clean & evaluable).sum()) / int(clean.sum()),
            "reviewed": int((clean & review).sum()),
            "evaluated_not_reviewed": int((clean & evaluable & ~review).sum()),
            "not_evaluated": int((clean & excluded).sum()),
        },
        "evaluation_state_counts": {
            "reviewed": int((state == "reviewed").sum()),
            "evaluated_not_reviewed": int((state == "evaluated_not_reviewed").sum()),
            "not_evaluated": int((state == "not_evaluated").sum()),
        },
        "caveat": (
            "not_evaluated vertices are NOT true negatives and NOT false negatives. "
            "The detector emitted no decision for them; a zero in the review mask there "
            "is an absence of coverage, not a negative call."
        ),
    }

    # ---------------- Domain B: detector-evaluable ----------------
    conf_B = _confusion(review[evaluable], gt[evaluable])
    n_eval = int(evaluable.sum())
    clean_eval = clean & evaluable
    report["domain_B_detector_evaluable"] = {
        "description": (
            "The only domain where the confusion matrix expresses detector decisions: "
            "every vertex here received a real score."
        ),
        "n": n_eval,
        "n_gt_positive": int((gt & evaluable).sum()),
        "n_clean": int(clean_eval.sum()),
        "confusion": conf_B,
        "clean_false_positive_count": int((review & clean_eval).sum()),
        "clean_false_positive_rate": int((review & clean_eval).sum()) / int(clean_eval.sum()),
        "fraction_of_evaluable_failure_sector_reviewed": (
            int((review & gt & evaluable).sum()) / int((gt & evaluable).sum())
        ),
        "fraction_of_evaluable_clean_surface_left_unflagged": (
            int((~review & clean_eval).sum()) / int(clean_eval.sum())
        ),
    }

    # ---------------- Domain C: literal emitted mask ----------------
    conf_C = _confusion(review[surface_valid], gt[surface_valid])
    report["domain_C_literal_emitted_mask"] = {
        "description": (
            "Conventional confusion matrix over all 100 surface-valid vertices, reading "
            "the emitted review mask literally."
        ),
        "coverage_confounded": True,
        "caveat": (
            "COVERAGE-CONFOUNDED. 36 of the 100 vertices carry an emitted zero only "
            "because the detector never scored them. Those zeros are counted here as "
            "negatives (16 as FN, 20 as TN), which credits and penalises the detector "
            "for decisions it did not make. This is reported for completeness only and "
            "is not the primary result."
        ),
        "confusion": conf_C,
        "fn_are_all_uncovered": bool(((~review & gt & surface_valid) == (gt & excluded)).all()),
        "tn_are_all_uncovered": bool(((~review & clean) == (clean & excluded)).all()),
    }

    # ---------------- review-condition attribution ----------------
    c_switch = switch >= SWITCH_THRESHOLD
    c_lowconf = confidence < CONFIDENCE_REVIEW_BELOW
    c_drift = drift >= REVIEW_CONDITION_SPACING_FRAC * spacing
    report["review_condition_attribution"] = {
        "review_rule": "valid & ((switch_score >= 0.5) | (confidence < 0.5) | (drift_score >= 0.35*spacing))",
        "big_drift_threshold_voxels": REVIEW_CONDITION_SPACING_FRAC * spacing,
        "n_evaluable": n_eval,
        "n_reviewed": int(review[evaluable].sum()),
        "switch_condition_true": int((c_switch & evaluable).sum()),
        "low_confidence_condition_true": int((c_lowconf & evaluable).sum()),
        "big_drift_condition_true": int((c_drift & evaluable).sum()),
        # review & ~switch_condition. This is 0 here only because switch_score
        # saturates to 1.0 on every evaluable vertex, not because the other two
        # conditions are weak. See `switch_rule_removed_diagnostic` below for
        # the separate counterfactual (what confidence/drift alone would flag).
        "reviewed_not_satisfying_switch_condition": int((review & ~c_switch & evaluable).sum()),
        "switch_condition_alone_reproduces_review": bool(
            (review[evaluable] == (c_switch & evaluable)[evaluable]).all()
        ),
        "switch_score_unique_values_on_evaluable": sorted(
            float(v) for v in np.unique(switch[evaluable])
        ),
    }

    # ---------------- diagnostic: switch rule removed from review OR-rule ----------------
    # NOT a proposed detector variant. This measures what the confidence/drift
    # conditions alone would flag, to show that switch saturation is not the
    # only source of over-review on this patch.
    c_no_switch = (c_lowconf | c_drift) & evaluable
    conf_no_switch = _confusion(c_no_switch[evaluable], gt[evaluable])
    clean_ev_ns = clean & evaluable
    report["switch_rule_removed_diagnostic"] = {
        "description": (
            "Diagnostic counterfactual only, not an alternative validated detector: "
            "valid & ((confidence < 0.5) | (drift_score >= 0.35*spacing)), i.e. the "
            "review OR-rule with the switch_score term removed."
        ),
        "rule": "valid & ((confidence < 0.5) | (drift_score >= 0.35*spacing))",
        "reviewed_count": int(c_no_switch.sum()),
        "confusion": conf_no_switch,
        "clean_left_unflagged": int((clean_ev_ns & ~c_no_switch).sum()),
        "clean_left_unflagged_of_clean_evaluable": (
            "%d / %d" % (int((clean_ev_ns & ~c_no_switch).sum()), int(clean_ev_ns.sum()))
        ),
    }

    # ---------------- signal comparison (evaluable only) ----------------
    ev = evaluable
    gt_ev = gt & ev
    cl_ev = clean & ev

    sig = {
        "domain": "detector-evaluable vertices only (n=%d)" % n_eval,
        "geom_offset_vs_signed_normal_offset": {
            **_corr(geom[ev], signed[ev]),
            "mae_voxels": float(np.mean(np.abs(geom[ev] - signed[ev]))),
            "sign_agreement_fraction": float(
                np.mean(np.sign(geom[ev]) == np.sign(signed[ev]))
            ),
            "sign_caveat": (
                "The package states the signed-offset sign is comparable only within "
                "this one trace, and ScrollAnchor's normal orientation is fixed by its "
                "own global mean-normal convention. Sign agreement is therefore not "
                "evidence of physical agreement."
            ),
            "geom_offset_abs_max_voxels": float(np.abs(geom[ev]).max()),
            "signed_normal_offset_abs_max_voxels": float(np.abs(signed[ev]).max()),
        },
        "abs_geom_offset_vs_distance_to_reference": {
            **_corr(np.abs(geom[ev]), dist[ev]),
            "mae_voxels": float(np.mean(np.abs(np.abs(geom[ev]) - dist[ev]))),
            "abs_geom_offset": _dist_stats(np.abs(geom[ev]), "abs(geom_offset)"),
            "distance_to_reference": _dist_stats(dist[ev], "distance_to_reference"),
        },
        "switch_classification_vs_gt": _confusion(c_switch[ev], gt[ev]),
        "review_mask_vs_gt": conf_B,
        "distributions_failure_vs_clean": {},
    }

    for name, arr in [
        ("confidence", confidence),
        ("drift_score", drift),
        ("switch_score", switch),
        ("chosen_offset", chosen),
        ("geom_offset", geom),
    ]:
        sig["distributions_failure_vs_clean"][name] = {
            "failure_sector": _dist_stats(arr[gt_ev], name + " | GT-positive"),
            "clean": _dist_stats(arr[cl_ev], name + " | clean"),
        }
    report["signal_comparison"] = sig

    # ---------------- switch-reference diagnosis ----------------
    # Measurement of the quantity the frozen code computes, to explain saturation.
    from scipy.ndimage import median_filter
    p64 = pts.astype(np.float64)
    ref = np.empty_like(p64)
    for c in range(3):
        ref[..., c] = median_filter(p64[..., c], size=31, mode="nearest")
    mag = np.linalg.norm(p64 - ref, axis=-1)
    report["switch_signal_diagnosis"] = {
        "code_path": "diagnostics._robust_residual_magnitude(points_xyz, valid, switch_smooth_window=31)",
        "observation_median_window": 31,
        "observation_grid_size": [int(H), int(W)],
        "reference_is_constant_across_grid": bool(np.allclose(ref, ref[0, 0])),
        "reference_point_xyz": [float(v) for v in ref[0, 0]],
        "sentinel_vertices_in_grid": int((~surface_valid).sum()),
        "sentinel_fraction_of_grid": float((~surface_valid).mean()),
        "valid_point_centroid_xyz": [float(v) for v in p64[surface_valid].mean(axis=0)],
        "switch_mag_on_evaluable_voxels": _dist_stats(mag[ev], "switch_mag"),
        "switch_ratio_on_evaluable": _dist_stats(mag[ev] / spacing, "switch_mag / spacing"),
        "switch_threshold_high": SWITCH_THRESHOLD,
        "measured_explanation": (
            "The 31-vertex median window exceeds the 14-vertex grid, so with mode='nearest' "
            "the 'local' reference collapses to a single constant point. Because the "
            "sentinel-filled invalid vertices (value -1) are included in that median and "
            "are replicated by the border mode, the reference lands exactly on (-1,-1,-1), "
            "outside the patch. switch_mag then measures distance from the patch to that "
            "point (hundreds of voxels), so switch_ratio saturates far above the 0.5 "
            "threshold at every evaluable vertex. The switch channel therefore carries no "
            "localisation information on this package."
        ),
    }

    # ---------------- per-vertex CSV ----------------
    csv_path = os.path.join(outdir, "per_vertex_metrics.csv")
    rows_idx, cols_idx = np.nonzero(surface_valid)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "row", "col", "generation", "x", "y", "z",
            "detector_evaluable", "evaluation_state",
            "gt_sector", "distance_to_reference", "signed_normal_offset",
            "review", "confidence", "drift_score", "switch_score",
            "chosen_offset", "geom_offset",
        ])
        for r, c in zip(rows_idx, cols_idx):
            is_ev = bool(evaluable[r, c])
            # Detector columns stay blank where no decision was made, so that an
            # absent measurement is never read as a measured zero.
            det = ["", "", "", "", "", ""]
            if is_ev:
                det = [
                    int(review[r, c]),
                    "%.8g" % confidence[r, c],
                    "%.8g" % drift[r, c],
                    "%.8g" % switch[r, c],
                    "%.8g" % chosen[r, c] if np.isfinite(chosen[r, c]) else "",
                    "%.8g" % geom[r, c],
                ]
            w.writerow([
                int(r), int(c), int(gens[r, c]),
                "%.6f" % surf.x[r, c], "%.6f" % surf.y[r, c], "%.6f" % surf.z[r, c],
                int(is_ev), state[r, c],
                int(gt[r, c]), "%.8g" % dist[r, c], "%.8g" % signed[r, c],
            ] + det)

    # ---------------- limitations ----------------
    report["scientific_limitations"] = [
        "The GT mask marks divergence of this trace from an accepted VC3D reference. It "
        "does not prove which physical sheet is correct, and does not prove a permanent "
        "sheet switch occurred.",
        "The package explicitly withdraws the w011 accepted surface as an evaluation "
        "reference; the reference used for distance_to_reference is not re-verifiable "
        "from the contents of this package alone.",
        "n = 100 surface-valid vertices from a single 14x14 patch on one scroll. All "
        "reported rates have wide binomial uncertainty and none of this generalises.",
        "The trace is coarse (step size 10 voxels). A 34-vertex sector is near the "
        "resolution limit for localisation, as the package itself notes.",
        "36 of 100 vertices received no detector decision, so 16 of 34 GT-positive "
        "vertices cannot be scored either way. Any single-number score over 100 vertices "
        "mixes detection behaviour with coverage loss.",
        "Because the review mask covers every evaluable vertex, precision, IoU and Dice "
        "on Domain B are fully determined by the GT prevalence within the covered block "
        "(18/64). They measure prevalence, not discrimination.",
        "Recall of 1.0 on Domain B is not evidence of localisation: a detector that "
        "flags everything achieves it by construction.",
        "The switch channel is saturated and therefore carries no per-vertex information "
        "on this package; correlations involving it are not meaningful.",
        "No confidence intervals or significance tests are reported because a single "
        "saturated, fully-flagged patch does not support them.",
    ]
    report["unsupported_claims"] = [
        "NOT SHOWN: that ScrollAnchor localised the failure sector. It flagged the whole "
        "covered block; the flag carries no positional information.",
        "NOT SHOWN: that the flagged vertices are physically wrong sheet assignments.",
        "NOT SHOWN: that the 16 uncovered GT-positive vertices would have been detected "
        "or missed had they been covered.",
        "NOT SHOWN: that geom_offset measures the same physical quantity as "
        "signed_normal_offset.",
        "NOT SHOWN: any generalisation to other scrolls, cubes, traces or grid sizes.",
        "NOT SHOWN: that the GT mask is a correct physical sheet labelling.",
    ]

    json_path = os.path.join(outdir, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_py(report), fh, indent=2)

    return report, {
        "surface_valid": surface_valid, "evaluable": evaluable, "excluded": excluded,
        "gt": gt, "review": review, "state": state,
    }, csv_path, json_path


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def make_figure(masks, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    sv = masks["surface_valid"]; ev = masks["evaluable"]
    gt = masks["gt"]; rev = masks["review"]; exc = masks["excluded"]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.4))

    def binpanel(ax, mask, title, color):
        img = np.full(mask.shape, 0.0)
        img[sv] = 1.0
        img[mask] = 2.0
        cmap = ListedColormap(["#f2f2f2", "#cccccc", color])
        ax.imshow(img, cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3),
                  interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(0, 14, 2)); ax.set_yticks(range(0, 14, 2))
        ax.tick_params(labelsize=7)

    binpanel(axes[0], gt, "Reference-divergence sector (34)\ndistance_to_reference > 4", "#d62728")
    binpanel(axes[1], ev, "Detector-evaluable (64)\nsurface.valid & normal_valid", "#1f77b4")
    binpanel(axes[2], rev & sv, "Review mask (64)\nall evaluable vertices", "#ff7f0e")

    # categorical outcome map
    cat = np.zeros(sv.shape, dtype=int)          # 0 = outside surface-valid
    cat[sv & ~gt & ev & ~rev] = 1                # evaluated, clean, not flagged
    cat[sv & gt & ev & ~rev] = 2                 # evaluated, GT, not flagged
    cat[sv & ~gt & ev & rev] = 3                 # flagged clean  (FP)
    cat[sv & gt & ev & rev] = 4                  # flagged GT     (TP)
    cat[sv & ~gt & exc] = 5                      # clean, no coverage
    cat[sv & gt & exc] = 6                       # GT, no coverage
    colors = ["#f2f2f2", "#9ecae1", "#fdd0a2", "#ff7f0e", "#d62728", "#dddddd", "#6a51a3"]
    labels = [
        "outside surface-valid",
        "evaluated clean, not flagged (TN)",
        "evaluated GT, not flagged (FN)",
        "flagged clean (FP)",
        "flagged labelled divergence (TP)",
        "clean, NOT EVALUATED",
        "labelled divergence, NOT EVALUATED",
    ]
    axes[3].imshow(cat, cmap=ListedColormap(colors),
                   norm=BoundaryNorm(np.arange(-.5, 7.5), 7), interpolation="nearest")
    axes[3].set_title("Outcome map\n(grey/purple = no detector decision)", fontsize=10)
    axes[3].set_xticks(range(0, 14, 2)); axes[3].set_yticks(range(0, 14, 2))
    axes[3].tick_params(labelsize=7)

    present = [i for i in range(7) if (cat == i).any()]
    fig.legend(
        handles=[Patch(facecolor=colors[i], edgecolor="#888", label=labels[i]) for i in present],
        loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "PHerc1667 candidate-07: frozen ScrollAnchor baseline vs labelled divergence sector "
        "(14x14 grid, 100 surface-valid, 64 detector-evaluable)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    path = os.path.join(outdir, "overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def self_check(report):
    """Lightweight internal consistency check; no production code is exercised."""
    a = report["domain_A_surface_valid"]
    b = report["domain_B_detector_evaluable"]
    c = report["domain_C_literal_emitted_mask"]
    v = report["input_validation"]
    checks = [
        ("surface-valid is 100", a["n_surface_valid"] == 100),
        ("evaluable + not_evaluated == 100",
         a["n_detector_evaluable"] + a["n_not_evaluated"] == 100),
        ("GT positive is 34", v["gt_positive_count"] == 34),
        ("GT split 18 covered + 16 uncovered == 34",
         a["gt_positive"]["reviewed"] + a["gt_positive"]["evaluated_not_reviewed"]
         + a["gt_positive"]["not_evaluated"] == 34),
        ("domain B confusion sums to 64",
         b["confusion"]["tp"] + b["confusion"]["fp"]
         + b["confusion"]["fn"] + b["confusion"]["tn"] == 64),
        ("domain C confusion sums to 100",
         c["confusion"]["tp"] + c["confusion"]["fp"]
         + c["confusion"]["fn"] + c["confusion"]["tn"] == 100),
        ("domain C FN are exactly the uncovered GT vertices", c["fn_are_all_uncovered"]),
        ("GT mask equals distance>4", v["gt_mask_equals_distance_gt_4_on_surface_valid"]),
        ("evaluable equals 3x3 erosion",
         report["detector_evaluable_derivation"]["equals_one_vertex_3x3_erosion_of_surface_valid"]),
        ("evaluable agrees with emitted finite chosen_offset",
         report["detector_evaluable_derivation"]["agrees_with_emitted_finite_chosen_offset"]),
        ("reproduction matches first pass exactly",
         report["reproduction_comparison"]["all_supplied_arrays_bytewise_identical"]
         and report["reproduction_comparison"]["diagnostics_summary_identical"]
         and report["reproduction_comparison"]["review_regions_identical"]),
    ]
    ok = True
    for name, res in checks:
        print("  [%s] %s" % ("PASS" if res else "FAIL", name))
        ok &= bool(res)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="data/external/pherc1667-candidate07")
    ap.add_argument("--reproduced", default="data/external/pherc1667-candidate07/reproduced-default")
    ap.add_argument("--output", default="results/pherc1667-candidate07-external-evaluation")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--package-zip", default=os.environ.get("PACKAGE_ZIP") or None)
    args = ap.parse_args(argv)

    report, masks, csv_path, json_path = evaluate(
        args.package, args.reproduced, args.output, args.repo, args.package_zip
    )
    fig_path = make_figure(masks, args.output)

    print("wrote:", json_path)
    print("wrote:", csv_path)
    print("wrote:", fig_path)
    print("\nself-check:")
    ok = self_check(report)
    print("\nself-check overall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
