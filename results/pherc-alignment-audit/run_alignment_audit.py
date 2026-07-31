"""Standalone alignment / mask-semantics audit for the PHerc 2D render run.

ANALYSIS ARTIFACT - not production code, not imported by the package.

Question under audit
--------------------
The tifxyz-derived validity mask flagged almost none of the visually obvious
black-edge candidates (1 of 200 exported regions touches invalid area; median
tifxyz boundary distance 646 processed px). Three explanations are possible and
this script deliberately does not assume one:

  1. the tifxyz validity mask and the render are spatially misaligned
  2. the visible black areas use different validity semantics from tifxyz
     invalid cells
  3. the conservative render-background extraction is itself insufficient

Method
------
* Reuse the completed detector run (regions.json / metadata.json). Detection is
  NOT rerun and scores are NOT recomputed.
* Reuse the shipped helpers ``load_render``, ``load_tifxyz_valid_mask``,
  ``project_valid_mask`` and ``boundary_distance`` so orientation, projection and
  distance conventions are identical to the run under audit.
* Sweep a fixed set of grayscale thresholds, keeping only near-black components
  connected to the outer raster border (sensitivity analysis, not an algorithm).
* Compare the projected tifxyz invalid mask against those render-derived
  backgrounds under four orientation variants (identity / hflip / vflip / both).
* Audit the eight requested candidates by RANK, resolving their component IDs
  from the current regions.json.

Everything is CPU-only and bounded in memory. Flip variants are taken as numpy
views and the Euclidean distance transform is flip-equivariant, so the expensive
tifxyz distance maps are computed once and reused for all four variants.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label as cc_label

# --------------------------------------------------------------------------- #
# Paths (derived from this file's location; cwd-independent)                    #
# --------------------------------------------------------------------------- #
AUDIT_DIR = Path(__file__).resolve().parent          # code/results/pherc-alignment-audit-*
CODE_ROOT = AUDIT_DIR.parent.parent                  # code/
REPO_ROOT = CODE_ROOT / "scroll-anchor"
sys.path.insert(0, str(REPO_ROOT / "src"))

RENDER_JPG = CODE_ROOT / (
    "PHercParis4-20260623163339-2.4um-0.22m-78keV-volume-"
    "20260411134726-alpha-overlay-combined-ds8.jpg"
)
TIFXYZ_DIR = CODE_ROOT / "data" / "tifxyz_normalized"
RUN_DIR = CODE_ROOT / "results" / "pherc-render-tifxyz-diagnostics-20260731"

from scroll_anchor.render2d import (  # noqa: E402  (path set above)
    RenderParams,
    boundary_distance,
    load_render,
    load_tifxyz_valid_mask,
    project_valid_mask,
)

# --------------------------------------------------------------------------- #
# Fixed audit parameters                                                        #
# --------------------------------------------------------------------------- #
DARK_THRESHOLDS = [1, 4, 8, 12, 16, 24, 32]          # grayscale 0-255, mandated set
ORIENTATIONS = {                                      # (flip_rows, flip_cols)
    "identity": (False, False),
    "hflip": (False, True),
    "vflip": (True, False),
    "hvflip": (True, True),
}
# A border-connected dark component smaller than this cannot plausibly be a
# render boundary region; it is dropped before any metric is computed.
MIN_COMPONENT_PX = 5_000
# Bounded subsampling for boundary-to-boundary distance statistics.
MAX_BOUNDARY_SAMPLES = 200_000
# Ranks requested by the audit brief (1-based positions in regions.json).
SUSPECTED_BLACK_EDGE_RANKS = [5, 6, 8, 10, 15, 16]
INTERIOR_CONTROL_RANKS = [7, 11]
AUDIT_RANKS = sorted(SUSPECTED_BLACK_EDGE_RANKS + INTERIOR_CONTROL_RANKS)
# "touches" uses the same <= 1 px convention as the shipped diagnostics.
TOUCH_PX = 1.0
OVERVIEW_DS = 3          # keeps the overview <= ~4005 px on the long axis
CROP_HALF = 220          # processed px around each candidate centroid

t_start = time.perf_counter()
log = lambda *a: print(f"[{time.perf_counter() - t_start:7.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------- #
# Small helpers                                                                 #
# --------------------------------------------------------------------------- #
def inner_boundary(mask: np.ndarray) -> np.ndarray:
    """4-neighbour inner contour of ``mask``

    A pixel is on the contour if it is set and has an unset 4-neighbour INSIDE
    the raster. The raster frame is therefore not automatically a contour, which
    matters because the edge-connected background necessarily includes the frame.
    """
    b = np.zeros_like(mask)
    b[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
    b[1:, :] |= mask[1:, :] & ~mask[:-1, :]
    b[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
    b[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
    return b


def edt_from(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (float32) to the nearest True pixel of ``mask``"""
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~mask).astype(np.float32)


def subsample_coords(rows: np.ndarray, cols: np.ndarray):
    """Deterministic stride subsample to at most MAX_BOUNDARY_SAMPLES points"""
    n = rows.size
    if n <= MAX_BOUNDARY_SAMPLES:
        return rows, cols, 1
    stride = int(np.ceil(n / MAX_BOUNDARY_SAMPLES))
    return rows[::stride], cols[::stride], stride


def flip_view(a: np.ndarray, flip_rows: bool, flip_cols: bool) -> np.ndarray:
    """Zero-copy flipped view (EDT is flip-equivariant, so maps can be reused)"""
    if flip_rows:
        a = a[::-1, :]
    if flip_cols:
        a = a[:, ::-1]
    return a


def flip_coords(rows, cols, shape, flip_rows: bool, flip_cols: bool):
    h, w = shape
    r = (h - 1 - rows) if flip_rows else rows
    c = (w - 1 - cols) if flip_cols else cols
    return r, c


def edge_connected_background(g8: np.ndarray, thr: int):
    """Near-black pixels connected to the outer raster border, size-filtered

    Returns ``(bg, stats)``. Deliberately conservative: dark papyrus interior
    that is not connected to the border is NOT treated as background.
    """
    dark = g8 <= thr
    dark_frac = float(dark.mean())
    labels, nlab = cc_label(dark)
    if nlab == 0:
        return np.zeros_like(dark), {
            "dark_fraction": dark_frac, "n_components": 0,
            "n_border_components": 0, "n_kept_components": 0,
        }

    border = np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
    ])
    border_labels = np.unique(border)
    border_labels = border_labels[border_labels != 0]

    counts = np.bincount(labels.ravel(), minlength=nlab + 1)
    keep = np.zeros(nlab + 1, dtype=bool)
    big_enough = border_labels[counts[border_labels] >= MIN_COMPONENT_PX]
    keep[big_enough] = True
    bg = keep[labels]
    del labels, dark
    return bg, {
        "dark_fraction": round(dark_frac, 8),
        "n_components": int(nlab),
        "n_border_components": int(border_labels.size),
        "n_kept_components": int(big_enough.size),
    }


# --------------------------------------------------------------------------- #
# 1. Load and verify inputs                                                     #
# --------------------------------------------------------------------------- #
log("loading existing run artifacts")
with open(RUN_DIR / "regions.json", encoding="utf-8") as fh:
    regions_doc = json.load(fh)
with open(RUN_DIR / "metadata.json", encoding="utf-8") as fh:
    run_meta = json.load(fh)
with open(RUN_DIR / "summary.json", encoding="utf-8") as fh:
    run_summary = json.load(fh)

regions = regions_doc["regions"]
n_candidates = len(regions)
scores = [float(r["score"]) for r in regions]
ordering_ok = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

jpg_shape = tuple(run_meta["jpg_shape_rowcol"])
proc_shape = tuple(run_meta["processed_shape_rowcol"])
tifxyz_shape_meta = tuple(run_meta["tifxyz_valid_mask"]["raster_shape_rowcol"])

log(f"candidates={n_candidates} ordering_non_increasing={ordering_ok}")
log(f"jpg={jpg_shape} processed={proc_shape} tifxyz(meta)={tifxyz_shape_meta}")

log("decoding render via the detector's own loader (identical orientation)")
params = RenderParams(working_downsample=int(run_meta["working_downsample"]))
f_img, jpg_shape_chk, proc_shape_chk, scale_row, scale_col = load_render(
    str(RENDER_JPG), params
)
assert tuple(jpg_shape_chk) == jpg_shape, (jpg_shape_chk, jpg_shape)
assert tuple(proc_shape_chk) == proc_shape, (proc_shape_chk, proc_shape)
# load_render returns float32 in [0,1] from an exact /255; recover 0-255 grayscale.
g8 = np.rint(f_img * 255.0).astype(np.uint8)
del f_img
H, W = g8.shape
log(f"grayscale processed raster {g8.shape} dtype={g8.dtype}")

# --------------------------------------------------------------------------- #
# 2. Reconstruct the projected tifxyz mask (identity orientation)               #
# --------------------------------------------------------------------------- #
log("loading + projecting tifxyz valid mask")
tif_valid_raw = load_tifxyz_valid_mask(str(TIFXYZ_DIR))
tifxyz_shape = tuple(int(v) for v in tif_valid_raw.shape)
tif_valid = project_valid_mask(tif_valid_raw, jpg_shape, proc_shape)
tif_invalid = ~tif_valid
tif_invalid_fraction = float(tif_invalid.mean())

scales = {
    "jpg_to_tifxyz_row": jpg_shape[0] / tifxyz_shape[0],
    "jpg_to_tifxyz_col": jpg_shape[1] / tifxyz_shape[1],
    "processed_to_tifxyz_row": proc_shape[0] / tifxyz_shape[0],
    "processed_to_tifxyz_col": proc_shape[1] / tifxyz_shape[1],
}
log(f"tifxyz={tifxyz_shape} invalid_fraction={tif_invalid_fraction:.6f} scales={scales}")

# Distance-to-invalid map, using the SAME helper/convention as the audited run.
tif_bdist = boundary_distance(tif_valid)
tif_contour = inner_boundary(tif_invalid)
tif_contour_rows, tif_contour_cols = np.nonzero(tif_contour)
tif_contour_edt = edt_from(tif_contour)
n_tif_contour = int(tif_contour_rows.size)
del tif_contour, tif_valid, tif_valid_raw
log(f"tifxyz contour pixels={n_tif_contour}")

# Reconstruction self-check: bbox-min of the recomputed map against the stored
# per-region value. The stored value is the min over the true component pixels,
# and bbox superset => recomputed_bbox_min <= stored for a faithful reconstruction.
recon_dev = []
for r in regions:
    r0, c0, r1, c1 = r["bbox_rowcol_processed"]
    recon_dev.append(
        float(tif_bdist[r0:r1 + 1, c0:c1 + 1].min()) - float(r["boundary_distance_px"])
    )
recon_dev = np.array(recon_dev)
recon_check = {
    "rule": "bbox-min of recomputed map minus stored component-min; must be <= 0",
    "max_deviation": float(recon_dev.max()),
    "min_deviation": float(recon_dev.min()),
    "median_deviation": float(np.median(recon_dev)),
    "n_violations_gt_0": int((recon_dev > 1e-6).sum()),
    "faithful": bool((recon_dev <= 1e-6).all()),
}
log(f"reconstruction check: {recon_check}")

# --------------------------------------------------------------------------- #
# 3-5. Threshold sweep: background masks, orientation metrics, candidate metrics #
# --------------------------------------------------------------------------- #
audit_rows = []
for rank in AUDIT_RANKS:
    reg = regions[rank - 1]
    r0, c0, r1, c1 = reg["bbox_rowcol_processed"]
    audit_rows.append({
        "rank": rank,
        "component_id": int(reg["id"]),
        "group": "suspected_black_edge" if rank in SUSPECTED_BLACK_EDGE_RANKS
                 else "interior_control",
        "score": float(reg["score"]),
        "direction": reg["direction"],
        "size_pixels_processed": int(reg["size_pixels_processed"]),
        "bbox_rowcol_processed": [r0, c0, r1, c1],
        "centroid_rowcol_jpg": reg["centroid_rowcol_jpg"],
        "tifxyz_boundary_distance_px": float(reg["boundary_distance_px"]),
        "tifxyz_invalid_overlap_fraction": float(reg["invalid_overlap_fraction"]),
        "tifxyz_touches_invalid_boundary": bool(reg["touches_invalid_boundary"]),
        "tifxyz_bbox_min_distance_px": round(
            float(tif_bdist[r0:r1 + 1, c0:c1 + 1].min()), 3),
        "per_threshold": {},
    })

orientation_rows = []
threshold_stats = {}

for thr in DARK_THRESHOLDS:
    log(f"--- threshold {thr} ---")
    bg, bstats = edge_connected_background(g8, thr)
    bg_fraction = float(bg.mean())
    log(f"    background fraction={bg_fraction:.6f} kept={bstats['n_kept_components']}"
        f" of {bstats['n_border_components']} border components")

    # Distance to ANY near-black pixel (ignores edge-connectivity). This is the
    # discriminator between "black area nearby but not border-connected" and
    # "no black area nearby at all".
    d_any_dark = boundary_distance(g8 > thr)
    any_dark_min = {
        row["rank"]: float(d_any_dark[
            row["bbox_rowcol_processed"][0]:row["bbox_rowcol_processed"][2] + 1,
            row["bbox_rowcol_processed"][1]:row["bbox_rowcol_processed"][3] + 1].min())
        for row in audit_rows
    }
    del d_any_dark

    # Distance to the edge-connected background, same convention as tifxyz.
    render_bdist = boundary_distance(~bg)
    for row in audit_rows:
        r0, c0, r1, c1 = row["bbox_rowcol_processed"]
        sub_bg = bg[r0:r1 + 1, c0:c1 + 1]
        sub_g = g8[r0:r1 + 1, c0:c1 + 1]
        dmin = float(render_bdist[r0:r1 + 1, c0:c1 + 1].min())
        row["per_threshold"][str(thr)] = {
            "render_bg_min_distance_px": round(dmin, 3),
            "render_bg_overlap_fraction": round(float(sub_bg.mean()), 6),
            "bbox_near_black_fraction": round(float((sub_g <= thr).mean()), 6),
            "any_dark_min_distance_px": round(any_dark_min[row["rank"]], 3),
            "touches_render_bg": bool(dmin <= TOUCH_PX),
        }
    del render_bdist

    # Orientation comparison against this background.
    bg_contour = inner_boundary(bg)
    bgc_rows, bgc_cols = np.nonzero(bg_contour)
    n_bg_contour = int(bgc_rows.size)
    bg_contour_edt = edt_from(bg_contour)
    del bg_contour
    s_bgc_r, s_bgc_c, bg_stride = subsample_coords(bgc_rows, bgc_cols)
    s_tif_r, s_tif_c, tif_stride = subsample_coords(tif_contour_rows, tif_contour_cols)

    bg_sum = float(bg.sum())
    for name, (fr, fc) in ORIENTATIONS.items():
        inv_v = flip_view(tif_invalid, fr, fc)
        inter = float(np.count_nonzero(bg & inv_v))
        inv_sum = float(inv_v.sum())
        union = bg_sum + inv_sum - inter
        # Distance from render-background contour pixels to the tifxyz contour.
        tif_edt_v = flip_view(tif_contour_edt, fr, fc)
        d_bg_to_tif = tif_edt_v[s_bgc_r, s_bgc_c]
        # Distance from tifxyz contour pixels (under the variant) to the bg contour.
        vr, vc = flip_coords(s_tif_r, s_tif_c, (H, W), fr, fc)
        d_tif_to_bg = bg_contour_edt[vr, vc]

        orientation_rows.append({
            "threshold": thr,
            "orientation": name,
            "iou": round(inter / union, 6) if union else 0.0,
            "render_bg_precision_wrt_tifxyz_invalid":
                round(inter / bg_sum, 6) if bg_sum else 0.0,
            "render_bg_recall_wrt_tifxyz_invalid":
                round(inter / inv_sum, 6) if inv_sum else 0.0,
            "median_dist_render_bg_boundary_to_tifxyz_boundary_px":
                round(float(np.median(d_bg_to_tif)), 3),
            "median_dist_tifxyz_boundary_to_render_bg_boundary_px":
                round(float(np.median(d_tif_to_bg)), 3),
            "p90_dist_render_bg_boundary_to_tifxyz_boundary_px":
                round(float(np.percentile(d_bg_to_tif, 90)), 3),
            "p90_dist_tifxyz_boundary_to_render_bg_boundary_px":
                round(float(np.percentile(d_tif_to_bg, 90)), 3),
            "n_render_bg_boundary_px": n_bg_contour,
            "n_tifxyz_boundary_px": n_tif_contour,
            "render_bg_boundary_stride": bg_stride,
            "tifxyz_boundary_stride": tif_stride,
        })

    threshold_stats[str(thr)] = {
        "background_fraction": round(bg_fraction, 8),
        "n_render_bg_boundary_px": n_bg_contour,
        **bstats,
    }
    del bg, bg_contour_edt

log("sweep complete")

# --------------------------------------------------------------------------- #
# Representative threshold: middle of the flattest background-fraction plateau  #
# --------------------------------------------------------------------------- #
fr_list = [threshold_stats[str(t)]["background_fraction"] for t in DARK_THRESHOLDS]
rel_change = []
for i, t in enumerate(DARK_THRESHOLDS):
    lo = fr_list[max(0, i - 1)]
    hi = fr_list[min(len(fr_list) - 1, i + 1)]
    base = max(fr_list[i], 1e-9)
    rel_change.append(abs(hi - lo) / base)
# Only consider thresholds that actually produce a non-trivial background.
eligible = [i for i, v in enumerate(fr_list) if v > 1e-4]
if eligible:
    rep_idx = min(eligible, key=lambda i: rel_change[i])
else:
    rep_idx = int(np.argmax(fr_list))
REPRESENTATIVE_THRESHOLD = DARK_THRESHOLDS[rep_idx]
threshold_selection = {
    "rule": ("threshold minimising the local relative change of background "
             "fraction across its neighbours (flattest plateau), restricted to "
             "thresholds whose background fraction exceeds 1e-4"),
    "background_fractions": {str(t): fr_list[i] for i, t in enumerate(DARK_THRESHOLDS)},
    "local_relative_change": {str(t): round(rel_change[i], 6)
                              for i, t in enumerate(DARK_THRESHOLDS)},
    "selected": REPRESENTATIVE_THRESHOLD,
}
log(f"representative threshold = {REPRESENTATIVE_THRESHOLD}")

# --------------------------------------------------------------------------- #
# Aggregate orientation verdicts                                                #
# --------------------------------------------------------------------------- #
def _best(metric: str, maximise: bool):
    per_or = {}
    for name in ORIENTATIONS:
        vals = [r[metric] for r in orientation_rows if r["orientation"] == name]
        per_or[name] = float(np.mean(vals))
    best = (max if maximise else min)(per_or, key=per_or.get)
    return best, per_or


best_by_metric = {}
for metric, maximise in [
    ("iou", True),
    ("render_bg_precision_wrt_tifxyz_invalid", True),
    ("render_bg_recall_wrt_tifxyz_invalid", True),
    ("median_dist_render_bg_boundary_to_tifxyz_boundary_px", False),
    ("median_dist_tifxyz_boundary_to_render_bg_boundary_px", False),
]:
    best, per_or = _best(metric, maximise)
    best_by_metric[metric] = {"best": best, "mean_by_orientation":
                              {k: round(v, 6) for k, v in per_or.items()}}

orientation_votes = [v["best"] for v in best_by_metric.values()]
metrics_agree = len(set(orientation_votes)) == 1
best_orientation = max(set(orientation_votes), key=orientation_votes.count)
log(f"best orientation={best_orientation} agree={metrics_agree} votes={orientation_votes}")

# --------------------------------------------------------------------------- #
# Candidate-level stability + conclusion                                        #
# --------------------------------------------------------------------------- #
for row in audit_rows:
    touches = [row["per_threshold"][str(t)]["touches_render_bg"] for t in DARK_THRESHOLDS]
    dists = [row["per_threshold"][str(t)]["render_bg_min_distance_px"]
             for t in DARK_THRESHOLDS]
    row["touches_render_bg_by_threshold"] = touches
    row["render_bg_min_distance_by_threshold"] = dists
    row["stable_across_thresholds"] = bool(len(set(touches)) == 1)
    row["n_thresholds_touching"] = int(sum(touches))
    row["min_distance_range_px"] = [min(dists), max(dists)]

suspected = [r for r in audit_rows if r["group"] == "suspected_black_edge"]
controls = [r for r in audit_rows if r["group"] == "interior_control"]

NEAR_PX = 5.0     # "close to the render-derived boundary"
INTERIOR_PX = 50.0  # "consistently interior"

suspected_close = all(
    all(r["per_threshold"][str(t)]["render_bg_min_distance_px"] <= NEAR_PX
        for t in DARK_THRESHOLDS if threshold_stats[str(t)]["background_fraction"] > 1e-4)
    for r in suspected
)
controls_interior = all(
    all(r["per_threshold"][str(t)]["render_bg_min_distance_px"] >= INTERIOR_PX
        for t in DARK_THRESHOLDS if threshold_stats[str(t)]["background_fraction"] > 1e-4)
    for r in controls
)
all_stable = all(r["stable_across_thresholds"] for r in audit_rows)

mean_iou_best = best_by_metric["iou"]["mean_by_orientation"][best_orientation]
mean_iou_identity = best_by_metric["iou"]["mean_by_orientation"]["identity"]
iou_uplift = mean_iou_best - mean_iou_identity
poor_agreement_all = all(
    v <= 0.20 for v in best_by_metric["iou"]["mean_by_orientation"].values()
)
bg_captures_anything = any(
    threshold_stats[str(t)]["background_fraction"] > 1e-4 for t in DARK_THRESHOLDS
)

if not bg_captures_anything:
    conclusion = "render_background_extraction_inconclusive"
    rationale = ("edge-connected near-black extraction produced essentially no "
                 "background at any threshold, so the render side carries no signal")
elif metrics_agree and best_orientation != "identity" and iou_uplift > 0.10 and all_stable:
    conclusion = "likely_spatial_misalignment"
    rationale = (f"variant {best_orientation} improves mean IoU by {iou_uplift:.3f} over "
                 "identity, consistently across metrics and thresholds")
elif poor_agreement_all and suspected_close and controls_interior and all_stable:
    conclusion = "likely_different_validity_semantics"
    rationale = ("no orientation variant achieves meaningful mask agreement, yet the "
                 "render-derived background separates the suspected black-edge "
                 "candidates from the interior controls stably across thresholds")
elif not all_stable:
    conclusion = "render_background_extraction_inconclusive"
    rationale = ("candidate classification flips with the dark threshold, so the "
                 "render-background extraction is not yet a stable reference")
else:
    conclusion = "mixed_or_inconclusive"
    rationale = "evidence does not satisfy any single interpretation cleanly"

log(f"conclusion={conclusion}")

# --------------------------------------------------------------------------- #
# 6. Figures                                                                    #
# --------------------------------------------------------------------------- #
log("rebuilding representative background for figures")
bg_rep, _ = edge_connected_background(g8, REPRESENTATIVE_THRESHOLD)
bg_rep_contour = inner_boundary(bg_rep)
tif_inv_best = flip_view(tif_invalid, *ORIENTATIONS[best_orientation])
tif_contour_best = inner_boundary(np.ascontiguousarray(tif_inv_best))
render_bdist_rep = boundary_distance(~bg_rep)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def block_max(m: np.ndarray, f: int) -> np.ndarray:
    """Block-max downsample a boolean mask so thin contours survive"""
    h, w = m.shape
    h2, w2 = (h // f) * f, (w // f) * f
    return m[:h2, :w2].reshape(h2 // f, f, w2 // f, f).any(axis=(1, 3))


log("writing mask_alignment_overview.png")
f_ds = OVERVIEW_DS
h2, w2 = (H // f_ds) * f_ds, (W // f_ds) * f_ds
base = g8[:h2, :w2].reshape(h2 // f_ds, f_ds, w2 // f_ds, f_ds).mean(axis=(1, 3))
rgb = np.repeat((base / 255.0)[:, :, None], 3, axis=2)
tif_c_ds = block_max(tif_contour_best, f_ds)
bg_c_ds = block_max(bg_rep_contour, f_ds)
rgb[tif_c_ds] = [1.0, 0.15, 0.15]     # tifxyz invalid contour  (red)
rgb[bg_c_ds] = [0.15, 0.75, 1.0]      # render background contour (cyan)

fig_w = rgb.shape[1] / 100.0
fig_h = rgb.shape[0] / 100.0 + 1.1
fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
ax = fig.add_axes([0.005, 0.01, 0.99, 0.90])
ax.imshow(rgb, interpolation="nearest")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(handles=[
    Line2D([0], [0], color="#ff2626", lw=3,
           label=f"projected tifxyz invalid contour ({best_orientation})"),
    Line2D([0], [0], color="#26bfff", lw=3,
           label=f"render edge-connected background contour (thr={REPRESENTATIVE_THRESHOLD})"),
], loc="upper right", fontsize=11, framealpha=0.9)
fig.suptitle(
    f"Mask alignment overview  |  processed {H}x{W} shown at 1/{f_ds}  |  "
    f"best orientation by aggregate metrics: {best_orientation}  |  "
    f"tifxyz invalid {tif_invalid_fraction:.3f} vs render background "
    f"{threshold_stats[str(REPRESENTATIVE_THRESHOLD)]['background_fraction']:.3f}",
    fontsize=13,
)
fig.savefig(AUDIT_DIR / "mask_alignment_overview.png", dpi=100)
plt.close(fig)

log("writing candidate_audit_crops.png")
ncol, nrow = 4, 2
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.3, nrow * 4.9))
axes = np.atleast_1d(axes).ravel()
for ax, row in zip(axes, audit_rows):
    r0, c0, r1, c1 = row["bbox_rowcol_processed"]
    cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
    tr, br = max(0, cr - CROP_HALF), min(H, cr + CROP_HALF)
    lc, rc = max(0, cc - CROP_HALF), min(W, cc + CROP_HALF)
    ax.imshow(g8[tr:br, lc:rc], cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    tc = tif_contour_best[tr:br, lc:rc]
    bc = bg_rep_contour[tr:br, lc:rc]
    ys, xs = np.nonzero(tc)
    if ys.size:
        ax.scatter(xs, ys, s=0.6, c="#ff2626", linewidths=0, label="_tif")
    ys, xs = np.nonzero(bc)
    if ys.size:
        ax.scatter(xs, ys, s=0.6, c="#26bfff", linewidths=0, label="_bg")
    ax.add_patch(Rectangle((c0 - lc, r0 - tr), max(2, c1 - c0), max(2, r1 - r0),
                           fill=False, edgecolor="#ffd000", linewidth=1.8))

    pt = row["per_threshold"][str(REPRESENTATIVE_THRESHOLD)]
    tag = "SUSPECT" if row["group"] == "suspected_black_edge" else "CONTROL"
    ax.set_title(
        f"rank {row['rank']}  id {row['component_id']}  [{tag}]\n"
        f"score {row['score']:.3f}   tifxyz dist {row['tifxyz_boundary_distance_px']:.0f} px\n"
        f"render-bg dist {pt['render_bg_min_distance_px']:.1f} px   "
        f"overlap {pt['render_bg_overlap_fraction']:.2f}\n"
        f"bbox near-black {pt['bbox_near_black_fraction']:.2f}   "
        f"stable {row['stable_across_thresholds']}",
        fontsize=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes[len(audit_rows):]:
    ax.axis("off")
fig.legend(handles=[
    Line2D([0], [0], color="#ff2626", lw=3, label="projected tifxyz invalid contour"),
    Line2D([0], [0], color="#26bfff", lw=3,
           label=f"render edge-connected background contour (thr={REPRESENTATIVE_THRESHOLD})"),
    Line2D([0], [0], color="#ffd000", lw=3, label="candidate bbox (detector, unchanged)"),
], loc="lower center", ncol=3, fontsize=9)
fig.suptitle(
    "Candidate audit crops - exploratory render anomalies, not confirmed errors "
    f"(representative dark threshold {REPRESENTATIVE_THRESHOLD})", fontsize=11)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(AUDIT_DIR / "candidate_audit_crops.png", dpi=125)
plt.close(fig)

# --------------------------------------------------------------------------- #
# 7. Machine-readable outputs                                                   #
# --------------------------------------------------------------------------- #
log("writing JSON/CSV outputs")
with open(AUDIT_DIR / "orientation_comparison.json", "w", encoding="utf-8") as fh:
    json.dump({
        "note": ("Diagnostic comparison only. No single metric is sufficient to "
                 "declare alignment."),
        "subsampling_rule": (f"deterministic stride subsample to at most "
                             f"{MAX_BOUNDARY_SAMPLES} boundary pixels per set; "
                             "stride = ceil(n / max), recorded per row"),
        "orientation_variants": list(ORIENTATIONS),
        "transpose_tested": False,
        "transpose_reason": "raster dimensions (627x2403 vs 3135x12015) make it incompatible",
        "per_threshold_per_orientation": orientation_rows,
        "best_by_metric": best_by_metric,
        "metrics_agree_on_one_orientation": metrics_agree,
        "best_orientation_overall": best_orientation,
        "mean_iou_uplift_over_identity": round(iou_uplift, 6),
    }, fh, indent=2)

with open(AUDIT_DIR / "candidate_metrics.json", "w", encoding="utf-8") as fh:
    json.dump({
        "note": ("Candidate footprint is approximated by its bounding box: "
                 "regions.json stores bbox and size but not the component pixel "
                 "mask, and diagnostics.npz is block-mean downsampled. The audited "
                 "regions are thin seams, so the bbox is a tight proxy. Detector "
                 "scores and ranking are reused unchanged."),
        "touch_rule_px": TOUCH_PX,
        "near_px": NEAR_PX,
        "interior_px": INTERIOR_PX,
        "representative_threshold": REPRESENTATIVE_THRESHOLD,
        "candidates": audit_rows,
    }, fh, indent=2)

csv_path = AUDIT_DIR / "candidate_metrics.csv"
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow([
        "rank", "component_id", "group", "score", "direction",
        "bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1",
        "centroid_row_jpg", "centroid_col_jpg",
        "tifxyz_boundary_distance_px", "tifxyz_invalid_overlap_fraction",
        "tifxyz_touches_invalid_boundary", "threshold",
        "render_bg_min_distance_px", "render_bg_overlap_fraction",
        "bbox_near_black_fraction", "any_dark_min_distance_px",
        "touches_render_bg", "stable_across_thresholds",
    ])
    for row in audit_rows:
        r0, c0, r1, c1 = row["bbox_rowcol_processed"]
        for thr in DARK_THRESHOLDS:
            pt = row["per_threshold"][str(thr)]
            w.writerow([
                row["rank"], row["component_id"], row["group"], row["score"],
                row["direction"], r0, c0, r1, c1,
                row["centroid_rowcol_jpg"][0], row["centroid_rowcol_jpg"][1],
                row["tifxyz_boundary_distance_px"],
                row["tifxyz_invalid_overlap_fraction"],
                row["tifxyz_touches_invalid_boundary"], thr,
                pt["render_bg_min_distance_px"], pt["render_bg_overlap_fraction"],
                pt["bbox_near_black_fraction"], pt["any_dark_min_distance_px"],
                pt["touches_render_bg"], row["stable_across_thresholds"],
            ])

with open(AUDIT_DIR / "audit_summary.json", "w", encoding="utf-8") as fh:
    json.dump({
        "format": "scroll-anchor.render-alignment-audit/v0",
        "note": ("Analysis-only audit. No repository code changed, no detector "
                 "rerun, no scores or ranking altered, no boundary penalty "
                 "introduced. Candidates remain exploratory."),
        "input_paths": {
            "source_render_jpg": str(RENDER_JPG),
            "tifxyz_normalized_dir": str(TIFXYZ_DIR),
            "existing_run_dir": str(RUN_DIR),
            "audit_output_dir": str(AUDIT_DIR),
        },
        "input_shapes": {
            "source_render_rowcol": list(jpg_shape),
            "processed_raster_rowcol": list(proc_shape),
            "tifxyz_raster_rowcol": list(tifxyz_shape),
            "tifxyz_shape_from_run_metadata": list(tifxyz_shape_meta),
            "shapes_consistent_with_run": bool(tifxyz_shape == tifxyz_shape_meta),
        },
        "scales": {k: round(v, 6) for k, v in scales.items()},
        "scales_exact_integer": {
            "jpg_to_tifxyz": bool(jpg_shape[0] % tifxyz_shape[0] == 0
                                  and jpg_shape[1] % tifxyz_shape[1] == 0),
            "processed_to_tifxyz": bool(proc_shape[0] % tifxyz_shape[0] == 0
                                        and proc_shape[1] % tifxyz_shape[1] == 0),
        },
        "candidate_count": n_candidates,
        "candidate_ordering_non_increasing_score": ordering_ok,
        "detector_run_reused": {
            "region_counts": run_summary["region_counts"],
            "exported_score_range": run_summary["exported_score_range"],
            "detection_rerun": False,
            "scores_recomputed": False,
        },
        "tifxyz_projection_reconstruction_check": recon_check,
        "tifxyz_invalid_fraction_processed": round(tif_invalid_fraction, 6),
        "thresholds_tested": DARK_THRESHOLDS,
        "threshold_stats": threshold_stats,
        "min_component_px": MIN_COMPONENT_PX,
        "orientation_variants_tested": list(ORIENTATIONS),
        "transpose_tested": False,
        "representative_threshold": REPRESENTATIVE_THRESHOLD,
        "representative_threshold_selection": threshold_selection,
        "best_orientation_by_metric": {k: v["best"] for k, v in best_by_metric.items()},
        "best_orientation_overall": best_orientation,
        "metrics_agree_on_one_orientation": metrics_agree,
        "mean_iou_by_orientation": best_by_metric["iou"]["mean_by_orientation"],
        "mean_iou_uplift_over_identity": round(iou_uplift, 6),
        "suspected_black_edge_ranks": SUSPECTED_BLACK_EDGE_RANKS,
        "interior_control_ranks": INTERIOR_CONTROL_RANKS,
        "suspected_black_edge_consistently_close_to_render_boundary": suspected_close,
        "interior_controls_remain_interior": controls_interior,
        "candidate_classification_stable_across_thresholds": all_stable,
        "candidate_component_ids_by_rank": {
            str(r["rank"]): r["component_id"] for r in audit_rows},
        "conclusion_status": conclusion,
        "conclusion_rationale": rationale,
        "limitations": [
            "Candidate footprints are approximated by their bounding boxes; "
            "regions.json does not store component pixel masks.",
            "Alignment between the tifxyz grid and the render remains an "
            "assumption; only the exact integer raster scale is verified.",
            "Edge-connected near-black extraction is a sensitivity probe, not a "
            "validated background segmentation.",
            "No CT evidence, no surface-normal geometry, no coordinate mapping.",
            "Candidates are neither confirmed surface errors nor confirmed false "
            "positives.",
        ],
    }, fh, indent=2)

log("done")
print(json.dumps({
    "conclusion": conclusion,
    "best_orientation": best_orientation,
    "metrics_agree": metrics_agree,
    "representative_threshold": REPRESENTATIVE_THRESHOLD,
    "suspected_close": suspected_close,
    "controls_interior": controls_interior,
    "all_stable": all_stable,
}, indent=2))
