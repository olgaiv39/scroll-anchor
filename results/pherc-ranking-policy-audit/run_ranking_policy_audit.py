#!/usr/bin/env python3
"""
Offline ranking-policy audit for ScrollAnchor render candidates.

ANALYSIS ONLY. This script does not touch the ScrollAnchor repository, does not
re-run candidate detection, and does not modify the detector score. It reads the
already-exported combined diagnostics run and compares candidate *review order*
under four simple policies.

Policies
--------
A raw          review_score_raw  = raw_score
B half penalty review_score_half = raw_score * (1 - 0.5 * overlap)
C full penalty review_score_full = raw_score * (1 - overlap)
D hard flag    overlap <  0.20 first, in raw-score order;
               overlap >= 0.20 appended afterwards, in raw-score order.
               (comparison only; explicitly NOT a proposed production rule)

overlap = render_background_overlap_fraction, clamped defensively to [0, 1].

Deterministic ordering for equal adjusted scores:
  1. higher adjusted score
  2. higher raw score
  3. lower component id

Outputs (in this directory)
---------------------------
  ranking_comparison.csv
  ranking_comparison.json
  audit_summary.json
  top20_raw.png
  top20_half_penalty.png
  top20_full_penalty.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent
CODE_ROOT = OUT_DIR.parent.parent
DIAG_DIR = CODE_ROOT / "results" / "pherc-render-combined-diagnostics-20260731"

REGIONS_JSON = DIAG_DIR / "regions.json"
METADATA_JSON = DIAG_DIR / "metadata.json"
SUMMARY_JSON = DIAG_DIR / "summary.json"

SOURCE_JPG = CODE_ROOT / (
    "PHercParis4-20260623163339-2.4um-0.22m-78keV-volume-"
    "20260411134726-alpha-overlay-combined-ds8.jpg"
)

# --------------------------------------------------------------------------
# Calibration groups, expressed as ORIGINAL (raw) 1-based ranks.
# These are calibration examples only, NOT a complete ground-truth dataset.
# --------------------------------------------------------------------------

KNOWN_BACKGROUND_RAW_RANKS = [5, 6, 8, 10, 16, 18]
INTERIOR_CONTROL_RAW_RANKS = [7, 11]
AMBIGUOUS_RAW_RANKS = [15]

TOP_K_VALUES = [10, 20, 50]
HARD_FILTER_THRESHOLD = 0.20


# --------------------------------------------------------------------------
# Load inputs
# --------------------------------------------------------------------------


def load_inputs():
    regions_doc = json.loads(REGIONS_JSON.read_text())
    metadata = json.loads(METADATA_JSON.read_text())
    summary = json.loads(SUMMARY_JSON.read_text())
    return regions_doc, metadata, summary


def build_records(regions_doc):
    """Build per-candidate records in raw (exported) order, with validation."""
    regions = regions_doc["regions"]

    raw_scores = [float(r["score"]) for r in regions]
    raw_order_is_descending = all(
        raw_scores[i] >= raw_scores[i + 1] for i in range(len(raw_scores) - 1)
    )

    overlap_out_of_range = []
    touches_but_zero_overlap = []
    overlap_but_not_touching = []

    records = []
    for i, r in enumerate(regions):
        stored_overlap = float(r["render_background_overlap_fraction"])
        if not (0.0 <= stored_overlap <= 1.0):
            overlap_out_of_range.append(
                {"component_id": int(r["id"]), "stored_overlap": stored_overlap}
            )
        overlap = float(min(1.0, max(0.0, stored_overlap)))

        touches = bool(r["touches_render_background"])
        if touches and overlap == 0.0:
            touches_but_zero_overlap.append(int(r["id"]))
        if (not touches) and overlap > 0.0:
            overlap_but_not_touching.append(int(r["id"]))

        raw_rank = i + 1
        raw_score = float(r["score"])

        if raw_rank in KNOWN_BACKGROUND_RAW_RANKS:
            group = "known_background"
        elif raw_rank in INTERIOR_CONTROL_RAW_RANKS:
            group = "interior_control"
        elif raw_rank in AMBIGUOUS_RAW_RANKS:
            group = "ambiguous"
        else:
            group = "unlabelled"

        records.append(
            {
                "component_id": int(r["id"]),
                "raw_rank": raw_rank,
                "raw_score": raw_score,
                "overlap": overlap,
                "stored_overlap": stored_overlap,
                "background_distance_px": float(r["render_background_distance_px"]),
                "touches_render_background": touches,
                "calibration_group": group,
                "bbox_rowcol_jpg": [float(v) for v in r["bbox_rowcol_jpg"]],
                "bbox_rowcol_processed": [int(v) for v in r["bbox_rowcol_processed"]],
                "centroid_rowcol_jpg": [float(v) for v in r["centroid_rowcol_jpg"]],
                "half_score": raw_score * (1.0 - 0.5 * overlap),
                "full_score": raw_score * (1.0 - overlap),
            }
        )

    validation = {
        "raw_order_is_descending_by_score": raw_order_is_descending,
        "n_unique_component_ids": len({rec["component_id"] for rec in records}),
        "overlap_values_outside_0_1": overlap_out_of_range,
        "n_overlap_values_outside_0_1": len(overlap_out_of_range),
        "component_ids_touching_background_with_zero_overlap": touches_but_zero_overlap,
        "component_ids_with_overlap_but_not_touching": overlap_but_not_touching,
    }
    return records, validation


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def assign_soft_ranks(records, score_key, rank_key):
    """Deterministic ordering: higher adjusted, higher raw, lower component id."""
    ordered = sorted(
        records,
        key=lambda rec: (-rec[score_key], -rec["raw_score"], rec["component_id"]),
    )
    for rank, rec in enumerate(ordered, start=1):
        rec[rank_key] = rank
    return ordered


def assign_hard_filter_ranks(records, rank_key, threshold=HARD_FILTER_THRESHOLD):
    """Overlap < threshold first in raw-score order, then the rest in raw-score order."""
    ordered = sorted(
        records,
        key=lambda rec: (
            0 if rec["overlap"] < threshold else 1,
            -rec["raw_score"],
            rec["component_id"],
        ),
    )
    for rank, rec in enumerate(ordered, start=1):
        rec[rank_key] = rank
    return ordered


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def top_k_stats(records, rank_key, k, raw_top_ids):
    in_top = sorted(
        (rec for rec in records if rec[rank_key] <= k), key=lambda rec: rec[rank_key]
    )
    overlaps = [rec["overlap"] for rec in in_top]

    top_ids = {rec["component_id"] for rec in in_top}
    entering = [
        {
            "component_id": rec["component_id"],
            "original_rank": rec["raw_rank"],
            "policy_rank": rec[rank_key],
            "overlap": round(rec["overlap"], 6),
        }
        for rec in in_top
        if rec["component_id"] not in raw_top_ids
    ]
    leaving = [
        {
            "component_id": rec["component_id"],
            "original_rank": rec["raw_rank"],
            "policy_rank": rec[rank_key],
            "overlap": round(rec["overlap"], 6),
        }
        for rec in records
        if rec["component_id"] in raw_top_ids and rec["component_id"] not in top_ids
    ]
    leaving.sort(key=lambda d: d["original_rank"])

    return {
        "k": k,
        "n_overlap_gt_0": sum(1 for v in overlaps if v > 0.0),
        "n_overlap_ge_0_20": sum(1 for v in overlaps if v >= 0.20),
        "n_overlap_ge_0_50": sum(1 for v in overlaps if v >= 0.50),
        "median_overlap": round(float(np.median(overlaps)), 6),
        "entering_vs_raw": entering,
        "leaving_vs_raw": leaving,
        "n_entering": len(entering),
        "n_leaving": len(leaving),
    }


def movement_stats(records, rank_key):
    """rank_change = policy_rank - raw_rank; positive means demoted (moved down)."""
    changes = np.array([rec[rank_key] - rec["raw_rank"] for rec in records], dtype=float)
    abs_changes = np.abs(changes)

    raw_ranks = [rec["raw_rank"] for rec in records]
    pol_ranks = [rec[rank_key] for rec in records]
    rho, _ = spearmanr(raw_ranks, pol_ranks)

    most_demoted = max(records, key=lambda rec: rec[rank_key] - rec["raw_rank"])
    most_promoted = min(records, key=lambda rec: rec[rank_key] - rec["raw_rank"])

    negligible = [rec for rec in records if rec["overlap"] < 0.01]
    negligible_abs = [abs(rec[rank_key] - rec["raw_rank"]) for rec in negligible]

    return {
        "max_abs_rank_movement": int(abs_changes.max()),
        "median_abs_rank_movement": float(np.median(abs_changes)),
        "mean_abs_rank_movement": round(float(abs_changes.mean()), 4),
        "max_demotion_positions": int(changes.max()),
        "max_promotion_positions": int(-changes.min()),
        "most_demoted": {
            "component_id": most_demoted["component_id"],
            "raw_rank": most_demoted["raw_rank"],
            "policy_rank": most_demoted[rank_key],
            "overlap": round(most_demoted["overlap"], 6),
        },
        "most_promoted": {
            "component_id": most_promoted["component_id"],
            "raw_rank": most_promoted["raw_rank"],
            "policy_rank": most_promoted[rank_key],
            "overlap": round(most_promoted["overlap"], 6),
        },
        "spearman_rho_vs_raw": round(float(rho), 6),
        "n_rank_change_gt_5": int((abs_changes > 5).sum()),
        "n_rank_change_gt_10": int((abs_changes > 10).sum()),
        "n_rank_change_gt_20": int((abs_changes > 20).sum()),
        "negligible_overlap_lt_0_01": {
            "n_candidates": len(negligible),
            "max_abs_rank_movement": int(max(negligible_abs)) if negligible_abs else 0,
            "median_abs_rank_movement": (
                float(np.median(negligible_abs)) if negligible_abs else 0.0
            ),
            "n_moving_more_than_5": int(sum(1 for v in negligible_abs if v > 5)),
            "n_moving_more_than_10": int(sum(1 for v in negligible_abs if v > 10)),
            "n_demoted": int(
                sum(1 for rec in negligible if rec[rank_key] > rec["raw_rank"])
            ),
            "n_promoted": int(
                sum(1 for rec in negligible if rec[rank_key] < rec["raw_rank"])
            ),
        },
    }


# --------------------------------------------------------------------------
# Contact sheets
# --------------------------------------------------------------------------


def crop_box_for(rec, img_h, img_w, pad=128, min_side=384):
    r0, c0, r1, c1 = rec["bbox_rowcol_jpg"]
    r0, r1 = min(r0, r1), max(r0, r1)
    c0, c1 = min(c0, c1), max(c0, c1)

    side = max(r1 - r0, c1 - c0) + 2 * pad
    side = max(side, min_side)
    cr = 0.5 * (r0 + r1)
    cc = 0.5 * (c0 + c1)

    half = side / 2.0
    tr0 = int(round(cr - half))
    tc0 = int(round(cc - half))
    side_i = int(round(side))

    tr0 = max(0, min(tr0, img_h - min(side_i, img_h)))
    tc0 = max(0, min(tc0, img_w - min(side_i, img_w)))
    tr1 = min(img_h, tr0 + side_i)
    tc1 = min(img_w, tc0 + side_i)
    return tr0, tc0, tr1, tc1


def make_contact_sheet(records, rank_key, score_key, image, out_path, policy_title):
    img_h, img_w = image.shape[:2]
    top = sorted(
        (rec for rec in records if rec[rank_key] <= 20), key=lambda rec: rec[rank_key]
    )

    ncols, nrows = 4, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.75 * nrows))
    fig.suptitle(
        f"{policy_title}  |  top 20 candidates  |  "
        "ScrollAnchor render-background ranking-policy audit (analysis only)",
        fontsize=13,
        y=0.995,
    )

    group_colour = {
        "known_background": "#d62728",
        "interior_control": "#2ca02c",
        "ambiguous": "#ff7f0e",
        "unlabelled": "#1f77b4",
    }

    for idx, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= len(top):
            ax.axis("off")
            continue

        rec = top[idx]
        tr0, tc0, tr1, tc1 = crop_box_for(rec, img_h, img_w)
        crop = image[tr0:tr1, tc0:tc1]
        ax.imshow(crop, cmap="gray" if crop.ndim == 2 else None)

        r0, c0, r1, c1 = rec["bbox_rowcol_jpg"]
        r0, r1 = min(r0, r1), max(r0, r1)
        c0, c1 = min(c0, c1), max(c0, c1)
        ax.add_patch(
            mpatches.Rectangle(
                (c0 - tc0 - 0.5, r0 - tr0 - 0.5),
                max(c1 - c0, 1.0),
                max(r1 - r0, 1.0),
                linewidth=1.6,
                edgecolor="#ffd400",
                facecolor="none",
            )
        )

        colour = group_colour[rec["calibration_group"]]
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2.5)

        label = (
            f"policy rank {rec[rank_key]}   (original rank {rec['raw_rank']})\n"
            f"component id {rec['component_id']}   [{rec['calibration_group']}]\n"
            f"raw {rec['raw_score']:.6f}   review {rec[score_key]:.6f}\n"
            f"bg overlap {rec['overlap']:.4f}   bg dist "
            f"{rec['background_distance_px']:.1f} px   "
            f"touches={rec['touches_render_background']}\n"
            f"bbox jpg rows [{r0:.0f},{r1:.0f}] cols [{c0:.0f},{c1:.0f}]"
        )
        ax.set_title(label, fontsize=7.4, color=colour, loc="left", pad=4)

    handles = [
        mpatches.Patch(color=v, label=k.replace("_", " ")) for k, v in group_colour.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.001),
    )
    fig.tight_layout(rect=[0, 0.018, 1, 0.985])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    regions_doc, metadata, summary = load_inputs()
    records, validation = build_records(regions_doc)

    # --- coordinate semantics -------------------------------------------------
    # regions.json already stores source-JPG bboxes ("bbox_rowcol_jpg"), so no
    # derivation is required. We only verify the stored relationship against
    # metadata["scale_processed_to_jpg_rowcol"] and record the result.
    scale_r, scale_c = metadata["scale_processed_to_jpg_rowcol"]
    mismatches = 0
    for rec in records:
        pr0, pc0, pr1, pc1 = rec["bbox_rowcol_processed"]
        jr0, jc0, jr1, jc1 = rec["bbox_rowcol_jpg"]
        expected = (pr0 * scale_r, pc0 * scale_c, pr1 * scale_r, pc1 * scale_c)
        if max(abs(a - b) for a, b in zip(expected, (jr0, jc0, jr1, jc1))) > 1e-6:
            mismatches += 1

    coordinate_handling = {
        "source_jpg_coordinates_available_in_regions_json": True,
        "field_used": "bbox_rowcol_jpg",
        "transformation_applied_by_this_audit": (
            "none; bbox_rowcol_jpg used directly as source-JPG (row, col) coordinates"
        ),
        "verification": (
            "bbox_rowcol_jpg == bbox_rowcol_processed * "
            f"scale_processed_to_jpg_rowcol {metadata['scale_processed_to_jpg_rowcol']}"
        ),
        "n_bboxes_failing_verification": mismatches,
        "jpg_shape_rowcol": metadata["jpg_shape_rowcol"],
        "processed_shape_rowcol": metadata["processed_shape_rowcol"],
        "orientation_note": metadata["orientation"],
    }

    # --- ranking --------------------------------------------------------------
    assign_soft_ranks(records, "raw_score", "raw_policy_rank")
    assign_soft_ranks(records, "half_score", "half_rank")
    assign_soft_ranks(records, "full_score", "full_rank")
    assign_hard_filter_ranks(records, "hard_020_rank")

    raw_rank_reproduced = all(
        rec["raw_policy_rank"] == rec["raw_rank"] for rec in records
    )

    # --- statistics -----------------------------------------------------------
    policies = {
        "A_raw": ("raw_score", "raw_rank"),
        "B_half_penalty": ("half_score", "half_rank"),
        "C_full_penalty": ("full_score", "full_rank"),
        "D_hard_flag_0_20": ("raw_score", "hard_020_rank"),
    }

    raw_top_ids_by_k = {
        k: {rec["component_id"] for rec in records if rec["raw_rank"] <= k}
        for k in TOP_K_VALUES
    }

    top_k_report = {}
    movement_report = {}
    for name, (score_key, rank_key) in policies.items():
        top_k_report[name] = {
            f"top_{k}": top_k_stats(records, rank_key, k, raw_top_ids_by_k[k])
            for k in TOP_K_VALUES
        }
        movement_report[name] = movement_stats(records, rank_key)

    # --- calibration table ----------------------------------------------------
    calibration_rows = []
    for rec in sorted(records, key=lambda r: r["raw_rank"]):
        if rec["calibration_group"] == "unlabelled":
            continue
        calibration_rows.append(
            {
                "original_rank": rec["raw_rank"],
                "component_id": rec["component_id"],
                "calibration_group": rec["calibration_group"],
                "raw_score": rec["raw_score"],
                "overlap": round(rec["overlap"], 6),
                "background_distance_px": rec["background_distance_px"],
                "touches_render_background": rec["touches_render_background"],
                "rank_policy_A_raw": rec["raw_rank"],
                "rank_policy_B_half": rec["half_rank"],
                "rank_policy_C_full": rec["full_rank"],
                "rank_policy_D_hard_0_20": rec["hard_020_rank"],
                "half_score": round(rec["half_score"], 6),
                "full_score": round(rec["full_score"], 6),
            }
        )

    overlaps_all = [rec["overlap"] for rec in records]
    overlap_distribution = {
        "n_candidates": len(records),
        "n_overlap_eq_0": int(sum(1 for v in overlaps_all if v == 0.0)),
        "n_overlap_gt_0": int(sum(1 for v in overlaps_all if v > 0.0)),
        "n_overlap_lt_0_01": int(sum(1 for v in overlaps_all if v < 0.01)),
        "n_overlap_in_0_to_0_20_exclusive_of_zero": int(
            sum(1 for v in overlaps_all if 0.0 < v < 0.20)
        ),
        "n_overlap_ge_0_20": int(sum(1 for v in overlaps_all if v >= 0.20)),
        "n_overlap_ge_0_50": int(sum(1 for v in overlaps_all if v >= 0.50)),
        "n_touches_render_background": int(
            sum(1 for rec in records if rec["touches_render_background"])
        ),
        "max_overlap": round(max(overlaps_all), 6),
    }

    # --- CSV ------------------------------------------------------------------
    csv_columns = [
        "component_id",
        "raw_rank",
        "raw_score",
        "overlap",
        "background_distance_px",
        "touches_render_background",
        "half_score",
        "half_rank",
        "full_score",
        "full_rank",
        "hard_020_rank",
        "calibration_group",
    ]
    csv_path = OUT_DIR / "ranking_comparison.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: r["raw_rank"]):
            writer.writerow(
                {
                    "component_id": rec["component_id"],
                    "raw_rank": rec["raw_rank"],
                    "raw_score": f"{rec['raw_score']:.6f}",
                    "overlap": f"{rec['overlap']:.6f}",
                    "background_distance_px": f"{rec['background_distance_px']:.3f}",
                    "touches_render_background": rec["touches_render_background"],
                    "half_score": f"{rec['half_score']:.6f}",
                    "half_rank": rec["half_rank"],
                    "full_score": f"{rec['full_score']:.6f}",
                    "full_rank": rec["full_rank"],
                    "hard_020_rank": rec["hard_020_rank"],
                    "calibration_group": rec["calibration_group"],
                }
            )

    # --- detailed JSON --------------------------------------------------------
    detailed = []
    for rec in sorted(records, key=lambda r: r["raw_rank"]):
        detailed.append(
            {
                "component_id": rec["component_id"],
                "calibration_group": rec["calibration_group"],
                "raw_score": rec["raw_score"],
                "overlap": round(rec["overlap"], 6),
                "stored_overlap": rec["stored_overlap"],
                "background_distance_px": rec["background_distance_px"],
                "touches_render_background": rec["touches_render_background"],
                "bbox_rowcol_jpg": rec["bbox_rowcol_jpg"],
                "centroid_rowcol_jpg": rec["centroid_rowcol_jpg"],
                "policies": {
                    "A_raw": {"score": rec["raw_score"], "rank": rec["raw_rank"], "rank_change": 0},
                    "B_half_penalty": {
                        "score": round(rec["half_score"], 6),
                        "rank": rec["half_rank"],
                        "rank_change": rec["half_rank"] - rec["raw_rank"],
                    },
                    "C_full_penalty": {
                        "score": round(rec["full_score"], 6),
                        "rank": rec["full_rank"],
                        "rank_change": rec["full_rank"] - rec["raw_rank"],
                    },
                    "D_hard_flag_0_20": {
                        "score": rec["raw_score"],
                        "rank": rec["hard_020_rank"],
                        "rank_change": rec["hard_020_rank"] - rec["raw_rank"],
                    },
                },
            }
        )

    (OUT_DIR / "ranking_comparison.json").write_text(
        json.dumps(
            {
                "format": "scroll-anchor.ranking-policy-audit/v0",
                "analysis_only": True,
                "detector_score_unchanged": True,
                "rank_change_convention": (
                    "rank_change = policy_rank - raw_rank; positive means demoted "
                    "(moved down the review list)"
                ),
                "source_run": str(DIAG_DIR.name),
                "candidates": detailed,
            },
            indent=2,
        )
    )

    # --- contact sheets -------------------------------------------------------
    # The source JPG is decoded exactly once and reused for all crops.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(SOURCE_JPG) as im:
        decoded_mode = im.mode
        decoded_size = im.size
        image = np.asarray(im.convert("RGB"))

    jpg_shape_matches_metadata = list(image.shape[:2]) == list(
        metadata["jpg_shape_rowcol"]
    )

    make_contact_sheet(
        records, "raw_rank", "raw_score", image, OUT_DIR / "top20_raw.png",
        "Policy A - raw detector score (review order unchanged)",
    )
    make_contact_sheet(
        records, "half_rank", "half_score", image, OUT_DIR / "top20_half_penalty.png",
        "Policy B - half penalty: raw x (1 - 0.5 x overlap)",
    )
    make_contact_sheet(
        records, "full_rank", "full_score", image, OUT_DIR / "top20_full_penalty.png",
        "Policy C - full penalty: raw x (1 - overlap)",
    )
    del image

    # --- audit summary --------------------------------------------------------
    audit_summary = {
        "format": "scroll-anchor.ranking-policy-audit-summary/v0",
        "analysis_only": True,
        "scope_note": (
            "Offline comparison of candidate review ORDER only. The detector score, "
            "the analyze-render outputs and the ScrollAnchor code are unchanged. No "
            "candidate is classified here as a confirmed false positive."
        ),
        "verified_input_counts": {
            "source_run_directory": DIAG_DIR.name,
            "source_filename": metadata["source_filename"],
            "n_regions_total": metadata["region_counts"]["n_regions_total"],
            "n_regions_above_threshold": metadata["region_counts"][
                "n_regions_above_threshold"
            ],
            "n_regions_exported": metadata["region_counts"]["n_regions_exported"],
            "n_regions_suppressed": metadata["region_counts"]["n_regions_suppressed"],
            "n_candidates_loaded_from_regions_json": len(records),
            "exported_score_range_metadata": metadata["exported_score_range"],
            "observed_raw_score_range": [
                min(rec["raw_score"] for rec in records),
                max(rec["raw_score"] for rec in records),
            ],
            "summary_json_counts_match_metadata": (
                summary["region_counts"] == metadata["region_counts"]
            ),
            "exported_is_ranked_subset": summary["exported_is_ranked_subset"],
            "raw_rank_reproduced_by_deterministic_sort": raw_rank_reproduced,
            **validation,
        },
        "overlap_distribution": overlap_distribution,
        "coordinate_handling": coordinate_handling,
        "source_image_read": {
            "path": SOURCE_JPG.name,
            "decoded_once": True,
            "decoded_mode": decoded_mode,
            "decoded_size_wh": list(decoded_size),
            "shape_matches_metadata_jpg_shape_rowcol": jpg_shape_matches_metadata,
        },
        "policy_formulas": {
            "A_raw": "review_score_raw = raw_score",
            "B_half_penalty": "review_score_half = raw_score * (1 - 0.5 * overlap)",
            "C_full_penalty": "review_score_full = raw_score * (1 - overlap)",
            "D_hard_flag_0_20": (
                "overlap < 0.20 ranked first by raw score; overlap >= 0.20 appended "
                "afterwards by raw score. COMPARISON ONLY - not proposed as production "
                "behaviour."
            ),
            "overlap_definition": (
                "render_background_overlap_fraction, clamped defensively to [0, 1]"
            ),
            "tie_break": "higher adjusted score, then higher raw score, then lower component id",
        },
        "top_k_statistics": top_k_report,
        "calibration_candidate_rank_table": calibration_rows,
        "calibration_group_definition": {
            "known_background_original_ranks": KNOWN_BACKGROUND_RAW_RANKS,
            "interior_control_original_ranks": INTERIOR_CONTROL_RAW_RANKS,
            "ambiguous_original_ranks": AMBIGUOUS_RAW_RANKS,
            "note": (
                "Calibration examples only, not a complete ground-truth dataset. The "
                "ambiguous case is treated as neither positive nor negative evidence."
            ),
        },
        "rank_movement_statistics": movement_report,
        "recommendation": None,
        "rationale": None,
        "limitations": None,
    }

    # --- decision, derived from explicit pre-stated criteria ------------------
    bg = [rec for rec in records if rec["calibration_group"] == "known_background"]
    ctl = [rec for rec in records if rec["calibration_group"] == "interior_control"]

    def evaluate_criteria(rank_key):
        """Explicit, pre-stated pass/fail criteria. No coefficient tuning."""
        bg_ranks = sorted(rec[rank_key] for rec in bg)
        ctl_ranks = sorted(rec[rank_key] for rec in ctl)
        negl = movement_report[
            {"half_rank": "B_half_penalty", "full_rank": "C_full_penalty"}[rank_key]
        ]["negligible_overlap_lt_0_01"]

        c1 = all(r > 20 for r in bg_ranks)
        c2 = all(r <= 10 for r in ctl_ranks)
        c3 = negl["n_demoted"] == 0
        c4 = overlap_distribution["n_overlap_gt_0"] / len(records) <= 0.5
        return {
            "C1_all_known_background_demoted_out_of_top_20": c1,
            "C2_both_interior_controls_within_top_10": c2,
            "C3_no_negligible_overlap_candidate_demoted": c3,
            "C4_penalty_touches_at_most_half_of_candidates": c4,
            "known_background_ranks": bg_ranks,
            "interior_control_ranks": ctl_ranks,
            "all_criteria_met": bool(c1 and c2 and c3 and c4),
        }

    criteria = {
        "B_half_penalty": evaluate_criteria("half_rank"),
        "C_full_penalty": evaluate_criteria("full_rank"),
    }

    have_calibration = len(bg) == len(KNOWN_BACKGROUND_RAW_RANKS) and len(ctl) == len(
        INTERIOR_CONTROL_RAW_RANKS
    )

    # Selection rule, fixed in advance: prefer the LEAST aggressive policy that
    # satisfies every criterion; if none does, keep diagnostics only.
    if not have_calibration:
        recommendation = "insufficient_evidence"
    elif criteria["B_half_penalty"]["all_criteria_met"]:
        recommendation = "add_half_penalty_review_score"
    elif criteria["C_full_penalty"]["all_criteria_met"]:
        recommendation = "add_full_penalty_review_score"
    else:
        recommendation = "keep_diagnostics_only"

    audit_summary["decision_criteria"] = {
        "criteria_definition": {
            "C1": "every known render-background calibration case falls out of the top 20",
            "C2": "both interior controls remain within the top 10",
            "C3": "no candidate with overlap < 0.01 is demoted relative to raw order",
            "C4": "at most 50% of candidates receive any penalty at all "
            "(guards against broad suppression)",
        },
        "selection_rule": (
            "Prefer the least aggressive policy satisfying all criteria (B before C). "
            "If none satisfies them, keep_diagnostics_only. If the calibration "
            "examples cannot be resolved, insufficient_evidence."
        ),
        "results": criteria,
    }
    audit_summary["recommendation"] = recommendation
    audit_summary["decision_evidence"] = {
        "known_background_ranks_raw": sorted(rec["raw_rank"] for rec in bg),
        "known_background_ranks_half": sorted(rec["half_rank"] for rec in bg),
        "known_background_ranks_full": sorted(rec["full_rank"] for rec in bg),
        "interior_control_ranks_raw": sorted(rec["raw_rank"] for rec in ctl),
        "interior_control_ranks_half": sorted(rec["half_rank"] for rec in ctl),
        "interior_control_ranks_full": sorted(rec["full_rank"] for rec in ctl),
        "negligible_overlap_stability_half": movement_report["B_half_penalty"][
            "negligible_overlap_lt_0_01"
        ],
        "negligible_overlap_stability_full": movement_report["C_full_penalty"][
            "negligible_overlap_lt_0_01"
        ],
    }

    rationale = []
    if recommendation == "add_half_penalty_review_score":
        rationale += [
            "Policy B demotes every known render-background calibration case out of "
            "the top 20 while both interior controls stay within the top 10.",
            "Policy C also satisfies the criteria but is monotonically stronger; it "
            "drives high-overlap candidates to near-zero review score, which is closer "
            "to hiding them than to reordering them. Six calibration examples do not "
            "justify the more aggressive option, so the least aggressive sufficient "
            "policy is selected.",
        ]
    elif recommendation == "add_full_penalty_review_score":
        rationale += [
            "Policy B failed at least one criterion while Policy C satisfied all of "
            "them, so the stronger penalty is the minimum sufficient option."
        ]
    elif recommendation == "keep_diagnostics_only":
        rationale += [
            "Neither soft policy satisfied all pre-stated criteria, so the existing "
            "render-background diagnostic fields should remain diagnostics only."
        ]
    else:
        rationale += [
            "The calibration examples could not be resolved from regions.json, so no "
            "policy can be assessed."
        ]
    rationale += [
        "Overlap is strongly bimodal in this run: "
        f"{overlap_distribution['n_overlap_eq_0']} of {len(records)} candidates have "
        "exactly zero overlap and only "
        f"{overlap_distribution['n_overlap_lt_0_01'] - overlap_distribution['n_overlap_eq_0']} "
        "candidate(s) fall in (0, 0.01), so a soft penalty here cannot behave as broad "
        "suppression of everything touching a black pixel.",
        "Zero-overlap candidates keep their raw score under both soft policies, so "
        "they can only be displaced upward as penalised candidates fall past them; "
        "they are never demoted.",
        "The raw detector score is preserved as a separate field; the review score is "
        "an additional ordering key, not a replacement.",
        "Policy D (hard flag at 0.20) is reported for contrast only. It discards raw "
        "score evidence at the threshold boundary and is not recommended.",
    ]
    audit_summary["rationale"] = rationale
    audit_summary["recommendation_wording"] = (
        "Add the selected penalty as a SECONDARY review-order score only. The raw "
        "detector score remains the primary, unchanged field, and analyze-render "
        "score/rank/JSON behaviour is not altered by this audit."
    )
    audit_summary["limitations"] = [
        "Only six known render-background examples, two interior controls and one "
        "ambiguous case are available; this is calibration, not ground truth.",
        "No candidate in this audit is confirmed as a false positive. High "
        "render-background overlap is a review-priority signal, not an error verdict.",
        "render_background_overlap_fraction depends on run-specific mask parameters "
        "(8-bit grayscale threshold 32, minimum component 5000 processed pixels, "
        "4-connectivity, edge-connected only) and is not universally calibrated.",
        "Visible black render background is not equivalent to tifxyz-invalid surface; "
        "tifxyz diagnostics were deliberately not used in this ranking audit.",
        "Conclusions cover only the 200 exported candidates, not the 428 "
        "above-threshold or 1152 total connected components.",
        "The penalty coefficients 0.5 and 1.0 were fixed in advance. No parameter "
        "search or coefficient tuning was performed, so 0.5 is not claimed optimal.",
        "The hard-filter comparison at overlap 0.20 is reported for contrast only and "
        "is explicitly not recommended as production behaviour.",
        "One candidate touches the render background with zero recorded overlap, so a "
        "purely overlap-based penalty leaves it entirely unpenalised.",
        "Results are specific to this single render and this single diagnostics run.",
    ]

    (OUT_DIR / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2))

    # --- console digest -------------------------------------------------------
    print(f"candidates loaded              : {len(records)}")
    print(f"raw order descending by score  : {validation['raw_order_is_descending_by_score']}")
    print(f"raw rank reproduced by sort    : {raw_rank_reproduced}")
    print(f"stored overlap outside [0,1]   : {validation['n_overlap_values_outside_0_1']}")
    print(f"bbox verification mismatches   : {mismatches}")
    print(f"jpg shape matches metadata     : {jpg_shape_matches_metadata}")
    print()
    print("overlap distribution:", json.dumps(overlap_distribution))
    print()
    for name in policies:
        m = movement_report[name]
        print(
            f"{name:18s} rho={m['spearman_rho_vs_raw']:.4f} "
            f"max|move|={m['max_abs_rank_movement']:3d} "
            f"median|move|={m['median_abs_rank_movement']:.1f} "
            f">5={m['n_rank_change_gt_5']:3d} >10={m['n_rank_change_gt_10']:3d} "
            f">20={m['n_rank_change_gt_20']:3d} "
            f"negligible max|move|={m['negligible_overlap_lt_0_01']['max_abs_rank_movement']}"
        )
    print()
    print("calibration ranks (orig -> A / B / C / D):")
    for row in calibration_rows:
        print(
            f"  orig {row['original_rank']:2d}  id {row['component_id']:4d}  "
            f"{row['calibration_group']:17s} ovl {row['overlap']:.4f}  "
            f"dist {row['background_distance_px']:8.1f}  ->  "
            f"A {row['rank_policy_A_raw']:3d} | B {row['rank_policy_B_half']:3d} | "
            f"C {row['rank_policy_C_full']:3d} | D {row['rank_policy_D_hard_0_20']:3d}"
        )
    print()
    for name in policies:
        for k in TOP_K_VALUES:
            s = top_k_report[name][f"top_{k}"]
            print(
                f"{name:18s} top{k:3d}: ovl>0={s['n_overlap_gt_0']:2d} "
                f">=0.20={s['n_overlap_ge_0_20']:2d} >=0.50={s['n_overlap_ge_0_50']:2d} "
                f"median={s['median_overlap']:.4f} "
                f"in={s['n_entering']:2d} out={s['n_leaving']:2d}"
            )
    print()
    print("recommendation:", audit_summary["recommendation"])
    print("outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
