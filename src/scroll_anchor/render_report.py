"""Report-only rebuild of the exploratory 2D render-anomaly review PDF.

This module regenerates ``report.pdf`` (and, when the source render is available,
``top_candidates.png``) from an existing results directory. It is deliberately
read-only with respect to the detector: it reads ``metadata.json``,
``summary.json``, ``regions.json`` and ``overlay.png`` and optionally decodes the
source JPG for larger crop panels. It never runs the detector, never reads or
recomputes ``diagnostics.npz``, and never rewrites the JSON, NPZ or overlay
artifacts. Only the report files are (over)written.
"""
from __future__ import annotations

import json
import os
import textwrap
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_URL = "https://github.com/olgaiv39/scroll-anchor"
PROJECT = "ScrollAnchor"
AUTHOR = "Olga Ivanova"

# Direction colours shared with the overlay (red = horizontal, blue = vertical).
_EDGE = {"horizontal": "#ff3c3c", "vertical": "#3ca0ff"}


# --------------------------------------------------------------------------- #
# Artifact loading (read-only)                                                #
# --------------------------------------------------------------------------- #
def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_results(results_dir: str) -> Tuple[Dict, Dict, List[Dict], Dict]:
    """Read the JSON artifacts. Returns (metadata, summary, regions, counts)."""
    meta_path = os.path.join(results_dir, "metadata.json")
    summ_path = os.path.join(results_dir, "summary.json")
    reg_path = os.path.join(results_dir, "regions.json")
    for pth in (meta_path, summ_path, reg_path):
        if not os.path.isfile(pth):
            raise FileNotFoundError(f"required artifact missing: {pth}")

    metadata = _read_json(meta_path)
    summary = _read_json(summ_path)
    regions_doc = _read_json(reg_path)
    regions = list(regions_doc.get("regions", []))
    # Rank defensively by score (descending); the export is already ranked.
    regions.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    counts = dict(regions_doc.get("region_counts", metadata.get("region_counts", {})))
    return metadata, summary, regions, counts


def _run_date(results_dir: str, metadata: Dict) -> str:
    """Run date from metadata if present, otherwise the metadata.json mtime."""
    for key in ("run_date", "created", "timestamp"):
        val = metadata.get(key)
        if isinstance(val, str) and val:
            return val
    mtime = os.path.getmtime(os.path.join(results_dir, "metadata.json"))
    return time.strftime("%Y-%m-%d", time.localtime(mtime))


# --------------------------------------------------------------------------- #
# Optional source-render crops                                                #
# --------------------------------------------------------------------------- #
def _load_render_image(render_path: str, metadata: Dict) -> Optional[np.ndarray]:
    """Decode the source JPG at the recorded working downsample (no detection).

    Returns a float image in [0, 1] whose shape matches
    ``processed_shape_rowcol`` so that ``bbox_rowcol_processed`` coordinates line
    up. Returns ``None`` if the render is unavailable or cannot be decoded.
    """
    if not render_path or not os.path.isfile(render_path):
        return None
    try:
        from .render2d import RenderParams, load_render

        ds = int(metadata.get("working_downsample", 2))
        f, _jpg, proc_shape, _sr, _sc = load_render(render_path, RenderParams(working_downsample=ds))
    except Exception:
        return None

    expected = metadata.get("processed_shape_rowcol")
    if expected and list(proc_shape) != [int(expected[0]), int(expected[1])]:
        # Coordinates would not line up; better to fall back than mis-place boxes.
        return None
    return f


def _crop_panel(f: np.ndarray, reg: Dict, half: int = 190):
    """Local context crop around a region bbox, with the bbox in crop coords."""
    r0, c0, r1, c1 = reg["bbox_rowcol_processed"]
    cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
    h, w = f.shape
    tr, br = max(0, cr - half), min(h, cr + half)
    lc, rc = max(0, cc - half), min(w, cc + half)
    crop = f[tr:br, lc:rc]
    box = (r0 - tr, c0 - lc, r1 - tr, c1 - lc)
    return crop, box


def _panel_title(reg: Dict, rank: int) -> str:
    jr, jc = reg["centroid_rowcol_jpg"]
    disp = reg.get("displacement_jpg_pixels")
    disp_s = "n/a" if disp is None else f"{float(disp):g} px"
    return (
        f"#{rank}  id {reg['id']}  score {float(reg['score']):.3f}\n"
        f"{reg['direction']}   2D shift {disp_s}\n"
        f"JPG row {jr:.0f}, col {jc:.0f}"
    )


# --------------------------------------------------------------------------- #
# Matplotlib setup / shared chrome                                            #
# --------------------------------------------------------------------------- #
def _require_matplotlib():
    try:
        import matplotlib
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for the PDF report; install the 'render' extra: "
            'pip install -e ".[render]"'
        ) from exc
    matplotlib.use("Agg")
    # Embed TrueType fonts (type 42) so text stays selectable and searchable.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"
    return matplotlib


A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = (11.69, 8.27)
_LEFT = 0.09
_MONO = "monospace"


def _footer(fig, page: int, total: int) -> None:
    fig.text(_LEFT, 0.028, f"{PROJECT}  -  {REPO_URL}", ha="left", va="center",
             fontsize=7.5, color="#555555")
    fig.text(1 - _LEFT, 0.028, f"Page {page} of {total}", ha="right", va="center",
             fontsize=7.5, color="#555555")


def _wrap(text: str, width: int = 96) -> str:
    return "\n".join(textwrap.fill(line, width=width) if line else ""
                     for line in text.split("\n"))


# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #
def _page_summary(pdf, metadata, counts, run_date) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=A4_PORTRAIT)
    y = 0.955
    fig.text(_LEFT, y, PROJECT, fontsize=26, weight="bold")
    fig.text(_LEFT, y - 0.038, "Exploratory 2D Render-Anomaly Review", fontsize=15)
    fig.text(_LEFT, y - 0.068, "PHercParis4 segment w110-112", fontsize=13, color="#333333")

    fig.text(_LEFT, y - 0.100, f"Author: {AUTHOR}", fontsize=10)
    fig.text(_LEFT, y - 0.122, "Repository:", fontsize=10)
    fig.text(_LEFT + 0.14, y - 0.122, REPO_URL, fontsize=9.5, family=_MONO)
    fig.text(_LEFT, y - 0.144, f"Run date: {run_date}", fontsize=10)
    fig.text(_LEFT, y - 0.166, "Source render:", fontsize=10)
    fig.text(_LEFT, y - 0.184, _wrap(str(metadata.get("source_filename", "")), 84),
             fontsize=8, family=_MONO, va="top", color="#333333")

    purpose = (
        "This report summarizes an exploratory, read-only 2D analysis of a single "
        "downsampled surface render (a JPG). The analysis flags candidate visual "
        "discontinuities that may correspond to sheet skips or local render shifts. "
        "It does not use surface-normal geometry or CT voxel evidence, so the "
        "candidates listed here are starting points for manual review, not confirmed "
        "segmentation errors."
    )
    fig.text(_LEFT, y - 0.225, _wrap(purpose, 96), fontsize=10, va="top")

    ax = fig.add_axes([_LEFT, 0.60, 1 - 2 * _LEFT, 0.045])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor="#fff3cd", edgecolor="#d39e00", linewidth=1.2))
    ax.text(0.5, 0.5, "Exploratory review candidates  -  not confirmed sheet skips",
            transform=ax.transAxes, ha="center", va="center", fontsize=12, weight="bold",
            color="#7a5c00")

    jr = metadata["jpg_shape_rowcol"]
    pr = metadata["processed_shape_rowcol"]
    factor = int(metadata.get("jpg_to_full_render_factor", 8))
    full_r, full_c = jr[0] * factor, jr[1] * factor
    exported = counts.get("n_regions_exported")
    rows = [
        ("JPG dimensions (row x col)", f"{jr[0]} x {jr[1]} px"),
        ("Processed dimensions (row x col)", f"{pr[0]} x {pr[1]} px"),
        ("Working downsample", f"{metadata.get('working_downsample')}  (half linear resolution)"),
        (f"Whole-render coverage (mapped x{factor})", f"{full_r} x {full_c} px"),
        ("Runtime", f"{metadata.get('runtime_seconds')} s"),
        ("Peak memory (RSS)", f"{metadata.get('peak_rss_mb')} MB"),
        ("Candidates exported for review", f"{exported}"),
    ]
    fig.text(_LEFT, 0.555, "Run summary", fontsize=12, weight="bold")
    yy = 0.525
    for label, value in rows:
        fig.text(_LEFT, yy, label, fontsize=9.5)
        fig.text(0.56, yy, value, fontsize=9.5, family=_MONO)
        yy -= 0.026

    statement = (
        "The complete downsampled JPG was scanned at half its linear resolution. "
        "This is a whole-render pass, not a partial crop."
    )
    fig.text(_LEFT, 0.33, _wrap(statement, 96), fontsize=10, va="top")

    attribution = (
        "Source render derived from Vesuvius Challenge open data (PHercParis4 segment "
        "20260623163339-w110-112). The source render is not included in this "
        "repository, and this report does not imply endorsement by the Vesuvius "
        "Challenge."
    )
    fig.text(_LEFT, 0.28, _wrap(attribution, 96), fontsize=8.5, va="top", color="#444444")

    _footer(fig, 1, TOTAL_PAGES)
    pdf.savefig(fig)
    plt.close(fig)


def _page_method(pdf, metadata, counts, page_num) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.text(_LEFT, 0.945, "Method and candidate counts", fontsize=15, weight="bold")

    method = (
        "The detector scans the render for short seams where the texture on one side "
        "aligns with the other only after a small lateral shift. It combines a fine "
        "and a coarse scale, weights each location by local texture reliability, and "
        "pools evidence along candidate seams. Connected pixels above a fixed "
        "pixel-level anomaly threshold form raw components; small components are "
        "removed by a minimum-size filter; the remaining components are ranked by a "
        "region score and the top-ranked subset is exported for review."
    )
    fig.text(_LEFT, 0.905, _wrap(method, 96), fontsize=10, va="top")

    fig.text(_LEFT, 0.74, "Candidate count funnel", fontsize=12, weight="bold")
    funnel = [
        ("Raw components above pixel anomaly threshold", counts.get("n_regions_total")),
        ("Components passing minimum size", counts.get("n_regions_above_threshold")),
        ("Ranked candidates exported for review", counts.get("n_regions_exported")),
        ("Additional passing candidates not exported due to cap",
         counts.get("n_regions_suppressed")),
        ("Export cap (max_regions)", counts.get("max_regions_cap")),
    ]
    yy = 0.71
    for label, value in funnel:
        fig.text(_LEFT, yy, label, fontsize=10)
        fig.text(0.72, yy, f"{value}", fontsize=10, family=_MONO)
        yy -= 0.030

    params = metadata.get("params", {})
    thr = params.get("anomaly_thr", 0.15)
    minpx = params.get("min_region_pixels", 40)
    sr = metadata.get("exported_score_range", [None, None])
    fig.text(_LEFT, 0.50, "Pixel threshold versus region score", fontsize=12, weight="bold")
    explain = (
        f"Two different numbers control this pipeline. The pixel anomaly threshold "
        f"(anomaly_thr = {thr}) is applied to the per-pixel anomaly map to decide "
        f"which pixels are candidates, and a minimum size of {minpx} processed pixels "
        f"removes tiny components. Each surviving component then receives a region "
        f"score, computed as its mean pixel anomaly multiplied by a texture-"
        f"reliability weight of at most one. A region score can therefore fall below "
        f"{thr} even though its pixels passed the pixel threshold. The exported score "
        f"range here is {sr[0]:.3f} to {sr[1]:.3f}. The exported regions are a ranked "
        f"subset for review, not a count of confirmed errors, and the raw component "
        f"total is not a count of sheet skips."
    )
    fig.text(_LEFT, 0.47, _wrap(explain, 96), fontsize=10, va="top")

    _footer(fig, page_num, TOTAL_PAGES)
    pdf.savefig(fig)
    plt.close(fig)


def _page_overview(pdf, overlay_path, page_num) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=A4_LANDSCAPE)
    fig.text(_LEFT, 0.95, "Full-render overview", fontsize=15, weight="bold")
    ax = fig.add_axes([0.03, 0.10, 0.94, 0.80])
    if overlay_path and os.path.isfile(overlay_path):
        ax.imshow(mpimg.imread(overlay_path))
    else:
        ax.text(0.5, 0.5, "overlay.png not found", ha="center", va="center")
    ax.axis("off")
    fig.text(_LEFT, 0.065,
             "Numbered boxes mark the top-ranked exported candidates over the whole "
             "downsampled render.", fontsize=9)
    fig.text(_LEFT, 0.045, "Legend:", fontsize=9, weight="bold")
    fig.text(_LEFT + 0.06, 0.045, "red = horizontal seam candidate", fontsize=9,
             color=_EDGE["horizontal"])
    fig.text(_LEFT + 0.32, 0.045, "blue = vertical seam candidate", fontsize=9,
             color=_EDGE["vertical"])
    _footer(fig, page_num, TOTAL_PAGES)
    pdf.savefig(fig)
    plt.close(fig)


def _page_crops(pdf, f, regions, page_num, page_label) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.text(_LEFT, 0.955, f"Enlarged candidate crops ({page_label})", fontsize=14,
             weight="bold")
    nrow, ncol = 3, 2
    for i, reg in enumerate(regions):
        rank = reg["_rank"]
        ax = fig.add_subplot(nrow, ncol, i + 1)
        crop, (br0, bc0, br1, bc1) = _crop_panel(f, reg)
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        col = _EDGE.get(str(reg["direction"]), "#ffd000")
        ax.add_patch(Rectangle((bc0, br0), max(2, bc1 - bc0), max(2, br1 - br0),
                               fill=False, edgecolor=col, linewidth=1.8))
        ax.set_title(_panel_title(reg, rank), fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(_LEFT, 0.045,
             "Crops are local context windows on the processed render. A genuine "
             "candidate shows a texture offset across the marked box.", fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.09, hspace=0.35,
                        wspace=0.12)
    _footer(fig, page_num, TOTAL_PAGES)
    pdf.savefig(fig)
    plt.close(fig)


def _page_crops_fallback(pdf, crops_path, page_num) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.text(_LEFT, 0.955, "Enlarged candidate crops", fontsize=14, weight="bold")
    ax = fig.add_axes([0.05, 0.06, 0.90, 0.86])
    if crops_path and os.path.isfile(crops_path):
        ax.imshow(mpimg.imread(crops_path))
    else:
        ax.text(0.5, 0.5, "top_candidates.png not found and source render "
                "unavailable", ha="center", va="center")
    ax.axis("off")
    _footer(fig, page_num, TOTAL_PAGES)
    pdf.savefig(fig)
    plt.close(fig)


def _page_table(pdf, regions, page_num) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=A4_LANDSCAPE)
    fig.text(_LEFT, 0.94, "Ranked candidates (top 20)", fontsize=15, weight="bold")

    col_labels = ["Rank", "ID", "Score", "Direction", "Size(px)", "JPG row",
                  "JPG col", "Full row", "Full col", "2D shift(px)"]
    body = []
    for rank, r in enumerate(regions[:20], start=1):
        jr, jc = r["centroid_rowcol_jpg"]
        fr, fc = r["mapped_full_render_rowcol"]
        disp = r.get("displacement_jpg_pixels")
        body.append([
            str(rank), str(r["id"]), f"{float(r['score']):.3f}", str(r["direction"]),
            str(r["size_pixels_processed"]), f"{jr:.0f}", f"{jc:.0f}",
            f"{fr:.0f}", f"{fc:.0f}", "n/a" if disp is None else f"{float(disp):g}",
        ])

    ax = fig.add_axes([0.04, 0.12, 0.92, 0.76])
    ax.axis("off")
    if body:
        table = ax.table(cellText=body, colLabels=col_labels, loc="upper center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.35)
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#3ca0ff")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#eef3f8")
    else:
        ax.text(0.5, 0.9, "No candidates exported.", ha="center")

    fig.text(_LEFT, 0.075,
             "Coordinates are JPG-pixel centroids and their mapped full-render "
             "positions. The 2D shift is a local estimate in JPG pixels, not a 3D "
             "voxel offset.", fontsize=8.5)
    _footer(fig, page_num, TOTAL_PAGES)
    pdf.savefig(fig)
    plt.close(fig)


def _page_score_overview(pdf, regions, counts, page_num) -> None:
    import matplotlib.pyplot as plt

    scores = [float(r["score"]) for r in regions]
    n_h = sum(1 for r in regions if str(r["direction"]) == "horizontal")
    n_v = sum(1 for r in regions if str(r["direction"]) == "vertical")
    n_disp = sum(1 for r in regions if r.get("displacement_jpg_pixels") is not None)
    n_nodisp = len(regions) - n_disp

    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.text(_LEFT, 0.955, "Score overview (exported candidates only)", fontsize=14,
             weight="bold")

    ax = fig.add_axes([0.10, 0.56, 0.82, 0.32])
    if scores:
        ax.hist(scores, bins=20, color="#3ca0ff", edgecolor="#1d5f99")
    ax.set_xlabel("Region score (mean anomaly x reliability)", fontsize=9.5)
    ax.set_ylabel("Exported candidates", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.set_title("Distribution of exported region scores", fontsize=10)

    fig.text(_LEFT, 0.46, "Direction and displacement", fontsize=12, weight="bold")
    lines = [
        ("Horizontal seam candidates", n_h),
        ("Vertical seam candidates", n_v),
        ("Candidates with a 2D shift estimate", n_disp),
        ("Candidates without a 2D shift estimate", n_nodisp),
    ]
    yy = 0.43
    for label, value in lines:
        fig.text(_LEFT, yy, label, fontsize=10)
        fig.text(0.62, yy, f"{value}", fontsize=10, family=_MONO)
        yy -= 0.030

    note = (
        "Scores cluster just above the export floor, which is expected for a "
        "conservative exploratory detector. A displacement is only reported when a "
        "region's per-pixel alignment is consistent enough to estimate one."
    )
    fig.text(_LEFT, 0.29, _wrap(note, 96), fontsize=9.5, va="top")
    _footer(fig, page_num, TOTAL_PAGES)
    pdf.savefig(fig)
    plt.close(fig)


def _page_checklist(pdf, metadata, page_num) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.text(_LEFT, 0.955, "Reviewer checklist and limitations", fontsize=15,
             weight="bold")

    fig.text(_LEFT, 0.90, "Suggested review steps", fontsize=12, weight="bold")
    steps = [
        "Open the full-render overview and locate the numbered boxes.",
        "For each candidate, inspect its crop for a genuine texture offset across "
        "the box versus an illumination or JPEG compression artifact.",
        "Cross-check promising candidates against the 3D segmentation in VC3D using "
        "the mapped full-render coordinates in the ranked table.",
        "Treat the ranking as a guide only. Low-scoring candidates near the export "
        "floor are frequently texture effects rather than sheet skips.",
    ]
    yy = 0.865
    for i, step in enumerate(steps, start=1):
        wrapped = _wrap(f"{i}. {step}", 92)
        fig.text(_LEFT, yy, wrapped, fontsize=10, va="top")
        yy -= 0.030 + 0.022 * wrapped.count("\n")

    fig.text(_LEFT, 0.55, "Limitations", fontsize=12, weight="bold")
    limitations = str(metadata.get("limitations", ""))
    fig.text(_LEFT, 0.52, _wrap(limitations, 96), fontsize=10, va="top")

    closing = (
        "This is a render-only 2D screening aid. It carries no surface-normal "
        "geometry and no CT evidence, and it makes no claim about 3D voxel "
        "displacement or corrected surface coordinates."
    )
    fig.text(_LEFT, 0.36, _wrap(closing, 96), fontsize=10, va="top", color="#333333")
    _footer(fig, page_num, TOTAL_PAGES)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Contact sheet (top_candidates.png)                                          #
# --------------------------------------------------------------------------- #
def _write_contact_sheet(path: str, f: np.ndarray, regions: List[Dict],
                         top: int = 12, ncol: int = 4) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sel = regions[:top]
    nrow = int(np.ceil(len(sel) / ncol)) if sel else 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.0, nrow * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, reg in zip(axes, sel):
        crop, (br0, bc0, br1, bc1) = _crop_panel(f, reg)
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        col = _EDGE.get(str(reg["direction"]), "#ffd000")
        ax.add_patch(Rectangle((bc0, br0), max(2, bc1 - bc0), max(2, br1 - br0),
                               fill=False, edgecolor=col, linewidth=1.6))
        ax.set_title(_panel_title(reg, reg["_rank"]), fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(sel):]:
        ax.axis("off")
    fig.suptitle("Top render-anomaly candidates (exploratory, not confirmed sheet skips)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
TOTAL_PAGES = 8  # set per-run in build_report before pages are drawn


def build_report(results_dir: str, render_path: Optional[str] = None) -> Dict[str, object]:
    """Rebuild report.pdf (and top_candidates.png if the render is available).

    Reads metadata.json, summary.json, regions.json and overlay.png from
    ``results_dir``. Overwrites only report.pdf and, when crops are regenerated,
    top_candidates.png. Never runs the detector and never touches the JSON, NPZ
    or overlay artifacts.
    """
    global TOTAL_PAGES
    _require_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages

    metadata, _summary, regions, counts = load_results(results_dir)
    run_date = _run_date(results_dir, metadata)
    overlay_path = os.path.join(results_dir, "overlay.png")
    crops_path = os.path.join(results_dir, "top_candidates.png")
    report_path = os.path.join(results_dir, "report.pdf")

    for rank, reg in enumerate(regions, start=1):
        reg["_rank"] = rank

    f = _load_render_image(render_path, metadata)
    used_render = f is not None
    if used_render:
        _write_contact_sheet(crops_path, f, regions)

    crop_pages = 2 if used_render else 1
    # Six fixed pages (summary, method, overview, table, score overview, checklist)
    # plus one or two crop pages.
    TOTAL_PAGES = 6 + crop_pages

    pdf_metadata = {
        "Title": "ScrollAnchor Exploratory 2D Render-Anomaly Review",
        "Author": AUTHOR,
        "Subject": ("Exploratory 2D render-anomaly review candidates for PHercParis4 "
                    "segment w110-112"),
        "Keywords": ("Vesuvius Challenge, PHercParis4, render anomaly, sheet skip, "
                     "exploratory, 2D"),
        "Creator": PROJECT,
    }

    with PdfPages(report_path, metadata=pdf_metadata) as pdf:
        page = 1
        _page_summary(pdf, metadata, counts, run_date); page += 1
        _page_method(pdf, metadata, counts, page); page += 1
        _page_overview(pdf, overlay_path, page); page += 1
        if used_render:
            first12 = regions[:12]
            _page_crops(pdf, f, first12[:6], page, "1 of 2"); page += 1
            _page_crops(pdf, f, first12[6:12], page, "2 of 2"); page += 1
        else:
            _page_crops_fallback(pdf, crops_path, page); page += 1
        _page_table(pdf, regions, page); page += 1
        _page_score_overview(pdf, regions, counts, page); page += 1
        _page_checklist(pdf, metadata, page); page += 1

    return {
        "report": report_path,
        "top_candidates": crops_path if used_render else None,
        "used_render": used_render,
        "n_pages": TOTAL_PAGES,
        "n_regions": len(regions),
        "counts": counts,
    }
