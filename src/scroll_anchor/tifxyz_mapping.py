"""Map exported 2D render candidates onto a compatible tifxyz raster

This is a *mapping* stage only:

    exported candidate -> source-render location -> tifxyz raster cell
    -> coordinate values read from x.tif, y.tif and z.tif

It does not rerun detection, does not touch candidate scores or ranking, does not
load a CT volume, and does not convert the read coordinates into any CT array
index. Nothing here classifies a candidate as a confirmed sheet skip or surface
error; candidates remain exploratory 2D visual anomalies.

Orientation between the render and the tifxyz raster is not recorded anywhere in
the render metadata, so it is never inferred. Only identity orientation is
supported, and only when the caller passes it explicitly.
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .render2d import load_tifxyz_coords

SCHEMA = "scroll-anchor.candidate-tifxyz-mapping/v0"

IDENTITY_ORIENTATION_ASSUMPTION = (
    "Identity orientation was assumed and was explicitly requested by the caller "
    "(--assume-identity-orientation): the tifxyz raster is treated as sharing the "
    "render's origin, row/column orientation and spatial extent, with no flip, no "
    "transpose, no crop and no offset. The render metadata does not record the "
    "tifxyz/render orientation, so this correspondence is NOT independently "
    "verified. If it is wrong, every mapped cell below is wrong."
)

PIXEL_CONTAINMENT_CONVENTION = (
    "Aligned half-open raster blocks: tifxyz_row = floor(jpg_row / "
    "source_to_tifxyz_row_scale) and tifxyz_col = floor(jpg_col / "
    "source_to_tifxyz_col_scale). Each tifxyz cell covers exactly one "
    "row_scale x col_scale block of source-JPG pixels, half-open on the high side. "
    "The mapped cell is the block CONTAINING the candidate centroid; it is not a "
    "subpixel-accurate surface position, and no interpolation is performed."
)

INVALID_CELL_CONVENTION = (
    "A tifxyz cell is valid only when x, y and z are all finite and none of them "
    "equals the exact -1 invalid sentinel. Zero is a legitimate coordinate value "
    "and stays valid. An invalid selected cell is reported as invalid and is never "
    "replaced by a nearby valid cell: there is no nearest-valid search or fallback."
)

COORDINATE_INTERPRETATION_WARNING = [
    "Values are read from x.tif, y.tif and z.tif, in that order.",
    "They are not yet converted to a CT NumPy array index.",
    "CT volume axis order and bounds remain unverified.",
    "An array may use z, y, x ordering, but this command does not assume that.",
]

LIMITATIONS = [
    "Mapping stage only: no CT volume is opened and no voxel is sampled.",
    "Identity orientation is assumed, not verified; see mapping_assumption.",
    "The mapped cell is a containing raster block, not a subpixel surface position.",
    "Candidates remain exploratory 2D visual anomalies, not confirmed sheet skips "
    "or confirmed surface errors.",
    "Candidate scores and exported ranks are passed through unchanged; this command "
    "never rescores, reorders, filters or suppresses candidates.",
    "Coordinates are read straight from the tifxyz rasters with no unit conversion, "
    "no scaling and no CT axis reordering.",
]


# --------------------------------------------------------------------------- #
# Input loading                                                               #
# --------------------------------------------------------------------------- #
def load_exported_candidates(regions_path: str) -> List[Dict[str, object]]:
    """Read the ranked candidate subset from an ``analyze-render`` regions.json

    The exported order is authoritative and is preserved exactly: exported rank is
    the 1-based position in this list. Rank is never assumed to equal component ID.
    """
    if not os.path.isfile(regions_path):
        raise FileNotFoundError(f"regions.json not found: {regions_path}")
    with open(regions_path) as fh:
        payload = json.load(fh)
    regions = payload.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError(f"{regions_path}: no exported candidate regions found")
    return regions


def load_render_metadata(metadata_path: str) -> Dict[str, object]:
    """Read the ``analyze-render`` metadata.json describing the source render"""
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    with open(metadata_path) as fh:
        metadata = json.load(fh)
    for key in ("jpg_shape_rowcol", "processed_shape_rowcol"):
        if key not in metadata:
            raise ValueError(f"{metadata_path}: missing required field '{key}'")
    return metadata


def _shape_pair(value: object, label: str) -> Tuple[int, int]:
    seq = list(value) if isinstance(value, (list, tuple)) else []
    if len(seq) != 2:
        raise ValueError(f"{label}: expected a [rows, cols] pair, got {value!r}")
    rows, cols = int(seq[0]), int(seq[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{label}: degenerate shape {rows}x{cols}")
    return rows, cols


# --------------------------------------------------------------------------- #
# Raster compatibility                                                        #
# --------------------------------------------------------------------------- #
def derive_integer_scale(
    source_shape: Tuple[int, int],
    tifxyz_shape: Tuple[int, int],
    label: str,
) -> Tuple[int, int]:
    """Derive the exact positive integer ``(row, col)`` scale ``source -> tifxyz``

    Rows and columns are checked independently, but each must divide exactly. There
    is deliberately no tolerance, rounding, crop, offset or per-axis resample: a
    near-miss raster (for example the flattened tifxyz parameterisation, whose grid
    is not an integer multiple of this render) is rejected rather than stretched,
    because stretching would silently misplace every mapped cell.
    """
    sh, sw = source_shape
    th, tw = tifxyz_shape
    if min(sh, sw, th, tw) < 1:
        raise ValueError(f"{label}: degenerate shapes source={sh}x{sw}, tifxyz={th}x{tw}")

    row_exact, col_exact = (sh % th == 0), (sw % tw == 0)
    row_scale, col_scale = sh // th, sw // tw
    if not (row_exact and col_exact and row_scale >= 1 and col_scale >= 1):
        raise ValueError(
            f"{label}: tifxyz raster {th}x{tw} is incompatible with {sh}x{sw}: "
            f"row scale {sh}/{th}={sh / th:.6f} (exact integer: {row_exact}), "
            f"column scale {sw}/{tw}={sw / tw:.6f} (exact integer: {col_exact}); "
            "an exact positive integer scale is required independently in each axis "
            "(no tolerance, crop, offset, flip, transpose or resampling). The tifxyz "
            "raster is never resampled to force a fit."
        )
    return int(row_scale), int(col_scale)


# --------------------------------------------------------------------------- #
# Candidate selection                                                         #
# --------------------------------------------------------------------------- #
def select_candidates(
    regions: Sequence[Dict[str, object]],
    ranks: Optional[Sequence[int]] = None,
    component_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, Dict[str, object]]]:
    """Resolve ``--rank`` / ``--component-id`` selectors to unique candidates

    Returns ``(exported_rank, region)`` pairs ordered by exported rank then
    component ID. Rank is the 1-based exported position and is never assumed to
    equal the component ID. Selecting the same candidate through both selectors
    yields one entry, not a duplicate. Unknown selectors fail loudly.
    """
    ranks = list(ranks or [])
    component_ids = list(component_ids or [])
    if not ranks and not component_ids:
        raise ValueError(
            "no candidate selected: pass at least one --rank or --component-id"
        )

    n = len(regions)
    by_id: Dict[int, int] = {}
    for idx, reg in enumerate(regions):
        by_id.setdefault(int(reg["id"]), idx)  # type: ignore[arg-type]

    chosen: Dict[int, Dict[str, object]] = {}  # exported rank -> region

    bad_ranks = sorted({r for r in ranks if r < 1 or r > n})
    if bad_ranks:
        raise ValueError(
            f"unknown exported rank(s) {bad_ranks}: this run exported {n} candidate(s), "
            f"so valid ranks are 1..{n}. Exported rank is the position in the ranked "
            "subset and is not the component ID."
        )
    for r in ranks:
        chosen[int(r)] = regions[int(r) - 1]

    bad_ids = sorted({int(c) for c in component_ids if int(c) not in by_id})
    if bad_ids:
        raise ValueError(
            f"unknown component ID(s) {bad_ids}: not present among the {n} exported "
            "candidate(s) in regions.json. Component ID is the connected-component "
            "label and is not the exported rank."
        )
    for c in component_ids:
        idx = by_id[int(c)]
        chosen[idx + 1] = regions[idx]  # same rank key -> deduplicated

    return sorted(chosen.items(), key=lambda kv: (kv[0], int(kv[1]["id"])))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Mapping                                                                     #
# --------------------------------------------------------------------------- #
def _centroids(
    region: Dict[str, object],
    proc_to_jpg_row: float,
    proc_to_jpg_col: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], str, str]:
    """Resolve the source-JPG and processed centroids used as mapping anchors

    The source-JPG centroid is the primary anchor whenever regions.json stores one.
    """
    jpg = region.get("centroid_rowcol_jpg")
    proc = region.get("centroid_rowcol_processed")

    if jpg is not None:
        jr, jc = float(jpg[0]), float(jpg[1])  # type: ignore[index]
        anchor = "centroid_rowcol_jpg"
    elif proc is not None:
        pr, pc = float(proc[0]), float(proc[1])  # type: ignore[index]
        jr, jc = pr * proc_to_jpg_row, pc * proc_to_jpg_col
        anchor = "derived from centroid_rowcol_processed"
    else:
        bbox = region.get("bbox_rowcol_jpg")
        if bbox is None:
            raise ValueError(
                f"component {region.get('id')}: no centroid_rowcol_jpg, "
                "centroid_rowcol_processed or bbox_rowcol_jpg to anchor the mapping"
            )
        r0, c0, r1, c1 = (float(v) for v in bbox)  # type: ignore[misc]
        jr, jc = 0.5 * (r0 + r1), 0.5 * (c0 + c1)
        anchor = "derived from bbox_rowcol_jpg centre"

    if proc is not None:
        pr, pc = float(proc[0]), float(proc[1])  # type: ignore[index]
        proc_source = "centroid_rowcol_processed"
    else:
        pr, pc = jr / proc_to_jpg_row, jc / proc_to_jpg_col
        proc_source = (
            "derived from the source-JPG centroid via metadata "
            "scale_processed_to_jpg_rowcol (regions.json stores no processed centroid)"
        )
    return (jr, jc), (pr, pc), anchor, proc_source


def _finite_or_none(value: float) -> Optional[float]:
    """JSON cannot carry NaN/Inf; report non-finite coordinates as null"""
    v = float(value)
    return v if math.isfinite(v) else None


def map_candidate(
    exported_rank: int,
    region: Dict[str, object],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    source_scale: Tuple[int, int],
    processed_scale: Tuple[int, int],
    proc_to_jpg: Tuple[float, float],
) -> Dict[str, object]:
    """Map one exported candidate to its containing tifxyz cell"""
    component_id = int(region["id"])  # type: ignore[arg-type]
    (jr, jc), (pr, pc), anchor, proc_source = _centroids(region, *proc_to_jpg)

    src_sr, src_sc = source_scale
    proc_sr, proc_sc = processed_scale

    # Primary path: source-JPG centroid, aligned half-open blocks.
    row_src, col_src = int(math.floor(jr / src_sr)), int(math.floor(jc / src_sc))
    # Second path: processed centroid and the processed->tifxyz scale.
    row_proc, col_proc = int(math.floor(pr / proc_sr)), int(math.floor(pc / proc_sc))

    if (row_src, col_src) != (row_proc, col_proc):
        raise ValueError(
            f"rank {exported_rank} (component {component_id}): the source-derived and "
            f"processed-derived tifxyz cells disagree - source path gives "
            f"({row_src}, {col_src}) from JPG centroid ({jr:.3f}, {jc:.3f}) at scale "
            f"{src_sr}x{src_sc}; processed path gives ({row_proc}, {col_proc}) from "
            f"processed centroid ({pr:.3f}, {pc:.3f}) at scale {proc_sr}x{proc_sc}. "
            "Refusing to pick one silently; the recorded shapes, scales or centroids "
            "are mutually inconsistent."
        )

    th, tw = int(valid.shape[0]), int(valid.shape[1])
    if not (0 <= row_src < th and 0 <= col_src < tw):
        raise ValueError(
            f"rank {exported_rank} (component {component_id}): mapped tifxyz cell "
            f"({row_src}, {col_src}) is outside the {th}x{tw} tifxyz raster "
            f"(JPG centroid ({jr:.3f}, {jc:.3f}), scale {src_sr}x{src_sc}). "
            "Nothing is clamped; this candidate cannot be mapped."
        )

    cell_valid = bool(valid[row_src, col_src])
    xv, yv, zv = float(x[row_src, col_src]), float(y[row_src, col_src]), float(z[row_src, col_src])

    entry: Dict[str, object] = {
        "exported_rank": exported_rank,
        "component_id": component_id,
        "score": float(region["score"]),  # type: ignore[arg-type]
        "direction": region.get("direction"),
        "centroid_rowcol_jpg": [jr, jc],
        "centroid_rowcol_processed": [pr, pc],
        "centroid_anchor": anchor,
        "processed_centroid_source": proc_source,
        "bbox_rowcol_jpg": list(region.get("bbox_rowcol_jpg") or []),  # type: ignore[arg-type]
        "bbox_rowcol_processed": list(region.get("bbox_rowcol_processed") or []),  # type: ignore[arg-type]
        "source_to_tifxyz_scale_rowcol": [src_sr, src_sc],
        "processed_to_tifxyz_scale_rowcol": [proc_sr, proc_sc],
        "tifxyz_row": row_src,
        "tifxyz_col": col_src,
        "tifxyz_cell_from_source_rowcol": [row_src, col_src],
        "tifxyz_cell_from_processed_rowcol": [row_proc, col_proc],
        "tifxyz_cell_paths_agree": True,
        "tifxyz_cell_valid": cell_valid,
        "x": _finite_or_none(xv),
        "y": _finite_or_none(yv),
        "z": _finite_or_none(zv),
        # Source-JPG footprint of the mapped cell, for the preview and for review.
        "tifxyz_cell_jpg_bbox_rowcol": [
            row_src * src_sr, col_src * src_sc,
            (row_src + 1) * src_sr, (col_src + 1) * src_sc,
        ],
        "mapping_assumption": IDENTITY_ORIENTATION_ASSUMPTION,
        "coordinate_interpretation_warning": COORDINATE_INTERPRETATION_WARNING,
    }
    return entry


# --------------------------------------------------------------------------- #
# Preview                                                                     #
# --------------------------------------------------------------------------- #
def _write_preview(path: str, render_path: str, entries: Sequence[Dict[str, object]]) -> None:
    """One readable source-JPG crop per mapped candidate

    Render-to-tifxyz mapping diagnostic only. This is NOT a CT overlay and carries
    no CT evidence. The source JPG is decoded exactly once and reused for all crops.
    """
    try:
        import matplotlib
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for the mapping preview; install the 'render' "
            'extra: pip install -e ".[render]"'
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .render2d import _require_pillow

    Image = _require_pillow()
    if not os.path.isfile(render_path):
        raise FileNotFoundError(f"render not found: {render_path}")

    prev_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(render_path) as im:  # decoded once, reused for every crop
            base = np.asarray(im.convert("L"))
    finally:
        Image.MAX_IMAGE_PIXELS = prev_limit

    img_h, img_w = base.shape[:2]
    n = len(entries)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.8 * nrows), squeeze=False)
    fig.suptitle(
        "Render-to-tifxyz candidate mapping (identity orientation assumed) - "
        "NOT a CT overlay and not CT validation",
        fontsize=12,
    )

    for idx, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= n:
            ax.axis("off")
            continue
        e = entries[idx]

        br0, bc0, br1, bc1 = (float(v) for v in e["bbox_rowcol_jpg"])  # type: ignore[misc]
        cr0, cc0, cr1, cc1 = (float(v) for v in e["tifxyz_cell_jpg_bbox_rowcol"])  # type: ignore[misc]
        jr, jc = (float(v) for v in e["centroid_rowcol_jpg"])  # type: ignore[misc]

        # Frame everything that must be visible: bbox, centroid and mapped cell.
        r_lo, r_hi = min(br0, br1, cr0, jr), max(br0, br1, cr1, jr)
        c_lo, c_hi = min(bc0, bc1, cc0, jc), max(bc0, bc1, cc1, jc)
        side = max(r_hi - r_lo, c_hi - c_lo) + 160.0
        rc, cc = 0.5 * (r_lo + r_hi), 0.5 * (c_lo + c_hi)
        half = side / 2.0
        tr0 = max(0, min(int(round(rc - half)), img_h - 1))
        tc0 = max(0, min(int(round(cc - half)), img_w - 1))
        tr1 = min(img_h, tr0 + int(round(side)))
        tc1 = min(img_w, tc0 + int(round(side)))

        ax.imshow(base[tr0:tr1, tc0:tc1], cmap="gray", interpolation="nearest")
        ax.add_patch(mpatches.Rectangle(
            (bc0 - tc0 - 0.5, br0 - tr0 - 0.5), max(bc1 - bc0, 1.0), max(br1 - br0, 1.0),
            linewidth=1.6, edgecolor="#ffd400", facecolor="none",
            label="candidate bbox",
        ))
        # Mapped tifxyz cell footprint: exactly the derived integer source scale.
        ax.add_patch(mpatches.Rectangle(
            (cc0 - tc0 - 0.5, cr0 - tr0 - 0.5), cc1 - cc0, cr1 - cr0,
            linewidth=1.8, edgecolor="#00e5ff", facecolor="none",
            label="mapped tifxyz cell",
        ))
        ax.plot(jc - tc0, jr - tr0, "+", color="#ff3b30", markersize=11, markeredgewidth=1.8)

        xs = "n/a" if e["x"] is None else f"{float(e['x']):.3f}"      # type: ignore[arg-type]
        ys = "n/a" if e["y"] is None else f"{float(e['y']):.3f}"      # type: ignore[arg-type]
        zs = "n/a" if e["z"] is None else f"{float(e['z']):.3f}"      # type: ignore[arg-type]
        ax.set_title(
            f"exported rank {e['exported_rank']}   component id {e['component_id']}\n"
            f"raw score {float(e['score']):.6f}\n"                     # type: ignore[arg-type]
            f"tifxyz row {e['tifxyz_row']}  col {e['tifxyz_col']}   "
            f"valid={e['tifxyz_cell_valid']}\n"
            f"x={xs}  y={ys}  z={zs}",
            fontsize=8.4, loc="left", pad=4,
        )

    handles = [
        mpatches.Patch(edgecolor="#ffd400", facecolor="none", label="candidate bbox"),
        mpatches.Patch(edgecolor="#00e5ff", facecolor="none", label="mapped tifxyz cell footprint"),
        # Marker proxy, not a patch: the centroid is drawn as a '+', not a box.
        Line2D([], [], linestyle="none", marker="+", color="#ff3b30",
               markersize=9, markeredgewidth=1.8, label="source-JPG centroid"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Top level                                                                   #
# --------------------------------------------------------------------------- #
CSV_COLUMNS = [
    "exported_rank",
    "component_id",
    "score",
    "direction",
    "centroid_jpg_row",
    "centroid_jpg_col",
    "centroid_processed_row",
    "centroid_processed_col",
    "source_to_tifxyz_row_scale",
    "source_to_tifxyz_col_scale",
    "processed_to_tifxyz_row_scale",
    "processed_to_tifxyz_col_scale",
    "tifxyz_row",
    "tifxyz_col",
    "tifxyz_cell_valid",
    "x",
    "y",
    "z",
]


def map_render_candidates(
    regions_path: str,
    metadata_path: str,
    tifxyz_dir: str,
    output_dir: str,
    ranks: Optional[Sequence[int]] = None,
    component_ids: Optional[Sequence[int]] = None,
    assume_identity_orientation: bool = False,
    render_path: Optional[str] = None,
) -> Dict[str, object]:
    """Map selected exported candidates onto a compatible tifxyz raster

    Writes ``candidate_tifxyz_mapping.json`` and ``candidate_tifxyz_mapping.csv``,
    plus ``candidate_tifxyz_mapping.png`` when ``render_path`` is supplied. The
    supplied regions.json and metadata.json are read-only and are never rewritten.
    """
    if not assume_identity_orientation:
        raise ValueError(
            "refusing to map candidates without an explicit orientation decision: the "
            "render metadata does not record how the tifxyz raster is oriented "
            "relative to the render, so identity orientation is never assumed "
            "silently. Re-run with --assume-identity-orientation to state that the "
            "tifxyz raster shares the render's origin, row/column orientation and "
            "extent (no flip, transpose, crop or offset). Flipped and transposed "
            "orientations are not implemented."
        )

    regions = load_exported_candidates(regions_path)
    metadata = load_render_metadata(metadata_path)

    jpg_shape = _shape_pair(metadata["jpg_shape_rowcol"], "jpg_shape_rowcol")
    proc_shape = _shape_pair(metadata["processed_shape_rowcol"], "processed_shape_rowcol")

    x, y, z, valid = load_tifxyz_coords(tifxyz_dir)
    tif_shape = (int(valid.shape[0]), int(valid.shape[1]))

    source_scale = derive_integer_scale(jpg_shape, tif_shape, "source JPG -> tifxyz")
    processed_scale = derive_integer_scale(proc_shape, tif_shape, "processed raster -> tifxyz")

    # Processed -> JPG scale, used only to reconstruct a processed centroid when
    # regions.json does not store one.
    recorded = metadata.get("scale_processed_to_jpg_rowcol")
    if recorded is not None:
        proc_to_jpg = (float(recorded[0]), float(recorded[1]))  # type: ignore[index]
    else:
        proc_to_jpg = (jpg_shape[0] / proc_shape[0], jpg_shape[1] / proc_shape[1])

    selected = select_candidates(regions, ranks, component_ids)
    entries = [
        map_candidate(rank, reg, x, y, z, valid, source_scale, processed_scale, proc_to_jpg)
        for rank, reg in selected
    ]
    entries.sort(key=lambda e: (int(e["exported_rank"]), int(e["component_id"])))  # type: ignore[arg-type]

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "candidate_tifxyz_mapping.json")
    csv_path = os.path.join(output_dir, "candidate_tifxyz_mapping.csv")

    payload = {
        "format": SCHEMA,
        "note": (
            "Render-to-tifxyz coordinate mapping for selected exported candidates. "
            "No CT volume was opened; no candidate was rescored, reordered or "
            "classified as a confirmed error."
        ),
        "inputs": {
            "regions": os.path.abspath(regions_path),
            "metadata": os.path.abspath(metadata_path),
            "tifxyz_dir": os.path.abspath(tifxyz_dir),
            "render": os.path.abspath(render_path) if render_path else None,
            "source_filename": metadata.get("source_filename"),
            "n_exported_candidates": len(regions),
        },
        "shapes": {
            "jpg_shape_rowcol": list(jpg_shape),
            "processed_shape_rowcol": list(proc_shape),
            "tifxyz_shape_rowcol": list(tif_shape),
        },
        "scales": {
            "source_to_tifxyz_rowcol": list(source_scale),
            "processed_to_tifxyz_rowcol": list(processed_scale),
            "derivation": (
                "Derived from the supplied shapes; never hardcoded. Each axis must "
                "divide exactly into a positive integer, checked independently for "
                "rows and columns. The tifxyz raster is never resampled."
            ),
        },
        "mapping_assumption": IDENTITY_ORIENTATION_ASSUMPTION,
        "identity_orientation_assumed": True,
        "orientation_independently_verified": False,
        "pixel_containment_convention": PIXEL_CONTAINMENT_CONVENTION,
        "invalid_cell_convention": INVALID_CELL_CONVENTION,
        "coordinate_interpretation_warning": COORDINATE_INTERPRETATION_WARNING,
        "selection": {
            "ranks_requested": [int(r) for r in (ranks or [])],
            "component_ids_requested": [int(c) for c in (component_ids or [])],
            "n_selected": len(entries),
            "ordering": "ascending exported rank, then component ID",
        },
        "candidates": entries,
        "limitations": LIMITATIONS,
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "exported_rank": e["exported_rank"],
                "component_id": e["component_id"],
                "score": f"{float(e['score']):.6f}",           # type: ignore[arg-type]
                "direction": e["direction"],
                "centroid_jpg_row": f"{float(e['centroid_rowcol_jpg'][0]):.3f}",        # type: ignore[index]
                "centroid_jpg_col": f"{float(e['centroid_rowcol_jpg'][1]):.3f}",        # type: ignore[index]
                "centroid_processed_row": f"{float(e['centroid_rowcol_processed'][0]):.3f}",  # type: ignore[index]
                "centroid_processed_col": f"{float(e['centroid_rowcol_processed'][1]):.3f}",  # type: ignore[index]
                "source_to_tifxyz_row_scale": source_scale[0],
                "source_to_tifxyz_col_scale": source_scale[1],
                "processed_to_tifxyz_row_scale": processed_scale[0],
                "processed_to_tifxyz_col_scale": processed_scale[1],
                "tifxyz_row": e["tifxyz_row"],
                "tifxyz_col": e["tifxyz_col"],
                "tifxyz_cell_valid": e["tifxyz_cell_valid"],
                "x": "" if e["x"] is None else f"{float(e['x']):.6f}",   # type: ignore[arg-type]
                "y": "" if e["y"] is None else f"{float(e['y']):.6f}",   # type: ignore[arg-type]
                "z": "" if e["z"] is None else f"{float(e['z']):.6f}",   # type: ignore[arg-type]
            })

    written = {"json": json_path, "csv": csv_path, "png": None}
    if render_path:
        png_path = os.path.join(output_dir, "candidate_tifxyz_mapping.png")
        _write_preview(png_path, render_path, entries)
        written["png"] = png_path

    return {
        "output_dir": output_dir,
        "files": written,
        "n_selected": len(entries),
        "tifxyz_shape_rowcol": list(tif_shape),
        "source_to_tifxyz_rowcol": list(source_scale),
        "processed_to_tifxyz_rowcol": list(processed_scale),
        "candidates": entries,
    }
