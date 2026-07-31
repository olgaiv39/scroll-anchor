"""Exploratory 2D surface-render analysis (CPU-only, classical image processing)

This module analyzes a single downsampled 2D grayscale surface render (a JPG) and
flags candidate *visual discontinuities* that MAY correspond to sheet skips or local
render shifts. It is deliberately separate from the 3D ``analyze`` pipeline.

Scientific scope and limits
---------------------------
A flat render carries no surface-normal geometry and no through-thickness CT
evidence, so this detector CANNOT report confirmed sheet switches, true 3D drift,
signed error along a surface normal, voxel displacement, corrected surface
coordinates, or validated natural annotation failures. It reports only 2D
candidates in render-pixel coordinates for manual community review.

Coordinates
-----------
Every region carries coordinates on the downloaded JPG (rows/cols). Full-render
coordinates are a documented multiplication of the JPG coordinates by the known
downsample factor (default 8) and are labelled as *mapped*, not verified VC3D
coordinates.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from scipy.ndimage import uniform_filter, zoom

# JPG pixel coordinates -> full-render pixel coordinates (exact 8x downsample).
JPG_TO_FULL_FACTOR = 8

# tifxyz stores "no surface here" as the -1 sentinel in x/y/z.
TIFXYZ_INVALID_SENTINEL = -1.0


@dataclass
class RenderParams:
    """Interpretable parameters for the exploratory 2D detector"""

    # Memory / working resolution policy.
    working_downsample: int = 2
    max_working_pixels: int = 60_000_000
    # Local contrast window (processed pixels) for the texture reliability weight.
    lc_window: int = 15
    # Seam evidence window: long parallel to the seam, thin across it.
    seam_long: int = 81
    seam_thin: int = 3
    # Small cross-seam probe shift used to compare the two sides.
    seam_shift: int = 1
    # Lateral lag search range: a translation seam aligns at a non-zero lag.
    max_lag: int = 8
    # Absolute local-contrast floor/ceiling for the texture reliability weight.
    tex_lo: float = 0.02
    tex_hi: float = 0.06
    # Detection threshold on the multi-scale anomaly (correlation-recovery gain,
    # 0..~1): a genuine lateral shift recovers correlation at a non-zero lag, while
    # continuous texture and broad illumination changes stay well below this.
    anomaly_thr: float = 0.15
    min_region_pixels: int = 40
    # Export/reporting cap on the number of ranked candidates written out. This is
    # NOT a detector threshold: it only bounds how many top-ranked regions are
    # exported. Raised from 60 to make the export clearly a ranked subset (with the
    # full response counts reported separately) rather than a fixed-size list.
    max_regions: int = 200
    # Fraction of each border to suppress.
    border_frac: float = 0.02
    # Coarse scale used for multi-scale agreement.
    multiscale_factor: int = 2
    # Report a displacement only when the region's per-pixel lag is this consistent.
    lag_consistency: float = 0.6
    # Diagnostics array budget (keeps diagnostics.npz compact).
    max_diag_pixels: int = 2_000_000
    # JPG -> full render factor (documented, not verified).
    jpg_to_full_factor: int = JPG_TO_FULL_FACTOR


@dataclass
class RenderDiagnostics:
    """Processed-resolution diagnostic fields"""

    anomaly: np.ndarray  # multi-scale seam anomaly in [0, ~1]
    texture: np.ndarray  # local contrast (std of the working image)
    seam_h: np.ndarray   # horizontal-seam evidence (fine scale)
    seam_v: np.ndarray   # vertical-seam evidence (fine scale)
    lag_h: np.ndarray    # best horizontal alignment lag (processed px)
    lag_v: np.ndarray    # best vertical alignment lag (processed px)
    agreement: np.ndarray  # per-pixel multi-scale agreement in [0, 1]
    horizontal: np.ndarray  # bool: horizontal seam dominates
    proc_shape: Tuple[int, int]


# --------------------------------------------------------------------------- #
# Coordinate helpers                                                          #
# --------------------------------------------------------------------------- #
def proc_to_jpg(row: float, col: float, scale_row: float, scale_col: float) -> Tuple[float, float]:
    """Map processed-resolution ``(row, col)`` to JPG ``(row, col)``"""
    return row * scale_row, col * scale_col


def jpg_to_full(row: float, col: float, factor: int = JPG_TO_FULL_FACTOR) -> Tuple[float, float]:
    """Map JPG ``(row, col)`` to *mapped* full-render ``(row, col)`` (x factor)"""
    return row * factor, col * factor


# --------------------------------------------------------------------------- #
# Small array utilities                                                       #
# --------------------------------------------------------------------------- #
def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Non-overlapping block-mean downsample by integer factor ``f``"""
    if f <= 1:
        return a
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    a = a[:h2, :w2]
    return a.reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))


def _upsample_to(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear upsample ``a`` to ``shape`` (no rotation/flip)"""
    if a.shape == shape:
        return a
    return zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)


# --------------------------------------------------------------------------- #
# Optional tifxyz-derived valid-surface mask (mesh boundaries)                 #
# --------------------------------------------------------------------------- #
def _read_coord_tif(path: str) -> np.ndarray:
    """Decode one single-channel 2D floating-point coordinate raster

    tifffile is tried first. The published rasters are LZW-compressed, and tifffile
    delegates LZW to the heavy optional ``imagecodecs`` package, so environments
    without it fail with "<COMPRESSION.LZW: 5> requires the 'imagecodecs' package".
    Pillow - already required by this module - decodes LZW natively, so it is used
    as a narrow fallback.

    Values are kept in their native floating-point range and are never rescaled or
    quantised to 8-bit, so the exact ``-1`` sentinel and any non-finite entries
    survive decoding unchanged.
    """
    try:
        import tifffile

        arr = tifffile.imread(path)
    except Exception as exc:  # LZW without imagecodecs, or tifffile unavailable
        Image = _require_pillow()
        try:
            with Image.open(path) as im:
                im.load()
                # mode "F" (or "I;16"/"I") decodes to its native dtype; asarray
                # copies out of the buffer before the handle is closed.
                arr = np.asarray(im)
        except Exception as pil_exc:
            raise ValueError(
                f"cannot decode tifxyz raster {path}: tifffile failed ({exc}); "
                f"Pillow fallback failed ({pil_exc})"
            ) from pil_exc

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a single-channel 2D TIFF, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_tifxyz_coords(
    directory: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a tifxyz ``x/y/z.tif`` triple plus its boolean valid-surface mask

    Returns ``(x, y, z, valid)``, all with the same 2D shape. A cell is valid only
    where all three world coordinates are finite and none of them is the ``-1``
    invalid sentinel. ``mask.tif`` is deliberately NOT consulted: the matching
    render dataset has none, so validity must come from the coordinate rasters
    themselves. ``meta.json`` is not read either.
    """
    coords = []
    for name in ("x.tif", "y.tif", "z.tif"):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"tifxyz directory is missing {name}: {directory}")
        coords.append(_read_coord_tif(path))

    x, y, z = coords
    if not (x.shape == y.shape == z.shape):
        raise ValueError(
            f"tifxyz x/y/z.tif shapes differ in {directory}: "
            f"x={x.shape}, y={y.shape}, z={z.shape}"
        )

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    sentinel = (
        (x == TIFXYZ_INVALID_SENTINEL)
        | (y == TIFXYZ_INVALID_SENTINEL)
        | (z == TIFXYZ_INVALID_SENTINEL)
    )
    return x, y, z, finite & ~sentinel


def load_tifxyz_valid_mask(directory: str) -> np.ndarray:
    """Derive a boolean valid-surface mask from a tifxyz ``x/y/z.tif`` triple"""
    return load_tifxyz_coords(directory)[3]


def project_valid_mask(
    valid: np.ndarray,
    render_shape: Tuple[int, int],
    proc_shape: Tuple[int, int],
) -> np.ndarray:
    """Project a tifxyz valid mask onto the working analysis raster

    Compatibility is exact in this revision: the source render must be an integer
    multiple of the tifxyz raster in BOTH axes, with the SAME factor in each
    (factor 1, i.e. identity, is allowed). A near-miss raster - for example a
    flattened tifxyz differing by a couple of rows and columns - is rejected rather
    than stretched onto the render, because stretching would silently misplace every
    mesh boundary. There is deliberately no tolerance, override, crop, offset, flip,
    transpose, or per-axis rescale.

    Resampling after the gate is centre-aligned nearest-neighbour only. The mask is
    categorical, so interpolation would invent partially-valid surface along every
    boundary.
    """
    th, tw = int(valid.shape[0]), int(valid.shape[1])
    rh, rw = int(render_shape[0]), int(render_shape[1])
    if min(th, tw, rh, rw) < 1:
        raise ValueError(f"degenerate raster shapes: tifxyz={th}x{tw}, render={rh}x{rw}")

    row_exact, col_exact = (rh % th == 0), (rw % tw == 0)
    sr, sc = rh // th, rw // tw
    if not (row_exact and col_exact and sr == sc):
        raise ValueError(
            f"tifxyz raster {th}x{tw} is incompatible with the source render {rh}x{rw}: "
            f"row scale {rh}/{th}={rh / th:.6f} (exact integer: {row_exact}, floor {sr}), "
            f"column scale {rw}/{tw}={rw / tw:.6f} (exact integer: {col_exact}, floor {sc}); "
            "this commit requires one exact shared integer raster scale in both axes "
            "(no tolerance, crop, offset, flip, transpose, or per-axis rescale)"
        )

    ph, pw = int(proc_shape[0]), int(proc_shape[1])
    ri = np.minimum(((np.arange(ph) + 0.5) * (th / ph)).astype(np.intp), th - 1)
    ci = np.minimum(((np.arange(pw) + 0.5) * (tw / pw)).astype(np.intp), tw - 1)
    return np.asarray(valid, dtype=bool)[ri][:, ci]


def boundary_distance(valid: np.ndarray) -> np.ndarray:
    """Euclidean distance (working pixels) from each pixel to the nearest invalid one

    Invalid pixels are outside-surface area and get distance 0. The raster is padded
    with an invalid ring so the render edge counts as a mesh boundary too, rather
    than as unbounded valid surface.
    """
    padded = np.zeros((valid.shape[0] + 2, valid.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = np.asarray(valid, dtype=bool)
    return distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)


# --------------------------------------------------------------------------- #
# Optional render-derived background mask (visible black margins)             #
# --------------------------------------------------------------------------- #
# Explicit 4-neighbour connectivity. Under 8-connectivity a diagonal chain of
# interior specks can leak out to the raster edge and drag real papyrus into the
# "background" set, so diagonal touches deliberately do NOT connect components.
_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def derive_render_background_mask(
    f: np.ndarray,
    grayscale_threshold: int,
    min_component_pixels: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Edge-connected near-black background of the processed render

    ``f`` is the working-resolution grayscale render in [0, 1] that the caller
    already holds, so the image is never reloaded or re-decoded.

    Only near-black components that reach an outer raster edge AND cover at least
    ``min_component_pixels`` are kept, so dark papyrus marks in the interior are
    never treated as background. The rule is deliberately blunt - no Otsu,
    adaptive thresholding, morphology, learned segmentation, or flood-fill
    heuristics - because both parameters are dataset- and resolution-dependent and
    must stay explicit per run rather than become hidden universal assumptions.

    Returns ``(background_mask, stats)``. The mask is boolean with exactly ``f``'s
    shape. This is a descriptive render property: visible black background is NOT
    the same thing as tifxyz invalid surface.
    """
    thr = int(grayscale_threshold)
    minpx = int(min_component_pixels)
    if thr != grayscale_threshold or not (0 <= thr <= 255):
        raise ValueError(
            "grayscale_threshold must be an integer in the inclusive 8-bit range "
            f"0..255, got {grayscale_threshold!r}"
        )
    if minpx != min_component_pixels or minpx < 1:
        raise ValueError(
            "min_component_pixels must be a positive integer count of processed "
            f"(working-resolution) pixels, got {min_component_pixels!r}"
        )

    a = np.asarray(f, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected a 2D processed render, got shape {a.shape}")
    # Quantise back to the 8-bit levels the render was decoded from, so the
    # threshold is a plain grayscale level rather than a float in disguise.
    g8 = np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)

    dark = g8 <= thr
    labels, nlab = cc_label(dark, structure=_FOUR_CONNECTED)
    if nlab == 0:
        bg = np.zeros(a.shape, dtype=bool)
        n_border = n_kept = 0
    else:
        border = np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
        border_labels = np.unique(border)
        border_labels = border_labels[border_labels != 0]
        counts = np.bincount(labels.ravel(), minlength=nlab + 1)
        kept_labels = border_labels[counts[border_labels] >= minpx]
        keep = np.zeros(nlab + 1, dtype=bool)
        keep[kept_labels] = True
        bg = keep[labels]
        n_border, n_kept = int(border_labels.size), int(kept_labels.size)

    stats: Dict[str, object] = {
        "grayscale_threshold_8bit": thr,
        "minimum_component_pixels_processed": minpx,
        "connectivity": 4,
        "near_black_fraction_processed": round(float(dark.mean()), 6),
        "background_fraction_processed": round(float(bg.mean()), 6),
        "n_near_black_components": int(nlab),
        "n_border_connected_components": n_border,
        "n_retained_components": n_kept,
    }
    return bg, stats


# --------------------------------------------------------------------------- #
# Core detector                                                               #
# --------------------------------------------------------------------------- #
def _shift(a: np.ndarray, s: int, axis: int) -> np.ndarray:
    """Shift by ``s`` along ``axis`` with edge replication (no wrap-around)"""
    if s == 0:
        return a
    out = np.roll(a, s, axis=axis)
    if axis == 0:
        if s > 0:
            out[:s, :] = a[:1, :]
        else:
            out[s:, :] = a[-1:, :]
    else:
        if s > 0:
            out[:, :s] = a[:, :1]
        else:
            out[:, s:] = a[:, -1:]
    return out


def _local_contrast(f: np.ndarray, win: int) -> np.ndarray:
    """Local standard deviation of ``f`` over a square window"""
    mu = uniform_filter(f, size=win)
    mu2 = uniform_filter(f * f, size=win)
    return np.sqrt(np.clip(mu2 - mu * mu, 0.0, None))


def _seam_direction(
    f: np.ndarray, axis_seam: int, p: RenderParams, max_lag: int, seam_long: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Directional seam evidence and best alignment lag for one seam orientation

    A translation seam (sheet skip / local render shift) is the signature we target:
    the two sides of the seam no longer line up at zero offset, but *recover*
    correlation at a non-zero lateral lag. Continuous texture already aligns at lag
    zero, so ``best_ncc - zero_ncc`` with a non-zero best lag responds to genuine
    shifts while staying quiet on ordinary texture and broad (DC) illumination
    changes. Evidence is pooled along the seam so only spatially coherent, thin-long
    discontinuities survive; isolated texture spikes wash out.

    ``axis_seam=1`` -> horizontal seam (compare rows, search a horizontal lag);
    ``axis_seam=0`` -> vertical seam (compare columns, search a vertical lag).
    """
    s = int(p.seam_shift)
    if axis_seam == 1:
        a = _shift(f, s, axis=0)
        b = _shift(f, -s, axis=0)
        win = (p.seam_thin, seam_long)
        lag_axis = 1
        pool = (1, seam_long)
    else:
        a = _shift(f, s, axis=1)
        b = _shift(f, -s, axis=1)
        win = (seam_long, p.seam_thin)
        lag_axis = 0
        pool = (seam_long, 1)

    # Side ``a`` is fixed; only its filtered moments are precomputed once.
    ma = uniform_filter(a, size=win)
    va = np.clip(uniform_filter(a * a, size=win) - ma * ma, 0.0, None)

    def _ncc(bk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mb = uniform_filter(bk, size=win)
        vb = np.clip(uniform_filter(bk * bk, size=win) - mb * mb, 0.0, None)
        cov = uniform_filter(a * bk, size=win) - ma * mb
        return cov / (np.sqrt(va * vb) + 1e-6), np.sqrt(np.minimum(va, vb))

    ncc0, min_std = _ncc(b)
    best = np.full(f.shape, -2.0, dtype=np.float32)
    best_lag = np.zeros(f.shape, dtype=np.float32)
    for k in range(-max_lag, max_lag + 1):
        ncc, _ = _ncc(_shift(b, k, lag_axis))
        upd = ncc > best
        best = np.where(upd, ncc, best)
        best_lag = np.where(upd, np.float32(k), best_lag)

    # The texture gate zeroes low-variance windows, where the correlation ratio is
    # ill-conditioned; there a genuine shift cannot be told from noise anyway.
    gate = np.clip((min_std - p.tex_lo) / (p.tex_hi - p.tex_lo), 0.0, 1.0)
    nonzero = (np.abs(best_lag) >= 1).astype(np.float32)
    ev = np.clip(best - ncc0, 0.0, None) * nonzero * gate
    ev = uniform_filter(ev, size=pool)  # spatial coherence along the seam
    return ev.astype(np.float32), best_lag.astype(np.float32)


def analyze_array(f: np.ndarray, params: Optional[RenderParams] = None) -> RenderDiagnostics:
    """Run the multi-scale seam detector on a working-resolution image in [0, 1]"""
    p = params or RenderParams()
    f = np.ascontiguousarray(f, dtype=np.float32)

    sd_f = _local_contrast(f, p.lc_window)

    # Fine scale evidence + alignment lags for both orientations.
    eh, lag_h = _seam_direction(f, 1, p, p.max_lag, p.seam_long)
    ev, lag_v = _seam_direction(f, 0, p, p.max_lag, p.seam_long)
    sal_f = np.maximum(eh, ev)

    # Coarse scale for multi-scale agreement (halved windows / lag search).
    mf = max(1, p.multiscale_factor)
    fc = _block_mean(f, mf)
    long_c = max(9, p.seam_long // mf)
    lag_c = max(2, p.max_lag // mf)
    eh_c, _ = _seam_direction(fc, 1, p, lag_c, long_c)
    ev_c, _ = _seam_direction(fc, 0, p, lag_c, long_c)
    sal_c = _upsample_to(np.maximum(eh_c, ev_c), f.shape)
    del eh_c, ev_c

    denom = np.maximum(sal_f, sal_c) + 1e-6
    agreement = np.minimum(sal_f, sal_c) / denom
    # Geometric mean requires a response at BOTH scales (kills single-scale spikes).
    anomaly = np.sqrt(np.clip(sal_f, 0.0, None) * np.clip(sal_c, 0.0, None))
    del sal_f, sal_c, denom

    # Border suppression (edge-shift artifacts + unreliable margins).
    bh = max(1, int(round(f.shape[0] * p.border_frac)))
    bw = max(1, int(round(f.shape[1] * p.border_frac)))
    anomaly[:bh, :] = 0.0
    anomaly[-bh:, :] = 0.0
    anomaly[:, :bw] = 0.0
    anomaly[:, -bw:] = 0.0

    return RenderDiagnostics(
        anomaly=anomaly.astype(np.float32),
        texture=sd_f.astype(np.float32),
        seam_h=eh.astype(np.float32),
        seam_v=ev.astype(np.float32),
        lag_h=lag_h,
        lag_v=lag_v,
        agreement=agreement.astype(np.float32),
        horizontal=(eh >= ev),
        proc_shape=(int(f.shape[0]), int(f.shape[1])),
    )


def _region_displacement(
    lag: np.ndarray, mask: np.ndarray, p: RenderParams
) -> Optional[float]:
    """Median |lag| over a region, only when the per-pixel lag is consistent"""
    vals = lag[mask]
    if vals.size == 0:
        return None
    med = float(np.median(vals))
    consistency = float(np.mean(np.abs(vals - med) <= 1.0))
    if consistency < p.lag_consistency or abs(med) < 1.0:
        return None
    return abs(med)


def extract_regions_with_stats(
    diag: RenderDiagnostics,
    scale_row: float,
    scale_col: float,
    params: Optional[RenderParams] = None,
    valid_mask: Optional[np.ndarray] = None,
    render_background_mask: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Threshold, cluster and score candidates; return (exported, counts)

    ``counts`` separates the full detector response from the exported subset:

    - ``n_regions_total``           connected components above ``anomaly_thr``
    - ``n_regions_above_threshold`` those also passing ``min_region_pixels``
    - ``n_regions_exported``        top-ranked subset kept after ``max_regions``
    - ``n_regions_suppressed``      above-threshold minus exported (cap overflow)
    - ``max_regions_cap``           the export cap in effect

    The exported list is therefore a RANKED SUBSET, not necessarily every response.

    An optional working-resolution ``valid_mask`` adds purely descriptive
    mesh-boundary fields to each exported region, and an optional
    ``render_background_mask`` adds purely descriptive render-background fields.
    The two are independent and neither affects the score, the threshold, the
    region set, or the ranking.
    """
    p = params or RenderParams()
    bdist = None
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != diag.anomaly.shape:
            raise ValueError(
                f"valid mask shape {valid_mask.shape} does not match the working "
                f"analysis shape {diag.anomaly.shape}"
            )
        bdist = boundary_distance(valid_mask)
    bgdist = None
    if render_background_mask is not None:
        render_background_mask = np.asarray(render_background_mask, dtype=bool)
        if render_background_mask.shape != diag.anomaly.shape:
            raise ValueError(
                f"render background mask shape {render_background_mask.shape} does "
                f"not match the working analysis shape {diag.anomaly.shape}"
            )
        # Same padded convention as the tifxyz distance: background pixels are 0
        # and the raster edge counts as background rather than unbounded interior.
        bgdist = boundary_distance(~render_background_mask)
    binary = diag.anomaly > p.anomaly_thr
    labels, n = cc_label(binary)
    regions: List[Dict[str, object]] = []
    factor = p.jpg_to_full_factor
    for lab in range(1, n + 1):
        m = labels == lab
        size = int(m.sum())
        if size < p.min_region_pixels:
            continue
        rows, cols = np.nonzero(m)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        cr, cc = float(rows.mean()), float(cols.mean())
        mean_anom = float(np.mean(diag.anomaly[m]))
        mean_agree = float(np.mean(diag.agreement[m]))
        mean_texw = float(
            np.mean(np.clip((diag.texture[m] - p.tex_lo) / (p.tex_hi - p.tex_lo), 0.0, 1.0))
        )
        reliability = float(np.clip(mean_agree * mean_texw, 0.0, 1.0))
        horiz = bool(np.mean(diag.horizontal[m]) >= 0.5)
        direction = "horizontal" if horiz else "vertical"

        jr0, jc0 = proc_to_jpg(r0, c0, scale_row, scale_col)
        jr1, jc1 = proc_to_jpg(r1, c1, scale_row, scale_col)
        jcr, jcc = proc_to_jpg(cr, cc, scale_row, scale_col)
        fr, fc = jpg_to_full(jcr, jcc, factor)

        # A horizontal seam is displaced laterally (columns); a vertical seam
        # vertically (rows). Report only when the region lag is consistent.
        lag = diag.lag_h if horiz else diag.lag_v
        disp_proc = _region_displacement(lag, m, p)
        disp_jpg = (
            round(disp_proc * (scale_col if horiz else scale_row), 3)
            if disp_proc is not None
            else None
        )

        score = round(mean_anom * (0.25 + 0.75 * reliability), 6)
        reg: Dict[str, object] = {
            "id": int(lab),
            "score": score,
            "anomaly_mean": round(mean_anom, 6),
            "reliability": round(reliability, 4),
            "direction": direction,
            "size_pixels_processed": size,
            "bbox_rowcol_processed": [r0, c0, r1, c1],
            "bbox_rowcol_jpg": [round(jr0, 2), round(jc0, 2), round(jr1, 2), round(jc1, 2)],
            "centroid_rowcol_jpg": [round(jcr, 2), round(jcc, 2)],
            "mapped_full_render_rowcol": [round(fr, 2), round(fc, 2)],
            "displacement_jpg_pixels": disp_jpg,
            "displacement_note": (
                "insufficient evidence" if disp_jpg is None else "local 2D estimate only"
            ),
        }
        if bdist is not None:
            # Descriptive only: how much of the candidate sits off-surface and how
            # close it comes to a mesh boundary, in processed (working) pixels.
            d_min = float(bdist[m].min())
            reg["boundary_distance_px"] = round(d_min, 3)
            reg["invalid_overlap_fraction"] = round(float(np.mean(~valid_mask[m])), 6)
            # <= 1 px means the region contains, or directly abuts, invalid area.
            reg["touches_invalid_boundary"] = bool(d_min <= 1.0)
        if bgdist is not None:
            # Descriptive only, and measured over the ACTUAL component pixels ``m``
            # rather than the bounding box, which would over-report on sparse or
            # diagonal candidates.
            bg_min = float(bgdist[m].min())
            reg["render_background_distance_px"] = round(bg_min, 3)
            reg["render_background_overlap_fraction"] = round(
                float(np.mean(render_background_mask[m])), 6
            )
            # Same one-pixel descriptive convention as the tifxyz field above.
            reg["touches_render_background"] = bool(bg_min <= 1.0)
        regions.append(reg)

    # Deterministic ranking: score desc, then id asc for stable tie-breaks.
    regions.sort(key=lambda r: (-float(r["score"]), int(r["id"])))
    exported = regions[: p.max_regions]
    counts = {
        "n_regions_total": int(n),
        "n_regions_above_threshold": len(regions),
        "n_regions_exported": len(exported),
        "n_regions_suppressed": len(regions) - len(exported),
        "max_regions_cap": int(p.max_regions),
    }
    return exported, counts


def extract_regions(
    diag: RenderDiagnostics,
    scale_row: float,
    scale_col: float,
    params: Optional[RenderParams] = None,
) -> List[Dict[str, object]]:
    """Threshold, cluster and score candidate discontinuity regions (ranked subset)"""
    regions, _ = extract_regions_with_stats(diag, scale_row, scale_col, params)
    return regions


# --------------------------------------------------------------------------- #
# Decoding with an explicit memory policy                                     #
# --------------------------------------------------------------------------- #
def _require_pillow():
    try:
        import PIL  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Pillow is required for render analysis; install the 'render' extra: "
            "pip install -e \".[render]\""
        ) from exc
    from PIL import Image

    return Image


def load_render(
    path: str, params: Optional[RenderParams] = None
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], float, float]:
    """Decode a JPG to a working-resolution grayscale float image in [0, 1]

    Returns ``(image, jpg_shape_rc, proc_shape_rc, scale_row, scale_col)`` where the
    scales map processed coordinates back to JPG coordinates. Uses Pillow ``draft``
    for decoder-level downsampling (JPEG cannot be truly tiled/random-accessed), so
    a full-resolution decode is avoided.
    """
    p = params or RenderParams()
    Image = _require_pillow()

    if not os.path.isfile(path):
        raise FileNotFoundError(f"render not found: {path}")

    # We enforce our own working-pixel budget below, so lift Pillow's global guard.
    prev_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        try:
            with Image.open(path) as header:
                ow, oh = header.size
        except Exception as exc:
            raise ValueError(f"cannot decode image: {path} ({exc})") from exc

        ds = max(1, int(p.working_downsample))
        tw, th = max(1, ow // ds), max(1, oh // ds)
        if th * tw > p.max_working_pixels:
            raise ValueError(
                f"working resolution {tw}x{th} ({th * tw} px) exceeds "
                f"max_working_pixels={p.max_working_pixels}; increase --working-downsample"
            )

        with Image.open(path) as im:
            im.draft("L", (tw, th))  # decoder-level power-of-two downscale hint
            im = im.convert("L")
            if (im.width, im.height) != (tw, th):
                im = im.resize((tw, th), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
    finally:
        Image.MAX_IMAGE_PIXELS = prev_limit

    ph, pw = arr.shape
    scale_row = oh / ph
    scale_col = ow / pw
    return arr, (oh, ow), (ph, pw), scale_row, scale_col


# --------------------------------------------------------------------------- #
# Outputs                                                                      #
# --------------------------------------------------------------------------- #
def _write_overlay(path: str, f: np.ndarray, regions: List[Dict[str, object]], p: RenderParams) -> None:
    """Draw ranked region boxes over a readable copy of the render (Pillow only)"""
    Image = _require_pillow()
    from PIL import ImageDraw

    # Downsize for a compact, readable overlay (max dimension ~2000 px).
    h, w = f.shape
    md = max(h, w)
    ov_ds = max(1, int(np.ceil(md / 2000)))
    base = (np.clip(f, 0, 1) * 255).astype(np.uint8)[::ov_ds, ::ov_ds]
    img = Image.fromarray(base, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = {"horizontal": (255, 60, 60), "vertical": (60, 160, 255)}
    for rank, reg in enumerate(regions[:30], start=1):
        r0, c0, r1, c1 = reg["bbox_rowcol_processed"]  # type: ignore[misc]
        x0, y0 = c0 / ov_ds, r0 / ov_ds
        x1, y1 = c1 / ov_ds, r1 / ov_ds
        col = colors.get(str(reg["direction"]), (255, 255, 0))
        draw.rectangle([x0, y0, x1, y1], outline=col, width=2)
        draw.text((x0 + 2, max(0, y0 - 10)), str(reg["id"]), fill=col)
    img.save(path)


def _diag_downsample_factor(shape: Tuple[int, int], budget: int) -> int:
    h, w = shape
    f = 1
    while (h // (f + 1)) * (w // (f + 1)) > budget and f < 32:
        f += 1
    return f


def _require_matplotlib():
    try:
        import matplotlib
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for the crop grid and PDF report; install the "
            "'render' extra: pip install -e \".[render]\""
        ) from exc
    matplotlib.use("Agg")
    return matplotlib


def _crop_context(f: np.ndarray, reg: Dict[str, object], half: int = 160):
    """Local context crop around a region centroid, with the bbox in crop coords"""
    r0, c0, r1, c1 = reg["bbox_rowcol_processed"]  # type: ignore[misc]
    cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
    h, w = f.shape
    tr, br = max(0, cr - half), min(h, cr + half)
    lc, rc = max(0, cc - half), min(w, cc + half)
    crop = f[tr:br, lc:rc]
    box = (r0 - tr, c0 - lc, r1 - tr, c1 - lc)  # bbox relative to the crop
    return crop, box


def _write_crop_grid(
    path: str, f: np.ndarray, regions: List[Dict[str, object]], p: RenderParams,
    top: int = 16, ncol: int = 4,
) -> None:
    """Contact sheet of the top-ranked candidates (needs matplotlib)"""
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sel = regions[:top]
    if not sel:
        # Still emit a placeholder so downstream reporting has a figure.
        fig = plt.figure(figsize=(8, 2))
        fig.text(0.5, 0.5, "No candidates passed the export filters.", ha="center")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return

    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.1, nrow * 3.4))
    axes = np.atleast_1d(axes).ravel()
    edge = {"horizontal": "#ff3c3c", "vertical": "#3ca0ff"}
    for ax, reg in zip(axes, sel):
        crop, (br0, bc0, br1, bc1) = _crop_context(f, reg)
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        col = edge.get(str(reg["direction"]), "#ffd000")
        ax.add_patch(Rectangle((bc0, br0), max(1, bc1 - bc0), max(1, br1 - br0),
                               fill=False, edgecolor=col, linewidth=1.6))
        jr, jc = reg["centroid_rowcol_jpg"]  # type: ignore[misc]
        fr, fc = reg["mapped_full_render_rowcol"]  # type: ignore[misc]
        disp = reg["displacement_jpg_pixels"]
        disp_s = "n/a" if disp is None else f"{disp:g}px"
        ax.set_title(
            f"id {reg['id']}  score {reg['score']:.3f}\n"
            f"{reg['direction']}  disp {disp_s}\n"
            f"jpg r{jr:.0f} c{jc:.0f}  full r{fr:.0f} c{fc:.0f}",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(sel):]:
        ax.axis("off")
    fig.suptitle("Top render-anomaly candidates (exploratory, not confirmed sheet skips)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _write_report(
    path: str, overlay_path: str, crops_path: str,
    regions: List[Dict[str, object]], counts: Dict[str, int],
    metadata: Dict[str, object], p: RenderParams,
) -> None:
    """Compact multi-page PDF review packet (needs matplotlib)"""
    _require_matplotlib()
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        # Page 1: title, context, params, limitations, counts.
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.5, 0.95, "ScrollAnchor exploratory render-anomaly review",
                 ha="center", fontsize=15, weight="bold")
        pr = metadata["processed_shape_rowcol"]
        jr = metadata["jpg_shape_rowcol"]
        sr = metadata["exported_score_range"]
        lines = [
            "Context",
            "  Exploratory 2D analysis of a single downsampled surface render (JPG).",
            "  It flags candidate visual discontinuities that MAY correspond to sheet",
            "  skips or local render shifts. It is not the 3D analyze pipeline.",
            "",
            "Source and run parameters",
            f"  file: {metadata['source_filename']}",
            f"  jpg shape (row, col): {jr[0]} x {jr[1]}",
            f"  processed shape (row, col): {pr[0]} x {pr[1]}",
            f"  working_downsample: {p.working_downsample}   "
            f"jpg->full factor: x{p.jpg_to_full_factor}",
            f"  anomaly_thr: {p.anomaly_thr}   min_region_pixels: {p.min_region_pixels}   "
            f"max_regions: {p.max_regions}",
            f"  runtime: {metadata['runtime_seconds']} s   "
            f"peak RSS: {metadata['peak_rss_mb']} MB",
            "",
            "Summary counts (a funnel, not a fixed issue count)",
            f"  n_regions_total (above anomaly threshold): {counts['n_regions_total']}",
            f"  n_regions_above_threshold (+ min size):    {counts['n_regions_above_threshold']}",
            f"  n_regions_exported (ranked subset):        {counts['n_regions_exported']}",
            f"  n_regions_suppressed (over the cap):       {counts['n_regions_suppressed']}",
            f"  max_regions_cap:                           {counts['max_regions_cap']}",
            f"  exported score range: {sr[0]:.3f} .. {sr[1]:.3f}",
            "",
            "Important limitations",
            "  Candidates are exploratory visual anomalies on a flat render. They are",
            "  NOT confirmed sheet skips. 2D displacement is in render (JPG) pixels",
            "  only. Full-render coordinates are a documented x{0} mapping, not verified"
            .format(p.jpg_to_full_factor),
            "  VC3D coordinates. No 3D voxel displacement or surface-normal error is",
            "  claimed. The exported set is a ranked subset; scores cluster just above",
            "  threshold, so many candidates are likely texture or illumination effects.",
        ]
        fig.text(0.08, 0.90, "\n".join(lines), ha="left", va="top",
                 fontsize=9.5, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: full overlay.
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        ax = fig.add_axes([0.02, 0.06, 0.96, 0.88])
        ax.imshow(mpimg.imread(overlay_path))
        ax.set_title("Overlay: ranked candidate boxes over the render "
                     "(red=horizontal, blue=vertical)", fontsize=10)
        ax.axis("off")
        fig.text(0.5, 0.02, "Boxes are the top-ranked exported candidates. "
                 "Not confirmed sheet skips.", ha="center", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: score table for the top rows.
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.96, "Top candidates (score-ranked)", ha="center",
                 fontsize=13, weight="bold")
        rows = regions[:20]
        col_labels = ["id", "score", "dir", "size", "jpg r,c", "full r,c", "disp(px)"]
        table = []
        for r in rows:
            jr2, jc2 = r["centroid_rowcol_jpg"]  # type: ignore[misc]
            fr2, fc2 = r["mapped_full_render_rowcol"]  # type: ignore[misc]
            disp = r["displacement_jpg_pixels"]
            table.append([
                r["id"], f"{r['score']:.3f}", str(r["direction"])[:4],
                r["size_pixels_processed"], f"{jr2:.0f},{jc2:.0f}",
                f"{fr2:.0f},{fc2:.0f}", "n/a" if disp is None else f"{disp:g}",
            ])
        ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
        ax.axis("off")
        if table:
            t = ax.table(cellText=table, colLabels=col_labels, loc="upper center",
                         cellLoc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            t.scale(1, 1.3)
        else:
            ax.text(0.5, 0.9, "No candidates exported.", ha="center")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: crop grid image.
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0.02, 0.03, 0.96, 0.92])
        ax.imshow(mpimg.imread(crops_path))
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def write_outputs(
    outdir: str,
    f: np.ndarray,
    diag: RenderDiagnostics,
    regions: List[Dict[str, object]],
    metadata: Dict[str, object],
    counts: Dict[str, int],
    params: RenderParams,
) -> Dict[str, str]:
    """Write overlay, regions, diagnostics, metadata, summary, crops and PDF report"""
    import json

    os.makedirs(outdir, exist_ok=True)
    paths = {
        "overlay": os.path.join(outdir, "overlay.png"),
        "regions": os.path.join(outdir, "regions.json"),
        "diagnostics": os.path.join(outdir, "diagnostics.npz"),
        "metadata": os.path.join(outdir, "metadata.json"),
        "summary": os.path.join(outdir, "summary.json"),
        "top_candidates": os.path.join(outdir, "top_candidates.png"),
        "report": os.path.join(outdir, "report.pdf"),
    }

    _write_overlay(paths["overlay"], f, regions, params)

    # Additive only when the matching option was used; otherwise the payload keys
    # are exactly the pre-existing schema. The two diagnostics are independent.
    used_tifxyz = bool(metadata.get("tifxyz_valid_mask_used", False))
    used_render_bg = bool(metadata.get("render_background_mask_used", False))

    regions_payload: Dict[str, object] = {
        "format": "scroll-anchor.render-candidates/v0",
        "note": (
            "Ranked SUBSET of candidate 2D visual discontinuities. Exploratory "
            "render anomalies, not confirmed sheet switches, 3D drift, or voxel "
            "displacement. See region_counts for the full response."
        ),
        "region_counts": counts,
    }
    if used_tifxyz:
        regions_payload["tifxyz_valid_mask_used"] = True
    if used_render_bg:
        regions_payload["render_background_mask_used"] = True
    regions_payload["regions"] = regions
    with open(paths["regions"], "w", encoding="utf-8") as fh:
        json.dump(regions_payload, fh, indent=2)

    # summary.json: the count funnel, front and centre, so the export reads as a
    # ranked subset rather than a definitive issue count.
    summary_payload: Dict[str, object] = {
        "source_filename": metadata.get("source_filename"),
        "region_counts": counts,
        "exported_is_ranked_subset": bool(counts["n_regions_suppressed"] > 0),
        "exported_score_range": metadata.get("exported_score_range"),
        "runtime_seconds": metadata.get("runtime_seconds"),
        "peak_rss_mb": metadata.get("peak_rss_mb"),
        "processed_shape_rowcol": metadata.get("processed_shape_rowcol"),
    }
    if used_tifxyz:
        summary_payload["tifxyz_valid_mask_used"] = True
    if used_render_bg:
        summary_payload["render_background_mask_used"] = True
    summary_payload["limitations"] = metadata.get("limitations")
    with open(paths["summary"], "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    # Compact diagnostics: downsample below the pixel budget.
    df = _diag_downsample_factor(diag.proc_shape, params.max_diag_pixels)
    np.savez_compressed(
        paths["diagnostics"],
        anomaly=_block_mean(diag.anomaly, df).astype(np.float32),
        texture=_block_mean(diag.texture, df).astype(np.float32),
        seam_h=_block_mean(diag.seam_h, df).astype(np.float32),
        seam_v=_block_mean(diag.seam_v, df).astype(np.float32),
        diag_downsample=np.int32(df),
        proc_shape=np.array(diag.proc_shape, dtype=np.int32),
    )

    with open(paths["metadata"], "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    # Crop grid + PDF are best-effort: they need matplotlib. If it is missing the
    # core artifacts above are still complete.
    try:
        _write_crop_grid(paths["top_candidates"], f, regions, params)
        _write_report(paths["report"], paths["overlay"], paths["top_candidates"],
                      regions, counts, metadata, params)
    except RuntimeError:
        paths.pop("top_candidates", None)
        paths.pop("report", None)

    return paths


def analyze_render(
    render_path: str,
    output_dir: str,
    params: Optional[RenderParams] = None,
    tifxyz_dir: Optional[str] = None,
    render_background_threshold: Optional[int] = None,
    render_background_min_component_pixels: Optional[int] = None,
) -> Dict[str, object]:
    """Full workflow: decode -> detect -> write outputs. Returns a summary dict

    ``tifxyz_dir`` optionally supplies the matching ``x/y/z.tif`` rasters. When
    given, a valid-surface mask is derived from them and each exported candidate
    gains descriptive mesh-boundary fields.

    ``render_background_threshold`` and ``render_background_min_component_pixels``
    optionally enable the render-derived background diagnostics. BOTH are required
    together and neither has a default: the values are dataset- and
    resolution-dependent, so they must be stated per run.

    Detection, scoring and ranking are unchanged in every combination.
    """
    p = params or RenderParams()
    t0 = time.perf_counter()

    if (render_background_threshold is None) != (
        render_background_min_component_pixels is None
    ):
        raise ValueError(
            "render-background diagnostics require BOTH render_background_threshold "
            "and render_background_min_component_pixels; got threshold="
            f"{render_background_threshold!r} and min_component_pixels="
            f"{render_background_min_component_pixels!r}. Neither has a default "
            "because both are dataset- and resolution-dependent."
        )

    f, jpg_shape, proc_shape, scale_row, scale_col = load_render(render_path, p)

    valid_mask = None
    tifxyz_info: Optional[Dict[str, object]] = None
    if tifxyz_dir is not None:
        raw_valid = load_tifxyz_valid_mask(tifxyz_dir)
        valid_mask = project_valid_mask(raw_valid, jpg_shape, proc_shape)
        tifxyz_info = {
            "raster_shape_rowcol": [int(raw_valid.shape[0]), int(raw_valid.shape[1])],
            "valid_fraction_tifxyz": round(float(raw_valid.mean()), 6),
            "valid_fraction_processed": round(float(valid_mask.mean()), 6),
            "validity_rule": "x, y and z all finite and none equal to -1 (no mask.tif used)",
            "projection": "centre-aligned nearest-neighbour to the working analysis raster",
            "boundary_distance_units": "processed (working-resolution) pixels",
            "effect_on_scoring": "none; diagnostics only in this revision",
            # Stated as ASSUMPTIONS, not findings: only the exact integer raster
            # scale is actually checked. Nothing here is derived from meta.json.
            "alignment_assumptions": {
                "same_raster_origin": True,
                "same_row_column_orientation": True,
                "same_spatial_extent": True,
                "axis_swap": False,
                "horizontal_flip": False,
                "vertical_flip": False,
                "crop_or_offset": False,
                "independently_verified": False,
                "note": (
                    "Assumed, not proven. Only an exact shared integer raster scale is "
                    "verified; the origin, orientation and extent correspondence between "
                    "the tifxyz grid and the render has NOT been independently confirmed "
                    "from dataset metadata."
                ),
            },
        }

    bg_mask = None
    bg_info: Optional[Dict[str, object]] = None
    if render_background_threshold is not None:
        bg_mask, bg_stats = derive_render_background_mask(
            f, render_background_threshold, render_background_min_component_pixels
        )
        bg_info = {
            "source": "edge-connected near-black pixels from the processed render",
            **bg_stats,
            "distance_units": "processed (working-resolution) pixels",
            "effect_on_scoring": "none; diagnostics only",
            "limitations": [
                "The grayscale threshold and minimum component size are explicit "
                "run parameters, not universally calibrated values; they depend on "
                "the dataset, the render and the working resolution.",
                "Visible black render background is NOT equivalent to tifxyz "
                "invalid surface: the two describe different things and can "
                "disagree over large areas.",
            ],
        }

    diag = analyze_array(f, p)
    regions, counts = extract_regions_with_stats(
        diag, scale_row, scale_col, p, valid_mask, bg_mask
    )

    runtime = time.perf_counter() - t0
    peak_mb = _peak_rss_mb()

    scores = [float(r["score"]) for r in regions]
    score_range = [round(min(scores), 6), round(max(scores), 6)] if scores else [0.0, 0.0]

    metadata = {
        "format": "scroll-anchor.render-metadata/v0",
        "source_filename": os.path.basename(render_path),
        "jpg_shape_rowcol": [int(jpg_shape[0]), int(jpg_shape[1])],
        "processed_shape_rowcol": [int(proc_shape[0]), int(proc_shape[1])],
        "working_downsample": int(p.working_downsample),
        "scale_processed_to_jpg_rowcol": [round(scale_row, 6), round(scale_col, 6)],
        "jpg_to_full_render_factor": int(p.jpg_to_full_factor),
        "orientation": "no rotation, flip, or axis swap; rows=Y, cols=X throughout",
        "params": asdict(p),
        "runtime_seconds": round(runtime, 3),
        "peak_rss_mb": peak_mb,
        "region_counts": counts,
        "exported_score_range": score_range,
        "limitations": (
            "Render-only 2D analysis. No surface-normal geometry or CT evidence. "
            "Candidates are exploratory visual anomalies on a render for manual "
            "review, not confirmed sheet switches, 3D drift, or voxel displacement. "
            "2D displacement is in render (JPG) pixels only. Full-render coordinates "
            "are a documented x{f} mapping, not verified VC3D coordinates.".format(
                f=p.jpg_to_full_factor
            )
        ),
    }
    # Only present when the option was used: a run without --tifxyz-dir must write
    # byte-for-byte the previous artifact schema, with no null/false placeholders.
    if tifxyz_info is not None:
        metadata["tifxyz_valid_mask_used"] = True
        metadata["tifxyz_valid_mask"] = tifxyz_info
    if bg_info is not None:
        metadata["render_background_mask_used"] = True
        metadata["render_background_mask"] = bg_info

    paths = write_outputs(output_dir, f, diag, regions, metadata, counts, p)
    return {
        "n_regions_exported": counts["n_regions_exported"],
        "n_regions_above_threshold": counts["n_regions_above_threshold"],
        "n_regions_total": counts["n_regions_total"],
        "exported_score_range": score_range,
        "runtime_seconds": metadata["runtime_seconds"],
        "peak_rss_mb": peak_mb,
        "processed_shape_rowcol": metadata["processed_shape_rowcol"],
        "tifxyz_valid_mask_used": bool(valid_mask is not None),
        "render_background_mask_used": bool(bg_mask is not None),
        "paths": paths,
    }


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource
        import sys

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kB, macOS reports bytes.
        return round(ru / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # pragma: no cover - platform dependent
        return None
