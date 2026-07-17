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
from scipy.ndimage import label as cc_label
from scipy.ndimage import uniform_filter, zoom

# JPG pixel coordinates -> full-render pixel coordinates (exact 8x downsample).
JPG_TO_FULL_FACTOR = 8


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
    max_regions: int = 60
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


def extract_regions(
    diag: RenderDiagnostics,
    scale_row: float,
    scale_col: float,
    params: Optional[RenderParams] = None,
) -> List[Dict[str, object]]:
    """Threshold, cluster and score candidate discontinuity regions"""
    p = params or RenderParams()
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
        regions.append(
            {
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
        )

    # Deterministic ranking: score desc, then id asc for stable tie-breaks.
    regions.sort(key=lambda r: (-float(r["score"]), int(r["id"])))
    return regions[: p.max_regions]


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


def write_outputs(
    outdir: str,
    f: np.ndarray,
    diag: RenderDiagnostics,
    regions: List[Dict[str, object]],
    metadata: Dict[str, object],
    params: RenderParams,
) -> Dict[str, str]:
    """Write overlay.png, regions.json, diagnostics.npz, metadata.json"""
    import json

    os.makedirs(outdir, exist_ok=True)
    paths = {
        "overlay": os.path.join(outdir, "overlay.png"),
        "regions": os.path.join(outdir, "regions.json"),
        "diagnostics": os.path.join(outdir, "diagnostics.npz"),
        "metadata": os.path.join(outdir, "metadata.json"),
    }

    _write_overlay(paths["overlay"], f, regions, params)

    with open(paths["regions"], "w", encoding="utf-8") as fh:
        json.dump(
            {
                "format": "scroll-anchor.render-candidates/v0",
                "note": (
                    "Candidate 2D visual discontinuities only. Not confirmed sheet "
                    "switches, 3D drift, or voxel displacement."
                ),
                "n_regions": len(regions),
                "regions": regions,
            },
            fh,
            indent=2,
        )

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

    return paths


def analyze_render(
    render_path: str, output_dir: str, params: Optional[RenderParams] = None
) -> Dict[str, object]:
    """Full workflow: decode -> detect -> write outputs. Returns a summary dict"""
    p = params or RenderParams()
    t0 = time.perf_counter()

    f, jpg_shape, proc_shape, scale_row, scale_col = load_render(render_path, p)
    diag = analyze_array(f, p)
    regions = extract_regions(diag, scale_row, scale_col, p)

    runtime = time.perf_counter() - t0
    peak_mb = _peak_rss_mb()

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
        "n_candidate_regions": len(regions),
        "limitations": (
            "Render-only 2D analysis. No surface-normal geometry or CT evidence. "
            "Candidates are visual discontinuities for manual review, not confirmed "
            "sheet switches, 3D drift, or voxel displacement. Full-render coordinates "
            "are a documented x{f} mapping, not verified VC3D coordinates.".format(
                f=p.jpg_to_full_factor
            )
        ),
    }

    paths = write_outputs(output_dir, f, diag, regions, metadata, p)
    return {
        "n_regions": len(regions),
        "runtime_seconds": metadata["runtime_seconds"],
        "peak_rss_mb": peak_mb,
        "processed_shape_rowcol": metadata["processed_shape_rowcol"],
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
