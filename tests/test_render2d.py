from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from scroll_anchor.render2d import (
    JPG_TO_FULL_FACTOR,
    RenderParams,
    analyze_array,
    extract_regions,
    jpg_to_full,
    load_render,
    proc_to_jpg,
)


def _texture(h=256, w=256, seed=0):
    """Spatially correlated grayscale texture in [0, 1] (structured, not white noise)"""
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma=3.0)
    a -= a.min()
    a /= a.max() + 1e-6
    return a


def _regions(img, params=None):
    p = params or RenderParams()
    diag = analyze_array(img, p)
    regs = extract_regions(diag, 1.0, 1.0, p)
    return diag, regs


def test_clean_texture_has_no_high_confidence_detections():
    diag, regs = _regions(_texture(seed=1))
    assert regs == []


def test_low_texture_has_no_detections():
    rng = np.random.default_rng(2)
    img = np.full((256, 256), 0.5, np.float32) + rng.normal(0, 0.005, (256, 256)).astype(np.float32)
    diag, regs = _regions(img)
    assert regs == []


def test_horizontal_discontinuity_detected():
    base = _texture(seed=3)
    img = base.copy()
    mid = img.shape[0] // 2
    img[mid:, :] = np.roll(base, 6, axis=1)[mid:, :]  # horizontal seam at mid row
    diag, regs = _regions(img)
    assert len(regs) >= 1
    overlapping = [r for r in regs if r["bbox_rowcol_processed"][0] <= mid <= r["bbox_rowcol_processed"][2]]
    assert overlapping, "no region overlaps the injected horizontal seam"
    assert any(r["direction"] == "horizontal" for r in overlapping)


def test_vertical_discontinuity_detected():
    base = _texture(seed=4)
    img = base.copy()
    mid = img.shape[1] // 2
    img[:, mid:] = np.roll(base, 6, axis=0)[:, mid:]  # vertical seam at mid col
    diag, regs = _regions(img)
    assert len(regs) >= 1
    overlapping = [r for r in regs if r["bbox_rowcol_processed"][1] <= mid <= r["bbox_rowcol_processed"][3]]
    assert overlapping, "no region overlaps the injected vertical seam"
    assert any(r["direction"] == "vertical" for r in overlapping)


def test_shifted_rectangle_boundary_scores_above_clean_interior():
    base = _texture(seed=5)
    shifted = np.roll(np.roll(base, 5, axis=0), 5, axis=1)
    img = base.copy()
    img[80:160, 90:170] = shifted[80:160, 90:170]
    diag, regs = _regions(img)
    assert len(regs) >= 1
    # Boundary anomaly must exceed the (translated but continuous) patch interior.
    boundary = diag.anomaly[78:82, 90:170].max()
    interior = diag.anomaly[110:130, 110:150].max()
    assert boundary > interior


def test_injected_seam_scores_above_clean_image():
    base = _texture(seed=6)
    seam = base.copy()
    mid = seam.shape[0] // 2
    seam[mid:, :] = np.roll(base, 6, axis=1)[mid:, :]
    _, clean_regs = _regions(base)
    _, seam_regs = _regions(seam)
    # The injected discontinuity must be detected; the clean image of the same
    # texture must not produce a competing region of comparable strength.
    assert len(seam_regs) >= 1
    clean_top = max((r["score"] for r in clean_regs), default=0.0)
    assert seam_regs[0]["score"] > clean_top


def test_proc_to_jpg_conversion():
    # working downsample of 2 -> scale 2 in each axis, no swap.
    assert proc_to_jpg(10.0, 20.0, 2.0, 2.0) == (20.0, 40.0)
    assert proc_to_jpg(3.0, 7.0, 4.0, 8.0) == (12.0, 56.0)


def test_jpg_to_full_factor_eight():
    assert JPG_TO_FULL_FACTOR == 8
    assert jpg_to_full(100.0, 200.0) == (800.0, 1600.0)
    assert jpg_to_full(1.0, 2.0, factor=8) == (8.0, 16.0)


def test_load_render_scale_and_no_axis_swap(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    # Non-square image so an axis swap would be detectable: width != height.
    h, w = 200, 320
    img = (_texture(h, w, seed=7) * 255).astype(np.uint8)
    p = tmp_path / "render.png"
    Image.fromarray(img, mode="L").save(str(p))

    arr, jpg_shape, proc_shape, sr, sc = load_render(str(p), RenderParams(working_downsample=2))
    assert jpg_shape == (h, w)  # (rows, cols) preserved, no swap
    assert proc_shape == (h // 2, w // 2)
    assert abs(sr - 2.0) < 1e-6 and abs(sc - 2.0) < 1e-6


def test_deterministic_ranking():
    base = _texture(seed=8)
    img = base.copy()
    mid = img.shape[0] // 2
    img[mid:, :] = np.roll(base, 6, axis=1)[mid:, :]
    p = RenderParams()
    diag = analyze_array(img, p)
    r1 = extract_regions(diag, 1.0, 1.0, p)
    r2 = extract_regions(diag, 1.0, 1.0, p)
    assert [r["id"] for r in r1] == [r["id"] for r in r2]
    assert [r["score"] for r in r1] == [r["score"] for r in r2]


def test_invalid_image_raises(tmp_path):
    pytest.importorskip("PIL.Image")
    bad = tmp_path / "not_an_image.txt"
    bad.write_text("this is not an image")
    with pytest.raises(ValueError):
        load_render(str(bad), RenderParams())
