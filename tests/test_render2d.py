from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from scroll_anchor.render2d import (
    JPG_TO_FULL_FACTOR,
    RenderDiagnostics,
    RenderParams,
    analyze_array,
    analyze_render,
    boundary_distance,
    derive_render_background_mask,
    extract_regions,
    extract_regions_with_stats,
    jpg_to_full,
    load_render,
    load_tifxyz_valid_mask,
    proc_to_jpg,
    project_valid_mask,
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


# --------------------------------------------------------------------------- #
# Optional tifxyz-derived valid-surface mask + mesh-boundary diagnostics       #
# --------------------------------------------------------------------------- #
NEW_FIELDS = ("boundary_distance_px", "invalid_overlap_fraction", "touches_invalid_boundary")


def _write_tifxyz(directory, x, y, z):
    """Write a bare x/y/z.tif triple (no mask.tif, matching the real render dataset)"""
    tifffile = pytest.importorskip("tifffile")
    directory.mkdir(parents=True, exist_ok=True)
    for name, arr in (("x.tif", x), ("y.tif", y), ("z.tif", z)):
        tifffile.imwrite(str(directory / name), np.asarray(arr, dtype=np.float32))
    return str(directory)


def _blob_diag(shape=(64, 64), box=(20, 20, 32, 32)):
    """Minimal diagnostics with one square response well above the export threshold"""
    zeros = np.zeros(shape, np.float32)
    anomaly = zeros.copy()
    r0, c0, r1, c1 = box
    anomaly[r0:r1, c0:c1] = 0.5
    return RenderDiagnostics(
        anomaly=anomaly,
        texture=np.full(shape, 0.06, np.float32),
        seam_h=zeros.copy(),
        seam_v=zeros.copy(),
        lag_h=zeros.copy(),
        lag_v=zeros.copy(),
        agreement=np.ones(shape, np.float32),
        horizontal=np.ones(shape, bool),
        proc_shape=shape,
    )


def test_valid_mask_derived_from_xyz_sentinel_and_nonfinite(tmp_path):
    x = np.ones((4, 5), np.float32)
    y = np.ones((4, 5), np.float32) * 2.0
    z = np.ones((4, 5), np.float32) * 3.0
    x[0, 0] = -1.0            # sentinel in x alone invalidates the pixel
    y[1, 1] = -1.0
    z[2, 2] = -1.0
    z[3, 3] = np.nan          # non-finite invalidates too
    x[3, 4] = np.inf

    d = _write_tifxyz(tmp_path / "tifxyz", x, y, z)
    valid = load_tifxyz_valid_mask(d)

    assert valid.dtype == np.bool_ and valid.shape == (4, 5)
    expected = np.ones((4, 5), bool)
    for rc in [(0, 0), (1, 1), (2, 2), (3, 3), (3, 4)]:
        expected[rc] = False
    assert np.array_equal(valid, expected)
    # -1 is the only sentinel: other negative coordinates stay valid.
    assert valid[0, 1] and valid[2, 3]


def test_valid_mask_negative_one_is_exact_not_thresholded(tmp_path):
    x = np.full((2, 2), -0.5, np.float32)
    y = np.full((2, 2), -2.0, np.float32)
    z = np.full((2, 2), -1.5, np.float32)
    valid = load_tifxyz_valid_mask(_write_tifxyz(tmp_path / "t", x, y, z))
    assert valid.all()


def _write_lzw_tif(path, arr):
    """Write a single-channel float32 TIFF with LZW compression (as the real data is)"""
    Image = pytest.importorskip("PIL.Image")
    im = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
    im.save(str(path), format="TIFF", compression="tiff_lzw")


def _lzw_tifxyz(directory):
    """LZW x/y/z triple mixing valid values, exact -1 sentinels and non-finite entries"""
    directory.mkdir(parents=True, exist_ok=True)
    x = np.array([[1.5, -1.0, 3.25], [np.nan, 2.0, -2.5]], np.float32)
    y = np.array([[10.5, 11.5, 12.5], [13.5, 14.5, 15.5]], np.float32)
    z = np.array([[20.5, 21.5, 22.5], [23.5, 24.5, -1.0]], np.float32)
    for name, a in (("x.tif", x), ("y.tif", y), ("z.tif", z)):
        _write_lzw_tif(directory / name, a)
    # (0,1) sentinel in x; (1,0) NaN in x; (1,2) sentinel in z. -2.5 stays valid.
    return np.array([[True, False, True], [False, True, False]])


def test_load_tifxyz_valid_mask_reads_lzw_float_tiffs(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    d = tmp_path / "lzw"
    expected = _lzw_tifxyz(d)

    with Image.open(str(d / "x.tif")) as im:
        assert im.info.get("compression") == "tiff_lzw"  # the case under repair
        assert im.mode == "F"  # float coordinates, not 8-bit image data

    valid = load_tifxyz_valid_mask(str(d))
    assert valid.dtype == np.bool_
    assert valid.shape == (2, 3)
    assert np.array_equal(valid, expected)


def test_lzw_tifxyz_decodes_via_pillow_when_tifffile_cannot(tmp_path, monkeypatch):
    """The Pillow fallback must cover environments without imagecodecs"""
    tifffile = pytest.importorskip("tifffile")
    d = tmp_path / "lzw_fallback"
    expected = _lzw_tifxyz(d)

    def _no_codec(*_a, **_kw):
        raise ValueError("<COMPRESSION.LZW: 5> requires the 'imagecodecs' package")

    monkeypatch.setattr(tifffile, "imread", _no_codec)

    valid = load_tifxyz_valid_mask(str(d))
    assert valid.dtype == np.bool_
    assert np.array_equal(valid, expected)


def test_pillow_fallback_preserves_float_values_and_sentinel(tmp_path, monkeypatch):
    """Coordinates must survive the fallback unquantised, so -1 stays exactly -1"""
    from scroll_anchor.render2d import _read_coord_tif

    tifffile = pytest.importorskip("tifffile")
    d = tmp_path / "vals"
    d.mkdir()
    arr = np.array([[-1.0, 0.5, 1234.5], [-1.25, np.inf, np.nan]], np.float32)
    _write_lzw_tif(d / "x.tif", arr)

    monkeypatch.setattr(tifffile, "imread", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("<COMPRESSION.LZW: 5> requires the 'imagecodecs' package")))

    out = _read_coord_tif(str(d / "x.tif"))
    assert out.dtype == np.float32
    assert out[0, 0] == -1.0            # exact sentinel, not rounded or rescaled
    assert out[0, 1] == pytest.approx(0.5)
    assert out[0, 2] == pytest.approx(1234.5)  # far outside any 8-bit range
    assert out[1, 0] == -1.25           # near-sentinel value stays distinct from -1
    assert np.isinf(out[1, 1]) and np.isnan(out[1, 2])


def test_read_coord_tif_reports_both_decoders_on_failure(tmp_path, monkeypatch):
    from scroll_anchor.render2d import _read_coord_tif

    tifffile = pytest.importorskip("tifffile")
    bad = tmp_path / "x.tif"
    bad.write_text("not a tiff")
    monkeypatch.setattr(tifffile, "imread", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("tifffile boom")))
    with pytest.raises(ValueError, match="cannot decode tifxyz raster"):
        _read_coord_tif(str(bad))


def test_tifxyz_channel_must_be_2d(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    d = tmp_path / "rgb"
    d.mkdir()
    tifffile.imwrite(str(d / "x.tif"), np.zeros((4, 4, 3), np.float32), photometric="rgb")
    tifffile.imwrite(str(d / "y.tif"), np.zeros((4, 4), np.float32))
    tifffile.imwrite(str(d / "z.tif"), np.zeros((4, 4), np.float32))
    with pytest.raises(ValueError, match="expected a single-channel 2D TIFF"):
        load_tifxyz_valid_mask(str(d))


def test_tifxyz_shape_mismatch_raises(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    d = tmp_path / "bad"
    d.mkdir()
    tifffile.imwrite(str(d / "x.tif"), np.zeros((4, 4), np.float32))
    tifffile.imwrite(str(d / "y.tif"), np.zeros((4, 4), np.float32))
    tifffile.imwrite(str(d / "z.tif"), np.zeros((4, 5), np.float32))
    with pytest.raises(ValueError, match="shapes differ"):
        load_tifxyz_valid_mask(str(d))


def test_tifxyz_missing_channel_raises(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    d = tmp_path / "partial"
    d.mkdir()
    tifffile.imwrite(str(d / "x.tif"), np.zeros((4, 4), np.float32))
    tifffile.imwrite(str(d / "y.tif"), np.zeros((4, 4), np.float32))
    with pytest.raises(FileNotFoundError, match="z.tif"):
        load_tifxyz_valid_mask(str(d))


def test_project_valid_mask_nearest_neighbour_upsample():
    valid = np.array([[True, False], [False, True]])
    # tifxyz 2x2 -> render 8x8 -> working raster 4x4 (shared 4x integer scaling).
    out = project_valid_mask(valid, (8, 8), (4, 4))
    expected = np.repeat(np.repeat(valid, 2, axis=0), 2, axis=1)
    # Nearest-neighbour only: block replication, no blended/partial validity.
    assert out.dtype == np.bool_
    assert np.array_equal(out, expected)


def test_project_valid_mask_downsample_and_non_square():
    valid = np.zeros((100, 160), bool)
    valid[50:, :] = True
    out = project_valid_mask(valid, (200, 320), (50, 80))
    assert out.shape == (50, 80)
    assert not out[:25, :].any() and out[25:, :].all()


# Real dataset rasters. Only the normalized grid divides the render exactly (x10).
NORMALIZED_SHAPE = (627, 2403)
FLATTENED_SHAPE = (629, 2405)
RENDER_SHAPE = (6270, 24030)
SMALL_PROC_SHAPE = (63, 241)


def test_project_valid_mask_accepts_real_normalized_raster():
    assert RENDER_SHAPE[0] // NORMALIZED_SHAPE[0] == 10
    assert RENDER_SHAPE[1] // NORMALIZED_SHAPE[1] == 10
    valid = np.ones(NORMALIZED_SHAPE, bool)
    valid[:10, :] = False
    out = project_valid_mask(valid, RENDER_SHAPE, SMALL_PROC_SHAPE)
    assert out.shape == SMALL_PROC_SHAPE and out.dtype == np.bool_
    assert out.any() and not out.all()


def test_project_valid_mask_rejects_real_flattened_raster():
    # 6270 / 629 and 24030 / 2405 are both non-integer: close, but not the same grid.
    valid = np.ones(FLATTENED_SHAPE, bool)
    with pytest.raises(ValueError, match="exact shared integer raster scale"):
        project_valid_mask(valid, RENDER_SHAPE, SMALL_PROC_SHAPE)


def test_project_valid_mask_accepts_identity_scale():
    valid = np.zeros((8, 12), bool)
    valid[4:, :] = True
    assert np.array_equal(project_valid_mask(valid, (8, 12), (8, 12)), valid)


def test_project_valid_mask_rejects_unequal_integer_factors():
    # Rows scale x2, columns x4: both exact integers, but not a shared factor.
    valid = np.ones((100, 160), bool)
    with pytest.raises(ValueError, match="exact shared integer raster scale"):
        project_valid_mask(valid, (200, 640), (50, 80))


def test_project_valid_mask_rejects_non_integer_scale():
    valid = np.ones((100, 160), bool)
    with pytest.raises(ValueError, match="exact shared integer raster scale"):
        project_valid_mask(valid, (250, 400), (50, 80))


def test_project_valid_mask_rejects_off_by_one_raster():
    # Deliberately NOT tolerated: an off-by-one grid is a different grid.
    valid = np.ones((100, 160), bool)
    with pytest.raises(ValueError, match="exact shared integer raster scale"):
        project_valid_mask(valid, (201, 320), (50, 80))


def test_project_valid_mask_rejects_aspect_mismatch():
    valid = np.ones((100, 100), bool)
    with pytest.raises(ValueError, match="incompatible with the source render"):
        project_valid_mask(valid, (200, 320), (50, 80))


def test_project_valid_mask_error_names_both_shapes_and_scales():
    valid = np.ones(FLATTENED_SHAPE, bool)
    with pytest.raises(ValueError) as excinfo:
        project_valid_mask(valid, RENDER_SHAPE, SMALL_PROC_SHAPE)
    msg = str(excinfo.value)
    assert "629x2405" in msg and "6270x24030" in msg
    assert "row scale" in msg and "column scale" in msg


def test_boundary_distance_zero_on_invalid_and_bounded_by_border():
    valid = np.ones((7, 7), bool)
    valid[3, 3] = False
    d = boundary_distance(valid)
    assert d[3, 3] == 0.0
    assert d[3, 4] == pytest.approx(1.0)
    assert d[3, 5] == pytest.approx(2.0)
    # The raster edge is padded as invalid, so border pixels are 1 px from "outside".
    assert d[0, 0] == pytest.approx(1.0)


def test_boundary_diagnostics_on_synthetic_region():
    diag = _blob_diag()  # single 12x12 response at rows/cols 20..31
    valid = np.ones((64, 64), bool)
    valid[:, 30:] = False  # invalid half overlaps the right 2 columns of the blob

    regs, _ = extract_regions_with_stats(diag, 1.0, 1.0, RenderParams(), valid)
    assert len(regs) == 1
    r = regs[0]
    assert r["touches_invalid_boundary"] is True
    assert r["boundary_distance_px"] == 0.0
    # 2 of 12 columns of the blob fall in the invalid area.
    assert r["invalid_overlap_fraction"] == pytest.approx(2 / 12, abs=1e-6)


def test_boundary_diagnostics_for_region_far_from_boundary():
    diag = _blob_diag()
    valid = np.ones((64, 64), bool)
    valid[:, 40:] = False  # invalid area clear of the blob (cols 20..31)

    regs, _ = extract_regions_with_stats(diag, 1.0, 1.0, RenderParams(), valid)
    r = regs[0]
    assert r["invalid_overlap_fraction"] == 0.0
    assert r["touches_invalid_boundary"] is False
    # Nearest invalid column is 40; the blob's right edge is column 31. This is
    # closer than the padded raster edge (21 px from the blob), so it dominates.
    assert r["boundary_distance_px"] == pytest.approx(9.0)


def test_valid_mask_does_not_change_scores_or_ranking():
    base = _texture(seed=9)
    img = base.copy()
    mid = img.shape[0] // 2
    img[mid:, :] = np.roll(base, 6, axis=1)[mid:, :]
    p = RenderParams()
    diag = analyze_array(img, p)

    valid = np.ones(diag.anomaly.shape, bool)
    valid[:, 200:] = False

    plain, c_plain = extract_regions_with_stats(diag, 1.0, 1.0, p)
    masked, c_masked = extract_regions_with_stats(diag, 1.0, 1.0, p, valid)

    assert c_plain == c_masked
    assert [r["id"] for r in plain] == [r["id"] for r in masked]
    assert [r["score"] for r in plain] == [r["score"] for r in masked]
    # The mask only ADDS fields; nothing pre-existing is rewritten.
    for a, b in zip(plain, masked):
        assert all(b[k] == v for k, v in a.items())


def test_no_diagnostics_fields_without_valid_mask():
    diag = _blob_diag()
    regs, _ = extract_regions_with_stats(diag, 1.0, 1.0, RenderParams())
    assert regs and all(k not in regs[0] for k in NEW_FIELDS)


def test_valid_mask_shape_mismatch_with_working_raster_raises():
    diag = _blob_diag()
    with pytest.raises(ValueError, match="does not match the working"):
        extract_regions_with_stats(diag, 1.0, 1.0, RenderParams(), np.ones((32, 32), bool))


def _seam_render(tmp_path, h=200, w=320, seed=11):
    Image = pytest.importorskip("PIL.Image")
    base = _texture(h, w, seed=seed)
    img = base.copy()
    img[h // 2:, :] = np.roll(base, 6, axis=1)[h // 2:, :]
    path = tmp_path / "render.png"  # lossless: keeps the injected seam intact
    Image.fromarray((img * 255).astype(np.uint8), mode="L").save(str(path))
    return str(path)


# Exact pre-tifxyz on-disk schemas. A run without --tifxyz-dir must still match.
PREV_REGIONS_KEYS = {"format", "note", "region_counts", "regions"}
PREV_SUMMARY_KEYS = {
    "source_filename", "region_counts", "exported_is_ranked_subset",
    "exported_score_range", "runtime_seconds", "peak_rss_mb",
    "processed_shape_rowcol", "limitations",
}
PREV_REGION_FIELDS = {
    "id", "score", "anomaly_mean", "reliability", "direction", "size_pixels_processed",
    "bbox_rowcol_processed", "bbox_rowcol_jpg", "centroid_rowcol_jpg",
    "mapped_full_render_rowcol", "displacement_jpg_pixels", "displacement_note",
}


def test_analyze_render_without_tifxyz_dir_preserves_previous_schema(tmp_path):
    import json

    render = _seam_render(tmp_path)
    summary = analyze_render(render, str(tmp_path / "out"), RenderParams(working_downsample=1))

    # The in-process return value may still carry the flag: it drives CLI logging.
    assert summary["tifxyz_valid_mask_used"] is False

    for key in ("regions", "summary", "metadata"):
        path = summary["paths"][key]
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        # No tifxyz key, block, or null/false placeholder anywhere in the artifact.
        assert "tifxyz" not in raw, f"{key}.json leaked a tifxyz key"
        for field in NEW_FIELDS:
            assert field not in raw, f"{key}.json leaked {field}"

    with open(summary["paths"]["regions"], encoding="utf-8") as fh:
        payload = json.load(fh)
    assert set(payload) == PREV_REGIONS_KEYS
    assert payload["format"] == "scroll-anchor.render-candidates/v0"
    assert payload["regions"]
    for r in payload["regions"]:
        assert set(r) == PREV_REGION_FIELDS

    with open(summary["paths"]["summary"], encoding="utf-8") as fh:
        assert set(json.load(fh)) == PREV_SUMMARY_KEYS

    with open(summary["paths"]["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["format"] == "scroll-anchor.render-metadata/v0"
    assert "tifxyz_valid_mask_used" not in meta
    assert "tifxyz_valid_mask" not in meta


def test_analyze_render_with_tifxyz_dir_adds_diagnostics_only(tmp_path):
    import json

    render = _seam_render(tmp_path)
    p = RenderParams(working_downsample=1)
    plain = analyze_render(render, str(tmp_path / "out_plain"), p)

    # tifxyz raster at half the render resolution (consistent 2x scaling).
    xyz = np.ones((100, 160), np.float32)
    invalid = np.zeros((100, 160), bool)
    invalid[:, 120:] = True
    x, y, z = xyz.copy(), xyz.copy() * 2, xyz.copy() * 3
    for a in (x, y, z):
        a[invalid] = -1.0
    d = _write_tifxyz(tmp_path / "tifxyz", x, y, z)

    masked = analyze_render(render, str(tmp_path / "out_masked"), p, tifxyz_dir=d)

    assert masked["tifxyz_valid_mask_used"] is True
    # Detection, ranking and counts are untouched by the mask.
    for k in ("n_regions_exported", "n_regions_above_threshold", "n_regions_total",
              "exported_score_range"):
        assert masked[k] == plain[k]

    with open(masked["paths"]["regions"], encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["tifxyz_valid_mask_used"] is True
    assert payload["format"] == "scroll-anchor.render-candidates/v0"  # unchanged
    assert set(payload) == PREV_REGIONS_KEYS | {"tifxyz_valid_mask_used"}
    assert payload["regions"]
    for r in payload["regions"]:
        assert set(r) == PREV_REGION_FIELDS | set(NEW_FIELDS)
        assert r["boundary_distance_px"] >= 0.0
        assert 0.0 <= r["invalid_overlap_fraction"] <= 1.0
        assert isinstance(r["touches_invalid_boundary"], bool)

    with open(masked["paths"]["summary"], encoding="utf-8") as fh:
        assert set(json.load(fh)) == PREV_SUMMARY_KEYS | {"tifxyz_valid_mask_used"}

    with open(masked["paths"]["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["tifxyz_valid_mask_used"] is True
    info = meta["tifxyz_valid_mask"]
    assert info["raster_shape_rowcol"] == [100, 160]
    assert info["valid_fraction_tifxyz"] == pytest.approx(120 / 160)
    assert info["valid_fraction_processed"] == pytest.approx(120 / 160)

    # Alignment is ASSUMED, and the artifact must say so explicitly.
    align = info["alignment_assumptions"]
    assert align["independently_verified"] is False
    assert align["same_raster_origin"] is True
    assert align["same_row_column_orientation"] is True
    assert align["same_spatial_extent"] is True
    assert align["axis_swap"] is False
    assert align["horizontal_flip"] is False
    assert align["vertical_flip"] is False
    assert align["crop_or_offset"] is False
    assert "independently confirmed" in align["note"]


def test_analyze_render_rejects_incompatible_tifxyz(tmp_path):
    render = _seam_render(tmp_path)  # 200x320
    square = np.ones((100, 100), np.float32)
    d = _write_tifxyz(tmp_path / "square", square, square, square)
    with pytest.raises(ValueError, match="incompatible with the source render"):
        analyze_render(render, str(tmp_path / "out"), RenderParams(working_downsample=1),
                       tifxyz_dir=d)


def test_cli_tifxyz_dir_defaults_to_none():
    from scroll_anchor.cli import build_parser

    args = build_parser().parse_args(
        ["analyze-render", "--render", "r.jpg", "--output", "o"]
    )
    assert args.tifxyz_dir is None
    args = build_parser().parse_args(
        ["analyze-render", "--render", "r.jpg", "--output", "o", "--tifxyz-dir", "t"]
    )
    assert args.tifxyz_dir == "t"


# --------------------------------------------------------------------------- #
# Optional render-derived background mask (visible black margins)             #
# --------------------------------------------------------------------------- #
RENDER_BG_FIELDS = (
    "render_background_distance_px",
    "render_background_overlap_fraction",
    "touches_render_background",
)


def _white(shape=(32, 32)):
    """Uniformly white processed render in [0, 1] (nothing near-black)"""
    return np.ones(shape, np.float32)


def _diag_from_mask(active):
    """Diagnostics whose single response is exactly ``active`` (any shape, not a box)"""
    active = np.asarray(active, bool)
    shape = active.shape
    zeros = np.zeros(shape, np.float32)
    return RenderDiagnostics(
        anomaly=np.where(active, np.float32(0.5), np.float32(0.0)),
        texture=np.full(shape, 0.06, np.float32),
        seam_h=zeros.copy(),
        seam_v=zeros.copy(),
        lag_h=zeros.copy(),
        lag_v=zeros.copy(),
        agreement=np.ones(shape, np.float32),
        horizontal=np.ones(shape, bool),
        proc_shape=shape,
    )


def test_border_connected_near_black_component_is_retained():
    img = _white()
    img[:8, :] = 0.0  # 8x32 = 256 px reaching the top raster edge
    bg, stats = derive_render_background_mask(img, 8, 100)

    expected = np.zeros((32, 32), bool)
    expected[:8, :] = True
    assert np.array_equal(bg, expected)
    assert stats["n_retained_components"] == 1
    assert stats["background_fraction_processed"] == pytest.approx(256 / 1024)


def test_internal_near_black_component_is_not_background():
    img = _white()
    img[:8, :] = 0.0          # border-connected margin
    img[15:25, 15:25] = 0.0   # interior dark mark, 100 px, touches no edge
    bg, stats = derive_render_background_mask(img, 8, 50)

    assert stats["n_near_black_components"] == 2
    assert stats["n_border_connected_components"] == 1
    assert stats["n_retained_components"] == 1
    # Dark papyrus marks in the interior must never become "background".
    assert not bg[15:25, 15:25].any()
    assert bg[:8, :].all()


def test_border_connected_component_below_minimum_size_is_dropped():
    img = _white()
    img[0, :8] = 0.0  # 8 px on the top edge
    bg, stats = derive_render_background_mask(img, 8, 100)

    assert not bg.any()
    assert stats["n_near_black_components"] == 1
    assert stats["n_border_connected_components"] == 1
    assert stats["n_retained_components"] == 0
    assert stats["background_fraction_processed"] == 0.0
    # The pixels are still counted as near-black; only the retention differs.
    # Fractions are reported rounded to 6 decimals, as elsewhere in the module.
    assert stats["near_black_fraction_processed"] == pytest.approx(8 / 1024, abs=1e-6)


def test_diagonal_touch_does_not_connect_under_four_connectivity():
    img = _white((16, 16))
    img[0:2, 0:2] = 0.0   # 4 px in the corner, touches both raster edges
    img[2:6, 2:6] = 0.0   # 16 px interior, meets the corner block only diagonally
    bg, stats = derive_render_background_mask(img, 8, 1)

    # Under 8-connectivity these would be one border-connected component.
    assert stats["n_near_black_components"] == 2
    assert stats["n_border_connected_components"] == 1
    assert stats["n_retained_components"] == 1
    assert bg[0:2, 0:2].all()
    assert not bg[2:6, 2:6].any()


def test_threshold_bounds_zero_and_255_are_accepted():
    img = _white()
    img[:4, :] = 0.0  # exactly 0 -> grayscale level 0
    bg0, stats0 = derive_render_background_mask(img, 0, 1)
    assert bg0[:4, :].all() and not bg0[4:, :].any()
    assert stats0["grayscale_threshold_8bit"] == 0

    bg255, stats255 = derive_render_background_mask(img, 255, 1)
    # Every level is <= 255, so the whole raster is one border-connected component.
    assert bg255.all()
    assert stats255["near_black_fraction_processed"] == 1.0
    assert stats255["n_near_black_components"] == 1


@pytest.mark.parametrize("bad", [-1, 256, 300])
def test_threshold_outside_8bit_range_is_rejected(bad):
    with pytest.raises(ValueError, match="0..255"):
        derive_render_background_mask(_white(), bad, 10)


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_minimum_component_size_is_rejected(bad):
    with pytest.raises(ValueError, match="positive integer"):
        derive_render_background_mask(_white(), 8, bad)


def test_background_mask_shape_dtype_and_statistics():
    img = _white((20, 40))
    img[:5, :] = 0.0            # 200 px border-connected
    img[10:12, 10:12] = 0.0     # 4 px interior
    bg, stats = derive_render_background_mask(img, 8, 50)

    assert bg.dtype == np.bool_ and bg.shape == (20, 40)
    assert stats == {
        "grayscale_threshold_8bit": 8,
        "minimum_component_pixels_processed": 50,
        "connectivity": 4,
        "near_black_fraction_processed": 0.255,   # 204 / 800
        "background_fraction_processed": 0.25,    # 200 / 800
        "n_near_black_components": 2,
        "n_border_connected_components": 1,
        "n_retained_components": 1,
    }


def _l_shaped_candidate():
    """An L-shaped response whose bbox contains pixels the component does not own"""
    active = np.zeros((64, 64), bool)
    active[20:32, 20:24] = True  # vertical arm
    active[28:32, 20:40] = True  # horizontal arm -> bbox rows 20..31, cols 20..39
    return active


def test_region_overlap_uses_component_pixels_not_bbox():
    active = _l_shaped_candidate()
    bg = np.zeros((64, 64), bool)
    bg[20:27, 30:40] = True  # inside the candidate bbox, outside the component

    diag = _diag_from_mask(active)
    regs, _ = extract_regions_with_stats(
        diag, 1.0, 1.0, RenderParams(), None, bg
    )
    assert len(regs) == 1
    r = regs[0]
    r0, c0, r1, c1 = r["bbox_rowcol_processed"]
    assert (r0, c0, r1, c1) == (20, 20, 31, 39)
    # A bbox-based calculation would report ~1/3 overlap here.
    assert bg[r0:r1 + 1, c0:c1 + 1].mean() > 0.25
    assert r["render_background_overlap_fraction"] == 0.0
    # Nearest component pixel is row 28 under the block's last row 26.
    assert r["render_background_distance_px"] == pytest.approx(2.0)
    assert r["touches_render_background"] is False


def test_candidate_inside_background_has_zero_distance_and_touches():
    diag = _blob_diag()  # 12x12 response at rows/cols 20..31
    bg = np.zeros((64, 64), bool)
    bg[:, 30:] = True  # covers the right 2 columns of the blob

    regs, _ = extract_regions_with_stats(diag, 1.0, 1.0, RenderParams(), None, bg)
    r = regs[0]
    assert r["render_background_distance_px"] == 0.0
    assert r["touches_render_background"] is True
    assert r["render_background_overlap_fraction"] == pytest.approx(2 / 12, abs=1e-6)


def test_interior_candidate_has_positive_distance_and_no_overlap():
    diag = _blob_diag()
    bg = np.zeros((64, 64), bool)
    bg[:, 40:] = True  # clear of the blob (cols 20..31)

    regs, _ = extract_regions_with_stats(diag, 1.0, 1.0, RenderParams(), None, bg)
    r = regs[0]
    assert r["render_background_overlap_fraction"] == 0.0
    assert r["touches_render_background"] is False
    # Column 40 is 9 px from the blob's right edge (col 31); the padded raster edge
    # is 21 px away, so the background dominates.
    assert r["render_background_distance_px"] == pytest.approx(9.0)


def test_render_background_mask_does_not_change_scores_or_ranking():
    base = _texture(seed=13)
    img = base.copy()
    mid = img.shape[0] // 2
    img[mid:, :] = np.roll(base, 6, axis=1)[mid:, :]
    p = RenderParams()
    diag = analyze_array(img, p)

    bg = np.zeros(diag.anomaly.shape, bool)
    bg[:, 200:] = True

    plain, c_plain = extract_regions_with_stats(diag, 1.0, 1.0, p)
    marked, c_marked = extract_regions_with_stats(diag, 1.0, 1.0, p, None, bg)

    assert c_plain == c_marked
    assert [r["id"] for r in plain] == [r["id"] for r in marked]
    assert [r["score"] for r in plain] == [r["score"] for r in marked]
    for a, b in zip(plain, marked):
        assert all(b[k] == v for k, v in a.items())


def test_no_render_background_fields_without_mask():
    diag = _blob_diag()
    regs, _ = extract_regions_with_stats(diag, 1.0, 1.0, RenderParams())
    assert regs and all(k not in regs[0] for k in RENDER_BG_FIELDS)


def test_render_background_mask_shape_mismatch_raises():
    diag = _blob_diag()
    with pytest.raises(ValueError, match="render background mask shape"):
        extract_regions_with_stats(
            diag, 1.0, 1.0, RenderParams(), None, np.ones((32, 32), bool)
        )


def test_tifxyz_and_render_background_diagnostics_coexist_on_a_region():
    diag = _blob_diag()  # blob at rows/cols 20..31
    valid = np.ones((64, 64), bool)
    valid[:, 30:] = False   # tifxyz invalid overlaps the blob's right 2 columns
    bg = np.zeros((64, 64), bool)
    bg[:, 40:] = True       # render background is clear of the blob

    regs, _ = extract_regions_with_stats(diag, 1.0, 1.0, RenderParams(), valid, bg)
    r = regs[0]
    assert set(NEW_FIELDS) | set(RENDER_BG_FIELDS) <= set(r)
    # Deliberately disagreeing masks: neither field set may be derived from the other.
    assert r["boundary_distance_px"] == 0.0
    assert r["touches_invalid_boundary"] is True
    assert r["invalid_overlap_fraction"] == pytest.approx(2 / 12, abs=1e-6)
    assert r["render_background_distance_px"] == pytest.approx(9.0)
    assert r["touches_render_background"] is False
    assert r["render_background_overlap_fraction"] == 0.0


def _margin_render(tmp_path, h=200, w=320, seed=14, margin=20):
    """Seam render with a deterministic black left margin (edge-connected background)"""
    Image = pytest.importorskip("PIL.Image")
    base = _texture(h, w, seed=seed)
    img = base.copy()
    img[h // 2:, :] = np.roll(base, 6, axis=1)[h // 2:, :]
    img[:, :margin] = 0.0
    path = tmp_path / "render_margin.png"  # lossless: keeps the margin exactly black
    Image.fromarray((img * 255).astype(np.uint8), mode="L").save(str(path))
    return str(path)


def test_analyze_render_without_render_background_args_preserves_previous_schema(tmp_path):
    import json

    render = _margin_render(tmp_path)
    summary = analyze_render(render, str(tmp_path / "out"), RenderParams(working_downsample=1))
    assert summary["render_background_mask_used"] is False

    for key in ("regions", "summary", "metadata"):
        with open(summary["paths"][key], encoding="utf-8") as fh:
            raw = fh.read()
        assert "render_background" not in raw, f"{key}.json leaked a render-background key"
        for field in RENDER_BG_FIELDS:
            assert field not in raw, f"{key}.json leaked {field}"

    with open(summary["paths"]["regions"], encoding="utf-8") as fh:
        payload = json.load(fh)
    assert set(payload) == PREV_REGIONS_KEYS
    assert payload["regions"]
    for r in payload["regions"]:
        assert set(r) == PREV_REGION_FIELDS

    with open(summary["paths"]["summary"], encoding="utf-8") as fh:
        assert set(json.load(fh)) == PREV_SUMMARY_KEYS

    with open(summary["paths"]["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert "render_background_mask_used" not in meta
    assert "render_background_mask" not in meta


def test_analyze_render_with_render_background_args_adds_diagnostics_only(tmp_path):
    import json

    render = _margin_render(tmp_path)
    p = RenderParams(working_downsample=1)
    plain = analyze_render(render, str(tmp_path / "out_plain"), p)
    marked = analyze_render(
        render, str(tmp_path / "out_bg"), p,
        render_background_threshold=8,
        render_background_min_component_pixels=500,
    )

    assert marked["render_background_mask_used"] is True
    assert marked["tifxyz_valid_mask_used"] is False
    for k in ("n_regions_exported", "n_regions_above_threshold", "n_regions_total",
              "exported_score_range"):
        assert marked[k] == plain[k]

    with open(marked["paths"]["regions"], encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["render_background_mask_used"] is True
    assert payload["format"] == "scroll-anchor.render-candidates/v0"  # unchanged
    assert set(payload) == PREV_REGIONS_KEYS | {"render_background_mask_used"}
    assert payload["regions"]
    for r in payload["regions"]:
        assert set(r) == PREV_REGION_FIELDS | set(RENDER_BG_FIELDS)
        assert r["render_background_distance_px"] >= 0.0
        assert 0.0 <= r["render_background_overlap_fraction"] <= 1.0
        assert isinstance(r["touches_render_background"], bool)

    with open(marked["paths"]["summary"], encoding="utf-8") as fh:
        assert set(json.load(fh)) == PREV_SUMMARY_KEYS | {"render_background_mask_used"}

    with open(marked["paths"]["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["render_background_mask_used"] is True
    info = meta["render_background_mask"]
    assert info["grayscale_threshold_8bit"] == 8
    assert info["minimum_component_pixels_processed"] == 500
    assert info["connectivity"] == 4
    assert info["source"].startswith("edge-connected near-black")
    assert info["effect_on_scoring"] == "none; diagnostics only"
    assert info["distance_units"] == "processed (working-resolution) pixels"
    # The injected 200x20 margin is the retained background.
    assert info["background_fraction_processed"] == pytest.approx(20 / 320, abs=0.01)
    assert info["n_retained_components"] >= 1
    assert any("not universally calibrated" in s for s in info["limitations"])
    assert any("NOT equivalent to tifxyz" in s for s in info["limitations"])
    # The threshold is a run parameter, not a detector parameter.
    assert "render_background_threshold" not in meta["params"]


@pytest.mark.parametrize("kwargs", [
    {"render_background_threshold": 8},
    {"render_background_min_component_pixels": 500},
])
def test_analyze_render_requires_both_render_background_args(tmp_path, kwargs):
    render = _margin_render(tmp_path)
    with pytest.raises(ValueError, match="require BOTH"):
        analyze_render(render, str(tmp_path / "out"), RenderParams(working_downsample=1),
                       **kwargs)


def test_analyze_render_with_tifxyz_and_render_background_together(tmp_path):
    import json

    render = _margin_render(tmp_path)  # 200x320
    p = RenderParams(working_downsample=1)
    plain = analyze_render(render, str(tmp_path / "out_plain"), p)

    # tifxyz raster at half the render resolution (consistent 2x scaling).
    xyz = np.ones((100, 160), np.float32)
    invalid = np.zeros((100, 160), bool)
    invalid[:, 120:] = True
    x, y, z = xyz.copy(), xyz.copy() * 2, xyz.copy() * 3
    for a in (x, y, z):
        a[invalid] = -1.0
    d = _write_tifxyz(tmp_path / "tifxyz_both", x, y, z)

    both = analyze_render(
        render, str(tmp_path / "out_both"), p, tifxyz_dir=d,
        render_background_threshold=8,
        render_background_min_component_pixels=500,
    )

    assert both["tifxyz_valid_mask_used"] is True
    assert both["render_background_mask_used"] is True
    for k in ("n_regions_exported", "n_regions_above_threshold", "n_regions_total",
              "exported_score_range"):
        assert both[k] == plain[k]

    with open(both["paths"]["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    # Both blocks present, neither overwriting the other.
    assert meta["tifxyz_valid_mask_used"] is True
    assert meta["render_background_mask_used"] is True
    assert meta["tifxyz_valid_mask"]["raster_shape_rowcol"] == [100, 160]
    assert meta["render_background_mask"]["grayscale_threshold_8bit"] == 8

    with open(both["paths"]["regions"], encoding="utf-8") as fh:
        payload = json.load(fh)
    assert set(payload) == PREV_REGIONS_KEYS | {
        "tifxyz_valid_mask_used", "render_background_mask_used"
    }
    assert payload["regions"]
    for r in payload["regions"]:
        assert set(r) == PREV_REGION_FIELDS | set(NEW_FIELDS) | set(RENDER_BG_FIELDS)

    with open(both["paths"]["summary"], encoding="utf-8") as fh:
        assert set(json.load(fh)) == PREV_SUMMARY_KEYS | {
            "tifxyz_valid_mask_used", "render_background_mask_used"
        }


def test_cli_render_background_options_default_to_none():
    from scroll_anchor.cli import build_parser

    args = build_parser().parse_args(
        ["analyze-render", "--render", "r.jpg", "--output", "o"]
    )
    assert args.render_background_threshold is None
    assert args.render_background_min_component_pixels is None

    args = build_parser().parse_args([
        "analyze-render", "--render", "r.jpg", "--output", "o",
        "--render-background-threshold", "32",
        "--render-background-min-component-pixels", "5000",
    ])
    assert args.render_background_threshold == 32
    assert args.render_background_min_component_pixels == 5000
