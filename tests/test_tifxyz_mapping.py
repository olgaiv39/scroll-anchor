"""Focused tests for the render-candidate -> tifxyz coordinate mapping stage

Only small synthetic arrays and images are used. The real PHerc render and the
real tifxyz rasters are never touched, and no CT volume is involved.
"""
import csv
import json

import numpy as np
import pytest

from scroll_anchor.cli import build_parser
from scroll_anchor.tifxyz_mapping import (
    derive_integer_scale,
    map_render_candidates,
    select_candidates,
)

# Synthetic geometry: JPG 40x60, processed 20x30, tifxyz 4x6.
# -> source scale 10x10, processed scale 5x5 (same relationship as the real run,
#    but derived from these shapes, never hardcoded in production code).
JPG_SHAPE = [40, 60]
PROC_SHAPE = [20, 30]
TIF_SHAPE = (4, 6)


def _write_tifxyz(directory, x, y, z):
    """Write a bare x/y/z.tif triple (no mask.tif, matching the real dataset)"""
    tifffile = pytest.importorskip("tifffile")
    directory.mkdir(parents=True, exist_ok=True)
    for name, arr in (("x.tif", x), ("y.tif", y), ("z.tif", z)):
        tifffile.imwrite(str(directory / name), np.asarray(arr, dtype=np.float32))
    return str(directory)


def _default_tifxyz(tmp_path, name="tifxyz"):
    """x = 100*row + col, y = x + 0.5, z = x + 0.25 over a 4x6 raster"""
    rows, cols = TIF_SHAPE
    base = (np.arange(rows)[:, None] * 100 + np.arange(cols)[None, :]).astype(np.float32)
    return _write_tifxyz(tmp_path / name, base, base + 0.5, base + 0.25), base


def _region(cid, score, jr, jc, direction="horizontal"):
    """One exported candidate whose JPG centroid is (jr, jc)"""
    return {
        "id": cid,
        "score": score,
        "direction": direction,
        "centroid_rowcol_jpg": [jr, jc],
        "bbox_rowcol_jpg": [jr - 1, jc - 4, jr + 1, jc + 4],
        "bbox_rowcol_processed": [(jr - 1) / 2, (jc - 4) / 2, (jr + 1) / 2, (jc + 4) / 2],
    }


def _inputs(tmp_path, regions=None, jpg_shape=None, proc_shape=None):
    """Write regions.json + metadata.json and return their paths"""
    regions = regions if regions is not None else [
        _region(244, 0.200628, 5.0, 25.0),    # -> tifxyz (0, 2)
        _region(391, 0.186059, 15.0, 45.0),   # -> tifxyz (1, 4)
        _region(566, 0.178003, 25.0, 5.0),    # -> tifxyz (2, 0)
    ]
    rp = tmp_path / "regions.json"
    rp.write_text(json.dumps({
        "format": "scroll-anchor.render-candidates/v0",
        "regions": regions,
    }))
    mp = tmp_path / "metadata.json"
    mp.write_text(json.dumps({
        "format": "scroll-anchor.render-metadata/v0",
        "source_filename": "synthetic.jpg",
        "jpg_shape_rowcol": jpg_shape or JPG_SHAPE,
        "processed_shape_rowcol": proc_shape or PROC_SHAPE,
        "scale_processed_to_jpg_rowcol": [2.0, 2.0],
    }))
    return str(rp), str(mp)


def _run(tmp_path, out="out", **kw):
    rp, mp = kw.pop("paths", None) or _inputs(tmp_path)
    td = kw.pop("tifxyz_dir", None) or _default_tifxyz(tmp_path)[0]
    kw.setdefault("assume_identity_orientation", True)
    return map_render_candidates(rp, mp, td, str(tmp_path / out), **kw)


# --------------------------------------------------------------------------- #
# 1-3. Selection, by rank, by component ID, and deduplication                 #
# --------------------------------------------------------------------------- #
def test_select_by_rank_does_not_confuse_rank_with_component_id():
    regions = [_region(244, 0.2, 5, 25), _region(391, 0.18, 15, 45)]
    picked = select_candidates(regions, ranks=[2])
    assert [(r, int(g["id"])) for r, g in picked] == [(2, 391)]


def test_select_by_component_id():
    regions = [_region(244, 0.2, 5, 25), _region(391, 0.18, 15, 45)]
    picked = select_candidates(regions, component_ids=[244])
    assert [(r, int(g["id"])) for r, g in picked] == [(1, 244)]


def test_select_deduplicates_same_candidate_from_both_selectors():
    regions = [_region(244, 0.2, 5, 25), _region(391, 0.18, 15, 45)]
    picked = select_candidates(regions, ranks=[1], component_ids=[244])
    assert [(r, int(g["id"])) for r, g in picked] == [(1, 244)]


def test_select_requires_at_least_one_selector():
    with pytest.raises(ValueError, match="at least one --rank or --component-id"):
        select_candidates([_region(1, 0.2, 5, 25)])


@pytest.mark.parametrize("kw,msg", [
    ({"ranks": [9]}, "unknown exported rank"),
    ({"component_ids": [999]}, "unknown component ID"),
])
def test_unknown_selectors_fail(kw, msg):
    with pytest.raises(ValueError, match=msg):
        select_candidates([_region(244, 0.2, 5, 25)], **kw)


# --------------------------------------------------------------------------- #
# 4. Exact integer scale derivation                                           #
# --------------------------------------------------------------------------- #
def test_exact_integer_scale_derivation_per_axis():
    assert derive_integer_scale((40, 60), (4, 6), "t") == (10, 10)
    assert derive_integer_scale((20, 30), (4, 6), "t") == (5, 5)
    # Rows and columns are independent: differing per-axis scales are allowed.
    assert derive_integer_scale((40, 60), (4, 3), "t") == (10, 20)


def test_scales_are_derived_from_shapes_not_hardcoded(tmp_path):
    """A different raster size yields different scales through the same code path"""
    base = np.zeros((2, 3), np.float32)  # divides both 40x60 and 20x30 exactly
    td = _write_tifxyz(tmp_path / "t23", base, base, base)
    info = _run(tmp_path, tifxyz_dir=td, ranks=[1])
    assert info["source_to_tifxyz_rowcol"] == [20, 20]
    assert info["processed_to_tifxyz_rowcol"] == [10, 10]
    assert info["candidates"][0]["tifxyz_row"] == 0
    assert info["candidates"][0]["tifxyz_col"] == 1


# --------------------------------------------------------------------------- #
# 5-6. The two coordinate paths must agree                                    #
# --------------------------------------------------------------------------- #
def test_source_and_processed_derived_cells_agree(tmp_path):
    info = _run(tmp_path, ranks=[1, 2, 3])
    for e in info["candidates"]:
        assert e["tifxyz_cell_from_source_rowcol"] == e["tifxyz_cell_from_processed_rowcol"]
        assert e["tifxyz_cell_paths_agree"] is True


def test_disagreeing_coordinate_paths_fail(tmp_path):
    """An inconsistent stored processed centroid must fail, not be silently chosen"""
    region = _region(244, 0.2, 5.0, 25.0)
    region["centroid_rowcol_processed"] = [2.5, 3.0]  # -> (0, 0), not (0, 2)
    paths = _inputs(tmp_path, regions=[region])
    with pytest.raises(ValueError, match="disagree"):
        _run(tmp_path, paths=paths, ranks=[1])


# --------------------------------------------------------------------------- #
# 7. Orientation must be stated explicitly                                    #
# --------------------------------------------------------------------------- #
def test_missing_identity_orientation_flag_fails(tmp_path):
    with pytest.raises(ValueError, match="assume-identity-orientation"):
        _run(tmp_path, ranks=[1], assume_identity_orientation=False)


# --------------------------------------------------------------------------- #
# 8-9. Incompatible rasters are rejected, never resampled                     #
# --------------------------------------------------------------------------- #
def test_non_integer_raster_scale_fails(tmp_path):
    """A near-miss raster (the flattened-parameterisation case) is rejected"""
    base = np.zeros((7, 11), np.float32)
    td = _write_tifxyz(tmp_path / "odd", base, base, base)
    with pytest.raises(ValueError, match="exact positive integer scale"):
        _run(tmp_path, tifxyz_dir=td, ranks=[1])


def test_mismatched_xyz_shapes_fail(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    d = tmp_path / "bad"
    d.mkdir()
    tifffile.imwrite(str(d / "x.tif"), np.zeros((4, 6), np.float32))
    tifffile.imwrite(str(d / "y.tif"), np.zeros((4, 6), np.float32))
    tifffile.imwrite(str(d / "z.tif"), np.zeros((4, 5), np.float32))
    with pytest.raises(ValueError, match="shapes differ"):
        _run(tmp_path, tifxyz_dir=str(d), ranks=[1])


# --------------------------------------------------------------------------- #
# 10-13. Coordinate extraction and the exact -1 sentinel                      #
# --------------------------------------------------------------------------- #
def test_valid_xyz_extraction_uses_containing_cell(tmp_path):
    info = _run(tmp_path, ranks=[1, 2, 3])
    got = {(e["exported_rank"], e["component_id"]): (e["tifxyz_row"], e["tifxyz_col"],
                                                     e["x"], e["y"], e["z"])
           for e in info["candidates"]}
    # x = 100*row + col by construction, y = x + 0.5, z = x + 0.25
    assert got[(1, 244)] == (0, 2, 2.0, 2.5, 2.25)
    assert got[(2, 391)] == (1, 4, 104.0, 104.5, 104.25)
    assert got[(3, 566)] == (2, 0, 200.0, 200.5, 200.25)
    assert all(e["tifxyz_cell_valid"] for e in info["candidates"])


def test_exact_minus_one_marks_cell_invalid_without_fallback(tmp_path):
    rows, cols = TIF_SHAPE
    base = (np.arange(rows)[:, None] * 100 + np.arange(cols)[None, :]).astype(np.float32)
    x = base.copy()
    x[0, 2] = -1.0  # the cell rank 1 maps to
    td = _write_tifxyz(tmp_path / "inv", x, base + 0.5, base + 0.25)

    info = _run(tmp_path, tifxyz_dir=td, ranks=[1, 2])
    first = info["candidates"][0]
    assert (first["exported_rank"], first["component_id"]) == (1, 244)
    # Reported as invalid, still the SELECTED cell: no nearest-valid substitution.
    assert first["tifxyz_cell_valid"] is False
    assert (first["tifxyz_row"], first["tifxyz_col"]) == (0, 2)
    assert first["x"] == -1.0
    # A neighbouring valid cell was not silently used instead.
    assert first["y"] == 2.5 and first["z"] == 2.25
    assert info["candidates"][1]["tifxyz_cell_valid"] is True


def test_zero_and_other_negatives_stay_valid(tmp_path):
    """Only exact -1 is the sentinel; 0.0 and -2.0 are legitimate coordinates"""
    rows, cols = TIF_SHAPE
    x = np.zeros((rows, cols), np.float32)
    y = np.full((rows, cols), -2.0, np.float32)
    z = np.zeros((rows, cols), np.float32)
    td = _write_tifxyz(tmp_path / "zero", x, y, z)
    info = _run(tmp_path, tifxyz_dir=td, ranks=[1])
    e = info["candidates"][0]
    assert e["tifxyz_cell_valid"] is True
    assert (e["x"], e["y"], e["z"]) == (0.0, -2.0, 0.0)


# --------------------------------------------------------------------------- #
# 14. Out-of-bounds mapping fails loudly, nothing is clamped                  #
# --------------------------------------------------------------------------- #
def test_out_of_bounds_mapping_fails(tmp_path):
    # Centroid beyond the recorded JPG extent -> cell outside the tifxyz raster.
    paths = _inputs(tmp_path, regions=[_region(244, 0.2, 5.0, 125.0)])
    with pytest.raises(ValueError, match="outside the 4x6 tifxyz raster"):
        _run(tmp_path, paths=paths, ranks=[1])


# --------------------------------------------------------------------------- #
# 15. Deterministic ordering and output contents                              #
# --------------------------------------------------------------------------- #
def test_deterministic_json_and_csv_ordering(tmp_path):
    """Selectors given out of order still produce rank-ascending output"""
    info = _run(tmp_path, ranks=[3, 1], component_ids=[391])
    assert [e["exported_rank"] for e in info["candidates"]] == [1, 2, 3]

    payload = json.loads(open(info["files"]["json"]).read())
    assert [c["exported_rank"] for c in payload["candidates"]] == [1, 2, 3]
    assert [c["component_id"] for c in payload["candidates"]] == [244, 391, 566]
    assert payload["format"] == "scroll-anchor.candidate-tifxyz-mapping/v0"
    assert payload["identity_orientation_assumed"] is True
    assert payload["orientation_independently_verified"] is False
    assert payload["shapes"]["tifxyz_shape_rowcol"] == [4, 6]
    assert payload["scales"]["source_to_tifxyz_rowcol"] == [10, 10]
    assert "not yet converted to a CT NumPy array index" in " ".join(
        payload["coordinate_interpretation_warning"])

    with open(info["files"]["csv"]) as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["exported_rank"]) for r in rows] == [1, 2, 3]
    assert [int(r["component_id"]) for r in rows] == [244, 391, 566]


def test_inputs_are_not_modified(tmp_path):
    rp, mp = _inputs(tmp_path)
    before = (open(rp).read(), open(mp).read())
    _run(tmp_path, paths=(rp, mp), ranks=[1])
    assert (open(rp).read(), open(mp).read()) == before


# --------------------------------------------------------------------------- #
# 16. The preview is optional                                                 #
# --------------------------------------------------------------------------- #
def test_preview_written_only_when_render_supplied(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("matplotlib")

    info = _run(tmp_path, out="no_png", ranks=[1])
    assert info["files"]["png"] is None
    assert not (tmp_path / "no_png" / "candidate_tifxyz_mapping.png").exists()

    render = tmp_path / "synthetic.jpg"
    Image.fromarray(np.random.default_rng(0).integers(
        0, 255, size=(JPG_SHAPE[0], JPG_SHAPE[1]), dtype=np.uint8), mode="L").save(render)
    info = _run(tmp_path, out="with_png", ranks=[1, 2], render_path=str(render))
    png = tmp_path / "with_png" / "candidate_tifxyz_mapping.png"
    assert info["files"]["png"] == str(png)
    assert png.is_file() and png.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 17. Scores and exported ranks pass through unchanged                        #
# --------------------------------------------------------------------------- #
def test_scores_and_ranks_pass_through_unchanged(tmp_path):
    rp, mp = _inputs(tmp_path)
    source = json.loads(open(rp).read())["regions"]
    info = _run(tmp_path, paths=(rp, mp), ranks=[1, 2, 3])

    for exported_rank, region in enumerate(source, start=1):
        e = next(c for c in info["candidates"] if c["exported_rank"] == exported_rank)
        assert e["component_id"] == region["id"]
        assert e["score"] == region["score"]
        assert e["direction"] == region["direction"]


# --------------------------------------------------------------------------- #
# CLI wiring                                                                  #
# --------------------------------------------------------------------------- #
def test_cli_parses_repeated_selectors_and_orientation_flag():
    args = build_parser().parse_args([
        "map-render-candidates",
        "--regions", "r.json", "--metadata", "m.json",
        "--tifxyz-dir", "t", "--output", "o",
        "--rank", "7", "--rank", "11", "--component-id", "566",
        "--assume-identity-orientation",
    ])
    assert args.rank == [7, 11]
    assert args.component_id == [566]
    assert args.assume_identity_orientation is True
    assert args.render is None


def test_cli_orientation_flag_defaults_off():
    args = build_parser().parse_args([
        "map-render-candidates",
        "--regions", "r.json", "--metadata", "m.json",
        "--tifxyz-dir", "t", "--output", "o", "--rank", "1",
    ])
    assert args.assume_identity_orientation is False
