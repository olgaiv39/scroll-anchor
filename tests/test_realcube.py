from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scroll_anchor import realcube as rc


def _sheet(shape, zc, ids=1):
    """A single-run sheet at depth zc along axis 0."""
    vol = np.zeros(shape, dtype=np.int64)
    vol[zc] = ids
    vol[zc + 1] = ids
    return vol


def test_medial_extraction_single_run_and_axis():
    inst = _sheet((10, 6, 6), zc=4) > 0
    ms = rc.extract_medial_surface(inst, roi_origin=(0, 0, 0))
    assert ms.proj_axis == 0
    assert ms.valid.all()
    np.testing.assert_allclose(ms.depth[ms.valid], 4.5, atol=1e-6)


def test_medial_rejects_double_crossing():
    inst = np.zeros((12, 4, 4), dtype=bool)
    inst[3] = True
    inst[9, 0, 0] = True  # a second crossing only for column (0,0)
    _, single = rc.medial_depth_on_grid(inst, 0)
    assert not single[0, 0]
    assert single[1, 1]


def test_grid_to_cube_coords_roundtrip():
    inst = _sheet((10, 5, 5), zc=4) > 0
    ms = rc.extract_medial_surface(inst, roi_origin=(100, 20, 30))
    surf = rc.medial_to_surface(ms)
    # Y, X follow grid rows/cols offset by origin; Z is medial depth + origin.
    assert surf.y[2, 3] == 20 + 2
    assert surf.x[2, 3] == 30 + 3
    np.testing.assert_allclose(surf.z[surf.valid], 100 + 4.5, atol=1e-6)


def test_drift_moves_patch_by_offset():
    inst = _sheet((10, 12, 12), zc=4) > 0
    ms = rc.extract_medial_surface(inst, roi_origin=(0, 0, 0))
    surf = rc.medial_to_surface(ms)
    corr = rc.make_drift(surf, offset=3.0, half=3, center_frac=(0.5, 0.5))
    disp = np.linalg.norm(corr.surface.points() - surf.points(), axis=-1)
    assert corr.region.any()
    np.testing.assert_allclose(disp[corr.region], 3.0, atol=1e-4)
    assert np.allclose(disp[~corr.region], 0.0, atol=1e-4)
    assert corr.info["n_vertices"] == int(corr.region.sum())


def test_switch_verifies_target_instance():
    shape = (16, 12, 12)
    src_id, tgt_id = 5, 7
    mask = np.zeros(shape, dtype=np.int64)
    mask[3] = src_id; mask[4] = src_id       # source sheet
    mask[10] = tgt_id; mask[11] = tgt_id     # target sheet
    src = rc.extract_medial_surface(mask == src_id, roi_origin=(0, 0, 0))
    surf = rc.medial_to_surface(src)
    dt, vt = rc.medial_depth_on_grid(mask == tgt_id, src.proj_axis)
    tgt = rc.MedialSurface(depth=dt, valid=vt, proj_axis=src.proj_axis, roi_origin=(0, 0, 0))

    corr = rc.make_switch(surf, tgt, mask, tgt_id, src_id, half=4)
    assert corr.region.any()
    # switched vertices now sit at the target medial depth (~10.5)
    np.testing.assert_allclose(corr.surface.z[corr.region], 10.5, atol=1e-6)
    assert corr.info["target_id"] == tgt_id


def test_switch_rejects_when_target_absent():
    shape = (16, 12, 12)
    src_id, tgt_id = 5, 7
    mask = np.zeros(shape, dtype=np.int64)
    mask[3] = src_id; mask[4] = src_id
    mask[10] = tgt_id; mask[11] = tgt_id
    src = rc.extract_medial_surface(mask == src_id, roi_origin=(0, 0, 0))
    surf = rc.medial_to_surface(src)
    dt, vt = rc.medial_depth_on_grid(mask == tgt_id, src.proj_axis)
    tgt = rc.MedialSurface(depth=dt, valid=vt, proj_axis=src.proj_axis, roi_origin=(0, 0, 0))
    # Erase the target labels: correspondences must all be rejected.
    mask_no_target = np.where(mask == tgt_id, 0, mask)
    corr = rc.make_switch(surf, tgt, mask_no_target, tgt_id, src_id, half=4)
    assert corr.info["n_vertices"] == 0
    assert not corr.region.any()


def _fake_diag(H=8, W=8):
    z = np.zeros((H, W))
    return SimpleNamespace(
        valid=np.ones((H, W), bool),
        switch_score=z.copy(),
        drift_score=z.copy(),
        chosen_offset=z.copy(),
        confidence=np.ones((H, W)),
        review=np.zeros((H, W), bool),
        estimated_spacing=8.0,
    )


def test_switch_metrics_harmful_and_prf():
    d = _fake_diag()
    region = np.zeros((8, 8), bool); region[2:4, 2:4] = True
    d.switch_score[region] = 1.0
    d.review[region] = True
    on_wrong = region.copy()
    m = rc.switch_metrics(d, region, d.valid, on_wrong)
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    # everything wrong is under review, so nothing harmful is accepted
    assert m["harmful_acceptance_rate"] == 0.0
    assert m["review_recall"] == 1.0


def test_drift_metrics_sign_and_mae():
    d = _fake_diag()
    region = np.zeros((8, 8), bool); region[5:7, 5:7] = True
    inj = np.zeros((8, 8)); inj[region] = 3.0
    d.drift_score[region] = 3.0
    d.chosen_offset[region] = -3.0  # points back toward the true sheet
    m = rc.drift_metrics(d, region, inj, d.valid, drift_min=1.0)
    assert m["displacement_mae"] == 0.0
    assert m["sign_accuracy"] == 1.0
    assert m["recall"] == 1.0
