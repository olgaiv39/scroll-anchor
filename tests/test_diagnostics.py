from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from scroll_anchor.config import DiagnosticsConfig
from scroll_anchor.diagnostics import (
    ProfileSelectionState, _hysteresis, _robust_residual_magnitude,
    compute_diagnostics,
)
from scroll_anchor.tifxyz import Surface, read_tifxyz, write_tifxyz


def _plane(H=9, W=9):
    rows, cols = np.indices((H, W), dtype=np.float32)
    return np.stack((cols, rows, 20.0 + 0.25 * cols), axis=-1)


def test_switch_residual_ignores_invalid_coordinate_values():
    points = _plane()
    valid = np.ones(points.shape[:2], dtype=bool)
    valid[3:6, 3:6] = False

    changed = points.copy()
    changed[~valid] = (1e6, -1e6, 5e5)

    actual = _robust_residual_magnitude(points, valid, window=5)
    altered = _robust_residual_magnitude(changed, valid, window=5)
    np.testing.assert_array_equal(actual, altered)


def test_switch_residual_matches_tifxyz_roundtrip(tmp_path):
    points = _plane()
    valid = np.ones(points.shape[:2], dtype=bool)
    valid[3:6, 3:6] = False
    surface = Surface(
        x=points[..., 0], y=points[..., 1], z=points[..., 2], valid=valid
    )

    path = tmp_path / "surface"
    write_tifxyz(str(path), surface)
    restored = read_tifxyz(str(path))

    before = _robust_residual_magnitude(surface.points(), surface.valid, window=5)
    after = _robust_residual_magnitude(restored.points(), restored.valid, window=5)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(
        _hysteresis(before / 8.0, high=0.5, low=0.35),
        _hysteresis(after / 8.0, high=0.5, low=0.35),
    )


def test_switch_residual_preserves_supported_all_valid_median():
    points = _plane()
    valid = np.ones(points.shape[:2], dtype=bool)
    ref = np.empty_like(points)
    for c in range(3):
        ref[..., c] = median_filter(points[..., c], size=5, mode="nearest")
    expected = np.linalg.norm(points - ref, axis=-1).astype(np.float32)

    actual = _robust_residual_magnitude(points, valid, window=5)
    np.testing.assert_array_equal(actual, expected)


def test_switch_residual_requires_window_support_on_small_grid():
    points = _plane(H=7, W=7)
    valid = np.ones((7, 7), dtype=bool)
    points[2:5, 2:5, 2] += 10.0

    actual = _robust_residual_magnitude(points, valid, window=31)
    np.testing.assert_array_equal(actual, np.zeros((7, 7), dtype=np.float32))


def test_geometric_residual_does_not_change_profile_confidence(monkeypatch):
    points = _plane(); valid = np.ones(points.shape[:2], dtype=bool)
    normals = np.zeros_like(points); normals[..., 2] = 1.0
    offsets = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    profiles = np.zeros((*valid.shape, 3), dtype=np.float32); profiles[..., 1] = 1.0
    cfg = DiagnosticsConfig(switch_smooth_window=31)

    monkeypatch.setattr(
        "scroll_anchor.diagnostics._grid_normal_residual",
        lambda *args: np.zeros(valid.shape, dtype=np.float32),
    )
    flat = compute_diagnostics(profiles, offsets, points, normals, valid, cfg)
    monkeypatch.setattr(
        "scroll_anchor.diagnostics._grid_normal_residual",
        lambda *args: np.full(valid.shape, 99.0, dtype=np.float32),
    )
    curved = compute_diagnostics(profiles, offsets, points, normals, valid, cfg)

    np.testing.assert_array_equal(flat.confidence, curved.confidence)
    np.testing.assert_array_equal(flat.drift_score, curved.drift_score)
    np.testing.assert_array_equal(flat.switch_score, curved.switch_score)


def test_profile_selection_state_preserves_peak_provenance():
    offsets = np.arange(-2.0, 3.0, dtype=np.float32)
    profiles = np.array([[
        [0.0, 0.0, 1.0, 0.0, 0.0],  # one detected local peak
        [0.0, 1.0, 0.0, 1.0, 0.0],  # multiple detected local peaks
        [0.0, 0.2, 0.4, 0.6, 1.0],  # no local peak: global-max fallback
        [0.5, 0.5, 0.5, 0.5, 0.5],  # unusable dynamic range
        [0.0, 0.0, 1.0, 0.0, 0.0],  # outside detector coverage
    ]], dtype=np.float32)
    valid = np.array([[True, True, True, True, False]])
    points = np.zeros((1, 5, 3), dtype=np.float32)
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0

    diag = compute_diagnostics(
        profiles, offsets, points, normals, valid,
        DiagnosticsConfig(peak_min_separation=2.0, switch_smooth_window=31),
    )

    assert diag.profile_selection_state.dtype == np.uint8
    assert diag.profile_selection_state.tolist() == [[
        ProfileSelectionState.LOCAL_PEAK_SINGLE,
        ProfileSelectionState.LOCAL_PEAK_MULTIPLE,
        ProfileSelectionState.GLOBAL_MAX_FALLBACK,
        ProfileSelectionState.PROFILE_UNUSABLE,
        ProfileSelectionState.NOT_EVALUATED,
    ]]
    assert diag.confidence[0, 2] == 1.0
    assert not diag.review[0, 2]
    assert diag.confidence[0, 3] == 0.0
    assert diag.review[0, 3]


def test_supported_segment_excludes_unsupported_peak_and_fallback():
    offsets = np.arange(-2.0, 3.0, dtype=np.float32)
    profiles = np.array([[[100.0, 0.0, 1.0, 0.0, 100.0],
                          [100.0, 0.0, 0.2, 0.4, 1.0]]], dtype=np.float32)
    support = np.array([[[False, True, True, True, False],
                         [False, True, True, True, False]]])
    valid = np.ones((1, 2), dtype=bool)
    points = np.zeros((1, 2, 3), dtype=np.float32)
    normals = np.zeros_like(points); normals[..., 2] = 1.0
    cfg = DiagnosticsConfig(peak_min_separation=2.0, switch_smooth_window=31)

    diag = compute_diagnostics(profiles, offsets, points, normals, valid, cfg,
                               sample_support=support)
    assert diag.chosen_offset.tolist() == [[0.0, 1.0]]
    assert diag.profile_selection_state.tolist() == [[
        ProfileSelectionState.LOCAL_PEAK_SINGLE,
        ProfileSelectionState.GLOBAL_MAX_FALLBACK,
    ]]


def test_missing_center_support_is_explicitly_unusable():
    offsets = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    profiles = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    support = np.array([[[True, False, True]]])
    valid = np.ones((1, 1), dtype=bool)
    points = np.zeros((1, 1, 3), dtype=np.float32)
    normals = np.zeros_like(points); normals[..., 2] = 1.0
    diag = compute_diagnostics(profiles, offsets, points, normals, valid,
                               DiagnosticsConfig(switch_smooth_window=31),
                               sample_support=support)
    assert diag.profile_selection_state[0, 0] == ProfileSelectionState.PROFILE_UNUSABLE
    assert np.isnan(diag.chosen_offset[0, 0])


def test_full_support_preserves_legacy_diagnostics_bitwise():
    offsets = np.arange(-2.0, 3.0, dtype=np.float32)
    profiles = np.array([[[0.0, 0.3, 1.0, 0.2, 0.0]]], dtype=np.float32)
    valid = np.ones((1, 1), dtype=bool)
    points = np.zeros((1, 1, 3), dtype=np.float32)
    normals = np.zeros_like(points); normals[..., 2] = 1.0
    cfg = DiagnosticsConfig(switch_smooth_window=31)
    legacy = compute_diagnostics(profiles, offsets, points, normals, valid, cfg)
    supported = compute_diagnostics(profiles, offsets, points, normals, valid, cfg,
                                    sample_support=np.ones_like(profiles, dtype=bool))
    for name in ("chosen_offset", "profile_selection_state", "drift_score", "switch_score",
                 "margin", "evidence", "contrast", "confidence", "review"):
        np.testing.assert_array_equal(getattr(legacy, name), getattr(supported, name))
