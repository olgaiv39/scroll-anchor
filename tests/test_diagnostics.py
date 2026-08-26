from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from scroll_anchor.diagnostics import _hysteresis, _robust_residual_magnitude
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
