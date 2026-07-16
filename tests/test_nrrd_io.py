from __future__ import annotations

import numpy as np
import pytest

nrrd = pytest.importorskip("nrrd")

from scroll_anchor.nrrd_io import load_cube, read_nrrd


def _internal(shape=(2, 3, 4)):
    z, y, x = np.indices(shape)
    return (z * 100 + y * 10 + x).astype(np.int16)


def _write(path, stored, space_directions, origin=(0, 0, 0)):
    header = {
        "space": "left-posterior-superior",
        "space directions": np.asarray(space_directions, dtype=float),
        "space origin": np.asarray(origin, dtype=float),
    }
    nrrd.write(str(path), stored, header)


def test_identity_axes(tmp_path):
    I = _internal()
    _write(tmp_path / "v.nrrd", I, np.eye(3))
    v = read_nrrd(str(tmp_path / "v.nrrd"))
    assert v.axis_perm == (0, 1, 2)
    np.testing.assert_array_equal(v.data, I)


def test_permuted_axes_are_restored(tmp_path):
    I = _internal()
    # stored axis order = (y, x, z) -> internal perm (1, 2, 0)
    stored = np.transpose(I, (1, 2, 0)).copy()
    sd = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    _write(tmp_path / "v.nrrd", stored, sd)
    v = read_nrrd(str(tmp_path / "v.nrrd"))
    assert v.axis_perm == (1, 2, 0)
    # read_nrrd must undo the permutation back to internal [z, y, x]
    np.testing.assert_array_equal(v.data, I)


def test_spacing_and_origin_reordered(tmp_path):
    I = _internal()
    stored = np.transpose(I, (1, 2, 0)).copy()
    sd = np.array([[0, 2, 0], [0, 0, 3], [4, 0, 0]], dtype=float)  # y:2, x:3, z:4
    _write(tmp_path / "v.nrrd", stored, sd, origin=(11, 22, 33))
    v = read_nrrd(str(tmp_path / "v.nrrd"))
    # internal (z, y, x) spacing
    np.testing.assert_allclose(v.spacing, (4.0, 2.0, 3.0))
    np.testing.assert_allclose(v.origin, (11.0, 22.0, 33.0))


def test_non_axis_aligned_raises(tmp_path):
    I = _internal()
    sd = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    _write(tmp_path / "v.nrrd", I, sd)
    with pytest.raises(ValueError):
        read_nrrd(str(tmp_path / "v.nrrd"))


def test_load_cube_mismatch_raises(tmp_path):
    a = _internal((2, 3, 4))
    b = _internal((2, 3, 5))
    _write(tmp_path / "a.nrrd", a, np.eye(3))
    _write(tmp_path / "b.nrrd", b, np.eye(3))
    with pytest.raises(ValueError):
        load_cube(str(tmp_path / "a.nrrd"), str(tmp_path / "b.nrrd"))
