from __future__ import annotations

import numpy as np

from scroll_anchor.tifxyz import Surface, read_tifxyz, write_tifxyz


def test_roundtrip_and_validity(tmp_path):
    H, W = 8, 6
    x = np.random.rand(H, W).astype(np.float32) + 1
    y = np.random.rand(H, W).astype(np.float32) + 1
    z = np.random.rand(H, W).astype(np.float32) + 1
    valid = np.ones((H, W), dtype=bool)
    valid[0, 0] = False
    s = Surface(x=x, y=y, z=z, valid=valid, scale=(2.0, 3.0), meta={"type": "seg"})

    d = tmp_path / "seg"
    write_tifxyz(str(d), s, extra_channels={"foo": np.zeros((H, W), np.float32)}, overwrite=True)
    r = read_tifxyz(str(d))

    assert r.shape == (H, W)
    assert r.scale == (2.0, 3.0)
    assert not r.valid[0, 0]
    np.testing.assert_allclose(r.x[valid], x[valid], rtol=1e-5)
    np.testing.assert_allclose(r.z[valid], z[valid], rtol=1e-5)


def test_zle_zero_invalidates(tmp_path):
    s = Surface(
        x=np.ones((4, 4), np.float32),
        y=np.ones((4, 4), np.float32),
        z=np.zeros((4, 4), np.float32),
        valid=np.ones((4, 4), bool),
    )
    d = tmp_path / "seg"
    write_tifxyz(str(d), s, overwrite=True)
    r = read_tifxyz(str(d))
    assert not r.valid.any()
