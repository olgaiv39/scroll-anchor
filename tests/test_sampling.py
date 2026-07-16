from __future__ import annotations

import numpy as np

from scroll_anchor.config import SamplingConfig
from scroll_anchor.sampling import offset_axis, sample_profiles
from scroll_anchor.volume import VolumeROI


def test_sample_world_trilinear_linear_field():
    Dz, Dy, Dx = 10, 5, 5
    zz = np.arange(Dz)[:, None, None] * np.ones((Dz, Dy, Dx))
    vol = VolumeROI.from_array(zz.astype(np.float32))
    pts = np.array([[2.0, 2.0, 3.5], [1.0, 1.0, 0.0]], dtype=np.float32)
    vals = vol.sample_world(pts)
    np.testing.assert_allclose(vals, [3.5, 0.0], atol=1e-5)


def test_origin_offset():
    vol = VolumeROI.from_array(np.ones((4, 4, 4), np.float32), origin=(10, 20, 30))
    inside = vol.sample_world(np.array([[31.0, 21.0, 11.0]], np.float32))
    outside = vol.sample_world(np.array([[0.0, 0.0, 0.0]], np.float32))
    assert abs(float(inside[0]) - 1.0) < 1e-5
    assert abs(float(outside[0])) < 1e-5


def test_profile_peaks_at_sheet():
    Dz = 12
    zz = np.arange(Dz)[:, None, None]
    vol_arr = np.exp(-((zz - 5.0) ** 2) / (2 * 1.5 ** 2)) * np.ones((Dz, 4, 4))
    vol = VolumeROI.from_array(vol_arr.astype(np.float32))
    H, W = 4, 4
    pts = np.zeros((H, W, 3), np.float32)
    pts[..., 0] = np.arange(W)[None, :]
    pts[..., 1] = np.arange(H)[:, None]
    pts[..., 2] = 7.0
    normals = np.zeros((H, W, 3), np.float32)
    normals[..., 2] = 1.0
    cfg = SamplingConfig(radius=6.0, step=0.5)
    profiles, offsets = sample_profiles(pts, normals, vol, cfg)
    peak_off = offsets[np.argmax(profiles[1, 1])]
    assert abs(peak_off - (-2.0)) <= 0.5
