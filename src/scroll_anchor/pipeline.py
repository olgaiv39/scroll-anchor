"""End-to-end analysis pipeline: surface + volume -> diagnostics"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .diagnostics import Diagnostics, compute_diagnostics
from .logging_setup import get_logger
from .normals import compute_normals
from .sampling import sample_profiles
from .tifxyz import Surface
from .volume import VolumeROI

log = get_logger(__name__)


@dataclass
class AnalysisResult:
    diagnostics: Diagnostics
    normals: np.ndarray
    profiles: np.ndarray
    offsets: np.ndarray
    points_xyz: np.ndarray


def analyze_surface(surface: Surface, volume: VolumeROI, config: RunConfig) -> AnalysisResult:
    """Run normal estimation, profile sampling and diagnostics for one surface"""
    normals, normal_valid = compute_normals(surface)
    valid = surface.valid & normal_valid
    points = surface.points()
    log.info(
        "analyzing surface %s (%d valid vertices)", surface.shape, int(valid.sum())
    )
    profiles, offsets = sample_profiles(
        points, normals, volume, config.sampling, chunk_rows=config.chunk_rows
    )
    diag = compute_diagnostics(
        profiles, offsets, points, normals, valid, config.diagnostics,
        correction=config.correction,
    )
    return AnalysisResult(
        diagnostics=diag, normals=normals, profiles=profiles, offsets=offsets, points_xyz=points
    )
