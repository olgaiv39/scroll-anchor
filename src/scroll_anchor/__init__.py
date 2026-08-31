"""ScrollAnchor: conservative surface-label diagnostics for volumetric papyrus CT"""
from __future__ import annotations

from .config import RunConfig
from .diagnostics import Diagnostics, ProfileSelectionState, compute_diagnostics
from .pipeline import AnalysisResult, analyze_surface
from .phase_score import PhasePatchScore, section_local_phase_patch_score
from .tifxyz import Surface, read_tifxyz, write_tifxyz
from .volume import VolumeROI

__version__ = "0.2.0"
__author__ = "Olga Ivanova"
__email__ = "ivolga.vak@gmail.com"

__all__ = [
    "RunConfig",
    "Diagnostics",
    "ProfileSelectionState",
    "compute_diagnostics",
    "AnalysisResult",
    "analyze_surface",
    "PhasePatchScore",
    "section_local_phase_patch_score",
    "Surface",
    "read_tifxyz",
    "write_tifxyz",
    "VolumeROI",
    "__version__",
    "__author__",
    "__email__",
]
