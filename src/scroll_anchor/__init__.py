"""ScrollAnchor: conservative surface-label diagnostics for volumetric papyrus CT"""
from __future__ import annotations

from .config import RunConfig
from .diagnostics import Diagnostics, compute_diagnostics
from .pipeline import AnalysisResult, analyze_surface
from .tifxyz import Surface, read_tifxyz, write_tifxyz
from .volume import VolumeROI

__version__ = "0.1.0"
__author__ = "Olga Ivanova"
__email__ = "ivolga.vak@gmail.com"

__all__ = [
    "RunConfig",
    "Diagnostics",
    "compute_diagnostics",
    "AnalysisResult",
    "analyze_surface",
    "Surface",
    "read_tifxyz",
    "write_tifxyz",
    "VolumeROI",
    "__version__",
    "__author__",
    "__email__",
]
