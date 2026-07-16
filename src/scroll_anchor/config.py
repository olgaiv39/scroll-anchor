"""Typed configuration for ScrollAnchor"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class SamplingConfig:
    """Normal-profile sampling settings"""

    radius: float = 12.0
    step: float = 0.5
    order: int = 1
    cval: float = 0.0


@dataclass
class DiagnosticsConfig:
    """Drift and sheet-switch scoring settings"""

    peak_min_prominence_frac: float = 0.15
    peak_min_separation: float = 2.0
    drift_min: float = 1.0
    sheet_spacing: Optional[float] = None
    switch_frac_of_spacing: float = 0.5
    smooth_window: int = 9
    # Must exceed the expected switched-patch diameter
    switch_smooth_window: int = 31
    margin_soft: float = 0.25


@dataclass
class ReviewConfig:
    """Review-region extraction settings"""

    confidence_review_below: float = 0.5
    min_region_vertices: int = 4
    max_regions: int = 200


@dataclass
class CorrectionConfig:
    """Conservative correction proposals, disabled by default"""

    enabled: bool = False
    min_confidence: float = 0.85
    max_offset: float = 6.0
    require_margin: float = 0.4


@dataclass
class RunConfig:
    """Top-level run configuration"""

    seed: int = 0
    chunk_rows: int = 128
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        data = dict(data or {})
        sampling = SamplingConfig(**(data.pop("sampling", {}) or {}))
        diagnostics = DiagnosticsConfig(**(data.pop("diagnostics", {}) or {}))
        review = ReviewConfig(**(data.pop("review", {}) or {}))
        correction = CorrectionConfig(**(data.pop("correction", {}) or {}))
        return cls(
            sampling=sampling,
            diagnostics=diagnostics,
            review=review,
            correction=correction,
            **data,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)
