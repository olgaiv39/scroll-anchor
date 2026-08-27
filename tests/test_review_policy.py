from __future__ import annotations

import numpy as np

from scroll_anchor.config import ReviewConfig
from scroll_anchor.diagnostics import Diagnostics
from scroll_anchor.report import apply_review


def _diag():
    z = np.zeros((1, 4), dtype=np.float32)
    return Diagnostics(
        valid=np.ones((1, 4), dtype=bool),
        chosen_offset=z.copy(),
        drift_score=np.array([[4.0, 0.0, 4.0, 0.0]], dtype=np.float32),
        switch_score=np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32),
        geom_offset=z.copy(), margin=z.copy(), evidence=z.copy(), contrast=z.copy(),
        confidence=np.array([[0.9, 0.9, 0.9, 0.2]], dtype=np.float32),
        review=np.zeros((1, 4), dtype=bool), correction_offset=z.copy(),
        estimated_spacing=8.0,
    )


def test_default_review_excludes_drift_only_and_preserves_diagnostics():
    diag = _diag()
    drift_before = diag.drift_score.copy()
    switch_before = diag.switch_score.copy()

    apply_review(diag, ReviewConfig())

    # drift-only, switch-only, drift+switch, low-confidence-only
    assert diag.review.tolist() == [[False, True, True, True]]
    assert np.array_equal(diag.drift_score, drift_before)
    assert np.array_equal(diag.switch_score, switch_before)


def test_legacy_drift_review_policy_is_explicit_opt_in():
    diag = _diag()

    apply_review(diag, ReviewConfig(include_drift_in_review=True))

    assert diag.review.tolist() == [[True, True, True, True]]
