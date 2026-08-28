from __future__ import annotations

import numpy as np

from scroll_anchor.config import ReviewConfig, RunConfig
from scroll_anchor.diagnostics import Diagnostics
from scroll_anchor.pipeline import analyze_surface
from scroll_anchor.report import apply_review, build_review_regions, write_reports
from scroll_anchor.synth import make_scene
from scroll_anchor.tifxyz import Surface


def _diag():
    z = np.zeros((1, 4), dtype=np.float32)
    return Diagnostics(
        valid=np.ones((1, 4), dtype=bool),
        chosen_offset=z.copy(),
        drift_score=np.array([[4.0, 0.0, 4.0, 0.0]], dtype=np.float32),
        switch_score=np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32),
        geom_offset=z.copy(), margin=z.copy(), evidence=z.copy(), contrast=z.copy(),
        confidence=np.array([[0.9, 0.9, 0.2, 0.2]], dtype=np.float32),
        review=np.zeros((1, 4), dtype=bool), correction_offset=z.copy(),
        review_low_confidence=np.zeros((1, 4), dtype=bool),
        review_switch=np.zeros((1, 4), dtype=bool),
        review_drift=np.zeros((1, 4), dtype=bool),
        estimated_spacing=8.0,
    )


def test_default_review_excludes_drift_only_and_preserves_diagnostics():
    diag = _diag()
    drift_before = diag.drift_score.copy()
    switch_before = diag.switch_score.copy()
    confidence_before = diag.confidence.copy()

    apply_review(diag, ReviewConfig())

    # drift-only, switch-only, low-confidence+switch, low-confidence-only
    assert diag.review.tolist() == [[False, True, True, True]]
    assert diag.review_low_confidence.tolist() == [[False, False, True, True]]
    assert diag.review_switch.tolist() == [[False, True, True, False]]
    assert not diag.review_drift.any()
    assert np.array_equal(diag.review, diag.review_low_confidence | diag.review_switch)
    assert np.array_equal(diag.drift_score, drift_before)
    assert np.array_equal(diag.switch_score, switch_before)
    assert np.array_equal(diag.confidence, confidence_before)


def test_legacy_drift_review_policy_is_explicit_opt_in():
    diag = _diag()

    apply_review(diag, ReviewConfig(include_drift_in_review=True))

    assert diag.review.tolist() == [[True, True, True, True]]
    assert diag.review_drift.tolist() == [[True, False, True, False]]
    assert np.array_equal(
        diag.review,
        diag.review_low_confidence | diag.review_switch | diag.review_drift,
    )


def test_review_cause_arrays_are_serialized_and_region_counts_match(tmp_path):
    diag = _diag()
    cfg = ReviewConfig(min_region_vertices=1)
    apply_review(diag, cfg)
    surface = Surface(
        x=np.zeros((1, 4), dtype=np.float32),
        y=np.zeros((1, 4), dtype=np.float32),
        z=np.ones((1, 4), dtype=np.float32),
        valid=diag.valid,
    )
    regions = build_review_regions(diag, cfg)

    write_reports(str(tmp_path), surface, diag, RunConfig(review=cfg), regions,
                  write_channels=False)

    for name, expected in {
        "review": diag.review,
        "review_low_confidence": diag.review_low_confidence,
        "review_switch": diag.review_switch,
        "review_drift": diag.review_drift,
    }.items():
        actual = np.load(tmp_path / "arrays" / (name + ".npy"))
        assert np.array_equal(actual, expected.astype(np.uint8))
    assert sum(r["review_switch_count"] for r in regions) == 2
    assert sum(r["review_low_confidence_count"] for r in regions) == 2


def test_pipeline_review_causes_match_report_recomputation():
    scene = make_scene(H=24, W=24, seed=7)
    cfg = RunConfig()
    diag = analyze_surface(scene.corrupt, scene.volume, cfg).diagnostics
    expected = (
        diag.review.copy(),
        diag.review_low_confidence.copy(),
        diag.review_switch.copy(),
        diag.review_drift.copy(),
    )
    raw_diagnostics = (
        diag.drift_score.copy(),
        diag.confidence.copy(),
        diag.switch_score.copy(),
    )

    apply_review(diag, cfg.review)

    assert all(np.array_equal(actual, recomputed) for actual, recomputed in zip(
        expected,
        (diag.review, diag.review_low_confidence, diag.review_switch, diag.review_drift),
    ))
    assert all(np.array_equal(before, after) for before, after in zip(
        raw_diagnostics,
        (diag.drift_score, diag.confidence, diag.switch_score),
    ))
