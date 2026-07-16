from __future__ import annotations

from scroll_anchor.config import RunConfig
from scroll_anchor.metrics import evaluate
from scroll_anchor.pipeline import analyze_surface
from scroll_anchor.report import apply_review
from scroll_anchor.synth import make_scene


def _run(enable_correction=False):
    cfg = RunConfig()
    cfg.correction.enabled = enable_correction
    scene = make_scene(H=64, W=64, seed=1)
    res = analyze_surface(scene.corrupt, scene.volume, cfg)
    apply_review(res.diagnostics, cfg.review)
    bench = evaluate(
        res.diagnostics, scene.gt, scene.sheet_model, scene.corrupt.points(),
        res.normals, res.profiles, res.offsets, cfg.diagnostics.drift_min,
    )
    return bench


def test_switch_detection_recall():
    b = _run()
    assert b.switch_detection["recall"] >= 0.8


def test_scroll_anchor_reduces_harmful_acceptance():
    b = _run()
    assert b.harmful_rate_scroll_anchor < b.harmful_rate_label_as_is
    assert b.harmful_rate_scroll_anchor <= 0.02


def test_clean_regions_stable():
    b = _run()
    assert b.clean_stability >= 0.9


def test_drift_displacement_recovered():
    b = _run()
    assert b.drift_displacement_mae <= 1.5
