"""Focused tests for render-report candidate captions

Synthetic regions only: no render, no tifxyz, no CT, no detection. These cover how
the two stored diagnostic sets are displayed, and confirm the caption never alters
the raw detector score or the candidate ordering.
"""

from scroll_anchor.render_report import _panel_title


def _region(**extra):
    """Minimal candidate record with the fields every caption already required"""
    reg = {
        "id": 566,
        "score": 0.178003,
        "direction": "vertical",
        "centroid_rowcol_jpg": [3504.0, 21903.0],
        "displacement_jpg_pixels": 12.0,
    }
    reg.update(extra)
    return reg


RENDER_BG = {
    "render_background_overlap_fraction": 0.929032,
    "render_background_distance_px": 0.0,
    "touches_render_background": True,
}
TIFXYZ = {
    "invalid_overlap_fraction": 0.0,
    "boundary_distance_px": 206.342,
    "touches_invalid_boundary": False,
}


def _lines(reg, rank=1):
    return _panel_title(reg, rank).split("\n")


# 1. no diagnostics -------------------------------------------------------------
def test_no_diagnostics_keeps_legacy_caption():
    lines = _lines(_region())
    assert lines == [
        "#1  id 566  score 0.178",
        "vertical   2D shift 12 px",
        "JPG row 3504, col 21903",
    ]
    # No placeholder line is invented for absent diagnostics.
    assert "render-bg" not in "\n".join(lines)
    assert "tifxyz-invalid" not in "\n".join(lines)


# 2. render-background only -----------------------------------------------------
def test_render_background_only():
    lines = _lines(_region(**RENDER_BG))
    assert len(lines) == 4
    assert lines[3] == "render-bg overlap 0.929 | distance 0.0 px | touches yes"
    assert not any("tifxyz-invalid" in ln for ln in lines)


# 3. tifxyz only ----------------------------------------------------------------
def test_tifxyz_only():
    lines = _lines(_region(**TIFXYZ))
    assert len(lines) == 4
    assert lines[3] == (
        "tifxyz-invalid overlap 0.000 | boundary distance 206.3 px | touches no"
    )
    assert not any("render-bg" in ln for ln in lines)


# 4. both sets ------------------------------------------------------------------
def test_both_sets_are_separate_lines():
    lines = _lines(_region(**RENDER_BG, **TIFXYZ))
    assert len(lines) == 5
    assert lines[3] == "render-bg overlap 0.929 | distance 0.0 px | touches yes"
    assert lines[4] == (
        "tifxyz-invalid overlap 0.000 | boundary distance 206.3 px | touches no"
    )
    # The two diagnostics stay semantically separate: never summed, averaged or
    # combined into a single agreement figure.
    assert lines[3] != lines[4]


# 5. zero values ----------------------------------------------------------------
def test_zero_and_false_are_present_data_not_absence():
    zeros = {
        "render_background_overlap_fraction": 0.0,
        "render_background_distance_px": 0.0,
        "touches_render_background": False,
        "invalid_overlap_fraction": 0.0,
        "boundary_distance_px": 0.0,
        "touches_invalid_boundary": False,
    }
    lines = _lines(_region(**zeros))
    assert lines[3] == "render-bg overlap 0.000 | distance 0.0 px | touches no"
    assert lines[4] == (
        "tifxyz-invalid overlap 0.000 | boundary distance 0.0 px | touches no"
    )


# 6. null distance --------------------------------------------------------------
def test_null_distance_reads_na_when_set_is_otherwise_present():
    reg = _region(
        render_background_overlap_fraction=0.25,
        render_background_distance_px=None,
        touches_render_background=False,
        invalid_overlap_fraction=0.5,
        boundary_distance_px=float("inf"),
        touches_invalid_boundary=True,
    )
    lines = _lines(reg)
    # A missing distance does not suppress the line: the rest of the set is real.
    assert lines[3] == "render-bg overlap 0.250 | distance n/a | touches no"
    # Non-finite distances are reported as n/a too, never as 'inf px'.
    assert lines[4] == (
        "tifxyz-invalid overlap 0.500 | boundary distance n/a | touches yes"
    )


# 7. unchanged ordering and raw score -------------------------------------------
def test_raw_score_and_ordering_unchanged():
    regs = [
        _region(id=215, score=0.181901, **RENDER_BG),
        _region(id=566, score=0.178003),
        _region(id=690, score=0.168743, **TIFXYZ),
    ]
    titles = [_panel_title(r, i + 1) for i, r in enumerate(regs)]

    # Ranks follow input order; the caption does not sort, filter or renumber.
    assert [t.split("\n")[0] for t in titles] == [
        "#1  id 215  score 0.182",
        "#2  id 566  score 0.178",
        "#3  id 690  score 0.169",
    ]

    # The displayed score is the raw stored score, not an adjusted one.
    for reg, title in zip(regs, titles):
        assert f"score {float(reg['score']):.3f}" in title.split("\n")[0]

    # Adding diagnostics never changes the score/identity line of a candidate.
    plain = _panel_title(_region(id=215, score=0.181901), 1).split("\n")[0]
    assert plain == titles[0].split("\n")[0]
