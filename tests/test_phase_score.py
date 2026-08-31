from scroll_anchor.phase_score import section_local_phase_patch_score


def test_adjacent_equal_q_from_disconnected_sections_do_not_merge():
    result = section_local_phase_patch_score(
        {(0, 0): 0, (0, 1): 0},
        {(0, 0): "left", (0, 1): "right"},
        {(0, 0), (0, 1)},
    )
    assert result.largest_region_size == 1
    assert result.score == 0.5


def test_adjacent_equal_q_inside_one_section_merges():
    result = section_local_phase_patch_score(
        {(0, 0): 0, (0, 1): 0},
        {(0, 0): "section", (0, 1): "section"},
        {(0, 0), (0, 1)},
    )
    assert result.largest_region_size == 2
    assert result.score == 1.0
