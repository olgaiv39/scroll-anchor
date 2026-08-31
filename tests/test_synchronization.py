from scroll_anchor.synchronization import (
    CanonicalEdgeGerm,
    DirectGaugeObservation,
    SynchronizationProblem,
    solve_synchronization,
)


def _solve(problem):
    return solve_synchronization(problem, max_time_seconds=5.0)


def test_ambiguity_group_exclusivity_and_null_feasibility():
    problem = SynchronizationProblem(
        components=("a", "b"),
        gauge_labels=(0, 1, 2),
        direct_observations=(),
        germs=(
            CanonicalEdgeGerm("ab_plus", "a", "b", 1, "east"),
            CanonicalEdgeGerm("ab_minus", "a", "b", -1, "east"),
        ),
    )
    solution = _solve(problem)
    assert len(solution.selected_germ_ids) == 1
    assert solution.null_group_count == 0
    assert solution.hard_constraint_violations == 0


def test_incompatible_endpoints_are_never_simultaneously_selected():
    problem = SynchronizationProblem(
        components=("a", "b", "c"),
        gauge_labels=(0, 1, 2),
        direct_observations=(DirectGaugeObservation("a", 0),),
        germs=(
            CanonicalEdgeGerm("ab", "a", "b", 1, "east"),
            CanonicalEdgeGerm("ac", "a", "c", 1, "north"),
        ),
        conflict_pairs=(("ab", "ac"),),
    )
    solution = _solve(problem)
    assert len(solution.selected_germ_ids) == 1
    assert solution.hard_constraint_violations == 0


def test_null_is_feasible_when_no_germ_can_match_domain():
    problem = SynchronizationProblem(
        components=("a", "b"),
        gauge_labels=(0, 1),
        direct_observations=(),
        germs=(CanonicalEdgeGerm("impossible", "a", "b", 7, "east"),),
    )
    solution = _solve(problem)
    assert solution.selected_germ_ids == ()
    assert solution.null_group_count == 1
    assert solution.objective_value == 0.5
    assert solution.hard_constraint_violations == 0


def test_compatible_selected_relation_beats_all_null():
    problem = SynchronizationProblem(
        components=("a", "b"),
        gauge_labels=(0, 1),
        direct_observations=(),
        germs=(CanonicalEdgeGerm("compatible", "a", "b", 1, "east"),),
    )
    solution = _solve(problem)
    assert solution.selected_germ_ids == ("compatible",)
    assert solution.objective_value == 0.0


def test_deterministic_solution_and_zero_hard_constraint_violations():
    problem = SynchronizationProblem(
        components=("a", "b", "c"),
        gauge_labels=(0, 1, 2),
        direct_observations=(DirectGaugeObservation("a", 0),),
        germs=(
            CanonicalEdgeGerm("ab", "a", "b", 1, "east"),
            CanonicalEdgeGerm("bc", "b", "c", 1, "east_2"),
            CanonicalEdgeGerm("ac_wrong", "a", "c", 0, "north"),
        ),
        conflict_pairs=(("bc", "ac_wrong"),),
    )
    first = _solve(problem)
    second = _solve(problem)
    assert first.q == second.q
    assert first.selected_germ_ids == second.selected_germ_ids
    assert first.objective_value == second.objective_value
    assert first.hard_constraint_violations == 0
