from itertools import product

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


def test_matches_independent_bruteforce_optimum_on_tiny_problem():
    problem = SynchronizationProblem(
        components=("a", "b", "c"),
        gauge_labels=(0, 1, 2),
        direct_observations=(
            DirectGaugeObservation("a", 0),
            DirectGaugeObservation("c", 2),
        ),
        germs=(
            CanonicalEdgeGerm("ab_plus", "a", "b", 1, "east"),
            CanonicalEdgeGerm("ab_zero", "a", "b", 0, "east"),
            CanonicalEdgeGerm("bc_plus", "b", "c", 1, "north"),
            CanonicalEdgeGerm("ac_zero", "a", "c", 0, "diagonal"),
        ),
        conflict_pairs=(("bc_plus", "ac_zero"),),
    )

    components = tuple(problem.components)
    germs = tuple(problem.germs)
    groups = {}
    for germ in germs:
        groups.setdefault(germ.ambiguity_group, []).append(germ)
    optimum = None
    optimal_assignments = set()
    for labels in product(problem.gauge_labels, repeat=len(components)):
        q = dict(zip(components, labels))
        for bits in product((0, 1), repeat=len(germs)):
            selected = frozenset(
                germ.germ_id for germ, bit in zip(germs, bits) if bit
            )
            if any(
                sum(germ.germ_id in selected for germ in group) > 1
                for group in groups.values()
            ):
                continue
            if any(left in selected and right in selected for left, right in problem.conflict_pairs):
                continue
            unary_cost = sum(q[obs.component] != obs.label for obs in problem.direct_observations)
            transport_cost = sum(
                germ.germ_id in selected
                and q[germ.target] - q[germ.source] != germ.transport
                for germ in germs
            )
            null_cost = 0.5 * sum(
                not any(germ.germ_id in selected for germ in group)
                for group in groups.values()
            )
            objective = unary_cost + transport_cost + null_cost
            assignment = (labels, selected)
            if optimum is None or objective < optimum:
                optimum = objective
                optimal_assignments = {assignment}
            elif objective == optimum:
                optimal_assignments.add(assignment)

    solution = _solve(problem)
    assert solution.objective_value == optimum
    assert (tuple(solution.q[component] for component in components), frozenset(solution.selected_germ_ids)) in optimal_assignments
    assert solution.hard_constraint_violations == 0


def test_germ_input_order_preserves_selected_relations_and_objective():
    germs = (
        CanonicalEdgeGerm("ab_plus", "a", "b", 1, "east"),
        CanonicalEdgeGerm("ab_zero", "a", "b", 0, "east"),
        CanonicalEdgeGerm("bc_plus", "b", "c", 1, "north"),
        CanonicalEdgeGerm("ac_zero", "a", "c", 0, "diagonal"),
    )
    kwargs = dict(
        components=("a", "b", "c"),
        gauge_labels=(0, 1, 2),
        direct_observations=(
            DirectGaugeObservation("a", 0),
            DirectGaugeObservation("c", 2),
        ),
        conflict_pairs=(("bc_plus", "ac_zero"),),
    )
    first = _solve(SynchronizationProblem(germs=germs, **kwargs))
    second = _solve(SynchronizationProblem(germs=tuple(reversed(germs)), **kwargs))
    assert frozenset(first.selected_germ_ids) == frozenset(second.selected_germ_ids)
    assert first.objective_value == second.objective_value
