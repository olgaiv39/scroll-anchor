"""Frozen single-valued gauge synchronization via CPU CP-SAT.

This module intentionally accepts already-constructed concrete components and
canonical edge germs.  It does not generate candidates, infer components, or
turn geometric diagnostics into identity constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, Mapping, Sequence, Tuple


ComponentId = Hashable


@dataclass(frozen=True)
class DirectGaugeObservation:
    """One independent soft K-reference observation for a component."""

    component: ComponentId
    label: int


@dataclass(frozen=True)
class CanonicalEdgeGerm:
    """One canonical concrete relative-transport observation."""

    germ_id: str
    source: ComponentId
    target: ComponentId
    transport: int
    ambiguity_group: str
    # A canonical relation is seen from both of its directed surface interfaces.
    # Keep the original field for backwards-compatible callers; when supplied,
    # all entries here receive the same at-most-one constraint.
    ambiguity_groups: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SynchronizationProblem:
    """Inputs to the frozen q_i + z_e formulation.

    ``conflict_pairs`` contains only locally incompatible, concrete-endpoint
    selections.  Forward/reverse duplicates must already have been canonicalized
    by the caller into one ``CanonicalEdgeGerm``.
    """

    components: Sequence[ComponentId]
    gauge_labels: Sequence[int]
    direct_observations: Sequence[DirectGaugeObservation]
    germs: Sequence[CanonicalEdgeGerm]
    conflict_pairs: Sequence[Tuple[str, str]] = ()
    # Optional exact compression of conflict_pairs.  Each item names one cached
    # surface endpoint, one concrete component at that endpoint, and all germs
    # using that component there.  At most one component-use literal is allowed
    # per endpoint; germs in the same item may still coexist.
    endpoint_component_uses: Sequence[Tuple[Hashable, ComponentId, Sequence[str]]] = ()
    # Optional exact relative-gauge quotient groups.  Callers supply connected
    # q-components and the original symmetric half-width B.
    relative_gauge_components: Sequence[Tuple[Sequence[ComponentId], int]] = ()


@dataclass(frozen=True)
class SynchronizationSolution:
    """A feasible CP-SAT solution and auditable frozen-objective statistics."""

    status: str
    objective_value: float
    best_objective_bound: float
    wall_time_seconds: float
    q: Mapping[ComponentId, int]
    selected_germ_ids: Tuple[str, ...]
    unary_mismatch_count: int
    selected_transport_mismatch_count: int
    null_group_count: int
    group_exclusivity_violations: int
    endpoint_conflict_violations: int

    @property
    def hard_constraint_violations(self) -> int:
        return self.group_exclusivity_violations + self.endpoint_conflict_violations


def _validate(problem: SynchronizationProblem) -> None:
    components = tuple(problem.components)
    labels = tuple(problem.gauge_labels)
    if not components or len(set(components)) != len(components):
        raise ValueError("components must be nonempty and unique")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("gauge_labels must be nonempty and unique")
    component_set = set(components)
    germ_ids = set()
    for observation in problem.direct_observations:
        if observation.component not in component_set:
            raise ValueError("direct observation refers to an unknown component")
    for germ in problem.germs:
        if germ.germ_id in germ_ids:
            raise ValueError("germ_id values must be unique")
        germ_ids.add(germ.germ_id)
        if germ.source not in component_set or germ.target not in component_set:
            raise ValueError("germ refers to an unknown component")
    for left, right in problem.conflict_pairs:
        if left == right or left not in germ_ids or right not in germ_ids:
            raise ValueError("conflict pairs must name two distinct known germs")
    use_keys = set()
    for endpoint, component, use_germs in problem.endpoint_component_uses:
        if component not in component_set:
            raise ValueError("endpoint component use refers to an unknown component")
        key = (endpoint, component)
        if key in use_keys:
            raise ValueError("endpoint component uses must be unique")
        use_keys.add(key)
        if not use_germs or any(germ_id not in germ_ids for germ_id in use_germs):
            raise ValueError("endpoint component uses must name known germs")


def solve_synchronization(
    problem: SynchronizationProblem,
    *,
    max_time_seconds: float = 1800.0,
) -> SynchronizationSolution:
    """Solve the frozen formulation with deterministic, single-worker CP-SAT.

    The integer objective is exactly twice the frozen objective:
    ``2*unary_mismatches + 2*selected_transport_mismatches + null_groups``.
    """
    _validate(problem)
    if max_time_seconds <= 0:
        raise ValueError("max_time_seconds must be positive")
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "CP-SAT requires the optional synchronization dependency: "
            "pip install -e '.[sync]'"
        ) from exc

    components = tuple(problem.components)
    labels = tuple(sorted(problem.gauge_labels))
    germs = tuple(problem.germs)
    groups: Dict[str, list] = {}
    for germ in germs:
        memberships = germ.ambiguity_groups or (germ.ambiguity_group,)
        for group in memberships:
            groups.setdefault(group, []).append(germ)

    model = cp_model.CpModel()
    contiguous = labels == tuple(range(labels[0], labels[-1] + 1))
    domain = None if contiguous else cp_model.Domain.FromValues(labels)
    q = {
        component: (
            model.NewIntVar(labels[0], labels[-1], "q_%d" % index)
            if contiguous else model.NewIntVarFromDomain(domain, "q_%d" % index)
        )
        for index, component in enumerate(components)
    }
    for index, (members, original_bound) in enumerate(problem.relative_gauge_components):
        members = tuple(members)
        if not members:
            continue
        model.Add(q[min(members)] == 0)
        q_min = model.NewIntVar(-2 * original_bound, 2 * original_bound, "gauge_min_%d" % index)
        q_max = model.NewIntVar(-2 * original_bound, 2 * original_bound, "gauge_max_%d" % index)
        model.AddMinEquality(q_min, [q[item] for item in members])
        model.AddMaxEquality(q_max, [q[item] for item in members])
        model.Add(q_max - q_min <= 2 * original_bound)
    z = {germ.germ_id: model.NewBoolVar("z_%d" % index)
         for index, germ in enumerate(germs)}
    if problem.relative_gauge_components:
        for value in q.values():
            model.AddHint(value, 0)
        for value in z.values():
            model.AddHint(value, 0)

    for group_germs in groups.values():
        model.Add(sum(z[germ.germ_id] for germ in group_germs) <= 1)
    if problem.endpoint_component_uses:
        uses_by_endpoint: Dict[Hashable, list] = {}
        for index, (endpoint, _component, use_germs) in enumerate(problem.endpoint_component_uses):
            use = model.NewBoolVar("endpoint_use_%d" % index)
            uses_by_endpoint.setdefault(endpoint, []).append(use)
            for germ_id in use_germs:
                model.AddImplication(z[germ_id], use)
        for endpoint_uses in uses_by_endpoint.values():
            if len(endpoint_uses) > 1:
                model.AddAtMostOne(endpoint_uses)
    else:
        for left, right in problem.conflict_pairs:
            model.Add(z[left] + z[right] <= 1)

    unary_mismatch = []
    for index, observation in enumerate(problem.direct_observations):
        mismatch = model.NewBoolVar("unary_mismatch_%d" % index)
        model.Add(q[observation.component] != observation.label).OnlyEnforceIf(mismatch)
        model.Add(q[observation.component] == observation.label).OnlyEnforceIf(
            mismatch.Not()
        )
        unary_mismatch.append(mismatch)

    selected_transport_mismatch = []
    for index, germ in enumerate(germs):
        relation_match = model.NewBoolVar("relation_match_%d" % index)
        model.Add(q[germ.target] - q[germ.source] == germ.transport).OnlyEnforceIf(
            relation_match
        )
        model.Add(q[germ.target] - q[germ.source] != germ.transport).OnlyEnforceIf(
            relation_match.Not()
        )
        mismatch = model.NewBoolVar("selected_transport_mismatch_%d" % index)
        model.AddImplication(mismatch, z[germ.germ_id])
        model.AddImplication(mismatch, relation_match.Not())
        model.AddBoolOr([z[germ.germ_id].Not(), relation_match, mismatch])
        model.AddImplication(z[germ.germ_id].Not(), mismatch.Not())
        selected_transport_mismatch.append(mismatch)

    # This is twice the frozen objective.  Each group contributes a constant
    # one minus its selected germ, so the constant can be restored in reporting.
    model.Minimize(
        2 * sum(unary_mismatch)
        + 2 * sum(selected_transport_mismatch)
        - sum(z.values())
    )

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = max_time_seconds
    status_code = solver.Solve(model)
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT did not return a feasible solution: %s" % solver.StatusName(status_code))

    q_values = {component: solver.Value(variable) for component, variable in q.items()}
    selected = tuple(germ.germ_id for germ in germs if solver.Value(z[germ.germ_id]))
    selected_set = set(selected)
    group_violations = sum(
        1 for group_germs in groups.values()
        if sum(germ.germ_id in selected_set for germ in group_germs) > 1
    )
    conflict_violations = sum(
        1 for left, right in problem.conflict_pairs
        if left in selected_set and right in selected_set
    )
    unary_count = sum(solver.Value(value) for value in unary_mismatch)
    transport_count = sum(solver.Value(value) for value in selected_transport_mismatch)
    null_groups = sum(
        1 for group_germs in groups.values()
        if not any(germ.germ_id in selected_set for germ in group_germs)
    )
    objective = unary_count + transport_count + 0.5 * null_groups
    return SynchronizationSolution(
        status=solver.StatusName(status_code),
        objective_value=float(objective),
        best_objective_bound=(solver.BestObjectiveBound() + len(groups)) / 2.0,
        wall_time_seconds=solver.WallTime(),
        q=q_values,
        selected_germ_ids=selected,
        unary_mismatch_count=unary_count,
        selected_transport_mismatch_count=transport_count,
        null_group_count=null_groups,
        group_exclusivity_violations=group_violations,
        endpoint_conflict_violations=conflict_violations,
    )
