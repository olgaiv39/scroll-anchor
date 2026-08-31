"""Section-local descriptive scoring for selected relative phase structure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Tuple


GridVertex = Tuple[int, int]


@dataclass(frozen=True)
class PhasePatchScore:
    """Threshold-free summary of selected non-modal phase structure."""

    active_vertex_count: int
    largest_region_size: int
    score: float


def section_local_phase_patch_score(
    vertex_labels: Mapping[GridVertex, int],
    vertex_sections: Mapping[GridVertex, Hashable],
    nonmodal_vertices: Iterable[GridVertex],
) -> PhasePatchScore:
    """Score same-q 4-neighbor non-modal regions within selected sections.

    ``vertex_labels`` must contain section-locally canonical q values and
    ``nonmodal_vertices`` must already have been determined from each section's
    modal q. Region connectivity requires equal section ID and equal q.
    """
    active = set(vertex_labels)
    if set(vertex_sections) != active:
        raise ValueError("vertex labels and section IDs must cover the same vertices")
    nonmodal = set(nonmodal_vertices)
    if not nonmodal.issubset(active):
        raise ValueError("non-modal vertices must be active vertices")

    largest = 0
    unseen = set(nonmodal)
    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 0
        q = vertex_labels[start]
        section = vertex_sections[start]
        while stack:
            vertex = stack.pop()
            size += 1
            for neighbor in (
                (vertex[0] - 1, vertex[1]),
                (vertex[0] + 1, vertex[1]),
                (vertex[0], vertex[1] - 1),
                (vertex[0], vertex[1] + 1),
            ):
                if (
                    neighbor in unseen
                    and vertex_labels[neighbor] == q
                    and vertex_sections[neighbor] == section
                ):
                    unseen.remove(neighbor)
                    stack.append(neighbor)

        largest = max(largest, size)

    return PhasePatchScore(
        active_vertex_count=len(active),
        largest_region_size=largest,
        score=largest / max(1, len(active)),
    )
