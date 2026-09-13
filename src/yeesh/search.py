from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .gates import propagate
from .metrics import mean_route_metrics, route_score


@dataclass(frozen=True)
class SearchResult:
    schedule: tuple[int, ...]
    score: float
    metrics: dict[str, float]


def _evaluate_schedule(
    operator: np.ndarray,
    examples: Sequence[np.ndarray],
    horizon: int,
    gate_times: Sequence[int],
    schedule: Sequence[int],
    gate_library: Sequence[np.ndarray],
    target_nodes: Sequence[int],
    competitor_nodes: Sequence[int],
) -> SearchResult:
    partial_times = tuple(gate_times[: len(schedule)])
    schedule = tuple(int(center) for center in schedule)
    states = [
        propagate(
            operator,
            example,
            horizon=horizon,
            gate_times=partial_times,
            schedule=schedule,
            gate_library=gate_library,
        )
        for example in examples
    ]
    metrics = mean_route_metrics(states, target_nodes, competitor_nodes)
    return SearchResult(schedule=schedule, score=route_score(metrics), metrics=metrics)


def beam_search(
    operator: np.ndarray,
    examples: Sequence[np.ndarray],
    horizon: int,
    gate_times: Sequence[int],
    gate_library: Sequence[np.ndarray],
    target_nodes: Sequence[int],
    competitor_nodes: Sequence[int],
    beam_width: int = 64,
) -> SearchResult:
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if not examples:
        raise ValueError("examples must not be empty")
    if not gate_times:
        return _evaluate_schedule(
            operator,
            examples,
            horizon,
            gate_times,
            (),
            gate_library,
            target_nodes,
            competitor_nodes,
        )

    beam = [
        _evaluate_schedule(
            operator,
            examples,
            horizon,
            gate_times,
            (),
            gate_library,
            target_nodes,
            competitor_nodes,
        )
    ]
    for _depth in range(len(gate_times)):
        candidates: list[SearchResult] = []
        for item in beam:
            for center in range(len(gate_library)):
                candidates.append(
                    _evaluate_schedule(
                        operator,
                        examples,
                        horizon,
                        gate_times,
                        item.schedule + (center,),
                        gate_library,
                        target_nodes,
                        competitor_nodes,
                    )
                )
        candidates.sort(key=lambda item: (-item.score, item.schedule))
        beam = candidates[:beam_width]
    return beam[0]


def best_fixed_schedule(
    operator: np.ndarray,
    examples: Sequence[np.ndarray],
    horizon: int,
    gate_times: Sequence[int],
    gate_library: Sequence[np.ndarray],
    target_nodes: Sequence[int],
    competitor_nodes: Sequence[int],
    beam_width: int | None = None,
) -> SearchResult:
    del beam_width
    if not gate_times:
        return _evaluate_schedule(
            operator,
            examples,
            horizon,
            gate_times,
            (),
            gate_library,
            target_nodes,
            competitor_nodes,
        )
    candidates = [
        _evaluate_schedule(
            operator,
            examples,
            horizon,
            gate_times,
            (center,) * len(gate_times),
            gate_library,
            target_nodes,
            competitor_nodes,
        )
        for center in range(len(gate_library))
    ]
    candidates.sort(key=lambda item: (-item.score, item.schedule))
    return candidates[0]
