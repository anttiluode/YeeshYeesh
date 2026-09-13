from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def build_gate_library(
    distances: np.ndarray,
    radius: int = 1,
    outside_keep: float = 0.35,
) -> list[np.ndarray]:
    distances = np.asarray(distances)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be square")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if not 0.0 <= outside_keep <= 1.0:
        raise ValueError("outside_keep must lie in [0, 1]")

    gates: list[np.ndarray] = []
    for center in range(distances.shape[0]):
        diagonal = np.where(distances[center] <= radius, 1.0, outside_keep)
        gates.append(np.diag(diagonal.astype(float)))
    return gates


def propagate(
    operator: np.ndarray,
    x0: np.ndarray,
    horizon: int,
    gate_times: Sequence[int] = (),
    schedule: Sequence[int] = (),
    gate_library: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    operator = np.asarray(operator)
    state = np.asarray(x0).copy()
    gate_times = tuple(int(t) for t in gate_times)
    schedule = tuple(int(c) for c in schedule)

    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be square")
    if state.shape != (operator.shape[0],):
        raise ValueError("x0 must match operator dimension")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    if len(gate_times) != len(schedule):
        raise ValueError("gate_times and schedule must have the same length")
    if gate_times != tuple(sorted(gate_times)) or len(set(gate_times)) != len(gate_times):
        raise ValueError("gate_times must be strictly increasing")
    if gate_times and (gate_times[0] < 1 or gate_times[-1] > horizon):
        raise ValueError("gate_times must lie within 1..horizon")
    if schedule and gate_library is None:
        raise ValueError("gate_library is required when schedule is nonempty")

    gate_index = 0
    for step in range(1, horizon + 1):
        state = operator @ state
        if gate_index < len(gate_times) and step == gate_times[gate_index]:
            center = schedule[gate_index]
            assert gate_library is not None
            if center < 0 or center >= len(gate_library):
                raise ValueError("schedule contains an invalid gate center")
            state = np.asarray(gate_library[center]) @ state
            gate_index += 1
    return state
