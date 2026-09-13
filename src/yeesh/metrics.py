from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .gates import propagate


def route_metrics(
    state: np.ndarray,
    target_nodes: Sequence[int],
    competitor_nodes: Sequence[int],
    eps: float = 1e-12,
) -> dict[str, float]:
    state = np.asarray(state)
    target_energy = float(np.sum(np.abs(state[list(target_nodes)]) ** 2))
    competitor_energy = float(np.sum(np.abs(state[list(competitor_nodes)]) ** 2))
    total_energy = float(np.sum(np.abs(state) ** 2))
    selectivity = target_energy / (target_energy + competitor_energy + eps)
    return {
        "target_energy": target_energy,
        "competitor_energy": competitor_energy,
        "total_energy": total_energy,
        "selectivity": float(selectivity),
    }


def route_score(metrics: dict[str, float]) -> float:
    return float(metrics["target_energy"] * (0.5 + metrics["selectivity"]))


def effective_operator(
    operator: np.ndarray,
    horizon: int,
    gate_times: Sequence[int],
    schedule: Sequence[int],
    gate_library: Sequence[np.ndarray],
) -> np.ndarray:
    operator = np.asarray(operator)
    dimension = operator.shape[0]
    dtype = complex if np.iscomplexobj(operator) else float
    result = np.zeros((dimension, dimension), dtype=dtype)
    for column in range(dimension):
        basis = np.zeros(dimension, dtype=dtype)
        basis[column] = 1.0
        result[:, column] = propagate(
            operator,
            basis,
            horizon=horizon,
            gate_times=gate_times,
            schedule=schedule,
            gate_library=gate_library,
        )
    return result


def mean_route_metrics(
    states: Sequence[np.ndarray],
    target_nodes: Sequence[int],
    competitor_nodes: Sequence[int],
) -> dict[str, float]:
    if not states:
        raise ValueError("states must not be empty")
    rows = [route_metrics(state, target_nodes, competitor_nodes) for state in states]
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
