from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .gates import build_gate_library, propagate
from .metrics import effective_operator, mean_route_metrics, route_score
from .search import beam_search, best_fixed_schedule
from .substrate import build_y_graph, diffusion_operator, wave_operator


@dataclass(frozen=True)
class ExperimentConfig:
    trunk_length: int = 5
    arm_length: int = 4
    wave_dt: float = 0.75
    diffusion_dt: float = 0.18
    diffusion_kappa: float = 0.9
    diffusion_leak: float = 0.02
    horizon: int = 24
    gate_times: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20)
    gate_radius: int = 1
    outside_keep: float = 0.35
    beam_width: int = 64
    train_amplitudes: tuple[float, ...] = (0.8, 1.0, 1.2)
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_noise_std: float = 0.01
    heldout_amplitudes: tuple[float, ...] = (0.65, 0.95, 1.35, 1.6)
    heldout_seeds: tuple[int, ...] = (100, 101, 102, 103)
    heldout_noise_std: float = 0.015
    random_repeats: int = 128
    random_seed: int = 12345


def make_examples(
    node_count: int,
    source: int,
    amplitudes: Sequence[float],
    seeds: Sequence[int],
    noise_std: float,
) -> list[np.ndarray]:
    if len(amplitudes) != len(seeds):
        raise ValueError("amplitudes and seeds must have the same length")
    if noise_std < 0:
        raise ValueError("noise_std must be nonnegative")
    examples: list[np.ndarray] = []
    for amplitude, seed in zip(amplitudes, seeds):
        rng = np.random.default_rng(seed)
        state = rng.normal(0.0, noise_std, size=node_count).astype(float)
        state[source] += float(amplitude)
        examples.append(state)
    return examples


def _evaluate_schedule(
    operator: np.ndarray,
    examples: Sequence[np.ndarray],
    horizon: int,
    gate_times: Sequence[int],
    schedule: Sequence[int],
    gate_library: Sequence[np.ndarray],
    target_nodes: Sequence[int],
    competitor_nodes: Sequence[int],
) -> dict[str, float]:
    states = [
        propagate(
            operator,
            example,
            horizon=horizon,
            gate_times=gate_times,
            schedule=schedule,
            gate_library=gate_library,
        )
        for example in examples
    ]
    metrics = mean_route_metrics(states, target_nodes, competitor_nodes)
    metrics["score"] = route_score(metrics)
    return metrics


def _metric_summary(rows: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("rows must not be empty")
    keys = tuple(rows[0].keys())
    data = {key: np.asarray([row[key] for row in rows], dtype=float) for key in keys}
    return {
        "median": {key: float(np.median(values)) for key, values in data.items()},
        "q90": {key: float(np.quantile(values, 0.90)) for key, values in data.items()},
        "max": {key: float(np.max(values)) for key, values in data.items()},
    }


def _operator_diagnostics(
    operator: np.ndarray,
    horizon: int,
    gate_times: Sequence[int],
    schedule: Sequence[int],
    gate_library: Sequence[np.ndarray],
) -> dict[str, Any]:
    free = np.linalg.matrix_power(operator, horizon)
    controlled = effective_operator(
        operator,
        horizon=horizon,
        gate_times=gate_times,
        schedule=schedule,
        gate_library=gate_library,
    )
    free_singular = np.linalg.svd(free, compute_uv=False)
    controlled_singular = np.linalg.svd(controlled, compute_uv=False)
    return {
        "frobenius_difference": float(np.linalg.norm(controlled - free, ord="fro")),
        "free_spectral_norm": float(free_singular[0]),
        "controlled_spectral_norm": float(controlled_singular[0]),
        "free_singular_values": [float(x) for x in free_singular[:8]],
        "controlled_singular_values": [float(x) for x in controlled_singular[:8]],
    }


def _run_condition(
    name: str,
    operator: np.ndarray,
    train_examples: Sequence[np.ndarray],
    heldout_examples: Sequence[np.ndarray],
    horizon: int,
    gate_times: Sequence[int],
    gate_library: Sequence[np.ndarray],
    target_nodes: Sequence[int],
    competitor_nodes: Sequence[int],
    beam_width: int,
    random_repeats: int,
    random_seed: int,
) -> dict[str, Any]:
    if random_repeats < 1:
        raise ValueError("random_repeats must be at least 1")

    optimized = beam_search(
        operator=operator,
        examples=train_examples,
        horizon=horizon,
        gate_times=gate_times,
        gate_library=gate_library,
        target_nodes=target_nodes,
        competitor_nodes=competitor_nodes,
        beam_width=beam_width,
    )
    fixed = best_fixed_schedule(
        operator=operator,
        examples=train_examples,
        horizon=horizon,
        gate_times=gate_times,
        gate_library=gate_library,
        target_nodes=target_nodes,
        competitor_nodes=competitor_nodes,
    )

    empty_times: tuple[int, ...] = ()
    empty_schedule: tuple[int, ...] = ()
    reverse_schedule = tuple(reversed(optimized.schedule))

    train = {
        "free": _evaluate_schedule(operator, train_examples, horizon, empty_times, empty_schedule, gate_library, target_nodes, competitor_nodes),
        "optimized": _evaluate_schedule(operator, train_examples, horizon, gate_times, optimized.schedule, gate_library, target_nodes, competitor_nodes),
        "fixed": _evaluate_schedule(operator, train_examples, horizon, gate_times, fixed.schedule, gate_library, target_nodes, competitor_nodes),
    }

    heldout_base = {
        "free": _evaluate_schedule(operator, heldout_examples, horizon, empty_times, empty_schedule, gate_library, target_nodes, competitor_nodes),
        "optimized": _evaluate_schedule(operator, heldout_examples, horizon, gate_times, optimized.schedule, gate_library, target_nodes, competitor_nodes),
        "fixed": _evaluate_schedule(operator, heldout_examples, horizon, gate_times, fixed.schedule, gate_library, target_nodes, competitor_nodes),
        "reverse": _evaluate_schedule(operator, heldout_examples, horizon, gate_times, reverse_schedule, gate_library, target_nodes, competitor_nodes),
    }

    rng = np.random.default_rng(random_seed)
    random_location_rows: list[dict[str, float]] = []
    random_order_rows: list[dict[str, float]] = []
    for _ in range(random_repeats):
        random_schedule = tuple(int(x) for x in rng.integers(0, len(gate_library), size=len(gate_times)))
        random_location_rows.append(_evaluate_schedule(operator, heldout_examples, horizon, gate_times, random_schedule, gate_library, target_nodes, competitor_nodes))

        shuffled = list(optimized.schedule)
        rng.shuffle(shuffled)
        random_order_rows.append(_evaluate_schedule(operator, heldout_examples, horizon, gate_times, tuple(shuffled), gate_library, target_nodes, competitor_nodes))

    random_location = _metric_summary(random_location_rows)
    random_order = _metric_summary(random_order_rows)
    heldout = dict(heldout_base)
    heldout["random_location_median"] = random_location["median"]
    heldout["random_order_median"] = random_order["median"]

    free_target = heldout["free"]["target_energy"]
    optimized_target = heldout["optimized"]["target_energy"]
    free_selectivity = heldout["free"]["selectivity"]
    optimized_selectivity = heldout["optimized"]["selectivity"]

    return {
        "name": name,
        "optimized_schedule": list(optimized.schedule),
        "fixed_schedule": list(fixed.schedule),
        "reverse_schedule": list(reverse_schedule),
        "training": train,
        "heldout": heldout,
        "random_location": {"repeats": random_repeats, **random_location},
        "random_order": {"repeats": random_repeats, **random_order},
        "operator_diagnostics": _operator_diagnostics(operator, horizon, gate_times, optimized.schedule, gate_library),
        "headline": {
            "target_energy_gain_over_free": float(optimized_target / (free_target + 1e-30)),
            "selectivity_gain_over_free": float(optimized_selectivity / (free_selectivity + 1e-30)),
            "beats_free_target_energy": bool(optimized_target > free_target),
            "beats_free_selectivity": bool(optimized_selectivity > free_selectivity),
        },
    }


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or ExperimentConfig()
    graph = build_y_graph(config.trunk_length, config.arm_length)
    gates = build_gate_library(graph.distances, radius=config.gate_radius, outside_keep=config.outside_keep)
    train_examples = make_examples(graph.adjacency.shape[0], graph.source, config.train_amplitudes, config.train_seeds, config.train_noise_std)
    heldout_examples = make_examples(graph.adjacency.shape[0], graph.source, config.heldout_amplitudes, config.heldout_seeds, config.heldout_noise_std)

    wave = wave_operator(graph.adjacency, config.wave_dt)
    diffusion = diffusion_operator(graph.adjacency, dt=config.diffusion_dt, kappa=config.diffusion_kappa, leak=config.diffusion_leak)

    common = dict(
        train_examples=train_examples,
        heldout_examples=heldout_examples,
        horizon=config.horizon,
        gate_times=config.gate_times,
        gate_library=gates,
        target_nodes=graph.target_left,
        competitor_nodes=graph.target_right,
        beam_width=config.beam_width,
        random_repeats=config.random_repeats,
        random_seed=config.random_seed,
    )
    return {
        "config": asdict(config),
        "graph": {
            "node_count": int(graph.adjacency.shape[0]),
            "source": graph.source,
            "left_arm": list(graph.left_arm),
            "right_arm": list(graph.right_arm),
            "target_left": list(graph.target_left),
            "target_right": list(graph.target_right),
        },
        "wave": _run_condition("wave", operator=wave, **common),
        "diffusion": _run_condition("diffusion", operator=diffusion, **common),
    }
