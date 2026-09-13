import numpy as np
import pytest

from yeesh.gates import build_gate_library
from yeesh.metrics import effective_operator, route_metrics, route_score
from yeesh.search import beam_search, best_fixed_schedule
from yeesh.substrate import build_y_graph, wave_operator


@pytest.fixture
def problem():
    graph = build_y_graph(5, 4)
    operator = wave_operator(graph.adjacency, dt=0.75)
    gates = build_gate_library(graph.distances, radius=1, outside_keep=0.35)
    examples = []
    for amplitude, seed in zip((0.8, 1.0, 1.2), (0, 1, 2)):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 0.01, size=operator.shape[0]).astype(complex)
        x[graph.source] += amplitude
        examples.append(x)
    return {
        "operator": operator,
        "examples": examples,
        "horizon": 24,
        "gate_times": (2, 5, 8, 11, 14, 17, 20),
        "gate_library": gates,
        "target_nodes": graph.target_left,
        "competitor_nodes": graph.target_right,
        "beam_width": 32,
    }


def test_route_metrics_report_energy_and_selectivity():
    state = np.array([1 + 0j, 2 + 0j, 1j, 0j])
    metrics = route_metrics(state, target_nodes=(0, 1), competitor_nodes=(2, 3))
    assert metrics["target_energy"] == pytest.approx(5.0)
    assert metrics["competitor_energy"] == pytest.approx(1.0)
    assert metrics["total_energy"] == pytest.approx(6.0)
    assert metrics["selectivity"] == pytest.approx(5.0 / 6.0)
    assert route_score(metrics) == pytest.approx(5.0 * (0.5 + 5.0 / 6.0))


def test_beam_search_is_deterministic(problem):
    a = beam_search(**problem)
    b = beam_search(**problem)
    assert a.schedule == b.schedule
    assert a.score == pytest.approx(b.score)


def test_beam_search_returns_one_center_per_gate_time(problem):
    result = beam_search(**problem)
    assert len(result.schedule) == len(problem["gate_times"])
    assert all(0 <= center < len(problem["gate_library"]) for center in result.schedule)


def test_best_fixed_schedule_repeats_one_center(problem):
    result = best_fixed_schedule(**problem)
    assert len(set(result.schedule)) == 1
    assert len(result.schedule) == len(problem["gate_times"])


def test_controlled_operator_is_not_renormalized(problem):
    result = beam_search(**problem)
    controlled = effective_operator(problem["operator"], problem["horizon"], problem["gate_times"], result.schedule, problem["gate_library"])
    assert np.linalg.norm(controlled, 2) <= 1.0 + 1e-10


def test_controlled_operator_differs_from_free_operator(problem):
    result = beam_search(**problem)
    controlled = effective_operator(problem["operator"], problem["horizon"], problem["gate_times"], result.schedule, problem["gate_library"])
    free = np.linalg.matrix_power(problem["operator"], problem["horizon"])
    assert np.linalg.norm(controlled - free, ord="fro") > 1e-6
