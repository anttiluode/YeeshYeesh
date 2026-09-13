import numpy as np
import pytest

from yeesh.gates import build_gate_library, propagate
from yeesh.substrate import build_y_graph, diffusion_operator, wave_operator


def test_y_graph_has_expected_branch_structure():
    graph = build_y_graph(5, 4)
    assert graph.adjacency.shape == (13, 13)
    assert np.allclose(graph.adjacency, graph.adjacency.T)
    assert graph.source == 0
    assert graph.left_arm == (5, 6, 7, 8)
    assert graph.right_arm == (9, 10, 11, 12)
    assert graph.target_left == (7, 8)
    assert graph.target_right == (11, 12)


def test_wave_operator_is_unitary():
    graph = build_y_graph(5, 4)
    operator = wave_operator(graph.adjacency, dt=0.75)
    ident = np.eye(operator.shape[0])
    assert np.allclose(operator.conj().T @ operator, ident, atol=1e-10)


def test_diffusion_operator_is_stable_and_nonnegative_for_canonical_parameters():
    graph = build_y_graph(5, 4)
    operator = diffusion_operator(graph.adjacency, dt=0.18, kappa=0.9, leak=0.02)
    assert np.min(operator) >= 0.0
    assert np.max(np.abs(np.linalg.eigvals(operator))) <= 1.0 + 1e-12


def test_every_gate_is_diagonal_and_contractive():
    graph = build_y_graph(5, 4)
    gates = build_gate_library(graph.distances, radius=1, outside_keep=0.35)
    assert len(gates) == graph.adjacency.shape[0]
    for gate in gates:
        assert np.allclose(gate, np.diag(np.diag(gate)))
        diagonal = np.diag(gate)
        assert np.max(diagonal) <= 1.0
        assert np.min(diagonal) >= 0.0
        assert np.linalg.norm(gate, 2) <= 1.0 + 1e-12


def test_propagate_rejects_mismatched_schedule_and_times():
    graph = build_y_graph(5, 4)
    operator = wave_operator(graph.adjacency, dt=0.75)
    gates = build_gate_library(graph.distances, radius=1, outside_keep=0.35)
    x0 = np.zeros(13, dtype=complex)
    x0[0] = 1.0
    with pytest.raises(ValueError, match="same length"):
        propagate(operator, x0, horizon=4, gate_times=(2, 4), schedule=(1,), gate_library=gates)


def test_propagate_never_renormalizes_a_contractive_schedule():
    graph = build_y_graph(5, 4)
    operator = wave_operator(graph.adjacency, dt=0.75)
    gates = build_gate_library(graph.distances, radius=1, outside_keep=0.35)
    x0 = np.zeros(13, dtype=complex)
    x0[0] = 1.0
    result = propagate(operator, x0, horizon=8, gate_times=(2, 5), schedule=(1, 5), gate_library=gates)
    assert np.linalg.norm(result) <= np.linalg.norm(x0) + 1e-12
