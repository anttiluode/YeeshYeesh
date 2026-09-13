from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraphSpec:
    adjacency: np.ndarray
    distances: np.ndarray
    source: int
    left_arm: tuple[int, ...]
    right_arm: tuple[int, ...]
    target_left: tuple[int, ...]
    target_right: tuple[int, ...]


def _all_pairs_distances(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    distances = np.full((n, n), fill_value=n + 1, dtype=int)
    for source in range(n):
        distances[source, source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in np.flatnonzero(adjacency[node]):
                candidate = distances[source, node] + 1
                if candidate < distances[source, neighbor]:
                    distances[source, neighbor] = candidate
                    queue.append(int(neighbor))
    return distances


def build_y_graph(trunk_length: int = 5, arm_length: int = 4) -> GraphSpec:
    if trunk_length < 2:
        raise ValueError("trunk_length must be at least 2")
    if arm_length < 2:
        raise ValueError("arm_length must be at least 2")

    node_count = trunk_length + 2 * arm_length
    adjacency = np.zeros((node_count, node_count), dtype=float)

    def connect(a: int, b: int) -> None:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    for node in range(trunk_length - 1):
        connect(node, node + 1)

    left_arm = tuple(range(trunk_length, trunk_length + arm_length))
    right_arm = tuple(range(trunk_length + arm_length, node_count))
    branch = trunk_length - 1
    connect(branch, left_arm[0])
    connect(branch, right_arm[0])

    for arm in (left_arm, right_arm):
        for a, b in zip(arm, arm[1:]):
            connect(a, b)

    return GraphSpec(
        adjacency=adjacency,
        distances=_all_pairs_distances(adjacency),
        source=0,
        left_arm=left_arm,
        right_arm=right_arm,
        target_left=left_arm[-2:],
        target_right=right_arm[-2:],
    )


def wave_operator(adjacency: np.ndarray, dt: float = 0.75) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("wave_operator requires symmetric coupling")
    if dt <= 0:
        raise ValueError("dt must be positive")

    hamiltonian = -adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * eigenvalues * dt)
    return (eigenvectors * phases) @ eigenvectors.T


def diffusion_operator(
    adjacency: np.ndarray,
    dt: float = 0.18,
    kappa: float = 0.9,
    leak: float = 0.02,
) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if dt <= 0 or kappa < 0 or leak < 0:
        raise ValueError("dt must be positive and kappa/leak nonnegative")

    degree = np.diag(adjacency.sum(axis=1))
    generator = kappa * (adjacency - degree) - leak * np.eye(adjacency.shape[0])
    operator = np.eye(adjacency.shape[0]) + dt * generator

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(operator))))
    if spectral_radius > 1.0 + 1e-10:
        raise ValueError("diffusion step is unstable; reduce dt")
    return operator
