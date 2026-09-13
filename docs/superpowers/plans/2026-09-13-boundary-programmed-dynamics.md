# Boundary-Programmed Dynamics V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible experiment showing whether repeated contractive spatial gates can program a frozen resonant graph operator, with monotone diffusion as a mechanistic negative control.

**Architecture:** A small Y graph defines two frozen propagators: a unitary complex wave operator and a stable real diffusion operator. A library of diagonal contractive gates is searched by deterministic beam search; matched controls and held-out perturbations determine whether a discovered movable schedule changes the effective input-output map without changing the substrate.

**Tech Stack:** Python 3.11+, NumPy, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-13-boundary-programmed-dynamics-design.md`

## Global Constraints

- The substrate operator is immutable during schedule search.
- Gates never amplify any state component and never renormalize the state.
- V0 searches only gate locations; gate times, radius, and strength are fixed.
- Random controls match optimized gate count, times, radius, and strength.
- Scientific outcomes are written to receipts; CI fails only on software/invariant failures.
- The README must explicitly reject a neuronal quantum-Zeno interpretation.

---

### Task 1: Frozen substrates and contractive gates

**Files:**
- Create: `src/yeesh/__init__.py`
- Create: `src/yeesh/substrate.py`
- Create: `src/yeesh/gates.py`
- Test: `tests/test_substrate_and_gates.py`

**Interfaces:**
- Produces: `build_y_graph(trunk_length, arm_length) -> GraphSpec`
- Produces: `wave_operator(adjacency, dt) -> np.ndarray`
- Produces: `diffusion_operator(adjacency, dt, kappa, leak) -> np.ndarray`
- Produces: `build_gate_library(distances, radius, outside_keep) -> list[np.ndarray]`
- Produces: `propagate(operator, x0, horizon, gate_times=(), schedule=(), gate_library=None) -> np.ndarray`

- [ ] **Step 1: Write failing invariant tests**

```python
def test_wave_operator_is_unitary():
    graph = build_y_graph(5, 4)
    U = wave_operator(graph.adjacency, 0.75)
    assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-10)


def test_every_gate_is_contractive():
    graph = build_y_graph(5, 4)
    gates = build_gate_library(graph.distances, radius=1, outside_keep=0.35)
    for P in gates:
        assert np.max(np.diag(P)) <= 1.0
        assert np.min(np.diag(P)) >= 0.0
        assert np.linalg.norm(P, 2) <= 1.0 + 1e-12
```

- [ ] **Step 2: Run tests and verify they fail because the package does not exist**

Run: `pytest tests/test_substrate_and_gates.py -v`
Expected: import failure.

- [ ] **Step 3: Implement graph, operators, gates, and propagation**

Use a symmetric adjacency matrix, all-pairs unweighted shortest-path distances, eigen-decomposition for `exp(-i H dt)`, explicit stable diffusion step, and diagonal gate masks. Validate schedule length against gate times and refuse a schedule without a gate library.

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/test_substrate_and_gates.py -v`
Expected: PASS.

---

### Task 2: Metrics and deterministic search

**Files:**
- Create: `src/yeesh/metrics.py`
- Create: `src/yeesh/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Produces: `route_metrics(state, target_nodes, competitor_nodes) -> dict[str, float]`
- Produces: `route_score(metrics) -> float`
- Produces: `effective_operator(operator, horizon, gate_times, schedule, gate_library) -> np.ndarray`
- Produces: `beam_search(operator, examples, horizon, gate_times, gate_library, target_nodes, competitor_nodes, beam_width) -> SearchResult`
- Produces: `best_fixed_schedule(...) -> SearchResult`

- [ ] **Step 1: Write failing tests for determinism and anti-cheating**

```python
def test_beam_search_is_deterministic(problem):
    a = beam_search(**problem)
    b = beam_search(**problem)
    assert a.schedule == b.schedule
    assert a.score == b.score


def test_controlled_operator_is_not_renormalized(problem):
    result = beam_search(**problem)
    M = effective_operator(problem["operator"], problem["horizon"],
                           problem["gate_times"], result.schedule,
                           problem["gate_library"])
    assert np.linalg.norm(M, 2) <= 1.0 + 1e-10
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_search.py -v`
Expected: missing metrics/search modules.

- [ ] **Step 3: Implement metrics and beam search**

Score each schedule by mean training `target_energy * (0.5 + selectivity)`. At each beam depth, simulate each partial schedule over the full horizon with only assigned gates active. Sort by descending score and lexicographic schedule for deterministic ties.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_search.py -v`
Expected: PASS.

---

### Task 3: Reproducible experiment and mechanistic controls

**Files:**
- Create: `src/yeesh/experiment.py`
- Test: `tests/test_experiment.py`

**Interfaces:**
- Produces: `ExperimentConfig` dataclass with canonical defaults.
- Produces: `make_examples(node_count, amplitudes, seeds, noise_std) -> list[np.ndarray]`
- Produces: `run_experiment(config) -> dict`

- [ ] **Step 1: Write failing end-to-end science-structure tests**

```python
def test_canonical_wave_result_beats_free_and_fixed():
    receipt = run_experiment(ExperimentConfig(random_repeats=32))
    held = receipt["wave"]["heldout"]
    assert held["optimized"]["target_energy"] > held["free"]["target_energy"]
    assert held["optimized"]["selectivity"] > held["free"]["selectivity"]
    assert held["optimized"]["target_energy"] > held["fixed"]["target_energy"]


def test_diffusion_control_does_not_create_target_energy_gain():
    receipt = run_experiment(ExperimentConfig(random_repeats=16))
    held = receipt["diffusion"]["heldout"]
    assert held["optimized"]["target_energy"] <= held["free"]["target_energy"] + 1e-12
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest tests/test_experiment.py -v`
Expected: missing experiment module.

- [ ] **Step 3: Implement canonical V0 orchestration**

Canonical defaults: trunk `5`, arm `4`, wave `dt=0.75`, diffusion `dt=0.18`, `kappa=0.9`, `leak=0.02`, horizon `24`, gate times `[2,5,8,11,14,17,20]`, radius `1`, outside keep `0.35`, beam width `64`. Training amplitudes `[0.8,1.0,1.2]` with seeds `[0,1,2]` and noise `0.01`; held-out amplitudes `[0.65,0.95,1.35,1.6]` with seeds `[100,101,102,103]` and noise `0.015`.

For each substrate compute free, optimized, best-fixed, reverse, random-location summary, random-order summary, effective-operator Frobenius difference, and leading singular values. Random schedules use a seeded NumPy generator.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_experiment.py -v`
Expected: PASS.

---

### Task 4: CLI, canonical receipt, and README

**Files:**
- Create: `run_experiment.py`
- Create: `README.md`
- Create: `results/.gitkeep`
- Create during verification: `results/v0_canonical.json`
- Test: `tests/test_cli.py`

**Interfaces:**
- CLI: `python run_experiment.py --out results/v0_canonical.json --random-repeats 128`

- [ ] **Step 1: Write a failing CLI smoke test**

Run the CLI in a temporary directory with `--random-repeats 4`; assert exit code zero and parseable JSON with `wave`, `diffusion`, and `config` keys.

- [ ] **Step 2: Implement CLI**

Use `argparse`, serialize NumPy values to ordinary JSON types, print a compact summary, and write the full receipt.

- [ ] **Step 3: Write README**

Document the question, anti-cheating rules, the resonant-vs-diffusion distinction, reproduction command, result interpretation, and explicit non-claim about quantum neuronal measurement.

- [ ] **Step 4: Run the canonical experiment**

Run: `python run_experiment.py --out results/v0_canonical.json --random-repeats 128`
Expected: a receipt in which wave optimized beats free/fixed on held-out target energy and selectivity, while diffusion optimized does not beat free diffusion on target energy.

- [ ] **Step 5: Run CLI test**

Run: `pytest tests/test_cli.py -v`
Expected: PASS.

---

### Task 5: CI and final verification

**Files:**
- Create: `pyproject.toml`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Package install: `python -m pip install -e .`
- Test command: `pytest -q`

- [ ] **Step 1: Add minimal package/test configuration**

Require Python `>=3.11`, NumPy, and pytest as a test extra. Configure pytest to add `src` to import path through editable installation rather than path hacks.

- [ ] **Step 2: Add GitHub Actions**

Run tests on Python 3.11 and 3.12 on push and pull request.

- [ ] **Step 3: Verify locally from a clean install**

Run: `python -m pip install -e '.[test]' && pytest -q`
Expected: all tests pass.

- [ ] **Step 4: Verify the canonical receipt again**

Run: `python run_experiment.py --out results/v0_canonical.json --random-repeats 128`
Expected: deterministic optimized schedules and the same headline comparisons.

- [ ] **Step 5: Commit and push the completed V0 branch**

Create a PR to `main`, verify CI, then merge if green.
