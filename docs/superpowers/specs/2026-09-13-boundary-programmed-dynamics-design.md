# Boundary-Programmed Dynamics — Design

## Purpose

Test one narrow claim:

> A fixed dynamical substrate can acquire a different effective input-output operator when it is repeatedly interrupted by spatially selective, time-dependent constraints.

This repository deliberately does **not** claim that neurons implement the quantum Zeno effect. The quantum-Zeno paper is motivation for the operator form `P_n F ... P_1 F`, not evidence for quantum cognition.

## Scientific question

Let a fixed substrate have one-step evolution operator `F`. Compare ordinary evolution

`x_T = F^T x_0`

with interrupted evolution

`x_T = P_K F^{d_K} ... P_2 F^{d_2} P_1 F^{d_1} x_0`,

where every `P_k` is a contractive spatial gate selected from a fixed library. The substrate `F` never changes during search.

The experiment asks whether searching only over gate **where**, **when**, and **strength** can produce a target-selective route that beats matched controls and survives held-out perturbations.

## Non-negotiable anti-cheating rules

1. The substrate parameters are frozen before controller search.
2. Gates are contractive: they may preserve or remove state magnitude, never amplify it.
3. No renormalization after a gate.
4. Search sees only training conditions.
5. Random controls use the same number of gate events and the same gate strength.
6. We report absolute retained target energy as well as target selectivity, so a controller cannot win by deleting nearly everything.
7. Science thresholds are recorded, not used to make CI fail. CI checks software invariants and deterministic reproducibility.

## V0 substrate

Use a small branched passive cable represented as a graph. Each node carries a scalar state. Free evolution is a stable leaky-diffusive step

`x[t+1] = F x[t] + u[t]`

with

`F = I + dt * (kappa * L - leak * I)`,

where `L` is the graph Laplacian with the sign convention that diffusion is stable.

The graph has one source trunk and two output arms. The experiment injects an initial localized pulse near the source and reads energy in two terminal output regions.

## Gate library

A gate is a diagonal matrix `P(c)` centered on graph node `c`.

- Nodes inside a radius around `c` are preserved.
- Nodes outside the protected region are multiplied by `outside_keep`, with `0 <= outside_keep <= 1`.
- Applying a gate never increases the L2 norm.

The controller is a sequence of gate centers at fixed gate times in V0. A later experiment may search gate times too, but V0 first establishes whether location scheduling alone changes the effective operator.

## Search

Use deterministic beam search over gate-center sequences.

At each gate event:

1. Expand each partial schedule with every candidate gate center.
2. Simulate all training examples to that event/final horizon.
3. Score each schedule by a predeclared objective combining:
   - absolute energy retained in the requested target region,
   - selectivity for the requested target over the competing output,
   - a small survival term preventing near-zero-state solutions.
4. Keep the top `beam_width` schedules.

Tie-breaking is deterministic.

## Conditions

The primary comparison contains:

- `free`: no gates.
- `fixed`: repeat the best single gate center at every gate event.
- `random_location`: matched gate count/strength, random centers.
- `random_order`: same centers as the optimized schedule, shuffled.
- `reverse`: optimized schedule applied in reverse order.
- `optimized`: beam-searched movable schedule.

Random controls are summarized over many seeded repeats.

## Generalization

Search uses a small training set with fixed pulse amplitudes and mild perturbations. Evaluation uses held-out amplitudes and independent additive input noise seeds. Because the system is intentionally simple, the claim is not broad general intelligence; the point is whether the schedule represents a reusable control motif rather than a single exact trajectory.

## Metrics

For each condition report:

- target energy,
- competitor energy,
- target selectivity = `target / (target + competitor + eps)`,
- total retained L2 energy,
- route score used by search,
- effective-operator difference `||M_schedule - F^T||_F`,
- singular values of the free and controlled effective operators.

The headline result is valid only if `optimized` improves target energy and selectivity together relative to free/fixed and beats the median random control on held-out examples.

## Software structure

- `src/yeesh/substrate.py` — graph construction and frozen evolution operator.
- `src/yeesh/gates.py` — contractive gate construction and schedule application.
- `src/yeesh/search.py` — deterministic beam search.
- `src/yeesh/metrics.py` — energy/selectivity/operator metrics.
- `src/yeesh/experiment.py` — reproducible V0 experiment orchestration.
- `run_experiment.py` — CLI entry point and result writer.
- `tests/` — invariants, search determinism, anti-cheating checks, and end-to-end smoke test.
- `.github/workflows/ci.yml` — unit tests on supported Python versions.

## Outputs

A run writes a JSON receipt containing configuration, optimized schedule, all control metrics, random-control distribution summary, held-out metrics, and effective-operator diagnostics. The README will contain the exact command needed to reproduce the canonical run.

## Interpretation boundary

A positive result would support this statement:

> Repeated selective constraints can program the effective dynamics of a fixed classical substrate.

It would **not** establish that an axon initial segment is a projective quantum measurement, that neurons use the quantum Zeno effect, or that this mechanism explains biological learning.
