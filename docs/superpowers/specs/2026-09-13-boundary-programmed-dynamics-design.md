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

The experiment asks whether searching only over gate **where** can produce a target-selective route that beats matched controls and survives held-out perturbations. Gate times and gate strength are fixed in V0 so location is the only searched variable.

## Non-negotiable anti-cheating rules

1. The substrate parameters are frozen before controller search.
2. Gates are contractive: they may preserve or remove state magnitude, never amplify it.
3. No renormalization after a gate.
4. Search sees only training conditions.
5. Random controls use the same number of gate events, gate times, and gate strength.
6. We report absolute target energy as well as target selectivity, so a controller cannot win by deleting nearly everything.
7. Science thresholds are recorded, not used to make CI fail. CI checks software invariants and deterministic reproducibility.

## V0 substrates

Use one small symmetric Y-shaped graph with a source trunk and two output arms. The initial condition is a localized source pulse and the readout is energy in terminal regions of the two arms.

V0 evaluates **two different fixed operators on the same graph**.

### Resonant/interfering operator

Construct a symmetric coupling matrix `H` from the graph adjacency and evolve a complex classical wave state with

`F_wave = exp(-i H dt)`.

`F_wave` is unitary to numerical precision. The point is not quantum biology; complex amplitudes are simply a compact representation of a linear resonant system with phase and interference.

This substrate can contain constructive and destructive interference. A contractive gate can therefore change later absolute target energy by changing which interfering components survive.

### Monotone diffusion control

Construct a stable leaky-diffusive operator

`F_diff = I + dt * (kappa * (A - D) - leak * I)`.

With nonnegative initial state and nonnegative stable propagation, contractive gates cannot create amplitude that free diffusion did not already contain. They may improve selectivity only by deleting competing activity. This is an intentional negative/mechanistic control.

The distinction is central: if absolute target-energy gain appears only in the resonant substrate, the result is evidence about interference-enabled control, not a generic property of every passive cable.

## Gate library

A gate is a diagonal matrix `P(c)` centered on graph node `c`.

- Nodes within graph-distance `radius` of `c` are preserved with multiplier `1`.
- Nodes outside the protected region are multiplied by `outside_keep`, with `0 <= outside_keep <= 1`.
- Applying a gate never increases the L2 norm.

The controller is a sequence of gate centers at fixed gate times. V0 searches only location schedules.

## Search

Use deterministic beam search over gate-center sequences.

At each gate event:

1. Expand each partial schedule with every candidate gate center.
2. Complete the remaining horizon with no future gates yet assigned.
3. Evaluate all training examples.
4. Score the partial schedule using `target_energy * (0.5 + selectivity)`.
5. Keep the top `beam_width` schedules with deterministic lexicographic tie-breaking.

The objective rewards absolute target energy and target-over-competitor selectivity simultaneously.

## Conditions

The primary comparison contains:

- `free`: no gates.
- `fixed`: repeat the best single gate center at every gate event, chosen on training data.
- `random_location`: matched gate count/strength/times, random centers.
- `random_order`: the optimized centers shuffled while keeping gate times fixed.
- `reverse`: optimized schedule applied in reverse order.
- `optimized`: beam-searched movable schedule.

Random controls are summarized over many seeded repeats.

## Generalization

Search uses a small training set with several source-pulse amplitudes and mild additive perturbations. Evaluation uses held-out amplitudes and independent noise seeds. The claim is not broad learning; the point is whether a discovered gate sequence is a reusable control motif rather than a single exact trajectory.

## Metrics

For each condition report:

- target energy,
- competitor energy,
- target selectivity = `target / (target + competitor + eps)`,
- total retained L2 energy,
- route score,
- effective-operator difference `||M_schedule - F^T||_F`,
- leading singular values of the free and controlled effective operators.

A strong V0 result requires the resonant `optimized` condition to improve both held-out target energy and selectivity relative to `free` and `fixed`, and to beat the median matched random-location control. The diffusion control is expected to be able to improve selectivity but **not** absolute target energy relative to free diffusion.

## Software structure

- `src/yeesh/substrate.py` — Y graph, graph distances, wave and diffusion operators.
- `src/yeesh/gates.py` — contractive gate construction and schedule propagation.
- `src/yeesh/search.py` — deterministic beam search and best-fixed control.
- `src/yeesh/metrics.py` — energy/selectivity/operator metrics.
- `src/yeesh/experiment.py` — reproducible V0 experiment and controls.
- `run_experiment.py` — CLI entry point and JSON result writer.
- `tests/` — invariants, determinism, anti-cheating checks, mechanistic controls, and end-to-end smoke test.
- `.github/workflows/ci.yml` — unit tests on supported Python versions.

## Outputs

A run writes a JSON receipt containing configuration, optimized schedules, all control metrics, random-control distribution summaries, held-out metrics, and effective-operator diagnostics. The README contains the exact command needed to reproduce the canonical run.

## Interpretation boundary

A positive result would support this statement:

> Repeated selective constraints can program the effective dynamics of a fixed classical resonant substrate, and phase/interference can make that control stronger than simple selective deletion in monotone diffusion.

It would **not** establish that an axon initial segment is a projective quantum measurement, that neurons use the quantum Zeno effect, or that this mechanism explains biological learning.
