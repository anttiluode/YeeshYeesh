# YeeshYeesh

**Can repeated constraints program a fixed dynamical substrate without changing the substrate itself?**

That is the whole V0 question.

This repo came from noticing the operator form behind spatial quantum-Zeno experiments,

```text
free evolution -> selective constraint -> free evolution -> selective constraint -> ...
```

and asking whether that form is useful *classically*. It is not a quantum-brain repo. The experiment below contains no quantum measurement, no wavefunction collapse, no learned substrate weights, and no claim that an axon initial segment implements a quantum Zeno effect.

The object under test is simply

```text
free:        F^T
controlled:  P_K F^dK ... P_2 F^d2 P_1 F^d1
```

where `F` is frozen and every `P_k` is contractive. Search is allowed to choose only **where** the spatial projection window is centered.

## The anti-cheating rules

- `F` is frozen before search.
- Gates may preserve or attenuate components; they can never amplify them.
- There is **no renormalization** after a gate.
- Gate times, radius, strength, and count are fixed in V0.
- Random controls use exactly the same gate count/times/strength.
- Search sees only the training perturbations.
- We report absolute target energy, not just target fraction, so deleting almost everything cannot masquerade as routing.

The gate itself is intentionally strong and abstract: it preserves a graph-distance window around a chosen center and attenuates the rest of the state by a fixed factor. That makes V0 a test of **projection-controlled dynamics**, not yet a realistic AIS model.

## Why there are two substrates

The first version of this idea was a leaky diffusion cable. That immediately exposed an important limitation.

For a positive monotone diffusion process, a contractive gate can improve *selectivity* by removing competing activity, but it cannot magically create more absolute target amplitude than ungated propagation. So V0 keeps diffusion as a negative/mechanistic control and adds a frozen **resonant linear wave substrate** with phase and interference.

Both use the same Y graph:

```text
source
  0--1--2--3--4
              |\
              | \
 left arm   5--6--7--8
 right arm  9-10-11-12
```

The target is the end of the left arm (`7, 8`); the competing readout is the end of the right arm (`11, 12`).

The wave propagator is

```text
F_wave = exp(-i H dt)
```

with symmetric graph coupling `H`. Complex amplitudes are just a compact representation of a classical linear resonant system with phase. The diffusion control is

```text
F_diff = I + dt * (kappa * (A - D) - leak * I)
```

with a stable nonnegative canonical step.

## Canonical V0 result

Run:

```bash
python -m pip install -e '.[test]'
python run_experiment.py --out results/v0_canonical.json --random-repeats 128
```

The committed canonical receipt gives the wave search schedule

```text
gate times:  [2, 5, 8, 11, 14, 17, 20]
centers:     [1, 5, 7,  5,  1,  1,  5]
```

on held-out amplitudes and independent noise seeds:

| condition | target energy | selectivity |
|---|---:|---:|
| wave free | 0.0267394 | 0.478930 |
| wave optimized | **0.0555838** | **0.903835** |
| wave best fixed | 0.00129696 | 0.499960 |
| wave reverse order | 0.0000217991 | 0.614180 |
| wave random-location median | 0.00000803436 | 0.475779 |
| wave random-order median | 0.0000557775 | 0.694540 |

So the searched schedule produces about **2.08x the free target energy** and **1.89x the free selectivity** on held-out inputs. It does this while remaining strongly contractive: total held-out state energy falls from `1.43913` free to `0.07237` controlled. In other words, the target gain is not hidden amplification; the schedule is changing which interfering components survive to arrive at the target.

The effective operator genuinely changes. The controlled-vs-free Frobenius distance is `3.54134`; the free wave operator has spectral norm `~1`, while the controlled operator is contractive with spectral norm `0.28554`.

### The diffusion control behaves differently

| condition | target energy | selectivity |
|---|---:|---:|
| diffusion free | **0.000235453** | 0.677487 |
| diffusion optimized | 0.0000152143 | **0.957011** |

The diffusion schedule becomes more selective, but retains only about **6.46%** of the free target energy. That is exactly the distinction V0 was designed to expose: selective deletion alone is not the same thing as interference-enabled routing.

## What this supports

The narrow supported statement is:

> Repeated selective constraints can program the effective input-output dynamics of a fixed classical resonant substrate, and phase/interference can make this stronger than simple selective deletion in monotone diffusion.

That is already useful. It says a controller need not rewrite the substrate matrix to rewrite its *effective* operator. It can instead choose a sequence of boundaries/projections around an otherwise trusted forward process.

## What this does **not** support

It does not show that:

- neurons are quantum computers,
- an AIS spike is a projective quantum measurement,
- the biological quantum Zeno effect is operating in a dendrite,
- passive dendritic cable physics alone provides this target-energy gain,
- the discovered gate is already a realistic soma/AIS mechanism.

The diffusion control is especially important for the dendrite question: a plain passive monotone cable does **not** reproduce the strong effect here. If biology has a related mechanism, the interesting ingredients would have to come from active/nonlinear/resonant dynamics and the actual local geometry of soma/AIS feedback, not from attaching quantum words to cable diffusion.

## Where Oja, dendrite, soma, and axon would enter next

V0 intentionally leaves them out so the operator claim can stand or fall cleanly.

- **Oja / synapse:** learn or normalize the input subspace before the frozen dynamics. The question would be whether the same gate motif works across learned input modes.
- **Dendrite:** replace the abstract resonant graph with cable + local NMDA dynamics and ask whether a comparable operator change survives.
- **Soma / AIS:** replace the global projection window with a genuinely local threshold/reset/boundary event. This is the key biological plausibility gate.
- **Axon:** treat the spike as the emitted readout after that classical boundary event, not as a quantum collapse.

The important failure condition is clear: if the effect disappears when the global projector is replaced by a local biologically plausible gate, the Zeno-inspired analogy was mathematically useful but biologically shallow. That is a good result too.

## Repository layout

```text
src/yeesh/substrate.py   Y graph + frozen wave/diffusion operators
src/yeesh/gates.py       contractive spatial gates + propagation
src/yeesh/metrics.py     route and effective-operator diagnostics
src/yeesh/search.py      deterministic beam search + fixed control
src/yeesh/experiment.py  canonical experiment + matched controls
run_experiment.py        reproducible CLI
results/v0_canonical.json committed result receipt
tests/                   invariants, controls, and end-to-end tests
```

CI tests Python 3.11 and 3.12. Scientific outcomes are recorded rather than used as arbitrary CI pass/fail thresholds; CI primarily protects the anti-cheating invariants and deterministic mechanism.
