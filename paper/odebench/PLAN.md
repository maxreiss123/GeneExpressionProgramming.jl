# Test plan: GeneExpressionProgramming.jl on ODEBench (arXiv:2310.05573)

Benchmarking this package's ODE regression against ODEFormer (d'Ascoli et al., ICLR 2024)
and the baselines evaluated there, on their benchmark and under their metric.

> **Status.** This is the plan as written before the runs, kept for its stated success
> criteria. What was run is in `odebench_gep.jl` and `RESULTS.md`, and differs from the
> plan as follows:
>
> * Route A only, with local-polynomial derivative targets (cubic, 7 samples; 15 with
>   noise). Neither the weak-form estimator nor Route B was run.
> * One seed per condition; population 600, 150 generations, three genes of head length
>   6, operators `+ - * / sin cos exp sqr`.
> * Linear scaling on, so no constant optimisation took part (in the recorded runs the
>   optimiser was also inactive because of a bug fixed since).
> * Candidates: the best model per component plus one second-best swap per component,
>   ranked by reconstruction R² on the clean training trajectory.
> * RK4 on the reference grid with a divergence guard, but not with one step cap for all
>   methods: GEP.jl integrates with h ≤ 10/2048, the Python baselines score with
>   h ≤ 10/1024.
> * Grid: σ ∈ {0, 0.01, 0.02, 0.05} × ρ ∈ {0, 0.5}. SINDy and PySR were re-run locally
>   (step 5); ODEFormer was not, and its numbers are quoted from a re-evaluation.
> * The benchmark JSON is vendored as `strogatz_extended.json`; the figures come from
>   `make_figure.py` and `make_figure_rows.py`.

## What the benchmark provides

The benchmark ships in the ODEFormer repository as `odeformer/odebench/strogatz_extended.json`
(vendored here); checked against that file:

* **63 autonomous ODE systems**: 23 one-dimensional, 28 two-dimensional, 10
  three-dimensional, 2 four-dimensional, from Strogatz's textbook and well-known named
  systems (RC circuit, logistic growth, Lotka–Volterra, Lorenz, Maxwell–Bloch, ...).
* Each system carries its symbolic form with substituted constants, **two initial
  conditions** and **reference trajectories of 512 points on t ∈ [0, 10]** per initial
  condition. The second trajectory serves the generalization task only, so nothing has
  to be re-integrated here.
* Operators: `+ - * / ^` plus `sin` (11 systems), `cos` (6), `exp` (4), and `log`, `cot`,
  `abs` (1 each). The constants (`consts`) go up to 130.

From the paper and its evaluation code:

* Corruption: **multiplicative noise** `y_i = (1 + ε) x_i`, `ε ~ N(0, σ²)`, and **random
  subsampling** that drops a fraction ρ of the 512 points; headline conditions σ = 0.05,
  ρ = 0.5.
* Two tasks, one metric. *Reconstruction*: integrate the inferred system from the
  observed trajectory's initial condition and compare against the clean reference.
  *Generalization*: integrate from the second, unseen initial condition. The score is the
  **fraction of systems with variance-weighted R² > 0.9**.
* Baselines in the paper: PySR, SINDy, ProGED, AFP, FFX, EHC — the functional-SR ones
  driven by finite-difference derivative targets with a tuned Savitzky–Golay filter.

Unverified at planning time: the full noise and subsampling grids beyond the headline
conditions, R² clipping conventions, and ODEFormer's candidate/beam settings. Step 0
addresses this by porting the paper's evaluation code rather than reimplementing the
metric from prose.

## How the method maps onto the task

The trajectory is `x(t) ∈ R^d` at 512 timestamps. Two regression routes:

**Route A — componentwise (primary).** Estimate derivative targets `ẋ_k(t)` from the
trajectory, then run `GepRegressor` once per component with features `x_1 ... x_d` and
`linear_scaling=true`. This is how the paper drives its functional-SR baselines, so it is
the like-for-like configuration: 1–4 small regressions of 512 samples each.

**Route B — vector form.** The state is one `Vec{d}` column and the target the
`Vec{d}`-valued `ẋ`; `GepTensorRegressor` fits a single vector equation with gene-wise
fitted coefficients through the loss-callback interface (as in
`examples/Main_streaming_chunks.jl`). One model instead of d, with coupling between
components expressed directly. Not how the baselines are run, so it would be reported
alongside Route A, never pooled.

**Derivative targets.** Both routes need `ẋ`. Three estimators, compared on the noisy
conditions because this choice is known to dominate at σ > 0:
1. central finite differences (the naive floor);
2. Savitzky–Golay smoothing then differentiation, window and order tuned per condition,
   as in the paper's baseline protocol;
3. **weak form**: integrate candidate expressions against test functions `(1 − s²)^β`
   over sliding windows of the trajectory, as in SPIDER (JFM 996 A25), so the derivative
   moves onto the test function by integration by parts and is never estimated from
   noisy data. The planned differentiator at σ = 0.05.

**Fitness** is derivative-matching MSE (cheap, per candidate). **Model selection** is
integration-based: the hall of fame's top k candidates are integrated from the training
initial condition and ranked by reconstruction R², which filters out expressions that
match `ẋ` pointwise but integrate badly.

## Protocol

* Conditions per system: clean (σ=0, ρ=0), noise (σ=0.05, ρ=0), subsampled (σ=0, ρ=0.5),
  both (σ=0.05, ρ=0.5); extended to the paper's full grids once step 0 has pinned them.
* 3 seeds per system and condition; the score of a (system, condition) cell is the
  median-R² seed, and the seed spread is reported — the paper's methods are deterministic
  at inference, ours is not.
* Search budget fixed across systems: population 1000, up to 500 generations with early
  stop, operators `+ - * / sin cos exp log abs sqr` (matching the benchmark's; `cot`
  enters as `cos/sin` and affects one system, id 35).
* Every candidate integration uses the same solver settings, with a step/time budget and
  a divergence guard — a candidate that blows up scores R² = −∞ for that trajectory, not
  an exception.
* Wall-clock per system recorded end to end (fit + selection), single machine, CPU only,
  alongside the published inference times.

## Deliverables and steps

0. **Port the metric.** Vendor the paper's evaluation functions (R², variance weighting,
   integration settings) from the ODEFormer repository and validate on one system end to
   end; pin down the full σ/ρ grids from the code. *(This step gates everything.)*
1. `download_odebench.py` — fetch and checksum the benchmark JSON; loader that applies
   noise/subsampling with a fixed RNG per (system, condition, seed).
2. `odebench_gep.jl` — Route A: derivative estimation (all three estimators behind a
   flag), componentwise fit, hall-of-fame integration selection, per-system JSON results.
3. `odebench_vector.jl` — Route B, on the loss-callback pattern of
   `examples/Main_streaming_chunks.jl`.
4. `summarize.py` + figures — accuracy against noise and subsampling (the paper's figure 4
   layout), P(R²>0.9) tables for both tasks, expression complexity, wall-clock. Published
   ODEFormer/PySR/SINDy/ProGED numbers quoted as published and marked as not re-run.
5. Optional: re-run SINDy and PySR under the same data loader, so that at least two
   baselines share the machine. ODEFormer is a pretrained pip package and worth trying on
   CPU; if it does not run, its published numbers stand, labelled as such.

## What success looks like, stated before running

* **Clean data:** competitive with PySR, a strong classical baseline in the paper, on
  P(R²>0.9), both tasks. If we cannot match PySR clean, the noise story is moot.
* **Noise/subsampling:** the paper's claim is that classical SR degrades badly at
  σ = 0.05 while ODEFormer does not. The weak-form estimator is our counter — the target
  is to hold P(R²>0.9) within 10 points of clean performance at σ = 0.05, which would
  place us above the classical baselines and in ODEFormer's bracket.
* **Speed:** end-to-end wall-clock per system in single-digit seconds on CPU, against
  ODEFormer's transformer inference; the integration-based selection may dominate the
  budget. Report it either way.
* **Symbolic recovery** (stricter than R², not in the paper): fraction of systems whose
  recovered expression is algebraically equivalent to the ground truth, checked by
  simplification. R² > 0.9 admits wrong equations that fit one trajectory.

## Known risks

* **Route B has no exact-form guarantee**: most ODEBench systems are componentwise
  heterogeneous (each `ẋ_k` a different expression), which a single vector equation over
  `Vec{d}` columns cannot always express with the current operator set. Route B is
  expected to win on symmetric or coupled systems and lose on heterogeneous ones; that
  split is itself a result.
* **Constant optimisation**: constants up to 130 and one trajectory of data mean that
  linear scaling alone may not reach nested constants (e.g. inside `sin`). The
  `Optim`-based constant tuner does not run together with linear scaling, so this needs a
  configuration without it, with the tuner's cost counted in the wall-clock.
* **Metric drift** invalidates everything — hence step 0.
* **Unverified grid details** (noise levels beyond 0.05, R² clipping) stay flagged until
  read from the paper's code, and any number depending on them is marked provisional.

## Sources

* ODEFormer: S. d'Ascoli et al., *ODEFormer: Symbolic Regression of Dynamical Systems
  with Transformers*, ICLR 2024, arXiv:2310.05573. Code and ODEBench:
  https://github.com/sdascoli/odeformer.
* Weak-form evaluation (SPIDER): Gurevich, Golden, Reinbold & Grigoriev, JFM 996 A25 (2024).
