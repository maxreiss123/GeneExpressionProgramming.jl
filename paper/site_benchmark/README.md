# GeneExpressionProgramming.jl vs. SITE (arXiv:2507.01466v1)

Reproduction of the benchmarks in

> T. Chen, H. Yang, W. Ma, J. Zhang, *Symbolic identification of tensor equations in
> multidimensional physical fields*, arXiv:2507.01466v1 (2025)

with this package, and a like-for-like comparison against the authors' implementation
([PistilReaper/SITE](https://github.com/PistilReaper/SITE); the runs here used its
release at [BUAA-MARS-group/SITE](https://github.com/BUAA-MARS-group/SITE), 32ab7c9)
run on the same machine.

SITE is related to this repository: its dimensional homogeneity check cites Reissmann et
al. (2025), the semantic-backpropagation constraint implemented here; its evolutionary
encoding follows M-GEP (a host chromosome for tensors, plasmid chromosomes for the
embedded scalars), and its coefficients come from a tensor linear regression (TLR).

All numbers below were measured on the same 4-core container (Julia 1.12.7 with 4
threads; Python 3.11 for SITE). The paper's own timings were taken on a 13th Gen Intel
Core i9-13900K and are quoted where useful.

> **Which code these numbers describe.** `results/` was rerun after three changes to the
> previous set. The dimensional check binds: every unscored individual is checked, the
> ones that fail are repaired (up to `correction_amount` of the population per
> generation, 0.3 in every harness here), and with a target dimension only homogeneous
> ones are scored; before, it acted almost only on the initial population. The constant
> optimiser runs (only the DSMC case asks for it); before, a bug kept it idle. And the
> genetic operators include gene averaging and the repaired transposition. The previous
> `results/` are in the git history and are quoted where the changes alter a conclusion.
> `results_pre_refactor/` holds the same matrix from the earlier, nondeterministic code
> (single draws). `results/tensor_gep_s1.json`, the tensor evaluator as released before
> the refactor, has no harness any more and was not rerun.
>
> Two later commits on main bear on these files. 1f7f583 scored a duplicate whose cached
> fitness was evicted in a parallel pass, and the tasks that pass spawned moved the seeds
> of every task spawned after them, the repair's among them: a seeded run with the check
> took another path. That pass runs on the calling thread again (466d6bf), and seed 1 of
> every configuration then reproduces these files exactly. 2ee8b7d holds every gene to the
> target under linear scaling and repairs the genes one by one, so the scaled scalar runs
> (`gep_ls_*`) were rerun with it, after a container restart, on a host of the same speed
> (seed 1 of the check-only run took 19.6 s there against the 19.9 s recorded here). Only
> those files record `winner_homogeneous`, the check applied to the winning model as it is
> scored (gene by gene under scaling); it holds for every winner.

## What is being compared

| | SITE | GeneExpressionProgramming.jl |
|---|---|---|
| encoding | host (tensor) + plasmid (scalar) chromosomes | (a) scalar chromosome, the tensor problem component-stacked, one row per `(sample, i, j)`; (b) tensor-native `GepTensorRegressor` |
| physical constraint | dimensional homogeneity check, invalid individuals get a large loss (soft rejection) | semantic backpropagation: individuals off the target dimension are *repaired* (`correct_genes!`), and only homogeneous ones are scored |
| coefficients | tensor linear regression (least squares per gene) and/or random numerical constants | gene-wise linear scaling (`linear_scaling=true`, the TLR analogue), symbolic constant terminals, random constants; `Optim`-based constant tuning (the DSMC case) |
| loss | mean relative `L2` error over the tensor components | the same (`make_site_loss`, a port of `loss_func` in `SITE.py`) |
| implementation | Python (geppy + DEAP) | Julia, multi-threaded |

Two Julia paths are benchmarked, with and without the dimensional homogeneity check
(DHC, `--config dhc|nodhc`) and with and without fitted coefficients (`run_all.sh` lists
the combinations):

* **scalar path** (`maxwell_gep.jl`, `GepRegressor`) — the component-stacked formulation.
  Exact for these targets, because each is a linear combination of the offered tensor
  terminals, but strictly less expressive than SITE's encoding: a stacked scalar
  chromosome cannot form a tensor inner product `A_ik B_kj`.
* **tensor path** (`maxwell_tensor_shadow.jl`, `GepTensorRegressor`) — evaluates whole
  batches into preallocated per-thread buffers and carries the check into the tensor
  operators; the closest counterpart to SITE's host/plasmid chromosome. The file name is
  historical; `SHADOW_ROOT` (default: this checkout) selects the source tree it loads.

The row *"GEP.jl tensor as released, one call per sample"* is the tensor evaluator this
package shipped before the refactor (`results/tensor_gep_s1.json`); its harness no
longer exists.

## The three cases

1. **Maxwell stress tensor** (paper Sec. 3.1, Tables 1 and 2) — the paper's data: 150
   samples from the authors' generator and seed (`export_site_data.py` follows
   `Maxwell_tlr_and_rnc_test.py`), plus the 5 % / 10 % / 20 % noise variants of
   `Maxwell_noise_test.py`.

       T_ij = eps_0 (E_i E_j - 1/2 E_k E_k delta_ij) + 1/mu_0 (B_i B_j - 1/2 B_k B_k delta_ij)

2. **Reynolds-stress transport in decaying isotropic turbulence** (Sec. 3.2, Table 3)

       dR_ij/dt = -(2/3) eps delta_ij

   The authors' repository does **not** contain the OpenFOAM dataset for this case
   (`data/processed_data.mat` is missing), so the data are reconstructed from the
   description in Sec. 3.2. Absolute coefficients are therefore only qualitatively
   comparable; the sub-sampling behaviour, which is what Table 3 reports, is.

3. **Constitutive relation from DSMC** (Sec. 4, Table 4) — the authors' DSMC
   lid-driven-cavity fields (`Kn = 0.005`, lid at 50 m/s and 337 m/s, 400x400 cells) are
   shipped, but the velocity gradients their TensorFlow network produces
   (`data/gradients.mat`) are not. `prepare_dsmc_data.py` rebuilds them with second-order
   finite differences on the uniform grid, restricted to the central 20 %–80 % of the
   domain that the paper samples from. A least-squares fit of the Newtonian model on the
   rebuilt compressible data gives `-1.888 mu S_ij + 0.660 mu D_kk delta_ij + 1.000 p
   delta_ij` against `-1.914 / +0.676 / +1.000` in the paper: the reconstruction agrees
   to within 2.5 % (1.4 % on the shear term).

---

# Results

Full tables: [`results/summary.md`](results/summary.md). Figures:
[`results/figures/`](results/figures). Raw per-run JSON: `results/*.json`.

## 1. Maxwell stress tensor — cost of one identification

| configuration | generations | wall-clock (s) | reached 1e-6 |
|---|---|---|---|
| **GEP.jl tensor, SI units + fitted coefficients** | 1 (median of 3) | **0.2** | **3/3 seeds** |
| **GEP.jl scalar, dimensional check + linear scaling** | 1 (median of 5) | 1.6 | **5/5 seeds** |
| **GEP.jl tensor, order check + fitted coefficients** | 57 (median of 3) | 3.9 | **3/3 seeds** |
| **GEP.jl scalar, dimensional check** | 134 (median of 5) | 11.3 | **5/5 seeds** |
| GEP.jl tensor, order check only | 829 (median of 3) | 25.1 | 3/3 seeds |
| GEP.jl tensor, no order check | 1170 (the converged seed) | 24.2 | 1/3 seeds |
| GEP.jl scalar, no dimensional check | 2000 (the cap) | 35.3 | 0/5 seeds |
| SITE, TLR + RNC (here) | 52 | 89.0 | yes |
| SITE, TLR only (here) | 501 (script cap) | 504.2 | no |
| SITE, RNC only (here) | 501 (script cap) | 672.5 | no |
| SITE, TLR + RNC — *paper*, i9-13900K | 23 | 20 | yes |
| SITE, TLR only — *paper* | 714 | 580 | yes |
| SITE, RNC only — *paper* | 2000 | 1666 | no |

Generations and wall-clock are medians over the converged seeds where any converged, over
all seeds otherwise (`fig1_convergence`, `fig2_cost`, `fig6_reliability`).

* **The Maxwell stress tensor is recovered exactly, on every seed of every configuration
  with a dimensional or order check.** With the check alone the scalar path takes 48 to
  205 generations (median 134, 11.3 s), and every winner projects onto the ground-truth
  terms as `+1, -1/2, +1, -1/2` to at least four digits. The winners are the textbook
  form plus terms that cancel, e.g. `eps0·EE + BB/mu0 − eps0·(delta·0.5)·E2 −
  BB·((0.5/mu0)/((BB/B2)/delta))` (seed 3, six digits).
* **With the check binding, it decides reliability.** With it, 5/5 seeds converge;
  without it, 0/5 within 2000 generations (best losses 8.4e-4 to 1.3e-2). In the previous
  runs, where the check acted almost only on the initial population, both converged on
  2/5 seeds. On the tensor path the order check lifts convergence from 1/3 to 3/3
  (Sec. 7).
* **Fitting the coefficients removes the search almost entirely.** With gene-wise linear
  scaling on top of the check, the scalar path reaches the tolerance within three
  generations on every seed (median 1, 1.6 s; 0.5 s before every gene had to take the
  target, which makes the repair of the initial population dearer), the tensor path with
  SI units in the first generation on all three (0.2 s), and the tensor path with the
  order check alone in 57 generations (3.9 s). The repair makes the initial population
  homogeneous, so it already contains the few admissible terms — under SI units
  `eps_0 E_iE_j`, `B_iB_j/mu_0` and their `delta_ij` traces, up to dimensionless factors —
  and least squares supplies the coefficients: the division of labour of the paper's TLR.
  Previously the same configurations took 66, 40 and 56 generations.
* **Generations are not comparable, wall-clock is.** SITE with TLR and RNC takes 52
  generations and 89.0 s on this machine; this package takes 0.2–1.6 s with fitted
  coefficients and 11.3 s without. SITE without either TLR or RNC does not converge
  within its script's cap (504 s and 672 s).
* **A binding check costs time per generation.** At population 1600 a scalar generation
  takes 94 ms with the check, the repair included, and 18 ms without (medians). In the
  previous runs, with the check idle, both took about 37 ms: evaluation is twice as fast
  as then, and the repair more than uses that up. It pays in generations: 134 against a
  search that does not converge.
* The reference implementation's released scripts cap the run at `n_gen = 500`, while the
  paper reports convergence of the TLR-only configuration at generation 714, so that row
  of Table 1 cannot be reproduced with the shipped script unless the cap is raised
  (`run_site_reference.py --generations`).

## 2. Robustness to noise (paper Table 2)

| data | eps0 E_iE_j | eps0 E_kE_k d_ij | B_iB_j/mu0 | B_kB_k d_ij/mu0 | mean rel. error |
|---|---|---|---|---|---|
| clean (n=5) | +1.0000 | -0.5000 | +1.0000 | -0.5000 | **0.00 %** |
| **5 % noise** (n=5) | +1.0000 | -0.5000 | +1.0012 | -0.4999 | **0.11 ± 0.09 %** |
| 10 % noise (n=5) | +1.0001 | -0.5000 | +1.0088 | -0.5068 | 0.57 ± 0.47 % |
| **20 % noise** (n=5) | +1.0004 | -0.5001 | +1.0061 | -0.5174 | **1.13 ± 0.53 %** |
| 5 % noise, linear scaling (n=5) | +0.9999 | -0.5001 | +0.9871 | -0.4963 | 0.74 ± 0.60 % |
| 10 % noise, linear scaling (n=5) | +0.9986 | -0.4997 | +1.0119 | -0.5065 | 0.72 ± 0.40 % |
| 20 % noise, linear scaling (n=5) | +0.9963 | -0.4983 | +1.0116 | -0.5188 | 1.87 ± 0.93 % |
| 5 % noise, tensor + SI units (n=3) | +0.9993 | -0.5002 | +1.0091 | -0.4994 | 0.39 ± 0.08 % |
| 10 % noise, tensor + SI units (n=3) | +0.9985 | -0.5003 | +0.9985 | -0.4932 | 0.43 ± 0.00 % |
| 20 % noise, tensor + SI units (n=3) | +0.9971 | -0.4996 | +1.0377 | -0.5138 | 1.73 ± 0.42 % |
| SITE, paper Table 2, 5 / 10 / 20 % | — | — | — | — | 0.25 / 0.45 / 1.53 % |

Coefficients are recovered by projecting each discovered model's prediction on the clean
data onto the four ground-truth tensor terms, so the comparison does not depend on how
the expression is written. Every configuration uses the dimensional check; the noise runs
never reach the tolerance, so each runs 2000 generations (`fig3_noise`,
`fig9_noise_paths`).

**With the check binding, the scalar search without fitted coefficients matches or beats
the reference implementation**: 0.11 %, 0.57 % and 1.13 % at 5, 10 and 20 % noise,
against SITE's 0.25 % (±0.27), 0.45 % (±0.55) and 1.53 % (±1.40) in the paper, with a
tighter spread at 5 and 20 %. In the previous runs, with the check idle, the same
configuration stood at 4.89 %, 4.50 % and 5.13 %. The admissible routes to the target
carry their coefficients as exact physical constants — `eps_0`, `mu_0` and the terminal
0.5 — which noise cannot move: the electric terms come out within 0.0004 of +1 and −1/2
at every level, and nearly all the error sits in the magnetic terms, about twelve times
smaller in magnitude on this data.

**Fitted coefficients do not help.** With linear scaling, every gene held to the target,
the error is 0.74 %, 0.72 % and 1.87 % (previously 0.31 %, 1.62 % and 6.21 %; 0.95 %,
0.40 % and 1.02 % when only the connected expression was held to it): above the unscaled
search at every level, where one free coefficient per gene takes up some of the noise that
the exact constants leave out. The tensor path with SI units and fitted coefficients gives
0.39 %, 0.43 % and 1.73 % (previously 0.34 %, 0.47 % and 2.08 %), close to the reference
implementation at every level and nearly unchanged by the binding unit check; its spread
at 10 % is again tight (±0.00 against ±0.55). These configurations differ in encoding,
head length (6 for the tensor runs, 8 for the scalar ones) and seed count (three against
five), so the comparison between them is loose. Against its own previous runs the scalar
search changed in the check, the genetic operators and the evaluator; that its winners now
carry exact constants follows from the check, which admits only homogeneous individuals.

## 3. Reynolds-stress transport (paper Table 3)

| data points | GEP.jl | SITE (paper, its own data) |
|---|---|---|
| 100 (100 %) | -0.666667 ± 9.9e-17 | -0.6617 |
| 75 (75 %) | -0.666667 ± 2.1e-16 | -(0.6617 ± 0.0001) |
| 50 (50 %) | -0.666667 ± 2.0e-16 | -(0.6617 ± 0.0002) |
| 25 (25 %) | -0.666667 ± 2.0e-16 | -(0.6617 ± 0.0004) |

25 sub-sampling seeds per size, median solve time **0.03 s** (median 1 generation;
previously 5 generations and 0.08 s). The package finds `-2/3` to machine precision at
every dataset size (previously to seven digits), with a scatter orders of magnitude below
the paper's — but the two runs are on different data, since the paper's OpenFOAM dataset
is not published and this is a reconstruction (`fig5_reynolds`).

It does *not* eliminate the two distractor terminals: 75 of the 100 runs (previously 93)
write the answer through `R_ij` and `k`, most often as `(eps + eps) - (eps + eps) -
R_ij·eps/k`. In decaying isotropic turbulence `R_ij = (2/3) k delta_ij`, so `R_ij/k` *is*
`(2/3) delta_ij` and the expression equals `-(2/3) eps delta_ij` on this data — which is
why the projection onto `eps delta_ij` returns `-2/3` exactly. The distractors are
absorbed into an equivalent form rather than dropped; only data from an anisotropic flow
would separate the two.

## 4. Constitutive relation from DSMC data (paper Table 4)

| case | GEP.jl (3 sub-samples) | paper, Table 4 |
|---|---|---|
| lid 50 m/s (incompressible) | **-1.858** mu S_ij + 0.028 mu D_kk d_ij + **1.000** p d_ij | -1.906 mu S_ij + 1.000 p d_ij |
| lid 337 m/s (compressible) | **-1.912** mu S_ij + 0.302 mu D_kk d_ij + **1.000** p d_ij | -1.914 mu S_ij + 0.676 mu D_kk d_ij + 1.000 p d_ij |

The constants of the best model are tuned every fifth generation (Nelder–Mead), each run now takes 10–13 s against 4–8 s then.

On the paper's own high-fidelity data the shear and pressure terms of the compressible
case are reproduced to 0.1 % (-1.912 ± 0.003 against -1.914; 1.000 against 1.000). The
`mu D_kk delta_ij` term is the unstable one: per seed it comes out as +0.335, +0.570 and
0.000 against the paper's +0.676. The paper itself singles this term out as much smaller
than the dominant contributions (its Sec. 4.3), so it is the one most exposed to the
finite-difference gradients used here in place of the authors' neural-network
differentiation, and three sub-samples are too few to pin it down.

The incompressible case, where the previous runs gave -1.765 ± 0.133 for the shear term,
now gives -1.858 ± 0.008 (-1.851, -1.857 and -1.867 per seed) against the paper's -1.906,
and 1.000 for the pressure term. The paper reports no `D_kk` term there, and the 0.028 ±
0.049 found here is consistent with zero: two of the three winners project onto it at
below 1e-3.

## 5. Runtime, with the one-off costs separated out

Every benchmark script runs a throw-away `fit!` before the timed one, so `solve_time_s`
contains no package loading, no precompilation and no JIT compilation.

| stage | seconds | paid |
|---|---|---|
| dependency precompilation (265 packages) | 237 | once per environment |
| package precompilation | 116.6 | once per source change |
| `using GeneExpressionProgramming` | 1.8 | once per process |
| first `fit!` (JIT) | 27.9 | once per process |
| second, identical `fit!` | 0.00 | steady state |
| **timed solve** (Maxwell, fastest converging configuration) | **0.2** | per identification |

So a cold start costs about 350 s once and ~30 s per process, and the identification
itself is what the tables above report (`fig2_cost`). The previous measurement, on Julia
1.11.5 with 199 dependencies, gave 205 s, 2.2 s, 3.9 s and 20.5 s for the first four
stages. In a separate measurement, removing the DynamicExpressions dependency took three
packages out of the manifest (DynamicExpressions, DispatchDoctor, Interfaces) but raised
the time to the first `fit!` from 18.6 s to 21.9 s: the batched stack machine is more
code to specialise than the tree walker was.

## 6. Speed and memory

Throughput on the 150-sample Maxwell case (`fig10_speed_memory`, panel a), counted as
population x generations per second of solve time (1600 per generation assumed for
SITE), so selection, variation and, where it runs, the repair are included:

| implementation | expressions/s |
|---|---|
| GEP.jl scalar (stacked, batched), no dimensional check | 90 732 |
| GEP.jl tensor (batched, preallocated), order check | 56 489 |
| GEP.jl scalar (stacked, batched), dimensional check | 16 979 |
| SITE (geppy + numpy), TLR + RNC | 953 |

**Speed is the claim that holds.** Against the reference implementation this package
processes candidates 18–95x faster and reaches the answer in 0.2 s against 89.0 s, both
measured on the same machine for the same task. With the check, the repair is the
largest cost of a scalar generation: it takes throughput from 90 732 to 16 979. Single
timings vary between runs (by up to 20 % in earlier repeats), so the throughput figures
are approximate; the time-to-solution table, a median over seeds, is the reliable one.

**Against DynamicExpressions the batched evaluator is now ahead at 150 samples too.** The
scalar command line without the check ran at 31.3 ms per generation through
DynamicExpressions (`results_pre_refactor/`) and at 35.4 ms through the previous batched
evaluator; it runs at 17.6 ms now. With the check the per-generation comparison no longer
holds still, since the repair, idle in both earlier sets, now takes most of a
generation. A first batched version, whose operands lived on a `Vector{Any}`, took
60.2 ms (26 585 expr/s), 1.8x the tree walker: profiled at 1350 rows, arithmetic was
3.5 % of its time and dispatch the rest. Compiling the alphabet into flat opcode tables
and walking a concretely typed stack (`compile_program` / `run_program!`) removed the
dispatch without changing a single result.

**Peak memory is a comparison SITE wins.** Resident high-water mark on the same run
(`measure_memory.py`, 200 generations at population 1600):

| | peak RSS | above its runtime's baseline |
|---|---|---|
| SITE (geppy + deap) | **128 MB** | 57 MB |
| GEP.jl scalar, batched buffers | 1029 MB | 601 MB |
| GEP.jl tensor, batched buffers | 1114 MB | 686 MB |

The Python interpreter with geppy, deap and numpy loaded starts at 71 MB; Julia with this
package loaded starts at 428 MB (previously 693 MB). Even setting the runtimes aside, the
search itself costs this package 10–12x what SITE's costs. At 150 samples the batched
evaluator does not help there: peak RSS is set by the runtime and the heap high-water
mark, not by per-generation churn.

**Allocation, and long data, are where the batched evaluator pays** (`fig10_speed_memory`,
panel b). Measured within Julia, where bytes are directly comparable, over 25 generations
at population 800 (`measure_allocation.jl`):

| samples | DynamicExpressions (earlier) | batched buffers (now) |
|---|---|---|
| 1 000 | 359 MB / 0.35 s | 73 MB / 0.15 s |
| 5 000 | 1 119 MB / 0.58 s | 87 MB / 0.21 s |
| 20 000 | 3 973 MB / 0.86 s | 141 MB / 0.54 s |
| 100 000 | 18 307 MB / 5.15 s | **455 MB** / 3.92 s |

DynamicExpressions allocated a fresh array for every node it evaluated, so its allocation
scaled with the data; the batched path writes into buffers allocated once per fit, and
its growth is those buffers. The DynamicExpressions column is the earlier measurement,
taken before that dependency was removed, on Julia 1.11.5 and with data and operators
that were not recorded; `measure_allocation.jl` fixes them (three features,
`y = x1 x2 - 0.5 x3`, `+ - * /`), so the comparison is indicative. This measurement also
found `fit!` building a fresh evaluation context for every epoch's validation of the
best model, and a second full set of per-thread buffers for the constant optimiser, which
uses only the input columns: together 1 344 MB per fit at 100 000 samples. The
validation context is now built once per fit and the optimiser shares the training
inputs, with the same results.

The per-thread `EvalScratch` in `src/TensorOps.jl` (four allocations fewer per call) cut
allocation 24-fold but not the wall-clock on the 150-sample case: medians of three runs
either side are 7.9 s and 8.0 s.

## 7. What the tensor path costs, and what the order check does

| tensor configuration (no fitted coefficients) | best loss per seed | note |
|---|---|---|
| with tensor-order check | **2.4e-7** (829 gen), **2.6e-7** (389 gen), **1.4e-7** (1282 gen) | 3/3 converged |
| without | 1000 (nothing valid), 1.8e-2, **2.3e-7** (1170 gen) | 1/3 converged |

In a flat tensor chromosome, scalars and tensors share one terminal set, so most random
expressions are type-invalid; SITE avoids this by construction, because a scalar can only
ever enter through a plasmid. Without the order check, seed 1 again produced **no valid
individual at all** in 2000 generations x 1600 individuals, and the three seeds scored
1.1 to 2.2 million invalid candidates. With the check binding, no invalid candidate was
scored and every seed converged, against two of three in the previous runs, where the
check acted almost only on the initial population. Fitting the gene coefficients remains
the larger lever: 57 generations against 829.

(The pre-refactor runs, against another package version on nondeterministic code, read
the other way round: no seed below the tolerance with the check, one without.)

## Notes on the tensor harness

`maxwell_tensor_shadow.jl` loads `src/` from `SHADOW_ROOT`, by default this checkout. It
still carries two workarounds from the package version it was first written against: a
warm-up population of 200, because that version's genetic operators indexed
`parents[1:100]`, and a second regressor for the clean-data prediction, because its
three-argument `predictT` did not work. Neither is needed with the current `src/`. Older
versions of `mul_t_unit_forward` also forwarded the larger of the two operand vectors
instead of composing their units, so tensor order and SI units could not both be carried
in one vector; `--units` relies on the current version, which composes them.

## Reproducing

```bash
# 1. the paper's data
python paper/site_benchmark/export_site_data.py
python paper/site_benchmark/prepare_dsmc_data.py /path/to/SITE      # case 3

# 2. everything, sequentially (so the wall-clock numbers stay comparable).
#    Stages whose result JSON already exists are skipped, so re-running resumes.
SITE_REPO=/path/to/SITE THREADS=4 DEPS_PRECOMPILE_S=237 bash paper/site_benchmark/run_all.sh
python paper/site_benchmark/measure_memory.py /path/to/SITE      # peak memory, allocation

# 3. tables and figures
python paper/site_benchmark/summarize.py > paper/site_benchmark/results/summary.md
python paper/site_benchmark/make_figures.py
```

Because stages with an existing result JSON are skipped, step 2 does nothing until the
committed `results/*.json` are moved away. `DEPS_PRECOMPILE_S` is the dependency
precompilation time from the Pkg log of the environment's `Pkg.instantiate()` (237 s
for the 265 packages here). The SITE side needs `pip install geppy deap`; the Julia side
only the packages declared in `Project.toml`. The DSMC CSVs are not committed — their
source data is GPL-3.0 — so `prepare_dsmc_data.py` has to be run against a clone of the
SITE repository before case 3 can be reproduced; the fits it prints for the rebuilt data
match the ones quoted above. The SITE runs reproduced the previous runs' generation
counts and losses exactly.

## Files

| file | purpose |
|---|---|
| `export_site_data.py` | writes the paper's Maxwell data (clean + 3 noise levels) and the reconstructed Reynolds data as CSV |
| `prepare_dsmc_data.py` | rebuilds the DSMC cavity terminal library from the authors' raw `.dat` fields |
| `maxwell_gep.jl` | Maxwell case with `GepRegressor` (`--config dhc\|nodhc`, `--scaling`, all noise levels) |
| `maxwell_tensor_shadow.jl` | Maxwell case with `GepTensorRegressor` (`--scaling`, `--units`; `SHADOW_ROOT` selects the source tree) |
| `reynolds_gep.jl` | Reynolds case incl. the sub-sampling study |
| `dsmc_gep.jl` | constitutive relation from the DSMC data |
| `run_site_reference.py` | runs the authors' scripts (from a copy, with the edits listed in its docstring) and records their timing |
| `measure_precompile.jl` | precompilation / load / JIT / steady-state breakdown |
| `measure_memory.py`, `measure_allocation.jl` | peak memory of SITE and both Julia paths; allocation per `fit!` against the data length |
| `summarize.py`, `make_figures.py` | tables and the ten paper-ready figures |
| `run_all.sh` | the full matrix, strictly sequential and resumable |
| `results_pre_refactor/` | the same matrix measured before the reproducibility fix and the evaluator refactor |
