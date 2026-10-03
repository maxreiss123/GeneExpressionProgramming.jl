# Results: PDE discovery — GEP.jl vs PDE-FIND and SITE

The four systems PDE-FIND (Rudy et al., Sci. Adv. 2017) was validated on — **heat,
Burgers, Korteweg–de Vries (KdV) and Kuramoto–Sivashinsky (KS)** — solved
pseudo-spectrally (`generate_data.py`), corrupted with additive noise of standard
deviation σ·std(u), and differentiated with Savitzky–Golay filters (`pde_common.py`).
Every method regresses u_t on the same feature matrices [u, u_x, u_xx, …], so the
comparison isolates the regression stage. The methods: GEP.jl in three routes
(pointwise, vectorised, and vectorised with units), PDE-FIND's sparse regression, and
SITE (Chen et al., J. Fluid Mech. 2025) with and without its dimensional check. One seed
per cell.

> **Provenance note.** This field-standard suite stands in for the benchmark of
> arXiv:2602.11630, whose paper could not be retrieved when these harnesses were written;
> its systems and metric can be added to the same harness.

**Metric: functional recovery.** A method's fitted right-hand side is evaluated on clean
derivative features at 20,000 grid points and compared with the analytic right-hand
side; R² > 0.99 counts as recovered. This scores the discovered operator, not its fit to
the noisy targets. The test points are drawn independently of the 5,000 training points,
not held out from them, so the two samples overlap; the score compares operators on clean
features and does not measure generalisation to unseen points. A second metric, term
recovery, checks the structure: whether a model holds the right terms (below).

## Functional-recovery R² (bold = recovered)

| PDE | σ | GEP.jl | GEP.jl (vector) | SITE | PDE-FIND | GEP.jl (vector+units) | SITE (units) |
|---|---|---|---|---|---|---|---|
| heat | 0 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| heat | 0.01 | **0.9993** | **0.9994** | **0.9993** | **0.9998** | **0.9998** | **0.9998** |
| heat | 0.05 | 0.9569 | 0.9566 | 0.9556 | 0.9560 | 0.9615 | 0.9615 |
| burgers | 0 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| burgers | 0.01 | **0.9973** | **0.9981** | **0.9972** | **0.9985** | **0.9991** | **0.9991** |
| burgers | 0.05 | **0.9927** | **0.9909** | 0.9896 | **0.9940** | **0.9984** | **0.9984** |
| kdv | 0 | **0.9999** | **0.9999** | **0.9999** | **1.0000** | **1.0000** | **1.0000** |
| kdv | 0.01 | 0.9234 | 0.9173 | 0.9278 | 0.9450 | 0.9719 | 0.9719 |
| kdv | 0.05 | 0.8775 | 0.8140 | 0.8709 | 0.8963 | 0.9858 | 0.9858 |
| ks | 0 | **0.9996** | **0.9996** | **0.9996** | **0.9996** | **0.9998** | **0.9998** |
| ks | 0.01 | -1.6996 | -1.9952 | -1.9355 | -2.1581 | -0.2167 | -0.2167 |
| ks | 0.05 | 0.5980 | 0.5503 | 0.6135 | -0.1837 | 0.2519 | 0.2519 |

GEP.jl's three routes, PDE-FIND and SITE with units recover the same seven cells: the
four clean ones, heat at σ = 0.01 and Burgers at σ = 0.01 and 0.05. SITE without units
recovers six; it misses Burgers at σ = 0.05, at 0.9896.

**The two unit-constrained columns are equal by construction, not by coincidence.** Under
the units only u·u_x, u_xx, u_xxx and, on KS, u_xxxx are admissible (see the
unit-constrained route). Both searches fit one least-squares weight per gene on the same
rows, and in every cell both winners' genes span all admissible terms, so both models are
the unique least-squares fit over them: the admissible-terms reference fit below. Expanded
into monomials, the two winners and that fit agree coefficient by coefficient to the six
printed digits in all 12 cells (`check_results.py`). These columns therefore show what the
library the units leave can do, not how the two searches compare; without units SITE and
GEP.jl find different models, and the other columns differ accordingly.

Every entry was checked independently of the harnesses (`check_results.py`): each
recorded winner, re-evaluated on the clean test features, reproduces its R² to within
7·10⁻⁵ (1.3·10⁻³ for PDE-FIND, whose expressions are printed to four decimals), and no
recovered mark changes.

## Term recovery (`term_recovery.py`)

R² scores the operator's values, not its structure: a model can clear 0.99 with spurious
terms that nearly cancel on the data, or miss it with the right terms under biased
coefficients. Here each winner is expanded into monomials of u, u_x, …, and a term counts
as found if its share, the root mean square of the term over the clean test points
relative to that of the true right-hand side, is at least 1 %. ✓: exactly the true
terms; −: a true term missing; +n: n spurious terms, the largest share in brackets.

| PDE | σ | GEP.jl | GEP.jl (vector) | SITE | PDE-FIND | GEP.jl (vector+units) | SITE (units) |
|---|---|---|---|---|---|---|---|
| heat | 0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| heat | 0.01 | +6 (12 %) | +7 (10 %) | +8 (11 %) | ✓ | ✓ | ✓ |
| heat | 0.05 | +7 (63 %) | +5 (70 %) | +8 (113 %) | +2 (80 %) | +2 (4.2 %) | +2 (4.2 %) |
| burgers | 0 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| burgers | 0.01 | +7 (4.2 %) | +4 (7.2 %) | +4 (4.8 %) | +6 (26 %) | +1 (3.3 %) | +1 (3.3 %) |
| burgers | 0.05 | +9 (28 %) | +6 (25 %) | −u_xx +11 (46 %) | +6 (68 %) | +1 (2.3 %) | +1 (2.3 %) |
| kdv | 0 | +1 (1.5 %) | ✓ | ✓ | +3 (3.2 %) | ✓ | ✓ |
| kdv | 0.01 | +3 (39 %) | +3 (45 %) | +3 (32 %) | +6 (110 %) | +1 (1.7 %) | +1 (1.7 %) |
| kdv | 0.05 | −u·u_x +5 (176 %) | +4 (151 %) | −u·u_x +5 (177 %) | +6 (100 %) | +1 (1.7 %) | +1 (1.7 %) |
| ks | 0 | +1 (1.6 %) | +1 (1.2 %) | ✓ | +2 (2.4 %) | ✓ | ✓ |
| ks | 0.01 | +3 (189 %) | +4 (130 %) | +4 (135 %) | +2 (156 %) | +1 (2.8 %) | +1 (2.8 %) |
| ks | 0.05 | +3 (308 %) | +3 (294 %) | +3 (319 %) | +4 (136 %) | +1 (3.1 %) | +1 (3.1 %) |

Exact term sets out of 12 (mean true positivity ratio TP / (TP + FN + FP)) at three
share thresholds:

| method | 0.1 % | 1 % | 5 % |
|---|---|---|---|
| GEP.jl | 1 (0.38) | 2 (0.47) | 5 (0.59) |
| GEP.jl (vector) | 1 (0.39) | 3 (0.52) | 4 (0.58) |
| SITE | 1 (0.33) | 4 (0.51) | 5 (0.59) |
| PDE-FIND | 2 (0.45) | 3 (0.53) | 5 (0.63) |
| GEP.jl (vector+units) | 4 (0.74) | 5 (0.79) | 12 (1.00) |
| SITE (units) | 4 (0.74) | 5 (0.79) | 12 (1.00) |

* **Clean data: the right terms, with small extras.** Every method finds every true term.
  SITE's four clean models and those of both unit-constrained routes hold nothing else;
  GEP.jl's pointwise route carries one extra term of 1.5–1.6 % on KdV and KS, its vector
  route one of 1.2 % on KS, PDE-FIND two or three of up to 3.2 %.
* **Noisy data without units: never quite the right set.** GEP.jl's two unconstrained
  routes, SITE and PDE-FIND hold spurious terms in every noisy cell, PDE-FIND on heat at
  σ = 0.01 excepted, the largest of them at 4 % to over 300 % of the right-hand side:
  large terms that nearly cancel. That includes cells scored as recovered: on heat at
  σ = 0.01 the pointwise route reaches R² = 0.9993 with six spurious terms, the largest
  (u_x²) at 12 %. The pointwise route and SITE also lose u·u_x on KdV at σ = 0.05, SITE
  u_xx on Burgers at σ = 0.05.
* **With units: every true term, in every cell.** Both unit-constrained routes keep all
  true terms in all 12 cells, and their only extras are the other admissible terms
  (u_xxx, u_xx or u·u_x) at 1.7–4.2 %, which the least-squares fit over the admissible
  terms gives some weight. Above a 5 % threshold both hold exactly the true terms in all
  12 cells, including the five where R² misses the bar: there the structure is right and
  the coefficients are biased by the derivative features (see the KS section).
* **The threshold moves the counts, not the ranking.** At 0.1 % few models are exact,
  since least squares leaves every candidate term some weight; at 5 % the unconstrained
  methods reach 4 or 5 of 12 and the unit-constrained ones 12. At every threshold the
  unit-constrained routes lead, and GEP.jl's unconstrained routes, SITE and PDE-FIND are
  level with one another.

## Wall-clock per cell (s)

| Method | Threads | Median | Range | Total |
|---|---|---|---|---|
| GEP.jl | 4 | 5.7 | 5.3–6.3 | 68.7 |
| GEP.jl | 1 | 10.7 | 9.8–11.6 | 128.9 |
| GEP.jl (vector) | 4 | 5.1 | 4.7–5.4 | 61.0 |
| GEP.jl (vector) | 1 | 10.2 | 9.3–10.5 | 121.1 |
| GEP.jl (vector+units) | 4 | 9.9 | 8.7–10.6 | 117.7 |
| GEP.jl (vector+units) | 1 | 19.8 | 17.0–21.6 | 234.4 |
| SITE | 1 | 433.3 | 368.9–522.7 | 5228.6 |
| SITE (units) | 1 | 202.0 | 190.2–242.6 | 2511.8 |
| PDE-FIND | 1 | 0.23 | 0.09–0.30 | 2.7 |

All runs on one 4-core machine, one at a time. Each GEP route ran on 4 threads and on
one. The pointwise and vector routes find the same models with either; the unit route's
repair draws from task-local random streams laid out by the thread count, so on one
thread its models differ in seven cells, with R² within 2·10⁻¹⁰ of the 4-thread run's.
SITE runs in one Python process and is serial, as released; PDE-FIND is a numpy linear
solve. A time covers the search and the pick among its best models; loading the data,
Julia's compilation (a warm-up run precedes the timed ones) and Python's imports are
excluded, and the setup left outside the timed region (a regressor, SITE's initial
population) takes under 0.3 s. The searches, GEP.jl's routes and SITE, get the same
budget of 200,000 candidates per cell; SITE ran all its 125 generations in every cell,
never reaching its tolerance of 1e-6. Repeated runs vary little: two runs of the vector
route on 4 threads both had a median of 5.07 s, their cells differing by up to 8 %.

* **Per core, GEP.jl's routes are 41 to 43 times faster than SITE, and 10 times with
  units.** On one thread the pointwise and vector routes take 10.7 s and 10.2 s per
  cell (medians) against SITE's 433 s, and the unit route 19.8 s against SITE's 202 s
  with units. On 4 threads the three routes take 5.7, 5.1 and 9.9 s. PDE-FIND is faster
  again (0.23 s): it runs 48 sparse regressions where the searches score 200,000
  candidates.
* **SITE's time goes to its Python machinery.** In a profile of 6 generations on heat
  at σ = 0.01, scoring individuals takes 56 % of the search: the regression step, which
  evaluates every gene from its string (37 %), and the dimension check, which evaluates
  it again (17 %, even when every dimension is zero). Copying offspring takes another
  17 %, and its tournaments of 200 take 15 %.
* **With units SITE takes half as long** (202 s median): an individual that fails its
  dimension check skips the regression. GEP.jl's unit route, which repairs individuals
  rather than rejecting them, takes twice as long as its vector route: in a profile on
  heat at σ = 0.01 the repair (`correct_genes!`) takes about as long as scoring.
* **The same models as before, in a fraction of the time.** The previous run, at
  206d046, took 168.6 s per cell for the vector route and 172.1 s with units, and 19.6 s
  for the pointwise route (4 threads; 38.6 s on one): on 4 threads against SITE's one,
  the vector routes were only 2.6 and 1.2 times faster. The rerun reproduces every R²
  and expression, and full searches on all three routes give the same fitness
  histories, best models and gene weights, bit for bit, on the old code and the new. The
  vector routes' time had gone to `predictT_scaled`, which evaluated every gene with the
  generic stack machine (a dynamic lookup per symbol, a norm check per terminal) into
  newly allocated columns; it now evaluates `Vector{Float64}` columns with the compiled
  evaluator into per-thread design matrices, and the harnesses' losses keep their
  prediction in a per-thread buffer (`predictT_scaled!`). Every route also stopped
  recounting the elites' symbols for each offspring's gene averaging, and the
  bookkeeping between evaluations (fitness-cache keys, karva strings, mutation donors,
  the population sort, the repair's comparison of dimensions) no longer builds
  throwaway strings, arrays or chromosomes.

## Reference fits (`oracle.py`)

Two least-squares fits show what the features allow. Both use the noisy training data
every method gets — the first 80 % of the training rows, on which the GEP harnesses fit
their gene weights — and are scored like the methods:

* **true terms**: the terms of the analytic right-hand side, refitted;
* **admissible terms**: every monomial with the dimension of u_t under the units of the
  unit-constrained route (below): u·u_x, u_xx, u_xxx and, where the features go that far
  (KS), u_xxxx.

| PDE | σ | true terms | admissible terms |
|---|---|---|---|
| heat | 0 | 1.0000 | 1.0000 |
| heat | 0.01 | 0.9998 | 0.9998 |
| heat | 0.05 | 0.9619 | 0.9615 |
| burgers | 0 | 1.0000 | 1.0000 |
| burgers | 0.01 | 0.9998 | 0.9991 |
| burgers | 0.05 | 0.9987 | 0.9984 |
| kdv | 0 | 1.0000 | 1.0000 |
| kdv | 0.01 | 0.9724 | 0.9719 |
| kdv | 0.05 | 0.9862 | 0.9858 |
| ks | 0 | 0.9998 | 0.9998 |
| ks | 0.01 | -0.2167 | -0.2167 |
| ks | 0.05 | 0.2525 | 0.2519 |

The five cells no method recovers — heat at σ = 0.05, KdV at σ = 0.01 and 0.05, KS at
σ = 0.01 and 0.05 — are exactly the cells where refitting the true terms also misses the
bar (R² 0.962, 0.972, 0.986, −0.22 and 0.25). There the derivative features, not the
regression stage, set the limit.

## Reading

* **Clean data: every method recovers the operator.** The pointwise route's winners
  contain every true term with coefficients within 0.4 % of the truth — heat
  `0.0999999·u_xx`; Burgers `−1.00094·u·u_x + 0.100033·u_xx`; KdV `−6.02067·u·u_x`
  (written `6.02067·u·(u_xxx − (u_x + u_xxx))`) and `−1.00037·u_xxx`; KS `−0.9977·u_xx`,
  `−0.9986·u_xxxx` and `−0.9991·u·u_x`, summed over three genes — next to residual genes
  with small weights. The vector route's winners hold the same terms at coefficients
  within 0.1 % of the truth, SITE's within 0.3 % (KdV: `−6.01625·u·u_x − 0.999129·u_xxx`).
* **Noisy data: the methods fail together.** Every method misses the same five cells,
  and at each of them the true terms refitted miss too (above): recovering the right
  structure would not clear the bar there either. SITE also misses Burgers at σ = 0.05
  (0.9896), where the true terms refitted reach 0.9987: a miss of its search, not of the
  features.
* **KS at σ = 0.05: a better fit, not the operator.** The pointwise and vector routes
  and SITE score 0.60, 0.55 and 0.61, against 0.25 for the true terms refitted and −0.18
  for PDE-FIND. They get there with terms other than the KS operator's (the winners are
  in `results/`), and none clears the bar.
* **What this suite cannot show.** The four right-hand sides lie in PDE-FIND's candidate
  library by construction. Right-hand sides outside a fixed library (rational or composed
  terms) are where a symbolic search can differ from sparse regression; they are not
  tested here.
* **What would lift the failing cells** is better derivative features, for every method
  alike: a spatial window chosen per noise level (it lifts the true-terms fit on KS at
  σ = 0.01 from −0.22 to 0.74, see below), or weak-form targets, which integrate the
  candidate terms against smooth test functions so that the derivatives fall on those
  (as in SPIDER, JFM 996 A25). Neither is implemented here.

## The vectorised route (`pde_gep_tensor.jl`)

Same conditions, split and metric; the engine is `GepTensorRegressor`. Each feature is
one column over all training samples, the fitness is a loss callback over those columns
(the interface of `examples/Main_streaming_chunks.jl`), and `predictT_scaled` solves one
least-squares weight per gene. Both GEP routes fit such gene-wise weights, so neither
searches coefficients: recovering `0.1·u_xx` means finding the term u_xx. The pointwise
route also has constant terminals (0, 0.5 and one random constant), the vector route
none. Scalar fields are the one-component case of the machinery that fits vector- and
tensor-valued equations.

It recovers the same seven cells as the pointwise route; at the noisy cells the two
differ by up to 0.3 in R² (KS at σ = 0.01: −1.70 and −2.00; KdV at σ = 0.05: 0.88 and
0.81). Per cell it costs about what the pointwise route costs (5.1 s and 5.7 s on
4 threads); in the previous run it cost 8.6 times as much, spent in `predictT_scaled`
(see the wall-clock notes).

## The unit-constrained route (`pde_gep_units.jl`)

The fields get physical units — u [m/s], x [m], t [s] — under which the textbook forms are
inhomogeneous, so three constants with units enter as all-ones feature columns: ν₂
[m²/s], ν₃ [m³/s] and ν₄ [m⁴/s]. The target is u_t [m/s²]. `considered_dimensions` and
`target_dimension` switch on the repair (`correct_genes!`) and the gate in `fit!`: an
individual that is not homogeneous after the repair is not scored. With `+` and `-` as
the only gene connectors every gene must carry the target dimension, so the gene-wise
weights are dimensionless.

Every terminal carries 1/s, so a monomial of dimension m/s² has exactly two factors, and
the only such monomials are u·u_x, ν₂·u_xx, ν₃·u_xxx and ν₄·u_xxxx. With the ν columns of
ones, every admissible model is a linear combination of u·u_x, u_xx, u_xxx and u_xxxx, and
the admissible-terms fit above is the best of them on the fitting rows. The spurious
u·u_xxx term (dimension 1/(m·s²)) that PDE-FIND selects for KS at σ = 0.01 is excluded.

1. **The constraint binds.** All 12 winners are homogeneous gene by gene. In the
   previously recorded run all 12 were not: the repair ran, but inhomogeneous individuals
   were still scored.
2. **The search finds the admissible-terms fit.** Its R² equals that fit's, to four
   decimals, in all 12 cells. It recovers the same seven cells as GEP.jl's other routes
   and PDE-FIND; at
   every noisy cell it comes within 0.001 of the true terms refitted, and it scores above
   every unconstrained method (tied with PDE-FIND on heat at σ = 0.01) except on KS at
   σ = 0.05, where the unconstrained searches fit better with terms outside the KS
   operator. On KS at σ = 0.01 it reaches −0.22, the value of the true terms refitted:
   excluding u·u_xxx removes the selection error, not the feature error.
3. **SITE finds the same fit, as it must.** Its dimensional check, under the same units,
   leaves the same admissible terms; its four host genes span them in every cell, and its
   per-gene least squares (TLR) on the same rows then yields the same projection. Its
   coefficients match this route's and the admissible-terms fit's to the six printed
   digits, and its R² matches this route's to within 2·10⁻¹⁰ (`check_results.py`). The
   equality follows from the constrained problem, a linear regression on three or four
   terms that both searches solve exactly; it does not show that the two searches behave
   alike.
4. **What this shows.** On this suite the units leave three or four candidate terms, and
   they include every true term. The gain over the unconstrained routes is that library's:
   least squares or PDE-FIND restricted to the same terms shares it. What the constraint
   contributes is deriving the library from the units instead of choosing it; its effect
   where the right-hand side lies outside such a library is not tested here.
5. **Cost.** The median time per cell is 1.9 times the vector route's on 4 threads
   and 2.0 times on one: the repair costs about as much as scoring (SITE: see the
   wall-clock notes).

## SITE (`site_baseline.py`)

SITE (T. Chen, H. Yang, W. Ma and J. Zhang, *J. Fluid Mech.* 1024, A34, 2025) identifies
tensor equations with a host–plasmid chromosome: host genes over tensor terminals, `p_`
nodes that scale a tensor by a scalar expression evolved in a plasmid, a dimensional
check that gives an inhomogeneous individual a loss of 1000, and tensor linear
regression (TLR) for one coefficient per host gene, the counterpart of the GEP routes'
gene-wise least squares. `site_baseline.py` imports the released `SITE.py` unmodified
([BUAA-MARS-group/SITE](https://github.com/BUAA-MARS-group/SITE) at 32ab7c9) and poses
each cell as SITE's one-component case: every feature enters the host as a 1×1 tensor
(`U`, `Ux`, …) and the plasmids as a scalar (`u`, `ux`, …), next to the identity `delta`.
The operators are the {+, −, ×} of the GEP routes (tensor sum, difference and inner
product and `p_` on the host; sum, difference and product on the plasmids) plus SITE's
random numerical constants. The other settings are the authors' Maxwell configuration
(head lengths 5 and 10, four host genes, tournaments of 200, 100 alien individuals,
tolerance 1e-6), with a population of 1600 over 125 generations: the 200,000 candidates
the GEP routes generate. The split, the pick among the four best models and the metric
are the GEP harnesses'. `--units` runs the dimensional check under the units of the
unit-constrained route (above); without it every dimension is zero, the authors' way of
switching the check off. The harness defines the tensor operators, as SITE's case scripts
do, with numpy's batched matrix product for the inner product: the same values, without
a Python loop over the 4,000 fitting rows.

1. **Without units SITE lands with the other unconstrained methods.** It recovers six
   cells, the seven of the others less Burgers at σ = 0.05, which it misses at 0.9896
   against their 0.9909–0.9940. At the other noisy cells its R² lies within the range of
   GEP.jl's two unconstrained routes and PDE-FIND, or at most 0.0004 below it, except on
   KS at σ = 0.05, where it scores highest (0.61, with terms outside the KS operator).
2. **With units it finds the same fit as GEP.jl** (see the unit-constrained route).
3. **Caveats.** SITE is built for tensor equations; on scalar fields its host–plasmid
   split has nothing to separate, so this suite tests its search, its regression and its
   dimensional check, not what sets it apart. Its settings are the authors' Maxwell
   settings, not tuned for these cells. Its expression printer (`my_compile`) can pair a
   gene's plasmids with the wrong `p_` nodes when the gene holds more than one: in a
   sample of 1040 genes per variant, 7 of the 37–38 genes with two or more. The fitness,
   the regression and the R² above do not depend on the printer; the expressions in
   `results/` are built by the harness in the evaluation order and checked against the
   scored predictions.

## Why mid-noise scores below high noise on KS

Every method scores lower on KS at σ = 0.01 than at σ = 0.05 (pointwise route −1.70 and
0.60, SITE −1.94 and 0.61, PDE-FIND −2.16 and −0.18), so the cause lies in the features.
The mechanism, from the true-terms fit (`oracle.py`, coefficients w for u_xx, u_xxxx and
u·u_x, all −1 in the operator):

1. **The KS right-hand side is a small residual of large terms.** On the clean test
   features the three terms have standard deviations 1.02, 1.78 and 1.12, their sum u_t
   0.30. R² against u_t therefore punishes coefficient errors that differ between terms
   far more than a common shrinkage.
2. **At σ = 0.01 the spatial window, chosen for σ = 0.05 (31 points), biases the terms
   unevenly.** The true terms refitted get w = (−0.70, −0.76, −0.51): the product term
   shrinks more than the linear ones, the cancellation breaks, and R² = −0.22 with the
   exact structure. Selection adds to that: PDE-FIND's model also holds the spurious term
   u·u_xxx, and deleting that one term moves its R² from −2.16 to 0.05.
3. **At σ = 0.05 the coefficients shrink further but more evenly**: w = (−0.47, −0.52,
   −0.37), R² = 0.25.
4. **The σ = 0.01 cell is not intrinsically hard.** With a 21-point spatial window the
   fit gets w = (−0.52, −0.54, −0.51), nearly uniform, and R² = 0.74 (15 points: 0.17;
   25 points: 0.54). The inversion comes from one window per noise regime; the protocol
   applies to every method alike, so the comparison stands.

## Reproducing

From the repository root (Python 3 with numpy; for SITE also scipy, sympy, geppy and
deap, and a clone of its repository; Julia with this repository's environment):

```bash
python paper/pdebench/generate_data.py      # reference fields -> data/*.npz, data/meta.json
python paper/pdebench/pde_common.py         # shared features -> data/conditions.json
python paper/pdebench/pdefind_baseline.py   # -> results/pdefind.json
julia --project=. --threads=4 paper/pdebench/pde_gep.jl          # -> results/gep.json
julia --project=. --threads=1 paper/pdebench/pde_gep.jl gep_1thread.json
julia --project=. --threads=4 paper/pdebench/pde_gep_tensor.jl   # -> results/gep_tensor.json
julia --project=. --threads=1 paper/pdebench/pde_gep_tensor.jl gep_tensor_1thread.json
julia --project=. --threads=4 paper/pdebench/pde_gep_units.jl    # -> results/gep_units.json
julia --project=. --threads=1 paper/pdebench/pde_gep_units.jl gep_units_1thread.json
git clone https://github.com/BUAA-MARS-group/SITE /path/to/SITE  # used at 32ab7c9
python paper/pdebench/site_baseline.py /path/to/SITE             # -> results/site.json
python paper/pdebench/site_baseline.py /path/to/SITE --units     # -> results/site_units.json
python paper/pdebench/make_table.py         # the two tables above
python paper/pdebench/oracle.py             # the reference fits and the KS diagnostics
python paper/pdebench/check_results.py      # every R² from its recorded model; the unit
                                            # winners against the admissible-terms fit
python paper/pdebench/term_recovery.py      # the term-recovery tables
```

The GEP harnesses seed Julia's global RNG. With 4 threads, reruns on a second machine
reproduced every R² and expression of the pointwise route and every R² of the vector
route; one thread gives the pointwise and vector routes' models unchanged, and the unit
route's up to its repair's random streams (see the wall-clock notes). The three GEP
routes, rerun on 4 threads at 097d40e, after the homogeneity and scheduling changes on
main (2ee8b7d, 1f7f583) and the serial re-scoring that keeps seeded runs as they were
(466d6bf), reproduced every R² and expression. SITE seeds Python's and numpy's generators;
reruns reproduced every R² of both SITE variants. Wall-clock times vary with the machine.
The SITE runs used Python 3.11 with numpy 2.4, scipy 1.17, sympy 1.14, geppy 0.1.3 and
deap 1.4.

The results recorded here before came from older code: the vector and unit routes' at
c59b5c0, the pointwise route's at a893d3d (whose numbers had not reached this page). The
pointwise and PDE-FIND reruns reproduced them exactly. The vector route's R² changed at
five noisy cells (KdV at σ = 0.05: 0.934 → 0.814), and the unit route's in all but two
clean cells, since its check now binds.
