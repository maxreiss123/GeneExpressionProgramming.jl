# Coefficient Tuning Against an Expensive Loss

With a CFD simulation as the cost function, every trial value of the constants of a model costs a simulation. A `ScreenedNelderMead` tunes them with few loss calls, inside a search or on their own, in one of three ways:

- **Nelder-Mead** (`screen = false`): the steps of `Optim.NelderMead()`.
- **Screened Nelder-Mead** (the default): a Gaussian process over the constants, fitted on the scored ones near the best, places every loss call at the minimum of its optimistic bound `mean - kappa * deviation` inside a trust region around the best constants. The region grows after improvements and shrinks after failures; once it is small, Nelder-Mead finishes on the loss.
- **Screened swarm** (`swarm_box`): a particle swarm explores a box first, one loss call per iteration at the position a Gaussian process ranks best, and the screened Nelder-Mead finishes once the swarm stalls.

The best constants are always ones the loss scored.

## Which One to Use

| the loss | use | measured |
| --- | --- | --- |
| smooth, and Nelder-Mead needs many steps | `ScreenedNelderMead()` | 1 % of the initial error after 14 to 35 loss calls on Lotka-Volterra calibrations of 2 to 4 coefficients, against 22 to 66 for Nelder-Mead; on quadratics in 5 to 8 constants after 53 to 102, where Nelder-Mead needed 107 or more than 150 |
| easy, or a narrow, curved valley | `ScreenedNelderMead(screen=false)` | Nelder-Mead converging further within the same calls: 0.01 % of the initial error after 32 calls on logistic growth (screened: 46), 49 on Rosenbrock (108), 78 on a quadratic of condition 100 (111); the fictive CFD closure calibration below |
| several minima in a known range | `ScreenedNelderMead(swarm_box=(lower, upper))` | an oscillator whose loss has minima 12 % apart: 1 % after 50 calls, against 94 (Nelder-Mead) and 98 (screened), and the global minimum from all 8 starts, the screened Nelder-Mead from 6; on smooth losses, 1.2 to 3.4 times the calls of the screened Nelder-Mead |

The numbers are medians over 8 starts with 150 loss calls (`benchmark/screened_constants.jl`, `benchmark/swarm_variants.jl`). Taken apart, the screening is what makes a swarm affordable (a swarm that scores every particle reached 1 % on 4 of the 13 problems, the screened one on 6), and the Nelder-Mead after it is what makes it converge. Skipping the trial points of Nelder-Mead that a process deems hopeless, the obvious alternative, saved no loss calls: those points lie outside the simplex, where the process knows least. The process costs milliseconds per loss call.

## Inside a Search

```julia
fit!(regressor, epochs, population_size, loss;
    constant_optimizer=ScreenedNelderMead(max_evaluations=30),   # loss calls per tuning
    optimization_epochs=5)
```

Every `optimization_epochs` epochs, if the best model has improved, its constants are tuned against the loss. One value per occurrence of a constant in the karva string is kept in the chromosome's `optimised_constants`, and the loss sees them only through an evaluation that applies them: `elem(ctx)`, `split_predict(elem, ctx, k)` or, for a `GepTensorRegressor`, `predictT(regressor, elem)` (not `elem.expression_raw`, the karva string alone); `equation_string` and `split_equations` print them. Several objectives are tuned by their mean, the order of the population. Since every model has constants of its own, a swarm takes a box relative to them: `swarm_box = r` spans `r` initial steps around each constant, about ±50 % of it for `r = 1`.

With a `surrogate`, only a model the loss has scored is tuned, and the tuning runs next to the screening, not through it. It calls the loss itself and places those calls by its own process and its own optimistic bound (`kappa`), so the acquisition of the screening (`GpScreen`) and `failure_above` do not apply to it. The screening does not archive or count these calls either: your own count of the loss calls minus `surrogate.evaluated_count` is what the tuning took. In `examples/Main_screened_constants.jl`, that was 58 of 492 solver calls. With `GpScreen(acquisition = :logei)` instead of the default, the search returned the same model after 607 calls, the tuning again taking 58.

## On Their Own

```julia
# the constants of one model: kept if they beat the ones it holds
result = optimize_constants!(model, loss; method=ScreenedNelderMead(max_evaluations=80))
model.optimised_constants, model.fitness

# any function of a vector, e.g. the coefficients of a closure in a simulation
result = simplex_search(p -> simulation_error(p), p0;
    method=ScreenedNelderMead(max_evaluations=100, swarm_box=(lower, upper)))
result.minimizer, result.minimum, result.evaluations, result.history
```

## When the Solver Diverges

Give a run that diverges a large error, or `Inf`. The Gaussian processes see the logarithm of the loss, so a large error marks the region as bad without swamping the rest, and a failed call counts as the worst value scored, so the process learns where the solver fails. On a fictive CFD closure calibration whose solver diverges next to the minimum, the screened search converged the same with 1e3, 1e10 or `Inf` for a diverged run; capping the large values or ranking the losses did not help. The tuning therefore needs no threshold. A screened search does: set its `failure_above` at the error of a diverged run, so that the screening learns from such a run where the solver fails, rather than fitting the error as a value ([Surrogate Screening](surrogate-screening.md#A-Loss-That-Diverges)).

## Example: a Closure with Fictive CFD in the Loop

`examples/Main_fictive_cfd_in_the_loop.jl` searches for a turbulence closure ``\nu_t(y, S, u_\tau, \nu)`` with a fictive CFD solver in the loop: screened with the settings for a loss that diverges, with two objectives, physical units and its constants tuned. The solver is not an actual CFD run but a stand-in for one that takes under a millisecond: it solves the 1D momentum balance of fully developed flow between two walls,

```math
(\nu + \nu_t(y, S))\, S = \tau(y), \qquad S = \frac{dU}{dy},
```

for a channel flow (``\tau = u_\tau^2 (1 - y/h)``) and a Couette flow (``\tau = u_\tau^2``) at the friction Reynolds number 550, by a damped fixed-point iteration that evaluates the closure at every iteration. A closure that makes ``\nu + \nu_t`` negative, not finite, or keeps the iteration from converging diverges, and the loss gives that flow the error 1e3. The reference profiles come from van Driest's mixing length.

The features carry their SI units, keyed by the names in `entered_features` (the regressor warns about a key that names no feature), and the search is held to the unit of ``\nu_t``:

```julia
units = Dict{Symbol,Vector{Float16}}(          # SI exponents [kg, m, s, K, mol, A, cd]
    :y => Float16[0, 1, 0, 0, 0, 0, 0],         # wall distance, m
    :S => Float16[0, 0, -1, 0, 0, 0, 0],        # shear rate, 1/s
    :u_tau => Float16[0, 1, -1, 0, 0, 0, 0],    # friction velocity, m/s
    :nu => Float16[0, 2, -1, 0, 0, 0, 0])       # viscosity, m^2/s
regressor = GepRegressor(4; entered_features=[:y, :S, :u_tau, :nu],
    entered_non_terminals=[:+, :-, :*, :/, :exp], considered_dimensions=units,
    gene_count=2, head_len=6, number_of_objectives=2, rounds=4)

# one objective per flow: the relative error of its velocity profile, 1e3 if it diverged;
# the solver evaluates elem(ctx), which applies tuned constants, at every iteration
function loss(elem, validate::Bool)
    isnan(mean(elem.fitness)) || validate || return
    local ctx = ctxs[Threads.threadid()]
    local S = ctx.nodes[s_sym]
    local diverged = solve!(S, () -> try elem(ctx) catch; nothing end)
    elem.fitness = flow_errors(S, diverged)
end

# the settings for a loss that diverges: a diverged run is a failure for the screening, not
# a value its processes fit, and the batch acquisition weighs every candidate by its chance
# to converge
surrogate = SurrogateScreening(regressor, probes; individuals_per_epoch=0.15, seed=1,
    failure_above=1e3, screen=GpScreen(acquisition=:qehvi))

fit!(regressor, 100, 400, loss; surrogate=surrogate,
    target_dimension=Float16[0, 2, -1, 0, 0, 0, 0], hof=10,
    constant_optimizer=ScreenedNelderMead(max_evaluations=30), optimization_epochs=10)
```

With the script's seed, the same at any thread count, the search made 4,078 fictive CFD runs in about two minutes on 4 threads, 1,922 of which diverged and 120 of which tuned constants. It returned ``\nu_t = 0.0874\, y^2 S - 0.600\, \nu``, Prandtl's mixing length ``(\kappa y)^2 S`` with ``\kappa = 0.296`` less a viscosity correction, homogeneous in m²/s, with relative errors of 2.9e-3 (channel) and 3.1e-3 (Couette). Over seeds 1 to 6 (`benchmark/acquisitions.jl tuned`), the best mean error of the two flows was 3.0e-3 to 3.8e-3. With the default screening, the search got below 4e-3 on 2 of the seeds and ended at 1.5e-2 to 3.5e-2 on the others ([A Loss That Diverges](surrogate-screening.md#A-Loss-That-Diverges)).

The script then calibrates van Driest's mixing length with a viscosity correction, ``\nu_t = (k y D)^2 S + c \nu`` with ``D = 1 - e^{-y^+/A}``, within bounds part of which diverge (``c < -1``), from 6 starts with 100 fictive CFD runs each. Every method found the reference constants ``k = 0.41``, ``A = 26``, ``c = 0``; plain Nelder-Mead reached the smallest median error (1.3e-7), and the swarm solved into the diverging part 13 times without harm. This loss is one valley, where the swarm brings nothing.

`examples/Main_screened_constants.jl` is the smaller example: the constants of a search for an ODE tuned in the loop, and four ODE coefficients calibrated with and without screening.

## Running the Examples

```bash
julia --project=. --threads=4 examples/Main_screened_constants.jl
julia --project=. --threads=4 examples/Main_fictive_cfd_in_the_loop.jl
```

---

*Continue to the [API Reference](../api-reference.md#Constants-Against-an-Expensive-Loss) for every option.*
