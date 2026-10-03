# Expensive Losses: Surrogate Screening

When a single loss call is expensive -- a solver in the loop, a simulation, an external program -- an epoch spends most of its calls on individuals that are visibly not worth one. A `SurrogateScreening`, passed to `fit!` as `surrogate`, spends the loss on a few promising individuals per epoch and lets Gaussian processes predict the loss of the others. It is the Julia counterpart of `gep.SurrogateBatchStrategy` of the Python package, with the defaults measured there.

This page screens a search whose loss integrates an ODE, then a search with two expressions and two objectives, the right-hand sides of a system of two ODEs. It closes with what the screening guarantees, the settings worth knowing, the diagnostics, and when the screening pays off. Both searches are runnable scripts: `examples/Main_surrogate_screening.jl` and `examples/Main_surrogate_multi_objective.jl`. The constants of a model, screened the same way, have a page of their own: [Coefficient Tuning](coefficient-tuning.md).

## How the Screening Works

Each epoch, the new individuals (one per karva string) go through four steps:

1. **Embedding.** The expression is evaluated on a small, fixed probe set, and its outputs (asinh-transformed) are the latent vector of the individual. Individuals that behave alike lie close together, however differently they are written. An expression that cannot be evaluated on every probe is scored as a crash, without a loss call.
2. **Prediction.** One Gaussian process per objective maps the latent vectors of the individuals scored so far to their losses, on a log10 scale.
3. **Selection.** An acquisition, by default the optimistic bound `mean - kappa * deviation`, picks the individuals the loss scores: `individuals_per_epoch` of them, a tenth of them at random instead, to explore.
4. **Imputation.** Every other individual receives the prediction of the processes, one float step behind the best loss scored so far on each objective, and competes with it in the selection.

Until enough individuals have been scored to fit a process (the warmup: six times the batch, at least 60), an epoch scores at most twice the batch (at least 20), picked at random, and the others receive the median loss of that batch. From 10 individuals per epoch on (read on a hundred individual epoch), each epoch also breeds three times the children the population takes, and the processes pick the ones that enter.

## A Single Objective: an ODE in the Loss

The search looks for `f` in `dx/dt = f(x)`, given trajectories of the logistic equation `f(x) = x - 0.5 x^2` from eight initial values. The loss integrates `dx/dt = f(x)` with the candidate `f` (RK4, all trajectories at once) and compares the result with the data, so every loss call costs a simulation.

```julia
using GeneExpressionProgramming
using Random
using Statistics

Random.seed!(1)

# the data: logistic growth from eight initial values, sampled every 0.1 up to t = 8
f_true(x) = x .- 0.5 .* x .^ 2
x0 = collect(range(0.05, 3.5; length=8))
dt, steps, every = 0.004, 2000, 25

function rk4(f, x0, dt, steps, every)
    x = copy(x0)
    out = zeros(length(x0), steps ÷ every)
    for s in 1:steps
        k1 = f(x)
        k2 = f(x .+ 0.5dt .* k1)
        k3 = f(x .+ 0.5dt .* k2)
        k4 = f(x .+ dt .* k3)
        x .+= dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        all(isfinite, x) || return nothing
        s % every == 0 && (out[:, s÷every] .= x)
    end
    return out
end

data = rk4(f_true, x0, dt, steps, every)

regressor = GepRegressor(1; entered_features=[:x], entered_non_terminals=[:+, :-, :*],
    gene_count=2, head_len=4)

# one buffer context per thread, whose input column (the state x) the solver overwrites at
# every stage: the compiled program reads the column in place
ctxs = thread_contexts(regressor.toolbox_, zeros(1, length(x0)))
x_sym = only(k for (k, n) in regressor.toolbox_.nodes if n isa InputSelector)
calls = Threads.Atomic{Int}(0)

# the loss runs on several threads at once: `local` keeps its variables its own
function loss(elem, validate::Bool)
    isnan(mean(elem.fitness)) || validate || return
    Threads.atomic_add!(calls, 1)
    local ctx = ctxs[Threads.threadid()]
    function rhs(x)
        ctx.nodes[x_sym] .= x
        local y = elem(ctx)
        return y isa AbstractVector ? copy(y) : fill(NaN, length(x))
    end
    local sim = rk4(rhs, x0, dt, steps, every)
    elem.fitness = isnothing(sim) ? (Inf,) : (mean(abs2, sim .- data),)
end

# the probe states span the range the trajectories visit; the latent vector of an
# individual is its f on them
probes = reshape(collect(range(0.0, 3.5; length=24)), 1, :)
surrogate = SurrogateScreening(regressor, probes;
    individuals_per_epoch=0.15,    # 15 % of the new individuals of an epoch
    seed=1)

fit!(regressor, 40, 200, loss; surrogate=surrogate)

best = regressor.best_models_[1]
println(best, "   loss ", best.fitness[1])
println("solver calls ", calls[], ", predictions ", surrogate.imputed_count)
```

The loss is the one a search without screening would use; the screening only decides which individuals reach it. Three choices matter:

- **The probes**: one row per feature and one column per probe sample, like the training data of `fit!`. A few dozen samples from the range that matters suffice; for an ODE, the states its trajectories visit.
- **The batch**: `individuals_per_epoch` is a count (10 by default), which pins the loss calls per epoch, or a share in `(0, 1)`, which adapts them to the new individuals of an epoch.
- **The seed**: `seed` fixes the random choices of the screening, so that a seeded search reproduces.

The acquisition, which picks the individuals the loss scores, is best left at its default for this search. [Choosing the Acquisition](#Choosing-the-Acquisition) tells when to change it, and [A Loss That Diverges](#A-Loss-That-Diverges) what to set for a solver that diverges.

`examples/Main_surrogate_screening.jl` runs this search with and without the screening, from the same seed. Both find `x - 0.5 x^2` exactly; the screened search after 776 solver calls, the other after 5,423.

## Several Expressions and Objectives: a System of ODEs

The setting of the Python tutorial `ext_eval_multi_objective_screened_gep.py`: one chromosome carries two expressions, and the loss sets one objective per expression. Here the expressions are the right-hand sides of the Lotka-Volterra system

```math
\frac{dx}{dt} = x - x y, \qquad \frac{dy}{dt} = x y - y
```

`split_karva(elem, 2)` splits the four genes of a chromosome into `f` (for `dx/dt`) and `g` (for `dy/dt`), `split_predict(elem, ctx, 2)` evaluates both in a buffer context, and `split_equations(elem, 2)` prints them. The loss integrates each equation with the other state read from the data -- `f` with the simulated `x` and the measured `y`, `g` with the measured `x` and the simulated `y` -- so the error of `x` judges `f` alone and the error of `y` judges `g` alone.

```julia
using GeneExpressionProgramming
using Random
using Statistics

Random.seed!(1)

# the data: six trajectories, sampled every 0.1 up to t = 8
x0 = [0.5, 1.5, 2.0, 1.0, 0.8, 0.3]
y0 = [1.0, 0.5, 1.5, 2.0, 0.3, 0.8]
n = length(x0)
dt, steps, every = 0.01, 800, 10

# RK4 for the pair (x, y), sampled every `every` steps; `rhs(x, y, k)` also gets the time
# of the stage in half steps, k dt/2, at which the loss reads the measured states
function rk4(rhs, x0, y0, dt, steps, every)
    x, y = copy(x0), copy(y0)
    xs, ys = zeros(length(x0), steps ÷ every), zeros(length(y0), steps ÷ every)
    for s in 0:steps-1
        k1x, k1y = rhs(x, y, 2s)
        k2x, k2y = rhs(x .+ 0.5dt .* k1x, y .+ 0.5dt .* k1y, 2s + 1)
        k3x, k3y = rhs(x .+ 0.5dt .* k2x, y .+ 0.5dt .* k2y, 2s + 1)
        k4x, k4y = rhs(x .+ dt .* k3x, y .+ dt .* k3y, 2s + 2)
        x = x .+ dt / 6 .* (k1x .+ 2 .* k2x .+ 2 .* k3x .+ k4x)
        y = y .+ dt / 6 .* (k1y .+ 2 .* k2y .+ 2 .* k3y .+ k4y)
        # a candidate that blows up is not worth integrating to the end
        (all(isfinite, x) && all(isfinite, y) && maximum(abs, x) < 1e6 && maximum(abs, y) < 1e6) ||
            return nothing
        if (s + 1) % every == 0
            xs[:, (s+1)÷every] .= x
            ys[:, (s+1)÷every] .= y
        end
    end
    return xs, ys
end

lotka_volterra(x, y, _) = (x .- x .* y, x .* y .- y)
x_data, y_data = rk4(lotka_volterra, x0, y0, dt, steps, every)
# the measured states at every half step, from a run at half the step: column k + 1 holds
# the time k dt/2
x_half, y_half = rk4(lotka_volterra, x0, y0, dt / 2, 2steps, 1)
x_measured, y_measured = hcat(x0, x_half), hcat(y0, y_half)

regressor = GepRegressor(2; entered_features=[:x, :y], entered_non_terminals=[:+, :-, :*],
    gene_connections=[:+, :-, :*], gene_count=4, head_len=4, number_of_objectives=2)

# one buffer context per thread with two sets of states side by side, (x, measured y) for
# f and (measured x, y) for g: one split_predict evaluates both expressions
ctxs = thread_contexts(regressor.toolbox_, zeros(2, 2n))
feature(i) = only(k for (k, nd) in regressor.toolbox_.nodes if nd isa InputSelector && nd.idx == i)
x_sym, y_sym = feature(1), feature(2)

function loss(elem, validate::Bool)
    isnan(mean(elem.fitness)) || validate || return
    local ctx = ctxs[Threads.threadid()]
    local xs, ys = ctx.nodes[x_sym], ctx.nodes[y_sym]
    function rhs(x, y, k)
        xs[1:n] .= x
        ys[1:n] .= @view y_measured[:, k+1]
        xs[n+1:2n] .= @view x_measured[:, k+1]
        ys[n+1:2n] .= y
        local f, g = split_predict(elem, ctx, 2)
        (f isa AbstractVector && g isa AbstractVector) || return (fill(NaN, n), fill(NaN, n))
        return f[1:n], g[n+1:2n]
    end
    local sim = rk4(rhs, x0, y0, dt, steps, every)
    elem.fitness = isnothing(sim) ? (Inf, Inf) :
                   (mean(abs2, sim[1] .- x_data), mean(abs2, sim[2] .- y_data))
end

# the probe states: 36 of the (x, y) states the trajectories visit
states = vcat(vec(x_data)', vec(y_data)')
probes = states[:, randperm(size(states, 2))[1:36]]
surrogate = SurrogateScreening(regressor, probes;
    expressions=2,                  # one latent block per expression
    objective_expressions=[1, 2],   # the error of x judges f alone, that of y judges g
    individuals_per_epoch=0.15,
    seed=1)

# only what the solver has scored may stop the search, never a prediction
solved(population, epoch) =
    any(c -> is_validated(surrogate, c) && maximum(c.fitness) < 1e-10, population)

fit!(regressor, 100, 200, loss; surrogate=surrogate, hof=10, break_condition=solved)

# the front: the non-dominated members of the hall of fame, which the solver has all
# scored; an error below 1e-10 is round-off and counts as solved
best = regressor.best_models_
front = best[calculate_fronts([max.(m.fitness, 1e-10) for m in best])[1]]
for m in unique(m -> m.fitness, front)
    f, g = split_equations(m, 2)
    println(m.fitness, "   dx/dt = ", f, "   dy/dt = ", g)
end
```

What changes against a single objective:

- **`expressions = 2`** embeds a chromosome with one latent block per expression: each part `split_karva` makes, evaluated on the probes. The whole karva string joins the parts with connectors the loss never uses, so its behaviour is not what the loss sees. The gene count has to be divisible by the number of expressions.
- **`objective_expressions = [1, 2]`** lets the process of each objective see the block of the expression it judges alone; the other block would only blur its distances. On this system, the rank correlation of predictions and held-out losses went from -0.21, 0.35, 0.18 and 0.29 to 0.19, 0.35, 0.54 and 0.56 in two runs, and the screened search solved both equations on 5 of 6 seeds instead of 2. An objective that depends on all expressions takes `0` or `:all`.
- **The pick** scalarizes the objectives by a random augmented Chebyshev weighting per epoch (ParEGO); `screen = GpScreen(acquisition = :ehvi)` ranks by the expected hypervolume improvement instead, and `:qehvi` by the joint expected hypervolume improvement of the batch (see [Choosing the Acquisition](#Choosing-the-Acquisition)).
- **The break condition** asks `is_validated(surrogate, c)`, so that only a loss value, never a prediction, can stop the search.
- **The result is a front**, not one model: `best_models_` holds the `hof` best models by the mean of their objectives, all of them scored by the loss (`validate_hof`), and `calculate_fronts` picks the non-dominated ones. A large `hof` costs a loss call per predicted member at the end.

`examples/Main_surrogate_multi_objective.jl` runs this search with and without the screening, and prints every tenth epoch the best scored errors and why the solver calls of that epoch were made. With its seed, the screened search solves both equations at epoch 54 after 1,196 solver calls, the search without screening at epoch 73 after 10,168. Over seeds 1 to 6 (with other probe states), the screened search solved five, after 800 to 2,050 solver calls; over seeds 1 to 3, the search without screening solved two, after 8,850 and 10,170.

Integrating each equation against the measured other state is what makes one objective judge one expression. Integrated as a coupled system, both errors depend on both expressions, and neither the screened search nor the one without screening solved the system reliably in 100 epochs.

### Objectives Known Without the Loss

An objective the loss computes from the chromosome alone, such as the size of the model, is better computed than predicted: a Gaussian process cannot tell the size of an expression from its behaviour. `exact_objectives` maps such an objective to its function; every predicted individual receives its exact value, and the acquisition sees it without uncertainty. The function has to return exactly what the loss sets:

```julia
size_of(c) = 0.01 * length(c.expression_raw)

# the loss sets (error of x, error of y, size_of(elem)); the regressor has
# number_of_objectives=3
surrogate = SurrogateScreening(regressor, probes;
    expressions=2,
    objective_expressions=[1, 2, :all],   # the size depends on both expressions
    exact_objectives=Dict(3 => size_of))  # computed, not predicted
```

## A Loss That Diverges

A solver in the loop often gives a run that diverged a large error. The loss is then smooth where the solver converges and jumps where it does not, and a Gaussian process fitted across the jump bends around it: it smears the jump into the converged region nearby and spends its uncertainty there. `failure_above` keeps the two apart:

```julia
# the loss gives a flow whose solve diverged the error 1e3
surrogate = SurrogateScreening(regressor, probes; individuals_per_epoch=0.15,
    failure_above=1e3,                     # a diverged run is a failure, not a value
    screen=GpScreen(acquisition=:qehvi))   # weighs each candidate by its chance to converge
```

- A diverged run teaches the `FeasibilityModel` where the solver fails, as a run that returns `Inf` does; the processes learn from the converged runs alone.
- An individual the loss has not scored is predicted between its process's prediction and `failure_above`, by its probability of diverging (in the transformed units), so a model that likely diverges does not survive on an optimistic guess.
- `:qehvi` weighs each candidate's improvement by that probability inside its samples; the other acquisitions fill the batch from the candidates the model considers runnable first.
- The individuals keep the fitness the loss gave them: the threshold only decides what the screening learns from it.

Set `failure_above` at or below the error your loss gives a diverged run, and above every error a converged run can reach. With several objectives, a call fails once any of them reaches the threshold, and then teaches the processes of none. A loss that returns `Inf` for a failed run needs no threshold, as a value that is not finite always counts as failed. Nor does the constant tuning of a search, which calls the loss outside the screening ([Coefficient Tuning](coefficient-tuning.md#When-the-Solver-Diverges)).

`examples/Main_fictive_cfd_in_the_loop.jl` searches a turbulence closure with these settings: a diverged run gets the error 1e3 per flow, and the constants of the best model are tuned every 10 epochs. Over seeds 1 to 6 (`benchmark/acquisitions.jl tuned`), the best mean relative error of the two flows was 3.0e-3 to 3.8e-3. The default screening got below 4e-3 on 2 of the seeds and ended at 1.5e-2 to 3.5e-2 on the others; `failure_above` with the default acquisition got there on 3, ending at about 1e-2 to 3.5e-2 on the others. Without the constant tuning (100 epochs, 4 seeds; `benchmark/acquisitions.jl`), `failure_above = 1e3` raised the median hypervolume of everything the solver scored, in log10 of the errors below a relative error of 1 in both flows, from 2.66 to 5.71 with the default acquisition and to 5.90 with `:qehvi`. The median best errors went from 2.7e-2 (channel) and 2.6e-2 (Couette) to 2.5e-3 and 5.1e-3.

The price is more diverged runs: in the example, a median of 1,726 of about 4,100 runs over the six seeds, against 821 of about 4,200 with the default screening. A loss that stops a run as soon as it diverges keeps these runs cheap.

## Choosing the Acquisition

The acquisition is the `screen` of the screening, `GpScreen(acquisition = ...)`. What to take, as measured on the searches of this page and on the closure of `examples/Main_fictive_cfd_in_the_loop.jl`:

| the search | `screen` | measured |
| --- | --- | --- |
| one objective | `GpScreen()`, the default `:lcb`; `:logei` and `:logei_believer` otherwise | the fewest loss calls: the ODE above found `f` after 776 solver calls, after 885 with `:logei_believer` and 939 with `:logei` |
| several objectives | `GpScreen()`, or `GpScreen(acquisition = :qehvi)` | a tie: the system of two ODEs solved on 4 of 6 seeds with either; with the seed of its example after 1,196 and 1,538 solver calls (`:ehvi`: 1,241) |
| several objectives, and a solver that gives a diverged run a large error | `GpScreen(acquisition = :qehvi)` with `failure_above` at that error, never without it | the best and steadiest on the closure, with its constant tuning and without (see [A Loss That Diverges](#A-Loss-That-Diverges)); without the threshold, erratic (hypervolumes of 2.06 to 6.20 over four seeds) |

`:ehvi` and `:qehvi` rank by the hypervolume of several objectives, which one objective does not have: a search whose loss sets one refuses them. With one objective and a solver that diverges, set `failure_above` the same way; which acquisition suits that case best was not measured. The acquisition does not reach the constant tuning of a search, which places its loss calls by a process of its own ([Coefficient Tuning](coefficient-tuning.md#Inside-a-Search)), and `:qehvi` costs about as much time per epoch as `:lcb` ([When the Screening Pays Off](#When-the-Screening-Pays-Off)).

`GpScreen(acquisition = :qehvi)` is the q-expected hypervolume improvement of Daulton, Balandat and Bakshy (2020), the batch form of `:ehvi`. The candidates are sampled jointly from their processes, and the batch is filled greedily: every pick is valued by the volume it adds to the front as the earlier picks of the same sample left it, averaged over the samples. Two candidates that behave alike are correlated in every sample, so the second adds nothing once the first is in, and a candidate that will likely fail is worth its improvement times its chance to succeed. `:ehvi` imputes the posterior mean of the earlier picks instead and samples each candidate alone; `:lcb`, the default, ranks by one random Chebyshev scalarization of the optimistic bounds per epoch. The paper's differentiable form serves acquisitions optimized by gradient over a continuous space; the screening ranks a finite set of bred individuals, so every candidate is valued directly.

On the two searches of `benchmark/acquisitions.jl` (output in `benchmark/acquisitions_results.txt`):

| acquisition | two ODEs: solved of 6, median solver calls | closure: median hypervolume, best errors | closure with `failure_above = 1e3` |
| --- | --- | --- | --- |
| `:lcb` (default) | 4, 1,551 | 2.66; 2.7e-2, 2.6e-2 | 5.71; 3.4e-3, 5.4e-3 |
| `:ehvi` | 3, 1,375 | 5.01; 5.6e-3, 6.1e-3 | 4.39; 7.2e-3, 6.9e-3 |
| `:qehvi` | 4, 1,503 | 3.51; 2.1e-2, 1.7e-2 | 5.90; 2.5e-3, 5.1e-3 |

With the threshold, `:qehvi` was the best and the steadiest on the closure (5.65 to 6.44 over the seeds, the default 5.00 to 6.21), with the fewest solver calls; on the two ODEs, whose failures are non-finite and need no threshold, it tied the default. Without the threshold the jump of the loss misleads the processes: `:ehvi` held up best, and `:qehvi` swung from 6.20 on one seed to 2.06 on another. Four and six seeds show which way the acquisitions lean, no more.

## What the Screening Guarantees

- **A prediction is never cached.** A copy of a predicted individual is screened again, while a copy of a scored one takes its loss times `penalty`, even once the fitness cache has dropped it.
- **No prediction claims a best loss.** A prediction is at least one float step worse than the best loss scored so far on every objective, so it cannot beat, tie or dominate the individual holding one.
- **The best individual of an epoch is scored** before its loss is recorded, selected with or shown to `break_condition`, and the returned hall of fame is scored at the end (`validate_hof = true`). Every member of `best_models_` carries a loss value: `is_validated(surrogate, m)` is `true`.
- **With several objectives, a prediction is provisional.** The individuals that carry one are screened again every epoch, next to the new ones, so that a later process can pick them for a loss call or give them a fresh prediction (`rescreen`). Without, the population fills up with stale predictions: in a search for an ODE system, 197 of 200 survivors carried one after 50 epochs, while the best scored losses had not moved since epoch 10. With one objective re-screening was a wash on the benchmark below, and it is off unless `rescreen = true`.
- **With several objectives, the scored holders of the best values survive.** The population survives by the mean of its objectives, which a prediction can beat without dominating anyone. The scored individuals that hold the best value of an objective are therefore kept right behind the leader, and the best scored value of every objective among the survivors never gets worse.
- **Only a screened search is affected.** `fit!` without `surrogate` scores every new individual, as it always does.

In your own callbacks (a `break_condition`, a `file_logger_callback`), `is_validated(surrogate, c)` tells a loss value from a prediction.

## Settings Worth Knowing

| Keyword | Default | What it does |
| --- | --- | --- |
| `individuals_per_epoch` | `10` | individuals the loss scores per screened epoch: a count, or a share in `(0, 1)` of the new individuals |
| `embedding` | `:expression` | `:genes` embeds one block per gene (`GeneEmbedder`), for a loss that scores the least-squares combination of the genes (`linear_scaling`) |
| `expressions` | `1` | latent blocks per chromosome, one per expression of a multi-expression loss |
| `objective_expressions` | `nothing` | the expression each objective judges, one entry per objective (`0` or `:all` for all of them) |
| `exact_objectives` | `nothing` | objectives computed from the chromosome instead of predicted, as `objective => function` |
| `screen` | `GpScreen()` | the ranking: `acquisition = :lcb` (default), `:logei`, `:logei_believer`, and for several objectives `:ehvi` and `:qehvi` (see [Choosing the Acquisition](#Choosing-the-Acquisition)); `kappa`, `fit` |
| `failure_above` | `nothing` | a loss value at or above which a loss call counts as failed, e.g. the error a diverged run gets (see [A Loss That Diverges](#A-Loss-That-Diverges)) |
| `budget_rule` | `:fixed` | `:uncertainty` scores only the individuals whose optimistic bound still beats the incumbent: a gain with several objectives, a collapse with one, as the Python package measured |
| `target_transform` | `:log10` | `:asinh` or `:none` for objectives that can be negative |
| `warmup_runs`, `warmup_batch` | six times the batch (at least 60), twice the batch (at least 20) | the warmup before the processes take over |
| `offspring_multiplier` | `nothing` | children bred per child the population takes, the processes picking which enter; by default 3 from 10 individuals per epoch on, else 1 |
| `rescreen` | `nothing` | screen the predictions again every epoch; `nothing` does so with several objectives only |
| `seed` | `0` | the random choices of the screening |

For a `GepTensorRegressor`, `SurrogateScreening(regressor, probes; components=3)` embeds with a `TensorEmbedder` on probe columns like those given to `allocate_buffers!`, `components` being the numbers per sample of an output (here a vector of 3); with several expressions, one number for all of them or one per expression. Any function `chromosome -> latent vector` (or `nothing`) works as an embedder: `SurrogateScreening(embedder; ...)`. [Surrogate Screening](../api-reference.md#Surrogate-Screening) in the API reference lists every option.

## Diagnostics

- `surrogate.evaluated_count`: loss calls made through the screening; `surrogate.imputed_count`: predictions handed out instead
- `surrogate.broken_count`: individuals scored as a crash because they could not be embedded; `surrogate.failed_count`: loss calls that returned a non-finite value, or one at or above `failure_above` (from 5 on, a model of which individuals the loss can score gates the batch, or weighs it with `:qehvi`)
- `surrogate.spearman_log`: the rank correlation of prediction and loss within each screened batch, of the first predicted objective; `surrogate.spearman_objectives` holds one such log per objective. A batch holds the individuals the processes rated most promising, which are hard to tell apart, so the values are low even where the screening works; follow them over the run rather than reading single ones
- `surrogate.last_batch`: the individuals the loss scored in the last epoch, as `(chromosome, fitness, reason)`, the reason being `:warmup`, `:acquisition`, `:explore` or `:validation`
- `archive_size(surrogate)`: individuals scored with a finite loss

A `file_logger_callback` sees every epoch after the screening, for instance to report where the loss calls went:

```julia
function report(population, epoch, _)
    epoch % 10 == 0 || return
    reasons = [entry.reason for entry in surrogate.last_batch]
    println("epoch ", epoch, ": ", length(reasons), " loss calls (",
        join(("$(count(==(r), reasons)) $r" for r in unique(reasons)), ", "), "), ",
        count(c -> !is_validated(surrogate, c), population), " individuals without a loss")
end

fit!(regressor, 100, 200, loss; surrogate=surrogate, file_logger_callback=report)
```

## Saving and Continuing

The screening is the memory of a search: every individual the loss has scored, with its latent vector and loss. Use a fresh one for every search. Passing the same one again continues from what it has learned, e.g. with a `load_state_callback`, and it can be saved alongside the population:

```julia
using Serialization

serialize("screening.jls", surrogate)
surrogate = deserialize("screening.jls")
```

## When the Screening Pays Off

Fitting the processes and ranking the new individuals costs time of its own, which grows with the archive of scored individuals (at most 1000), with the population and with the objectives. Measured on one thread, outside the loss and after the compilation, in seconds per epoch:

| search | without screening | screened, `:lcb` | screened, `:qehvi` |
| --- | --- | --- | --- |
| the ODE above, a batch of 10 | 0.010 | 0.032 | – |
| the ODE above, 15 % of the new individuals | 0.010 | 0.051 | – |
| the system of two ODEs, 15 % | 0.012 | 0.20 | 0.17 |
| the closure of `examples/Main_fictive_cfd_in_the_loop.jl`, 15 % of 400 individuals, with its constant tuning | 0.20 | 0.67 | 0.69, with `failure_above` |

The screening saves time where a loss call costs more than that over the individuals it spares: a solver, a simulation, an external program. With a cheap loss it works too, but the time goes into the processes instead. With several objectives, `:qehvi` costs about as much as the default.

On three benchmark functions (10 seeds each, 200 individuals, 40 epochs, a custom MSE loss; `benchmark/surrogate_screening.jl`), screening 15 % of the new individuals beat scoring every individual at about 6.5 times fewer loss calls, and a batch of 10 per epoch beat a search without screening at the same number of loss calls on every function. Median MSE of the returned model:

| | loss calls | ``x_1^2 + x_1 x_2 - 2x_2^2`` | Nguyen-4 | Nguyen-7 |
| --- | --- | --- | --- | --- |
| every individual scored | 5,165-5,681 | 7.9e-2 (1 exact) | 2.1e-2 | 1.0e-3 |
| screened, 15 % of the new individuals | 751-876 | 7.4e-3 (5 exact) | 8.1e-3 | 8.9e-4 |
| screened, 10 per epoch | 438-472 | 9.4e-2 (1 exact) | 1.4e-2 | 3.4e-3 |
| no screening, population 30, 20 epochs | 415-429 | 2.9e-1 | 9.9e-2 | 7.8e-3 |

(exact: MSE below 1e-10, of 10 seeds.)

A screened search is still an evolutionary search: it can settle in a local optimum like any other. Of the six seeds of the system above, one found `dy/dt` but settled at `dx/dt ≈ 1 - y`, with an error of `x` of 0.04.

## The Constants of a Model

A Gaussian process screens the constants of a model just as well: `fit!(...; constant_optimizer=ScreenedNelderMead())` tunes the constants of the best model against the loss, with the loss calls placed by a process over the constants, and `optimize_constants!` and `simplex_search` do so on their own. In a screened search, the tuning calls the loss next to the screening, with a process and an acquisition of its own: `screen` and `failure_above` do not reach it, and `surrogate.evaluated_count` does not count its calls. [Coefficient Tuning](coefficient-tuning.md) shows when to use which variant, what to do about runs that diverge, and a closure model searched with a fictive CFD solver in the loop.

## Running the Examples

From a clone of the repository:

```bash
julia --project=. --threads=4 examples/Main_surrogate_screening.jl
julia --project=. --threads=4 examples/Main_surrogate_multi_objective.jl
```

Each runs its search with and without the screening and reports the solver calls; the first search of a session includes the compilation.

---

*Continue to [Coefficient Tuning](coefficient-tuning.md), or to the [API Reference](../api-reference.md#Surrogate-Screening) for every option.*
