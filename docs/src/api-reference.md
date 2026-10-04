# API Reference

The public functions, types and modules of GeneExpressionProgramming.jl, by functionality. Everything listed is exported by `GeneExpressionProgramming` unless it is written with its module prefix.

## Regressors

### GepRegressor

The regressor for scalar symbolic regression.

```julia
GepRegressor(feature_amount::Int; kwargs...)
```

**Parameters:**
- `feature_amount::Int`: Number of input features
- `entered_features::Vector{Symbol} = Symbol[]`: Feature names; `x1, x2, ...` when empty
- `entered_non_terminals::Vector{Symbol} = [:+, :-, :*, :/]`: Functions (see [Function Sets](#Function-Sets))
- `gene_connections::Vector{Symbol} = [:+, :-, :*, :/]`: Connectors, the functions that join the genes; only those also in `entered_non_terminals` are used
- `entered_terminal_nums::Vector{Symbol} = [Symbol(0.5), Symbol(0.0)]`: Constant terminals
- `rnd_count::Int = 1`: Number of further constant terminals, drawn uniformly from [0, 1)
- `node_type::Type = Float64`: Numeric type of the constant terminals
- `gene_count::Int = 3`: Number of genes per chromosome
- `head_len::Int = 6`: Head length of each gene (the tail is `head_len + 1` long)
- `tail_weigths = [0.6, 0.2, 0.2]`: Sampling weight of each feature, each fixed constant and each random constant when terminals are drawn
- `head_weigths = nothing`: Accepted; not used
- `number_of_objectives::Int = 1`: Number of objectives; more than one needs the `fit!` method with a custom loss, and selects with NSGA-II
- `considered_dimensions::Dict{Symbol,Vector{Float16}} = Dict()`: Physical dimensions of the features and constants, keyed by their symbols (`:x1`, `:x2`, ... or the `entered_features`; `Symbol(0.5)`, ... for the `entered_terminal_nums`); unlisted ones are dimensionless. When given, the regressor builds the library that the unit repair draws on, which `fit!` needs for a `target_dimension`
- `max_permutations_lib::Int = 10000`: New library expressions kept per build round
- `rounds::Int = 4`: Library build rounds; library expressions have up to `rounds + 1` symbols
- `preamble_syms::Vector{Symbol} = Symbol[]`: Terminals appended to every chromosome, one per gene, outside its expression

**Fields:**
- `best_models_::Vector{Chromosome}`: The best models of the last `fit!`, best first
- `fitness_history_`: Training history: `train_loss` and `val_loss`, one fitness tuple per epoch
- `toolbox_::Toolbox`: The GEP configuration (alphabet, operators, probabilities)
- `token_dto_`: The library and unit rules used by the repair (`nothing` without `considered_dimensions`)

**Example:**
```julia
regressor = GepRegressor(3;
                        gene_count=3,
                        head_len=5,
                        entered_non_terminals=[:+, :-, :*, :/, :sin, :cos])
```

### GepTensorRegressor

Regressor for models over scalars, vectors and higher-order tensors ([Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl) types).

```julia
GepTensorRegressor(feature_amount::Int; kwargs...)
```

**Parameters:**
- `feature_amount::Int`: Number of input features (scalar and tensor-valued alike)
- `problem_dimension::Int = 2`: Spatial dimension of the tensors
- `feature_names::Vector{String} = String[]`: Names of the features, used when printing; `x1, x2, ...` when empty
- `entered_non_terminals::Vector{Symbol} = [:+, :-, :*, :/]`: Functions, including the tensor operations (see [Tensor Operations](#Tensor-Operations))
- `gene_connections::Vector{Symbol} = [:+, :*]`: Connectors; only those also in `entered_non_terminals` are used
- `entered_terminal_nums::Vector{<:AbstractFloat} = Float64[]`: Constant terminals
- `rnd_count::Int = 0`: Number of further constant terminals, drawn uniformly from `rnd_limits = (-0.1, 0.1)`
- `gene_count::Int = 2`: Number of genes per chromosome
- `head_len::Int = 3`: Head length of each gene
- `number_of_objectives::Int = 1`: Number of objectives
- `head_tail_balance::Real = 0.6`: Weight of the binary functions when head symbols are drawn; `tail_weigths = [0.7, 0.2, 0.1]`: as for `GepRegressor`
- `considered_dimensions`, `max_permutations_lib = 10000`, `rounds = 5`: As for `GepRegressor`, with two differences: the dimensions are keyed `:x1`, `:x2`, ... in feature order whatever `feature_names` says, and each carries the tensor order in front of the SI exponents; constants are dimensionless. Only the functions with tensor unit rules can be used with dimensions (see [Tensor Operations](#Tensor-Operations))
- `higher_dim_feature_amount::Int = 0`: Accepted; not used

**Fields:**
- `best_models_`, `fitness_history_`, `toolbox_`, `token_dto_`: As for `GepRegressor`
- `input_values`: The input columns, set by `allocate_buffers!`
- `buffers`: The per-thread evaluation buffers, set by `allocate_buffers!`

**Example:**
```julia
regressor = GepTensorRegressor(5;
                              problem_dimension=3,
                              gene_count=3,
                              head_len=4,
                              entered_non_terminals=[:+, :-, :*],
                              feature_names=["x1", "x2", "U1", "U2", "U3"])
# x1, x2: scalar columns; u1, u2, u3: columns of Tensor{1,3} (see Tensor Regression)
allocate_buffers!(regressor, (x1, x2, u1, u2, u3))    # one column per feature
```

## Training

### fit!

Train a regressor. There are three methods:

1. Scalar regression on data arrays:

```julia
fit!(regressor::GepRegressor, epochs::Int, population_size::Int, x_train::AbstractArray,
     y_train::AbstractArray; kwargs...)
```

`x_train` holds one row per feature and one column per sample (hence the transposes in the examples). Throws an `ArgumentError` when a function or terminal has no batched counterpart.

**Keyword Arguments:**
- `x_test = nothing`, `y_test = nothing`: Held-out data for the validation loss (the training data when not given)
- `loss_fun::Union{String,Function} = "mse"`: Loss to minimise: a name (see [Loss Functions](#Loss-Functions)) or a function `(y_true, y_pred) -> Real`
- `loss_fun_validation::Union{String,Function} = "mse"`: Loss reported on the held-out data
- `hof::Int = 3`: Number of best models returned in `best_models_`
- `optimization_epochs::Int = 100`: Every this many epochs, if the best model has improved since the last time, Nelder-Mead tunes each occurrence of a constant in it on the training loss; values that lower the loss are kept in its `optimised_constants`. Not with `linear_scaling`
- `max_iterations::Int = 1000`: Iterations of that constant optimisation
- `linear_scaling::Bool = false`: Score each chromosome as the least-squares combination of its genes, one coefficient per gene (stored in `scaling_weights`), so that evolution searches for the structure only; the model is then the weighted sum of its genes. Needs `:+` and `:*` among the functions. With a `target_dimension`, every gene is held to it, not only the connected expression, since the coefficients are dimensionless only then
- `target_dimension::Union{Vector{Float16},Nothing} = nothing`: Target physical dimension; only homogeneous individuals are scored, the others are repaired (needs `considered_dimensions`)
- `correction_epochs::Int = 1`: Run the repair every this many epochs
- `correction_amount::Real = 1.0`: The most individuals repaired per correction epoch, as a fraction of the population
- `cycles::Int = 10`: Repair attempts per individual
- `lib_seed_amount::Real = 0.5`: Fraction of the initial population seeded with library expressions of the target dimension
- `penalty::AbstractFloat = 2.0`: Factor on the cached fitness of a new individual whose karva string has been scored before
- `population_sampling_multiplier::Int = 1`: Above 1, the initial population is picked from this many times more random chromosomes, by the mean of their predictions on random probe data (`select_n_samples_lhs`). The selection is meant as Latin hypercube sampling, but its target points are not stratified
- `break_condition = nothing`: `(population, epoch) -> Bool`, stops the run when `true`
- `file_logger_callback = nothing`: `(population, epoch, selected_members)`, called every epoch
- `save_state_callback = nothing`: `(population, strategy)`, called every epoch
- `load_state_callback = nothing`: `() -> (population, start_epoch)`, resumes a run
- `surrogate = nothing`: A [`SurrogateScreening`](#Surrogate-Screening) for an expensive loss: the loss scores only the individuals it picks each epoch, and the others get the prediction of a Gaussian process
- `opt_method_const`, `n_starts`, `buffered`: Accepted; not used

2. Custom loss over chromosomes, for several objectives or any evaluation of your own:

```julia
fit!(regressor::GepRegressor, epochs::Int, population_size::Int, loss_function::Function; kwargs...)
```

`loss_function(elem::Chromosome, validate::Bool)` sets `elem.fitness` to a tuple with one entry per objective (as many as `number_of_objectives`); its return value is ignored. It is called inside the threaded fitness loop for the unscored chromosomes (fitness `NaN`), once per distinct karva string, and each epoch with `validate = true` for the best one. The loop hands the chromosomes out one at a time to whichever thread is free, so a loss whose cost varies from one chromosome to the next keeps every thread busy; a call stays on one thread and no other call shares its `Threads.threadid()` meanwhile, even if the loss waits (on an external solver, say). Evaluate the chromosome in a per-thread context (see [thread_contexts](#thread_contexts)), and catch errors -- an exception thrown by the loss stops `fit!`.

**Keyword Arguments:**
- `loss_function_validation = nothing`: `(elem, validate) -> Tuple`, called for the best chromosome each epoch; its return value is recorded as the validation loss (without it, the training fitness is recorded)
- `hof`, `target_dimension`, `correction_epochs`, `correction_amount`, `cycles`, `lib_seed_amount`, `penalty`, `break_condition`, `file_logger_callback`, `save_state_callback`, `load_state_callback`: As above
- `gene_wise_dimension::Bool = false`: Hold every gene to `target_dimension` rather than the connected expression, for a loss that scores the least-squares combination of the genes
- `surrogate = nothing`: A [`SurrogateScreening`](#Surrogate-Screening): the loss is called only for the individuals it picks each epoch (and for a best or returned model that carries a prediction); the others get the prediction of a Gaussian process
- `constant_optimizer = nothing`: A [`ScreenedNelderMead`](#Constants-Against-an-Expensive-Loss): every `optimization_epochs::Int = 100` epochs, if the best model has improved since the last time, its constants are tuned against `loss_function` by Nelder-Mead, screened by a Gaussian process or not (`optimize_constants!`), within `max_evaluations` loss calls. The loss must evaluate the chromosome by a path that applies tuned constants: `elem(ctx)` or `split_predict(elem, ctx, k)`. With a `surrogate`, the model tuned is the best one the loss has scored, never one that carries a prediction
- `optimizer_function_`, `opt_method_const`, `max_iterations`, `n_starts`: Accepted; not used

This method has no `population_sampling_multiplier` keyword; `runGep`'s default of 100 applies, so the initial population is picked from 100 × `population_size` random chromosomes as described above.

3. Tensor regression:

```julia
fit!(regressor::GepTensorRegressor, epochs::Int, population_size::Int, loss_function::Function; kwargs...)
```

The loss has the same form as in 2.; it typically evaluates with `predictT`. Call `allocate_buffers!` first. Tournaments are of 0.3 % of the population (at least 3), and duplicates take the cached fitness times 2.0 (`runGep`'s default `penalty`).

**Keyword Arguments:**
- `hof`, `target_dimension`, `correction_epochs`, `correction_amount`, `cycles`, `lib_seed_amount`, `break_condition`, `file_logger_callback`, `save_state_callback`, `load_state_callback`: As above
- `gene_wise_dimension::Bool = false`: Hold every gene to `target_dimension` rather than the connected expression; a loss that scores with `predictT_scaled` needs it unless the gene connectors are only `:+` and `:-`
- `population_sampling_multiplier = 1`: Keep it at 1: the sampling probes candidates with the scalar evaluator, which does not accept a tensor toolbox (with a `surrogate`, the sampling uses its `TensorEmbedder` instead and works)
- `surrogate = nothing`: A [`SurrogateScreening`](#Surrogate-Screening) built with `SurrogateScreening(regressor, probes)`, as for method 2.
- `constant_optimizer = nothing`, `optimization_epochs::Int = 100`: As for method 2, for a loss that evaluates the chromosome with `predictT(regressor, elem)`, which applies tuned constants

**Examples** (`x_train`, `y_train`, `x_test`, `y_test`, `target_dim` and the losses as defined on this page and in the examples):
```julia
# Basic regression
fit!(regressor, 1000, 1000, x_train', y_train; loss_fun="mse")

# With validation data
fit!(regressor, 1000, 1000, x_train', y_train;
     x_test=x_test', y_test=y_test, loss_fun="rmse")

# With physical dimensions (a regressor built with considered_dimensions)
fit!(regressor, 1000, 1000, x_train', y_train;
     target_dimension=target_dim)

# Custom (e.g. multi-objective or tensor) loss
fit!(regressor, 100, 500, multi_objective_loss)
```

## Prediction and Evaluation

### Calling a regressor or a chromosome

```julia
(regressor::GepRegressor)(x_data)     # the best model's predictions
(chromosome::Chromosome)(x_data)      # any chromosome's predictions
(chromosome::Chromosome)(ctx)         # predictions in a prebuilt evaluation context
```

`x_data` holds one row per feature and one column per sample. The predictions include the fitted constants (`optimised_constants`) or gene coefficients (`scaling_weights`) when the model has them.

**Example:**
```julia
predictions = regressor(x_test')
second_best = regressor.best_models_[2](x_test')
```

### thread_contexts

```julia
thread_contexts(toolbox, x_data; std_return_type=Float64)
buffer_context(toolbox, x_data; std_return_type=Float64)
```

`buffer_context` builds an evaluation context for `x_data` (one row per feature): the input columns, the operators, and the buffers the evaluator writes into. `thread_contexts` builds one per thread id, for losses that run inside the threaded fitness loop. Both work for `GepRegressor` toolboxes only; for a tensor toolbox, whose callbacks are operator objects, `buffer_context` returns `nothing`, and a tensor loss evaluates with `predictT` instead.

```julia
using Statistics

ctxs = thread_contexts(regressor.toolbox_, x_train')

function loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        y_pred = elem(ctxs[Threads.threadid()])
        # ... set elem.fitness from y_pred
    end
end
```

The result of `elem(ctx)` lives in the context's buffers; use it (or copy it) before the next evaluation in the same context.

### Tensor prediction

```julia
allocate_buffers!(regressor::GepTensorRegressor, data_x; std_return_type=Float64)
predictT(regressor::GepTensorRegressor, rek_string::Vector)
predictT(regressor::GepTensorRegressor, chromosome::Chromosome)
predictT(regressor::GepTensorRegressor, rek_string::Vector, x_data::Vector)
predictT(regressor::GepTensorRegressor, rek_string::Vector, new_input_values::Dict)
predictT_scaled(regressor::GepTensorRegressor, chromosome::Chromosome, target)
predictT_scaled!(out, regressor::GepTensorRegressor, chromosome::Chromosome, target)
gene_bases(regressor::GepTensorRegressor, chromosome::Chromosome)
```

- `allocate_buffers!` stores the input columns (`data_x`: one column per feature, as a tuple or `Vector{Any}`) and allocates the per-thread buffers for them (`std_return_type` is not used). Required before `fit!`.
- `predictT(regressor, rek_string)` evaluates a karva string (`chromosome.expression_raw`) on those columns, in the calling thread's buffers; use or copy the result before the next evaluation on the same thread.
- `predictT(regressor, chromosome)` does the same for a chromosome, with its tuned constants (`optimised_constants`) if it has them, which a loss needs for `optimize_constants!` and the `constant_optimizer` of `fit!` to reach it; with tuned constants, the expression runs on the allocating path of the evaluator, into fresh arrays.
- `predictT(regressor, rek_string, x_data)` evaluates it on new data (one column per feature, in the same order), into fresh arrays; constants are broadcast to the new sample count.
- `predictT(regressor, rek_string, new_input_values)` evaluates it on the stored columns, with the column of each terminal `index => value` in the `Dict` set to `value` in every sample.
- `predictT_scaled` fits one least-squares coefficient per gene against `target` -- the tensor counterpart of `linear_scaling` -- and returns the scaled prediction; the coefficients are stored in `scaling_weights`. Genes whose output does not match `target` in length and element type are left out; it returns `nothing` when no gene matches or the fit is not finite. The coefficients are dimensionless only if every gene has the target dimension (`gene_wise_dimension=true` in `fit!`). When every column and `target` are `Vector{Float64}`, the genes are evaluated by the compiled evaluator into preallocated buffers, with the same result.
- `predictT_scaled!` writes that prediction into `out` (a vector like `target`) and returns it; with one `out` per thread slot, a loss allocates no prediction per candidate (second example).
- `gene_bases` evaluates every gene on its own, into fresh arrays.

**Example** (the columns as in [Tensor Regression](examples/tensor-regression.md)):
```julia
best = regressor.best_models_[1]
pred_train = predictT(regressor, best.expression_raw)
# x1_new, x2_new, u1_new, u2_new, u3_new: new columns, one per feature, in the same order
pred_new = predictT(regressor, best.expression_raw, Any[x1_new, x2_new, u1_new, u2_new, u3_new])
```

**Example** (a scaled loss on `Vector{Float64}` columns and target `y`, one prediction buffer per thread slot):
```julia
preds = [similar(y) for _ in 1:thread_slots()]
function loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        pred = predictT_scaled!(preds[Threads.threadid()], regressor, elem, y)
        if pred isa AbstractVector && allfinite(pred)
            pred .-= y                                  # the residual, in place
            elem.fitness = (sum(abs2, pred) / length(y),)
        else
            elem.fitness = (1e6,)
        end
    end
end
```

## Inspecting Models

A model is a `Chromosome`. Printing it (`println`, `string`) writes it as an equation, with every binary operation parenthesised and with fitted constants or gene coefficients in place.

```julia
best = regressor.best_models_[1]
println(best)                    # e.g. ((x1 * x1) + (x2 * (x1 - (x2 + x2))))
best.fitness                     # tuple, one entry per objective
best.expression_raw              # the karva string (Vector{Int8}) the evaluator runs
best.optimised_constants         # tuned constants, or nothing
best.scaling_weights             # gene coefficients (linear scaling), or nothing
best.dimension_homogene          # true once the model is known to meet the target dimension
```

### equation_string

```julia
equation_string(chromosome::Chromosome)
```

The string `show` prints: the model as an equation, with fitted constants in place, or as a weighted sum of genes when it carries gene coefficients; both rounded to 6 significant digits.

### print_karva_strings

```julia
print_karva_strings(chromosome::Chromosome; split_len::Int=1)
```

The expression as a string, built from the karva string alone -- without fitted constants or gene coefficients. `split_len = k > 1` skips the first `k - 1` connectors and returns the partial results, last gene first, instead of one string; with `k` equal to the gene count that is one string per gene.

### fitness / set_fitness!

```julia
fitness(chromosome::Chromosome)
set_fitness!(chromosome::Chromosome, value::Tuple)
```

Get or set the fitness tuple.

### Training history

```julia
history = regressor.fitness_history_
train = [history.train_loss[i][1] for i in eachindex(history.train_loss) if isassigned(history.train_loss, i)]
val = [history.val_loss[i][1] for i in eachindex(history.val_loss) if isassigned(history.val_loss, i)]
```

One tuple per epoch, for the best model of that epoch; the epochs after a `break_condition` stopped the run are unassigned.

## Surrogate Screening

For an expensive loss (a solver in the loop, a simulation), a Gaussian process decides which individuals of an epoch the loss scores and predicts the loss of the others. It is the Julia counterpart of `gep.SurrogateBatchStrategy` of the Python package, with its defaults. The types below are exported; the functions they are built from live in the submodule `GepSurrogate`. The example [Surrogate Screening](examples/surrogate-screening.md) walks through a search with one objective and one with several expressions and objectives.

### SurrogateScreening

```julia
SurrogateScreening(embedder; screen=GpScreen(), individuals_per_epoch=10, kwargs...)
SurrogateScreening(regressor::GepRegressor, probes::AbstractMatrix; embedding=:expression,
    expressions=1, objective_expressions=nothing, kwargs...)
SurrogateScreening(regressor::GepTensorRegressor, probes::AbstractVector; embedding=:expression,
    expressions=1, objective_expressions=nothing, components=1, kwargs...)
```

Pass it to `fit!` (or `runGep`) as `surrogate`. `embedder` maps a chromosome to its latent vector, or to `nothing` for an expression that cannot be evaluated. The regressor methods build it on `probes`: one row per feature and one column per probe sample for a `GepRegressor` (like `x_train`), one column per feature for a `GepTensorRegressor` (like the data of `allocate_buffers!`); `embedding = :genes` embeds one block per gene, and `expressions = k` one block per expression of a chromosome that carries `k` of them (below). A few dozen probe samples from the relevant range suffice.

Each epoch, the new individuals (one per karva string) are embedded; one that cannot be embedded is scored as a crash without a loss call. Then:
- **Warmup**, until `warmup_runs` individuals (six times the batch, at least 60) have a finite loss: at most `warmup_batch` of them (twice the batch, at least 20; `nothing` for all) are scored, picked uniformly (or by Latin hypercube sampling with `warmup_lhs=true`), and the others get the median loss of that batch.
- **Screened epochs**: a `GpScreen` is fitted on the scored individuals (the last `archive_cap = 1000`); the loss scores `individuals_per_epoch` individuals (a count, or a share in `(0, 1)` of the new individuals), `explore_fraction = 0.1` of them picked as in the warmup, the rest by the acquisition; every other individual gets the prediction `mean + impute_beta * deviation` (`impute_beta = 0`).
- A prediction is at least one float step worse than the best loss scored so far on every objective (the incumbent), and it is never cached: a copy of a predicted individual is screened again, while a copy of a scored one takes its loss times `penalty`, even once the fitness cache has dropped it.
- A prediction is provisional: with `rescreen = true`, the default with several objectives (`rescreen = nothing`), the individuals that carry one are screened again at the start of every epoch, next to the new ones, so a later process can pick them for a loss call or give them a fresh prediction. The batch stays a count or a share of the new individuals. Without, a surviving prediction is never looked at again, and in a search with several objectives the population fills up with stale ones; with one objective, re-screening was a wash on the benchmark of the package.
- The best individual of an epoch is scored if it carries a prediction, before it is selected with; the returned hall of fame is scored at the end (`validate_hof = true`). The population is not sorted again: a best whose loss turns out worse than its prediction (a run that diverged, say) still leads it, so `population[1]` in a callback is not always the best scored individual, and the epoch records the best scored one in `fitness_history_`.
- From `min_failures = 5` failed loss calls (a non-finite value, or one at or above `failure_above`) on, a `FeasibilityModel` gates each batch towards the individuals the loss can score; `:qehvi` weighs the batch by it instead.
- `failure_above` (default `nothing`) is the loss value at or above which a call counts as failed, e.g. `1e3` for a loss that gives a diverged run 1e3: such a call teaches the feasibility model, not the processes, which see the converged runs alone, and an individual the loss has not scored is predicted toward `failure_above` by its probability of failing (in the transformed units). The individuals keep the fitness the loss gave them. Set it at most at the error of a diverged run and above every error a converged run reaches; with several objectives, a call fails once any of them reaches it. A loss that returns `Inf` for a failed run needs none.
- With `budget_rule = :uncertainty`, an epoch scores only the individuals whose optimistic bound still beats the incumbent, between `min_individuals` and `individuals_per_epoch` of them (measured by the Python package as a gain with several objectives and as a collapse with one). `budget_decay` schedules the batch from `warmup_batch` down to `individuals_per_epoch`.
- An epoch breeds `offspring_multiplier` times the children the population takes (by default 3 from 10 individuals per epoch on, read on a hundred individual epoch, and 1 below), and the process picks which enter (`GepSurrogate.preselect`).
- An oversampled initial population (`population_sampling_multiplier > 1`) is picked by Latin hypercube sampling over the latent vectors (`characterize_initial = true`).
- `target_transform = :log10` brings the losses into the space the processes work in; use `:asinh` or `:none` for objectives that can be negative.

**Several objectives and several expressions.** A loss that sets several objectives (`number_of_objectives = k`) is screened with one process per objective. The pick scalarizes them by a random augmented Chebyshev weighting per epoch (ParEGO), or ranks by the expected hypervolume improvement (`GpScreen(acquisition = :ehvi)`), or by the joint expected hypervolume improvement of the batch (`:qehvi`). A prediction is clamped behind the best scored value of every objective, so no prediction dominates the individual holding one; and since the population survives by its mean fitness, which a prediction can beat without dominating anyone, the scored individuals that hold the best value of an objective are kept right behind the leader, where the next generation does not replace them (`GepSurrogate.keep_best_scored!`). The best scored value of every objective among the survivors thus never gets worse. An objective the loss computes from the chromosome alone, e.g. a size, is better computed than predicted: `exact_objectives = Dict(2 => c -> 0.01 * length(c.expression_raw))` gives every predicted individual that value (without a clamp), and the acquisition sees it without uncertainty. The function has to return exactly the value the loss sets. A chromosome that carries several expressions, split by `split_karva` in the loss (`split_predict` evaluates them), is embedded with one block per expression: `SurrogateScreening(regressor, probes; expressions = 2)`, which needs a gene count divisible by the number of expressions. The whole karva string joins the parts with connectors the loss never uses, so its behaviour is not what the loss sees. Where each objective judges one expression, `objective_expressions = [1, 2]` (one entry per objective, `0` or `:all` for one that depends on all of them) lets the process of an objective see the latent block of its expression alone (it sets the `inputs` of the `GpScreen`, `GepSurrogate.expression_blocks`). The result of such a search is a front: take the non-dominated members of `best_models_` (`calculate_fronts`), which the loss has all scored (`validate_hof`); a large `hof` costs a loss call per predicted member at the end.

**Diagnostics:** `evaluated_count` (loss calls made through the screening), `imputed_count` (predictions), `broken_count`, `failed_count`, `spearman_log` (rank correlation of prediction and loss per screened batch, of the first predicted objective with several), `spearman_objectives` (the same, one log per objective), `last_batch` (the individuals scored in the last epoch, as `(chromosome, fitness, reason)`, the reason being `:warmup`, `:acquisition`, `:explore` or `:validation`), `archive_size(s)`, `is_validated(s, chromosome)` (the individual carries a loss, not a prediction).

```julia
using Random

regressor = GepRegressor(2; number_of_objectives=1)
probes = x_train[:, randperm(size(x_train, 2))[1:36]]     # x_train: one row per feature
surrogate = SurrogateScreening(regressor, probes; individuals_per_epoch=0.15, seed=1)
fit!(regressor, 100, 500, expensive_loss; surrogate=surrogate)

@show surrogate.evaluated_count surrogate.imputed_count
@show all(is_validated(surrogate, m) for m in regressor.best_models_)   # true
```

### GpScreen

```julia
GpScreen(; acquisition=:lcb, kappa=1.0, rho=0.05, scalarize_per_pick=false, fit=false, nugget=1e-6,
    inputs=nothing)
```

Ranks the pending individuals with a Gaussian process per objective. `acquisition` is `:lcb` (the optimistic bound `mean - kappa * deviation`, the default the Python study measured best), `:logei` (the top of the log expected improvement), `:logei_believer` (greedy LogEI with the kriging believer between the picks) `:ehvi` (expected hypervolume improvement, several objectives, every pick joining the front at its predicted mean) or `:qehvi` (the q-expected hypervolume improvement of Daulton, Balandat and Bakshy (2020): the candidates sampled jointly, the batch filled greedily with every pick valued in each sample against the front the earlier picks of that sample left, a candidate's probability of failing weighed inside the samples, against a reference a tenth of the front's spread beyond its worst values). `:ehvi` and `:qehvi` need several objectives: a search whose loss sets one refuses them (`GepSurrogate.check_acquisition`), as does `SurrogateScreening(regressor, probes; screen)`. [Choosing the Acquisition](examples/surrogate-screening.md#Choosing-the-Acquisition) tells which to take for which search: the default for one objective, the default or `:qehvi` for several, and `:qehvi` with `failure_above` for a solver that gives a diverged run a large error. With several objectives the other acquisitions pick by a random augmented Chebyshev scalarization (weight `rho`), drawn per fit or, with `scalarize_per_pick`, per pick. `fit = true` chooses the length scale and noise of the processes by the marginal likelihood over a grid. `inputs` gives the latent coordinates the process of each objective sees (one range or index vector per objective, `nothing` for all); with it, `:lcb` scalarizes the bounds of the processes of the objectives by the weights of the epoch. A custom screen is any object with methods of `GepSurrogate.fit_screen!`, `predict_screen`, `select_screen` and `plausible_screen`.

### GaussianProcess and FeasibilityModel

```julia
GaussianProcess(X, y; nugget=1e-6, fit=false)      # X: d × n, one latent vector per column
FeasibilityModel(X, labels; prior=2.0, bandwidth=0.25)
```

Exact Gaussian process regression with a radial basis kernel whose length scale is the median distance between distinct points (or fitted, with `fit = true`); `GepSurrogate.posterior(gp, Xq)` gives the standardized mean and deviation, `posterior_covariance(gp, Xq)` the mean and the joint covariance, `unstandardize`, `expected_improvement`, `log_expected_improvement` and `believe` build on it. `FeasibilityModel` is a kernel-weighted average of the outcomes (1 for a usable loss, 0 for a failure), pulled to the overall rate by `prior` pseudo observations; call it on a `d × m` matrix for `m` probabilities.

### SemanticEmbedder, GeneEmbedder, TensorEmbedder

```julia
SemanticEmbedder(toolbox, probes::AbstractMatrix; transform=:asinh, expressions=1)
GeneEmbedder(toolbox, probes::AbstractMatrix; transform=:asinh, count=nothing)
TensorEmbedder(toolbox, probes::AbstractVector; components=1, transform=:asinh, per_gene=false,
    expressions=1)
```

Map a chromosome to the transformed outputs of its expression on the probe set (`SemanticEmbedder`), of its genes one block each (`GeneEmbedder`), or of a tensor expression flattened to `components` numbers per sample (`TensorEmbedder`), or to `nothing` if an output is not finite. With `expressions = k`, the chromosome is split into its `k` expressions (`split_karva`) and embedded with one block per expression; for a `TensorEmbedder`, `components` is then one number for all of them or a vector with one per expression. Constants tuned by an optimiser and gene coefficients are not applied, as an individual is embedded before it is scored.

## Constants Against an Expensive Loss

The constants of a model, tuned against an expensive custom loss (a solver in the loop, a CFD simulation) by Nelder-Mead, plain or screened by a Gaussian process over the constants. The functions they are built from live in the submodule `GepSimplex`. [Coefficient Tuning](examples/coefficient-tuning.md) shows when to use which variant, with measurements, and walks through both uses, inside a search and on their own.

### ScreenedNelderMead

```julia
ScreenedNelderMead(; max_evaluations=60, screen=true, kappa=1.0, target_transform=:log10,
    fit=true, polish_radius=1/32, tolerance=1e-8, seed=0, swarm_box=nothing, particles=10)
```

How `optimize_constants!` and `simplex_search` minimize, within `max_evaluations` loss calls (the initial simplex included). Both modes start as `Optim.NelderMead()` does: the given constants and one vertex per constant, moved by half its value plus 0.025 (by 0.025 where that vanishes, at -0.05).
- `screen = false`: Nelder-Mead on the loss, the steps of `Optim.NelderMead()` (the adaptive parameters of Gao and Han, its stopping rule).
- `screen = true`: every further loss call goes where a Gaussian process over the constants (a `GaussianProcess` with `fit` choosing its length scale and noise by the marginal likelihood, on the `target_transform` of the loss) expects the minimum: the minimum of `mean - kappa * deviation` inside a trust region around the best constants scored, in units of the initial steps. The region doubles after two improvements in a row and halves after `max(2, ceil(n / 2))` failures in a row; the bound turns greedy as it shrinks below the initial steps. A failed call (not finite) counts as the worst value scored. Below `polish_radius` initial steps, Nelder-Mead finishes on the loss from scored constants near the best ones.
- `tolerance`: Nelder-Mead stops when the deviation of the values at the vertices, times `sqrt(n / (n + 1))`, falls below it, as in Optim; `seed` seeds the random starts of the search on the process and the swarm.
- `swarm_box`: a particle swarm screened by a process explores a box first, for a loss with several minima: `(lower, upper)`, which must contain the start, or a number `r` of initial steps around the start (the form for `fit!`, whose models differ in their constants). The start and a Latin hypercube of the box, `particles` points in all, are scored; then every iteration moves every particle (constriction coefficients, the best of the swarm) and the loss scores the new position the process ranks best, one call per iteration. After `2n + 4` calls without a gain of 0.1 %, or at half the budget, the trust region search continues from the best point on every point scored; it is not bound to the box. It needs `screen = true`.

Which variant suits which loss, with measurements: [Coefficient Tuning](examples/coefficient-tuning.md#Which-One-to-Use).

### optimize_constants!

```julia
optimize_constants!(chromosome::Chromosome, loss::Function; method=ScreenedNelderMead(),
    objective=nothing) -> Union{SimplexSearch,Nothing}
```

Tunes the constants of `chromosome` against `loss(chromosome, validate)`, the custom loss of a search: one parameter per occurrence of a constant in `expression_raw` (`constant_positions(chromosome)`), starting from the values the chromosome holds. Every candidate is set as the chromosome's `optimised_constants` and scored by `loss(chromosome, true)`, so the loss must evaluate by a path that applies them: `chromosome(ctx)`, `split_predict`, `predictT(regressor, chromosome)`. `objective` turns the fitness tuple into the value minimized: `nothing` for the mean of its entries (the order of the population), an index, or a function of the tuple. If the best constants beat the ones held, they stay in `optimised_constants` and the chromosome takes the fitness the loss gave them; otherwise, and if the loss throws, the chromosome is left as it was. Returns `nothing` for a chromosome without constants. As the `constant_optimizer` of a screened search it calls the loss next to the screening: the acquisition and `failure_above` of the `SurrogateScreening` do not apply, and its calls are not in `evaluated_count`.

### simplex_search and SimplexSearch

```julia
simplex_search(f, x0::AbstractVector{<:Real}; method=ScreenedNelderMead()) -> SimplexSearch
```

Minimizes `f(x)`, a number for a vector, e.g. the error of a simulation for the coefficients `x`, within `method.max_evaluations` calls; a value that is not finite counts as the worst. The `SimplexSearch` holds the best point `f` scored (`minimizer`, `minimum`), the calls (`evaluations`), how many of them the process placed (`proposals`), the best value after every call (`history`), and every point scored with its value (`points`, `values`).

```julia
calibrate(p) = simulation_error(p)                 # one simulation per call
result = simplex_search(calibrate, [1.0, 0.5, 2.0]; method=ScreenedNelderMead(max_evaluations=50))
result.minimizer, result.minimum, result.evaluations
```

## Evaluation Strategies

`fit!` builds one of these and hands it to `runGep`; they are only needed to call `runGep` directly.

### StandardRegressionStrategy

Evaluates scalar models on data arrays.

```julia
StandardRegressionStrategy{T}(operators, x_data, y_data, x_data_test, y_data_test,
    loss_function::Function;
    validation_loss_function=nothing,
    secOptimizer=nothing,
    break_condition=nothing,
    penalty::T=zero(T),
    crash_value::T=typemax(T),
    linear_scaling::Bool=false,
    buffered=nothing) where {T<:AbstractFloat}
```

`buffered` is the evaluation context from `build_buffers(regressor, x_data)` (without it every candidate scores `crash_value`); `crash_value` is the fitness of a candidate that cannot be evaluated, `Inf` for floating-point `T`. `operators`, `x_data` and `penalty` are stored but not used.

### GenericRegressionStrategy

Hands the chromosomes to a user loss, supporting several objectives.

```julia
GenericRegressionStrategy(operators, number_of_objectives::Int, loss_function::Function;
    validation_loss_function=nothing,
    secOptimizer=nothing,
    break_condition=nothing)
```

## Core GEP Functions

### runGep

The evolutionary loop.

```julia
runGep(epochs::Int, population_size::Int, toolbox::Toolbox, evalStrategy::EvaluationStrategy;
    hof::Int=3,
    correction_callback=nothing,
    homogeneity_check=nothing,
    population_seeder=nothing,
    correction_epochs::Int=1,
    correction_amount::Real=0.6,
    tourni_size::Int=3,
    optimization_epochs::Int=500,
    file_logger_callback=nothing,
    save_state_callback=nothing,
    load_state_callback=nothing,
    population_sampling_multiplier::Int=100,
    inputs_::Int=0,
    cache_size::Int=10000,
    penalty::AbstractFloat=2.0,
    surrogate=nothing)
```

Some defaults differ from `fit!`'s (`correction_amount`, `optimization_epochs`, `population_sampling_multiplier`); `cache_size` is the capacity of the fitness cache, keyed by karva string. `surrogate` is a [`SurrogateScreening`](#Surrogate-Screening); what it changes in the steps below is listed there.

**Returns:** `(best, history)`: the `hof` best chromosomes and the training history.

Each epoch:
1. With a dimensional target, new individuals are checked and, if needed, repaired (`correction_callback`, `homogeneity_check`); the others get the worst fitness instead of being scored
2. New individuals are scored in parallel, once per distinct karva string, each thread taking the next one as soon as it is free; duplicates of known karva strings take the cached fitness times `penalty`
3. The population is ranked by mean fitness; every `optimization_epochs` epochs, if the best has improved since the last time, the strategy's `secOptimizer` tunes it
4. The best is re-scored with `validate = true`, its training and validation loss are recorded, and `break_condition(population, epoch)` may stop the run
5. Parents are selected (tournament of `tourni_size`, or NSGA-II with several objectives) and, unless this is the last epoch, their offspring take ranks `population_size - m` to `population_size - 1` (see [Population Dynamics](core-concepts.md#Population-Dynamics))

### Toolbox

The GEP configuration: alphabet, arities, operators, gene layout and operator probabilities.

```julia
Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8},
       gene_connections::Vector{Int8}, callbacks::Dict, nodes::OrderedDict,
       gep_probs::Dict{String,AbstractFloat};
       unary_prob::Real=0.1, preamble_syms=Int8[],
       number_of_objectives::Int=1, operators_=nothing,
       function_complile=compile_djl_datatype,
       tail_weights_=nothing, head_tail_balance::Real=0.5, ...)
```

**Fields (selection):**
- `gene_count`, `head_len`: Gene layout
- `symbols::OrderedDict{Int8,Int8}`: Every symbol and its arity
- `gene_connections::Vector{Int8}`: The connector symbols
- `headsyms`, `tailsyms`: Symbols allowed in heads and tails
- `callbacks`: The operator of every function symbol
- `nodes`: The terminal behind every terminal symbol (feature selector or constant)
- `gen_start_indices::Vector{Int}`: Position of every gene in a chromosome
- `gep_probs`: Operator probabilities; for the regressors, the dictionary `RegressionWrapper.GENE_COMMON_PROBS` itself (see [Genetic Operators](#Genetic-Operators))
- `fitness_reset::Tuple`: The worst fitness (all `Inf`) and the unscored fitness (all `NaN`)

### Chromosome

An individual: a linear chromosome and the model it encodes.

```julia
Chromosome(genes::Vector{Int8}, toolbox::Toolbox, compile::Bool=false)
```

**Fields:**
- `genes::Vector{Int8}`: The connectors, then the genes (head and tail each), then the preamble symbols, if any
- `expression_raw::Vector{Int8}`: The karva string: the connectors, then each gene's active part in prefix order
- `fitness::Tuple`: Fitness, one entry per objective (`NaN` until scored)
- `compiled::Bool`: Whether `expression_raw` has been resolved; editing `genes` does not reset it
- `dimension_homogene::Bool`: Set once the expression is known to meet the target dimension
- `optimised_constants`, `scaling_weights`: Fitted constants and gene coefficients, or `nothing`
- `toolbox::Toolbox`: The configuration it was built with

`chromosome.compiled_function`, where older code read the compiled expression, still works: it gives the model, which prints as `equation_string(chromosome)` and predicts like `chromosome(x)` when called as `m(x)` or `m(x, operators)` (`x` with one row per feature).

### compile_expression!

```julia
compile_expression!(chromosome::Chromosome; force_compile::Bool=false)
```

Resolves the genes into `expression_raw`, which is what the evaluator runs, if the chromosome is not compiled yet or `force_compile` is set; this clears `optimised_constants` and `scaling_weights` and resets the fitness to unscored. After editing `genes` directly, call it with `force_compile=true`.

### generate_gene / generate_chromosome / generate_population

```julia
generate_gene(headsyms, tailsyms, headlen, tail_weights, head_weights; rng=...)
generate_chromosome(toolbox::Toolbox; rng=...)
generate_population(number::Int, toolbox::Toolbox)
```

Random genes, compiled chromosomes and populations for a toolbox.

### split_karva / split_predict / split_equations

```julia
split_karva(chromosome::Chromosome, coeffs::Int=2)
split_positions(chromosome::Chromosome, coeffs::Int=2)
split_predict(chromosome::Chromosome, ctx, coeffs::Int=2)
split_equations(chromosome::Chromosome, coeffs::Int=2)
```

`split_karva` splits the chromosome into `coeffs` karva strings of consecutive genes, each with its own connectors (the first `coeffs - 1` connectors are dropped, and each part takes `gene_count ÷ coeffs` genes) -- the building block of template models with several factors, and of a chromosome that carries several expressions, e.g. the right-hand sides of a system of equations. `split_positions` gives the positions of the symbols of each part in `expression_raw`. `split_predict` evaluates the parts in a buffer context (`buffer_context`, one per thread from `thread_contexts`) and returns their outputs, copied out of the context's buffers; an entry that is not a vector marks a part that cannot be evaluated. `split_equations` writes each part as an equation string. Both apply the tuned constants of the chromosome (`optimised_constants`), the equations rounded to 6 significant digits.

### EvoSelection.SelectedMembers

The result of a selection: `indices::Vector{Int}` of the selected individuals, and `fronts` (the Pareto fronts by rank, after NSGA-II; empty after tournament selection).

## Loss Functions

### Built-in Loss Functions

`get_loss_function(name)` returns the function behind a name (an unknown name throws a `KeyError`); `fit!` accepts the names directly (`loss_fun="mse"`). Every one takes `(y_true, y_pred)`, two arrays of equal length and element type.

- `"mse"`: Mean squared error
- `"rmse"`: Root mean squared error
- `"mae"`: Mean absolute error
- `"nrmse"`: Root mean squared error divided by the standard deviation of `y_true`
- `"srsme"`: Relative root mean squared error (residuals divided by the target's magnitude)
- `"r2_score"`: Coefficient of determination -- a score, higher is better
- `"r2_score_f"`: `r2_score` on data rescaled by a power of ten, for very large or small magnitudes
- `"xi_core"`: A rank correlation modelled on Chatterjee's ξ -- a score

The search minimises its loss; use the scores for reporting.

### Custom Loss Functions

#### Single-Objective Custom Loss
```julia
using Statistics

# regressor, x_data (one row per sample), y_data, epochs and population_size as in the examples
# mean absolute error, weighting the residuals of large targets less
function custom_loss(y_true, y_pred)
    return mean(abs.(y_true .- y_pred) ./ (1 .+ abs.(y_true)))
end

fit!(regressor, epochs, population_size, x_data', y_data; loss_fun=custom_loss)
```

#### Multi-Objective Custom Loss
```julia
using Statistics

# n_features, x_data (one row per sample), y_data, epochs and population_size as in the examples
regressor = GepRegressor(n_features; number_of_objectives=2)
ctxs = thread_contexts(regressor.toolbox_, x_data')

function multi_objective_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        y_pred = try
            elem(ctxs[Threads.threadid()])
        catch
            nothing
        end
        if y_pred isa AbstractVector && all(isfinite, y_pred)
            mse = mean(abs2, y_data .- y_pred)                     # objective 1: accuracy
            complexity = 0.01 * length(elem.expression_raw)      # objective 2: size, scaled
            elem.fitness = (mse, complexity)
        else
            elem.fitness = (Inf, Inf)
        end
    end
end

fit!(regressor, epochs, population_size, multi_objective_loss)
```

`Inf` is a valid penalty: tournament selection leaves out individuals whose (first) objective is not finite, falling back to the whole population when none is finite, and NSGA-II ranks tuples with more non-finite entries behind.

#### Tensor Custom Loss
```julia
using LinearAlgebra, Statistics

# regressor: a GepTensorRegressor after allocate_buffers!; target_tensors: the target column
function tensor_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        predictions = try
            predictT(regressor, elem.expression_raw)
        catch
            nothing
        end
        if predictions isa AbstractVector && length(predictions) == length(target_tensors) &&
           eltype(predictions) == eltype(target_tensors)
            total_error = sum(norm(predictions[i] - target_tensors[i])^2
                              for i in eachindex(target_tensors))
            elem.fitness = (total_error / length(target_tensors),)
        else
            elem.fitness = (1e6,)    # e.g. a scalar where a vector belongs
        end
    end
end
```

#### Template loss
Template losses enable multi-expression formulations or fixed templates like $f(x_1,..x_n)=v(x_1,x_2) + g(x_3)$: `split_karva` splits a chromosome into groups of genes, and `calc_stack_batch_tensor` evaluates each group on the input columns (for a `GepRegressor`, `split_predict(elem, ctx, 2)` evaluates the groups in a buffer context, as `examples/Main_surrogate_multi_objective.jl` does). Here with a `GepTensorRegressor`, whose toolbox holds the evaluator's operator objects:

```julia
using LinearAlgebra, Statistics

# inputs: Dict{Int8,Any}, one column per terminal symbol -- regressor.input_values after
# allocate_buffers!; t1, t2: your two basis columns; a_true: the target column
function template_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        try
            g1, g2 = split_karva(elem, 2)
            p1 = calc_stack_batch_tensor(g1, regressor.toolbox_.callbacks, inputs, nothing)
            p2 = calc_stack_batch_tensor(g2, regressor.toolbox_.callbacks, inputs, nothing)
            a_pred = @. p1 * t1 + p2 * t2
            elem.fitness = (norm(a_true - a_pred),)
        catch
            elem.fitness = (1e6,)
        end
    end
end
```

## Selection Methods

### Tournament Selection

The selection for a single objective: each parent is the best of `tourni_size` individuals drawn with replacement from those with a finite fitness, identical fitness values counting once; if no fitness is finite, from the whole population. The best individual is always among the parents. `fit!` uses tournaments of 3 % of the population (0.3 % for tensor regression), at least 3.

### NSGA-II Selection

The selection for several objectives (`GepRegressor(n; number_of_objectives=2)` with a custom loss): Pareto ranks by non-dominated sorting, and tournaments of 3 decided by rank, then by crowding distance.

`dominates_(a, b)` tells whether fitness tuple `a` dominates `b` (a tuple with fewer non-finite entries dominates one with more), which is how to pick the non-dominated models out of `best_models_`.

## Genetic Operators

The probabilities and rates of the genetic operators are the entries of `RegressionWrapper.GENE_COMMON_PROBS`. Every toolbox holds this dictionary itself, not a copy, so a change applies to existing regressors too:

```julia
using GeneExpressionProgramming
probs = GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS

probs["mutation_rate"] = 0.1
```

`list_all_genetic_params()` returns a copy of the current values. Crossover and fusion probabilities are per pair of parents, the others per offspring. [Core Concepts](core-concepts.md#Genetic-Operators) describes each operator.

| Key | Default | Meaning |
| --- | --- | --- |
| `one_point_cross_over_prob` | 0.5 | one-point crossover |
| `two_point_cross_over_prob` | 0.4 | two-point crossover |
| `mutation_prob` | 1.0 | point mutation |
| `mutation_rate` | 0.15 | a mutation redraws `round(0.15 × length)` positions, drawn with replacement |
| `inversion_prob` | 0.1 | inversion of head symbols |
| `insertion_prob` | 0.1 | a random terminal written into a head position |
| `root_insertion_prob` | 0.1 | rotation of a gene's head, changing its root |
| `reverse_insertion_tail` | 0.0 | rotation within a gene's tail |
| `gene_transposition_prob` | 0.1 | exchange of two tail segments |
| `gene_averaging_prob` | 1.0 | gene averaging towards the consensus of the elite |
| `gene_averaging_rate` | 0.3 | probability per position of taking the consensus symbol |
| `gene_averaging_elite_frac` | 0.3 | elite size, as a fraction of the mating size (at least 3) |
| `dominant_fusion_prob`, `rezessiv_fusion_prob`, `fusion_prob` | 0.0 each | fusion operators |
| `dominant_fusion_rate`, `rezessiv_fusion_rate`, `fusion_rate` | 0.1, 0.1, 0.0 | share of positions a fusion draws |
| `mating_size` | 0.7 | offspring per epoch, as a fraction of the population size |

## Function Sets

### Scalar Functions

Every function in `FUNCTION_LIB_COMMON` can be entered by its symbol:

```julia
basic_functions = [:+, :-, :*, :/]
power_functions = [:sqr, :sqrt, :^]
exp_log_functions = [:exp, :log, :log10, :log2]
trig_functions = [:sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh, :acosh, :atanh]
other_functions = [:abs, :sign, :floor, :ceil, :round, :min, :max]
```

Under physical dimensions, `min` and `max` take operands of one dimension, `abs` keeps its operand's, `sign` takes any and gives a dimensionless result, and `floor`, `ceil` and `round`, like `exp` or `sin`, take dimensionless operands only: rounding a quantity with units would give a result that depends on the units it is expressed in.

`list_all_functions()` lists each with its arity and its unit rules; `list_all_arity()`, `list_all_forward_handlers()` and `list_all_backward_handlers()` return copies of the single tables. `set_function!(sym, func)`, `set_arity!(sym, arity::Int8)`, `set_forward_handler!(sym, handler)`, `set_backward_handler!(sym, handler)` and `update_function!(sym; func, arity, forward_handler, backward_handler)` change an existing entry, for regressors built afterwards; a symbol that is not in the library throws an `ArgumentError`. The batched evaluator runs only functions it has an operator node for (the functions behind `TENSOR_NODES`). A regressor with any other function cannot be used: the data method of `fit!` throws an `ArgumentError`, so does predicting with it, `buffer_context` and `thread_contexts` return `nothing`, and the custom-loss method of `fit!` fails while it samples the initial population.

```julia
# abs of dimensionless operands only
update_function!(:abs; forward_handler=zero_unit_forward, backward_handler=zero_unit_backward)
```

### Tensor Operations

On top of the scalar functions, `GepTensorRegressor` accepts:

- **Products**: `:*` (products with a scalar), `:dot` (single contraction), `:dcontract` (double contraction), `:otimes` (outer product), `:crossp` (cross product of 3D vectors), `:hadamard` (element-wise)
- **Invariants and norms**: `:tr`, `:det`, `:norm`
- **Parts and derived tensors**: `:symmetric`, `:skew`, `:dev`, `:vol`, `:inv`, `:tdot` and `:dott` (both A·Aᵀ), `:lap`

With `considered_dimensions`, only the functions with tensor unit rules can be entered: `:+`, `:-`, `:*`, `:/`, `:inv`, `:dot`, `:crossp`, `:tr`, `:det`, `:dcontract`, `:lap`, `:hadamard`, `:sqrt`, `:norm`, `:log`, `:exp`, `:sin` and `:cos`.

The tensors are Tensors.jl types:

```julia
using Tensors

vector_3d = rand(Tensor{1,3})
matrix_3x3 = rand(Tensor{2,3})
```

## Physical Dimensionality

### Dimension Representation

Physical dimensions are 7-element `Float16` vectors of SI exponents, in the order [kg, m, s, K, mol, A, cd]:

```julia
# [Mass, Length, Time, Temperature, Amount of substance, Current, Luminous intensity]
velocity_dim = Float16[0, 1, -1, 0, 0, 0, 0]    # [L T⁻¹]
force_dim = Float16[1, 1, -2, 0, 0, 0, 0]       # [M L T⁻²]
energy_dim = Float16[1, 2, -2, 0, 0, 0, 0]      # [M L² T⁻²]
current_density_dim = Float16[0, -2, 0, 0, 0, 1, 0]   # [I L⁻²]
```

On the tensor path the vector carries the tensor order in front of these.

### Dimensional Constraints

```julia
# x_data: a mass, a length and a time per row (sample); y_data: a velocity per sample;
# epochs and population_size as in the examples
feature_dims = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[1, 0, 0, 0, 0, 0, 0],    # Mass
    :x2 => Float16[0, 1, 0, 0, 0, 0, 0],    # Length
    :x3 => Float16[0, 0, 1, 0, 0, 0, 0],    # Time
)

target_dim = Float16[0, 1, -1, 0, 0, 0, 0]  # Velocity

regressor = GepRegressor(3;
                        considered_dimensions=feature_dims,
                        max_permutations_lib=10000)

fit!(regressor, epochs, population_size, x_data', y_data;
     target_dimension=target_dim)
```

A constant terminal takes a dimension under its symbol, e.g. `Symbol(9.807) => get_constant_dims("g")` with `entered_terminal_nums=[Symbol(9.807)]`.

### Checking and Repairing Expressions

```julia
is_dimensionally_homogeneous(expression::Vector{Int8}, target_dimension, token_dto)
dimensional_homogeneity_distance(expression::Vector{Int8}, target_dimension, token_dto)
is_gene_wise_homogeneous(expression::Vector{Int8}, target_dimension, token_dto, gene_count)
gene_dimensions(expression::Vector{Int8}, gene_count, token_dto)
correct_genes!(genes, start_indices, expression, target_dimension, token_dto;
               cycles=5, gene_len=0, head_len=0, connectors=nothing, work_limit=800,
               gene_wise=false)
sample_lib_expression(target_dimension, token_dto; max_len, exact_only=false, head_len=0)
```

- `is_dimensionally_homogeneous` / `dimensional_homogeneity_distance`: Forward check of a karva string against a target (the distance is `Inf16` for an expression with an inconsistency anywhere)
- `is_gene_wise_homogeneous` / `gene_dimensions`: The same gene by gene, the check a model scored as the weighted sum of its genes (linear scaling) needs
- `correct_genes!`: The repair `fit!` uses. It pushes the target down the expression and applies the cheapest fixes -- swapping terminals or operators, splitting a requirement between operands, replacing subexpressions with library expressions -- within the gene layout (`gene_len`, `head_len`, `connectors`). Returns `(distance, success)` and leaves `genes` untouched on failure; recompile the chromosome after a success. With `gene_wise=true` every gene is repaired to the target on its own, and the connectors become `+` or `-` if `connectors` offer one
- `sample_lib_expression`: A random library-backed expression of (or near) a dimension, in prefix order

**Example** (`regressor` and `target_dim` as above):
```julia
tb, dto = regressor.toolbox_, regressor.token_dto_
c = generate_chromosome(tb)
_, ok = correct_genes!(c.genes, tb.gen_start_indices, c.expression_raw, target_dim, dto;
                       gene_len=2 * tb.head_len + 1, head_len=tb.head_len,
                       connectors=tb.gene_connections, cycles=10)
ok && compile_expression!(c; force_compile=true)
is_dimensionally_homogeneous(c.expression_raw, target_dim, dto)
```

### Physical Constants

```julia
get_constant(name)          # (value, dimension)
get_constant_value(name)
get_constant_dims(name)
physical_constants          # Dict of common constants, e.g. "c", "G", "h", "k_B", "e"
physical_constants_all      # a larger set; the get_constant* functions look up physical_constants only
```

### Units from a Description File

`get_feature_dims_json(case_data, feature_names, case_name)` and `get_target_dim_json(case_data, case_name)` read the dimensions of an equation from a parsed JSON description such as `assets/case_dsc.json`, as `paper/ConstraintViaSBP.jl` does. The feature dimensions come keyed by the names in `feature_names` as `String`s; convert the keys to `Symbol`s for `considered_dimensions`.

## Utility Functions

### train_test_split

```julia
train_test_split(X::AbstractMatrix{T}, y::AbstractVector{T}; train_ratio::T=0.9, consider::Int=1)
```

Shuffles the rows of `X` (one sample per row) and `y` with the global RNG, splits them at `train_ratio`, and keeps every `consider`-th row of each part.

**Returns:** `(x_train, y_train, x_test, y_test)`

**Example:**
```julia
x_train, y_train, x_test, y_test = train_test_split(X, y; train_ratio=0.8)
```

### Other Utilities

- `minmax_scale(X; feature_range=(0, 1))`: A copy of `X` with each column mapped linearly onto `feature_range`
- `isclose(a, b; rtol=1e-5, atol=1e-8)`: `abs(a - b) <= atol + rtol * abs(b)`
- `save_state(filename, state)` / `load_state(filename)`: Serialize and restore, for example a population in `save_state_callback` / `load_state_callback`
- `thread_slots()`: The number of per-thread slots buffers are sized by (`Threads.maxthreadid()`)
- `allfinite(A)`: `all(isfinite, A)` for a float array, in one vectorised pass without early exit
- `calc_stack_batch_tensor(rek_string, callbacks, inputs, buffers)`: The evaluator itself: runs a karva string on input columns

## Error Handling

### Common Errors

#### ArgumentError: an operator or terminal in this regressor has no batched counterpart
A function in the regressor has no node in the batched evaluator (see [Function Sets](#Function-Sets)).

#### ArgumentError: a target_dimension needs the features' dimensions
`fit!` got a `target_dimension` for a regressor built without `considered_dimensions`.

#### ArgumentError: linear_scaling writes the scaled model as a sum of weighted genes
`linear_scaling=true` needs `:+` and `:*` among the functions.

#### KeyError from get_loss_function
The loss name is not one of the [built-in losses](#Built-in-Loss-Functions).

#### An error inside a custom loss
An exception thrown inside a custom loss is not caught by the search: it stops `fit!`. Catch evaluation errors inside the loss and assign a penalty.

## Configuration Example

A larger search with more functions and held-out data (`x_train`, `x_test`: 5 features, one row per sample; `y_train`, `y_test`: the targets):

```julia
regressor = GepRegressor(
    5;                                    # 5 input features
    gene_count = 3,                       # 3 genes per chromosome
    head_len = 8,                         # Longer expressions
    entered_non_terminals = [:+, :-, :*, :/, :sin, :cos, :exp]
)

fit!(regressor, 1500, 2000, x_train', y_train;   # 1500 epochs, population of 2000
     x_test = x_test',
     y_test = y_test,
     loss_fun = "rmse")
```

For additional examples and use cases, refer to the examples, starting with [Basic Regression](examples/basic-regression.md).
