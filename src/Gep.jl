"""
    GepRegression

The evolutionary loop of Gene Expression Programming, [`runGep`](@ref), with the fitness
evaluation, dimension repair and population update it is built from.

`RegressionWrapper.fit!` is the usual entry point: it builds the toolbox, the evaluation
strategy and the dimension callbacks, then calls `runGep`. Called directly:

```julia
# loss(elem, validate) sets elem.fitness; see GenericRegressionStrategy
strategy = GenericRegressionStrategy(nothing, 1, loss)
best, history = runGep(epochs, population_size, toolbox, strategy; hof=3)
```

See also: [`GepEntities.Toolbox`](@ref), [`GepEntities.Chromosome`](@ref),
[`GepEntities.GenericRegressionStrategy`](@ref),
[`GepEntities.StandardRegressionStrategy`](@ref).
"""
module GepRegression


using ..GepUtils
using ..GepEntities
using ..TensorRegUtils
using ..LossFunction
using ..EvoSelection
using ..GepSurrogate


using Random
using Statistics
using LinearAlgebra
using ProgressMeter
using OrderedCollections
using Logging
using Distributions
using Printf
using LRUCache
using Base.Threads: SpinLock
using .Threads
using ThreadsX
export runGep, gene_basis, gene_basis!, solve_scaling, buffered_predict


"""
    buffered_predict(elem::Chromosome, b)

Evaluate `elem` with the batched evaluator in the calling thread's buffers of `b`, the
context built by `RegressionWrapper.build_buffers`: with the compiled program `b.program`
if there is one, with `calc_stack_batch_tensor` otherwise. Both write into the same
buffers, so the result is valid until the next evaluation on this thread. A chromosome
with `optimised_constants` is evaluated with them, on the allocating path
([`evaluate_with_constants`](@ref)). A result that is not a vector (e.g. `NaN`) marks an
expression that cannot be evaluated.
"""
@inline function buffered_predict(elem::Chromosome, b)
    isnothing(elem.optimised_constants) ||
        return evaluate_with_constants(elem, b, elem.optimised_constants)
    tid = Threads.threadid()
    prog = b.program
    isnothing(prog) || return run_program!(elem.expression_raw, prog,
        b.fast[tid], b.stacks[tid])
    return calc_stack_batch_tensor(elem.expression_raw, b.callbacks, b.nodes,
                                   b.pools[tid])
end

"""
    gene_basis(elem::Chromosome, ctx, n)

Design matrix `G` (`n` samples × genes) whose `j`-th column is gene `j` evaluated on its
own, without the connectors: the basis of the least-squares fit of the gene coefficients
([`solve_scaling`](@ref)). Returns `nothing` if a gene does not evaluate to a vector.

Each column is copied out as soon as it is computed, so one buffer pool serves all genes.
That pool is the calling thread's `ctx.gene_pools` entry, separate from the one
[`buffered_predict`](@ref) uses, so a prediction still held in those buffers survives.
"""
@inline function gene_basis(elem::Chromosome, ctx, n::Int)
    raw = _karva_raw(elem; split=true)
    return gene_basis!(Matrix{Float64}(undef, n, length(raw) - 1), raw, ctx)
end

"""
    gene_basis!(G, raw, ctx)

[`gene_basis`](@ref) into a given `G` (`n × genes`), for the split karva string
`raw = _karva_raw(elem; split=true)`. Returns `G`, or `nothing` if a gene does not evaluate
to a vector.
"""
@inline function gene_basis!(G::Matrix{Float64}, raw, ctx)
    # the per-thread lookups stay outside `@inbounds`, so an out-of-range thread id
    # raises a BoundsError instead of reading past the end of the pool vector
    tid = Threads.threadid()
    prog = ctx.program
    pool = ctx.gene_pools[tid]
    fast = ctx.gene_fast[tid]
    stack = ctx.stacks[tid]
    @inbounds for j in 2:length(raw)
        col = if isnothing(prog)
            calc_stack_batch_tensor(collect(raw[j]), ctx.callbacks, ctx.nodes, pool)
        else
            run_program!(raw[j], prog, fast, stack)
        end
        col isa AbstractVector || return nothing
        G[:, j-1] .= col
    end
    return G
end

"""
    solve_scaling(G, y)

Least-squares gene coefficients `w` for `G * w ≈ y`, from the normal equations with a
ridge of about `1e-12` times the mean diagonal of `G'G`: duplicated genes make columns of
`G` collinear, and the ridge only breaks such ties.
"""
@inline function solve_scaling(G::Matrix{Float64}, y::AbstractVector)
    k = size(G, 2)
    A = G' * G
    b = G' * y
    lambda = 1e-12 * (tr(A) / k + eps())
    @inbounds for i in 1:k
        A[i, i] += lambda
    end
    return A \ b
end

"""
    compute_fitness(elem::Chromosome, evalArgs::StandardRegressionStrategy;
        validate::Bool=false)

Score `elem` on the training data if it is unscored or `validate` is set:
`elem.fitness = (loss_function(y_data, prediction),)`. With `linear_scaling` the
prediction is the least-squares combination of the genes ([`gene_basis`](@ref),
[`solve_scaling`](@ref)), whose coefficients are stored in `elem.scaling_weights`;
otherwise it comes from [`buffered_predict`](@ref). A candidate that cannot be evaluated,
or throws, gets `(crash_value,)`.
"""
@inline function compute_fitness(elem::Chromosome, evalArgs::StandardRegressionStrategy; validate::Bool=false)
    try
        if isnan(mean(elem.fitness)) || validate
            if evalArgs.linear_scaling
                ctx = evalArgs.buffered
                n = length(evalArgs.y_data)
                raw = _karva_raw(elem; split=true)
                m = length(raw) - 1
                tid = Threads.threadid()
                # the calling thread's design matrix and prediction from `build_buffers`,
                # instead of allocating both for every candidate
                own = hasproperty(ctx, :designs) && 1 <= m <= length(ctx.designs[tid]) &&
                      size(ctx.designs[tid][m], 1) == n
                G = gene_basis!(own ? ctx.designs[tid][m] : Matrix{Float64}(undef, n, m),
                    raw, ctx)
                if isnothing(G) || !allfinite(G)
                    elem.fitness = (evalArgs.crash_value,)
                    return
                end
                w = solve_scaling(G, vec(evalArgs.y_data))
                if !all(isfinite, w)
                    elem.fitness = (evalArgs.crash_value,)
                    return
                end
                elem.scaling_weights = w
                pred = own ? mul!(ctx.preds[tid], G, w) : G * w
                elem.fitness = (evalArgs.loss_function(evalArgs.y_data, pred),)
            else
                y_pred = buffered_predict(elem, evalArgs.buffered)
                elem.fitness = y_pred isa AbstractVector ?
                               (evalArgs.loss_function(evalArgs.y_data, y_pred),) :
                               (evalArgs.crash_value,)
            end
        end
    catch e
        elem.fitness = (evalArgs.crash_value,)
    end
end

@inline function compute_fitness_validation(elem::Chromosome, evalArgs::StandardRegressionStrategy; validate::Bool=false)
    try
        if isnan(mean(elem.fitness)) || validate
            # predict with the coefficients and constants fitted on the training data;
            # never refit them here
            ctx = evalArgs.validation_ctx
            y_pred = isnothing(ctx) ? elem(evalArgs.x_data_test) : elem(ctx)
            return (evalArgs.validation_loss_function(evalArgs.y_data_test, y_pred),)
        end
    catch e
        return (evalArgs.crash_value,)
    end
end

# TODO: further strategies for wrapping custom functions
@inline function compute_fitness(elem::Chromosome, evalArgs::Union{GenericRegressionStrategy}; validate::Bool=false)
    evalArgs.loss_function(elem, validate)
end

@inline function compute_fitness_validation(elem::Chromosome, evalArgs::Union{GenericRegressionStrategy}; validate::Bool=false)
    evalArgs.validation_loss_function(elem, validate)
end

@inline function modify_fitness(t::Tuple, penalty::AbstractFloat)
    return ntuple(i -> t[i] * penalty, length(t))
end

"""
    foreach_balanced(f, items)

Call `f(item)` for every element of `items` on all threads, handing the items out one at
a time to whichever thread is free. `Threads.@threads` over the items would instead split
them into one fixed chunk per thread (`:static` and `:dynamic` alike): when their costs
differ, the threads with cheap chunks go idle while the rest work through theirs one item
at a time.

There is one worker per thread, pinned to it (`Threads.@threads :static` over the workers,
not over the items), so `threadid()` does not change during a call of `f` and no two calls
in progress share it: buffers indexed by `threadid()` stay safe even if `f` yields, e.g.
while it waits for an external solver. Once a call throws, no further items are started,
and the exception is rethrown as by `Threads.@threads`. Like `:static`, it cannot be
nested in another threaded loop.

The workers are started even for no items, as `Threads.@threads` starts its tasks for an
empty range: a spawn moves on the seed that later tasks draw their random numbers from, so
the number of tasks spawned is part of what makes a seeded run reproduce.
"""
function foreach_balanced(f, items::AbstractVector)
    n = length(items)
    next = Threads.Atomic{Int}(1)
    failed = Threads.Atomic{Bool}(false)
    Threads.@threads :static for _ in 1:min(n, Threads.nthreads())
        while !failed[]
            k = Threads.atomic_add!(next, 1)
            k > n && break
            try
                f(items[k])
            catch
                failed[] = true
                rethrow()
            end
        end
    end
    return nothing
end


"""
    sort_by_fitness!(population)

`sort!(population, by = x -> mean(x.fitness))` with each key computed once rather than
per comparison: the same order, since both sorts are stable.
"""
function sort_by_fitness!(population::Vector{Chromosome})
    fits = [mean(x.fitness) for x in population]
    permute!(population, sortperm(fits))
    return population
end

"""
    breed!(offspring, parents, population, toolbox, n, mating_size, generation,
        max_generation)

Breed `n` offspring from `parents[1:n]` into `offspring[1:n]`, pair by pair in parallel
([`genetic_operations!`](@ref), one RNG per pair split from `toolbox.master_rng`), and
compile them. The elite pool of `gene_averaging!` is the front of the sorted `population`,
sized by `mating_size`.
"""
function breed!(offspring::Vector{Chromosome}, parents::Vector{Chromosome},
    population::Vector{Chromosome}, toolbox::Toolbox, n::Int, mating_size::Int,
    generation::Int, max_generation::Int)
    subkeys = split_rng(toolbox.master_rng,div(n,2)+1)

    # `population` is sorted, so its front is the elite pool `gene_averaging!` reads its
    # consensus from: the fraction `gene_averaging_elite_frac` (0.05 if unset) of the
    # mating size, rounded down, at least 3 and at most the whole population. The elites
    # do not change while the offspring are bred, so their symbol frequencies are counted
    # once here rather than once per offspring (`top_k = 3`, `gene_averaging!`'s default).
    elite_frac = get(toolbox.gep_probs, "gene_averaging_elite_frac", 0.05)
    n_elites = min(length(population), max(3, floor(Int, elite_frac * mating_size)))
    elites = ConsensusSampler([e.genes for e in @view population[1:n_elites]], 3)

    @inbounds Threads.@threads for i in 1:2:n-1
        offspring[i] = parents[i]
        offspring[i+1] = parents[i+1]

        genetic_operations!(offspring, i, toolbox;
            generation=generation, max_generation=max_generation, parents=parents,
            elites=elites, rng=subkeys[div(i,2)+1])

        compile_expression!(offspring[i]; force_compile=true)
        compile_expression!(offspring[i+1]; force_compile=true)

    end
    return offspring
end

"""
    perform_step!(population::Vector{Chromosome}, parents::Vector{Chromosome},
        next_gen::Vector{Chromosome}, toolbox::Toolbox, mating_size::Int,
        generation::Int, max_generation::Int)

Breed `mating_size` offspring from `parents[1:mating_size]` into `next_gen` ([`breed!`](@ref):
pair by pair in parallel, one RNG per pair split from `toolbox.master_rng`, compiled), and
insert them into the sorted `population`. With `m = mating_size`, the offspring take
positions `end-2m:end-m-1`, whose previous occupants move to `end-m:end-1`, replacing the
individuals there; `population[end]` is kept. `generation` and `max_generation` are passed
on to `genetic_operations!`, which does not use them.
"""
@inline function perform_step!(population::Vector{Chromosome}, parents::Vector{Chromosome}, next_gen::Vector{Chromosome},
    toolbox::Toolbox, mating_size::Int, generation::Int, max_generation::Int)
    breed!(next_gen, parents, population, toolbox, mating_size, mating_size, generation,
        max_generation)

    Threads.@threads for i in eachindex(next_gen)
        try
            population[end-i] = population[end-mating_size-i]
            population[end-mating_size-i] = next_gen[i]
        catch e
            error_message = sprint(showerror, e, catch_backtrace())
            @error "Error in perform_step!: $error_message"
        end
    end
end


"""
    seeded_by(f, seed)

`f()` with the generator of the calling task seeded by `seed`, and restored afterwards.
The repair draws from that generator, and a threaded pass runs it in tasks whose number
and generators depend on the thread count: seeded by the individual, a repair is the
same whatever task runs it, and the task's own stream is left as it was.
"""
function seeded_by(f, seed)
    rng = Random.default_rng()
    state = copy(rng)
    Random.seed!(rng, seed)
    try
        return f()
    finally
        copy!(rng, state)
    end
end

"""
    perform_correction_callback!(population::Vector{Chromosome}, epoch::Int,
        correction_epochs::Int, correction_amount::Real,
        correction_callback::Union{Function,Nothing};
        homogeneity_check::Union{Function,Nothing}=nothing, penalty::AbstractFloat=1.0)

Hold the unscored individuals of `population` to the target dimension before they are
scored; does nothing without a `correction_callback`.

1. Every epoch, each compiled, unscored individual not yet flagged `dimension_homogene`
   is tested, in parallel, with `homogeneity_check(expression_raw)`; a pass sets the flag.
2. Every `correction_epochs` epochs, the first `ceil(correction_amount *
   length(population))` of those that failed (all of them, without a check), in index
   order, are repaired in parallel by `correction_callback(genes, gen_start_indices,
   expression_raw, epoch)`, e.g. semantic backpropagation (`correct_genes!`), which
   returns `(distance, success)` and edits `genes` in place; the generator of the task
   is seeded by the genes and the epoch for it ([`seeded_by`](@ref)), so that a repair
   does not depend on the thread count. A successful repair is
   recompiled, and flagged only if the recompiled expression passes
   `homogeneity_check`, so the flag never rests on the repair's own claim; a failed or
   unconfirmed one gets the worst fitness (`fitness_reset[1]`).

`runGep` gives the worst fitness to every individual left unflagged, so only homogeneous
ones are scored. `penalty` is not used.
"""
@inline function perform_correction_callback!(population::Vector{Chromosome}, epoch::Int, correction_epochs::Int, correction_amount::Real,
    correction_callback::Union{Function,Nothing};
    homogeneity_check::Union{Function,Nothing}=nothing, penalty::AbstractFloat=1.0)

    isnothing(correction_callback) && return

    # pass 1, the read-only check, in parallel: each call writes only its own slot of
    # `needs`, a Vector{Bool} because the bits of a BitVector share words between tasks.
    # The new individuals sit together near the end of the population, so they are handed
    # out one at a time; fixed chunks of the whole range would leave them to the last threads
    needs = zeros(Bool, length(population))
    unchecked = [i for i in eachindex(population) if !population[i].dimension_homogene &&
                 population[i].compiled && isnan(mean(population[i].fitness))]
    foreach_balanced(unchecked) do i
        c = population[i]
        if !isnothing(homogeneity_check) && homogeneity_check(c.expression_raw)
            c.dimension_homogene = true
        else
            needs[i] = true
        end
    end
    epoch % correction_epochs == 0 || return

    # pass 2, on correction epochs: repair the first candidates, up to the budget
    candidates = findall(needs)
    repair_budget = min(length(candidates), Int(ceil(length(population) * correction_amount)))
    ThreadsX.foreach(view(candidates, 1:repair_budget)) do i
        c = population[i]
        _, correction = seeded_by(hash(c.genes, hash(epoch))) do
            correction_callback(c.genes, c.toolbox.gen_start_indices, c.expression_raw, epoch)
        end
        # a recompilation that fails leaves the old expression and the worst fitness
        correction && compile_expression!(c; force_compile=true)
        if correction && isnan(mean(c.fitness)) &&
           (isnothing(homogeneity_check) || homogeneity_check(c.expression_raw))
            c.dimension_homogene = true
            @debug "Dimension correction successful"
        else
            c.fitness = c.toolbox.fitness_reset[1]
        end
    end
end

"""
    demote_clones!(population)

Move every individual whose fitness exactly equals that of an earlier one behind all the
others, keeping both groups in order, and return `population`, which should be sorted by
fitness. Exact ties are, in practice, semantic copies (the same predictions from a
different karva string), which the fitness cache, keyed by the karva string, misses.

Survival keeps the top of the ranking, so without this the neutral variants of the best
individual (a model times zero, say) can fill it. Tournament selection already counts
equal fitness values once; copies still survive when too few distinct individuals exist.
"""
function demote_clones!(population::AbstractVector{Chromosome})
    seen = Set{Tuple}()
    distinct = Chromosome[]
    clones = Chromosome[]
    for c in population
        if c.fitness in seen
            push!(clones, c)
        else
            push!(seen, c.fitness)
            push!(distinct, c)
        end
    end
    isempty(clones) && return population
    n = length(distinct)
    population[1:n] = distinct
    population[n+1:end] = clones
    return population
end

"""
    select_brood_parents(selected, fits, n; rng)

Extend an NSGA-II selection, which picks as many parents as there are individuals, to `n`
parents for an oversampled epoch, by further selections over the same fitness values.
"""
function select_brood_parents(selected, fits::Vector{Tuple}, n::Int; rng::AbstractRNG)
    while length(selected.indices) < n
        append!(selected.indices, nsga_selection(fits; rng=rng).indices)
    end
    return selected
end

"""
    perform_brood_step!(population, parents, brood, next_gen, toolbox, mating_size,
        generation, max_generation, surrogate; correction_callback=nothing,
        homogeneity_check=nothing, correction_epochs=1, correction_amount=1.0)

The step of an oversampled epoch: breed `length(brood)` children from `parents` into
`brood` ([`breed!`](@ref)), let the surrogate pick the `mating_size` that enter the
population (`GepSurrogate.preselect`), and insert them as [`perform_step!`](@ref) does. With
a dimensional target the brood is checked and repaired first
([`perform_correction_callback!`](@ref), for the epoch the children are scored in), since
the pick ranks the expressions that will be scored, and a child that is not homogeneous is
taken only where too few others are left. Should the pick return fewer children, the
places left over keep their individuals.
"""
function perform_brood_step!(population::Vector{Chromosome}, parents::Vector{Chromosome},
    brood::Vector{Chromosome}, next_gen::Vector{Chromosome}, toolbox::Toolbox,
    mating_size::Int, generation::Int, max_generation::Int, surrogate;
    correction_callback::Union{Function,Nothing}=nothing,
    homogeneity_check::Union{Function,Nothing}=nothing,
    correction_epochs::Int=1, correction_amount::Real=1.0)
    breed!(brood, parents, population, toolbox, length(brood), mating_size, generation,
        max_generation)
    perform_correction_callback!(brood, generation + 1, correction_epochs, correction_amount,
        correction_callback; homogeneity_check=homogeneity_check)
    # a child that already holds a fitness could not be compiled or repaired, and one that
    # is not homogeneous will be given the worst fitness
    failed = [!isnan(mean(c.fitness)) ||
              (!isnothing(correction_callback) && !c.dimension_homogene) for c in brood]
    kept = preselect(surrogate, brood, mating_size; failed=failed)
    for (k, j) in enumerate(kept)
        next_gen[k] = brood[j]
    end
    for i in 1:length(kept)
        try
            population[end-i] = population[end-mating_size-i]
            population[end-mating_size-i] = next_gen[i]
        catch e
            error_message = sprint(showerror, e, catch_backtrace())
            @error "Error in perform_brood_step!: $error_message"
        end
    end
end

"""
    validate_leaders!(population, surrogate, evalStrategy, toolbox, hof;
        correction_callback=nothing)

Score the predicted individuals among the first `hof` of the sorted `population` with the
loss, record them in the surrogate and sort again, until the first `hof` carry no
prediction, so that a hall of fame reports losses only. An individual whose fitness is not
finite is not a prediction and is left alone, as is one that is not homogeneous under a
dimensional target.
"""
function validate_leaders!(population::Vector{Chromosome}, surrogate,
    evalStrategy::EvaluationStrategy, toolbox::Toolbox, hof::Int;
    correction_callback::Union{Function,Nothing}=nothing)
    surrogate.validate_hof || return population
    for _ in eachindex(population)
        todo = [i for i in 1:min(hof, length(population))
                if !is_validated(surrogate, population[i]) &&
                   all(isfinite, population[i].fitness) &&
                   (isnothing(correction_callback) || population[i].dimension_homogene)]
        isempty(todo) && break
        # the loss scores unscored individuals, so the prediction is dropped first
        for i in todo
            population[i].fitness = toolbox.fitness_reset[2]
        end
        foreach_balanced(i -> compute_fitness(population[i], evalStrategy), todo)
        for i in todo
            record_validation!(surrogate, population[i])
        end
        sort_by_fitness!(population)
        isnothing(correction_callback) || demote_clones!(population)
    end
    return population
end

"""
    runGep(epochs::Int, population_size::Int, toolbox::Toolbox,
        evalStrategy::EvaluationStrategy; hof::Int=3, correction_callback=nothing,
        homogeneity_check=nothing, population_seeder=nothing, correction_epochs::Int=1,
        correction_amount::Real=0.6, tourni_size::Int=3, optimization_epochs::Int=500,
        file_logger_callback=nothing, save_state_callback=nothing,
        load_state_callback=nothing, population_sampling_multiplier::Int=100,
        inputs_::Int=0, cache_size::Int=10000, penalty::AbstractFloat=2.0,
        surrogate=nothing)

Evolve a population for up to `epochs` epochs, scoring it with `evalStrategy`, and return
`(best, history)`: the hall of fame (the `hof` best chromosomes) and an
`OptimizationHistory` of the best individual's training and validation fitness per epoch.

The population holds `population_size + m` chromosomes, with the mating size `m` equal to
`ceil(population_size * gep_probs["mating_size"])` rounded down to even. Only the first
`population_size` are scored and take part in selection.

# Keyword arguments
- `correction_callback`, `homogeneity_check`: a dimensional target, as the repair
  `(genes, gen_start_indices, expression_raw, epoch) -> (distance, success)` and the
  check `expression_raw -> Bool` (see [`perform_correction_callback!`](@ref)). With a
  correction callback only homogeneous individuals are scored, and exact fitness copies
  rank behind distinct individuals ([`demote_clones!`](@ref)).
- `correction_epochs`, `correction_amount`: repair every `correction_epochs` epochs, at
  most `correction_amount` of `population_size` individuals
- `population_seeder`: `population -> nothing`, called on the initial population unless
  the run resumes at a later epoch
- `tourni_size`: tournament size, for one objective (NSGA-II selection otherwise)
- `optimization_epochs`: every this many epochs, `evalStrategy.secOptimizer(population)`
  runs if the best fitness improved since its last run
- `file_logger_callback`: `(population[1:population_size], epoch, selected) -> nothing`,
  called every epoch
- `save_state_callback`: `(population, evalStrategy) -> nothing`, called every epoch
- `load_state_callback`: `() -> (population, start_epoch)`, replaces the random start
- `population_sampling_multiplier`: above 1, a fresh initial population is chosen from
  `population_size * population_sampling_multiplier` random chromosomes by Latin
  hypercube sampling of their mean predictions on uniform random probe data: a
  `100 × K` matrix read as 100 feature rows and `K` samples, `K = inputs_` (10 if 0)
- `cache_size`: capacity of the fitness cache, keyed by karva string
- `penalty`: a chromosome whose karva string is cached, or is queued earlier in the same
  epoch, gets the cached fitness times `penalty`
- `surrogate`: a `GepSurrogate.SurrogateScreening`, for an expensive loss: of the
  individuals step 2 would score, the loss scores only the few the surrogate picks, and the
  others get its prediction (see below)

# Each epoch
1. Check and repair the new individuals ([`perform_correction_callback!`](@ref)).
2. Score the unscored among the first `population_size`, in parallel, once per distinct
   karva string; each thread takes the next individual as soon as it is free
   (`foreach_balanced`).
3. Sort the population by mean fitness (and demote clones, with a dimensional target);
   run the secondary optimiser when due.
4. Re-score the best with `validate = true`, record its training and validation fitness,
   and stop if `break_condition(population[1:population_size], epoch)` holds.
5. Select parents and, unless this is the last epoch, breed the next generation
   ([`perform_step!`](@ref)).

# With a surrogate
- Step 2 embeds the individuals it would score (one that cannot be embedded gets the
  worst fitness at once), the loss scores the ones the surrogate picks, in parallel, and
  every other one gets a prediction (`GepSurrogate.screen_epoch!`, `commit_epoch!`). A
  prediction is never cached; a copy of a string the loss has scored takes that loss, with
  the penalty, even once the fitness cache has dropped it. A prediction is provisional
  (by default with several objectives): an individual that carries one is scored again in
  step 2 of the next epoch, where the surrogate may pick it for the loss
  (`GepSurrogate.rescreen_predictions!`).
- Step 3 keeps, with several objectives, every scored individual that holds the best value
  of an objective among the survivors: a prediction can beat it by its mean fitness
  without dominating it (`GepSurrogate.keep_best_scored!`).
- Step 4 re-scores the best only if it carries a prediction. The population is not sorted
  again, so a best whose loss turns out worse than its prediction (a run that diverged,
  say) still leads it, and the epoch records the scored individual with the lowest mean
  fitness instead (`GepSurrogate.best_scored_index`).
- Step 5 breeds `GepSurrogate.brood_size` children, from as many parents, and the
  surrogate picks the `m` that enter ([`perform_brood_step!`](@ref)).
- An oversampled initial population is picked over the latent vectors of the surrogate
  (`GepSurrogate.characterize`), and the predicted members of the returned hall of fame
  are scored at the end ([`validate_leaders!`](@ref)).
"""
@inline function runGep(epochs::Int,
    population_size::Int,
    toolbox::Toolbox,
    evalStrategy::EvaluationStrategy;
    hof::Int=3,
    correction_callback::Union{Function,Nothing}=nothing,
    homogeneity_check::Union{Function,Nothing}=nothing,
    population_seeder::Union{Function,Nothing}=nothing,
    correction_epochs::Int=1,
    correction_amount::Real=0.6,
    tourni_size::Int=3,
    optimization_epochs::Int=500,
    file_logger_callback::Union{Function,Nothing}=nothing,
    save_state_callback::Union{Function,Nothing}=nothing,
    load_state_callback::Union{Function,Nothing}=nothing,
    population_sampling_multiplier::Int=100,
    inputs_::Int=0,
    cache_size::Int=10000,
    penalty::AbstractFloat=2.0,
    surrogate::Union{SurrogateScreening,Nothing}=nothing)

    isnothing(surrogate) || check_acquisition(surrogate.screen, length(toolbox.fitness_reset[1]))
    recorder = HistoryRecorder(epochs, Tuple)
    mating_ = toolbox.gep_probs["mating_size"]
    mating_size = Int(ceil(population_size * mating_))
    mating_size = mating_size % 2 == 0 ? mating_size : mating_size - 1
    # a surrogate may breed more children than the population takes and pick among them
    brood = isnothing(surrogate) ? mating_size : brood_size(surrogate, mating_size)
    fits_representation = Vector{Tuple}(undef, population_size)
    # keyed by a copy of the karva string: the chromosome's own vector may be recompiled
    fit_cache = LRU{Vector{Int8},Tuple}(maxsize=cache_size)
    # scoring scratch, reused across epochs; `fit_cache` is only touched in serial passes
    pending = Dict{Vector{Int8},Int}()
    work = Int[]
    dups = Tuple{Int,Vector{Int8}}[]
    redo = Int[]

    initial_size = population_sampling_multiplier <= 1 ? population_size + mating_size : population_size * population_sampling_multiplier
    population, start_epoch = isnothing(load_state_callback) ? (generate_population(initial_size, toolbox), 1) : load_state_callback()
    if start_epoch <= 1 && population_sampling_multiplier > 1
        if isnothing(surrogate) || !surrogate.characterize_initial
            prob_dataset = rand(toolbox.master_rng,100, inputs_ == 0 ? 10 : inputs_)'
            population = population[equation_characterization_default(population, population_size + mating_size, prob_dataset')]
        else
            # the start fills the latent space the search is screened in
            population = population[characterize(surrogate, population, population_size + mating_size)]
        end
    end
    # seeds only a fresh population -- a state loaded mid-run is already evolved
    if start_epoch <= 1 && !isnothing(population_seeder)
        population_seeder(population)
    end

    next_gen = Vector{eltype(population)}(undef, mating_size)
    brood_gen = brood > mating_size ? Vector{eltype(population)}(undef, brood) : next_gen
    progBar = Progress(epochs; showspeed=true, desc="Training: ")
    prev_best = toolbox.fitness_reset[1]

    for epoch in start_epoch:epochs
        same = Atomic{Int}(0)
        # a prediction is provisional: the individuals that carry one are screened again,
        # with what the surrogate has learned since
        rescreened = isnothing(surrogate) ? nothing : rescreen_predictions!(surrogate,
            population, population_size, toolbox.fitness_reset[2])
        perform_correction_callback!(population[1:population_size], epoch, correction_epochs, correction_amount,
            correction_callback; homogeneity_check=homogeneity_check)

        # Decide serially, by index, which individuals to evaluate, so that nothing depends
        # on thread scheduling and seeded runs reproduce: the first occurrence of a karva
        # string gets the true fitness, later ones the penalised copy, and cache lookups
        # (which reorder the LRU) happen in a fixed order.
        empty!(pending)
        empty!(work)
        empty!(dups)
        @inbounds for i in 1:population_size
            isnan(mean(population[i].fitness)) || continue
            # with a dimensional target only homogeneous individuals are scored, whatever
            # the loss; the rest get the worst fitness
            if !isnothing(correction_callback) && !population[i].dimension_homogene
                population[i].fitness = toolbox.fitness_reset[1]
                continue
            end
            raw = population[i].expression_raw
            # with a surrogate, a string the loss has scored is a copy even once the cache
            # has dropped it, since the surrogate keeps every loss it has seen
            if haskey(fit_cache, raw) || haskey(pending, raw) ||
               (!isnothing(surrogate) && known_expression(surrogate, population[i]))
                push!(dups, (i, copy(raw)))
            else
                pending[copy(raw)] = i
                push!(work, i)
            end
        end

        # evaluation writes into per-thread buffers indexed by `threadid()`: the workers
        # of `foreach_balanced` stay on their threads, and each takes the next individual
        # when it is free, so an expensive one holds up only the thread it runs on
        if isnothing(surrogate)
            foreach_balanced(i -> compute_fitness(population[i], evalStrategy), work)
            scored = work
        else
            # the loss scores the individuals the surrogate picks, and the others get its
            # prediction, which is never cached: a copy is screened again
            plan = screen_epoch!(surrogate, population, work, toolbox.fitness_reset[1];
                rescreened=rescreened)
            foreach_balanced(i -> compute_fitness(population[i], evalStrategy),
                evaluated_indices(plan))
            commit_epoch!(surrogate, population, plan)
            scored = cached_indices(plan)
        end

        @inbounds for i in scored
            # a loss may leave an individual unscored (NaN); caching that would pass NaN
            # on to every later copy
            isnan(mean(population[i].fitness)) && continue
            fit_cache[copy(population[i].expression_raw)] = population[i].fitness
        end
        # a copy inherits the cached fitness, scaled by `penalty`. A string cached when it
        # was queued but evicted since is scored again, once, on this thread: the tasks a
        # threaded pass spawns would move the seeds of every task spawned after them, the
        # repair's among them, so whether a string was evicted would change what a seed
        # produces
        empty!(redo)
        @inbounds for (i, key) in dups
            atomic_add!(same, 1)
            cached = get(fit_cache, key, nothing)
            isnothing(cached) && !isnothing(surrogate) && (cached = rescore_known(surrogate, key))
            if !isnothing(cached)
                population[i].fitness = modify_fitness(cached, penalty)
            elseif !haskey(pending, key)
                pending[key] = i
                push!(redo, i)
            end
        end
        if !isempty(redo)
            for i in redo
                compute_fitness(population[i], evalStrategy)
                isnothing(surrogate) || record_validation!(surrogate, population[i])
            end
            @inbounds for i in redo
                isnan(mean(population[i].fitness)) && continue
                fit_cache[copy(population[i].expression_raw)] = population[i].fitness
            end
        end
        # the other copies of a string scored in this epoch but not in the cache take the
        # fitness of the individual scored for it; if the loss left that one unscored,
        # they stay unscored too rather than calling the loss again on the same string
        @inbounds for (i, key) in dups
            isnan(mean(population[i].fitness)) || continue
            j = get(pending, key, 0)
            (j == 0 || j == i) && continue
            held = population[j].fitness
            isnan(mean(held)) && continue
            population[i].fitness = modify_fitness(held, penalty)
            # the copy of a prediction is a prediction, and screened again like one
            !isnothing(surrogate) && carries_prediction(surrogate, population[j]) &&
                mark_prediction!(surrogate, population[i])
        end

        sort_by_fitness!(population)
        # with a dimensional target, repair can turn many offspring into scored neutral
        # variants of the best, which would crowd out the rest
        isnothing(correction_callback) || demote_clones!(population)
        # the population survives by its mean fitness, which a prediction can beat without
        # dominating anyone: the scored holders of the best values survive regardless
        isnothing(surrogate) || keep_best_scored!(surrogate, population, population_size)

        Threads.@threads for index in eachindex(population[1:population_size])
            fits_representation[index] = population[index].fitness
        end

        if !isnothing(evalStrategy.secOptimizer) && epoch % optimization_epochs == 0 && population[1].fitness < prev_best
            evalStrategy.secOptimizer(population)
            fits_representation[1] = population[1].fitness
            prev_best = fits_representation[1]
        end

        # `validate = true` forces evaluation, so with a dimensional target a best that is
        # not homogeneous is not re-scored
        if isnothing(correction_callback) || population[1].dimension_homogene
            if isnothing(surrogate)
                compute_fitness(population[1], evalStrategy; validate=true)
            elseif !is_validated(surrogate, population[1])
                # a best that only carries a prediction is scored by the loss, and that
                # loss is what the epoch selects with
                compute_fitness(population[1], evalStrategy; validate=true)
                record_validation!(surrogate, population[1])
                fits_representation[1] = population[1].fitness
                isnan(mean(population[1].fitness)) ||
                    (fit_cache[copy(population[1].expression_raw)] = population[1].fitness)
            end
        end
        # the epoch records its best scored individual: a best scored above whose loss turned
        # out worse than its prediction (a run that diverged, say) still leads the population
        lead = isnothing(surrogate) ? 1 : best_scored_index(surrogate, population, population_size)
        val_loss = if isnothing(evalStrategy.validation_loss_function) ||
                      !(isnothing(correction_callback) || population[lead].dimension_homogene)
            population[lead].fitness
        else
            compute_fitness_validation(population[lead], evalStrategy; validate=true)
        end
        record!(recorder, epoch, fits_representation[lead], val_loss)

        ProgressMeter.update!(progBar, epoch, showvalues=[
            (:epoch_, @sprintf("%.0f", epoch)),
            (:duplicates_per_epoch, @sprintf("%.0f", same[])),
            (:train_loss, @sprintf("%.6e", mean(fits_representation[lead]))),
            (:validation_loss, @sprintf("%.6e", mean(val_loss)))
        ])

        !isnothing(evalStrategy.break_condition) && evalStrategy.break_condition(population[1:population_size], epoch) && break


        # an oversampled epoch selects a parent per child it breeds
        if length(fits_representation[1]) == 1
            selectedMembers = tournament_selection(fits_representation, brood, tourni_size; rng=toolbox.master_rng)
        else
            selectedMembers = nsga_selection(fits_representation; rng=toolbox.master_rng)
            brood > mating_size && select_brood_parents(selectedMembers, fits_representation,
                brood; rng=toolbox.master_rng)
        end

        !isnothing(file_logger_callback) && file_logger_callback(population[1:population_size], epoch, selectedMembers)
        !isnothing(save_state_callback) && save_state_callback(population, evalStrategy)

        if epoch < epochs
            parents = population[selectedMembers.indices]
            if brood > mating_size
                perform_brood_step!(population, parents, brood_gen, next_gen, toolbox,
                    mating_size, epoch, epochs, surrogate;
                    correction_callback=correction_callback,
                    homogeneity_check=homogeneity_check,
                    correction_epochs=correction_epochs, correction_amount=correction_amount)
            else
                perform_step!(population, parents, next_gen, toolbox, mating_size, epoch, epochs)
            end
        end

    end

    sort_by_fitness!(population)
    isnothing(correction_callback) || demote_clones!(population)
    # the hall of fame reports losses, not predictions
    isnothing(surrogate) || validate_leaders!(population, surrogate, evalStrategy, toolbox,
        hof; correction_callback=correction_callback)

    best = population[1:hof]
    close_recorder!(recorder)
    return best, recorder.history
end

end
