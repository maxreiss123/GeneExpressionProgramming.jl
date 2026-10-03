"""
    GepEntities

Core data structures and genetic operators of Gene Expression Programming (GEP).

A [`Chromosome`](@ref) is a vector of `Int8` symbols: `gene_count - 1` connectors, then
`gene_count` genes of `2 * head_len + 1` symbols (head + tail), then, optionally,
`gene_count` preamble symbols. [`compile_expression!`](@ref) resolves it into its karva
string (`expression_raw`): the connectors, then each gene's active part, in prefix order.
The batched evaluator runs karva strings on data columns.

A [`Toolbox`](@ref) holds the alphabet, gene layout, operator probabilities and RNG a
population is built and varied with; an [`EvaluationStrategy`](@ref) tells `runGep` how
to score it. The genetic operators modify chromosomes in place;
[`genetic_operations!`](@ref) applies them to a pair of offspring.
"""
module GepEntities


export Chromosome, Toolbox, EvaluationStrategy, StandardRegressionStrategy, GenericRegressionStrategy
export fitness, set_fitness!
export generate_gene, compile_expression!, generate_chromosome, generate_population
export buffer_context, thread_contexts, equation_string, constant_positions, evaluate_with_constants
export genetic_operations!, replicate, gene_inversion!, gene_mutation!, gene_one_point_cross_over!, gene_two_point_cross_over!, gene_fussion!, gene_averaging!, split_karva, print_karva_strings
export split_predict, split_equations, split_positions
export retrieve_coefficients
export prob_equation_djl, equation_characterization_default, _karva_raw


using ..GepUtils
using ..TensorRegUtils
using OrderedCollections
using StatsBase
using Random
using Random123

const STD_RNG = MersenneTwister()

"""
    EvaluationStrategy

How `runGep` scores chromosomes: [`StandardRegressionStrategy`](@ref) or
[`GenericRegressionStrategy`](@ref).
"""
abstract type EvaluationStrategy end

"""
    StandardRegressionStrategy{T}(operators, x_data, y_data, x_data_test, y_data_test,
        loss_function; validation_loss_function=nothing, secOptimizer=nothing,
        break_condition=nothing, penalty=zero(T), crash_value=typemax(T),
        linear_scaling=false, buffered=nothing, validation_ctx=nothing)

Single-objective scoring on data arrays: `loss_function(y_true, y_pred)` on the training
targets, `validation_loss_function` (default `loss_function`) on the test data.

- `buffered`: batched-evaluator context from `build_buffers`, holding the training
  inputs; without it every candidate scores `crash_value`
- `validation_ctx`: a [`buffer_context`](@ref) on `x_data_test`, reused for every
  validation; without it each validation builds a fresh one
- `linear_scaling`: fit one least-squares coefficient per gene before scoring
- `secOptimizer`: `population -> nothing`, run by `runGep` on the sorted population
- `break_condition`: `(population, epoch) -> Bool`; `true` stops the run
- `crash_value`: the fitness of a candidate that cannot be evaluated
- `operators`, `x_data` and `penalty` are stored but not used
"""
struct StandardRegressionStrategy{T<:AbstractFloat} <: EvaluationStrategy
    operators::Any
    number_of_objectives::Int
    x_data::AbstractArray{T}
    y_data::AbstractArray{T}
    x_data_test::AbstractArray{T}
    y_data_test::AbstractArray{T}
    loss_function::Function
    validation_loss_function::Function
    secOptimizer::Union{Function,Nothing}
    break_condition::Union{Function,Nothing}
    penalty::T
    crash_value::T
    linear_scaling::Bool
    buffered::Any
    validation_ctx::Any

    function StandardRegressionStrategy{T}(operators::Any,
        x_data::AbstractArray,
        y_data::AbstractArray,
        x_data_test::AbstractArray,
        y_data_test::AbstractArray,
        loss_function::Function;
        validation_loss_function::Union{Nothing,Function}=nothing,
        secOptimizer::Union{Function,Nothing}=nothing,
        break_condition::Union{Function,Nothing}=nothing,
        penalty::T=zero(T),
        crash_value::T=typemax(T),
        linear_scaling::Bool=false,
        buffered=nothing,
        validation_ctx=nothing) where {T<:AbstractFloat}
        new(operators,
            1,
            x_data,
            y_data,
            x_data_test,
            y_data_test,
            loss_function,
            isnothing(validation_loss_function) ? loss_function : validation_loss_function,
            secOptimizer,
            break_condition,
            penalty,
            crash_value,
            linear_scaling,
            buffered,
            validation_ctx
        )
    end

end

"""
    GenericRegressionStrategy(operators, number_of_objectives::Int, loss_function::Function;
        validation_loss_function=nothing, secOptimizer=nothing, break_condition=nothing)

Scoring by a user loss over chromosomes, for any number of objectives. `runGep` calls
`loss_function(elem, validate)` on unscored chromosomes in its threaded fitness loop (once
per distinct karva string), and with `validate = true` on the best of each epoch. The loop
hands the chromosomes out one at a time to whichever thread is free; a call stays on one
thread, and no other call shares its `threadid()` meanwhile, so buffers indexed by
`threadid()` are safe even in a loss that waits, e.g. on an external solver. The loss
sets `elem.fitness` to a tuple, one entry per objective; its return value is ignored, and
an exception it throws stops the run. `validation_loss_function(elem, validate)`, if given,
returns the validation fitness tuple. `operators` and `number_of_objectives` are stored
but not used.

See also: [`thread_contexts`](@ref).
"""
struct GenericRegressionStrategy <: EvaluationStrategy
    operators::Any
    number_of_objectives::Int
    loss_function::Function
    validation_loss_function::Union{Function,Nothing}
    secOptimizer::Union{Function,Nothing}
    break_condition::Union{Function,Nothing}

    function GenericRegressionStrategy(operators::Any, number_of_objectives::Int, loss_function::Function;
        validation_loss_function::Union{Function,Nothing}=nothing,
        secOptimizer::Union{Function,Nothing}=nothing,
        break_condition::Union{Function,Nothing}=nothing)
        new(operators, number_of_objectives, loss_function, validation_loss_function,
            secOptimizer, break_condition)
    end
end

"""
    Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8},
        gene_connections::Vector{Int8}, callbacks::Dict, nodes::OrderedDict,
        gep_probs::Dict{String,AbstractFloat}; unary_prob::Real=0.1, preamble_syms=Int8[],
        number_of_objectives::Int=1, operators_=nothing,
        function_complile=compile_djl_datatype, tail_weights_=nothing,
        head_tail_balance::Real=0.5, master_rng::AbstractRNG=STD_RNG,
        constant_indices=nothing)

The GEP configuration a population shares: alphabet, gene layout, operator probabilities
and RNG.

# Fields
- `gene_count`, `head_len`: genes per chromosome and head length; a gene has
  `2 * head_len + 1` symbols
- `symbols`, `arrity_by_id`: the same dictionary, every symbol with its arity
- `gene_connections`: connector symbols
- `headsyms`: symbols drawn for heads: binary operators, unary operators, then
  `tailsyms`, in the order of `head_weights`
- `tailsyms`: terminal symbols other than preamble symbols
- `callbacks`: the operator of every operator symbol (a function, or an operator object
  on the tensor path)
- `nodes`: the terminal of every terminal symbol (an `InputSelector` or a constant)
- `gen_start_indices`: position of each gene's first symbol in `genes`
- `gep_probs`: operator probabilities and rates (see `genetic_operations!`) and the
  mating fraction `mating_size`, by name
- `fitness_reset`: `(worst, unscored)`: all-`Inf` and all-`NaN` tuples with
  `number_of_objectives` entries
- `preamble_syms`, `len_preamble`: preamble symbols and their count
- `tail_weights`, `head_weights`: sampling weights. `tail_weights_`, or equal weights
  without it, for the tail; for the head, binary operators share `head_tail_balance`,
  unary operators `unary_prob`, and each tail symbol gets its tail weight times
  `1 - head_tail_balance - unary_prob` (`unary_prob` counts as 0 without unary operators)
- `master_rng`: source of the per-task RNGs; `split_rng` requires a `Threefry4x`
- `constant_indices`: constant symbols, for `retrieve_coefficients`
- `operators_`, `compile_function_` (from `function_complile`): stored but not used
"""
struct Toolbox
    gene_count::Int
    head_len::Int
    symbols::OrderedDict{Int8,Int8}
    gene_connections::Vector{Int8}
    headsyms::Vector{Int8}
    tailsyms::Vector{Int8}
    arrity_by_id::OrderedDict{Int8,Int8}
    callbacks::Dict
    nodes::OrderedDict
    gen_start_indices::Vector{Int}
    gep_probs::Dict{String,AbstractFloat}
    fitness_reset::Tuple
    preamble_syms::Vector{Int8}
    len_preamble::Int8
    operators_::Any
    compile_function_::Union{Function,Nothing}
    tail_weights::Union{Weights,Nothing}
    head_weights::Union{Weights,Nothing}
    master_rng::AbstractRNG
    constant_indices::Union{Vector{Int8},Nothing}


    function Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8}, gene_connections::Vector{Int8},
        callbacks::Dict, nodes::OrderedDict, gep_probs::Dict{String,AbstractFloat};
        unary_prob::Real=0.1, preamble_syms=Int8[],
        number_of_objectives::Int=1, operators_::Any=nothing,
        function_complile::Union{Function,Nothing}=compile_djl_datatype,
        tail_weights_::Union{Weights,Nothing}=nothing,
        head_tail_balance::Real=0.5,
        master_rng::AbstractRNG=STD_RNG, constant_indices::Union{Vector{Int8},Nothing}=nothing)

        fitness_reset = (
            ntuple(_ -> Inf, number_of_objectives),
            ntuple(_ -> NaN, number_of_objectives)
        )
        gene_len = head_len * 2 + 1
        headsyms = [key for (key, arity) in symbols if arity == 2]
        unary_syms = [key for (key, arity) in symbols if arity == 1]
        up = isempty(unary_syms) ? 0 : unary_prob

        tailsyms = [key for (key, arity) in symbols if arity < 1 && !(key in preamble_syms)]
        len_preamble = length(preamble_syms)
        gen_start_indices = [gene_count + (gene_len * (i - 1)) for i in 1:gene_count]

        tail_weights = isnothing(tail_weights_) ? weights([1 / length(tailsyms) for _ in 1:length(tailsyms)]) : tail_weights_
        head_weights = weights([
            fill(head_tail_balance / length(headsyms), length(headsyms));
            fill(unary_prob / length(unary_syms), length(unary_syms));
            tail_weights .* (1 - head_tail_balance - up)
        ])


        head_syms = vcat([headsyms, unary_syms, tailsyms]...)
        
        new(gene_count, head_len, symbols, gene_connections, head_syms, tailsyms, symbols,
            callbacks, nodes, gen_start_indices, gep_probs, fitness_reset, preamble_syms, len_preamble, operators_,
            function_complile,
            tail_weights, head_weights, master_rng, constant_indices)
    end
end

"""
    Chromosome(genes::Vector{Int8}, toolbox::Toolbox, compile::Bool=false)

An individual: a chromosome and the model it encodes. With `compile = true` the karva
string is resolved at once (see [`compile_expression!`](@ref)).

# Fields
- `genes`: the connectors, the genes (head + tail each), then the preamble symbols, if
  any
- `fitness`: a tuple, one entry per objective; all `NaN` until scored
- `toolbox`: the configuration it was built with
- `compiled`: whether `expression_raw` has been resolved (editing `genes` does not reset
  it)
- `expression_raw`: the karva string
- `dimension_homogene`: set once the expression is known to have the target dimension,
  by the check or by repair
- `chromo_id`: `-1`; not used
- `scaling_weights`: least-squares gene coefficients under linear scaling, else `nothing`
- `optimised_constants`: tuned values of the constant occurrences, in the order of
  [`constant_positions`](@ref), or `nothing`

`chromosome.compiled_function`, where older code read the compiled expression, gives a
[`CompiledModel`](@ref): the model, printed as its equation and callable on data.
"""
mutable struct Chromosome
    genes::Vector{Int8}
    fitness::Tuple
    toolbox::Toolbox
    compiled::Bool
    expression_raw::Vector{Int8}
    dimension_homogene::Bool
    chromo_id::Int
    scaling_weights::Union{Nothing,Vector{Float64}}
    optimised_constants::Union{Nothing,Vector{Float64}}

    function Chromosome(genes::Vector{Int8}, toolbox::Toolbox, compile::Bool=false)
        obj = new()
        obj.genes = genes
        obj.fitness = toolbox.fitness_reset[2]
        obj.toolbox = toolbox
        obj.compiled = false
        obj.dimension_homogene = false
        obj.chromo_id = -1
        obj.expression_raw = Int8[]
        obj.scaling_weights = nothing
        obj.optimised_constants = nothing
        if compile
            compile_expression!(obj)
        end
        return obj
    end
end



"""
    buffer_context(toolbox, x_data; std_return_type=Float64)

Evaluation context for the batched evaluator on `x_data` (one row per feature, one column
per sample): operator objects, one input column per terminal, one buffer pool, and a
compiled program when the alphabet allows one. Built from the toolbox alone, so a fitted
model can be evaluated without its regressor. Returns `nothing` when an operator or
terminal has no batched counterpart, as for a tensor toolbox, whose callbacks are
operator objects already.

A context has one set of buffers, so it serves one thread; see [`thread_contexts`](@ref).
"""
function buffer_context(toolbox, x_data::AbstractArray; std_return_type::Type=Float64)
    n = size(x_data, 2)
    callbacks = Dict{Int8,Any}()
    for (idx, f) in toolbox.callbacks
        haskey(TENSOR_NODE_BY_FUNCTION, f) || return nothing
        callbacks[idx] = TENSOR_NODE_BY_FUNCTION[f]()
    end
    nodes = Dict{Int8,Any}()
    for (idx, nd) in toolbox.nodes
        col = if nd isa Number
            fill(std_return_type(nd), n)
        elseif nd isa InputSelector
            Vector{std_return_type}(@view x_data[nd.idx, :])
        else
            return nothing
        end
        nodes[idx] = col
    end
    V = Vector{std_return_type}
    gene_buff = (toolbox.gene_count + 1) * toolbox.head_len
    pool = Dict{Type,NTuple}(V => Tuple([zeros(std_return_type, n) for _ in 1:gene_buff]))
    program = compile_program(callbacks, nodes, V)
    return (callbacks=callbacks, nodes=nodes, pool=pool, program=program,
        fast=collect(V, pool[V]), stack=sizehint!(V[], gene_buff + 2))
end

"""
    ctx_eval(rek_string, ctx)

Evaluate a karva string, or one gene's part of it, in a [`buffer_context`](@ref): with its
compiled program if it has one, with `calc_stack_batch_tensor` otherwise. Both write into
the context's buffers, so consume or copy a result before the next evaluation in the same
context. A result that is not a vector (e.g. `NaN`) marks an expression that cannot be
evaluated.
"""
@inline function ctx_eval(rek_string::AbstractVector{Int8}, ctx)
    prog = ctx.program
    isnothing(prog) || return run_program!(rek_string, prog, ctx.fast, ctx.stack)
    return calc_stack_batch_tensor(collect(rek_string), ctx.callbacks, ctx.nodes, ctx.pool)
end

"""
    thread_contexts(toolbox, x_data; std_return_type=Float64)

One [`buffer_context`](@ref) per thread id, for a custom loss that evaluates chromosomes
inside `runGep`'s threaded fitness loop:

```julia
ctxs = thread_contexts(regressor.toolbox_, x_train)
loss(elem, validate) = ... elem(ctxs[Threads.threadid()]) ...
```

The vector has `thread_slots()` entries, not `nthreads()`: since Julia 1.12,
`Threads.@threads` can hand out ids above `nthreads()`.
"""
function thread_contexts(toolbox, x_data::AbstractArray; std_return_type::Type=Float64)
    return [buffer_context(toolbox, x_data; std_return_type=std_return_type)
            for _ in 1:thread_slots()]
end

"""
    (chromosome::Chromosome)(x_data)

Predict on `x_data` (one row per feature, one column per sample) with the batched
evaluator, in a fresh [`buffer_context`](@ref). Tuned constants or gene coefficients are
applied as by the `ctx` method. Throws an `ArgumentError` when an operator or terminal
has no batched counterpart.
"""
function (chromosome::Chromosome)(x_data::AbstractArray)
    ctx = buffer_context(chromosome.toolbox, x_data)
    isnothing(ctx) && throw(ArgumentError(
        "an operator or terminal of this model has no batched counterpart"))
    return chromosome(ctx)
end

"""
    (chromosome::Chromosome)(ctx::NamedTuple)

Predict on the data the [`buffer_context`](@ref) `ctx` was built for, reusing its buffers;
this is the form for a custom loss (see [`thread_contexts`](@ref)). Uses
`optimised_constants` if set, otherwise `scaling_weights` if set, otherwise the plain
karva string. The result may live in the context's buffers: consume or copy it before the
next evaluation in the same context.
"""
function (chromosome::Chromosome)(ctx::NamedTuple)
    if !isnothing(chromosome.optimised_constants)
        return evaluate_with_constants(chromosome, ctx, chromosome.optimised_constants)
    end
    if isnothing(chromosome.scaling_weights)
        return ctx_eval(chromosome.expression_raw, ctx)
    end
    raw = _karva_raw(chromosome; split=true)
    acc = nothing
    for (j, w) in enumerate(chromosome.scaling_weights)
        # each gene is folded into `acc` before the next one reuses the buffers
        part = ctx_eval(raw[j+1], ctx)
        part isa AbstractVector || return part
        acc = isnothing(acc) ? w .* part : acc .+ w .* part
    end
    return acc
end

"""
    CompiledModel

What `chromosome.compiled_function` returns, for code written when `Chromosome` kept its
compiled expression in that field: the chromosome as a model. It prints as
[`equation_string`](@ref), and `m(x_data)` or `m(x_data, operators)` predicts like
`chromosome(x_data)`, `x_data` holding one row per feature as before; `operators` is not
used. A chromosome that is not compiled prints as `(not compiled)`.
"""
struct CompiledModel
    chromosome::Chromosome
end

function Base.show(io::IO, m::CompiledModel)
    getfield(m.chromosome, :compiled) || return print(io, "(not compiled)")
    print(io, equation_string(m.chromosome))
end

(m::CompiledModel)(x_data, _...) = m.chromosome(x_data)

# a literal field name folds the comparison away, so field access stays a plain getfield
@inline function Base.getproperty(chromosome::Chromosome, name::Symbol)
    name === :compiled_function && return CompiledModel(chromosome)
    return getfield(chromosome, name)
end

Base.propertynames(::Chromosome, private::Bool=false) =
    (fieldnames(Chromosome)..., :compiled_function)

"""
    stringify_key(callback)

Key of a toolbox callback in `FUNCTION_STRINGIFY` / `TENSOR_STRINGIFY`: the name of a
function, or the type name of an operator object (for which `Symbol` would give
`Symbol("AdditionNode()")`).
"""
@inline stringify_key(callback) =
    callback isa Function ? Symbol(callback) : nameof(typeof(callback))

"""
    stringify_callbacks(toolbox)

The string-rendering counterpart of every toolbox callback that has one, keyed by symbol.
"""
function stringify_callbacks(toolbox)
    callbacks = Dict{Int8,Function}()
    for (key, value) in toolbox.callbacks
        sym = stringify_key(value)
        haskey(FUNCTION_STRINGIFY, sym) && (callbacks[key] = FUNCTION_STRINGIFY[sym])
        haskey(TENSOR_STRINGIFY, sym) && (callbacks[key] = TENSOR_STRINGIFY[sym])
    end
    return callbacks
end

"""
    equation_string(chromosome)

The model as an equation string: with tuned constants in place of the drawn ones (which
are kept if no symbol ids are left to hold the tuned values), or, with gene coefficients,
as the weighted sum of its genes, which is what a scaled model computes. Tuned constants
and coefficients are rounded to 6 significant digits.
"""
function equation_string(chromosome::Chromosome)
    if !isnothing(chromosome.optimised_constants)
        tb = chromosome.toolbox
        callbacks = stringify_callbacks(tb)
        leaves = Dict{Int8,Any}(k => v for (k, v) in tb.nodes)
        expr = copy(chromosome.expression_raw)
        # one fresh symbol per occurrence, numbered past every symbol in use, operators
        # included, or an occurrence would be read as the operator that shares its id
        positions = constant_positions(chromosome)
        next = Int(max(maximum(keys(leaves)), maximum(keys(tb.arrity_by_id); init=Int8(0)))) + 1
        next + length(positions) - 1 <= typemax(Int8) ||
            return string(print_karva_strings(chromosome))
        for (k, pos) in enumerate(positions)
            sym = Int8(next + k - 1)
            leaves[sym] = round(chromosome.optimised_constants[k]; sigdigits=6)
            expr[pos] = sym
        end
        return string(compile_djl_datatype(expr, tb.arrity_by_id, callbacks, leaves, 1))
    end
    isnothing(chromosome.scaling_weights) &&
        return string(print_karva_strings(chromosome))
    raw = _karva_raw(chromosome; split=true)
    tb = chromosome.toolbox
    callbacks = stringify_callbacks(tb)
    parts = String[]
    for (j, w) in enumerate(chromosome.scaling_weights)
        body = string(compile_djl_datatype(collect(raw[j+1]), tb.arrity_by_id, callbacks,
            tb.nodes, 1))
        # parenthesise the gene unless its rendering already is
        startswith(body, '(') && endswith(body, ')') || (body = string('(', body, ')'))
        sign = j == 1 ? (w < 0 ? "-" : "") : (w < 0 ? " - " : " + ")
        push!(parts, string(sign, round(abs(w); sigdigits=6), " * ", body))
    end
    return join(parts)
end

function Base.show(io::IO, chromosome::Chromosome)
    print(io, chromosome.compiled ? equation_string(chromosome) :
              "Chromosome(uncompiled, $(length(chromosome.genes)) genes)")
end


"""
    constant_positions(chromosome)

Positions in the karva string that hold a numeric constant. Constants are tuned per
occurrence, so a constant symbol that occurs twice has two positions.
"""
function constant_positions(chromosome::Chromosome)
    tb = chromosome.toolbox
    return [pos for (pos, sym) in enumerate(chromosome.expression_raw)
            if get(tb.nodes, sym, nothing) isa Number]
end

"""
    evaluate_with_constants(chromosome, ctx, constants)

Evaluate the chromosome with its constant occurrences set to `constants`, in the order of
[`constant_positions`](@ref), leaving the shared toolbox untouched. `ctx` needs only
`callbacks` and `nodes`. Used by the constant optimiser and for every model with
`optimised_constants`.

Each occurrence becomes a fresh terminal symbol, outside the alphabet a compiled program
covers, so this runs on the allocating path of the evaluator (no buffer pool). Throws an
`ArgumentError` when no symbol ids are left for the occurrences.
"""
function evaluate_with_constants(chromosome::Chromosome, ctx, constants::AbstractVector)
    return evaluate_karva_constants(chromosome.expression_raw, constant_positions(chromosome),
        constants, ctx.callbacks, ctx.nodes)
end

# a karva string with the symbols at `positions` set to `constants`, on the allocating path
function evaluate_karva_constants(rek::AbstractVector{Int8}, positions::AbstractVector{Int},
    constants::AbstractVector, callbacks, ctx_nodes)
    nodes = copy(ctx_nodes)
    isempty(positions) && return calc_stack_batch_tensor(collect(rek), callbacks, nodes, nothing)

    # one fresh terminal per occurrence, so the occurrences vary independently; numbered
    # past every symbol in use, operators included, as in `equation_string`
    expr = collect(rek)
    next = Int(max(maximum(keys(nodes)), maximum(keys(callbacks); init=Int8(0)))) + 1
    next + length(positions) - 1 <= typemax(Int8) ||
        throw(ArgumentError("no free symbol ids left for the constant occurrences"))
    n = length(first(values(ctx_nodes)))
    for (k, pos) in enumerate(positions)
        sym = Int8(next + k - 1)
        nodes[sym] = fill(Float64(constants[k]), n)
        expr[pos] = sym
    end
    return calc_stack_batch_tensor(expr, callbacks, nodes, nothing)
end

"""
    compile_expression!(chromosome::Chromosome; force_compile::Bool=false)

Resolve the genes into the karva string (`expression_raw`) if the chromosome is not
compiled yet or `force_compile` is set. This clears `scaling_weights` and
`optimised_constants` and resets the fitness to unscored; if the genes cannot be
resolved, the fitness is set to the worst value (`toolbox.fitness_reset[1]`) instead.
"""
@inline function compile_expression!(chromosome::Chromosome; force_compile::Bool=false)
    if !chromosome.compiled || force_compile
        try
            chromosome.expression_raw = _karva_raw(chromosome)
            chromosome.scaling_weights = nothing
            chromosome.optimised_constants = nothing
            chromosome.fitness = chromosome.toolbox.fitness_reset[2]
            chromosome.compiled = true
        catch e
            chromosome.fitness = chromosome.toolbox.fitness_reset[1]
        end
    end
end

"""
    fitness(chromosome::Chromosome)

The chromosome's fitness tuple.
"""
function fitness(chromosome::Chromosome)
    return chromosome.fitness
end


"""
    set_fitness!(chromosome::Chromosome, value::Tuple)

Set the chromosome's fitness tuple.
"""
function set_fitness!(chromosome::Chromosome, value::Tuple)
    chromosome.fitness = value
end

"""
    _karva_raw(chromosome::Chromosome; split::Bool=false)

The chromosome's karva string: the connectors, then each gene's active part, i.e. its
shortest prefix that is a complete prefix expression (found by counting open argument
slots; a gene that never completes is taken whole). Preamble symbols are left out.

Returns a `Vector{Int8}`, or with `split = true` the pieces
`[connectors, gene_1, ..., gene_n]` as views into `chromosome.genes`.

# Example
With `+`, `*` (arity 2) as symbols 1, 2, `x1`, `x2` as 3, 4, `gene_count = 2` and
`head_len = 2`, the genes `[1, 2, 3, 4, 3, 4, 3, 1, 4, 4, 3]` (connector, gene 1,
gene 2) give `[1, 2, 3, 4, 3]`, i.e. `+(*(x1, x2), x1)`.
"""
@inline function _karva_raw(chromosome::Chromosome; split::Bool=false)
    tb = chromosome.toolbox
    gene_len = tb.head_len * 2 + 1
    gene_count = tb.gene_count
    genes = chromosome.genes
    checkbounds(genes, gene_count * gene_len + gene_count - 1)

    lens = Vector{Int}(undef, gene_count)
    total = gene_count - 1
    for g in 1:gene_count
        lens[g] = active_gene_length(genes, gene_count + (g - 1) * gene_len, gene_len,
            tb.arrity_by_id)
        total += lens[g]
    end

    if split
        pieces = Vector{typeof(view(genes, 1:0))}(undef, gene_count + 1)
        pieces[1] = view(genes, 1:gene_count-1)
        for g in 1:gene_count
            start = gene_count + (g - 1) * gene_len
            pieces[g+1] = view(genes, start:start+lens[g]-1)
        end
        return pieces
    end
    out = Vector{Int8}(undef, total)
    copyto!(out, 1, genes, 1, gene_count - 1)
    pos = gene_count
    for g in 1:gene_count
        copyto!(out, pos, genes, gene_count + (g - 1) * gene_len, lens[g])
        pos += lens[g]
    end
    return out
end

"""
    active_gene_length(genes, start, gene_len, arity)

Length of the active part of the gene `genes[start:start+gene_len-1]`: the first `k` at
which the open argument slots, `1 + sum(arity[s] - 1)` over its first `k` symbols, reach
zero, or `gene_len` if they never do. Every symbol is looked up, the inactive ones too, so
an unknown symbol throws a `KeyError` wherever it is.
"""
@inline function active_gene_length(genes::Vector{Int8}, start::Int, gene_len::Int, arity)
    slots = 1
    len = 0
    for k in 0:gene_len-1
        a = Int(arity[genes[start+k]])
        if len == 0
            slots += a - 1
            slots == 0 && (len = k + 1)
        end
    end
    return len == 0 ? gene_len : len
end

"""
    split_karva(chromosome::Chromosome, coeffs::Int=2)

Split the chromosome into `coeffs` independent karva strings, e.g. for template models:
the first `coeffs - 1` connectors are dropped, and each part takes the next
`gene_count ÷ coeffs` genes and the next `gene_count ÷ coeffs - 1` connectors.
"""
@inline function split_karva(chromosome::Chromosome, coeffs::Int=2)
    raw = _karva_raw(chromosome; split=true)
    connectors = popfirst!(raw)[coeffs:end]
    gene_count_per_factor = div(chromosome.toolbox.gene_count, coeffs)
    retval = []
    for _ in 1:coeffs
        temp_cons = splice!(connectors, 1:gene_count_per_factor-1)
        temp_genes = reduce(vcat, splice!(raw, 1:gene_count_per_factor))
        push!(retval, vcat([temp_cons, temp_genes]...))
    end
    return retval
end

"""
    split_positions(chromosome::Chromosome, coeffs::Int=2)

The positions in `expression_raw` of the symbols of each part [`split_karva`](@ref) makes,
in its order: `chromosome.expression_raw[split_positions(chromosome, k)[j]]` is
`split_karva(chromosome, k)[j]`.
"""
function split_positions(chromosome::Chromosome, coeffs::Int=2)
    gene_count = chromosome.toolbox.gene_count
    raw = _karva_raw(chromosome; split=true)
    lens = [length(raw[g+1]) for g in 1:gene_count]
    # the connectors come first, then the active part of every gene
    starts = cumsum(vcat(gene_count, lens[1:end-1]))
    per = div(gene_count, coeffs)
    connectors = collect(coeffs:gene_count-1)
    parts = Vector{Vector{Int}}()
    for j in 1:coeffs
        cons = splice!(connectors, 1:per-1)
        genes = reduce(vcat, [collect(starts[g]:starts[g]+lens[g]-1) for g in (j-1)*per+1:j*per])
        push!(parts, vcat(cons, genes))
    end
    return parts
end

# the tuned constants of each part split_karva makes: the positions in the part and the
# values, or nothing without tuned constants
function split_constants(chromosome::Chromosome, coeffs::Int)
    isnothing(chromosome.optimised_constants) && return nothing
    value_at = Dict(zip(constant_positions(chromosome), chromosome.optimised_constants))
    return [let local_positions = [k for (k, p) in enumerate(pos) if haskey(value_at, p)]
                (local_positions, [value_at[pos[k]] for k in local_positions])
            end for pos in split_positions(chromosome, coeffs)]
end

"""
    split_predict(chromosome::Chromosome, ctx, coeffs::Int=2)

The predictions of a multi-expression model: each of the `coeffs` expressions
[`split_karva`](@ref) splits the chromosome into, evaluated in the [`buffer_context`](@ref)
`ctx` (e.g. this thread's of [`thread_contexts`](@ref)) and copied out of its buffers, so
all of them are valid at once, with the tuned constants of the chromosome
(`optimised_constants`) if it has them. An entry that is not a vector (e.g. `NaN`) marks
an expression that cannot be evaluated; an evaluation that throws, as the logarithm of a
negative number does, throws here too.
"""
function split_predict(chromosome::Chromosome, ctx, coeffs::Int=2)
    parts = split_karva(chromosome, coeffs)
    tuned = split_constants(chromosome, coeffs)
    return [let out = isnothing(tuned) ? ctx_eval(part, ctx) :
                      evaluate_karva_constants(part, tuned[j][1], tuned[j][2], ctx.callbacks,
                          ctx.nodes)
                out isa AbstractVector ? copy(out) : out
            end for (j, part) in enumerate(parts)]
end

"""
    split_equations(chromosome::Chromosome, coeffs::Int=2)

The equation of each of the `coeffs` expressions [`split_karva`](@ref) splits the
chromosome into, as strings, in order, with the tuned constants (`optimised_constants`,
rounded to 6 significant digits) if it has them: how to print a multi-expression model.
(`print_karva_strings(chromosome; split_len=k)` groups the genes differently: it leaves
the last `k - 1` genes on their own.)
"""
function split_equations(chromosome::Chromosome, coeffs::Int=2)
    tb = chromosome.toolbox
    callbacks = stringify_callbacks(tb)
    parts = split_karva(chromosome, coeffs)
    tuned = split_constants(chromosome, coeffs)
    isnothing(tuned) &&
        return [string(compile_djl_datatype(part, tb.arrity_by_id, callbacks, tb.nodes, 1))
                for part in parts]
    out = String[]
    for (j, part) in enumerate(parts)
        leaves = Dict{Int8,Any}(k => v for (k, v) in tb.nodes)
        expr = collect(part)
        positions, values = tuned[j]
        # one fresh symbol per occurrence, as in `equation_string`
        next = Int(max(maximum(keys(leaves)), maximum(keys(tb.arrity_by_id); init=Int8(0)))) + 1
        next + length(positions) - 1 <= typemax(Int8) ||
            throw(ArgumentError("no free symbol ids left for the constant occurrences"))
        for (k, pos) in enumerate(positions)
            sym = Int8(next + k - 1)
            leaves[sym] = round(values[k]; sigdigits=6)
            expr[pos] = sym
        end
        push!(out, string(compile_djl_datatype(expr, tb.arrity_by_id, callbacks, leaves, 1)))
    end
    return out
end

"""
    print_karva_strings(chromosome::Chromosome; split_len::Int=1)

Render the karva string with the stringify callbacks, without tuned constants or gene
coefficients (see [`equation_string`](@ref)). With `split_len = k > 1` the first `k - 1`
connectors are skipped and the partial results are returned, last gene first, instead of
one string.
"""
@inline function print_karva_strings(chromosome::Chromosome; split_len::Int=1)
    callback_ = stringify_callbacks(chromosome.toolbox)

    return compile_djl_datatype(
        chromosome.expression_raw,
        chromosome.toolbox.arrity_by_id,
        callback_,
        chromosome.toolbox.nodes,
        split_len)
end

"""
    retrieve_coefficients(chromo::Chromosome; ignore_indices=nothing)

Map each symbol of `toolbox.constant_indices` that occurs in the karva string to its
position there (the last one, for a symbol that occurs more than once). Requires
`constant_indices` to be set; `ignore_indices` is not used.
"""
@inline function retrieve_coefficients(chromo::Chromosome; ignore_indices::Union{Vector{Int8},Nothing}=nothing)
    const_indices = chromo.toolbox.constant_indices
    return Dict{Int8,Int}(elem => idx for (idx, elem) in enumerate(chromo.expression_raw) if elem in const_indices)
end

"""
    generate_gene(headsyms::Vector{Int8}, tailsyms::Vector{Int8}, headlen::Int,
        tail_weights::Weights, head_weights::Weights; rng::AbstractRNG=STD_RNG)

A random gene: `headlen` head symbols drawn from `headsyms` by `head_weights`, then
`headlen + 1` tail symbols drawn from `tailsyms` by `tail_weights`, with replacement.
"""
@inline function generate_gene(headsyms::Vector{Int8}, tailsyms::Vector{Int8}, headlen::Int,
    tail_weights::Weights, head_weights::Weights; rng::AbstractRNG=STD_RNG)
    head = sample(rng,headsyms, head_weights, headlen)
    tail = sample(rng,tailsyms, tail_weights, headlen + 1)
    return vcat(head, tail)
end



# TODO: adapt to tensor generation
"""
    generate_chromosome(toolbox::Toolbox; rng::AbstractRNG=STD_RNG)

A random, compiled chromosome: `gene_count - 1` connectors drawn uniformly, `gene_count`
genes from [`generate_gene`](@ref), and `gene_count` preamble symbols if the toolbox has
any.
"""
@inline function generate_chromosome(toolbox::Toolbox; rng::AbstractRNG=STD_RNG)
    return Chromosome(generate_genes(toolbox; rng=rng), toolbox, true)
end

"""
    generate_genes(toolbox::Toolbox; rng::AbstractRNG=STD_RNG)

The symbol vector of [`generate_chromosome`](@ref), drawn from `rng` the same way, without
building a chromosome.
"""
@inline function generate_genes(toolbox::Toolbox; rng::AbstractRNG=STD_RNG)
    connectors = rand(rng, toolbox.gene_connections, toolbox.gene_count - 1)
    genes = vcat([generate_gene(toolbox.headsyms, toolbox.tailsyms, toolbox.head_len, toolbox.tail_weights,
        toolbox.head_weights; rng=rng) for _ in 1:toolbox.gene_count]...)
    if !isempty(toolbox.preamble_syms)
        return vcat(connectors, genes, sample(rng, toolbox.preamble_syms, toolbox.gene_count))
    end
    return vcat(connectors, genes)
end



"""
    generate_population(number::Int, toolbox::Toolbox)

`number` random chromosomes, generated in parallel, each with its own RNG split from
`toolbox.master_rng`.
"""
@inline function generate_population(number::Int, toolbox::Toolbox)
    population = Vector{Chromosome}(undef, number)
    subkeys = split_rng(toolbox.master_rng, number)
    Threads.@threads for i in 1:number
        @inbounds population[i] = generate_chromosome(toolbox; rng=subkeys[i])
    end
    return population
end


@inline function create_operator_masks(gene_seq_alpha::Vector{Int8}, gene_seq_beta::Vector{Int8}, pb::Real=0.2; 
    rng::AbstractRNG=STD_RNG)
    alpha_operator = zeros(Int8, length(gene_seq_alpha))
    beta_operator = zeros(Int8, length(gene_seq_beta))
    indices_alpha = rand(rng,1:length(gene_seq_alpha), min(round(Int, (pb * length(gene_seq_alpha))), length(gene_seq_alpha)))
    indices_beta = rand(rng,1:length(gene_seq_beta), min(round(Int, (pb * length(gene_seq_beta))), length(gene_seq_beta)))
    alpha_operator[indices_alpha] .= Int8(1)
    beta_operator[indices_beta] .= Int8(1)
    return alpha_operator, beta_operator
end

@inline function create_operator_point_one_masks(gene_seq_alpha::Vector{Int8}, gene_seq_beta::Vector{Int8}, toolbox::Toolbox; 
    rng::AbstractRNG=STD_RNG)
    alpha_operator = zeros(Int8, length(gene_seq_alpha))
    beta_operator = zeros(Int8, length(gene_seq_beta))
    head_len = toolbox.head_len
    gene_len = head_len * 2 + 1

    for i in toolbox.gen_start_indices
        ref = i
        mid = ref + gene_len ÷ 2

        point1 = rand(rng,ref:mid)
        point2 = rand(rng,(mid+1):(ref+gene_len-1))
        alpha_operator[point1:point2] .= Int8(1)

        point1 = rand(rng,ref:mid)
        point2 = rand(rng,(mid+1):(ref+gene_len-1))
        beta_operator[point1:point2] .= Int8(1)
    end

    return alpha_operator, beta_operator
end


@inline function create_operator_point_two_masks(gene_seq_alpha::Vector{Int8}, gene_seq_beta::Vector{Int8}, toolbox::Toolbox; 
    rng::AbstractRNG=STD_RNG)
    alpha_operator = zeros(Int8, length(gene_seq_alpha))
    beta_operator = zeros(Int8, length(gene_seq_beta))
    head_len = toolbox.head_len
    gene_len = head_len * 2 + 1

    for i in toolbox.gen_start_indices
        start = i
        quarter = start + gene_len ÷ 4
        half = start + gene_len ÷ 2
        end_gene = start + gene_len - 1


        point1 = rand(rng,start:quarter)
        point2 = rand(rng,quarter+1:half)
        point3 = rand(rng,half+1:end_gene)
        alpha_operator[point1:point2] .= Int8(1)
        alpha_operator[point3:end_gene] .= Int8(1)


        point1 = rand(rng,start:end_gene)
        point2 = rand(rng,point1:end_gene)
        beta_operator[point1:point2] .= Int8(1)
        beta_operator[point2+1:end_gene] .= Int8(1)
    end

    return alpha_operator, beta_operator
end

"""
    replicate(chromosome1::Chromosome, chromosome2::Chromosome, toolbox)

Uncompiled, unscored copies of the two chromosomes (`genes` copied), built with
`toolbox`.
"""
@inline function replicate(chromosome1::Chromosome, chromosome2::Chromosome, toolbox)
    return [Chromosome(copy(chromosome1.genes), toolbox), Chromosome(copy(chromosome2.genes), toolbox)]
end


"""
    gene_dominant_fusion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2;
        rng::AbstractRNG=STD_RNG)

Each chromosome, independently, draws `round(Int, pb * length(genes))` positions with
replacement (connectors and preamble included) and takes there the larger of the two
parents' symbol ids.
"""
@inline function gene_dominant_fusion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2; rng::AbstractRNG=STD_RNG)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb; rng=rng)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i]
        child_2_genes[i] = beta_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

"""
    gen_rezessiv!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2;
        rng::AbstractRNG=STD_RNG)

As [`gene_dominant_fusion!`](@ref), taking the smaller symbol id.
"""
@inline function gen_rezessiv!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2; rng::AbstractRNG=STD_RNG)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb; rng=rng)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i]
        child_2_genes[i] = beta_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

"""
    gene_fussion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2;
        rng::AbstractRNG=STD_RNG)

As [`gene_dominant_fusion!`](@ref), taking the mean `(a + b) ÷ 2` of the two symbol ids in
`Int8` arithmetic. The result need not be valid at its position: at a connector position
it can be another operator, and ids summing past 127 wrap around.
"""
@inline function gene_fussion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2; rng::AbstractRNG=STD_RNG)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb; rng=rng)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? Int8((gene_seq_alpha[i] + gene_seq_beta[i]) ÷ 2) : gene_seq_alpha[i]
        child_2_genes[i] = beta_operator[i] == 1 ? Int8((gene_seq_alpha[i] + gene_seq_beta[i]) ÷ 2) : gene_seq_beta[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

"""
    gene_one_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome;
        rng::AbstractRNG=STD_RNG)

Segment crossover: in every gene, each chromosome keeps its own symbols on one random
segment, from a position among the gene's first `head_len + 1` symbols to one among its
last `head_len`, and takes the other parent's symbols everywhere else, connectors and
preamble included. The two chromosomes draw their segments independently.
"""
@inline function gene_one_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome; rng::AbstractRNG=STD_RNG)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_point_one_masks(gene_seq_alpha, gene_seq_beta, chromosome1.toolbox; rng=rng)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? gene_seq_alpha[i] : gene_seq_beta[i]
        child_2_genes[i] = beta_operator[i] == 1 ? gene_seq_beta[i] : gene_seq_alpha[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

"""
    gene_two_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome;
        rng::AbstractRNG=STD_RNG)

Segment crossover as [`gene_one_point_cross_over!`](@ref), with other segments per gene:
chromosome 1 keeps `[p1, p2]` and `[p3, end]`, with `p1` in the gene's first quarter,
`p2` in its second and `p3` in its second half (by integer division of the gene length);
chromosome 2 keeps `[p, end]` for a random position `p`. Everything else, connectors and
preamble included, comes from the other parent.
"""
@inline function gene_two_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome; rng::AbstractRNG=STD_RNG)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_point_two_masks(gene_seq_alpha, gene_seq_beta, chromosome1.toolbox; rng=rng)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? gene_seq_alpha[i] : gene_seq_beta[i]
        child_2_genes[i] = beta_operator[i] == 1 ? gene_seq_beta[i] : gene_seq_alpha[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

"""
    gene_mutation!(chromosome1::Chromosome, pb::Real=0.25; rng::AbstractRNG=STD_RNG)

Point mutation: `round(Int, pb * length(genes))` positions drawn with replacement
(connectors and preamble included) take the symbol at the same position of a freshly
generated chromosome, so each position keeps a symbol valid for it.
"""
@inline function gene_mutation!(chromosome1::Chromosome, pb::Real=0.25; rng::AbstractRNG=STD_RNG)
    gene_seq_alpha = chromosome1.genes
    alpha_operator, _ = create_operator_masks(gene_seq_alpha, gene_seq_alpha, pb; rng=rng)
    mutation_seq_1 = generate_genes(chromosome1.toolbox; rng=rng)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        gene_seq_alpha[i] = alpha_operator[i] == 1 ? mutation_seq_1[i] : gene_seq_alpha[i]
    end
end

"""
    gene_inversion!(chromosome1::Chromosome; rng::AbstractRNG=STD_RNG)

Reverse `genes[start:head_len]`, `start` being the first position of a random gene. For
the first gene that is positions `gene_count:head_len`, its whole head only when
`gene_count == 1` and nothing when `gene_count > head_len`; for any other gene the range
is empty.
"""
@inline function gene_inversion!(chromosome1::Chromosome; rng::AbstractRNG=STD_RNG)
    start_1 = rand(rng,chromosome1.toolbox.gen_start_indices)
    reverse!(@view chromosome1.genes[start_1:chromosome1.toolbox.head_len])
end

"""
    gene_insertion!(chromosome::Chromosome; rng::AbstractRNG=STD_RNG)

Overwrite one random head position of a random gene with a tail symbol drawn uniformly;
nothing is shifted.
"""
@inline function gene_insertion!(chromosome::Chromosome; rng::AbstractRNG=STD_RNG)
    start_1 = rand(rng,chromosome.toolbox.gen_start_indices)
    insert_pos = rand(rng,start_1:(start_1+chromosome.toolbox.head_len-1))
    insert_sym = rand(rng,chromosome.toolbox.tailsyms)
    chromosome.genes[insert_pos] = insert_sym
end

"""
    root_insertion!(chromosome::Chromosome; rng::AbstractRNG=STD_RNG)

Rotate the head of a random gene cyclically by a random shift in `1:head_len-1`, which
changes the gene's root. Throws for `head_len == 1`.
"""
@inline function root_insertion!(chromosome::Chromosome; rng::AbstractRNG=STD_RNG)
    start_1 = rand(rng,chromosome.toolbox.gen_start_indices)
    rolled_array = circshift(chromosome.genes[start_1:start_1+chromosome.toolbox.head_len-1], rand(rng,1:chromosome.toolbox.head_len-1))
    chromosome.genes[start_1:start_1+chromosome.toolbox.head_len-1] = rolled_array
end

"""
    reverse_insertion_tail!(chromosome::Chromosome; rng::AbstractRNG=STD_RNG)

Rotate the tail of a random gene, without its first symbol, cyclically by a random shift
in `1:head_len-1`. Throws for `head_len == 1`.
"""
@inline function reverse_insertion_tail!(chromosome::Chromosome; rng::AbstractRNG=STD_RNG)
    start_1 = rand(rng,chromosome.toolbox.gen_start_indices) + chromosome.toolbox.head_len + 1
    rolled_array = circshift(chromosome.genes[start_1:start_1+chromosome.toolbox.head_len-1], rand(rng,1:chromosome.toolbox.head_len-1))
    chromosome.genes[start_1:start_1+chromosome.toolbox.head_len-1] = rolled_array
end


"""
    gene_transposition!(chromosome::Chromosome, len::Int=5; rng::AbstractRNG=STD_RNG)

Swap, element by element, two segments of `min(len, head_len + 1)` symbols, each within
the tail of a random gene (the two genes may coincide, and the segments may then overlap).
"""
@inline function gene_transposition!(chromosome::Chromosome, len::Int=5; rng::AbstractRNG=STD_RNG)
    toolbox = chromosome.toolbox
    head_len = toolbox.head_len
    gene_len = head_len * 2 + 1
    gen_start_indices = toolbox.gen_start_indices

    source_start = rand(rng,gen_start_indices)
    target_start = rand(rng,gen_start_indices)

    # a segment must fit in a tail, which has gene_len - head_len symbols
    segment_len = min(len, gene_len - head_len)

    source_pos = rand(rng,source_start+head_len:(source_start + gene_len-segment_len))
    target_pos = rand(rng,target_start+head_len:(target_start + gene_len-segment_len))

    for i in 0:(segment_len - 1)
        chromosome.genes[source_pos + i], chromosome.genes[target_pos + i] = chromosome.genes[target_pos + i], chromosome.genes[source_pos + i]
    end
end


"""
    gene_averaging!(chromosome::Chromosome, elites::AbstractVector{Chromosome}, rate::Real;
        top_k::Int=3, rng::AbstractRNG=STD_RNG)
    gene_averaging!(chromosome::Chromosome, consensus::ConsensusSampler, rate::Real;
        rng::AbstractRNG=STD_RNG)

Pull the chromosome toward the elite as a group: draw a consensus from `elites` -- per
position, one of the `top_k` most frequent elite symbols, sampled by frequency
(`one_hot_mean`) -- and replace each position by the consensus symbol with probability
`rate`. A consensus symbol occurs at its position in some elite, so it is valid there.
The elites are only read; with none, nothing changes. The second form takes the elites'
frequencies precomputed, `ConsensusSampler([e.genes for e in elites], top_k)`, for many
offspring of the same elites; both forms make the same draws.
"""
@inline function gene_averaging!(chromosome::Chromosome, elites::AbstractVector{Chromosome},
    rate::Real; top_k::Int=3, rng::AbstractRNG=STD_RNG)
    isempty(elites) && return
    gene_averaging!(chromosome, ConsensusSampler([e.genes for e in elites], top_k), rate;
        rng=rng)
end

@inline function gene_averaging!(chromosome::Chromosome, consensus::ConsensusSampler,
    rate::Real; rng::AbstractRNG=STD_RNG)
    isempty(consensus) && return
    draw = consensus_draw(consensus; rng=rng)
    limit = min(length(chromosome.genes), length(draw))
    @inbounds for i in 1:limit
        if rand(rng) < rate
            chromosome.genes[i] = draw[i]
        end
    end
end


"""
    genetic_operations!(space_next::Vector{Chromosome}, i::Int, toolbox::Toolbox;
        generation::Int64, max_generation::Int64, parents::Vector{Chromosome},
        elites=nothing, rng::AbstractRNG=STD_RNG)

Replace `space_next[i]` and `space_next[i+1]` by uncompiled copies ([`replicate`](@ref))
and vary them. In this order, each operator fires with the probability under its key in
`toolbox.gep_probs`, on the pair or on each chromosome with its own draw:

- `one_point_cross_over_prob`: `gene_one_point_cross_over!` on the pair
- `two_point_cross_over_prob`: `gene_two_point_cross_over!` on the pair
- `mutation_prob`: `gene_mutation!` on each, rate `mutation_rate`
- `dominant_fusion_prob`: `gene_dominant_fusion!` on the pair, rate `dominant_fusion_rate`
- `rezessiv_fusion_prob`: `gen_rezessiv!` on the pair, rate `rezessiv_fusion_rate`
- `fusion_prob`: `gene_fussion!` on the pair, rate `fusion_rate`
- `inversion_prob`: `gene_inversion!` on each
- `insertion_prob`: `gene_insertion!` on each
- `root_insertion_prob`: `root_insertion!` on each
- `reverse_insertion_tail`: `reverse_insertion_tail!` on each
- `gene_transposition_prob`: `gene_transposition!` on each
- `gene_averaging_prob`: `gene_averaging!` on each, rate `gene_averaging_rate`, when
  `elites` (the elite chromosomes, or their `ConsensusSampler` for `top_k = 3`) is
  non-empty

The two `gene_averaging` keys default to 0; every other key must be present.
`generation`, `max_generation` and `parents` are not used.
"""
@inline function genetic_operations!(space_next::Vector{Chromosome}, i::Int, toolbox::Toolbox; 
    generation::Int64, max_generation::Int64, parents::Vector{Chromosome},
    elites::Union{AbstractVector{Chromosome},ConsensusSampler,Nothing}=nothing,
    rng::AbstractRNG=STD_RNG)
    # vary copies: the parents are still members of the population
    space_next[i:i+1] = replicate(space_next[i], space_next[i+1], toolbox)
    rand_space = rand(rng,20)


    if rand_space[1] < toolbox.gep_probs["one_point_cross_over_prob"]
        gene_one_point_cross_over!(space_next[i], space_next[i+1], rng=rng)
    end

    if rand_space[2] < toolbox.gep_probs["two_point_cross_over_prob"]
        gene_two_point_cross_over!(space_next[i], space_next[i+1], rng=rng)
    end

    if rand_space[3] < toolbox.gep_probs["mutation_prob"]
        gene_mutation!(space_next[i], toolbox.gep_probs["mutation_rate"];rng=rng)
    end

    if rand_space[4] < toolbox.gep_probs["mutation_prob"]
        gene_mutation!(space_next[i+1], toolbox.gep_probs["mutation_rate"]; rng=rng)
    end

    if rand_space[5] < toolbox.gep_probs["dominant_fusion_prob"]
        gene_dominant_fusion!(space_next[i], space_next[i+1], toolbox.gep_probs["dominant_fusion_rate"]; rng=rng)
    end

    if rand_space[6] < toolbox.gep_probs["rezessiv_fusion_prob"]
        gen_rezessiv!(space_next[i], space_next[i+1], toolbox.gep_probs["rezessiv_fusion_rate"];rng=rng)
    end

    if rand_space[7] < toolbox.gep_probs["fusion_prob"]
        gene_fussion!(space_next[i], space_next[i+1], toolbox.gep_probs["fusion_rate"];rng=rng)
    end

    if rand_space[8] < toolbox.gep_probs["inversion_prob"]
        gene_inversion!(space_next[i];rng=rng)
    end

    if rand_space[9] < toolbox.gep_probs["inversion_prob"]
        gene_inversion!(space_next[i+1];rng=rng)
    end

    if rand_space[10] < toolbox.gep_probs["insertion_prob"]
        gene_insertion!(space_next[i];rng=rng)
    end

    if rand_space[11] < toolbox.gep_probs["insertion_prob"]
        gene_insertion!(space_next[i+1];rng=rng)
    end

    if rand_space[12] < toolbox.gep_probs["root_insertion_prob"]
        root_insertion!(space_next[i];rng=rng)
    end

    if rand_space[13] < toolbox.gep_probs["root_insertion_prob"]
        root_insertion!(space_next[i+1];rng=rng)
    end

    if rand_space[14] < toolbox.gep_probs["reverse_insertion_tail"]
        reverse_insertion_tail!(space_next[i];rng=rng)
    end

    if rand_space[15] < toolbox.gep_probs["reverse_insertion_tail"]
        reverse_insertion_tail!(space_next[i+1];rng=rng)
    end

    if rand_space[16] < toolbox.gep_probs["gene_transposition_prob"]
        gene_transposition!(space_next[i];rng=rng)
    end

    if rand_space[18] < toolbox.gep_probs["gene_transposition_prob"]
        gene_transposition!(space_next[i+1];rng=rng)
    end

    if !isnothing(elites) && !isempty(elites)
        if rand_space[17] < get(toolbox.gep_probs, "gene_averaging_prob", 0.0)
            gene_averaging!(space_next[i], elites, get(toolbox.gep_probs, "gene_averaging_rate", 0.0); rng=rng)
        end
        if rand_space[19] < get(toolbox.gep_probs, "gene_averaging_prob", 0.0)
            gene_averaging!(space_next[i+1], elites, get(toolbox.gep_probs, "gene_averaging_rate", 0.0); rng=rng)
        end
    end

end

"""
    prob_equation_djl(chromosome::Chromosome, coeff_count::Int,
        prob_data_set::AbstractArray)

A `coeff_count × 1` feature column for [`equation_characterization_default`](@ref): the
mean prediction on `prob_data_set` in the last row, zeros above it; `Inf` marks a
chromosome that is uncompiled or cannot be evaluated.
"""
@inline function prob_equation_djl(chromosome::Chromosome, coeff_count::Int, prob_data_set::AbstractArray)
    ret_val = zeros(coeff_count, 1)
    if !chromosome.compiled
        ret_val[:, 1] .= Inf
    else
        try
            pred = chromosome(prob_data_set)
            ret_val[coeff_count, 1] = pred isa AbstractVector ? mean(pred) : Inf
        catch e
            ret_val[:, 1] .= Inf
        end
    end
    return ret_val
end


"""
    equation_characterization_default(population::Vector{Chromosome}, n_samples::Int,
        prob_dataset::AbstractArray)

Indices of `n_samples` chromosomes chosen by Latin hypercube sampling
(`select_n_samples_lhs`) over their [`prob_equation_djl`](@ref) features on
`prob_dataset`, with one row per preamble symbol (one row without any); chromosomes with
non-finite features are not chosen.
"""
@inline function equation_characterization_default(population::Vector{Chromosome}, n_samples::Int, prob_dataset::AbstractArray)
    len_extented_pop = length(population)
    coeff_count = isempty(population[1].toolbox.preamble_syms) ? 1 : length(population[1].toolbox.preamble_syms)
    features = zeros(coeff_count, len_extented_pop)
    Threads.@threads for p_index in eachindex(population)
        try
            features[:, p_index] .= prob_equation_djl(population[p_index], coeff_count, prob_dataset)
        catch
            features[:, p_index] .= Inf
        end
    end
    indices = select_n_samples_lhs(features, n_samples)
    return indices
end

end
