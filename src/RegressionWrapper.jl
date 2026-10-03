"""
    RegressionWrapper

The user-facing regressors, built on the GEP core:

- `GepRegressor` evolves scalar models. `fit!` trains it on data arrays (one row per
  feature, one column per sample) or against a custom loss over chromosomes, e.g. with
  several objectives; `regressor(x)` predicts with the best model.
- `GepTensorRegressor` evolves models over scalars, vectors and higher-order tensors,
  scored by a custom loss through `predictT` after `allocate_buffers!`.

Given `considered_dimensions`, a regressor builds the library of dimensionally consistent
subexpressions. `fit!` with a `target_dimension` then seeds part of the initial population
from the library and repairs individuals by semantic backpropagation (SBP), and only
homogeneous individuals are scored.

The module also holds the unit rules of the function library
(`FUNCTION_LIB_FORWARD_COMMON`, `FUNCTION_LIB_BACKWARD_COMMON`), the genetic operator
defaults (`GENE_COMMON_PROBS`) and the functions that list or change the library
(`list_all_functions`, `update_function!`, ...).
"""
module RegressionWrapper


export GepRegressor, GepTensorRegressor
export allocate_buffers!, predictT, predictT_scaled, predictT_scaled!, gene_bases
export build_buffers, BUFFERED_EVAL_MIN_SAMPLES
export create_function_entries, create_feature_entries, create_constants_entries, create_physical_operations
export GENE_COMMON_PROBS, FUNCTION_LIB_BACKWARD_COMMON, FUNCTION_LIB_FORWARD_COMMON
export fit!

export list_all_functions, list_all_arity, list_all_forward_handlers,
    list_all_backward_handlers, list_all_genetic_params,
    set_function!, set_arity!, set_forward_handler!, set_backward_handler!,
    update_function!


using ..GepEntities
using ..LossFunction
using ..EvoSelection

using ..GepRegression
using ..SBPUtils
using ..GepUtils
using ..TensorRegUtils
using ..GepSurrogate
using ..GepSimplex
using OrderedCollections
using LinearAlgebra
using StatsBase
using Distributions
using Tensors
using Random123
using Random
# Nelder-Mead for the constant optimiser in `fit!`
using Optim

"""
    FUNCTION_LIB_FORWARD_COMMON::Dict{Symbol,Function}

Forward unit rule of each function in `FUNCTION_LIB_COMMON`: the dimension of the result
from the dimensions of the operands. A `GepRegressor` built with `considered_dimensions`
takes the rules of its functions from here; change one with `set_forward_handler!`.

- `equal_unit_forward`: operands and result share one dimension (`+`, `-`, `min`, `max`)
- `mul_unit_forward`, `div_unit_forward`: the exponents add or subtract (`*`, `/`)
- `zero_unit_forward`: dimensionless operands and result (`exp`, the logarithms, the
  trigonometric and hyperbolic functions, `^`, and `floor`, `ceil`, `round`, whose result
  on a quantity with units would depend on the units it is expressed in)
- `arbitrary_unit_forward`: the result keeps the operand's dimension (`abs`)
- `sign_unit_forward`: an operand of any dimension, a dimensionless result (`sign`)
- `sqr_unit_forward` doubles the exponents (`sqr`); `sqrt` halves them with its inverse,
  `sqr_unit_backward`

A rule must hold for the function itself: if the operands' units are rescaled, the result
must rescale as its dimension says (`abs(λx) = λ abs(x)`, but `floor(λx) ≠ λ floor(x)`),
or a model the check accepts is not dimensionally homogeneous.
"""
const FUNCTION_LIB_FORWARD_COMMON = Dict{Symbol,Function}(
    :+ => equal_unit_forward,
    :- => equal_unit_forward,
    :* => mul_unit_forward,
    :/ => div_unit_forward,
    :min => equal_unit_forward,
    :max => equal_unit_forward, :abs => arbitrary_unit_forward,
    :floor => zero_unit_forward,
    :ceil => zero_unit_forward,
    :round => zero_unit_forward, :exp => zero_unit_forward,
    :log => zero_unit_forward,
    :log10 => zero_unit_forward,
    :log2 => zero_unit_forward, :sin => zero_unit_forward,
    :cos => zero_unit_forward,
    :tan => zero_unit_forward,
    :asin => zero_unit_forward,
    :acos => zero_unit_forward,
    :atan => zero_unit_forward, :sinh => zero_unit_forward,
    :cosh => zero_unit_forward,
    :tanh => zero_unit_forward,
    :asinh => zero_unit_forward,
    :acosh => zero_unit_forward,
    :sqr => sqr_unit_forward,
    :atanh => zero_unit_forward, :sqrt => sqr_unit_backward, :sign => sign_unit_forward,
    # the unit of x^y depends on the value of y, so both operands must be dimensionless;
    # `sqr` squares with units
    :^ => zero_unit_forward
)

"""
    FUNCTION_LIB_BACKWARD_COMMON::Dict{Symbol,Function}

Backward unit rule of each function in `FUNCTION_LIB_COMMON`: the operand dimensions that
give a required result dimension. Semantic backpropagation (SBP) applies them to unary
functions and to binary ones such as `^`; `+`, `-`, `*`, `/`, `min` and `max` it inverts
from their forward rules. The rules mirror `FUNCTION_LIB_FORWARD_COMMON`: `sqrt` doubles
with `sqr_unit_forward`, `abs` keeps its operand's dimension with the identity
`arbitrary_unit_forward`, and `sign` asks for a dimensionless operand (any would do, a
dimensionless one is the one it names). Change a rule with `set_backward_handler!`.
"""
const FUNCTION_LIB_BACKWARD_COMMON = Dict{Symbol,Function}(
    :+ => equal_unit_backward,
    :- => equal_unit_backward,
    :* => mul_unit_backward,
    :/ => div_unit_backward,
    :min => equal_unit_backward,
    :max => equal_unit_backward, :abs => arbitrary_unit_forward,
    :floor => zero_unit_backward,
    :ceil => zero_unit_backward,
    :round => zero_unit_backward, :exp => zero_unit_backward,
    :log => zero_unit_backward,
    :log10 => zero_unit_backward,
    :log2 => zero_unit_backward, :sin => zero_unit_backward,
    :cos => zero_unit_backward,
    :tan => zero_unit_backward,
    :asin => zero_unit_backward,
    :acos => zero_unit_backward,
    :atan => zero_unit_backward, :sinh => zero_unit_backward,
    :cosh => zero_unit_backward,
    :tanh => zero_unit_backward,
    :asinh => zero_unit_backward,
    :acosh => zero_unit_backward,
    :sqr => sqr_unit_backward,
    :atanh => zero_unit_backward, :sqrt => sqr_unit_forward, :sign => sign_unit_backward,
    :^ => zero_unit_backward
)



"""
    GENE_COMMON_PROBS::Dict{String,AbstractFloat}

Rates of the genetic operators, with their defaults. Every toolbox holds this dictionary
itself, not a copy, so an edit applies to existing regressors too. Probabilities are per
offspring, except for crossover and fusion, which are per pair of parents.

- `one_point_cross_over_prob` (0.5), `two_point_cross_over_prob` (0.4): crossover
- `mutation_prob` (1.0): mutation; `mutation_rate` (0.15): share of positions redrawn
- `inversion_prob` (0.1): head inversion
- `insertion_prob` (0.1): a terminal written into a head position
- `root_insertion_prob` (0.1): rotation of a gene's head
- `reverse_insertion_tail` (0.0): rotation within a gene's tail
- `gene_transposition_prob` (0.1): exchange of two tail segments
- `gene_averaging_prob` (1.0): exchange with a consensus of the elite;
  `gene_averaging_rate` (0.3): probability per position; `gene_averaging_elite_frac`
  (0.3): elite size as a fraction of the mating pool, at least 3
- `dominant_fusion_prob`, `rezessiv_fusion_prob`, `fusion_prob` (0.0 each), with the
  rates `dominant_fusion_rate` (0.1), `rezessiv_fusion_rate` (0.1) and `fusion_rate`
  (0.0): fusion operators
- `mating_size` (0.7): offspring per epoch, as a fraction of the population size
"""
const GENE_COMMON_PROBS = Dict{String,AbstractFloat}(
    "one_point_cross_over_prob" => 0.5,
    "two_point_cross_over_prob" => 0.4,
    "mutation_prob" => 1.0,
    "mutation_rate" => 0.15,
    "dominant_fusion_prob" => 0.0,
    "dominant_fusion_rate" => 0.1,
    "rezessiv_fusion_prob" => 0.0,
    "rezessiv_fusion_rate" => 0.1,
    "fusion_prob" => 0.0,
    "fusion_rate" => 0.0,
    "inversion_prob" => 0.1,
    "insertion_prob" => 0.1,
    "root_insertion_prob" => 0.1,
    "reverse_insertion_tail" => 0.0,
    "gene_transposition_prob" => 0.1,
    "gene_averaging_prob" => 1.0,
    # rate and elite fraction: the best setting of the sweep in
    # benchmark/nguyen_averaging_results.txt
    "gene_averaging_rate" => 0.3,
    "gene_averaging_elite_frac" => 0.3,
    "mating_size" => 0.7)

const SymbolDict = OrderedDict{Int8,Int8}
const CallbackDict = Dict{Int8,Function}
const OrderedCallBackDict = OrderedDict{Int8,Function}
const NodeDict = OrderedDict{Int8,Any}
const DimensionDict = OrderedDict{Int8,Vector{Float16}}


"""
    make_lib_seeder(toolbox, token_dto, target_dimension, lib_seed_amount; gene_wise=false)

Population seeder for `runGep`: writes expressions sampled from the library into the genes
of the first `lib_seed_amount` fraction of a new population, so that part of it is
homogeneous from the start. If the toolbox has `+` or `-` connectors, the seeds are joined
by those only and every gene takes the target dimension; otherwise the first gene takes it
and the others are dimensionless, unless `gene_wise` asks for the target in every gene (see
`make_dimension_contract`). An individual keeps its random genes when the library has no
expression for one of them.

Returns `nothing` (no seeding) without a target, without a library (`token_dto`) or when
`lib_seed_amount <= 0`.
"""
function make_lib_seeder(toolbox, token_dto, target_dimension, lib_seed_amount;
    gene_wise::Bool=false)
    (isnothing(target_dimension) || isnothing(token_dto) || lib_seed_amount <= 0) && return nothing
    return function (population::Vector{Chromosome})
        gene_len = toolbox.head_len * 2 + 1
        rng = toolbox.master_rng
        zdim = zeros(Float16, length(target_dimension))

        # the individual must be homogeneous as a whole: joined by equal-unit connectors
        # (+, -), every gene takes the target dimension; joined by * or /, the first gene
        # takes it and the dimensionless rest leave it unchanged -- unless every gene must
        # take it
        fwd = token_dto.tokenLib.physical_operation_dict[]
        eq_conns = [c for c in toolbox.gene_connections
                    if get(fwd, c, nothing) === equal_unit_forward]
        conns = isempty(eq_conns) ? toolbox.gene_connections : eq_conns
        isempty(conns) && return
        gene_dims = isempty(eq_conns) && !gene_wise ?
                    vcat([target_dimension], fill(zdim, length(toolbox.gen_start_indices) - 1)) :
                    fill(target_dimension, length(toolbox.gen_start_indices))

        # only exact dimensions are seeded, since a near miss cannot pass the homogeneity
        # gate; an individual with a gene the library cannot fill stays as drawn
        n_seed = min(length(population), Int(floor(length(population) * lib_seed_amount)))
        @inbounds for i in 1:n_seed
            c = population[i]
            exprs = Vector{Vector{Int8}}(undef, length(gene_dims))
            complete = true
            for (g, dim) in enumerate(gene_dims)
                # operators stay within the head, so the seeded gene is an ordinary gene
                # for the genetic operators
                expr = sample_lib_expression(convert(Vector{Float16}, dim), token_dto;
                    max_len=gene_len, head_len=toolbox.head_len, exact_only=true, rng=rng)
                expr === nothing && (complete = false; break)
                exprs[g] = expr
            end
            complete || continue
            for k in 1:length(toolbox.gen_start_indices)-1
                c.genes[k] = rand(rng, conns)
            end
            for (g, start) in enumerate(toolbox.gen_start_indices)
                c.genes[start:start+length(exprs[g])-1] = exprs[g]
            end
            compile_expression!(c; force_compile=true)
        end
    end
end

"""
    sbp_constants(const_syms, nodes)

Indices of the constant terminals the library is built from: those in `const_syms` (a
vector of indices or a dictionary keyed by them) whose value is not zero. A zero never
serves a repair: it is neutral under `+` and `-`, annihilates a product and makes a
quotient infinite. Evolution can still use it.
"""
sbp_constants(const_syms, nodes) =
    Int8[idx for idx in (const_syms isa AbstractDict ? keys(const_syms) : const_syms)
         if !(nodes[idx] isa Number && iszero(nodes[idx]))]

"""
    make_dimension_contract(toolbox, token_dto, target_dimension, cycles; gene_wise=false)

The `(correction_callback, homogeneity_check)` pair with which `runGep` holds a population
to `target_dimension`, or `(nothing, nothing)` without a target:

- `correction_callback(genes, start_indices, expression, generation)` repairs an
  individual in place with `correct_genes!` (`cycles` attempts), keeping its genes within
  their length with operators in the head and its connectors among the toolbox's
  `gene_connections`; `generation` is not used.
- `homogeneity_check(expression)` is the read-only `is_dimensionally_homogeneous`, which
  flags individuals that are homogeneous without repair, and which `runGep` also applies
  to every repair before flagging it.

With `gene_wise`, for a model scored as a least-squares combination of its genes (linear
scaling), every gene is held to the target instead of the expression as a whole: the
coefficients of such a model are dimensionless only then, whatever the connectors. The
check then asks every gene to have the target dimension (`is_gene_wise_homogeneous`) and,
if the toolbox has `+` or `-` connectors, the whole expression too; the repair is
`correct_genes!` with `gene_wise`, which meets both.

Throws an `ArgumentError` for a target without a library (`token_dto === nothing`, i.e.
the regressor was built without `considered_dimensions`).
"""
function make_dimension_contract(toolbox, token_dto, target_dimension, cycles::Int;
    gene_wise::Bool=false)
    isnothing(target_dimension) && return nothing, nothing
    isnothing(token_dto) && throw(ArgumentError(
        "a target_dimension needs the features' dimensions: construct the regressor with " *
        "considered_dimensions"))
    target = convert(Vector{Float16}, target_dimension)
    gene_len = 2 * toolbox.head_len + 1
    correction = (genes, start_indices, expression, generation) -> correct_genes!(
        genes, start_indices, expression, target, token_dto;
        cycles=cycles, gene_len=gene_len, head_len=toolbox.head_len,
        connectors=toolbox.gene_connections, gene_wise=gene_wise)
    gene_wise || return correction,
        expression -> is_dimensionally_homogeneous(expression, target, token_dto)
    # with equal-unit connectors, genes of the target dimension join into it as a whole
    fwd = token_dto.tokenLib.physical_operation_dict[]
    joined = any(c -> get(fwd, c, nothing) === equal_unit_forward, toolbox.gene_connections)
    check = expression ->
        is_gene_wise_homogeneous(expression, target, token_dto, toolbox.gene_count) &&
        (!joined || is_dimensionally_homogeneous(expression, target, token_dto))
    return correction, check
end

function create_physical_operations(entered_non_terminals::Vector{Symbol},
    idx_functions::Vector{Int8}; 
    function_forward::Dict{Symbol,Function}=FUNCTION_LIB_FORWARD_COMMON,
    function_backward::Dict{Symbol,Function}=FUNCTION_LIB_BACKWARD_COMMON,
    function_lib_common::Dict{Symbol,T1}=FUNCTION_LIB_COMMON) where {T1<:Union{Function,Type}}
    forward_funs = OrderedCallBackDict()
    backward_funs = CallbackDict()
    point_ops = Int8[]

    for (idx, elem) in zip(idx_functions, entered_non_terminals)
        if !haskey(function_lib_common, elem)
            @debug "Symbol: " elem " is ignored"
            continue
        end
        forward_funs[idx] = function_forward[elem]
        backward_funs[idx] = function_backward[elem]
        if elem == :* || elem == :/
            push!(point_ops, idx)
        end

    end

    return forward_funs, backward_funs, point_ops
end

function create_function_entries(
    entered_non_terminals::Vector{Symbol},
    gene_connections_raw::Vector{Symbol},
    start_idx::Int8=Int8(1)
)::Tuple{SymbolDict,CallbackDict,Vector{Function},Vector{Function},Vector{Int8},Int8}

    utilized_symbols = SymbolDict()
    callbacks = CallbackDict()
    binary_operations = Function[]
    unary_operations = Function[]
    gene_connections = Int8[]
    cur_idx = start_idx

    for (idx, elem) in enumerate(entered_non_terminals)
        if !haskey(FUNCTION_LIB_COMMON, elem)
            @info "Symbol: " elem " is ignored"
            continue
        end

        utilized_symbols[idx] = ARITY_LIB_COMMON[elem]
        callbacks[idx] = FUNCTION_LIB_COMMON[elem]

        if ARITY_LIB_COMMON[elem] == 2
            push!(binary_operations, FUNCTION_LIB_COMMON[elem])
        elseif ARITY_LIB_COMMON[elem] == 1
            push!(unary_operations, FUNCTION_LIB_COMMON[elem])
        end

        elem in gene_connections_raw && push!(gene_connections, idx)
        cur_idx += 1
    end

    return utilized_symbols, callbacks, binary_operations, unary_operations, gene_connections, cur_idx
end


function create_feature_entries(
    entered_terminals_features::Vector{Symbol},
    dimensions_to_consider::Dict{Symbol,Vector{Float16}},
    node_type::Type,
    start_idx::Int8
)::Tuple{SymbolDict,NodeDict,DimensionDict,Int8}

    utilized_symbols = SymbolDict()
    nodes = NodeDict()
    dimension_information = DimensionDict()
    cur_idx = start_idx

    for (idx, elem) in enumerate(entered_terminals_features)
        utilized_symbols[cur_idx] = 0
        nodes[cur_idx] = InputSelector(idx, string(elem))
        dimension_information[cur_idx] = get(dimensions_to_consider, elem, ZERO_DIM)
        cur_idx += 1
    end

    return utilized_symbols, nodes, dimension_information, cur_idx
end


function create_constants_entries(
    entered_terminal_nums::Vector{Symbol},
    rnd_count::Int,
    dimensions_to_consider::Dict{Symbol,Vector{Float16}},
    node_type::Type,
    start_idx::Int8,
    rng::AbstractRNG
)::Tuple{SymbolDict,NodeDict,DimensionDict,Int8}

    utilized_symbols = SymbolDict()
    nodes = NodeDict()
    dimension_information = DimensionDict()
    cur_idx = start_idx


    for elem in entered_terminal_nums
        utilized_symbols[cur_idx] = 0
        nodes[cur_idx] = parse(node_type, string(elem))
        dimension_information[cur_idx] = get(dimensions_to_consider, elem, ZERO_DIM)
        cur_idx += 1
    end


    for _ in 1:rnd_count
        utilized_symbols[cur_idx] = 0
        nodes[cur_idx] = node_type(rand(rng))
        dimension_information[cur_idx] = ZERO_DIM
        cur_idx += 1
    end

    return utilized_symbols, nodes, dimension_information, cur_idx
end


# preamble symbols: terminals appended to every chromosome, one per gene, outside its
# expression
function create_preamble_entries(
    preamble_syms_raw::Vector{Symbol},
    dimensions_to_consider::Dict{Symbol,Vector{Float16}},
    node_type::Type,
    start_idx::Int8
)::Tuple{SymbolDict,NodeDict,DimensionDict,Vector{Int8},Int8}

    utilized_symbols = SymbolDict()
    nodes = NodeDict()
    dimension_information = DimensionDict()
    preamble_syms = Int8[]
    cur_idx = start_idx

    for elem in preamble_syms_raw
        utilized_symbols[cur_idx] = 0
        nodes[cur_idx] = InputSelector(Int(cur_idx), string(elem))
        dimension_information[cur_idx] = get(dimensions_to_consider, elem, ZERO_DIM)
        push!(preamble_syms, cur_idx)
        cur_idx += 1
    end

    return utilized_symbols, nodes, dimension_information, preamble_syms, cur_idx
end


function merge_collections(
    func_symbols::SymbolDict,
    feat_symbols::SymbolDict,
    const_symbols::SymbolDict,
    preamble_symbols::SymbolDict
)::SymbolDict
    merged = SymbolDict()
    for dict in (func_symbols, feat_symbols, const_symbols, preamble_symbols)
        merge!(merged, dict)
    end
    return merged
end

"""
    BUFFERED_EVAL_MIN_SAMPLES

Not read by the package: `fit!` always uses the batched evaluator and ignores its
`buffered` keyword.
"""
const BUFFERED_EVAL_MIN_SAMPLES = 5_000


"""
    GepRegressor(feature_amount::Int; kwargs...)

Regressor for scalar models over `feature_amount` features. Train it with `fit!`;
`regressor(x)` predicts with the best model.

# Keyword arguments
- `entered_features::Vector{Symbol}=Symbol[]`: feature names; `[:x1, :x2, ...]` when empty
- `entered_non_terminals::Vector{Symbol}=[:+, :-, :*, :/]`: functions, from
  `FUNCTION_LIB_COMMON`; others are skipped with a message
- `gene_connections::Vector{Symbol}=[:+, :-, :*, :/]`: connectors; only those also in
  `entered_non_terminals` are used
- `entered_terminal_nums::Vector{Symbol}=[Symbol(0.5), Symbol(0.0)]`: constant terminals
- `rnd_count::Int=1`: further constant terminals, drawn uniformly from [0, 1)
- `node_type::Type=Float64`: numeric type of the constant terminals
- `gene_count::Int=3`: genes per chromosome
- `head_len::Int=6`: head length of each gene; the tail is `head_len + 1` long
- `tail_weigths=[0.6, 0.2, 0.2]`: sampling weights of a feature, a fixed constant and a
  random constant, per symbol
- `head_weigths=nothing`: accepted; not used
- `number_of_objectives::Int=1`: objectives; more than one needs the `fit!` method with a
  custom loss, and selection is then by NSGA-II
- `considered_dimensions::Dict{Symbol,Vector{Float16}}=Dict()`: dimensions of the features
  and constants, keyed by their symbols (`entered_features`, `entered_terminal_nums`);
  unlisted ones are dimensionless. When given, the regressor builds the library that
  semantic backpropagation (SBP) draws on, which `fit!` needs for a `target_dimension`.
- `max_permutations_lib::Int=10000`: new library expressions kept per build round
- `rounds::Int=4`: build rounds; library expressions have up to `rounds + 1` symbols
- `preamble_syms::Vector{Symbol}=Symbol[]`: terminals appended to every chromosome, one per
  gene, outside its expression

# Fields
- `best_models_`: the best chromosomes of the last `fit!`, best first
- `fitness_history_`: training and validation loss per epoch
- `toolbox_`: the GEP configuration
- `token_dto_`: the library and unit rules for SBP; `nothing` without
  `considered_dimensions`
"""
mutable struct GepRegressor
    toolbox_::Toolbox
    operators_::NamedTuple
    dimension_information_::OrderedDict{Int8,Vector{Float16}}
    best_models_::Union{Nothing,Vector{Chromosome}}
    fitness_history_::Any
    token_dto_::Union{TokenDto,Nothing}


    function GepRegressor(feature_amount::Int;
        entered_features::Vector{Symbol}=Vector{Symbol}(),
        entered_non_terminals::Vector{Symbol}=[:+, :-, :*, :/],
        entered_terminal_nums::Vector{Symbol}=[Symbol(0.5),Symbol(0.0)],
        gene_connections::Vector{Symbol}=[:+, :-, :*, :/],
        considered_dimensions::Dict{Symbol,Vector{Float16}}=Dict{Symbol,Vector{Float16}}(),
        rnd_count::Int=1,
        node_type::Type=Float64,
        gene_count::Int=3,
        head_len::Int=6,
        preamble_syms::Vector{Symbol}=Symbol[],
        max_permutations_lib::Int=10000, rounds::Int=4,
        number_of_objectives::Int=1,
        head_weigths::Union{Vector{<:AbstractFloat},Nothing}=nothing,
        tail_weigths::Union{Vector{<:AbstractFloat},Nothing}=[0.6,0.2,0.2]
    )        
        master_rng = Threefry4x(UInt64, (UInt64(rand(0:2000)), UInt64(0), UInt64(0), UInt64(0)))
        tail_count = feature_amount + rnd_count + length(entered_terminal_nums)
        tail_weigths_ = [tail_weigths[1]/tail_count for _ in 1:feature_amount]
        append!(tail_weigths_, fill(tail_weigths[2]/tail_count, length(entered_terminal_nums)))
        append!(tail_weigths_, fill(tail_weigths[3]/tail_count, rnd_count))

        entered_features_ = isempty(entered_features) ?
                            [Symbol("x$i") for i in 1:feature_amount] : entered_features
        # a key that names no feature, constant or preamble symbol is ignored, and a target
        # dimension can then be out of reach of every individual
        named = Set{Symbol}(vcat(entered_features_, entered_terminal_nums, preamble_syms))
        unnamed = sort!([k for k in keys(considered_dimensions) if !(k in named)])
        isempty(unnamed) || @warn("considered_dimensions has keys that name no feature or " *
            "constant terminal, and are ignored; the features are $(entered_features_)",
            unnamed)

        func_syms, callbacks, binary_ops, unary_ops, gene_connections_, cur_idx = create_function_entries(
            entered_non_terminals, gene_connections
        )

        feat_syms, feat_nodes, feat_dims, cur_idx = create_feature_entries(
            entered_features_, considered_dimensions, node_type, cur_idx
        )

        const_syms, const_nodes, const_dims, cur_idx = create_constants_entries(
            entered_terminal_nums, rnd_count, considered_dimensions, node_type, cur_idx, master_rng
        )

        pre_syms, pre_nodes, pre_dims, preamble_syms_, cur_idx = create_preamble_entries(
            preamble_syms, considered_dimensions, node_type, cur_idx
        )


        utilized_symbols = merge_collections(func_syms, feat_syms, const_syms, pre_syms)
        
        nodes = merge!(NodeDict(), feat_nodes, const_nodes, pre_nodes)
        
        dimension_information = merge!(DimensionDict(), feat_dims, const_dims, pre_dims)
        
        operators = (binary=binary_ops, unary=unary_ops)

        if !isempty(considered_dimensions)
            # the handlers are keyed by the symbol's index in the chromosome alphabet,
            # which is what func_syms carries
            idx_funs = Int8[idx for (idx, _) in func_syms]
            forward_funs, backward_funs, point_ops = create_physical_operations(entered_non_terminals, idx_funs)
            token_lib = TokenLib(
                dimension_information,
                forward_funs,
                utilized_symbols
            )
            idx_features = [idx for (idx, _) in feat_syms]
            idx_funs = [idx for (idx, _) in func_syms]
            idx_const = sbp_constants(const_syms, nodes)

            lib = create_lib(token_lib,
                idx_features,
                idx_funs,
                idx_const;
                rounds=rounds, max_permutations=max_permutations_lib)
            token_dto = TokenDto(token_lib, point_ops, lib, backward_funs, gene_count; head_len=head_len - 1)
        else
            token_dto = nothing
        end

        toolbox = Toolbox(gene_count, head_len, utilized_symbols, gene_connections_,
            callbacks, nodes, GENE_COMMON_PROBS; preamble_syms=preamble_syms_, number_of_objectives=number_of_objectives,
            operators_=operators, tail_weights_=weights(tail_weigths_), master_rng=master_rng)

        obj = new()
        obj.toolbox_ = toolbox
        obj.operators_ = operators
        obj.dimension_information_ = dimension_information
        obj.token_dto_ = token_dto
        return obj
    end
end


"""
    build_buffers(regressor::GepRegressor, x_data; std_return_type=Float64)

Buffer context for evaluating the regressor's chromosomes on `x_data` (one row per feature,
one column per sample) with the batched evaluator. Per-thread entries are indexed by
`Threads.threadid()`. Fields:

- `callbacks`, `nodes`, `pools`: the operator objects, one column per terminal, and the
  buffer pools of `calc_stack_batch_tensor`
- `program`, `fast`, `stacks`: the alphabet compiled for the monomorphic evaluator
  (`run_program!`) with its buffers and stacks; `program` is `nothing` when
  `compile_program` declines, and evaluation then uses `calc_stack_batch_tensor`
- `gene_pools`, `gene_fast`: a second set of buffers, into which gene-wise linear scaling
  evaluates the genes (`gene_basis`), apart from the buffers of an ordinary evaluation
- `designs`, `preds`: per thread, a design matrix `n × m` for every gene count `m` and a
  prediction column, which linear scaling fills instead of allocating them

Returns `nothing` when a function has no batched counterpart (`TENSOR_NODE_BY_FUNCTION`)
or a terminal is neither a feature nor a number; `fit!` then throws an `ArgumentError`.
"""
function build_buffers(regressor::GepRegressor, x_data::AbstractArray;
    std_return_type::Type=Float64)
    n = size(x_data, 2)

    callbacks = Dict{Int8,Any}()
    for (idx, f) in regressor.toolbox_.callbacks
        haskey(TENSOR_NODE_BY_FUNCTION, f) || return nothing
        callbacks[idx] = TENSOR_NODE_BY_FUNCTION[f]()
    end

    nodes = Dict{Int8,Any}()
    for (idx, nd) in regressor.toolbox_.nodes
        col = if nd isa InputSelector
            Vector{std_return_type}(@view x_data[nd.idx, :])
        elseif nd isa Number
            fill(std_return_type(nd), n)
        else
            return nothing
        end
        nodes[idx] = col
    end

    V = Vector{std_return_type}
    gene_buff = (regressor.toolbox_.gene_count + 1) * regressor.toolbox_.head_len
    # one pool per thread *id*, not per worker thread -- see `thread_slots`
    nslots = thread_slots()
    pools = [Dict{Type,NTuple}(V => Tuple([zeros(std_return_type, n) for _ in 1:gene_buff]))
             for _ in 1:nslots]
    gene_pools = [Dict{Type,NTuple}(V => Tuple([zeros(std_return_type, n) for _ in 1:gene_buff]))
                  for _ in 1:nslots]

    program = compile_program(callbacks, nodes, V)
    fast = [collect(V, p[V]) for p in pools]
    gene_fast = [collect(V, p[V]) for p in gene_pools]
    stacks = [sizehint!(V[], gene_buff + 2) for _ in 1:nslots]
    designs = [[Matrix{Float64}(undef, n, m) for m in 1:regressor.toolbox_.gene_count]
               for _ in 1:nslots]
    preds = [Vector{Float64}(undef, n) for _ in 1:nslots]

    return (callbacks=callbacks, nodes=nodes, pools=pools, gene_pools=gene_pools,
        program=program, fast=fast, gene_fast=gene_fast, stacks=stacks,
        designs=designs, preds=preds)
end

# TODO => adapt probs for occurrence of !
"""
    GepTensorRegressor(feature_amount::Int; kwargs...)

Regressor for models over scalars, vectors and higher-order tensors (Tensors.jl), with
`feature_amount` features, scalar and tensor-valued alike. Store the training columns with
`allocate_buffers!` before `fit!`; a loss typically scores a chromosome with
`predictT(regressor, chromosome.expression_raw)`.

# Keyword arguments
- `problem_dimension::Int=2`: spatial dimension of the tensors
- `feature_names::Vector{String}=String[]`: feature names for printing; `x1, x2, ...` when
  empty
- `entered_non_terminals::Vector{Symbol}=[:+, :-, :*, :/]`: functions, from `TENSOR_NODES`
- `gene_connections::Vector{Symbol}=[:+, :*]`: connectors; only those also in
  `entered_non_terminals` are used
- `entered_terminal_nums::Vector{<:AbstractFloat}=Float64[]`: constant terminals
- `rnd_count::Int=0`: further constant terminals, drawn uniformly from
  `rnd_limits::Tuple=(-0.1, 0.1)`
- `gene_count::Int=2`: genes per chromosome
- `head_len::Int=3`: head length of each gene
- `number_of_objectives::Int=1`: objectives
- `head_tail_balance::Real=0.6`: weight of the binary functions when head symbols are drawn
- `tail_weigths=[0.7, 0.2, 0.1]`: sampling weights of a feature, a fixed constant and a
  random constant, per symbol
- `considered_dimensions::Dict{Symbol,Vector{Float16}}=Dict()`: feature dimensions, keyed
  `:x1, :x2, ...` in feature order whatever `feature_names` says; each is the tensor order
  followed by the SI exponents. Constants are dimensionless. When given, the library for
  SBP is built as for `GepRegressor` (`max_permutations_lib::Int=10000`,
  `rounds::Int=5`); unit rules exist only for the functions in
  `FUNCTION_LIB_FORWARD_COMMON_TENSOR`.
- `higher_dim_feature_amount::Int=0`: accepted; not used

# Fields
- `best_models_`, `fitness_history_`, `toolbox_`, `token_dto_`: as for `GepRegressor`
- `input_values`, `buffers`: the input columns and per-thread buffers set by
  `allocate_buffers!`
"""
mutable struct GepTensorRegressor
    toolbox_::Toolbox
    problem_dimension::Int
    best_models_::Union{Nothing,Vector{Chromosome}}
    fitness_history_::Any
    buffers::Union{Array{Dict{Type,NTuple}},Nothing}
    input_values::Union{Nothing,Dict}
    mod_input_values::Union{Nothing,Dict}
    token_dto_::Union{TokenDto,Nothing}
    dimension_information_::OrderedDict{Int8,Vector{Float16}}
    # compiled evaluation for `predictT_scaled` and `predictT` (see `ScalarColumnCache`)
    scalar_cache_::Any

    function GepTensorRegressor(feature_amount::Int;
        problem_dimension::Int=2,
        higher_dim_feature_amount::Int=0,
        entered_non_terminals::Vector{Symbol}=[:+, :-, :*, :/],
        entered_terminal_nums::Vector{<:AbstractFloat}=Float64[],
        gene_connections::Vector{Symbol}=[:+, :*],
        rnd_count::Int=0,
        rnd_limits::Tuple=(-0.1, 0.1),
        gene_count::Int=2,
        head_len::Int=3,
        number_of_objectives::Int=1,
        feature_names::Vector{String}=String[],
        head_tail_balance::Real=0.6,
        tail_weigths::Union{Vector{<:AbstractFloat},Nothing}=[0.7, 0.2, 0.1],
        considered_dimensions::Dict{Symbol,Vector{Float16}}=Dict{Symbol,Vector{Float16}}(),
        max_permutations_lib::Int=10000, rounds::Int=5
    )
        master_rng = Threefry4x(UInt64, (UInt64(rand(0:2000)), UInt64(0), UInt64(0), UInt64(0)))
        tail_count = feature_amount + rnd_count + length(entered_terminal_nums)
        tail_weigths_ = [tail_weigths[1] / tail_count for _ in 1:feature_amount]
        append!(tail_weigths_, fill(tail_weigths[2] / tail_count, length(entered_terminal_nums)))
        append!(tail_weigths_, fill(tail_weigths[3] / tail_count, rnd_count))

        # symbol indices: features, then fixed and random constants, then the functions
        dimension_information = DimensionDict()
        cur_idx = Int8(1)
        nodes = OrderedDict{Int8,Any}()
        utilized_symbols = SymbolDict()
        callbacks = Dict{Int8,Any}()
        gene_connections_ = Int8[]

        feat_syms = Int8[]
        func_syms = Int8[]
        const_syms = Int8[]

        # a dimension here is the tensor order followed by the SI exponents, one entry
        # longer than ZERO_DIM; dimensionless terminals take the length of the given ones
        zdim = isempty(considered_dimensions) ? ZERO_DIM :
               zeros(Float16, length(first(values(considered_dimensions))))

        for _ in 1:feature_amount
            feature_name = isempty(feature_names) ? "x$cur_idx" : feature_names[cur_idx]
            nodes[cur_idx] = InputSelector(cur_idx, feature_name)
            utilized_symbols[cur_idx] = Int8(0)
            dimension_information[cur_idx] = get(considered_dimensions, Symbol("x$cur_idx"), zdim)
            push!(feat_syms, cur_idx)
            cur_idx += 1
        end

        # constants
        for elem in entered_terminal_nums
            nodes[cur_idx] = elem
            utilized_symbols[cur_idx] = Int8(0)
            dimension_information[cur_idx] = get(considered_dimensions, elem, zdim)
            push!(const_syms, cur_idx)
            cur_idx += 1
        end

        for _ in 1:rnd_count
            nodes[cur_idx] = rand(master_rng, Uniform(rnd_limits[1], rnd_limits[2]))
            utilized_symbols[cur_idx] = Int8(0)
            dimension_information[cur_idx] = zdim
            push!(const_syms, cur_idx)
            cur_idx += 1
        end

        # functions: index => operator object
        for elem in entered_non_terminals
            callbacks[cur_idx] = TENSOR_NODES[elem]()
            utilized_symbols[cur_idx] = TENSOR_NODES_ARITY[elem]
            if elem in gene_connections
                push!(gene_connections_, cur_idx)
            end
            push!(func_syms, cur_idx)
            cur_idx += 1
        end

        if !isempty(considered_dimensions)
            idx_features = feat_syms
            idx_funs = func_syms
            idx_const = sbp_constants(const_syms, nodes)
            forward_funs, backward_funs, point_ops = create_physical_operations(
                entered_non_terminals, idx_funs; function_forward=FUNCTION_LIB_FORWARD_COMMON_TENSOR,
                function_backward=FUNCTION_LIB_BACKWARD_COMMON_TENSOR, function_lib_common=TENSOR_NODES)
            token_lib = TokenLib(
                dimension_information,
                forward_funs,
                utilized_symbols
            )


            lib = create_lib(token_lib,
                idx_features,
                idx_funs,
                idx_const;
                rounds=rounds, max_permutations=max_permutations_lib)
            token_dto = TokenDto(token_lib, point_ops, lib, backward_funs, gene_count; head_len=head_len - 1)
        else
            token_dto = nothing
        end


        toolbox = Toolbox(gene_count, head_len, utilized_symbols, gene_connections_,
            callbacks, nodes, GENE_COMMON_PROBS; number_of_objectives=number_of_objectives,
            operators_=nothing, head_tail_balance=head_tail_balance,
            tail_weights_=weights(tail_weigths_), function_complile=(args...) -> nothing, master_rng=master_rng,
            constant_indices=const_syms)

        obj = new()
        obj.toolbox_ = toolbox
        obj.problem_dimension = problem_dimension
        obj.buffers = nothing
        obj.input_values = nothing
        obj.token_dto_ = token_dto
        obj.scalar_cache_ = nothing
        return obj
    end
end


# Unit rules of the tensor functions (`GepTensorRegressor`), on dimensions that carry the
# tensor order in front of the SI exponents. With `considered_dimensions`, only the
# functions listed here can be used. `inv` keeps the order and inverts the units;
# `hadamard`, an elementwise product, keeps the order and multiplies the units.
const FUNCTION_LIB_FORWARD_COMMON_TENSOR = Dict{Symbol,Function}(
    :+ => equal_unit_forward,
    :- => equal_unit_forward,
    :* => mul_t_unit_forward,
    :/ => div_t_unit_forward,
    :inv => inv_t_unit_forward,
    :dot => contraction_unit_forward,
    :crossp => crossp_unit_forward,
    :tr => zero_unit_forward,
    :det => zero_unit_forward,
    :dcontract => double_contraction_unit_forward,
    :lap => symmetric_contraction_forward,
    :hadamard => hadamard_unit_forward,
    :sqrt => zero_unit_forward,
    :norm => zero_unit_forward,
    :log => zero_unit_forward,
    :exp => zero_unit_forward,
    :sin => zero_unit_forward,
    :cos => zero_unit_forward
)


const FUNCTION_LIB_BACKWARD_COMMON_TENSOR = Dict{Symbol,Function}(
    :+ => equal_unit_backward,
    :- => equal_unit_backward,
    :* => mul_t_unit_backward,
    :/ => div_t_unit_backward,
    :inv => inv_t_unit_backward,
    :dot => contraction_unit_backward,
    :crossp => crossp_unit_backward,
    :tr => tr_unit_backward,
    :det => tr_unit_backward,
    :dcontract => double_contraction_unit_backward,
    :lap => symmetric_contraction_backward,
    :hadamard => hadamard_unit_backward,
    :sqrt => zero_unit_backward,
    :norm => arbitrary_unit_backward,
    :log => zero_unit_backward,
    :exp => zero_unit_backward,
    :sin => zero_unit_backward,
    :cos => zero_unit_backward
)



"""
    fit!(regressor::GepRegressor, epochs::Int, population_size::Int, x_train::AbstractArray,
         y_train::AbstractArray; kwargs...)

Evolve `regressor` on data for `epochs` epochs: `x_train` holds one row per feature and one
column per sample, `y_train` one target per sample. The best `hof` chromosomes end up in
`regressor.best_models_`, best first, and the loss history in `regressor.fitness_history_`.
Parents are chosen by tournaments of `max(3, ceil(0.03 * population_size))`. Throws an
`ArgumentError` when a function or terminal has no batched counterpart.

# Keyword arguments
- `x_test`, `y_test`: data for the validation loss; the training data when not given
- `loss_fun="mse"`: the loss to minimise, a name for `get_loss_function` or a function
  `(y_true, y_pred) -> Real`; `loss_fun_validation="mse"`: the validation loss, likewise
- `hof::Int=3`: number of best models kept
- `optimization_epochs::Int=100`: every this many epochs, if the best model has improved
  since the last time, Nelder-Mead tunes each occurrence of a constant in it on the
  training loss (`max_iterations::Int=1000` iterations); values that lower the loss are
  kept in its `optimised_constants`. Not with `linear_scaling`.
- `linear_scaling::Bool=false`: score each chromosome as the least-squares combination of
  its genes, one coefficient per gene (kept in its `scaling_weights`), so that evolution
  searches for the structure only. Needs `:+` and `:*` among the entered non-terminals.
  With a `target_dimension`, every gene is held to it (not only the connected
  expression), since the coefficients are dimensionless only then; see
  `make_dimension_contract`.
- `target_dimension::Union{Vector{Float16},Nothing}=nothing`: dimension of the target;
  needs a regressor built with `considered_dimensions`. Only homogeneous individuals are
  scored; the others are repaired by semantic backpropagation (SBP):
  - `correction_epochs::Int=1`: repair every this many epochs
  - `correction_amount::Real=1.0`: the most individuals repaired per correction epoch, as
    a fraction of the population
  - `cycles::Int=10`: repair attempts per individual
  - `lib_seed_amount::Real=0.5`: fraction of the initial population seeded from the
    library (see `make_lib_seeder`)
- `penalty::AbstractFloat=2.0`: factor on the fitness of a new individual whose karva
  string has been scored before
- `population_sampling_multiplier::Int=1`: above 1, the initial population is picked by
  Latin hypercube sampling from this many times more random candidates
- `break_condition`: `(population, epoch) -> Bool`; `true` stops the run
- `file_logger_callback`: `(population, epoch, selected)`, called every epoch
- `save_state_callback`: `(population, strategy)`, called every epoch
- `load_state_callback`: `() -> (population, start_epoch)`, to resume a run
- `surrogate::Union{SurrogateScreening,Nothing}=nothing`: for an expensive loss, a
  surrogate screening (e.g. `SurrogateScreening(regressor, probes)`): the loss scores only
  the individuals it picks each epoch, and the others get the prediction of a Gaussian
  process; see `GepSurrogate`
- `opt_method_const`, `n_starts`, `buffered`: accepted; not used
"""
function fit!(regressor::GepRegressor, epochs::Int, population_size::Int, x_train::AbstractArray,
    y_train::AbstractArray; x_test::Union{AbstractArray,Nothing}=nothing, y_test::Union{AbstractArray,Nothing}=nothing,
    optimization_epochs::Int=100,
    hof::Int=3, loss_fun::Union{String,Function}="mse",
    loss_fun_validation::Union{String, Function}="mse",
    correction_epochs::Int=1, correction_amount::Real=1.0,
    opt_method_const::Symbol=:cg,
    target_dimension::Union{Vector{Float16},Nothing}=nothing,
    cycles::Int=10, max_iterations::Int=1000, n_starts::Int=3,
    break_condition::Union{Function,Nothing}=nothing,
    file_logger_callback::Union{Function,Nothing}=nothing,
    save_state_callback::Union{Function,Nothing}=nothing,
    load_state_callback::Union{Function,Nothing}=nothing, 
    population_sampling_multiplier::Int=1, 
    lib_seed_amount::Real=0.5,
    penalty::AbstractFloat = 2.0,
    linear_scaling::Bool=false,
    buffered::Union{Bool,Symbol}=:auto,
    surrogate::Union{SurrogateScreening,Nothing}=nothing
)
    buffered_ctx = build_buffers(regressor, x_train)
    isnothing(buffered_ctx) && throw(ArgumentError(
        "an operator or terminal in this regressor has no batched counterpart, so its " *
        "candidates cannot be evaluated"))

    if linear_scaling
        ops = string.(regressor.operators_.binary)
        if !("+" in ops && "*" in ops)
            throw(ArgumentError(
                "linear_scaling writes the scaled model as a sum of weighted genes, so " *
                "it needs :+ and :* among the entered non-terminals; got $(ops)"))
        end
    end

    # a scaled model is the weighted sum of its genes, so every gene must take the target
    correction_callback, homogeneity_check = make_dimension_contract(
        regressor.toolbox_, regressor.token_dto_, target_dimension, cycles;
        gene_wise=linear_scaling)

    # constant optimiser (the strategy's `secOptimizer`): Nelder-Mead over the constant
    # occurrences of the best individual, each a separate parameter. It needs only the
    # operators and the input columns, which `evaluate_with_constants` does not modify.
    const_ctx = buffered_ctx
    function optimizer_wrapper(population::Vector{Chromosome})
        elem = population[1]
        positions = constant_positions(elem)
        isempty(positions) && return
        x0 = Float64[Float64(regressor.toolbox_.nodes[elem.expression_raw[p]])
                     for p in positions]
        lossfn = loss_fun isa String ? get_loss_function(loss_fun) : loss_fun
        ctx = (callbacks=const_ctx.callbacks, nodes=const_ctx.nodes)
        function opt_step(v::AbstractVector)
            pred = evaluate_with_constants(elem, ctx, v)
            pred isa AbstractVector || return Inf
            l = lossfn(y_train, pred)
            return isfinite(l) ? l : Inf
        end
        try
            res = Optim.optimize(opt_step, x0, Optim.NelderMead(),
                Optim.Options(; iterations=max_iterations, show_trace=false))
            if res.minimum < mean(elem.fitness)
                elem.fitness = (res.minimum,)
                elem.optimised_constants = Optim.minimizer(res)
            end
        catch e
            @debug "ignored constant optimisation" exception = e
        end
    end

    x_valid = !isnothing(x_test) ? x_test : x_train
    evalStrat = StandardRegressionStrategy{typeof(first(x_train))}(
        regressor.operators_,
        x_train,
        y_train,
        x_valid,
        !isnothing(y_test) ? y_test : y_train,
        loss_fun isa String ? get_loss_function(loss_fun) : loss_fun;
        validation_loss_function = loss_fun_validation isa String ? get_loss_function(loss_fun_validation) : loss_fun_validation, 
        # the optimiser tunes the connected expression, which a scaled model does not use
        secOptimizer=linear_scaling ? nothing : optimizer_wrapper,
        break_condition=break_condition,
        linear_scaling=linear_scaling,
        buffered=buffered_ctx,
        # the best of every epoch is validated in one context, not a new one each time
        validation_ctx=buffer_context(regressor.toolbox_, x_valid)
    )

    best, history = runGep(epochs,
        population_size,
        regressor.toolbox_,
        evalStrat;
        hof=hof,
        correction_callback=correction_callback,
        homogeneity_check=homogeneity_check,
        population_seeder=make_lib_seeder(regressor.toolbox_, regressor.token_dto_,
            target_dimension, lib_seed_amount; gene_wise=linear_scaling),
        correction_epochs=correction_epochs,
        correction_amount=correction_amount,
        tourni_size=max(Int(ceil(population_size * 0.03)), 3),
        optimization_epochs=optimization_epochs,
        file_logger_callback=file_logger_callback,
        save_state_callback=save_state_callback,
        load_state_callback=load_state_callback,
        population_sampling_multiplier=population_sampling_multiplier,
        penalty = penalty,
        surrogate=surrogate
    )

    regressor.best_models_ = best
    regressor.fitness_history_ = history
end

"""
    fit!(regressor::GepRegressor, epochs::Int, population_size::Int,
         loss_function::Function; kwargs...)

Evolve `regressor` against a custom loss, e.g. with several objectives (selection by
NSGA-II). `loss_function(chromosome, validate::Bool)` sets `chromosome.fitness` to a tuple
with one entry per objective. It is called for every unscored chromosome (fitness `NaN`),
from several threads at once, and each epoch with `validate = true` for the best one; to
evaluate, use a per-thread context from `thread_contexts`.

Keywords as for the data method: `hof`, `target_dimension`, `correction_epochs`,
`correction_amount`, `cycles`, `lib_seed_amount`, `penalty`, `break_condition`,
`file_logger_callback`, `save_state_callback`, `load_state_callback`. Further:

- `loss_function_validation=nothing`: `(chromosome, validate) -> tuple`; its return value
  for the best chromosome is recorded as the validation loss each epoch
- `gene_wise_dimension::Bool=false`: hold every gene to `target_dimension` rather than the
  connected expression, for a loss that scores the least-squares combination of the genes
  (see `make_dimension_contract`)
- `surrogate::Union{SurrogateScreening,Nothing}=nothing`: for an expensive loss, e.g. a
  solver in the loop, a surrogate screening: `loss_function` is called only for the
  individuals it picks each epoch (and for a best or hall of fame member that carries a
  prediction), and the others get the prediction of a Gaussian process; see
  `SurrogateScreening`
- `constant_optimizer::Union{ScreenedNelderMead,Nothing}=nothing`: tunes the constants of
  the best chromosome against `loss_function`, every `optimization_epochs::Int=100` epochs
  if it has improved since, by Nelder-Mead, screened by a Gaussian process or not (see
  `optimize_constants!`); the loss must evaluate the chromosome in a way that applies tuned
  constants. With a `surrogate`, it tunes the best chromosome the loss has scored, passing
  over the ones that carry a prediction.
- `optimizer_function_`, `opt_method_const`, `max_iterations`, `n_starts`: accepted; not
  used
"""
function fit!(regressor::GepRegressor, epochs::Int, population_size::Int, loss_function::Function;
    optimizer_function_::Union{Function,Nothing}=nothing,
    loss_function_validation::Union{Function, Nothing}=nothing,
    optimization_epochs::Int=100,
    hof::Int=3,
    correction_epochs::Int=1,
    correction_amount::Real=1.0,
    opt_method_const::Symbol=:nd,
    target_dimension::Union{Vector{Float16},Nothing}=nothing,
    cycles::Int=10, max_iterations::Int=150, n_starts::Int=5,
    break_condition::Union{Function,Nothing}=nothing,
    file_logger_callback::Union{Function,Nothing}=nothing,
    save_state_callback::Union{Function,Nothing}=nothing,
    load_state_callback::Union{Function,Nothing}=nothing,
    lib_seed_amount::Real=0.5,
    penalty::AbstractFloat = 2.0,
    gene_wise_dimension::Bool=false,
    surrogate::Union{SurrogateScreening,Nothing}=nothing,
    constant_optimizer::Union{ScreenedNelderMead,Nothing}=nothing
)

    correction_callback, homogeneity_check = make_dimension_contract(
        regressor.toolbox_, regressor.token_dto_, target_dimension, cycles;
        gene_wise=gene_wise_dimension)

    evalStrat = GenericRegressionStrategy(
        regressor.operators_,
        length(regressor.toolbox_.fitness_reset[1]),
        loss_function;
        validation_loss_function = loss_function_validation,
        secOptimizer=leader_constant_search(constant_optimizer, loss_function, surrogate,
            population_size),
        break_condition=break_condition
    )

    best, history = runGep(epochs,
        population_size,
        regressor.toolbox_,
        evalStrat;
        hof=hof,
        correction_callback=correction_callback,
        homogeneity_check=homogeneity_check,
        population_seeder=make_lib_seeder(regressor.toolbox_, regressor.token_dto_,
            target_dimension, lib_seed_amount; gene_wise=gene_wise_dimension),
        correction_epochs=correction_epochs,
        correction_amount=correction_amount,
        tourni_size=max(Int(ceil(population_size * 0.03)), 3),
        optimization_epochs=optimization_epochs,
        file_logger_callback=file_logger_callback,
        save_state_callback=save_state_callback,
        load_state_callback=load_state_callback,
        penalty = penalty,
        surrogate=surrogate
    )

    regressor.best_models_ = best
    regressor.fitness_history_ = history
end

"""
    fit!(regressor::GepTensorRegressor, epochs::Int, population_size::Int,
         loss_function::Function; kwargs...)

Evolve a tensor regressor against a custom loss of the same form as for `GepRegressor`,
typically evaluating with `predictT`; call `allocate_buffers!` first. Parents are chosen by
tournaments of `max(3, ceil(0.003 * population_size))`, or by NSGA-II with several
objectives. Keywords as for the `GepRegressor` methods: `hof`, `target_dimension`,
`correction_epochs`, `correction_amount`, `cycles`, `lib_seed_amount`,
`population_sampling_multiplier`, `break_condition`, `file_logger_callback`,
`save_state_callback`, `load_state_callback`, and `gene_wise_dimension`, which a loss
scoring with `predictT_scaled` needs unless the gene connectors are only `+` and `-`. A
`surrogate` (e.g. `SurrogateScreening(regressor, probes)`, embedding with a
`TensorEmbedder`) screens an expensive loss as for `GepRegressor`, and a
`constant_optimizer` (a `ScreenedNelderMead`) tunes the constants of the best chromosome
every `optimization_epochs::Int=100` epochs as for `GepRegressor`, for a loss that
evaluates with `predictT(regressor, chromosome)`.
"""
function fit!(regressor::GepTensorRegressor, epochs::Int, population_size::Int, loss_function::Function;
    hof::Int=3,
    correction_epochs::Int=1,
    correction_amount::Real=1.0,
    cycles::Int=10,
    target_dimension::Union{Vector{Float16},Nothing}=nothing,
    break_condition::Union{Function,Nothing}=nothing,
    file_logger_callback::Union{Function,Nothing}=nothing,
    save_state_callback::Union{Function,Nothing}=nothing,
    load_state_callback::Union{Function,Nothing}=nothing,
    population_sampling_multiplier::Int=1,
    lib_seed_amount::Real=0.5,
    gene_wise_dimension::Bool=false,
    surrogate::Union{SurrogateScreening,Nothing}=nothing,
    constant_optimizer::Union{ScreenedNelderMead,Nothing}=nothing,
    optimization_epochs::Int=100
)

    correction_callback, homogeneity_check = make_dimension_contract(
        regressor.toolbox_, regressor.token_dto_, target_dimension, cycles;
        gene_wise=gene_wise_dimension)

    evalStrat = GenericRegressionStrategy(
        nothing,
        length(regressor.toolbox_.fitness_reset[1]),
        loss_function;
        secOptimizer=leader_constant_search(constant_optimizer, loss_function, surrogate,
            population_size),
        break_condition=break_condition
    )

    best, history = runGep(epochs,
        population_size,
        regressor.toolbox_,
        evalStrat;
        hof=hof,
        correction_callback=correction_callback,
        homogeneity_check=homogeneity_check,
        population_seeder=make_lib_seeder(regressor.toolbox_, regressor.token_dto_,
            target_dimension, lib_seed_amount; gene_wise=gene_wise_dimension),
        correction_epochs=correction_epochs,
        correction_amount=correction_amount,
        tourni_size=max(Int(ceil(population_size * 0.003)), 3),
        file_logger_callback=file_logger_callback,
        save_state_callback=save_state_callback,
        load_state_callback=load_state_callback,
        population_sampling_multiplier=population_sampling_multiplier,
        optimization_epochs=optimization_epochs,
        surrogate=surrogate
    )

    regressor.best_models_ = best
    regressor.fitness_history_ = history
end

"""
    leader_constant_search(method, loss_function, surrogate, population_size)

The secondary optimiser of a search against a custom loss: `optimize_constants!` with
`method` on the first chromosome of the population with a finite fitness that the loss has
scored, or `nothing` without a `method`. Without a `surrogate` that is the best one; with
one, the chromosomes that carry a prediction are passed over, so that the memory of the
surrogate holds losses of drawn constants only, and the search stays among the survivors.
"""
function leader_constant_search(method, loss_function::Function, surrogate,
    population_size::Int)
    isnothing(method) && return nothing
    return function (population::Vector{Chromosome})
        for chromosome in view(population, 1:min(population_size, length(population)))
            isnothing(surrogate) || is_validated(surrogate, chromosome) || continue
            all(isfinite, chromosome.fitness) || continue
            optimize_constants!(chromosome, loss_function; method=method)
            break
        end
        return nothing
    end
end

# one entry of objective_expressions per objective, checked before the loss is ever called
function check_objective_expressions(toolbox, kwargs)
    tie = get(kwargs, :objective_expressions, nothing)
    isnothing(tie) && return
    k = length(toolbox.fitness_reset[1])
    length(tie) == k || throw(ArgumentError(
        "objective_expressions names $(length(tie)) expressions for $k objectives"))
    return
end

"""
    SurrogateScreening(regressor::GepRegressor, probes::AbstractMatrix;
        embedding=:expression, expressions=1, transform=:asinh, kwargs...)

A [`SurrogateScreening`](@ref) for the chromosomes of `regressor`, embedded on `probes`
(one row per feature, one column per probe sample, like the training data of `fit!`; a
few dozen samples from the relevant range suffice, e.g. a random subset of the training
data). `embedding = :expression` embeds the expression (`SemanticEmbedder`); a loss that
splits every chromosome into `expressions` expressions (`split_karva(elem, expressions)`)
gets one block per expression, and with `objective_expressions` (one expression per
objective, e.g. `[1, 2]`) the process of each objective sees the block of its expression
alone. `embedding = :genes` embeds one block per gene (`GeneEmbedder`), which suits a loss
that scores the least-squares combination of the genes. `transform` is the transform of
the probe outputs; the other keywords go to `SurrogateScreening`.
"""
function GepSurrogate.SurrogateScreening(regressor::GepRegressor, probes::AbstractMatrix;
    embedding::Symbol=:expression, expressions::Integer=1,
    transform::Union{Symbol,AbstractString}=:asinh, kwargs...)
    check_objective_expressions(regressor.toolbox_, kwargs)
    embedder = if embedding === :expression
        SemanticEmbedder(regressor.toolbox_, probes; transform=transform,
            expressions=expressions)
    elseif embedding === :genes
        GeneEmbedder(regressor.toolbox_, probes; transform=transform)
    else
        throw(ArgumentError("the embedding $embedding is unknown, use :expression or :genes"))
    end
    return SurrogateScreening(embedder; kwargs...)
end

"""
    SurrogateScreening(regressor::GepTensorRegressor, probes::AbstractVector;
        embedding=:expression, expressions=1, components=1, transform=:asinh, kwargs...)

A [`SurrogateScreening`](@ref) for the chromosomes of a tensor regressor, embedded with a
`TensorEmbedder` on `probes` (one column per feature, in feature order, over the probe
samples, like the data given to `allocate_buffers!`). `components` is the number of
components of an output per sample (1 for a scalar, `dim` for a vector, `dim^2` for a
second-order tensor), one number or one per expression of a template loss that splits
every chromosome into `expressions` expressions; `embedding = :genes` embeds one block per
gene.
"""
function GepSurrogate.SurrogateScreening(regressor::GepTensorRegressor, probes::AbstractVector;
    embedding::Symbol=:expression, expressions::Integer=1,
    components::Union{Integer,AbstractVector{<:Integer}}=1,
    transform::Union{Symbol,AbstractString}=:asinh, kwargs...)
    embedding in (:expression, :genes) || throw(ArgumentError(
        "the embedding $embedding is unknown, use :expression or :genes"))
    check_objective_expressions(regressor.toolbox_, kwargs)
    embedder = TensorEmbedder(regressor.toolbox_, probes; components=components,
        transform=transform, per_gene=embedding === :genes, expressions=expressions)
    return SurrogateScreening(embedder; kwargs...)
end



"""
    gene_bases(regressor::GepTensorRegressor, chromosome::Chromosome)

The output of each gene of `chromosome`, evaluated on its own on the columns stored by
`allocate_buffers!`: one batch per gene, the basis for fitting gene coefficients
(`predictT_scaled`). The genes are evaluated into fresh arrays rather than the per-thread
buffers, since all their outputs are needed at once.
"""
function gene_bases(regressor::GepTensorRegressor, chromosome::Chromosome)
    raw = _karva_raw(chromosome; split=true)
    return [calc_stack_batch_tensor(collect(raw[j]), regressor.toolbox_.callbacks,
                                    regressor.input_values, nothing)
            for j in 2:length(raw)]
end

"""
    flatten_components(batch)

The components of a batch of tensors (or numbers) in one `Float64` vector, sample by
sample, for a scalar least-squares solve.
"""
@inline function flatten_components(batch::AbstractVector)
    n = length(batch)
    n == 0 && return Float64[]
    k = length(first(batch))
    out = Vector{Float64}(undef, n * k)
    @inbounds for (i, t) in enumerate(batch)
        for c in 1:k
            out[(i-1)*k+c] = Float64(t[c])
        end
    end
    return out
end

"""
    ScalarColumnCache

State for evaluating a `GepTensorRegressor` whose input columns are all `Vector{Float64}`
with the compiled evaluator (`compile_program`, `run_program!`) instead of
`calc_stack_batch_tensor`: the compiled alphabet and, per thread slot, the operator
buffers, the stack, and one design matrix `n × m` for every gene count `m`, so that
`predictT_scaled!` allocates no column. Both evaluators apply the same kernels in the same
order, so results are bit-identical.

It is built from `input_values` on first use and rebuilt when that dictionary, or a column
in it, is replaced. `program` is `nothing` when the compiled evaluator does not apply: a
column that is not a finite `Vector{Float64}` of the common length (the generic evaluator
rejects a terminal whose `norm` is not finite), a terminal without a column, or an
operator `compile_program` does not take.
"""
struct ScalarColumnCache
    inputs::Any
    columns::Vector{Pair{Int8,Any}}
    n::Int
    program::Union{EvalProgram{Vector{Float64}},Nothing}
    fast::Vector{Vector{Vector{Float64}}}
    stacks::Vector{Vector{Vector{Float64}}}
    designs::Vector{Vector{Matrix{Float64}}}
end

const SCALAR_CACHE_LOCK = ReentrantLock()

function build_scalar_cache(regressor::GepTensorRegressor)
    inputs = regressor.input_values
    columns = inputs isa Dict ? Pair{Int8,Any}[k => v for (k, v) in inputs] : Pair{Int8,Any}[]
    none = ScalarColumnCache(inputs, columns, 0, nothing, [], [], [])
    isempty(columns) && return none
    tb = regressor.toolbox_
    n = length(last(first(columns)))
    all(((_, c),) -> c isa Vector{Float64} && length(c) == n && isfinite(norm(c)), columns) ||
        return none
    all(((sym, arity),) -> arity != 0 || haskey(inputs, sym), tb.symbols) || return none
    program = compile_program(tb.callbacks, inputs, Vector{Float64})
    isnothing(program) && return none
    # enough buffers for a whole karva string, the connectors included
    nbuf = (tb.gene_count + 1) * tb.head_len
    slots = thread_slots()
    return ScalarColumnCache(inputs, columns, n, program,
        [[zeros(n) for _ in 1:nbuf] for _ in 1:slots],
        [Vector{Float64}[] for _ in 1:slots],
        [[Matrix{Float64}(undef, n, m) for m in 1:tb.gene_count] for _ in 1:slots])
end

@inline function cache_current(cache::ScalarColumnCache, inputs)
    cache.inputs === inputs || return false
    inputs isa Dict || return true
    length(inputs) == length(cache.columns) || return false
    for (k, v) in cache.columns
        get(inputs, k, nothing) === v || return false
    end
    return true
end

function scalar_cache(regressor::GepTensorRegressor)
    cache = regressor.scalar_cache_
    cache isa ScalarColumnCache && cache_current(cache, regressor.input_values) &&
        return cache
    return lock(SCALAR_CACHE_LOCK) do
        cache = regressor.scalar_cache_
        if !(cache isa ScalarColumnCache && cache_current(cache, regressor.input_values))
            cache = build_scalar_cache(regressor)
            regressor.scalar_cache_ = cache
        end
        cache
    end
end

"""
    compiled_gene_basis(cache, chromosome)

The design matrix of `predictT_scaled`: gene `j` of `chromosome`, evaluated on its own by
the compiled evaluator, in column `j` of this thread's preallocated `n × genes` matrix
(valid until the next call on the same thread). Returns `nothing` if a gene cannot be
evaluated, and the caller then takes the generic route, which reports that case as before.
"""
function compiled_gene_basis(cache::ScalarColumnCache, chromosome::Chromosome)
    raw = _karva_raw(chromosome; split=true)
    m = length(raw) - 1
    tid = Threads.threadid()
    designs = cache.designs[tid]
    1 <= m <= length(designs) || return nothing
    G = designs[m]
    fast = cache.fast[tid]
    stack = cache.stacks[tid]
    for j in 1:m
        col = run_program!(raw[j+1], cache.program, fast, stack)
        col isa Vector{Float64} || return nothing
        @inbounds G[:, j] .= col
    end
    return G
end

"""
    combine_genes!(out, G, w)

`out = G * w`, accumulated per sample in the order of `predictT_scaled!`'s generic loop,
`(w[1] * G[i, 1] + w[2] * G[i, 2]) + ...`, so both give the same bits. Up to four genes
the loop over genes is unrolled and each sample is summed in one pass.
"""
function combine_genes!(out::Vector{Float64}, G::Matrix{Float64}, w::Vector{Float64})
    m = size(G, 2)
    m == 1 && return combine_genes!(out, G, w, Val(1))
    m == 2 && return combine_genes!(out, G, w, Val(2))
    m == 3 && return combine_genes!(out, G, w, Val(3))
    m == 4 && return combine_genes!(out, G, w, Val(4))
    @inbounds out .= w[1] .* view(G, :, 1)
    @inbounds for j in 2:m
        out .+= w[j] .* view(G, :, j)
    end
    return out
end

@inline function combine_genes!(out::Vector{Float64}, G::Matrix{Float64},
    w::Vector{Float64}, ::Val{M}) where {M}
    ws = ntuple(j -> w[j], Val(M))
    @inbounds for i in eachindex(out)
        acc = ws[1] * G[i, 1]
        for j in 2:M
            acc += ws[j] * G[i, j]
        end
        out[i] = acc
    end
    return out
end

"""
    predictT_scaled(regressor::GepTensorRegressor, chromosome::Chromosome, target)

Prediction of `chromosome` as a least-squares combination of its genes, one coefficient per
gene fitted against `target`: the tensor counterpart of `linear_scaling`, and of the tensor
linear regression of SITE (arXiv:2507.01466). Genes whose output does not match `target`
in length and element type are left out; the coefficients of the others are stored in
`chromosome.scaling_weights`. Returns `nothing` when no gene matches or the fit is not
finite. The coefficients are dimensionless only if every gene used has the target
dimension: with a `target_dimension`, fit with `gene_wise_dimension=true` unless the gene
connectors are only `+` and `-`.

When every input column is a `Vector{Float64}` and so is `target`, the genes are evaluated
by the compiled evaluator into preallocated buffers (`ScalarColumnCache`), with the same
result. [`predictT_scaled!`](@ref) writes the prediction into a given vector instead.
"""
predictT_scaled(regressor::GepTensorRegressor, chromosome::Chromosome,
    target::AbstractVector) = predictT_scaled!(similar(target), regressor, chromosome, target)

"""
    predictT_scaled!(out, regressor::GepTensorRegressor, chromosome::Chromosome, target)

[`predictT_scaled`](@ref), with the prediction written into `out`, a vector like `target`
(`similar(target)`), which is returned. On `nothing` the contents of `out` are undefined.
A loss that keeps one `out` per thread (indexed by `threadid()`, with
[`thread_slots`](@ref) entries) scores Float64 columns without allocating per candidate.
"""
function predictT_scaled!(out::AbstractVector, regressor::GepTensorRegressor,
    chromosome::Chromosome, target::AbstractVector)
    axes(out) == axes(target) ||
        throw(DimensionMismatch("out has axes $(axes(out)), target $(axes(target))"))
    cache = scalar_cache(regressor)
    if !isnothing(cache.program) && target isa Vector{Float64} && length(target) == cache.n
        G = compiled_gene_basis(cache, chromosome)
        if !isnothing(G)
            # the generic route below, on the same values: G holds the gene outputs
            allfinite(G) || return nothing
            w = solve_scaling(G, target)
            all(isfinite, w) || return nothing
            chromosome.scaling_weights = w
            return combine_genes!(out, G, w)
        end
    end

    bases = gene_bases(regressor, chromosome)
    usable = [b for b in bases if b isa AbstractVector && length(b) == length(target) &&
              eltype(b) <: eltype(target)]
    isempty(usable) && return nothing

    G = reduce(hcat, (flatten_components(b) for b in usable))
    all(isfinite, G) || return nothing
    w = solve_scaling(G, flatten_components(target))
    all(isfinite, w) || return nothing
    chromosome.scaling_weights = w

    @inbounds for i in eachindex(target)
        acc = w[1] * usable[1][i]
        for j in 2:length(usable)
            acc += w[j] * usable[j][i]
        end
        out[i] = acc
    end
    return out
end

"""
    predictT(regressor::GepTensorRegressor, rek_string::Vector)

Evaluate a karva string (a chromosome's `expression_raw`) on the columns stored by
`allocate_buffers!`, in the calling thread's buffers; the form for a loss. The result may
be one of those buffers: use or copy it before the next evaluation on the same thread.
Returns `NaN`, or throws, when the expression cannot be evaluated. Scalar columns are
evaluated by the compiled evaluator, as in `predictT_scaled`.
"""
@inline function predictT(regressor::GepTensorRegressor, rek_string::Vector)
    cache = scalar_cache(regressor)
    if !isnothing(cache.program) && eltype(rek_string) === Int8
        tid = Threads.threadid()
        col = run_program!(rek_string, cache.program, cache.fast[tid], cache.stacks[tid])
        col isa Vector{Float64} && return col
    end
    return calc_stack_batch_tensor(rek_string, regressor.toolbox_.callbacks, regressor.input_values, regressor.buffers[Threads.threadid()])
end

"""
    predictT(regressor::GepTensorRegressor, chromosome::Chromosome)

[`predictT`](@ref) of the chromosome's karva string, with its tuned constants
(`optimised_constants`) if it has them, which a loss needs for `optimize_constants!` to
reach it. With tuned constants the expression runs on the allocating path of the
evaluator, into fresh arrays.
"""
function predictT(regressor::GepTensorRegressor, chromosome::Chromosome)
    isnothing(chromosome.optimised_constants) &&
        return predictT(regressor, chromosome.expression_raw)
    return evaluate_with_constants(chromosome,
        (callbacks=regressor.toolbox_.callbacks, nodes=regressor.input_values),
        chromosome.optimised_constants)
end

"""
    predictT(regressor::GepTensorRegressor, rek_string::Vector, new_input_values::Dict)

Evaluate a karva string on the stored columns with some replaced: for each
`index => value` in `new_input_values`, the column of terminal `index` (an existing key of
`regressor.input_values`) takes `value` in every sample.
"""
@inline function predictT(regressor::GepTensorRegressor, rek_string::Vector, new_input_values::Dict)
    input_val_cp = copy(regressor.input_values)
    for (key, elem) in new_input_values
        input_val_cp[key] = ones(length(input_val_cp[key])) .* elem
    end
    return calc_stack_batch_tensor(rek_string, regressor.toolbox_.callbacks, input_val_cp, regressor.buffers[Threads.threadid()])
end

"""
    predictT(regressor::GepTensorRegressor, rek_string::Vector, x_data::Vector)

Evaluate a karva string on new data: `x_data` holds one column per feature, in feature
order, like the data given to `allocate_buffers!` (which must have run). Constants are
broadcast to the new sample count, and the result is written to fresh arrays, since the
buffers are sized for the training data.
"""
@inline function predictT(regressor::GepTensorRegressor, rek_string::Vector, x_data::Vector)
    n = length(x_data[1])
    inputs = Dict{Int8,Any}()
    for (index, elem) in enumerate(x_data)
        inputs[Int8(index)] = elem
    end
    for (key, elem) in regressor.input_values
        if !haskey(inputs, key)
            inputs[key] = fill(elem[1], n)
        end
    end
    return calc_stack_batch_tensor(rek_string, regressor.toolbox_.callbacks, inputs, nothing)
end

"""
    allocate_buffers!(regressor::GepTensorRegressor, data_x; std_return_type=Float64)

Store the input columns for `predictT` and allocate per-thread buffers sized to them.
`data_x` holds one column (a vector over the samples) per feature, in feature order; the
columns may differ in element type -- numbers, `Vec`s, second-order tensors -- so pass a
`Tuple` or a `Vector{Any}`. Constant terminals become constant columns. The buffers cover
scalars and tensors of order 1 to 4 in `problem_dimension` dimensions; `std_return_type`
is not used.
"""
allocate_buffers!(regressor::GepTensorRegressor, data_x::Tuple; std_return_type::Type=Float64) =
    allocate_buffers!(regressor, Any[data_x...]; std_return_type=std_return_type)

function allocate_buffers!(regressor::GepTensorRegressor, data_x::AbstractArray; std_return_type::Type=Float64)
    gene_count = regressor.toolbox_.gene_count
    head_len = regressor.toolbox_.head_len

    gene_buff = (gene_count + 1) * head_len
    n_samples = length(data_x[1])
    prob_dim = regressor.problem_dimension
    buffers = [Dict{Type,NTuple}(
        Vector{Float64} => Tuple([zeros(Float64, n_samples) for _ in 1:gene_buff]),
        Vector{Vec{prob_dim}} => Tuple([zeros(Vec{prob_dim}, n_samples) for _ in 1:gene_buff]),
        Vector{Tensor{1,prob_dim}} => Tuple([zeros(Vec{prob_dim}, n_samples) for _ in 1:gene_buff]),
        Vector{Tensor{2,prob_dim}} => Tuple([zeros(Tensor{2,prob_dim}, n_samples) for _ in 1:gene_buff]),
        Vector{Tensor{3,prob_dim}} => Tuple([zeros(Tensor{3,prob_dim}, n_samples) for _ in 1:gene_buff]),
        Vector{Tensor{4,prob_dim}} => Tuple([zeros(Tensor{4,prob_dim}, n_samples) for _ in 1:gene_buff])
    ) for _ in 1:thread_slots()]


    regressor.buffers = buffers

    inputs = Dict{Int8,Any}(
        key => value for (key, value) in enumerate(data_x)
    )


    for key in keys(regressor.toolbox_.nodes)
        if !(key in keys(inputs))
            inputs[key] = regressor.toolbox_.nodes[key] * ones(n_samples)
        end
    end

    regressor.input_values = inputs

end

"""
    (regressor::GepRegressor)(x_data::AbstractArray; ensemble::Bool=false)

Predictions of the best model, `regressor.best_models_[1]`, for `x_data` (one row per
feature, one column per sample), with its fitted constants or gene coefficients applied.
`ensemble` is accepted but not used.
"""
function (regressor::GepRegressor)(x_data::AbstractArray; ensemble::Bool=false)
    return regressor.best_models_[1](x_data)
end



"""
    list_all_functions() -> Dict{Symbol,NamedTuple}

Every function of `FUNCTION_LIB_COMMON` with its arity and unit rules, as a named tuple
`(function_, arity, forward_handler, backward_handler)`.

```julia
list_all_functions()[:sin].arity    # 1
```
"""
function list_all_functions()
    return Dict(sym => (
        function_=FUNCTION_LIB_COMMON[sym],
        arity=ARITY_LIB_COMMON[sym],
        forward_handler=FUNCTION_LIB_FORWARD_COMMON[sym],
        backward_handler=FUNCTION_LIB_BACKWARD_COMMON[sym]
    ) for sym in keys(FUNCTION_LIB_COMMON))
end

"""
    list_all_arity() -> Dict{Symbol,Int8}

A copy of `ARITY_LIB_COMMON`: the arity of every function in the library.
"""
function list_all_arity()
    return Dict(k => v for (k, v) in ARITY_LIB_COMMON)
end

"""
    list_all_forward_handlers() -> Dict{Symbol,Function}

A copy of `FUNCTION_LIB_FORWARD_COMMON`: the forward unit rule of every function.
"""
function list_all_forward_handlers()
    return Dict(k => v for (k, v) in FUNCTION_LIB_FORWARD_COMMON)
end

"""
    list_all_backward_handlers() -> Dict{Symbol,Function}

A copy of `FUNCTION_LIB_BACKWARD_COMMON`: the backward unit rule of every function.
"""
function list_all_backward_handlers()
    return Dict(k => v for (k, v) in FUNCTION_LIB_BACKWARD_COMMON)
end

"""
    list_all_genetic_params() -> Dict{String,AbstractFloat}

A copy of `GENE_COMMON_PROBS`, the rates of the genetic operators.

```julia
list_all_genetic_params()["mutation_prob"]    # 1.0
```
"""
function list_all_genetic_params()
    return Dict(k => v for (k, v) in GENE_COMMON_PROBS)
end

# Setters

"""
    set_function!(sym::Symbol, func::Function)

Replace the function of `sym` in `FUNCTION_LIB_COMMON`, for regressors built afterwards.
Throws an `ArgumentError` when `sym` is not in the library. The batched evaluator knows
only the functions in `TENSOR_NODE_BY_FUNCTION`, so the data method of `fit!` rejects a
regressor built with any other.
"""
function set_function!(sym::Symbol, func::Function)
    haskey(FUNCTION_LIB_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    FUNCTION_LIB_COMMON[sym] = func
    return nothing
end

"""
    set_arity!(sym::Symbol, arity::Int8)

Set the arity of `sym` in `ARITY_LIB_COMMON`, for regressors built afterwards; `arity` is
an `Int8`, such as `Int8(2)`. Throws an `ArgumentError` when `sym` is not in the library or
`arity` is not 1 or 2.
"""
function set_arity!(sym::Symbol, arity::Int8)
    haskey(ARITY_LIB_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    arity in (1, 2) || throw(ArgumentError("Arity must be 1 or 2"))
    ARITY_LIB_COMMON[sym] = arity
    return nothing
end

"""
    set_forward_handler!(sym::Symbol, handler::Function)

Replace the forward unit rule of `sym` in `FUNCTION_LIB_FORWARD_COMMON`, for regressors
built afterwards. Throws an `ArgumentError` when `sym` has no rule there.
"""
function set_forward_handler!(sym::Symbol, handler::Function)
    haskey(FUNCTION_LIB_FORWARD_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    FUNCTION_LIB_FORWARD_COMMON[sym] = handler
    return nothing
end

"""
    set_backward_handler!(sym::Symbol, handler::Function)

Replace the backward unit rule of `sym` in `FUNCTION_LIB_BACKWARD_COMMON`, for regressors
built afterwards. Throws an `ArgumentError` when `sym` has no rule there.
"""
function set_backward_handler!(sym::Symbol, handler::Function)
    haskey(FUNCTION_LIB_BACKWARD_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    FUNCTION_LIB_BACKWARD_COMMON[sym] = handler
    return nothing
end

"""
    update_function!(sym::Symbol; func=nothing, arity=nothing, forward_handler=nothing,
                     backward_handler=nothing)

Apply the given changes to `sym` with `set_function!`, `set_arity!` (`arity` is an
`Int8`), `set_forward_handler!` and `set_backward_handler!`, in that order. Throws an
`ArgumentError` when `sym` is not in `FUNCTION_LIB_COMMON` or a setter rejects its
argument; the changes made before the error are kept.

```julia
# abs of dimensionless operands only
update_function!(:abs; forward_handler=zero_unit_forward,
                 backward_handler=zero_unit_backward)
```
"""
function update_function!(sym::Symbol;
    func::Union{Function,Nothing}=nothing,
    arity::Union{Int8,Nothing}=nothing,
    forward_handler::Union{Function,Nothing}=nothing,
    backward_handler::Union{Function,Nothing}=nothing)
    haskey(FUNCTION_LIB_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))

    if !isnothing(func)
        set_function!(sym, func)
    end
    if !isnothing(arity)
        set_arity!(sym, arity)
    end
    if !isnothing(forward_handler)
        set_forward_handler!(sym, forward_handler)
    end
    if !isnothing(backward_handler)
        set_backward_handler!(sym, backward_handler)
    end

    return nothing
end


end
