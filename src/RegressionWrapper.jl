"""
    RegressionWrapper

A high-level wrapper module for Gene Expression Programming (GEP) regression that provides
simplified interfaces and utilities for symbolic regression tasks with physical dimension support.

# Features
- User-friendly interface for GEP regression via `GepRegressor`
- Support for physical dimensions and dimensional homogeneity
- Extensive mathematical function library
- Customizable gene structure and operations
- Forward and backward dimension propagation
- Ensemble prediction capabilities
- Constant optimization
- Feature and constant dimension handling

# Exports
## Main Components
- `GepRegressor`: Main regression class
- `fit!`: Training function
- `create_function_entries`: Function symbol creation
- `create_feature_entries`: Feature symbol creation
- `create_constants_entries`: Constant symbol creation
- `create_physical_operations`: Physical operation setup

## Constants
- `GENE_COMMON_PROBS`: Default genetic operation probabilities
- `FUNCTION_LIB_BACKWARD_COMMON`: Backward dimension propagation functions
- `FUNCTION_LIB_FORWARD_COMMON`: Forward dimension propagation functions

# Usage Example
```julia
# Create regressor
regressor = GepRegressor(
    2;                              # number of features
    entered_features=[:x1, :x2],    # feature names
    gene_count=3,                   # genes per individual
    head_len=6                      # head length per gene
)

# Train model
fit!(regressor, 
    100,                           # epochs
    50,                            # population size
    X_train, y_train;
    x_test=X_test,
    y_test=y_test,
    loss_fun="mse"
)

# Make predictions
predictions = regressor(X_test)
```


# Dimensional Analysis Example
```julia
using GeneExpressionProgramming

# Load data (features: charge density, charge, vector potential, mass)
data = Matrix(CSV.read("data.txt", DataFrame))
num_features = size(data, 2) - 1

# Define physical dimensions using SI base units [kg, m, s, K, mol, A, cd]
feature_dims = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, -3, 1, 0, 0, 1, 0],   # charge density: A⋅s/m³
    :x2 => Float16[0, 0, 1, 0, 0, 1, 0],    # charge: A⋅s
    :x3 => Float16[1, 1, -2, 0, 0, -1, 0],  # vector potential: kg⋅m/s²⋅A
    :x4 => Float16[1, 0, 0, 0, 0, 0, 0]     # mass: kg
)

# Target dimension (electric conductivity: A/m²)
target_dim = Float16[0, -2, 0, 0, 0, 1, 0]

# Create and train regressor
regressor = GepRegressor(num_features; considered_dimensions=feature_dims)
fit!(regressor, epochs, population_size, x_train', y_train;
     x_test=x_test', y_test=y_test,
     target_dimension=target_dim,
     correction_epochs=5,    # Apply dimension correction every 5 epochs
     correction_amount=0.1)  # Correct 10% of population
```

# Implementation Details
## Type Aliases
- `SymbolDict = OrderedDict{Int8,Int8}`
- `CallbackDict = Dict{Int8,Function}`
- `OrderedCallBackDict = OrderedDict{Int8,Function}`
- `NodeDict = OrderedDict{Int8,Any}`
- `DimensionDict = OrderedDict{Int8,Vector{Float16}}`

## Dependencies
- Internal: GepEntities, LossFunction, GepRegression, SBPUtils, GepUtils
- External: DynamicExpressions, OrderedCollections


See also:
## Core GEP Components
- [`GepRegression.runGep`](@ref): Core GEP algorithm implementation
- [`GepEntities.Chromosome`](@ref): Solution representation
- [`GepEntities.Toolbox`](@ref): Algorithm configuration

## Dimension Handling Functions
### Forward Operations
- [`SBPUtils.equal_unit_forward`](@ref): Dimension equality checking
- [`SBPUtils.mul_unit_forward`](@ref): Dimension multiplication
- [`SBPUtils.div_unit_forward`](@ref): Dimension division
- [`SBPUtils.zero_unit_forward`](@ref): Zero dimension checking
- [`SBPUtils.sqr_unit_forward`](@ref): Dimension squaring
- [`SBPUtils.arbitrary_unit_forward`](@ref): Direct dimension passing

### Backward Operations
- [`SBPUtils.equal_unit_backward`](@ref): Backward equality propagation
- [`SBPUtils.mul_unit_backward`](@ref): Backward multiplication propagation
- [`SBPUtils.div_unit_backward`](@ref): Backward division propagation
- [`SBPUtils.zero_unit_backward`](@ref): Backward zero propagation
- [`SBPUtils.sqr_unit_backward`](@ref): Backward square propagation

"""

module RegressionWrapper


export GepRegressor, GepTensorRegressor
export create_function_entries, create_feature_entries, create_constants_entries, create_physical_operations
export GENE_COMMON_PROBS, FUNCTION_LIB_BACKWARD_COMMON, FUNCTION_LIB_FORWARD_COMMON
export fit!

export list_all_functions, list_all_arity, list_all_forward_handlers,
    list_all_backward_handlers, list_all_genetic_params,
    set_function!, set_arity!, set_forward_handler!, set_backward_handler!,
    update_function!, vec_add, vec_mul


using ..GepEntities
using ..LossFunction
using ..EvoSelection

using ..GepRegression
using ..SBPUtils
using ..GepUtils
using ..TensorRegUtils
using DynamicExpressions
using OrderedCollections
using LinearAlgebra
using StatsBase


"""
    FUNCTION_LIB_FORWARD_COMMON::Dict{Symbol,Function}

Dictionary mapping functions to their forward unit handling implementations.
Defines how units are propagated through operations in forward mode.

Available unit handlers:
- `equal_unit_forward`: For operations requiring equal units
- `mul_unit_forward`: For multiplication operations
- `div_unit_forward`: For division operations
- `arbitrary_unit_forward`: For operations that work with any unit
- `zero_unit_forward`: For operations requiring dimensionless input
- `sqr_unit_forward`: For square root operations

When adding new functions, define appropriate unit handling behavior here.
"""
const FUNCTION_LIB_FORWARD_COMMON = Dict{Symbol,Function}(
    :+ => equal_unit_forward,
    :- => equal_unit_forward,
    :* => mul_unit_forward,
    :/ => div_unit_forward,
    :min => equal_unit_forward,
    :max => equal_unit_forward, :abs => arbitrary_unit_forward,
    :floor => arbitrary_unit_forward,
    :ceil => arbitrary_unit_forward,
    :round => arbitrary_unit_forward, :exp => zero_unit_forward,
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
    :atanh => zero_unit_forward, :sqrt => sqr_unit_backward, :sign => arbitrary_unit_forward
)

"""
    FUNCTION_LIB_BACKWARD_COMMON::Dict{Symbol,Function}

Dictionary mapping functions to their backward unit handling implementations.
Defines how units are propagated through operations in backward mode.

Available unit handlers:
- `equal_unit_backward`: For operations requiring equal units
- `mul_unit_backward`: For multiplication operations
- `div_unit_backward`: For division operations
- `arbitrary_unit_forward`: For operations that work with any unit
- `zero_unit_backward`: For operations requiring dimensionless input
- `sqr_unit_backward`: For square root operations

When adding new functions, define appropriate unit handling behavior here.
"""
const FUNCTION_LIB_BACKWARD_COMMON = Dict{Symbol,Function}(
    :+ => equal_unit_backward,
    :- => equal_unit_backward,
    :* => mul_unit_backward,
    :/ => div_unit_backward,
    :min => equal_unit_backward,
    :max => equal_unit_backward, :abs => arbitrary_unit_forward,
    :floor => arbitrary_unit_forward,
    :ceil => arbitrary_unit_forward,
    :round => arbitrary_unit_forward, :exp => zero_unit_backward,
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
    :atanh => zero_unit_backward, :sqrt => sqr_unit_forward, :sign => arbitrary_unit_forward
)



"""
    GENE_COMMON_PROBS::Dict{String,AbstractFloat}

Dictionary containing default probabilities and parameters for genetic algorithm operations.

# Parameters
- `one_point_cross_over_prob`: Probability of single-point crossover (0.4)
- `two_point_cross_over_prob`: Probability of two-point crossover (0.3)
- `mutation_prob`: Probability of mutation occurring (0.9)
- `mutation_rate`: Rate of mutation when it occurs (0.05)
- `dominant_fusion_prob`: Probability of dominant fusion (0.1)
- `dominant_fusion_rate`: Rate of dominant fusion (0.1)
- `rezessiv_fusion_prob`: Probability of recessive fusion (0.1)
- `rezessiv_fusion_rate`: Rate of recessive fusion (0.1)
- `fusion_prob`: Probability of general fusion (0.0)
- `fusion_rate`: Rate of general fusion (0.0)
- `inversion_prob`: Probability of inversion (0.1)
- `gene_averaging_prob`: Probability for a chromosome to be considered for averaging (1.0)
- `gene_averaging_rate`: Rate of positions in the chromosome for beeing averaged
- `mating_size`: Relative size of mating pool (0.5)

These values can be adjusted to fine-tune the genetic algorithm's behavior.
"""
const GENE_COMMON_PROBS = Dict{String,AbstractFloat}(
    "one_point_cross_over_prob" => 0.5,
    "two_point_cross_over_prob" => 0.4,
    "mutation_prob" => 1.0,
    "mutation_rate" => 0.1,
    "dominant_fusion_prob" => 0.0,
    "dominant_fusion_rate" => 0.1,
    "rezessiv_fusion_prob" => 0.0,
    "rezessiv_fusion_rate" => 0.1,
    "fusion_prob" => 0.0,
    "fusion_rate" => 0.0,
    "inversion_prob" => 0.0,
    "reverse_insertion" => 0.1,
    "reverse_insertion_tail" => 0.0,
    "gene_transposition" => 0.0,
    "gene_averaging_prob" => 0.0,
    "gene_averaging_rate" => 0.05,
    "mating_size" => 0.7)

const SymbolDict = OrderedDict{Int8,Int8}
const CallbackDict = Dict{Int8,Function}
const OrderedCallBackDict = OrderedDict{Int8,Function}
const NodeDict = OrderedDict{Int8,Any}
const DimensionDict = OrderedDict{Int8,Vector{Float16}}


function create_physical_operations(entered_non_terminals::Vector{Symbol})
    forward_funs = OrderedCallBackDict()
    backward_funs = CallbackDict()
    point_ops = Int8[]

    for (idx, elem) in enumerate(entered_non_terminals)
        if !haskey(FUNCTION_LIB_COMMON, elem)
            @info "Symbol: " elem " is ignored"
            continue
        end
        forward_funs[idx] = FUNCTION_LIB_FORWARD_COMMON[elem]
        backward_funs[idx] = FUNCTION_LIB_BACKWARD_COMMON[elem]
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
        nodes[cur_idx] = Node{node_type}(feature=idx)
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
    start_idx::Int8
)::Tuple{SymbolDict,NodeDict,DimensionDict,Int8}

    utilized_symbols = SymbolDict()
    nodes = NodeDict()
    dimension_information = DimensionDict()
    cur_idx = start_idx


    for elem in entered_terminal_nums
        utilized_symbols[cur_idx] = 0
        nodes[cur_idx] = Node{node_type}(; val=parse(node_type, string(elem)))
        dimension_information[cur_idx] = get(dimensions_to_consider, elem, ZERO_DIM)
        cur_idx += 1
    end


    for _ in 1:rnd_count
        utilized_symbols[cur_idx] = 0
        nodes[cur_idx] = Node{node_type}(; val=rand())
        dimension_information[cur_idx] = ZERO_DIM
        cur_idx += 1
    end

    return utilized_symbols, nodes, dimension_information, cur_idx
end


#preamble syms are just dummies
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
        nodes[cur_idx] = Node{node_type}(feature=cur_idx)
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
    GepRegressor(feature_amount::Int; kwargs...)

Create a Gene Expression Programming regressor for symbolic regression.

# Arguments
- `feature_amount::Int`: Number of input features

# Keyword Arguments
- `entered_features::Vector{Symbol}=[]`: Custom feature names. Defaults to `[x1, x2, ...]`
- `entered_non_terminals::Vector{Symbol}=[:+, :-, :*, :/]`: Available operations
- `entered_terminal_nums::Vector{Symbol}=[Symbol(0.0), Symbol(0.5)]`: Constant terms
- `gene_connections::Vector{Symbol}=[:+, :-, :*, :/]`: Operations for connecting genes
- `considered_dimensions::Dict{Symbol,Vector{Float16}}=Dict()`: Physical dimensions for features/constants
- `rnd_count::Int=1`: Number of random constants
- `node_type::Type=Float64`: Data type for calculations
- `gene_count::Int=3`: Number of genes
- `head_len::Int=6`: Length of head section in genes
- `preamble_syms::Vector{Symbol}=Symbol[]`: Preamble symbols
- `max_permutations_lib::Int=10000`: Maximum permutations for dimension library
- `rounds::Int=4`: Rounds for dimension library creation
- `number_of_objectives::Int=1`: Defines the number of objectives considered by the search
- `head_weigths=nothing`: Defines the weights for the different function - ∑(head_weigths)==1
- `tail_weigths=[0.6,0.2,0.2]`: Defines the weights for the different utilized symbols - ∑(tail-weights)==1
"""
mutable struct GepRegressor
    toolbox_::Toolbox
    operators_::OperatorEnum
    dimension_information_::OrderedDict{Int8,Vector{Float16}}
    best_models_::Union{Nothing,Vector{Chromosome}}
    fitness_history_::Any
    token_dto_::Union{TokenDto,Nothing}


    function GepRegressor(feature_amount::Int;
        entered_features::Vector{Symbol}=Vector{Symbol}(),
        entered_non_terminals::Vector{Symbol}=[:+, :-, :*, :/],
        entered_terminal_nums::Vector{Symbol}=[Symbol(0.0), Symbol(0.5)],
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
        tail_count = feature_amount + rnd_count + length(entered_terminal_nums)
        tail_weigths_ = [tail_weigths[1]/tail_count for _ in 1:feature_amount]
        append!(tail_weigths_, fill(tail_weigths[2]/tail_count, length(entered_terminal_nums)))
        append!(tail_weigths_, fill(tail_weigths[3]/tail_count, rnd_count))

        entered_features_ = isempty(entered_features) ?
                            [Symbol("x$i") for i in 1:feature_amount] : entered_features

        func_syms, callbacks, binary_ops, unary_ops, gene_connections_, cur_idx = create_function_entries(
            entered_non_terminals, gene_connections
        )

        feat_syms, feat_nodes, feat_dims, cur_idx = create_feature_entries(
            entered_features_, considered_dimensions, node_type, cur_idx
        )

        const_syms, const_nodes, const_dims, cur_idx = create_constants_entries(
            entered_terminal_nums, rnd_count, considered_dimensions, node_type, cur_idx
        )

        pre_syms, pre_nodes, pre_dims, preamble_syms_, cur_idx = create_preamble_entries(
            preamble_syms, considered_dimensions, node_type, cur_idx
        )


        utilized_symbols = merge_collections(func_syms, feat_syms, const_syms, pre_syms)
        
        nodes = merge!(NodeDict(), feat_nodes, const_nodes, pre_nodes)
        
        dimension_information = merge!(DimensionDict(), feat_dims, const_dims, pre_dims)
        
        operators = OperatorEnum(binary_operators=binary_ops, unary_operators=unary_ops)

        if !isempty(considered_dimensions)
            forward_funs, backward_funs, point_ops = create_physical_operations(entered_non_terminals)
            token_lib = TokenLib(
                dimension_information,
                forward_funs,
                utilized_symbols
            )
            idx_features = [idx for (idx, _) in feat_syms]
            idx_funs = [idx for (idx, _) in func_syms]
            idx_const = [idx for (idx, _) in const_syms]

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
            operators_=operators, tail_weights_=weights(tail_weigths_))

        obj = new()
        obj.toolbox_ = toolbox
        obj.operators_ = operators
        obj.dimension_information_ = dimension_information
        obj.token_dto_ = token_dto
        return obj
    end
end

"""
    GepTensorRegressor

A Gene Expression Programming (GEP) regressor that evolves higher order mathematical expressions (e.g tensor-based). 

# Fields
- `toolbox_::Toolbox`: Contains configuration and operators for GEP evolution
- `best_models_::Union{Nothing,Vector{Chromosome}}`: Best models found during evolution
- `fitness_history_::Any`: History of fitness values during training

# Constructor
    GepTensorRegressor(feature_amount::Int; kwargs...)

Create a new GEP tensor regressor with specified number of input features.

# Arguments
- `scalar_feature_amount::Int`: Number of input features representing scalar quantities
- `higher_dim_feature_amount::Int`: Number of input features representing hihger quantities

# Keyword Arguments 
- `entered_non_terminals::Vector{Symbol}=[:+, :-, :*, :/]`: Available mathematical operators
- `entered_terminal_nums::Vector{<:AbstractFloat}=[0.0, 0.5]`: Constants available as terminals
- `gene_connections::Vector{Symbol}=[:+, :-, :*, :/]`: Operators for connecting genes
- `rnd_count::Int=1`: Number of random constant terminals to generate
- `gene_count::Int=3`: Number of genes in each chromosome 
- `head_len::Int=6`: Length of the head section in each gene
- `number_of_objectives::Int=1`: Number of optimization objectives

The regressor uses GEP to evolve tensor-based mathematical expressions that map input features 
to target values. It supports multiple genes connected by operators and can optimize for 
multiple objectives.

# Implementation Details
- Uses InputSelector nodes for features
- Combines fixed and random constant terminals
- Maps operators to TENSOR_NODES callbacks
- Uses TENSOR_NODES_ARITY for operator arity
- Compiles expressions to Flux networks via compile_to_flux_network
"""
#TODO => adapt probs for occurance of !
mutable struct GepTensorRegressor
    toolbox_::Toolbox
    best_models_::Union{Nothing,Vector{Chromosome}}
    fitness_history_::Any


    function GepTensorRegressor(scalar_feature_amount::Int;
        higher_dim_feature_amount::Int=0,
        entered_non_terminals::Vector{Symbol}=[:+, :-, :*, :/],
        entered_terminal_nums::Vector{<:AbstractFloat}=Float64[],
        gene_connections::Vector{Symbol}=[:+, :*],
        rnd_count::Int=0,
        gene_count::Int=2,
        head_len::Int=3,
        number_of_objectives::Int=1,
        feature_names::Vector{String}=String[]
    )
        #Creating the feature Nodes -> asuming a data dict pointing to 
        cur_idx = Int8(1)
        nodes = OrderedDict{Int8,Any}()
        utilized_symbols = SymbolDict()
        callbacks = Dict{Int8,Any}()
        gene_connections_ = Int8[]
        tensor_syms_idx = Int8[]
        tensor_function_idx = Int8[]
        for _ in 1:scalar_feature_amount
            feature_name = isempty(feature_names) ? "x$cur_idx" : feature_names[cur_idx]
            nodes[cur_idx] = InputSelector(cur_idx, feature_name)
            utilized_symbols[cur_idx] = Int8(0)
            cur_idx += 1
        end

        for _ in 1:higher_dim_feature_amount
            nodes[cur_idx] = InputSelector(cur_idx)
            utilized_symbols[cur_idx] = Int8(0)
            cur_idx += 1
        end

        #Creating the const_nodes
        for elem in entered_terminal_nums
            nodes[cur_idx] = elem
            utilized_symbols[cur_idx] = Int8(0)
            cur_idx += 1
        end

        for _ in 1:rnd_count
            nodes[cur_idx] = rand()
            utilized_symbols[cur_idx] = Int8(0)
            cur_idx += 1
        end

        #callback - index => function
        for elem in entered_non_terminals
            @show elem
            callbacks[cur_idx] = TENSOR_NODES[elem]
            utilized_symbols[cur_idx] = TENSOR_NODES_ARITY[elem]
            if elem in gene_connections
                push!(gene_connections_, cur_idx)
            end
            cur_idx += 1
        end

        toolbox = Toolbox(gene_count, head_len, utilized_symbols, gene_connections_,
            callbacks, nodes, GENE_COMMON_PROBS; number_of_objectives=number_of_objectives,
            operators_=nothing, function_complile=compile_to_flux_network)

        obj = new()
        obj.toolbox_ = toolbox
        return obj
    end
end


"""
    fit!(regressor::GepRegressor, epochs::Int, population_size::Int, x_train::AbstractArray, 
         y_train::AbstractArray; kwargs...)

Train the GEP regressor model.

# Arguments
- `regressor::GepRegressor`: The regressor instance
- `epochs::Int`: Number of evolutionary generations
- `population_size::Int`: Size of the population
- `x_train::AbstractArray`: Training features
- `y_train::AbstractArray`: Training targets

# Keyword Arguments
- `x_test::AbstractArray`: Test features
- `y_test::AbstractArray`: Test targets
- `optimization_epochs::Int=500`: Number of epochs for constant optimization
- `hof::Int=3`: Number of best models to keep
- `loss_fun::Union{String,Function}="mse"`: Loss function ("mse", "mae", or custom function)
- `correction_epochs::Int=1`: Epochs between dimension corrections
- `correction_amount::Real=1.0`: Fraction of population to correct for the dimensioal homogeneity
- `tourni_size::Int=3`: Tournament selection size
- `opt_method_const::Symbol=:cg`: Optimization method for constants
- `target_dimension::Union{Vector{Float16},Nothing}=nothing`: Target physical dimension
"""
function fit!(regressor::GepRegressor, epochs::Int, population_size::Int, x_train::AbstractArray,
    y_train::AbstractArray; x_test::Union{AbstractArray,Nothing}=nothing, y_test::Union{AbstractArray,Nothing}=nothing,
    optimization_epochs::Int=100,
    hof::Int=3, loss_fun::Union{String,Function}="mse",
    correction_epochs::Int=1, correction_amount::Real=0.05,
    opt_method_const::Symbol=:cg,
    target_dimension::Union{Vector{Float16},Nothing}=nothing,
    cycles::Int=10, max_iterations::Int=1000, n_starts::Int=3,
    break_condition::Union{Function,Nothing}=nothing,
    file_logger_callback::Union{Function,Nothing}=nothing,
    save_state_callback::Union{Function,Nothing}=nothing,
    load_state_callback::Union{Function,Nothing}=nothing, 
    population_sampling_multiplier::Int=1
)

    correction_callback = if !isnothing(target_dimension)
        (genes, start_indices, expression) -> correct_genes!(
            genes,
            start_indices,
            expression,
            target_dimension,
            regressor.token_dto_;
            cycles=cycles
        )
    else
        nothing
    end

    @inline function optimizer_function_(sub_tree::Node)
        y_pred, flag = eval_tree_array(sub_tree, x_train, regressor.operators_)
        return get_loss_function("mse")(y_pred, y_train)
    end

    function optimizer_wrapper(population::Vector{Chromosome})
        try
            eqn, result = optimize_constants!(population[1].compiled_function, optimizer_function_;
                opt_method=opt_method_const, max_iterations=max_iterations, n_restarts=n_starts)
            population[1].fitness = (result,)
            population[1].compiled_function = eqn
        catch e
            @show "Ignored constant opt." e
        end
    end

    evalStrat = StandardRegressionStrategy{typeof(first(x_train))}(
        regressor.operators_,
        x_train,
        y_train,
        !isnothing(x_test) ? x_test : x_train,
        !isnothing(y_test) ? y_test : y_train,
        get_loss_function(loss_fun);
        secOptimizer=optimizer_wrapper,
        break_condition=break_condition
    )

    best, history = runGep(epochs,
        population_size,
        regressor.toolbox_,
        evalStrat;
        hof=hof,
        correction_callback=correction_callback,
        correction_epochs=correction_epochs,
        correction_amount=correction_amount,
        tourni_size=max(Int(ceil(population_size * 0.03)), 3),
        optimization_epochs=optimization_epochs,
        file_logger_callback=file_logger_callback,
        save_state_callback=save_state_callback,
        load_state_callback=load_state_callback,
        population_sampling_multiplier=population_sampling_multiplier
    )

    regressor.best_models_ = best
    regressor.fitness_history_ = history
end

function fit!(regressor::GepRegressor, epochs::Int, population_size::Int, loss_function::Function;
    optimizer_function_::Union{Function,Nothing}=nothing,
    optimization_epochs::Int=100,
    hof::Int=3,
    correction_epochs::Int=1,
    correction_amount::Real=0.3,
    opt_method_const::Symbol=:nd,
    target_dimension::Union{Vector{Float16},Nothing}=nothing,
    cycles::Int=10, max_iterations::Int=150, n_starts::Int=5,
    break_condition::Union{Function,Nothing}=nothing,
    file_logger_callback::Union{Function,Nothing}=nothing,
    save_state_callback::Union{Function,Nothing}=nothing,
    load_state_callback::Union{Function,Nothing}=nothing
)

    correction_callback = if !isnothing(target_dimension)
        (genes, start_indices, expression) -> correct_genes!(
            genes,
            start_indices,
            expression,
            target_dimension,
            regressor.token_dto_;
            cycles=cycles
        )
    else
        nothing
    end


    function optimizer_wrapper(population::Vector{Chromosome})
        try
            eqn, result = optimize_constants!(population[1].compiled_function, optimizer_function_;
                opt_method=opt_method_const, max_iterations=max_iterations, n_restarts=n_starts)
            population[1].fitness = result
            population[1].compiled_function = eqn
        catch e
            @show "Ignored constant opt."
        end
    end

    evalStrat = GenericRegressionStrategy(
        regressor.operators_,
        length(regressor.toolbox_.fitness_reset[1]),
        loss_function;
        secOptimizer=nothing,
        break_condition=break_condition
    )

    best, history = runGep(epochs,
        population_size,
        regressor.toolbox_,
        evalStrat;
        hof=hof,
        correction_callback=correction_callback,
        correction_epochs=correction_epochs,
        correction_amount=correction_amount,
        tourni_size=max(Int(ceil(population_size * 0.03)), 3),
        optimization_epochs=optimization_epochs,
        file_logger_callback=file_logger_callback,
        save_state_callback=save_state_callback,
        load_state_callback=load_state_callback
    )

    regressor.best_models_ = best
    regressor.fitness_history_ = history
end

function fit!(regressor::GepTensorRegressor, epochs::Int, population_size::Int, loss_function::Function;
    hof::Int=3,
    break_condition::Union{Function,Nothing}=nothing,
    file_logger_callback::Union{Function, Nothing}=nothing, 
    save_state_callback::Union{Function, Nothing}=nothing,
    load_state_callback::Union{Function, Nothing}=nothing
)

    evalStrat = GenericRegressionStrategy(
        nothing,
        length(regressor.toolbox_.fitness_reset[1]),
        loss_function;
        secOptimizer=nothing,
        break_condition=break_condition
    )

    best, history = runGep(epochs,
        population_size,
        regressor.toolbox_,
        evalStrat;
        hof=hof,
        tourni_size=max(Int(ceil(population_size * 0.003)), 3),
        file_logger_callback=file_logger_callback,
        save_state_callback=save_state_callback,
        load_state_callback=load_state_callback
    )

    regressor.best_models_ = best
    regressor.fitness_history_ = history
end

"""
    (regressor::GepRegressor)(x_data::AbstractArray; ensemble::Bool=false)

Make predictions using the trained regressor.

# Arguments
- `x_data::AbstractArray`: Input features

# Keyword Arguments
- `ensemble::Bool=false`: Whether to use ensemble predictions

# Returns
- Predicted values for the input features -> only employable when using compile_djl_datatype
"""
function (regressor::GepRegressor)(x_data::AbstractArray; ensemble::Bool=false)
    return regressor.best_models_[1].compiled_function(x_data, regressor.operators_)
end



"""
    list_all_functions() -> Dict{Symbol, NamedTuple}

List all functions in the library with their complete information including arity and handlers.

# Returns
- Dictionary mapping function symbols to NamedTuples containing:
  - `function`: The actual function
  - `arity`: Number of arguments
  - `forward_handler`: Forward unit handling function
  - `backward_handler`: Backward unit handling function

# Examples
```julia
funcs = list_all_functions()
sin_info = funcs[:sin]
println(sin_info.arity)  # 1
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
    list_all_arity() -> Dict{Symbol, Int8}

List all functions and their arities.

# Returns
- Dictionary mapping function symbols to their arity values

# Examples
```julia
arities = list_all_arity()
println(arities[:+])  # 2
```
"""
function list_all_arity()
    return Dict(k => v for (k, v) in _ARITY_LIB_COMMON)
end

"""
    list_all_forward_handlers() -> Dict{Symbol, Function}

List all functions and their forward unit handlers.

# Returns
- Dictionary mapping function symbols to their forward unit handling functions

# Examples
```julia
handlers = list_all_forward_handlers()
sin_handler = handlers[:sin]
```
"""
function list_all_forward_handlers()
    return Dict(k => v for (k, v) in _FUNCTION_LIB_FORWARD_COMMON)
end

"""
    list_all_backward_handlers() -> Dict{Symbol, Function}

List all functions and their backward unit handlers.

# Returns
- Dictionary mapping function symbols to their backward unit handling functions

# Examples
```julia
handlers = list_all_backward_handlers()
sin_handler = handlers[:sin]
```
"""
function list_all_backward_handlers()
    return Dict(k => v for (k, v) in _FUNCTION_LIB_BACKWARD_COMMON)
end

"""
    list_all_genetic_params() -> Dict{String, AbstractFloat}

List all genetic algorithm parameters and their current values.

# Returns
- Dictionary mapping parameter names to their current values

# Examples
```julia
params = list_all_genetic_params()
println(params["mutation_prob"])  # 0.9
```
"""
function list_all_genetic_params()
    return Dict(k => v for (k, v) in _GENE_COMMON_PROBS)
end

# Setters

"""
    set_function!(sym::Symbol, func::Function)

Set or update a function in the library. Requires the function to already exist in the library.

# Arguments
- `sym::Symbol`: Symbol representing the function
- `func::Function`: New function implementation

# Examples
```julia
set_function!(:sin, new_sin_implementation)
```

# Throws
- ArgumentError if the function symbol doesn't exist in the library
"""
function set_function!(sym::Symbol, func::Function)
    haskey(_UNCTION_LIB_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    FUNCTION_LIB_COMMON[sym] = func
    return nothing
end

"""
    set_arity!(sym::Symbol, arity::Int8)

Set or update the arity for a function. Requires the function to already exist in the library.

# Arguments
- `sym::Symbol`: Symbol representing the function
- `arity::Int8`: New arity value (must be 1 or 2)

# Examples
```julia
set_arity!(:custom_func, 2)
```

# Throws
- ArgumentError if the function symbol doesn't exist or arity is invalid
"""
function set_arity!(sym::Symbol, arity::Int8)
    haskey(ARITY_LIB_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    arity in (1, 2) || throw(ArgumentError("Arity must be 1 or 2"))
    ARITY_LIB_COMMON[sym] = arity
    return nothing
end

"""
    set_forward_handler!(sym::Symbol, handler::Function)

Set or update the forward unit handler for a function.

# Arguments
- `sym::Symbol`: Symbol representing the function
- `handler::Function`: New forward unit handling function

# Examples
```julia
set_forward_handler!(:custom_func, zero_unit_forward)
```

# Throws
- ArgumentError if the function symbol doesn't exist
"""
function set_forward_handler!(sym::Symbol, handler::Function)
    haskey(FUNCTION_LIB_FORWARD_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    FUNCTION_LIB_FORWARD_COMMON[sym] = handler
    return nothing
end

"""
    set_backward_handler!(sym::Symbol, handler::Function)

Set or update the backward unit handler for a function.

# Arguments
- `sym::Symbol`: Symbol representing the function
- `handler::Function`: New backward unit handling function

# Examples
```julia
set_backward_handler!(:custom_func, zero_unit_backward)
```

# Throws
- ArgumentError if the function symbol doesn't exist
"""
function set_backward_handler!(sym::Symbol, handler::Function)
    haskey(FUNCTION_LIB_BACKWARD_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    FUNCTION_LIB_BACKWARD_COMMON[sym] = handler
    return nothing
end

"""
    update_function!(sym::Symbol; 
                    func::Union{Function,Nothing}=nothing,
                    arity::Union{Int8,Nothing}=nothing,
                    forward_handler::Union{Function,Nothing}=nothing,
                    backward_handler::Union{Function,Nothing}=nothing)

Update multiple aspects of a function at once.

# Arguments
- `sym::Symbol`: Symbol representing the function
- `func::Function`: (optional) New function implementation
- `arity::Int8`: (optional) New arity value
- `forward_handler::Function`: (optional) New forward unit handler
- `backward_handler::Function`: (optional) New backward unit handler

# Examples
```julia
update_function!(:custom_func, 
                func=new_implementation,
                arity=2,
                forward_handler=new_forward_handler)
```

# Throws
- ArgumentError if the function symbol doesn't exist or parameters are invalid
"""
function update_function!(sym::Symbol;
    func::Union{Function,Nothing}=nothing,
    arity::Union{Int8,Nothing}=nothing,
    forward_handler::Union{Function,Nothing}=nothing,
    backward_handler::Union{Function,Nothing}=nothing)
    haskey(_FUNCTION_LIB_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))

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
