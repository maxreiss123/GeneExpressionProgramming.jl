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
- `FUNCTION_LIB_COMMON`: Available mathematical functions

# Function Library
Includes extensive mathematical operations:
- Basic arithmetic: +, -, *, /, ^
- Comparisons: min, max
- Rounding: floor, ceil, round
- Exponential: exp, log, log10, log2
- Trigonometric: sin, cos, tan, asin, acos, atan
- Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
- Special: sqr, sqrt, sign, abs

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


export GepRegressor
export create_function_entries, create_feature_entries, create_constants_entries, create_physical_operations
export GENE_COMMON_PROBS, FUNCTION_LIB_BACKWARD_COMMON, FUNCTION_LIB_FORWARD_COMMON, FUNCTION_LIB_COMMON
export fit!

export list_all_functions, list_all_arity, list_all_forward_handlers, 
       list_all_backward_handlers, list_all_genetic_params,
       set_function!, set_arity!, set_forward_handler!, set_backward_handler!,
       update_function!

include("Entities.jl")
include("Gep.jl")
include("Losses.jl")
include("PhyConstants.jl")
include("Sbp.jl")
include("Selection.jl")
include("Util.jl")


using .GepEntities
using .LossFunction
using .GepEntities
using .EvoSelection

using .GepRegression
using .SBPUtils
using .GepUtils
using DynamicExpressions
using OrderedCollections

const Toolbox = GepRegression.GepEntities.Toolbox
const TokenDto = SBPUtils.TokenDto

function sqr(x::Vector{T}) where {T<:AbstractFloat}
    return x .* x
end

function sqr(x::T) where {T<:Union{AbstractFloat,Node{<:AbstractFloat}}}
    return x * x
end


"""
    FUNCTION_LIB_COMMON::Dict{Symbol,Function}

Dictionary mapping function symbols to their corresponding functions.
Contains basic mathematical operations, trigonometric, and other common functions.

# Available Functions
- Basic arithmetic: `+`, `-`, `*`, `/`, `^`
- Comparison: `min`, `max`
- Rounding: `floor`, `ceil`, `round`
- Exponential & Logarithmic: `exp`, `log`, `log10`, `log2`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Other: `abs`, `sqr`, `sqrt`, `sign`

To add a new function, ensure you also add corresponding entries in `ARITY_LIB_COMMON`,
`FUNCTION_LIB_FORWARD_COMMON`, and `FUNCTION_LIB_BACKWARD_COMMON`.
"""
const FUNCTION_LIB_COMMON = Dict{Symbol,Function}(
    :+ => +,
    :- => -,
    :* => *,
    :/ => /,
    :^ => ^,
    :min => min,
    :max => max, :abs => abs,
    :floor => floor,
    :ceil => ceil,
    :round => round, :exp => exp,
    :log => log,
    :log10 => log10,
    :log2 => log2, :sin => sin,
    :cos => cos,
    :tan => tan,
    :asin => asin,
    :acos => acos,
    :atan => atan, :sinh => sinh,
    :cosh => cosh,
    :tanh => tanh,
    :asinh => asinh,
    :acosh => acosh,
    :atanh => atanh, :sqr => sqr,
    :sqrt => sqrt, :sign => sign
)

"""
    ARITY_LIB_COMMON::Dict{Symbol,Int8}

Dictionary specifying the number of arguments (arity) for each function in the library.
- Value of 1 indicates unary functions (e.g., `sin`, `cos`, `abs`)
- Value of 2 indicates binary functions (e.g., `+`, `-`, `*`, `/`)

When adding new functions to `FUNCTION_LIB_COMMON`, ensure to specify their arity here.
"""
const ARITY_LIB_COMMON = Dict{Symbol,Int8}(
    :+ => 2,
    :- => 2,
    :* => 2,
    :/ => 2,
    :^ => 2,
    :min => 2,
    :max => 2, :abs => 1,
    :floor => 1,
    :ceil => 1,
    :round => 1,
    :exp => 1,
    :log => 1,
    :log10 => 1,
    :log2 => 1,
    :sin => 1,
    :cos => 1,
    :tan => 1,
    :asin => 1,
    :acos => 1,
    :atan => 1,
    :sinh => 1,
    :cosh => 1,
    :tanh => 1,
    :asinh => 1,
    :acosh => 1,
    :atanh => 1,
    :sqrt => 1,
    :sqr => 1
)

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
- `mating_size`: Relative size of mating pool (0.5)
- `penalty_consideration`: Weight of penalty in fitness evaluation (0.2)

These values can be adjusted to fine-tune the genetic algorithm's behavior.
"""
const GENE_COMMON_PROBS = Dict{String,AbstractFloat}(
    "one_point_cross_over_prob" => 0.4,
    "two_point_cross_over_prob" => 0.3,
    "mutation_prob" => 0.9,
    "mutation_rate" => 0.05,
    "dominant_fusion_prob" => 0.1,
    "dominant_fusion_rate" => 0.2,
    "rezessiv_fusion_prob" => 0.1,
    "rezessiv_fusion_rate" => 0.2,
    "fusion_prob" => 0.1,
    "fusion_rate" => 0.2,
    "inversion_prob" => 0.1,
    "reverse_insertion" => 0.05,
    "reverse_insertion_tail" => 0.05,
    "mating_size" => 0.5,
    "penalty_consideration" => 0.2)

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
        if elem == :mul || elem == :/
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
        nodes[cur_idx] = parse(node_type, string(elem))
        dimension_information[cur_idx] = get(dimensions_to_consider, elem, ZERO_DIM)
        cur_idx += 1
    end


    for _ in 1:rnd_count
        utilized_symbols[cur_idx] = 0
        nodes[cur_idx] = rand()
        dimension_information[cur_idx] = ZERO_DIM
        cur_idx += 1
    end

    return utilized_symbols, nodes, dimension_information, cur_idx
end


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
        nodes[cur_idx] = Node{AbstractArray}(feature=cur_idx)
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
"""
mutable struct GepRegressor
    toolbox_::Toolbox
    operators_::OperatorEnum
    dimension_information_::OrderedDict{Int8,Vector{Float16}}
    best_models_::Union{Nothing,Vector{GepRegression.GepEntities.Chromosome}}
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
        gene_count::Int=2,
        head_len::Int=10,
        preamble_syms::Vector{Symbol}=Symbol[],
        max_permutations_lib::Int=10000, rounds::Int=4
    )

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

        toolbox = GepRegression.GepEntities.Toolbox(gene_count, head_len, utilized_symbols, gene_connections_,
            callbacks, nodes, GENE_COMMON_PROBS; preamble_syms=preamble_syms_)

        obj = new()
        obj.toolbox_ = toolbox
        obj.operators_ = operators
        obj.dimension_information_ = dimension_information
        obj.token_dto_ = token_dto
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
function fit!(regressor::GepRegressor, epochs::Int, population_size, x_train::AbstractArray,
    y_train::AbstractArray; x_test::Union{AbstractArray,Nothing}=nothing, y_test::Union{AbstractArray,Nothing}=nothing,
    optimization_epochs::Int=500,
    hof::Int=3, loss_fun::Union{String,Function}="mse",
    correction_epochs::Int=1, correction_amount::Real=0.05,
    tourni_size::Int=3, opt_method_const::Symbol=:cg,
    target_dimension::Union{Vector{Float16},Nothing}=nothing,
    cycles::Int=10
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

    
    best, history = runGep(epochs,
        population_size,
        regressor.operators_,
        x_train,
        y_train,
        regressor.toolbox_;
        hof=hof,
        x_data_test=!isnothing(x_test) ? x_test : x_train,
        y_data_test=!isnothing(y_test) ? y_test : y_train,
        loss_fun_=loss_fun,
        correction_callback=correction_callback,
        correction_epochs=correction_epochs,
        correction_amount=correction_amount,
        tourni_size=tourni_size,
        opt_method_const=opt_method_const,
        optimisation_epochs=optimization_epochs)

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
- Predicted values for the input features
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
        function_ = _FUNCTION_LIB_COMMON[sym],
        arity = _ARITY_LIB_COMMON[sym],
        forward_handler = _FUNCTION_LIB_FORWARD_COMMON[sym],
        backward_handler = _FUNCTION_LIB_BACKWARD_COMMON[sym]
    ) for sym in keys(_FUNCTION_LIB_COMMON))
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
    haskey(_FUNCTION_LIB_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    _FUNCTION_LIB_COMMON[sym] = func
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
    haskey(_ARITY_LIB_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    arity in (1, 2) || throw(ArgumentError("Arity must be 1 or 2"))
    _ARITY_LIB_COMMON[sym] = arity
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
    haskey(_FUNCTION_LIB_FORWARD_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    _FUNCTION_LIB_FORWARD_COMMON[sym] = handler
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
    haskey(_FUNCTION_LIB_BACKWARD_COMMON, sym) || throw(ArgumentError("Function $sym not found in library"))
    _FUNCTION_LIB_BACKWARD_COMMON[sym] = handler
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
