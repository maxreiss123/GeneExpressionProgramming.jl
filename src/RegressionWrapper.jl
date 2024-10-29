module RegressionWrapper


export GepRegressor
export create_function_entries, create_feature_entries, create_constants_entries, create_physical_operations
export GENE_COMMON_PROBS, FUNCTION_LIB_BACKWARD_COMMON, FUNCTION_LIB_FORWARD_COMMON, FUNCTION_LIB_COMMON
export fit!

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
    :atanh => zero_unit_forward, :sqrt => sqr_unit_forward, :sign => arbitrary_unit_forward
)

const FUNCTION_LIB_BACKWARD_COMMON = Dict{Symbol,Function}(
    :+ => equal_unit_backward,
    :- => equal_unit_backward,
    :* => mul_unit_backward,
    :/ => div_unit_backward,
    :min => equal_unit_backward,
    :max => equal_unit_backward, :abs => arbitrary_unit_forward,
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
    :atanh => zero_unit_forward, :sqrt => sqr_unit_backward, :sign => arbitrary_unit_forward
)


const GENE_COMMON_PROBS = Dict{String,AbstractFloat}(
    "one_point_cross_over_prob" => 0.4,
    "two_point_cross_over_prob" => 0.3,
    "mutation_prob" => 0.9,
    "mutation_rate" => 0.05,
    "dominant_fusion_prob" => 0.1,
    "dominant_fusion_rate" => 0.1,
    "rezessiv_fusion_prob" => 0.1,
    "rezessiv_fusion_rate" => 0.1,
    "fusion_prob" => 0.0,
    "fusion_rate" => 0.0,
    "inversion_prob" => 0.1,
    "mating_size" => 0.5,
    "penalty_consideration" => 0.0)

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
        head_len::Int=8,
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
- `correction_amount::Real=1.0`: Fraction of population to correct
- `tourni_size::Int=3`: Tournament selection size
- `opt_method_const::Symbol=:cg`: Optimization method for constants
- `target_dimension::Union{Vector{Float16},Nothing}=nothing`: Target physical dimension
"""
function fit!(regressor::GepRegressor, epochs::Int, population_size, x_train::AbstractArray,
    y_train::AbstractArray; x_test::AbstractArray, y_test::AbstractArray,
    optimization_epochs::Int=500,
    hof::Int=3, loss_fun::Union{String,Function}="mse",
    correction_epochs::Int=1, correction_amount::Real=0.3,
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
        x_data_test=x_test,
        y_data_test=y_test,
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



end