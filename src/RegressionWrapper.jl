module RegressionWrapper


include("Entities.jl")
include("Gep.jl")
include("Losses.jl")
include("PhyConstants.jl")
include("Sbp.jl")
include("Selection.jl")
include("Util.jl")


using .GepEntities
using .LossFunction
using .GepUtils
using .EvoSelection
using .GepRegression
using .SBPUtils
using DynamicExpressions
using OrderedCollections


export GepRegressor
export create_function_entries, create_feature_entries, create_constants_entries, create_physical_operations

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
    "one_point_cross_over_prob" => 0.5,
    "two_point_cross_over_prob" => 0.5,
    "mutation_prob" => 0.9,
    "mutation_rate" => 0.1,
    "dominant_fusion_prob" => 0.1,
    "dominant_fusion_rate" => 0.1,
    "rezessiv_fusion_prob" => 0.1,
    "rezessiv_fusion_rate" => 0.1,
    "fusion_prob" => 0.0,
    "fusion_rate" => 0.0,
    "inversion_prob" => 0.1
)

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


@inline function corr_call_back!(genes::Vector{Int8}, start_indices::Vector{Int}, expression::Vector{Int8}, token_dto::TokenDto; cycles::Int=5)
    return correct_genes!(genes, start_indices, expression,
        convert.(Float16, target_dim), token_dto; cycles=cycles)
end

mutable struct GepRegressor
    utilized_symbols_::OrderedDict{Int8,Int8}
    operators_::OperatorEnum
    callbacks_::Dict{Int8,Function}
    nodes_::OrderedDict{Int8,Any}
    gene_connections_::Vector{Int8}
    gene_count_::Int
    head_len_::Int
    preamble_syms_::Vector{Int8}
    dimension_information_::OrderedDict{Int8,Vector{Float16}}
    best_models_::Union{Nothing,Vector{Chromosome}}
    fitness_history_::Vector{AbstractFloat}
    token_dto::Union{TokenDto,Nothing}
    target_dim::Vector{Float16}

    function GepRegressor(feature_amount::Int;
        entered_features::Vector{Symbol}=Vector{Symbol}(),
        entered_non_terminals::Vector{Symbol}=[:+, :-, :*, :/, :sqrt],
        entered_terminal_nums::Vector{Symbol}=[Symbol(0.0), Symbol(0.5)],
        gene_connections::Vector{Symbol}=[:+, :-, :*, :/],
        considered_dimensions::Dict{Symbol,Vector{Float16}}=Dict{Symbol,Vector{Float16}}(),
        target_dimension::Vector{Float16} = Vector{Float16}(),
        rnd_count::Int=2,
        node_type::Type=Float64,
        gene_count::Int=3,
        head_len::Int=8,
        preamble_syms::Vector{Symbol}=Symbol[],
        max_permutations_lib::Int=10
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
            idx_features = [idx for (idx,_) in feat_syms]
            idx_funs = [idx for (idx,_) in func_syms]
            idx_const = [idx for (idx,_) in const_syms]

            lib = create_lib(token_lib,
                idx_features,
                idx_funs,
                idx_const;
                rounds=2, max_permutations=max_permutations_lib)
                token_dto = TokenDto(token_lib, point_ops, lib, backward_funs, gene_count; head_len=head_len- 1)
        else
            token_dto = nothing
        end


        new(
            utilized_symbols,
            operators,
            callbacks,
            nodes,
            gene_connections_,
            gene_count,
            head_len,
            preamble_syms_,
            dimension_information,
            nothing,
            Vector{AbstractFloat}(),
            token_dto,
            target_dimension
        )
    end
end

function fit!(regressor::GepRegressor, dataset::AbstractArray; train_split::Real=0.8)

end

function predict(regressor::GepRegressor, x_data::AbstractArray; ensembe::Bool=False)

end

function eval(regressor::GepRegressor, x_datat::AbstractArray; plot::Bool=False)

end



end