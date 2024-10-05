module GepUtils


using OrderedCollections
using DynamicExpressions
using LinearAlgebra
using Optim
using LineSearches
using Zygote
using Serialization

export Toolbox
export find_indices_with_sum, compile_djl_datatype, optimize_constants, minmax_scale, float16_scale, isclose
export save_state, load_state



struct Toolbox
    gene_count::Int
    head_len::Int
    symbols::OrderedDict{Int8,Int8}
    gene_connections::Vector{Int8}
    headsyms::Vector{Int8}
    unary_syms::Vector{Int8}
    tailsyms::Vector{Int8}
    arrity_by_id::OrderedDict{Int8,Int8}
    callbacks::Dict
    nodes::OrderedDict
    gen_start_indices::Vector{Int}
    gep_probs::Dict{String,AbstractFloat}
    unary_prob::Real
    fitness_reset::Tuple
    preamble_syms::Vector{Int8}
    len_preamble::Int8


    function Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8}, gene_connections::Vector{Int8},
        callbacks::Dict, nodes::OrderedDict, gep_probs::Dict{String,AbstractFloat};
        unary_prob::Real=0.4, fitness_reset::Tuple=(Inf, NaN), preamble_syms=Int8[])
        gene_len = head_len * 2 + 1
        headsyms = [key for (key, arity) in symbols if arity == 2]
        unary_syms = [key for (key, arity) in symbols if arity == 1]
        tailsyms = [key for (key, arity) in symbols if arity < 1 && !(key in preamble_syms)]
        len_preamble = length(preamble_syms) == 0 ? 0 : gene_count
        gen_start_indices = [gene_count + len_preamble + (gene_len * (i - 1)) for i in 1:gene_count] #depending on the usage should shift everthing 
        new(gene_count, head_len, symbols, gene_connections, headsyms, unary_syms, tailsyms, symbols,
            callbacks, nodes, gen_start_indices, gep_probs, unary_prob, fitness_reset, preamble_syms, len_preamble)
    end
end

function isclose(a::T, b::T; rtol::T=1e-5, atol::T=1e-8) where {T<:Number}
    return abs(a - b) <= (atol + rtol * abs(b))
end

function fast_sqrt_32(x::Real)
    i = reinterpret(UInt32, x)
    i = 0x1fbd1df5 + (i >> 1)
    return reinterpret(Real, i)
end

function float32_scale(arr::AbstractArray{T}) where {T<:AbstractFloat}
    min_magnitude = Float32(6.1e-5)
    max_magnitude = Float32(65504)
    scaled = _minmax_scale!(copy(arr), feature_range=(min_magnitude, max_magnitude))
    return Float16.(scaled)
end


function find_indices_with_sum(arr::SubArray, target_sum::Int, num_indices::Int)
    if arr[1] == 0
        return [1]
    end
    cum_sum = cumsum(arr)
    indices = findall(x -> x == target_sum, cum_sum)
    if length(indices) >= num_indices
        return indices[1:num_indices]
    else
        return [length(arr)]
    end
end

function compile_djl_datatype(rek_string::Vector, arity_map::OrderedDict, callbacks::Dict, nodes::OrderedDict)
    #just let it fail when it becomes invalid, because then the equation is not that use ful
    stack = []
    for elem in reverse(rek_string)
        if get(arity_map, elem, 0) == 2
            op1 = (temp = pop!(stack); temp isa Int8 ? nodes[temp] : temp)
            op2 = (temp = pop!(stack); temp isa Int8 ? nodes[temp] : temp)
            ops = callbacks[elem]
            push!(stack, ops(op1, op2))
        elseif get(arity_map, elem, 0) == 1
            op1 = (temp = pop!(stack); temp isa Int8 ? nodes[temp] : temp)
            ops = callbacks[elem]
            push!(stack, ops(op1))
        else
            push!(stack, elem)
        end
    end
    return last(stack)
end

function retrieve_constants_from_node(node::Node)
    constants = AbstractFloat[]
    for op in node
        if op isa AbstractNode && op.degree == 0 && op.constant
            push!(constants, convert(AbstractFloat, op.val))
        end
    end
    constants
end


function optimize_constants(
    node::Node,
    x_data::AbstractArray{T},
    y_data::AbstractArray{T},
    loss::Function,
    operators::AbstractOperatorEnum;
    opt_method::Symbol=:cg,
    max_iterations::Int=250,
    n_restarts::Int=3
) where {T<:AbstractFloat}

    nconst = count_constants(node)

    if nconst == 0
        return node, 0.0
    end

    function f(tree::Node)
        y_pred, flag = eval_tree_array(tree, x_data, operators)
        return loss(y_pred, y_data)
    end

    baseline = f(node)
    best_node = deepcopy(node)
    best_loss = baseline

    algorithm = if opt_method == :newton
        Optim.Newton(; linesearch=LineSearches.BackTracking())
    elseif opt_method == :cg
        Optim.ConjugateGradient(; linesearch=LineSearches.BackTracking())
    else
        Optim.NelderMead()
    end

    optimizer_options = Optim.Options(; iterations=max_iterations, show_trace=false)

    for i in 0:n_restarts
        current_node = i == 0 ? node : deepcopy(node)

        if i > 0
            foreach(current_node) do n
                if n.degree == 0 && n.constant
                    n.val = n.val * (1 + 0.5 * randn())
                end
            end
        end

        result = Optim.optimize(f, current_node, algorithm, optimizer_options)

        if result.minimum < best_loss
            best_node = result.minimizer
            best_loss = result.minimum
        end
    end

    return best_node, best_loss
end

function _minmax_scale!(X::AbstractArray{T}; feature_range=(zero(T), one(T))) where {T<:AbstractFloat}
    min_vals = minimum(X, dims=1)
    max_vals = maximum(X, dims=1)
    range_width = max_vals .- min_vals

    a, b = feature_range
    scale = (b - a) ./ range_width

    @inbounds @simd for j in axes(X, 2)
        if range_width[j] ≈ zero(T)
            X[:, j] .= (a + b) / 2
        else
            @simd for i in axes(X, 1)
                X[i, j] = (X[i, j] - min_vals[j]) * scale[j] + a
            end
        end
    end

    return X
end

function minmax_scale(X::AbstractArray{T}; feature_range=(zero(T), one(T))) where {T<:AbstractFloat}
    return _minmax_scale!(copy(X); feature_range=feature_range)
end

function save_state(filename::String, state::Any)
    open(filename, "w") do io
        serialize(io, state)
    end
end

function load_state(filename::String)
    open(filename, "r") do io
        return deserialize(io)
    end
end

end