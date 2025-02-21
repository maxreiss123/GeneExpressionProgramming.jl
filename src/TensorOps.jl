module TensorRegUtils

using Flux, LinearAlgebra, OrderedCollections, ChainRulesCore, Tensors, PrecompileTools



# Abstract base type with parametric types for improved type stability
abstract type AbstractOperationNode{T} end

# Input selector with strict typing
struct InputSelector{T<:Integer}
    idx::T
end

@inline function (l::InputSelector{T})(x::Tuple) where {T}
    @inbounds x[l.idx]
end

@inline function (l::InputSelector{T})(x::Any) where {T}
    @inbounds x
end

# Macro to generate specialized operation nodes with functors
macro generate_operation_node(name)
    return quote
        struct $(esc(name)){T<:Union{Nothing,Chain}} <: AbstractOperationNode{T}
            chain::T
            $(esc(name))(chain=nothing) = new{typeof(chain)}(chain)
        end
        Flux.Functors.@functor $(esc(name))
    end
end

# Generate concrete operation nodes
@generate_operation_node AdditionNode
@generate_operation_node SubtractionNode
@generate_operation_node MultiplicationNode
@generate_operation_node DivisionNode
@generate_operation_node PowerNode
@generate_operation_node MinNode
@generate_operation_node MaxNode
@generate_operation_node InversionNode
@generate_operation_node TraceNode
@generate_operation_node DeterminantNode
@generate_operation_node SymmetricNode
@generate_operation_node SkewNode
@generate_operation_node VolumetricNode
@generate_operation_node DeviatricNode
@generate_operation_node TdotNode
@generate_operation_node DottNode
@generate_operation_node DoubleContractionNode
@generate_operation_node DeviatoricNode

# Specialized nodes with their functors
struct ConstantNode{T<:Number,C<:Union{Nothing,Chain}} <: AbstractOperationNode{C}
    value::T
    chain::C
    ConstantNode(value::T, chain=nothing) where {T<:Number} = new{T,typeof(chain)}(value, chain)
end
Flux.Functors.@functor ConstantNode

struct UnaryNode{F<:Function,C<:Union{Nothing,Chain}} <: AbstractOperationNode{C}
    operation::F
    chain::C
    UnaryNode(operation::F, chain=nothing) where {F<:Function} = new{F,typeof(chain)}(operation, chain)
end
Flux.Functors.@functor UnaryNode

# Operation implementations
@inline function (l::AdditionNode{T})(x::Union{Tensor,SymmetricTensor},
    y::Union{Tensor,SymmetricTensor}) where {T}
    @fastmath (x + y)::Union{Tensor,SymmetricTensor}
end

@inline function (l::AdditionNode{T})(x::Number, y::Number) where {T}
    @fastmath (x + y)::Number
end


@inline function (l::AdditionNode{T})(x::Any, y::Any) where {T}
    Inf::Number
end

@inline function (l::MultiplicationNode{T})(x::Any, y::Any) where {T}
    Inf::Number
end

@inline function (l::SubtractionNode{T})(x::Union{Tensor,SymmetricTensor}, y::Union{Tensor,SymmetricTensor}) where {T}
    @fastmath (x - y)::Union{Tensor,SymmetricTensor}
end


@inline function (l::SubtractionNode{T})(x::Number, y::Number) where {T}
    @fastmath (x - y)::Number
end

@inline function (l::SubtractionNode{T})(x::Any, y::Any) where {T}
    Inf::Number
end

@inline function (l::MultiplicationNode{T})(x::Number, y::Number) where {T}
    @fastmath (x * y)::Number
end

@inline function (l::MultiplicationNode{T})(x::Union{Tensor,SymmetricTensor}, y::Number) where {T}
    @fastmath (x * y)::Union{Tensor,SymmetricTensor}
end

@inline function (l::MultiplicationNode{T})(x::Number, y::Union{Tensor,SymmetricTensor}) where {T}
    @fastmath (x * y)::Union{Tensor,SymmetricTensor}
end



@inline function (l::MultiplicationNode{T})(x::Union{Tensor,SymmetricTensor}, y::Union{Tensor,SymmetricTensor}) where {T}
    @fastmath dot(x, y)::Union{Tensor,SymmetricTensor,Number}
end

@inline function (l::DivisionNode{T})(x::Union{Tensor,SymmetricTensor,Number}, y::Number) where {T}
    @fastmath (x / y)::Union{Tensor,SymmetricTensor,Number}
end

@inline function (l::DivisionNode{T})(x::Any, y::Any) where {T}
    Inf::Number
end

@inline function (l::PowerNode{T})(x::Union{Tensor,SymmetricTensor,Number}, y::Number) where {T}
    @fastmath (x^y)::Union{Tensor,SymmetricTensor}
end

@inline function (l::DoubleContractionNode{T})(x, y) where {T}
    @fastmath dcontract(x, y)
end

@inline function (l::DeviatoricNode{T})(x) where {T}
    @fastmath dev(x)
end

@inline function (l::MinNode{T})(x, y) where {T}
    @fastmath min(x, y)
end

@inline function (l::MaxNode{T})(x, y) where {T}
    @fastmath max(x, y)
end

@inline function (l::InversionNode{T})(x) where {T}
    @fastmath inv(x)
end

@inline function (l::TraceNode{T})(x) where {T}
    @fastmath tr(x)
end

@inline function (l::DeterminantNode{T})(x) where {T}
    @fastmath det(x)::Number
end

@inline function (l::SymmetricNode{T})(x::Union{Tensor,SymmetricTensor}) where {T}
    @fastmath symmetric(x)::Union{Tensor,SymmetricTensor}
end

@inline function (l::SkewNode{T})(x::Union{Tensor,SymmetricTensor}) where {T}
    @fastmath skew(x)::Union{Tensor,SymmetricTensor}
end

@inline function (l::VolumetricNode{T})(x) where {T}
    @fastmath vol(x)
end

@inline function (l::DeviatricNode{T})(x) where {T}
    @fastmath dev(x)
end

@inline function (l::TdotNode{T})(x) where {T}
    @fastmath tdot(x)
end

@inline function (l::DottNode{T})(x) where {T}
    @fastmath dott(x)
end

@inline function (l::ConstantNode{V,T})() where {V,T}
    l.value
end

@inline function (l::UnaryNode{F,T})(x) where {F,T}
    @fastmath l.operation.(x)
end


@inline function (l::AdditionNode{T})(x::AbstractVector, y::AbstractVector) where {T}
    map((a, b) -> l(a, b), x, y)::AbstractVector
end

@inline function (l::AdditionNode{T})(x::AbstractVector{Number}, y::AbstractVector{Number}) where {T}
    (x .+ y)::AbstractVector{Number}
end

@inline function (l::SubtractionNode{T})(x::AbstractVector, y::AbstractVector) where {T}
    map((a, b) -> l(a, b), x, y)::AbstractVector
end

@inline function (l::MultiplicationNode{T})(x::AbstractVector, y::AbstractVector) where {T}
    map((a, b) -> l(a, b), x, y)::AbstractVector
end

@inline function (l::DivisionNode{T})(x::AbstractVector, y::Number) where {T}
    map(a -> l(a, y), x)::AbstractVector
end

@inline function (l::DivisionNode{T})(x::AbstractVector, y::AbstractVector) where {T}
    map((a, b) -> l(a, b), x, y)::AbstractVector
end

@inline function (l::PowerNode{T})(x::AbstractVector, y::Number) where {T}
    map(a -> l(a, y), x)::AbstractVector
end

@inline function (l::PowerNode{T})(x::AbstractVector, y::AbstractVector) where {T}
    map((a, b) -> l(a, b), x, y)::AbstractVector
end

@inline function (l::MinNode{T})(x::AbstractVector, y::AbstractVector) where {T}
    map((a, b) -> l(a, b), x, y)::AbstractVector
end

@inline function (l::MaxNode{T})(x::AbstractVector, y::AbstractVector) where {T}
    map((a, b) -> l(a, b), x, y)::AbstractVector
end

@inline function (l::TraceNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end

@inline function (l::DeterminantNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end

@inline function (l::SymmetricNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end

@inline function (l::SkewNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end

@inline function (l::VolumetricNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end

@inline function (l::DeviatricNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end

@inline function (l::InversionNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end

@inline function (l::TdotNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end

@inline function (l::DottNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end

@inline function (l::DeviatoricNode{T})(x::AbstractVector) where {T}
    map(l, x)::AbstractVector
end



function compile_to_flux_network(rek_string::Vector, arity_map::OrderedDict, callbacks::Dict, nodes::OrderedDict, pre_len::Int)
    stack = []
    for elem in reverse(rek_string)
        if get(arity_map, elem, 0) == 2
            op1 = pop!(stack)
            op2 = pop!(stack)
            node = callbacks[elem]()
            push!(stack, (inputs) -> node(op1(inputs), op2(inputs)))
        elseif get(arity_map, elem, 0) == 1
            op = pop!(stack)
            node = callbacks[elem]()
            push!(stack, (inputs) -> node(op(inputs)))
        else
            if nodes[elem] isa Number
                num = nodes[elem]
                push!(stack, (inputs) -> ConstantNode(num)())
            else
                idx = nodes[elem].idx
                push!(stack, (inputs) -> InputSelector(idx)(inputs))
            end
        end

    end
    return Chain(pop!(stack))
end

function string()

end

# Constant mappings
const TENSOR_NODES = Dict{Symbol,Type}(
    :+ => AdditionNode,
    :- => SubtractionNode,
    :* => MultiplicationNode,
    :/ => DivisionNode,
    :^ => PowerNode,
    :min => MinNode,
    :max => MaxNode,
    :inv => InversionNode,
    :tr => TraceNode,
    :det => DeterminantNode,
    :symmetric => SymmetricNode,
    :skew => SkewNode,
    :vol => VolumetricNode,
    :dev => DeviatricNode,
    :tdot => TdotNode,
    :dott => DottNode,
    :dcontract => DoubleContractionNode,
    :deviator => DeviatoricNode
)

const TENSOR_NODES_ARITY = Dict{Symbol,Int8}(
    :+ => 2, :- => 2, :* => 2, :/ => 2, :^ => 2,
    :min => 2, :max => 2,
    :inv => 1, :tr => 1, :det => 1,
    :symmetric => 1, :skew => 1, :vol => 1, :dev => 1,
    :tdot => 1, :dott => 1, :dcontract => 2, :deviator => 1
)

# Exports
export InputSelector
export AdditionNode, SubtractionNode, MultiplicationNode, DivisionNode, PowerNode
export MinNode, MaxNode, InversionNode
export TraceNode, DeterminantNode, SymmetricNode, SkewNode
export VolumetricNode, DeviatricNode, TdotNode, DottNode
export DoubleContractionNode, DeviatoricNode
export ConstantNode, UnaryNode
export compile_to_flux_network
export TENSOR_NODES, TENSOR_NODES_ARITY


@setup_workload begin
    dim = 3
    t2 = rand(Tensor{2,dim})
    vec3 = rand(Tensor{1,dim})
    const_val = 2.0
    nodes = OrderedDict{Int8,Any}(
        5 => InputSelector(1),
        6 => InputSelector(2),
        7 => const_val
    )
    arity_map = OrderedDict{Int8,Int}(
        1 => 2,  # + (AdditionNode)
        2 => 2,  # * (MultiplicationNode)
        3 => 2,  # dcontract
        4 => 1   # tr
    )
    callbacks = Dict{Int8,Any}(
        1 => AdditionNode,
        2 => MultiplicationNode,
        3 => DoubleContractionNode,
        4 => TraceNode
    )

    expressions = [
        Int8[2, 5, 5],
        Int8[2, 5, 5],
        Int8[1, 5, 5],
        Int8[2, 6, 7],       
        Int8[1, 5, 6],       
        Int8[3, 5, 5],       
        Int8[4, 5]           
    ]
    inputs = (t2, vec3, const_val)
    inputs2 = ([t2 for _ in 1:10],[vec3 for _ in 1:10],[const_val for _ in 1:10])
    @compile_workload begin
        for expr in expressions
            net = compile_to_flux_network(expr, arity_map, callbacks, nodes, 0)
            try
                net(inputs)
                net(inputs2)
            catch
                # Ignore runtime dimension mismatch for precompile
            end
        end
    end
end

end