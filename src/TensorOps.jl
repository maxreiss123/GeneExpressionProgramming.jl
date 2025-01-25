module TensorRegUtils

using Flux, LinearAlgebra, OrderedCollections, ChainRulesCore

using Tensors

# Types
export OperationNode
export InputSelector
export AdditionNode, SubtractionNode, MultiplicationNode, DivisionNode, PowerNode
export MinNode, MaxNode, InversionNode
export TraceNode, DeterminantNode, SymmetricNode, SkewNode
export VolumetricNode, DeviatricNode, TdotNode, DottNode
export DoubleContractionNode, DeviatoricNode
export ConstantNode, UnaryNode

export compile_to_flux_network

export TENSOR_NODES, TENSOR_NODES_ARITY


abstract type OperationNode end

struct InputSelector
    idx::Int
end

@inline function (l::InputSelector)(x::Tuple)
    x[l.idx]
end

struct AdditionNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    AdditionNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct SubtractionNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    SubtractionNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct MultiplicationNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    MultiplicationNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct DivisionNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    DivisionNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct PowerNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    PowerNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct MinNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    MinNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct MaxNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    MaxNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct InversionNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    InversionNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct TraceNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    TraceNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct DeterminantNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    DeterminantNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct SymmetricNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    SymmetricNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct SkewNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    SkewNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct VolumetricNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    VolumetricNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct DeviatricNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    DeviatricNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct TdotNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    TdotNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct DottNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    DottNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct DoubleContractionNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    DoubleContractionNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct DeviatoricNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    DeviatoricNode(chain=Chain(); use_cuda::Bool=false) = new(chain, use_cuda)
end

struct ConstantNode <: OperationNode
    value::Number
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    ConstantNode(value::Number, chain=Chain(); use_cuda::Bool=false) = new(value, chain, use_cuda)
end


struct UnaryNode <: OperationNode
    operation::Function
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    UnaryNode(operation::Function, chain=Chain(); use_cuda::Bool=false) =
        new(operation, chain, use_cuda)
end


const NODES = [
    AdditionNode,
    SubtractionNode,
    MultiplicationNode,
    DivisionNode,
    PowerNode,
    MinNode,
    MaxNode,
    InversionNode,
    TraceNode,
    DeterminantNode,
    SymmetricNode,
    SkewNode,
    VolumetricNode,
    DeviatricNode,
    TdotNode,
    DottNode,
    DoubleContractionNode,
    DeviatoricNode,
    ConstantNode,
    UnaryNode
]

for T in NODES
    @eval Flux.Functors.@functor $T
end

@inline function (l::AdditionNode)(x::Union{Tensor,SymmetricTensor}, y::Union{Tensor,SymmetricTensor})
    @fastmath return x + y
end

@inline function (l::AdditionNode)(x::Vector, y::Vector)
    @fastmath return x + y
end

@inline function (l::AdditionNode)(x::Number, y::Number)
    @fastmath return x + y
end

@inline function (l::SubtractionNode)(x::Union{Tensor,SymmetricTensor}, y::Union{Tensor,SymmetricTensor})
    @fastmath return x - y
end


@inline function (l::SubtractionNode)(x::Vector, y::Vector)
    @fastmath return x - y
end

@inline function (l::SubtractionNode)(x::Number, y::Number)
    @fastmath return x - y
end

@inline function (l::MultiplicationNode)(x::Number, y::Number)
    @fastmath return y * x
end

@inline function (l::MultiplicationNode)(x::Union{Tensor,SymmetricTensor}, y::Number)
    @fastmath return y * x
end

@inline function (l::MultiplicationNode)(x::Number, y::Union{Tensor,SymmetricTensor})
    @fastmath return y * x
end

@inline function (l::MultiplicationNode)(x::Union{Tensor,SymmetricTensor}, y::Union{Tensor,SymmetricTensor})
    @fastmath return dot(x, y)
end

@inline function (l::DivisionNode)(x::Union{Tensor,SymmetricTensor}, y::Vector)
    @fastmath return x / y
end


@inline function (l::PowerNode)(x::Union{Tensor,SymmetricTensor,Number}, y::Number)
    @fastmath return x^y
end

@inline function (l::DoubleContractionNode)(x, y)
    @fastmath return dcontract(x, y)
end

@inline function (l::DeviatoricNode)(x)
    @fastmath return dev(x)
end


@inline function (l::MinNode)(x, y)
    @fastmath return min(x, y)
end

@inline function (l::MaxNode)(x, y)
    @fastmath return max(x, y)
end

@inline function (l::InversionNode)(x)
    @fastmath return inv(x)
end

@inline function (l::TraceNode)(x)
    @fastmath return tr(x)
end

@inline function (l::DeterminantNode)(x)
    @fastmath return det(x)
end

@inline function (l::SymmetricNode)(x)
    @fastmath return symmetric(x)
end

@inline function (l::SkewNode)(x)
    @fastmath return skew(x)
end

@inline function (l::VolumetricNode)(x)
    @fastmath return vol(x)
end

@inline function (l::DeviatricNode)(x)
    @fastmath return dev(x)
end

@inline function (l::TdotNode)(x)
    @fastmath return tdot(x)
end

@inline function (l::DottNode)(x)
    @fastmath return dott(x)
end

@inline function (l::ConstantNode)(x)
    return l.value
end


@inline function (l::UnaryNode)(x)
    @fastmath return l.operation.(x)
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
                push!(stack, _ -> num)
            else
                idx = nodes[elem].idx
                push!(stack, (inputs) -> InputSelector(idx)(inputs))
            end
        end

    end
    return Chain(pop!(stack))
end


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
    :+ => 2,
    :- => 2,
    :* => 2,
    :/ => 2,
    :^ => 2,
    :min => 2,
    :max => 2,
    :inv => 1,
    :tr => 1,
    :det => 1,
    :symmetric => 1,
    :skew => 1,
    :vol => 1,
    :dev => 1,
    :tdot => 1,
    :dott => 1,
    :dcontract => 2,
    :deviator => 1
)


"""

Later on cuda Module

@inline function ensure_cuda(x, use_cuda::Bool)
    if use_cuda && !isa(x, CuArray)
        cu(x)
    elseif !use_cuda && isa(x, CuArray)
        cpu(x)
    else
        return x
    end
end

Example Method usage:

struct DottNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    DottNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

notes:
if rek comes to a constant: ConstantNode(elem, nothing, use_cuda=use_cuda)
if rek comes to an x -> InputSelector(i)
if unary -> initializes Arity1Node(nothing, use_cuda=use_cuda) 
if binary -> initaliszed Arity2Node(nothing, use_cuda=use_cuda) 

@assert sp == 1 "Invalid expression: stack error"

"""

end