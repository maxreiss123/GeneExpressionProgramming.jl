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

# Core functions
export compile_to_flux_network

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

@inline function (l::AdditionNode)(x::Tensor, y::Tensor)
    result = x .+ y
    return result
end

@inline function (l::AdditionNode)(x::Vector, y::Vector)
    result = x .+ y
    return result
end

@inline function (l::AdditionNode)(x::Number, y::Number)
    result = x .+ y
    return result
end

@inline function (l::SubtractionNode)(x::Tensor, y::Tensor)
    result = x .- y
    return result
end


@inline function (l::SubtractionNode)(x::Vector, y::Vector)
    result = x .- y
    return result
end

@inline function (l::SubtractionNode)(x::Number, y::Number)
    result = x .- y
    return result
end

@inline function (l::MultiplicationNode)(x::Number, y::Number)
    result =  y .* x
    return result
end

@inline function (l::MultiplicationNode)(x::Tensor, y::Number)
    result =  y * x
    return result
end

@inline function (l::MultiplicationNode)(x::Number, y::Tensor)
    result =  y * x
    return result
end

@inline function (l::MultiplicationNode)(x::Tensor, y::Tensor)
    result =  dot(x,y)
    return result
end

@inline function (l::DivisionNode)(x::Tensor, y::Vector)
    result = x ./ y
    return result
end


@inline function (l::PowerNode)(x::Union{Tensor,Number}, y::Number)
    result = x ^ y
    return result
end

@inline function (l::DoubleContractionNode)(x, y)
    result = dcontract(x, y)
    return result
end

@inline function (l::DeviatoricNode)(x)
    result = dev(x)
    return result
end


@inline function (l::MinNode)(x, y)
    result = min.(x, y)
    return result
end

@inline function (l::MaxNode)(x, y)
    result = max.(x, y)
    return result
end

@inline function (l::InversionNode)(x)
    result = inv(x)
    return result
end

@inline function (l::TraceNode)(x)
    result = tr(x)
    return result
end

@inline function (l::DeterminantNode)(x)
    result = det(x)
    return result
end

@inline function (l::SymmetricNode)(x)
    result = symmetric(x)
    return result
end

@inline function (l::SkewNode)(x)
    result = skew(x)
    return result
end

@inline function (l::VolumetricNode)(x)
    result = vol(x)
    return result
end

@inline function (l::DeviatricNode)(x)
    result = dev(x)
    return result
end

@inline function (l::TdotNode)(x)
    result = tdot(x)
    return result
end

@inline function (l::DottNode)(x)
    result = dott(x)
    return result
end

@inline function (l::ConstantNode)(x)
    return l.value
end


@inline function (l::UnaryNode)(x)
    result = l.operation.(x)
    return result
end

function compile_to_flux_network(rek_string::Vector, arity_map::OrderedDict, callbacks::Dict, nodes::OrderedDict; use_cuda::Bool=false)
    stack = []
    inputs_idx = Dict{Int8, Int8}()
    idx = 1
    for elem in reverse(rek_string)
        if get(arity_map, elem, 0) == 2
            op1 = pop!(stack)
            op2 = pop!(stack)
            node = callbacks[elem]()
            push!(stack, (inputs) -> node(op1(inputs), op2(inputs)))
            @show elem
        elseif get(arity_map, elem, 0) == 1
            op = pop!(stack)
            node = callbacks[elem]()
            push!(stack, (inputs) -> node(op(inputs)))
        else
            if get(inputs_idx, elem, 0) == 0
                inputs_idx[elem] = idx    
                idx +=1
            end
            if nodes[elem] isa Number
                push!(stack, _ -> nodes[elem])
                #push!(stack, (inputs) -> ConstantNode(inputs_idx[elem])(inputs))
            else
                push!(stack, (inputs) -> InputSelector(inputs_idx[elem])(inputs))
            end
        end
        
    end
    return Chain(pop!(stack)), inputs_idx
end

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