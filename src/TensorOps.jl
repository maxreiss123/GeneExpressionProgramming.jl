"""
    TensorRegUtils

Operator nodes and the batched evaluator. Each operator is a node type (`AdditionNode`,
`DotProductNode`, ...) callable on single values (scalars, Tensors.jl tensors) and, in
batched form, on data columns; `return_type` gives its result type.
`calc_stack_batch_tensor` evaluates a karva string over columns, writing intermediates into
preallocated per-thread buffers; `compile_program` and `run_program!` do the same without
dynamic dispatch when all columns share one type. The `TENSOR_*` tables map operator
symbols and functions to node types, arities and renderers.
"""
module TensorRegUtils

# Imports
import Base: string
using LinearAlgebra, OrderedCollections, ChainRulesCore, Tensors

# Exports
export InputSelector
export AdditionNode, SubtractionNode, MultiplicationNode, DivisionNode, PowerNode
export MinNode, MaxNode, InversionNode, DotProductNode
export TraceNode, DeterminantNode, SymmetricNode, SkewNode
export VolumetricNode, TdotNode, DottNode
export DoubleContractionNode, DeviatoricNode
export ConstantNode, UnaryNode, CrossProductNode, LapNode
export calc_stack_batch_tensor, OuterProductNode, HadamardNode, SqrtNode, NormNode
export TENSOR_NODES, TENSOR_NODES_ARITY, TENSOR_STRINGIFY, return_type
export TENSOR_NODE_BY_FUNCTION
export eval_op!
export EvalProgram, compile_program, run_program!

"""
    ∘(a::AbstractTensor{N,dim}, b::AbstractTensor{N,dim}) where {N,dim}

Hadamard (elementwise) product of two tensors of the same order and spatial dimension,
returned as a `Tensor`. Both must be `Tensor`s: the method multiplies the stored `data`
tuples, and a `SymmetricTensor` stores only its independent components.
"""
function Base.:∘(a::AbstractTensor{N,dim}, b::AbstractTensor{N,dim}) where {N,dim}
    return Tensor{N,dim}(a.data .* b.data)
end


# Second-order identity by spatial dimension, for `LapNode`.
const UNIT_TENSOR = Dict(
    2 => one(SymmetricTensor{2,2}),
    3 => one(SymmetricTensor{2,3})
)

# Node types by arity: `Arity0Node` terminals, `Arity1Node` unary and `Arity2Node` binary
# operators.
abstract type AbstractOperationNode end
abstract type Arity0Node <: AbstractOperationNode end
abstract type Arity1Node <: AbstractOperationNode end
abstract type Arity2Node <: AbstractOperationNode end

"""
    InputSelector(idx[, name])

Terminal standing for input feature `idx`, displayed as `name` (by default `x1`, `x2`,
...). Called on a tuple it returns element `idx`; any other argument is returned unchanged.
"""
struct InputSelector{T<:Integer} <: Arity0Node
    idx::T
    name::String
end
function InputSelector(idx::T) where {T<:Integer}
    InputSelector{T}(idx, "x$idx")
end
function Base.string(node::InputSelector)
    "$(node.name)"
end
# Rendered equations interpolate and `join` terminals, which prints them through `show`,
# not through the one-argument `string` above.
Base.show(io::IO, node::InputSelector) = print(io, node.name)
@inline function (l::InputSelector{T})(x::Tuple) where {T}
    @inbounds x[l.idx]
end
@inline function (l::InputSelector{T})(x::Any) where {T}
    @inbounds x
end

"""
    ConstantNode(value)

Terminal holding the constant `value`; called with one to three arguments it returns
`value`.
"""
struct ConstantNode{T<:Number} <: Arity0Node
    value::T
    ConstantNode(value::T) where {T<:Number} = new{T}(value)
end
@inline function (l::ConstantNode)(x::Any)
    @fastmath l.value
end
@inline function (l::ConstantNode)(x::Any, y::Any)
    @fastmath l.value
end
@inline function (l::ConstantNode)(x::Any, y::Any, z::Any)
    @fastmath l.value
end

"""
    UnaryNode(operation)

Unary operator applying the scalar function `operation`. `return_type` gives it no column
type, so `calc_stack_batch_tensor` applies it (elementwise) only when called without
buffers.
"""
struct UnaryNode{F<:Function} <: Arity1Node
    operation::F
    UnaryNode(operation::F) where {F<:Function} = new{F}(operation)
end
@inline function (l::UnaryNode{F})(x::Any) where {F}
    @fastmath l.operation(x)
end
@inline function (l::UnaryNode{F})(x::Any, y::Any) where {F}
    @fastmath Inf::Number
end
@inline function (l::UnaryNode{F})(x::Any, y::Any, z::Any) where {F}
    @fastmath Inf::Number
end

# Defines the operator node type `name <: arity` (`Arity1Node` or `Arity2Node`) with
# catch-all methods for its single-value and batched forms that return `Inf`, so operands
# without a specific method give a non-finite value instead of a `MethodError`.
macro generate_operation_node(name, arity)
    if arity == :Arity1Node
        quote
            struct $(esc(name)) <: $(esc(arity)) end
            @inline function (l::$(esc(name)))(x::Any)
                @fastmath Inf::Number
            end
            @inline function (l::$(esc(name)))(x::Any, y::Any)
                @fastmath Inf::Number
            end
        end
    elseif arity == :Arity2Node
        quote
            struct $(esc(name)) <: $(esc(arity)) end
            @inline function (l::$(esc(name)))(x::Any, y::Any)
                @fastmath Inf::Number
            end
            @inline function (l::$(esc(name)))(x::Any, y::Any, z::Any)
                @fastmath Inf::Number
            end
        end
    end
end

# Operator node types
@generate_operation_node AdditionNode Arity2Node
@generate_operation_node SubtractionNode Arity2Node
@generate_operation_node MultiplicationNode Arity2Node
@generate_operation_node DivisionNode Arity2Node
@generate_operation_node DoubleContractionNode Arity2Node
@generate_operation_node DotProductNode Arity2Node
@generate_operation_node PowerNode Arity2Node
@generate_operation_node MinNode Arity2Node
@generate_operation_node MaxNode Arity2Node
@generate_operation_node InversionNode Arity1Node
@generate_operation_node DeterminantNode Arity1Node
@generate_operation_node SymmetricNode Arity1Node
@generate_operation_node SkewNode Arity1Node
@generate_operation_node VolumetricNode Arity1Node
@generate_operation_node DeviatoricNode Arity1Node
@generate_operation_node TdotNode Arity1Node
@generate_operation_node TraceNode Arity1Node
@generate_operation_node DottNode Arity1Node
@generate_operation_node CrossProductNode Arity2Node
@generate_operation_node LapNode Arity1Node
@generate_operation_node OuterProductNode Arity2Node
@generate_operation_node HadamardNode Arity2Node
@generate_operation_node SqrtNode Arity1Node
@generate_operation_node NormNode Arity1Node

@generate_operation_node SinNode Arity1Node
@generate_operation_node CosNode Arity1Node
@generate_operation_node ExpNode Arity1Node
@generate_operation_node LogNode Arity1Node
@generate_operation_node TanNode Arity1Node


# Index-based variants over a preallocated `stack` (top at `stack_idx`) and one buffer tuple
# (`counter_buffer` indexes the next unused buffer). Nothing in the package calls them.
# Unlike `calc_stack_batch_tensor`, the binary method applies `op(second, top)`, and the
# unary method returns `stack` where the others return `stack_idx`.
@inline function eval_op!(op::Arity2Node, stack::Vector, stack_idx::Int, counter_buffer::Int, buffers::NTuple)
    stack_idx -= 2
    op1 = stack[stack_idx+1]
    op2 = stack[stack_idx+2]
    buffer_ = buffers[counter_buffer]
    counter_buffer += 1
    op(op1, op2, buffer_)
    stack_idx += 1
    stack[stack_idx] = buffer_
    return stack_idx, counter_buffer
end

@inline function eval_op!(op::Arity1Node, stack::Vector, stack_idx::Int, counter_buffer::Int, buffers::NTuple)
    stack_idx -= 1
    op1 = stack[stack_idx+1]
    buffer_ = buffers[counter_buffer]
    counter_buffer += 1
    op(op1, buffer_)
    stack_idx += 1
    stack[stack_idx] = buffer_
    return stack, counter_buffer
end

@inline function eval_op!(op::Union{Vector,Arity0Node}, stack::Vector, stack_idx::Int, counter_buffer::Int, buffers::NTuple)
    stack_idx += 1
    stack[stack_idx] = op
    return stack_idx, counter_buffer
end

@inline function eval_op!(op::Union{Vector,Arity0Node}, stack::Vector, stack_idx::Int)
    stack_idx += 1
    stack[stack_idx] = op
    return stack_idx
end

@inline function eval_op!(op::Any, stack::Vector, stack_idx::Int, counter_buffer::Int, buffers::NTuple)
    stack_idx += 1
    stack[stack_idx] = op
    return stack_idx, counter_buffer
end

"""
    eval_op!(op, stack, used_buffers, buffers) -> Bool

One step of `calc_stack_batch_tensor`. A terminal column `op` is pushed onto `stack`. An
operator pops its operands, top of stack first, and pushes its result: with `buffers` (a
pool of preallocated columns keyed by column type) the result is written into the next
unused buffer of type `return_type(op, ...)`, `used_buffers` mapping each type to the
index of that buffer; with `buffers === nothing` it is freshly allocated. Returns `false`,
pushing nothing, when a terminal's `norm` is not finite or no pool matches the result type.
"""
@inline function eval_op!(op::Arity2Node, stack::Vector, used_buffers::Dict, buffers::Dict{Type,NTuple})
    op1 = pop!(stack)
    op2 = pop!(stack)
    type_check = TensorRegUtils.return_type(op, typeof(op1), typeof(op2))
    @debug "(1)-- $(Threads.threadid()) -- $op - type operands: $(typeof(op1)) - $(typeof(op2)) - estimate return $type_check"
    buffer_pool = get(buffers, type_check, nothing)
    @debug "(2)-- $(Threads.threadid()) -- Pool located: " !(isnothing(buffer_pool))
    if !(isnothing(buffer_pool))
        buffer_idx = used_buffers[type_check]
        buffer_ = buffer_pool[buffer_idx]
        @debug "(3)-- $(Threads.threadid()) -- Type buffer: $(typeof(buffer_))"
        @debug "(3.1)-- $(Threads.threadid()) -- o1 norm: $(norm(op1))"
        @debug "(3.2)-- $(Threads.threadid()) -- o2 norm: $(norm(op2))"
        result = op(op1, op2, buffer_)
        @debug "(4.1)-- $(Threads.threadid()) -- o1 norm: $(norm(op1))"
        @debug "(4.2)-- $(Threads.threadid()) -- o2 norm: $(norm(op2))"
        @debug "(4.3)-- $(Threads.threadid()) -- result norm: $(norm(result))"
        @debug "(4.3)-- $(Threads.threadid()) -- buffer norm: $(norm(buffer_))"
        used_buffers[type_check] += 1
        push!(stack, result)
        return true
    end
    return false
end

@inline function eval_op!(op::Arity1Node, stack::Vector, used_buffers::Dict, buffers::Dict{Type,NTuple})
    op1 = pop!(stack)
    type_check = TensorRegUtils.return_type(op, typeof(op1))
    @debug "-- $(Threads.threadid()) -- $op - type operands: $(typeof(op1)) - estimate return $type_check"
    buffer_pool = get(buffers, type_check, nothing)
    if !(isnothing(buffer_pool))
        buffer_idx = used_buffers[type_check]
        buffer_ = buffer_pool[buffer_idx]
        @debug "-- $(Threads.threadid()) -- Type buffer: $(typeof(buffer_))"
        result = op(op1, buffer_)
        used_buffers[type_check] += 1
        push!(stack, result)
        return true
    end
    return false
end

# TODO: handle `ConstantNode` and `InputSelector` terminals, which throw here (`norm` cannot
# iterate them); callers currently pass every terminal as a data column.
@inline function eval_op!(op::Any, stack::Vector, used_buffers::Dict, buffers::Union{Dict{Type,NTuple},Nothing})
    !(isfinite(norm(op))) && return false
    push!(stack, op)
    return true
end


@inline function eval_op!(op::Arity2Node, stack::Vector, used_buffers::Dict, buffers::Nothing)
    op1 = pop!(stack)
    op2 = pop!(stack)
    result = op.(op1, op2)
    push!(stack, result)
    return true
end

@inline function eval_op!(op::Arity1Node, stack::Vector, used_buffers::Dict, buffers::Nothing)
    op1 = pop!(stack)
    result = op.(op1)
    push!(stack, result)
    return true
end


# --------------------------------------------------------------------------------
#  Operator methods
#
#  `op(x)` and `op(x, y)` apply an operator to single values. The batched forms `op(x, b)`
#  and `op(x, y, b)` apply it elementwise over columns (vectors holding one value per
#  sample), write the result into the buffer `b` and return `b`. `Tensor` means Tensors.jl's
#  `Tensor` type, which includes `Vec` but not `SymmetricTensor`; "second-order tensor"
#  means either kind. Scalar functions throw a `DomainError` outside their domain, as in
#  Base.
# --------------------------------------------------------------------------------

"""
    (::AdditionNode)(x, y)
    (::AdditionNode)(x, y, b)

`x + y` for two scalars, two scalar columns, or two `Tensor`s of the same order and
spatial dimension. Batched: scalar columns, where one operand may be a single scalar, or
two `Tensor` columns.
"""
@inline function (l::AdditionNode)(x::T1, y::T2) where {T1<:Number,T2<:Number}
    return x + y
end

@inline function (l::AdditionNode)(x::Vector{T1}, y::Vector{T2}) where {T1<:Number,T2<:Number}
    return x .+ y
end

@inline function (l::AdditionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x .+ y
    return b
end

@inline function (l::AdditionNode)(x::Vector{T1}, y::T2, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x .+ y
    return b
end

@inline function (l::AdditionNode)(x::T1, y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x .+ y
    return b
end

@inline function (l::AdditionNode)(x::T1, y::T2) where {M,N,T1<:Tensor{M,N},T2<:Tensor{M,N}}
    return x + y
end

@inline function (l::AdditionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,N,T1<:Tensor{M,N},
    T2<:Tensor{M,N},T3<:Tensor{M,N}}
    b .= x .+ y
    return b
end

"""
    (::SubtractionNode)(x, y)
    (::SubtractionNode)(x, y, b)

`x - y` for two scalars or two `Tensor`s of the same order and spatial dimension.
Batched: scalar columns, where one operand may be a single scalar, or two `Tensor`
columns.
"""
@inline function (l::SubtractionNode)(x::T1, y::T2) where {T1<:Number,T2<:Number}
    return x - y
end

@inline function (l::SubtractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x .- y
    return b
end

@inline function (l::SubtractionNode)(x::Vector{T1}, y::T2, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x .- y
    return b
end

@inline function (l::SubtractionNode)(x::T1, y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x .- y
    return b
end

@inline function (l::SubtractionNode)(x::T1, y::T2) where {M,N,T1<:Tensor{M,N},T2<:Tensor{M,N}}
    return x - y
end

@inline function (l::SubtractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,N,T1<:Tensor{M,N},
    T2<:Tensor{M,N},T3<:Tensor{M,N}}
    @inbounds b .= x .- y
    return b
end


"""
    (::MultiplicationNode)(x, y)
    (::MultiplicationNode)(x, y, b)

`x * y` for two scalars, or a scalar and a `Tensor` in either order. Batched: the same
pairings over columns, where one scalar operand may be a single value.
"""
@inline function (l::MultiplicationNode)(x::T1, y::T2) where {T1<:Number,T2<:Number}
    return (x * y)
end

@inline function (l::MultiplicationNode)(x::T1, y::T2) where {T1<:Number,T2<:Tensor}
    return (x * y)
end

@inline function (l::MultiplicationNode)(x::T1, y::T2) where {T1<:Tensor,T2<:Number}
    return (x * y)
end


@inline function (l::MultiplicationNode)(x::Vector{T1}, y::T2, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= y .* x
    return b
end

@inline function (l::MultiplicationNode)(x::T1, y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x .* y
    return b
end

@inline function (l::MultiplicationNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x .* y
    return b
end

@inline function (l::MultiplicationNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {T1<:Number,T2<:Tensor,T3<:Tensor}
    @inbounds b .= x .* y
    return b
end

@inline function (l::MultiplicationNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {T1<:Number,T2<:Tensor,T3<:Tensor}
    @inbounds b .= x .* y
    return b
end


@inline function (l::MultiplicationNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,N,T1<:Tensor{M,N},
    T2<:Number,T3<:Tensor{M,N}}
    @inbounds b .= x .* y
    return b
end

@inline function (l::MultiplicationNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,N,T1<:Tensor{M,N},T2<:Number,T3<:Tensor{M,N}}
    @inbounds b .= x .* y
    return b
end


"""
    (::DivisionNode)(x, y)
    (::DivisionNode)(x, y, b)

`x / y` for two scalars, or a `Tensor` divided by a scalar. Batched: the same pairings
over columns, where one scalar operand may be a single value.
"""
@inline function (l::DivisionNode)(x::T1, y::T2) where {T1<:Number,T2<:Number}
    return x ./ y
end

@inline function (l::DivisionNode)(x::Vector{T1}, y::T2, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x ./ y
    return b
end

@inline function (l::DivisionNode)(x::T1, y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= x ./ y
    return b
end

@inline function (l::DivisionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= l.(x, y)
    return b
end


@inline function (l::DivisionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {T1<:Tensor,T2<:Number,T3<:Tensor}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DivisionNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {T1<:Tensor,T2<:Number,T3<:Tensor}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DivisionNode)(x::T1, y::T2) where {T1<:Tensor,T2<:Number}
    return x / y
end

"""
    (::DotProductNode)(x, y)
    (::DotProductNode)(x, y, b)

Single contraction `x ⋅ y` (`dot`) of two `Tensor`s of order one or two with the same
spatial dimension; for two scalars, `x * y`. Batched: the same tensor pairings over
columns, where in a pairing with a `Vec` one operand may be a single tensor.
"""
@inline function (l::DotProductNode)(x::T1, y::T2) where {T1<:Number,T2<:Number}
    return x * y
end

@inline function (l::DotProductNode)(x::T1, y::T2) where {T1<:Vec,T2<:Vec}
    return dot(x, y)
end

@inline function (l::DotProductNode)(x::T1, y::T2) where {M,T1<:Vec{M},T2<:Tensor{2,M}}
    return dot(x, y)
end

@inline function (l::DotProductNode)(x::T1, y::T2) where {M,T1<:Tensor{2,M},T2<:Vec{M}}
    return dot(x, y)
end

@inline function (l::DotProductNode)(x::T1, y::T2) where {M,T1<:Tensor{2,M},T2<:Tensor{2,M}}
    return dot(x, y)
end

# TODO: for two operands of the same type this method and the `Vec`/`Vec` or
# `Tensor{2}`/`Tensor{2}` method above both match, which is likely a method ambiguity
# (check with `Test.detect_ambiguities`).
@inline function (l::DotProductNode)(x::T1, y::T1) where {M,N,T1<:Tensor{M,N}}
    return dot(x, y)
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Vec{M},T2<:Vec{M},T3<:Number}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Vec{M},T2<:Tensor{2,M},T3<:Vec{M}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Vec{M},T3<:Vec{M}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{2,M},T3<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,T1<:Vec{M},T2<:Vec{M},T3<:Number}
    @inbounds b .= l.(x, y)
    return b
end


@inline function (l::DotProductNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Vec{M},T2<:Vec{M},T3<:Number}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::T1, y::Vector{T2}, b::Vector{T1}) where {M,T1<:Vec{M},T2<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::T2, b::Vector{T1}) where {M,T1<:Vec{M},T2<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::T2, y::Vector{T1}, b::Vector{T1}) where {M,T1<:Vec{M},T2<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T2}, y::T1, b::Vector{T1}) where {M,T1<:Vec{M},T2<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

# Operands of order three or four.
# TODO: these broadcast `l` over the elements, but `DotProductNode` has no single-value
# method for these orders, so each element gets the `Inf` catch-all and storing it in `b`
# throws. The order-3 ⋅ order-2 method also takes a `Tensor{2}` buffer, although that
# product (and its `return_type`) is of order 3.
@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{4,M},
    T2<:Tensor{2,M},T3<:Tensor{4,M}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{4,M},T3<:Tensor{4,M}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {N,T1<:Vec{N},T2<:Tensor{3,N},T3<:Tensor{2,N}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {N,T2<:Vec{N},T1<:Tensor{3,N},T3<:Tensor{2,N}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {N,T1<:Tensor{2,N},T2<:Tensor{3,N},T3<:Tensor{3,N}}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::DotProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {N,T2<:Tensor{2,N},T1<:Tensor{3,N},T3<:Tensor{2,N}}
    @inbounds b .= l.(x, y)
    return b
end

"""
    (::DoubleContractionNode)(x, y)
    (::DoubleContractionNode)(x, y, b)

Double contraction `x : y` (`dcontract`) of two `Tensor`s of order two to four with the
same spatial dimension, in every pairing except order three with order three. Batched:
the same pairings over columns, where one operand may be a single tensor.
"""
function (l::DoubleContractionNode)(x::T1, y::T2) where {M,T1<:Tensor{2,M},T2<:Tensor{2,M}}
    return dcontract(x, y)
end

function (l::DoubleContractionNode)(x::T1, y::T2) where {M,T1<:Tensor{2,M},T2<:Tensor{4,M}}
    return dcontract(x, y)
end

function (l::DoubleContractionNode)(x::T1, y::T2) where {M,T1<:Tensor{4,M},T2<:Tensor{2,M}}
    return dcontract(x, y)
end

function (l::DoubleContractionNode)(x::T1, y::T2) where {M,T1<:Tensor{2,M},T2<:Tensor{3,M}}
    return dcontract(x, y)
end

function (l::DoubleContractionNode)(x::T1, y::T2) where {M,T1<:Tensor{3,M},T2<:Tensor{2,M}}
    return dcontract(x, y)
end

function (l::DoubleContractionNode)(x::T1, y::T2) where {M,T1<:Tensor{3,M},T2<:Tensor{4,M}}
    return dcontract(x, y)
end

function (l::DoubleContractionNode)(x::T1, y::T2) where {M,T1<:Tensor{4,M},T2<:Tensor{3,M}}
    return dcontract(x, y)
end

function (l::DoubleContractionNode)(x::T1, y::T2) where {M,T1<:Tensor{4,M},T2<:Tensor{4,M}}
    return dcontract(x, y)
end

# Batched, column : column
function (l::DoubleContractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{2,M},T3<:Number}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{4,M},T3<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{4,M},T2<:Tensor{2,M},T3<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{3,M},T3<:Vec{M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{3,M},T2<:Tensor{2,M},T3<:Vec{M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{3,M},T2<:Tensor{4,M},T3<:Tensor{3,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{4,M},T2<:Tensor{3,M},T3<:Tensor{3,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{4,M},T2<:Tensor{4,M},T3<:Tensor{4,M}}
    @inbounds b .= l.(x, y)
    return b
end

# Batched, column : single tensor
function (l::DoubleContractionNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{2,M},T3<:Number}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{4,M},T3<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,T1<:Tensor{4,M},T2<:Tensor{2,M},T3<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{3,M},T3<:Vec{M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,T1<:Tensor{3,M},T2<:Tensor{2,M},T3<:Vec{M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,T1<:Tensor{3,M},T2<:Tensor{4,M},T3<:Tensor{3,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,T1<:Tensor{4,M},T2<:Tensor{3,M},T3<:Tensor{3,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::Vector{T1}, y::T2, b::Vector{T3}) where {M,T1<:Tensor{4,M},T2<:Tensor{4,M},T3<:Tensor{4,M}}
    @inbounds b .= l.(x, y)
    return b
end

# Batched, single tensor : column
function (l::DoubleContractionNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{2,M},T3<:Number}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{4,M},T3<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{4,M},T2<:Tensor{2,M},T3<:Tensor{2,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{2,M},T2<:Tensor{3,M},T3<:Vec{M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{3,M},T2<:Tensor{2,M},T3<:Vec{M}}
    # TODO: computes nothing; `b` is returned unchanged.
    return b
end

function (l::DoubleContractionNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{3,M},T2<:Tensor{4,M},T3<:Tensor{3,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{4,M},T2<:Tensor{3,M},T3<:Tensor{3,M}}
    @inbounds b .= l.(x, y)
    return b
end

function (l::DoubleContractionNode)(x::T1, y::Vector{T2}, b::Vector{T3}) where {M,T1<:Tensor{4,M},T2<:Tensor{4,M},T3<:Tensor{4,M}}
    @inbounds b .= l.(x, y)
    return b
end


"""
    (::PowerNode)(x, y)
    (::PowerNode)(x, y, b)

`x ^ y` for two scalars. Batched: over scalar columns, where one operand may be a single
scalar.
"""
@inline function (l::PowerNode)(x::T1,
    y::T2) where {T1<:Number,T2<:Number}
    return (x^y)
end

@inline function (l::PowerNode)(x::Vector{T1},
    y::T2, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::PowerNode)(x::T1,
    y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::PowerNode)(x::Vector{T1},
    y::Vector{T2}, b::Vector{T1}) where {T1<:Number,T2<:Number}
    @inbounds b .= l.(x, y)
    return b
end



"""
    (::MinNode)(x, y)
    (::MinNode)(x, y, b)

`min(x, y)` for two scalars of the same type. Batched: over scalar columns, where one
operand may be a single scalar.
"""
@inline function (l::MinNode)(x::T1,
    y::T1) where {T1<:Number}
    return min(x, y)
end

@inline function (l::MinNode)(x::Vector{T1},
    y::T1, b::Vector{T1}) where {T1<:Number}
    b .= l.(x, y)
    return b
end

@inline function (l::MinNode)(x::T1,
    y::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    b .= l.(x, y)
    return b
end


@inline function (l::MinNode)(x::Vector{T1},
    y::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    b .= l.(x, y)
    return b
end




"""
    (::MaxNode)(x, y)
    (::MaxNode)(x, y, b)

`max(x, y)` for two scalars of the same type. Batched: over scalar columns, where one
operand may be a single scalar.
"""
@inline function (l::MaxNode)(x::T1,
    y::T1) where {T1<:Number}
    return max(x, y)
end

@inline function (l::MaxNode)(x::Vector{T1},
    y::T1, b::Vector{T1}) where {T1<:Number}
    b .= l.(x, y)
    return b
end

@inline function (l::MaxNode)(x::T1,
    y::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    b .= l.(x, y)
    return b
end


@inline function (l::MaxNode)(x::Vector{T1},
    y::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    b .= l.(x, y)
    return b
end



"""
    (::SqrtNode)(x)
    (::SqrtNode)(x, b)

`sqrt(x)` for a scalar. Batched: over a scalar column.
"""
@inline function (l::SqrtNode)(x::T1) where {T1<:Number}
    return sqrt(x)
end

@inline function (l::SqrtNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    @inbounds b .= l.(x)
    return b
end


"""
    (::LogNode)(x)
    (::LogNode)(x, b)

Natural logarithm `log(x)` for a scalar. Batched: over a scalar column.
"""
@inline function (l::LogNode)(x::T1) where {T1<:Number}
    return log(x)
end

@inline function (l::LogNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    @inbounds b .= l.(x)
    return b
end

"""
    (::ExpNode)(x)
    (::ExpNode)(x, b)

`exp(x)` for a scalar. Batched: over a scalar column.
"""
@inline function (l::ExpNode)(x::T1) where {T1<:Number}
    return exp(x)
end

@inline function (l::ExpNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    @inbounds b .= l.(x)
    return b
end

"""
    (::SinNode)(x)
    (::SinNode)(x, b)

`sin(x)` for a scalar. Batched: over a scalar column.
"""
@inline function (l::SinNode)(x::T1) where {T1<:Number}
    return sin(x)
end

@inline function (l::SinNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    @inbounds b .= l.(x)
    return b
end



"""
    (::CosNode)(x)
    (::CosNode)(x, b)

`cos(x)` for a scalar. Batched: over a scalar column.
"""
@inline function (l::CosNode)(x::T1) where {T1<:Number}
    return cos(x)
end

@inline function (l::CosNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    @inbounds b .= l.(x)
    return b
end


"""
    (::TanNode)(x)
    (::TanNode)(x, b)

`tan(x)` for a scalar. Batched: over a scalar column.
"""
@inline function (l::TanNode)(x::T1) where {T1<:Number}
    return tan(x)
end

@inline function (l::TanNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:Number}
    @inbounds b .= l.(x)
    return b
end


"""
    (::NormNode)(x)
    (::NormNode)(x, b)

`norm(x)`: the absolute value of a scalar, the Frobenius norm of a `Tensor`. Batched: over
a column of either, into a scalar column.
"""
@inline function (l::NormNode)(x::T1) where {T1<:Union{<:Tensor,Number}}
    return norm(x)
end

@inline function (l::NormNode)(x::Vector{T1}, b::Vector{T2}) where {T1<:Union{<:Tensor,<:Number},T2<:Number}
    @inbounds b .= l.(x)
    return b
end



"""
    (::TdotNode)(x)
    (::TdotNode)(x, b)

`x ⋅ xᵀ` for a second-order `Tensor` `x`: the same product as `DottNode`, whereas
Tensors.jl's `tdot(x)` is `xᵀ ⋅ x`. Batched: over a column, into a buffer of the same
element type.
"""
@inline function (l::TdotNode)(x::T1) where {T1<:Tensor}
    return dot(x, x')
end

@inline function (l::TdotNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:Tensor}
    @inbounds b .= l.(x)
    return b
end

"""
    (::LapNode)(x)
    (::LapNode)(x, b)

`I : x` for a third-order `Tensor` `x` of spatial dimension 2 or 3: the double contraction
of the identity with the first two indices of `x`, giving a `Vec`. Batched: over a column
of such tensors.
"""
@inline function (l::LapNode)(x::T1) where {N,T1<:Tensor{3,N}}
    return dcontract(UNIT_TENSOR[N], x)
end

@inline function (l::LapNode)(x::Vector{T1}, b::Vector{T2}) where {N,T1<:Tensor{3,N},T2<:Vec{N}}
    @inbounds b .= l.(x)
    return b
end

"""
    (::TraceNode)(x)
    (::TraceNode)(x, b)

`tr(x)` for a second-order tensor. Batched: over a column, into a scalar column.
"""
@inline function (l::TraceNode)(x::T1) where {T1<:SecondOrderTensor}
    return tr(x)
end

@inline function (l::TraceNode)(x::Vector{T1}, b::Vector{T2}) where {T1<:SecondOrderTensor,T2<:Number}
    @inbounds b .= l.(x)
    return b
end

"""
    (::DeterminantNode)(x)
    (::DeterminantNode)(x, b)

`det(x)` for a second-order tensor. Batched: over a column, into a scalar column.
"""
@inline function (l::DeterminantNode)(x::T1) where {T1<:SecondOrderTensor}
    return det(x)
end

@inline function (l::DeterminantNode)(x::Vector{T1}, b::Vector{T2}) where {T1<:SecondOrderTensor,T2<:Number}
    @inbounds b .= l.(x)
    return b
end

"""
    (::InversionNode)(x)
    (::InversionNode)(x, b)

`inv(x)` for a second-order tensor. Batched: over a column, into a buffer of the same
element type.
"""
@inline function (l::InversionNode)(x::T1) where {T1<:SecondOrderTensor}
    return inv(x)
end

@inline function (l::InversionNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:SecondOrderTensor}
    @inbounds b .= l.(x)
    return b
end

"""
    (::SymmetricNode)(x)
    (::SymmetricNode)(x, b)

`symmetric(x)`, the symmetric part of a second-order tensor or the minor-symmetric part of a
fourth-order one. Batched: over a column, into a buffer of the same element type.
"""
@inline function (l::SymmetricNode)(x::T1) where {T1<:Union{SecondOrderTensor,FourthOrderTensor}}
    return symmetric(x)
end

@inline function (l::SymmetricNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:Union{SecondOrderTensor,FourthOrderTensor}}
    @inbounds b .= l.(x)
    return b
end

"""
    (::SkewNode)(x)
    (::SkewNode)(x, b)

`skew(x)`, the skew-symmetric part of a second-order tensor. Batched: over a column, into a
buffer of the same element type.
"""
@inline function (l::SkewNode)(x::T1) where {T1<:SecondOrderTensor}
    return skew(x)
end

@inline function (l::SkewNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:SecondOrderTensor}
    @inbounds b .= l.(x)
    return b
end

"""
    (::DeviatoricNode)(x)
    (::DeviatoricNode)(x, b)

`dev(x)`, the deviatoric part of a second-order tensor. Batched: over a column, into a
buffer of the same element type.
"""
@inline function (l::DeviatoricNode)(x::T1) where {T1<:SecondOrderTensor}
    return dev(x)
end

@inline function (l::DeviatoricNode)(x::Vector{T1}, b::Vector{T1}) where {T1<:SecondOrderTensor}
    @inbounds b .= l.(x)
    return b
end

"""
    (::CrossProductNode)(x, y)
    (::CrossProductNode)(x, y, b)

Cross product `x × y` of two `Vec`s. Tensors.jl returns a `Vec{3}` for any input
dimension, and the batched form writes into a buffer of the element type of `x`, so it
works only on `Vec{3}` columns.
"""
@inline function (l::CrossProductNode)(x::T1, y::T2) where {T1<:Vec,T2<:Vec}
    return cross(x, y)
end

@inline function (l::CrossProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T1}
) where {T1<:Vec,T2<:Vec}
    @inbounds b .= l.(x, y)
    return b
end


"""
    (::OuterProductNode)(x, y)
    (::OuterProductNode)(x, y, b)

Outer product `x ⊗ y` (`otimes`) of two `Vec`s, giving a second-order tensor, or of two
second-order tensors, giving a fourth-order one. Batched: over columns.
"""
@inline function (l::OuterProductNode)(x::Vec, y::Vec)
    return otimes(x, y)
end

@inline function (l::OuterProductNode)(x::SecondOrderTensor, y::SecondOrderTensor)
    return otimes(x, y)
end

@inline function (l::OuterProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {T1<:Vec,
    T2<:Vec,T3<:SecondOrderTensor}
    @inbounds b .= l.(x, y)
    return b
end

@inline function (l::OuterProductNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T3}) where {T1<:SecondOrderTensor,T2<:SecondOrderTensor,T3<:FourthOrderTensor}
    @inbounds b .= l.(x, y)
    return b
end


"""
    (::HadamardNode)(x, y)
    (::HadamardNode)(x, y, b)

Elementwise product `x ∘ y` of two `Tensor`s of the same order and spatial dimension.
Batched: over columns, where one operand may be a single tensor.
"""
@inline function (l::HadamardNode)(x::T1, y::T2) where {M,N,T1<:Tensor{M,N},
    T2<:Tensor{M,N}}
    return x ∘ y
end

@inline function (l::HadamardNode)(x::Vector{T1}, y::Vector{T2}, b::Vector{T1}) where {M,N,T1<:Tensor{M,N},
    T2<:Tensor{M,N}}
    @inbounds b .= l.(x, y)
    return b
end


@inline function (l::HadamardNode)(x::T1, y::Vector{T2}, b::Vector{T1}) where {M,N,T1<:Tensor{M,N},
    T2<:Tensor{M,N}}
    @inbounds b .= l.(x, y)
    return b
end


@inline function (l::HadamardNode)(x::Vector{T1}, y::T2, b::Vector{T1}) where {M,N,T1<:Tensor{M,N},
    T2<:Tensor{M,N}}
    @inbounds b .= l.(x, y)
    return b
end


"""
    (::VolumetricNode)(x)
    (::VolumetricNode)(x, b)

`vol(x)`, the volumetric part of a second-order tensor. Batched: over a column.
"""
@inline function (l::VolumetricNode)(x::T1) where {T1<:SecondOrderTensor}
    return vol(x)
end

@inline function (l::VolumetricNode)(x::Vector{T1}, b::Vector{T2}) where {N,T1<:SecondOrderTensor{N},T2<:SecondOrderTensor{N}}
    @inbounds b .= l.(x)
    return b
end

"""
    (::DottNode)(x)
    (::DottNode)(x, b)

`dott(x)`, that is `x ⋅ xᵀ`, for a second-order tensor. Batched: over a column.
"""
@inline function (l::DottNode)(x::T1) where {N,T1<:SecondOrderTensor{N}}
    return dott(x)
end

@inline function (l::DottNode)(x::Vector{T1}, b::Vector{T2}) where {N,T1<:SecondOrderTensor{N},T2<:SecondOrderTensor{N}}
    @inbounds b .= l.(x)
    return b
end


"""
    return_type(op, T) -> Type
    return_type(op, T1, T2) -> Type

Type of the result of `op` applied to operands of type `T` (or `T1`, `T2`).
`calc_stack_batch_tensor` chooses the buffer pool for a result by it, and `compile_program`
tests closure with it. Tensor column results have abstract element types, e.g.
`Vector{Tensor{2,N}}`, matching the pool keys of `allocate_buffers!`. Operand types without
a method of their own give `Float64`, which no pool is keyed on (pools hold columns), so
the evaluator rejects the operation; `ConstantNode` and `UnaryNode` always give `Float64`.
"""
@inline return_type(::ConstantNode, ::Type) = Float64

@inline return_type(::UnaryNode, ::Type) = Float64

@inline return_type(::TraceNode, ::Type{T1}) where {N,T1<:SecondOrderTensor{N}} = Float64
@inline return_type(::TraceNode, ::Type{Vector{T1}}) where {N,T1<:SecondOrderTensor{N}} = Vector{eltype(T1)}
@inline return_type(::TraceNode, ::Type) = Float64

@inline return_type(::DeviatoricNode, ::Type{T1}) where {N,T1<:SecondOrderTensor{N}} = Tensor{2,N}
@inline return_type(::DeviatoricNode, ::Type{Vector{T1}}) where {N,T1<:SecondOrderTensor{N}} = Vector{Tensor{2,N}}
@inline return_type(::DeviatoricNode, ::Type) = Float64

@inline return_type(::VolumetricNode, ::Type{T1}) where {N,T1<:SecondOrderTensor{N}} = Tensor{2,N}
@inline return_type(::VolumetricNode, ::Type{Vector{T1}}) where {N,T1<:SecondOrderTensor{N}} = Vector{Tensor{2,N}}
@inline return_type(::VolumetricNode, ::Type) = Float64

@inline return_type(::SkewNode, ::Type{T1}) where {N,T1<:SecondOrderTensor{N}} = Tensor{2,N}
@inline return_type(::SkewNode, ::Type{Vector{T1}}) where {N,T1<:SecondOrderTensor{N}} = Vector{Tensor{2,N}}
@inline return_type(::SkewNode, ::Type) = Float64

@inline return_type(::SymmetricNode, ::Type{T1}) where {N,T1<:Tensor{2,N}} = Tensor{2,N}
@inline return_type(::SymmetricNode, ::Type{T1}) where {N,T1<:Tensor{4,N}} = Tensor{4,N}
@inline return_type(::SymmetricNode, ::Type{Vector{T1}}) where {N,T1<:Tensor{2,N}} = Vector{Tensor{2,N}}
@inline return_type(::SymmetricNode, ::Type{Vector{T1}}) where {N,T1<:Tensor{4,N}} = Vector{Tensor{4,N}}
@inline return_type(::SymmetricNode, ::Type) = Float64

@inline return_type(::InversionNode, ::Type{T1}) where {N,T1<:SecondOrderTensor{N}} = Tensor{2,N}
@inline return_type(::InversionNode, ::Type{Vector{T1}}) where {N,T1<:SecondOrderTensor{N}} = Vector{Tensor{2,N}}
@inline return_type(::InversionNode, ::Type) = Float64

@inline return_type(::CrossProductNode, ::Type{T1}, ::Type{T2}) where {N,T1<:Vec{N},T2<:Vec{N}} = Vec{N}
@inline return_type(::CrossProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Vec{N},T2<:Vec{N}} = Vector{Vec{N}}
@inline return_type(::CrossProductNode, ::Type, ::Type) = Float64

# TODO: no methods for the (2,4), (3,4) and (4,3) order pairings, which the kernels
# support, so the evaluator rejects them.
@inline return_type(::DoubleContractionNode, ::Type{T1}, ::Type{T2}) where {N,T1<:SecondOrderTensor{N},T2<:SecondOrderTensor{N}} = Float64
@inline return_type(::DoubleContractionNode, ::Type{T1}, ::Type{T2}) where {N,T1<:Tensor{4,N},T2<:Tensor{2,N}} = Tensor{2,N}
@inline return_type(::DoubleContractionNode, ::Type{T1}, ::Type{T2}) where {N,T1<:SymmetricTensor{4,N},T2<:Tensor{2,N}} = Tensor{2,N}
@inline return_type(::DoubleContractionNode, ::Type{T1}, ::Type{T2}) where {N,T1<:Tensor{4,N},T2<:SymmetricTensor{2,N}} = Tensor{2,N}
@inline return_type(::DoubleContractionNode, ::Type{T1}, ::Type{T2}) where {N,T1<:SymmetricTensor{4,N},T2<:SymmetricTensor{2,N}} = Tensor{2,N}
@inline return_type(::DoubleContractionNode, ::Type{T1}, ::Type{T2}) where {N,T1<:Tensor{4,N},T2<:Tensor{4,N}} = Tensor{4,N}
@inline return_type(::DoubleContractionNode, ::Type{T1}, ::Type{T2}) where {N,T1<:SymmetricTensor{4,N},T2<:SymmetricTensor{4,N}} = Tensor{4,N}
@inline return_type(::DoubleContractionNode, ::Type{T1}, ::Type{T2}) where {N,T1<:Tensor{2,N},T2<:Tensor{3,N}} = Vec{N}
@inline return_type(::DoubleContractionNode, ::Type{T1}, ::Type{T2}) where {N,T1<:Tensor{3,N},T2<:Tensor{2,N}} = Vec{N}
@inline return_type(::DoubleContractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:SecondOrderTensor{N},T2<:SecondOrderTensor{N}} = Vector{Float64}
@inline return_type(::DoubleContractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Tensor{4,N},T2<:Tensor{2,N}} = Vector{Tensor{2,N}}
@inline return_type(::DoubleContractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:SymmetricTensor{4,N},T2<:Tensor{2,N}} = Vector{Tensor{2,N}}
@inline return_type(::DoubleContractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Tensor{4,N},T2<:SymmetricTensor{2,N}} = Vector{Tensor{2,N}}
@inline return_type(::DoubleContractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:SymmetricTensor{4,N},T2<:SymmetricTensor{2,N}} = Vector{Tensor{2,N}}
@inline return_type(::DoubleContractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Tensor{4,N},T2<:Tensor{4,N}} = Vector{Tensor{4,N}}
@inline return_type(::DoubleContractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:SymmetricTensor{4,N},T2<:SymmetricTensor{4,N}} = Vector{Tensor{4,N}}
@inline return_type(::DoubleContractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Tensor{2,N},T2<:Tensor{3,N}} = Vector{Vec{N}}
@inline return_type(::DoubleContractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Tensor{3,N},T2<:Tensor{2,N}} = Vector{Vec{N}}
@inline return_type(::DoubleContractionNode, ::Type, ::Type) = Float64

@inline return_type(::DotProductNode, ::Type{T1}, ::Type{T2}) where {N,T1<:Vec{N},T2<:Vec{N}} = Float64
@inline return_type(::DotProductNode, ::Type{T1}, ::Type{T2}) where {N,T1<:Vec{N},T2<:SecondOrderTensor{N}} = Vec{N}
@inline return_type(::DotProductNode, ::Type{T1}, ::Type{T2}) where {N,T1<:SecondOrderTensor{N},T2<:Vec{N}} = Vec{N}
@inline return_type(::DotProductNode, ::Type{T1}, ::Type{T2}) where {N,T1<:SecondOrderTensor{N},T2<:SecondOrderTensor{N}} = Tensor{2,N}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Vec{N},T2<:Vec{N}} = Vector{Float64}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Vec{N},T2<:SecondOrderTensor{N}} = Vector{Vec{N}}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:SecondOrderTensor{N},T2<:Vec{N}} = Vector{Vec{N}}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:SecondOrderTensor{N},T2<:SecondOrderTensor{N}} = Vector{Tensor{2,N}}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:FourthOrderTensor{N},T2<:SecondOrderTensor{N}} = Vector{Tensor{4,N}}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T2<:FourthOrderTensor{N},T1<:SecondOrderTensor{N}} = Vector{Tensor{4,N}}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Vec{N},T2<:Tensor{3,N}} = Vector{Tensor{2,N}}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T2<:Vec{N},T1<:Tensor{3,N}} = Vector{Tensor{2,N}}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:SecondOrderTensor{N},T2<:Tensor{3,N}} = Vector{Tensor{3,N}}
@inline return_type(::DotProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T2<:SecondOrderTensor{N},T1<:Tensor{3,N}} = Vector{Tensor{3,N}}
@inline return_type(::DotProductNode, ::Type, ::Type) = Float64

@inline return_type(::PowerNode, ::Type{T1}, ::Type{T2}) where {T1<:Number,T2<:Number} = T1
@inline return_type(::PowerNode, ::Type{Vector{T1}}, ::Type{T2}) where {T1<:Number,T2<:Number} = Vector{T1}
@inline return_type(::PowerNode, ::Type{T1}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{T1}
@inline return_type(::PowerNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{T1}
@inline return_type(::PowerNode, ::Type, ::Type) = Float64

@inline return_type(::DivisionNode, ::Type{T1}, ::Type{T2}) where {N,M,T1<:Tensor{N,M},T2<:Number} = Tensor{N,M}
@inline return_type(::DivisionNode, ::Type{T1}, ::Type{T2}) where {T1<:Number,T2<:Number} = Float64
@inline return_type(::DivisionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,M,T1<:Tensor{N,M},T2<:Number} = Vector{Tensor{N,M}}
@inline return_type(::DivisionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::DivisionNode, ::Type{Vector{T1}}, ::Type{T2}) where {N,M,T1<:Tensor{N,M},T2<:Number} = Vector{Tensor{N,M}}
@inline return_type(::DivisionNode, ::Type{Vector{T1}}, ::Type{T2}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::DivisionNode, ::Type{T1}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::DivisionNode, ::Type, ::Type) = Float64

# TODO: on scalar columns the result is `Vector{Float64}` whatever the operands' element
# type, as for `+`, `-`, `/`, `min` and `max`; columns of another float type find no pool.
@inline return_type(::MultiplicationNode, ::Type{T1}, ::Type{T2}) where {T1<:Number,T2<:Number} = Float64
@inline return_type(::MultiplicationNode, ::Type{Vector{T1}}, ::Type{T2}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::MultiplicationNode, ::Type{T1}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::MultiplicationNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::MultiplicationNode, ::Type{T1}, ::Type{T2}) where {N,M,T1<:Tensor{N,M},T2<:Number} = Tensor{N,M}
@inline return_type(::MultiplicationNode, ::Type{Vector{T1}}, ::Type{T2}) where {N,M,T1<:Tensor{N,M},T2<:Number} = Vector{Tensor{N,M}}
@inline return_type(::MultiplicationNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,M,T1<:Tensor{N,M},T2<:Number} = Vector{Tensor{N,M}}
@inline return_type(::MultiplicationNode, ::Type{Vector{T2}}, ::Type{Vector{T1}}) where {N,M,T1<:Tensor{N,M},T2<:Number} = Vector{Tensor{N,M}}
@inline return_type(::MultiplicationNode, ::Type{T2}, ::Type{T1}) where {N,M,T1<:Tensor{N,M},T2<:Number} = Tensor{N,M}
@inline return_type(::MultiplicationNode, ::Type{T2}, ::Type{Vector{T1}}) where {N,M,T1<:Tensor{N,M},T2<:Number} = Vector{Tensor{N,M}}
@inline return_type(::MultiplicationNode, ::Type, ::Type) = Float64

@inline return_type(::HadamardNode, ::Type{T1}, ::Type{T2}) where {N,M,T1<:Tensor{N,M},T2<:Tensor{N,M}} = Tensor{N,M}
@inline return_type(::HadamardNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,M,T1<:Tensor{N,M},T2<:Tensor{N,M}} = Vector{Tensor{N,M}}
@inline return_type(::HadamardNode, ::Type{Vector{T1}}, ::Type{T2}) where {N,M,T1<:Tensor{N,M},T2<:Tensor{N,M}} = Vector{Tensor{N,M}}
@inline return_type(::HadamardNode, ::Type{T1}, ::Type{Vector{T2}}) where {N,M,T1<:Tensor{N,M},T2<:Tensor{N,M}} = Vector{Tensor{N,M}}
@inline return_type(::HadamardNode, ::Type, ::Type) = Float64

@inline return_type(::SubtractionNode, ::Type{T1}, ::Type{T2}) where {T1<:Number,T2<:Number} = Float64
@inline return_type(::SubtractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::SubtractionNode, ::Type{T1}, ::Type{T2}) where {M,N,T1<:Tensor{M,N},T2<:Tensor{M,N}} = Tensor{M,N}
@inline return_type(::SubtractionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {M,N,T1<:Tensor{M,N},T2<:Tensor{M,N}} = Vector{Tensor{M,N}}
@inline return_type(::SubtractionNode, ::Type, ::Type) = Float64

@inline return_type(::AdditionNode, ::Type{T1}, ::Type{T2}) where {T1<:Number,T2<:Number} = Float64
@inline return_type(::AdditionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::AdditionNode, ::Type{T1}, ::Type{T2}) where {M,N,T1<:Tensor{M,N},T2<:Tensor{M,N}} = Tensor{M,N}
@inline return_type(::AdditionNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {M,N,T1<:Tensor{M,N},T2<:Tensor{M,N}} = Vector{Tensor{M,N}}
@inline return_type(::AdditionNode, ::Type, ::Type) = Float64

@inline return_type(::LapNode, ::Type{T1}) where {N,T1<:Tensor{3,N}} = Vec{N}
@inline return_type(::LapNode, ::Type{Vector{T1}}) where {N,T1<:Tensor{3,N}} = Vector{Vec{N}}
@inline return_type(::LapNode, ::Type) = Float64

@inline return_type(::OuterProductNode, ::Type{T1}, ::Type{T2}) where {N,T1<:Vec{N},T2<:Vec{N}} = Tensor{2,N}
@inline return_type(::OuterProductNode, ::Type{T1}, ::Type{T2}) where {N,T1<:SecondOrderTensor{N},T2<:SecondOrderTensor{N}} = Tensor{4,N}
@inline return_type(::OuterProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:Vec{N},T2<:Vec{N}} = Vector{Tensor{2,N}}
@inline return_type(::OuterProductNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {N,T1<:SecondOrderTensor{N},T2<:SecondOrderTensor{N}} = Vector{Tensor{4,N}}
@inline return_type(::OuterProductNode, ::Type, ::Type) = Float64

@inline return_type(::NormNode, ::Type{T1}) where {T1<:AbstractFloat} = T1
@inline return_type(::NormNode, ::Type{T1}) where {N,M,T1<:Tensor{N,M}} = eltype(T1)
@inline return_type(::NormNode, ::Type{Vector{T1}}) where {T1<:AbstractFloat} = Vector{eltype(T1)}
@inline return_type(::NormNode, ::Type{Vector{T1}}) where {N,M,T1<:Tensor{N,M}} = Vector{eltype(T1)}
@inline return_type(::NormNode, ::Type) = Float64

@inline return_type(::SqrtNode, ::Type{T1}) where {T1<:AbstractFloat} = T1
@inline return_type(::SqrtNode, ::Type{Vector{T1}}) where {T1<:AbstractFloat} = Vector{eltype(T1)}
@inline return_type(::SqrtNode, ::Type) = Float64

@inline return_type(::DeterminantNode, ::Type{T1}) where {N,T1<:SecondOrderTensor{N}} = Float64
@inline return_type(::DeterminantNode, ::Type{Vector{T1}}) where {N,T1<:SecondOrderTensor{N}} = Vector{Float64}
@inline return_type(::DeterminantNode, ::Type) = Float64

@inline return_type(::TdotNode, ::Type{T1}) where {N,T1<:SecondOrderTensor{N}} = Tensor{2,N}
@inline return_type(::TdotNode, ::Type{Vector{T1}}) where {N,T1<:SecondOrderTensor{N}} = Vector{Tensor{2,N}}
@inline return_type(::TdotNode, ::Type) = Float64

@inline return_type(::DottNode, ::Type{T1}) where {N,T1<:SecondOrderTensor{N}} = Tensor{2,N}
@inline return_type(::DottNode, ::Type{Vector{T1}}) where {N,T1<:SecondOrderTensor{N}} = Vector{Tensor{2,N}}
@inline return_type(::DottNode, ::Type) = Float64

@inline return_type(::MinNode, ::Type{T1}, ::Type{T2}) where {T1<:Number,T2<:Number} = Float64
@inline return_type(::MinNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::MinNode, ::Type{Vector{T1}}, ::Type{T2}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::MinNode, ::Type{T1}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::MinNode, ::Type, ::Type) = Float64

@inline return_type(::MaxNode, ::Type{T1}, ::Type{T2}) where {T1<:Number,T2<:Number} = Float64
@inline return_type(::MaxNode, ::Type{Vector{T1}}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::MaxNode, ::Type{Vector{T1}}, ::Type{T2}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::MaxNode, ::Type{T1}, ::Type{Vector{T2}}) where {T1<:Number,T2<:Number} = Vector{Float64}
@inline return_type(::MaxNode, ::Type, ::Type) = Float64

@inline return_type(::ExpNode, ::Type{T1}) where {T1<:AbstractFloat} = T1
@inline return_type(::ExpNode, ::Type{Vector{T1}}) where {T1<:AbstractFloat} = Vector{eltype(T1)}
@inline return_type(::ExpNode, ::Type) = Float64

@inline return_type(::LogNode, ::Type{T1}) where {T1<:AbstractFloat} = T1
@inline return_type(::LogNode, ::Type{Vector{T1}}) where {T1<:AbstractFloat} = Vector{eltype(T1)}
@inline return_type(::LogNode, ::Type) = Float64

@inline return_type(::SinNode, ::Type{T1}) where {T1<:AbstractFloat} = T1
@inline return_type(::SinNode, ::Type{Vector{T1}}) where {T1<:AbstractFloat} = Vector{eltype(T1)}
@inline return_type(::SinNode, ::Type) = Float64

@inline return_type(::CosNode, ::Type{T1}) where {T1<:AbstractFloat} = T1
@inline return_type(::CosNode, ::Type{Vector{T1}}) where {T1<:AbstractFloat} = Vector{eltype(T1)}
@inline return_type(::CosNode, ::Type) = Float64

@inline return_type(::TanNode, ::Type{T1}) where {T1<:AbstractFloat} = T1
@inline return_type(::TanNode, ::Type{Vector{T1}}) where {T1<:AbstractFloat} = Vector{eltype(T1)}
@inline return_type(::TanNode, ::Type) = Float64



"""
    EvalScratch

The operand stack and buffer counters of `calc_stack_batch_tensor`, reused across calls
instead of allocated per call. `EVAL_SCRATCH` holds one per thread id (see `__init__`).
"""
mutable struct EvalScratch
    stack::Vector{Any}
    counters::Dict{Type,Int}
end

let
    global EVAL_SCRATCH = [EvalScratch(Any[], Dict{Type,Int}()) for _ in 1:50]
end

# `EVAL_SCRATCH` is built at precompile time, so its length cannot follow the session's
# thread count. `threadid()` runs up to `maxthreadid()`, which exceeds `nthreads()` when
# interactive threads exist, so at load time the vector is extended to `maxthreadid()`
# entries if shorter (`nthreads()` where `maxthreadid` is not defined).
function __init__()
    n = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads()
    while length(EVAL_SCRATCH) < n
        push!(EVAL_SCRATCH, EvalScratch(Any[], Dict{Type,Int}()))
    end
end

"""
    calc_stack_batch_tensor(rek_string, callbacks, nodes, buffers)

Evaluate the karva string `rek_string` over whole data columns, right to left as a stack
machine: a symbol in `callbacks` (symbol => operator node) pops its operands, top of stack
first, and pushes its result; any other symbol pushes its column from `nodes` (symbol =>
data column). With `buffers`, this thread's pools of preallocated columns keyed by column
type, each result goes into the next unused buffer of type `return_type(op, ...)`. With
`buffers === nothing` results are freshly allocated, and operands an operator does not
support yield `Inf` entries.

Returns the column left on top of the stack, or `NaN` when a result type has no pool or a
terminal's `norm` is not finite. The result may alias a buffer or a column of `nodes`, so
it must be used or copied before the next evaluation with the same buffers. Throws on a
malformed karva string: an unknown symbol, too few operands, more operators than buffers.
"""
@inline function calc_stack_batch_tensor(rek_string::Vector,
    callbacks::Dict, nodes::Dict, buffers::Union{Dict{Type,NTuple},Nothing})
    scratch = EVAL_SCRATCH[Threads.threadid()]
    stack = scratch.stack
    used_buffers = scratch.counters
    empty!(stack)
    empty!(used_buffers)
    if !isnothing(buffers)
        for T in keys(buffers)
            used_buffers[T] = 1
        end
    end
    @inbounds for elem in Iterators.reverse(rek_string)
        ops = get(callbacks, elem, elem)
        ops = ops isa Int8 ? nodes[ops] : ops
        success = eval_op!(ops, stack, used_buffers, buffers)
        !(success) && return NaN
    end
    return last(stack)
end


# --------------------------------------------------------------------------------
#  Elementwise scalar functions
#
#  The functions of `FUNCTION_LIB_COMMON` that have no node above. Each is an elementwise
#  map over a scalar column, so one generated block defines them all. `sqr` is not a Base
#  function (it is `GepUtils.sqr`), so its body is written out as `x * x`.
# --------------------------------------------------------------------------------
const ELEMENTWISE_UNARY = (
    (:AbsNode, :abs), (:FloorNode, :floor), (:CeilNode, :ceil), (:RoundNode, :round),
    (:SignNode, :sign), (:Log10Node, :log10), (:Log2Node, :log2), (:SqrNode, :sqr),
    (:AsinNode, :asin), (:AcosNode, :acos), (:AtanNode, :atan),
    (:SinhNode, :sinh), (:CoshNode, :cosh), (:TanhNode, :tanh),
    (:AsinhNode, :asinh), (:AcoshNode, :acosh), (:AtanhNode, :atanh),
)

for (nodename, fname) in ELEMENTWISE_UNARY
    op = fname === :sqr ? :(x * x) : :($fname(x))
    @eval begin
        struct $nodename <: Arity1Node end
        @inline function (l::$nodename)(x::T1) where {T1<:Number}
            return $op
        end
        @inline function (l::$nodename)(x::Vector{T1}, b::Vector{T1}) where {T1<:Number}
            @inbounds b .= l.(x)
            return b
        end
        # Catch-alls as in `@generate_operation_node`: a non-scalar operand yields `Inf`.
        @inline function (l::$nodename)(x::Any)
            @fastmath Inf::Number
        end
        @inline function (l::$nodename)(x::Any, y::Any)
            @fastmath Inf::Number
        end
        @inline return_type(::$nodename, ::Type{T1}) where {T1<:AbstractFloat} = T1
        @inline return_type(::$nodename, ::Type{Vector{T1}}) where {T1<:AbstractFloat} =
            Vector{eltype(T1)}
        @inline return_type(::$nodename, ::Type) = Float64
        export $nodename
    end
end

# ---------------------------------------------------------------------------------
#  Monomorphic evaluator
#
#  `calc_stack_batch_tensor` keeps its operands on a `Vector{Any}` and resolves every
#  operator by dynamic dispatch. When all terminal columns and all intermediates have one
#  concrete type `V` -- in practice the scalar `Vector{Float64}` path -- that indirection
#  is unnecessary: `compile_program` turns the alphabet into flat opcode tables and
#  `run_program!` walks the karva string over a `Vector{V}` stack with the same order,
#  buffer sequence and kernels, so both give bit-identical results. What the tables cannot
#  represent makes `compile_program` return `nothing`, and callers then use
#  `calc_stack_batch_tensor`.
# ---------------------------------------------------------------------------------

# Opcodes follow tuple order, which is the order `apply2!`/`apply1!` test them in, so the
# four arithmetic operators, the most frequent in a search, come first.
const BINARY_OPS = (AdditionNode, SubtractionNode, MultiplicationNode, DivisionNode,
    PowerNode, MinNode, MaxNode, DotProductNode, DoubleContractionNode,
    CrossProductNode, OuterProductNode, HadamardNode)
const UNARY_OPS = (SqrtNode, ExpNode, LogNode, SinNode, CosNode, TanNode, NormNode,
    InversionNode, TraceNode, DeterminantNode, SymmetricNode, SkewNode,
    VolumetricNode, DeviatoricNode, TdotNode, DottNode, LapNode,
    AbsNode, FloorNode, CeilNode, RoundNode, SignNode, Log10Node, Log2Node,
    SqrNode, AsinNode, AcosNode, AtanNode, SinhNode, CoshNode, TanhNode,
    AsinhNode, AcoshNode, AtanhNode)

const BINARY_CODE = Dict{DataType,Int8}(T => Int8(i) for (i, T) in enumerate(BINARY_OPS))
const UNARY_CODE = Dict{DataType,Int8}(T => Int8(i) for (i, T) in enumerate(UNARY_OPS))

# `apply2!(code, a, b, out)` and `apply1!(code, a, out)` call operator number `code` of
# `BINARY_OPS`/`UNARY_OPS` through a chain of comparisons generated from the tuples; they
# return `true` once the operator has been called and `false` for an unknown code. Each
# operator sits behind its own `@noinline` barrier so that the `@inline` operator methods
# are not all inlined into one large chain function, which optimises poorly.
for (fname, ops, args) in ((:apply2!, BINARY_OPS, (:a, :b, :out)),
    (:apply1!, UNARY_OPS, (:a, :out)))
    body = :(return false)
    for (i, NT) in Iterators.reverse(collect(enumerate(ops)))
        barrier = Symbol(fname, :_, i)
        @eval @noinline function $barrier($(args...))
            $(Expr(:call, NT(), args...))
            return true
        end
        body = Expr(:if, :(code === $(Int8(i))),
            :(return $(Expr(:call, barrier, args...))), body)
    end
    @eval @inline function $fname(code::Int8, $(args...))
        $body
    end
end

"""
    EvalProgram{V}

An alphabet compiled by `compile_program` for `run_program!`. Each field is indexed by
`Int8` symbol id, so resolving a symbol is an array read rather than a `Dict` lookup:

* `kind` -- `0` terminal, `1` unary operator, `2` binary operator, `-1` unused id
* `code` -- the operator's index in `UNARY_OPS` or `BINARY_OPS` (`0` otherwise)
* `column` -- the terminal's data column (undefined for other ids)
"""
struct EvalProgram{V}
    kind::Vector{Int8}
    code::Vector{Int8}
    column::Vector{V}
end

"""
    compile_program(callbacks, nodes, ::Type{V}) -> Union{EvalProgram{V},Nothing}

Compile the alphabet given by `callbacks` (symbol => operator node) and `nodes` (symbol =>
data column) for `run_program!`. Returns `nothing` when `nodes` is empty, a symbol id is
below 1, a terminal column is not a `V`, or an operator is not in `BINARY_OPS`/`UNARY_OPS`
or not closed over `V` (`return_type` of `V` operands is not `V`). Closure is judged by
`return_type`, the function `calc_stack_batch_tensor` uses to pick a buffer pool. A binary
operator's `return_type` on two `V` columns is `V` only for `V = Vector{Float64}` (results
on scalar columns are `Vector{Float64}`, on tensor columns they have abstract element
types), so an alphabet with a binary operator compiles only for that `V`.
"""
function compile_program(callbacks::Dict, nodes::Dict, ::Type{V}) where {V}
    isempty(nodes) && return nothing
    maxsym = Int(max(maximum(keys(nodes)),
        isempty(callbacks) ? typemin(Int8) : maximum(keys(callbacks))))
    maxsym > 0 || return nothing

    kind = fill(Int8(-1), maxsym)
    code = zeros(Int8, maxsym)
    column = Vector{V}(undef, maxsym)

    for (sym, col) in nodes
        col isa V || return nothing
        1 <= sym <= maxsym || return nothing
        column[sym] = col
        kind[sym] = Int8(0)
    end

    for (sym, op) in callbacks
        1 <= sym <= maxsym || return nothing
        if op isa Arity2Node
            haskey(BINARY_CODE, typeof(op)) || return nothing
            return_type(op, V, V) === V || return nothing
            kind[sym] = Int8(2)
            code[sym] = BINARY_CODE[typeof(op)]
        elseif op isa Arity1Node
            haskey(UNARY_CODE, typeof(op)) || return nothing
            return_type(op, V) === V || return nothing
            kind[sym] = Int8(1)
            code[sym] = UNARY_CODE[typeof(op)]
        else
            return nothing
        end
    end
    return EvalProgram{V}(kind, code, column)
end

"""
    run_program!(rek_string, prog, buffers, stack) -> V or NaN

Evaluate the karva string `rek_string` with the compiled `prog`, as
`calc_stack_batch_tensor` does: right to left, operands popped top of stack first (a
binary operator is applied as `op(top, second)`), each operator's result written into the
next unused element of `buffers`. `stack` is scratch space, emptied on entry.

Returns the column left on top of the stack, which may alias an element of `buffers` or a
terminal column, or `NaN` for a malformed karva string: an unknown symbol, too few
operands, more operators than buffers, or an empty string. `calc_stack_batch_tensor`
throws in these cases instead.
"""
function run_program!(rek_string::AbstractVector{Int8}, prog::EvalProgram{V},
    buffers::Vector{V}, stack::Vector{V}) where {V}
    empty!(stack)
    nbuf = length(buffers)
    used = 0
    nsym = length(prog.kind)
    @inbounds for i in length(rek_string):-1:1
        sym = rek_string[i]
        (1 <= sym <= nsym) || return NaN
        k = prog.kind[sym]
        if k === Int8(0)
            push!(stack, prog.column[sym])
        elseif k === Int8(2)
            length(stack) >= 2 || return NaN
            used += 1
            used <= nbuf || return NaN
            op1 = pop!(stack)
            op2 = pop!(stack)
            out = buffers[used]
            apply2!(prog.code[sym], op1, op2, out) || return NaN
            push!(stack, out)
        elseif k === Int8(1)
            isempty(stack) && return NaN
            used += 1
            used <= nbuf || return NaN
            op1 = pop!(stack)
            out = buffers[used]
            apply1!(prog.code[sym], op1, out) || return NaN
            push!(stack, out)
        else
            return NaN
        end
    end
    isempty(stack) && return NaN
    return last(stack)
end

# Operator tables

"""
    TENSOR_NODES

Operator node type by operator symbol, the symbols `GepTensorRegressor` accepts in
`entered_non_terminals`.
"""
const TENSOR_NODES = Dict{Symbol,Type}(
    :+ => AdditionNode,
    :- => SubtractionNode,
    :* => MultiplicationNode,
    :/ => DivisionNode,
    :^ => PowerNode,
    :min => MinNode,
    :max => MaxNode,
    :inv => InversionNode,
    :dot => DotProductNode,
    :tr => TraceNode,
    :det => DeterminantNode,
    :symmetric => SymmetricNode,
    :skew => SkewNode,
    :vol => VolumetricNode,
    :tdot => TdotNode,
    :dott => DottNode,
    :dcontract => DoubleContractionNode,
    :dev => DeviatoricNode,
    :crossp => CrossProductNode,
    :lap => LapNode,
    :otimes => OuterProductNode,
    :hadamard => HadamardNode,
    :sqrt => SqrtNode,
    :norm => NormNode,
    :sin => SinNode,
    :cos => CosNode,
    :tan => TanNode,
    :exp => ExpNode,
    :log => LogNode,
    :abs => AbsNode,
    :floor => FloorNode,
    :ceil => CeilNode,
    :round => RoundNode,
    :sign => SignNode,
    :log10 => Log10Node,
    :log2 => Log2Node,
    :sqr => SqrNode,
    :asin => AsinNode,
    :acos => AcosNode,
    :atan => AtanNode,
    :sinh => SinhNode,
    :cosh => CoshNode,
    :tanh => TanhNode,
    :asinh => AsinhNode,
    :acosh => AcoshNode,
    :atanh => AtanhNode
)

"""
    TENSOR_NODE_BY_FUNCTION

Operator node type by the Julia function it evaluates, derived from `TENSOR_NODES`. Used to
translate the function callbacks of a scalar regressor's toolbox, taken from
`FUNCTION_LIB_COMMON`, into operator nodes.
"""
const TENSOR_NODE_BY_FUNCTION = begin
    d = Dict{Any,Any}()
    # Each symbol is resolved in `Base`, then in `GepUtils` (included before this module),
    # which defines `sqr`, the one function of `FUNCTION_LIB_COMMON` not in Base. Symbols
    # found in neither are left out.
    extra = isdefined(parentmodule(@__MODULE__), :GepUtils) ?
            getfield(parentmodule(@__MODULE__), :GepUtils) : nothing
    for (sym, T) in TENSOR_NODES
        f = if isdefined(Base, sym)
            getfield(Base, sym)
        elseif !isnothing(extra) && isdefined(extra, sym)
            getfield(extra, sym)
        else
            nothing
        end
        isnothing(f) || (d[f] = T)
    end
    d
end

"""
    TENSOR_STRINGIFY

Renderer by operator node type name (`nameof(typeof(node))`): a function that renders the
operator applied to its operands, used to print equations.
"""
const TENSOR_STRINGIFY = Dict{Symbol,Function}(
    :AdditionNode => (args...) -> "($(string(args[1])) + $(string(args[2])))",
    :SubtractionNode => (args...) -> "($(string(args[1])) - $(string(args[2])))",
    :MultiplicationNode => (args...) -> "($(string(args[1])) * $(string(args[2])))",
    :DivisionNode => (args...) -> "($(string(args[1])) / $(string(args[2])))",
    :PowerNode => (args...) -> "($(string(args[1]))^$(string(args[2])))",
    :MinNode => (args...) -> "min($(string(args[1])),$(string(args[2])))",
    :MaxNode => (args...) -> "max($(string(args[1])),$(string(args[2])))",
    :InversionNode => A -> "inv($(string(A)))",
    :DotProductNode => (args...) -> "dot($(string(args[1])),$(string(args[2])))",
    :TraceNode => A -> "tr($(string(A)))",
    :DeterminantNode => A -> "det($(string(A)))",
    :SymmetricNode => A -> "sym($(string(A)))",
    :SkewNode => A -> "skew($(string(A)))",
    :VolumetricNode => A -> "vol($(string(A)))",
    :DeviatoricNode => A -> "dev($(string(A)))",
    :TdotNode => (args...) -> "($(string(args[1]))·$(string(args[end]))ᵀ)",
    :DottNode => (args...) -> "($(string(args[1]))ᵀ·$(string(args[end])))",
    :DoubleContractionNode => (A, B) -> "($(string(A)):$(string(B)))",
    :CrossProductNode => (A, B) -> "($(string(A))×$(string(B)))",
    :LapNode => A -> "(I:($(string(A))))",
    :OuterProductNode => (A, B) -> "($(string(A))⊗$(string(B)))",
    :HadamardNode => (A, B) -> "($(string(A))∘$(string(B)))",
    :SqrtNode => A -> "sqrt($(string(A)))",
    :NormNode => A -> "norm($(string(A)))",
    :SinNode => A -> "sin($(string(A)))",
    :CosNode => A -> "cos($(string(A)))",
    :TanNode => A -> "tan($(string(A)))",
    :ExpNode => A -> "exp($(string(A)))",
    :LogNode => A -> "log($(string(A)))",
    :AbsNode => A -> "abs($(string(A)))",
    :FloorNode => A -> "floor($(string(A)))",
    :CeilNode => A -> "ceil($(string(A)))",
    :RoundNode => A -> "round($(string(A)))",
    :SignNode => A -> "sign($(string(A)))",
    :Log10Node => A -> "log10($(string(A)))",
    :Log2Node => A -> "log2($(string(A)))",
    :SqrNode => A -> "sqr($(string(A)))",
    :AsinNode => A -> "asin($(string(A)))",
    :AcosNode => A -> "acos($(string(A)))",
    :AtanNode => A -> "atan($(string(A)))",
    :SinhNode => A -> "sinh($(string(A)))",
    :CoshNode => A -> "cosh($(string(A)))",
    :TanhNode => A -> "tanh($(string(A)))",
    :AsinhNode => A -> "asinh($(string(A)))",
    :AcoshNode => A -> "acosh($(string(A)))",
    :AtanhNode => A -> "atanh($(string(A)))",
)

"""
    TENSOR_NODES_ARITY

Arity by operator symbol, for the symbols of `TENSOR_NODES`.
"""
const TENSOR_NODES_ARITY = Dict{Symbol,Int8}(
    :+ => 2, :- => 2, :* => 2, :/ => 2, :^ => 2,
    :min => 2, :max => 2,
    :inv => 1, :dot => 2,
    :crossp => 2,
    :tr => 1, :det => 1,
    :symmetric => 1, :skew => 1, :vol => 1, :dev => 1,
    :tdot => 1, :dott => 1, :dcontract => 2, :deviator => 1, :lap => 1, :otimes => 2, :hadamard => 2,
    :sqrt => 1, :norm => 1, :sin => 1, :cos => 1, :tan => 1, :exp => 1, :log => 1,
    :abs => 1, :floor => 1, :ceil => 1, :round => 1, :sign => 1, :log10 => 1,
    :log2 => 1, :sqr => 1, :asin => 1, :acos => 1, :atan => 1, :sinh => 1,
    :cosh => 1, :tanh => 1, :asinh => 1, :acosh => 1, :atanh => 1)
end