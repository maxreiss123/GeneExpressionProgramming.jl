using Flux, LinearAlgebra, OrderedCollections
using Tensors

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
    AdditionNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct SubtractionNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    SubtractionNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct MultiplicationNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    MultiplicationNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct DivisionNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    DivisionNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct PowerNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    PowerNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct MinNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    MinNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct MaxNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    MaxNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct ContractionNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    ContractionNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct TmagnitudeNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    TmagnitudeNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct InnerProductNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    InnerProductNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct OuterProductNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    OuterProductNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct KroneckerNode <: OperationNode
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    KroneckerNode(chain=nothing; use_cuda::Bool=false) = new(chain, use_cuda)
end

struct UnaryNode <: OperationNode
    operation::Function
    chain::Union{Nothing,Chain}
    use_cuda::Bool
    UnaryNode(operation::Function, chain=nothing; use_cuda::Bool=false) = 
        new(operation, chain, use_cuda)
end


for T in subtypes(OperationNode)
    @eval Flux.Functors.@functor $T
end

# Helper for CUDA operations
@inline function ensure_cuda(x, use_cuda::Bool)
    #if use_cuda && !isa(x, CuArray)
    #    cu(x)
    #elseif !use_cuda && isa(x, CuArray)
   #     cpu(x)
   # else
   return x
   # end
end

@inline function (l::AdditionNode)(x, y)
    @show size(x) == size(y)
    size(x) == size(y) || throw(DimensionMismatch("Sizes must match exactly. Got sizes $(size(x)) and $(size(y))"))
    x, y = ensure_cuda.((x, y), l.use_cuda)
    result = x .+ y  
    l.use_cuda ? result : cpu(result)
end

@inline function (l::SubtractionNode)(x, y)
    size(x) == size(y) || throw(DimensionMismatch("Sizes must match exactly. Got sizes $(size(x)) and $(size(y))"))
    x, y = ensure_cuda.((x, y), l.use_cuda)
    result = x .- y 
    l.use_cuda ? result : cpu(result)
end

@inline function (l::MultiplicationNode)(x::AbstractArray, y::Number)
    x = ensure_cuda(x, l.use_cuda)
    result = x * y 
    l.use_cuda ? result : cpu(result)
end

@inline function (l::MultiplicationNode)(x::AbstractArray, y::AbstractArray)
    x, y = ensure_cuda.((x, y), l.use_cuda)
    result = x * y 
    l.use_cuda ? result : cpu(result)
end

@inline function (l::MultiplicationNode)(x, y)
    x, y = ensure_cuda.((x, y), l.use_cuda)
    result = x * y 
    l.use_cuda ? result : cpu(result)
end

@inline function (l::DivisionNode)(x::Number, y::AbstractArray)
    y = ensure_cuda(y, l.use_cuda)
    result = x / y
    l.use_cuda ? result : cpu(result)
end

@inline function (l::DivisionNode)(x::AbstractArray, y::Number)
    x = ensure_cuda(x, l.use_cuda)
    result = x / y
    l.use_cuda ? result : cpu(result)
end

@inline function (l::PowerNode)(x, y)
    x, y = ensure_cuda.((x, y), l.use_cuda)
    result = x .^ y
    l.use_cuda ? result : cpu(result)
end

@inline function (l::MinNode)(x, y)
    x, y = ensure_cuda.((x, y), l.use_cuda)
    result = min.(x, y)
    l.use_cuda ? result : cpu(result)
end

@inline function (l::MaxNode)(x, y)
    x, y = ensure_cuda.((x, y), l.use_cuda)
    result = max.(x, y)
    l.use_cuda ? result : cpu(result)
end

@inline function (l::UnaryNode)(x)
    x = ensure_cuda(x, l.use_cuda)
    result = l.operation.(x)
    l.use_cuda ? result : cpu(result)
end

@inline function (l::ContractionNode)(x)
    x = ensure_cuda(x, l.use_cuda)
    result = sum(x, dims=1:(ndims(x)-1))
    l.use_cuda ? result : cpu(result)
end

@inline function (l::TmagnitudeNode)(x)
    x = ensure_cuda(x, l.use_cuda)
    result = sqrt.(sum(abs2.(x), dims=1:(ndims(x)-1)))
    l.use_cuda ? result : cpu(result)
end

@inline function (l::InnerProductNode)(x, y)
    x, y = ensure_cuda.((x, y), l.use_cuda)
    result = sum(x .* y, dims=1:(ndims(x)-1))
    l.use_cuda ? result : cpu(result)
end

@inline function (l::OuterProductNode)(x, y)
    x, y = ensure_cuda.((x, y), l.use_cuda)
    x_reshape = reshape(x, size(x)..., 1, :)
    y_reshape = reshape(y, 1, size(y)...)
    result = x_reshape .* y_reshape
    l.use_cuda ? result : cpu(result)
end

@inline function (l::KroneckerNode)(x)
    x = ensure_cuda(x, l.use_cuda)
    batch_size = size(x)[end]
    I_mat = Matrix(I, 3, 3)
    I_mat = l.use_cuda ? cu(I_mat) : I_mat
    result = similar(x, size(x, 1) * 3, size(x, 2) * 3, batch_size)
    result = l.use_cuda ? cu(result) : result
    
    for b in 1:batch_size
        result[:, :, b] = kron(x[:, :, b], I_mat)
    end
    return result
end

# Operation mappings
const UNARY_OPS = Dict(
    :exp => exp, :log => log, :sqrt => sqrt,
    :sin => sin, :cos => cos, :tan => tan,
    :asin => asin, :acos => acos, :atan => atan,
    :sinh => sinh, :cosh => cosh, :tanh => tanh,
    :asinh => asinh, :acosh => acosh, :atanh => atanh,
    :abs => abs, :sign => sign,
    :floor => floor, :ceil => ceil, :round => round
)

const BINARY_OPS = Dict(
    :+ => AdditionNode,
    :- => SubtractionNode,
    :* => MultiplicationNode,
    :/ => DivisionNode,
    :^ => PowerNode,
    :min => MinNode,
    :max => MaxNode
)

const TENSOR_OPS = Dict(
    :contract => ContractionNode,
    :magnitude => TmagnitudeNode,
    :kronecker => KroneckerNode,
    :inner => InnerProductNode,
    :outer => OuterProductNode
)

# Optimized network compilation
function compile_to_flux_network(rek_string::Vector, arity_map::OrderedDict; use_cuda::Bool=false)
    stack = Vector{Any}(undef, length(rek_string))
    sp = 0
    input_count = count(x -> x isa Int8, rek_string)
    inputs = [InputSelector(i) for i in 1:input_count]
    
    @inbounds for elem in rek_string
        if elem isa Int8
            sp += 1
            stack[sp] = inputs[elem]
        else
            arity = get(arity_map, elem, 0)
            if arity == 2 && sp >= 2
                op1 = stack[sp]
                op2 = stack[sp-1]
                sp -= 1
                
                if haskey(BINARY_OPS, elem)
                    node = BINARY_OPS[elem](nothing, use_cuda=use_cuda)
                    stack[sp] = x -> node(op2(x), op1(x))
                elseif haskey(TENSOR_OPS, elem)
                    node = TENSOR_OPS[elem](nothing, use_cuda=use_cuda)
                    stack[sp] = x -> node(op2(x), op1(x))
                end
            elseif arity == 1 && sp >= 1
                op = stack[sp]
                if haskey(UNARY_OPS, elem)
                    node = UnaryNode(UNARY_OPS[elem], nothing, use_cuda=use_cuda)
                    stack[sp] = x -> node(op(x))
                elseif haskey(TENSOR_OPS, elem)
                    node = TENSOR_OPS[elem](nothing, use_cuda=use_cuda)
                    stack[sp] = x -> node(op(x))
                end
            end
        end
    end
    
    @assert sp == 1 "Invalid expression: stack error"
    return Chain(stack[1])
end

# Test functions
function test_binary_ops(batch_size::Int=32)
    # Test addition with batching
    add_node = AdditionNode()
    x = rand(Float32, 10, batch_size)  # (features, batch)
    y = rand(Float32, 10, batch_size)
    result = add_node(x, y)
    @assert size(result, 2) == batch_size "Batch dimension not preserved"
    
    # Test multiplication with batching
    mul_node = MultiplicationNode()
    result = mul_node(x, y)
    @assert size(result, 2) == batch_size "Batch dimension not preserved"
    
    return (addition=result, multiplication=result)
end

function test_tensor_ops(batch_size::Int=32)
    # Test inner product with batching
    inner_node = InnerProductNode()
    x = rand(Float32, 10, batch_size)
    y = rand(Float32, 10, batch_size)
    result = inner_node(x, y)
    @assert size(result)[end] == batch_size "Batch dimension not preserved"
    
    # Test contraction with batching
    contract_node = ContractionNode()
    tensor = rand(Float32, 5, 5, batch_size)
    result = contract_node(tensor)
    @assert size(result)[end] == batch_size "Batch dimension not preserved"
    
    return (inner_product=result, contraction=result)
end

function test_complete_network(batch_size::Int=32, use_cuda::Bool=false)
    rek_string = [Int8(1), Int8(2), :*, Int8(3), :+]
    
    arity_map = OrderedDict{Any,Int}(
        :+ => 2,
        :* => 2
    )
    
    network = compile_to_flux_network(rek_string, arity_map, use_cuda=use_cuda)
    
    x1 = rand(Float32, 100, batch_size)
    x2 = rand(Float32, 100, batch_size)
    x3 = rand(Float32, 100, batch_size)
    
    if use_cuda
        x1, x2, x3 = cu.((x1, x2, x3))
    end
    
    result = network((x1, x2, x3))
    @assert size(result)[end] == batch_size "Batch dimension not preserved in network"
    
    return network, result
end

# Run tests
function run_all_tests(batch_size::Int=32)
    println("Testing binary operations...")
    binary_results = test_binary_ops(batch_size)
    
    println("Testing tensor operations...")
    tensor_results = test_tensor_ops(batch_size)
    
    println("Testing complete network...")
    network, complete_results = test_complete_network(batch_size)
    
    println("All tests completed successfully!")
    return (binary=binary_results, tensor=tensor_results, network=complete_results)
end

# Run all tests
results = run_all_tests(10000000)