#=
OBSOLETE: kept as the record behind benchmark/Benchmark.md; it no longer runs.

Compared DynamicExpressions.jl with the former Flux-network path
(`TensorRegUtils.compile_to_flux_network`) on one tensor-valued expression, single-threaded
(JULIA_NUM_THREADS=1). Both have since been replaced by the batched evaluator
(`calc_stack_batch_tensor`): DynamicExpressions is no longer a dependency and
`compile_to_flux_network` no longer exists. The expression is adapted from the
DynamicExpressions.jl README (https://github.com/SymbolicML/DynamicExpressions.jl).
=#
using DynamicExpressions
using DynamicExpressions: @declare_expression_operator
using BenchmarkTools
using LinearAlgebra

include("../src/TensorOps.jl")
using .TensorRegUtils
using Tensors
using OrderedCollections
using Flux


T = Union{Float64,Vector{Float64},Tensor}
vec_add(x::Tensor, y::Tensor) = @fastmath x + y;
vec_square(x::Tensor) = @fastmath dot(x,x);


@declare_expression_operator(vec_add, 2);
@declare_expression_operator(vec_square, 1);


operators = GenericOperatorEnum(; binary_operators=[vec_add], unary_operators=[vec_square]);

# x1 + x1 + x1 . x1 on a 3x3 tensor of ones
variable_names = ["x1"]
c1 = Expression(Node{T}(; val=ones(Tensor{2,3})); operators, variable_names);  
expression = vec_add(vec_add(vec_square(c1), c1), c1);

X = ones(Tensor{2,3});


# the same expression as a karva string (1 = +, 2 = *, 5 = x1) for the Flux network
c1_ = ones(Tensor{2,3});
inputs = (c1_,);

arity_map = OrderedDict{Int8,Int}(
    1 => 2,  # Addition
    2 => 2  # Multiplication
);

callbacks = Dict{Int8,Any}(
    Int8(1) => AdditionNode,
    Int8(2) => MultiplicationNode
);

nodes = OrderedDict{Int8,Any}(
    Int8(5) => InputSelector(1)
);

# every entry of the result is 5: (ones * ones) + ones + ones, with * the single contraction
tests_n = 100000
@show "Benchmark expression"
expression(X)  
@btime for _ in 1:tests_n
    expression(X)  
end

# recorded: 83.021 ms (1798979 allocations: 187.67 MiB)

rek_string = Int8[1, 1, 2, 5, 5, 5, 5];
network = TensorRegUtils.compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0);
@show "Benchmark network"
result = network(inputs)
@btime for _ in 1:tests_n
    result = network(inputs)
end

# recorded: 11.703 ms (998979 allocations: 59.49 MiB), i.e. about 7x faster than
# DynamicExpressions.jl on this expression