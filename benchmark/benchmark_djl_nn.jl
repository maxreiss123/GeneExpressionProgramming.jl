using DynamicExpressions
using DynamicExpressions: @declare_expression_operator
using BenchmarkTools
using LinearAlgebra

include("../src/TensorOps.jl")
using .TensorRegUtils
using Tensors
using OrderedCollections
using Flux


"""
Benchmark for comparing higher dim structures for tensor regression - run test with - export JULIA_NUM_THREADS=1
example from: https://github.com/SymbolicML/DynamicExpressions.jl with changes according to utilize tensors
"""


T = Union{Float64,Vector{Float64},Tensor}
vec_add(x::Tensor, y::Tensor) = @fastmath x + y;
vec_square(x::Tensor) = @fastmath dot(x,x);


@declare_expression_operator(vec_add, 2);
@declare_expression_operator(vec_square, 1);


operators = GenericOperatorEnum(; binary_operators=[vec_add], unary_operators=[vec_square]);

# Construct the expression:
variable_names = ["x1"]
c1 = Expression(Node{T}(; val=ones(Tensor{2,3})); operators, variable_names);  
expression = vec_add(vec_add(vec_square(c1), c1), c1);

X = ones(Tensor{2,3});


# create the inputs for Flux
c1_ = ones(Tensor{2,3});
inputs = (c1_,);

# create the arity map
arity_map = OrderedDict{Int8,Int}(
    1 => 2,  # Addition
    2 => 2  # Multiplication
);

#assign the callbacks
callbacks = Dict{Int8,Any}(
    Int8(1) => AdditionNode,
    Int8(2) => MultiplicationNode
);

#define nodes
nodes = OrderedDict{Int8,Any}(
    Int8(5) => InputSelector(1)
);

# Evaluate - expression 
# Solution => [[5.0 5.0 5.0], [5.0 5.0 5.0], [5.0 5.0 5.0]]
tests_n = 100000
@show "Benchmark expression"
expression(X)  
@btime for _ in 1:tests_n
    expression(X)  
end

#83.021 ms (1798979 allocations: 187.67 MiB)

rek_string = Int8[1, 1, 2, 5, 5, 5, 5];
network = TensorRegUtils.compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0);
@show "Benchmark network"
result = network(inputs)
@btime for _ in 1:tests_n
    result = network(inputs)
end

#11.703 ms (998979 allocations: 59.49 MiB)


#Conclusion ≈ 7 times faster than DynamicExpressions.jl for such structures