# Tensor evaluation: DynamicExpressions.jl vs a Flux network (obsolete)

> **Historical record.** This compared two ways of evaluating a tensor-valued expression
> that the package no longer has: DynamicExpressions.jl, and a Flux network compiled from a
> karva string (`TensorRegUtils.compile_to_flux_network`). Both were replaced by the batched
> evaluator (`calc_stack_batch_tensor`); DynamicExpressions is no longer a dependency and
> `compile_to_flux_network` no longer exists, so `benchmark_djl_nn.jl` does not run. The
> numbers below are kept as recorded, single-threaded (`JULIA_NUM_THREADS=1`).

## DynamicExpressions.jl

Expression adapted from the DynamicExpressions.jl README
(https://github.com/SymbolicML/DynamicExpressions.jl) to use `Tensors` as the data type.

```julia
using DynamicExpressions
using DynamicExpressions: @declare_expression_operator
using BenchmarkTools
using LinearAlgebra

# Operations
vec_add(x::Tensor, y::Tensor) = @fastmath x + y
vec_square(x::Tensor) = @fastmath dot(x,x)

@declare_expression_operator(vec_add, 2)
@declare_expression_operator(vec_square, 1)

# Build expression
operators = GenericOperatorEnum(
    binary_operators=[vec_add], 
    unary_operators=[vec_square]
)
variable_names = ["x1"]
c1 = Expression(Node{T}(; val=ones(Tensor{2,3})); operators, variable_names);  
expression = vec_add(vec_add(vec_square(c1), c1), c1);
X = ones(Tensor{2,3});

# Evaluate the expression:

tests_n = 100000
@show "Benchmark expression"
expression(X)  # [[5.0 5.0 5.0], [5.0 5.0 5.0], [5.0 5.0 5.0]]
@btime for _ in 1:tests_n
    expression(X)  
end

# 83.021 ms (1798979 allocations: 187.67 MiB)
```

## Flux network compiled from the karva string

```julia 
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

rek_string = Int8[1, 1, 2, 5, 5, 5, 5];
network = TensorRegUtils.compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0);
@show "Benchmark network"
result = network(inputs) # [[5.0 5.0 5.0], [5.0 5.0 5.0], [5.0 5.0 5.0]]
@btime for _ in 1:tests_n
    result = network(inputs)
end

#11.703 ms (998979 allocations: 59.49 MiB)

```

## Conclusion (as recorded)

- The Flux network was about 7x faster (11.7 ms against 83.0 ms for 100 000 evaluations).
- It allocated about 3.2x less memory (59.49 MiB against 187.67 MiB) in 1.8x fewer
  allocations.
