"""
    GepUtils

A utility module providing essential functions and types for Gene Expression Programming (GEP)
operations, including optimization, history tracking, data manipulation, and state management.

# Core Features
## Optimization
- Constant optimization with multiple algorithms
- Node compilation and manipulation
- Distance calculations and scaling

## History Recording
- Asynchronous history tracking
- Training metrics recording
- Optimization history management
- Progress monitoring

## Data Handling
- Train-test splitting
- Minmax scaling
- Data type conversion
- State serialization

# Main Types
## History Management
- `OptimizationHistory`: Stores training metrics and statistics
- `HistoryRecorder`: Asynchronous recorder for optimization metrics

# Main Functions
## Optimization
- `optimize_constants!`: Optimize constant values in expressions
- `compile_djl_datatype`: Compile recursive expressions
- `retrieve_constants_from_node`: Extract constants from nodes

## Scaling and Metrics
- `minmax_scale`: Scale data to specific range
- `float16_scale`: Scale to Float16 range
- `isclose`: Approximate equality comparison

## History Recording
- `create_history_recorder`: Initialize recording
- `record!`: Record optimization step
- `close_recorder!`: Finalize recording
- `get_history_arrays`: Extract history data

## Data Management
- `train_test_split`: Split dataset for training/testing
- `save_state`, `load_state`: State persistence

## Constants
- `FUNCTION_LIB_COMMON`: Available mathematical functions

# Function Library
Includes extensive mathematical operations:
- Basic arithmetic: +, -, *, /, ^
- Comparisons: min, max
- Rounding: floor, ceil, round
- Exponential: exp, log, log10, log2
- Trigonometric: sin, cos, tan, asin, acos, atan
- Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
- Special: sqr, sqrt, sign, abs
- Tensorfunctions: 

# Usage Example
```julia
# History recording
recorder = HistoryRecorder(100, Float64)  # 100 epochs
record!(recorder, epoch, train_loss, val_loss, fitness_vector)
close_recorder!(recorder)

# Data scaling
scaled_data = minmax_scale(data, feature_range=(0.0, 1.0))

# Train-test split
x_train, y_train, x_test, y_test = train_test_split(X, y, train_ratio=0.8)

# Constant optimization
optimized_node, final_loss = optimize_constants!(
    node,
    loss_function;
    opt_method=:cg,
    max_iterations=250
)
```

# Implementation Details
## Performance Optimizations
- Thread-safe operations via channels
- SIMD optimizations where applicable
- Efficient memory management
- Asynchronous history recording

## Dependencies
- `OrderedCollections`: Ordered data structures
- `DynamicExpressions`: Expression handling
- `LinearAlgebra`: Matrix operations
- `Optim`: Optimization algorithms
- `LineSearches`: Line search methods
- `Zygote`: Automatic differentiation
- `Serialization`: State persistence
- `Statistics`: Statistical computations
- `Flux`: ML-Package
- `Tensors`: mathematical objects of higher order
- `Random`: Random number generation
- `CUDA`: Extension to run on CUDA-cores
"""
module GepUtils

export find_indices_with_sum, compile_djl_datatype, optimize_constants!, minmax_scale, float16_scale, isclose
export save_state, load_state
export create_history_recorder, record_history!, record!, close_recorder!
export HistoryRecorder, OptimizationHistory, get_history_arrays, one_hot_mean, FUNCTION_STRINGIFY
export train_test_split, select_n_samples_lhs
export FUNCTION_LIB_COMMON, ARITY_LIB_COMMON
export TensorNode, compile_network

using OrderedCollections
using DynamicExpressions
using LinearAlgebra
using Optim
using LineSearches
using Zygote
using Serialization
using Statistics
using Random
using Tensors
using Flux
using StatsBase
using NearestNeighbors
using Base.Threads: @spawn


function sqr(x::Vector{T}) where {T<:AbstractFloat}
    return x .* x
end

function sqr(x::T) where {T<:Union{AbstractFloat,Node{<:AbstractFloat}}}
    return x * x
end

"""
    FUNCTION_LIB_COMMON::Dict{Symbol,Function}

Dictionary mapping function symbols to their corresponding functions.
Contains basic mathematical operations, trigonometric, and other common functions.

# Available Functions
- Basic arithmetic: `+`, `-`, `*`, `/`, `^`
- Comparison: `min`, `max`
- Rounding: `floor`, `ceil`, `round`
- Exponential & Logarithmic: `exp`, `log`, `log10`, `log2`
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- Other: `abs`, `sqr`, `sqrt`, `sign`
- Tensor Functions: 

To add a new function, ensure you also add corresponding entries in `ARITY_LIB_COMMON`,
`FUNCTION_LIB_FORWARD_COMMON`, and `FUNCTION_LIB_BACKWARD_COMMON`.
"""
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


const FUNCTION_STRINGIFY = Dict{Symbol,Function}(
    :+ => (args...) -> join(args, " + "),
    :- => (args...) -> length(args) == 1 ? "-$(args[1])" : join(args, " - "),
    :* => (args...) -> join(args, " * "),
    :/ => (args...) -> join(args, " / "),
    :^ => (args...) -> "$(args[1])^$(args[2])",
    :min => (args...) -> "min($(join(args, ", ")))",
    :max => (args...) -> "max($(join(args, ", ")))",
    :abs => a -> "|$a|",
    :round => a -> "round($a)",
    :exp => a -> "e^($a)",
    :log => a -> "ln($a)",
    :log10 => a -> "log₁₀($a)",
    :log2 => a -> "log₂($a)",
    :sin => a -> "sin($a)",
    :cos => a -> "cos($a)",
    :tan => a -> "tan($a)",
    :asin => a -> "arcsin($a)",
    :acos => a -> "arccos($a)",
    :atan => a -> "arctan($a)",
    
    :sinh => a -> "sinh($a)",
    :cosh => a -> "cosh($a)",
    :tanh => a -> "tanh($a)",
    :asinh => a -> "arcsinh($a)",
    :acosh => a -> "arccosh($a)",
    :atanh => a -> "arctanh($a)",
    
    :sqr => a -> "($a)²",
    :sqrt => a -> "√($a)",
    :sign => a -> "sign($a)"
)

"""
    ARITY_LIB_COMMON::Dict{Symbol,Int8}

Dictionary specifying the number of arguments (arity) for each function in the library.
- Value of 1 indicates unary functions (e.g., `sin`, `cos`, `abs`)
- Value of 2 indicates binary functions (e.g., `+`, `-`, `*`, `/`)

When adding new functions to `FUNCTION_LIB_COMMON`, ensure to specify their arity here.
"""
const ARITY_LIB_COMMON = Dict{Symbol,Int8}(
    :+ => 2,
    :- => 2,
    :* => 2,
    :/ => 2,
    :^ => 2,
    :min => 2,
    :max => 2,
    :abs => 1,
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

struct OptimizationHistory{T<:Union{AbstractFloat,Tuple}}
    train_loss::Vector{T}
    val_loss::Vector{T}
    train_mean::Vector{T}
    train_std::Vector{T}

    function OptimizationHistory(epochs::Int, ::Type{T}) where {T<:Union{AbstractFloat,Tuple}}
        return new{T}(
            Vector{T}(undef, epochs),
            Vector{T}(undef, epochs),
            Vector{T}(undef, epochs),
            Vector{T}(undef, epochs)
        )
    end
end

function Base.iterate(hist::OptimizationHistory, state::Int=1)
    if state > length(hist.train_loss)
        return nothing
    end
    return (
        (
            train_loss=hist.train_loss[state],
            val_loss=hist.val_loss[state],
            train_mean=hist.train_mean[state],
            train_std=hist.train_std[state]
        ),
        state + 1
    )
end

Base.length(hist::OptimizationHistory) = length(hist.train_loss)

Base.size(hist::OptimizationHistory) = (length(hist.train_loss),)

function Base.getindex(hist::OptimizationHistory, i::Int)
    return (
        train_loss=hist.train_loss[i],
        val_loss=hist.val_loss[i],
        train_mean=hist.train_mean[i],
        train_std=hist.train_std[i]
    )
end


Base.firstindex(hist::OptimizationHistory) = 1
Base.lastindex(hist::OptimizationHistory) = length(hist)


function Base.show(io::IO, hist::OptimizationHistory)
    println(io, "OptimizationHistory{$(eltype(hist.train_loss))} with $(length(hist)) epochs")
end


function get_history_arrays(hist::OptimizationHistory)
    return (
        train_loss=hist.train_loss,
        val_loss=hist.val_loss,
        train_mean=hist.train_mean,
        train_std=hist.train_std
    )
end


"""
    HistoryRecorder{T<:AbstractFloat}

A thread-safe structure for asynchronous recording of optimization history during
GEP evolution, using channels for communication between optimization and recording tasks.

# Fields
- `channel::Channel{Tuple{Int,T,T,Vector{T}}}`: Communication channel for metrics
  - Tuple format: (epoch, train_loss, validation_loss, fitness_vector)
- `task::Task`: Asynchronous task handling the recording process
- `history::OptimizationHistory{T}`: Storage for optimization metrics

# Constructor
```julia
HistoryRecorder(
    epochs::Int,
    ::Type{T};
    buffer_size::Int=32
) where {T<:AbstractFloat}
```

# Arguments
- `epochs::Int`: Number of epochs to record
- `T`: Numeric type for metrics (e.g., Float64)
- `buffer_size::Int=32`: Channel buffer size for async communication

# Example Usage
```julia
# Create recorder for 100 epochs using Float64
recorder = HistoryRecorder(100, Float64)

# Record metrics for each epoch
for epoch in 1:100
    train_loss = compute_training_loss()
    val_loss = compute_validation_loss()
    fitness_vector = get_population_fitness()
    
    record!(recorder, epoch, train_loss, val_loss, fitness_vector)
end

# Close recorder and wait for completion
close_recorder!(recorder)

# Access recorded history
final_history = recorder.history
```

# Thread Safety
- Uses channels for thread-safe communication
- Spawns separate task for recording
- Ensures non-blocking metric recording
- Maintains data consistency

# Performance Notes
## Buffer Size
- Default 32 provides balance between memory and performance
- Increase for high-frequency recording
- Decrease for memory-constrained environments

## Memory Management
- Preallocates history arrays
- Reuses metric tuples
- Minimizes allocation during recording

# Notes
- Automatically spawns recording task on creation
- Must be closed with `close_recorder!` to ensure proper cleanup
- Supports any AbstractFloat type
- Channel depth can be adjusted for different recording patterns
"""
struct HistoryRecorder{T<:Union{AbstractFloat,Tuple}}
    channel::Channel{Tuple{Int,T,T,Vector{T}}}
    task::Task
    history::OptimizationHistory{T}

    function HistoryRecorder(epochs::Int, ::Type{T}; buffer_size::Int=32) where {T<:Union{AbstractFloat,Tuple}}
        history = OptimizationHistory(epochs, T)
        channel = Channel{Tuple{Int,T,T,Vector{T}}}(buffer_size)
        task = @spawn record_history!(channel, history)
        return new{T}(channel, task, history)
    end
end


@inline function tuple_agg(entries::Vector{T}, fun::Function) where {T<:Tuple}
    isempty(entries) && return entries[1]
    N = length(first(entries))
    L = length(entries)

    vectors = ntuple(i -> Vector{Float64}(undef, L), N)

    for (j, entry) in enumerate(entries)
        for i in 1:length(entry)
            vectors[i][j] = entry[i]
        end
    end
    return tuple(i -> fun(vectors[i]), N)
end

@inline function record_history!(
    channel::Channel{Tuple{Int,T,T,Vector{T}}},
    history::OptimizationHistory{T}
) where {T<:Union{AbstractFloat,Tuple}}
    for (epoch, train_loss, val_loss, fit_vector) in channel
        @inbounds begin
            history.train_loss[epoch] = train_loss
            history.val_loss[epoch] = val_loss
            history.train_mean[epoch] = tuple_agg(fit_vector, mean)
            history.train_std[epoch] = tuple_agg(fit_vector, std)
        end
    end
end

@inline function record!(
    recorder::HistoryRecorder{T},
    epoch::Int,
    train_loss::T,
    val_loss::T,
    fit_vector::Vector{T}
) where {T<:Union{AbstractFloat,Tuple}}
    put!(recorder.channel, (epoch, train_loss, val_loss, fit_vector))
end

@inline function close_recorder!(recorder::HistoryRecorder)
    close(recorder.channel)
    wait(recorder.task)
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

"""
    compile_djl_datatype(
        rek_string::Vector,
        arity_map::OrderedDict,
        callbacks::Dict,
        nodes::OrderedDict
    )

Compiles a reverse Polish notation (postfix) expression into an executable form using
a stack-based algorithm with support for unary and binary operations.

# Arguments
- `rek_string::Vector`: Expression in reverse Polish notation
- `arity_map::OrderedDict`: Maps symbols to their arities (number of operands)
- `callbacks::Dict`: Maps symbols to their corresponding operations
- `nodes::OrderedDict`: Maps terminal symbols to their node representations

# Returns
The compiled expression as a DynamicExpressions.Node object

# Algorithm
1. Initializes empty stack
2. Processes expression in reverse order:
   - For binary operators (arity 2):
     * Pops two operands
     * Applies operation
     * Pushes result
   - For unary operators (arity 1):
     * Pops one operand
     * Applies operation
     * Pushes result
   - For terminals (arity 0):
     * Pushes directly to stack
3. Returns final stack element

# Example
```julia
# Define components
rek_string = [1, 2, :+, 3, :*]  # represents (1 + 2) * 3 -> Examplified - in our application, we use tokenized version of that
arity_map = OrderedDict(
    :+ => 2,
    :* => 2
)
callbacks = Dict(
    :+ => +,
    :* => *
)
nodes = OrderedDict(
    1 => Node(1.0),
    2 => Node(2.0),
    3 => Node(3.0)
)

# Compile expression
result = compile_djl_datatype(rek_string, arity_map, callbacks, nodes)
# Returns Node representing (1 + 2) * 3
```

# Error Handling
- Allows expression to fail if invalid
- Invalid expressions may occur from:
  * Stack underflow
  * Unknown operators
  * Mismatched arities
  * Invalid node references

# Implementation Notes
## Stack Operations
- Uses pop! for operand retrieval
- Uses push! for result storage
- Handles Int8 to Node conversion

## Type Handling
- Supports Int8 terminal symbols
- Converts terminals via nodes dictionary
- Preserves operation types from callbacks

## Performance Considerations
- Single pass through expression
- Minimal memory allocation
- Direct operation application
- Early failure for invalid expressions

See also: [`DynamicExpressions.Node`](@ref)
"""
function compile_djl_datatype(rek_string::Vector, arity_map::OrderedDict, callbacks::Dict, nodes::OrderedDict, pre_len::Int)
    stack = []
    for elem in reverse(rek_string[pre_len:end])
        if get(arity_map, elem, 0) == 2
            op1 = pop!(stack)
            op2 = pop!(stack)
            ops = callbacks[elem]
            push!(stack, ops(op1, op2))
        elseif get(arity_map, elem, 0) == 1
            op1 = pop!(stack)
            ops = callbacks[elem]
            push!(stack, ops(op1))
        else
            push!(stack, elem isa Int8 ? nodes[elem] : elem)
        end
    end
    return pre_len == 1 ? last(stack) : stack
end

@inline function retrieve_constants_from_node(node::Node)
    constants = AbstractFloat[]
    for op in node
        if op isa AbstractNode && op.degree == 0 && op.constant
            push!(constants, convert(AbstractFloat, op.val))
        end
    end
    constants
end


"""
    optimize_constants!(
        node::Node,
        loss::Function;
        opt_method::Symbol=:cg,
        max_iterations::Int=250,
        n_restarts::Int=3
    )

Optimizes constant values in a symbolic expression tree to minimize a given loss function.

# Arguments
- `node::Node`: Expression tree containing constants to optimize
- `loss::Function`: Loss function to minimize
- `opt_method::Symbol=:cg`: Optimization method (:newton, :cg, or other for NelderMead)
- `max_iterations::Int=250`: Maximum iterations per optimization attempt
- `n_restarts::Int=3`: Number of random restarts to avoid local minima

# Returns
Tuple containing:
- `best_node::Node`: Expression tree with optimized constants
- `best_loss::Float64`: Final loss value achieved

# Optimization Methods
## Available Algorithms
- `:newton`: Newton's method with backtracking line search
- `:cg`: Conjugate Gradient with backtracking line search
- `other`: Nelder-Mead simplex method (default fallback)

## Random Restart Strategy
1. First attempt uses original constants
2. Subsequent restarts randomly perturb constants:
   - Multiplication by (1 + 0.5 * randn())
   - Targets only degree-0 constant nodes
   - Preserves variable nodes

# Example
```julia
# Create expression with constants
expr = Node(*, [
    Node(1.5),  # constant to optimize
    Node(x, degree=1)  # variable
])

# Define loss function
loss(node) = sum((node(x_data) .- y_data).^2)

# Optimize constants
optimized_expr, final_loss = optimize_constants!(
    expr,
    loss;
    opt_method=:cg,
    max_iterations=500,
    n_restarts=5
)
```

# Implementation Notes
## Performance
- `@inline` directive for function inlining
- Early return for expressions without constants
- Efficient constant counting and modification
- Minimal memory allocation during optimization

## Algorithm Selection
- Newton's method: Second-order optimization
- Conjugate Gradient: First-order optimization
- Nelder-Mead: Derivative-free optimization

## Optimization Process
1. Count constants in expression
2. Establish baseline loss
3. For each restart:
   - Create new expression copy (except first attempt)
   - Randomly perturb constants (except first attempt)
   - Optimize using selected algorithm
   - Update best result if improved
4. Return best found solution

# Notes
- Modifies input node during optimization
- Uses Optim.jl for optimization algorithms
- Supports automatic differentiation through Zygote
- Multiple restarts help avoid local minima

See also: [`DynamicExpressions.Node`](@ref), [`Optim.optimize`](@ref), [`LineSearches.BackTracking`](@ref)
"""
@inline function optimize_constants!(
    node::Node,
    loss::Function;
    opt_method::Symbol=:cg,
    max_iterations::Int=250,
    n_restarts::Int=3
)

    nconst = count_constant_nodes(node)
    baseline = loss(node)

    if nconst == 0
        return node, baseline
    end

    
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
        #needs to be revised!
        x0, refs = get_scalar_constants(current_node)


        function opt_step(x::AbstractVector)
            set_scalar_constants!(current_node,x, refs)
            loss(current_node)
        end
        result = Optim.optimize(opt_step, x0, algorithm, optimizer_options)

        if result.minimum < best_loss
            best_node = current_node
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
    temp_filename = filename * ".tmp"
    open(temp_filename, "w") do io
        serialize(io, state)
        flush(io)
    end
    mv(temp_filename, filename; force=true)
    return true
end

function load_state(filename::String)
    open(filename, "r") do io
        return deserialize(io)
    end
end


function train_test_split(
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    train_ratio::T=0.9,
    consider::Int=1
) where {T<:AbstractFloat}

    data = hcat(X, y)


    data = data[shuffle(1:size(data, 1)), :]


    split_point = floor(Int, size(data, 1) * train_ratio)


    data_train = data[1:split_point, :]
    data_test = data[(split_point+1):end, :]


    x_train = T.(data_train[1:consider:end, 1:(end-1)])
    y_train = T.(data_train[1:consider:end, end])

    x_test = T.(data_test[1:consider:end, 1:(end-1)])
    y_test = T.(data_test[1:consider:end, end])

    return x_train, y_train, x_test, y_test
end

function one_hot_mean(vectors::Vector{Vector{T}}, k::Int) where T <: Integer
    if isempty(vectors)
        return T[]
    end
    
    max_value = maximum(maximum(v) for v in vectors if !isempty(v))
    
    max_length = maximum(length(v) for v in vectors)
    
    frequency_matrix = zeros(Float64, max_length, max_value)
    
    position_counts = zeros(Int, max_length)
    
    for vec in vectors
        for (i, val) in enumerate(vec)
            frequency_matrix[i, convert(Int, val)] += 1
            position_counts[i] += 1
        end
    end
    
    for i in 1:max_length
        if position_counts[i] > 0
            frequency_matrix[i, :] ./= position_counts[i]
        end
    end
    
    result = Vector{T}(undef, max_length)
    
    for i in 1:max_length
        sorted_indices = sortperm(frequency_matrix[i, :], rev=true)
        if k == 1
            result[i] = convert(T, sorted_indices[1])
        else
            top_k_indices = sorted_indices[1:min(k, length(sorted_indices))]
            top_k_probs = frequency_matrix[i, top_k_indices]
            top_k_probs = top_k_probs ./ sum(top_k_probs)
            result[i] = convert(T, sample(top_k_indices, Weights(top_k_probs)))
        end
    end
    
    return result
end


function select_closest_points(lhs_points, normalized_features, n_samples)
    selected_indices = zeros(Int, n_samples)
    remaining_indices = Set(1:size(normalized_features, 2))
    
    # Build KD-tree once with all points
    kdtree = KDTree(normalized_features)
    
    for i in 1:n_samples
        # Find nearest neighbors among remaining points
        idxs, dists = knn(kdtree, lhs_points[:, i], 1, true, j -> j ∉ remaining_indices)
        best_idx = idxs[1]
        
        selected_indices[i] = best_idx
        delete!(remaining_indices, best_idx)
    end
    
    return selected_indices
end

"""
    select_n_samples_lhs(equastacked_features, n_samples; seed=nothing)

Select a subset of equations using Latin Hypercube Sampling based on their characteristics.

Parameters:
- `stacked_features`: Array of equation information -> (n_features × n_probes)
- `n_samples`: Number of equations to select
- `seed`: Optional random seed for reproducibility

Returns:
- Indices of selected equations
"""
function select_n_samples_lhs(stacked_features::AbstractArray, n_samples::Int)
    _,test_len = size(stacked_features)
    invalid_mask = falses(test_len)
    for i in 1:test_len
        if any(isnan.(stacked_features[:, i])) || any(isinf.(stacked_features[:, i])) 
            invalid_mask[i] = true
        end
    end
    valid_indices = findall(.!invalid_mask)
    valid_features = stacked_features[:, valid_indices]
    normalized_features = normalize_features(valid_features)
    n_features, n_probes = size(normalized_features)

    bins = zeros(n_features, n_samples, 2)
    bins[:, :, 1] .= ((1:n_samples)' .- 1) ./ n_samples
    bins[:, :, 2] .= (1:n_samples)' ./ n_samples


    for i in 1:n_features
        shuffle!(view(bins,i,:,:))
    end

    rand_vals = rand(n_features, n_samples)
    lhs_points = bins[:,:,1] + rand_vals .* (bins[:,:,2] - bins[:,:,1])

    selected_indices = select_closest_points(lhs_points, normalized_features, n_samples)
    return valid_indices[selected_indices]
end


function normalize_features(features)

    feature_mins = minimum(features, dims=2)
    feature_maxs = maximum(features, dims=2)
    feature_ranges = feature_maxs .- feature_mins
    
    normalized = similar(features)
    
    normalized = @. ifelse(
        feature_ranges > 0,
        (features - feature_mins) / feature_ranges,
        0.5
    )
    
    return normalized
end

end