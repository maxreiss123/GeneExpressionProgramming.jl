"""
    GepUtils

Utilities shared by the GEP modules:

- the scalar function library `FUNCTION_LIB_COMMON`, with its arities
  (`ARITY_LIB_COMMON`) and renderers (`FUNCTION_STRINGIFY`);
- karva-string helpers: `compile_djl_datatype`, a fold used to render equations, and
  `find_indices_with_sum`, which finds where a gene's expression ends;
- history recording: `HistoryRecorder`, `OptimizationHistory`, `record!`,
  `close_recorder!`, `get_history_arrays`;
- data helpers: `train_test_split`, `minmax_scale`, `select_n_samples_lhs`,
  `one_hot_mean`;
- `save_state`/`load_state`, `split_rng`, `thread_slots` and `isclose`.
"""
module GepUtils

export find_indices_with_sum, compile_djl_datatype, minmax_scale, isclose
export save_state, load_state
export record_history!, record!, close_recorder!
export HistoryRecorder, OptimizationHistory, get_history_arrays, one_hot_mean, FUNCTION_STRINGIFY
export ConsensusSampler, consensus_draw
export train_test_split, select_n_samples_lhs, thread_slots, allfinite
export FUNCTION_LIB_COMMON, ARITY_LIB_COMMON
export split_rng

using OrderedCollections
using LinearAlgebra
using Optim
using LineSearches
using Serialization
using Statistics
using Random
using Tensors
using StatsBase
using NearestNeighbors
using Random123
using Base.Threads: @spawn


function sqr(x::Vector{T}) where {T<:AbstractFloat}
    return x .* x
end

function sqr(x::T) where {T<:Number}
    return x * x
end

"""
    FUNCTION_LIB_COMMON::Dict{Symbol,Function}

The scalar functions a regressor can be built from, keyed by name:

- arithmetic: `+`, `-`, `*`, `/`, `^`, `min`, `max`
- rounding: `floor`, `ceil`, `round`
- exponential and logarithmic: `exp`, `log`, `log10`, `log2`
- trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- other: `abs`, `sqr` (the square), `sqrt`, `sign`

A new function also needs its arity in `ARITY_LIB_COMMON`, a renderer in
`FUNCTION_STRINGIFY`, dimension handlers in `FUNCTION_LIB_FORWARD_COMMON` and
`FUNCTION_LIB_BACKWARD_COMMON`, and a batched operator in `TENSOR_NODES`.
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


"""
    FUNCTION_STRINGIFY::Dict{Symbol,Function}

Renderers for the functions of `FUNCTION_LIB_COMMON`, under the same names; each builds
a string from its arguments. Infix operators parenthesise their result, since flat text
would lose the nesting that the prefix-order karva string carries.
"""
const FUNCTION_STRINGIFY = Dict{Symbol,Function}(
    :+ => (args...) -> "(" * join(args, " + ") * ")",
    :- => (args...) -> length(args) == 1 ? "(-$(args[1]))" : "(" * join(args, " - ") * ")",
    :* => (args...) -> "(" * join(args, " * ") * ")",
    :/ => (args...) -> "(" * join(args, " / ") * ")",
    :^ => (args...) -> "($(args[1])^$(args[2]))",
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
    :sign => a -> "sign($a)",
    :floor => a -> "floor($a)",
    :ceil => a -> "ceil($a)"
)

"""
    ARITY_LIB_COMMON::Dict{Symbol,Int8}

Arity of each function in `FUNCTION_LIB_COMMON`: 2 for `+`, `-`, `*`, `/`, `^`, `min`
and `max`, 1 for the rest.
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
    :sign => 1,
    :sqr => 1
)

"""
    OptimizationHistory(epochs::Int, T)

Training and validation loss per epoch, preallocated for `epochs` epochs. `T` is the
loss type: a float, or `Tuple` for fitness tuples. Indexing and iteration yield
`(train_loss=..., val_loss=...)`. Epochs never recorded, e.g. after an early stop, stay
uninitialised.
"""
struct OptimizationHistory{T<:Union{AbstractFloat,Tuple}}
    train_loss::Vector{T}
    val_loss::Vector{T}

    function OptimizationHistory(epochs::Int, ::Type{T}) where {T<:Union{AbstractFloat,Tuple}}
        return new{T}(
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
            val_loss=hist.val_loss[state]
        ),
        state + 1
    )
end

Base.length(hist::OptimizationHistory) = length(hist.train_loss)

Base.size(hist::OptimizationHistory) = (length(hist.train_loss),)

function Base.getindex(hist::OptimizationHistory, i::Int)
    return (
        train_loss=hist.train_loss[i],
        val_loss=hist.val_loss[i]
    )
end


Base.firstindex(hist::OptimizationHistory) = 1
Base.lastindex(hist::OptimizationHistory) = length(hist)


function Base.show(io::IO, hist::OptimizationHistory)
    println(io, "OptimizationHistory{$(eltype(hist.train_loss))} with $(length(hist)) epochs")
end


"""
    get_history_arrays(hist::OptimizationHistory)

The recorded losses as `(train_loss=..., val_loss=...)`, one vector each (not copies).
"""
function get_history_arrays(hist::OptimizationHistory)
    return (
        train_loss=hist.train_loss,
        val_loss=hist.val_loss
    )
end


"""
    HistoryRecorder(epochs::Int, T; buffer_size::Int=32)

Records the training and validation loss of each epoch on a background task. `record!`
puts `(epoch, train_loss, val_loss)` on `channel`, a `Channel{Tuple{Int,T,T}}` holding
up to `buffer_size` entries; `task`, spawned by the constructor, writes them into
`history::OptimizationHistory{T}`. `T` is the loss type: a float, or `Tuple` for
fitness tuples, as `runGep` records them. Call `close_recorder!` before reading
`history`.

# Example
```julia
recorder = HistoryRecorder(100, Float64)
for epoch in 1:100
    record!(recorder, epoch, train_loss, val_loss)
end
close_recorder!(recorder)
recorder.history.train_loss
```
"""
struct HistoryRecorder{T<:Union{AbstractFloat,Tuple}}
    channel::Channel{Tuple{Int,T,T}}
    task::Task
    history::OptimizationHistory{T}

    function HistoryRecorder(epochs::Int, ::Type{T}; buffer_size::Int=32) where {T<:Union{AbstractFloat,Tuple}}
        history = OptimizationHistory(epochs, T)
        channel = Channel{Tuple{Int,T,T}}(buffer_size)
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

"""
    record_history!(channel, history::OptimizationHistory)

Write each `(epoch, train_loss, val_loss)` taken from `channel` into `history` until the
channel is closed and drained. The task of a `HistoryRecorder` runs this.
"""
@inline function record_history!(
    channel::Channel{Tuple{Int,T,T}},
    history::OptimizationHistory{T}
) where {T<:Union{AbstractFloat,Tuple}}
    for (epoch, train_loss, val_loss) in channel
        @inbounds begin
            history.train_loss[epoch] = train_loss
            history.val_loss[epoch] = val_loss
        end
    end
end

"""
    record!(recorder::HistoryRecorder{T}, epoch::Int, train_loss::T, val_loss::T)

Queue the losses of `epoch` for recording. Blocks only while the channel is full.
"""
@inline function record!(
    recorder::HistoryRecorder{T},
    epoch::Int,
    train_loss::T,
    val_loss::T
) where {T<:Union{AbstractFloat,Tuple}}
    put!(recorder.channel, (epoch, train_loss, val_loss))
end

"""
    close_recorder!(recorder::HistoryRecorder)

Close the recorder's channel and wait until every queued epoch is in `recorder.history`.
"""
@inline function close_recorder!(recorder::HistoryRecorder)
    close(recorder.channel)
    wait(recorder.task)
end


"""
    allfinite(A::AbstractArray{<:AbstractFloat})

`all(isfinite, A)` without a branch per element: `x - x` is `0` for a finite `x` and
`NaN` for `Inf`, `-Inf` and `NaN`, so the sum of these terms is zero exactly when every
element is finite, and it cannot overflow.
"""
@inline function allfinite(A::AbstractArray{T}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i in eachindex(A)
        s += A[i] - A[i]
    end
    return iszero(s)
end

"""
    isclose(a, b; rtol=1e-5, atol=1e-8)

`abs(a - b) <= atol + rtol * abs(b)`: the tolerance is relative to `b`, so the test is
not symmetric. `a`, `b`, `rtol` and `atol` must share one type, so with the default
tolerances `a` and `b` must be `Float64`.
"""
function isclose(a::T, b::T; rtol::T=1e-5, atol::T=1e-8) where {T<:Number}
    return abs(a - b) <= (atol + rtol * abs(b))
end

"""
    thread_slots()

The length a per-thread buffer vector needs to be indexed by `Threads.threadid()`:
`Threads.maxthreadid()`, or `Threads.nthreads()` on Julia versions without it.

`nthreads()` counts only the default thread pool, while `threadid()` numbers the threads
of every pool. Julia 1.12 starts one interactive thread by default and gives it id 1, so
on a plain launch the only worker thread has id 2 and a vector of `nthreads()` slots is
one short.
"""
@inline thread_slots() = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() :
                         Threads.nthreads()

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


"""
    find_indices_with_sum(arr::SubArray, target_sum::Int, num_indices::Int)

The first `num_indices` positions at which the cumulative sum of `arr` equals
`target_sum`, or `[length(arr)]` if there are fewer; `[1]` whenever `arr[1] == 0`.
`_karva_raw` passes a gene's arities, less one from the second position on, with
`target_sum = 0`: the cumulative sum counts the open argument slots, so its first zero
is where the gene's expression ends.
"""
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
    compile_djl_datatype(rek_string, arity_map, callbacks, nodes, pre_len)

Fold the karva string `rek_string` (prefix order) from right to left on a stack. A token
of arity 2 or 1 in `arity_map` pops its operands, the first operand from the top, and
pushes `callbacks[token](operands...)`. Any other token is a leaf: an `Int8` is replaced
by `nodes[token]`, anything else is pushed as it is. Input is not validated; a
malformed string or a token without a callback throws.

Folding starts at position `pre_len`. With `pre_len == 1` the top of the stack is
returned: the whole expression, folded. Otherwise the stack itself is returned; with
`pre_len = gene_count`, which skips a chromosome's `gene_count - 1` connectors, it holds
one entry per gene, last gene first.

With the renderers of `FUNCTION_STRINGIFY` as callbacks the fold returns the equation
as a string, which is how `print_karva_strings` uses it.

# Example
```julia
using OrderedCollections
arity = OrderedDict{Int8,Int8}(1 => 2, 2 => 2, 3 => 0, 4 => 0)
callbacks = Dict{Int8,Function}(1 => FUNCTION_STRINGIFY[:*], 2 => FUNCTION_STRINGIFY[:+])
nodes = OrderedDict{Int8,Any}(3 => "x1", 4 => 2.0)
compile_djl_datatype(Int8[1, 2, 3, 4, 3], arity, callbacks, nodes, 1)
# "((x1 + 2.0) * x1)"
```
"""
function compile_djl_datatype(rek_string::Vector, arity_map::AbstractDict, callbacks::AbstractDict,
    nodes::AbstractDict, pre_len::Int)
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


function _minmax_scale!(X::AbstractArray{T}; feature_range=(zero(T), one(T))) where {T<:AbstractFloat}
    min_vals = minimum(X, dims=1)
    max_vals = maximum(X, dims=1)
    range_width = max_vals .- min_vals

    a, b = feature_range
    scale = (b - a) ./ range_width

    @inbounds for j in axes(X, 2)
        if range_width[j] ≈ zero(T)
            X[:, j] .= (a + b) / 2
        else
            for i in axes(X, 1)
                X[i, j] = (X[i, j] - min_vals[j]) * scale[j] + a
            end
        end
    end

    return X
end

"""
    minmax_scale(X; feature_range=(zero(T), one(T)))

A copy of `X` with each column mapped linearly onto `feature_range`; a constant column
becomes the midpoint of the range.
"""
function minmax_scale(X::AbstractArray{T}; feature_range=(zero(T), one(T))) where {T<:AbstractFloat}
    return _minmax_scale!(copy(X); feature_range=feature_range)
end

"""
    save_state(filename::String, state)

Serialize `state` to `filename`. The data goes to `filename * ".tmp"` first and is then
moved into place, so an interrupted write leaves an existing file intact. Returns `true`.
"""
function save_state(filename::String, state::Any)
    temp_filename = filename * ".tmp"
    open(temp_filename, "w") do io
        serialize(io, state)
        flush(io)
    end
    mv(temp_filename, filename; force=true)
    return true
end

"""
    load_state(filename::String)

Deserialize and return the object that `save_state` wrote to `filename`.
"""
function load_state(filename::String)
    open(filename, "r") do io
        return deserialize(io)
    end
end


"""
    train_test_split(X, y; train_ratio=0.9, consider=1)

Shuffle the rows of `X` (samples × features) together with `y` and split them: the first
`floor(Int, n * train_ratio)` of the `n` rows train, the rest test. `consider` keeps
every `consider`-th row of each part. Returns `(x_train, y_train, x_test, y_test)`.

`X`, `y` and `train_ratio` share the element type `T`, so for data other than `Float64`
pass `train_ratio` as a `T`. The shuffle uses the global RNG.
"""
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

"""
    ConsensusSampler(vectors, k)

The per-position distributions `one_hot_mean` draws from, computed once for a set of
integer vectors: at each position, the (at most) `k` most frequent values there, most
frequent first (the smallest value on a tie), and their frequencies normalised to sum 1.
[`consensus_draw`](@ref) makes the draws. Values must be positive, since they index the
count table.
"""
struct ConsensusSampler{T<:Integer}
    k::Int
    top::Vector{Vector{Int}}
    probs::Vector{Vector{Float64}}
end

function ConsensusSampler(vectors::AbstractVector{<:AbstractVector{T}}, k::Int) where T<:Integer
    isempty(vectors) && return ConsensusSampler{T}(k, Vector{Int}[], Vector{Float64}[])
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
    top = Vector{Vector{Int}}(undef, max_length)
    probs = Vector{Vector{Float64}}(undef, max_length)
    for i in 1:max_length
        sorted_indices = sortperm(frequency_matrix[i, :], rev=true)
        top[i] = sorted_indices[1:min(k, length(sorted_indices))]
        top_k_probs = frequency_matrix[i, top[i]]
        probs[i] = top_k_probs ./ sum(top_k_probs)
    end
    return ConsensusSampler{T}(k, top, probs)
end

Base.isempty(s::ConsensusSampler) = isempty(s.top)

"""
    consensus_draw(sampler::ConsensusSampler; rng=Random.default_rng())

One consensus vector: at each position the most frequent value when `k == 1`, otherwise
one of the `k` most frequent, drawn with probability proportional to its count (one draw
from `rng` per position).
"""
function consensus_draw(s::ConsensusSampler{T}; rng::AbstractRNG=Random.default_rng()) where T
    result = Vector{T}(undef, length(s.top))
    for i in eachindex(s.top)
        result[i] = s.k == 1 ? convert(T, s.top[i][1]) :
                    convert(T, sample(rng, s.top[i], Weights(s.probs[i])))
    end
    return result
end

"""
    one_hot_mean(vectors::Vector{Vector{T}}, k::Int; rng=Random.default_rng())

Positionwise consensus of integer vectors, as long as the longest one. At each position
the values found there are counted; with `k == 1` the most frequent is taken (the
smallest on a tie), otherwise one of the `k` most frequent is drawn with probability
proportional to its count. Values must be positive, since they index the count table.
For many draws from the same vectors, build a [`ConsensusSampler`](@ref) once.
"""
function one_hot_mean(vectors::Vector{Vector{T}}, k::Int;
    rng::AbstractRNG=Random.default_rng()) where T <: Integer
    isempty(vectors) && return T[]
    return consensus_draw(ConsensusSampler(vectors, k); rng=rng)
end


function select_closest_points(lhs_points, normalized_features, n_samples)
    selected_indices = zeros(Int, n_samples)
    remaining_indices = Set(1:size(normalized_features, 2))
    
    # one tree over all columns; the search skips the ones already selected
    kdtree = KDTree(normalized_features)
    
    for i in 1:n_samples
        idxs, dists = knn(kdtree, lhs_points[:, i], 1, true, j -> j ∉ remaining_indices)
        best_idx = idxs[1]
        
        selected_indices[i] = best_idx
        delete!(remaining_indices, best_idx)
    end
    
    return selected_indices
end

"""
    select_n_samples_lhs(stacked_features::AbstractArray, n_samples::Int)

Choose `n_samples` columns of `stacked_features` (features × candidates, e.g. one column
per equation) spread over the feature space, and return their indices. Columns with a
`NaN` or `Inf` are dropped, each feature is min-max normalised to [0, 1] (to 0.5 where
it is constant), and each of `n_samples` random target points takes the nearest column
not yet chosen. The target points are meant as a Latin hypercube design, but the bin
shuffle mixes bin bounds across strata, so they are not stratified. `n_samples` must not
exceed the number of valid columns. Randomness comes from the global RNG.
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

"""
    split_rng(master_rng::Threefry4x, n::Int)

`n` new `Threefry4x` generators, one per parallel task, each keyed with four 32-bit
values drawn from `master_rng`, which advances. Drawing the keys serially keeps a seeded
run reproducible however the tasks are scheduled.
"""
function split_rng(master_rng::Threefry4x, n::Int)
    typeof(master_rng)
    subkeys = Vector{Threefry4x}(undef, n)
    for i in 1:n
        new_seed = (UInt64(rand(master_rng, UInt32)), UInt64(rand(master_rng, UInt32)), UInt64(rand(master_rng, UInt32)), UInt64(rand(master_rng, UInt32)))
        subkeys[i] = Threefry4x(UInt64, new_seed)
    end
    return subkeys
end



end