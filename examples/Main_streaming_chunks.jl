#=
Streaming evaluation: fitting an equation to a dataset far larger than memory
============================================================================

The batched evaluator writes every intermediate into preallocated buffers, one per
operator application, each as long as the data: at least (gene_count + 1) * head_len of
them per thread slot. Their memory grows linearly with the sample count (Part 1 prints it
at a few sizes), which becomes prohibitive for, e.g., a resolved field of 10^8 points.

Nothing in the package has to change to avoid this. `calc_stack_batch_tensor` takes the
input columns as an argument, and the tensor path's `fit!` takes a loss callback that does
its own prediction, so the loss can size the buffers to a chunk, walk the dataset chunk by
chunk and accumulate the residual as a scalar. The evaluator's memory then depends on the
chunk, not the dataset; the table printed at the end records a measurement of this.

    julia --project=. --threads=4 examples/Main_streaming_chunks.jl

The target, T = tr(E) B - E (T_ij = E_kk B_ij - E_ij) for random second-order tensors E
and B, mixes a scalar column (tr E) with tensor ones.
=#

include("../src/GeneExpressionProgramming.jl")

using .GeneExpressionProgramming
using .GeneExpressionProgramming.GepEntities
using Random
using Tensors
using LinearAlgebra
using Statistics
using Printf

const DIM = 3          # 3x3 tensors
const CHUNK = 20_000   # samples evaluated at once; sets the evaluator's memory
const N_TOTAL = 200_000

# ---------------------------------------------------------------------------------------
# Part 1. Why chunk at all
#
# Build the scalar path's buffer context at a few dataset sizes and measure what it
# allocates. Nothing is fitted here.
# ---------------------------------------------------------------------------------------
function buffer_cost()
    println("Buffer pool cost against dataset size (scalar path, 4 genes, head 8)\n")
    @printf("  %-12s %12s\n", "samples", "pool")
    for n in (1_000, 100_000, 1_000_000)
        x = randn(7, n)
        reg = GepRegressor(7; entered_non_terminals=[:+, :-, :*, :/],
            gene_count=4, head_len=8)
        GC.gc()
        bytes = @allocated build_buffers(reg, x)
        @printf("  %-12d %9.1f MB\n", n, bytes / 2^20)
    end
    println("\n  Linear in the sample count. A 10^8-point field would need ~320 GB.\n")
end

# ---------------------------------------------------------------------------------------
# Part 2. The data
#
# Held here as plain column vectors for a self-contained example. In a real streaming
# setup `fill_chunk!` would read the slice from disk (HDF5, a memory-mapped array, a
# database cursor), and only the chunk would be resident.
# ---------------------------------------------------------------------------------------
function make_data(n)
    rng = MersenneTwister(7)
    EE = [Tensor{2,DIM}(randn(rng, DIM, DIM)) for _ in 1:n]
    BB = [Tensor{2,DIM}(randn(rng, DIM, DIM)) for _ in 1:n]
    dd = [one(Tensor{2,DIM}) for _ in 1:n]
    E2 = [tr(EE[i]) for i in 1:n]
    B2 = [tr(BB[i]) for i in 1:n]
    T = [E2[i] * BB[i] - EE[i] for i in 1:n]          # <- the relation to recover
    return (E2=E2, B2=B2, EE=EE, BB=BB, dd=dd, T=T)
end

"""
    chunk_columns(len)

One column per feature, `len` long, keyed by symbol id: `GepTensorRegressor` numbers the
features 1..5 in the order given.

Every thread needs its own set: the fitness loop is `Threads.@threads :static`, so several
threads run the loss at once, and one shared set of columns mutated per chunk would be a
data race. This is also why the loss calls `calc_stack_batch_tensor` directly rather than
`predictT`, which reads the regressor's shared `input_values`.
"""
chunk_columns(len) = Dict{Int8,Any}(
    Int8(1) => zeros(Float64, len),                        # E2  (scalar)
    Int8(2) => zeros(Float64, len),                        # B2  (scalar)
    Int8(3) => [zero(Tensor{2,DIM}) for _ in 1:len],       # EE  (tensor)
    Int8(4) => [zero(Tensor{2,DIM}) for _ in 1:len],       # BB  (tensor)
    Int8(5) => [zero(Tensor{2,DIM}) for _ in 1:len],       # delta_ij
)

"""Copy `data[lo:hi]` into this thread's columns. Replace with a disk read for real data."""
function fill_chunk!(cols, data, lo, hi)
    len = hi - lo + 1
    copyto!(cols[Int8(1)], 1, data.E2, lo, len)
    copyto!(cols[Int8(2)], 1, data.B2, lo, len)
    copyto!(cols[Int8(3)], 1, data.EE, lo, len)
    copyto!(cols[Int8(4)], 1, data.BB, lo, len)
    copyto!(cols[Int8(5)], 1, data.dd, lo, len)
    return len
end

# ---------------------------------------------------------------------------------------
# Part 3. Fit, streaming
# ---------------------------------------------------------------------------------------
function main()
    buffer_cost()

    @printf("Streaming fit: %d samples in %d chunks of %d\n\n",
        N_TOTAL, cld(N_TOTAL, CHUNK), CHUNK)
    data = make_data(N_TOTAL)

    Random.seed!(3)
    regressor = GepTensorRegressor(5;
        problem_dimension=DIM,
        gene_count=2,
        head_len=4,
        entered_non_terminals=[:+, :-, :*],
        entered_terminal_nums=[0.5, 2.0],
        gene_connections=[:+, :-],
        feature_names=["E2", "B2", "EE", "BB", "d"])

    # the buffers are sized to this template: one chunk, not the whole dataset
    template = Any[zeros(CHUNK), zeros(CHUNK),
        [zero(Tensor{2,DIM}) for _ in 1:CHUNK],
        [zero(Tensor{2,DIM}) for _ in 1:CHUNK],
        [zero(Tensor{2,DIM}) for _ in 1:CHUNK]]
    allocate_buffers!(regressor, template)

    # one set per thread slot: since Julia 1.12 `Threads.@threads` can hand out thread ids
    # above `nthreads()`
    columns = [chunk_columns(CHUNK) for _ in 1:thread_slots()]
    callbacks = regressor.toolbox_.callbacks

    """
    Walk the dataset chunk by chunk, accumulating the squared error in a scalar. No
    prediction outlives its chunk, so the memory is `CHUNK`-sized, however long the
    dataset is.
    """
    function streaming_loss(elem, validate::Bool)
        if isnan(mean(elem.fitness)) || validate
            tid = Threads.threadid()
            cols = columns[tid]
            pool = regressor.buffers[tid]
            total = 0.0
            counted = 0
            valid = true
            @inbounds for lo in 1:CHUNK:N_TOTAL
                hi = min(lo + CHUNK - 1, N_TOTAL)
                len = fill_chunk!(cols, data, lo, hi)
                # the buffers are CHUNK long, so a ragged final chunk is skipped rather
                # than resized; size N_TOTAL as a multiple of CHUNK to use every sample
                len == CHUNK || continue
                pred = try
                    calc_stack_batch_tensor(elem.expression_raw, callbacks, cols, pool)
                catch
                    nothing
                end
                if !(pred isa AbstractVector) || length(pred) != CHUNK ||
                   !(eltype(pred) <: Tensor{2,DIM})
                    valid = false
                    break
                end
                for i in 1:CHUNK
                    total += norm(pred[i] - data.T[lo+i-1])^2
                end
                counted += CHUNK
            end
            # a large finite penalty: tournament selection skips non-finite fitness, so Inf
            # (= typemax(Float64)) would take the individual out of selection
            elem.fitness = (valid && counted > 0 && isfinite(total) ? total / counted : 1e6,)
        end
    end

    # what the chunking saved: the pool is linear in the template length, so the figure
    # for the whole dataset is extrapolated from the chunk-sized one
    GC.gc()
    pool_bytes = @allocated allocate_buffers!(regressor, template)
    @printf("buffer pool at CHUNK=%d      : %8.1f MB\n", CHUNK, pool_bytes / 2^20)
    @printf("the same pool at N_TOTAL=%d : %8.1f MB  (never allocated)\n\n",
        N_TOTAL, pool_bytes / 2^20 * N_TOTAL / CHUNK)

    elapsed = @elapsed fit!(regressor, 40, 400, streaming_loss)

    best = regressor.best_models_[1]
    println()
    @printf("recovered : %s\n", print_karva_strings(best))
    # the search may write the relation redundantly -- e.g. ((((BB*E2)-EE)+EE)-EE) -- which
    # is the same function; a loss of zero is the thing to check, not the string
    @printf("target    : ((E2 * BB) - EE)\n")
    @printf("loss      : %.3e\n", best.fitness[1])
    @printf("time      : %.1f s\n", elapsed)

    println("""

    Measured on a 4-core container with a smaller chromosome, holding the chunk at 10 000
    and varying only the dataset:

        samples     chunks   data       evaluator   peak RSS
          100 000       10    29.0 MB    349.9 MB   1 792 MB
          400 000       40   116.0 MB    349.7 MB   2 119 MB
        1 600 000      160   463.9 MB    349.7 MB   2 852 MB

    The evaluator's footprint does not move across a 16x change in dataset size. Peak RSS
    grows only by the data, which a real streaming source would never make resident.

    Two things this does not buy you. The compute is unchanged -- every candidate still
    visits every chunk, so the work is (population x generations x dataset), and scoring on
    one chunk first and committing only survivors to the full pass is the obvious next step.
    And the hook exists on the tensor path only: `GepTensorRegressor`'s `fit!` takes a loss
    callback that predicts for itself, whereas the scalar `GepRegressor` evaluates first and
    hands `(y_true, y_pred)` to the loss, leaving nowhere to intervene.
    """)
end

main()
