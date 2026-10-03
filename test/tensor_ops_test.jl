using Test
using Tensors
using OrderedCollections
using Random
using Statistics
using LinearAlgebra

# The tensor path evaluates a karva string with a stack machine over the whole batch of
# samples (`calc_stack_batch_tensor`), optionally into preallocated buffers. These tests
# check the operator semantics through that evaluator.
@testset "TensorRegUtils" begin
    dim, n = 3, 5
    t2 = [rand(Tensor{2,dim}) for _ in 1:n]
    v3 = [rand(Vec{dim}) for _ in 1:n]
    scalars = fill(2.0, n)

    callbacks = Dict{Int8,Any}(
        Int8(1) => AdditionNode(),
        Int8(2) => MultiplicationNode(),
        Int8(3) => DoubleContractionNode(),
        Int8(4) => TraceNode(),
        Int8(8) => SubtractionNode(),
    )
    # terminals: an index that is not an operator is looked up here
    nodes = Dict{Int8,Any}(
        Int8(5) => t2,
        Int8(6) => v3,
        Int8(7) => scalars,
    )

    run(rek) = calc_stack_batch_tensor(rek, callbacks, nodes, nothing)

    @testset "Basic Operations" begin
        # scalar multiplication, applied across the batch
        @test run(Int8[2, 6, 7]) ≈ v3 .* scalars

        # adding a second-order tensor to a vector is not a valid combination; the
        # evaluator signals it with a non-finite result rather than throwing, so the
        # fitness function can reject the individual
        invalid = run(Int8[1, 5, 6])
        @test invalid === NaN || all(!isfinite, invalid)

        @test run(Int8[8, 5, 5]) ≈ t2 .- t2
    end

    @testset "Tensor Operations" begin
        @test run(Int8[3, 5, 5]) ≈ [dcontract(a, a) for a in t2]
        @test run(Int8[4, 5]) ≈ [tr(a) for a in t2]
    end

    @testset "Complex Expressions" begin
        # tr(t2 + t2), i.e. the operators compose through the stack
        @test run(Int8[4, 1, 5, 5]) ≈ [tr(a + a) for a in t2]

        # (v3 * c) + v3
        @test run(Int8[1, 2, 6, 7, 6]) ≈ v3 .* scalars .+ v3
    end

    @testset "Buffered evaluation matches unbuffered" begin
        # predictT evaluates into such buffers; the result must match the allocating path
        gene_buff = 16
        buffers = Dict{Type,NTuple}(
            Vector{Float64} => Tuple([zeros(Float64, n) for _ in 1:gene_buff]),
            Vector{Vec{dim}} => Tuple([zeros(Vec{dim}, n) for _ in 1:gene_buff]),
            Vector{Tensor{1,dim}} => Tuple([zeros(Vec{dim}, n) for _ in 1:gene_buff]),
            Vector{Tensor{2,dim}} => Tuple([zeros(Tensor{2,dim}, n) for _ in 1:gene_buff]),
            Vector{Tensor{3,dim}} => Tuple([zeros(Tensor{3,dim}, n) for _ in 1:gene_buff]),
            Vector{Tensor{4,dim}} => Tuple([zeros(Tensor{4,dim}, n) for _ in 1:gene_buff]),
        )
        for rek in (Int8[2, 6, 7], Int8[3, 5, 5], Int8[4, 5], Int8[4, 1, 5, 5])
            @test calc_stack_batch_tensor(rek, callbacks, nodes, buffers) ≈ run(rek)
        end
    end
end

# The batched evaluator through GepTensorRegressor: allocate_buffers! sizes the per-thread
# buffers to the data, and predictT evaluates a karva string (`expression_raw`) in them.
@testset "GepTensorRegressor end to end" begin
    dim, n = 3, 30
    rng = MersenneTwister(1)
    rvec(r) = Vec{dim}(ntuple(_ -> rand(r), dim))
    x = [[rvec(rng) for _ in 1:n], [rvec(rng) for _ in 1:n]]
    y = [x[1][i] + x[2][i] for i in 1:n]

    build() = GepTensorRegressor(2; problem_dimension=dim,
        entered_non_terminals=Symbol[:+, :-, :*],
        gene_connections=Symbol[:+, :-],
        gene_count=2, head_len=4, feature_names=["a", "b"])

    make_loss(reg) = function (elem, validate::Bool)
        if isnan(mean(elem.fitness)) || validate
            l = try
                pred = predictT(reg, elem.expression_raw)
                (pred isa AbstractVector && length(pred) == n &&
                 eltype(pred) <: Vec{dim}) ? sum(norm.(pred .- y)) / n : 1e6
            catch
                1e6
            end
            elem.fitness = (isfinite(l) ? l : 1e6,)
        end
        return elem.fitness
    end

    reg = build()
    allocate_buffers!(reg, x)
    fit!(reg, 5, 200, make_loss(reg); hof=1)

    @test reg.best_models_ !== nothing
    @test !isnan(mean(reg.best_models_[1].fitness))

    # predictT on data the regressor was not fitted on: a different sample count, and
    # compared against the same expression evaluated from scratch on those columns
    rek = reg.best_models_[1].expression_raw
    m = n + 7
    x_new = [[rvec(rng) for _ in 1:m], [rvec(rng) for _ in 1:m]]
    fresh = predictT(reg, rek, x_new)
    @test fresh isa AbstractVector && length(fresh) == m
    direct = calc_stack_batch_tensor(rek, reg.toolbox_.callbacks,
        Dict{Int8,Any}(Int8(1) => x_new[1], Int8(2) => x_new[2]), nothing)
    @test fresh == direct
    @test copy(predictT(reg, rek)) != fresh[1:n]
end
