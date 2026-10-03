using Test
using Random
using Statistics
using LinearAlgebra

# Gene-wise linear scaling fits one least-squares coefficient per gene before scoring, so
# evolution searches for the structure and the coefficients are solved for rather than
# built from constants: the analogue of the tensor linear regression in arXiv:2507.01466.
@testset "Linear scaling" begin

    @testset "recovers coefficients no constant terminal supplies" begin
        Random.seed!(1)
        rng = MersenneTwister(1)
        n = 100
        x = randn(rng, 2, n)
        # the constant terminals are 0.5, 0.0 and one random value in [0, 1), so an
        # unscaled run has to build 3.7 and -1.25 from arithmetic and generally misses them
        y = 3.7 .* x[1, :] .^ 2 .- 1.25 .* x[2, :]

        function run(ls)
            Random.seed!(1)
            reg = GepRegressor(2; entered_features=[:x1, :x2],
                entered_non_terminals=[:+, :-, :*], gene_count=2, head_len=3)
            fit!(reg, 60, 800, x, y; loss_fun="mse", linear_scaling=ls)
            return reg
        end

        scaled = run(true)
        plain = run(false)
        @test mean(scaled.best_models_[1].fitness) < mean(plain.best_models_[1].fitness)
        @test mean(scaled.best_models_[1].fitness) < 1e-12

        # the coefficients are part of the returned model, which predicts with them
        @test vec(scaled(x)) ≈ y atol = 1e-6
    end

    @testset "the constant optimiser leaves scaled models alone" begin
        # a scaled model predicts with its gene weights, so constants tuned on the unscaled
        # expression would describe a different model than the one scored and returned
        rng = MersenneTwister(1)
        x = randn(rng, 2, 100)
        y = 3.7 .* x[1, :] .^ 2 .- 1.25 .* x[2, :] .+ 0.3
        for seed in 1:6
            Random.seed!(seed)
            reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=2, head_len=3)
            fit!(reg, 15, 200, x, y; loss_fun="mse", linear_scaling=true, optimization_epochs=1)
            @test all(m -> m.optimised_constants === nothing, reg.best_models_)
        end
    end

    @testset "least squares solve" begin
        rng = MersenneTwister(2)
        G = randn(rng, 50, 3)
        w = [2.0, -1.5, 0.25]
        @test solve_scaling(G, G * w) ≈ w atol = 1e-8

        # duplicated genes make columns collinear; the solve has to stay finite
        Gd = hcat(G[:, 1], G[:, 1], G[:, 2])
        @test all(isfinite, solve_scaling(Gd, Gd * [1.0, 1.0, 1.0]))
    end

    @testset "gene basis has one column per gene" begin
        Random.seed!(3)
        rng = MersenneTwister(3)
        n = 20
        x = randn(rng, 2, n)
        y = x[1, :] .+ x[2, :]
        reg = GepRegressor(2; entered_features=[:x1, :x2],
            entered_non_terminals=[:+, :-, :*], gene_count=3, head_len=3)
        fit!(reg, 5, 200, x, y; loss_fun="mse")
        ctx = build_buffers(reg, x)
        G = gene_basis(reg.best_models_[1], ctx, n)
        @test size(G) == (n, 3)
        @test all(isfinite, G)
    end

    @testset "needs + and * to write the scaled model" begin
        rng = MersenneTwister(4)
        x = randn(rng, 2, 20)
        y = vec(sum(x, dims=1))
        reg = GepRegressor(2; entered_features=[:x1, :x2],
            entered_non_terminals=[:-, :/], gene_count=2, head_len=3)
        @test_throws ArgumentError fit!(reg, 5, 200, x, y; loss_fun="mse", linear_scaling=true)
    end
end

# The batched stack machine is the only evaluator: it walks the karva string the search
# evolves and writes intermediates into preallocated buffers. A chromosome is the model,
# callable on new data.
@testset "Buffered evaluation" begin
    rng = MersenneTwister(11)
    n = 400
    x = randn(rng, 3, n)
    y = 2.0 .* x[1, :] .- x[2, :] .* x[3, :]

    Random.seed!(11)
    reg = GepRegressor(3; entered_features=[:x1, :x2, :x3],
        entered_non_terminals=[:+, :-, :*, :/], gene_count=3, head_len=5)
    fit!(reg, 10, 300, x, y; loss_fun="mse")
    ctx = build_buffers(reg, x)

    for m in reg.best_models_
        direct = GeneExpressionProgramming.GepRegression.buffered_predict(m, ctx)
        @test direct isa AbstractVector
        # the model called on data agrees with its evaluation in the search's context
        @test m(x) ≈ direct
    end

    @testset "buffer context" begin
        reg = GepRegressor(3; entered_features=[:x1, :x2, :x3],
            entered_non_terminals=[:+, :-, :*, :/], gene_count=2, head_len=4)
        ctx = build_buffers(reg, x)
        @test ctx !== nothing
        # one pool per thread id; since Julia 1.12 the ids can exceed nthreads()
        @test length(ctx.pools) == thread_slots() >= Threads.nthreads()
        # one column per terminal, each as long as the data
        @test all(v -> length(v) == n, values(ctx.nodes))
    end

    @testset ":auto picks by sample count" begin
        # `fit!` ignores its `buffered` keyword, so nothing picks by sample count; only
        # the exported constant is checked
        @test BUFFERED_EVAL_MIN_SAMPLES > 0
        @test size(x, 2) < BUFFERED_EVAL_MIN_SAMPLES
    end
end
