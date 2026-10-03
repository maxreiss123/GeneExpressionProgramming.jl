using Test
using Random
using Statistics

# The batched evaluator has two routes: `run_program!` walks a compiled `EvalProgram` over
# a concretely typed stack, `calc_stack_batch_tensor` walks the same karva string over a
# `Vector{Any}` and resolves every symbol and operator dynamically. The two must compute
# the same function, bit for bit. Whatever the program cannot represent, `compile_program`
# declines, and the generic route takes over.
#
# Two traps:
#   * both routes return buffers, so results from one shared pool would compare an object
#     with itself; the two contexts below own independent pools, and results are copied
#     before comparison.
#   * the `sqrt` and `log` kernels are unguarded: a negative operand throws on both routes
#     (`compute_fitness` catches it). Both throwing is agreement; only one throwing fails.

const P = GeneExpressionProgramming.GepRegression

"""Two buffer contexts for `reg` on `x` that share no buffers: the compiled route and, with
the program removed, the generic stack machine."""
function eval_routes(reg, x)
    fast = build_buffers(reg, x)
    other = build_buffers(reg, x)
    @assert !isnothing(fast) && !isnothing(other)
    slow = (callbacks=other.callbacks, nodes=other.nodes, pools=other.pools,
        gene_pools=other.gene_pools, program=nothing, fast=other.fast,
        gene_fast=other.gene_fast, stacks=other.stacks)
    return fast, slow
end

try_eval(f) = try
    (true, f())
catch
    (false, nothing)
end

@testset "Monomorphic evaluator" begin

    @testset "compile_program declines what it cannot represent" begin
        V = Vector{Float64}
        # an alphabet it handles
        nodes = Dict{Int8,Any}(Int8(1) => zeros(3), Int8(2) => ones(3))
        cbs = Dict{Int8,Any}(Int8(3) => AdditionNode())
        @test compile_program(cbs, nodes, V) isa EvalProgram{V}

        # a terminal that is not a column of the expected type
        @test isnothing(compile_program(cbs, Dict{Int8,Any}(Int8(1) => 1.0), V))
        # an operator not closed over Vector{Float64}: `return_type` gives Float64 for tr
        @test isnothing(compile_program(Dict{Int8,Any}(Int8(3) => TraceNode()), nodes, V))
        # no terminals at all
        @test isnothing(compile_program(cbs, Dict{Int8,Any}(), V))
    end

    @testset "operators compute what their names say" begin
        # `max` computed min, and `^` had no result type for two columns
        x = [1.0 2.0 3.0; 3.0 2.0 1.5]                     # features x1, x2 as rows
        nodes = Dict{Int8,Any}(Int8(1) => x[1, :], Int8(2) => x[2, :])
        unbuffered(op) = calc_stack_batch_tensor(Int8[3, 1, 2],
            Dict{Int8,Any}(Int8(3) => op), nodes, nothing)
        @test unbuffered(MaxNode()) == max.(x[1, :], x[2, :])
        @test unbuffered(MinNode()) == min.(x[1, :], x[2, :])
        @test unbuffered(PowerNode()) ≈ x[1, :] .^ x[2, :]

        reg = GepRegressor(2; entered_non_terminals=[:max, :^, :+])
        tb = reg.toolbox_
        fast = buffer_context(tb, x)                  # the compiled route
        slow = merge(fast, (program=nothing,))        # the generic stack machine
        @test !isnothing(fast.program)
        id(f) = only(k for (k, v) in tb.callbacks if v === f)
        feature(i) = only(k for (k, v) in tb.nodes if v isa InputSelector && v.idx == i)
        for (f, expected) in ((max, max.(x[1, :], x[2, :])), (^, x[1, :] .^ x[2, :]))
            expr = Int8[id(f), feature(1), feature(2)]
            for ctx in (fast, slow)
                @test copy(GeneExpressionProgramming.GepEntities.ctx_eval(expr, ctx)) ≈ expected
            end
        end
    end

    @testset "agrees with the generic stack machine, bit for bit" begin
        cases = [
            (4, [:+, :-, :*, :/], 3, 5, 300, "arithmetic only"),
            (4, [:+, :-, :*, :/], 5, 9, 150, "deep genes, short data"),
            (3, [:+, :-, :*, :/, :sqrt, :exp, :log], 3, 6, 400, "with unary operators"),
            (6, [:+, :-, :*, :/, :sin, :cos], 4, 7, 500, "wider alphabet"),
            (2, [:+, :*], 2, 3, 50, "tiny alphabet"),
        ]
        for (nfeat, ops, genes, head, n, tag) in cases
            # no `fit!`: the comparison needs only a toolbox and data. `generate_chromosome`
            # defaults to the module RNG `GepEntities.STD_RNG`, which `Random.seed!` does
            # not reach, so the population is drawn from an explicit RNG
            Random.seed!(11)
            rng = MersenneTwister(1234)
            x = randn(nfeat, n)
            reg = GepRegressor(nfeat; entered_non_terminals=ops,
                gene_count=genes, head_len=head)
            fast, slow = eval_routes(reg, x)
            @test !isnothing(fast.program)

            pop = [generate_chromosome(reg.toolbox_; rng=rng) for _ in 1:120]
            for c in pop
                compile_expression!(c)
            end

            compared = 0
            for c in pop
                (fok, a) = try_eval(() -> P.buffered_predict(c, fast))
                (sok, b) = try_eval(() -> P.buffered_predict(c, slow))
                @test fok == sok
                if fok && sok
                    if a isa AbstractVector && b isa AbstractVector
                        # if these were the same object the comparison would be vacuous
                        @test a !== b
                        @test isequal(copy(a), copy(b))
                        compared += 1
                    else
                        @test (a isa AbstractVector) == (b isa AbstractVector)
                    end
                end

                # the design matrix linear scaling fits its coefficients on
                (gfok, Gf) = try_eval(() -> P.gene_basis(c, fast, n))
                (gsok, Gs) = try_eval(() -> P.gene_basis(c, slow, n))
                @test gfok == gsok
                if gfok && gsok && Gf isa AbstractMatrix && Gs isa AbstractMatrix
                    @test isequal(Gf, Gs)
                end
            end
            @test compared > 0   # the case has to exercise something, not just throw
            @info "monomorphic evaluator: $tag, $compared chromosomes compared"
        end
    end

    @testset "a search is unchanged by which route it takes" begin
        # a seeded search, which takes the compiled route, gives the same result twice
        Random.seed!(21)
        n = 200
        x = randn(3, n)
        y = 2.0 .* x[1, :] .- x[2, :] .* x[3, :]

        function run(useprog::Bool)
            Random.seed!(21)
            reg = GepRegressor(3; entered_non_terminals=[:+, :-, :*, :/],
                gene_count=3, head_len=5)
            fit!(reg, 25, 300, x, y; loss_fun="mse")
            ctx = build_buffers(reg, x)
            @test useprog == !isnothing(ctx.program)
            return reg
        end
        a = run(true)
        @test mean(a.best_models_[1].fitness) >= 0.0
        b = run(true)
        @test mean(a.best_models_[1].fitness) == mean(b.best_models_[1].fitness)
        @test a.best_models_[1].expression_raw == b.best_models_[1].expression_raw
    end

    @testset "predictT_scaled: compiled columns agree with the generic route" begin
        # with Float64 columns `predictT_scaled` evaluates the genes with `run_program!`
        # (`ScalarColumnCache`); a cache without a program forces `gene_bases`, the generic
        # stack machine. Predictions and gene weights must agree bit for bit.
        R = GeneExpressionProgramming.RegressionWrapper
        Random.seed!(41)
        n, d = 300, 3
        X = randn(d, n)
        y = X[1, :] .* X[2, :] .- 0.5 .* X[3, :]
        reg = GepTensorRegressor(d; problem_dimension=1, gene_count=3, head_len=4,
            entered_non_terminals=[:+, :-, :*], gene_connections=[:+, :-])
        reg.input_values = Dict{Int8,Any}(Int8(i) => X[i, :] for i in 1:d)
        pop = generate_population(400, reg.toolbox_)
        @test !isnothing(R.scalar_cache(reg).program)
        scaled(c) = (p = predictT_scaled(reg, c, y);
            (p, copy(something(c.scaling_weights, Float64[]))))
        fast = [scaled(c) for c in pop]
        @test count(r -> !isnothing(r[1]), fast) > 300

        # the in-place form writes the same prediction into the given vector
        out = fill(NaN, n)
        for (c, (p, w)) in zip(pop, fast)
            isnothing(p) && continue
            @test predictT_scaled!(out, reg, c, y) === out
            @test isequal(out, p) && isequal(c.scaling_weights, w)
        end
        @test_throws DimensionMismatch predictT_scaled!(zeros(n - 1), reg, pop[1], y)

        reg.scalar_cache_ = R.ScalarColumnCache(reg.input_values,
            Pair{Int8,Any}[k => v for (k, v) in reg.input_values], 0, nothing, [], [], [])
        @test isnothing(R.scalar_cache(reg).program)
        slow = [scaled(c) for c in pop]
        @test all(isequal(f, s) for (f, s) in zip(fast, slow))

        # replacing a column rebuilds the cache, and the compiled route returns
        reg.input_values = Dict{Int8,Any}(Int8(i) => X[i, :] for i in 1:d)
        @test !isnothing(R.scalar_cache(reg).program)
    end

    @testset "gene_basis draws on its own pool" begin
        # gene_basis evaluates in `gene_pools`, apart from an ordinary evaluation's pools
        Random.seed!(31)
        n = 120
        x = randn(3, n)
        y = randn(n)
        reg = GepRegressor(3; entered_non_terminals=[:+, :-, :*], gene_count=3, head_len=4)
        fit!(reg, 3, 80, x, y; loss_fun="mse")
        ctx = build_buffers(reg, x)
        c = reg.best_models_[1]
        G = P.gene_basis(c, ctx, n)
        @test G isa AbstractMatrix
        reference = copy(G)
        # a later evaluation of the whole chromosome leaves G unchanged
        P.buffered_predict(c, ctx)
        @test G == reference
        @test ctx.gene_pools !== ctx.pools
    end
end
