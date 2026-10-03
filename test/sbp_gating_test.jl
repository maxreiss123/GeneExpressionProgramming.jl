#=
Semantic backpropagation (SBP) behind a loss gated on `elem.dimension_homogene`: the
read-only forward check, repairs whose writeback that check accepts, library seeding, a
search that keeps supplying homogeneous candidates to such a loss, and the demotion of
fitness copies behind distinct individuals.
=#

using Test
using Random
using Statistics

@testset "SBP gating contract" begin
    # velocity = distance / time: two features with dims, target m/s
    dims = Dict{Symbol,Vector{Float16}}(
        :x1 => Float16[0, 1, 0, 0, 0, 0, 0],    # m  ([kg, m, s, K, mol, A, cd])
        :x2 => Float16[0, 0, 1, 0, 0, 0, 0])    # s
    target = Float16[0, 1, -1, 0, 0, 0, 0]      # m/s

    Random.seed!(11)
    reg = GepRegressor(2;
        considered_dimensions=dims,
        entered_non_terminals=[:+, :-, :*, :/],
        gene_count=2, head_len=3,
        max_permutations_lib=2000, rounds=5)
    tb = reg.toolbox_
    dto = reg.token_dto_

    # the scalar constructor registers operators before features, so ids are looked
    # up rather than assumed: features by their dimension, operators by callback name
    dimdict = dto.tokenLib.physical_dimension_dict[]
    x1 = only(k for (k, v) in dimdict if v == dims[:x1] && !haskey(tb.callbacks, k))
    x2 = only(k for (k, v) in dimdict if v == dims[:x2] && !haskey(tb.callbacks, k))
    id_div = only(k for (k, v) in tb.callbacks if string(v) == "/")
    id_add = only(k for (k, v) in tb.callbacks if string(v) == "+")

    @testset "forward check" begin
        # x1 / x2 -> m/s : homogeneous against the target
        @test is_dimensionally_homogeneous(Int8[id_div, x1, x2], target, dto)
        # x1 + x2 -> inconsistent
        @test !is_dimensionally_homogeneous(Int8[id_add, x1, x2], target, dto)
        # x1 alone -> consistent expression, wrong dimension
        @test !is_dimensionally_homogeneous(Int8[x1], target, dto)
        # computing the distance, which the check uses, leaves the expression unchanged
        expr = Int8[id_add, x1, x2]
        snapshot = copy(expr)
        dimensional_homogeneity_distance(expr, target, dto)
        @test expr == snapshot
    end

    @testset "repair writeback is verified by the forward check" begin
        Random.seed!(12)
        pop = GeneExpressionProgramming.GepEntities.generate_population(300, tb)
        foreach(c -> GeneExpressionProgramming.GepEntities.compile_expression!(c; force_compile=true), pop)
        repaired = 0
        for c in pop
            is_dimensionally_homogeneous(c.expression_raw, target, dto) && continue
            _, ok = correct_genes!(c.genes, tb.gen_start_indices, c.expression_raw,
                target, dto; cycles=10)
            if ok
                GeneExpressionProgramming.GepEntities.compile_expression!(c; force_compile=true)
                # a claimed repair must hold up under the read-only check
                @test is_dimensionally_homogeneous(c.expression_raw, target, dto)
                repaired += 1
            end
        end
        # m/s is reachable, so some repairs must succeed
        @test repaired > 0
    end

    @testset "library seeding" begin
        # expressions sampled near the target are internally consistent (finite distance)
        Random.seed!(21)
        for _ in 1:20
            expr = sample_lib_expression(target, dto; max_len=7)
            @test expr isa Vector{Int8}
            @test dimensional_homogeneity_distance(expr, target, dto) < Inf16
        end

        # seeding half the population with exact library expressions adds homogeneous ones
        Random.seed!(22)
        seeder = GeneExpressionProgramming.RegressionWrapper.make_lib_seeder(
            tb, dto, target, 0.5)
        @test seeder !== nothing
        pop = GeneExpressionProgramming.GepEntities.generate_population(200, tb)
        foreach(c -> GeneExpressionProgramming.GepEntities.compile_expression!(c; force_compile=true), pop)
        before = count(c -> is_dimensionally_homogeneous(c.expression_raw, target, dto), pop)
        seeder(pop)
        after = count(c -> is_dimensionally_homogeneous(c.expression_raw, target, dto), pop)
        @test after > before
        @test after >= 20            # at least a fifth of the seeded half
        # no target -> no seeder
        @test GeneExpressionProgramming.RegressionWrapper.make_lib_seeder(
            tb, dto, nothing, 0.5) === nothing
    end

    @testset "gated loss keeps evolving" begin
        Random.seed!(13)
        n = 60
        x = vcat(abs.(randn(1, n)) .+ 1.0, abs.(randn(1, n)) .+ 1.0)
        y = vec(x[1, :] ./ x[2, :])

        reg2 = GepRegressor(2;
            considered_dimensions=dims,
            entered_non_terminals=[:+, :-, :*, :/],
            gene_count=2, head_len=3,
            max_permutations_lib=2000, rounds=5)
        ctxs = thread_contexts(reg2.toolbox_, x)
        n_evals = Threads.Atomic{Int}(0)
        n_violations = Threads.Atomic{Int}(0)
        function gated_loss(elem, validate::Bool)
            try
                if isnan(mean(elem.fitness)) && elem.dimension_homogene || validate
                    elem.dimension_homogene || Threads.atomic_add!(n_violations, 1)
                    pred = elem(ctxs[Threads.threadid()])
                    elem.fitness = pred isa AbstractVector ?
                                   (sqrt(get_loss_function("mse")(y, pred)),) :
                                   (typemax(Float64),)
                    Threads.atomic_add!(n_evals, 1)
                end
            catch
                elem.fitness = (typemax(Float64),)
            end
        end

        fit!(reg2, 20, 200, gated_loss; target_dimension=target, correction_amount=0.5)
        best = reg2.best_models_[1]
        @test n_evals[] > 200                    # evolution kept supplying candidates
        @test n_violations[] == 0                # nothing non-homogeneous was evaluated
        @test best.dimension_homogene
        @test isfinite(best.fitness[1])
        @test is_dimensionally_homogeneous(best.expression_raw, target, reg2.token_dto_)
        # copies of a fitness rank behind distinct models, so the three returned models
        # have distinct fitness
        @test allunique(m.fitness for m in reg2.best_models_)
    end

    @testset "clones of a better fitness rank behind distinct individuals" begin
        Random.seed!(31)
        pop = GeneExpressionProgramming.GepEntities.generate_population(6, tb)
        for (c, f) in zip(pop, (1.0, 1.0, 1.0, 2.0, 2.0, 3.0))
            c.fitness = (f,)
        end
        ids = objectid.(pop)
        GeneExpressionProgramming.GepRegression.demote_clones!(pop)
        @test [c.fitness[1] for c in pop] == [1.0, 2.0, 3.0, 1.0, 1.0, 2.0]
        # the first of each fitness keeps its place in the ranking, the copies keep
        # their order behind all distinct individuals
        @test objectid.(pop) == ids[[1, 4, 6, 2, 3, 5]]
        # nothing to do without copies
        distinct = pop[1:3]
        @test objectid.(GeneExpressionProgramming.GepRegression.demote_clones!(distinct)) ==
              ids[[1, 4, 6]]
    end
end
