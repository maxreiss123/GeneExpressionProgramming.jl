#=
The gene-averaging operator: each position of an offspring is replaced, with probability
`rate`, by the same position of a consensus of the elites (`one_hot_mean`: per position, a
sample from the `top_k` most frequent symbols there).
=#

using Test
using Random
using Statistics

@testset "gene averaging (collective exchange)" begin
    Random.seed!(41)
    reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=2, head_len=4)
    tb = reg.toolbox_
    E = GeneExpressionProgramming.GepEntities

    @testset "consensus semantics" begin
        rng = MersenneTwister(1)
        elites = E.generate_population(6, tb)

        # identical elites at rate 1.0: the consensus is the elite, so the offspring
        # becomes a copy of it
        clone_genes = copy(elites[1].genes)
        foreach(e -> e.genes .= clone_genes, elites)
        child = E.generate_chromosome(tb; rng=rng)
        gene_averaging!(child, elites, 1.0; rng=rng)
        @test child.genes == clone_genes

        # rate 0.0: untouched
        child2 = E.generate_chromosome(tb; rng=rng)
        snapshot = copy(child2.genes)
        gene_averaging!(child2, elites, 0.0; rng=rng)
        @test child2.genes == snapshot

        # diverse elites, top_k=1: every exchanged position carries the
        # positional mode, so each symbol appears at that position in some elite
        elites2 = E.generate_population(8, tb)
        child3 = E.generate_chromosome(tb; rng=rng)
        gene_averaging!(child3, elites2, 1.0; top_k=1, rng=rng)
        for i in eachindex(child3.genes)
            @test any(e.genes[i] == child3.genes[i] for e in elites2)
        end

        # deterministic under a seeded rng
        c_a = E.generate_chromosome(tb; rng=MersenneTwister(9))
        c_b = E.generate_chromosome(tb; rng=MersenneTwister(9))
        gene_averaging!(c_a, elites2, 0.5; rng=MersenneTwister(3))
        gene_averaging!(c_b, elites2, 0.5; rng=MersenneTwister(3))
        @test c_a.genes == c_b.genes
    end

    @testset "wired into evolution" begin
        # with the operator on, a plain regression on the target of Main_min_example.jl
        # still reaches a low loss
        Random.seed!(42)
        x = randn(Float64, 100, 2)
        y = @. x[:, 1] * x[:, 1] + x[:, 1] * x[:, 2] - 2 * x[:, 2] * x[:, 2]
        reg2 = GepRegressor(2)
        @test reg2.toolbox_.gep_probs["gene_averaging_prob"] > 0   # active by default
        fit!(reg2, 60, 300, x', y; loss_fun="mse")
        @test isfinite(reg2.best_models_[1].fitness[1])
        @test reg2.best_models_[1].fitness[1] < 1.0
    end
end
