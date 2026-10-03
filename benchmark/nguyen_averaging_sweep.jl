#=
Parameter sweep for the collective gene-averaging operator on the Nguyen suite.

Two knobs: `gene_averaging_rate`, the per-position probability that an offspring takes
the elite consensus symbol, and `gene_averaging_elite_frac`, the size of the consensus
pool: the best `elite_frac x mating-pool size` individuals of the population, at least 3.
Problems, data and solve criterion as in nguyen_averaging.jl, but at population 100 and
250 epochs; 25 seeds, identical per-seed data in every cell, `gene_averaging_prob = 1.0`
throughout. Reported per cell: total exact solves, and the two groups the operator moved
most in the recorded runs, the polynomials N2-N4 (up) and the bivariate N9-N10 (down).

The recorded sweep (nguyen_averaging_results.txt) adds five cells outside the grid in
`main` (rate/elite_frac 0.3/0.3, 0.2/0.5, 0.3/0.5, 0.4/0.3, 0.5/0.3); the best of them,
0.3/0.3, became the package default. It predates the fix that made the constant
optimiser work, so a rerun can differ.

    julia --project=. benchmark/nguyen_averaging_sweep.jl
=#

include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))
using .GeneExpressionProgramming
using Random
using Statistics
using Printf
const RW = GeneExpressionProgramming.RegressionWrapper

const SEEDS = 25
const POP = 100
const EPOCHS = 250
const OPS = [:+, :-, :*, :/, :sin, :cos, :exp, :log]

const PROBLEMS = [
    ("Nguyen-1", x -> x[1]^3 + x[1]^2 + x[1], 1, (-1.0, 1.0), 20),
    ("Nguyen-2", x -> x[1]^4 + x[1]^3 + x[1]^2 + x[1], 1, (-1.0, 1.0), 20),
    ("Nguyen-3", x -> x[1]^5 + x[1]^4 + x[1]^3 + x[1]^2 + x[1], 1, (-1.0, 1.0), 20),
    ("Nguyen-4", x -> x[1]^6 + x[1]^5 + x[1]^4 + x[1]^3 + x[1]^2 + x[1], 1, (-1.0, 1.0), 20),
    ("Nguyen-5", x -> sin(x[1]^2) * cos(x[1]) - 1, 1, (-1.0, 1.0), 20),
    ("Nguyen-6", x -> sin(x[1]) + sin(x[1] + x[1]^2), 1, (-1.0, 1.0), 20),
    ("Nguyen-7", x -> log(x[1] + 1) + log(x[1]^2 + 1), 1, (0.0, 2.0), 20),
    ("Nguyen-8", x -> sqrt(x[1]), 1, (0.0, 4.0), 20),
    ("Nguyen-9", x -> sin(x[1]) + sin(x[2]^2), 2, (0.0, 1.0), 100),
    ("Nguyen-10", x -> 2 * sin(x[1]) * cos(x[2]), 2, (0.0, 1.0), 100),
]

function make_data(fn, nvars, range_, n, seed)
    rng = MersenneTwister(1000 + seed)
    lo, hi = range_
    x = lo .+ (hi - lo) .* rand(rng, n, nvars)
    y = [fn(view(x, i, :)) for i in 1:n]
    return x, y
end

function run_cell(rate, elite_frac)
    RW.GENE_COMMON_PROBS["gene_averaging_prob"] = 1.0
    RW.GENE_COMMON_PROBS["gene_averaging_rate"] = rate
    RW.GENE_COMMON_PROBS["gene_averaging_elite_frac"] = elite_frac
    per_problem = Int[]
    for prob in PROBLEMS
        name, fn, nvars, range_, n = prob
        solved = 0
        for seed in 1:SEEDS
            x, y = make_data(fn, nvars, range_, n, seed)
            Random.seed!(seed)
            reg = GepRegressor(nvars; entered_non_terminals=OPS)
            fit!(reg, EPOCHS, POP, x', y; loss_fun="mse")
            reg.best_models_[1].fitness[1] < 1e-10 && (solved += 1)
        end
        push!(per_problem, solved)
    end
    return per_problem
end

function main()
    rates = [0.02, 0.05, 0.1, 0.2]
    elite_fracs = [0.05, 0.15, 0.3]       # x mating pool (70 here): 3, 10, 21 elites

    @printf("%-6s %-6s | %6s | %10s | %10s | per-problem (N1..N10)\n",
        "rate", "elites", "total", "N2-N4", "N9-N10")
    println("-"^88)

    # baseline: run_cell sets the probability back to 1.0, but at rate 0.0 the operator
    # exchanges nothing, so it is off in effect
    RW.GENE_COMMON_PROBS["gene_averaging_prob"] = 0.0
    base = run_cell(0.0, 0.05)
    RW.GENE_COMMON_PROBS["gene_averaging_prob"] = 1.0
    @printf("%-6s %-6s | %3d/250 | %7d/75 | %7d/50 | %s\n",
        "off", "-", sum(base), sum(base[2:4]), sum(base[9:10]), join(base, " "))
    flush(stdout)

    for elite_frac in elite_fracs, rate in rates
        pp = run_cell(rate, elite_frac)
        @printf("%-6g %-6g | %3d/250 | %7d/75 | %7d/50 | %s\n",
            rate, elite_frac, sum(pp), sum(pp[2:4]), sum(pp[9:10]), join(pp, " "))
        flush(stdout)
    end
end

main()
