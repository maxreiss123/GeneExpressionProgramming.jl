#=
Nguyen benchmark: the collective gene-averaging operator on vs off.

The ten Nguyen problems (Uy et al., GPEM 2011): 20 uniform samples each (100 on [0, 1]^2
for the bivariate Nguyen-9/10), function set {+, -, *, /, sin, cos, exp, log}. 25 seeds per
arm, identical data per seed in both arms, population 200, 500 epochs. A problem counts as
solved when the best model's final training MSE is below 1e-10. The ON arm sets
`gene_averaging_prob = 1.0` and keeps the package defaults for `gene_averaging_rate` and
`gene_averaging_elite_frac` (0.3 and 0.3, chosen by nguyen_averaging_sweep.jl).

nguyen_averaging_results.txt also holds runs at population 100 with 100 and 250 epochs,
made with POP/EPOCHS set accordingly and the defaults of the time (rate 0.05, 3 elites).
All recorded runs predate the fix that made the constant optimiser work (it now tunes the
best model's constants every 100 epochs), so a rerun can differ.

    julia --project=. benchmark/nguyen_averaging.jl
=#

include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))
using .GeneExpressionProgramming
using Random
using Statistics
using Printf
const RW = GeneExpressionProgramming.RegressionWrapper

const SEEDS = 25
const POP = 200
const EPOCHS = 500
const OPS = [:+, :-, :*, :/, :sin, :cos, :exp, :log]

# name, target, number of variables, sample range, sample count
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
    rng = MersenneTwister(1000 + seed)          # same data in both arms
    lo, hi = range_
    x = lo .+ (hi - lo) .* rand(rng, n, nvars)
    y = [fn(view(x, i, :)) for i in 1:n]
    return x, y
end

function run_arm(prob, avg_prob)
    name, fn, nvars, range_, n = prob
    RW.GENE_COMMON_PROBS["gene_averaging_prob"] = avg_prob
    finals = Float64[]
    t0 = time()
    for seed in 1:SEEDS
        x, y = make_data(fn, nvars, range_, n, seed)
        Random.seed!(seed)
        reg = GepRegressor(nvars; entered_non_terminals=OPS)
        fit!(reg, EPOCHS, POP, x', y; loss_fun="mse")
        push!(finals, reg.best_models_[1].fitness[1])
    end
    solved = count(<(1e-10), finals)
    return solved, median(finals), (time() - t0) / SEEDS
end

function main()
    @printf("%-10s | %13s | %13s | %11s | %11s\n",
        "problem", "solved OFF", "solved ON", "med MSE OFF", "med MSE ON")
    println("-"^70)
    tot_off = 0
    tot_on = 0
    t_off = 0.0
    t_on = 0.0
    for prob in PROBLEMS
        s0, m0, dt0 = run_arm(prob, 0.0)
        s1, m1, dt1 = run_arm(prob, 1.0)
        tot_off += s0; tot_on += s1; t_off += dt0; t_on += dt1
        @printf("%-10s | %10d/25 | %10d/25 | %11.2e | %11.2e\n",
            prob[1], s0, s1, m0, m1)
        flush(stdout)
    end
    println("-"^70)
    @printf("%-10s | %10d/250 | %10d/250 |  mean %4.2fs |  mean %4.2fs per run\n",
        "TOTAL", tot_off, tot_on, t_off / length(PROBLEMS), t_on / length(PROBLEMS))
end

main()
