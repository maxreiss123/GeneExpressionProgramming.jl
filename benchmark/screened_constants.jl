#=
Screened Nelder-Mead: loss calls to calibrate the constants of a model.

Every problem is a loss of a few constants: calibrations of the coefficients of an ODE
(logistic growth, Lotka-Volterra with 2, 3 or all 4 coefficients free and the others
known, and two oscillators whose loss has several minima) against trajectories, where
every loss call solves the ODE (the stand-in for a CFD simulation), and, for reference,
Rosenbrock and rotated quadratics. Each runs from 8 starts (the same starts for every
method) with a budget of 150 loss calls, with Nelder-Mead on the loss
(`ScreenedNelderMead(screen=false)`, which takes the steps of `Optim.NelderMead()`), with
the screened search (`ScreenedNelderMead()`), and with the screened swarm in a box of one
initial step around the start before it (`ScreenedNelderMead(swarm_box=1.0)`).

Reported, as the median over the starts: the loss calls until the excess loss
f - f* falls to 1e-2, 1e-4 and 1e-6 of the excess at the start ("-": not within the
budget), the share of the initial excess left after 20, 40 and 80 calls, the seconds
per run, which for the screened searches is mostly the work on the Gaussian process, and
how many starts got to 1e-4 of the excess at the start within the budget. The data of the
noisy calibrations carry Gaussian noise of 0.02; f* is then the minimum a long run of
Nelder-Mead finds from the true coefficients. The oscillators, x'' = -k x - c x' (and
- a x^3 for the Duffing one), are observed up to t = 60 with little damping: their loss in
the stiffness k has minima about 12 % apart, and the starts lie within 30 % of the truth.

    julia --project=. benchmark/screened_constants.jl

screened_constants_results.txt holds the output of a run.
=#

include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))
using .GeneExpressionProgramming
using LinearAlgebra
using Printf
using Random
using Statistics

const STARTS = 8
const BUDGET = 150

include(joinpath(@__DIR__, "calibration_problems.jl"))

const METHODS = [
    ("Nelder-Mead", ScreenedNelderMead(screen=false, max_evaluations=BUDGET)),
    ("screened", ScreenedNelderMead(max_evaluations=BUDGET)),
    ("screened+swarm", ScreenedNelderMead(max_evaluations=BUDGET, swarm_box=1.0)),
]

# ---------------------------------------------------------------------------------------
#  The runs
# ---------------------------------------------------------------------------------------

calls_to(excess, level) = something(findfirst(<=(level), excess), Inf)
fmt_calls(v) = isfinite(v) ? @sprintf("%5d", round(Int, v)) : "    -"

println("# julia --project=. benchmark/screened_constants.jl (Julia $(VERSION)), ",
    "median over $STARTS starts, $BUDGET loss calls")
for (name, f, fstar, start) in PROBLEMS
    @printf("== %s (f* = %.4g) ==\n", name, fstar)
    @printf("%-14s | %5s %5s %5s | %8s %8s %8s | %6s | %5s\n", "method", "1e-2", "1e-4",
        "1e-6", "@20", "@40", "@80", "s/run", "1e-4")
    println("-"^82)
    for (label, method) in METHODS
        rng = MersenneTwister(11)
        rows = NTuple{7,Float64}[]
        seconds = 0.0
        for _ in 1:STARTS
            x0 = start(rng)
            initial = f(x0) - fstar
            seconds += @elapsed result = simplex_search(f, x0; method=method)
            # the share of the initial excess left after every call
            excess = (result.history .- fstar) ./ initial
            left(k) = excess[min(k, end)]
            push!(rows, (calls_to(excess, 1e-2), calls_to(excess, 1e-4),
                calls_to(excess, 1e-6), left(20), left(40), left(80), excess[end]))
        end
        med(k) = median(row[k] for row in rows)
        @printf("%-14s | %s %s %s | %8.1e %8.1e %8.1e | %6.3f | %3d/%d\n", label,
            fmt_calls(med(1)), fmt_calls(med(2)), fmt_calls(med(3)), med(4), med(5), med(6),
            seconds / STARTS, count(row -> row[7] <= 1e-4, rows), STARTS)
    end
    println()
    flush(stdout)
end
