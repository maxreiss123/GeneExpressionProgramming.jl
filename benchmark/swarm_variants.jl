#=
Particle swarms against Nelder-Mead on the calibration problems of
benchmark/screened_constants.jl, in one table.

Five methods, each with a budget of 150 loss calls from the same 8 starts per problem:

- Nelder-Mead: `ScreenedNelderMead(screen=false)`, the steps of `Optim.NelderMead()`
- screened NM: `ScreenedNelderMead()`, the trust region search on a Gaussian process
- full swarm: a particle swarm that scores every particle every iteration
- screened swarm: a particle swarm whose loss calls a Gaussian process picks, one per
  iteration, the position it ranks best by mean - kappa * deviation
- screened swarm + NM: `ScreenedNelderMead(swarm_box=1.0)`, the screened swarm until it
  stalls (2n + 4 calls without a gain of 0.1 %, at most half the budget), then the
  screened Nelder-Mead from its best point, on every point it scored

The three swarms are the same swarm: 10 particles in a box of one initial step around the
start (the box of `swarm_box=1.0`), the start and a Latin hypercube of the box as their
first positions, the constriction coefficients of Clerc and Kennedy, the global best, and
walls that stop a particle. The full and the screened swarm run to the end of the budget;
they differ only in which positions the loss scores, and the screened swarm and the last
method only in the end.

Each cell holds the median number of loss calls until the excess loss f - f* falls to
1 % of the excess at the start ("-": not within the budget), and how many starts got to
0.01 % within it.

    julia --project=. benchmark/swarm_variants.jl

swarm_variants_results.txt holds the output of a run.
=#

include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))
using .GeneExpressionProgramming
using .GeneExpressionProgramming.GepSurrogate: GaussianProcess, posterior, unstandardize
using LinearAlgebra
using Printf
using Random
using Statistics

const STARTS = 8
const BUDGET = 150

include(joinpath(@__DIR__, "calibration_problems.jl"))

const GS = GeneExpressionProgramming.GepSimplex

"""
    swarm(f, x0; screened, particles=10, kappa=1.0, seed=0) -> best value after every call

The swarm of `ScreenedNelderMead(swarm_box=1.0)` (`GepSimplex.swarm_search!`) run to the
end of the budget: screened, the loss scores one position per iteration, the one a
Gaussian process over every point scored ranks best; else it scores every particle.
"""
function swarm(f, x0; screened::Bool, particles::Int=10, kappa::Float64=1.0, seed::Int=0)
    e = GS.Evaluations(f, BUDGET)
    n = length(x0)
    lower, upper = GS.swarm_bounds(1.0, x0)
    width = upper .- lower
    unit(x) = (x .- lower) ./ width
    rng = MersenneTwister(seed)
    m = particles
    X = [copy(x0)]
    strata = [randperm(rng, m - 1) for _ in 1:n]
    for i in 1:m-1
        push!(X, lower .+ width .* [(strata[j][i] - rand(rng)) / (m - 1) for j in 1:n])
    end
    V = [(lower .+ width .* rand(rng, n) .- X[i]) ./ 2 for i in 1:m]
    P = deepcopy(X)
    Pf = fill(Inf, m)
    function score!(i)
        y = GS.evaluate!(e, X[i])
        y < Pf[i] && ((P[i], Pf[i]) = (copy(X[i]), y))
    end
    for i in 1:m
        GS.remaining(e) > 0 && score!(i)
    end
    while GS.remaining(e) > 0
        finite = findall(isfinite, e.Y)
        isempty(finite) && break
        g = P[argmin(Pf)]
        for i in 1:m
            r1, r2 = rand(rng, n), rand(rng, n)
            V[i] = GS.CHI .* (V[i] .+ GS.ACCELERATION .* r1 .* (P[i] .- X[i]) .+
                              GS.ACCELERATION .* r2 .* (g .- X[i]))
            X[i] = X[i] .+ V[i]
            outside = (X[i] .< lower) .| (X[i] .> upper)
            X[i] = clamp.(X[i], lower, upper)
            V[i][outside] .= 0.0
        end
        if screened
            worst = maximum(e.Y[finite])
            values = [isfinite(y) ? y : worst for y in e.Y]
            gp = GaussianProcess(reduce(hcat, unit.(e.X)), log10.(max.(values, 1e-30)); fit=true)
            mu, sd = unstandardize(gp, posterior(gp, reduce(hcat, unit.(X)))...)
            scored = unit.(e.X)
            order = sortperm(mu .- kappa .* sd)
            k = findfirst(i -> all(z -> maximum(abs.(z .- unit(X[i]))) >= 1e-9, scored), order)
            isnothing(k) && break
            score!(order[k])
        else
            for i in 1:m
                GS.remaining(e) > 0 && score!(i)
            end
        end
    end
    return e.best
end

const METHODS = [
    ("Nelder-Mead", (f, x0) -> simplex_search(f, x0;
        method=ScreenedNelderMead(screen=false, max_evaluations=BUDGET)).history),
    ("screened NM", (f, x0) -> simplex_search(f, x0;
        method=ScreenedNelderMead(max_evaluations=BUDGET)).history),
    ("full swarm", (f, x0) -> swarm(f, x0; screened=false)),
    ("screened swarm", (f, x0) -> swarm(f, x0; screened=true)),
    ("screened swarm + NM", (f, x0) -> simplex_search(f, x0;
        method=ScreenedNelderMead(max_evaluations=BUDGET, swarm_box=1.0)).history),
]

calls_to(excess, level) = something(findfirst(<=(level), excess), Inf)

println("# julia --project=. benchmark/swarm_variants.jl (Julia $(VERSION)), $STARTS starts, ",
    "$BUDGET loss calls: median calls to 1 % of the initial excess, starts that got to 0.01 %")
@printf("%-50s |", "problem")
for (label, _) in METHODS
    @printf(" %19s |", label)
end
println()
println("-"^(52 + 22 * length(METHODS)))
for (name, f, fstar, start) in PROBLEMS
    @printf("%-50s |", name)
    for (_, run) in METHODS
        rng = MersenneTwister(11)
        to_1, to_1e4 = Float64[], 0
        for _ in 1:STARTS
            x0 = start(rng)
            excess = (run(f, x0) .- fstar) ./ (f(x0) - fstar)
            push!(to_1, calls_to(excess, 1e-2))
            to_1e4 += excess[end] <= 1e-4
        end
        med = median(to_1)
        @printf(" %15s %d/%d |", isfinite(med) ? string(round(Int, med)) : "-", to_1e4, STARTS)
    end
    println()
    flush(stdout)
end
