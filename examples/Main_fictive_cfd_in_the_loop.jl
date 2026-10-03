#=
A turbulence closure searched with a fictive CFD solver in the loop: the search is
screened, has two objectives, a loss of its own and physical units, and the loss is not
smooth, since a run that diverges returns a large error.

    julia --project=. --threads=4 examples/Main_fictive_cfd_in_the_loop.jl

The fictive CFD solver is not an actual CFD run but a stand-in for one that takes under a
millisecond: the 1D momentum balance of fully developed turbulent flow between two walls,
(nu + nu_t(y, S)) S = tau(y) with S = dU/dy, for a channel flow (tau = u_tau^2 (1 - y / h))
and a Couette flow (tau = u_tau^2) at Re_tau = 550, solved for S by a damped fixed-point
iteration that evaluates the closure nu_t at every iteration. A closure that makes
nu + nu_t negative or not finite, or keeps the iteration from converging, diverges, and
that flow gets the error 1e3. The reference profiles come from van Driest's mixing length,
(kappa y (1 - exp(-y+ / A)))^2 S, kappa = 0.41, A = 26.

1. The search for nu_t(y, S, u_tau, nu): the features carry their SI units and every
   scored model is held to m^2/s; one objective per flow, the relative error of its
   velocity profile; a `SurrogateScreening` scores 15 % of the new individuals of an
   epoch, with the settings for a loss that diverges (`failure_above` at the error of a
   diverged run, the batch acquisition `:qehvi`), and the constants of the best model are
   tuned against the fictive CFD solver every 10 epochs (`constant_optimizer`).
2. The calibration of van Driest's mixing length with a viscosity correction,
   nu_t = (k y D)^2 S + c nu, in bounds of which the part c < -1 diverges, by Nelder-Mead,
   the screened Nelder-Mead and the screened swarm (`swarm_box`), from the same starts.

With the script's seed, the same at any thread count, the search makes 4,078 fictive CFD
runs in about two minutes on 4 threads, 1,922 of which diverge and 120 of which tune
constants, and returns nu_t = 0.0874 y^2 S - 0.600 nu, Prandtl's mixing length
(kappa y)^2 S with kappa = 0.296 less a viscosity correction, homogeneous in m^2/s, with
relative errors of 2.9e-3 (channel) and 3.1e-3 (Couette); it prints with the four
constants it carries, and with u_tau / u_tau. Over seeds 1 to 6 the best mean error of the
two flows was 3.0e-3 to 3.8e-3 (benchmark/acquisitions.jl tuned); with the default
screening (no `failure_above`, `:lcb`) the search got below 4e-3 on 2 of the seeds and
ended at 1.5e-2 to 3.5e-2 on the others, with half the diverged runs. Every calibration
finds k = 0.41, A = 26, c = 0: plain Nelder-Mead reaches the smallest median error
(1.3e-7), and the swarm, which scores points across the bounds, solves into the diverging
part 13 times without harm. This loss is one valley, where the swarm brings nothing; it is
meant for a loss with several minima.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using Printf
using Random
using Statistics

# ---------------------------------------------------------------------------------------
#  The flows and the fictive CFD solver
# ---------------------------------------------------------------------------------------

const NU = 1.5e-5                       # kinematic viscosity of air, m^2/s
const H = 0.05                          # half the distance of the walls, m
const U_TAU = 550 * NU / H              # friction velocity at Re_tau = 550, m/s
const N = 48                            # grid points per flow, clustered at the wall
const Y = H .* (1 .- cos.(range(0, pi / 2; length=N))) .^ 1.6
const DIVERGED = 1e3                    # the error of a solve that diverged

# both flows side by side, as the columns the closure is evaluated on: the total shear
# stress (over the density) of the channel flow, then of the Couette flow
const FLOW = [1:N, N+1:2N]
const TAU = vcat(U_TAU^2 .* (1 .- Y ./ H), fill(U_TAU^2, N))

"""
    solve!(S, closure!) -> Vector{Bool}

The shear rate of both flows by a damped fixed-point iteration of (nu + nu_t) S = tau,
in place in `S`, with `closure!()` returning nu_t at the current `S` (or `nothing`).
Returns which flows diverged.
"""
function solve!(S, closure!; iterations=300, relaxation=0.5, tolerance=1e-9)
    S .= TAU ./ NU                      # the laminar profile
    running = trues(2)
    diverged = falses(2)
    for _ in 1:iterations
        nut = closure!()
        if !(nut isa AbstractVector)
            diverged .|= running
            break
        end
        for k in 1:2
            running[k] || continue
            r = FLOW[k]
            if !all(isfinite, view(nut, r)) || any(<=(0), NU .+ view(nut, r))
                running[k] = false
                diverged[k] = true
                continue
            end
            Sk = view(S, r)
            step = relaxation .* (view(TAU, r) ./ (NU .+ view(nut, r)) .- Sk)
            Sk .+= step
            maximum(abs, step) < tolerance * maximum(abs, Sk) && (running[k] = false)
        end
        any(running) || break
    end
    return diverged .| running           # not converged counts as diverged
end

velocity(S) = vcat(0.0, cumsum(0.5 .* (S[1:end-1] .+ S[2:end]) .* diff(Y)))

# the error of each flow: the relative mean squared error of its velocity profile
function flow_errors(S, diverged)
    return ntuple(k -> diverged[k] ? DIVERGED :
                       mean(abs2, velocity(S[FLOW[k]]) .- U_REF[k]) / mean(abs2, U_REF[k]), 2)
end

# the reference: van Driest's mixing length
van_driest(p, y, S, ut) = (p[1] .* y .* (1 .- exp.(-y .* ut ./ (p[2] * NU)))) .^ 2 .* S .+ p[3] * NU
const Y2 = vcat(Y, Y)
const UT2 = fill(U_TAU, 2N)
const S_REF = let S = zeros(2N)
    solve!(S, () -> van_driest([0.41, 26.0, 0.0], Y2, S, UT2))
    S
end
const U_REF = [velocity(S_REF[FLOW[k]]) for k in 1:2]

# ---------------------------------------------------------------------------------------
#  1. The search: screened, two objectives, held to the unit of nu_t
# ---------------------------------------------------------------------------------------

Random.seed!(1)

# SI exponents [kg, m, s, K, mol, A, cd], keyed by the names of the features
units = Dict{Symbol,Vector{Float16}}(
    :y => Float16[0, 1, 0, 0, 0, 0, 0],             # wall distance, m
    :S => Float16[0, 0, -1, 0, 0, 0, 0],            # shear rate, 1/s
    :u_tau => Float16[0, 1, -1, 0, 0, 0, 0],        # friction velocity, m/s
    :nu => Float16[0, 2, -1, 0, 0, 0, 0])           # viscosity, m^2/s
nut_unit = Float16[0, 2, -1, 0, 0, 0, 0]            # nu_t, m^2/s

regressor = GepRegressor(4; entered_features=[:y, :S, :u_tau, :nu],
    entered_non_terminals=[:+, :-, :*, :/, :exp], considered_dimensions=units,
    gene_count=2, head_len=6, number_of_objectives=2, rounds=4)

# one context per thread with both flows side by side, columns y, S, u_tau, nu; the solver
# overwrites the column of S at every iteration
columns = vcat(Y2', zeros(1, 2N), UT2', fill(NU, 1, 2N))
ctxs = thread_contexts(regressor.toolbox_, columns)
feature(i) = only(k for (k, nd) in regressor.toolbox_.nodes if nd isa InputSelector && nd.idx == i)
s_sym = feature(2)
calls = Threads.Atomic{Int}(0)
diverged_calls = Threads.Atomic{Int}(0)

# the loss runs on several threads at once: `local` keeps its variables its own
function loss(elem, validate::Bool)
    isnan(mean(elem.fitness)) || validate || return
    Threads.atomic_add!(calls, 1)
    local ctx = ctxs[Threads.threadid()]
    local S = ctx.nodes[s_sym]
    # elem(ctx) applies the constants the constant search tunes
    local diverged = solve!(S, () -> try
        elem(ctx)
    catch                               # e.g. a domain error: the solve diverged
        nothing
    end)
    any(diverged) && Threads.atomic_add!(diverged_calls, 1)
    elem.fitness = flow_errors(S, diverged)
end

# the probes of the screening: 36 states (y, S, u_tau, nu) the reference flows pass through
states = vcat(Y2', S_REF', UT2', fill(NU, 1, 2N))
probes = states[:, randperm(2N)[1:36]]
# a run that diverged gets the error 1e3: for the screening a failure, which teaches it
# where the solver diverges, not a value its processes fit; the batch acquisition weighs
# every candidate by its chance to converge
surrogate = SurrogateScreening(regressor, probes; individuals_per_epoch=0.15, seed=1,
    failure_above=DIVERGED, screen=GpScreen(acquisition=:qehvi))

function report(population, epoch, _)
    epoch % 20 == 0 || return
    known = [c.fitness for c in population if is_validated(surrogate, c) && maximum(c.fitness) < DIVERGED]
    isempty(known) && return
    @printf("  epoch %3d | best error channel %.2e, Couette %.2e | fictive CFD runs %5d, %d diverged\n",
        epoch, minimum(first, known), minimum(last, known), calls[], diverged_calls[])
end

println("1. the search for a closure nu_t(y, S, u_tau, nu):")
seconds = @elapsed fit!(regressor, 100, 400, loss; surrogate=surrogate,
    target_dimension=nut_unit, hof=10, file_logger_callback=report,
    constant_optimizer=ScreenedNelderMead(max_evaluations=30), optimization_epochs=10)

best = regressor.best_models_
front = unique(m -> m.fitness, best[calculate_fronts([m.fitness for m in best])[1]])
println("  the front (error of the channel flow, of the Couette flow), every model in m^2/s:")
for m in sort(front; by=m -> m.fitness[1])
    @printf("    %.2e  %.2e   nu_t = %s   (homogeneous: %s)\n", m.fitness..., m,
        is_dimensionally_homogeneous(m.expression_raw, nut_unit, regressor.token_dto_))
end
# the screening counts the runs it made; the others tuned constants
println("  fictive CFD runs  ", calls[], ", ", diverged_calls[], " of them diverged, ",
    calls[] - surrogate.evaluated_count, " of them on the constants")
println("  time              ", round(seconds; digits=1), " s")

# ---------------------------------------------------------------------------------------
#  2. The calibration of a closure form with three constants, part of whose bounds diverge
# ---------------------------------------------------------------------------------------

S_cal = zeros(2N)
function calibration_loss(p)
    diverged = solve!(S_cal, () -> van_driest(p, Y2, S_cal, UT2))
    return mean(flow_errors(S_cal, diverged))
end

lower, upper = [0.2, 5.0, -1.5], [0.6, 50.0, 1.0]
methods = [
    ("Nelder-Mead", ScreenedNelderMead(screen=false, max_evaluations=100)),
    ("screened", ScreenedNelderMead(max_evaluations=100)),
    ("screened+swarm", ScreenedNelderMead(max_evaluations=100, swarm_box=(lower, upper))),
]
rng = MersenneTwister(2)
starts = map(1:6) do _
    p = lower .+ (upper .- lower) .* rand(rng, 3)
    p[3] = -0.9 + 1.8 * rand(rng)                       # a correction the solver survives
    p
end
println("\n2. the calibration of (k, A, c) in nu_t = (k y D)^2 S + c nu, 100 fictive CFD runs from each of 6 starts")
println("   (the reference flows have k = 0.41, A = 26, c = 0)")
@printf("   %-14s | %12s | %10s | %8s | %s\n", "", "median error", "below 1e-4", "diverged",
    "best (k, A, c)")
for (label, method) in methods
    results = [simplex_search(calibration_loss, p0; method=method) for p0 in starts]
    best_run = results[argmin([r.minimum for r in results])]
    @printf("   %-14s | %12.1e | %6d of 6 | %8d | (%.3f, %4.1f, %6.3f)\n", label,
        median(r.minimum for r in results), count(r -> r.minimum < 1e-4, results),
        sum(r -> count(>=(DIVERGED / 2), r.values), results), best_run.minimizer...)
end
