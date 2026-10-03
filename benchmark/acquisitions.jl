#=
The acquisitions of a screening with several objectives, on two searches whose loss runs a
solver:

- `ode`: the system of two ODEs of examples/Main_surrogate_multi_objective.jl, one
  expression and one objective per equation; a search stops once the solver has scored an
  individual that solves both (errors below 1e-10). Seeds 1 to 6, each with its own probe
  states; reported: the solver calls until solved ("-": not within 100 epochs).
- `closure`: the turbulence closure of examples/Main_fictive_cfd_in_the_loop.jl, with the
  fictive CFD solver in the loop, whose diverged runs get the error 1e3 per flow; 100
  epochs, without the constant search, with and without `failure_above=1e3`. Seeds 1 to
  4; reported: the hypervolume of the front of every individual the solver has scored, in
  log10 of the errors against the corner (1, 1), i.e. how far below a relative error of 1
  the front reaches in both flows (larger is better), after 25, 50 and 100 epochs, the
  best error per flow, and the solver calls, the diverged ones among them.
- `tuned`: the closure search as the example runs it, with the constant search (every 10
  epochs, 30 solver calls at most), with the default screening, with `failure_above=1e3`
  and with `failure_above=1e3` and `:qehvi`. Seeds 1 to 6, seed 1 being the example's;
  reported: the front of the hall of fame the search returns (its best mean error, its
  best error per flow, the hypervolume as above), and the solver calls, the diverged ones
  and those on the constants among them.

Acquisitions: `:lcb` (the default, a Chebyshev scalarization of the optimistic bounds per
epoch), `:ehvi` (expected hypervolume improvement, every pick joining the front at its
mean) and `:qehvi` (the joint expected hypervolume improvement of the batch).

    julia --project=. --threads=4 benchmark/acquisitions.jl [ode | closure | tuned]

acquisitions_results.txt holds the output of a run.
=#

include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))
using .GeneExpressionProgramming
using .GeneExpressionProgramming.GepSurrogate: hypervolume
using Printf
using Random
using Statistics

const ACQUISITIONS = (:lcb, :ehvi, :qehvi)
const which = isempty(ARGS) ? "all" : ARGS[1]

# ---------------------------------------------------------------------------------------
#  The system of two ODEs
# ---------------------------------------------------------------------------------------

module TwoOdes
using ..GeneExpressionProgramming
using Random
using Statistics

const x0 = [0.5, 1.5, 2.0, 1.0, 0.8, 0.3]
const y0 = [1.0, 0.5, 1.5, 2.0, 0.3, 0.8]
const n = length(x0)
const dt = 0.01
const steps = 800
const every = 10

function rk4(rhs, x0, y0, dt, steps, every)
    x, y = copy(x0), copy(y0)
    xs, ys = zeros(length(x0), steps ÷ every), zeros(length(y0), steps ÷ every)
    for s in 0:steps-1
        k1x, k1y = rhs(x, y, 2s)
        k2x, k2y = rhs(x .+ 0.5dt .* k1x, y .+ 0.5dt .* k1y, 2s + 1)
        k3x, k3y = rhs(x .+ 0.5dt .* k2x, y .+ 0.5dt .* k2y, 2s + 1)
        k4x, k4y = rhs(x .+ dt .* k3x, y .+ dt .* k3y, 2s + 2)
        x = x .+ dt / 6 .* (k1x .+ 2 .* k2x .+ 2 .* k3x .+ k4x)
        y = y .+ dt / 6 .* (k1y .+ 2 .* k2y .+ 2 .* k3y .+ k4y)
        (all(isfinite, x) && all(isfinite, y) && maximum(abs, x) < 1e6 && maximum(abs, y) < 1e6) ||
            return nothing
        if (s + 1) % every == 0
            xs[:, (s+1)÷every] .= x
            ys[:, (s+1)÷every] .= y
        end
    end
    return xs, ys
end

lotka_volterra(x, y, _) = (x .- x .* y, x .* y .- y)
const x_data, y_data = rk4(lotka_volterra, x0, y0, dt, steps, every)
const x_half, y_half = rk4(lotka_volterra, x0, y0, dt / 2, 2steps, 1)
const x_measured, y_measured = hcat(x0, x_half), hcat(y0, y_half)
const solved = 1e-10

"""
    search(seed, acquisition) -> (solved epoch or 0, solver calls)
"""
function search(seed::Int, acquisition::Symbol)
    Random.seed!(seed)
    regressor = GepRegressor(2; entered_features=[:x, :y], entered_non_terminals=[:+, :-, :*],
        gene_connections=[:+, :-, :*], gene_count=4, head_len=4, number_of_objectives=2)
    ctxs = thread_contexts(regressor.toolbox_, zeros(2, 2n))
    feature(i) = only(k for (k, nd) in regressor.toolbox_.nodes if nd isa InputSelector && nd.idx == i)
    x_sym, y_sym = feature(1), feature(2)
    calls = Threads.Atomic{Int}(0)
    function loss(elem, validate::Bool)
        isnan(mean(elem.fitness)) || validate || return
        Threads.atomic_add!(calls, 1)
        local ctx = ctxs[Threads.threadid()]
        local xs, ys = ctx.nodes[x_sym], ctx.nodes[y_sym]
        function rhs(x, y, k)
            xs[1:n] .= x
            ys[1:n] .= @view y_measured[:, k+1]
            xs[n+1:2n] .= @view x_measured[:, k+1]
            ys[n+1:2n] .= y
            local f, g = split_predict(elem, ctx, 2)
            (f isa AbstractVector && g isa AbstractVector) || return (fill(NaN, n), fill(NaN, n))
            return f[1:n], g[n+1:2n]
        end
        local sim = rk4(rhs, x0, y0, dt, steps, every)
        elem.fitness = isnothing(sim) ? (Inf, Inf) :
                       (mean(abs2, sim[1] .- x_data), mean(abs2, sim[2] .- y_data))
    end
    states = vcat(vec(x_data)', vec(y_data)')
    probes = states[:, randperm(size(states, 2))[1:36]]
    surrogate = SurrogateScreening(regressor, probes; expressions=2, objective_expressions=[1, 2],
        individuals_per_epoch=0.15, screen=GpScreen(acquisition=acquisition), seed=seed)
    reached = Ref(0)
    done = Ref(false)
    function stop(population, epoch)
        reached[] = epoch
        done[] = any(c -> is_validated(surrogate, c) && maximum(c.fitness) < solved, population)
        return done[]
    end
    fit!(regressor, 100, 200, loss; surrogate=surrogate, hof=10, break_condition=stop)
    return done[] ? reached[] : 0, calls[]
end
end

# ---------------------------------------------------------------------------------------
#  The turbulence closure, with the fictive CFD solver in the loop
# ---------------------------------------------------------------------------------------

module Closure
using ..GeneExpressionProgramming
using ..GeneExpressionProgramming.GepSurrogate: hypervolume, pareto_points
using Random
using Statistics

const NU = 1.5e-5
const H = 0.05
const U_TAU = 550 * NU / H
const N = 48
const Y = H .* (1 .- cos.(range(0, pi / 2; length=N))) .^ 1.6
const DIVERGED = 1e3
const FLOW = [1:N, N+1:2N]
const TAU = vcat(U_TAU^2 .* (1 .- Y ./ H), fill(U_TAU^2, N))

function solve!(S, closure!; iterations=300, relaxation=0.5, tolerance=1e-9)
    S .= TAU ./ NU
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
    return diverged .| running
end

velocity(S) = vcat(0.0, cumsum(0.5 .* (S[1:end-1] .+ S[2:end]) .* diff(Y)))
van_driest(p, y, S, ut) = (p[1] .* y .* (1 .- exp.(-y .* ut ./ (p[2] * NU)))) .^ 2 .* S .+ p[3] * NU
const Y2 = vcat(Y, Y)
const UT2 = fill(U_TAU, 2N)
const S_REF = let S = zeros(2N)
    solve!(S, () -> van_driest([0.41, 26.0, 0.0], Y2, S, UT2))
    S
end
const U_REF = [velocity(S_REF[FLOW[k]]) for k in 1:2]

flow_errors(S, diverged) = ntuple(k -> diverged[k] ? DIVERGED :
    mean(abs2, velocity(S[FLOW[k]]) .- U_REF[k]) / mean(abs2, U_REF[k]), 2)

# the volume the scored front dominates in log10 of the errors, below the corner (1, 1)
function front_volume(fitnesses)
    isempty(fitnesses) && return 0.0
    P = reduce(hcat, [log10.(max.(collect(f), 1e-12)) for f in fitnesses])
    return hypervolume(pareto_points(P), [0.0, 0.0])
end

"""
    search(seed, acquisition, failure_above; tuned=false) -> named tuple
"""
function search(seed::Int, acquisition::Symbol, failure_above; tuned::Bool=false)
    Random.seed!(seed)
    units = Dict{Symbol,Vector{Float16}}(
        :y => Float16[0, 1, 0, 0, 0, 0, 0], :S => Float16[0, 0, -1, 0, 0, 0, 0],
        :u_tau => Float16[0, 1, -1, 0, 0, 0, 0], :nu => Float16[0, 2, -1, 0, 0, 0, 0])
    regressor = GepRegressor(4; entered_features=[:y, :S, :u_tau, :nu],
        entered_non_terminals=[:+, :-, :*, :/, :exp], considered_dimensions=units,
        gene_count=2, head_len=6, number_of_objectives=2, rounds=4)
    columns = vcat(Y2', zeros(1, 2N), UT2', fill(NU, 1, 2N))
    ctxs = thread_contexts(regressor.toolbox_, columns)
    s_sym = only(k for (k, nd) in regressor.toolbox_.nodes if nd isa InputSelector && nd.idx == 2)
    calls = Threads.Atomic{Int}(0)
    diverged_calls = Threads.Atomic{Int}(0)
    # every fitness the solver returned
    seen = Tuple{Float64,Float64}[]
    seen_lock = ReentrantLock()
    function loss(elem, validate::Bool)
        isnan(mean(elem.fitness)) || validate || return
        Threads.atomic_add!(calls, 1)
        local ctx = ctxs[Threads.threadid()]
        local S = ctx.nodes[s_sym]
        local diverged = solve!(S, () -> try
            elem(ctx)
        catch
            nothing
        end)
        any(diverged) && Threads.atomic_add!(diverged_calls, 1)
        elem.fitness = flow_errors(S, diverged)
        lock(() -> push!(seen, elem.fitness), seen_lock)
    end
    states = vcat(Y2', S_REF', UT2', fill(NU, 1, 2N))
    probes = states[:, randperm(2N)[1:36]]
    surrogate = SurrogateScreening(regressor, probes; individuals_per_epoch=0.15,
        failure_above=failure_above, screen=GpScreen(acquisition=acquisition), seed=seed)
    volumes = Dict{Int,Float64}()
    report(population, epoch, _) = epoch in (25, 50) && (volumes[epoch] = front_volume(seen))
    tuning = tuned ? (constant_optimizer=ScreenedNelderMead(max_evaluations=30), optimization_epochs=10) : (;)
    fit!(regressor, 100, 400, loss; surrogate=surrogate, target_dimension=Float16[0, 2, -1, 0, 0, 0, 0],
        hof=10, file_logger_callback=report, tuning...)
    volumes[100] = front_volume(seen)
    # the front of the hall of fame, what the search returns
    best = regressor.best_models_
    front = [Tuple(m.fitness) for m in best[calculate_fronts([m.fitness for m in best])[1]]]
    return (volumes=volumes, calls=calls[], diverged=diverged_calls[],
        channel=minimum(first, seen), couette=minimum(last, seen),
        on_constants=calls[] - surrogate.evaluated_count, front=front,
        front_mean=minimum(mean, front), front_volume=front_volume(front))
end
end

# ---------------------------------------------------------------------------------------
#  The runs
# ---------------------------------------------------------------------------------------

println("# julia --project=. --threads=$(Threads.nthreads()) benchmark/acquisitions.jl (Julia $(VERSION))")

if which in ("all", "ode")
    seeds = 1:6
    println("\n== two ODEs: solver calls until both equations are solved (\"-\": not within 100 epochs) ==")
    @printf("%-8s |%s | %6s | %13s\n", "", join([@sprintf("%7s", "seed $s") for s in seeds]),
        "solved", "median calls")
    for acquisition in ACQUISITIONS
        cells = String[]
        solved_calls = Int[]
        for seed in seeds
            epoch, calls = TwoOdes.search(seed, acquisition)
            push!(cells, epoch > 0 ? @sprintf("%7d", calls) : @sprintf("%7s", "-"))
            epoch > 0 && push!(solved_calls, calls)
        end
        @printf("%-8s |%s | %3d of %d | %13s\n", acquisition, join(cells), length(solved_calls),
            length(seeds), isempty(solved_calls) ? "-" : string(round(Int, median(solved_calls))))
        flush(stdout)
    end
end

if which in ("all", "closure")
    seeds = 1:4
    println("\n== closure: hypervolume of the scored front in log10 errors below (1, 1) after 25, 50 and 100 epochs, ",
        "best errors, solver calls (diverged) ==")
    for failure_above in (nothing, 1e3), acquisition in ACQUISITIONS
        label = "$(acquisition)$(isnothing(failure_above) ? "" : ", failure_above=1e3")"
        rows = [Closure.search(seed, acquisition, failure_above) for seed in seeds]
        for (seed, r) in zip(seeds, rows)
            @printf("%-26s seed %d | %5.2f %5.2f %5.2f | channel %.2e Couette %.2e | %5d calls (%4d diverged)\n",
                label, seed, r.volumes[25], r.volumes[50], r.volumes[100], r.channel, r.couette,
                r.calls, r.diverged)
        end
        @printf("%-26s median | %5.2f %5.2f %5.2f | channel %.2e Couette %.2e | %5d calls (%4d diverged)\n\n",
            label, median(r.volumes[25] for r in rows), median(r.volumes[50] for r in rows),
            median(r.volumes[100] for r in rows), median(r.channel for r in rows),
            median(r.couette for r in rows), round(Int, median(r.calls for r in rows)),
            round(Int, median(r.diverged for r in rows)))
        flush(stdout)
    end
end

if which in ("all", "tuned")
    seeds = 1:6
    println("\n== closure with the constant search of the example: the front of the hall of fame (best mean error, ",
        "best errors, hypervolume), solver calls (diverged, on the constants) ==")
    for (acquisition, failure_above) in ((:lcb, nothing), (:lcb, 1e3), (:qehvi, 1e3))
        label = "$(acquisition)$(isnothing(failure_above) ? "" : ", failure_above=1e3")"
        rows = [Closure.search(seed, acquisition, failure_above; tuned=true) for seed in seeds]
        for (seed, r) in zip(seeds, rows)
            @printf("%-26s seed %d | mean %.2e | channel %.2e Couette %.2e | %5.2f | %5d calls (%4d diverged, %3d on the constants)\n",
                label, seed, r.front_mean, minimum(first, r.front), minimum(last, r.front),
                r.front_volume, r.calls, r.diverged, r.on_constants)
        end
        @printf("%-26s median | mean %.2e | channel %.2e Couette %.2e | %5.2f | %5d calls (%4d diverged, %3d on the constants)\n\n",
            label, median(r.front_mean for r in rows), median(minimum(first, r.front) for r in rows),
            median(minimum(last, r.front) for r in rows), median(r.front_volume for r in rows),
            round(Int, median(r.calls for r in rows)), round(Int, median(r.diverged for r in rows)),
            round(Int, median(r.on_constants for r in rows)))
        flush(stdout)
    end
end
