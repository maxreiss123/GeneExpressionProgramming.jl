#=
The constants of a model against an expensive loss, here an ODE solve (RK4) per call:
Nelder-Mead, screened by a Gaussian process or not.

    julia --project=. --threads=4 examples/Main_screened_constants.jl

1. Inside a search: f in dx/dt = f(x) from trajectories of the logistic equation
   f(x) = 1.3 x - 0.54 x^2. The constants a model draws (0.5, 0 and a random one) do not
   make 1.3 and 0.54, so `fit!` tunes the constants of the best model against the solver
   every 5 epochs (`constant_optimizer`), with a `SurrogateScreening` for the search; the
   loss evaluates with `elem(ctx)`, which applies tuned constants.
2. On its own: the four coefficients of the Lotka-Volterra system calibrated with
   `simplex_search`, plain (`screen=false`) and screened, from the same four starts within
   25 % of the truth, 100 solver calls each.

With the script's seed the search returns 0.539888 (0.555192 - x) x + x, i.e.
1.29974 x - 0.539888 x^2, with an error of 9.0e-9 after 492 solver calls, 58 of them on
the constants. The screened calibration reaches 1 % of the error at the start within 12
to 45 solver calls, the plain one within 57 to 93.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using Printf
using Random
using Statistics

# RK4 for the state vector z, sampled every `every` steps; nothing for a run that blows up
function rk4(rhs, z0, dt, steps, every)
    z = copy(z0)
    out = zeros(length(z0), steps ÷ every)
    for s in 1:steps
        k1 = rhs(z)
        k2 = rhs(z .+ 0.5dt .* k1)
        k3 = rhs(z .+ 0.5dt .* k2)
        k4 = rhs(z .+ dt .* k3)
        z = z .+ dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        (all(isfinite, z) && maximum(abs, z) < 1e6) || return nothing
        s % every == 0 && (out[:, s÷every] .= z)
    end
    return out
end

# ---------------------------------------------------------------------------------------
#  1. Inside a search: the constants of the best model, tuned against the solver
# ---------------------------------------------------------------------------------------

x0 = collect(range(0.05, 3.5; length=8))
data = rk4(x -> 1.3 .* x .- 0.54 .* x .^ 2, x0, 0.004, 2000, 25)

Random.seed!(3)
regressor = GepRegressor(1; entered_features=[:x], entered_non_terminals=[:+, :-, :*],
    gene_count=2, head_len=4)
# one buffer context per thread, whose input column (the state x) the solver overwrites
ctxs = thread_contexts(regressor.toolbox_, zeros(1, length(x0)))
x_sym = only(k for (k, nd) in regressor.toolbox_.nodes if nd isa InputSelector)
calls = Threads.Atomic{Int}(0)
tuning_calls = Threads.Atomic{Int}(0)

# the loss runs on several threads at once: `local` keeps its variables its own
function loss(elem, validate::Bool)
    isnan(mean(elem.fitness)) || validate || return
    Threads.atomic_add!(calls, 1)
    # a model with optimised_constants is being tuned, or was
    isnothing(elem.optimised_constants) || Threads.atomic_add!(tuning_calls, 1)
    local ctx = ctxs[Threads.threadid()]
    function rhs(x)
        ctx.nodes[x_sym] .= x
        local y = elem(ctx)                  # with the tuned constants, if it has them
        return y isa AbstractVector ? copy(y) : fill(NaN, length(x))
    end
    local sim = rk4(rhs, x0, 0.004, 2000, 25)
    elem.fitness = isnothing(sim) ? (Inf,) : (mean(abs2, sim .- data),)
end

probes = reshape(collect(range(0.0, 3.5; length=24)), 1, :)
surrogate = SurrogateScreening(regressor, probes; individuals_per_epoch=0.15, seed=1)
seconds = @elapsed fit!(regressor, 40, 200, loss; surrogate=surrogate,
    constant_optimizer=ScreenedNelderMead(max_evaluations=30), optimization_epochs=5)

best = regressor.best_models_[1]
println("1. the search, the constants of its best model tuned every 5 epochs:")
println("  best f(x)     ", best)
@printf("  error         %.3e\n", best.fitness[1])
println("  constants     ", round.(something(best.optimised_constants, Float64[]); sigdigits=6))
println("  solver calls  ", calls[], ", ", tuning_calls[], " of them on the constants")
println("  time          ", round(seconds; digits=1), " s")

# ---------------------------------------------------------------------------------------
#  2. On its own: the coefficients of a model, calibrated against the solver
# ---------------------------------------------------------------------------------------

z0 = [0.5, 1.0, 1.5, 2.0, 0.8, 0.3, 1.0, 0.5, 1.5, 2.0, 0.3, 0.8]   # x of six states, then y
truth = [1.1, 0.9, 1.0, 1.2]
function predator_prey(p)
    return z -> begin
        x, y = z[1:6], z[7:12]
        vcat(p[1] .* x .- p[2] .* x .* y, p[4] .* x .* y .- p[3] .* y)
    end
end
trajectories = rk4(predator_prey(truth), z0, 0.01, 800, 10)
# the loss of a calibration: one simulation per call, 1e3 for one that blows up
function calibration_loss(p)
    sim = rk4(predator_prey(p), z0, 0.01, 800, 10)
    return isnothing(sim) ? 1e3 : mean(abs2, sim .- trajectories)
end

println("\n2. the 4 coefficients of the predator-prey model, 100 solver calls per start:")
println("  solver calls until the error is 1 % and 0.01 % of the error at the start, and the")
println("  share of it left after 20 and 40 calls")
@printf("  %-13s | %5s %6s | %8s %8s | %s\n", "start", "1 %", "0.01 %", "@20", "@40",
    "coefficients (true: $truth)")
rng = MersenneTwister(4)
for start in 1:4
    p0 = truth .* (0.75 .+ 0.5 .* rand(rng, 4))
    for screen in (false, true)
        result = simplex_search(calibration_loss, p0;
            method=ScreenedNelderMead(screen=screen, max_evaluations=100))
        left = result.history ./ result.values[1]
        calls_to(share) = something(findfirst(<=(share), left), "-")
        @printf("  %d %-11s | %5s %6s | %8.1e %8.1e | %s\n", start,
            screen ? "screened" : "Nelder-Mead", calls_to(1e-2), calls_to(1e-4), left[20],
            left[40], round.(result.minimizer; digits=4))
    end
end
