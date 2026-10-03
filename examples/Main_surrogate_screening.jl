#=
Surrogate screening of an expensive loss: the right-hand side of an ODE, recovered from
trajectories by a loss that integrates every candidate.

    julia --project=. --threads=4 examples/Main_surrogate_screening.jl

The search looks for f in dx/dt = f(x), given trajectories of the logistic equation
f(x) = x - 0.5 x^2 from eight initial values. The loss is a solver in the loop: it
integrates dx/dt = f(x) with the candidate f (RK4, all eight trajectories at once) and
compares the result with the data, so every loss call costs a simulation.

A `SurrogateScreening` lets the solver run only for a few individuals per epoch. Every new
individual is embedded as the behaviour of its f on a few probe states, a Gaussian process
maps these latent vectors to the losses of the individuals simulated so far, and the
process picks the individuals worth a simulation; the others get its prediction. The
script runs the same search with and without the screening, from the same seed, and
reports the solver calls: both find f exactly, the screened search after 776 solver
calls, the other after 5,423.

The screening costs time of its own (fitting the process, predicting and ranking): here,
on one thread, 0.05 s per epoch against 0.01 s for the loop without it, so it saves time
where a loss call costs more than that over the individuals it spares, i.e. for a real
solver. The simulation here takes about 6 ms; the first search of a session includes the
compilation of either path.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using Random
using Statistics

# the data: logistic growth from eight initial values, sampled every 0.1 up to t = 8
f_true(x) = x .- 0.5 .* x .^ 2
x0 = collect(range(0.05, 3.5; length=8))
dt = 0.004
steps = 2000
every = 25

function rk4(f, x0, dt, steps, every)
    x = copy(x0)
    out = zeros(length(x0), steps ÷ every)
    for s in 1:steps
        k1 = f(x)
        k2 = f(x .+ 0.5dt .* k1)
        k3 = f(x .+ 0.5dt .* k2)
        k4 = f(x .+ dt .* k3)
        x .+= dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        all(isfinite, x) || return nothing
        s % every == 0 && (out[:, s÷every] .= x)
    end
    return out
end

data = rk4(f_true, x0, dt, steps, every)

epochs = 40
population_size = 200

function search(; screened::Bool)
    Random.seed!(1)        # both searches start alike
    regressor = GepRegressor(1; entered_features=[:x], entered_non_terminals=[:+, :-, :*],
        gene_count=2, head_len=4)

    # one buffer context per thread, whose input column (the state x) the solver
    # overwrites at every stage: the compiled program reads the column in place
    ctxs = thread_contexts(regressor.toolbox_, zeros(1, length(x0)))
    x_sym = only(k for (k, n) in regressor.toolbox_.nodes if n isa InputSelector)
    calls = Threads.Atomic{Int}(0)

    # the loss runs on several threads at once: `local` keeps its variables its own, even
    # where the enclosing function assigns a variable of the same name
    function loss(elem, validate::Bool)
        isnan(mean(elem.fitness)) || validate || return
        Threads.atomic_add!(calls, 1)
        local ctx = ctxs[Threads.threadid()]
        function rhs(x)
            ctx.nodes[x_sym] .= x
            local y = elem(ctx)
            return y isa AbstractVector ? copy(y) : fill(NaN, length(x))
        end
        local sim = rk4(rhs, x0, dt, steps, every)
        elem.fitness = isnothing(sim) ? (Inf,) : (mean(abs2, sim .- data),)
    end

    surrogate = nothing
    if screened
        # the probe states span the range the trajectories visit; the latent vector of an
        # individual is its f on them
        probes = reshape(collect(range(0.0, 3.5; length=24)), 1, :)
        surrogate = SurrogateScreening(regressor, probes;
            individuals_per_epoch=0.15,    # 15 percent of the new individuals of an epoch
            seed=1)
    end

    time = @elapsed fit!(regressor, epochs, population_size, loss; surrogate=surrogate)
    return regressor, calls[], time, surrogate
end

for screened in (false, true)
    regressor, calls, time, surrogate = search(screened=screened)
    best = regressor.best_models_[1]
    println(screened ? "\nwith the surrogate screening:" : "\nwithout a surrogate:")
    println("  best f(x)     ", best)
    println("  loss          ", best.fitness[1])
    println("  solver calls  ", calls)
    println("  time          ", round(time; digits=1), " s")
    if screened
        println("  predictions   ", surrogate.imputed_count)
        println("  rank corr.    ", round(mean(surrogate.spearman_log); digits=2),
            " (prediction vs loss, per screened batch)")
    end
end
