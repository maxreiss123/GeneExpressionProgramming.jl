#=
Surrogate screening of a search with two expressions and two objectives: the right-hand
sides of a predator-prey system, recovered from trajectories by a loss that integrates
every candidate.

    julia --project=. --threads=4 examples/Main_surrogate_multi_objective.jl

The data are trajectories of the Lotka-Volterra system

    dx/dt = x - x y        (prey)
    dy/dt = x y - y        (predator)

from six initial states. One chromosome carries both right-hand sides: `split_karva`
splits its four genes into two expressions, f for dx/dt and g for dy/dt, and the loss
sets one objective per expression, the error of the prey and of the predator
trajectories -- the setting of the Python tutorial
`tutorials/ext_eval_multi_objective_screened_gep.py`. The loss is a solver in the loop:
it integrates each equation with the other state read from the data (f with the simulated
x and the measured y, g with the measured x and the simulated y), so each objective
judges one expression, and every loss call costs a simulation.

A `SurrogateScreening` with `expressions=2` embeds an individual with one latent block per
expression, each expression evaluated on a few probe states. One Gaussian process per
objective predicts its loss, from the block of the expression the objective judges
(`objective_expressions=[1, 2]`), and a random Chebyshev scalarization of the two picks the
individuals worth a simulation (ParEGO). A prediction stays behind the best simulated value
of each objective, and it is provisional: an individual that carries one is screened again
every epoch, until the solver has scored it or it dies.

The result of a search with several objectives is a front, not one model. The script
prints the non-dominated members of the hall of fame, which the solver has all scored,
and where the solver calls went; a search stops once a scored individual solves both
equations. It runs the same search with and without the screening.

With the script's seed, the screened search solves both equations at epoch 54 after 1,196
solver calls, the one without screening at epoch 73 after 10,168. Over seeds 1 to 6, with
other probe states, the screened search solved five, after 800 to 2,050 solver calls; over
seeds 1 to 3 the search without screening solved two, after 8,850 and 10,170 calls (each
at most 100 epochs). Without the screening the best error of y in the population also got
worse at times, e.g. at epoch 60: the population survives by its mean error, and the
individual that had solved y alone fell behind. The screening keeps the scored holders of
the best values, as a prediction could otherwise push them out.
`surrogate.spearman_objectives` holds how well the processes ranked each batch they
picked, per objective.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using Printf
using Random
using Statistics

# the data: six trajectories, sampled every 0.1 up to t = 8
x0 = [0.5, 1.5, 2.0, 1.0, 0.8, 0.3]
y0 = [1.0, 0.5, 1.5, 2.0, 0.3, 0.8]
n = length(x0)
dt = 0.01
steps = 800
every = 10

# RK4 for the pair (x, y), sampled every `every` steps; `rhs(x, y, k)` also gets the time
# of the stage in half steps, k dt/2, at which the loss reads the measured states
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
        # a candidate that blows up is not worth integrating to the end
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
x_data, y_data = rk4(lotka_volterra, x0, y0, dt, steps, every)
# the measured states at every half step, from a run at half the step: column k + 1 holds
# the time k dt/2
x_half, y_half = rk4(lotka_volterra, x0, y0, dt / 2, 2steps, 1)
x_measured, y_measured = hcat(x0, x_half), hcat(y0, y_half)

epochs = 100
population_size = 200
solved = 1e-10         # both errors below this: the system is found

function search(; screened::Bool)
    Random.seed!(1)        # both searches start alike
    regressor = GepRegressor(2; entered_features=[:x, :y], entered_non_terminals=[:+, :-, :*],
        gene_connections=[:+, :-, :*], gene_count=4, head_len=4, number_of_objectives=2)

    # one buffer context per thread with two sets of states side by side, (x, measured y)
    # for f and (measured x, y) for g, which the solver overwrites at every stage: one
    # split_predict evaluates both expressions
    ctxs = thread_contexts(regressor.toolbox_, zeros(2, 2n))
    feature(i) = only(k for (k, nd) in regressor.toolbox_.nodes if nd isa InputSelector && nd.idx == i)
    x_sym, y_sym = feature(1), feature(2)
    calls = Threads.Atomic{Int}(0)

    # the loss runs on several threads at once: `local` keeps its variables its own, even
    # where the enclosing function assigns a variable of the same name
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

    surrogate = nothing
    if screened
        # the probe states: 36 of the (x, y) states the trajectories visit; the latent
        # vector of an individual is its f and its g on them
        states = vcat(vec(x_data)', vec(y_data)')
        probes = states[:, randperm(size(states, 2))[1:36]]
        surrogate = SurrogateScreening(regressor, probes;
            expressions=2,                 # one latent block per expression
            objective_expressions=[1, 2],  # the error of x judges f alone, that of y g
            individuals_per_epoch=0.15,    # 15 percent of the new individuals of an epoch
            seed=1)
    end
    scored(c) = isnothing(surrogate) || is_validated(surrogate, c)

    # every tenth epoch: the best scored errors in the population, and the solver calls
    # of the epoch by why the screening chose them
    function report(population, epoch, _)
        epoch % 10 == 0 || return
        known = [c.fitness for c in population if scored(c) && all(isfinite, c.fitness)]
        isempty(known) && return
        @printf("  epoch %3d | best error x %.2e, y %.2e | solver calls %5d", epoch,
            minimum(first, known), minimum(last, known), calls[])
        if !isnothing(surrogate)
            batch = surrogate.last_batch
            reasons = join(("$(count(e -> e.reason === r, batch)) $r"
                            for r in (:warmup, :acquisition, :explore, :validation)
                            if any(e -> e.reason === r, batch)), ", ")
            # a prediction is finite; an individual scored as a crash without a call is not
            @printf(" | this epoch %d (%s) | predicted %d of %d", length(batch), reasons,
                count(c -> !scored(c) && all(isfinite, c.fitness), population),
                length(population))
        end
        println()
    end

    # only what the solver has scored may stop the search, never a prediction
    reached = Ref(0)
    stop(population, epoch) = (reached[] = epoch;
        any(c -> scored(c) && maximum(c.fitness) < solved, population))

    time = @elapsed fit!(regressor, epochs, population_size, loss; surrogate=surrogate,
        hof=10, file_logger_callback=report, break_condition=stop)
    return regressor, calls[], reached[], time, surrogate
end

for screened in (false, true)
    println(screened ? "\nwith the surrogate screening:" : "\nwithout a surrogate:")
    regressor, calls, reached, time, surrogate = search(screened=screened)
    best = regressor.best_models_
    # the non-dominated members of the hall of fame, each with its errors once; an error
    # below `solved` is round-off, and counts as solved
    floored = [max.(m.fitness, solved) for m in best]
    front = unique(m -> m.fitness, best[calculate_fronts(floored)[1]])
    println("  front (error x, error y):")
    for m in front
        f, g = split_equations(m, 2)
        @printf("    %.2e  %.2e   dx/dt = %s   dy/dt = %s\n", m.fitness..., f, g)
    end
    println("  epochs        ", reached, " of ", epochs)
    println("  solver calls  ", calls)
    println("  time          ", round(time; digits=1), " s")
    if screened
        println("  predictions   ", surrogate.imputed_count)
        println("  all scored    ", all(m -> is_validated(surrogate, m), best))
    end
end
