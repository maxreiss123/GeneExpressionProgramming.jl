#=
Surrogate screening: loss calls against quality.

A custom loss (the MSE of the model on 100 samples) counts its calls, and the same search
runs with every individual scored, with a `SurrogateScreening`, and without screening at a
number of loss calls like the screened one's. 10 seeds per arm, identical data per seed in
every arm; population 200 and 40 epochs unless an arm says otherwise; 3 genes, head length
6. The screening embeds on 36 of the samples. Reported: the median and first quartile of
the MSE of the returned model (every returned model is scored, never predicted), the
seeds that reached an MSE below 1e-10, and the mean number of loss calls.

    julia --project=. --threads=4 benchmark/surrogate_screening.jl

surrogate_screening_results.txt holds the output of a run.
=#

include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))
using .GeneExpressionProgramming
using Random
using Statistics
using Printf

const SEEDS = 1:10

# name, target, number of variables, sample range, function set
const PROBLEMS = [
    ("x1^2+x1*x2-2*x2^2", x -> x[1, :] .^ 2 .+ x[1, :] .* x[2, :] .- 2 .* x[2, :] .^ 2, 2,
        (-1.0, 1.0), [:+, :-, :*, :/]),
    ("Nguyen-4", x -> x[1, :] .^ 6 .+ x[1, :] .^ 5 .+ x[1, :] .^ 4 .+ x[1, :] .^ 3 .+
                      x[1, :] .^ 2 .+ x[1, :], 1, (-1.0, 1.0), [:+, :-, :*, :/]),
    ("Nguyen-7", x -> log.(x[1, :] .+ 1) .+ log.(x[1, :] .^ 2 .+ 1), 1, (0.0, 2.0),
        [:+, :-, :*, :/, :log, :exp]),
]

# name, epochs, population size, individuals per screened epoch (nothing: no screening)
const ARMS = [
    ("every individual scored", 40, 200, nothing),
    ("screened, 15 % of the new", 40, 200, 0.15),
    ("screened, 10 per epoch", 40, 200, 10),
    ("no screening, pop 30, 20 epochs", 20, 30, nothing),
    ("no screening, pop 60, 14 epochs", 14, 60, nothing),
]

function run_one(problem, seed, epochs, population, budget)
    _, target, nvar, (lo, hi), ops = problem
    Random.seed!(seed)
    rng = MersenneTwister(seed)
    x = lo .+ (hi - lo) .* rand(rng, nvar, 100)
    y = target(x)
    regressor = GepRegressor(nvar; entered_non_terminals=ops, gene_count=3, head_len=6)
    ctxs = thread_contexts(regressor.toolbox_, x)
    mse = get_loss_function("mse")
    calls = Threads.Atomic{Int}(0)
    # the loss runs on several threads at once, so its variables must be its own: `local`
    # keeps a name the enclosing function also uses from being shared between the calls
    function loss(elem, validate::Bool)
        isnan(mean(elem.fitness)) || validate || return
        Threads.atomic_add!(calls, 1)
        local pred = try
            elem(ctxs[Threads.threadid()])
        catch                              # e.g. the log of a negative number
            nothing
        end
        elem.fitness = pred isa AbstractVector && all(isfinite, pred) ? (mse(y, pred),) : (Inf,)
    end
    surrogate = isnothing(budget) ? nothing :
                SurrogateScreening(regressor, x[:, randperm(rng, 100)[1:36]];
                    individuals_per_epoch=budget, seed=seed)
    redirect_stderr(devnull) do
        fit!(regressor, epochs, population, loss; surrogate=surrogate)
    end
    final = regressor.best_models_[1](x)
    return (mse=final isa AbstractVector ? mse(y, final) : Inf, calls=calls[])
end

for problem in PROBLEMS
    println("== ", problem[1], ", ", length(SEEDS), " seeds ==")
    @printf("%-33s | %9s | %9s | %6s | %6s\n", "arm", "med MSE", "q25 MSE", "solved", "calls")
    println("-"^75)
    for (name, epochs, population, budget) in ARMS
        runs = [run_one(problem, seed, epochs, population, budget) for seed in SEEDS]
        errors = [r.mse for r in runs]
        @printf("%-33s | %9.3e | %9.3e | %3d/%-2d | %6.0f\n", name, median(errors),
            quantile(errors, 0.25), count(<(1e-10), errors), length(errors),
            mean(r.calls for r in runs))
    end
    println()
end
