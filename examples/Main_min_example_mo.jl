#=
Multi-objective symbolic regression (NSGA-II) of y = x1² + x1·x2 - 2·x2²: accuracy against
expression length.

    julia --project=. --threads=4 examples/Main_min_example_mo.jl

With more than one objective the loss is a callback that evaluates the chromosome itself
and stores a tuple of objectives in `elem.fitness`. It runs inside the threaded fitness
loop, so it evaluates in per-thread buffer contexts built once (`thread_contexts`) rather
than allocating one per call.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using Random
using Statistics

Random.seed!(1)

epochs = 200
population_size = 500
number_features = 2

x_data = randn(Float64, 100, number_features)
y_data = @. x_data[:, 1] * x_data[:, 1] + x_data[:, 1] * x_data[:, 2] - 2 * x_data[:, 2] * x_data[:, 2]

regressor = GepRegressor(number_features; number_of_objectives=2)

# one buffer context per thread slot, built once for the training data
ctxs = thread_contexts(regressor.toolbox_, x_data')
mse = get_loss_function("mse")

function loss_new(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        y_pred = elem(ctxs[Threads.threadid()])
        # a prediction that is not a finite vector scores Inf (typemax) on both objectives,
        # which NSGA-II ranks behind every tuple with fewer non-finite entries
        elem.fitness = y_pred isa AbstractVector && all(isfinite, y_pred) ?
                       (mse(y_data, y_pred), length(elem.expression_raw) * 0.01) :
                       (typemax(Float64), typemax(Float64))
    end
end

fit!(regressor, epochs, population_size, loss_new)

println("\nBest models (mse, length/100):")
for m in regressor.best_models_
    println("  ", m.fitness, "  ", m)
end

pred = regressor(x_data')
println("max error of the first: ", maximum(abs.(pred .- y_data)))
