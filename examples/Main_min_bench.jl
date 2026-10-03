#=
Timing of a small scalar search -- y = x1² + x1·x2 - 2·x2², 1000 training and 200 held-out
test samples -- run twice: the first `fit!` includes compilation, the second is the
steady-state cost.

    julia --project=. --threads=4 examples/Main_min_bench.jl
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using Random
using Printf

target(x) = @. x[:, 1] * x[:, 1] + x[:, 1] * x[:, 2] - 2 * x[:, 2] * x[:, 2]

epochs = 100
population_size = 100
number_features = 2

for run in 1:2
    Random.seed!(0)
    x_data = randn(Float64, 1000, number_features)
    x_data_test = randn(Float64, 200, number_features)

    regressor = GepRegressor(number_features)
    t = @elapsed fit!(regressor, epochs, population_size, x_data', target(x_data);
        loss_fun="mse", x_test=x_data_test', y_test=target(x_data_test))

    best = regressor.best_models_[1]
    @printf("\nrun %d (%s): %.2f s, %.2f ms per epoch on %d thread(s)\n", run,
        run == 1 ? "with compilation" : "steady state", t, 1e3 * t / epochs, Threads.nthreads())
    @printf("  train mse %.3e, test mse %.3e\n", best.fitness[1],
        get_loss_function("mse")(target(x_data_test), regressor(x_data_test')))
    println("  ", best)
end
