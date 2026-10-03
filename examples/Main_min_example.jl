#=
Minimal symbolic regression: recover y = x1² + x1·x2 - 2·x2² from 100 noise-free samples.

    julia --project=. examples/Main_min_example.jl

Plots.jl is not a dependency of the package. When it is installed in the active or the
default environment, the script also plots the predictions and the loss history.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using Random

Random.seed!(1)

# generations and population size
epochs = 1000
population_size = 100

number_features = 2

x_data = randn(Float64, 100, number_features)
y_data = @. x_data[:, 1] * x_data[:, 1] + x_data[:, 1] * x_data[:, 2] - 2 * x_data[:, 2] * x_data[:, 2]

# the regressor takes the data with one row per feature, hence the transposes
regressor = GepRegressor(number_features)
fit!(regressor, epochs, population_size, x_data', y_data; loss_fun="mse")

pred = regressor(x_data')

best = regressor.best_models_[1]
println("best model : ", best)                   # the equation, as a string
println("train mse  : ", best.fitness[1])
println("max error  : ", maximum(abs.(pred .- y_data)))

if Base.find_package("Plots") !== nothing
    include(joinpath(@__DIR__, "plot_results.jl"))
    plot_results(regressor, pred, y_data)
end
