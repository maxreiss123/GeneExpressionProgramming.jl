#=
Symbolic regression on data from a CSV file whose last column is the target -- here the
Feynman equation III.21.20, J = -ρ q A / m, with 1 % noise.

    julia --project=. --threads=4 examples/Main_min_with_csv.jl

`Main_physical_dimensions.jl` fits the same data with the features' units and the target's
dimension enforced.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using CSV
using DataFrames
using Random

Random.seed!(1)

epochs = 300
population_size = 1000

data = Matrix(CSV.read(joinpath(@__DIR__, "..", "paper", "srsd", "feynman-III.21.20\$0.01.txt"),
    DataFrame))
data = data[all.(x -> !any(isnan, x), eachrow(data)), :]
num_cols = size(data, 2)

# a shuffled 90/10 split; `consider=4` keeps every 4th row of each part
x_train, y_train, x_test, y_test = train_test_split(data[:, 1:num_cols-1], data[:, num_cols];
    consider=4)

# every column except the last is a feature
regressor = GepRegressor(num_cols - 1)

fit!(regressor, epochs, population_size, x_train', y_train; x_test=x_test', y_test=y_test,
    loss_fun="mse")

pred = regressor(x_test')
best = regressor.best_models_[1]
println("best model : ", best)
println("train mse  : ", best.fitness[1])
println("test r2    : ", get_loss_function("r2_score")(y_test, pred))

if Base.find_package("Plots") !== nothing
    include(joinpath(@__DIR__, "plot_results.jl"))
    plot_results(regressor, pred, y_test)
end
