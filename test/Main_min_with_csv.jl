include("../src/VGeneExpressionProgramming.jl")
using .VGeneExpressionProgramming
using DynamicExpressions
using OrderedCollections
using BenchmarkTools
using CSV
using DataFrames
using Random
using Plots

Random.seed!(1)

#Example Call
#start with some GEP hyper-params:
epochs = 2500
population_size = 2000


# Data file, here is expected to be a csv, where the columns are in the order x1,x2...xk, y 
data = Matrix(CSV.read("paper/srsd/feynman-III.21.20\$0.txt", DataFrame))
data = data[all.(x -> !any(isnan, x), eachrow(data)), :]
num_cols = size(data, 2)


# Perform a simple train test split
x_train, y_train, x_test, y_test = train_test_split(data[:, 1:num_cols-1], data[:, num_cols]; consider=4)


#define the features, here the numbers of the first two cols - here we add the feature dims and a maximum of permutations per tree high
regressor = GepRegressor(num_cols)


fit!(regressor, epochs, population_size, x_train', y_train; x_test=x_test', y_test=y_test,
    loss_fun="mse")

pred = regressor(x_test')

@show regressor.best_models_[1].compiled_function
@show regressor.best_models_[1].fitness


#Making a nice plot - data vs
pred_vs_actual = scatter(vec(pred), vec(y_test),
    xlabel="Actual Values",
    ylabel="Predicted Values",
    label="Predictions ",
    title="Predictions vs Actual - Symbolic Regression");


plot!(pred_vs_actual, vec(y_test), vec(y_test),
    label="Prediction Comparison",
    color=:red)

#train loss vs validation loss
train_validation = plot(
    regressor.fitness_history_.train_loss,
    label="Training Loss",
    ylabel="Loss",
    xlabel="Epoch",
    linewidth=2
);

plot!(
    train_validation,
    regressor.fitness_history_.val_loss,
    label="Validation Loss",
    linewidth=2
)
