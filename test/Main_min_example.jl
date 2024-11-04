include("../src/GeneExpressionProgramming.jl")

using .GeneExpressionProgramming
using Random
using Plots

Random.seed!(1)

#Define the iterations for the algorithm and the population size
epochs = 1000
population_size = 1000

#Number of features which needs to be inserted
number_features = 2

x_data = randn(Float64, 100, number_features)
y_data = @. x_data[:,1] * x_data[:,1] + x_data[:,1] * x_data[:,2] - 2 * x_data[:,1] * x_data[:,2]

#define the 
regressor = GepRegressor(number_features)
fit!(regressor, epochs, population_size, x_data', y_data; loss_fun="mse")

pred = regressor(x_data')

@show regressor.best_models_[1].compiled_function
@show regressor.best_models_[1].fitness


#Making a nice plot - data vs
pred_vs_actual = scatter(vec(pred), vec(y_data), 
xlabel="Actual Values", 
ylabel="Predicted Values",
label="Predictions ",
title="Predictions vs Actual - Symbolic Regression");


plot!(pred_vs_actual, vec(y_data), vec(y_data), 
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
