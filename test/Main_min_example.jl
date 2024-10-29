include("../src/VGeneExpressionProgramming.jl")

using .VGeneExpressionProgramming

epochs = 1000
population_size = 1000

number_features = 2

x_data = randn(Float64, number_features, 100)
y_data = @. x_data[1,:] * x_data[1,:] + x_data[1,:] * x_data[2,:] - 2 * x_data[2,:] * x_data[2,:]


regressor = GepRegressor(number_features)
fit!(regressor, epochs, population_size, x_data', y_data; loss_fun="mse")

@show regressor.best_models_[1].compiled_function
@show regressor.best_models_[1].fitness