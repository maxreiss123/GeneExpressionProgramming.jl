include("../src/GeneExpressionProgramming.jl")

using .GeneExpressionProgramming
using Random
using Plots
using DynamicExpressions
using Statistics


Random.seed!(1)

#Define the iterations for the algorithm and the population size
epochs = 1000
population_size = 1000

#Number of features which needs to be inserted
number_features = 2

x_data = randn(Float64, 100, number_features)
y_data = @. x_data[:,1] * x_data[:,1] + x_data[:,1] * x_data[:,2] - 2 * x_data[:,2] * x_data[:,2]

#define the 
regressor = GepRegressor(number_features; number_of_objectives=2)

@inline function loss_new(elem,validate::Bool)
    try
        if isnan(mean(elem.fitness)) || validate
            y_pred = elem.compiled_function(x_data', regressor.operators_)
            elem.fitness = (get_loss_function("mse")(y_data, y_pred), length(elem.expression_raw)*0.01)
        end
    catch e
        elem.fitness = (typemax(Float64),typemax(Float64))
    end
end

fit!(regressor, epochs, population_size, loss_new)


pred = regressor(x_data')

@show regressor.best_models_[1].compiled_function
@show regressor.best_models_[1].fitness