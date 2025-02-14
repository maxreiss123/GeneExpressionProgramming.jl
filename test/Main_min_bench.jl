include("../src/GeneExpressionProgramming.jl")

using .GeneExpressionProgramming
using Random
using Plots
using BenchmarkTools

Random.seed!(1)

#Define the iterations for the algorithm and the population size
epochs = 100
population_size = 20

#Number of features which needs to be inserted
number_features = 2

x_data = randn(Float64, 10000, number_features)
y_data = @. x_data[:,1] * x_data[:,1] + x_data[:,1] * x_data[:,2] - 2 * x_data[:,2] * x_data[:,2] * rand()

#define the 
regressor = GepRegressor(number_features)
@btime fit!(regressor, epochs, population_size, x_data', y_data; loss_fun="mse")