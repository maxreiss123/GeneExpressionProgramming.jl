#=
Symbolic regression held to physical units by semantic backpropagation (SBP).

    julia --project=. --threads=4 examples/Main_physical_dimensions.jl

The data is Feynman III.21.20, the electric current density of a superconductor,
J = -ρ q A / m, with 1 % noise: ρ the charge density, q the charge, A the magnetic vector
potential and m the mass. Every feature carries its SI dimension, and the search is given
the target's. Up to half the initial population is seeded with library expressions of the
target dimension. Each generation, new individuals whose dimension misses the target are
repaired in place -- the requirement is pushed down the expression tree and the parts that
cannot meet it are rewritten with dimensionally consistent subexpressions -- and only
homogeneous individuals are scored.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using CSV
using DataFrames
using Random

Random.seed!(1)

epochs = 300
population_size = 1000

# 4 feature columns and the target
data = Matrix(CSV.read(joinpath(@__DIR__, "..", "paper", "srsd", "feynman-III.21.20\$0.01.txt"),
    DataFrame))
data = data[all.(x -> !any(isnan, x), eachrow(data)), :]
num_cols = size(data, 2)

x_train, y_train, x_test, y_test = train_test_split(data[:, 1:num_cols-1], data[:, num_cols];
    consider=4)

# SI exponents [kg, m, s, K, mol, A, cd] (the unit order used in OpenFOAM)
target_dim = Float16[0, -2, 0, 0, 0, 1, 0]          # J: A/m²

# the columns are rho_c_0, q, A_vec, m; features are named x1 ... xn in column order
feature_dims = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, -3, 1, 0, 0, 1, 0],            # ρ: A·s/m³
    :x2 => Float16[0, 0, 1, 0, 0, 1, 0],             # q: A·s
    :x3 => Float16[1, 1, -2, 0, 0, -1, 0],           # A: kg·m/(s²·A)
    :x4 => Float16[1, 0, 0, 0, 0, 0, 0],             # m: kg
)

# `rounds` bounds the library of dimensionally consistent subexpressions the repair draws
# on to rounds + 1 symbols; composing two of them under * or / extends its reach to
# expressions about twice that long
regressor = GepRegressor(num_cols - 1; considered_dimensions=feature_dims,
    max_permutations_lib=10000, rounds=5)

fit!(regressor, epochs, population_size, x_train', y_train; x_test=x_test', y_test=y_test,
    loss_fun="mse", target_dimension=target_dim)

pred = regressor(x_test')
best = regressor.best_models_[1]
println("best model   : ", best)
println("train mse    : ", best.fitness[1])
println("test r2      : ", get_loss_function("r2_score")(y_test, pred))
println("homogeneous  : ", best.dimension_homogene &&
                            is_dimensionally_homogeneous(best.expression_raw, target_dim,
                                regressor.token_dto_))
