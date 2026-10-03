#=
Tensor regression: recover a = 0.5·u1 + x2·u2 + 2·u3 from 3D vector and scalar columns.

    julia --project=. --threads=4 examples/Main_tensor_regression.jl

The tensor path takes one column per feature -- scalars and `Tensors.jl` vectors side by
side -- and a loss callback, which evaluates each chromosome's karva string in the
regressor's preallocated buffers with `predictT`.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using LinearAlgebra
using Random
using Statistics
using Tensors

Random.seed!(789)

n_samples = 300
x1 = fill(2.0, n_samples)                     # a constant scalar column
x2 = randn(n_samples)                         # a varying scalar column
u1 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]
u2 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]
u3 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]
a_true = [0.5 * u1[i] + x2[i] * u2[i] + 2.0 * u3[i] for i in 1:n_samples]

regressor = GepTensorRegressor(5;
    problem_dimension=3,                      # 3D tensors
    gene_count=3,                             # one gene per additive term
    head_len=4,
    entered_non_terminals=[:+, :-, :*],
    entered_terminal_nums=[0.5, 2.0],
    gene_connections=[:+, :-],
    feature_names=["x1", "x2", "U1", "U2", "U3"])

# per-thread buffers sized to the training data; predictT evaluates into them
allocate_buffers!(regressor, (x1, x2, u1, u2, u3))

# a large finite penalty: tournament selection skips non-finite fitness, so an Inf
# penalty would take the individual out of selection altogether
const PENALTY = 1e6

function tensor_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        try
            pred = predictT(regressor, elem.expression_raw)
            # a type-invalid chromosome -- a scalar where a vector belongs -- comes back as
            # something other than a vector of 3D tensors
            if pred isa AbstractVector && length(pred) == n_samples && eltype(pred) <: Tensor{1,3}
                l = mean(norm(pred[i] - a_true[i])^2 for i in 1:n_samples)
                elem.fitness = (isfinite(l) ? l : PENALTY,)
            else
                elem.fitness = (PENALTY,)
            end
        catch
            elem.fitness = (PENALTY,)
        end
    end
end

t = @elapsed fit!(regressor, 60, 600, tensor_loss)

best = regressor.best_models_[1]
println("\nbest expression : ", print_karva_strings(best))
println("mean sq. error  : ", best.fitness[1])
println("time            : ", round(t; digits=1), " s")

# prediction on new data: one column per feature, any number of samples
m = 50
new_cols = Any[fill(2.0, m), randn(m), [Tensor{1,3}(randn(3)) for _ in 1:m],
    [Tensor{1,3}(randn(3)) for _ in 1:m], [Tensor{1,3}(randn(3)) for _ in 1:m]]
truth = [0.5 * new_cols[3][i] + new_cols[2][i] * new_cols[4][i] + 2.0 * new_cols[5][i] for i in 1:m]
pred_new = predictT(regressor, best.expression_raw, new_cols)
println("max error, new : ", maximum(norm(pred_new[i] - truth[i]) for i in 1:m))
