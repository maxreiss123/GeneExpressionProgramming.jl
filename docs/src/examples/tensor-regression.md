# Tensor Regression

Tensor regression evolves expressions over vector and tensor data, for applications such as computational mechanics and fluid dynamics, where the relationships involve vectors and higher-order tensors. GeneExpressionProgramming.jl supports it either with fixed templates of several expressions, in the manner of M-GEP [1] -- see the [template loss](../api-reference.md#Template-loss) -- or with a free search over tensor operations, shown here.

## Tensor Operations in GeneExpressionProgramming.jl

`GepTensorRegressor` evaluates candidates with the same batched evaluator as the scalar path, over [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl) types, with buffers allocated once per thread and tensor type. Its operations include vector addition and subtraction, products with a scalar, dot and cross products, double contraction, element-wise products, traces, determinants, norms and the parts of a tensor; [Tensor Operations](../api-reference.md#Tensor-Operations) lists them.

## Tensor Regression Example

The example recovers a vector-valued relationship between velocity-like vectors and scalars.

```julia
using GeneExpressionProgramming
using Random
using Tensors
using LinearAlgebra
using Statistics

Random.seed!(789)

println("Target: a = 0.5*u1 + x2*u2 + 2*u3")
println("where u1, u2, u3 are 3D vectors and x2 is a scalar")

# ---------------------------------------------------------------- data --------
n_samples = 300

x1 = [2.0 for _ in 1:n_samples]   # a constant scalar column
x2 = randn(n_samples)             # a varying scalar column
u1 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]
u2 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]
u3 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]

a_true = [0.5 * u1[i] + x2[i] * u2[i] + 2.0 * u3[i] for i in 1:n_samples]

# One column per feature. Unlike the scalar path, which takes a matrix, the tensor path
# takes a column per feature because the columns have different element types -- scalars
# here, 3D vectors there. A tuple or a `Vector{Any}` both work.
inputs = (x1, x2, u1, u2, u3)

# ------------------------------------------------------------ regressor -------
regressor = GepTensorRegressor(5;
    problem_dimension=3,                       # 3D tensors
    gene_count=3,                              # one gene per additive term
    head_len=4,
    entered_non_terminals=[:+, :-, :*],
    entered_terminal_nums=[0.5, 2.0],          # constants the search may use
    gene_connections=[:+, :-],
    feature_names=["x1", "x2", "U1", "U2", "U3"])

# Evaluation buffers are allocated once, against the data the search will score on, and
# reused for every candidate. This must be called before `fit!`.
allocate_buffers!(regressor, inputs)

# --------------------------------------------------------------- loss ---------
# The tensor path takes a loss *callback* rather than a loss name: it is handed each
# chromosome and sets its fitness. `predictT` evaluates the chromosome's karva string
# through the preallocated buffers.
#
# A candidate that cannot be scored gets a large finite penalty, which keeps it eligible
# for the tournaments; `Inf` would work too -- tournament selection leaves non-finite
# fitness values out, and falls back to the whole population when none is finite.
const PENALTY = 1e6

@inline function tensor_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        try
            pred = predictT(regressor, elem.expression_raw)
            # a type-invalid chromosome -- a scalar where a vector belongs, say -- comes
            # back as something other than a vector of 3D tensors
            if pred isa AbstractVector && length(pred) == n_samples &&
               eltype(pred) <: Tensor{1,3}
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

# ------------------------------------------------------------ evolution -------
epochs = 60
population_size = 600

training_time = @elapsed fit!(regressor, epochs, population_size, tensor_loss)
println("Training completed in $(round(training_time, digits=2)) seconds")

# -------------------------------------------------------------- results ------
best = regressor.best_models_[1]
println("Best expression : ", print_karva_strings(best))
println("Best fitness    : ", best.fitness[1])

pred = predictT(regressor, best.expression_raw)
vector_errors = [norm(pred[i] - a_true[i]) for i in 1:n_samples]
println("Mean vector error: ", mean(vector_errors))
println("Max vector error : ", maximum(vector_errors))

total_variance = sum(norm(a_true[i] - mean(a_true))^2 for i in 1:n_samples)
residual_variance = sum(norm(pred[i] - a_true[i])^2 for i in 1:n_samples)
println("R² equivalent    : ", 1 - residual_variance / total_variance)
```

With this seed the search recovers the target exactly, as

```
(((U3 + (U2 * x2)) + U3) + (U1 * 0.5))
```

which is `2*U3 + x2*U2 + 0.5*U1` — the relationship the data was generated from, at a
mean squared error of about `1e-31`.

To predict on new data, pass one column per feature, in the same order and of any length: `predictT(regressor, best.expression_raw, Any[x1_new, x2_new, u1_new, u2_new, u3_new])`. In a loss, `predictT_scaled(regressor, elem, a_true)` returns the prediction with one least-squares coefficient per gene -- the tensor counterpart of `linear_scaling` -- and stores the coefficients in `elem.scaling_weights`; it returns `nothing` when no gene matches the target in length and element type, or the fit is not finite.

## Working with Other Tensor Orders

The same setup handles second-order tensors; only the columns, the operations and the expected element type in the loss change:

```julia
# Second-order tensor columns: stresses, strains, velocity gradients
sigma = [rand(Tensor{2,3}) for _ in 1:n_samples]
epsilon = [rand(Tensor{2,3}) for _ in 1:n_samples]

regressor2 = GepTensorRegressor(2;
    problem_dimension=3,
    entered_non_terminals=[:+, :-, :*, :dot, :tr, :dev],
    feature_names=["sigma", "epsilon"])
allocate_buffers!(regressor2, (sigma, epsilon))

# in the loss, test for eltype(pred) <: Tensor{2,3} (or <: Number for a scalar target)
```

## Physical Dimensions on the Tensor Path

With `considered_dimensions`, each feature dimension carries the tensor order in front of the SI exponents, keyed `:x1`, `:x2`, ... in feature order; constants are dimensionless. Only the functions with tensor unit rules can then be entered: `:+`, `:-`, `:*`, `:/`, `:inv`, `:dot`, `:crossp`, `:tr`, `:det`, `:dcontract`, `:lap`, `:hadamard`, `:sqrt`, `:norm`, `:log`, `:exp`, `:sin` and `:cos`.

## References
 
[1] Weatheritt, J., Sandberg, R. D. (2016)  A novel evolutionary algorithm applied to algebraic modifications of the RANS stress–strain relationship. Journal of Computational Physics, 325, 22-37

---

*Next: [Surrogate Screening](surrogate-screening.md)*
