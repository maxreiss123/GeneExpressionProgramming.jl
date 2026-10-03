# Physical Dimensionality and Semantic Backpropagation

Given the physical dimensions of the features and of the target, GeneExpressionProgramming.jl holds the search to dimensionally homogeneous expressions: semantic backpropagation (SBP) repairs candidates towards the target dimension, and only homogeneous candidates are scored. [Core Concepts](../core-concepts.md#Physical-Dimensionality-and-Semantic-Backpropagation) describes the unit rules, the library and the repair.

## Understanding Physical Dimensionality

### Dimensional Analysis Fundamentals

A physical equation must be dimensionally homogeneous: all its terms have the same physical dimension. The base dimensions of the International System of Units (SI) are:

1. **Mass (M)**: kilogram [kg]
2. **Length (L)**: meter [m]  
3. **Time (T)**: second [s]
4. **Temperature (Θ)**: kelvin [K]
5. **Amount of Substance (N)**: mole [mol]
6. **Electric Current (I)**: ampere [A]
7. **Luminous Intensity (J)**: candela [cd]

Every physical quantity has a dimension composed of these, for example:
- Velocity: [L T⁻¹]
- Force: [M L T⁻²]
- Energy: [M L² T⁻²]
- Electric Charge: [I T]

### Dimensional Representation in GeneExpressionProgramming.jl

A dimension is a `Float16` vector of SI exponents, one component per base unit, in the order listed above -- [kg, m, s, K, mol, A, cd], the order OpenFOAM uses and the one the package's physical constants (`get_constant_dims`) are given in:

```julia
# Dimension vector: [M, L, T, Θ, N, I, J]
velocity_dim = Float16[0, 1, -1, 0, 0, 0, 0]    # [L T⁻¹]
force_dim = Float16[1, 1, -2, 0, 0, 0, 0]       # [M L T⁻²]
energy_dim = Float16[1, 2, -2, 0, 0, 0, 0]      # [M L² T⁻²]
charge_dim = Float16[0, 0, 1, 0, 0, 1, 0]       # [I T]
```

## Complete Physical Dimensionality Example

The example recovers the electric current density in superconductivity, J = -ρqA/m (Feynman Lectures III 21.20), with ρ the charge density, q the electric charge, A the magnetic vector potential and m the mass. A shorter version is `examples/Main_physical_dimensions.jl`.

```julia
using GeneExpressionProgramming
using Random
using CSV
using DataFrames
using Statistics
using Plots

# Set random seed for reproducibility
Random.seed!(42)

# Target: the current density J, in A m⁻²
target_dim = Float16[0, -2, 0, 0, 0, 1, 0]

# Feature dimensions, keyed by the feature names x1 ... x4
feature_dims = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, -3, 1, 0, 0, 1, 0],   # ρ (charge density) [A s m⁻³]
    :x2 => Float16[0, 0, 1, 0, 0, 1, 0],    # q (electric charge) [A s]
    :x3 => Float16[1, 1, -2, 0, 0, -1, 0],  # A (magnetic vector potential) [kg m s⁻² A⁻¹]
    :x4 => Float16[1, 0, 0, 0, 0, 0, 0],    # m (mass) [kg]
)

# Data generated from J = -ρ * q * A / m, shipped with the repository (run from the
# repository root); the columns are rho_c_0, q, A_vec, m and the target
data = Matrix(CSV.read("./paper/srsd/feynman-III.21.20\$0.txt", DataFrame))
num_cols = size(data, 2)
x_train, y_train, x_test, y_test = train_test_split(data[:, 1:num_cols-1], data[:, num_cols]; consider=4)

# Evolution parameters
epochs = 1000
population_size = 1000
num_features = num_cols - 1 

# Regressor with the feature dimensions; it builds the library the repair draws on
regressor = GepRegressor(
    num_features;
    considered_dimensions=feature_dims,
    max_permutations_lib=10000,  # New library expressions kept per round
    rounds=7                     # Library expressions have up to rounds + 1 symbols
)

# Fit with the target dimension
training_time = @elapsed fit!(regressor, epochs, population_size, x_train', y_train; 
                              x_test=x_test', y_test=y_test, 
                              loss_fun="mse", 
                              target_dimension=target_dim)
println("Training completed in $(round(training_time, digits=2)) seconds")

# The best model
best_model = regressor.best_models_[1]
println("Best evolved expression: ", best_model)
println("Fitness (MSE): ", best_model.fitness[1])
println("Dimensionally homogeneous: ",
        is_dimensionally_homogeneous(best_model.expression_raw, target_dim, regressor.token_dto_))

# Predictions and metrics
train_predictions = regressor(x_train')
test_predictions = regressor(x_test')

r2 = get_loss_function("r2_score")
for (name, y, p) in (("Training", y_train, train_predictions), ("Test", y_test, test_predictions))
    println("$name: MSE = $(round(get_loss_function("mse")(y, p), sigdigits=6)), ",
            "R² = $(round(r2(y, p), digits=6)), ",
            "max error = $(round(maximum(abs.(y .- p)), sigdigits=6))")
end

# Comparison with the true relationship
true_predictions_test = -x_test[:, 1] .* x_test[:, 2] .* x_test[:, 3] ./ x_test[:, 4]
println("Correlation with the true relationship (test): ",
        round(cor(true_predictions_test, test_predictions), digits=6))

# Prediction accuracy and residuals
all_y = vcat(y_train, y_test)
all_pred = vcat(train_predictions, test_predictions)
min_val, max_val = extrema(vcat(all_y, all_pred))

p1 = scatter(y_train, train_predictions,
             xlabel="Actual Current Density", ylabel="Predicted Current Density",
             title="Actual vs Predicted", alpha=0.6, label="Training Data", markersize=3)
scatter!(p1, y_test, test_predictions, alpha=0.6, color=:red, label="Test Data", markersize=3)
plot!(p1, [min_val, max_val], [min_val, max_val],
      color=:black, linestyle=:dash, label="Perfect Prediction")

p2 = scatter(all_pred, all_y .- all_pred,
             xlabel="Predicted Values", ylabel="Residuals",
             title="Residual Analysis", alpha=0.6, legend=false, markersize=3)
hline!(p2, [0], color=:black, linestyle=:dash)

final_plot = plot(p1, p2, layout=(2,1), size=(800, 700))
savefig(final_plot, "physical_dimensionality_analysis.png")
```

## Settings

- **Feature and constant dimensions**: `considered_dimensions` is keyed by the feature symbols (`:x1`, `:x2`, ... or the `entered_features`) and by the symbols of the constant terminals (`Symbol(0.5)`, ...); unlisted features and constants, and the random constants, are dimensionless. A physical constant can enter as a terminal with its dimension, e.g. `entered_terminal_nums=[Symbol(9.807)]` with `Symbol(9.807) => get_constant_dims("g")`.
- **Library**: built when the regressor is constructed, from the features, the non-zero constants and the functions. `rounds` bounds the length of its expressions (`rounds + 1` symbols) and `max_permutations_lib` the new expressions kept per round; larger values let the repair find more replacements, at the cost of build time and memory.
- **Repair** (keywords of `fit!`): `target_dimension` switches it on; `correction_epochs` (default 1) sets how often the repair runs, `correction_amount` (default 1.0) the most individuals repaired per correction epoch, as a fraction of the population, and `cycles` (default 10) the attempts per individual. `lib_seed_amount` (default 0.5) is the fraction of the initial population seeded with library expressions of the target dimension.
- **Linear scaling**: with `linear_scaling=true` the model is the weighted sum of its genes, which is homogeneous only if every gene carries the target dimension; the check and the repair therefore go gene by gene. A custom loss that scores the genes' least-squares combination (`predictT_scaled`) needs `gene_wise_dimension=true` in `fit!`.
- **Checking a model**: `is_dimensionally_homogeneous(model.expression_raw, target_dim, regressor.token_dto_)`; `dimensional_homogeneity_distance` gives the distance to the target instead. For a scaled model, `is_gene_wise_homogeneous(model.expression_raw, target_dim, regressor.token_dto_, gene_count)` checks every gene.

The check is a cheap forward pass; the repair looks its replacements up in an index built once per regressor. If the repair still dominates the run time, lower `correction_amount` or raise `correction_epochs`; individuals that are not repaired get the worst fitness instead of being scored.

## Dimensions in Other Domains

```julia
# Fluid dynamics: pressure drop in pipe flow
feature_dims_fluid = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, 0, 0, 0, 0, 0, 0],     # Friction Factor
    :x2 => Float16[1, -3, 0, 0, 0, 0, 0],    # ρ (density)  
    :x3 => Float16[0, 1, -1, 0, 0, 0, 0],    # v (velocity) 
    :x4 => Float16[0, 1, 0, 0, 0, 0, 0],     # L (length)  
    :x5 => Float16[0, 1, 0, 0, 0, 0, 0]      # D (diameter)
)
target_dim_fluid = Float16[1, -1, -2, 0, 0, 0, 0]    # ΔP (pressure drop) [kg m⁻¹ s⁻²]

# Heat transfer: heat flux q = h⋅ΔT
feature_dims_heat_flux = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[1, 0, -3, -1, 0, 0, 0],   # h (heat transfer coefficient) [W m⁻² K⁻¹]
    :x2 => Float16[0, 0, 0, 1, 0, 0, 0],     # ΔT (temperature difference) [K]
)
target_dim_heat_flux = Float16[1, 0, -3, 0, 0, 0, 0] # q [W m⁻²]

# Electromagnetism: wave propagation
feature_dims_em = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, 1, -1, 0, 0, 0, 0],    # c (speed of light) 
    :x2 => Float16[0, 0, -1, 0, 0, 0, 0],    # f (frequency)
    :x3 => Float16[0, 1, 0, 0, 0, 0, 0],     # λ (wavelength)
    :x4 => Float16[-1, -3, 4, 0, 0, 2, 0],   # ε₀ (permittivity)
    :x5 => Float16[1, 1, -2, 0, 0, -2, 0],   # μ₀ (permeability)
)
target_dim_em = Float16[0, 1, -1, 0, 0, 0, 0]        # speed [m s⁻¹]
```

Where the physics involves dimensionless groups, such as the Reynolds number, they can enter as derived, dimensionless features.

---

*Next: [Tensor Regression](tensor-regression.md)*
