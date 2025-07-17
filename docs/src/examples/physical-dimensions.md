# Physical Dimensionality and Semantic Backpropagation

One of the most innovative features of GeneExpressionProgramming.jl is its support for physical dimensionality constraints through semantic backpropagation. This feature ensures that evolved expressions respect physical units and dimensional homogeneity, making it particularly valuable for scientific and engineering applications where physical laws must be respected.

## Understanding Physical Dimensionality

### Dimensional Analysis Fundamentals

Physical dimensionality is based on the principle that any valid physical equation must be dimensionally homogeneous - all terms must have the same physical dimensions. The fundamental dimensions in the International System of Units (SI) are:

1. **Mass (M)**: kilogram [kg]
2. **Length (L)**: meter [m]  
3. **Time (T)**: second [s]
4. **Temperature (K)**: kelvin [K]
5. **Electric Current (I)**: ampere [A]
6. **Amount of Substance (N)**: mole [mol]
7. **Luminous Intensity (J)**: candela [cd]

Any physical quantity can be expressed as a combination of these fundamental dimensions. For example:
- Velocity: [L T⁻¹]
- Force: [M L T⁻²]
- Energy: [M L² T⁻²]
- Electric Charge: [A T]

### Dimensional Representation in GeneExpressionProgramming.jl

Dimensions are represented as vectors where each component corresponds to a fundamental unit:

```julia
# Dimension vector: [M, L, T, K, I, N, J]
velocity_dim = [0, 1, -1, 0, 0, 0, 0]    # [L T⁻¹]
force_dim = [1, 1, -2, 0, 0, 0, 0]       # [M L T⁻²]
energy_dim = [1, 2, -2, 0, 0, 0, 0]      # [M L² T⁻²]
charge_dim = [0, 0, 0, 1, 1, 0, 0]       # [A T]
```

## Complete Physical Dimensionality Example

This example demonstrates how to use physical dimensionality constraints to discover the relationship for electric current density in superconductivity, based on the Feynman Lectures equation J = -ρqmA (Feynman III 21.20).

```julia
using GeneExpressionProgramming
using Random
using CSV
using DataFrames
using Statistics
using Plots

# Set random seed for reproducibility
Random.seed!(42)

println("=== Physical Dimensionality Example ===")
println("Discovering: J = -ρqA/m (Feynman III 21.20)")
println("J: electric current density, ρ: charge density, q: electric charge")
println("m: mass, A: magnetic vector potential")
println()

# Define target dimension for electric current density J
# J has dimensions of [A m⁻²] = [0, -2, 0, 0,1, 0, 0]
target_dim = Float16[0, -2, 0, 0, 1, 0, 0]  # Current density: A/m²

println("Target dimension (J): [M, L, T, K, I, N, J] = $target_dim")
println("Physical meaning: Ampere per square meter [A m⁻²]")
println()

# Define dimensions for input features
# Based on the Feynman equation variables
feature_dims = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, -3, 1, 0, 1, 0, 0],   # ρ (charge density)
    :x2 => Float16[0, 0, 1, 0, 1, 0, 0],    # q (electric charge)
    :x3 => Float16[1, 1, -2, 0, -1, 0, 0],  # A (magnetic vector potential)
    :x4 => Float16[1, 0, 0, 0, 0, 0, 0],    # m (mass)
)

println("Feature dimensions:")
for (var, dim) in feature_dims
    println("  $var: $dim")
end
println()

# Physical interpretation of features
interpretations = Dict(
    :x1 => "ρ (charge density) [A s m⁻³]",
    :x2 => "q (electric charge) [A s]", 
    :x3 => "A (magnetic vector potential) [kg m s⁻² A⁻¹]",
    :x4 => "m (mass) [kg]"
)

println("Physical interpretation:")
for (var, interp) in interpretations
    println("  $var: $interp")
end
println()

# Generate synthetic data based on the known relationship
# J = -ρ * q * A / m 
n_samples = 500

println("Generating synthetic data...")

# J = -ρ * q * A / m 
n_samples = 5000

# Load the data from the file =>  rho_c_0,q,A_vec,m,target
data = Matrix(CSV.read("./paper/srsd/feynman-III.21.20\$0.txt", DataFrame))
num_cols = size(data, 2)
x_train, y_train, x_test, y_test = train_test_split(data[:, 1:num_cols-1], data[:, num_cols]; consider=4)

println("Training samples: $(length(y_train))")
println("Test samples: $(length(y_test))")
println()

# Evolution parameters
epochs = 1000
population_size = 1000
num_features = num_cols - 1 

# Create regressor with dimensional constraints
regressor = GepRegressor(
    num_features;
    considered_dimensions=feature_dims,
    max_permutations_lib=10000,  # Increase for more complex expressions
    rounds=7                     # Tree depth for dimensional checking
)

println("Regressor configuration:")
println("  Features: $num_features")
println("  Dimensional constraints: enabled")
println("  Max permutations: 10000")
println("  Tree depth (rounds): 7")
println()

println("Starting dimensionally-constrained evolution...")
start_time = time()

# Fit with dimensional constraints
fit!(regressor, epochs, population_size, x_train', y_train; 
     x_test=x_test', y_test=y_test, 
     loss_fun="mse", 
     target_dimension=target_dim)

training_time = time() - start_time
println("Training completed in $(round(training_time, digits=2)) seconds")
println()

# Analyze results
println("=== Results Analysis ===")

# Get the best model
best_model = regressor.best_models_[1]
best_expression = best_model.compiled_function
best_fitness = best_model.fitness

println("Best evolved expression:")
println("  $best_expression")
println("  Fitness (MSE): $best_fitness")
println()

# Make predictions
train_predictions = regressor(x_train')
test_predictions = regressor(x_test')

# Calculate comprehensive metrics
function calculate_detailed_metrics(y_true, y_pred, dataset_name)
    mse = mean((y_true .- y_pred).^2)
    mae = mean(abs.(y_true .- y_pred))
    rmse = sqrt(mse)
    
    # R² score
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r2 = 1 - ss_res / ss_tot
    
    # Mean Absolute Percentage Error
    mape = mean(abs.((y_true .- y_pred) ./ y_true)) * 100
    
    # Maximum error
    max_error = maximum(abs.(y_true .- y_pred))
    
    println("$dataset_name Performance:")
    println("  MSE:        $(round(mse, sigdigits=6))")
    println("  MAE:        $(round(mae, sigdigits=6))")
    println("  RMSE:       $(round(rmse, sigdigits=6))")
    println("  R²:         $(round(r2, digits=6))")
    println("  MAPE:       $(round(mape, digits=2))%")
    println("  Max Error:  $(round(max_error, sigdigits=6))")
    println()
    
    return (mse=mse, mae=mae, rmse=rmse, r2=r2, mape=mape, max_error=max_error)
end

train_metrics = calculate_detailed_metrics(y_train, train_predictions, "Training")
test_metrics = calculate_detailed_metrics(y_test, test_predictions, "Test")

# Visualization
println("Creating visualizations...")

# 1. Prediction accuracy plot
p1 = scatter(y_train, train_predictions,
             xlabel="Actual Current Density",
             ylabel="Predicted Current Density", 
             title="Training Set: Actual vs Predicted",
             alpha=0.6,
             color=:blue,
             label="Training Data",
             markersize=3)

scatter!(p1, y_test, test_predictions,
         alpha=0.6,
         color=:red,
         label="Test Data",
         markersize=3)

# Perfect prediction line
all_y = vcat(y_train, y_test)
all_pred = vcat(train_predictions, test_predictions)
min_val = min(minimum(all_y), minimum(all_pred))
max_val = max(maximum(all_y), maximum(all_pred))

plot!(p1, [min_val, max_val], [min_val, max_val],
      color=:black, linestyle=:dash, linewidth=2, label="Perfect Prediction")

# 2. Residual analysis
train_residuals = y_train .- train_predictions
test_residuals = y_test .- test_predictions

p2 = scatter(train_predictions, train_residuals,
             xlabel="Predicted Values",
             ylabel="Residuals",
             title="Residual Analysis",
             alpha=0.6,
             color=:blue,
             label="Training",
             markersize=3)

scatter!(p2, test_predictions, test_residuals,
         alpha=0.6,
         color=:red,
         label="Test",
         markersize=3)

hline!(p2, [0], color=:black, linestyle=:dash, linewidth=2, label="Zero Line")

# 3. Feature importance analysis (correlation with target)
feature_names = ["ρ", "q", "A", "m"]
correlations = [cor(x_train[:, i], y_train) for i in 1:4]

p3 = bar(feature_names, abs.(correlations),
         xlabel="Features",
         ylabel="Absolute Correlation with Target",
         title="Feature Importance",
         color=:lightblue,
         legend=false)


# Combine plots
final_plot = plot(p1, p2, p3, layout=(3,1), size=(1000, 800))
savefig(final_plot, "physical_dimensionality_analysis.png")
println("Analysis plot saved as 'physical_dimensionality_analysis.png'")

# 5. Detailed comparison with true relationship
println("=== Detailed Comparison with True Relationship ===")

# Calculate predictions using the true relationship
true_predictions_train = -x_train[:, 1] .* x_train[:, 2] .* x_train[:, 3] .* x_train[:, 4]
true_predictions_test = -x_test[:, 1] .* x_test[:, 2] .* x_test[:, 3] .* x_test[:, 4]

# Compare evolved vs true relationship
comparison_plot = plot(layout=(1,2), size=(1000, 400))

# Training comparison
scatter!(comparison_plot[1], true_predictions_train, train_predictions,
         xlabel="True Relationship Predictions",
         ylabel="Evolved Expression Predictions",
         title="Training: Evolved vs True Relationship",
         alpha=0.6,
         color=:blue,
         legend=false,
         markersize=3)

# Test comparison  
scatter!(comparison_plot[2], true_predictions_test, test_predictions,
         xlabel="True Relationship Predictions", 
         ylabel="Evolved Expression Predictions",
         title="Test: Evolved vs True Relationship",
         alpha=0.6,
         color=:red,
         legend=false,
         markersize=3)

# Perfect agreement lines
for i in 1:2
    min_val = i == 1 ? minimum(true_predictions_train) : minimum(true_predictions_test)
    max_val = i == 1 ? maximum(true_predictions_train) : maximum(true_predictions_test)
    plot!(comparison_plot[i], [min_val, max_val], [min_val, max_val],
          color=:black, linestyle=:dash, linewidth=2)
end

savefig(comparison_plot, "evolved_vs_true_comparison.png")
println("Evolved vs true relationship comparison saved as 'evolved_vs_true_comparison.png'")

# Calculate agreement metrics
train_agreement = cor(true_predictions_train, train_predictions)
test_agreement = cor(true_predictions_test, test_predictions)

println("Agreement with true relationship:")
println("  Training correlation: $(round(train_agreement, digits=6))")
println("  Test correlation: $(round(test_agreement, digits=6))")
println()
```


## Real-World Applications

### 1. Fluid Dynamics

```julia
# Discovering relationships in fluid mechanics
# Example: Pressure drop in pipe flow

feature_dims_fluid = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, 0, 0, 0, 0, 0, 0],     # Friction Factor
    :x2 => Float16[1, -3, 0, 0, 0, 0, 0],    # ρ (density)  
    :x3 => Float16[0, 1, -1, 0, 0, 0, 0],    # v (velocity) 
    :x4 => Float16[0, 1, 0, 0, 0, 0, 0],     # L (length)  
    :x5 => Float16[0, 1, 0, 0, 0, 0, 0]     # D (diameter)
)

# Target: Pressure [ kg m⁻¹ s⁻²]
target_dim_fluid = Float16[1, -1, -2, 0, 0, 0, 0]  # ΔP (pressure drop)
```

### 2. Heat Transfer

```julia
# Discovering heat transfer
# Example: Heat flux

feature_dims_heat_flux = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[1, 0, -3, 0, -1, 0, 0],   # h (heat transfer coefficient) 
    :x2 => Float16[0, 0, 0, 0, 1, 0, 0],     # ΔT (temperature difference)
)

# Target: Heat flux q = h⋅ΔT [W m⁻²]
target_dim_heat_flux = Float16[1, 0, -3, 0, 0, 0, 0]
```

### 3. Electromagnetic Theory

```julia
# Discovering electromagnetic relationships
# Example: Electromagnetic wave propagation

feature_dims_em = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, 1, -1, 0, 0, 0, 0],    # c (speed of light) 
    :x2 => Float16[0, 0, -1, 0, 0, 0, 0],    # f (frequency)
    :x3 => Float16[0, 1, 0, 0, 0, 0, 0],     # λ (wavelength)
    :x4 => Float16[-1, -3, 4, 0, 2, 0, 0],   # ε₀ (permittivity)
    :x5 => Float16[1, 1, -2, 0, -2, 0, 0],   # μ₀ (permeability)
)

# Target: Speed [m s⁻¹]
target_dim_em = Float16[0, 1, -1, 0, 0, 0, 0]
```

## Benefits of Dimensional Constraints

### 1. Physical Plausibility
- Ensures evolved expressions make physical sense
- Prevents dimensionally inconsistent solutions
- Maintains scientific validity

### 2. Search Space Reduction
- Eliminates large portions of invalid solution space
- Focuses search on physically meaningful expressions
- Improves convergence efficiency

### 3. Interpretability
- Results have clear physical meaning
- Easier to validate against known physics
- Facilitates scientific understanding

### 4. Robustness
- Reduces overfitting to specific datasets
- Improves generalization to new conditions
- Increases confidence in extrapolation


## Common Challenges and Solutions

### Challenge 1: Complex Dimensional Relationships
**Problem**: Some physical relationships involve complex dimensional dependencies
**Solution**: Use hierarchical approach, break into simpler sub-problems

### Challenge 2: Dimensionless Numbers
**Problem**: Important physics often involves dimensionless groups
**Solution**: Include dimensionless combinations as derived features

### Challenge 3: Unit System Consistency
**Problem**: Mixed unit systems can cause confusion
**Solution**: Standardize on SI units throughout analysis

### Challenge 4: Computational Overhead
**Problem**: Dimensional checking adds computational cost
**Solution**: Optimize dimensional propagation algorithms, use caching

Physical dimensionality constraints in GeneExpressionProgramming.jl provide a powerful tool for discovering physically meaningful relationships while ensuring scientific validity. This feature makes the package particularly valuable for scientific and engineering applications where physical laws must be respected.

---

*Next: [Tensor Regression](tensor-regression.md)*

