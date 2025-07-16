# Basic Symbolic Regression

This example demonstrates the fundamental usage of GeneExpressionProgramming.jl for symbolic regression tasks. We'll walk through a complete workflow from data generation to result analysis, providing detailed explanations and best practices along the way.

## Problem Setup

For this example, we'll work with a synthetic dataset where we know the true underlying function. This allows us to evaluate how well the algorithm can rediscover the known mathematical relationship. The target function we'll use is:

```
f(x₁, x₂) = x₁² + x₁ × x₂ - 2 × x₁ × x₂
```

This function combines polynomial terms and demonstrates the algorithm's ability to discover both quadratic relationships and interaction terms between variables.

## Complete Example Code

```julia
using GeneExpressionProgramming
using Random
using Statistics
using Plots

# Set random seed for reproducibility
Random.seed!(42)

println("=== Basic Symbolic Regression Example ===")
println("Target function: f(x₁, x₂) = x₁² + x₁×x₂ - 2×x₁×x₂")
println()

# Problem parameters
number_features = 2
n_samples = 200
noise_level = 0.05

# Generate training data
println("Generating training data...")
x_train = randn(Float64, n_samples, number_features)
y_train = @. x_train[:,1]^2 + x_train[:,1] * x_train[:,2] - 2 * x_train[:,1] * x_train[:,2]

# Add noise to make the problem more realistic
y_train += noise_level * randn(n_samples)

# Generate separate test data
n_test = 50
x_test = randn(Float64, n_test, number_features)
y_test = @. x_test[:,1]^2 + x_test[:,1] * x_test[:,2] - 2 * x_test[:,1] * x_test[:,2]
y_test += noise_level * randn(n_test)

println("Training samples: $n_samples")
println("Test samples: $n_test")
println("Noise level: $noise_level")
println()

# Evolution parameters
epochs = 1000
population_size = 1000

println("Evolution parameters:")
println("  Epochs: $epochs")
println("  Population size: $population_size")
println()

# Create and configure the regressor
regressor = GepRegressor(number_features)

println("Starting evolution...")
start_time = time()

# Train the model
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="mse")

training_time = time() - start_time
println("Training completed in $(round(training_time, digits=2)) seconds")
println()

# Make predictions
train_predictions = regressor(x_train')
test_predictions = regressor(x_test')

# Calculate performance metrics
function calculate_metrics(y_true, y_pred)
    mse = mean((y_true .- y_pred).^2)
    mae = mean(abs.(y_true .- y_pred))
    rmse = sqrt(mse)
    
    # R² score
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r2 = 1 - ss_res / ss_tot
    
    return (mse=mse, mae=mae, rmse=rmse, r2=r2)
end

train_metrics = calculate_metrics(y_train, train_predictions)
test_metrics = calculate_metrics(y_test, test_predictions)

# Display results
println("=== Results ===")
println("Best evolved expression:")
println("  ", regressor.best_models_[1].compiled_function)
println()

println("Training Performance:")
println("  MSE:  $(round(train_metrics.mse, digits=6))")
println("  MAE:  $(round(train_metrics.mae, digits=6))")
println("  RMSE: $(round(train_metrics.rmse, digits=6))")
println("  R²:   $(round(train_metrics.r2, digits=6))")
println()

println("Test Performance:")
println("  MSE:  $(round(test_metrics.mse, digits=6))")
println("  MAE:  $(round(test_metrics.mae, digits=6))")
println("  RMSE: $(round(test_metrics.rmse, digits=6))")
println("  R²:   $(round(test_metrics.r2, digits=6))")
println()

# Fitness evolution plot
if hasfield(typeof(regressor), :fitness_history_) && 
   !isnothing(regressor.fitness_history_) && 
   hasfield(typeof(regressor.fitness_history_), :train_loss)
    
    fitness_history = [elem[1] for elem in regressor.fitness_history_.train_loss]
    
    p1 = plot(1:length(fitness_history), fitness_history,
              xlabel="Generation",
              ylabel="Best Fitness (MSE)",
              title="Evolution Progress",
              legend=false,
              linewidth=2,
              color=:blue)
    
    # Log scale for better visualization
    p2 = plot(1:length(fitness_history), fitness_history,
              xlabel="Generation",
              ylabel="Best Fitness (MSE)",
              title="Evolution Progress (Log Scale)",
              legend=false,
              linewidth=2,
              color=:blue,
              yscale=:log10)
    
    plot(p1, p2, layout=(1,2), size=(800, 300))
    savefig("fitness_evolution.png")
    println("Fitness evolution plot saved as 'fitness_evolution.png'")
end

# Prediction accuracy plots
p3 = scatter(y_train, train_predictions,
             xlabel="Actual Values",
             ylabel="Predicted Values",
             title="Training Set: Actual vs Predicted",
             alpha=0.6,
             color=:blue,
             label="Training Data")

# Add perfect prediction line
min_val = min(minimum(y_train), minimum(train_predictions))
max_val = max(maximum(y_train), maximum(train_predictions))
plot!(p3, [min_val, max_val], [min_val, max_val],
      color=:red, linestyle=:dash, linewidth=2, label="Perfect Prediction")

p4 = scatter(y_test, test_predictions,
             xlabel="Actual Values",
             ylabel="Predicted Values",
             title="Test Set: Actual vs Predicted",
             alpha=0.6,
             color=:green,
             label="Test Data")

plot!(p4, [min_val, max_val], [min_val, max_val],
      color=:red, linestyle=:dash, linewidth=2, label="Perfect Prediction")

plot(p3, p4, layout=(1,2), size=(800, 300))
savefig("prediction_accuracy.png")
println("Prediction accuracy plot saved as 'prediction_accuracy.png'")

# Residual analysis
train_residuals = y_train .- train_predictions
test_residuals = y_test .- test_predictions

p5 = scatter(train_predictions, train_residuals,
             xlabel="Predicted Values",
             ylabel="Residuals",
             title="Training Residuals",
             alpha=0.6,
             color=:blue,
             label="Training")
hline!(p5, [0], color=:red, linestyle=:dash, linewidth=2, label="Zero Line")

p6 = scatter(test_predictions, test_residuals,
             xlabel="Predicted Values",
             ylabel="Residuals",
             title="Test Residuals",
             alpha=0.6,
             color=:green,
             label="Test")
hline!(p6, [0], color=:red, linestyle=:dash, linewidth=2, label="Zero Line")

plot(p5, p6, layout=(1,2), size=(800, 300))
savefig("residual_analysis.png")
println("Residual analysis plot saved as 'residual_analysis.png'")

println()
println("=== Analysis Complete ===")
```

## Detailed Code Explanation

### Data Generation

The example begins by generating synthetic data with a known mathematical relationship. This approach allows us to evaluate the algorithm's performance objectively since we know the ground truth.

```julia
x_train = randn(Float64, n_samples, number_features)
y_train = @. x_train[:,1]^2 + x_train[:,1] * x_train[:,2] - 2 * x_train[:,1] * x_train[:,2]
y_train += noise_level * randn(n_samples)
```

The input features are drawn from a standard normal distribution, which provides a good range of values for testing the algorithm. The target function combines:
- A quadratic term: `x₁²`
- An interaction term: `x₁ × x₂`
- A scaled interaction term: `-2 × x₁ × x₂`

Adding noise makes the problem more realistic and tests the algorithm's robustness to measurement errors.

### Regressor Configuration

```julia
regressor = GepRegressor(number_features)
```

The `GepRegressor` is initialized with the number of input features. By default, it uses sensible parameter values that work well for most problems:
- Population size: 1000
- Gene count: 2
- Head length: 7
- Function set: Basic arithmetic operations (+, -, *, /)

### Training Process

```julia
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="mse")
```

The training process uses the `fit!` function with:
- **epochs**: Number of generations for evolution
- **population_size**: Number of individuals in each generation
- **x_train'**: Transposed feature matrix (features as rows)
- **y_train**: Target values
- **loss_fun**: Loss function ("mse" for mean squared error)

### Performance Evaluation

The example includes comprehensive performance evaluation:

```julia
function calculate_metrics(y_true, y_pred)
    mse = mean((y_true .- y_pred).^2)
    mae = mean(abs.(y_true .- y_pred))
    rmse = sqrt(mse)
    
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r2 = 1 - ss_res / ss_tot
    
    return (mse=mse, mae=mae, rmse=rmse, r2=r2)
end
```

This function calculates multiple metrics:
- **MSE**: Mean Squared Error - penalizes large errors more heavily
- **MAE**: Mean Absolute Error - robust to outliers
- **RMSE**: Root Mean Squared Error - in the same units as the target
- **R²**: Coefficient of determination - proportion of variance explained

## Visualization and Analysis

The example generates several plots for analysis:

### 1. Fitness Evolution
Shows how the best fitness improves over generations, helping you understand convergence behavior.

### 2. Prediction Accuracy
Scatter plots comparing actual vs. predicted values, with a perfect prediction line for reference.

### 3. Residual Analysis
Plots residuals (prediction errors) against predicted values to identify patterns in the errors.

## Parameter Sensitivity

### Population Size Effects

```julia
# Small population (fast but limited exploration)
regressor_small = GepRegressor(number_features)
fit!(regressor_small, 500, 100, x_train', y_train; loss_fun="mse")

# Large population (thorough exploration but slower)
regressor_large = GepRegressor(number_features)
fit!(regressor_large, 200, 2000, x_train', y_train; loss_fun="mse")
```

### Function Set Customization

```julia
# Extended function set
regressor_extended = GepRegressor(number_features; 
                                 function_set=[:+, :-, :*, :/, :sin, :cos, :exp])
fit!(regressor_extended, epochs, population_size, x_train', y_train; loss_fun="mse")
```

## Common Issues and Solutions

### Issue 1: Poor Convergence

**Symptoms**: High MSE, low R², fitness plateaus early

**Solutions**:
- Increase population size or number of generations
- Adjust mutation/crossover rates
- Try different random seeds
- Check data quality and scaling

### Issue 2: Overfitting

**Symptoms**: Good training performance, poor test performance

**Solutions**:
- Use cross-validation
- Reduce expression complexity (shorter head length)
- Add regularization through multi-objective optimization
- Increase training data size

### Issue 3: Slow Convergence

**Symptoms**: Fitness improves very slowly

**Solutions**:
- Increase mutation rate for more exploration
- Use larger population size
- Check for proper data scaling
- Consider different loss functions

## Advanced Variations

### Custom Loss Function

```julia
function custom_loss(y_true, y_pred)
    # Huber loss (robust to outliers)
    delta = 1.0
    residual = abs.(y_true .- y_pred)
    return mean(ifelse.(residual .<= delta, 
                       0.5 * residual.^2, 
                       delta * (residual .- 0.5 * delta)))
end

# Use custom loss function
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun=custom_loss)
```

### Early Stopping

```julia
function fit_with_early_stopping!(regressor, max_epochs, population_size, x_train, y_train;
                                  patience=50, min_improvement=1e-6)
    best_fitness = Inf
    patience_counter = 0
    
    for epoch in 1:max_epochs
        fit!(regressor, 1, population_size, x_train, y_train; loss_fun="mse")
        current_fitness = regressor.best_models_[1].fitness[1]
        
        if current_fitness < best_fitness - min_improvement
            best_fitness = current_fitness
            patience_counter = 0
        else
            patience_counter += 1
        end
        
        if patience_counter >= patience
            println("Early stopping at epoch $epoch")
            break
        end
    end
end
```

### Cross-Validation

```julia
function cross_validate(X, y, k_folds=5)
    n_samples = size(X, 1)
    fold_size = div(n_samples, k_folds)
    scores = Float64[]
    
    for fold in 1:k_folds
        # Create train/validation split
        val_start = (fold - 1) * fold_size + 1
        val_end = min(fold * fold_size, n_samples)
        
        val_indices = val_start:val_end
        train_indices = setdiff(1:n_samples, val_indices)
        
        X_train_fold = X[train_indices, :]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices, :]
        y_val_fold = y[val_indices]
        
        # Train model
        regressor_fold = GepRegressor(size(X, 2))
        fit!(regressor_fold, 500, 500, X_train_fold', y_train_fold; loss_fun="mse")
        
        # Evaluate
        y_pred_fold = regressor_fold(X_val_fold')
        mse_fold = mean((y_val_fold .- y_pred_fold).^2)
        push!(scores, mse_fold)
        
        println("Fold $fold MSE: $(round(mse_fold, digits=6))")
    end
    
    println("Mean CV MSE: $(round(mean(scores), digits=6)) ± $(round(std(scores), digits=6))")
    return scores
end

# Perform cross-validation
cv_scores = cross_validate(x_train, y_train)
```

## Best Practices Summary

1. **Always use separate test data** for unbiased performance evaluation
2. **Monitor convergence** through fitness evolution plots
3. **Analyze residuals** to identify systematic errors
4. **Use appropriate metrics** for your specific problem type
5. **Consider cross-validation** for robust performance estimates
6. **Visualize results** to gain insights into model behavior
7. **Start with simple parameters** and gradually increase complexity

This basic example provides a solid foundation for understanding GeneExpressionProgramming.jl. The principles and techniques demonstrated here apply to more complex scenarios covered in the advanced examples.

---

*Next: [Multi-Objective Optimization](multi-objective.md)*

