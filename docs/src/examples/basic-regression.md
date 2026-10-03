# Basic Symbolic Regression

This example runs the complete workflow on synthetic data, from data generation to the analysis of the result.

## Problem Setup

The data comes from a known function, so the result can be compared with the truth:

```
f(x₁, x₂) = x₁² + x₁ × x₂ - 2 × x₂²
```

It combines two quadratic terms and an interaction term between the variables.

## Complete Example Code

```julia
using GeneExpressionProgramming
using Random
using Plots

# Set random seed for reproducibility
Random.seed!(42)

# Problem parameters
number_features = 2
n_samples = 200
noise_level = 0.05

# Training data, one row per sample
x_train = randn(Float64, n_samples, number_features)
y_train = @. x_train[:,1]^2 + x_train[:,1] * x_train[:,2] - 2 * x_train[:,2]^2
y_train += noise_level * randn(n_samples)

# Separate test data
n_test = 50
x_test = randn(Float64, n_test, number_features)
y_test = @. x_test[:,1]^2 + x_test[:,1] * x_test[:,2] - 2 * x_test[:,2]^2
y_test += noise_level * randn(n_test)

# Evolution parameters
epochs = 1000
population_size = 1000

# Create the regressor and train it
regressor = GepRegressor(number_features)

training_time = @elapsed fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="mse")
println("Training completed in $(round(training_time, digits=2)) seconds")

# Make predictions
train_predictions = regressor(x_train')
test_predictions = regressor(x_test')

# Performance metrics, with the package's losses and scores
function calculate_metrics(y_true, y_pred)
    return (mse=get_loss_function("mse")(y_true, y_pred),
            mae=get_loss_function("mae")(y_true, y_pred),
            rmse=get_loss_function("rmse")(y_true, y_pred),
            r2=get_loss_function("r2_score")(y_true, y_pred))
end

train_metrics = calculate_metrics(y_train, train_predictions)
test_metrics = calculate_metrics(y_test, test_predictions)

println("Best evolved expression: ", regressor.best_models_[1])
println("Training: ", train_metrics)
println("Test:     ", test_metrics)

# Loss of the best model per epoch (epochs after an early stop would be unassigned)
history = regressor.fitness_history_.train_loss
fitness_history = [history[i][1] for i in eachindex(history) if isassigned(history, i)]

plot(fitness_history,
     xlabel="Generation",
     ylabel="Best Fitness (MSE)",
     title="Evolution Progress",
     legend=false,
     yscale=:log10)
savefig("fitness_evolution.png")

# Prediction accuracy plots
min_val = min(minimum(y_train), minimum(train_predictions))
max_val = max(maximum(y_train), maximum(train_predictions))

p1 = scatter(y_train, train_predictions,
             xlabel="Actual Values", ylabel="Predicted Values",
             title="Training Set", alpha=0.6, label="Training Data")
plot!(p1, [min_val, max_val], [min_val, max_val],
      color=:red, linestyle=:dash, label="Perfect Prediction")

p2 = scatter(y_test, test_predictions,
             xlabel="Actual Values", ylabel="Predicted Values",
             title="Test Set", alpha=0.6, color=:green, label="Test Data")
plot!(p2, [min_val, max_val], [min_val, max_val],
      color=:red, linestyle=:dash, label="Perfect Prediction")

plot(p1, p2, layout=(1,2), size=(800, 300))
savefig("prediction_accuracy.png")

# Residual analysis
p3 = scatter(train_predictions, y_train .- train_predictions,
             xlabel="Predicted Values", ylabel="Residuals",
             title="Training Residuals", alpha=0.6, label="Training")
hline!(p3, [0], color=:red, linestyle=:dash, label="Zero Line")

p4 = scatter(test_predictions, y_test .- test_predictions,
             xlabel="Predicted Values", ylabel="Residuals",
             title="Test Residuals", alpha=0.6, color=:green, label="Test")
hline!(p4, [0], color=:red, linestyle=:dash, label="Zero Line")

plot(p3, p4, layout=(1,2), size=(800, 300))
savefig("residual_analysis.png")
```

## Code Explanation

### Regressor Configuration

```julia
regressor = GepRegressor(number_features)
```

The `GepRegressor` is initialized with the number of input features. Its defaults are:
- Gene count: 3
- Head length: 6
- Function set: `+`, `-`, `*`, `/` (also the connectors)
- Constants: 0.0 and 0.5, plus one random constant drawn from [0, 1)

The population size is not a property of the regressor; it is passed to `fit!`.

### Training Process

```julia
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="mse")
```

- **epochs**: Number of generations
- **population_size**: Number of individuals in each generation
- **x_train'**: The feature matrix with one row per feature, hence the transpose
- **y_train**: Target values
- **loss_fun**: Loss function ("mse" for mean squared error)

### Performance Evaluation

The metrics come from `get_loss_function`:
- **MSE**: Mean squared error, which penalizes large errors more heavily
- **MAE**: Mean absolute error, less sensitive to outliers
- **RMSE**: Root mean squared error, in the same units as the target
- **R²**: Coefficient of determination, the proportion of variance explained

The plots show the loss of the best model per generation, the predictions against the actual values, and the residuals against the predictions, where a pattern points to a systematic error.

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
                                 entered_non_terminals=[:+, :-, :*, :/, :sin, :cos, :exp])
fit!(regressor_extended, epochs, population_size, x_train', y_train; loss_fun="mse")
```

## Advanced Variations

### Custom Loss Function

```julia
using Statistics

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

Every call of `fit!` starts a new search, so early stopping belongs inside the run: `break_condition(population, epoch)` is called after every epoch with the ranked population, and the run stops when it returns `true`.

```julia
function make_early_stopping(; patience=50, min_improvement=1e-6)
    best_fitness = Inf
    since_improvement = 0
    return function (population, epoch)
        current = population[1].fitness[1]
        if current < best_fitness - min_improvement
            best_fitness = current
            since_improvement = 0
        else
            since_improvement += 1
        end
        return since_improvement >= patience || current < 1e-12
    end
end

fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="mse",
     break_condition=make_early_stopping(patience=50))
```

### Cross-Validation

```julia
using Statistics

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

---

*Next: [Multi-Objective Optimization](multi-objective.md)*
