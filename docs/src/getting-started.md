# Getting Started

This guide provides a comprehensive introduction to using GeneExpressionProgramming.jl for symbolic regression tasks. Whether you're new to genetic programming or an experienced practitioner, this guide will help you understand the core concepts and get you running your first symbolic regression experiments quickly.

## What is Gene Expression Programming?

Gene Expression Programming (GEP) is an evolutionary algorithm that evolves mathematical expressions to solve symbolic regression problems. Unlike traditional genetic programming that uses tree structures, GEP uses linear chromosomes that are translated into expression trees, providing several advantages including faster processing and more efficient genetic operations.

The key innovation of GEP lies in its separation of genotype (linear chromosome) and phenotype (expression tree), allowing for more flexible and efficient evolution of mathematical expressions. This package implements GEP with modern Julia optimizations and additional features like multi-objective optimization and physical dimensionality constraints.

## Basic Workflow

The typical workflow for using GeneExpressionProgramming.jl follows these steps:

1. **Data Preparation**: Organize your input features and target values
2. **Regressor Creation**: Initialize a GEP regressor with appropriate parameters
3. **Model Fitting**: Train the regressor using evolutionary algorithms
4. **Prediction**: Use the trained model to make predictions on new data
5. **Analysis**: Examine the evolved expressions and their performance

Let's walk through each step with practical examples.

## Your First Symbolic Regression

### Step 1: Import Required Packages

```julia
using GeneExpressionProgramming
using Random
using Plots  # Optional, for visualization
```

### Step 2: Generate Sample Data

For this example, we'll create a synthetic dataset with a known mathematical relationship:

```julia
# Set random seed for reproducibility
Random.seed!(42)

# Define the number of features
number_features = 2

# Generate random input data
n_samples = 100
x_data = randn(Float64, n_samples, number_features)

# Define the true function: f(x1, x2) = x1² + x1*x2 - 2*x1*x2
y_data = @. x_data[:,1]^2 + x_data[:,1] * x_data[:,2] - 2 * x_data[:,1] * x_data[:,2]

# Add some noise to make it more realistic
y_data += 0.1 * randn(n_samples)
```

### Step 3: Create and Configure the Regressor

```julia
# Create a GEP regressor
regressor = GepRegressor(number_features)

# Define evolution parameters
epochs = 1000          # Number of generations
population_size = 1000 # Size of the population
```

The `GepRegressor` constructor accepts various parameters to customize the evolutionary process. For now, we're using the default settings, which work well for most problems.

### Step 4: Train the Model

```julia
# Fit the regressor to the data
fit!(regressor, epochs, population_size, x_data', y_data; loss_fun="mse")
```

Note that we transpose `x_data` because GeneExpressionProgramming.jl expects features as rows and samples as columns, following Julia's column-major convention.

### Step 5: Make Predictions and Analyze Results

```julia
# Make predictions on the training data
predictions = regressor(x_data')

# Display the best evolved expression
println("Best expression: ", regressor.best_models_[1].compiled_function)
println("Fitness (MSE): ", regressor.best_models_[1].fitness)

# Calculate R² score
function r_squared(y_true, y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - ss_res / ss_tot
end

r2 = r_squared(y_data, predictions)
println("R² Score: ", r2)
```

### Step 6: Visualize Results (Optional)

```julia
# Create a scatter plot comparing actual vs predicted values
scatter(y_data, predictions, 
        xlabel="Actual Values", 
        ylabel="Predicted Values",
        title="Actual vs Predicted Values",
        legend=false,
        alpha=0.6)

# Add perfect prediction line
plot!([minimum(y_data), maximum(y_data)], 
      [minimum(y_data), maximum(y_data)], 
      color=:red, 
      linestyle=:dash,
      label="Perfect Prediction")
```

## Understanding the Results

When you run the above code, GeneExpressionProgramming.jl will evolve mathematical expressions over the specified number of generations. The algorithm will try to find expressions that minimize the mean squared error between predictions and actual values.

The output will show you:
- **Best Expression**: The mathematical formula that best fits your data
- **Fitness**: The loss value (lower is better for MSE)
- **R² Score**: Coefficient of determination (closer to 1 is better)

### Interpreting Evolved Expressions

The evolved expressions use standard mathematical notation:
- `x1`, `x2`, etc. represent your input features
- Common operators include `+`, `-`, `*`, `/`, `^`
- Functions like `sin`, `cos`, `exp`, `log` may appear depending on the function set

For example, an evolved expression might look like:
```
x1 * x1 + x1 * x2 - 2.0 * x1 * x2
```

This closely matches our original function, demonstrating the algorithm's ability to discover the underlying mathematical relationship.

## Working with Real Data

### Loading Data from Files

For real-world applications, you'll typically load data from files:

```julia
using CSV, DataFrames

# Load data from CSV
df = CSV.read("your_data.csv", DataFrame)

# Extract features and target
feature_columns = [:feature1, :feature2, :feature3]  # Adjust column names
target_column = :target

x_data = Matrix(df[:, feature_columns])
y_data = df[:, target_column]

# Get number of features
number_features = length(feature_columns)
```

### Data Preprocessing

Before training, consider preprocessing your data:

```julia
# Normalize features (optional but often helpful)
using Statistics

function normalize_features(X)
    X_norm = copy(X)
    for i in 1:size(X, 2)
        col_mean = mean(X[:, i])
        col_std = std(X[:, i])
        if col_std > 0
            X_norm[:, i] = (X[:, i] .- col_mean) ./ col_std
        end
    end
    return X_norm
end

x_data_normalized = normalize_features(x_data)
```

### Train-Test Split

For proper evaluation, split your data into training and testing sets:

```julia
function train_test_split(X, y; test_ratio=0.2, random_state=42)
    Random.seed!(random_state)
    n_samples = size(X, 1)
    n_test = round(Int, n_samples * test_ratio)
    
    indices = randperm(n_samples)
    test_indices = indices[1:n_test]
    train_indices = indices[n_test+1:end]
    
    return X[train_indices, :], X[test_indices, :], y[train_indices], y[test_indices]
end

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
```

## Customizing the Regressor

### Basic Parameters

The `GepRegressor` constructor accepts several parameters to customize the evolutionary process:

```julia
regressor = GepRegressor(
    number_features;
    gene_count = 2,             # Number of genes per chromosome
    head_len = 7,                # Head length of genes
    rnd_count = 2, 		# assign the number of utilized random numbers
    tail_weigths = [0.6,0.2,0.2],  # assign utlized prob. for the sampled symbols =>  [features, constants, random numbers]
    gene_connections=[:+, :-, :*, :/], #defines how the genes can be connnected
    entered_terminal_nums = [Symbol(0.0), Symbol(0.5)] # define some constant values
    
)
```

### Function Set Customization

You can customize the function set used in evolution:

```julia
# Define custom function set
custom_functions = [
    :+, :-, :*, :/,              # Basic arithmetic
    :sin, :cos, :exp, :log,      # Transcendental functions
    :sqrt, :abs                  # Other functions
]

regressor = GepRegressor(number_features; entered_non_terminals=custom_functions)
```

### Loss Function Options

GeneExpressionProgramming.jl supports various loss functions:

```julia
# Mean Squared Error (default)
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="mse")

# Mean Absolute Error
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="mae")

# Root Mean Squared Error
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="rmse")
```

## Monitoring Training Progress

### Fitness History

You can monitor the training progress by accessing the fitness history:

```julia
# After training
fitness_history = [elem[1] for elem in regressor.fitness_history_.train_loss] # save as tuple within the history

# Plot fitness over generations
plot(1:length(fitness_history), fitness_history,
     xlabel="Generation",
     ylabel="Fitness (MSE)",
     title="Training Progress",
     legend=false)
```


## Best Practices

### 1. Start Simple

Begin with basic parameters and gradually increase complexity:
- Start with smaller population sizes (100-500) for quick experiments
- Use fewer generations initially to test your setup
- Gradually increase complexity as needed

### 2. Data Quality

Ensure your data is clean and well-prepared:
- Remove or handle missing values appropriately
- Consider feature scaling for better convergence
- Ensure sufficient data for the complexity of the target function

### 3. Parameter Tuning

Experiment with different parameter combinations:
- **Population Size**: Larger populations explore more solutions but require more computation
- **Generations**: More generations allow for better solutions but take longer
- **Gene Count**: More genes can represent more complex functions
- **Head Length**: Longer heads allow for more complex expressions

### 4. Validation

Always validate your results on unseen data:
- Use train-test splits or cross-validation
- Check for overfitting by comparing training and test performance
- Consider the interpretability of evolved expressions


## Common Issues and Solutions

### Issue 1: Poor Convergence

If the algorithm doesn't find good solutions:
- Increase population size or number of generations
- Adjust mutation and crossover rates
- Try different selection methods
- Check data quality and preprocessing

### Issue 2: Overly Complex Expressions

If evolved expressions are too complex:
- Reduce head length or gene count
- Use multi-objective optimization to balance accuracy and complexity
- Implement expression simplification post-processing

### Issue 3: Slow Performance

If training is too slow:
- Reduce population size for initial experiments
- Use fewer generations with early stopping
- Consider parallel processing options
- Profile your code to identify bottlenecks

## Next Steps

Now that you understand the basics, explore more advanced features:

1. **[Multi-Objective Optimization](examples/multi-objective.md)**: Balance accuracy and complexity
2. **[Physical Dimensionality](examples/physical-dimensions.md)**: Ensure dimensional consistency
3. **[Tensor Regression](examples/tensor-regression.md)**: Work with vector and matrix data

## Summary

In this guide, you learned:
- The basic workflow for symbolic regression with GeneExpressionProgramming.jl
- How to prepare data and configure the regressor
- How to train models and interpret results
- Best practices for successful symbolic regression
- Common issues and their solutions

The power of GeneExpressionProgramming.jl lies in its ability to automatically discover mathematical relationships in your data while providing interpretable results. As you become more familiar with the package, you can explore its advanced features to tackle more complex problems and achieve better results.

---

*Continue to [Core Concepts](core-concepts.md) to deepen your understanding of the underlying algorithms and theory.*

