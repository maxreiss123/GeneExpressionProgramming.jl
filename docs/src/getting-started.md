# Getting Started

This guide runs a first symbolic regression and shows the options you are most likely to change. [Core Concepts](core-concepts.md) explains the algorithm behind them.

## What is Gene Expression Programming?

Gene Expression Programming (GEP) is an evolutionary algorithm that evolves mathematical expressions. A model is a fixed-length linear chromosome: genes, each a head (functions and terminals) and a tail (terminals only), joined by connectors. Each gene is read in prefix order until its expression is complete; the tail, which holds only terminals, is long enough for that. GeneExpressionProgramming.jl evaluates the expressed part of the chromosome, its karva string, directly on data columns, without building an expression tree. It adds multi-objective selection and physical-dimension constraints.

## Your First Symbolic Regression

### Step 1: Load the Package

```julia
using GeneExpressionProgramming
using Random
```

Start Julia with several threads (`julia --threads=auto`) to evaluate the population in parallel.

### Step 2: Generate Sample Data

A synthetic dataset with a known relationship:

```julia
# Set random seed for reproducibility
Random.seed!(42)

# Define the number of features
number_features = 2

# Generate random input data, one row per sample
n_samples = 100
x_data = randn(Float64, n_samples, number_features)

# Define the true function: f(x1, x2) = x1² + x1*x2 - 2*x2²
y_data = @. x_data[:,1]^2 + x_data[:,1] * x_data[:,2] - 2 * x_data[:,2]^2

# Add some noise
y_data += 0.1 * randn(n_samples)
```

### Step 3: Create the Regressor

```julia
# Create a GEP regressor
regressor = GepRegressor(number_features)

# Define evolution parameters
epochs = 1000          # Number of generations
population_size = 1000 # Size of the population
```

The defaults are 3 genes with a head length of 6, the functions `+ - * /` (also the connectors), and as constants 0.0, 0.5 and one random constant drawn from [0, 1). [Customizing the Regressor](#Customizing-the-Regressor) shows how to change them.

### Step 4: Train the Model

```julia
# Fit the regressor to the data
fit!(regressor, epochs, population_size, x_data', y_data; loss_fun="mse")
```

`fit!` expects the features with one row per feature and one column per sample, hence the transpose of `x_data`.

### Step 5: Make Predictions and Analyze Results

```julia
# Make predictions on the training data
predictions = regressor(x_data')

# Display the best evolved expression
println("Best expression: ", regressor.best_models_[1])
println("Fitness (MSE): ", regressor.best_models_[1].fitness[1])

# R² with the package's score
r2 = get_loss_function("r2_score")(y_data, predictions)
println("R² Score: ", r2)
```

### Step 6: Visualize Results (Optional)

Plots.jl is not a dependency of the package; install it into your environment (`Pkg.add("Plots")`) to draw the results.

```julia
using Plots

# Create a scatter plot comparing actual vs predicted values
scatter(y_data, predictions, 
        xlabel="Actual Values", 
        ylabel="Predicted Values",
        title="Actual vs Predicted Values",
        label="Predictions",
        alpha=0.6)

# Add perfect prediction line
plot!([minimum(y_data), maximum(y_data)], 
      [minimum(y_data), maximum(y_data)], 
      color=:red, 
      linestyle=:dash,
      label="Perfect Prediction")
```

## Understanding the Results

- `regressor.best_models_` holds the `hof` best models of the run (3 by default), best first; `regressor(x)` predicts with the first.
- A model's `fitness` is a tuple with one entry per objective, here the training MSE (lower is better). A model whose karva string repeats another's carries that fitness times the duplicate `penalty` (see [Population Dynamics](core-concepts.md#Population-Dynamics)).
- The R² score is the coefficient of determination (closer to 1 is better).

### Interpreting Evolved Expressions

Printing a model writes it as an equation:
- `x1`, `x2`, etc. are the input features (or the names given in `entered_features`)
- Every binary operation is parenthesised, so the printed form is unambiguous
- Other functions appear in mathematical notation if they are in the function set: `sin(…)`, `e^(…)` for `exp`, `ln(…)` for `log`, `√(…)`, `(…)²` for `sqr`, `|…|` for `abs`

For example, an evolved expression might look like:
```
((x1 * x1) + (x2 * (x1 - (x2 + x2))))
```

which is x1² + x1·x2 − 2·x2², the original function, though not in its simplest form. When the constant optimiser has tuned the constants of a model, they are printed with their fitted values (`regressor.best_models_[1].optimised_constants` holds them).

## Working with Real Data

### Loading Data from Files

```julia
using CSV, DataFrames

# Load data from CSV
df = CSV.read("your_data.csv", DataFrame)                # your file

# Extract features and target
feature_columns = [:feature1, :feature2, :feature3]      # your column names
target_column = :target

x_data = Matrix{Float64}(df[:, feature_columns])         # one row per sample
y_data = Float64.(df[:, target_column])

# Get number of features
number_features = length(feature_columns)
```

Features and target should share one floating-point type, as `train_test_split` requires.

### Data Preprocessing

`minmax_scale` maps each column onto the interval from 0 to 1, or the one `feature_range` sets; the evolved expressions are then in the scaled variables.

```julia
x_data_scaled = minmax_scale(x_data)
```

### Train-Test Split

The package exports `train_test_split`, which shuffles the rows (one sample per row) and splits them at `train_ratio`:

```julia
x_train, y_train, x_test, y_test = train_test_split(x_data, y_data; train_ratio=0.8)

# the held-out set is the validation data reported during training
fit!(regressor, epochs, population_size, x_train', y_train;
     x_test=x_test', y_test=y_test, loss_fun="mse")
```

`consider=k` keeps every k-th row of both parts, to subsample large files.

## Customizing the Regressor

### Basic Parameters

```julia
regressor = GepRegressor(
    number_features;
    gene_count = 2,                     # Number of genes per chromosome (default 3)
    head_len = 7,                       # Head length of genes (default 6)
    rnd_count = 2,                      # Number of random constants (default 1)
    tail_weigths = [0.6, 0.2, 0.2],     # Sampling weight of each feature, fixed constant and random constant
    gene_connections = [:+, :-, :*, :/], # Connectors; only those also among the functions are used
    entered_terminal_nums = [Symbol(0.0), Symbol(0.5)] # Constant terminals
)
```

### Function Set Customization

```julia
# Define custom function set
custom_functions = [
    :+, :-, :*, :/,              # Basic arithmetic
    :sin, :cos, :exp, :log,      # Transcendental functions
    :sqrt, :abs                  # Other functions
]

regressor = GepRegressor(number_features; entered_non_terminals=custom_functions)
```

[Function Sets](api-reference.md#Function-Sets) lists every available function. A candidate that cannot be evaluated, e.g. the `log` of a negative number, gets the worst fitness (`Inf`).

### Loss Function Options

```julia
# Mean Squared Error (default)
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="mse")

# Mean Absolute Error
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="mae")

# Root Mean Squared Error
fit!(regressor, epochs, population_size, x_train', y_train; loss_fun="rmse")

# Any function of (y_true, y_pred) that returns a number to minimise
fit!(regressor, epochs, population_size, x_train', y_train;
     loss_fun=(y, p) -> maximum(abs.(y .- p)))
```

The named losses are `"mse"`, `"rmse"`, `"mae"`, `"nrmse"`, `"srsme"`, `"r2_score"`, `"r2_score_f"` and `"xi_core"` (see [Loss Functions](api-reference.md#Loss-Functions)); the search minimises, so use the error measures to drive it and the scores to report on it.

## Monitoring Training Progress

```julia
# After training: one tuple per epoch; epochs after a break condition stopped the run are unassigned
train_loss = regressor.fitness_history_.train_loss
fitness_history = [train_loss[i][1] for i in eachindex(train_loss) if isassigned(train_loss, i)]

# Plot fitness over generations (Plots.jl)
plot(1:length(fitness_history), fitness_history,
     xlabel="Generation",
     ylabel="Fitness (MSE)",
     title="Training Progress",
     legend=false)
```

## Tuning

- **Search effort**: `population_size` and `epochs` (arguments of `fit!`); a `break_condition=(population, epoch) -> Bool` stops a run early when it returns `true` (see [Early Stopping](examples/basic-regression.md#Early-Stopping)).
- **Expression size**: `head_len` and `gene_count` bound it; to trade accuracy against size explicitly, use [several objectives](examples/multi-objective.md).
- **Genetic operators**: their rates are in `RegressionWrapper.GENE_COMMON_PROBS` (see [Genetic Operators](api-reference.md#Genetic-Operators)).
- **Coefficients**: `linear_scaling=true` solves one least-squares coefficient per gene, so evolution only searches for the structure (needs `:+` and `:*` among the functions).
- **Threads**: start Julia with more threads (`julia --threads=auto`).

## Next Steps

1. **[Multi-Objective Optimization](examples/multi-objective.md)**: Balance accuracy and complexity
2. **[Physical Dimensionality](examples/physical-dimensions.md)**: Ensure dimensional consistency
3. **[Tensor Regression](examples/tensor-regression.md)**: Work with vector and matrix data
4. **[Surrogate Screening](examples/surrogate-screening.md)**: Search with an expensive loss, such as a solver in the loop

---

*Continue to [Core Concepts](core-concepts.md) for the algorithm behind these options.*
