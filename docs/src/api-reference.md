# API Reference

This comprehensive API reference provides detailed documentation for all public functions, types, and modules in GeneExpressionProgramming.jl. The API is organized by functionality to help you quickly find the components you need.

## Core Types

### GepRegressor

The main regressor for scalar symbolic regression tasks.

```julia
GepRegressor(number_features::Int; kwargs...)
```

**Parameters:**
- `number_features::Int`: Number of input features
- `gene_count::Int = 2`: Number of genes per chromosome
- `head_len::Int = 7`: Head length of each gene
- `rnd_count::Int = 5`: Amount of rondom numbers beeing considered  
- `entered_non_terminals::Vector{Symbol} = [:+, :-, :*, :/]`: Available functions
- `gene_connections::Vector{Symbol} = [:+, :-, :*, :/]`: Functions for connecting the genes
- `number_of_objectives::Int = 1`: Number of objectives (1 for single-objective)
- `considered_dimensions::Dict{Symbol,Vector{Float16}} = Dict()`: Physical dimensions
- `max_permutations_lib::Int = 1000`: Maximum permutations for dimensional analysis
- `rounds::Int = 5`: Tree depth for dimensional checking


**Fields:**
- `best_models_::Vector`: Best evolved models
- `fitness_history_`: Training history (if available)

**Example:**
```julia
regressor = GepRegressor(3; 
                        gene_count=3,
                        head_len=5,
                        entered_non_terminals=[:+, :-, :*, :/, :sin, :cos])
```

### GepTensorRegressor

Specialized regressor for tensor (vector/matrix) symbolic regression.

```julia
GepTensorRegressor(number_features::Int, gene_count::Int, head_len::Int; kwargs...)
```

**Parameters:**
- `number_features::Int`: Number of input features
- `gene_count::Int`: Number of genes per chromosome
- `head_len::Int`: Head length of each gene
- `feature_names::Vector{String} = []`: Names for features (for interpretability)

**Example:**
```julia
regressor = GepTensorRegressor(5, 2, 3; 
                              feature_names=["x1", "x2", "U1", "U2", "U3"])
```

## Core Functions

### fit!

Train the GEP regressor on data.

```julia
fit!(regressor, epochs::Int, population_size::Int, x_data, y_data; kwargs...)
fit!(regressor, epochs::Int, population_size::Int, loss_function)
```

**Parameters:**
- `regressor`: GepRegressor or GepTensorRegressor instance
- `epochs::Int`: Number of generations to evolve
- `population_size::Int`: Population size for evolution
- `x_data`: Input features (features as rows, samples as columns)
- `y_data`: Target values
- `loss_function`: Custom loss function (for tensor regression or multi objective)

**Keyword Arguments:**
- `x_test = nothing`: Test features for validation
- `y_test = nothing`: Test targets for validation
- `loss_fun::Function = "function"`: Loss function self defined by the user to guide the search
- `target_dimension = nothing`: Target physical dimension

**Examples:**
```julia
# Basic regression
fit!(regressor, 1000, 1000, x_train', y_train; loss_fun="mse")

# With validation data
fit!(regressor, 1000, 1000, x_train', y_train; 
     x_test=x_test', y_test=y_test, loss_fun="rmse")

# With physical dimensions
fit!(regressor, 1000, 1000, x_train', y_train; 
     target_dimension=target_dim)

# Tensor regression with custom loss
fit!(regressor, 100, 500, custom_loss_function)
```

### Prediction

Make predictions using trained regressor.

```julia
(regressor::GepRegressor)(x_data)
(regressor::GepTensorRegressor)(input_data)
```

**Parameters:**
- `x_data`: Input features (features as rows, samples as columns)
- `input_data`: Input data tuple for tensor regression

**Returns:**
- Predictions as vector (scalar regression) or vector of tensors (tensor regression)

**Examples:**
```julia
# Scalar predictions
predictions = regressor(x_test')

# Tensor predictions
tensor_predictions = tensor_regressor(input_tuple)
```

## Utility Functions

### Data Utilities

#### train_test_split

```julia
train_test_split(X, y; test_ratio=0.2, random_state=42)
```

Split data into training and testing sets.

**Parameters:**
- `X`: Feature matrix
- `y`: Target vector
- `test_ratio::Float64 = 0.2`: Proportion of data for testing
- `random_state::Int = 42`: Random seed

**Returns:**
- `(X_train, X_test, y_train, y_test)`: Split data

**Example:**
```julia
X_train, X_test, y_train, y_test = train_test_split(X, y; test_ratio=0.3)
```

### Expression Utilities

#### print_karva_strings

```julia
print_karva_strings(solution)
```

Print the Karva notation representation of an evolved solution.

**Parameters:**
- `solution`: Evolved solution from `best_models_`

**Example:**
```julia
best_solution = regressor.best_models_[1]
print_karva_strings(best_solution)
```

## Loss Functions

### Built-in Loss Functions

The package provides several built-in loss functions accessible via string names:

#### "mse" - Mean Squared Error
```julia
mse(y_true, y_pred) = mean((y_true .- y_pred).^2)
```

#### "mae" - Mean Absolute Error
```julia
mae(y_true, y_pred) = mean(abs.(y_true .- y_pred))
```

#### "rmse" - Root Mean Squared Error
```julia
rmse(y_true, y_pred) = sqrt(mean((y_true .- y_pred).^2))
```

### Custom Loss Functions

For advanced applications, you can define custom loss functions:

#### Single-Objective Custom Loss
```julia
function custom_loss(y_true, y_pred)
    # Your custom loss calculation
    return loss_value::Float64
end

# Use with fit!
fit!(regressor, epochs, population_size, x_data', y_data; loss_fun=custom_loss)
```

#### Multi-Objective Custom Loss
```julia
@inline function multi_objective_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        model = elem.compiled_function
        
        try
            y_pred = model(x_data')
            
            # Objective 1: Accuracy
            mse = mean((y_true .- y_pred).^2)
            
            # Objective 2: Complexity
            complexity = expression_complexity(model) # expression complexity needs to be defined by the user
            
            elem.fitness = (mse, complexity)
        catch
            elem.fitness = (typemax(Float64), typemax(Float64))
        end
    end
end

# Use with multi-objective regressor
regressor = GepRegressor(n_features; number_of_objectives=2)
fit!(regressor, epochs, population_size, multi_objective_loss)
```

#### Tensor Custom Loss
```julia
@inline function tensor_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        model = elem.compiled_function
        
        try
            predictions = model(input_data)
            
            # Calculate tensor-specific loss
            total_error = 0.0
            for i in 1:length(target_tensors)
                error = norm(predictions[i] - target_tensors[i])^2
                total_error += error
            end
            
            elem.fitness = (total_error / length(target_tensors),)
        catch
            elem.fitness = (typemax(Float64),)
        end
    end
end
```

## Selection Methods

### Tournament Selection

Default selection method that chooses the best individul based on the tournament selections.

**Configuration:**
```julia
regressor = GepRegressor(n_features)
```

### NSGA-II Selection

Multi-objective selection using Non-dominated Sorting Genetic Algorithm II.

**Configuration:**
```julia
regressor = GepRegressor(n_features; 
                        number_of_objectives=2)
```

## Genetic Operators

### Genetic Operators

The package implements several genetic operators. Here the can be adjusted in advance using the dictinary `GENE_COMMON_PROBS`, which is available after loading the `GeneExpressionProgramming.jl`

- **Point Mutation**: Random symbol replacement
- **Inversion**: Sequence reversal
- **IS Transposition**: Insertion sequence transposition
- **RIS Transposition**: Root insertion sequence transposition

**Configuration:**
```julia
using GeneExpressionProgramming

GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["mutation_prob"] = 1.0 # Probability for a chromosome of facing a mutation
GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["mutation_rate"] = 0.1 # Proportion of the gene beeing changed


GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["inversion_prob"] = 0.1 # Setting the prob. for the operation to take place 
GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["reverse_insertion_tail"] = 0.1 # Setting  IS 
GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["reverse_insertion"] = 0.1 # Setting RIS
GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["gene_transposition"] = 0.0  # Setting Transposition


```

### Crossover Operators

Available crossover operators: Similar to the gene

- **One-Point Crossover**: Single crossover point
- **Two-Point Crossover**: Two crossover points

**Configuration:**
```julia
using GeneExpressionProgramming

GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["one_point_cross_over_prob"] = 0.5 # Setting the one-point crossover
GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["two_point_cross_over_prob"] = 0.3 # Setting the two-point crossover
```

## Function Sets

### Basic Arithmetic
```julia
basic_functions = [:+, :-, :*, :/]
```

### Extended Mathematical Functions
```julia
extended_functions = [:+, :-, :*, :/, :sin, :cos, :tan, :exp, :log, :sqrt, :abs]
```

### Power Functions
```julia
power_functions = [:^, :sqrt]
```

### Trigonometric Functions
```julia
trig_functions = [:sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh]
```

## Physical Dimensionality

### Dimension Representation

Physical dimensions are represented as 7-element vectors corresponding to SI base units:

```julia
# [Mass, Length, Time, Temperature, Current, Amount, Luminosity]
velocity_dim = Float16[0, 1, -1, 0, 0, 0, 0]    # [L T⁻¹]
force_dim = Float16[1, 1, -2, 0, 0, 0, 0]       # [M L T⁻²]
energy_dim = Float16[1, 2, -2, 0, 0, 0, 0]      # [M L² T⁻²]
```

### Dimensional Constraints

```julia
feature_dims = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[1, 0, 0, 0, 0, 0, 0],    # Mass
    :x2 => Float16[0, 1, 0, 0, 0, 0, 0],    # Length
    :x3 => Float16[0, 0, 1, 0, 0, 0, 0],    # Time
)

target_dim = Float16[0, 1, -1, 0, 0, 0, 0]  # Velocity

regressor = GepRegressor(3; 
                        considered_dimensions=feature_dims,
                        max_permutations_lib=10000)

fit!(regressor, epochs, population_size, x_data', y_data; 
     target_dimension=target_dim)
```

## Tensor Operations (under constructions)

### Supported Tensor Types

The tensor regression module supports various tensor types through Tensors.jl:

```julia
using Tensors

# Vectors (rank-1 tensors)
vector_3d = rand(Tensor{1,3})

# Matrices (rank-2 tensors)  
matrix_2x2 = rand(Tensor{2,2})

# Higher-order tensors
tensor_3x3x3 = rand(Tensor{3,3})
```

### Tensor Operations

Available tensor operations include:

- **Element-wise operations**: `+`, `-`, `*`, `/`
- **Tensor products**: `⊗` (outer product)
- **Contractions**: `⋅` (dot product), `⊡` (double contraction)
- **Norms**: `norm()`, `tr()` (trace)
- **Decompositions**: `eigen()`, `svd()`

## Error Handling

### Common Error

#### ArgumentError: collection must be non-empty
Thrown when the argument vector for the selection process is empty. This happens, when all the loss returns `Inf` for all fit values. 

## Performance Tuning

### Memory Management

```julia
# Monitor memory usage
using Profile

@profile fit!(regressor, epochs, population_size, x_data', y_data)
Profile.print()

# Force garbage collection
GC.gc()
```

## Configuration Examples

### Basic Configuration
```julia
regressor = GepRegressor(3)
fit!(regressor, 1000, 1000, x_data', y_data)
```

### Advanced Configuration
```julia
regressor = GepRegressor(
    5;                                    # 5 input features
    population_size = 2000,               # Large population
    gene_count = 3,                       # 3 genes per chromosome
    head_len = 8,                         # Longer expressions
    entered_non_terminals = [:+, :-, :*, :/, :sin, :cos, :exp]
)

fit!(regressor, 1500, 2000, x_train', y_train;
     x_test = x_test', 
     y_test = y_test,
     loss_fun = "rmse")
```

### Multi-Objective Configuration
```julia
regressor = GepRegressor(
    3;
    number_of_objectives = 2,
    population_size = 1500,
    gene_count = 2,
    head_len = 6
)

fit!(regressor, 1000, 1500, loss_function=multi_objective_loss)
```

### Physical Dimensionality Configuration
```julia
feature_dims = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[1, 0, 0, 0, 0, 0, 0],  # Mass
    :x2 => Float16[0, 1, 0, 0, 0, 0, 0],  # Length
    :x3 => Float16[0, 0, 1, 0, 0, 0, 0],  # Time
)

regressor = GepRegressor(
    3;
    considered_dimensions = feature_dims,
    max_permutations_lib = 15000,
    rounds = 8
)

target_dim = Float16[1, 1, -2, 0, 0, 0, 0]  # Force

fit!(regressor, 1200, 1200, x_data', y_data;
     target_dimension = target_dim)
```

### Tensor Regression Configuration
```julia
regressor = GepTensorRegressor(
    5,                                    # 5 features
    gene_count = 2,			  # Inserting the number of genes
    head_len = 5, 			  # Inserting the head_len for each gene
    feature_names = ["scalar1", "scalar2", "vector1", "vector2", "matrix1"]
)

fit!(regressor, 150, 800, tensor_loss_function)
```

## Version Information

```julia
# Get package version
using Pkg
Pkg.status("GeneExpressionProgramming")

# Check for updates
Pkg.update("GeneExpressionProgramming")
```

## Debugging and Diagnostics



### Fitness History
```julia
# Access fitness evolution
if hasfield(typeof(regressor), :fitness_history_)
    history = regressor.fitness_history_
    train_loss = [elem[1] for elem in history.train_loss]
    plot(train_loss)
end
```

### Expression Analysis
```julia
# Analyze best expressions
for (i, model) in enumerate(regressor.best_models_)
    println("Model $i: $(model.compiled_function)")
    println("Fitness: $(model.fitness)")
end
```

This API reference provides comprehensive coverage of all public interfaces in GeneExpressionProgramming.jl. For additional examples and use cases, refer to the [Examples](./examples/).

---

*For the most up-to-date API documentation, always refer to the package source code and docstrings.*

