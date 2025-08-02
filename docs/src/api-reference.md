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
- `rnd_count::Int = 5`: Amount of random numbers being considered  
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

### StandardRegressionStrategy

A strategy for evaluating standard regression tasks with typed floating-point data.

```julia
struct StandardRegressionStrategy{T<:AbstractFloat} <: EvaluationStrategy
```

**Fields:**
- `operators::OperatorEnum`: Available operators for the strategy
- `number_of_objectives::Int`: Number of optimization objectives
- `x_data::AbstractArray{T}`: Training input data
- `y_data::AbstractArray{T}`: Training target data
- `x_data_test::AbstractArray{T}`: Test input data
- `y_data_test::AbstractArray{T}`: Test target data
- `loss_function::Function`: Primary loss function
- `validation_loss_function::Function`: Validation loss function
- `secOptimizer::Union{Function,Nothing}`: Secondary optimizer (if any)
- `break_condition::Union{Function,Nothing}`: Condition to stop evolution
- `penalty::T`: Penalty value for regularization
- `crash_value::T`: Value assigned on evaluation failure

**Constructor:**
```julia
StandardRegressionStrategy{T}(operators::OperatorEnum,
    x_data::AbstractArray,
    y_data::AbstractArray,
    x_data_test::AbstractArray,
    y_data_test::AbstractArray,
    loss_function::Function;
    validation_loss_function::Union{Nothing,Function}=nothing,
    secOptimizer::Union{Function,Nothing}=nothing,
    break_condition::Union{Function,Nothing}=nothing,
    penalty::T=zero(T),
    crash_value::T=typemax(T)) where {T<:AbstractFloat}
```

**Example:**
```julia
strategy = StandardRegressionStrategy{Float64}(
    OperatorEnum([:+, :-, :*, :/]),
    x_train, y_train, x_test, y_test,
    mse;
    penalty=0.1,
    crash_value=Inf
)
```

### GenericRegressionStrategy

A flexible strategy for generic regression tasks, supporting multi-objective optimization.

```julia
struct GenericRegressionStrategy <: EvaluationStrategy
```

**Fields:**
- `operators::Union{OperatorEnum,Nothing}`: Available operators (optional)
- `number_of_objectives::Int`: Number of optimization objectives
- `loss_function::Function`: Primary loss function
- `validation_loss_function::Union{Function,Nothing}`: Validation loss function
- `secOptimizer::Union{Function,Nothing}`: Secondary optimizer (if any)
- `break_condition::Union{Function,Nothing}`: Condition to stop evolution

**Constructor:**
```julia
GenericRegressionStrategy(operators::Union{OperatorEnum,Nothing}, number_of_objectives::Int, loss_function::Function;
    validation_loss_function::Union{Function,Nothing}=nothing,
    secOptimizer::Union{Function,Nothing}=nothing,
    break_condition::Union{Function,Nothing}=nothing)
```

**Example:**
```julia
strategy = GenericRegressionStrategy(
    nothing,
    2,
    multi_objective_loss;
    validation_loss_function=validate_loss
)
```

### Toolbox

Contains parameters and operations for GEP algorithm execution.

```julia
struct Toolbox
```

**Fields:**
- `gene_count::Int`: Number of genes per chromosome
- `head_len::Int`: Head length for each gene
- `symbols::OrderedDict{Int8,Int8}`: Available symbols and their arities
- `gene_connections::Vector{Int8}`: How genes connect
- `headsyms::Vector{Int8}`: Symbols allowed in head
- `tailsyms::Vector{Int8}`: Symbols allowed in tail
- `arrity_by_id::OrderedDict{Int8,Int8}`: Symbol arities
- `callbacks::Dict`: Operation callbacks
- `nodes::OrderedDict`: Node definitions
- `gen_start_indices::Vector{Int}`: Gene start positions
- `gep_probs::Dict{String,AbstractFloat}`: Operation probabilities
- `fitness_reset::Tuple`: Default fitness values
- `preamble_syms::Vector{Int8}`: Preamble symbols
- `len_preamble::Int8`: Preamble length
- `operators_::Union{OperatorEnum,Nothing}`: Operator definitions
- `compile_function_::Union{Function,Nothing}`: Compilation function
- `tail_weights::Union{Weights,Nothing}`: Probability for tail symbols
- `head_weights::Union{Weights,Nothing}`: Probability for head symbols

**Constructor:**
```julia
Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8}, 
       gene_connections::Vector{Int8}, callbacks::Dict, nodes::OrderedDict, 
       gep_probs::Dict{String,AbstractFloat};
       unary_prob::Real=0.1, preamble_syms=Int8[],
       number_of_objectives::Int=1, operators_::Union{OperatorEnum,Nothing}=nothing,
       function_complile::Union{Function,Nothing}=compile_djl_datatype,
       tail_weights_::Union{Weights,Nothing}=nothing,
       head_tail_balance::Real=0.5)
```

**Example:**
```julia
toolbox = Toolbox(
    3, 5, OrderedDict{Int8,Int8}(1 => 2, 2 => 0), 
    Int8[1], Dict(), OrderedDict(), Dict("mutation_prob" => 0.1);
    unary_prob=0.2
)
```

### Chromosome

Represents an individual solution in GEP.

```julia
mutable struct Chromosome
```

**Fields:**
- `genes::Vector{Int8}`: Genetic material
- `fitness::Tuple`: Fitness score
- `toolbox::Toolbox`: Reference to toolbox
- `compiled_function::Any`: Compiled expression
- `compiled::Bool`: Compilation status
- `expression_raw::Vector{Int8}`: Raw expression
- `dimension_homogene::Bool`: Dimensional homogeneity
- `chromo_id::Int`: Chromosome identifier

**Constructor:**
```julia
Chromosome(genes::Vector{Int8}, toolbox::Toolbox, compile::Bool=false)
```

**Example:**
```julia
chromosome = Chromosome(Int8[1,2,3,4,5], toolbox, true)
```


### SelectedMembers

Represents the selected individuals obtained from NSGA-II .

```julia
struct SelectedMembers
```

**Fields:**
- `indices::Vector{Int}`: Sorted indices according to the current population
- `fronts::Dict{Int,Vector{Int}}`: Listing of the pareto front


## Core Functions

### fit!

Train the GEP regressor model.

There are multiple overloads for different use cases:

1. Standard scalar regression with training data:

```julia
fit!(regressor::GepRegressor, epochs::Int, population_size::Int, x_train::AbstractArray, 
     y_train::AbstractArray; kwargs...)
```

Trains using provided data arrays, with optional validation data and dimensional constraints.

**Arguments:**
- `regressor::GepRegressor`: The regressor instance
- `epochs::Int`: Number of evolutionary generations
- `population_size::Int`: Size of the population
- `x_train::AbstractArray`: Training features
- `y_train::AbstractArray`: Training targets

**Keyword Arguments:**
- `x_test::Union{AbstractArray,Nothing}=nothing`: Test features for validation
- `y_test::Union{AbstractArray,Nothing}=nothing`: Test targets for validation
- `optimization_epochs::Int=100`: Number of epochs for constant optimization
- `hof::Int=3`: Number of best models to keep in hall of fame
- `loss_fun::Union{String,Function}="mse"`: Loss function ("mse", "mae", "rmse" or custom)
- `loss_fun_validation::Union{String,Function}="mse"`: Validation loss function
- `correction_epochs::Int=1`: Interval for dimensional corrections
- `correction_amount::Real=0.05`: Fraction of population to correct for homogeneity
- `opt_method_const::Symbol=:cg`: Method for constant optimization (:cg, :nd, etc.)
- `target_dimension::Union{Vector{Float16},Nothing}=nothing`: Target physical dimension for homogeneity
- `cycles::Int=10`: Cycles for dimension correction
- `max_iterations::Int=1000`: Max iterations for optimizer
- `n_starts::Int=3`: Number of optimizer restarts
- `break_condition::Union{Function,Nothing}=nothing`: Function to check for early stopping
- `file_logger_callback::Union{Function,Nothing}=nothing`: Callback for logging to file
- `save_state_callback::Union{Function,Nothing}=nothing`: Callback to save evolution state
- `load_state_callback::Union{Function,Nothing}=nothing`: Callback to load evolution state
- `population_sampling_multiplier::Int=1`: Multiplier for initial population sampling
- `penalty::AbstractFloat=2.0`: Penalty factor for duplicate functions

2. Custom loss for scalar or multi-objective regression:

```julia
fit!(regressor::GepRegressor, epochs::Int, population_size::Int, loss_function::Function; kwargs...)
```

Allows custom loss for guiding evolution, useful for multi-objective or non-standard fitness.

**Arguments:**
- `regressor::GepRegressor`: The regressor instance
- `epochs::Int`: Number of generations
- `population_size::Int`: Population size
- `loss_function::Function`: Custom loss function that sets fitness

**Keyword Arguments:**
- `optimizer_function_::Union{Function,Nothing}=nothing`: Function for secondary optimization
- `loss_function_validation::Union{Function,Nothing}=nothing`: Validation loss
- `optimization_epochs::Int=100`: Constant optimization epochs
- `hof::Int=3`: Hall of fame size
- `correction_epochs::Int=1`: Dimension correction interval
- `correction_amount::Real=0.3`: Correction fraction
- `opt_method_const::Symbol=:nd`: Constant optimization method
- `target_dimension::Union{Vector{Float16},Nothing}=nothing`: Target dimension
- `cycles::Int=10`: Correction cycles
- `max_iterations::Int=150`: Optimizer iterations
- `n_starts::Int=5`: Optimizer restarts
- `break_condition::Union{Function,Nothing}=nothing`: Early stop condition
- `file_logger_callback::Union{Function,Nothing}=nothing`: Logging callback
- `save_state_callback::Union{Function,Nothing}=nothing`: Save state callback
- `load_state_callback::Union{Function,Nothing}=nothing`: Load state callback
- `penalty::AbstractFloat=2.0`: Duplicate penalty

3. For tensor regression:

```julia
fit!(regressor::GepTensorRegressor, epochs::Int, population_size::Int, loss_function::Function; kwargs...)
```

Trains tensor regressor with custom tensor-specific loss.

**Arguments:**
- `regressor::GepTensorRegressor`: The tensor regressor
- `epochs::Int`: Generations
- `population_size::Int`: Population size
- `loss_function::Function`: Custom tensor loss

**Keyword Arguments:**
- `hof::Int=3`: Hall of fame
- `break_condition::Union{Function,Nothing}=nothing`: Stop condition
- `file_logger_callback::Union{Function,Nothing}=nothing`: Logger
- `save_state_callback::Union{Function,Nothing}=nothing`: Save state
- `load_state_callback::Union{Function,Nothing}=nothing`: Load state

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

### compile_expression!

Compiles chromosome's genes into executable function using DynamicExpressions.

```julia
compile_expression!(chromosome::Chromosome; force_compile::Bool=false)
```

**Parameters:**
- `chromosome::Chromosome`: Chromosome to compile
- `force_compile::Bool=false`: Force recompilation

**Effects:**
Updates chromosome's `compiled_function` and related fields.

**Example:**
```julia
compile_expression!(chromosome, force_compile=true)
```

### fitness

Get chromosome's fitness value.

```julia
fitness(chromosome::Chromosome)
```

**Returns:**
Fitness value or tuple

**Example:**
```julia
fit_value = fitness(chromosome)
```

### set_fitness!

Set chromosome's fitness value.

```julia
set_fitness!(chromosome::Chromosome, value::Tuple)
```

**Parameters:**
- `chromosome::Chromosome`: Target chromosome
- `value::Tuple`: New fitness value

**Example:**
```julia
set_fitness!(chromosome, (0.5, 0.3))
```

### _karva_raw

Convert a chromosome's genes into Karva notation (K-expression).

```julia
_karva_raw(chromosome::Chromosome; split::Bool=false)
```

**Parameters:**
- `chromosome::Chromosome`: The chromosome to convert
- `split::Bool=false`: Whether to split the expression by genes

**Returns:**
Vector{Int8} representing the K-expression, or list of vectors if split=true

**Example:**
```julia
k_expression = _karva_raw(chromosome)
```

### split_karva

Split a chromosome's Karva expression into parts.

```julia
split_karva(chromosome::Chromosome, coeffs::Int=2)
```

**Parameters:**
- `chromosome::Chromosome`: The chromosome to process
- `coeffs::Int=2`: Number of parts to split into

**Returns:**
List of vectors representing split K-expressions

**Example:**
```julia
split_expressions = split_karva(chromosome, coeffs=3)
```

### generate_gene

Generate a single gene for GEP.

```julia
generate_gene(headsyms::Vector{Int8}, tailsyms::Vector{Int8}, headlen::Int,
    tail_weights::Weights, head_weights::Weights)
```

**Parameters:**
- `headsyms::Vector{Int8}`: Symbols for head
- `tailsyms::Vector{Int8}`: Symbols for tail
- `headlen::Int`: Head length
- `tail_weights::Weights`: Probability weights for tail symbols
- `head_weights::Weights`: Probability weights for head symbols

**Returns:**
Vector{Int8} representing gene

**Example:**
```julia
gene = generate_gene(headsyms, tailsyms, 5, tail_weights, head_weights)
```

### generate_chromosome

Generate a new chromosome using toolbox configuration.

```julia
generate_chromosome(toolbox::Toolbox)
```

**Parameters:**
- `toolbox::Toolbox`: Toolbox configuration

**Returns:**
New Chromosome instance

**Example:**
```julia
chromosome = generate_chromosome(toolbox)
```

### perform_step!

Performs one evolutionary step in the GEP algorithm, creating and evaluating new chromosomes.

```julia
perform_step!(population::Vector{Chromosome}, parents::Vector{Chromosome}, 
    next_gen::Vector{Chromosome}, toolbox::Toolbox, mating_size::Int)
```

**Arguments:**
- `population::Vector{Chromosome}`: Current population of chromosomes
- `parents::Vector{Chromosome}`: Selected parent chromosomes for breeding
- `next_gen::Vector{Chromosome}`: Buffer for storing newly created chromosomes
- `toolbox::Toolbox`: Contains genetic operators and algorithm parameters
- `mating_size::Int`: Number of chromosomes to create in this step

**Details:**
- Processes parents in pairs to create new chromosomes
- Applies genetic operations to create offspring
- Compiles expressions for new chromosomes
- Updates population with new chromosomes
- Operations are performed in parallel using multiple threads

### perform_correction_callback!

Applies correction operations to ensure dimensional homogeneity in chromosomes.

```julia
perform_correction_callback!(population::Vector{Chromosome}, epoch::Int, 
    correction_epochs::Int, correction_amount::Real,
    correction_callback::Union{Function,Nothing})
```

**Arguments:**
- `population::Vector{Chromosome}`: Current population of chromosomes
- `epoch::Int`: Current epoch number
- `correction_epochs::Int`: Frequency of correction operations
- `correction_amount::Real`: Proportion of population to apply corrections to
- `correction_callback::Union{Function,Nothing}`: Function that performs the actual correction

**Details:**
- Executes corrections periodically (every correction_epochs)
- Processes a subset of the population determined by correction_amount
- Applies corrections to dimensionally heterogeneous chromosomes
- Updates chromosome compilation and dimensional homogeneity flags

### runGep

Main function that executes the GEP algorithm using a specified evaluation strategy.

```julia
runGep(epochs::Int, population_size::Int, toolbox::Toolbox, evalStrategy::EvaluationStrategy;
    hof::Int=3, correction_callback::Union{Function,Nothing}=nothing,
    correction_epochs::Int=1, correction_amount::Real=0.6,
    tourni_size::Int=3)
```

**Arguments:**
- `epochs::Int`: Number of evolutionary epochs to run
- `population_size::Int`: Size of the chromosome population
- `toolbox::Toolbox`: Contains genetic operators and algorithm parameters
- `evalStrategy::EvaluationStrategy`: Strategy for evaluating chromosomes, handling fitness computation, and optimization

**Optional Arguments:**
- `hof::Int=3`: Number of best solutions to return (Hall of Fame size)
- `correction_callback::Union{Function,Nothing}=nothing`: Function for dimensional homogeneity correction
- `correction_epochs::Int=1`: Frequency of correction operations
- `correction_amount::Real=0.6`: Proportion of population for correction
- `tourni_size::Int=3`: Tournament selection size
- `file_logger_callback::Union{Function,Nothing}=nothing`: Callback for extra logging, expected inputs `file_logger_callback(population::Vector{Chromosome}, epoch::Int, selectedMembers::SelectedMembers)`
- `save_state_callback::Union{Function,Nothing}=nothing`: Callback for save a population, expected inputs `save_state_callback(population::Vector{Chromosome},evalStrategy::EvaluationStrategy)`
- `load_state_callback::Union{Function,Nothing}=nothing`: Callback for loading a population, expected return `tuple(population::Vector{Chromosome},startepoch::Int)`
- ` population_sampling_multiplier::Int=100`: Expansionfactor on the population to employ Latin-Hypercupe
- `cache_size::Int=10000`: Functions stored in cache, (Limit! To mitigate cache blow up)
- `penalty::AbstractFloat=2.0`: Employs a fit penalty for functions that has been seen for the second time

**Returns:**
`Tuple{Vector{Chromosome}, Any}`: Returns best solutions and training history

**Details:**
1. Initializes population and evolution parameters
2. For each epoch:
   - Applies dimensional homogeneity corrections if provided
   - Computes fitness for all chromosomes using evaluation strategy
   - Sorts population based on fitness
   - Applies secondary optimization if specified in strategy
   - Records training progress
   - Checks break condition from evaluation strategy
   - Performs selection and creates new generation
3. Returns final solutions and training history

Progress is monitored through a progress bar showing:
- Current epoch
- Training loss
- Validation loss

The evolution process stops when either:
- Maximum epochs is reached
- Break condition specified in evaluation strategy is met => needs to be informed as break_condition(population, epoch)

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
print_karva_strings(chromosome::Chromosome; split_len::Int=1)
```

Print the Karva notation representation of a chromosome.

**Parameters:**
- `chromosome::Chromosome`: Chromosome to print
- `split_len::Int=1`: Length for splitting output

**Example:**
```julia
print_karva_strings(chromosome)
```

### optimize_constants!

Optimizes constant values in a symbolic expression tree to minimize a given loss function.

```julia
optimize_constants!(
    node::Node,
    loss::Function;
    opt_method::Symbol=:cg,
    max_iterations::Int=250,
    n_restarts::Int=3
)
```

**Arguments:**
- `node::Node`: Expression tree containing constants to optimize
- `loss::Function`: Loss function to minimize
- `opt_method::Symbol=:cg`: Optimization method (:newton, :cg, or other for NelderMead)
- `max_iterations::Int=250`: Maximum iterations per optimization attempt
- `n_restarts::Int=3`: Number of random restarts to avoid local minima

**Returns:**
Tuple containing:
- `best_node::Node`: Expression tree with optimized constants
- `best_loss::Float64`: Final loss value achieved

**Example:**
```julia
# Create expression with constants
expr = Node(*, [
    Node(1.5),  # constant to optimize
    Node(x, degree=1)  # variable
])

# Define loss function
loss(node) = sum((node(x_data) .- y_data).^2)

# Optimize constants
optimized_expr, final_loss = optimize_constants!(
    expr,
    loss;
    opt_method=:cg,
    max_iterations=500,
    n_restarts=5
)
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

#### Template loss
Template loss enables multiexpression formulation or fixed defined template functions like $f(x_1,..x_n)=v(x_1,x_2) + g(x_3)$. 
```julia
@inline function loss_new(elem, validate::Bool)
    try
        if isnan(mean(elem.fitness)) || validate

            g1 = elem.compiled_function[1](x_data', regressor.operators_)
            g2 = elem.compiled_function[2](x_data', regressor.operators_)

            a_pred = @. (g1' * t1) + (g2' * t2)
            loss = abs(norm(a_true - a_pred))
            elem.fitness = (loss,)
        end
    catch e
        @error "something wnt wrong" exception = (e, catch_backtrace())
        elem.fitness = (typemax(Float64),)
    end
end
```

## Selection Methods

### Tournament Selection

Default selection method that chooses the best individual based on tournament selections.

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

The package implements several genetic operators. These can be adjusted in advance using the dictionary `GENE_COMMON_PROBS`, which is available after loading the `GeneExpressionProgramming.jl`

- **Point Mutation**: Random symbol replacement
- **Inversion**: Sequence reversal
- **IS Transposition**: Insertion sequence transposition
- **RIS Transposition**: Root insertion sequence transposition

**Configuration:**
```julia
using GeneExpressionProgramming

GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["mutation_prob"] = 1.0 # Probability for a chromosome of facing a mutation
GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["mutation_rate"] = 0.1 # Proportion of the gene being changed

GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["inversion_prob"] = 0.1 # Setting the prob. for the operation to take place 
GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["insertion_prob"] = 0.1 # Setting IS 
GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["root_insertion_prob"] = 0.1 # Setting RIS
GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["gene_transposition"] = 0.0  # Setting Transposition
```

### Crossover Operators

Available crossover operators:

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

## Tensor Operations (under construction)

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
Thrown when the argument vector for the selection process is empty. This happens when all the loss returns `Inf` for all fit values.

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
    gene_count = 2,                       # Number of genes
    head_len = 5,                         # Head length for each gene
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