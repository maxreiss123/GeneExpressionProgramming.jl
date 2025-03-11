module GeneExpressionProgramming

"""
    GeneExpressionProgramming

A comprehensive module for Gene Expression Programming (GEP) implementation in Julia,
providing tools for symbolic regression and evolutionary computation.

# Module Structure
- `GepEntities`: Core data structures for GEP chromosomes and genetic operations
- `Gep`: Main GEP algorithm implementation
- `Losses`: Loss functions for fitness evaluation
- `PhyConstants`: Physical constants and dimensional analysis utilities
- `Sbp`: Symbolic computation and library management
- `Selection`: Selection mechanisms for evolutionary processes
- `Util`: Utility functions for GEP operations
- `RegressionWrapper`: High-level wrapper for regression tasks

# Main Exports

## Core GEP Components
- `Chromosome`, `Toolbox`: Basic building blocks for GEP
- `runGep`: Main function for running GEP algorithm
- `GepRegressor`: High-level regression interface

## Selection Mechanisms
- `selection_NSGA`: NSGA-II selection algorithm
- `basic_tournament_selection`: Tournament selection implementation
- `dominates_`, `fast_non_dominated_sort`: NSGA-II utility functions
- `calculate_fronts`, `determine_ranks`: Front calculation utilities
- `assign_crowding_distance`: Diversity preservation

## Symbolic Processing
- `TokenLib`, `TokenDto`, `LibEntry`, `TempComputeTree`: Symbolic computation structures
- `create_lib`, `create_compute_tree`: Library and tree creation utilities
- `propagate_necessary_changes!`, `correct_genes!`: Tree manipulation functions

## Utility Functions
- `find_indices_with_sum`, `compile_djl_datatype`: Data processing utilities
- `optimize_constants!`: Constants optimization
- `isclose`: Numerical comparison
- `save_state`, `load_state`: State persistence
- `train_test_split`: Data splitting utility

## Dimensional Analysis
- `physical_constants`, `physical_constants_all`: Physical constants databases
- `get_constant`, `get_constant_value`, `get_constant_dims`: Constants retrieval
- Unit operation functions:
  - Forward propagation: `equal_unit_forward`, `mul_unit_forward`, `div_unit_forward`
  - Backward propagation: `zero_unit_backward`, `mul_unit_backward`, `div_unit_backward`
  - Special operations: `sqr_unit_forward`, `sqr_unit_backward`

## History Recording
- `HistoryRecorder`, `OptimizationHistory`: History tracking structures
- `record_history!`, `record!`: Recording functions
- `close_recorder!`, `get_history_arrays`: History management

# Example Usage
```julia
using GeneExpressionProgramming

# Create a GEP regressor
regressor = GepRegressor(feature_count)

# Fit the model
fit!(regressor, population_size, epochs,X_train, y_train)

# Access physical constants
constant = get_constant("speed_of_light")
value = get_constant_value(constant)
dims = get_constant_dims(constant)
```

# Notes
- The module integrates dimensional analysis with symbolic regression
- Supports multi-objective optimization through NSGA-II
- Provides both low-level GEP operations and high-level regression interface
- Includes comprehensive utilities for tree manipulation and unit consistency
"""

include("Util.jl")
include("TensorOps.jl")
include("Entities.jl")
include("Losses.jl")
include("Selection.jl")
include("Sbp.jl")
# First export the submodules themselves
export GepUtils, TensorRegUtils, GepEntities, LossFunction, EvoSelection, SBPUtils

include("Gep.jl")
include("PhyConstants.jl")
include("RegressionWrapper.jl")
export PhysicalConstants, GepRegression, RegressionWrapper

# Import core functions for tensor regression
import .TensorRegUtils:
    InputSelector,
    AdditionNode, SubtractionNode, MultiplicationNode, DivisionNode, PowerNode,
    MinNode, MaxNode, InversionNode,
    TraceNode, DeterminantNode, SymmetricNode, SkewNode,
    VolumetricNode, DeviatricNode, TdotNode, DottNode,
    DoubleContractionNode, DeviatoricNode,
    ConstantNode, UnaryNode,
    compile_to_flux_network,
    TENSOR_NODES, TENSOR_NODES_ARITY, TENSOR_STRINGIFY

# Import GEP core functionality
import .GepRegression:
    runGep

# Import loss functions
import .LossFunction:
    get_loss_function

# Import utilities
import .GepUtils:
    find_indices_with_sum,
    compile_djl_datatype,
    optimize_constants!,
    minmax_scale,
    isclose,
    save_state,
    load_state,
    record_history!,
    record!,
    close_recorder!,
    HistoryRecorder,
    OptimizationHistory,
    get_history_arrays,
    train_test_split,
    ARITY_LIB_COMMON,
    FUNCTION_LIB_COMMON,
    FUNCTION_STRINGIFY,
    one_hot_mean,
    select_n_samples_lhs

# Import selection mechanisms
import .EvoSelection:
    tournament_selection,
    nsga_selection,
    dominates_,
    fast_non_dominated_sort,
    calculate_fronts,
    determine_ranks,
    assign_crowding_distance

# Import physical constants functionality
import .PhysicalConstants:
    physical_constants,
    physical_constants_all,
    get_constant,
    get_constant_value,
    get_constant_dims

# Import symbolic computation utilities
import .SBPUtils:
    TokenLib,
    TokenDto,
    LibEntry,
    TempComputeTree,
    create_lib,
    create_compute_tree,
    propagate_necessary_changes!,
    calculate_vector_dimension!,
    flush!,
    flatten_dependents,
    correct_genes!,
    equal_unit_forward,
    mul_unit_forward,
    div_unit_forward,
    zero_unit_backward,
    zero_unit_forward,
    sqr_unit_backward,
    sqr_unit_forward,
    mul_unit_backward,
    div_unit_backward,
    equal_unit_backward,
    get_feature_dims_json,
    get_target_dim_json,
    retrieve_coeffs_based_on_similarity

# Import core GEP entities
import .GepEntities:
    Chromosome,
    Toolbox,
    EvaluationStrategy,
    StandardRegressionStrategy,
    GenericRegressionStrategy,
    fitness,
    set_fitness!,
    generate_gene,
    compile_expression!,
    generate_chromosome,
    generate_population,
    genetic_operations!,
    split_karva,
    print_karva_strings

# Import regression wrapper functionality
import .RegressionWrapper:
    GepRegressor,
    GepTensorRegressor,
    fit!,
    list_all_functions,
    list_all_arity,
    list_all_forward_handlers,
    list_all_backward_handlers,
    list_all_genetic_params,
    set_function!,
    set_arity!,
    set_forward_handler!,
    set_backward_handler!,
    update_function!,
    create_physical_operations,
    create_function_entries,
    create_constants_entries,
    create_feature_entries


# Export GEP core functionality
export runGep, EvaluationStrategy, StandardRegressionStrategy, GenericRegressionStrategy

# Export structures for the tensor regression
export InputSelector,
    AdditionNode, SubtractionNode, MultiplicationNode, DivisionNode, PowerNode,
    MinNode, MaxNode, InversionNode,
    TraceNode, DeterminantNode, SymmetricNode, SkewNode,
    VolumetricNode, DeviatricNode, TdotNode, DottNode,
    DoubleContractionNode, DeviatoricNode,
    ConstantNode, UnaryNode,
    compile_to_flux_network,
    TENSOR_NODES, TENSOR_NODES_ARITY, TENSOR_STRINGIFY


# Export core GEP entities and operations
export Chromosome, Toolbox, fitness, set_fitness!,
    generate_gene, compile_expression!, generate_chromosome, generate_population,
    genetic_operations!, split_karva, print_karva_strings

# Export regression components
export GepRegressor, GepTensorRegressor, fit!,
    list_all_functions, list_all_arity, list_all_forward_handlers,
    list_all_backward_handlers, list_all_genetic_params,
    set_function!, set_arity!, set_forward_handler!, set_backward_handler!,
    update_function!, create_physical_operations, create_function_entries, create_constants_entries,
    create_feature_entries

# Export loss functions
export get_loss_function

# Export selection mechanisms
export tournament_selection, nsga_selection, dominates_,
    fast_non_dominated_sort, calculate_fronts,
    determine_ranks, assign_crowding_distance

# Export physical constants functionality
export physical_constants, physical_constants_all,
    get_constant, get_constant_value, get_constant_dims

# Export symbolic computation types
export TokenLib, TokenDto, LibEntry, TempComputeTree

# Export symbolic computation utilities
export create_lib, create_compute_tree,
    propagate_necessary_changes!, calculate_vector_dimension!,
    flush!, flatten_dependents, correct_genes!,
    get_feature_dims_json, get_target_dim_json,
    retrieve_coeffs_based_on_similarity

# Export unit handling operations
export equal_unit_forward, mul_unit_forward, div_unit_forward,
    zero_unit_backward, zero_unit_forward,
    sqr_unit_backward, sqr_unit_forward,
    mul_unit_backward, div_unit_backward, equal_unit_backward

# Export general utilities
export find_indices_with_sum, compile_djl_datatype,
    optimize_constants!, minmax_scale, isclose,
    save_state, load_state,
    train_test_split, one_hot_mean, select_n_samples_lhs

# Export history recording functionality
export HistoryRecorder, OptimizationHistory,
    record_history!, record!, close_recorder!,
    get_history_arrays

# Export common libraries
export ARITY_LIB_COMMON, FUNCTION_LIB_COMMON, FUNCTION_STRINGIFY


end
