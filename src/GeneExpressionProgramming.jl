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
- `minmax_scale`, `float16_scale`: Scaling functions
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
- `create_history_recorder`, `record_history!`, `record!`: Recording functions
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

include("Entities.jl")
include("Gep.jl")
include("Losses.jl")
include("PhyConstants.jl")
include("Sbp.jl")
include("Selection.jl")
include("Util.jl")
include("RegressionWrapper.jl")

export GepEntities, LossFunction, PhysicalConstants, GepUtils, GepRegression, RegressionWrapper

using .GepRegression
export runGep, EvaluationStrategy, StandardRegressionStrategy, GenericRegressionStrategy

using .LossFunction
export get_loss_function


using .GepUtils
export find_indices_with_sum, compile_djl_datatype, optimize_constants!, minmax_scale, float16_scale, isclose
export save_state, load_state, create_history_recorder, record_history!, record!, close_recorder!
export HistoryRecorder, OptimizationHistory, get_history_arrays
export train_test_split

using .EvoSelection
export tournament_selection, nsga_selection, dominates_, fast_non_dominated_sort, calculate_fronts, determine_ranks, assign_crowding_distance


using .PhysicalConstants
export physical_constants, physical_constants_all
export get_constant, get_constant_value, get_constant_dims


using .SBPUtils
export TokenLib, TokenDto, LibEntry, TempComputeTree
export create_lib, create_compute_tree, propagate_necessary_changes!, calculate_vector_dimension!, flush!, calculate_vector_dimension!, flatten_dependents
export propagate_necessary_changes!, correct_genes!
export equal_unit_forward, mul_unit_forward, div_unit_forward, zero_unit_backward, zero_unit_forward, sqr_unit_backward, sqr_unit_forward, mul_unit_backward, div_unit_backward, equal_unit_backward
export get_feature_dims_json, get_target_dim_json, retrieve_coeffs_based_on_similarity


using .GepEntities
export Chromosome, Toolbox
export AbstractSymbol, FunctionalSymbol, BasicSymbol, SymbolConfig
export fitness, set_fitness!
export generate_gene, generate_preamle!, compile_expression!, generate_chromosome, generate_population 
export genetic_operations!


using .RegressionWrapper
export GepRegressor, fit!
export list_all_functions, list_all_arity, list_all_forward_handlers, 
       list_all_backward_handlers, list_all_genetic_params,
       set_function!, set_arity!, set_forward_handler!, set_backward_handler!,
       update_function!

end
