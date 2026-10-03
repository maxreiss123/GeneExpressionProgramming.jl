"""
    GeneExpressionProgramming

Symbolic regression by Gene Expression Programming (GEP). A model is a chromosome: genes,
each a head and a tail, joined by connector operators. Its karva string, the resolved
token vector, is evaluated on data columns by a batched evaluator, a stack machine with
preallocated per-thread buffers.

The entry points are `GepRegressor`, for scalar models, and `GepTensorRegressor`, for
models over vectors and tensors (Tensors.jl); both are trained with `fit!`. Given the
physical dimensions of the features and a target dimension, semantic backpropagation
(SBP) repairs candidates towards the target, and only homogeneous ones are scored.

```julia
using GeneExpressionProgramming

x = randn(200, 2)                                   # one row per sample
y = @. x[:, 1]^2 + x[:, 1] * x[:, 2]
x_train, y_train, x_test, y_test = train_test_split(x, y)

regressor = GepRegressor(2)
fit!(regressor, 100, 500, x_train', y_train;        # epochs, population size, and the
     x_test=x_test', y_test=y_test)                 # data with one row per feature
println(regressor.best_models_[1])                  # the best model as an equation
y_pred = regressor(x_test')
```

For an expensive loss, a `SurrogateScreening` passed to `fit!` as `surrogate` lets the
loss score only a few promising individuals per epoch and a Gaussian process predict the
loss of the others, and `optimize_constants!` tunes the constants of a model against such
a loss by Nelder-Mead, screened by a Gaussian process (`ScreenedNelderMead`).

Submodules: `GepEntities` (chromosome, toolbox, genetic operators), `GepRegression` (the
evolutionary loop, `runGep`), `TensorRegUtils` (batched evaluator), `SBPUtils` (dimensions
and SBP), `GepSurrogate` (surrogate screening of expensive losses), `GepSimplex`
(Nelder-Mead for the constants of a model, screened or not), `RegressionWrapper`
(regressors), `LossFunction`, `EvoSelection`, `GepUtils`, `PhysicalConstants`.
"""
module GeneExpressionProgramming

include("Util.jl")
include("TensorOps.jl")
include("Entities.jl")
include("Losses.jl")
include("Selection.jl")
include("Sbp.jl")
include("Surrogate.jl")
include("Simplex.jl")
# Export the submodules
export GepUtils, TensorRegUtils, GepEntities, LossFunction, EvoSelection, SBPUtils, GepSurrogate,
    GepSimplex

include("Gep.jl")
include("PhyConstants.jl")
include("RegressionWrapper.jl")
export PhysicalConstants, GepRegression, RegressionWrapper

# Import the batched evaluator and its operator nodes
import .TensorRegUtils:
    InputSelector,
    AdditionNode, SubtractionNode, MultiplicationNode, DivisionNode, PowerNode,
    MinNode, MaxNode, InversionNode,
    TraceNode, DeterminantNode, SymmetricNode, SkewNode,
    VolumetricNode, TdotNode, DottNode,
    DoubleContractionNode, DeviatoricNode,
    ConstantNode, UnaryNode,
    CrossProductNode, LapNode, OuterProductNode, HadamardNode, SqrtNode, NormNode,
    DotProductNode,
    calc_stack_batch_tensor, eval_op!, return_type,
    EvalProgram, compile_program, run_program!,
    TENSOR_NODES, TENSOR_NODES_ARITY, TENSOR_STRINGIFY

# Import the evolutionary loop and gene-wise linear scaling
import .GepRegression:
    runGep,
    gene_basis,
    solve_scaling

# Import loss functions
import .LossFunction:
    get_loss_function

# Import utilities
import .GepUtils:
    find_indices_with_sum,
    compile_djl_datatype,
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
    select_n_samples_lhs,
    split_rng,
    thread_slots,
    allfinite

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

# Import dimensions, unit rules and semantic backpropagation (SBP)
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
    retrieve_coeffs_based_on_similarity,
    dimensional_homogeneity_distance,
    is_dimensionally_homogeneous,
    gene_dimensions,
    is_gene_wise_homogeneous,
    sign_unit_forward,
    sign_unit_backward,
    sample_lib_expression

# Import the surrogate screening of expensive losses
import .GepSurrogate:
    SurrogateScreening,
    GpScreen,
    GaussianProcess,
    FeasibilityModel,
    SemanticEmbedder,
    GeneEmbedder,
    TensorEmbedder,
    archive_size,
    is_validated

# Import Nelder-Mead for the constants of a model with an expensive loss
import .GepSimplex:
    ScreenedNelderMead,
    SimplexSearch,
    simplex_search,
    optimize_constants!

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
    gene_averaging!,
    split_karva,
    split_predict,
    split_equations,
    split_positions,
    print_karva_strings,
    equation_string,
    constant_positions,
    buffer_context,
    thread_contexts

# Import the regressors and the function-library helpers
import .RegressionWrapper:
    GepRegressor,
    GepTensorRegressor,
    fit!,
    allocate_buffers!,
    predictT,
    predictT_scaled,
    predictT_scaled!,
    gene_bases,
    build_buffers,
    BUFFERED_EVAL_MIN_SAMPLES,
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

# Export the batched evaluator and its operator nodes
export InputSelector,
    AdditionNode, SubtractionNode, MultiplicationNode, DivisionNode, PowerNode,
    MinNode, MaxNode, InversionNode,
    TraceNode, DeterminantNode, SymmetricNode, SkewNode,
    VolumetricNode, TdotNode, DottNode,
    DoubleContractionNode, DeviatoricNode,
    ConstantNode, UnaryNode,
    CrossProductNode, LapNode, OuterProductNode, HadamardNode, SqrtNode, NormNode,
    DotProductNode,
    calc_stack_batch_tensor, eval_op!, return_type,
    EvalProgram, compile_program, run_program!,
    TENSOR_NODES, TENSOR_NODES_ARITY, TENSOR_STRINGIFY


# Export core GEP entities and operations
export Chromosome, Toolbox, fitness, set_fitness!,
    generate_gene, compile_expression!, generate_chromosome, generate_population,
    genetic_operations!, gene_averaging!, split_karva, split_predict, split_equations,
    split_positions,
    print_karva_strings, equation_string, constant_positions, buffer_context,
    thread_contexts

# Export regression components
export GepRegressor, GepTensorRegressor, fit!, allocate_buffers!, predictT,
    predictT_scaled, predictT_scaled!, gene_bases, build_buffers, BUFFERED_EVAL_MIN_SAMPLES,
    gene_basis, solve_scaling,
    list_all_functions, list_all_arity, list_all_forward_handlers,
    list_all_backward_handlers, list_all_genetic_params,
    set_function!, set_arity!, set_forward_handler!, set_backward_handler!,
    update_function!, create_physical_operations, create_function_entries, create_constants_entries,
    create_feature_entries

# Export loss functions
export get_loss_function

# Export the surrogate screening
export SurrogateScreening, GpScreen, GaussianProcess, FeasibilityModel,
    SemanticEmbedder, GeneEmbedder, TensorEmbedder,
    archive_size, is_validated

# Export Nelder-Mead for the constants of a model
export ScreenedNelderMead, SimplexSearch, simplex_search, optimize_constants!

# Export selection mechanisms
export tournament_selection, nsga_selection, dominates_,
    fast_non_dominated_sort, calculate_fronts,
    determine_ranks, assign_crowding_distance

# Export physical constants functionality
export physical_constants, physical_constants_all,
    get_constant, get_constant_value, get_constant_dims

# Export the SBP data structures
export TokenLib, TokenDto, LibEntry, TempComputeTree

# Export the library build, the repair and the homogeneity checks
export create_lib, create_compute_tree,
    propagate_necessary_changes!, calculate_vector_dimension!,
    flush!, flatten_dependents, correct_genes!,
    dimensional_homogeneity_distance, is_dimensionally_homogeneous,
    gene_dimensions, is_gene_wise_homogeneous,
    sample_lib_expression,
    get_feature_dims_json, get_target_dim_json,
    retrieve_coeffs_based_on_similarity

# Export the unit rules
export equal_unit_forward, mul_unit_forward, div_unit_forward,
    zero_unit_backward, zero_unit_forward,
    sqr_unit_backward, sqr_unit_forward,
    sign_unit_forward, sign_unit_backward,
    mul_unit_backward, div_unit_backward, equal_unit_backward

# Export general utilities
export find_indices_with_sum, compile_djl_datatype,
    minmax_scale, isclose,
    save_state, load_state,
    train_test_split, one_hot_mean, select_n_samples_lhs, split_rng, thread_slots,
    allfinite

# Export history recording functionality
export HistoryRecorder, OptimizationHistory,
    record_history!, record!, close_recorder!,
    get_history_arrays

# Export the function library
export ARITY_LIB_COMMON, FUNCTION_LIB_COMMON, FUNCTION_STRINGIFY


end
