module VGeneExpressionProgramming

include("Entities.jl")
include("Gep.jl")
include("Losses.jl")
include("PhyConstants.jl")
include("Sbp.jl")
include("Selection.jl")
include("Util.jl")


using .GepRegressor
export Chromosome
export runGep


using .LossFunction
export get_loss_function


using .GepUtils
export find_indices_with_sum, compile_djl_datatype, optimize_constants!, minmax_scale, float16_scale, isclose
export save_state, load_state


using .EvoSelection
export selection_NSGA, basic_tournament_selection, dominates_, fast_non_dominated_sort, calculate_fronts, determine_ranks, assign_crowding_distance


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


end
