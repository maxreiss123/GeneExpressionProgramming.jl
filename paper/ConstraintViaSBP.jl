using VGeneExpressionProgramming
using DynamicExpressions
using OrderedCollections
using CSV 
using DataFrames
using FileIO
using Random
using Logging
using Dates
using JSON


function sqr(x::Vector{T}) where T<:AbstractFloat
    return x .* x
end

function sqr(x::T) where T<:Union{AbstractFloat, Node{<:AbstractFloat}}
    return x * x
end

function create_symbol_config(
    features_names::Vector{String},
    constants::Vector{T};
    feature_dims::Dict{String, Vector{Float16}} = Dict{String,Vector{Float16}}()
) where T <: AbstractFloat
    operators_djl =  OperatorEnum(; binary_operators=[*,/,+,-], unary_operators=[sqr,sqrt,sin,cos,exp,log])
    nodes_djl = OrderedDict{Int8, Any}()

    point_operations_idx = Int8[1, 2]  # correspond to the indices of the first two operations
    
    basic_symbols = OrderedDict{Int8, BasicSymbol}()
    functional_symbols = OrderedDict{Int8, FunctionalSymbol}()
    constant_symbols = OrderedDict{Int8, BasicSymbol}()

    std_phy_dim = zeros(Float16, 7)

    functional_symbols[1] = FunctionalSymbol("*", std_phy_dim, 1, 2, (*), mul_unit_forward,mul_unit_backward )
    functional_symbols[2] = FunctionalSymbol("/", std_phy_dim, 2, 2, (/), div_unit_forward, div_unit_backward)
    functional_symbols[3] = FunctionalSymbol("+", std_phy_dim, 3, 2, (+), equal_unit_forward, equal_unit_backward)
    functional_symbols[4] = FunctionalSymbol("-", std_phy_dim, 4, 2, (-), equal_unit_forward, equal_unit_backward)
    functional_symbols[5] = FunctionalSymbol("sqr", std_phy_dim, 5, 1, sqr, sqr_unit_forward, sqr_unit_backward )
    functional_symbols[6] = FunctionalSymbol("sqrt", std_phy_dim, 6, 1, sqrt, sqr_unit_backward,sqr_unit_forward )
    functional_symbols[7] = FunctionalSymbol("exp", std_phy_dim, 7, 1, exp, zero_unit_forward, zero_unit_backward)
    functional_symbols[8] = FunctionalSymbol("log", std_phy_dim, 8, 1, log, zero_unit_forward, zero_unit_backward)
    functional_symbols[9] = FunctionalSymbol("sin", std_phy_dim, 9, 1, sin, zero_unit_forward, zero_unit_backward)
    functional_symbols[10] = FunctionalSymbol("cos", std_phy_dim, 10, 1, cos, zero_unit_forward, zero_unit_backward)


    offset = length(functional_symbols)
    for (index, feature_name) in enumerate(features_names)
        phy_dim = get(feature_dims, feature_name, std_phy_dim)
        basic_symbols[index+offset] = BasicSymbol(feature_name, phy_dim, index+offset, 0, true)
        nodes_djl[index+offset] = Node{Float64}(; feature=index)
    end

    offset += length(basic_symbols)

    for (index, elem) in enumerate(constants)
        key = string(elem)
        constant_symbols[index+offset] = BasicSymbol(key, std_phy_dim, index+offset, 0, false)
        nodes_djl[index+offset] = parse(Float64, key)
    end

    callbacks = Dict{Int8,Function}(symbol.index => symbol.arithmetic_operation for (_, symbol) in functional_symbols)
    symbol_arity_mapping = OrderedDict(symbol.index => symbol.arity for (_, symbol) in merge(functional_symbols, basic_symbols, constant_symbols))
    physical_operation_dict = OrderedDict(symbol.index => symbol.forward_function for (_, symbol) in functional_symbols)
    physical_dimension_dict = OrderedDict(symbol.index => symbol.unit for (_, symbol) in merge(functional_symbols, basic_symbols, constant_symbols))

    features_idx = [symbol.index for (_, symbol) in basic_symbols]
    functions_idx = [symbol.index for (_, symbol) in functional_symbols]
    constants_idx = [symbol.index for (_, symbol) in constant_symbols]    
    inverse_operations = Dict(symbol.index => symbol.reverse_function for (_, symbol) in functional_symbols if !isnothing(symbol.reverse_function))

    return SymbolConfig(
        basic_symbols,
        constant_symbols,
        functional_symbols,
        callbacks,
        operators_djl,
        nodes_djl,
        symbol_arity_mapping,
        physical_operation_dict,
        physical_dimension_dict,
        features_idx,
        functions_idx,
        constants_idx,
        point_operations_idx,
        inverse_operations
    )
end

function setup_logger(log_file_path::String)
    mkpath(dirname(log_file_path))
    logger = SimpleLogger(open(log_file_path, "a"))
    global_logger(logger)
end

function log_error(error_message::String, exception::Exception)
    @error "$(now()) - $error_message" exception=exception stack=catch_backtrace()
end

function read_all_csvs(folder_path::String)

    csv_files = filter(f->endswith(lowercase(f), ".txt"), readdir(folder_path))
    csv_files = reverse(csv_files)
    framesDict = OrderedDict{String, Matrix}()

    for file in csv_files
        file_path = joinpath(folder_path, file)
        @show file_path
        df = CSV.read(file_path, DataFrame, header=true)
        key = split(file)[1]
        framesDict[key] = Matrix(df)
    end
    return framesDict
end

function save_results_to_csv(file_name::String, results::DataFrame)
    if isfile(file_name)
        CSV.write(file_name, results, append=true, header=false)
    else
        CSV.write(file_name, results)
    end
end


function get_or_create_test_data(test_data_dict, equation_name, x_data_test, y_data_test)
    if !haskey(test_data_dict, equation_name)
        test_data_dict[equation_name] = (x_data_test, y_data_test)
    end
    return test_data_dict[equation_name]
end

function main()
    framesDict_ = read_all_csvs("./test/srsd")
    case_data = JSON.parsefile("./assets/case_dsc.json")
    log_path = "error.log"
    setup_logger(log_path)


    file_name_save = "test_gep_on_srsd_p10.csv"
    penalty_consideration = 0
    cycles = 1
    cost_function = "mse"

    gep_probs = Dict{String, AbstractFloat}(
    "one_point_cross_over_prob" => 0.6,
    "two_point_cross_over_prob" => 0.5,
    "mutation_prob" => 1,
    "mutation_rate" => 0.25,
    "dominant_fusion_prob" => 0.1,
    "dominant_fusion_rate" => 0.2,
    "rezessiv_fusion_prob" => 0.1,
    "rezessiv_fusion_rate" => 0.2,
    "fusion_prob" => 0.1,
    "fusion_rate" => 0.2,
    "inversion_prob" => 0.1
    )
    

    for seed in 1:10
        test_data_dict = Dict{String, Tuple{Matrix{Float64}, Vector{Float64}}}()
        for (name, data) in framesDict_
                case_identifier_name = uppercase(split(name,"-")[2])
                case_name = string(split(case_identifier_name,"\$")[1])
                noise_level = string(split(case_identifier_name,"\$")[2][1:end-4])


                if case_name in keys(case_data)
                    @show ("Current case: ", case_name)
                    #gep_params
                    pop_size = 1500
                    generations = 1500
                    gene_count = 3
                    head_length = 6 
                    consider = 1

                    results = DataFrame(Seed=[],
                        Name = String[], NoiseLeve=String[], Fitness = Float64[], Equation = String[], R2_test = Float64[], 
                        R2_train = Float64[], Runtime = Float64[], LibCreattime = Float64[], Dimensional_Homogeneity = Bool[], Target = Any[])
                    
                    Random.seed!(seed)
                    num_cols = size(data,2)
                    connection_syms = Int8[1,2,3,4]
                    feature_names = ["x"*string(i) for i in 1:num_cols-1]
                    constants = [0.0, 0.5]

                    println(feature_names)
                    println(case_name)
                    phy_dims = get_feature_dims_json(case_data, feature_names, case_name)
                    target_dim = get_target_dim_json(case_data, case_name)

                    #based on the name - dims are obtained from a dictionary
                    config = create_symbol_config(
                        feature_names,constants;feature_dims=phy_dims
                    )

                    #create a library
                    token_lib = TokenLib(config.physical_dimension_dict, 
                                        config.physical_operation_dict, 
                                        config.symbol_arity_mapping)

                    start_time = time_ns()
                    lib = create_lib(token_lib, 
                                    config.features_idx, 
                                    config.functions_idx, 
                                    config.constants_idx; 
                                    rounds=head_length-2, max_permutations=100000)
                    total_len_lib = sum(length(entry) for entry in values(lib))
                    @show ("Lib Entries:" , total_len_lib)
                    end_time_lib = (time_ns() - start_time)/1e9
                    token_dto = TokenDto(token_lib, config.point_operations_idx, lib, config.inverse_operations, gene_count; head_len=head_length-2)



                    function corr_call_back!(genes::Vector{Int8}, start_indices::Vector{Int}, expression::Vector{Int8})
                        return correct_genes!(genes, start_indices, expression, 
                        convert.(Float16,target_dim), token_dto; cycles=cycles) 
                    end


                    utilized_syms  = merge!(
                        OrderedDict(key => symbol.arity for (key, symbol) in config.functional_symbols),
                        OrderedDict(key => symbol.arity for (key, symbol) in config.basic_symbols),
                        OrderedDict(key => symbol.arity for (key, symbol) in config.constant_symbols)
                    )
                    data = data[shuffle(1:size(data, 1)), :]
                    #data = minmax_scale(data,feature_range=(-1.0, 1.0))
                    data_train = data[1:floor(Int, size(data,1) * 0.75),:]
                    data_test = data[floor(Int, size(data,1) * 0.75):end,:]
		            x_data = Float64.(data_train[1:consider:end ,1:(num_cols-1)])
                    y_data = Float64.(data_train[1:consider:end, num_cols])

                    x_data_test, y_data_test = get_or_create_test_data(
                        test_data_dict,
                        case_name,
                        Float64.(data_test[1:consider:end, 1:(num_cols-1)]),
                        Float64.(data_test[1:consider:end, num_cols])
                    )

                    start_time = time_ns()
                    best,_=runGep(pop_size, generations,
                                gene_count,head_length,utilized_syms, config.operators_djl, 
                                config.callbacks, 
                                config.nodes_djl, 
                                x_data',y_data, connection_syms, gep_probs; correction_callback=corr_call_back!, 
                                loss_fun_=cost_function, 
                                penalty_consideration=penalty_consideration, x_data_test=x_data_test', 
                                y_data_test=y_data_test, hof=1)
                    
                    end_time = (time_ns() - start_time)/1e9
                    
                    #log_results
                    for elem in best
                    push!(results, (seed, case_name, noise_level, elem.fitness, string(elem.compiled_function), 
                        elem.fitness_r2_train, elem.fitness_r2_test, end_time, end_time_lib, elem.dimension_homogene, target_dim))
                    end
		    @show best[1].fitness_r2_test
                    save_results_to_csv(file_name_save,results)
                end
            end
    end
    close(global_logger().stream)
end

main()
