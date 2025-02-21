include("../src/GeneExpressionProgramming.jl")
using .GeneExpressionProgramming
using DynamicExpressions
using OrderedCollections
using CSV
using DataFrames
using FileIO
using Random
using Logging
using Dates
using JSON
using Statistics


function break_condition(population, epoch)
    return isclose(mean(population[1].fitness), 0.0)
end

function loss_new(eqn::Node, operators::OperatorEnum, x_data::AbstractArray, y_data::AbstractArray)
    try
        y_pred = eqn(x_data, operators)
        return get_loss_function("r2_score")(y_data, y_pred)
    catch e
        return zero(Float64)
    end
end


function setup_logger(log_file_path::String)
    mkpath(dirname(log_file_path))
    logger = SimpleLogger(open(log_file_path, "a"))
    global_logger(logger)
end

function log_error(error_message::String, exception::Exception)
    @error "$(now()) - $error_message" exception = exception stack = catch_backtrace()
end

function read_all_csvs(folder_path::String)

    csv_files = filter(f -> endswith(lowercase(f), ".txt"), readdir(folder_path))
    csv_files = reverse(csv_files)
    framesDict = OrderedDict{String,Matrix}()

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
    framesDict_ = read_all_csvs("./paper/srsd")
    case_data = JSON.parsefile("./assets/case_dsc.json")
    log_path = "error.log"
    setup_logger(log_path)

    file_name_save = "test_gep_on_srsd_p.csv"


    for seed in 1:10
        test_data_dict = Dict{String,Tuple{Matrix{Float64},Vector{Float64}}}()
        for (name, data) in framesDict_
            case_identifier_name = uppercase(split(name, "-")[2])
            case_name = string(split(case_identifier_name, "\$")[1])
            noise_level = string(split(case_identifier_name, "\$")[2][1:end-4])


            if case_name in keys(case_data)
                @show ("Current case: ", case_name)
                #gep_params
                epochs = 1000
                population_size = 1500

                results = DataFrame(Seed=[],
                    Name=String[], NoiseLeve=String[], Fitness=Float64[], Equation=String[], R2_test=Float64[],
                    R2_train=Float64[], Runtime=Float64[], Dimensional_Homogeneity=Bool[], Target=Any[])

                Random.seed!(seed)
                num_cols = size(data, 2)
                feature_names = ["x$i" for i in 1:num_cols-1]


                println(feature_names)
                println(case_name)
                phy_dims = get_feature_dims_json(case_data, feature_names, case_name)
                phy_dims = Dict{Symbol,Vector{Float16}}(Symbol(x_n) => dim_n for (x_n, dim_n) in phy_dims)
                target_dim = get_target_dim_json(case_data, case_name)

                print(phy_dims)

                x_train, y_train, x_test, y_test = train_test_split(data[:, 1:num_cols-1], data[:, num_cols]; consider=4)

                x_test, y_test = get_or_create_test_data(
                    test_data_dict,
                    case_name,
                    x_test,
                    y_test
                )

                start_time = time_ns()

                regressor = GepRegressor(num_cols - 1;
                    considered_dimensions=phy_dims,
                    entered_non_terminals=[:+, :-, :*, :/, :sqrt, :sin, :cos, :exp, :log],
                    max_permutations_lib=10000, rounds=7, number_of_objectives=1)


                @inline function loss_new_(elem, validate::Bool)
                    try
                        if isnan(mean(elem.fitness)) || validate
                            y_pred = elem.compiled_function(x_train', regressor.operators_)
                            return (get_loss_function("mse")(y_train, y_pred),)
                        else
                            return (elem.fitness, length(elem.expression_raw) * elem.fitness)
                            #return (elem.fitness,)
                        end
                    catch e
                        return (typemax(Float64),typemax(Float64))
                    end
                end


                #perform the regression by entering epochs, population_size, the feature cols, the target col and the loss function
                fit!(regressor, epochs, population_size, x_train', y_train;
                    x_test=x_test', y_test=y_test', target_dimension=target_dim,
                    loss_fun="mse", break_condition=break_condition)

                end_time = (time_ns() - start_time) / 1e9
                elem = regressor.best_models_[1]
                fitness_r2_train = loss_new(elem.compiled_function, regressor.operators_, x_train', y_train)
                fitness_r2_test = loss_new(elem.compiled_function, regressor.operators_, x_test', y_test)

                #log_results
                push!(results, (seed, case_name, noise_level, mean(elem.fitness), string(elem.compiled_function),
                    fitness_r2_train, fitness_r2_test, end_time, elem.dimension_homogene, target_dim))

                @show fitness_r2_test
                save_results_to_csv(file_name_save, results)
            end
        end
    end
    close(global_logger().stream)
end

main()
