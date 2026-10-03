#=
The Feynman experiments of "Constraining genetic symbolic regression via semantic
backpropagation" (Reissmann et al., GPEM 2025), run against the current src/.

    julia --project=. --threads=4 paper/ConstraintViaSBP.jl [seeds] [epochs] [population]

Defaults are the paper's budget: 100 seeds, 1000 epochs, population 1000. A quick check
of the whole pipeline:

    julia --project=. --threads=4 paper/ConstraintViaSBP.jl 1 50 200

Every file in `paper/srsd` whose equation has an entry in `assets/case_dsc.json` (feature
and target dimensions) is fitted once per seed by a `GepRegressor` with three genes of
head length 6 and the target dimension set: the search repairs individuals by semantic
backpropagation (SBP) and scores only homogeneous ones. The fitness is the square root of
the training RMSE; there is no constant optimisation (the loss-callback form of `fit!`
has no target to tune against). `train_test_split(...; consider=10)` keeps every tenth
row of a shuffled 90/10 split. Files are read in reverse name order, so an equation's
noise-free file comes first, and its test rows are reused for the noisy variants.

One row per fit is appended to `paper/results/test_gep_on_srsd_p_full_scale.csv`; the log
goes to `paper/results/error.log`. `paper/srsd` ships two equations (III.19.51 and
III.21.20) at noise levels 0, 0.001, 0.01 and 0.1; further SRSD-Feynman files go there,
named `feynman-<equation>$<noise>.txt`.
=#
include(joinpath(@__DIR__, "..", "src", "GeneExpressionProgramming.jl"))
using .GeneExpressionProgramming
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
    return isclose(mean(population[1].fitness[1]), 0.0)
end

# R^2 of a fitted model on (x_data, y_data), 0 when it cannot be evaluated. Calling the
# chromosome on data builds an evaluation context per call, which is fine outside the
# search loop.
function loss_new(elem, x_data::AbstractArray, y_data::AbstractArray)
    try
        y_pred = elem(x_data)
        y_pred isa AbstractVector || return zero(Float64)
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

function main(; seeds::Int=100, epochs::Int=1000, population_size::Int=1000)
    root = joinpath(@__DIR__, "..")
    framesDict_ = read_all_csvs(joinpath(root, "paper", "srsd"))
    case_data = JSON.parsefile(joinpath(root, "assets", "case_dsc.json"))
    results_dir = joinpath(root, "paper", "results")
    setup_logger(joinpath(results_dir, "error.log"))

    file_name_save = joinpath(results_dir, "test_gep_on_srsd_p_full_scale.csv")


    for seed in 1:seeds
        test_data_dict = Dict{String,Tuple{Matrix{Float64},Vector{Float64}}}()
        for (name, data) in framesDict_
            case_identifier_name = uppercase(split(name, "-")[2])
            case_name = string(split(case_identifier_name, "\$")[1])
            noise_level = string(split(case_identifier_name, "\$")[2][1:end-4])


            if case_name in keys(case_data)
                @show ("Current case: ", case_name)
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

                x_train, y_train, x_test, y_test = train_test_split(data[:, 1:num_cols-1], data[:, num_cols]; consider=10)

                x_test, y_test = get_or_create_test_data(
                    test_data_dict,
                    case_name,
                    x_test,
                    y_test
                )

                start_time = time_ns()

                regressor = GepRegressor(num_cols - 1;
                    considered_dimensions=phy_dims, gene_count=3, head_len=6,
                    entered_non_terminals=[:+, :-, :*, :/, :sqrt, :sin, :cos, :exp, :log],
                    max_permutations_lib=20000, rounds=5, number_of_objectives=1)

                # one evaluation context per thread slot, allocated once: the loss runs
                # inside the threaded fitness loop, and per-call context creation
                # would allocate the whole buffer pool for every candidate
                eval_ctxs = thread_contexts(regressor.toolbox_, x_train')

                @inline function loss_new_(elem, validate::Bool)
                    try
                        # with a target dimension the search only hands homogeneous
                        # individuals to the loss; the flag test here is kept as the
                        # paper's formulation of the contract
                        if isnan(mean(elem.fitness)) && elem.dimension_homogene || validate
                            y_pred = elem(eval_ctxs[Threads.threadid()])
                            if y_pred isa AbstractVector
                                # sqrt of the RMSE, i.e. MSE^(1/4): ranks as the MSE does
                                fit = sqrt(get_loss_function("rmse")(y_train, y_pred))
                                elem.fitness = (fit,)
                            else
                                elem.fitness = (typemax(Float64),)
                            end
                        end
                    catch e
                        elem.fitness = (typemax(Float64),)
                    end
                end


                # loss-callback form of fit!: the loss closes over the training data; SBP
                # may repair up to half the population per generation, up to 30 attempts
                # per individual
		fit!(regressor, epochs, population_size, loss_new_; target_dimension=target_dim,
                break_condition=break_condition, correction_amount=0.5, cycles=30)

                end_time = (time_ns() - start_time) / 1e9
                elem = regressor.best_models_[1]
                fitness_r2_train = loss_new(elem, x_train', y_train)
                fitness_r2_test = loss_new(elem, x_test', y_test)

                # one row in the column order above: R2_test before R2_train (earlier
                # versions of this script wrote the two swapped)
                push!(results, (seed, case_name, noise_level, mean(elem.fitness), equation_string(elem),
                    fitness_r2_test, fitness_r2_train, end_time, elem.dimension_homogene, target_dim))

                @show fitness_r2_test
                save_results_to_csv(file_name_save, results)
            end
        end
    end
    close(global_logger().stream)
end

let args = parse.(Int, ARGS)
    main(; (k => v for (k, v) in zip((:seeds, :epochs, :population_size), args))...)
end
