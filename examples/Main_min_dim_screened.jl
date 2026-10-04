include("FileHandler.jl")

using GeneExpressionProgramming
using Random
using .FileHandler
using Logging
using DelimitedFiles          # for the probe states below; Pkg.add("DelimitedFiles") if missing

global_logger(ConsoleLogger(stderr, Logging.Debug))

# The probe states of the screening, from the baseline runs as ParaView exports them (Save
# Data as CSV): one row per point with I1_mean..I4_mean and the six components of each of
# T1_mean..T4_mean. Every point gives one sample per component, made of the invariants and
# the same component of T1..T4, in the order of the features x1..x8 (I1..I4, T1..T4); the
# coordinates (Points:0..2) are not used. The screening embeds a model by its values on 40
# samples drawn from these.
baseline_files = ["baseline_features.csv"]          # e.g. one export per case, C1 and C2

function baseline_samples(file)
    data, header = readdlm(file, ',', Float64; header=true)
    names = strip.(vec(header), '"')
    function col(name)
        j = findfirst(==(name), names)
        isnothing(j) && error("$file has no column $name; it has $(join(names, ", "))")
        return j
    end
    invariants = data[:, [col("I$(k)_mean") for k in 1:4]]
    samples = reduce(vcat, [hcat(invariants, data[:, [col("T$(k)_mean:$c") for k in 1:4]])
                            for c in 0:5])
    # of use: finite, and the tensors do not all vanish (where they do, every model, being
    # linear in T1..T4, is zero, and the sample tells the models apart by nothing)
    keep = [all(isfinite, samples[i, :]) && any(!iszero, samples[i, 5:8])
            for i in axes(samples, 1)]
    return samples[keep, :]
end

baseline = reduce(vcat, [baseline_samples(f) for f in baseline_files])
size(baseline, 1) >= 40 || error("the baseline gives $(size(baseline, 1)) usable samples, " *
                                 "the screening draws 40")

for seed in 1:10
    @show "Current seed: $seed"
    number_of_objective = 2
    epochs = 100
    population_size = 50
    number_features = 8
    gene_count = 3
    openfoam_conversion = Dict{Int8,String}()
    feature_meaning = Dict{String,String}("x1" => "I1", "x2" => "I2", "x3" => "I3", "x4" => "I4",
        "x5" => "T1", "x6" => "T2", "x7" => "T3", "x8" => "T4")
    Random.seed!(seed)

    feature_dims = Dict{Symbol,Vector{Float16}}(
        :x1 => Float16[0, 0, 0, 0, 0, 0, 0],
        :x2 => Float16[0, 0, 0, 0, 0, 0, 0],
        :x3 => Float16[0, 1, 0, 0, 0, 0, 0],
        :x4 => Float16[0, 1, 0, 0, 0, 0, 0],
        :x5 => Float16[0, 0, 0, 0, 0, 0, 1],
        :x6 => Float16[0, 1, 0, 0, 0, 0, 1],
        :x7 => Float16[0, 1, 0, 0, 0, 0, 1],
        :x8 => Float16[0, 1, 0, 0, 0, 0, 1])      # the 7th slot marks the tensors
    target_dim = Float16[0, 3, 0, 0, 0, 0, 1]

    log_name = "optimization_trace$(seed).csv"
    # Simulated: true for errors the CFD computed, false for a prediction of the screening
    log_header = "Seed,Epoch,Equation,EqnToken," *
                 join(["Error$i" for i in 1:number_of_objective], ",") *
                 ",Comment,Chromo_id,Pareto,Simulated"

    python_path = "/home/student.unimelb.edu.au/reissmannm/anaconda3/envs/phd/bin/python"
    config_cfd_loop = Dict(
        "name" => "cfd_loop",
        "running_template_path" => "running_template",
        "input_template_name" => "input_template",
        "eval_template_path" => "eval_template",
        "output_folder_name" => "output",
        "target_path" => ["C1", "C2"],
        "eval_model_file" => "eval_model.py")
    fileMan = FileManager(config_cfd_loop)
    remove_run_folders = false           # true: delete a run folder once its errors are read
    loss_calls = Threads.Atomic{Int}(0)
    cfd_runs = Threads.Atomic{Int}(0)

    logger = open(log_name, "w")
    println(logger, log_header)
    flush(logger)

    regressor = GepRegressor(number_features;
        gene_count=gene_count,
        head_len=9,
        entered_non_terminals=[:+, :-, :*],
        gene_connections=[:+, :-],
        entered_terminal_nums=[Symbol(0.5), Symbol(0.0), Symbol(2.0)],
        number_of_objectives=number_of_objective,
        rnd_count=3,                     # random constants
        considered_dimensions=feature_dims,
        max_permutations_lib=50000000,
        rounds=9)

    # translation into the tokens of OpenFOAM: functions, features and constants
    for (key, elem) in regressor.toolbox_.callbacks
        openfoam_conversion[key] = string(elem)
    end
    for (key, elem) in regressor.toolbox_.nodes
        openfoam_conversion[key] = get(feature_meaning, string(elem), string(elem))
    end
    for (key, value) in openfoam_conversion
        value == "/" && (openfoam_conversion[key] = "div")
    end

    # The tokens OpenFOAM receives: the karva string, and once the constant optimiser has
    # tuned the model, the tuned value of every constant in its place (one value per
    # occurrence, in the order of constant_positions).
    function openfoam_tokens(chromo)
        tokens = [openfoam_conversion[s] for s in chromo.expression_raw]
        if !isnothing(chromo.optimised_constants)
            for (pos, value) in zip(constant_positions(chromo), chromo.optimised_constants)
                tokens[pos] = string(value)
            end
        end
        return tokens
    end

    # The screening: the CFD runs for 15 % of the new individuals of an epoch, a Gaussian
    # process per objective predicts the errors of the others.
    probes = permutedims(baseline[randperm(size(baseline, 1))[1:40], :])
    surrogate = SurrogateScreening(regressor, probes;
        individuals_per_epoch=0.15,
        screen=GpScreen(acquisition=:qehvi),   # two objectives, a solver that can fail
        failure_above=9999.0,                  # eval_model.py gives a diverged run 9999: a
        #                                      # failure for the screening, not an error to fit
        seed=seed)

    function log_population(population, epoch, selected_members)
        rearranged_pareto = Dict(v => k for (k, vs) in selected_members.fronts for v in vs)
        # "best": the simulated individual with the lowest sum of errors. That is not always
        # population[1]: a model ranked first by its prediction is simulated before the
        # selection, and when that run diverges (9999) it still stands first
        simulated = [is_validated(surrogate, c) && all(isfinite, c.fitness) for c in population]
        best = any(simulated) ?
               argmin(i -> simulated[i] ? sum(population[i].fitness) : Inf, eachindex(population)) : 0
        for (index, elem) in enumerate(population)
            results = Any[seed, epoch, equation_string(elem), join(openfoam_tokens(elem), ";")]
            append!(results, [string(f) for f in elem.fitness])
            push!(results, index == best ? "best" : "", elem.chromo_id,
                get(rearranged_pareto, index, 0), is_validated(surrogate, elem))
            println(logger, join(results, ","))
        end
        flush(logger)
    end

    # The errors of every input OpenFOAM has run, by the input: a model whose tokens, tuned
    # constants included, ran before is not run again. Every tuning of the constants starts
    # with the constants the model holds, whose errors are known.
    known_results = Dict{String,Tuple{Tuple,Int}}()
    known_lock = ReentrantLock()             # the loss runs on several threads at once

    # The loss of the search, of the screening and of the constant optimiser. It runs the CFD
    # for an individual whose fitness is unset (NaN) and on every call with validate = true:
    # the constant optimiser calls it once per trial of the constants, on the same model.
    function cost_function_external!(chromo, validate::Bool)
        any(isnan, chromo.fitness) || validate || return
        Threads.atomic_add!(loss_calls, 1)
        worst = regressor.toolbox_.fitness_reset[1]          # Inf for every objective
        if !chromo.dimension_homogene
            chromo.fitness = worst
            return
        end
        tokens = openfoam_tokens(chromo)
        key = join(tokens, ",")
        known = lock(() -> get(known_results, key, nothing), known_lock)
        if !isnothing(known)
            chromo.fitness, chromo.chromo_id = known
            return
        end
        ind_id = get_next_id(fileMan)                         # atomic: thread-safe
        Threads.atomic_add!(cfd_runs, 1)
        result = worst
        try
            target_path = create_cfd_instance(fileMan, ind_id, [tokens])
            execute_external_file(fileMan, target_path; python=python_path)
            # a failed case is the worst error: NaN would mean "not scored" to the search
            result = map(v -> isfinite(v) ? v : Inf, retrieve_value(fileMan, target_path))
            remove_run_folders && deconstruct_cfd_instance(fileMan, target_path)
        catch e
            @warn "CFD run $ind_id failed" exception = e
        end
        all(isfinite, result) && lock(() -> (known_results[key] = (result, ind_id)), known_lock)
        chromo.chromo_id = ind_id
        chromo.fitness = result
    end

    try
        fit!(regressor, epochs, population_size, cost_function_external!;
            file_logger_callback=log_population,
            target_dimension=target_dim, correction_amount=1.0, cycles=500,
            surrogate=surrogate,
            constant_optimizer=ScreenedNelderMead(max_evaluations=20),   # CFD runs per tuning
            optimization_epochs=5)
    catch e
        @error "Something went wrong" exception = (e, catch_backtrace())
    end
    close(logger)

    println("seed $seed: $(loss_calls[]) loss calls, $(surrogate.evaluated_count) by the ",
        "screening and $(loss_calls[] - surrogate.evaluated_count) by the constant optimiser; ",
        "$(cfd_runs[]) of them ran the CFD, the others reused a result; ",
        "$(surrogate.imputed_count) predictions instead of runs")
    best = regressor.best_models_
    for m in unique(m -> m.fitness, best[calculate_fronts([m.fitness for m in best])[1]])
        println("  ", m.fitness, "   ", equation_string(m), "   ", join(openfoam_tokens(m), " "))
    end
end
