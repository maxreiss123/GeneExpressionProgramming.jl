module GepRegressor

include("Losses.jl")
include("Util.jl")
include("Selection.jl")
include("Entities.jl")

using .LossFunction
using .GepUtils
using .EvoSelection
using .SymbolicEntities

using Random
using Statistics
using LinearAlgebra
using ProgressMeter
using OrderedCollections
using DynamicExpressions
using Logging
using Printf

export runGep

@inline function compute_fitness(elem::Chromosome, operators::OperatorEnum, x_data::AbstractArray{T},
    y_data::AbstractArray{T}, loss_function::Function,
    crash_value::T; validate::Bool=false, penalty_consideration::Real=0.0) where {T<:AbstractFloat}
    try
        if isnan(elem.fitness) || validate
            y_pred = elem.compiled_function(x_data, operators)
            return loss_function(y_data, y_pred) + penalty_consideration
        else
            return elem.fitness
        end
    catch e
        return crash_value
    end
end

@inline function perform_step!(population::Vector{Chromosome}, parents::Vector{Chromosome}, next_gen::Vector{Chromosome},
    toolbox::Toolbox, mating_size::Int)

    @inbounds Threads.@threads for i in 1:2:mating_size-1
        next_gen[i] = parents[i]
        next_gen[i+1] = parents[i+1]

        genetic_operations!(next_gen, i, toolbox)

        compile_expression!(next_gen[i]; force_compile=true)
        compile_expression!(next_gen[i+1]; force_compile=true)

    end

    Threads.@threads for i in 1:mating_size-1
        try
            population[end-i] = next_gen[i]
        catch
            @show "sth went wrong"
        end
    end
end

@inline function perform_correction_callback!(population::Vector{Chromosome}, epoch::Int, correction_epochs::Int, correction_amount::Real,
    correction_callback::Union{Function,Nothing})

    if !isnothing(correction_callback) && epoch % correction_epochs == 0
        pop_amount = Int(ceil(length(population) * correction_amount))
        Threads.@threads for i in 1:pop_amount
            if isnan(population[i].fitness)
                distance, correction = correction_callback(population[i].genes, population[i].toolbox.gen_start_indices,
                    population[i].expression_raw)
                if correction
                    compile_expression!(population[i]; force_compile=true)
                    population[i].dimension_homogene = true
                else
                    population[i].penalty += distance
                end
            end
        end
    end
end


function runGep(epochs::Int,
    population_size::Int,
    gene_count::Int,
    head_len::Int,
    symbols::OrderedDict,
    operators::OperatorEnum,
    callbacks::Dict,
    nodes::OrderedDict,
    x_data::AbstractArray{T},
    y_data::AbstractArray{T},
    gene_connections::Vector{Int8},
    gep_probs::Dict{String,AbstractFloat};
    hof::Int=3,
    x_data_test::Union{AbstractArray{T},Nothing}=nothing,
    y_data_test::Union{AbstractArray{T},Nothing}=nothing,
    seed::Int=0,
    loss_fun_str::String="mae",
    mating_::Real=0.5,
    correction_callback::Union{Function,Nothing}=nothing,
    correction_epochs::Int=1,
    correction_amount::Real=0.6,
    tourni_size::Int=3, penalty_consideration::Real=0.0,
    opt_method_const::Symbol=:cg,
    preamble_syms=Int8[],
    optimisation_epochs::Int=500) where {T<:AbstractFloat}

    loss_fun::Function = get_loss_function(loss_fun_str)
    recorder = HistoryRecorder(epochs, Float64)

    if isnothing(x_data_test) || isnothing(y_data_test)
        x_data_test = x_data
        y_data_test = y_data
    end

    function optimizer_function(sub_tree::Node)
        y_pred, flag = eval_tree_array(sub_tree, x_data, operators)
        return get_loss_function("mse")(y_pred, y_data)
    end

    

    Random.seed!(seed)
    mating_size = Int(ceil(population_size * mating_))
    mating_size = mating_size % 2 == 0 ? mating_size : mating_size - 1
    toolbox = Toolbox(gene_count, head_len, symbols, gene_connections,
        callbacks, nodes, gep_probs; preamble_syms=preamble_syms)

    population = generate_population(population_size, toolbox)
    next_gen = Vector{eltype(population)}(undef, mating_size)
    progBar = Progress(epochs; showspeed=true, desc="Training: ")

    prev_best = -1
    
    for epoch in 1:epochs
        perform_correction_callback!(population, epoch, correction_epochs, correction_amount, correction_callback)

        Threads.@threads for i in eachindex(population)
            if isnan(population[i].fitness)
                population[i].fitness = compute_fitness(population[i], operators, x_data, y_data, loss_fun, typemax(T);
                    penalty_consideration=population[i].penalty * penalty_consideration)
            end
        end

        sort!(population, by=x -> x.fitness)

        try
            if (prev_best == -1 || prev_best > population[1].fitness) && epoch % optimisation_epochs == 0
                eqn, result = optimize_constants!(population[1].compiled_function, optimizer_function;
                    opt_method=opt_method_const, max_iterations=250, n_restarts=3)
                population[1].fitness = result
                population[1].compiled_function = eqn
                prev_best = result
            end
        catch
            @show "Opt. issue"
        end


        fits_representation = [chromo.fitness for chromo in population]
        best_r = compute_fitness(population[1], operators, x_data, y_data,
            get_loss_function("r2_score"), zero(T); validate=true)
        val_loss = compute_fitness(population[1], operators, x_data_test, y_data_test, loss_fun, typemax(T); validate=true)
        record!(recorder, epoch, fits_representation[1], val_loss, fits_representation)


        ProgressMeter.update!(progBar, epoch, showvalues = [
            (:train_loss, @sprintf("%.6f", fits_representation[1])),
            (:validation_loss, @sprintf("%.6f", val_loss))
        ])


        if isclose(best_r, one(T))
            break
        end

        if epoch < epochs
            indices = basic_tournament_selection(fits_representation[1:mating_size], tourni_size, mating_size)
            parents = population[indices]
            perform_step!(population, parents, next_gen, toolbox, mating_size)
        end

    end

    best = population[1:hof]
    for elem in best
        elem.fitness_r2_train = compute_fitness(elem, operators, x_data, y_data, get_loss_function("r2_score"), zero(T); validate=true)
        if !isnothing(x_data_test)
            elem.fitness_r2_test = compute_fitness(elem, operators, x_data_test, y_data_test, get_loss_function("r2_score"), zero(T); validate=true)
        end
    end
    
    close_recorder!(recorder)
    return best, recorder.history
end
end