module GepRegression



include("Losses.jl")
include("Util.jl")
include("Selection.jl")
include("Entities.jl")

using .LossFunction
export runGep

using .GepUtils
using .EvoSelection
using .GepEntities

using Random
using Statistics
using LinearAlgebra
using ProgressMeter
using OrderedCollections
using DynamicExpressions
using Logging
using Printf



const Chromosome = GepEntities.Chromosome
const Toolbox = GepEntities.Toolbox


"""
    compute_fitness(elem::Chromosome, operators::OperatorEnum, x_data::AbstractArray{T},
        y_data::AbstractArray{T}, loss_function::Function, crash_value::T; 
        validate::Bool=false, penalty_consideration::Real=0.0) where {T<:AbstractFloat}

Computes the fitness score for a chromosome using the specified loss function.

# Arguments
- `elem::Chromosome`: The chromosome whose fitness needs to be computed
- `operators::OperatorEnum`: The set of mathematical operators available for expression evaluation
- `x_data::AbstractArray{T}`: Input features for fitness computation
- `y_data::AbstractArray{T}`: Target values for fitness computation
- `loss_function::Function`: The loss function used to compute fitness
- `crash_value::T`: Default value returned if computation fails
- `validate::Bool=false`: If true, forces recomputation of fitness even if already calculated
- `penalty_consideration::Real=0.0`: Additional penalty term added to the fitness score

# Returns
Returns the computed fitness value (loss + penalty) or crash_value if computation fails

# Details
- Checks if fitness needs to be computed (if NaN or validate=true)
- Evaluates the chromosome's compiled function on input data
- Applies loss function and adds any penalty consideration
- Returns crash_value if any errors occur during computation
"""
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

"""
    perform_step!(population::Vector{Chromosome}, parents::Vector{Chromosome}, 
        next_gen::Vector{Chromosome}, toolbox::Toolbox, mating_size::Int)

Performs one evolutionary step in the GEP algorithm, creating and evaluating new chromosomes.

# Arguments
- `population::Vector{Chromosome}`: Current population of chromosomes
- `parents::Vector{Chromosome}`: Selected parent chromosomes for breeding
- `next_gen::Vector{Chromosome}`: Buffer for storing newly created chromosomes
- `toolbox::Toolbox`: Contains genetic operators and algorithm parameters
- `mating_size::Int`: Number of chromosomes to create in this step

# Details
- Processes parents in pairs to create new chromosomes
- Applies genetic operations to create offspring
- Compiles expressions for new chromosomes
- Updates population with new chromosomes
- Operations are performed in parallel using multiple threads
"""
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


"""
    perform_correction_callback!(population::Vector{Chromosome}, epoch::Int, 
        correction_epochs::Int, correction_amount::Real,
        correction_callback::Union{Function,Nothing})

Applies correction operations to ensure dimensional homogeneity in chromosomes.

# Arguments
- `population::Vector{Chromosome}`: Current population of chromosomes
- `epoch::Int`: Current epoch number
- `correction_epochs::Int`: Frequency of correction operations
- `correction_amount::Real`: Proportion of population to apply corrections to
- `correction_callback::Union{Function,Nothing}`: Function that performs the actual correction

# Details
- Executes corrections periodically (every correction_epochs)
- Processes a subset of the population determined by correction_amount
- Applies corrections to dimensionally heterogeneous chromosomes
- Updates chromosome compilation and dimensional homogeneity flags
"""
@inline function perform_correction_callback!(population::Vector{Chromosome}, epoch::Int, correction_epochs::Int, correction_amount::Real,
    correction_callback::Union{Function,Nothing})

    if !isnothing(correction_callback) && epoch % correction_epochs == 0
        pop_amount = Int(ceil(length(population) * correction_amount))
        Threads.@threads for i in 1:pop_amount
            if !(population[i].dimension_homogene)
                distance, correction = correction_callback(population[i].genes, population[i].toolbox.gen_start_indices,
                    population[i].expression_raw)
                if correction
                    compile_expression!(population[i]; force_compile=true)
                    population[i].dimension_homogene = true
                end
            end
        end
    end
end



"""
    runGep(epochs::Int, population_size::Int, operators::OperatorEnum,
        x_data::AbstractArray{T}, y_data::AbstractArray{T}, toolbox::Toolbox;
        hof::Int=3, x_data_test::Union{AbstractArray{T},Nothing}=nothing,
        y_data_test::Union{AbstractArray{T},Nothing}=nothing,
        loss_fun_::Union{String,Function}="mae",
        correction_callback::Union{Function,Nothing}=nothing,
        correction_epochs::Int=1, correction_amount::Real=0.6,
        tourni_size::Int=3, opt_method_const::Symbol=:cg,
        optimisation_epochs::Int=500) where {T<:AbstractFloat}

Main function that executes the GEP algorithm for regression problems.

# Arguments
- `epochs::Int`: Number of evolutionary epochs to run
- `population_size::Int`: Size of the chromosome population
- `operators::OperatorEnum`: Available mathematical operators
- `x_data::AbstractArray{T}`: Training input features
- `y_data::AbstractArray{T}`: Training target values
- `toolbox::Toolbox`: Contains genetic operators and algorithm parameters
- `hof::Int=3`: Number of best solutions to return (Hall of Fame size)
- `x_data_test::Union{AbstractArray{T},Nothing}`: Optional test input features
- `y_data_test::Union{AbstractArray{T},Nothing}`: Optional test target values
- `loss_fun_::Union{String,Function}="mae"`: Loss function for fitness computation
- `correction_callback::Union{Function,Nothing}`: Function for dimensional homogeneity correction
- `correction_epochs::Int=1`: Frequency of correction operations
- `correction_amount::Real=0.6`: Proportion of population for correction
- `tourni_size::Int=3`: Tournament selection size
- `opt_method_const::Symbol=:cg`: Optimization method for constant optimization
- `optimisation_epochs::Int=500`: Frequency of constant optimization

# Returns
Tuple{Vector{Chromosome}, Any}: Returns best solutions and training history
Best solutions include both training and test R² scores when test data is provided

# Details
1. Initializes population and evolution parameters
2. For each epoch:
   - Applies dimensional homogeneity corrections
   - Computes fitness for all chromosomes
   - Sorts population by fitness
   - Optimizes constants for best solution periodically
   - Records training progress
   - Performs tournament selection
   - Creates new generation through genetic operations
3. Computes final R² scores for best solutions
4. Returns best solutions and training history

Progress is monitored through a progress bar showing:
- Current epoch
- Training loss
- Validation loss
Early stopping occurs if perfect R² score is achieved
"""
function runGep(epochs::Int,
    population_size::Int,
    operators::OperatorEnum,
    x_data::AbstractArray{T},
    y_data::AbstractArray{T},
    toolbox::Toolbox;
    hof::Int=3,
    x_data_test::Union{AbstractArray{T},Nothing}=nothing,
    y_data_test::Union{AbstractArray{T},Nothing}=nothing,
    loss_fun_::Union{String,Function}="mae",
    correction_callback::Union{Function,Nothing}=nothing,
    correction_epochs::Int=1,
    correction_amount::Real=0.6,
    tourni_size::Int=3,
    opt_method_const::Symbol=:cg,
    optimisation_epochs::Int=500) where {T<:AbstractFloat}

    loss_fun = typeof(loss_fun_) == String ? get_loss_function(loss_fun_) : loss_fun_

    recorder = HistoryRecorder(epochs, Float64)

    function optimizer_function(sub_tree::Node)
        y_pred, flag = eval_tree_array(sub_tree, x_data, operators)
        return get_loss_function("mse")(y_pred, y_data)
    end

    penalty_consideration = convert(Real,toolbox.gep_probs["penalty_consideration"])
    mating_ = toolbox.gep_probs["mating_size"]
    mating_size = Int(ceil(population_size * mating_))
    mating_size = mating_size % 2 == 0 ? mating_size : mating_size - 1
    fits_representation = Vector{T}(undef, population_size)

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
                    opt_method=opt_method_const, max_iterations=150, n_restarts=5)
                population[1].fitness = result
                population[1].compiled_function = eqn
                prev_best = result
            end
        catch
            @show "Opt. issue"
        end

        Threads.@threads for index in eachindex(population)
            fits_representation[index] =  population[index].fitness
        end

        best_r = compute_fitness(population[1], operators, x_data, y_data,
            get_loss_function("r2_score"), zero(T); validate=true)
        val_loss = compute_fitness(population[1], operators, x_data_test, y_data_test, loss_fun, typemax(T); validate=true)
        record!(recorder, epoch, fits_representation[1], val_loss, fits_representation)


        ProgressMeter.update!(progBar, epoch, showvalues=[
            (:epoch_, @sprintf("%.0f", epoch)),
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