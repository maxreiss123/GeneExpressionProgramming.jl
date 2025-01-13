"""
    GepRegression

A module implementing Gene Expression Programming (GEP) for symbolic regression 

# Features
- Symbolic regression using Gene Expression Programming
- Dimensional homogeneity enforcement
- Constant optimization
- Validation on test data

# Dependencies
## Internal Modules
- `LossFunction`: Various loss metrics
- `GepUtils`: Utility functions
- `EvoSelection`: Selection mechanisms
- `GepEntities`: Core GEP data structures

## External Packages
- `Random`: For stochastic operations
- `Statistics`: Statistical computations
- `LinearAlgebra`: Matrix operations
- `ProgressMeter`: Progress visualization
- `OrderedCollections`: Ordered data structures
- `DynamicExpressions`: Expression handling
- `Logging`: Error and debug logging
- `Printf`: Formatted output

# Usage Example
```julia
using GepRegression

# Setup parameters
toolbox = Toolbox(...)
operators = OperatorEnum(...)

# Run GEP regression
best_solutions, history = runGep(
    100,                    # epochs
    50,                     # population size
    operators,
    x_train, y_train,      # training data
    toolbox;
    x_data_test=x_test,    # test data
    y_data_test=y_test,
    loss_fun_="rmse"       # loss function
)
```

See also: 
- [`GepEntities.Chromosome`](@ref): Individual solution representation in GEP
- [`GepEntities.Toolbox`](@ref): GEP algorithm parameters and operations
- [`GepEntities.genetic_operations!`](@ref): Genetic modification operations
- [`GepEntities.compile_expression!`](@ref): Chromosome expression compilation
- [`GepEntities.generate_chromosome`](@ref): New chromosome creation
- [`GepEntities.generate_population`](@ref): Initial population generation
- [`GepEntities.fitness`](@ref): Fitness value access
- [`GepEntities.set_fitness!`](@ref): Fitness value modification

#TODO need to create different strategies for wrapping costum functions
"""

module GepRegression


include("Losses.jl")
include("Util.jl")
include("Selection.jl")
include("Entities.jl")

using .LossFunction
export runGep, EvaluationStrategy, StandardRegressionStrategy, GenericRegressionStrategy

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
using Base.Threads: SpinLock

const Chromosome = GepEntities.Chromosome
const Toolbox = GepEntities.Toolbox

abstract type EvaluationStrategy end

struct StandardRegressionStrategy{T<:AbstractFloat} <: EvaluationStrategy
    operators::OperatorEnum
    number_of_objectives::Int
    x_data::AbstractArray{T}
    y_data::AbstractArray{T}
    x_data_test::AbstractArray{T}
    y_data_test::AbstractArray{T}
    loss_function::Function
    secOptimizer::Union{Function,Nothing}
    break_condition::Union{Function,Nothing}
    penalty::T
    crash_value::T

    function StandardRegressionStrategy{T}(operators::OperatorEnum,
        x_data::AbstractArray,
        y_data::AbstractArray,
        x_data_test::AbstractArray,
        y_data_test::AbstractArray,
        loss_function::Function;
        secOptimizer::Union{Function,Nothing}=nothing,
        break_condition::Union{Function,Nothing}=nothing,
        penalty::T=zero(T),
        crash_value::T=typemax(T)) where {T<:AbstractFloat}
        new(operators,
            1,
            x_data,
            y_data,
            x_data_test,
            y_data_test,
            loss_function,
            secOptimizer,
            break_condition,
            penalty,
            crash_value
        )
    end

end

struct GenericRegressionStrategy <: EvaluationStrategy
    operators::OperatorEnum
    number_of_objectives::Int
    loss_function::Function
    secOptimizer::Union{Function,Nothing}
    break_condition::Union{Function,Nothing}

    function GenericRegressionStrategy(operators::OperatorEnum, number_of_objectives::Int, loss_function::Function;
        secOptimizer::Union{Function,Nothing}, break_condition::Union{Function,Nothing})
        new(operators, number_of_objectives, loss_function, secOptimizer, break_condition)
    end
end


#redesign -> compute fitness should return fitness and crash, we just need to insert the chromosome
"""
    compute_fitness(elem::Chromosome, operators::OperatorEnum, x_data::AbstractArray{T},
        y_data::AbstractArray{T}, loss_function::Function, crash_value::T; 
        validate::Bool=false) where {T<:AbstractFloat}

Computes the fitness score for a chromosome using the specified loss function.

# Arguments
- `elem::Chromosome`: The chromosome whose fitness needs to be computed
- `operators::OperatorEnum`: The set of mathematical operators available for expression evaluation
- `x_data::AbstractArray{T}`: Input features for fitness computation
- `y_data::AbstractArray{T}`: Target values for fitness computation
- `loss_function::Function`: The loss function used to compute fitness
- `crash_value::T`: Default value returned if computation fails
- `validate::Bool=false`: If true, forces recomputation of fitness even if already calculated

# Returns
Returns the computed fitness value (loss) or crash_value if computation fails

# Details
- Checks if fitness needs to be computed (if NaN or validate=true)
- Evaluates the chromosome's compiled function on input data
- Returns crash_value if any errors occur during computation
"""
@inline function compute_fitness(elem::Chromosome, evalArgs::StandardRegressionStrategy; validate::Bool=false)
    try
        if isnan(mean(elem.fitness)) || validate
            y_pred = elem.compiled_function(evalArgs.x_data, evalArgs.operators)
            return (evalArgs.loss_function(evalArgs.y_data, y_pred),)
        else
            return (elem.fitness,)
        end
    catch e
        return (evalArgs.crash_value,)
    end
end

@inline function compute_fitness(elem::Chromosome, evalArgs::GenericRegressionStrategy; validate::Bool=false)
    return evalArgs.loss_function(elem, validate)
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
        catch e
            error_message = sprint(showerror, e, catch_backtrace())
            @error "Error in perform_step!: $error_message"
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
            if !(population[i].dimension_homogene) && population[i].compiled
                distance, correction = correction_callback(population[i].genes, population[i].toolbox.gen_start_indices,
                    population[i].expression_raw)
                if correction
                    compile_expression!(population[i]; force_compile=true)
                    population[i].dimension_homogene = true
                else
                    population[i].fitness += distance
                end
            end
        end
    end
end


"""
    runGep(epochs::Int, population_size::Int, toolbox::Toolbox, evalStrategy::EvaluationStrategy;
        hof::Int=3, correction_callback::Union{Function,Nothing}=nothing,
        correction_epochs::Int=1, correction_amount::Real=0.6,
        tourni_size::Int=3)

Main function that executes the GEP algorithm using a specified evaluation strategy.

# Arguments
- `epochs::Int`: Number of evolutionary epochs to run
- `population_size::Int`: Size of the chromosome population
- `toolbox::Toolbox`: Contains genetic operators and algorithm parameters
- `evalStrategy::EvaluationStrategy`: Strategy for evaluating chromosomes, handling fitness computation, and optimization

# Optional Arguments
- `hof::Int=3`: Number of best solutions to return (Hall of Fame size)
- `correction_callback::Union{Function,Nothing}=nothing`: Function for dimensional homogeneity correction
- `correction_epochs::Int=1`: Frequency of correction operations
- `correction_amount::Real=0.6`: Proportion of population for correction
- `tourni_size::Int=3`: Tournament selection size

# Returns
`Tuple{Vector{Chromosome}, Any}`: Returns best solutions and training history

# Details
1. Initializes population and evolution parameters
2. For each epoch:
   - Applies dimensional homogeneity corrections if provided
   - Computes fitness for all chromosomes using evaluation strategy
   - Sorts population based on fitness
   - Applies secondary optimization if specified in strategy
   - Records training progress
   - Checks break condition from evaluation strategy
   - Performs selection and creates new generation
3. Returns best solutions and training history

Progress is monitored through a progress bar showing:
- Current epoch
- Training loss
- Validation loss

The evolution process stops when either:
- Maximum epochs is reached
- Break condition specified in evaluation strategy is met => needs to be informed as break_condition(population, epoch)
"""
function runGep(epochs::Int,
    population_size::Int,
    toolbox::Toolbox,
    evalStrategy::EvaluationStrategy;
    hof::Int=3,
    correction_callback::Union{Function,Nothing}=nothing,
    correction_epochs::Int=1,
    correction_amount::Real=0.6,
    tourni_size::Int=3,
    optimization_epochs::Int=500)

    recorder = HistoryRecorder(epochs, Tuple)
    mating_ = toolbox.gep_probs["mating_size"]
    mating_size = Int(ceil(population_size * mating_))
    mating_size = mating_size % 2 == 0 ? mating_size : mating_size - 1
    fits_representation = Vector{Tuple}(undef, population_size)
    fit_cache = Dict{Vector{Int8},Tuple}()
    cache_lock = SpinLock()

    population = generate_population(population_size, toolbox)
    next_gen = Vector{eltype(population)}(undef, mating_size)
    progBar = Progress(epochs; showspeed=true, desc="Training: ")

    prev_best = (typemax(Float64),)
    for epoch in 1:epochs
        perform_correction_callback!(population, epoch, correction_epochs, correction_amount, correction_callback)

        Threads.@threads for i in eachindex(population)
            if isnan(mean(population[i].fitness)) 
                cache_value = get(fit_cache, population[i].expression_raw, nothing)
                if isnothing(cache_value)
                    population[i].fitness = compute_fitness(population[i], evalStrategy)
                    lock(cache_lock)
                        fit_cache[population[i].expression_raw] = population[i].fitness
                    unlock(cache_lock)
                else
                    population[i].fitness = cache_value
                end
            end
        end
        sort!(population, by=x -> mean(x.fitness))

        Threads.@threads for index in eachindex(population)
            fits_representation[index] = population[index].fitness
        end

        if !isnothing(evalStrategy.secOptimizer) && epochs % optimization_epochs == 0 && population[1].fitness < prev_best
            evalStrategy.secOptimizer(population)
            fits_representation[1] = population[1].fitness
            prev_best = fits_representation[1]
        end

        val_loss = compute_fitness(population[1], evalStrategy; validate=true)
        record!(recorder, epoch, fits_representation[1], val_loss, fits_representation)


        ProgressMeter.update!(progBar, epoch, showvalues=[
            (:epoch_, @sprintf("%.0f", epoch)),
            (:train_loss, @sprintf("%.6e", mean(fits_representation[1]))),
            (:validation_loss, @sprintf("%.6e", mean(val_loss)))
        ])


        if !isnothing(evalStrategy.break_condition) && evalStrategy.break_condition(population, epoch)
            break
        end

        if epoch < epochs
            selectedMembers = selection(fits_representation, mating_size, tourni_size)
            parents = population[selectedMembers.indices]
            perform_step!(population, parents, next_gen, toolbox, mating_size)
        end

    end

    best = population[1:hof]
    close_recorder!(recorder)
    return best, recorder.history
end

end
