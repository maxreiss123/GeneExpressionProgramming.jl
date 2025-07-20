"""
    GepEntities

A module implementing core data structures and genetic operations for Gene Expression
Programming (GEP).


## Evolution Types
- `Chromosome`: Individual solution representation
- `Toolbox`: Container for GEP algorithm parameters and operations

# Features
## Symbol Management
- Flexible symbol representation system
- Support for physical dimensions
- Arithmetic operation mapping
- Forward/reverse function handling

## Genetic Operations
### Crossover Operations
- One-point crossover
- Two-point crossover
- Gene fusion (dominant/recessive) - experimental

### Mutation Operations
- Random mutation
- Gene inversion
- Gene insertion
- Reverse insertion
- Tail insertion


# Implementation Details
## Performance Optimizations
- Inlined genetic operations
- View-based array operations

## Memory Management
- Efficient gene replication through vector representation
- In-place modifications

## Type System
- Int8-based gene representation
- AbstractFloat fitness values
- Flexible symbol representation

# Exports
## Types
- `Chromosome`, `Toolbox`, `EvaluationStrategy``, `StandardRegressionStrategy`, `GenericRegressionStrategy`

## Functions
### Core Operations
- `fitness`, `set_fitness!`
- `generate_gene`, `compile_expression!`
- `generate_chromosome`, `generate_population`

### Genetic Operations
- `genetic_operations!`, `replicate`
- `gene_inversion!`, `gene_mutation!`
- `gene_one_point_cross_over!`, `gene_two_point_cross_over!` - 
- `gene_fussion!` - experimental

# Dependencies
- `GepUtils`: Utility functions
- `OrderedCollections`: Ordered data structures

# Notes
- All genetic operations are implemented as in-place modifications
- Chromosomes maintain their own compilation state

"""
#todo introduce gene blending
module GepEntities


export Chromosome, Toolbox, EvaluationStrategy, StandardRegressionStrategy, GenericRegressionStrategy
export fitness, set_fitness!
export generate_gene, compile_expression!, generate_chromosome, generate_population
export genetic_operations!, replicate, gene_inversion!, gene_mutation!, gene_one_point_cross_over!, gene_two_point_cross_over!, gene_fussion!, split_karva, print_karva_strings
export prob_equation_djl, equation_characterization_default, _karva_raw


using ..GepUtils
using ..TensorRegUtils
using OrderedCollections
using DynamicExpressions
using StatsBase


abstract type EvaluationStrategy end

struct StandardRegressionStrategy{T<:AbstractFloat} <: EvaluationStrategy
    operators::OperatorEnum
    number_of_objectives::Int
    x_data::AbstractArray{T}
    y_data::AbstractArray{T}
    x_data_test::AbstractArray{T}
    y_data_test::AbstractArray{T}
    loss_function::Function
    validation_loss_function::Function
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
        validation_loss_function::Union{Nothing,Function}=nothing,
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
            isnothing(validation_loss_function) ? loss_function : validation_loss_function,
            secOptimizer,
            break_condition,
            penalty,
            crash_value
        )
    end

end

struct GenericRegressionStrategy <: EvaluationStrategy
    operators::Union{OperatorEnum,Nothing}
    number_of_objectives::Int
    loss_function::Function
    validation_loss_function::Function
    secOptimizer::Union{Function,Nothing}
    break_condition::Union{Function,Nothing}

    function GenericRegressionStrategy(operators::Union{OperatorEnum,Nothing}, number_of_objectives::Int, loss_function::Function;
        validation_loss_function::Union{Function,Nothing},
        secOptimizer::Union{Function,Nothing}, break_condition::Union{Function,Nothing})
        new(operators, number_of_objectives, loss_function, isnothing(validation_loss_function) ? loss_function : validation_loss_function,
            secOptimizer, break_condition)
    end
end

"""
    Toolbox

Contains parameters and operations for GEP algorithm execution.

# Fields
- `gene_count::Int`: Number of genes per chromosome
- `head_len::Int`: Head length for each gene
- `symbols::OrderedDict{Int8,Int8}`: Available symbols and their arities
- `gene_connections::Vector{Int8}`: How genes connect
- `headsyms::Vector{Int8}`: Symbols allowed in head
- `tailsyms::Vector{Int8}`: Symbols allowed in tail
- `arrity_by_id::OrderedDict{Int8,Int8}`: Symbol arities
- `callbacks::Dict`: Operation callbacks
- `nodes::OrderedDict`: Node definitions
- `gen_start_indices::Vector{Int}`: Gene start positions
- `gep_probs::Dict{String,AbstractFloat}`: Operation probabilities
- `fitness_reset::Tuple`: Default fitness values
- `preamble_syms::Vector{Int8}`: Preamble symbols
- `len_preamble::Int8`: Preamble length
- `tail_weights::Union{Weights,Nothing}` Defines the probability for tail symbols
- `head_weigths::Union{Weights,Nothing}` Defines the probability for head symbols 

# Constructor
    Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8}, 
           gene_connections::Vector{Int8}, callbacks::Dict, nodes::OrderedDict, 
           gep_probs::Dict{String,AbstractFloat}; 
           fitness_reset::Tuple=(Inf, NaN), preamble_syms=Int8[],
           function_complile::Function=compile_djl_datatype,
           tail_weights_::Union{Weights,Nothing}=nothing,
           head_tail_balance::Real=0.2)
"""
struct Toolbox
    gene_count::Int
    head_len::Int
    symbols::OrderedDict{Int8,Int8}
    gene_connections::Vector{Int8}
    headsyms::Vector{Int8}
    tailsyms::Vector{Int8}
    arrity_by_id::OrderedDict{Int8,Int8}
    callbacks::Dict
    nodes::OrderedDict
    gen_start_indices::Vector{Int}
    gep_probs::Dict{String,AbstractFloat}
    fitness_reset::Tuple
    preamble_syms::Vector{Int8}
    len_preamble::Int8
    operators_::Union{OperatorEnum,Nothing}
    compile_function_::Union{Function,Nothing}
    tail_weights::Union{Weights,Nothing}
    head_weights::Union{Weights,Nothing}


    function Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8}, gene_connections::Vector{Int8},
        callbacks::Dict, nodes::OrderedDict, gep_probs::Dict{String,AbstractFloat};
        unary_prob::Real=0.1, preamble_syms=Int8[],
        number_of_objectives::Int=1, operators_::Union{OperatorEnum,Nothing}=nothing,
        function_complile::Union{Function,Nothing}=compile_djl_datatype,
        tail_weights_::Union{Weights,Nothing}=nothing,
        head_tail_balance::Real=0.5)

        fitness_reset = (
            ntuple(_ -> Inf, number_of_objectives),
            ntuple(_ -> NaN, number_of_objectives)
        )
        gene_len = head_len * 2 + 1
        headsyms = [key for (key, arity) in symbols if arity == 2]
        unary_syms = [key for (key, arity) in symbols if arity == 1]
        up = isempty(unary_syms) ? 0 : unary_prob

        tailsyms = [key for (key, arity) in symbols if arity < 1 && !(key in preamble_syms)]
        len_preamble = length(preamble_syms)
        gen_start_indices = [gene_count + (gene_len * (i - 1)) for i in 1:gene_count]

        tail_weights = isnothing(tail_weights_) ? weights([1 / length(tailsyms) for _ in 1:length(tailsyms)]) : tail_weights_
        head_weights = weights([
            fill(head_tail_balance / length(headsyms), length(headsyms));
            fill(unary_prob / length(unary_syms), length(unary_syms));
            tail_weights .* (1 - head_tail_balance - up)
        ])


        #ensure_buffer_size!(head_len, gene_count)
        head_syms = vcat([headsyms, unary_syms, tailsyms]...)
        new(gene_count, head_len, symbols, gene_connections, head_syms, tailsyms, symbols,
            callbacks, nodes, gen_start_indices, gep_probs, fitness_reset, preamble_syms, len_preamble, operators_,
            function_complile,
            tail_weights, head_weights)
    end
end

"""
    Chromosome

Represents an individual solution in GEP.

# Fields
- `genes::Vector{Int8}`: Genetic material
- `fitness::Union{AbstractFloat,Tuple}`: Fitness score
- `toolbox::Toolbox`: Reference to toolbox
- `compiled_function::Any`: Compiled expression
- `compiled::Bool`: Compilation status
- `fitness_r2_train::AbstractFloat`: R² score on training
- `fitness_r2_test::AbstractFloat`: R² score on testing
- `expression_raw::Vector{Int8}`: Raw expression
- `dimension_homogene::Bool`: Dimensional homogeneity
- `chromo_id::Int`: Chromosome identifier

# Constructor
    Chromosome(genes::Vector{Int8}, toolbox::Toolbox, compile::Bool=false)
"""
mutable struct Chromosome
    genes::Vector{Int8}
    fitness::Tuple
    toolbox::Toolbox
    compiled_function::Any
    compiled::Bool
    expression_raw::Vector{Int8}
    dimension_homogene::Bool
    chromo_id::Int

    function Chromosome(genes::Vector{Int8}, toolbox::Toolbox, compile::Bool=false)
        obj = new()
        obj.genes = genes
        obj.fitness = toolbox.fitness_reset[2]
        obj.toolbox = toolbox
        obj.compiled = false
        obj.dimension_homogene = false
        obj.chromo_id = -1
        obj.expression_raw = Int8[]
        if compile
            compile_expression!(obj)
        end
        return obj
    end
end


"""
    compile_expression!(chromosome::Chromosome; force_compile::Bool=false)

Compiles chromosome's genes into executable function - using the types from the DynamicExpressions.

# Arguments
- `chromosome::Chromosome`: Chromosome to compile
- `force_compile::Bool=false`: Force recompilation

# Effects
Updates chromosome's compiled_function and related fields
"""
@inline function compile_expression!(chromosome::Chromosome; force_compile::Bool=false)
    if !chromosome.compiled || force_compile
        try
            expression = _karva_raw(chromosome)
            expression_tree = chromosome.toolbox.compile_function_(expression, chromosome.toolbox.symbols, chromosome.toolbox.callbacks,
                chromosome.toolbox.nodes, max(chromosome.toolbox.len_preamble, 1))
            chromosome.compiled_function = expression_tree
            chromosome.expression_raw = expression
            chromosome.fitness = chromosome.toolbox.fitness_reset[2]
            chromosome.compiled = true
        catch e
            #@error "something went wrong" exception = (e, catch_backtrace())
            chromosome.fitness = chromosome.toolbox.fitness_reset[1]
        end
    end
end

"""
    fitness(chromosome::Chromosome)

Get chromosome's fitness value.

# Returns
Fitness value or tuple
"""
function fitness(chromosome::Chromosome)
    return chromosome.fitness
end


"""
    set_fitness!(chromosome::Chromosome, value::AbstractFloat)

Set chromosome's fitness value.

# Arguments
- `chromosome::Chromosome`: Target chromosome
- `value::AbstractFloat`: New fitness value
"""
function set_fitness!(chromosome::Chromosome, value::Tuple)
    chromosome.fitness = value
end

"""
    _karva_raw(chromosome::Chromosome)

Convert a chromosome's genes into Karva notation (K-expression) by identifying active genes and their connections.

# Arguments
- `chromosome::Chromosome`: The chromosome to convert, containing:
  - `genes`: Vector of gene symbols
  - `toolbox`: Configuration with gene length, count, and arity mappings

# Returns
Vector{Int8} representing the K-expression of the chromosome

# Algorithm
1. Calculate gene dimensions:
   - `gene_len = head_len * 2 + 1` (total length of each gene)
   - `gene_count` (number of genes in chromosome)
   - `len_preamble` (length of preamble section)

2. Extract gene components:
   - Connection symbols between genes (`connectionsym`)
   - Main gene content (`genes`)

3. Process each gene:
   - Map symbols to their arities
   - Create sliding window over gene content
   - Decrement arities (except first position)
   - Find cutoff point where sum of arities becomes zero
   - Extract active portion of gene

4. Combine processed genes:
   - First element: connection symbols
   - Subsequent elements: active portions of each gene
   - Concatenate all elements into final expression

# Examples
```julia
# For a chromosome with genes [1,2,3,4,5,6,7] and gene length 3:
# - Connection symbols: [1]
# - Gene content: [2,3,4,5,6,7]
# - If arities are [2,1,0] for first gene
# Result might be: [1,2,3,4] (connection + active portion)
```

"""
@inline function _karva_raw(chromosome::Chromosome; split::Bool=false)
    gene_len = chromosome.toolbox.head_len * 2 + 1
    gene_count = chromosome.toolbox.gene_count

    connectionsym = @view chromosome.genes[1:gene_count-1]
    genes = @view chromosome.genes[gene_count:gene_count*(gene_len)+gene_count-1]

    arity_gene_ = map(x -> chromosome.toolbox.arrity_by_id[x], genes)
    rolled_indices = Vector{Any}(undef, div(length(arity_gene_), gene_len) + 1)
    rolled_indices[1] = connectionsym

    @inbounds for (idx, i) in enumerate(1:gene_len:length(arity_gene_))
        window = @view arity_gene_[i:i+gene_len-1]
        window[2:end] .-= 1
        indices = find_indices_with_sum(window, 0, 1)
        rolled_indices[idx+1] = @view genes[i:i+first(indices)-1]
    end

    !split && return vcat(rolled_indices...)
    return rolled_indices
end

@inline function split_karva(chromosome::Chromosome, coeffs::Int=2)
    raw = _karva_raw(chromosome; split=true)
    connectors = popfirst!(raw)[coeffs:end]
    gene_count_per_factor = div(chromosome.toolbox.gene_count, coeffs)
    retval = []
    for _ in 1:coeffs
        temp_cons = splice!(connectors, 1:gene_count_per_factor-1)
        temp_genes = reduce(vcat, splice!(raw, 1:gene_count_per_factor))
        push!(retval, vcat([temp_cons, temp_genes]...))
    end
    return retval
end

@inline function print_karva_strings(chromosome::Chromosome; split_len::Int=1)
    callback_ = Dict{Int8,Function}()
    flag = false
    for (key, value) in chromosome.toolbox.callbacks
        if Symbol(value) in keys(FUNCTION_STRINGIFY)
            callback_[key] = FUNCTION_STRINGIFY[Symbol(value)]
        elseif Symbol(value) in keys(TENSOR_STRINGIFY)
            callback_[key] = TENSOR_STRINGIFY[Symbol(value)]
            flag = true
        end
        @debug "Found - $flag - $value"
    end

    return compile_djl_datatype(
        chromosome.expression_raw,
        chromosome.toolbox.arrity_by_id,
        callback_,
        chromosome.toolbox.nodes,
        split_len)
end

"""
    generate_gene(headsyms::Vector{Int8}, tailsyms::Vector{Int8}, headlen::Int; 
                 unarys::Vector{Int8}=[], unary_prob::Real=0.2)

Generate a single gene for GEP.

# Arguments
- `headsyms::Vector{Int8}`: Symbols for head
- `tailsyms::Vector{Int8}`: Symbols for tail
- `headlen::Int`: Head length
- `tail_weights::Union{Weights,Nothing}` Defines the probability for tail symbols
- `head_weigths::Union{Weights,Nothing}` Defines the probability for head symbols 

# Returns
Vector{Int8} representing gene
"""
@inline function generate_gene(headsyms::Vector{Int8}, tailsyms::Vector{Int8}, headlen::Int,
    tail_weights::Weights, head_weights::Weights)
    head = sample(headsyms, head_weights, headlen)
    tail = sample(tailsyms, tail_weights, headlen + 1)
    return vcat(head, tail)
end



"""
    generate_chromosome(toolbox::Toolbox)

Generate a new chromosome using toolbox configuration.

# Returns => need to be adapted for tensor gen
New Chromosome instance
"""
@inline function generate_chromosome(toolbox::Toolbox)
    connectors = rand(toolbox.gene_connections, toolbox.gene_count - 1)
    genes = vcat([generate_gene(toolbox.headsyms, toolbox.tailsyms, toolbox.head_len, toolbox.tail_weights,
        toolbox.head_weights) for _ in 1:toolbox.gene_count]...)
    if !isempty(toolbox.preamble_syms)
        return Chromosome(vcat(connectors, genes, sample(toolbox.preamble_syms, toolbox.gene_count)), toolbox, true)
    end
    return Chromosome(vcat(connectors, genes), toolbox, true)
end



"""
    generate_population(number::Int, toolbox::Toolbox)

Generate initial population of chromosomes.

# Arguments
- `number::Int`: Population size
- `toolbox::Toolbox`: Toolbox configuration

# Returns
Vector of Chromosomes
"""
@inline function generate_population(number::Int, toolbox::Toolbox)
    population = Vector{Chromosome}(undef, number)

    Threads.@threads for i in 1:number
        @inbounds population[i] = generate_chromosome(toolbox)
    end

    return population
end


@inline function create_operator_masks(gene_seq_alpha::Vector{Int8}, gene_seq_beta::Vector{Int8}, pb::Real=0.2)
    alpha_operator = zeros(Int8, length(gene_seq_alpha))
    beta_operator = zeros(Int8, length(gene_seq_beta))
    indices_alpha = rand(1:length(gene_seq_alpha), min(round(Int, (pb * length(gene_seq_alpha))), length(gene_seq_alpha)))
    indices_beta = rand(1:length(gene_seq_beta), min(round(Int, (pb * length(gene_seq_beta))), length(gene_seq_beta)))
    alpha_operator[indices_alpha] .= Int8(1)
    beta_operator[indices_beta] .= Int8(1)
    return alpha_operator, beta_operator
end

@inline function create_operator_point_one_masks(gene_seq_alpha::Vector{Int8}, gene_seq_beta::Vector{Int8}, toolbox::Toolbox)
    alpha_operator = zeros(Int8, length(gene_seq_alpha))
    beta_operator = zeros(Int8, length(gene_seq_beta))
    head_len = toolbox.head_len
    gene_len = head_len * 2 + 1

    for i in toolbox.gen_start_indices
        ref = i
        mid = ref + gene_len ÷ 2

        point1 = rand(ref:mid)
        point2 = rand((mid+1):(ref+gene_len-1))
        alpha_operator[point1:point2] .= Int8(1)

        point1 = rand(ref:mid)
        point2 = rand((mid+1):(ref+gene_len-1))
        beta_operator[point1:point2] .= Int8(1)
    end

    return alpha_operator, beta_operator
end


@inline function create_operator_point_two_masks(gene_seq_alpha::Vector{Int8}, gene_seq_beta::Vector{Int8}, toolbox::Toolbox)
    alpha_operator = zeros(Int8, length(gene_seq_alpha))
    beta_operator = zeros(Int8, length(gene_seq_beta))
    head_len = toolbox.head_len
    gene_len = head_len * 2 + 1

    for i in toolbox.gen_start_indices
        start = i
        quarter = start + gene_len ÷ 4
        half = start + gene_len ÷ 2
        end_gene = start + gene_len - 1


        point1 = rand(start:quarter)
        point2 = rand(quarter+1:half)
        point3 = rand(half+1:end_gene)
        alpha_operator[point1:point2] .= Int8(1)
        alpha_operator[point3:end_gene] .= Int8(1)


        point1 = rand(start:end_gene)
        point2 = rand(point1:end_gene)
        beta_operator[point1:point2] .= Int8(1)
        beta_operator[point2+1:end_gene] .= Int8(1)
    end

    return alpha_operator, beta_operator
end

@inline function replicate(chromosome1::Chromosome, chromosome2::Chromosome, toolbox)
    return [Chromosome(deepcopy(chromosome1.genes), toolbox), Chromosome(deepcopy(chromosome2.genes), toolbox)]
end


@inline function gene_dominant_fusion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i]
        child_2_genes[i] = beta_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

@inline function gen_rezessiv!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i]
        child_2_genes[i] = beta_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

@inline function gene_fussion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? Int8((gene_seq_alpha[i] + gene_seq_beta[i]) ÷ 2) : gene_seq_alpha[i]
        child_2_genes[i] = beta_operator[i] == 1 ? Int8((gene_seq_alpha[i] + gene_seq_beta[i]) ÷ 2) : gene_seq_beta[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

@inline function gene_one_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_point_one_masks(gene_seq_alpha, gene_seq_beta, chromosome1.toolbox)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? gene_seq_alpha[i] : gene_seq_beta[i]
        child_2_genes[i] = beta_operator[i] == 1 ? gene_seq_beta[i] : gene_seq_alpha[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

@inline function gene_two_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_point_two_masks(gene_seq_alpha, gene_seq_beta, chromosome1.toolbox)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? gene_seq_alpha[i] : gene_seq_beta[i]
        child_2_genes[i] = beta_operator[i] == 1 ? gene_seq_beta[i] : gene_seq_alpha[i]
    end

    chromosome1.genes = child_1_genes
    chromosome2.genes = child_2_genes
end

@inline function gene_mutation!(chromosome1::Chromosome, pb::Real=0.25)
    gene_seq_alpha = chromosome1.genes
    alpha_operator, _ = create_operator_masks(gene_seq_alpha, gene_seq_alpha, pb)
    mutation_seq_1 = generate_chromosome(chromosome1.toolbox)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        gene_seq_alpha[i] = alpha_operator[i] == 1 ? mutation_seq_1.genes[i] : gene_seq_alpha[i]
    end
end

@inline function gene_inversion!(chromosome1::Chromosome)
    start_1 = rand(chromosome1.toolbox.gen_start_indices)
    reverse!(@view chromosome1.genes[start_1:chromosome1.toolbox.head_len])
end

@inline function gene_insertion!(chromosome::Chromosome)
    start_1 = rand(chromosome.toolbox.gen_start_indices)
    insert_pos = rand(start_1:(start_1+chromosome.toolbox.head_len-1))
    insert_sym = rand(chromosome.toolbox.tailsyms)
    chromosome.genes[insert_pos] = insert_sym
end

@inline function reverse_insertion!(chromosome::Chromosome)
    start_1 = rand(chromosome.toolbox.gen_start_indices)
    rolled_array = circshift(chromosome.genes[start_1:start_1+chromosome.toolbox.head_len-1], rand(1:chromosome.toolbox.head_len-1))
    chromosome.genes[start_1:start_1+chromosome.toolbox.head_len-1] = rolled_array
end

@inline function reverse_insertion_tail!(chromosome::Chromosome)
    start_1 = rand(chromosome.toolbox.gen_start_indices) + chromosome.toolbox.head_len + 1
    rolled_array = circshift(chromosome.genes[start_1:start_1+chromosome.toolbox.head_len-1], rand(1:chromosome.toolbox.head_len-1))
    chromosome.genes[start_1:start_1+chromosome.toolbox.head_len-1] = rolled_array
end


"""
    genetic_operations!(space_next::Vector{Chromosome}, i::Int, toolbox::Toolbox)

Apply genetic operations to chromosomes.

# Arguments
- `space_next::Vector{Chromosome}`: Population buffer
- `i::Int`: Starting index
- `toolbox::Toolbox`: Toolbox configuration

# Effects
Modifies chromosomes in place applying various genetic operations based on probabilities
"""

"""
    gene_mutation!(chromosome1::Chromosome, pb::Real=0.25)
    gene_inversion!(chromosome1::Chromosome)
    gene_insertion!(chromosome::Chromosome)
    reverse_insertion!(chromosome::Chromosome)
    reverse_insertion_tail!(chromosome::Chromosome)
    gene_one_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome)
    gene_two_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome)
    gene_dominant_fusion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    gen_rezessiv!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    gene_fussion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    gene_insertion!(chromosome1::Chromosome)
    reverse_insertion!(chromosome1::Chromosome)
    reverse_insertion_tail!(chromosome1::Chromosome)
    

Genetic operators for chromosome modification.

# Arguments
- `chromosome1`, `chromosome2`: Target chromosomes
- `pb`: Probability of modification

# Effects
Modify chromosome genes in place
"""
@inline function genetic_operations!(space_next::Vector{Chromosome}, i::Int, toolbox::Toolbox; generation::Int64, max_generation::Int64, parents::Vector{Chromosome})
    #allocate them within the space - create them once instead of n time 
    space_next[i:i+1] = replicate(space_next[i], space_next[i+1], toolbox)
    rand_space = rand(15)


    if rand_space[1] < toolbox.gep_probs["one_point_cross_over_prob"]
        gene_one_point_cross_over!(space_next[i], space_next[i+1])
    end

    if rand_space[2] < toolbox.gep_probs["two_point_cross_over_prob"]
        gene_two_point_cross_over!(space_next[i], space_next[i+1])
    end

    if rand_space[3] < toolbox.gep_probs["mutation_prob"]
        gene_mutation!(space_next[i], toolbox.gep_probs["mutation_rate"])
    end

    if rand_space[4] < toolbox.gep_probs["mutation_prob"]
        gene_mutation!(space_next[i+1], toolbox.gep_probs["mutation_rate"])
    end

    if rand_space[5] < toolbox.gep_probs["dominant_fusion_prob"]
        gene_dominant_fusion!(space_next[i], space_next[i+1], toolbox.gep_probs["fusion_rate"])
    end

    if rand_space[6] < toolbox.gep_probs["rezessiv_fusion_prob"]
        gen_rezessiv!(space_next[i], space_next[i+1], toolbox.gep_probs["rezessiv_fusion_rate"])
    end

    if rand_space[7] < toolbox.gep_probs["fusion_prob"]
        gene_fussion!(space_next[i], space_next[i+1], toolbox.gep_probs["fusion_rate"])
    end

    if rand_space[8] < toolbox.gep_probs["inversion_prob"]
        gene_inversion!(space_next[i])
    end

    if rand_space[9] < toolbox.gep_probs["inversion_prob"]
        gene_inversion!(space_next[i+1])
    end

    if rand_space[10] < toolbox.gep_probs["inversion_prob"]
        gene_insertion!(space_next[i])
    end

    if rand_space[11] < toolbox.gep_probs["inversion_prob"]
        gene_insertion!(space_next[i+1])
    end

    if rand_space[12] < toolbox.gep_probs["reverse_insertion"]
        reverse_insertion!(space_next[i])
    end

    if rand_space[13] < toolbox.gep_probs["reverse_insertion"]
        reverse_insertion!(space_next[i+1])
    end

    if rand_space[14] < toolbox.gep_probs["reverse_insertion_tail"]
        reverse_insertion_tail!(space_next[i+1])
    end

    if rand_space[15] < toolbox.gep_probs["reverse_insertion_tail"]
        reverse_insertion_tail!(space_next[i+1])
    end

end

@inline function prob_equation_djl(chromosome::Chromosome, coeff_count::Int, prob_data_set::AbstractArray)
    ret_val = zeros(coeff_count, 1)
    if !chromosome.compiled
        ret_val[:, 1] .= Inf
    else
        try
            if coeff_count > 1
                for e_index in 1:coeff_count
                    ret_val[e_index, 1] = mean(eval_tree_array(chromosome.compiled_function[e_index], prob_data_set, chromosome.toolbox.operators_)[1])
                end
            else
                ret_val[coeff_count, 1] = mean(eval_tree_array(chromosome.compiled_function, prob_data_set, chromosome.toolbox.operators_)[1])
            end
        catch e
            @error "Something went wrong $e"
            ret_val[:, 1] .= Inf
        end
    end
    return ret_val
end


@inline function equation_characterization_default(population::Vector{Chromosome}, n_samples::Int, prob_dataset::AbstractArray)
    len_extented_pop = length(population)
    coeff_count = isempty(population[1].toolbox.preamble_syms) ? 1 : length(population[1].toolbox.preamble_syms)
    features = zeros(coeff_count, len_extented_pop)
    Threads.@threads for p_index in eachindex(population)
        try
            features[:, p_index] .= prob_equation_djl(population[p_index], coeff_count, prob_dataset)
        catch
            features[:, p_index] .= Inf
        end
    end
    indices = select_n_samples_lhs(features, n_samples)
    return indices
end

end
