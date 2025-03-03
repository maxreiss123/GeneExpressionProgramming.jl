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
module GepEntities


export Chromosome, Toolbox, EvaluationStrategy, StandardRegressionStrategy, GenericRegressionStrategy
export fitness, set_fitness!
export generate_gene, compile_expression!, generate_chromosome, generate_population
export genetic_operations!, replicate, gene_inversion!, gene_mutation!, gene_one_point_cross_over!, gene_two_point_cross_over!, gene_fussion!, split_karva, print_karva_strings

using ..GepUtils
using ..TensorRegUtils
using OrderedCollections
using DynamicExpressions
using StatsBase


"""
    Memory Buffer for reducing allocation during runtime! 
"""
struct GeneticBuffers
    alpha_operator::Vector{Int8}
    beta_operator::Vector{Int8}
    child_1_genes::Vector{Int8}
    child_2_genes::Vector{Int8}
    rolled_indices::Vector{Any}
    arity_gene::Vector{Int8}
end


const THREAD_BUFFERS = let
    default_size = 256
    [GeneticBuffers(
        zeros(Int8, default_size),
        zeros(Int8, default_size),
        Vector{Int8}(undef, default_size),
        Vector{Int8}(undef, default_size),
        Vector{Any}(undef, default_size),
        Vector{Int8}(undef, default_size)
    ) for _ in 1:Threads.nthreads()]
end


function ensure_buffer_size!(head_len::Int, gene_count::Int)
    gene_len = head_len * 2 + 1
    total_gene_length = gene_count - 1 + gene_count * gene_len

    for buffer in THREAD_BUFFERS
        if length(buffer.alpha_operator) < total_gene_length
            resize!(buffer.alpha_operator, total_gene_length)
            resize!(buffer.beta_operator, total_gene_length)
            resize!(buffer.child_1_genes, total_gene_length)
            resize!(buffer.child_2_genes, total_gene_length)
            resize!(buffer.rolled_indices, gene_count + 1)
            resize!(buffer.arity_gene, gene_count * gene_len)
        end
    end
end

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
    operators::Union{OperatorEnum,Nothing}
    number_of_objectives::Int
    loss_function::Function
    secOptimizer::Union{Function,Nothing}
    break_condition::Union{Function,Nothing}

    function GenericRegressionStrategy(operators::Union{OperatorEnum,Nothing}, number_of_objectives::Int, loss_function::Function;
        secOptimizer::Union{Function,Nothing}, break_condition::Union{Function,Nothing})
        new(operators, number_of_objectives, loss_function, secOptimizer, break_condition)
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
    compile_function_::Function
    tail_weights::Union{Weights,Nothing}
    head_weights::Union{Weights,Nothing}


    function Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8}, gene_connections::Vector{Int8},
        callbacks::Dict, nodes::OrderedDict, gep_probs::Dict{String,AbstractFloat};
        unary_prob::Real=0.1, preamble_syms=Int8[],
        number_of_objectives::Int=1, operators_::Union{OperatorEnum,Nothing}=nothing, 
        function_complile::Function=compile_djl_datatype,
        tail_weights_::Union{Weights,Nothing}=nothing, 
        head_tail_balance::Real=0.8)

        fitness_reset = (
            ntuple(_ -> Inf, number_of_objectives),
            ntuple(_ -> NaN, number_of_objectives)
        )
        gene_len = head_len * 2 + 1
        headsyms = [key for (key, arity) in symbols if arity == 2]
        unary_syms = [key for (key, arity) in symbols if arity == 1]
        up = isempty(unary_syms) ? unary_prob : 0

        tailsyms = [key for (key, arity) in symbols if arity < 1 && !(key in preamble_syms)]
        len_preamble = length(preamble_syms)
        gen_start_indices = [gene_count + (gene_len * (i - 1)) for i in 1:gene_count]
        

        tail_weights = isnothing(tail_weights_) ? weights([1/length(tailsyms) for _ in 1:length(tailsyms)]) : tail_weights_
        head_weights = weights([
            fill(head_tail_balance/length(headsyms), length(headsyms));
            fill(unary_prob/length(unary_syms), length(unary_syms));
            tail_weights .* (1-head_tail_balance-up)
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
    genes = chromosome.genes[gene_count:end]

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
    gene_count_per_factor = div(chromosome.toolbox.gene_count,coeffs)
    retval = []
    for _ in 1:coeffs
        temp_cons = splice!(connectors, 1:gene_count_per_factor-1)
        temp_genes = reduce(vcat, splice!(raw,1:gene_count_per_factor))
        push!(retval,vcat([temp_cons, temp_genes]...))
    end
    return retval
end

@inline function print_karva_strings(chromosome::Chromosome)
    coeff_count = length(chromosome.toolbox.preamble_syms)
    callback_ = Dict{Int8, Function}()

    for (key, value) in chromosome.toolbox.callbacks
        if Symbol(value) in keys(FUNCTION_STRINGIFY)
            callback_[key] = FUNCTION_STRINGIFY[Symbol(value)]
        elseif Symbol(value) in keys(TENSOR_STRINGIFY)
            callback_[key] = TENSOR_STRINGIFY[Symbol(value)]
        end
    end

    return compile_djl_datatype(
        chromosome.expression_raw, 
        chromosome.toolbox.arrity_by_id,
        callback_, 
        chromosome.toolbox.nodes, 
        1)
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
    head = sample(headsyms,head_weights,headlen)
    tail = sample(tailsyms,tail_weights,headlen + 1)
    return vcat(head, tail)
end



"""
    generate_chromosome(toolbox::Toolbox)

Generate a new chromosome using toolbox configuration.

# Returns
New Chromosome instance
"""
@inline function generate_chromosome(toolbox::Toolbox)
    connectors = rand(toolbox.gene_connections, toolbox.gene_count - 1)
    genes = vcat([generate_gene(toolbox.headsyms, toolbox.tailsyms, toolbox.head_len, toolbox.tail_weights,
        toolbox.head_weights) for _ in 1:toolbox.gene_count]...)
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
    buffer = THREAD_BUFFERS[Threads.threadid()]
    len_a = length(gene_seq_alpha)

    buffer.alpha_operator[1:len_a] .= zeros(Int8)
    buffer.beta_operator[1:len_a] .= zeros(Int8)

    indices_alpha = rand(1:len_a, min(round(Int, (pb * len_a)), len_a))
    indices_beta = rand(1:len_a, min(round(Int, (pb * len_a)), len_a))

    buffer.alpha_operator[indices_alpha] .= Int8(1)
    buffer.beta_operator[indices_beta] .= Int8(1)
end


@inline function create_operator_point_one_masks(gene_seq_alpha::Vector{Int8}, gene_seq_beta::Vector{Int8}, toolbox::Toolbox)
    buffer = THREAD_BUFFERS[Threads.threadid()]
    len_a = length(gene_seq_alpha)

    buffer.alpha_operator[1:len_a] .= zeros(Int8)
    buffer.beta_operator[1:len_a] .= zeros(Int8)

    head_len = toolbox.head_len
    gene_len = head_len * 2 + 1

    for i in toolbox.gen_start_indices
        ref = i
        mid = ref + gene_len ÷ 2

        point1 = rand(ref:mid)
        point2 = rand((mid+1):(ref+gene_len-1))
        buffer.alpha_operator[point1:point2] .= Int8(1)

        point1 = rand(ref:mid)
        point2 = rand((mid+1):(ref+gene_len-1))
        buffer.beta_operator[point1:point2] .= Int8(1)
    end
end


@inline function create_operator_point_two_masks(gene_seq_alpha::Vector{Int8}, gene_seq_beta::Vector{Int8}, toolbox::Toolbox)
    buffer = THREAD_BUFFERS[Threads.threadid()]
    len_a = length(gene_seq_alpha)

    buffer.alpha_operator[1:len_a] .= zeros(Int8)
    buffer.beta_operator[1:len_a] .= zeros(Int8)
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
        buffer.alpha_operator[point1:point2] .= Int8(1)
        buffer.alpha_operator[point3:end_gene] .= Int8(1)


        point1 = rand(start:end_gene)
        point2 = rand(point1:end_gene)
        buffer.beta_operator[point1:point2] .= Int8(1)
        buffer.beta_operator[point2+1:end_gene] .= Int8(1)
    end


end

@inline function replicate(chromosome1::Chromosome, chromosome2::Chromosome, toolbox)
    return [Chromosome(deepcopy(chromosome1.genes), toolbox), Chromosome(deepcopy(chromosome2.genes), toolbox)]
end


@inline function gene_dominant_fusion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    buffer = THREAD_BUFFERS[Threads.threadid()]
    len_a = length(chromosome1.genes)
    create_operator_masks(chromosome1.genes, chromosome2.genes, pb)

    @inbounds @simd for i in eachindex(chromosome1.genes)
        buffer.child_1_genes[i] = buffer.alpha_operator[i] == 1 ? max(chromosome1.genes[i], chromosome2.genes[i]) : chromosome1.genes[i]
        buffer.child_2_genes[i] = buffer.beta_operator[i] == 1 ? max(chromosome1.genes[i], chromosome2.genes[i]) : chromosome2.genes[i]
    end

    chromosome1.genes .= @view buffer.child_1_genes[1:len_a]
    chromosome2.genes .= @view buffer.child_2_genes[1:len_a]
end

@inline function gen_rezessiv!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    buffer = THREAD_BUFFERS[Threads.threadid()]
    len_a = length(chromosome1.genes)
    create_operator_masks(chromosome1.genes, chromosome2.genes, pb)

    @inbounds @simd for i in eachindex(chromosome1.genes)
        buffer.child_1_genes[i] = buffer.alpha_operator[i] == 1 ? min(chromosome1.genes[i], chromosome2.genes[i]) : chromosome1.genes[i]
        buffer.child_2_genes[i] = buffer.beta_operator[i] == 1 ? min(chromosome1.genes[i], chromosome2.genes[i]) : chromosome2.genes[i]
    end

    chromosome1.genes .= @view buffer.child_1_genes[1:len_a]
    chromosome2.genes .= @view buffer.child_2_genes[1:len_a]
end

@inline function gene_fussion!(chromosome1::Chromosome, chromosome2::Chromosome, pb::Real=0.2)
    buffer = THREAD_BUFFERS[Threads.threadid()]
    len_a = length(chromosome1.genes)
    create_operator_masks(chromosome1.genes, chromosome2.genes, pb)

    @inbounds @simd for i in eachindex(chromosome1.genes)
        buffer.child_1_genes[i] = buffer.alpha_operator[i] == 1 ? Int8((chromosome1.genes[i] + chromosome2.genes[i]) ÷ 2) : chromosome1.genes[i]
        buffer.child_2_genes[i] = buffer.beta_operator[i] == 1 ? Int8((chromosome1.genes[i] + chromosome2.genes[i]) ÷ 2) : chromosome2.genes[i]
    end

    chromosome1.genes .= @view buffer.child_1_genes[1:len_a]
    chromosome2.genes .= @view buffer.child_2_genes[1:len_a]
end

@inline function gene_one_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome)
    buffer = THREAD_BUFFERS[Threads.threadid()]
    len_a = length(chromosome1.genes)
    create_operator_point_one_masks(chromosome1.genes, chromosome2.genes, chromosome1.toolbox)

    @inbounds @simd for i in eachindex(chromosome1.genes)
        buffer.child_1_genes[i] = buffer.alpha_operator[i] == 1 ? chromosome1.genes[i] : chromosome2.genes[i]
        buffer.child_2_genes[i] = buffer.beta_operator[i] == 1 ? chromosome2.genes[i] : chromosome1.genes[i]
    end

    chromosome1.genes .= @view buffer.child_1_genes[1:len_a]
    chromosome2.genes .= @view buffer.child_2_genes[1:len_a]
end

@inline function gene_two_point_cross_over!(chromosome1::Chromosome, chromosome2::Chromosome)
    buffer = THREAD_BUFFERS[Threads.threadid()]
    len_a = length(chromosome1.genes)
    create_operator_point_two_masks(chromosome1.genes, chromosome2.genes, chromosome1.toolbox)

    @inbounds @simd for i in eachindex(chromosome1.genes)
        buffer.child_1_genes[i] = buffer.alpha_operator[i] == 1 ? chromosome1.genes[i] : chromosome2.genes[i]
        buffer.child_2_genes[i] = buffer.beta_operator[i] == 1 ? chromosome2.genes[i] : chromosome1.genes[i]
    end

    chromosome1.genes .= @view buffer.child_1_genes[1:len_a]
    chromosome2.genes .= @view buffer.child_2_genes[1:len_a]
end

@inline function gene_mutation!(chromosome1::Chromosome, pb::Real=0.25)
    buffer = THREAD_BUFFERS[Threads.threadid()]
    create_operator_masks(chromosome1.genes, chromosome1.genes, pb)
    buffer.child_1_genes[1:length(chromosome1.genes)] .= generate_chromosome(chromosome1.toolbox).genes[1:length(chromosome1.genes)]

    @inbounds @simd for i in eachindex(chromosome1.genes)
        chromosome1.genes[i] = buffer.alpha_operator[i] == 1 ? buffer.child_1_genes[i] : chromosome1.genes[i]
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


@inline function gene_fussion_extent!(chromosome1::Chromosome, parents::Vector{Chromosome}, pb::Real=0.2; topk::Int=1)
    buffer = THREAD_BUFFERS[Threads.threadid()]
    len_a = length(chromosome1.genes)
    genes2 = one_hot_mean([p.genes for p in parents], topk)

    create_operator_masks(chromosome1.genes, genes2, pb)

    @inbounds @simd for i in eachindex(chromosome1.genes)
        buffer.child_1_genes[i] = buffer.alpha_operator[i] == 1 ? genes2[i] : chromosome1.genes[i]
    end

    chromosome1.genes .= @view buffer.child_1_genes[1:len_a]
end


"""
    diversity_injection!(chromosome::Chromosome, diversity_factor::Real=0.5)

Injects diversity by randomly replacing portions of genes with completely new material.
Useful for avoiding premature convergence.

# Arguments
- `chromosome::Chromosome`: Target chromosome
- `diversity_factor::Real=0.5`: Controls how much of the chromosome to randomize
"""
@inline function diversity_injection!(chromosome::Chromosome, diversity_factor::Real=0.5)
    gene_len = chromosome.toolbox.head_len * 2 + 1
    gene_count = chromosome.toolbox.gene_count
    
    genes_to_randomize = max(1, round(Int, diversity_factor * gene_count))
    
    genes_indices = sample(1:gene_count, genes_to_randomize, replace=false)
    
    for gene_idx in genes_indices
        gene_start = chromosome.toolbox.gen_start_indices[gene_idx]
        
        new_gene = generate_gene(
            chromosome.toolbox.headsyms, 
            chromosome.toolbox.tailsyms, 
            chromosome.toolbox.head_len,
            chromosome.toolbox.tail_weights,
            chromosome.toolbox.head_weights
        )
        
        chromosome.genes[gene_start:(gene_start + gene_len - 1)] .= new_gene
    end
end

"""
    adaptive_mutation!(chromosome::Chromosome, generation::Int, max_generations::Int, pb_start::Real=0.4, pb_end::Real=0.1)

Adaptive mutation operator that adjusts mutation rate based on generation progress.
Higher mutation rate early for exploration, lower later for exploitation.

# Arguments
- `chromosome::Chromosome`: Target chromosome
- `generation::Int`: Current generation
- `max_generations::Int`: Maximum generations for the run
- `pb_start::Real=0.4`: Starting probability (higher for exploration)
- `pb_end::Real=0.1`: Ending probability (lower for exploitation)
"""
@inline function adaptive_mutation!(chromosome::Chromosome, generation::Int, max_generations::Int; 
        pb_start::Real=0.4, pb_end::Real=0.1)
    progress = generation / max_generations
    adaptive_pb = pb_start - (pb_start - pb_end) * progress
    
    buffer = THREAD_BUFFERS[Threads.threadid()]
    create_operator_masks(chromosome.genes, chromosome.genes, adaptive_pb)
    buffer.child_1_genes[1:length(chromosome.genes)] .= generate_chromosome(chromosome.toolbox).genes[1:length(chromosome.genes)]

    @inbounds @simd for i in eachindex(chromosome.genes)
        chromosome.genes[i] = buffer.alpha_operator[i] == 1 ? buffer.child_1_genes[i] : chromosome.genes[i]
    end
end

"""
    gene_transposition!(chromosome::Chromosome, len::Int=3)

Transposes a small segment of genes from one position to another within the same chromosome,
preserving their order but changing their context.

# Arguments
- `chromosome::Chromosome`: Target chromosome
- `len::Int=3`: Length of segment to transpose
"""
@inline function gene_transposition!(chromosome::Chromosome, len::Int=3)
    gene_len = chromosome.toolbox.head_len * 2 + 1
    gene_count = chromosome.toolbox.gene_count
    
    source_gene_idx = rand(1:gene_count)
    target_gene_idx = rand([i for i in 1:gene_count if i != source_gene_idx])
    
    source_start = chromosome.toolbox.gen_start_indices[source_gene_idx]
    target_start = chromosome.toolbox.gen_start_indices[target_gene_idx]
    
    segment_len = min(len, gene_len - 1)
    
    source_pos = rand(source_start:(source_start + gene_len - segment_len - 1))
    target_pos = rand(target_start:(target_start + gene_len - segment_len - 1))
    
    segment = copy(chromosome.genes[source_pos:(source_pos + segment_len - 1)])
    
    target_region = chromosome.genes[target_pos:(target_pos + segment_len - 1)]
    chromosome.genes[target_pos:(target_pos + segment_len - 1)] .= segment
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

Genetic operators for chromosome modification.

# Arguments
- `chromosome1`, `chromosome2`: Target chromosomes
- `pb`: Probability of modification

# Effects
Modify chromosome genes in place
"""
@inline function genetic_operations!(space_next::Vector{Chromosome}, i::Int, toolbox::Toolbox;
        generation::Int=0, max_generation::Int=0, parents::Union{Vector{Chromosome},Nothing}=nothing)
    #allocate them within the space - create them once instead of n time 
    space_next[i:i+1] = replicate(space_next[i], space_next[i+1], toolbox)
    rand_space = rand(17)


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

    if rand_space[16] < toolbox.gep_probs["gene_transposition"]
        gene_transposition!(space_next[i])
    end

    if rand_space[17] < toolbox.gep_probs["gene_transposition"]
        gene_transposition!(space_next[i+1])
    end

    if rand_space[16] < toolbox.gep_probs["gene_averaging_prob"]
        gene_fussion_extent!(space_next[i], parents, toolbox.gep_probs["gene_averaging_rate"])
    end

    if rand_space[17] < toolbox.gep_probs["gene_averaging_prob"]
        gene_fussion_extent!(space_next[i+1], parents, toolbox.gep_probs["gene_averaging_rate"])
    end


    

end
end
