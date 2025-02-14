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
export genetic_operations!, replicate, gene_inversion!, gene_mutation!, gene_one_point_cross_over!, gene_two_point_cross_over!, gene_fussion!

include("Util.jl")
include("TensorOps.jl")


using .GepUtils
using .TensorRegUtils
using OrderedCollections
using DynamicExpressions


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
    default_size = 1000
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
- `unary_syms::Vector{Int8}`: Unary operators
- `tailsyms::Vector{Int8}`: Symbols allowed in tail
- `arrity_by_id::OrderedDict{Int8,Int8}`: Symbol arities
- `callbacks::Dict`: Operation callbacks
- `nodes::OrderedDict`: Node definitions
- `gen_start_indices::Vector{Int}`: Gene start positions
- `gep_probs::Dict{String,AbstractFloat}`: Operation probabilities
- `unary_prob::Real`: Unary operator probability
- `fitness_reset::Tuple`: Default fitness values
- `preamble_syms::Vector{Int8}`: Preamble symbols
- `len_preamble::Int8`: Preamble length

# Constructor
    Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8}, 
           gene_connections::Vector{Int8}, callbacks::Dict, nodes::OrderedDict, 
           gep_probs::Dict{String,AbstractFloat}; unary_prob::Real=0.4, 
           fitness_reset::Tuple=(Inf, NaN), preamble_syms=Int8[])
"""
struct Toolbox
    gene_count::Int
    head_len::Int
    symbols::OrderedDict{Int8,Int8}
    gene_connections::Vector{Int8}
    headsyms::Vector{Int8}
    unary_syms::Vector{Int8}
    tailsyms::Vector{Int8}
    arrity_by_id::OrderedDict{Int8,Int8}
    callbacks::Dict
    nodes::OrderedDict
    gen_start_indices::Vector{Int}
    gep_probs::Dict{String,AbstractFloat}
    unary_prob::Real
    fitness_reset::Tuple
    preamble_syms::Vector{Int8}
    len_preamble::Int8
    operators_::Union{OperatorEnum,Nothing}
    compile_function_::Function


    function Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8}, gene_connections::Vector{Int8},
        callbacks::Dict, nodes::OrderedDict, gep_probs::Dict{String,AbstractFloat};
        unary_prob::Real=0.1, preamble_syms=Int8[],
        number_of_objectives::Int=1, operators_::Union{OperatorEnum,Nothing}=nothing, function_complile::Function=compile_djl_datatype)

        fitness_reset = (
            ntuple(_ -> Inf, number_of_objectives),
            ntuple(_ -> NaN, number_of_objectives)
        )
        gene_len = head_len * 2 + 1
        headsyms = [key for (key, arity) in symbols if arity == 2]
        unary_syms = [key for (key, arity) in symbols if arity == 1]
        tailsyms = [key for (key, arity) in symbols if arity < 1 && !(key in preamble_syms)]
        len_preamble = length(preamble_syms)
        gen_start_indices = [gene_count + (gene_len * (i - 1)) for i in 1:gene_count]
        ensure_buffer_size!(head_len, gene_count)
        new(gene_count, head_len, symbols, gene_connections, headsyms, unary_syms, tailsyms, symbols,
            callbacks, nodes, gen_start_indices, gep_probs, unary_prob, fitness_reset, preamble_syms, len_preamble, operators_, function_complile)
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
            @error "something went wrong" exception = (e, catch_backtrace())
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

@inline function _karva_raw(chromosome::Chromosome)
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

    return vcat(rolled_indices...)
end


"""
    generate_gene(headsyms::Vector{Int8}, tailsyms::Vector{Int8}, headlen::Int; 
                 unarys::Vector{Int8}=[], unary_prob::Real=0.2)

Generate a single gene for GEP.

# Arguments
- `headsyms::Vector{Int8}`: Symbols for head
- `tailsyms::Vector{Int8}`: Symbols for tail
- `headlen::Int`: Head length
- `unarys::Vector{Int8}=[]`: Unary operators
- `unary_prob::Real=0.2`: Unary operator probability

# Returns
Vector{Int8} representing gene
"""
@inline function generate_gene(headsyms::Vector{Int8}, tailsyms::Vector{Int8}, headlen::Int;
    unarys::Vector{Int8}=[], unary_prob::Real=0.2, tensor_prob::Real=0.2)
    if !isempty(unarys) && rand() < unary_prob
        heads = vcat(headsyms, tailsyms)
        push!(heads, rand(unarys))
    else
        heads = headsyms
    end

    head = rand(heads, headlen)
    tail = rand(tailsyms, headlen + 1)
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
    genes = vcat([generate_gene(toolbox.headsyms, toolbox.tailsyms, toolbox.head_len; unarys=toolbox.unary_syms,
        unary_prob=toolbox.unary_prob) for _ in 1:toolbox.gene_count]...)
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
@inline function genetic_operations!(space_next::Vector{Chromosome}, i::Int, toolbox::Toolbox)
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
end