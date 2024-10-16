module SymbolicEntities

include("Util.jl")

using .GepUtils
using OrderedCollections

export Chromosome, Toolbox
export AbstractSymbol, FunctionalSymbol, BasicSymbol, SymbolConfig
export fitness, set_fitness!
export generate_gene, generate_preamle!, compile_expression!, generate_chromosome, generate_population 
export genetic_operations!



abstract type AbstractSymbol end

struct BasicSymbol <: AbstractSymbol
    representation::Union{String,Real}
    unit::Vector{Float16}
    index::Int
    arity::Int8
    feature::Bool
end

# Concrete type for functional symbols
struct FunctionalSymbol <: AbstractSymbol
    representation::String
    unit::Vector{Float16}
    index::Int
    arity::Int8
    arithmetic_operation::Function
    forward_function::Union{Function, Nothing}
    reverse_function::Union{Function, Nothing}
end


struct SymbolConfig
    basic_symbols::OrderedDict{Int8, BasicSymbol}
    constant_symbols::OrderedDict{Int8, BasicSymbol}
    functional_symbols::OrderedDict{Int8, FunctionalSymbol}
    callbacks::Dict{Int8,Function}
    operators_djl::Any
    nodes_djl::OrderedDict{Int8, Any}
    symbol_arity_mapping::OrderedDict{Int8, Int8}
    physical_operation_dict::Union{OrderedDict{Int8, Function},Nothing}
    physical_dimension_dict::OrderedDict{Int8, Vector{Float16}}
    features_idx::Vector{Int8}
    functions_idx::Vector{Int8}
    constants_idx::Vector{Int8}
    point_operations_idx::Vector{Int8}
    inverse_operations::Union{Dict{Int8, Function},Nothing}
end


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


    function Toolbox(gene_count::Int, head_len::Int, symbols::OrderedDict{Int8,Int8}, gene_connections::Vector{Int8},
        callbacks::Dict, nodes::OrderedDict, gep_probs::Dict{String,AbstractFloat};
        unary_prob::Real=0.4, fitness_reset::Tuple=(Inf, NaN), preamble_syms=Int8[])
        gene_len = head_len * 2 + 1
        headsyms = [key for (key, arity) in symbols if arity == 2]
        unary_syms = [key for (key, arity) in symbols if arity == 1]
        tailsyms = [key for (key, arity) in symbols if arity < 1 && !(key in preamble_syms)]
        len_preamble = length(preamble_syms) == 0 ? 0 : gene_count
        gen_start_indices = [gene_count + len_preamble + (gene_len * (i - 1)) for i in 1:gene_count] #depending on the usage should shift everthing 
        new(gene_count, head_len, symbols, gene_connections, headsyms, unary_syms, tailsyms, symbols,
            callbacks, nodes, gen_start_indices, gep_probs, unary_prob, fitness_reset, preamble_syms, len_preamble)
    end
end




mutable struct Chromosome
    genes::Vector{Int8}
    fitness::Union{AbstractFloat,Tuple}
    toolbox::Toolbox
    compiled_function::Any
    compiled::Bool
    fitness_r2_train::AbstractFloat
    fitness_r2_test::AbstractFloat
    expression_raw::Vector{Int8}
    dimension_homogene::Bool
    penalty::Real
    chromo_id::Int

    function Chromosome(genes::Vector{Int8}, toolbox::Toolbox, compile::Bool=false)
        obj = new()
        obj.genes = genes

        obj.fitness = toolbox.fitness_reset[2]
        obj.toolbox = toolbox
        obj.fitness_r2_train = 0.0
        obj.fitness_r2_test = 0.0
        obj.compiled = false
        obj.dimension_homogene = false
        obj.penalty = 0.0
        obj.chromo_id = -1
        if compile
            compile_expression!(obj)
        end
        return obj
    end
end

function compile_expression!(chromosome::Chromosome; force_compile::Bool=false)
    if !chromosome.compiled || force_compile
        try
            expression = _karva_raw(chromosome)
            expression_tree = compile_djl_datatype(expression, chromosome.toolbox.symbols, chromosome.toolbox.callbacks,
                chromosome.toolbox.nodes)
            chromosome.compiled_function = expression_tree
            chromosome.expression_raw = expression
            chromosome.fitness = chromosome.toolbox.fitness_reset[2]
            chromosome.compiled = true
        catch 
            #print(stacktrace())
            #becomes when the composition does not make sense according to algebraic rules!
            chromosome.fitness = chromosome.toolbox.fitness_reset[1]
        end
    end
end


function fitness(chromosome::Chromosome)
    return chromosome.fitness
end

function set_fitness!(chromosome::Chromosome, value::AbstractFloat)
    chromosome.fitness = value
end


function _karva_raw(chromosome::Chromosome)
    gene_len = chromosome.toolbox.head_len * 2 + 1
    gene_count = chromosome.toolbox.gene_count
    len_preamble = chromosome.toolbox.len_preamble

    connectionsym = @view chromosome.genes[1+len_preamble:gene_count+len_preamble-1]
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

function generate_gene(headsyms::Vector{Int8}, tailsyms::Vector{Int8}, headlen::Int; unarys::Vector{Int8}=[], unary_prob::Real=0.2)
    if !isempty(unarys) && rand() < unary_prob
        heads = vcat(headsyms)
        push!(heads, rand(unarys))
    else
        heads = headsyms
    end

    head = rand(heads, headlen)
    tail = rand(tailsyms, headlen + 1)
    return vcat(head, tail)
end


function generate_chromosome(toolbox::Toolbox)
    connectors = rand(toolbox.gene_connections, toolbox.gene_count - 1)
    genes = vcat([generate_gene(toolbox.headsyms, toolbox.tailsyms, toolbox.head_len; unarys=toolbox.unary_syms,
        unary_prob=toolbox.unary_prob) for _ in 1:toolbox.gene_count]...)
    return Chromosome(vcat(connectors, genes), toolbox, true)
end

function generate_population(number::Int, toolbox::Toolbox)
    population = Vector{Chromosome}(undef, number)
     for i in 1:number
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
    reverse!(@view chromosome1.genes[start_1:chromosome1.toolbox.head_len-1])
end

@inline function gene_insertion!(chromosome::Chromosome)
    start_1 = rand(chromosome.toolbox.gen_start_indices)
    insert_pos = rand(start_1:(start_1+chromosome.toolbox.head_len-1))
    insert_sym = rand(chromosome.toolbox.tailsyms)
    chromosome.genes[insert_pos] = insert_sym
end

@inline function reverse_insertion!(chromosome::Chromosome)
    start_1 = rand(chromosome.toolbox.gen_start_indices)
    rolled_array = circshift(chromosome.genes[start_1:chromosome.toolbox.head_len-1], rand(1:chromosome.toolbox.head_len-1))
    chromosome.genes[start_1:chromosome.toolbox.head_len-1] = rolled_array
end

@inline function genetic_operations!(space_next::Vector{Chromosome}, i::Int, toolbox::Toolbox)
    #allocate them within the space - create them once instead of n time 
    space_next[i:i+1] = replicate(space_next[i], space_next[i+1], toolbox)
    rand_space = rand(13)


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

    if rand_space[12] < toolbox.gep_probs["inversion_prob"]
        reverse_insertion!(space_next[i])
    end

    if rand_space[13] < toolbox.gep_probs["inversion_prob"]
        reverse_insertion!(space_next[i+1])
    end

end
end