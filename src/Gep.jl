module GepRegressor

include("Losses.jl")
include("Util.jl")
include("Selection.jl")

using .LossFunction
using .GepUtils
using .EvoSelection

using Random
using Statistics
using LinearAlgebra
using ProgressMeter
using OrderedCollections
using DynamicExpressions
using Logging


export Chromosome
export runGep



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
        heads = vcat(headsyms,tailsyms)
        push!(heads, rand(unarys))
    else
        heads = headsyms
    end

    head = rand(heads, headlen)
    tail = rand(tailsyms, headlen + 1)
    return vcat(head, tail)
end

function generate_preamle!(toolbox::Toolbox, preamble::Vector{Int8})
    if !isempty(toolbox.preamble_syms)
        append!(preamble, rand(toolbox.preamble_syms, toolbox.gene_count))
    end
end

function generate_chromosome(toolbox::Toolbox)
    connectors = rand(toolbox.gene_connections, toolbox.gene_count - 1)
    preamble_sym = Int8[]
    generate_preamle!(toolbox, preamble_sym)
    genes = vcat([generate_gene(toolbox.headsyms, toolbox.tailsyms, toolbox.head_len; unarys=toolbox.unary_syms,
        unary_prob=toolbox.unary_prob) for _ in 1:toolbox.gene_count]...)
    return Chromosome(vcat(preamble_sym, connectors, genes), toolbox, true)
end


function generate_population(number::Int, toolbox::Toolbox)
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


@inline function gene_dominant_fusion(chromosome1::Chromosome, chromosome2::Chromosome, toolbox::Toolbox, pb::Real=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i]
        child_2_genes[i] = beta_operator[i] == 1 ? max(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i]
    end

    return [Chromosome(child_1_genes, toolbox), Chromosome(child_2_genes, toolbox)]
end

@inline function gen_rezessiv(chromosome1::Chromosome, chromosome2::Chromosome, toolbox::Toolbox, pb::Real=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_alpha[i]
        child_2_genes[i] = beta_operator[i] == 1 ? min(gene_seq_alpha[i], gene_seq_beta[i]) : gene_seq_beta[i]
    end

    return [Chromosome(child_1_genes, toolbox), Chromosome(child_2_genes, toolbox)]
end

@inline function gene_fussion(chromosome1::Chromosome, chromosome2::Chromosome, toolbox::Toolbox, pb::Real=0.2)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_masks(gene_seq_alpha, gene_seq_beta, pb)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? Int8((gene_seq_alpha[i] + gene_seq_beta[i]) ÷ 2) : gene_seq_alpha[i]
        child_2_genes[i] = beta_operator[i] == 1 ? Int8((gene_seq_alpha[i] + gene_seq_beta[i]) ÷ 2) : gene_seq_beta[i]
    end

    return [Chromosome(child_1_genes, toolbox), Chromosome(child_2_genes,toolbox)]
end

@inline function gene_one_point_cross_over(chromosome1::Chromosome, chromosome2::Chromosome, toolbox::Toolbox)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_point_one_masks(gene_seq_alpha, gene_seq_beta, chromosome1.toolbox)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? gene_seq_alpha[i] : gene_seq_beta[i]
        child_2_genes[i] = beta_operator[i] == 1 ? gene_seq_beta[i] : gene_seq_alpha[i]
    end

    return [Chromosome(child_1_genes, toolbox), Chromosome(child_2_genes, toolbox)]
end

@inline function gene_two_point_cross_over(chromosome1::Chromosome, chromosome2::Chromosome, toolbox::Toolbox)
    gene_seq_alpha = chromosome1.genes
    gene_seq_beta = chromosome2.genes
    alpha_operator, beta_operator = create_operator_point_two_masks(gene_seq_alpha, gene_seq_beta, chromosome1.toolbox)

    child_1_genes = similar(gene_seq_alpha)
    child_2_genes = similar(gene_seq_beta)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? gene_seq_alpha[i] : gene_seq_beta[i]
        child_2_genes[i] = beta_operator[i] == 1 ? gene_seq_beta[i] : gene_seq_alpha[i]
    end

    return [Chromosome(child_1_genes, toolbox), Chromosome(child_2_genes, toolbox)]
end

@inline function gene_mutation(chromosome1::Chromosome, toolbox::Toolbox, pb::Real=0.25)
    gene_seq_alpha = chromosome1.genes
    alpha_operator, _ = create_operator_masks(gene_seq_alpha, gene_seq_alpha, pb)
    mutation_seq_1 = generate_chromosome(chromosome1.toolbox)

    child_1_genes = similar(gene_seq_alpha)

    @inbounds @simd for i in eachindex(gene_seq_alpha)
        child_1_genes[i] = alpha_operator[i] == 1 ? mutation_seq_1.genes[i] : gene_seq_alpha[i]
    end

    return Chromosome(child_1_genes, toolbox)
end

@inline function gene_inversion(chromosome1::Chromosome, toolbox::Toolbox)
    start_1 = rand(chromosome1.toolbox.gen_start_indices)
    gene_1 = copy(chromosome1.genes)
    reverse!(@view gene_1[start_1:chromosome1.toolbox.head_len-1])
    return Chromosome(gene_1, toolbox)
end

@inline function gene_insertion(chromosome::Chromosome, toolbox::Toolbox)
    start_1 = rand(chromosome.toolbox.gen_start_indices)
    insert_pos = rand(start_1:(start_1+chromosome.toolbox.head_len-1))
    insert_sym = rand(chromosome.toolbox.tailsyms)
    gene_1 = copy(chromosome.genes)
    gene_1[insert_pos] = insert_sym
    return Chromosome(gene_1, toolbox)
end

@inline function compute_fitness(elem::Chromosome, operators::OperatorEnum, x_data::AbstractArray{T}, y_data::AbstractArray{T}, loss_function::Function,
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


@inline function genetic_operations!(space_next::Vector{Chromosome}, i::Int, toolbox::Toolbox)
    if rand() < toolbox.gep_probs["one_point_cross_over_prob"]
        space_next[i:i+1] = gene_one_point_cross_over(space_next[i], space_next[i+1],toolbox)
    end

    if rand() < toolbox.gep_probs["two_point_cross_over_prob"]
        space_next[i:i+1] = gene_two_point_cross_over(space_next[i], space_next[i+1],toolbox)
    end

    if rand() < toolbox.gep_probs["mutation_prob"]
        space_next[i] = gene_mutation(space_next[i], toolbox,toolbox.gep_probs["mutation_rate"])
    end

    if rand() < toolbox.gep_probs["mutation_prob"]
        space_next[i+1] = gene_mutation(space_next[i+1], toolbox, toolbox.gep_probs["mutation_rate"])
    end

    if rand() < toolbox.gep_probs["dominant_fusion_prob"]
        space_next[i:i+1] = gene_dominant_fusion(space_next[i], space_next[i+1], toolbox,toolbox.gep_probs["fusion_rate"])
    end

    if rand() < toolbox.gep_probs["rezessiv_fusion_prob"]
        space_next[i:i+1] = gen_rezessiv(space_next[i], space_next[i+1], toolbox, toolbox.gep_probs["rezessiv_fusion_rate"])
    end

    if rand() < toolbox.gep_probs["fusion_prob"]
        space_next[i:i+1] = gene_fussion(space_next[i], space_next[i+1], toolbox,toolbox.gep_probs["fusion_rate"])
    end

    if rand() < toolbox.gep_probs["inversion_prob"]
        space_next[i] = gene_inversion(space_next[i], toolbox)
    end

    if rand() < toolbox.gep_probs["inversion_prob"]
        space_next[i+1] = gene_inversion(space_next[i+1], toolbox)
    end

    if rand() < toolbox.gep_probs["inversion_prob"]
        space_next[i] = gene_insertion(space_next[i], toolbox)
    end

    if rand() < toolbox.gep_probs["inversion_prob"]
        space_next[i+1] = gene_insertion(space_next[i+1], toolbox)
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
    preamble_syms=Int8[]) where {T<:AbstractFloat}
    loss_fun::Function = get_loss_function(loss_fun_str)

    if isnothing(x_data_test) || isnothing(y_data_test)
        x_data_test = x_data
        y_data_test = y_data
    end

    Random.seed!(seed)
    mating_size = Int(ceil(population_size * mating_))
    mating_size = mating_size % 2 == 0 ? mating_size : mating_size - 1
    toolbox = Toolbox(gene_count, head_len, symbols, gene_connections,
        callbacks, nodes, gep_probs; preamble_syms=preamble_syms)

    population = generate_population(population_size, toolbox)
    next_gen = Vector{eltype(population)}(undef, mating_size)

    prev_best = -1
    @showprogress for epoch in 1:epochs
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

        Threads.@threads for i in eachindex(population)
            if isnan(population[i].fitness)
                population[i].fitness = compute_fitness(population[i], operators, x_data, y_data, loss_fun, typemax(T);
                    penalty_consideration=population[i].penalty * penalty_consideration)
            end
        end

        sort!(population, by=x -> x.fitness)

	try
            if (prev_best == -1 || prev_best > population[1].fitness) && epoch % 500 == 0
                eqn, result = optimize_constants(population[1].compiled_function,
                    x_data, y_data, get_loss_function(loss_fun_str), operators; opt_method=opt_method_const, max_iterations=250, n_restarts=3)
                population[1].fitness = result
                population[1].compiled_function = eqn
                prev_best = result
            end
	catch 
		@show "Opt. issue"
	end


        if epoch < epochs
            fits_representation = [chromo.fitness for chromo in population]
            indices = basic_tournament_selection(fits_representation, tourni_size, mating_size)
            parents = population[indices]
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
            best_r = compute_fitness(population[1], operators, x_data, y_data,
                get_loss_function("r2_score"), zero(T); validate=true)
            if isclose(best_r, one(T))
                break
            end

        end
    end

    best = sort(population, by=x -> x.fitness)[1:hof]
    

    for elem in best
        elem.fitness_r2_train = compute_fitness(elem, operators, x_data, y_data, get_loss_function("r2_score"), zero(T); validate=true)
        if !isnothing(x_data_test)
            elem.fitness_r2_test = compute_fitness(elem, operators, x_data_test, y_data_test, get_loss_function("r2_score"), zero(T); validate=true)
        end
    end
    return best
end
end
