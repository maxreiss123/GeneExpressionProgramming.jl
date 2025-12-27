module EvoSelection
using LinearAlgebra
using Random

export tournament_selection, nsga_selection, dominates_, fast_non_dominated_sort, calculate_fronts, determine_ranks, assign_crowding_distance
export moga_selection, determine_moga_ranks

struct SelectedMembers
    indices::Vector{Int}
    fronts::Dict{Int,Vector{Int}}
end

const STD_RNG = MersenneTwister()

@inline function tournament_selection(population::AbstractArray{Tuple}, number_of_winners::Int, tournament_size::Int; rng::AbstractRNG=STD_RNG)
    selected_indices = Vector{Int}(undef, number_of_winners)
    valid_indices_ = findall(x -> isfinite(x[1]), population)
    valid_indices = []
    doubles = Set()
    for elem in valid_indices_
        if !(population[elem] in doubles)
            push!(doubles, population[elem])
            push!(valid_indices, elem)
        end
    end

    for index in 1:number_of_winners
        if index == number_of_winners
            selected_indices[index] = 1
        else
            contenders = rand(rng, valid_indices, min(tournament_size, length(valid_indices)))
            winner = reduce((best, contender) -> population[contender] <= population[best] ? contender : best, contenders)
            selected_indices[index] = winner
        end
    end
    return SelectedMembers(selected_indices, Dict{Int,Vector{Int}}())
end

function count_infinites(t::Tuple)
    return count(isinf, t) + count(isnan, t)
end

function dominates_(a::Tuple, b::Tuple)
    a_inf_count = count_infinites(a)
    b_inf_count = count_infinites(b)

    if b_inf_count > a_inf_count
        return true
    elseif a_inf_count > b_inf_count
        return false
    end

    better_in_at_least_one = false
    not_worse_in_any = true

    for i in eachindex(a)
        ai, bi = a[i], b[i]
        if ai < bi
            better_in_at_least_one = true
        elseif ai > bi
            not_worse_in_any = false
        end
    end
    return better_in_at_least_one && not_worse_in_any
end

@inline function determine_ranks(pop::Vector{T}) where {T<:Tuple}
    n = length(pop)
    dom_list = [Int[] for _ in 1:n]
    rank = zeros(Int, n)
    dom_count = zeros(Int, n)

    for i in 1:n
        for j in i+1:n
            i_dominates_j = dominates_(pop[i], pop[j])
            j_dominates_i = dominates_(pop[j], pop[i])

            if i_dominates_j
                push!(dom_list[i], j)
                dom_count[j] += 1
            elseif j_dominates_i
                push!(dom_list[j], i)
                dom_count[i] += 1
            end
        end
        if dom_count[i] == 0
            rank[i] = 1
        end
    end

    k = 2
    while any(==(k - 1), rank)
        for p in 1:n
            if rank[p] == k - 1
                for q in dom_list[p]
                    dom_count[q] -= 1
                    if dom_count[q] == 0
                        rank[q] = k
                    end
                end
            end
        end
        k += 1
    end
    return rank
end

@inline function fast_non_dominated_sort(population::Vector{T}) where {T<:Tuple}
    ranks = determine_ranks(population)
    pop_indices = [(index, rank) for (index, rank) in enumerate(ranks)]
    sort!(pop_indices, by=x -> x[2])
    return [elem[1] for elem in pop_indices]
end

@inline function calculate_fronts(population::Vector{T}) where {T<:Tuple}
    ranks = determine_ranks(population)
    min_rank = minimum(ranks)
    max_rank = maximum(ranks)

    fronts = [Int[] for _ in min_rank:max_rank]

    for (i, r) in enumerate(ranks)
        push!(fronts[r-min_rank+1], i)
    end

    filter!(!isempty, fronts)
    return fronts
end

@inline function assign_crowding_distance(front::Vector{Int}, population::Vector{T}) where {T<:Tuple}
    n = length(front)
    objectives_count = length(first(population))

    distances = Dict{Int,Float64}()
    for i in front
        distances[i] = 0.0
    end

    for m in 1:objectives_count
        values = [population[i][m] for i in front]
        if length(unique(values)) < length(values)
            indexed_values = [(val, idx) for (idx, val) in enumerate(values)]
            sorted_indices = sortperm(indexed_values, by=x -> (x[1], x[2]))
            sorted_front = [front[i] for i in sorted_indices]
        else
            sorted_front = sort(front, by=i -> population[i][m])
        end

        distances[sorted_front[1]] = Inf
        distances[sorted_front[end]] = Inf

        if n > 2
            obj_range = population[sorted_front[end]][m] - population[sorted_front[1]][m]
            if obj_range > 0
                for i in 2:n-1
                    prev = population[sorted_front[i-1]][m]
                    next = population[sorted_front[i+1]][m]
                    distances[sorted_front[i]] += (next - prev) / obj_range
                end
            else
                for i in sorted_front
                    distances[i] = Inf
                end
                break
            end
        end
    end
    return distances
end

function tournament_selection_nsga(pop_indices::Vector{Int}, ranks::Vector{Int},
    crowding_distances::Dict{Int,Float64}, number_of_winners::Int, tournament_size::Int; rng::AbstractRNG=STD_RNG)
    selected_indices = Vector{Int}(undef, number_of_winners)

    finite_distances = [d for d in values(crowding_distances) if isfinite(d)]
    if !isempty(finite_distances)
        d_max = maximum(finite_distances)
        d_min = minimum(finite_distances)
        d_range = d_max - d_min
        normalized_distances = Dict{Int,Float64}()
        for (i, d) in crowding_distances
            if isfinite(d) && d_range > 0
                normalized_distances[i] = 0.99 * (d - d_min) / d_range
            else
                normalized_distances[i] = isfinite(d) ? 0.0 : 0.99
            end
        end
    else
        normalized_distances = Dict(i => 0.0 for i in pop_indices)
    end

    for i in 1:number_of_winners
        contenders = rand(rng, pop_indices, tournament_size)
        winner = reduce(contenders; init=contenders[1]) do best, contender
            if ranks[contender] < ranks[best]
                contender
            elseif ranks[contender] > ranks[best]
                best
            else
                normalized_distances[contender] > normalized_distances[best] ? contender : best
            end
        end
        selected_indices[i] = winner
    end
    return selected_indices
end

function nsga_selection(population::Vector{T}; tournament_size::Int=3, rng::AbstractRNG=STD_RNG) where {T<:Tuple}
    pop_size = length(population)

    ranks = determine_ranks(population)
    fronts = calculate_fronts(population)
    crowding_distances = Dict{Int,Float64}()
    for front in fronts
        front_distances = assign_crowding_distance(front, population)
        for (i, d) in front_distances
            crowding_distances[i] = d
        end
    end

    all_indices = collect(1:pop_size)
    selected_indices = tournament_selection_nsga(all_indices, ranks, crowding_distances, pop_size, tournament_size; rng=rng)

    return SelectedMembers(selected_indices, Dict(enumerate(fronts)))
end

"""
    determine_moga_ranks(pop)

Calculates the Pareto Rank for MOGA.
Rank = 1 + (number of individuals that dominate the current individual).
"""
@inline function determine_moga_ranks(pop::Vector{T}) where {T<:Tuple}
    n = length(pop)
    ranks = ones(Int, n)
    
    # Compare every individual with every other individual
    for i in 1:n
        for j in 1:n
            if i != j && dominates_(pop[j], pop[i]) # If j dominates i
                ranks[i] += 1
            end
        end
    end
    return ranks
end

function calculate_shared_fitness(population::Vector{T}, ranks::Vector{Int}, sigma_share::Float64) where {T<:Tuple}
    n = length(population)
    
    # --- 1. Assign Raw Fitness (Interpolated) ---
    # We assign higher fitness to lower ranks (better individuals).
    # Logic: Sort by rank, assign linear values N down to 1, then average for ties.
    
    indices = sortperm(ranks)
    sorted_ranks = ranks[indices]
    raw_fitness = zeros(Float64, n)
    
    # Linear values to distribute
    linear_values = Float64.(n:-1:1)
    
    i = 1
    while i <= n
        j = i
        # Find the end of the current rank block
        while j < n && sorted_ranks[j+1] == sorted_ranks[i]
            j += 1
        end
        
        block_sum = sum(linear_values[k] for k in i:j)
        avg_fit = block_sum / (j - i + 1)
        
        for k in i:j
            raw_fitness[indices[k]] = avg_fit
        end
        i = j + 1
    end

    num_obj = length(first(population))
    
    objs = [[p[m] for p in population] for m in 1:num_obj]
    mins = [minimum(col) for col in objs]
    maxs = [maximum(col) for col in objs]
    ranges = [maxs[m] - mins[m] for m in 1:num_obj]
    
    ranges = [r == 0 ? 1.0 : r for r in ranges]

    norm_pop = Matrix{Float64}(undef, n, num_obj)
    for i in 1:n
        for m in 1:num_obj
            norm_pop[i, m] = (population[i][m] - mins[m]) / ranges[m]
        end
    end

    shared_fitness = zeros(Float64, n)
    
    for i in 1:n
        niche_count = 0.0
        for j in 1:n
            dist_sq = 0.0
            for m in 1:num_obj
                dist_sq += (norm_pop[i, m] - norm_pop[j, m])^2
            end
            d = sqrt(dist_sq)

            if d < sigma_share
                niche_count += 1.0 - (d / sigma_share)
            end
        end
        
        shared_fitness[i] = raw_fitness[i] / niche_count
    end

    return shared_fitness
end

function moga_selection(population::Vector{T}; tournament_size::Int=3, sigma_share::Float64=0.1, rng::AbstractRNG=STD_RNG) where {T<:Tuple}
    pop_size = length(population)
    
    ranks = determine_moga_ranks(population)
    
    fitness = calculate_shared_fitness(population, ranks, sigma_share)
    
    selected_indices = Vector{Int}(undef, pop_size)
    pop_indices = collect(1:pop_size)
    
    for i in 1:pop_size
        contenders = rand(rng, pop_indices, tournament_size)
        
        winner = reduce(contenders) do best, contender
            fitness[contender] > fitness[best] ? contender : best
        end
        selected_indices[i] = winner
    end

    unique_ranks = sort(unique(ranks))
    fronts = Dict{Int, Vector{Int}}()
    for r in unique_ranks
        fronts[r] = findall(x -> x == r, ranks)
    end

    return SelectedMembers(selected_indices, fronts)
end

end