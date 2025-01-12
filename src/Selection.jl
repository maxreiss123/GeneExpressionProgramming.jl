module EvoSelection
using LinearAlgebra

export selection, dominates_, fast_non_dominated_sort, calculate_fronts, determine_ranks, assign_crowding_distance


struct SelectedMembers
    indices::Vector{Int}
    fronts::Dict{Int,Vector{Int}}
end

#Note: selection is constructed to allways return a list of indices => {just care about the data not the objects}
@inline function selection(population::AbstractArray{T}, number_of_winners::Int, tournament_size::Int) where {T<:Number}
    selected_indices = Vector{Int}(undef, number_of_winners)
    valid_indices_ = findall(isfinite, population)
    valid_indices = []
    doubles = Set()
    for elem in valid_indices_
    	if !(population[elem] in doubles)
    		push!(doubles,population[elem])
    		push!(valid_indices, elem)
    	end
    end
    
    Threads.@threads for index in 1:number_of_winners-1
        contenders = rand(valid_indices, tournament_size)
        winner = reduce((best, contender) -> population[contender] < population[best] ? contender : best, contenders)
        selected_indices[index] = winner
    end
    selected_indices[end] = 1
    return SelectedMembers(selected_indices,Dict{Int,Vector{Int}}())
end


function count_infinites(t::Tuple)
    return count(isinf, t) + count(isnan, t)
end

function ≪(a::T, b::U) where {T<:Number,U<:Number}
    if isnan(a) || isnan(b) || isinf(a) || isinf(b)
        return false
    else
        return (a <= b * 0.1) || (a < 0 && b >= 0)
    end
end

function dominates_(a::Tuple, b::Tuple)
    all_smaller = true
    one_significant_smaller = false

    a_inf_count = count_infinites(a)
    b_inf_count = count_infinites(b)

    if b_inf_count > a_inf_count
        return true
    elseif a_inf_count > b_inf_count
        return false
    end


    @inbounds for i in eachindex(a)
        ai, bi = a[i], b[i]
        if ai ≪ bi
            one_significant_smaller = true
        elseif ai <= bi
            all_smaller = true & all_smaller
        elseif bi ≪ ai || ai > bi
            all_smaller = false
        end
    end
    return one_significant_smaller || all_smaller
end


@inline function determine_ranks(pop::Vector{T}) where T<:Tuple
    n = length(pop)

    dom_list = [ Int[] for i in 1:n ]
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

    k = UInt16(2)
    while any(==(k-one(UInt16)), (rank[p] for p in 1:n)) 
        for p in 1:n
            if rank[p] == k-one(UInt16)
                for q in dom_list[p]
                    dom_count[q] -= one(UInt16)
                    if dom_count[q] == zero(UInt16)
                        rank[q] = k
                    end
                end
            end
        end
        k += one(UInt16)
    end
    return rank
end

@inline function fast_non_dominated_sort(population::Vector{T}) where {T<:Tuple}
    ranks = determine_ranks(population)
    pop_indices = [(index,rank) for (index, rank) in enumerate(ranks)]
    sort!(pop_indices, by = x -> x[2])
    return [elem[1] for elem in pop_indices]
end

@inline function calculate_fronts(population::Vector{T}) where {T<:Tuple}
    ranks = determine_ranks(population)
    min_rank = minimum(unique(ranks)) 
    max_rank = maximum(unique(ranks))
    if min_rank == 0
        ranks = [rank == 0 ? max_rank+1 : rank for rank in ranks]
    end
    fronts = [Int[] for i in eachindex(unique(ranks))]

    for (i, r) in enumerate(ranks)
        push!(fronts[r], i)
    end
    return fronts
end

@inline function assign_crowding_distance(front::Vector{Int}, population::Vector{T}) where {T<:Tuple}
    n = length(front)
    objectives_count = length(first(population))

    distances = Dict{Int,Float64}()
    for i in front
        distances[i] = 0.0
    end
    # only looking to the direct neighbour!
    for m in 1:objectives_count
        sorted_front = sort(front, by=i -> population[i][m])
        distances[sorted_front[1]] = distances[sorted_front[end]] = Inf

        if n > 2
            obj_range = population[sorted_front[end]][m] - population[sorted_front[1]][m]
            if obj_range > 0
                for i in 2:n-1
                    prev = population[sorted_front[i-1]][m]
                    next = population[sorted_front[i+1]][m]
                    distances[sorted_front[i]] += (next - prev) / obj_range
                end
            end
        end
    end
    return distances
end


@inline function selection(population::Vector{T}) where {T<:Tuple}
    fronts = calculate_fronts(population)
    n_fronts = length(fronts)

    selected_indices = Int[]
    pareto_fronts = Dict{Int,Vector{Int}}()

    estimated_size = length(population)
    selected_indices = Vector{Int}(undef, estimated_size)
    current_idx = 1


    @inbounds for front_idx in 1:n_fronts
        front = fronts[front_idx]
        crowding_distances = assign_crowding_distance(front, population)

        sorted_front = sort(front, by=i -> crowding_distances[i], rev=true)
        front_size = length(sorted_front)
        copyto!(selected_indices, current_idx, sorted_front, 1, front_size)
        
        pareto_fronts[front_idx] = sorted_front
        current_idx +=front_size
    end
    resize!(selected_indices, current_idx - 1)

    return SelectedMembers(selected_indices, pareto_fronts)

end

end
