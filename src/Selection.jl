module EvoSelection
using LinearAlgebra

export tournament_selection, nsga_selection, dominates_, fast_non_dominated_sort, calculate_fronts, determine_ranks, assign_crowding_distance

struct SelectedMembers
    indices::Vector{Int}
    fronts::Dict{Int,Vector{Int}}
end

@inline function tournament_selection(population::AbstractArray{Tuple}, number_of_winners::Int, tournament_size::Int)
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
            selected_indices[index] = 1  # Keep the original behavior
        else
            contenders = rand(valid_indices, min(tournament_size, length(valid_indices)))
            winner = reduce((best, contender) -> population[contender] < population[best] ? contender : best, contenders)
            selected_indices[index] = winner
        end
    end
    return SelectedMembers(selected_indices, Dict{Int,Vector{Int}}())
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

    for i in eachindex(a)
        ai, bi = a[i], b[i]
        if ai ≪ bi
            one_significant_smaller = true
        elseif ai <= bi
            all_smaller = all_smaller
        else
            all_smaller = false
        end
    end
    return one_significant_smaller || all_smaller
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
        sorted_front = sort(front, by=i -> population[i][m])

        # Set extreme points to infinity
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
            end
        end
    end
    return distances
end


function tournament_selection_nsga(pop_indices::Vector{Int}, ranks::Vector{Int}, crowding_distances::Dict{Int,Float64}, number_of_winners::Int, tournament_size::Int)
    selected_indices = Vector{Int}(undef, number_of_winners)
    for i in 1:number_of_winners
        contenders = rand(pop_indices, tournament_size)
        winner = reduce(contenders; init=contenders[1]) do best, contender
            if ranks[contender] < ranks[best]
                contender
            elseif ranks[contender] > ranks[best]
                best
            else
                crowding_distances[contender] > crowding_distances[best] ? contender : best
            end
        end
        selected_indices[i] = winner
    end
    return selected_indices
end

function nsga_selection(population::Vector{T}; tournament_size::Int=2) where {T<:Tuple}
    pop_size = length(population)

    # Compute ranks for all individuals
    ranks = determine_ranks(population)

    # Compute fronts for tracking and crowding distances
    fronts = calculate_fronts(population)
    crowding_distances = Dict{Int,Float64}()
    for front in fronts
        front_distances = assign_crowding_distance(front, population)
        for (i, d) in front_distances
            crowding_distances[i] = d
        end
    end

    all_indices = collect(1:pop_size)
    selected_indices = tournament_selection_nsga(all_indices, ranks, crowding_distances, pop_size, tournament_size)

    @debug selected_indices
    @debug population[selected_indices]

    return SelectedMembers(selected_indices, Dict(enumerate(fronts)))
end


end