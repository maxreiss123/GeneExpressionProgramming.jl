"""
    EvoSelection

Parent selection for the GEP loop: tournament selection for one objective, NSGA-II
(non-dominated sorting with crowding distance) for several. A fitness is a tuple of
losses, all minimised. Both selections return a `SelectedMembers`.
"""
module EvoSelection
using LinearAlgebra
using Random

export tournament_selection, nsga_selection, dominates_, fast_non_dominated_sort, calculate_fronts, determine_ranks, assign_crowding_distance

"""
    SelectedMembers

Result of a selection: `indices`, the selected population indices in selection order,
and `fronts`, the Pareto fronts by rank (`k => indices of rank k`), empty after
tournament selection.
"""
struct SelectedMembers
    indices::Vector{Int}
    fronts::Dict{Int,Vector{Int}}
end

const STD_RNG = MersenneTwister()

"""
    tournament_selection(population, number_of_winners, tournament_size; rng=STD_RNG)

Select `number_of_winners` indices of `population`, a vector of fitness tuples. Each
tournament draws `tournament_size` eligible contenders (fewer if fewer are eligible)
with replacement, and the smallest fitness wins. Eligible are the individuals with a
finite first objective, and of several with the same fitness only the first; if none
qualifies, all are eligible. The last slot always holds index 1, the best individual
when `population` is sorted, as `runGep` sorts it. Returns a `SelectedMembers` with empty
`fronts`.
"""
@inline function tournament_selection(population::AbstractArray{Tuple}, number_of_winners::Int, tournament_size::Int; rng::AbstractRNG=STD_RNG)
    selected_indices = Vector{Int}(undef, number_of_winners)
    valid_indices_ = findall(x -> isfinite(x[1]), population)
    valid_indices = Int[]
    doubles = Set()
    for elem in valid_indices_
        if !(population[elem] in doubles)
            push!(doubles, population[elem])
            push!(valid_indices, elem)
        end
    end

    # A generation without a finite fitness is normal, e.g. on a hard problem, with an
    # Inf penalty, or among the mostly invalid chromosomes of a first generation. Drawing
    # from the whole population keeps the search going; drawing from an empty set throws.
    isempty(valid_indices) && (valid_indices = collect(eachindex(population)))

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

"""
    dominates_(a::Tuple, b::Tuple)

Whether fitness `a` dominates `b`, all objectives minimised. A tuple with fewer
non-finite entries (`Inf`, `NaN`) dominates one with more; with equally many, `a`
dominates if it is nowhere worse and somewhere better.
"""
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

"""
    determine_ranks(pop::Vector{<:Tuple})

The Pareto rank of each fitness under `dominates_`: 1 for the non-dominated, `k` for
those non-dominated once ranks `1:k-1` are removed.
"""
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

"""
    fast_non_dominated_sort(population::Vector{<:Tuple})

The population indices ordered by Pareto rank, rank 1 first.
"""
@inline function fast_non_dominated_sort(population::Vector{T}) where {T<:Tuple}
    ranks = determine_ranks(population)
    pop_indices = [(index, rank) for (index, rank) in enumerate(ranks)]
    sort!(pop_indices, by=x -> x[2])
    return [elem[1] for elem in pop_indices]
end

"""
    calculate_fronts(population::Vector{<:Tuple})

The Pareto fronts, best first: entry `k` lists the indices of rank `k`.
"""
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

"""
    assign_crowding_distance(front::Vector{Int}, population::Vector{<:Tuple})

The crowding distance of each member of `front`, as a `Dict` from population index to
distance. Per objective the members are sorted, the two extremes get `Inf`, and each
interior member adds the gap between its neighbours divided by the objective's range on
the front. If an objective is constant on a front of more than two members, all of them
get `Inf` and the remaining objectives are skipped.
"""
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

"""
    tournament_selection_nsga(pop_indices, ranks, crowding_distances, number_of_winners,
        tournament_size; rng=STD_RNG)

Crowded-comparison tournaments: each of `number_of_winners` draws `tournament_size`
contenders from `pop_indices` with replacement; the lower rank wins, on equal rank the
larger crowding distance, otherwise the first drawn. Finite distances are rescaled to
[0, 0.99] and `Inf` counts as 0.99. Returns the winners' indices.
"""
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

"""
    nsga_selection(population::Vector{<:Tuple}; tournament_size=3, rng=STD_RNG)

NSGA-II selection on fitness tuples: sort `population` into Pareto fronts, compute the
crowding distances within each, and run `length(population)` crowded-comparison
tournaments of `tournament_size`. Returns a `SelectedMembers` holding the winners and
the fronts.
"""
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

end