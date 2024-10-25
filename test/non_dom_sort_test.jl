using Test

include("../src/Selection.jl")

using .EvoSelection


function create_population(data)
    return data
end

# Test dominates_ function
@testset "dominates_ function" begin
    @test dominates_((1, 2), (2, 3)) == true
    @test dominates_((1, NaN), (1, 2)) == false
    @test dominates_((1, 2), (NaN, 3)) == true
    @test dominates_((1, 2, 3), (2, 3, 4)) == true
    @test dominates_((1, 2, 3), (1, 2, 3)) == true
    @test dominates_((1, 2, 3), (0, 3, 4)) == false
end

# Test fast_non_dominated_sort function
@testset "fast_non_dominated_sort function" begin
    # 2 objectives
    pop2 = create_population([
        (1, 5), 
        (2, 4), 
        (3, 3), 
        (4, 2), 
        (5, 1),
        (1, 4), 
        (2, 3), 
        (3, 2), 
        (4, 1),
        (1, 3), 
        (2, 2), 
        (3, 1),
        (1, 2), 
        (2, 1), # 
        (1, 1)  #front 1
    ])
    fronts2 = calculate_fronts(pop2)
    @test length(fronts2) == 5
    @test fronts2[1] == [15]
    @test Set(fronts2[2]) == Set([13, 14])
    @test Set(fronts2[3]) == Set([10, 11, 12])
    @test Set(fronts2[4]) == Set([6, 7, 8, 9])
    @test Set(fronts2[5]) == Set([1, 2, 3, 4, 5])

    # 3 objectives
    pop3 = create_population([
        (1, 1, 1), (2, 2, 2), (3, 3, 3),
        (1, 2, 3), (2, 3, 1), (3, 1, 2),
        (1, 3, 2), (2, 1, 3), (3, 2, 1)
    ])
    fronts3 = calculate_fronts(pop3)
    
    @show fronts3
    @test length(fronts3) == 3
    @test Set(fronts3[1]) == Set([1])
    @test Set(fronts3[2]) == Set([2, 4, 5, 6, 7, 8, 9])
end

# Test assign_crowding_distance function
@testset "assign_crowding_distance function" begin
    # 2 objectives
    pop2 = create_population([(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)])
    front2 = [1, 2, 3, 4, 5]
    distances2 = assign_crowding_distance(front2, pop2)
    @test distances2[1] == Inf
    @test distances2[5] == Inf
    @test distances2[3] ≈ 1

    # 3 objectives
    pop3 = create_population([(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (2, 3, 1), (3, 1, 2)])
    front3 = [1, 4, 5, 6]
    distances3 = assign_crowding_distance(front3, pop3)
    @test distances3[1] == Inf
    @test distances3[6] == Inf
    @test sum(values(distances3)) > 0
end

# Test selection_NSGA function
@testset "selection_NSGA function" begin
    # 2 objectives
    pop2 = create_population([
        (1, 5), (2, 4), (3, 3), (4, 2), (5, 1),
        (1, 4), (2, 3), (3, 2), (4, 1),
        (1, 3), (2, 2), (3, 1),
        (1, 2), (2, 1),
        (1, 1)
    ])
    selected2, fronts2 = selection_NSGA(pop2, 10)
    @test length(selected2) == 10
    @test 15 in selected2  # Preserve the best!! alllllways 
    @test length(fronts2) == 4

    # 3 objectives
    pop3 = create_population([
        (1, 1, 1), (2, 2, 2), (3, 3, 3),
        (1, 2, 3), (2, 3, 1), (3, 1, 2),
        (1, 3, 2), (2, 1, 3), (3, 2, 1)
    ])
    selected3, fronts3 = selection_NSGA(pop3, 5)
    @test length(selected3) == 5
    @test 1 in selected3  # The best individual should always be selected
    @test length(fronts3) == 2
end