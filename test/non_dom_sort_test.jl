using Test

function create_population(data)
    return data
end

# Test dominates_ function
@testset "dominates_ function" begin
    @test dominates_((1, 2), (2, 3)) == true          # (1, 2) dominates (2, 3)
    @test dominates_((1, NaN), (1, 2)) == false       # NaN handling: (1, NaN) does not dominate
    @test dominates_((1, 2), (NaN, 3)) == true        # (1, 2) dominates (NaN, 3)
    @test dominates_((1, 2, 3), (2, 3, 4)) == true    # 3 objectives, all better
    @test dominates_((1, 2, 3), (1, 2, 3)) == true    # Equal objectives, should return true per definition
    @test dominates_((1, 2, 3), (0, 3, 4)) == false   # (0, 3, 4) is better in first objective
end


# Test calculate_fronts function
@testset "calculate_fronts function" begin
    # 2 objectives
    pop2 = create_population([
        (1, 5), (2, 4), (3, 3), (4, 2), (5, 1),  # Front 5
        (1, 4), (2, 3), (3, 2), (4, 1),          # Front 4
        (1, 3), (2, 2), (3, 1),                  # Front 3
        (1, 2), (2, 1),                           # Front 2
        (1, 1)                                    # Front 1
    ])
    fronts2 = calculate_fronts(pop2)
    @test length(fronts2) == 5                        # 5 non-dominated fronts
    @test fronts2[1] == [15]                          # Front 1: (1, 1)
    @test Set(fronts2[2]) == Set([13, 14])            # Front 2: (1, 2), (2, 1)
    @test Set(fronts2[3]) == Set([10, 11, 12])        # Front 3: (1, 3), (2, 2), (3, 1)
    @test Set(fronts2[4]) == Set([6, 7, 8, 9])        # Front 4: (1, 4), (2, 3), (3, 2), (4, 1)
    @test Set(fronts2[5]) == Set([1, 2, 3, 4, 5])     # Front 5: (1, 5), (2, 4), (3, 3), (4, 2), (5, 1)

    # 3 objectives
    pop3 = create_population([
        (1, 1, 1), (2, 2, 2), (3, 3, 3),
        (1, 2, 3), (2, 3, 1), (3, 1, 2),
        (1, 3, 2), (2, 1, 3), (3, 2, 1)
    ])
    fronts3 = calculate_fronts(pop3)
    @test length(fronts3) == 3                        # 3 non-dominated fronts
    @test Set(fronts3[1]) == Set([1])                 # Front 1: (1, 1, 1)
    @test Set(fronts3[2]) == Set([2, 4, 5, 6, 7, 8, 9])  # Front 2: all others except (3, 3, 3)
end

# Test assign_crowding_distance function
@testset "assign_crowding_distance function" begin
    # 2 objectives
    pop2 = create_population([(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)])
    front2 = [1, 2, 3, 4, 5]
    distances2 = assign_crowding_distance(front2, pop2)
    @test distances2[1] == Inf                        # Boundary point (1, 5)
    @test distances2[5] == Inf                        # Boundary point (5, 1)
    @test distances2[3] ≈ 1                           # Middle point (3, 3) has finite distance

    # 3 objectives
    pop3 = create_population([(1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (2, 3, 1), (3, 1, 2)])
    front3 = [1, 4, 5, 6]
    distances3 = assign_crowding_distance(front3, pop3)
    @test distances3[1] == Inf                        # Boundary point (1, 1, 1)
    @test distances3[6] == Inf                        # Boundary point (3, 1, 2)
    @test sum(values(distances3)) > 0                 # Sum of distances should be positive
end


# Test nsga_selection function
@testset "nsga_selection function" begin
    # 2 objectives
    pop2 = create_population([
        (1, 5), (2, 4), (3, 3), (4, 2), (5, 1),
        (1, 4), (2, 3), (3, 2), (4, 1),
        (1, 3), (2, 2), (3, 1),
        (1, 2), (2, 1),
        (1, 1)
    ])
    selected2 = nsga_selection(pop2)
    @test length(selected2.indices) == 15             # Matches population size
    @test all(i -> 1 <= i <= 15, selected2.indices)   # All indices are valid
    @test length(selected2.fronts) == 5               # Correct number of fronts

    # 3 objectives
    pop3 = create_population([
        (1, 1, 1), (2, 2, 2), (3, 3, 3),
        (1, 2, 3), (2, 3, 1), (3, 1, 2),
        (1, 3, 2), (2, 1, 3), (3, 2, 1)
    ])
    selected3 = nsga_selection(pop3)
    @test length(selected3.indices) == 9              # Matches population size
    @test all(i -> 1 <= i <= 9, selected3.indices)    # All indices are valid
    @test length(selected3.fronts) == 3               # Correct number of fronts
end