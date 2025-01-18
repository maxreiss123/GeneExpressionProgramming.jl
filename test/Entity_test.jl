include("../src/Entities.jl")

using .GepEntities
using DynamicExpressions
using Test
using OrderedCollections
using Random
operators =  OperatorEnum(; binary_operators=[+, -, *, /], unary_operators=[sqrt])

# Helper function to create test toolbox
function create_test_toolbox()
    

    symbols = OrderedDict{Int8,Int8}(
        1 => 2,  # Addition (+)
        2 => 2,  # Multiplication (*)
        3 => 1,  # SQRT
        4 => 0,  # Variable x
        5 => 0   # Constant 1.0
    )
    
    callbacks = Dict{Int8,Function}(
        1 => +,
        2 => *,
        3 => sqrt
    )
    
    nodes = OrderedDict{Int8,Any}(
        4 => Node{Float64}(feature=1),
        5 => 1.0
    )
    
    gep_probs = Dict{String,AbstractFloat}(
        "mutation_prob" => 0.2,
        "mutation_rate" => 0.1,
        "inversion_prob" => 0.1,
        "one_point_cross_over_prob" => 0.3,
        "two_point_cross_over_prob" => 0.3,
        "dominant_fusion_prob" => 0.2,
        "rezessiv_fusion_prob" => 0.2,
        "fusion_prob" => 0.2,
        "fusion_rate" => 0.1,
        "rezessiv_fusion_rate" => 0.1
    )
    
    return Toolbox(2, 3, symbols, Int8[1], callbacks, nodes, gep_probs)
end

@testset "SymbolicEntities Tests" begin
    @testset "Basic Setup" begin
        toolbox = create_test_toolbox()
        
        @test toolbox.gene_count == 2
        @test toolbox.head_len == 3
        @test length(toolbox.headsyms) == 2  # + and *
        @test length(toolbox.unary_syms) == 1  # square
        @test length(toolbox.tailsyms) == 2  # x and 1.0
    end

    @testset "Chromosome Creation" begin
        toolbox = create_test_toolbox()
        Random.seed!(42)  # For reproducibility
        chromosome = generate_chromosome(toolbox)
        
        @test chromosome isa Chromosome
        @test length(chromosome.genes) == (toolbox.gene_count - 1 + 
            toolbox.gene_count * (2 * toolbox.head_len + 1))
        @test chromosome.compiled == true
        @test chromosome.dimension_homogene == false
        @test isnan(chromosome.fitness[1])
    end

    @testset "Function Compilation" begin
        toolbox = create_test_toolbox()
        Random.seed!(42)
        chromosome = generate_chromosome(toolbox)
        compile_expression!(chromosome, force_compile=true)
        
        @test chromosome.compiled == true
        @test chromosome.compiled_function !== nothing
        
        # Test function evaluation
        if !isnothing(chromosome.compiled_function)
            result = chromosome.compiled_function([1.0], operators)
            @test typeof(result[1]) <: Real
        end
    end
    
    @testset "Population Generation" begin
        toolbox = create_test_toolbox()
        population_size = 10
        population = generate_population(population_size, toolbox)
        
        @test length(population) == population_size
        @test all(x -> x isa Chromosome, population)
        @test length(unique([p.genes for p in population])) == population_size
    end
end