include("../src/RegressionWrapper.jl")

using .RegressionWrapper
using Test
using OrderedCollections
using DynamicExpressions
using Random

Random.seed!(1)

@testset "GepRegressor Tests" begin

    @testset "Function Entries Creation" begin
        # Test create_function_entries
        non_terminals = [:+, :-, :*, :/, :exp]
        gene_connections = [:+, :*]
        
        syms, callbacks, binary_ops, unary_ops, gene_conns, idx = create_function_entries(
            non_terminals, gene_connections, Int8(1)
        )
        
        @test syms isa OrderedDict{Int8,Int8}
        @test callbacks isa Dict{Int8,Function}
        @test length(binary_ops) == 4  # +, -, *, /
        @test length(unary_ops) == 1   # exp
        @test length(gene_conns) == 2  # +, *
    end

    @testset "Feature Entries Creation" begin
        features = [:x1, :x2, :x3]
        dimensions = Dict{Symbol,Vector{Float16}}()
        
        syms, nodes, dims, idx = create_feature_entries(
            features, dimensions, Float64, Int8(1)
        )
        
        @test length(syms) == 3
        @test all(v -> v == 0, values(syms))
        @test all(n -> n isa Node, values(nodes))
        @test length(dims) == 3
    end

    @testset "Constants Entries Creation" begin
        constants = [Symbol(1), Symbol(2.5)]
        dimensions = Dict{Symbol,Vector{Float16}}()
        
        syms, nodes, dims, idx = create_constants_entries(
            constants, 2, dimensions, Float64, Int8(1)
        )
        
        @test length(syms) == 4  # 2 constants + 2 random
        @test all(v -> v == 0, values(syms))
        @test length(nodes) == 4
        @test nodes[1] == 1.0
        @test nodes[2] == 2.5
    end

    @testset "Physical Operations" begin
        non_terminals = [:+, :-, :*, :/, :sqrt]
        forward_funs, backward_funs, point_ops = create_physical_operations(non_terminals)
        
        @test forward_funs isa OrderedDict{Int8,Function}
        @test backward_funs isa Dict{Int8,Function}
        @test point_ops isa Vector{Int8}
    end

    @testset "Dimension Handling" begin
        dimensions = Dict(
            :x1 => Float16[1, 0, 0, 0, 0, 0, 0],
            :x2 => Float16[0, 1, 0, 0, 0, 0, 0]
        )
        
        regressor = GepRegressor(
            2,
            entered_features=[:x, :y],
            considered_dimensions=dimensions
        )
        
        @test !isnothing(regressor.token_dto_)
        @test length(regressor.dimension_information_) > 0
        @test all(v -> v isa Vector{Float16}, values(regressor.dimension_information_))
    end

    @testset "Basic Training" begin
        X = rand(10, 2)
        y = 2 .* X[:, 1] .+ X[:, 2]
        
        regressor = GepRegressor(2)
        
        fit!(regressor, 100, 1000, X, y)
        @test !isempty(regressor.fitness_history_.train_loss)
        @test length(regressor.best_models_) > 0
    end
    
    @testset "Training with Physical Dimensions" begin
        X = rand(50, 2)
        y = X[:, 1] .* 2 .+ X[:, 2]
        
        dimensions = Dict(
            :x1 => Float16[1,0,0,0,0,0,0],
            :x2 => Float16[0,1,0,0,0,0,0]
        )
        
        regressor = GepRegressor(2,
            entered_features=[:x1, :x2],
            considered_dimensions=dimensions)
            
        @test_nowarn fit!(regressor, 10, 20, X, y)
        @test !isnothing(regressor.token_dto_)
        @test !isnothing(regressor.best_models_)
    end
    
    @testset "Training with Different Loss Functions" begin
        X = rand(50, 2)
        y = X[:, 1] .* 2 .+ X[:, 2]
        
        regressor = GepRegressor(2)
        
        @test_nowarn fit!(regressor, 10, 20, X, y, loss_fun="mse")
        
        @test_nowarn fit!(regressor, 10, 20, X, y, loss_fun="mae")
        
    end
end
