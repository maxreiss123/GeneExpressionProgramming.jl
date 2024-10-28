include("../src/RegressionWrapper.jl")
using Test
using OrderedCollections
using DynamicExpressions

using .RegressionWrapper

@testset "GepRegressor Tests" begin
    @testset "Basic Construction" begin
        # Test default construction
        @test_nowarn GepRegressor(3)
        
        # Test with minimal parameters
        regressor = GepRegressor(2)
        @test regressor.gene_count_ == 3
        @test regressor.head_len_ == 8
        @test !isnothing(regressor.utilized_symbols_)
        @test !isnothing(regressor.operators_)
    end

    @testset "Function Entries Creation" begin
        # Test create_function_entries
        non_terminals = [:+, :-, :*, :/, :sqrt]
        gene_connections = [:+, :*]
        
        syms, callbacks, binary_ops, unary_ops, gene_conns, idx = create_function_entries(
            non_terminals, gene_connections, Int8(1)
        )
        
        @test syms isa OrderedDict{Int8,Int8}
        @test callbacks isa Dict{Int8,Function}
        @test length(binary_ops) == 4  # +, -, *, /
        @test length(unary_ops) == 1   # sqrt
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

    @testset "Complex Construction" begin
        # Test with more complex parameters
        regressor = GepRegressor(
            3,
            entered_features=[:x, :y, :z],
            entered_non_terminals=[:+, :-, :*, :/],
            entered_terminal_nums=[Symbol(1.0), Symbol(2.0)],
            gene_count=5,
            head_len=10,
            considered_dimensions=Dict(
                :x => Float16[1, 0, 0, 0, 0, 0, 0],
                :y => Float16[0, 1, 0, 0, 0, 0, 0],
                :z => Float16[0, 0, 1, 0, 0, 0, 0]
            )
        )
        
        @test length(regressor.utilized_symbols_) > 0
        @test length(regressor.nodes_) > 0
        @test length(regressor.dimension_information_) > 0
        @test regressor.gene_count_ == 5
        @test regressor.head_len_ == 10
    end

    @testset "Dimension Handling" begin
        dimensions = Dict(
            :x => Float16[1, 0, 0, 0, 0, 0, 0],
            :y => Float16[0, 1, 0, 0, 0, 0, 0]
        )
        
        regressor = GepRegressor(
            2,
            entered_features=[:x, :y],
            considered_dimensions=dimensions
        )
        
        @test !isnothing(regressor.token_dto)
        @test length(regressor.dimension_information_) > 0
        @test all(v -> v isa Vector{Float16}, values(regressor.dimension_information_))
    end
end