using Test
using OrderedCollections
using Random

Random.seed!(1)

@testset "GepRegressor Tests" begin

    @testset "Function Entries Creation" begin
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
        @test all(n -> n isa InputSelector, values(nodes))
        @test length(dims) == 3
    end

    @testset "Constants Entries Creation" begin
        constants = [Symbol(1), Symbol(2.5)]
        dimensions = Dict{Symbol,Vector{Float16}}()
        
        syms, nodes, dims, idx = create_constants_entries(
            constants, 2, dimensions, Float64, Int8(1), MersenneTwister()
        )
        
        @test length(syms) == 4  # 2 constants + 2 random
        @test all(v -> v == 0, values(syms))
        @test length(nodes) == 4
        # constants are stored as plain numbers
        @test nodes[1] == 1.0
        @test nodes[2] == 2.5
    end

    @testset "Physical Operations" begin
        non_terminals = [:+, :-, :*, :/, :sqrt]
        # the handlers are keyed by the symbol ids the caller supplies
        idx_funs = Int8[1, 2, 3, 4, 5]
        forward_funs, backward_funs, point_ops = create_physical_operations(non_terminals, idx_funs)

        @test forward_funs isa OrderedDict{Int8,Function}
        @test backward_funs isa Dict{Int8,Function}
        @test point_ops isa Vector{Int8}
    end

    @testset "Dimension Handling" begin
        # keyed by the names of the features
        dimensions = Dict(
            :x => Float16[1, 0, 0, 0, 0, 0, 0],
            :y => Float16[0, 1, 0, 0, 0, 0, 0]
        )
        
        regressor = GepRegressor(
            2,
            entered_features=[:x, :y],
            considered_dimensions=dimensions
        )
        
        @test !isnothing(regressor.token_dto_)
        @test length(regressor.dimension_information_) > 0
        @test all(v -> v isa Vector{Float16}, values(regressor.dimension_information_))
        @test count(v -> v in values(dimensions), values(regressor.dimension_information_)) == 2

        # a key that names no feature is ignored, with a warning
        @test_logs (:warn, r"name no feature") GepRegressor(2; entered_features=[:x, :y],
            considered_dimensions=Dict(:x1 => Float16[1, 0, 0, 0, 0, 0, 0]), rounds=1,
            max_permutations_lib=100)
        @test_logs GepRegressor(2; considered_dimensions=Dict(:x1 => Float16[1, 0, 0, 0, 0, 0, 0],
            Symbol(0.5) => Float16[0, 1, 0, 0, 0, 0, 0]), rounds=1, max_permutations_lib=100)
    end

    @testset "Basic Training" begin
        X = rand(10, 2)
        y = 2 .* X[:, 1] .+ X[:, 2]
        
        regressor = GepRegressor(2)
        
        fit!(regressor, 100, 1000, X', y)
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
            
        fit!(regressor, 10, 20, X', y)
        @test !isnothing(regressor.token_dto_)
        @test !isnothing(regressor.best_models_)
    end
end

@testset "Constant optimisation" begin
    # y = 3.7 x1 needs a constant no terminal holds (they are 0.5, 0.0 and one random
    # value in [0, 1)), so the optimiser has work to do
    Random.seed!(4)
    x = randn(2, 80)
    y = 3.7 .* x[1, :]
    reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*, :/], rnd_count=1)
    fit!(reg, 30, 200, x, y; loss_fun="mse", optimization_epochs=5)
    best = reg.best_models_[1]
    @test any(m -> m.optimised_constants !== nothing, reg.best_models_)
    # the stored score is the score of the model as it predicts
    for m in reg.best_models_
        @test m.fitness[1] ≈ get_loss_function("mse")(y, m(x)) rtol = 1e-6 atol = 1e-12
    end
    @test best.fitness[1] < 1e-6
    # and a model with optimised constants prints
    for m in filter(m -> m.optimised_constants !== nothing, reg.best_models_)
        @test !isempty(sprint(show, m))
    end
end

@testset "Function library management" begin
    # the accessors read the library, and the setters reject unknown names
    params = list_all_genetic_params()
    @test params["mutation_prob"] ==
          GeneExpressionProgramming.RegressionWrapper.GENE_COMMON_PROBS["mutation_prob"]
    @test_throws ArgumentError set_function!(:no_such_function, identity)
    @test_throws ArgumentError update_function!(:no_such_function; func=identity)
    # re-registering an entry with what it already holds leaves the library unchanged
    f = FUNCTION_LIB_COMMON[:sin]
    @test set_function!(:sin, f) === nothing
    @test update_function!(:sin; func=f, arity=Int8(1)) === nothing
    @test FUNCTION_LIB_COMMON[:sin] === f
    @test ARITY_LIB_COMMON[:sin] == 1
    RW = GeneExpressionProgramming.RegressionWrapper
    @test list_all_arity()[:sin] == 1
    @test list_all_forward_handlers()[:*] === RW.FUNCTION_LIB_FORWARD_COMMON[:*]
    @test list_all_backward_handlers()[:sqrt] === RW.FUNCTION_LIB_BACKWARD_COMMON[:sqrt]
end

@testset "Physical constants" begin
    @test physical_constants_all["Z_0"][2] == physical_constants["Z_0"][2]   # ohm
    @test physical_constants_all["G_0"][2] == Float16[-1, -2, 3, 0, 0, 2, 0]  # siemens
    @test get_constant_dims("e") == Float16[0, 0, 1, 0, 0, 1, 0]              # A s
end
