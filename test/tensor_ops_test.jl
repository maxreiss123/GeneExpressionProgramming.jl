using BenchmarkTools
using Test
using Tensors
using OrderedCollections
using Flux

@testset "TensorRegUtils" begin
    # Test setup
    dim = 3
    t2 = rand(Tensor{2,dim})
    vec3 = rand(Tensor{1,dim})
    const_val = 2.0

    data_ = Dict(
        5 => t2,
        6 => vec3,
        7 => const_val
    )
    inputs = (t2, vec3, const_val)


    arity_map = OrderedDict{Int8,Int}(
        1 => 2,  # Addition
        2 => 2,  # Multiplication
        3 => 2,  # Double Contraction
        4 => 1   # Trace
    )

    callbacks = Dict{Int8,Any}(
        Int8(1) => AdditionNode,
        Int8(2) => MultiplicationNode,
        Int8(3) => DoubleContractionNode,
        Int8(4) => TraceNode
    )

    @testset "Basic Operations" begin
        nodes = OrderedDict{Int8,Any}(
            Int8(5) => InputSelector(1),
            Int8(6) => InputSelector(2),
            Int8(7) => const_val
        )

        # Scalar multiplication
        rek_string = Int8[2, 6, 7]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        result = network(inputs)
        @test result ≈ vec3 * const_val

        # Addition with dimension mismatch
        rek_string = Int8[1, 5, 6]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test_throws DimensionMismatch network(inputs)
    end

    @testset "Tensor Operations" begin
        nodes = OrderedDict{Int8,Any}(
            Int8(5) => InputSelector(1),
            Int8(6) => InputSelector(2),
            Int8(7) => const_val
        )

        # Double contraction
        rek_string = Int8[3, 5, 5]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        result = network(inputs)
        @test result ≈ dcontract(t2, t2)

        # Trace
        rek_string = Int8[4, 5]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        result = network(inputs)
        @test result ≈ tr(t2)
    end

    @testset "Complex Expressions" begin
        nodes = OrderedDict{Int8,Any}(
            Int8(5) => InputSelector(1),
            Int8(6) => InputSelector(2),
            Int8(7) => const_val
        )

        # (vec3 * const_val) + vec3
        rek_string = Int8[1, 2, 6, 7, 6]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)

        result = network(inputs)
        @test result ≈ (vec3 * const_val) + vec3

        # (t2 * vec3) + const_val - should fail
        rek_string = Int8[1, 2, 5, 6, 7]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test !isfinite(network(inputs))

        # tr(t2) * const_val + tr(t2)
        rek_string = Int8[1, 2, 4, 5, 7, 4, 5]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        result = network(inputs)
        @test result ≈ tr(t2) * const_val + tr(t2)
    end

end

@testset "All Node Operations" begin
    dim = 3
    t2 = rand(Tensor{2,dim})
    t2_2 = rand(Tensor{2,dim})
    vec3 = rand(Vec{dim})
    const_val = 2.0

    data_ = Dict(19 => t2, 20 => t2_2, 21 => vec3, 22 => const_val)

    inputs = (t2, t2_2, vec3, const_val)

    arity_map = OrderedDict{Int8,Int}(
        1 => 2, 2 => 2, 3 => 2, 4 => 2, 5 => 2, 6 => 2, 7 => 2,  # Binary ops
        8 => 1, 9 => 1, 10 => 1, 11 => 1, 12 => 1, 13 => 1,      # Unary ops part 1
        14 => 1, 15 => 1, 16 => 1, 17 => 2, 18 => 1              # Unary ops part 2 + DC
    )

    callbacks = Dict{Int8,Any}(
        Int8(1) => AdditionNode,
        Int8(2) => SubtractionNode,
        Int8(3) => MultiplicationNode,
        Int8(4) => DivisionNode,
        Int8(5) => PowerNode,
        Int8(6) => MinNode,
        Int8(7) => MaxNode,
        Int8(8) => InversionNode,
        Int8(9) => TraceNode,
        Int8(10) => DeterminantNode,
        Int8(11) => SymmetricNode,
        Int8(12) => SkewNode,
        Int8(13) => VolumetricNode,
        Int8(14) => DeviatricNode,
        Int8(15) => TdotNode,
        Int8(16) => DottNode,
        Int8(17) => DoubleContractionNode,
        Int8(18) => DeviatoricNode
    )

    @testset "Basic Node Operations" begin
        nodes = OrderedDict{Int8,Any}(
            Int8(19) => InputSelector(1),
            Int8(20) => InputSelector(2),
            Int8(21) => InputSelector(3),
            Int8(22) => const_val
        )

        # Test 1: Addition
        rek_string = Int8[1, 19, 20]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test network(inputs) ≈ t2 + t2_2

        # Test 2: Subtraction 
        rek_string = Int8[2, 19, 20]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test network(inputs) ≈ t2 - t2_2

        # Test 3: Multiplication with constant
        rek_string = Int8[3, 19, 22]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test network(inputs) ≈ t2 * const_val
    end

    @testset "Tensor Operations" begin
        nodes = OrderedDict{Int8,Any}(
            Int8(19) => InputSelector(1),
            Int8(20) => InputSelector(2)
        )

        # Test 1: Double Contraction
        rek_string = Int8[17, 19, 20]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test network(inputs) ≈ dcontract(t2, t2_2)

        # Test 2: Deviatoric
        rek_string = Int8[14, 19]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test network(inputs) ≈ dev(t2)

        # Test 3: Trace + Symmetric
        rek_string = Int8[9, 11, 19]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test network(inputs) ≈ tr(symmetric(t2))
    end

    @testset "Complex Compositions" begin
        nodes = OrderedDict{Int8,Any}(
            Int8(19) => InputSelector(1),
            Int8(20) => InputSelector(2),
            Int8(22) => const_val
        )

        # Test 1: (t2 ⊡ t2_2) * const_val
        rek_string = Int8[3, 17, 19, 20, 22]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test network(inputs) ≈ dcontract(t2, t2_2) * const_val

        # Test 2: dev(symmetric(t2)) + t2_2
        rek_string = Int8[1, 14, 11, 19, 20]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test network(inputs) ≈ dev(symmetric(t2)) + t2_2

        # Test 3: tr(t2) * tr(t2_2)
        rek_string = Int8[3, 9, 19, 9, 20]
        network = compile_to_flux_network(rek_string, arity_map, callbacks, nodes, 0)
        @test network(inputs) ≈ tr(t2) * tr(t2_2)
    end
end