
using Test
using Random
using OrderedCollections
Random.seed!(1)

# symbols 1-6 are operators, 7 and 8 the features x1 (kg) and x2 (m)
function create_token_lib_test()
    physical_dimension_dict = OrderedDict{Int8, Vector{Float16}}(
        1 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # mul
        2 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # div
        3 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # add
        4 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sub
        5 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sqr
        6 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sin
        7 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x1
        8 => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # x2
    )

    physical_operation_dict = OrderedDict{Int8, Function}(
        1 => (mul_unit_forward),
        2 => (div_unit_forward),
        3 => (equal_unit_forward),          
        4 => (equal_unit_forward),
        5 => (sqr_unit_forward),
        6 => (zero_unit_forward)     
    )

    symbol_arity_mapping = OrderedDict{Int8, Int8}(
        1 => 2, 2 => 2, 3 => 2, 4 => 2,
        5 => 1, 6 => 1,
        7 => 0, 8 => 0
    )

    return TokenLib(physical_dimension_dict, physical_operation_dict, symbol_arity_mapping)
end


@testset "SBP Tests" begin
    @testset "TokenLib Creation" begin
        token_lib = create_token_lib_test()
        @test length(token_lib.physical_dimension_dict[]) == 8
        @test length(token_lib.physical_operation_dict[]) == 6
        @test length(token_lib.symbol_arity_mapping[]) == 8
    end

    @testset "Library Creation" begin
        token_lib = create_token_lib_test()
        features = Int8[7, 8]  # x1, x2
        functions = Int8[1, 2, 3, 4, 5, 6]  # mul, div, add, sub, sqr, sin
        constants = Int8[]
        lib = create_lib(token_lib, features, functions, constants; rounds=8, max_permutations=10000)
        total_len_lib = sum(length(entry) for entry in values(lib))
        @show ("Lib Entries:" , total_len_lib)
        @test !isempty(lib)
    end

    @testset "Compute Tree Creation" begin
        token_lib = create_token_lib_test()
        features = Int8[7, 8]  # x1, x2
        functions = Int8[1, 2, 3, 4, 5, 6]  # mul, div, add, sub, sqr, sin
        constants = Int8[]
        lib = create_lib(token_lib, features, functions, constants; rounds=6, max_permutations=10000)
        point_operations = Int8[1,2]
        inverse_operation = Dict{Int8, Function}(
            1 =>  (mul_unit_backward),
            2 =>  (div_unit_backward),
            5 => (sqr_unit_backward),
            6 => (zero_unit_backward)
        )
        token_dto = TokenDto(token_lib, point_operations, lib, inverse_operation, 1)

        expression = Int8[1,7,7]  # mul(x1, x1)
        tree = create_compute_tree(expression, token_dto)
        @test tree.symbol == 1  # mul
        @test length(tree.depend_on) == 2
        @test all(dep == 7 for dep in tree.depend_on)  # both dependencies are x1
    end

    @testset "Vector Dimension Calculation" begin
        token_lib = create_token_lib_test()
        features = Int8[7, 8]  # x1, x2
        functions = Int8[1, 2, 3, 4, 5, 6]  # mul, div, add, sub, sqr, sin
        constants = Int8[]
        lib = create_lib(token_lib, features, functions, constants; rounds=6, max_permutations=10000)
        point_operations = [1,2]
        inverse_operation = Dict{Int8, Function}(
            1 =>  (mul_unit_backward),
            2 =>  (div_unit_backward),
            5 => (sqr_unit_backward),
            6 => (zero_unit_backward)
        )
        token_dto = TokenDto(token_lib, point_operations, lib, inverse_operation, 1)

        expression = Int8[1,7,7]  # mul(x1, x1)
        tree = create_compute_tree(expression, token_dto)
        dimension = calculate_vector_dimension!(tree)
        @test dimension == [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end

    @testset "Enforce Backpropagation complex" begin
        token_lib = create_token_lib_test()
        features = Int8[7, 8]  # x1, x2
        functions = Int8[1, 2, 3, 4, 5, 6]  # mul, div, add, sub, sqr, sin
        constants = Int8[]
        lib = create_lib(token_lib, features, functions, constants; rounds=8, max_permutations=100000)
        point_operations = Int8[1,2]
        inverse_operation = Dict{Int8, Function}(
            1 =>  (mul_unit_backward),
            2 =>  (div_unit_backward),
            5 => (sqr_unit_backward),
            6 => (zero_unit_backward)
        )
        token_dto = TokenDto(token_lib, point_operations, lib, inverse_operation, 1)
        expression = Int8[1,2,3,7,7,8,8]  # mul(div(add(x1, x1), x2), x2): [1, 0, ...]
        tree = create_compute_tree(expression, token_dto)
        dimension = calculate_vector_dimension!(tree)
        propagate_necessary_changes!(tree, convert.(Float16,[1.0,-1.0,0.0,0.0,0.0, 0.0, 0.0]))
        dimension = calculate_vector_dimension!(tree)
        @test dimension == [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end

    @testset "Enforce Backpropagation simple" begin
        token_lib = create_token_lib_test()
        features = Int8[7, 8]  # x1, x2
        functions = Int8[1, 2, 3, 4, 5, 6]  # mul, div, add, sub, sqr, sin
        constants = Int8[]
        lib = create_lib(token_lib, features, functions, constants; rounds=6, max_permutations=10000)
        point_operations = Int8[1,2]
        inverse_operation = Dict{Int8, Function}(
            1 =>  (mul_unit_backward),
            2 =>  (div_unit_backward),
            5 => (sqr_unit_backward),
            6 => (zero_unit_backward)
        )
        token_dto = TokenDto(token_lib, point_operations, lib, inverse_operation, 1)
        expression = Int8[1,7,7] # mul(x1, x1): [2, 0, ...], repaired to [1, 1, 0, ...]
        tree = create_compute_tree(expression, token_dto)
        propagate_necessary_changes!(tree, convert.(Float16,[1.0,1.0,0.0,0.0,0.0, 0.0, 0.0]))
        dimension = calculate_vector_dimension!(tree)
        @test dimension == [1.0, 1.0, 0.0, 0.0,0.0, 0.0, 0.0]
    end

    @testset "Correct gene" begin
        token_lib = create_token_lib_test()
        features = Int8[7, 8]  # x1, x2
        functions = Int8[1, 2, 3, 4, 5, 6]  # mul, div, add, sub, sqr, sin
        constants = Int8[]
        lib = create_lib(token_lib, features, functions, constants; rounds=6, max_permutations=10000)
        point_operations = Int8[1,2]
        inverse_operation = Dict{Int8, Function}(
            1 =>  (mul_unit_backward),
            2 =>  (div_unit_backward),
            5 => (sqr_unit_backward),
            6 => (zero_unit_backward)
        )
        token_dto = TokenDto(token_lib, point_operations, lib, inverse_operation, 0)
        gene = Int8[1,4,1,2,7,8,7,3,4,7,7,8,1,4,8,7,8]
        gene_orig = copy(gene)
        @show "Genes before test" gene
        
        indices = [1] 
        # correct_genes! returns (distance, success): `correction` is the remaining distance
        correction, distance = correct_genes!(gene, indices, gene, Float16[1.0,1.0,0.0,0.0, 0.0, 0.0, 0.0] , token_dto; cycles=10)
        @show "Genes after test" gene
        @show "Correction" correction
        @test correction < eps(Float16)
        @test gene != gene_orig
    end


    @testset "calculate_vector_dimension!" begin
        token_lib = create_token_lib_test()
        features = Int8[7, 8]  # x1, x2
        functions = Int8[1, 2, 3, 4, 5, 6]  # mul, div, add, sub, sqr, sin
        constants = Int8[]
        lib = create_lib(token_lib, features, functions, constants; rounds=6, max_permutations=10000)
        point_operations = Int8[1, 2]
        inverse_operation = Dict{Int8, Function}(
            1 =>  (mul_unit_backward),
            2 =>  (div_unit_backward),
            5 => (sqr_unit_backward),
            6 => (zero_unit_backward)
        )
        token_dto = TokenDto(token_lib, point_operations, lib, inverse_operation, 1)
    
        # x1 is [1, 0, ...], x2 is [0, 1, ...]

        # mul(x1, x1): the exponents add
        tree1 = TempComputeTree(Int8(1), Int8[7, 7], Float16[], token_dto)
        dim1 = calculate_vector_dimension!(tree1)
        @test dim1 == [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
        # mul(x1, div(x2, x1)): x1 + (x2 - x1)
        tree2 = TempComputeTree(Int8(1), [
            Int8(7),
            TempComputeTree(Int8(2), Int8[8, 7], Float16[], token_dto)
        ], Float16[], token_dto)
        dim2 = calculate_vector_dimension!(tree2)
        @test dim2 == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
        # sqr(x1): the exponents double
        tree3 = TempComputeTree(Int8(5), Int8[7], Float16[], token_dto)
        dim3 = calculate_vector_dimension!(tree3)
        @test dim3 == [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    

        # add(mul(x1, x2), mul(x1, x2)): equal operand dimensions carry through the sum
        tree4 = TempComputeTree(Int8(3), [
            TempComputeTree(Int8(1), Int8[7, 8], Float16[], token_dto),
            TempComputeTree(Int8(1), Int8[7, 8], Float16[], token_dto)
        ], Float16[], token_dto)
        dim4 = calculate_vector_dimension!(tree4)
        @show dim4
        @test dim4 == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


        # sin(x1): sin needs a dimensionless operand, so the dimension is inconsistent (Inf)
        tree5 = TempComputeTree(Int8(6), Int8[7], Float16[], token_dto)
        dim5 = calculate_vector_dimension!(tree5)
        @test dim5 == Float16[Inf, Inf, Inf, Inf, Inf, Inf, Inf]  
    
    
        # flush! clears a stored dimension; the tree recomputes it from its operands
        tree6 = TempComputeTree(Int8(1), Int8[7, 8], Float16[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], token_dto)
        flush!(tree6)
        dim6 = calculate_vector_dimension!(tree6)
        @test dim6 == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end
end
