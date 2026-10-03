using Test
using Random

# On the tensor path a dimension vector carries the tensor order in slot 1 and the SI
# exponents after it, one slot more than on the scalar path. The backward unit rules must
# agree with the forward ones, or semantic backpropagation proposes a subtree whose forward
# dimension is not the one it asked for: a backward rule followed by the matching forward
# rule reproduces the expected dimension.
const SBP = GeneExpressionProgramming.SBPUtils

@testset "Tensor dimensions" begin
    f16(v) = Float16.(v)

    # [tensor order, kg, m, s, K, mol, A, cd]
    force = f16([1, 1, 1, -2, 0, 0, 0, 0])   # order 1, kg m/s^2
    mass = f16([0, 1, 0, 0, 0, 0, 0, 0])     # order 0, kg
    accel = f16([1, 0, 1, -2, 0, 0, 0, 0])   # order 1, m/s^2
    stress = f16([2, 1, -1, -2, 0, 0, 0, 0]) # order 2, kg/(m s^2)

    @testset "forward composes order and units" begin
        @test SBP.mul_t_unit_forward(mass, accel) == force
        @test SBP.div_t_unit_forward(force, mass) == accel
        # without units only the order composes
        @test SBP.mul_t_unit_forward(f16([0, 0]), f16([2, 0])) == f16([2, 0])
    end

    @testset "backward round trips through forward" begin
        for (u1, u2, expected) in ((mass, accel, force),
                                   (accel, mass, force),
                                   (mass, stress, f16([2, 2, -1, -2, 0, 0, 0, 0])))
            a, b = SBP.mul_t_unit_backward(u1, u2, expected)
            @test SBP.mul_t_unit_forward(a, b) == expected
        end

        a, b = SBP.div_t_unit_backward(force, mass, force)
        @test SBP.div_t_unit_forward(a, b) == force
    end

    @testset "length-matched empty and zero dimensions" begin
        # ZERO_DIM and EMPTY_DIM have the scalar width (7); the tensor path needs its own
        @test length(SBP.zero_dim(8)) == 8
        @test all(iszero, SBP.zero_dim(8))
        @test SBP.has_inf16(SBP.empty_dim(8))
        @test !SBP.has_inf16(SBP.zero_dim(8))
    end

    @testset "order and units split and rejoin" begin
        order, units = SBP.split_dim(force)
        @test order == 1
        @test units == f16([1, 1, -2, 0, 0, 0, 0])
        @test SBP.join_dim(order, units) == force
    end

    @testset "seeds joined by + carry the target in every gene" begin
        # the tensor path's connectors are operator objects, so `+` is recognised by its
        # unit rule; seeds must be homogeneous as a whole
        d(m_, s_) = f16([0, 0, m_, s_, 0, 0, 0, 0])
        dims = Dict{Symbol,Vector{Float16}}(:x1 => d(1, -1), :x2 => d(0, -1),
            :x3 => d(-1, -1), :x4 => d(2, -1))        # u, u_x, u_xx, nu2
        target = d(1, -2)                            # u_t
        Random.seed!(5)
        reg = GepTensorRegressor(4; problem_dimension=1, gene_count=3, head_len=4,
            entered_non_terminals=[:+, :-, :*], gene_connections=[:+, :-],
            considered_dimensions=dims)
        tb, dto = reg.toolbox_, reg.token_dto_
        seeder = GeneExpressionProgramming.RegressionWrapper.make_lib_seeder(
            tb, dto, target, 1.0)
        pop = GeneExpressionProgramming.GepEntities.generate_population(40, tb)
        foreach(c -> GeneExpressionProgramming.GepEntities.compile_expression!(c;
            force_compile=true), pop)
        seeder(pop)
        @test all(c -> is_dimensionally_homogeneous(c.expression_raw, target, dto), pop)
    end
end
