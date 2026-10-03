using Test
using Random
using Statistics

# A seeded run gives the same result every time. The fitness loop is threaded: start Julia
# with several threads (`julia -t 4`) for this to cover the parallel case. The population
# is wide so that duplicate expressions, which the loop scores from its cache with a
# penalty, are likely to occur.
@testset "Reproducibility" begin
    # exp(x1) * x2 is out of reach without exp, so the loss stays above zero; on a target
    # solved exactly, both runs would sit at zero and the comparison would be vacuous
    function run_once(; seed::Int=1, epochs::Int=30, pop::Int=800)
        Random.seed!(seed)
        rng = MersenneTwister(seed)
        n = 60
        x = randn(rng, 2, n)
        y = @views exp.(x[1, :]) .* x[2, :]

        reg = GepRegressor(2; entered_features=[:x1, :x2],
            entered_non_terminals=[:+, :-, :*],
            gene_count=3, head_len=5)
        fit!(reg, epochs, pop, x, y; loss_fun="mse")
        return mean(reg.best_models_[1].fitness)
    end

    a = run_once()
    b = run_once()
    @test isfinite(a)
    @test a > 0          # the check below is only meaningful on an unconverged run
    @test a == b

    @testset "a dimensional repair does not depend on the task that runs it" begin
        # the repair pass runs in tasks whose number depends on the thread count: a repair
        # must be the same whichever task runs it and in whatever order, and must leave the
        # random stream of the calling task as it was
        GR = GeneExpressionProgramming.GepRegression
        Random.seed!(7)
        reg = GepRegressor(2; entered_features=[:x, :y], entered_non_terminals=[:+, :-, :*, :/],
            considered_dimensions=Dict(:x => Float16[0, 1, 0, 0, 0, 0, 0],
                :y => Float16[0, 0, 1, 0, 0, 0, 0]),
            gene_count=2, head_len=4, rounds=2, max_permutations_lib=500)
        repair, check = GeneExpressionProgramming.RegressionWrapper.make_dimension_contract(
            reg.toolbox_, reg.token_dto_, Float16[0, 1, -1, 0, 0, 0, 0], 10)
        pop = generate_population(60, reg.toolbox_)
        forward, backward = deepcopy(pop), reverse(deepcopy(pop))
        Random.seed!(1)
        GR.perform_correction_callback!(forward, 1, 1, 1.0, repair; homogeneity_check=check)
        Random.seed!(2)
        GR.perform_correction_callback!(backward, 1, 1, 1.0, repair; homogeneity_check=check)
        @test count(c -> c.dimension_homogene, forward) > 0
        @test [c.genes for c in forward] == reverse([c.genes for c in backward])
        Random.seed!(3)
        untouched = rand(5)
        Random.seed!(3)
        GR.perform_correction_callback!(deepcopy(pop), 1, 1, 1.0, repair; homogeneity_check=check)
        @test rand(5) == untouched
    end

    @testset "duplicates are ranked, not left NaN" begin
        # the returned models carry numeric fitness: a NaN reaching `sort!` would leave
        # the population in arbitrary order
        Random.seed!(3)
        rng = MersenneTwister(3)
        n = 40
        x = randn(rng, 2, n)
        y = @views x[1, :] .+ x[2, :]
        reg = GepRegressor(2; entered_features=[:x1, :x2],
            entered_non_terminals=[:+, :-, :*],
            gene_count=2, head_len=3)
        fit!(reg, 20, 400, x, y; loss_fun="mse")
        @test all(m -> !isnan(mean(m.fitness)), reg.best_models_)
    end
end

# The tensor unit rules compose both operands: slot 1 carries the tensor order, and the SI
# exponents add under a product or contraction and subtract under a quotient.
const SBP = GeneExpressionProgramming.SBPUtils

@testset "Tensor unit composition" begin
    f16(v) = Float16.(v)
    # slot 1 is the tensor order, then the first four SI exponents [kg, m, s, K]
    vel = f16([1.0, 0.0, 1.0, -1.0, 0.0])    # order 1, m/s
    mass = f16([0.0, 1.0, 0.0, 0.0, 0.0])    # order 0, kg
    grad = f16([2.0, 0.0, 0.0, -1.0, 0.0])   # order 2, 1/s

    @test SBP.mul_t_unit_forward(mass, vel) == f16([1, 1, 1, -1, 0])       # kg m/s
    @test SBP.div_t_unit_forward(vel, mass) == f16([1, -1, 1, -1, 0])      # m/(s kg)
    @test SBP.contraction_unit_forward(grad, vel) == f16([1, 0, 1, -2, 0]) # m/s^2
    @test SBP.crossp_unit_forward(vel, vel) == f16([1, 0, 2, -2, 0])       # m^2/s^2

    # without units only the order composes: a scalar times a second-order tensor
    @test SBP.mul_t_unit_forward(f16([0, 0]), f16([2, 0])) == f16([2, 0])

    # invalid combinations give an inconsistent (Inf) dimension
    @test SBP.has_inf16(SBP.mul_t_unit_forward(vel, grad))
    @test SBP.has_inf16(SBP.contraction_unit_forward(mass, mass))
end
