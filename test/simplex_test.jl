using Test
using Random
using Statistics
using LinearAlgebra
using Optim

const GS = GeneExpressionProgramming.GepSimplex
const RW = GeneExpressionProgramming.RegressionWrapper
const SUR = GeneExpressionProgramming.GepSurrogate

# a function of a vector that counts its calls: the calls are what the screening saves
function simplex_counted(f)
    calls = Ref(0)
    return (x -> (calls[] += 1; f(x))), calls
end

simplex_rosenbrock(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
simplex_scaled(x) = sum((i * (x[i] - i))^2 for i in eachindex(x))

# four constants, condition number 10, the axes rotated
const SIMPLEX_A = let
    Q = Matrix(qr(randn(MersenneTwister(5), 4, 4)).Q)
    Q * Diagonal(exp.(range(0, log(10.0); length=4))) * Q'
end
const SIMPLEX_CENTER = [-1.0, -0.3, 0.3, 1.0]
simplex_rotated(x) = dot(x .- SIMPLEX_CENTER, SIMPLEX_A * (x .- SIMPLEX_CENTER))

# two basins: the global minimum 0 at (1.2, 0.8), a local one near (-0.8, -0.6)
simplex_twowell(x) = sum(abs2, x .- [1.2, 0.8]) * sum(abs2, x .- [-0.8, -0.6]) +
                     0.2 * sum(abs2, x .- [1.2, 0.8])

# a regressor, data with a coefficient to find, and a counted custom loss with the MSE and
# the size of the model as objectives
function simplex_problem(; seed::Int=4)
    Random.seed!(seed)
    rng = MersenneTwister(seed)
    x = randn(rng, 2, 40)
    y = @views 1.7 .* x[1, :] .* x[2, :] .+ 0.3 .* x[1, :]
    reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=2, head_len=4,
        number_of_objectives=2)
    ctxs = thread_contexts(reg.toolbox_, x)
    calls = Threads.Atomic{Int}(0)
    function loss(elem, validate::Bool)
        if isnan(mean(elem.fitness)) || validate
            Threads.atomic_add!(calls, 1)
            p = elem(ctxs[Threads.threadid()])
            elem.fitness = p isa AbstractVector && all(isfinite, p) ?
                           (mean(abs2, p .- y), 0.01 * length(elem.expression_raw)) : (Inf, Inf)
        end
    end
    return (x=x, y=y, reg=reg, loss=loss, calls=calls)
end

drawn_constants(c) = [Float64(c.toolbox.nodes[c.expression_raw[p]]) for p in constant_positions(c)]

# the loss of a copy of `c` with the constants `constants` (nothing: the drawn ones)
function simplex_rescore(c, loss, constants)
    d = deepcopy(c)
    d.optimised_constants = constants
    d.fitness = d.toolbox.fitness_reset[2]
    loss(d, true)
    return d.fitness
end

@testset "Screened Nelder-Mead" begin

    @testset "options" begin
        @test_throws ArgumentError ScreenedNelderMead(max_evaluations=0)
        @test_throws ArgumentError ScreenedNelderMead(kappa=-1)
        @test_throws ArgumentError ScreenedNelderMead(polish_radius=0)
        @test_throws ArgumentError ScreenedNelderMead(polish_radius=2)
        @test_throws ArgumentError ScreenedNelderMead(target_transform=:cube)
        @test ScreenedNelderMead(target_transform="asinh").target_transform === :asinh
        @test_throws ArgumentError simplex_search(sum, Float64[])
    end

    @testset "the initial simplex is Optim's" begin
        S = GS.initial_simplex([2.0, 0.0, -3.0])
        @test S == [[2.0, 0.0, -3.0], [3.025, 0.0, -3.0], [2.0, 0.025, -3.0], [2.0, 0.0, -4.475]]
        # Optim's move vanishes at -0.05, which leaves its simplex flat
        @test GS.initial_simplex([-0.05, 1.0])[2] == [-0.05 + 0.025, 1.0]
    end

    @testset "plain mode takes the steps of Optim.NelderMead" begin
        for (f, x0) in ((simplex_rosenbrock, [-1.2, 1.0]), (simplex_scaled, [0.3, -0.5, 1.0, 2.0]))
            g, calls = simplex_counted(f)
            res = Optim.optimize(g, x0, Optim.NelderMead(), Optim.Options(iterations=1000))
            mine = simplex_search(f, x0; method=ScreenedNelderMead(screen=false,
                max_evaluations=10_000))
            @test mine.evaluations == calls[]
            @test isapprox(mine.minimizer, Optim.minimizer(res); rtol=1e-12)
            @test mine.proposals == 0
        end
        # where Optim's simplex is flat, this one is not
        r = simplex_search(simplex_scaled, [-0.05, 0.7]; method=ScreenedNelderMead(screen=false,
            max_evaluations=400))
        @test r.minimum < 1e-6
    end

    @testset "the budget, and the best point the loss scored" begin
        for screen in (false, true), budget in (1, 2, 5, 40)
            g, calls = simplex_counted(simplex_rotated)
            r = simplex_search(g, zeros(4); method=ScreenedNelderMead(screen=screen,
                max_evaluations=budget))
            @test r.evaluations == calls[] <= budget
            @test length(r.points) == length(r.values) == length(r.history) == r.evaluations
            @test r.minimum == minimum(r.values) == r.history[end]
            @test r.minimizer in r.points
            @test simplex_rotated(r.minimizer) == r.minimum
            @test issorted(r.history; rev=true)
            @test r.proposals <= r.evaluations
        end
        r = simplex_search(simplex_rotated, zeros(4); method=ScreenedNelderMead(max_evaluations=40))
        # the process placed every call after the initial simplex, until the polish
        @test r.evaluations == 40 && r.proposals > 20
        @test occursin("loss calls", sprint(show, r))
    end

    @testset "a failed loss call is the worst value" begin
        # not finite where the first constant is below 0.5, next to the minimum at (0.6, 2)
        f(x) = x[1] < 0.5 ? NaN : (x[1] - 0.6)^2 + (x[2] - 2)^2
        plain, screened = [simplex_search(f, [1.5, 0.5]; method=ScreenedNelderMead(screen=screen,
            max_evaluations=100)) for screen in (false, true)]
        for r in (plain, screened)
            @test !any(isnan, r.values) && any(isinf, r.values)
            @test r.minimum < 1e-6
        end
        # the process learns where the loss fails
        @test count(isinf, screened.values) < count(isinf, plain.values)
    end

    @testset "the screening saves loss calls" begin
        rng = MersenneTwister(3)
        plain, screened = Float64[], Float64[]
        for _ in 1:6
            x0 = 4 .* rand(rng, 4) .- 2
            push!(plain, simplex_search(simplex_rotated, x0;
                method=ScreenedNelderMead(screen=false, max_evaluations=40)).minimum)
            push!(screened, simplex_search(simplex_rotated, x0;
                method=ScreenedNelderMead(max_evaluations=40)).minimum)
        end
        @test median(screened) < 0.5 * median(plain)
        @test count(screened .< plain) >= 4
    end

    @testset "the screened swarm" begin
        @test_throws ArgumentError ScreenedNelderMead(swarm_box=1.0, screen=false)
        @test_throws ArgumentError ScreenedNelderMead(swarm_box=0)
        @test_throws ArgumentError ScreenedNelderMead(swarm_box=-1.0)
        @test_throws ArgumentError ScreenedNelderMead(swarm_box=([1.0], [0.0]))
        @test_throws DimensionMismatch ScreenedNelderMead(swarm_box=([0.0, 0.0], [1.0]))
        @test_throws ArgumentError ScreenedNelderMead(swarm_box="wide")
        @test_throws ArgumentError ScreenedNelderMead(particles=2)
        @test ScreenedNelderMead(swarm_box=([0, 0], [1, 1])).swarm_box == ([0.0, 0.0], [1.0, 1.0])
        @test ScreenedNelderMead(swarm_box=2).swarm_box === 2.0
        box = ([-2.0, -2.0], [2.0, 2.0])
        @test_throws DimensionMismatch simplex_search(simplex_twowell, [0.0, 0.0, 0.0];
            method=ScreenedNelderMead(swarm_box=box))
        @test_throws ArgumentError simplex_search(simplex_twowell, [3.0, 0.0];
            method=ScreenedNelderMead(swarm_box=box))

        # the budget, the best point scored, and a swarm that starts at the start and
        # inside its box: the start and a Latin hypercube of the box, one point per stratum
        for budget in (1, 5, 12, 60)
            g, calls = simplex_counted(simplex_twowell)
            r = simplex_search(g, [-0.8, -0.6]; method=ScreenedNelderMead(max_evaluations=budget,
                swarm_box=box, particles=6))
            @test r.evaluations == calls[] <= budget
            @test r.minimum == minimum(r.values) == r.history[end] && r.minimizer in r.points
            @test issorted(r.history; rev=true)
            first_swarm = r.points[1:min(6, end)]
            @test first_swarm[1] == [-0.8, -0.6]
            @test all(x -> all(box[1] .<= x .<= box[2]), first_swarm)
            if budget >= 6
                for j in 1:2
                    @test sort(floor.(Int, (getindex.(first_swarm[2:6], j) .+ 2) ./ 4 .* 5)) == 0:4
                end
            end
        end
        # a relative box: r initial steps around the start
        lower, upper = GS.swarm_bounds(2.0, [1.0, -0.05])
        @test lower ≈ [1.0 - 2 * 0.525, -0.05 - 2 * 0.025] && upper ≈ [1.0 + 2 * 0.525, -0.05 + 2 * 0.025]

        # from the basin of a local minimum, Nelder-Mead and the screened search stay in it,
        # the swarm finds the global one
        rng = MersenneTwister(4)
        starts = [[-0.8, -0.6] .+ 0.3 .* randn(rng, 2) for _ in 1:10]
        found(method) = count(x0 -> simplex_search(simplex_twowell, x0; method=method).minimum < 1e-4,
            starts)
        @test found(ScreenedNelderMead(screen=false, max_evaluations=60)) == 0
        @test found(ScreenedNelderMead(max_evaluations=60)) == 0
        @test found(ScreenedNelderMead(max_evaluations=60, swarm_box=box)) == 10
    end

    @testset "the constants of a chromosome" begin
        prob = simplex_problem()
        tb = prob.reg.toolbox_
        pop = generate_population(60, tb)
        for c in pop
            c.fitness = tb.fitness_reset[2]
            prob.loss(c, false)
        end
        bare = first(c for c in pop if isempty(constant_positions(c)))
        held = bare.fitness
        @test isnothing(optimize_constants!(bare, prob.loss))
        @test bare.fitness == held && isnothing(bare.optimised_constants)

        candidates = [c for c in pop if !isempty(constant_positions(c)) && all(isfinite, c.fitness)]
        originals = deepcopy(candidates)
        improved = 0
        for c in candidates[1:min(end, 8)]
            held = c.fitness
            before = prob.calls[]
            r = optimize_constants!(c, prob.loss; method=ScreenedNelderMead(max_evaluations=15))
            @test r isa SimplexSearch
            @test prob.calls[] - before == r.evaluations <= 15
            @test r.points[1] == drawn_constants(c)
            if r.minimum < r.values[1]
                improved += 1
                @test c.optimised_constants == r.minimizer
                @test mean(c.fitness) == r.minimum
                # the fitness is the loss of the tuned constants, and evaluation applies them
                @test simplex_rescore(c, prob.loss, c.optimised_constants) == c.fitness
                @test simplex_rescore(c, prob.loss, nothing) != c.fitness
                @test mean(abs2, c(prob.x) .- prob.y) ≈ c.fitness[1]
                # a second search starts from the tuned constants
                r2 = optimize_constants!(c, prob.loss; method=ScreenedNelderMead(screen=false,
                    max_evaluations=6))
                @test r2.points[1] == r.minimizer
                @test mean(c.fitness) <= r.minimum
            else
                @test isnothing(c.optimised_constants) && c.fitness == held
            end
        end
        @test improved >= 3

        # the objective that is minimized: an index or a function of the fitness
        c = deepcopy(originals[1])
        r = optimize_constants!(c, prob.loss; objective=1, method=ScreenedNelderMead(max_evaluations=12))
        @test r.values[1] == originals[1].fitness[1]
        @test r.minimum < r.values[1] ? c.fitness[1] == r.minimum : isnothing(c.optimised_constants)
        c = deepcopy(originals[1])
        r = optimize_constants!(c, prob.loss; objective=f -> 10 * f[1],
            method=ScreenedNelderMead(screen=false, max_evaluations=12))
        @test r.values[1] ≈ 10 * originals[1].fitness[1]

        # with a swarm in a box of one initial step around the constants the chromosome holds
        c = deepcopy(originals[1])
        r = optimize_constants!(c, prob.loss; method=ScreenedNelderMead(max_evaluations=25,
            swarm_box=1.0, particles=5))
        lower, upper = GS.swarm_bounds(1.0, drawn_constants(originals[1]))
        @test r.evaluations <= 25 && r.points[1] == drawn_constants(originals[1])
        @test all(x -> all(lower .<= x .<= upper), r.points[1:5])
        @test r.minimum < r.values[1] ? c.optimised_constants == r.minimizer &&
                                        mean(c.fitness) == r.minimum :
              isnothing(c.optimised_constants)

        # a loss that ignores the constants: nothing improves, the chromosome is left alone
        c = deepcopy(originals[1])
        held = c.fitness
        flat(elem, validate) = (elem.fitness = (1.0, 1.0))
        r = optimize_constants!(c, flat; method=ScreenedNelderMead(max_evaluations=10))
        @test r.evaluations == 10 && r.minimum == r.values[1]
        @test isnothing(c.optimised_constants) && c.fitness == held

        # a loss that throws: the chromosome is restored and the error passes on
        n_calls = Ref(0)
        failing(elem, validate) = (n_calls[] += 1; n_calls[] > 3 && error("solver failed");
                                   prob.loss(elem, validate))
        @test_throws ErrorException optimize_constants!(c, failing)
        @test isnothing(c.optimised_constants) && c.fitness == held
    end

    @testset "tuned constants in the parts of a multi-expression model" begin
        Random.seed!(2)
        x = randn(MersenneTwister(2), 2, 30)
        reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=4, head_len=3)
        ctx = buffer_context(reg.toolbox_, x)
        pop = generate_population(40, reg.toolbox_)
        for c in pop, k in (1, 2, 4)
            @test [c.expression_raw[p] for p in split_positions(c, k)] ==
                  [collect(part) for part in split_karva(c, k)]
        end
        same(a, b) = a isa AbstractVector && b isa AbstractVector ? a ≈ b : isequal(a, b)
        tested = 0
        for c in pop
            positions = constant_positions(c)
            length(positions) >= 2 || continue
            tested += 1
            untuned = split_predict(c, ctx, 2)
            drawn = drawn_constants(c)
            # the drawn constants as tuned ones change nothing (tuned constants are printed
            # to 6 significant digits)
            c.optimised_constants = copy(drawn)
            @test all(same.(split_predict(c, ctx, 2), untuned))
            drawn_equations = split_equations(c, 2)
            # with one part, the part is the model
            c.optimised_constants = drawn .+ 0.37
            @test same(split_predict(c, ctx, 1)[1], c(ctx))
            @test split_equations(c, 1)[1] == equation_string(c)
            # the constants of the first part move it alone
            first_part = Set(split_positions(c, 2)[1])
            c.optimised_constants = [p in first_part ? v + 0.5 : v for (p, v) in zip(positions, drawn)]
            moved = split_predict(c, ctx, 2)
            @test same(moved[2], untuned[2])
            equations = split_equations(c, 2)
            @test equations[2] == drawn_equations[2]
            any(in(first_part), positions) && @test occursin(string(round(drawn[findfirst(in(first_part),
                positions)] + 0.5; sigdigits=6)), equations[1])
            c.optimised_constants = nothing
        end
        @test tested >= 10
    end

    @testset "predictT with the tuned constants of a chromosome" begin
        Random.seed!(5)
        n = 30
        x1, x2 = randn(n), randn(n)
        reg = GepTensorRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=2, head_len=4,
            feature_names=["x1", "x2"], entered_terminal_nums=[0.5, 1.0], rnd_count=2)
        allocate_buffers!(reg, [x1, x2])
        tb = reg.toolbox_
        tested = 0
        for c in generate_population(40, tb)
            positions = constant_positions(c)
            isempty(positions) && continue
            untuned = predictT(reg, c.expression_raw)
            untuned isa AbstractVector || continue
            untuned = copy(untuned)
            tested += 1
            @test predictT(reg, c) ≈ untuned
            c.optimised_constants = drawn_constants(c)
            @test predictT(reg, c) ≈ untuned
            # every occurrence one up is every constant symbol one up
            c.optimised_constants = drawn_constants(c) .+ 1.0
            symbols = unique(c.expression_raw[positions])
            @test predictT(reg, c) ≈ predictT(reg, c.expression_raw,
                Dict(sym => tb.nodes[sym] + 1.0 for sym in symbols))
        end
        @test tested >= 5
    end

    @testset "fit! tunes the best chromosome" begin
        Random.seed!(3)
        rng = MersenneTwister(3)
        x = randn(rng, 2, 50)
        y = @views 1.7 .* x[1, :] .* x[2, :] .+ 0.3 .* x[1, :]
        reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=2, head_len=4)
        ctxs = thread_contexts(reg.toolbox_, x)
        tuned_calls = Threads.Atomic{Int}(0)
        function loss(elem, validate::Bool)
            if isnan(mean(elem.fitness)) || validate
                isnothing(elem.optimised_constants) || Threads.atomic_add!(tuned_calls, 1)
                p = elem(ctxs[Threads.threadid()])
                elem.fitness = p isa AbstractVector && all(isfinite, p) ?
                               (mean(abs2, p .- y),) : (Inf,)
            end
        end
        fit!(reg, 20, 80, loss; constant_optimizer=ScreenedNelderMead(max_evaluations=20),
            optimization_epochs=4)
        @test tuned_calls[] > 0
        for m in reg.best_models_
            @test isapprox(m.fitness[1], mean(abs2, m(x) .- y); rtol=1e-8, atol=1e-20)
        end
    end

    @testset "with a surrogate, only a chromosome the loss scored is tuned" begin
        prob = simplex_problem(; seed=6)
        tb = prob.reg.toolbox_
        s = SurrogateScreening(prob.reg, prob.x[:, 1:16]; individuals_per_epoch=6,
            warmup_runs=20, seed=2)
        pop = [c for c in generate_population(60, tb) if !isempty(constant_positions(c))]
        foreach(c -> prob.loss(c, false), pop)
        scored, predicted = [c for c in pop if all(isfinite, c.fitness)][1:2]
        SUR.record_validation!(s, scored)
        predicted.fitness = (0.0, 0.0)
        SUR.mark_prediction!(s, predicted)
        @test is_validated(s, scored) && !is_validated(s, predicted)
        seen = Set{UInt}()
        function watched(elem, validate)
            push!(seen, objectid(elem))
            prob.loss(elem, validate)
        end
        hook = RW.leader_constant_search(ScreenedNelderMead(max_evaluations=8), watched, s, 2)
        hook([predicted, scored])
        @test seen == Set([objectid(scored)])
        @test predicted.fitness == (0.0, 0.0) && isnothing(predicted.optimised_constants)
        @test isnothing(RW.leader_constant_search(nothing, watched, s, 2))

        # a whole search: the memory of the surrogate keeps the losses of drawn constants
        s = SurrogateScreening(prob.reg, prob.x[:, 1:16]; individuals_per_epoch=6,
            warmup_runs=20, seed=3)
        fit!(prob.reg, 12, 60, prob.loss; surrogate=s, hof=10,
            constant_optimizer=ScreenedNelderMead(max_evaluations=10), optimization_epochs=2)
        @test any(m -> !isnothing(m.optimised_constants), prob.reg.best_models_)
        for m in prob.reg.best_models_
            @test is_validated(s, m)
            isnothing(m.optimised_constants) && continue
            @test isequal(SUR.rescore_known(s, m.expression_raw),
                simplex_rescore(m, prob.loss, nothing))
            @test m.fitness == simplex_rescore(m, prob.loss, m.optimised_constants)
        end
    end
end
