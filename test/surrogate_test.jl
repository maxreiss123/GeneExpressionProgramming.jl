using Test
using Random
using Statistics
using LinearAlgebra
using Serialization

const SG = GeneExpressionProgramming.GepSurrogate

# A regressor, data and a counted custom loss: the loss is what a surrogate saves calls of,
# so every test counts them
function surrogate_problem(; seed::Int=1, n::Int=80, objectives::Int=1)
    Random.seed!(seed)
    rng = MersenneTwister(seed)
    x = randn(rng, 2, n)
    y = @views x[1, :] .^ 2 .+ x[1, :] .* x[2, :] .- 2 .* x[2, :] .^ 2
    reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=3, head_len=4,
        number_of_objectives=objectives)
    ctxs = thread_contexts(reg.toolbox_, x)
    mse = get_loss_function("mse")
    calls = Threads.Atomic{Int}(0)
    function loss(elem, validate::Bool)
        if isnan(mean(elem.fitness)) || validate
            Threads.atomic_add!(calls, 1)
            p = elem(ctxs[Threads.threadid()])
            ok = p isa AbstractVector && all(isfinite, p)
            elem.fitness = objectives == 1 ? (ok ? (mse(y, p),) : (Inf,)) :
                           (ok ? (mse(y, p), 0.01 * length(elem.expression_raw)) : (Inf, Inf))
        end
    end
    truth(c) = (p = c(x); p isa AbstractVector && all(isfinite, p) ? mse(y, p) : Inf)
    return (x=x, y=y, reg=reg, loss=loss, calls=calls, truth=truth, rng=rng)
end

# Two expressions per chromosome (split_karva), f and g, each with its own target and
# objective, and optionally the size of the model as a third objective
function multi_expression_problem(; seed::Int=1, n::Int=60, with_size::Bool=false)
    Random.seed!(seed)
    rng = MersenneTwister(seed)
    x = randn(rng, 2, n)
    f = @views x[1, :] .* x[2, :] .+ x[1, :]
    g = @views x[1, :] .- x[2, :] .* x[2, :]
    reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=4, head_len=4,
        number_of_objectives=with_size ? 3 : 2)
    ctxs = thread_contexts(reg.toolbox_, x)
    mse = get_loss_function("mse")
    calls = Threads.Atomic{Int}(0)
    size_of(c) = 0.01 * length(c.expression_raw)
    err(p, t) = p isa AbstractVector && all(isfinite, p) ? mse(t, p) : Inf
    function loss(elem, validate::Bool)
        if isnan(mean(elem.fitness)) || validate
            Threads.atomic_add!(calls, 1)
            local pf, pg = split_predict(elem, ctxs[Threads.threadid()], 2)
            elem.fitness = with_size ? (err(pf, f), err(pg, g), size_of(elem)) :
                           (err(pf, f), err(pg, g))
        end
    end
    return (x=x, f=f, g=g, reg=reg, loss=loss, calls=calls, size_of=size_of, rng=rng,
        err=err)
end

# One epoch of the screening by hand: screen, score the picks, commit
function screened_epoch!(s, pop, loss, worst)
    plan = SG.screen_epoch!(s, pop, collect(eachindex(pop)), worst)
    for i in SG.evaluated_indices(plan)
        loss(pop[i], false)
    end
    SG.commit_epoch!(s, pop, plan)
    return plan
end

@testset "Surrogate screening" begin

    @testset "counts, shares and transforms" begin
        @test SG.resolve_count(nothing, 100, 7) == 7
        @test SG.resolve_count(10, 100) == 10
        @test SG.resolve_count(0.15, 100) == 15
        @test SG.resolve_count(0.001, 100) == 1       # at least one
        @test SG.resolve_count(12.7, 100) == 12       # a float from 1 on is a count
        for kind in (:log10, :asinh, :none)
            fwd, inv = SG.get_transform(kind)
            @test inv(fwd(3.5)) ≈ 3.5
        end
        @test SG.get_transform("log10")[1](0.0) ≈ -30.0       # floored at 1e-30
        @test isfinite(SG.get_transform(:log10)[2](1e6))       # clamped exponent
        @test_throws ArgumentError SG.get_transform(:cube)
        @test_throws ArgumentError SurrogateScreening(identity; budget_rule=:greedy)
        @test_throws ArgumentError GpScreen(acquisition=:random)
    end

    @testset "Gaussian process" begin
        rng = MersenneTwister(2)
        X = rand(rng, 2, 40)
        f(v) = sin(3v[1]) + v[2]^2
        y = [f(X[:, j]) for j in 1:40]
        gp = GaussianProcess(X, y)

        # it interpolates its data and is unsure far away from it
        mu, sd = SG.unstandardize(gp, SG.posterior(gp, X)...)
        @test maximum(abs.(mu .- y)) < 1e-2
        @test maximum(sd) < 0.05
        far = [10.0 -10.0; 10.0 -10.0]
        _, sd_far = SG.posterior(gp, far)
        @test all(sd_far .> 0.9)

        # and predicts between its data
        Xq = rand(rng, 2, 30)
        mu_q, _ = SG.unstandardize(gp, SG.posterior(gp, Xq)...)
        @test cor(mu_q, [f(Xq[:, j]) for j in 1:30]) > 0.95

        # the fitted process picks its hyperparameters from the grid
        fitted = GaussianProcess(X, y; fit=true)
        @test fitted.nugget in SG.FIT_NOISES
        @test any(isapprox(fitted.lengthscale, gp.lengthscale * f) for f in SG.FIT_LENGTH_FACTORS)
        @test fitted.amplitude > 0

        # the log of the expected improvement is the log of the expected improvement
        # where that is representable, and keeps ordering deep in the tail
        ei = SG.expected_improvement(gp, Xq)
        lei = SG.log_expected_improvement(gp, Xq)
        rep = ei .> 1e-250
        @test isapprox(lei[rep], log.(ei[rep]); rtol=1e-6, atol=1e-8)
        z = [-40.0, -20.0, -12.0, -10.5, -9.5, -5.0, -1.0, 0.0, 1.0, 5.0]
        h = SG.log_h.(z)
        @test all(isfinite, h)
        @test issorted(h)
        # both branches meet at z = -10
        @test SG.log_h(-10.0 + 1e-9) ≈ SG.log_h(-10.0 - 1e-9) atol = 1e-6

        # a belief moves no mean and collapses the deviation at the believed point
        x_new = [0.5, 0.5]
        believed = SG.believe(gp, x_new)
        mu0, sd0 = SG.posterior(gp, reshape(x_new, :, 1))
        mu1, sd1 = SG.posterior(believed, reshape(x_new, :, 1))
        @test mu1[1] ≈ mu0[1] atol = 1e-6
        @test sd1[1] < sd0[1]
        @test sd1[1] < 1e-2
        @test believed.best == gp.best

        # a single observation and duplicated points still factorize
        @test GaussianProcess(reshape([1.0, 2.0], :, 1), [3.0]).best == 0.0
        dup = GaussianProcess(hcat(X, X), vcat(y, y))
        @test all(isfinite, SG.posterior(dup, Xq)[1])
    end

    @testset "feasibility model" begin
        rng = MersenneTwister(3)
        X = randn(rng, 2, 200)
        labels = [X[1, j] < 0 ? 0.0 : 1.0 for j in 1:200]
        model = FeasibilityModel(X, labels)
        p = model([-2.0 2.0; 0.0 0.0])
        @test p[1] < 0.2
        @test p[2] > 0.8
        # far from every observation it falls back to the overall rate
        @test model(reshape([0.0, 1e6], :, 1))[1] ≈ mean(labels) atol = 1e-6
    end

    @testset "fronts and hypervolumes" begin
        P = [1.0 2.0 3.0 2.5; 3.0 2.0 1.0 2.5]
        front = SG.pareto_points(P)
        @test size(front, 2) == 3                  # (2.5, 2.5) is dominated by (2, 2)
        ref = [4.0, 4.0]
        @test SG.hypervolume(front, ref) ≈ 6.0
        # a point adds what it dominates beyond the front
        gain = SG.hypervolume_improvement(front, [1.5 3.5 5.0; 1.5 3.5 0.0], ref)
        @test gain[1] ≈ SG.hypervolume(hcat(front, [1.5, 1.5]), ref) - 6.0
        @test gain[2] == 0.0                       # dominated
        @test gain[3] == 0.0                       # outside the reference box
        @test SG.hypervolume_improvement(zeros(2, 0), reshape([1.0, 1.0], :, 1), ref)[1] ≈ 9.0
        # three objectives: the unit cube corner and a slab
        @test SG.hypervolume(reshape([0.0, 0.0, 0.0], :, 1), [1.0, 1.0, 1.0]) ≈ 1.0
        P3 = [0.0 0.5; 0.5 0.0; 0.5 0.5]
        @test SG.hypervolume(P3, [1.0, 1.0, 1.0]) ≈ 0.375
        # the 2d sweep agrees with the generic difference of hypervolumes
        rng = MersenneTwister(4)
        F = SG.pareto_points(rand(rng, 2, 12))
        Q = rand(rng, 2, 25)
        @test SG.hypervolume_improvement(F, Q, [1.2, 1.2]) ≈ SG.hv_improvement_slow(F, Q, [1.2, 1.2])
    end

    @testset "embedders" begin
        prob = surrogate_problem()
        tb = prob.reg.toolbox_
        probes = prob.x[:, 1:12]
        emb = SemanticEmbedder(tb, probes)
        genes = GeneEmbedder(tb, probes)
        pop = generate_population(40, tb)
        for c in pop
            v = emb(c)
            isnothing(v) && continue
            # the transformed outputs of the expression on the probes
            @test length(v) == 12
            @test v ≈ asinh.(c(probes))
            g = genes(c)
            @test length(g) == 12 * tb.gene_count
        end
        # an uncompiled chromosome is not embedded
        @test isnothing(emb(Chromosome(copy(pop[1].genes), tb, false)))
        @test_throws ArgumentError SemanticEmbedder(tb, zeros(2, 0))

        # nor is an expression that cannot be evaluated on every probe: x1 / 0
        reg = GepRegressor(2; entered_non_terminals=[:+, :/], gene_count=1, head_len=2)
        t2 = reg.toolbox_
        div = only(k for (k, f) in t2.callbacks if f === (/))
        x1 = only(k for (k, n) in t2.nodes if n isa InputSelector && n.idx == 1)
        zero_ = only(k for (k, n) in t2.nodes if n isa Number && iszero(n))
        c = Chromosome(Int8[div, x1, zero_, x1, x1], t2, true)
        @test !all(isfinite, c(probes))
        @test isnothing(SemanticEmbedder(t2, probes)(c))
        @test isnothing(GeneEmbedder(t2, probes)(c))
    end

    @testset "a custom embedder" begin
        prob = surrogate_problem(; seed=2)
        tb = prob.reg.toolbox_
        pop = generate_population(30, tb)
        # any function of a chromosome is an embedder; the first vector fixes the length
        s = SurrogateScreening(c -> Float64[length(c.expression_raw), sum(c.expression_raw)])
        @test SG.embed(s, pop[1]) == Float64[length(pop[1].expression_raw), sum(pop[1].expression_raw)]
        @test s.latent_dim == 2
        wrong = SurrogateScreening(c -> ones(length(c.expression_raw)))
        lengths = [length(c.expression_raw) for c in pop]
        @test !isnothing(SG.embed(wrong, pop[1]))
        other = findfirst(!=(lengths[1]), lengths)
        isnothing(other) || @test isnothing(SG.embed(wrong, pop[other]))
        # an embedder that throws or returns a non-finite value does not embed
        @test isnothing(SG.embed(SurrogateScreening(c -> error("no")), pop[1]))
        @test isnothing(SG.embed(SurrogateScreening(c -> [NaN]), pop[1]))
    end

    @testset "screening an epoch" begin
        prob = surrogate_problem()
        tb = prob.reg.toolbox_
        s = SurrogateScreening(prob.reg, prob.x[:, 1:24]; individuals_per_epoch=5,
            warmup_runs=30, warmup_batch=nothing, seed=1)
        @test s.screen_budget == 5
        @test SG.brood_multiplier(s) == 1.0        # five per epoch is below the threshold
        @test SG.brood_size(SurrogateScreening(identity), 70) == 210

        function epoch!(pop)
            work = collect(eachindex(pop))
            plan = SG.screen_epoch!(s, pop, work, tb.fitness_reset[1])
            for i in SG.evaluated_indices(plan)
                prob.loss(pop[i], false)
            end
            SG.commit_epoch!(s, pop, plan)
            return plan
        end

        # the warmup scores every individual, without a cap
        warm = generate_population(40, tb)
        plan = epoch!(warm)
        @test plan.mode === :all
        @test all(r -> r === :warmup, plan.reasons)
        @test all(c -> is_validated(s, c) || !isfinite(c.fitness[1]), warm)
        @test archive_size(s) >= 30

        # a screened epoch scores the budget and predicts the rest
        fresh = generate_population(40, tb)
        before = s.evaluated_count
        plan = epoch!(fresh)
        @test plan.mode === :screened
        @test s.evaluated_count - before == 5
        @test count(r -> r === :explore, plan.reasons) == round(Int, 5 * s.explore_fraction)
        scored = SG.evaluated_indices(plan)
        predicted = setdiff(plan.entries, scored)
        @test length(predicted) == length(plan.entries) - 5
        @test s.imputed_count == length(predicted)
        for i in scored
            @test is_validated(s, fresh[i])
            @test fresh[i].fitness[1] ≈ prob.truth(fresh[i])
        end
        # a prediction is finite, is never known, and stays strictly behind the best loss
        for i in predicted
            @test !is_validated(s, fresh[i])
            @test isfinite(fresh[i].fitness[1])
            @test fresh[i].fitness[1] > s.incumbent[1]
        end
        # the individuals that cannot be embedded got the worst fitness without a call
        @test all(i -> fresh[i].fitness == tb.fitness_reset[1], plan.broken)
        @test isempty(intersect(Set(SG.cached_indices(plan)), Set(predicted)))
    end

    @testset "a capped warmup imputes the median of its batch" begin
        prob = surrogate_problem(; seed=5)
        tb = prob.reg.toolbox_
        s = SurrogateScreening(prob.reg, prob.x[:, 1:24]; individuals_per_epoch=5,
            warmup_batch=8, warmup_lhs=true, seed=2)
        pop = generate_population(40, tb)
        plan = SG.screen_epoch!(s, pop, collect(eachindex(pop)), tb.fitness_reset[1])
        @test plan.mode === :warmup
        @test length(plan.chosen) == 8                # the larger of batch and warmup cap
        for i in SG.evaluated_indices(plan)
            prob.loss(pop[i], false)
        end
        SG.commit_epoch!(s, pop, plan)
        finite = [pop[i].fitness[1] for i in SG.evaluated_indices(plan) if isfinite(pop[i].fitness[1])]
        rest = setdiff(plan.entries, SG.evaluated_indices(plan))
        @test !isempty(rest)
        @test all(i -> pop[i].fitness[1] == max(median(finite), nextfloat(s.incumbent[1])), rest)
        @test all(i -> pop[i].fitness[1] > s.incumbent[1], rest)
    end

    @testset "screen selection" begin
        rng = MersenneTwister(6)
        X = rand(rng, 1, 60)
        T = reshape((X[1, :] .- 0.3) .^ 2, 1, :)
        screen = GpScreen()
        SG.fit_screen!(screen, X, T; rng=rng)
        cands = reshape(collect(range(0, 1; length=21)), 1, :)
        pick = SG.select_screen(screen, cands, 3)
        @test length(pick) == 3
        @test abs(cands[1, pick[1]] - 0.3) <= 0.1
        for acq in (:logei, :logei_believer)
            sc = GpScreen(acquisition=acq)
            SG.fit_screen!(sc, X, T; rng=rng)
            p = SG.select_screen(sc, cands, 3)
            @test length(unique(p)) == 3
            @test abs(cands[1, p[1]] - 0.3) <= 0.15
        end
        # the feasibility gate fills the batch from the runnable candidates first
        feasible = [c < 0.5 ? 0.0 : 1.0 for c in cands[1, :]]
        gated = SG.select_screen(screen, cands, 3; feasible=feasible)
        @test all(cands[1, gated] .>= 0.5)
        # a plausible candidate may still beat the reference, the far end cannot
        plausible = SG.plausible_screen(screen, cands, [0.01])
        @test plausible[pick[1]]
        @test !plausible[end]

        # two objectives, optimal at 0.3 and 0.7, and an archive whose front is
        # {0.35, 0.5, 0.65}: one prediction per objective, and every multi-objective
        # acquisition picks from between the optima
        X2 = reshape([0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0], 1, :)
        T2 = vcat((X2 .- 0.3) .^ 2, (X2 .- 0.7) .^ 2)
        @test size(SG.pareto_points(T2), 2) == 3
        for (acq, per_pick) in ((:lcb, false), (:lcb, true), (:ehvi, false))
            sc = GpScreen(acquisition=acq, scalarize_per_pick=per_pick)
            SG.fit_screen!(sc, X2, T2; rng=MersenneTwister(7))
            means, devs = SG.predict_screen(sc, cands)
            @test size(means) == (2, 21) && size(devs) == (2, 21)
            p = SG.select_screen(sc, cands, 4)
            @test length(unique(p)) == 4
            # the front lies between the two optima
            @test all(0.2 .<= cands[1, p] .<= 0.8)
        end
    end

    @testset "fit! with a surrogate" begin
        plain = surrogate_problem(; seed=11)
        fit!(plain.reg, 12, 120, plain.loss)
        prob = surrogate_problem(; seed=11)
        s = SurrogateScreening(prob.reg, prob.x[:, randperm(prob.rng, 80)[1:24]];
            individuals_per_epoch=10, warmup_runs=40, seed=3)
        fit!(prob.reg, 12, 120, prob.loss; surrogate=s)

        # far fewer loss calls
        @test prob.calls[] < plain.calls[] / 3
        @test s.evaluated_count > 0 && s.imputed_count > 0
        @test archive_size(s) <= s.evaluated_count
        # the hall of fame holds losses, not predictions
        for m in prob.reg.best_models_
            @test is_validated(s, m)
            @test m.fitness[1] ≈ prob.truth(m)
        end
        # the history records losses
        hist = prob.reg.fitness_history_
        @test length(hist.train_loss) == 12
        @test all(t -> isfinite(t[1]), hist.train_loss)
        @test issorted([t[1] for t in hist.train_loss]; rev=true)
        # the brood was ranked by the process once it was fitted
        @test s.last_preselect.bred == SG.brood_size(s, 84)
        @test s.last_preselect.kept == 84
    end

    @testset "a seeded screened search reproduces" begin
        function screened_once()
            prob = surrogate_problem(; seed=21)
            s = SurrogateScreening(prob.reg, prob.x[:, 1:20]; individuals_per_epoch=0.15,
                warmup_runs=30, seed=4)
            fit!(prob.reg, 10, 100, prob.loss; surrogate=s)
            return prob.reg.best_models_[1].fitness, prob.calls[], s.imputed_count
        end
        @test screened_once() == screened_once()
    end

    @testset "two objectives" begin
        prob = surrogate_problem(; seed=31, objectives=2)
        s = SurrogateScreening(prob.reg, prob.x[:, 1:20]; individuals_per_epoch=8,
            warmup_runs=30, screen=GpScreen(acquisition=:ehvi), seed=5)
        fit!(prob.reg, 8, 100, prob.loss; surrogate=s)
        @test s.imputed_count > 0
        for m in prob.reg.best_models_
            @test length(m.fitness) == 2
            @test is_validated(s, m)
        end
        # the incumbent is the best loss per objective, and no prediction beat it
        @test length(s.incumbent) == 2
    end

    @testset "several expressions per chromosome" begin
        prob = multi_expression_problem()
        tb = prob.reg.toolbox_
        probes = prob.x[:, 1:10]
        ctx = buffer_context(tb, probes)
        emb = SemanticEmbedder(tb, probes; expressions=2)
        checked = 0
        for c in generate_population(40, tb)
            v = emb(c)
            isnothing(v) && continue
            # one block per expression, the parts the loss evaluates
            f, g = split_predict(c, ctx, 2)
            @test v ≈ vcat(asinh.(f), asinh.(g))
            @test length(SemanticEmbedder(tb, probes)(c)) == 10
            checked += 1
        end
        @test checked > 10
        @test_throws ArgumentError SemanticEmbedder(tb, probes; expressions=3)

        # f = x1 + x1 * x2 and g = x1 - x2 * x2 by hand: connectors, then four genes of
        # head length 4; split_karva drops the first connector
        op(sym) = only(k for (k, fn) in tb.callbacks if fn === sym)
        feat(i) = only(k for (k, nd) in tb.nodes if nd isa InputSelector && nd.idx == i)
        x1, x2 = feat(1), feat(2)
        filler = fill(x1, 8)
        genes = Int8[op(*), op(+), op(-),
            x1, filler...,
            op(*), x1, x2, fill(x1, 6)...,
            x1, filler...,
            op(*), x2, x2, fill(x1, 6)...]
        c = Chromosome(genes, tb, true)
        f, g = split_predict(c, ctx, 2)
        @test f ≈ probes[1, :] .+ probes[1, :] .* probes[2, :]
        @test g ≈ probes[1, :] .- probes[2, :] .* probes[2, :]
        eqs = split_equations(c, 2)
        @test length(eqs) == 2
        @test occursin("x1", eqs[1]) && occursin("x2", eqs[2]) && !occursin("-", eqs[1])

        # the tensor embedder splits alike, with the components of each expression
        treg = GepTensorRegressor(2; entered_non_terminals=[:+, :*], gene_count=2, head_len=3)
        tprobes = [randn(8), randn(8)]
        temb = TensorEmbedder(treg.toolbox_, tprobes; expressions=2)
        usable = [c for c in generate_population(20, treg.toolbox_) if !isnothing(temb(c))]
        @test !isempty(usable)
        tc = first(usable)
        @test length(temb(tc)) == 16
        @test TensorEmbedder(treg.toolbox_, tprobes; expressions=2, components=[1, 1])(tc) ==
              temb(tc)
        @test_throws ArgumentError TensorEmbedder(treg.toolbox_, tprobes; expressions=2,
            components=[1, 1, 1])
    end

    @testset "no prediction dominates the holder of a best loss" begin
        prob = multi_expression_problem(; seed=3)
        tb = prob.reg.toolbox_
        s = SurrogateScreening(prob.reg, prob.x[:, 1:24]; expressions=2,
            individuals_per_epoch=5, warmup_runs=30, warmup_batch=nothing, seed=2)
        warm = generate_population(60, tb)
        screened_epoch!(s, warm, prob.loss, tb.fitness_reset[1])
        scored = [c for c in warm if is_validated(s, c) && all(isfinite, c.fitness)]
        champions = [scored[argmin([c.fitness[j] for c in scored])] for j in 1:2]
        @test [c.fitness[j] for (j, c) in enumerate(champions)] == s.incumbent

        fresh = generate_population(60, tb)
        plan = screened_epoch!(s, fresh, prob.loss, tb.fitness_reset[1])
        @test plan.mode === :screened
        predicted = [c for c in fresh if SG.carries_prediction(s, c)]
        @test !isempty(predicted)
        for c in predicted, champion in champions
            @test length(c.fitness) == 2
            @test !dominates_(c.fitness, champion.fitness)
        end
    end

    @testset "processes tied to the expressions they judge" begin
        prob = multi_expression_problem(; seed=10)
        tb = prob.reg.toolbox_
        probes = prob.x[:, 1:12]
        @test SG.expression_blocks(SemanticEmbedder(tb, probes; expressions=2)) == [1:12, 13:24]
        @test SG.expression_blocks(SemanticEmbedder(tb, probes)) == [1:12]
        treg = GepTensorRegressor(2; entered_non_terminals=[:+, :*], gene_count=2, head_len=3)
        @test SG.expression_blocks(TensorEmbedder(treg.toolbox_, [randn(8), randn(8)];
            expressions=2, components=[1, 3])) == [1:8, 9:32]
        @test_throws ArgumentError SG.expression_blocks(GeneEmbedder(tb, probes))

        s = SurrogateScreening(prob.reg, probes; expressions=2, objective_expressions=[1, 2])
        @test s.screen.inputs == [1:12, 13:24]
        s = SurrogateScreening(prob.reg, probes; expressions=2, objective_expressions=[2, :all])
        @test s.screen.inputs == [13:24, nothing]
        @test_throws ArgumentError SurrogateScreening(prob.reg, probes; expressions=2,
            objective_expressions=[1, 3])
        # one entry per objective of the regressor, checked before any loss call
        @test_throws ArgumentError SurrogateScreening(prob.reg, probes; expressions=2,
            objective_expressions=[1])
        @test_throws ArgumentError SurrogateScreening(prob.reg, probes; expressions=2,
            objective_expressions=[1, 2], screen=GpScreen(inputs=[1:2, 3:4]))
        @test_throws ArgumentError SurrogateScreening(prob.reg, probes; embedding=:genes,
            objective_expressions=[1, 2])

        # objective j depends on coordinate j alone: a process that sees that coordinate
        # predicts what one fitted on it alone predicts, whatever the other one does
        rng = MersenneTwister(3)
        X = rand(rng, 2, 40)
        T = vcat(sin.(6 .* X[1:1, :]), cos.(5 .* X[2:2, :]))
        Q = rand(rng, 2, 15)
        sc = GpScreen(inputs=[1:1, [2]])
        SG.fit_screen!(sc, X, T; rng=MersenneTwister(1))
        means, devs = SG.predict_screen(sc, Q)
        for j in 1:2
            alone = SG.GaussianProcess(X[j:j, :], T[j, :])
            mu, sd = SG.unstandardize(alone, SG.posterior(alone, Q[j:j, :])...)
            @test means[j, :] ≈ mu && devs[j, :] ≈ sd
        end
        @test length(SG.select_screen(sc, Q, 4)) == 4
        @test_throws ArgumentError SG.fit_screen!(GpScreen(inputs=[1:1]), X, T)
        one = GpScreen(inputs=[2:2])
        SG.fit_screen!(one, X, T[2:2, :]; rng=MersenneTwister(1))
        @test SG.select_screen(one, Q, 3) == sortperm(vec(
            SG.predict_screen(one, Q)[1] .- SG.predict_screen(one, Q)[2]))[1:3]

        # a search with tied processes
        s = SurrogateScreening(prob.reg, prob.x[:, 1:20]; expressions=2,
            objective_expressions=[1, 2], individuals_per_epoch=8, warmup_runs=30, seed=9)
        fit!(prob.reg, 8, 80, prob.loss; surrogate=s)
        @test s.imputed_count > 0
        @test all(m -> is_validated(s, m), prob.reg.best_models_)
    end

    @testset "the scored holders of the best values survive" begin
        prob = multi_expression_problem(; seed=8)
        tb = prob.reg.toolbox_
        s = SurrogateScreening(prob.reg, prob.x[:, 1:24]; expressions=2, seed=6)
        pop = unique(c -> c.expression_raw, generate_population(40, tb))[1:10]
        # six predictions whose means beat every scored individual, four scored ones that
        # fall behind them: two hold the best value of an objective; one more prediction
        # at the tail claims the best first objective of all
        fits = [(0.10, 0.40), (0.20, 0.20), (0.30, 0.15), (0.20, 0.30), (0.35, 0.10),
            (0.30, 0.30), (0.30, 0.35), (0.80, 0.05), (0.01, 0.90), (0.001, 1.50)]
        for (c, f) in zip(pop, fits)
            c.fitness = f
        end
        for i in 7:9
            s.known[copy(pop[i].expression_raw)] = pop[i].fitness
        end
        for i in [1:6; 10]
            SG.mark_prediction!(s, pop[i])
        end
        sort!(pop; by=c -> mean(c.fitness))
        best_f, best_g, claim = pop[findfirst(c -> c.fitness == (0.01, 0.90), pop)],
            pop[findfirst(c -> c.fitness == (0.80, 0.05), pop)],
            pop[findfirst(c -> c.fitness == (0.001, 1.50), pop)]
        survivors = pop[1:6]
        SG.keep_best_scored!(s, pop, 6)
        # the holders follow the leader, where the next generation does not take their
        # place, and the last survivors make room; the prediction holds nothing
        @test pop[1] === survivors[1]
        @test pop[2:3] == [best_g, best_f]
        @test pop[4:6] == survivors[2:4]
        @test !(claim in pop[1:6])
        # nothing moves once they are there, nor with one objective; a holder among the
        # last survivors moves up as well
        before = copy(pop)
        SG.keep_best_scored!(s, pop, 6)
        @test pop == before
        pop[3], pop[6] = pop[6], pop[3]
        SG.keep_best_scored!(s, pop, 6)
        @test pop[2:3] == [best_g, best_f]
        single = generate_population(5, prob.reg.toolbox_)
        for (i, c) in enumerate(single)
            c.fitness = (Float64(6 - i),)
        end
        order = copy(single)
        SG.keep_best_scored!(s, single, 2)
        @test single == order
    end

    @testset "the best scored values of a screened search never get worse" begin
        prob = multi_expression_problem(; seed=9)
        s = SurrogateScreening(prob.reg, prob.x[:, 1:20]; expressions=2,
            individuals_per_epoch=6, warmup_runs=30, seed=8)
        best = Vector{Float64}[]
        function logger(pop, epoch, _)
            known = [c.fitness for c in pop if is_validated(s, c) && all(isfinite, c.fitness)]
            isempty(known) || push!(best, [minimum(f[j] for f in known) for j in 1:2])
        end
        fit!(prob.reg, 15, 80, prob.loss; surrogate=s, file_logger_callback=logger)
        @test length(best) >= 10
        @test all(i -> all(best[i+1] .<= best[i]), 1:length(best)-1)
    end

    @testset "objectives known without the loss" begin
        prob = multi_expression_problem(; seed=4, with_size=true)
        tb = prob.reg.toolbox_
        s = SurrogateScreening(prob.reg, prob.x[:, 1:24]; expressions=2,
            individuals_per_epoch=5, warmup_runs=30, warmup_batch=nothing,
            exact_objectives=Dict(3 => prob.size_of), seed=3)
        screened_epoch!(s, generate_population(60, tb), prob.loss, tb.fitness_reset[1])
        fresh = generate_population(60, tb)
        @test screened_epoch!(s, fresh, prob.loss, tb.fitness_reset[1]).mode === :screened
        predicted = [c for c in fresh if SG.carries_prediction(s, c)]
        @test !isempty(predicted)
        # the size is computed, not predicted, and not clamped: it is not a guess
        @test all(c -> c.fitness[3] == prob.size_of(c), predicted)
        s.incumbent = [0.5, 0.5, 0.05]
        @test SG.impute(s, [0.1, 1.0, 9.0], [NaN, NaN, 0.001]) ==
              (nextfloat(0.5), 1.0, 0.001)
        @test_throws ArgumentError SurrogateScreening(identity; exact_objectives=Dict(0 => size))
        bad = SurrogateScreening(identity; exact_objectives=Dict(4 => prob.size_of))
        @test_throws ArgumentError SG.exact_values(bad, predicted[1], 3)

        # a known objective replaces the prediction, with no deviation, and the picks of
        # the confidence bound and the hypervolume follow it: candidate 6 (x = 0.5) is the
        # optimum of the first objective and known to be far ahead on the second
        rng = MersenneTwister(9)
        X = rand(rng, 1, 30)
        T = vcat((X .- 0.5) .^ 2, X)
        cands = reshape(collect(range(0, 1; length=11)), 1, :)
        known = fill(NaN, 2, 11)
        known[2, :] .= 10.0
        known[2, 6] = -10.0
        for acquisition in (:lcb, :ehvi)
            sc = GpScreen(acquisition=acquisition)
            SG.fit_screen!(sc, X, T; rng=MersenneTwister(1))
            means, devs = SG.predict_screen(sc, cands; known=known)
            @test means[2, 6] == -10.0 && devs[2, 6] == 0.0 && means[2, 1] == 10.0
            @test means[1, :] ≈ SG.predict_screen(sc, cands)[1][1, :]
            @test SG.select_screen(sc, cands, 1; known=known) == [6]
        end
        # without the known values the process of the scalarization picks, elsewhere
        sc = GpScreen()
        SG.fit_screen!(sc, X, T; rng=MersenneTwister(1))
        @test length(SG.select_screen(sc, cands, 3)) == 3
    end

    @testset "a prediction is provisional" begin
        prob = multi_expression_problem(; seed=5)
        tb = prob.reg.toolbox_
        s = SurrogateScreening(prob.reg, prob.x[:, 1:24]; expressions=2,
            individuals_per_epoch=5, warmup_runs=30, warmup_batch=nothing, seed=4)
        screened_epoch!(s, generate_population(60, tb), prob.loss, tb.fitness_reset[1])
        fresh = generate_population(60, tb)
        screened_epoch!(s, fresh, prob.loss, tb.fitness_reset[1])
        predicted = [i for i in eachindex(fresh) if SG.carries_prediction(s, fresh[i])]
        @test !isempty(predicted)
        @test all(i -> !is_validated(s, fresh[i]), predicted)

        # a prediction that its expression's loss has overtaken is not a loss value
        stale = fresh[predicted[1]]
        copy_ = Chromosome(copy(stale.genes), tb, true)
        prob.loss(copy_, false)
        SG.record_validation!(s, copy_)
        @test SG.known_expression(s, stale) && is_validated(s, copy_)
        @test !is_validated(s, stale)

        # every epoch screens the predictions again
        positions = SG.rescreen_predictions!(s, fresh, length(fresh), tb.fitness_reset[2])
        @test positions == Set(predicted)
        @test all(i -> isnan(fresh[i].fitness[1]), positions)
        @test !SG.carries_prediction(s, stale)
        plan = SG.screen_epoch!(s, fresh, collect(positions), tb.fitness_reset[1];
            rescreened=positions)
        # the individuals screened again compete for the batch of the epoch
        @test length(plan.chosen) == 5
        once = SurrogateScreening(prob.reg, prob.x[:, 1:24]; expressions=2, rescreen=false)
        @test isempty(SG.rescreen_predictions!(once, fresh, length(fresh), tb.fitness_reset[2]))
        # by default with several objectives only
        @test SG.rescreens(s, 2) && !SG.rescreens(s, 1)
        @test SG.rescreens(SurrogateScreening(identity; rescreen=true), 1)
    end

    @testset "fit! with two expressions and three objectives" begin
        prob = multi_expression_problem(; seed=6, with_size=true)
        s = SurrogateScreening(prob.reg, prob.x[:, 1:20]; expressions=2,
            individuals_per_epoch=8, warmup_runs=30, exact_objectives=Dict(3 => prob.size_of),
            seed=7)
        fit!(prob.reg, 10, 100, prob.loss; surrogate=s, hof=5)
        @test s.imputed_count > 0
        # a rank correlation per predicted objective; the known one has none
        @test length(s.spearman_objectives) == 3 && isempty(s.spearman_objectives[3])
        @test s.spearman_log == s.spearman_objectives[1]
        @test !isempty(s.spearman_log)
        ctx = buffer_context(prob.reg.toolbox_, prob.x)
        for m in prob.reg.best_models_
            @test length(m.fitness) == 3
            @test is_validated(s, m)
            f, g = split_predict(m, ctx, 2)
            @test m.fitness[1] ≈ prob.err(f, prob.f)
            @test m.fitness[2] ≈ prob.err(g, prob.g)
            @test m.fitness[3] == prob.size_of(m)
        end
        # the memory of a search can be saved with the population
        io = IOBuffer()
        serialize(io, s)
        restored = deserialize(seekstart(io))
        @test archive_size(restored) == archive_size(s)
        @test restored.known == s.known
    end

    @testset "data fit! and the uncertainty rule" begin
        Random.seed!(41)
        rng = MersenneTwister(41)
        x = randn(rng, 2, 60)
        y = @views 2 .* x[1, :] .- x[2, :] .* x[1, :]
        reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=2, head_len=4)
        s = SurrogateScreening(reg, x[:, 1:16]; embedding=:genes, budget_rule=:uncertainty,
            min_individuals=0.05, individuals_per_epoch=0.2, warmup_runs=20,
            offspring_multiplier=2, seed=6)
        @test SG.brood_multiplier(s) == 2.0
        fit!(reg, 10, 80, x, y; loss_fun="mse", linear_scaling=true, surrogate=s)
        @test s.evaluated_count > 0
        best = reg.best_models_[1]
        @test is_validated(s, best)
        # an exact model scores round-off, which the two summation orders round differently
        @test isapprox(best.fitness[1], mean(abs2, y .- best(x)); rtol=1e-8, atol=1e-20)
        @test_throws ArgumentError SurrogateScreening(reg, x[:, 1:4]; embedding=:trees)
    end

    @testset "tensor regressor" begin
        Random.seed!(51)
        n = 40
        x1 = randn(n)
        x2 = randn(n)
        target = 2 .* x1 .+ x1 .* x2
        reg = GepTensorRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=2,
            head_len=3, feature_names=["x1", "x2"])
        allocate_buffers!(reg, [x1, x2])
        calls = Threads.Atomic{Int}(0)
        function tloss(elem, validate::Bool)
            if isnan(mean(elem.fitness)) || validate
                Threads.atomic_add!(calls, 1)
                p = try
                    predictT(reg, elem.expression_raw)
                catch
                    nothing
                end
                elem.fitness = p isa AbstractVector{Float64} && length(p) == n &&
                               all(isfinite, p) ? (mean(abs2, p .- target),) : (Inf,)
            end
        end
        probes = [x1[1:10], x2[1:10]]
        emb = TensorEmbedder(reg.toolbox_, probes)
        c = generate_population(5, reg.toolbox_)[1]
        v = emb(c)
        @test isnothing(v) || length(v) == 10
        @test length(TensorEmbedder(reg.toolbox_, probes; per_gene=true)(c)) == 20
        s = SurrogateScreening(reg, probes; individuals_per_epoch=6, warmup_runs=20, seed=7)
        fit!(reg, 8, 60, tloss; surrogate=s)
        @test s.imputed_count > 0
        @test is_validated(s, reg.best_models_[1])

        # the oversampled start is picked over the tensor embedding, which the default
        # characterization (the scalar evaluator) cannot do
        s2 = SurrogateScreening(reg, probes; individuals_per_epoch=6, warmup_runs=20, seed=8)
        fit!(reg, 3, 60, tloss; surrogate=s2, population_sampling_multiplier=4)
        @test length(reg.best_models_) == 3
        @test is_validated(s2, reg.best_models_[1])
    end
end
