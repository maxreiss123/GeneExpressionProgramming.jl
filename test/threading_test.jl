#=
Per-thread buffers are indexed by `Threads.threadid()`, which can exceed `nthreads()`:
since Julia 1.12 a session, a plain `julia` launch included, has an interactive thread with
the low id beside its worker pool, so `Threads.@threads` hands out ids up to
`nthreads() + 1`. These tests size the buffers and evaluate from inside the same kind of
loop the search uses.
=#

using Test
using Random
using Statistics

@testset "per-thread buffers" begin
    ids = zeros(Int, 8 * Threads.nthreads())
    Threads.@threads :static for i in eachindex(ids)
        ids[i] = Threads.threadid()
    end
    @test thread_slots() >= Threads.nthreads()
    @test maximum(ids) <= thread_slots()

    Random.seed!(3)
    x = randn(2, 40)
    reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*, :/], gene_count=3, head_len=5)
    ctx = build_buffers(reg, x)
    @test length(ctx.pools) == length(ctx.fast) == length(ctx.stacks) ==
          length(ctx.gene_fast) == thread_slots()

    pop = generate_population(96, reg.toolbox_)
    preds = Vector{Any}(undef, length(pop))
    bases = Vector{Any}(undef, length(pop))
    Threads.@threads :static for i in eachindex(pop)
        p = GeneExpressionProgramming.GepRegression.buffered_predict(pop[i], ctx)
        preds[i] = p isa AbstractVector ? copy(p) : p
        bases[i] = GeneExpressionProgramming.GepRegression.gene_basis(pop[i], ctx, size(x, 2))
    end
    @test all(p -> p isa AbstractVector && length(p) == size(x, 2), preds)
    @test all(G -> G isa Matrix && size(G) == (size(x, 2), 3), bases)
    # the pooled evaluation agrees with the allocating one
    @test all(i -> isequal(preds[i], pop[i](x)), eachindex(pop))

    ctxs = thread_contexts(reg.toolbox_, x)
    @test length(ctxs) == thread_slots()
    out = Vector{Any}(undef, length(pop))
    Threads.@threads :static for i in eachindex(pop)
        p = pop[i](ctxs[Threads.threadid()])
        out[i] = p isa AbstractVector ? copy(p) : p
    end
    @test all(i -> isequal(out[i], preds[i]), eachindex(pop))
end

@testset "search evaluates every candidate" begin
    Random.seed!(5)
    x = randn(2, 60)
    y = x[1, :] .+ x[2, :]
    reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*, :/], gene_count=2, head_len=4)
    fit!(reg, 40, 200, x, y; loss_fun="mse")
    # with every candidate scored, whichever thread evaluates it, this target is found
    @test reg.best_models_[1].fitness[1] < 1e-10
    @test maximum(abs.(reg(x) .- y)) < 1e-6
end

#=
The fitness loop hands out individuals one at a time to whichever thread is free
(`foreach_balanced`). `Threads.@threads` over the individuals would give each thread one
fixed chunk instead: with uneven costs (a loss that runs a solver, say) the threads with
cheap chunks idle while the rest work through theirs one individual at a time.
=#
const foreach_balanced = GeneExpressionProgramming.GepRegression.foreach_balanced

@testset "balanced loop" begin
    n = 40 * Threads.nthreads() + 3
    seen = zeros(Int, n)
    foreach_balanced(k -> (seen[k] += 1), collect(1:n))
    @test all(==(1), seen)
    foreach_balanced(k -> error("not called"), Int[])

    # a call keeps its thread id across a yield, and no two calls in progress share one,
    # so buffers indexed by `threadid()` are safe even for a loss that waits
    busy = [Threads.Atomic{Bool}(false) for _ in 1:thread_slots()]
    clashes = Threads.Atomic{Int}(0)
    moved = Threads.Atomic{Int}(0)
    foreach_balanced(collect(1:8 * Threads.nthreads())) do _
        tid = Threads.threadid()
        Threads.atomic_cas!(busy[tid], false, true) && Threads.atomic_add!(clashes, 1)
        sleep(0.002)
        Threads.threadid() == tid || Threads.atomic_add!(moved, 1)
        busy[tid][] = false
    end
    @test clashes[] == 0
    @test moved[] == 0

    @test_throws Exception foreach_balanced(k -> k == 3 && error("stop"), collect(1:10))

    if Threads.nthreads() > 1
        # the first call waits until every other item is done: with one fixed chunk per
        # thread, the rest of its own chunk could not start and the wait would time out
        n = 16 * Threads.nthreads()
        done = Threads.Atomic{Int}(0)
        first = Threads.Atomic{Bool}(true)
        timed_out = Threads.Atomic{Bool}(false)
        foreach_balanced(collect(1:n)) do _
            if Threads.atomic_xchg!(first, false)
                t0 = time()
                while done[] < n - 1
                    time() - t0 > 60 && (timed_out[] = true; break)
                    sleep(0.001)
                end
            end
            Threads.atomic_add!(done, 1)
        end
        @test !timed_out[]
        @test done[] == n
    end
end

@testset "an expensive individual holds up only its own thread" begin
    if Threads.nthreads() > 1
        Random.seed!(9)
        x = randn(2, 30)
        y = x[1, :] .* x[2, :]
        reg = GepRegressor(2; entered_non_terminals=[:+, :-, :*], gene_count=2, head_len=3)
        ctxs = thread_contexts(reg.toolbox_, x)
        slow = Threads.Atomic{Bool}(false)
        slow_end = Ref(Inf)
        ends = zeros(10_000)
        n_done = Threads.Atomic{Int}(0)
        function loss(elem, validate)
            (isnan(mean(elem.fitness)) || validate) || return
            # the first individual of the timed run takes far longer than the others
            first = !validate && Threads.atomic_xchg!(slow, false)
            first && sleep(2.0)
            p = elem(ctxs[Threads.threadid()])
            elem.fitness = p isa AbstractVector ? (mean(abs2, p .- y),) : (Inf,)
            if first
                slow_end[] = time()
            elseif !validate
                ends[Threads.atomic_add!(n_done, 1)+1] = time()
            end
            return
        end
        fit!(reg, 1, 200, loss)             # compiles the loss and the loop
        n_done[] = 0
        slow[] = true
        fit!(reg, 1, 200, loss)
        @test n_done[] > 100
        # the other threads scored every other individual while the slow one ran
        @test count(>(slow_end[]), ends[1:n_done[]]) == 0
    end
end

@testset "copies are scored once per string and epoch" begin
    # a cache of three entries evicts almost every string before its copies come up, and
    # the loss leaves half of the strings unscored (NaN): copies of an evicted string are
    # scored again once, copies of a string left NaN are not scored again
    Random.seed!(21)
    x = randn(2, 20)
    y = x[1, :] .- x[2, :]
    reg = GepRegressor(2; entered_non_terminals=[:+, :-], gene_count=1, head_len=2)
    ctxs = thread_contexts(reg.toolbox_, x)
    epoch = Ref(1)
    calls_epoch = zeros(Int, 100_000)
    calls_key = zeros(UInt, 100_000)
    n_calls = Threads.Atomic{Int}(0)
    left_nan(raw) = isodd(hash(raw) >> 7)
    function loss(elem, validate)
        (isnan(mean(elem.fitness)) || validate) || return
        if !validate
            k = Threads.atomic_add!(n_calls, 1) + 1
            calls_epoch[k] = epoch[]
            calls_key[k] = hash(elem.expression_raw)
        end
        left_nan(elem.expression_raw) && return
        p = elem(ctxs[Threads.threadid()])
        elem.fitness = p isa AbstractVector ? (mean(abs2, p .- y),) : (Inf,)
        return
    end
    unscored_copies = Ref(0)
    wrong = Ref(0)
    logger = function (pop, e, _)
        for c in pop
            if left_nan(c.expression_raw)
                isnan(mean(c.fitness)) && (unscored_copies[] += 1)
            else
                isnan(mean(c.fitness)) && (wrong[] += 1)
            end
        end
        epoch[] += 1
    end
    strategy = GenericRegressionStrategy(nothing, 1, loss)
    runGep(12, 300, reg.toolbox_, strategy; cache_size=3, file_logger_callback=logger,
        population_sampling_multiplier=1)
    pairs = [(calls_epoch[k], calls_key[k]) for k in 1:n_calls[]]
    @test length(unique(pairs)) == length(pairs)
    @test wrong[] == 0
    @test unscored_copies[] > 0
end
