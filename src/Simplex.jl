"""
    GepSimplex

Nelder-Mead for the constants of a model whose loss is expensive, e.g. a CFD simulation in
the loop: plain, as `Optim.NelderMead()` does it, or screened by a Gaussian process over
the constants, which decides where the loss runs, optionally after a screened particle
swarm in a box, for a loss with several minima ([`ScreenedNelderMead`](@ref)).
[`optimize_constants!`](@ref) tunes the constants of a chromosome with it against the
custom loss of a search; [`simplex_search`](@ref) minimizes any function of a vector.
"""
module GepSimplex

using LinearAlgebra
using Random
using Statistics

using ..GepEntities: Chromosome, constant_positions
using ..GepSurrogate: GaussianProcess, posterior, unstandardize, get_transform

export ScreenedNelderMead, SimplexSearch, simplex_search, optimize_constants!

"""
    ScreenedNelderMead(; max_evaluations=60, screen=true, kappa=1.0,
        target_transform=:log10, fit=true, polish_radius=1/32, tolerance=1e-8, seed=0,
        swarm_box=nothing, particles=10)

How [`simplex_search`](@ref) and [`optimize_constants!`](@ref) minimize: Nelder-Mead,
screened by a Gaussian process or not, within `max_evaluations` calls of the loss. Both
start as `Optim.NelderMead()` does: the given constants and one vertex per constant, moved
by half its value plus 0.025 (by 0.025 where that move vanishes, at -0.05).

With `screen = false` it is Nelder-Mead on the loss, with the steps of `Optim.NelderMead()`
(the adaptive parameters of Gao and Han, its stopping rule).

With `screen = true` every further loss call goes where a Gaussian process over the
constants expects the minimum: the process ([`GaussianProcess`](@ref), `fit` choosing its
length scale and noise) is fitted on the scored constants near the best ones, a failed call
counting as the worst value scored, and the loss scores the minimum of the optimistic bound
`mean - kappa * deviation` inside a trust region around the best constants, found by
Nelder-Mead on the process. The region, at first as wide as the initial steps, doubles
after two improvements in a row and halves after `max(2, ceil(n / 2))` failures in a row,
and the bound turns greedy as it shrinks; below `polish_radius` initial steps, Nelder-Mead
finishes on the loss from scored constants near the best one.

With `swarm_box`, a particle swarm explores a box first: `(lower, upper)`, or a number `r`
of initial steps around the start (the form for `fit!`, whose models differ in their
constants). The start and a Latin hypercube of the box, `particles` points, are scored;
then every iteration moves every particle (the constriction coefficients of Clerc and
Kennedy, the global best) and the loss scores the new position the process ranks best, one
call per iteration; a particle's best changes on scored values only. After `2n + 4` calls
without a gain of 0.1 %, or at half the budget, the trust region search continues from the
best point on every point scored; it is not bound to the box.

The screening saves loss calls where Nelder-Mead needs many steps: on calibrations of 2 to
4 coefficients of the Lotka-Volterra equations it reached 1 % of the initial error with 37
to 64 % of the calls of Nelder-Mead (`benchmark/screened_constants.jl`). Where Nelder-Mead
is fast anyway or the minimum lies in a narrow valley, `screen = false` converges further
within the same calls. The swarm pays off for a loss with several minima (an oscillator: 50
calls instead of 94 to 98, the global minimum from every start) and costs 1.2 to 3.4 times
the calls on a smooth one (`benchmark/swarm_variants.jl`). The best constants are always ones
the loss scored, and a solve that diverges may give a large error or `Inf`: the process
sees the logarithm of the loss, so it only marks the region as bad.

- `max_evaluations`: loss calls, the initial simplex included
- `screen`: place the loss calls by the process, or run plain Nelder-Mead
- `kappa`: weight of the deviation in the optimistic bound
- `target_transform`: transform of the loss values before the process sees them (see
  `GepSurrogate.TRANSFORMS`); `:asinh` or `:none` for a loss that can be negative
- `fit`: choose the length scale and noise of the process by the marginal likelihood,
  else take the median heuristic and a fixed nugget
- `polish_radius`: the width of the region, in initial steps, below which Nelder-Mead
  finishes on the loss
- `tolerance`: Nelder-Mead stops when the standard deviation of the values at the vertices
  (times `sqrt(n / (n + 1))`, as in Optim) falls below it
- `seed`: seed of the random starts of the search on the process, and of the swarm
- `swarm_box`: `nothing` for no swarm, `(lower, upper)`, or a number of initial steps
  around the start; it needs `screen = true`
- `particles`: the size of the swarm, at least 3
"""
struct ScreenedNelderMead
    max_evaluations::Int
    screen::Bool
    kappa::Float64
    target_transform::Symbol
    fit::Bool
    polish_radius::Float64
    tolerance::Float64
    seed::Int
    swarm_box::Union{Nothing,Float64,Tuple{Vector{Float64},Vector{Float64}}}
    particles::Int
end

function ScreenedNelderMead(; max_evaluations::Integer=60, screen::Bool=true, kappa::Real=1.0,
    target_transform::Union{Symbol,AbstractString}=:log10, fit::Bool=true,
    polish_radius::Real=1 / 32, tolerance::Real=1e-8, seed::Integer=0, swarm_box=nothing,
    particles::Integer=10)
    max_evaluations >= 1 || throw(ArgumentError("max_evaluations is at least one loss call"))
    kappa >= 0 || throw(ArgumentError("kappa weighs a deviation, it is not negative"))
    0 < polish_radius <= 1 || throw(ArgumentError(
        "polish_radius is a share of the initial steps, in (0, 1]"))
    get_transform(target_transform)
    box = check_swarm_box(swarm_box)
    isnothing(box) || screen || throw(ArgumentError(
        "the swarm is screened by the process: swarm_box needs screen = true"))
    particles >= 3 || throw(ArgumentError("a swarm needs at least 3 particles"))
    return ScreenedNelderMead(Int(max_evaluations), screen, Float64(kappa),
        Symbol(target_transform), fit, Float64(polish_radius), Float64(tolerance), Int(seed),
        box, Int(particles))
end

# nothing, a positive number of initial steps around the start, or (lower, upper)
check_swarm_box(::Nothing) = nothing
function check_swarm_box(r::Real)
    isfinite(r) && r > 0 || throw(ArgumentError(
        "a relative swarm_box is a positive number of initial steps"))
    return Float64(r)
end
function check_swarm_box(box::Tuple{AbstractVector{<:Real},AbstractVector{<:Real}})
    lower, upper = Vector{Float64}(box[1]), Vector{Float64}(box[2])
    length(lower) == length(upper) || throw(DimensionMismatch(
        "the bounds of swarm_box differ in length"))
    all(isfinite, lower) && all(isfinite, upper) && all(lower .< upper) || throw(ArgumentError(
        "swarm_box = (lower, upper) needs finite bounds with lower < upper"))
    return (lower, upper)
end
check_swarm_box(box) = throw(ArgumentError(
    "swarm_box is nothing, a number of initial steps or a tuple (lower, upper), not $(typeof(box))"))

"""
    SimplexSearch

The result of [`simplex_search`](@ref) and [`optimize_constants!`](@ref): the best point
the loss scored (`minimizer`, `minimum`), the loss calls (`evaluations`), how many of them
the Gaussian process placed or picked from the swarm (`proposals`), the best value after
every call (`history`), and every point scored with its value (`points`, `values`).
"""
struct SimplexSearch
    minimizer::Vector{Float64}
    minimum::Float64
    evaluations::Int
    proposals::Int
    history::Vector{Float64}
    points::Vector{Vector{Float64}}
    values::Vector{Float64}
end

function Base.show(io::IO, r::SimplexSearch)
    print(io, "SimplexSearch(minimum ", r.minimum, " after ", r.evaluations, " loss calls, ",
        r.proposals, " of them placed by the process)")
end

# ---------------------------------------------------------------------------------------
#  Bookkeeping of the loss calls
# ---------------------------------------------------------------------------------------

mutable struct Evaluations
    f::Any
    budget::Int
    X::Vector{Vector{Float64}}
    Y::Vector{Float64}
    best::Vector{Float64}
    proposals::Int
end

Evaluations(f, budget::Int) = Evaluations(f, budget, Vector{Float64}[], Float64[], Float64[], 0)

remaining(e::Evaluations) = e.budget - length(e.Y)

function evaluate!(e::Evaluations, x::AbstractVector)
    y = Float64(e.f(x))
    # NaN would break every comparison of the simplex: a failed call is the worst value
    isfinite(y) || (y = Inf)
    push!(e.X, Vector{Float64}(x))
    push!(e.Y, y)
    push!(e.best, min(isempty(e.best) ? Inf : e.best[end], y))
    return y
end

# ---------------------------------------------------------------------------------------
#  Nelder-Mead, as Optim.NelderMead() does it
# ---------------------------------------------------------------------------------------

# the initial simplex of Optim's AffineSimplexer(a = 0.025, b = 0.5): vertex j + 1 moves
# constant j to 1.5 x + 0.025, computed as Optim does; a move that vanishes (x = -0.05) is
# 0.025 instead, since it would leave the simplex flat
function initial_simplex(x0::AbstractVector)
    S = [Vector{Float64}(x0) for _ in 1:length(x0)+1]
    for j in eachindex(x0)
        x = Float64(x0[j])
        moved = 1.5 * x + 0.025
        abs(moved - x) < 1e-9 * (1 + abs(x)) && (moved = x + 0.025)
        S[j+1][j] = moved
    end
    return S
end

# the simplex with the vertices x and x + steps[j] e_j
function axis_simplex(x::AbstractVector, steps::AbstractVector)
    S = [Vector{Float64}(x) for _ in 1:length(x)+1]
    for j in eachindex(x)
        S[j+1][j] += steps[j]
    end
    return S
end

# the adaptive parameters of Gao and Han: reflection, expansion, contraction, shrink
nm_parameters(n::Int) = (1.0, 1.0 + 2 / n, 0.75 - 1 / 2n, 1.0 - 1 / n)

centroid(S, h::Int) = sum(S[i] for i in eachindex(S) if i != h) ./ (length(S) - 1)

# Optim's stopping value: the deviation of the values at the vertices
converged(F, n::Int, tolerance::Float64) = sqrt(var(F) * (n / (n + 1))) <= tolerance

"""
    nelder_mead!(e, S, F, tolerance; iterations=10_000)

Nelder-Mead from the simplex `S` with the values `F`, until the values converge, the
iterations or the budget of `e` run out. Every point it moves a vertex to is scored first.
"""
function nelder_mead!(e::Evaluations, S::Vector{Vector{Float64}}, F::Vector{Float64},
    tolerance::Float64; iterations::Int=10_000)
    n = length(S[1])
    m = n + 1
    α, β, γ, δ = nm_parameters(n)
    it = 0
    while it < iterations && remaining(e) > 0 && !converged(F, n, tolerance)
        it += 1
        order = sortperm(F)
        l, sh, h = order[1], order[n], order[m]
        c = centroid(S, h)
        xr = c .+ α .* (c .- S[h])
        fr = evaluate!(e, xr)
        if fr < F[l]
            if remaining(e) == 0
                S[h], F[h] = xr, fr
                break
            end
            xe = c .+ β .* (xr .- c)
            fe = evaluate!(e, xe)
            if fe < fr
                S[h], F[h] = xe, fe
            else
                S[h], F[h] = xr, fr
            end
        elseif fr < F[sh]
            S[h], F[h] = xr, fr
        else
            remaining(e) == 0 && break
            if fr < F[h]
                # outside contraction
                xc = c .+ γ .* (xr .- c)
                fc = evaluate!(e, xc)
                shrink = !(fc < fr)
            else
                # inside contraction
                xc = c .- γ .* (xr .- c)
                fc = evaluate!(e, xc)
                shrink = !(fc < F[h])
            end
            if shrink
                for i in order[2:end]
                    remaining(e) == 0 && break
                    S[i] = S[l] .+ δ .* (S[i] .- S[l])
                    F[i] = evaluate!(e, S[i])
                end
            else
                S[h], F[h] = xc, fc
            end
        end
    end
    return S, F
end

# as Optim does after its iterations: the centroid of the best vertices, scored
function finish!(e::Evaluations, S, F)
    remaining(e) > 0 || return
    evaluate!(e, centroid(S, argmax(F)))
end

function plain_search!(e::Evaluations, x0::Vector{Float64}, method::ScreenedNelderMead)
    S = initial_simplex(x0)
    F = Float64[]
    for x in S
        remaining(e) > 0 || return
        push!(F, evaluate!(e, x))
    end
    S, F = nelder_mead!(e, S, F, method.tolerance)
    finish!(e, S, F)
end

# ---------------------------------------------------------------------------------------
#  The screened search
# ---------------------------------------------------------------------------------------

# the minimum of g over the box center +- radius, by Nelder-Mead on g from every start;
# leaving the box costs in proportion to the distance
function box_minimum(g, center::Vector{Float64}, radius::Float64,
    starts::Vector{Vector{Float64}})
    n = length(center)
    lo, hi = center .- radius, center .+ radius
    boxed(u) = g(clamp.(u, lo, hi)) + 1e3 * sum(max.(abs.(u .- center) .- radius, 0.0))
    best, best_value = copy(center), Inf
    for s in starts
        inner = Evaluations(boxed, 60 * n)
        S = axis_simplex(s, fill(radius / 2, n))
        F = [evaluate!(inner, x) for x in S]
        nelder_mead!(inner, S, F, 1e-10)
        u = clamp.(inner.X[argmin(inner.Y)], lo, hi)
        v = g(u)
        v < best_value && ((best, best_value) = (u, v))
    end
    return best
end

# n + 1 scored points near the best one that span a simplex, picked greedily for volume
# (in units of the initial steps), or nothing if they are too flat
function archive_simplex(e::Evaluations, scale::Vector{Float64}, b::Int)
    n = length(scale)
    z(i) = (e.X[i] .- e.X[b]) ./ scale
    order = sortperm([norm(z(i)) for i in eachindex(e.X)])
    pool = [i for i in order[1:min(end, 8n + 8)] if i != b && isfinite(e.Y[i])]
    chosen = Int[]
    for _ in 1:n
        best_i, best_v = 0, 0.0
        for i in pool
            i in chosen && continue
            E = reduce(hcat, [z(j) for j in vcat(chosen, i)])
            v = sqrt(max(det(E' * E), 0.0))
            v > best_v && ((best_i, best_v) = (i, v))
        end
        best_i == 0 && return nothing
        push!(chosen, best_i)
    end
    E = reduce(hcat, [z(j) for j in chosen])
    edge = maximum(norm.(eachcol(E)))
    sqrt(max(det(E' * E), 0.0)) < 1e-6 * edge^n && return nothing
    return vcat(b, chosen)
end

function screened_search!(e::Evaluations, x0::Vector{Float64}, method::ScreenedNelderMead)
    for x in initial_simplex(x0)
        remaining(e) > 0 || return
        evaluate!(e, x)
    end
    trust_region_search!(e, x0, method)
end

# the initial steps of the simplex at x: the units of the search
initial_steps(x::AbstractVector) = (S = initial_simplex(x); [abs(S[j+1][j] - x[j]) for j in eachindex(x)])

# the trust region search on the process and the polish by Nelder-Mead, on the points the
# evaluations hold, in units of the initial steps at x0
function trust_region_search!(e::Evaluations, x0::Vector{Float64}, method::ScreenedNelderMead)
    n = length(x0)
    scale = initial_steps(x0)
    forward, _ = get_transform(method.target_transform)
    rng = MersenneTwister(method.seed)
    radius, max_radius = 1.0, 8.0
    shrink_after = max(2, cld(n, 2))
    successes = failures = 0
    while remaining(e) > 0 && radius >= method.polish_radius
        finite = findall(isfinite, e.Y)
        length(finite) >= 2 || break
        b = finite[argmin(e.Y[finite])]
        # a failed call counts as the worst value scored, so that the process learns where
        # the loss fails rather than proposing there again
        worst = maximum(e.Y[finite])
        values = [isfinite(y) ? y : worst for y in e.Y]
        Z = [x ./ scale for x in e.X]
        c = Z[b]
        # the process sees the scored constants near the region, at least 2n + 2 of them
        distance = [maximum(abs.(z .- c)) for z in Z]
        near = findall(<=(2radius), distance)
        if length(near) < 2n + 2
            near = sortperm(distance)[1:min(2n + 2, length(Z))]
        end
        gp = GaussianProcess(reduce(hcat, Z[near]), forward.(values[near]); fit=method.fit)
        k = method.kappa * min(1.0, radius)
        function bound(u)
            mu, sd = unstandardize(gp, posterior(gp, reshape(u, :, 1))...)
            return mu[1] - k * sd[1]
        end
        starts = [c]
        for i in near[sortperm(e.Y[near])][1:min(3, length(near))]
            all(abs.(Z[i] .- c) .<= radius) && push!(starts, Z[i])
        end
        for _ in 1:6
            push!(starts, c .+ radius .* (2 .* rand(rng, n) .- 1))
        end
        u = box_minimum(bound, c, radius, starts)
        # a proposal the loss has scored already brings nothing new: look closer
        if minimum(maximum(abs.(u .- z)) for z in Z) < 1e-9
            radius /= 2
            continue
        end
        y = evaluate!(e, u .* scale)
        e.proposals += 1
        if y < e.Y[b] - 1e-3 * abs(e.Y[b])
            successes += 1
            failures = 0
            successes >= 2 && ((radius, successes) = (min(2radius, max_radius), 0))
        else
            failures += 1
            successes = 0
            failures >= shrink_after && ((radius, failures) = (radius / 2, 0))
        end
    end
    remaining(e) > 0 || return
    # Nelder-Mead finishes on the loss, from scored constants near the best ones if they
    # span a simplex, else from new ones one step of the last region away
    finite = findall(isfinite, e.Y)
    b = isempty(finite) ? 1 : finite[argmin(e.Y[finite])]
    chosen = archive_simplex(e, scale, b)
    if isnothing(chosen)
        S = axis_simplex(e.X[b], max(radius, method.polish_radius) .* scale)
        F = [e.Y[b]]
        for x in S[2:end]
            remaining(e) > 0 || return
            push!(F, evaluate!(e, x))
        end
    else
        S = [copy(e.X[i]) for i in chosen]
        F = [e.Y[i] for i in chosen]
    end
    S, F = nelder_mead!(e, S, F, method.tolerance)
    finish!(e, S, F)
end

# ---------------------------------------------------------------------------------------
#  The screened swarm
# ---------------------------------------------------------------------------------------

# the constriction coefficient and the acceleration of Clerc and Kennedy
const CHI = 0.7298
const ACCELERATION = 2.05

# the bounds of the swarm: absolute, or r initial steps around the start
function swarm_bounds(box::Float64, x0::Vector{Float64})
    steps = initial_steps(x0)
    return x0 .- box .* steps, x0 .+ box .* steps
end
function swarm_bounds(box::Tuple{Vector{Float64},Vector{Float64}}, x0::Vector{Float64})
    lower, upper = box
    length(lower) == length(x0) || throw(DimensionMismatch(
        "swarm_box has $(length(lower)) bounds for $(length(x0)) constants"))
    all(lower .<= x0 .<= upper) || throw(ArgumentError("the start lies outside swarm_box"))
    return lower, upper
end

"""
    swarm_search!(e, x0, lower, upper, method)

A particle swarm in the box `lower .. upper`, screened by a Gaussian process: the start
and a Latin hypercube of the box are scored; then every iteration moves every particle
(constriction coefficients, the global best), the process over every point scored ranks
the new positions by the optimistic bound, and the loss scores the first one it has not
scored, so one loss call per iteration. A particle's best changes on scored values only.
It stops once `2n + 4` calls in a row brought no gain of 0.1 %, or at half the budget.
"""
function swarm_search!(e::Evaluations, x0::Vector{Float64}, lower::Vector{Float64},
    upper::Vector{Float64}, method::ScreenedNelderMead)
    n = length(x0)
    m = method.particles
    width = upper .- lower
    unit(x) = (x .- lower) ./ width
    forward, _ = get_transform(method.target_transform)
    rng = MersenneTwister(method.seed)
    X = [copy(x0)]
    strata = [randperm(rng, m - 1) for _ in 1:n]
    for i in 1:m-1
        push!(X, lower .+ width .* [(strata[j][i] - rand(rng)) / (m - 1) for j in 1:n])
    end
    V = [(lower .+ width .* rand(rng, n) .- X[i]) ./ 2 for i in 1:m]
    P = deepcopy(X)
    Pf = fill(Inf, m)
    function score!(i)
        y = evaluate!(e, X[i])
        y < Pf[i] && ((P[i], Pf[i]) = (copy(X[i]), y))
        return y
    end
    for i in 1:m
        remaining(e) > 0 || return
        score!(i)
    end
    best = minimum(e.Y)
    last_gain = length(e.Y)
    while remaining(e) > 0 && 2length(e.Y) < e.budget && length(e.Y) - last_gain < 2n + 4
        finite = findall(isfinite, e.Y)
        isempty(finite) && return
        g = P[argmin(Pf)]
        for i in 1:m
            r1, r2 = rand(rng, n), rand(rng, n)
            V[i] = CHI .* (V[i] .+ ACCELERATION .* r1 .* (P[i] .- X[i]) .+
                           ACCELERATION .* r2 .* (g .- X[i]))
            X[i] = X[i] .+ V[i]
            # a particle stops at the wall it hits
            outside = (X[i] .< lower) .| (X[i] .> upper)
            X[i] = clamp.(X[i], lower, upper)
            V[i][outside] .= 0.0
        end
        worst = maximum(e.Y[finite])
        values = [isfinite(y) ? y : worst for y in e.Y]
        gp = GaussianProcess(reduce(hcat, unit.(e.X)), forward.(values); fit=method.fit)
        mu, sd = unstandardize(gp, posterior(gp, reduce(hcat, unit.(X)))...)
        # the most promising position the loss has not scored yet
        scored = unit.(e.X)
        order = sortperm(mu .- method.kappa .* sd)
        k = findfirst(i -> all(z -> maximum(abs.(z .- unit(X[i]))) >= 1e-9, scored), order)
        isnothing(k) && return
        y = score!(order[k])
        e.proposals += 1
        (!isfinite(best) && isfinite(y) || y < best - 1e-3 * abs(best)) && (last_gain = length(e.Y))
        best = min(best, y)
    end
end

"""
    simplex_search(f, x0; method=ScreenedNelderMead()) -> SimplexSearch

Minimize `f(x)` (a number for a vector) from `x0` within `method.max_evaluations` calls of
`f`, by Nelder-Mead, screened by a Gaussian process or not, with a screened swarm in
`method.swarm_box` before it or not (see [`ScreenedNelderMead`](@ref)). A call that
returns a value that is not finite counts as the worst value. The result is the best point
`f` scored. A `swarm_box = (lower, upper)` must contain `x0`.
"""
function simplex_search(f, x0::AbstractVector{<:Real}; method::ScreenedNelderMead=ScreenedNelderMead())
    x = Vector{Float64}(x0)
    isempty(x) && throw(ArgumentError("there is nothing to optimize in an empty vector"))
    e = Evaluations(f, method.max_evaluations)
    if !method.screen
        plain_search!(e, x, method)
    elseif isnothing(method.swarm_box)
        screened_search!(e, x, method)
    else
        # the swarm explores the box, then the trust region search finishes from the best
        # point the swarm scored, on every point it scored
        swarm_search!(e, x, swarm_bounds(method.swarm_box, x)..., method)
        remaining(e) > 0 && trust_region_search!(e, copy(e.X[argmin(e.Y)]), method)
    end
    b = argmin(e.Y)
    return SimplexSearch(copy(e.X[b]), e.Y[b], length(e.Y), e.proposals, copy(e.best),
        e.X, e.Y)
end

# ---------------------------------------------------------------------------------------
#  The constants of a chromosome
# ---------------------------------------------------------------------------------------

scalarizer(::Nothing) = fitness -> mean(fitness)
scalarizer(j::Integer) = fitness -> fitness[j]
scalarizer(f::Function) = f

"""
    optimize_constants!(chromosome, loss; method=ScreenedNelderMead(), objective=nothing)
        -> SimplexSearch

Tune the constants of `chromosome` against `loss(chromosome, validate)`, the custom loss
of a search (a CFD simulation, say), with Nelder-Mead, screened by a Gaussian process or
not (see [`ScreenedNelderMead`](@ref)). Every occurrence of a constant in the expression
is one parameter (see `constant_positions`), as for the constant optimiser of `fit!` on
data; the search starts from the values the chromosome holds.

For every candidate, the constants are set as the chromosome's `optimised_constants` and
the loss is called with `validate = true`; it sees them as long as it evaluates the
chromosome by `chromosome(ctx)`, `split_predict`, `predictT(regressor, chromosome)`,
`equation_string` or `split_equations`. `objective` turns the fitness tuple into the value
minimized: `nothing` for the mean of its entries (the order of the population), an index,
or a function of the tuple. If the best constants beat the ones the chromosome held, they
stay in `optimised_constants` and the chromosome takes the fitness the loss gave them;
otherwise the chromosome is left as it was. Returns the [`SimplexSearch`](@ref), or
`nothing` for a chromosome without constants.
"""
function optimize_constants!(chromosome::Chromosome, loss::Function;
    method::ScreenedNelderMead=ScreenedNelderMead(), objective=nothing)
    positions = constant_positions(chromosome)
    isempty(positions) && return nothing
    tb = chromosome.toolbox
    held_constants = chromosome.optimised_constants
    held_fitness = chromosome.fitness
    x0 = isnothing(held_constants) ?
         Float64[Float64(tb.nodes[chromosome.expression_raw[p]]) for p in positions] :
         copy(held_constants)
    scalar = scalarizer(objective)
    fitness_of = Dict{Vector{Float64},Tuple}()
    function f(x)
        chromosome.optimised_constants = Vector{Float64}(x)
        chromosome.fitness = tb.fitness_reset[2]
        loss(chromosome, true)
        fitness_of[Vector{Float64}(x)] = chromosome.fitness
        return scalar(chromosome.fitness)
    end
    result = try
        simplex_search(f, x0; method=method)
    catch
        chromosome.optimised_constants = held_constants
        chromosome.fitness = held_fitness
        rethrow()
    end
    # the first call scored the constants the chromosome held
    if result.minimum < result.values[1]
        chromosome.optimised_constants = result.minimizer
        chromosome.fitness = fitness_of[result.minimizer]
    else
        chromosome.optimised_constants = held_constants
        chromosome.fitness = held_fitness
    end
    return result
end

end
