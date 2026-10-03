"""
    GepSurrogate

Surrogate screening of expensive losses, the Julia counterpart of `gep.surrogate` in the
Python package. A search whose loss is expensive (a solver in the loop, a simulation)
pays one loss call per new individual and epoch, and most of those calls are spent on
individuals that are visibly not worth one. With a [`SurrogateScreening`](@ref) passed to
`runGep` (or `fit!`) as `surrogate`:

1. every new individual of an epoch is embedded into a latent space: its expression is
   evaluated on a small, fixed probe set, and the transformed outputs are its latent
   vector ([`SemanticEmbedder`](@ref), [`GeneEmbedder`](@ref), [`TensorEmbedder`](@ref)),
   so individuals that behave alike lie close together however they are written;
2. a Gaussian process ([`GaussianProcess`](@ref)) maps the latent vectors of the
   individuals scored so far to their (transformed) losses, and an acquisition function
   ([`GpScreen`](@ref)) picks the few individuals the loss scores;
3. every other individual receives the prediction of the process instead, strictly worse
   than the best loss scored so far, and competes with it.

A predicted fitness is never cached: a copy of a predicted individual is screened again,
the best individual of an epoch is scored by the loss before it is recorded, and the hall
of fame `runGep` returns holds scored individuals only. Until the archive of scored
individuals is large enough to carry a process (the warmup), epochs are capped by a space
filling pick instead.

The screening is the zero-dependency variant of the Python module: LinearAlgebra,
Statistics and Distributions only. Its defaults are the ones the Python package measured
(see the docstrings there): a batch of ten individuals per epoch, an optimistic confidence
bound as the acquisition, a tenth of the batch spent on exploration, and a threefold brood
from which the process picks the offspring that enter the population.
"""
module GepSurrogate

using ..GepUtils
using ..TensorRegUtils
using ..GepEntities
using LinearAlgebra
using Statistics
using Random
using StatsBase: corspearman
using Distributions: Normal, cdf, pdf, logcdf

export SurrogateScreening, GpScreen, GaussianProcess, FeasibilityModel
export SemanticEmbedder, GeneEmbedder, TensorEmbedder
export posterior, unstandardize, expected_improvement, log_expected_improvement, believe
export fit_screen!, predict_screen, select_screen, plausible_screen
export pareto_points, hypervolume, hypervolume_improvement
export resolve_count, get_transform
export archive_size, is_validated, brood_multiplier, brood_size
export screen_epoch!, commit_epoch!, evaluated_indices, cached_indices
export record_validation!, rescore_known, preselect, characterize
export carries_prediction, mark_prediction!, known_expression, rescreen_predictions!
export keep_best_scored!, expression_blocks

# ----------------------------------------------------------------------------------------
#  Transforms and counts
# ----------------------------------------------------------------------------------------

# floor of the loss values before the logarithm, i.e. the resolution of the targets of the
# process
const FLOOR = 1e-30

log10_forward(x::Real) = log10(max(Float64(x), FLOOR))
# the exponent is clamped: a process extrapolating into a region it has never seen can
# predict an exponent beyond any float, and such an individual has to lose the
# competition, not overflow it
log10_inverse(x::Real) = 10.0^min(Float64(x), 300.0)
asinh_forward(x::Real) = asinh(Float64(x))
asinh_inverse(x::Real) = sinh(clamp(Float64(x), -700.0, 700.0))
identity_transform(x::Real) = Float64(x)

"""
    TRANSFORMS

The transforms bringing loss values (or probe outputs) into the space the process works
in, by name, as `(forward, inverse)` pairs of scalar functions: `:log10` (the default for
losses, which span decades; values are floored at `1e-30`), `:asinh` (compresses large
values and keeps their sign; the default for probe outputs) and `:none`.
"""
const TRANSFORMS = Dict{Symbol,Tuple{Function,Function}}(
    :log10 => (log10_forward, log10_inverse),
    :asinh => (asinh_forward, asinh_inverse),
    :none => (identity_transform, identity_transform),
)

"""
    get_transform(kind) -> (forward, inverse)

The transform pair of [`TRANSFORMS`](@ref) named `kind` (a `Symbol` or a string); throws
an `ArgumentError` for an unknown name.
"""
function get_transform(kind::Symbol)
    haskey(TRANSFORMS, kind) || throw(ArgumentError(
        "the transform $kind is unknown, use one of $(join(sort!(collect(keys(TRANSFORMS))), ", "))"))
    return TRANSFORMS[kind]
end
get_transform(kind::AbstractString) = get_transform(Symbol(kind))

"""
    resolve_count(value, total, default=nothing)

An absolute count from a count or a share: `value` is an absolute number (an integer), a
share of `total` (a float in `(0, 1)`), or `nothing`, which gives `default`. The result is
at least one. A share adapts the count to the individuals an epoch proposes, an absolute
number pins the number of loss calls.
"""
function resolve_count(value, total::Integer, default=nothing)
    isnothing(value) && return default
    value isa AbstractFloat && 0 < value < 1 && return max(1, round(Int, value * total))
    return max(1, floor(Int, value))
end

# a share stays a share, everything else becomes an absolute count
count_or_share(value::Integer) = max(1, Int(value))
count_or_share(value::Real) = 0 < value < 1 ? Float64(value) : max(1, floor(Int, value))

# ----------------------------------------------------------------------------------------
#  Distances and kernels, on matrices with one column per point
# ----------------------------------------------------------------------------------------

"""
    square_distances(A, B)

Squared euclidean distances between the columns of `A` (`d × m`) and of `B` (`d × n`), as
an `m × n` matrix, floored at zero.
"""
function square_distances(A::AbstractMatrix, B::AbstractMatrix)
    sa = vec(sum(abs2, A; dims=1))
    sb = vec(sum(abs2, B; dims=1))
    D = sa .+ sb' .- 2 .* (A' * B)
    return max.(D, 0.0)
end

"""
    median_distance(X)

Median of the positive pairwise distances between the first 256 columns of `X`, or 1 when
there are none: the length scale of the median heuristic.
"""
function median_distance(X::AbstractMatrix)
    sub = X[:, 1:min(size(X, 2), 256)]
    D = sqrt.(square_distances(sub, sub))
    positive = D[D.>0.0]
    return isempty(positive) ? 1.0 : median(positive)
end

rbf_kernel(A::AbstractMatrix, B::AbstractMatrix, lengthscale::Real) =
    exp.(-0.5 .* square_distances(A, B) ./ lengthscale^2)

"""
    robust_cholesky(K, jitter)

The Cholesky factor of `K + jitter * I`, with the jitter raised tenfold (up to 1e-2)
while the factorization fails, and the jitter that was used.
"""
function robust_cholesky(K::AbstractMatrix, jitter::Float64)
    j = jitter
    while true
        A = Matrix(K)
        @inbounds for i in axes(A, 1)
            A[i, i] += j
        end
        F = cholesky(Symmetric(A); check=false)
        issuccess(F) && return F.L, j
        j >= 1e-2 && throw(PosDefException(1))
        j = min(10 * j, 1e-2)
    end
end

# ----------------------------------------------------------------------------------------
#  Gaussian process
# ----------------------------------------------------------------------------------------

# the grid a fitted process searches: multiples of the median pairwise distance, and the
# noise variances between a nearly noise free loss and one whose values carry a tenth of
# the target variance
const FIT_LENGTH_FACTORS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
const FIT_NOISES = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)

"""
    GaussianProcess(X, y; nugget=1e-6, fit=false, length_factors=FIT_LENGTH_FACTORS,
        noises=FIT_NOISES, fit_subset=400)

Exact Gaussian process regression of `y` on the columns of `X` (`d × n`, one latent vector
per column). The process is kept small: the inputs standardized per dimension, the targets
to zero mean and unit variance, a radial basis kernel whose length scale is the median
pairwise distance of the data, and a fixed nugget. There is nothing to fit iteratively, so
the process is cheap enough to be rebuilt every epoch.

With `fit = true` the length scale (a multiple of the median heuristic from
`length_factors`) and the noise (from `noises`) are chosen by the marginal likelihood over
the grid, with the amplitude of the kernel solved in closed form at every grid point, on
the last `fit_subset` observations. The Python package measured it as clearly better
regression and as a tie as a screen (a screen ranks, and ranking by the posterior mean
suffers little from a wrong length scale), which is why it is off by default.

See [`posterior`](@ref), [`expected_improvement`](@ref),
[`log_expected_improvement`](@ref), [`believe`](@ref).
"""
struct GaussianProcess
    x_mean::Vector{Float64}
    x_scale::Vector{Float64}
    X::Matrix{Float64}
    y_mean::Float64
    y_scale::Float64
    y_std::Vector{Float64}
    lengthscale::Float64
    nugget::Float64
    amplitude::Float64
    # what the diagonal of the factorized kernel carries on top of the unit variance
    diagonal::Float64
    L::LowerTriangular{Float64,Matrix{Float64}}
    alpha::Vector{Float64}
    # the best (lowest) standardized observation, the reference of the improvement
    best::Float64
end

function GaussianProcess(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
    nugget::Real=1e-6, fit::Bool=false, length_factors=FIT_LENGTH_FACTORS,
    noises=FIT_NOISES, fit_subset::Integer=400)
    X = Matrix{Float64}(X)
    y = Vector{Float64}(y)
    size(X, 2) == length(y) || throw(DimensionMismatch(
        "the process needs one target per column of X, got $(size(X, 2)) columns and $(length(y)) targets"))
    isempty(y) && throw(ArgumentError("a Gaussian process needs at least one observation"))

    x_mean = vec(mean(X; dims=2))
    x_scale = vec(std(X; dims=2, corrected=false)) .+ 1e-12
    Xs = (X .- x_mean) ./ x_scale

    y_mean = mean(y)
    y_scale = std(y; corrected=false) + 1e-12
    y_std = (y .- y_mean) ./ y_scale

    lengthscale = median_distance(Xs)
    noise = Float64(nugget)
    # the amplitude of the kernel: one where the standardized targets are taken at face
    # value, the fitted variance of the process otherwise; it scales the deviations, not
    # the mean
    amplitude = 1.0
    if fit
        lengthscale, noise, amplitude = fit_hyperparameters(Xs, y_std, lengthscale, noise,
            length_factors, noises, fit_subset)
    end

    L, diagonal = robust_cholesky(rbf_kernel(Xs, Xs, lengthscale), noise + 1e-8)
    alpha = L' \ (L \ y_std)
    return GaussianProcess(x_mean, x_scale, Xs, y_mean, y_scale, y_std, lengthscale, noise,
        amplitude, diagonal, L, alpha, minimum(y_std))
end

"""
    fit_hyperparameters(Xs, y_std, lengthscale, nugget, length_factors, noises, subset)

The length scale, noise and amplitude that maximize the marginal likelihood over the grid
`length_factors × noises`. For a fixed kernel shape the likelihood is maximized by the
empirical variance of the whitened targets, which is one solve of a factorization the grid
point has computed anyway, so only a two dimensional grid is searched: small enough to be
searched exhaustively, and free of the local optima a gradient fit would have to survive
on an archive that changes every epoch.
"""
function fit_hyperparameters(Xs::Matrix{Float64}, y_std::Vector{Float64}, lengthscale::Float64,
    nugget::Float64, length_factors, noises, subset::Integer)
    take = min(size(Xs, 2), Int(subset))
    Xf = Xs[:, end-take+1:end]
    yf = y_std[end-take+1:end]
    # the distances are the same for every grid point, only the exponential is repeated
    S = square_distances(Xf, Xf)
    best = (-Inf, lengthscale, nugget, 1.0)
    for factor in length_factors
        ell = lengthscale * Float64(factor)
        K = exp.(-0.5 .* S ./ ell^2)
        for noise in noises
            F = cholesky(Symmetric(K + (Float64(noise) + 1e-8) * I); check=false)
            issuccess(F) || continue
            w = F.L \ yf
            # the profile likelihood: the amplitude that maximizes it, then the likelihood
            # at that amplitude
            amplitude = max(dot(w, w) / take, 1e-12)
            evidence = -0.5 * take * log(amplitude) - sum(log, diag(F.L))
            evidence > best[1] && (best = (evidence, ell, Float64(noise), amplitude))
        end
    end
    return best[2], best[3], best[4]
end

standardize_inputs(gp::GaussianProcess, Xq::AbstractMatrix) =
    (Matrix{Float64}(Xq) .- gp.x_mean) ./ gp.x_scale

"""
    posterior(gp::GaussianProcess, Xq) -> (mean, deviation)

Posterior mean and standard deviation at the columns of `Xq` (`d × m`), in the
standardized units of the targets; [`unstandardize`](@ref) turns them into the units of
the data.
"""
function posterior(gp::GaussianProcess, Xq::AbstractMatrix)
    Q = standardize_inputs(gp, Xq)
    cross = rbf_kernel(Q, gp.X, gp.lengthscale)
    mu = cross * gp.alpha
    V = gp.L \ Matrix(cross')
    variance = max.(1.0 .- vec(sum(abs2, V; dims=1)), 1e-12)
    # the amplitude cancels in the mean and scales the variance, which is why the
    # factorization never carries it
    return mu, sqrt.(gp.amplitude .* variance)
end

"""
    unstandardize(gp::GaussianProcess, mean, deviation) -> (mean, deviation)

A standardized prediction of [`posterior`](@ref) in the units of the data.
"""
unstandardize(gp::GaussianProcess, mu, sd) = (mu .* gp.y_scale .+ gp.y_mean, sd .* gp.y_scale)

const STD_NORMAL = Normal()

"""
    expected_improvement(gp::GaussianProcess, Xq)

Expected improvement below the best observation at the columns of `Xq`, in standardized
units. It underflows to an exact zero for every candidate the process considers clearly
worse than the best; [`log_expected_improvement`](@ref) keeps the ordering there.
"""
function expected_improvement(gp::GaussianProcess, Xq::AbstractMatrix)
    mu, sd = posterior(gp, Xq)
    gap = gp.best .- mu
    z = gap ./ sd
    return gap .* cdf.(STD_NORMAL, z) .+ sd .* pdf.(STD_NORMAL, z)
end

"""
    log_h(z)

`log(φ(z) + z Φ(z))`, the logarithm of the expected improvement at unit deviation, for any
`z`. For `z > 0` the two terms are added directly. Below, the cancellation is avoided
through the Mills ratio `r = Φ(z) / φ(z)`: `φ + z Φ = φ (1 + z r)`, with `1 + z r` taken
from the logarithms of `Φ` and `φ` down to `z = -10`, and from its asymptotic series
`1/z² - 3/z⁴ + 15/z⁶ - ...` below, where the logarithms lose the digits the difference
needs.
"""
function log_h(z::Float64)
    z > 0 && return log(pdf(STD_NORMAL, z) + z * cdf(STD_NORMAL, z))
    log_phi = -0.5 * z^2 - 0.5 * log(2π)
    if z > -10
        mills = exp(logcdf(STD_NORMAL, z) - log_phi)
        return log_phi + log(max(1.0 + z * mills, floatmin(Float64)))
    end
    t = 1 / z^2
    inner = t * (1 - t * (3 - t * (15 - t * (105 - t * 945))))
    return log_phi + log(max(inner, floatmin(Float64)))
end

"""
    log_expected_improvement(gp::GaussianProcess, Xq)

Logarithm of the expected improvement at the columns of `Xq`: the LogEI acquisition of
Ament et al. (2023). It ranks a candidate like the expected improvement wherever that is
representable, and keeps the ordering deep in the tail where the expected improvement
underflows to a tie.
"""
function log_expected_improvement(gp::GaussianProcess, Xq::AbstractMatrix)
    mu, sd = posterior(gp, Xq)
    z = (gp.best .- mu) ./ sd
    return log.(sd) .+ log_h.(z)
end

"""
    believe(gp::GaussianProcess, x) -> GaussianProcess

A copy of the process conditioned on its own prediction at the point `x`: the kriging
believer of greedy batch acquisition. The mean surface does not move, the deviation around
`x` collapses, and the next pick of a batch is therefore pushed away from the ones already
chosen. The best observation is not updated, a belief is not a loss call. The factor grows
by one row, so the update costs one solve instead of a refit.
"""
function believe(gp::GaussianProcess, x::AbstractVector)
    xs = (Vector{Float64}(x) .- gp.x_mean) ./ gp.x_scale
    cross = vec(rbf_kernel(gp.X, reshape(xs, :, 1), gp.lengthscale))
    solved = gp.L \ cross
    mu = dot(cross, gp.alpha)
    pivot = sqrt(max(1.0 + gp.diagonal - dot(solved, solved), 1e-12))
    n = length(gp.y_std)
    L = zeros(n + 1, n + 1)
    L[1:n, 1:n] .= gp.L
    L[n+1, 1:n] .= solved
    L[n+1, n+1] = pivot
    Lt = LowerTriangular(L)
    y_std = vcat(gp.y_std, mu)
    alpha = Lt' \ (Lt \ y_std)
    return GaussianProcess(gp.x_mean, gp.x_scale, hcat(gp.X, xs), gp.y_mean, gp.y_scale, y_std,
        gp.lengthscale, gp.nugget, gp.amplitude, gp.diagonal, Lt, alpha, gp.best)
end

# ----------------------------------------------------------------------------------------
#  Feasibility
# ----------------------------------------------------------------------------------------

"""
    FeasibilityModel(X, labels; prior=2.0, bandwidth=0.25)

Probability that the loss returns a finite value for an individual. An expensive loss does
not only cost, it also fails (a case diverges, a run hits its timeout), and such a call
costs as much as a useful one while carrying no value the regression could learn from.
The model is the classification counterpart of the process, kept as small: a kernel
weighted average of the outcomes `labels` (1 for a finite value, 0 for a failure) at the
columns of `X` (a Nadaraya-Watson estimator), pulled toward the overall success rate by
`prior` pseudo observations, so a region nobody has probed is neither condemned nor
trusted. The kernel is `bandwidth` times the median pairwise distance wide: the failures
of a loss are a property of a region, and a wide kernel smears that boundary. Call it on a
`d × m` matrix to get `m` probabilities.
"""
struct FeasibilityModel
    x_mean::Vector{Float64}
    x_scale::Vector{Float64}
    X::Matrix{Float64}
    labels::Vector{Float64}
    lengthscale::Float64
    prior::Float64
    base_rate::Float64
end

function FeasibilityModel(X::AbstractMatrix{<:Real}, labels::AbstractVector{<:Real};
    prior::Real=2.0, bandwidth::Real=0.25)
    X = Matrix{Float64}(X)
    x_mean = vec(mean(X; dims=2))
    x_scale = vec(std(X; dims=2, corrected=false)) .+ 1e-12
    Xs = (X .- x_mean) ./ x_scale
    labels = Vector{Float64}(labels)
    return FeasibilityModel(x_mean, x_scale, Xs, labels,
        Float64(bandwidth) * median_distance(Xs), Float64(prior), mean(labels))
end

function (m::FeasibilityModel)(Xq::AbstractMatrix)
    Q = (Matrix{Float64}(Xq) .- m.x_mean) ./ m.x_scale
    W = rbf_kernel(Q, m.X, m.lengthscale)
    return (W * m.labels .+ m.prior * m.base_rate) ./ (vec(sum(W; dims=2)) .+ m.prior)
end

# ----------------------------------------------------------------------------------------
#  Pareto fronts and hypervolumes, on matrices with one column per point (minimized)
# ----------------------------------------------------------------------------------------

"""
    pareto_points(P)

The non-dominated columns of `P` (`k × n`, one objective vector per column, minimized).
"""
function pareto_points(P::AbstractMatrix)
    n = size(P, 2)
    keep = trues(n)
    for i in 1:n
        keep[i] || continue
        for j in 1:n
            (j == i || !keep[j]) && continue
            # everything column i is at least as good in everywhere and strictly better in
            # somewhere is dominated by it
            weakly = true
            strictly = false
            for r in axes(P, 1)
                P[r, j] < P[r, i] && (weakly = false; break)
                P[r, j] > P[r, i] && (strictly = true)
            end
            weakly && strictly && (keep[j] = false)
        end
    end
    return P[:, keep]
end

"""
    hypervolume(P, reference)

The volume of the objective space between the columns of `P` (`k × n`, minimized) and the
`reference` corner that the points dominate. Two objectives are swept exactly; more are
sliced along the last objective, recursively.
"""
function hypervolume(P::AbstractMatrix, reference::AbstractVector)
    pts = [Vector{Float64}(P[:, j]) for j in axes(P, 2) if all(P[:, j] .< reference)]
    return hv_recursive(pts, Vector{Float64}(reference))
end

function hv_recursive(pts::Vector{Vector{Float64}}, ref::Vector{Float64})
    isempty(pts) && return 0.0
    k = length(ref)
    k == 1 && return ref[1] - minimum(p[1] for p in pts)
    if k == 2
        sorted = sort(pts; by=p -> (p[1], p[2]))
        volume = 0.0
        low = ref[2]
        for (i, p) in enumerate(sorted)
            low = min(low, p[2])
            right = i < length(sorted) ? sorted[i+1][1] : ref[1]
            volume += max(right - p[1], 0.0) * (ref[2] - low)
        end
        return volume
    end
    sorted = sort(pts; by=p -> p[k])
    volume = 0.0
    for i in eachindex(sorted)
        upper = i < length(sorted) ? sorted[i+1][k] : ref[k]
        height = upper - sorted[i][k]
        height > 0 || continue
        volume += height * hv_recursive([p[1:k-1] for p in sorted[1:i]], ref[1:k-1])
    end
    return volume
end

"""
    hypervolume_improvement(front, points, reference)

The volume each column of `points` (`k × m`) would add to the hypervolume of `front`
(`k × p`, non-dominated) with respect to `reference`; zero where a point is dominated by
the front or outside the reference box. Two objectives are computed by a sweep along the
first objective, more by the difference of two [`hypervolume`](@ref)s.
"""
function hypervolume_improvement(front::AbstractMatrix, points::AbstractMatrix,
    reference::AbstractVector)
    size(points, 1) == 2 || return hv_improvement_slow(front, points, reference)
    m = size(points, 2)
    volume = zeros(m)
    ref = Vector{Float64}(reference)
    if size(front, 2) == 0
        for j in 1:m
            p = view(points, :, j)
            all(p .< ref) && (volume[j] = prod(ref .- p))
        end
        return volume
    end
    order = sortperm(view(front, 1, :))
    xs = front[1, order]
    # the staircase the front dominates, read as the lowest second objective reached up to
    # every step of the first
    lows = accumulate(min, front[2, order])
    for j in 1:m
        px, py = points[1, j], points[2, j]
        (px < ref[1] && py < ref[2]) || continue
        slot = searchsortedlast(xs, px)
        # dominated by the front where a step to its left already reaches below it
        slot > 0 && lows[slot] <= py && continue
        # the sweep runs from the point to the reference corner, cut at every step of the
        # staircase right of it
        ceiling = slot > 0 ? lows[slot] : ref[2]
        start = px
        acc = 0.0
        for s in slot+1:length(xs)
            acc += max(xs[s] - start, 0.0) * max(min(ceiling, ref[2]) - py, 0.0)
            start = xs[s]
            ceiling = lows[s]
        end
        acc += max(ref[1] - start, 0.0) * max(min(ceiling, ref[2]) - py, 0.0)
        volume[j] = acc
    end
    return volume
end

function hv_improvement_slow(front::AbstractMatrix, points::AbstractMatrix,
    reference::AbstractVector)
    base = hypervolume(front, reference)
    return [all(points[:, j] .< reference) ?
            max(hypervolume(hcat(front, points[:, j]), reference) - base, 0.0) : 0.0
            for j in axes(points, 2)]
end

# ----------------------------------------------------------------------------------------
#  Embedders
# ----------------------------------------------------------------------------------------

"""
    expression_parts(chromosome, expressions)

The karva strings of the `expressions` expressions a chromosome carries, in order: its
whole karva string for one, `split_karva(chromosome, expressions)` for several (each part
takes `gene_count ÷ expressions` consecutive genes and their connectors, the first
`expressions - 1` connectors join nothing).
"""
expression_parts(chromosome::Chromosome, expressions::Int) =
    expressions == 1 ? [chromosome.expression_raw] : split_karva(chromosome, expressions)

function check_expressions(toolbox::Toolbox, expressions::Integer)
    expressions >= 1 || throw(ArgumentError("a chromosome carries at least one expression"))
    toolbox.gene_count % expressions == 0 || throw(ArgumentError(
        "$(toolbox.gene_count) genes do not split into $expressions expressions of equal " *
        "size; use a gene_count divisible by the number of expressions"))
    return Int(expressions)
end

"""
    SemanticEmbedder(toolbox, probes; transform=:asinh, expressions=1)

Embeds a chromosome as the behaviour of its expression: the karva string is evaluated on
the probe set and the transformed outputs are the latent vector, so expressions that
behave alike lie close together however they are written. `probes` holds one row per
feature and one column per probe sample, like the training data of `fit!`; a few dozen
samples drawn from the physically relevant range (e.g. a random subset of the training
data) are enough. `transform` (see [`TRANSFORMS`](@ref)) compresses large outputs; the
default keeps their sign.

A chromosome that carries several expressions, split by `split_karva(chromosome,
expressions)` in a multi-expression or template loss, is embedded with one block per
expression, in expression order: the whole karva string joins the parts with connectors
the loss never uses, so it is not what the loss sees.

Call it on a chromosome to get its latent vector, or `nothing` for an expression that
cannot be evaluated on every probe (a non-finite output), which the screening scores as a
crash without calling the loss. Constants tuned by an optimiser and gene coefficients are
not applied: an individual is embedded before it is scored. The evaluation needs the
batched evaluator (a scalar toolbox, e.g. of a `GepRegressor`); see
[`TensorEmbedder`](@ref) otherwise. One buffer context per thread slot is kept, so it may
be called from several threads.
"""
struct SemanticEmbedder
    contexts::Vector{Any}
    block_width::Int
    expressions::Int
    transform::Symbol
    forward::Function
end

function SemanticEmbedder(toolbox::Toolbox, probes::AbstractMatrix{<:Real};
    transform::Union{Symbol,AbstractString}=:asinh, expressions::Integer=1)
    size(probes, 2) > 0 || throw(ArgumentError("the probe set of an embedder is empty"))
    expressions = check_expressions(toolbox, expressions)
    contexts = thread_contexts(toolbox, Matrix{Float64}(probes))
    any(isnothing, contexts) && throw(ArgumentError(
        "an operator or terminal of this toolbox has no batched counterpart; embed its " *
        "chromosomes with a TensorEmbedder"))
    forward, _ = get_transform(transform)
    return SemanticEmbedder(contexts, size(probes, 2), expressions, Symbol(transform), forward)
end

"""
    GeneEmbedder(toolbox, probes; transform=:asinh, count=nothing)

As [`SemanticEmbedder`](@ref), with one latent block per gene, in gene order, for the
first `count` genes (all by default). It is the embedding for a search that reads the
genes of an individual one by one, e.g. with linear scaling, where the model is a
least-squares combination of the genes: two individuals whose genes behave alike then lie
close together whatever their connectors are. The order is kept, gene `k` is block `k`,
so the genes of a multi-expression chromosome stay grouped by expression.
"""
struct GeneEmbedder
    contexts::Vector{Any}
    block_width::Int
    count::Union{Int,Nothing}
    transform::Symbol
    forward::Function
end

function GeneEmbedder(toolbox::Toolbox, probes::AbstractMatrix{<:Real};
    transform::Union{Symbol,AbstractString}=:asinh, count::Union{Integer,Nothing}=nothing)
    inner = SemanticEmbedder(toolbox, probes; transform=transform)
    return GeneEmbedder(inner.contexts, inner.block_width,
        isnothing(count) ? nothing : Int(count), inner.transform, inner.forward)
end

# the transformed output of one evaluation, or nothing if it is not a finite vector over
# the probes; the output lives in the buffers of the context, so it is copied
function embed_block(out, width::Int, forward::Function)
    out isa AbstractVector{<:Real} || return nothing
    length(out) == width || return nothing
    block = Vector{Float64}(undef, width)
    @inbounds for i in 1:width
        v = forward(out[i])
        isfinite(v) || return nothing
        block[i] = v
    end
    return block
end

function (e::SemanticEmbedder)(chromosome::Chromosome)
    chromosome.compiled || return nothing
    ctx = e.contexts[Threads.threadid()]
    parts = try
        expression_parts(chromosome, e.expressions)
    catch
        return nothing
    end
    width = e.block_width
    vector = Vector{Float64}(undef, length(parts) * width)
    for (k, part) in enumerate(parts)
        out = try
            GepEntities.ctx_eval(part, ctx)
        catch
            return nothing
        end
        # each part is copied out of the buffers before the next one reuses them
        block = embed_block(out, width, e.forward)
        isnothing(block) && return nothing
        vector[(k-1)*width+1:k*width] .= block
    end
    return vector
end

function (e::GeneEmbedder)(chromosome::Chromosome)
    chromosome.compiled || return nothing
    ctx = e.contexts[Threads.threadid()]
    raw = try
        _karva_raw(chromosome; split=true)
    catch
        return nothing
    end
    genes = length(raw) - 1
    count = isnothing(e.count) ? genes : min(e.count, genes)
    vector = Vector{Float64}(undef, count * e.block_width)
    for g in 1:count
        out = try
            GepEntities.ctx_eval(raw[g+1], ctx)
        catch
            return nothing
        end
        block = embed_block(out, e.block_width, e.forward)
        isnothing(block) && return nothing
        vector[(g-1)*e.block_width+1:g*e.block_width] .= block
    end
    return vector
end

"""
    TensorEmbedder(toolbox, probes; components=1, transform=:asinh, per_gene=false,
        expressions=1)

The embedder for a tensor toolbox (`GepTensorRegressor`), whose expressions the batched
evaluator runs on columns of numbers and tensors. `probes` holds one column per feature,
in feature order, over the probe samples, like the data given to `allocate_buffers!`;
constant terminals become constant columns. The output is flattened sample by sample into
`components` numbers per sample (1 for a scalar, `dim` for a vector, `dim^2` for a
second-order tensor): an expression whose output has another shape is left unembedded,
i.e. scored as a crash.

With `expressions = k`, the chromosome is split into its `k` expressions
(`split_karva`), e.g. the coefficient functions of a template loss, and embedded with one
block per expression; `components` is then one number for all of them or one per
expression. With `per_gene = true`, one block per gene is embedded instead (a gene has the
components of its expression), and a gene whose output has another shape contributes a
block of zeros, as `predictT_scaled` leaves such a gene out. The expressions are evaluated
into fresh arrays, so it may be called from several threads.
"""
struct TensorEmbedder
    callbacks::Dict
    inputs::Dict{Int8,Any}
    samples::Int
    components::Vector{Int}
    expressions::Int
    genes_per_expression::Int
    per_gene::Bool
    transform::Symbol
    forward::Function
end

function TensorEmbedder(toolbox::Toolbox, probes::AbstractVector;
    components::Union{Integer,AbstractVector{<:Integer}}=1,
    transform::Union{Symbol,AbstractString}=:asinh, per_gene::Bool=false,
    expressions::Integer=1)
    isempty(probes) && throw(ArgumentError("the probe set of an embedder is empty"))
    samples = length(first(probes))
    samples > 0 || throw(ArgumentError("the probe set of an embedder is empty"))
    expressions = check_expressions(toolbox, expressions)
    comps = components isa Integer ? fill(Int(components), expressions) : Vector{Int}(components)
    length(comps) == expressions || throw(ArgumentError(
        "components holds one number per expression: $expressions expected, got $(length(comps))"))
    inputs = Dict{Int8,Any}(Int8(k) => col for (k, col) in enumerate(probes))
    # the terminals that are not features are constants, broadcast to the probe samples
    for (key, node) in toolbox.nodes
        haskey(inputs, key) || (inputs[key] = node * ones(samples))
    end
    forward, _ = get_transform(transform)
    return TensorEmbedder(toolbox.callbacks, inputs, samples, comps, expressions,
        toolbox.gene_count ÷ expressions, per_gene, Symbol(transform), forward)
end

# the components of a batch of numbers or tensors, sample by sample, or nothing when the
# batch does not have `components` per sample over `samples` samples
function flatten_batch(out, samples::Int, components::Int)
    out isa AbstractVector || return nothing
    length(out) == samples || return nothing
    flat = Vector{Float64}(undef, samples * components)
    @inbounds for (i, t) in enumerate(out)
        length(t) == components || return nothing
        for c in 1:components
            flat[(i-1)*components+c] = Float64(t[c])
        end
    end
    return flat
end

function tensor_block(e::TensorEmbedder, rek::AbstractVector{Int8}, components::Int)
    out = try
        calc_stack_batch_tensor(collect(rek), e.callbacks, e.inputs, nothing)
    catch
        return nothing
    end
    flat = flatten_batch(out, e.samples, components)
    isnothing(flat) && return nothing
    return embed_block(flat, e.samples * components, e.forward)
end

function (e::TensorEmbedder)(chromosome::Chromosome)
    chromosome.compiled || return nothing
    if !e.per_gene
        parts = try
            expression_parts(chromosome, e.expressions)
        catch
            return nothing
        end
        blocks = Vector{Float64}[]
        for (k, part) in enumerate(parts)
            block = tensor_block(e, part, e.components[k])
            isnothing(block) && return nothing
            push!(blocks, block)
        end
        return reduce(vcat, blocks)
    end
    raw = try
        _karva_raw(chromosome; split=true)
    catch
        return nothing
    end
    blocks = Vector{Float64}[]
    usable = false
    for g in 2:length(raw)
        components = e.components[cld(g - 1, e.genes_per_expression)]
        block = tensor_block(e, raw[g], components)
        usable |= !isnothing(block)
        push!(blocks, isnothing(block) ? zeros(e.samples * components) : block)
    end
    # a model none of whose genes can be evaluated is broken
    return usable ? reduce(vcat, blocks) : nothing
end

"""
    expression_blocks(embedder) -> Vector{UnitRange{Int}}

The positions of the latent block of each expression in the vectors of `embedder`, in
expression order: one block of the probe outputs per expression for a
[`SemanticEmbedder`](@ref), and for a [`TensorEmbedder`](@ref) (not `per_gene`) a block of
the components of each expression. Other embedders have no blocks by expression.
"""
function expression_blocks(e::SemanticEmbedder)
    w = e.block_width
    return [(j-1)*w+1:j*w for j in 1:e.expressions]
end

function expression_blocks(e::TensorEmbedder)
    e.per_gene && throw(ArgumentError(
        "a TensorEmbedder with per_gene embeds genes, not expressions, block by block"))
    ends = cumsum(e.samples .* e.components)
    return [(j == 1 ? 1 : ends[j-1] + 1):ends[j] for j in eachindex(ends)]
end

expression_blocks(e) = throw(ArgumentError(
    "the latent vectors of a $(typeof(e)) have no blocks by expression; give the inputs " *
    "of the GpScreen instead"))

# ----------------------------------------------------------------------------------------
#  The screen: ranks the pending individuals with Gaussian processes
# ----------------------------------------------------------------------------------------

"""
    ACQUISITIONS

The acquisitions [`GpScreen`](@ref) selects with: `:lcb` (the optimistic confidence bound
`mean - kappa * deviation`, the default), `:logei` (the top of the LogEI ranking),
`:logei_believer` (greedy batch LogEI with the kriging believer between the picks) and
`:ehvi` (the expected hypervolume improvement, for several objectives).
"""
const ACQUISITIONS = (:lcb, :logei, :logei_believer, :ehvi)

# samples of the Monte Carlo estimate of the expected hypervolume improvement, and the
# margin of its reference point as a share of the spread of the observed front
const EHVI_SAMPLES = 96
const EHVI_MARGIN = 0.1

"""
    GpScreen(; nugget=1e-6, rho=0.05, acquisition=:lcb, kappa=1.0,
        scalarize_per_pick=false, fit=false, inputs=nothing)

Ranks the pending individuals with Gaussian processes: one [`GaussianProcess`](@ref) per
objective for the predictions and, for the selection, the process of the objective (one
objective) or the process of a random augmented Chebyshev scalarization of the objectives
(several, with augmentation weight `rho`), drawn anew at every fit, which rotates its
attention over the front like ParEGO.

The default acquisition is the optimistic confidence bound `mean - kappa * deviation`, and
the choice is empirical (the Nguyen study of the Python package): the screening loop is
not pure Bayesian optimization, its job is to hand true losses to the would-be parents of
the next generation, and an exploitation-dominant acquisition serves that job best. The
expected improvement family fell behind random screening there: the classic form because
its improvement underflows to a tie for most members of an evolved population, the log
form because it resolves that tail toward large deviations, i.e. toward behavioural
outliers, which in a symbolic search are mostly broken models. See [`ACQUISITIONS`](@ref)
for the others.

- `scalarize_per_pick`: several objectives and `:lcb` only; draw a fresh scalarization
  for every pick of a batch rather than one per epoch (measured as a tie)
- `fit`: fit the length scale and noise of the processes by the marginal likelihood (see
  [`GaussianProcess`](@ref))
- `inputs`: the latent coordinates the process of each objective sees, one entry per
  objective, a range or vector of indices or `nothing` for all of them (the default for
  every objective). An objective that judges one expression of a chromosome that carries
  several learns from the block of that expression alone, undisturbed by the others: see
  `objective_expressions` of [`SurrogateScreening`](@ref), which sets it. With `inputs`,
  `:lcb` picks by the scalarization of the epoch over the bounds of the processes of the
  objectives rather than by the process of the scalarization, which sees every coordinate.

Objectives that are known without the loss (`exact_objectives` of a
[`SurrogateScreening`](@ref), e.g. the size of a model) reach the screen as `known`
values: they replace the prediction of their objective, with no deviation. With known
values, `:lcb` scalarizes the optimistic bounds of the objectives with the weights of the
epoch instead of asking the process of the scalarization, and `:ehvi` samples only the
unknown objectives; the LogEI acquisitions rank by the process of the scalarization, which
learned the known objectives from the archive.

A custom screen is any object with methods of [`fit_screen!`](@ref),
[`predict_screen`](@ref), [`select_screen`](@ref) and [`plausible_screen`](@ref).
"""
mutable struct GpScreen
    nugget::Float64
    fit::Bool
    rho::Float64
    acquisition::Symbol
    kappa::Float64
    scalarize_per_pick::Bool
    models::Vector{GaussianProcess}
    chooser::Union{GaussianProcess,Nothing}
    targets::Matrix{Float64}
    target_mean::Vector{Float64}
    target_scale::Vector{Float64}
    # the scalarization of the epoch, for several objectives
    weights::Vector{Float64}
    # the latent coordinates the process of each objective sees (nothing: all)
    inputs::Union{Nothing,Vector{Any}}
    rng::AbstractRNG

    function GpScreen(; nugget::Real=1e-6, rho::Real=0.05, acquisition::Symbol=:lcb,
        kappa::Real=1.0, scalarize_per_pick::Bool=false, fit::Bool=false,
        inputs::Union{Nothing,AbstractVector}=nothing)
        acquisition in ACQUISITIONS || throw(ArgumentError(
            "the acquisition $acquisition is unknown, use one of $(join(ACQUISITIONS, ", "))"))
        new(Float64(nugget), fit, Float64(rho), acquisition, Float64(kappa), scalarize_per_pick,
            GaussianProcess[], nothing, zeros(0, 0), Float64[], Float64[], Float64[],
            isnothing(inputs) ? nothing : Vector{Any}(inputs), MersenneTwister(0))
    end
end

# the rows of `X` the process of objective `j` sees
objective_input(screen::GpScreen, j::Int, X::AbstractMatrix) =
    (isnothing(screen.inputs) || isnothing(screen.inputs[j])) ? X : X[screen.inputs[j], :]

# the rows of `X` the process the selection asks sees: the objective's with one objective,
# all of them for the process of a scalarization
chooser_input(screen::GpScreen, X::AbstractMatrix) =
    length(screen.models) == 1 ? objective_input(screen, 1, X) : X

"""
    fit_screen!(screen, X, targets; rng)

Build the processes of `screen` from the archive: latent vectors `X` (`d × n`) and
transformed loss values `targets` (`k × n`, one row per objective). `rng` draws the
scalarization of several objectives.
"""
function fit_screen!(screen::GpScreen, X::AbstractMatrix, targets::AbstractMatrix;
    rng::AbstractRNG=screen.rng)
    k = size(targets, 1)
    isnothing(screen.inputs) || length(screen.inputs) == k || throw(ArgumentError(
        "the screen has inputs for $(length(screen.inputs)) objectives, the loss sets $k"))
    screen.models = [GaussianProcess(objective_input(screen, j, X), targets[j, :];
                         nugget=screen.nugget, fit=screen.fit) for j in 1:k]
    screen.target_mean = vec(mean(targets; dims=2))
    screen.target_scale = vec(std(targets; dims=2, corrected=false)) .+ 1e-12
    screen.targets = Matrix{Float64}(targets)
    screen.rng = rng
    if k == 1
        screen.chooser = screen.models[1]
        screen.weights = [1.0]
        return screen
    end
    weights = rand(rng, k) .+ 1e-12
    weights ./= sum(weights)
    screen.weights = weights
    weighted = (targets .- screen.target_mean) ./ screen.target_scale .* weights
    scalar = vec(maximum(weighted; dims=1)) .+ screen.rho .* vec(sum(weighted; dims=1))
    screen.chooser = GaussianProcess(X, scalar; nugget=screen.nugget)
    return screen
end

# whether a matrix of known values holds any
has_known(known) = !isnothing(known) && any(!isnan, known)

"""
    predict_screen(screen, X; known=nothing) -> (means, deviations)

The predictions of every objective at the columns of `X` (`d × m`), in the transformed
units of the targets, as two `k × m` matrices. `known` (`k × m`, transformed, `NaN` where
unknown) replaces the prediction wherever it holds a value, with no deviation.
"""
function predict_screen(screen::GpScreen, X::AbstractMatrix; known=nothing)
    k = length(screen.models)
    m = size(X, 2)
    means = Matrix{Float64}(undef, k, m)
    deviations = Matrix{Float64}(undef, k, m)
    for (j, model) in enumerate(screen.models)
        mu, sd = unstandardize(model, posterior(model, objective_input(screen, j, X))...)
        means[j, :] .= mu
        deviations[j, :] .= sd
    end
    if has_known(known)
        size(known) == (k, m) || throw(DimensionMismatch(
            "known holds $(size(known)) values for $k objectives of $m candidates"))
        @inbounds for i in eachindex(means)
            isnan(known[i]) && continue
            means[i] = known[i]
            deviations[i] = 0.0
        end
    end
    return means, deviations
end

"""
    plausible_screen(screen, X, reference; known=nothing) -> BitVector

Which columns of `X` could still beat `reference` (one transformed value per objective):
those whose optimistic bound `mean - kappa * deviation` lies below it on at least one
objective. Everything else the process claims to know well enough to skip. `known` as for
[`predict_screen`](@ref).
"""
function plausible_screen(screen::GpScreen, X::AbstractMatrix, reference::AbstractVector;
    known=nothing)
    means, deviations = predict_screen(screen, X; known=known)
    bounds = means .- screen.kappa .* deviations
    return vec(any(bounds .< reference; dims=1))
end

"""
    select_screen(screen, X, n; feasible=nothing, known=nothing) -> Vector{Int}

The indices of the `n` most promising columns of `X`, the most promising first, by the
acquisition of the screen. `feasible`, one probability per column that the loss returns a
finite value (a [`FeasibilityModel`](@ref)), gates the batch: it is filled from the
columns with a probability of at least one half before any other, and among those the
acquisition ranks as always. `known` (`k × m`, transformed, `NaN` where unknown) holds the
objectives known without the loss (see [`GpScreen`](@ref)).
"""
function select_screen(screen::GpScreen, X::AbstractMatrix, n::Integer; feasible=nothing,
    known=nothing)
    X = Matrix{Float64}(X)
    m = size(X, 2)
    n = min(Int(n), m)
    n <= 0 && return Int[]
    isnothing(feasible) || return select_feasible(screen, X, n, feasible; known=known)
    multi = length(screen.models) > 1
    multi && screen.acquisition === :ehvi && return select_ehvi(screen, X, n; known=known)
    if multi && screen.acquisition === :lcb
        screen.scalarize_per_pick && return select_scalarized(screen, X, n; known=known)
        # the scalarization of the epoch, over bounds some of which are known exactly or
        # come from processes that see a part of the latent vector each
        (has_known(known) || !isnothing(screen.inputs)) &&
            return select_weighted(screen, X, n, screen.weights; known=known)
    end
    X = chooser_input(screen, X)
    if screen.acquisition === :lcb
        mu, sd = posterior(screen.chooser, X)
        return sortperm(mu .- screen.kappa .* sd)[1:n]
    elseif screen.acquisition === :logei
        return sortperm(log_expected_improvement(screen.chooser, X); rev=true)[1:n]
    end
    # :logei_believer, and :ehvi of a single objective: greedy LogEI, the process
    # conditioned on its own prediction at every pick
    model = screen.chooser
    remaining = collect(1:m)
    chosen = Int[]
    while !isempty(remaining) && length(chosen) < n
        scores = log_expected_improvement(model, X[:, remaining])
        winner = remaining[argmax(scores)]
        push!(chosen, winner)
        filter!(!=(winner), remaining)
        (!isempty(remaining) && length(chosen) < n) && (model = believe(model, X[:, winner]))
    end
    return chosen
end

function select_feasible(screen, X::AbstractMatrix, n::Int, feasible; threshold::Real=0.5,
    known=nothing)
    feasible = Vector{Float64}(feasible)
    eligible = findall(>=(threshold), feasible)
    if length(eligible) >= n
        picks = select_screen(screen, X[:, eligible], n;
            known=has_known(known) ? known[:, eligible] : nothing)
        return eligible[picks]
    end
    # too few runnable individuals: the batch is completed by the ones the model considers
    # least hopeless
    chosen = copy(eligible)
    rest = setdiff(1:size(X, 2), eligible)
    order = sortperm(feasible[rest]; rev=true)
    append!(chosen, rest[order[1:min(length(order), n - length(chosen))]])
    return chosen[1:min(n, length(chosen))]
end

# the optimistic bounds of every objective in units of the archive, the scale the
# Chebyshev scalarizations weigh
function scaled_bounds(screen::GpScreen, X::AbstractMatrix; known=nothing)
    means, deviations = predict_screen(screen, X; known=known)
    return (means .- screen.kappa .* deviations .- screen.target_mean) ./ screen.target_scale
end

chebyshev(weighted::AbstractMatrix, rho::Float64) =
    vec(maximum(weighted; dims=1)) .+ rho .* vec(sum(weighted; dims=1))

"""
    select_weighted(screen, X, n, weights; known=nothing)

The `n` columns of `X` with the lowest augmented Chebyshev scalarization, by `weights`, of
their optimistic bounds: the pick of the epoch's scalarization where some objectives are
known exactly, which the process of the scalarization could only approximate.
"""
function select_weighted(screen::GpScreen, X::AbstractMatrix, n::Int, weights::AbstractVector;
    known=nothing)
    scalar = chebyshev(scaled_bounds(screen, X; known=known) .* weights, screen.rho)
    return sortperm(scalar)[1:n]
end

"""
    select_scalarized(screen, X, n; known=nothing)

A batch of several objectives with one fresh Chebyshev scalarization per pick, in the
space of the optimistic bounds, as ParEGO draws its weights per evaluation.
"""
function select_scalarized(screen::GpScreen, X::AbstractMatrix, n::Int; known=nothing)
    bounds = scaled_bounds(screen, X; known=known)
    k, m = size(bounds)
    remaining = collect(1:m)
    chosen = Int[]
    while !isempty(remaining) && length(chosen) < n
        weights = rand(screen.rng, k) .+ 1e-12
        weights ./= sum(weights)
        scalar = chebyshev(bounds[:, remaining] .* weights, screen.rho)
        winner = remaining[argmin(scalar)]
        push!(chosen, winner)
        filter!(!=(winner), remaining)
    end
    return chosen
end

"""
    reference_point(screen)

The corner the hypervolume is measured against: the worst observed value per objective
plus a tenth of the spread, so that an individual at the edge of the front still carries
a volume.
"""
function reference_point(screen::GpScreen)
    worst = vec(maximum(screen.targets; dims=2))
    spread = worst .- vec(minimum(screen.targets; dims=2))
    return worst .+ EHVI_MARGIN .* ifelse.(spread .> 0, spread, 1.0)
end

"""
    select_ehvi(screen, X, n; samples=EHVI_SAMPLES, known=nothing)

A batch by the expected hypervolume improvement over the front of the archive, estimated
by sampling the independent posteriors of the objectives (a known objective is not
sampled). The batch is filled greedily, and every pick joins the front at its predicted
mean (the kriging believer of the single objective case), which stops a batch from
crowding into one corner.
"""
function select_ehvi(screen::GpScreen, X::AbstractMatrix, n::Int; samples::Int=EHVI_SAMPLES,
    known=nothing)
    means, deviations = predict_screen(screen, X; known=known)
    k, m = size(means)
    front = pareto_points(screen.targets)
    reference = reference_point(screen)
    # the posterior samples of candidate j, one per column
    drawn = [means[:, j] .+ deviations[:, j] .* randn(screen.rng, k, samples) for j in 1:m]
    remaining = collect(1:m)
    chosen = Int[]
    while !isempty(remaining) && length(chosen) < n
        scores = [mean(hypervolume_improvement(front, drawn[j], reference)) for j in remaining]
        winner = remaining[argmax(scores)]
        push!(chosen, winner)
        filter!(!=(winner), remaining)
        front = pareto_points(hcat(front, means[:, winner]))
    end
    return chosen
end

# ----------------------------------------------------------------------------------------
#  The screening policy
# ----------------------------------------------------------------------------------------

# how many children an epoch breeds per child it keeps where the screening ranks them, and
# the individuals per epoch (on the scale of a hundred individual epoch) from which on it
# does: the Python package measured the threefold brood as a win at fifteen and a tie at
# five, and put the boundary at ten
const OFFSPRING_MULTIPLIER = 3.0
const BROOD_THRESHOLD = 10

"""
    SurrogateScreening(embedder; kwargs...)

Spends an expensive loss only on the promising individuals of an epoch and predicts the
loss of the others with a Gaussian process; pass it to `fit!` or `runGep` as `surrogate`.
`embedder` maps a chromosome to its latent vector, or to `nothing` for an expression that
cannot be evaluated: a [`SemanticEmbedder`](@ref), a [`GeneEmbedder`](@ref), a
[`TensorEmbedder`](@ref), or any function of that form. `SurrogateScreening(regressor,
probes; ...)` builds the embedder from a regressor.

Each epoch the new, distinct individuals are embedded (those that cannot be are scored as
a crash, without a loss call) and then:

- during the warmup, i.e. until `warmup_runs` individuals have been scored with a finite
  loss, at most `warmup_batch` of them are scored, picked at random (by Latin hypercube
  sampling over the latent vectors with `warmup_lhs`), and the others receive the median
  loss of that batch;
- after it, a [`GpScreen`](@ref) is fitted on the archive of scored individuals, the loss
  scores `individuals_per_epoch` of them, picked by the acquisition, a share
  `explore_fraction` of them as in the warmup instead, and every other one
  receives the prediction `mean + impute_beta * deviation`, made strictly worse than the
  best loss scored so far on every objective (no individual the loss has not seen may
  claim to beat or tie the incumbent).

A prediction is never cached, and the best individual of an epoch is scored before it is
recorded; with `validate_hof` the hall of fame is scored as well. A prediction is also
provisional: with `rescreen`, the default with several objectives, an individual that
carries one is screened again every epoch, with the process of that epoch (the batch
stays a count or a share of the new individuals). Without, a prediction that survives is
never looked at again, and the population fills up with stale ones: in a two-objective
search the clamp sends every optimistic prediction to the same corner of the front, and
in a search for an ODE system 197 of 200 survivors carried a prediction after 50 epochs,
61 of them in that corner, while the best scored losses had not moved since epoch 10.
With one objective, where the incumbent leads the population and is scored every epoch,
screening the predictions again was a wash on the benchmark of the package (better on
three of its six screened arms, worse on two, even on one). Once `min_failures` scored
individuals had a non-finite loss, the batch is filled from the individuals a
[`FeasibilityModel`](@ref) considers runnable first.

The screening is the memory of a search: every individual the loss has scored, with its
latent vector and loss. Use a fresh one for a new search (or another regressor); passing
the same one again continues from what it has learned, e.g. with a `load_state_callback`,
and it can be saved alongside the population with `Serialization`.

# Keyword arguments
- `screen=GpScreen()`: ranks the pending individuals
- `individuals_per_epoch=10`: individuals the loss scores per screened epoch, a count or a
  share (a float in `(0, 1)`) of the individuals the epoch proposes; `screen_budget` is
  the same under the name of the acquisition literature
- `budget_rule=:fixed`: `:uncertainty` scores only the individuals whose optimistic bound
  still beats the incumbent, between `min_individuals` and `individuals_per_epoch` of
  them; measured as the better way to spend the loss where objectives stay open (several
  objectives) and as a collapse with a single objective
- `min_individuals=1`: floor of the batch under the uncertainty rule, a count or a share
- `min_failures=5`: failed loss calls before the feasibility gate starts
- `explore_fraction=0.1`: share of the batch picked as in the warmup, to explore
- `warmup_runs`: scored individuals before the screening starts, by default six times
  the batch, at least 60
- `warmup_batch`: most individuals scored per warmup epoch, by default twice the batch,
  at least 20; a share of the epoch is accepted, and `nothing` scores every individual
  during the warmup
- `warmup_lhs=false`: pick the warmup and exploration individuals by Latin hypercube
  sampling over their latent vectors instead of uniformly
- `budget_decay=nothing`: epochs of an exponential schedule of the batch, from
  `warmup_batch` down to `individuals_per_epoch`
- `impute_beta=0.0`: deviations added to a prediction, i.e. the pessimism of the
  imputation
- `archive_cap=1000`: most recent scored individuals the processes are built on
- `offspring_multiplier=nothing`: children bred per child the population takes, the
  process picking which enter (see [`brood_multiplier`](@ref)); `nothing` decides by the
  batch, 1 switches the brood off
- `characterize_initial=true`: with an oversampled initial population
  (`population_sampling_multiplier > 1`), pick it by Latin hypercube sampling over the
  latent vectors, so the start fills the space the search is screened in
- `validate_hof=true`: score the predicted members of the returned hall of fame
- `rescreen=nothing`: hand the individuals that carry a prediction back to the screening
  at the start of every epoch, next to the new ones, so that the process can pick them for
  a loss call or give them a fresh prediction; `false` keeps a prediction until the
  individual dies, as the Python package does, and `nothing` screens again with several
  objectives only (see above)
- `target_transform=:log10`: transform of the loss values before the processes see them
  (see [`TRANSFORMS`](@ref)); `:asinh` or `:none` for losses that can be negative
- `exact_objectives=nothing`: objectives known without the loss, as pairs
  `objective => chromosome -> value` (a `Dict` or a vector of pairs), e.g.
  `Dict(2 => c -> 0.01 * length(c.expression_raw))` for a size objective. A predicted
  individual gets their value instead of a prediction (and no clamp, it is not a guess),
  and the acquisition sees them exactly; a Gaussian process cannot predict the size of an
  expression from its behaviour. The value has to be the one the loss sets.
- `objective_expressions=nothing`: for a chromosome that carries several expressions, the
  expression each objective judges, one entry per objective (`0` or `:all` for one that
  depends on all of them, e.g. a size), e.g. `[1, 2]` when the loss sets one objective per
  expression. The process of an objective then sees the latent block of its expression
  alone (the `inputs` of the [`GpScreen`](@ref), [`expression_blocks`](@ref)) instead of
  the whole latent vector, whose other blocks only blur its distances.
- `seed=0`: seed of the random choices of the screening

# Several objectives and several expressions
A loss that sets several objectives is screened with one process per objective, a random
Chebyshev scalarization per epoch for the pick (ParEGO) or the expected hypervolume
improvement (`GpScreen(acquisition=:ehvi)`), and a prediction is clamped behind the best
scored value of every objective, so no prediction dominates the individual holding one.
The population survives by its mean fitness, though, which a prediction can beat without
dominating anyone, so the scored individuals that hold the best value of an objective
survive regardless ([`keep_best_scored!`](@ref)). A chromosome that carries several
expressions (split by `split_karva` in the loss) is embedded with one block per
expression: pass `expressions` to the embedder or to `SurrogateScreening(regressor,
probes; expressions=k)`. Where every objective judges one of them, `objective_expressions`
lets its process see that block alone: in two searches for a system of two ODEs, one
objective per equation, the rank correlations of the predictions with held-out losses
went from -0.21, 0.35, 0.18 and 0.29 to 0.19, 0.35, 0.54 and 0.56.

# Diagnostics
- `evaluated_count`: loss calls made through the screening; `imputed_count`: predictions
  handed out instead; `broken_count`: individuals scored as a crash for their embedding;
  `failed_count`: loss calls that returned a non-finite value
- `spearman_log`: rank correlation of the prediction and the realized loss, per screened
  batch -- the running diagnostic of the surrogate; with several objectives, of the first
  one the processes predict
- `spearman_objectives`: the same for every objective, one log each (empty for an
  objective in `exact_objectives`)
- `last_batch`: the individuals the loss scored in the last epoch, as named tuples
  `(chromosome, fitness, reason)`, the reason being `:warmup`, `:acquisition`, `:explore`
  or `:validation`
- [`archive_size`](@ref), [`is_validated`](@ref)
"""
mutable struct SurrogateScreening
    embedder::Any
    screen::Any
    screen_budget::Union{Int,Float64}
    budget_rule::Symbol
    min_individuals::Union{Int,Float64}
    min_failures::Int
    explore_fraction::Float64
    warmup_runs::Int
    warmup_batch::Union{Int,Float64,Nothing}
    warmup_lhs::Bool
    budget_decay::Union{Float64,Nothing}
    impute_beta::Float64
    archive_cap::Int
    offspring_multiplier::Union{Float64,Nothing}
    characterize_initial::Bool
    validate_hof::Bool
    target_transform::Symbol
    forward::Function
    inverse::Function
    exact_objectives::Dict{Int,Any}
    rng::MersenneTwister

    # the memory of the surrogate: the scored individuals and what the loss returned
    archive_vectors::Vector{Vector{Float64}}
    archive_targets::Vector{Vector{Float64}}
    risk_vectors::Vector{Vector{Float64}}
    risk_labels::Vector{Float64}
    feasibility::Union{FeasibilityModel,Nothing}
    incumbent::Union{Vector{Float64},Nothing}
    known::Dict{Vector{Int8},Tuple}
    # the prediction each individual that carries one was given; weak, so the screening
    # holds no individual alive
    predictions::WeakKeyDict{Chromosome,Tuple}
    rescreen::Union{Bool,Nothing}
    # length of the latent vectors, set by the first embedding (0 before)
    latent_dim::Int
    screen_ready::Bool
    epochs_seen::Int

    evaluated_count::Int
    imputed_count::Int
    broken_count::Int
    failed_count::Int
    spearman_log::Vector{Float64}
    spearman_objectives::Vector{Vector{Float64}}
    last_batch::Vector{Any}
    last_preselect::NamedTuple
end

function SurrogateScreening(embedder;
    screen=GpScreen(),
    individuals_per_epoch::Union{Real,Nothing}=nothing,
    screen_budget::Real=10,
    budget_rule::Symbol=:fixed,
    min_individuals::Real=1,
    min_failures::Integer=5,
    explore_fraction::Real=0.1,
    warmup_runs::Union{Integer,Nothing}=nothing,
    warmup_batch::Union{Real,Nothing,Symbol}=:auto,
    warmup_lhs::Bool=false,
    budget_decay::Union{Real,Nothing}=nothing,
    impute_beta::Real=0.0,
    archive_cap::Integer=1000,
    offspring_multiplier::Union{Real,Nothing}=nothing,
    characterize_initial::Bool=true,
    validate_hof::Bool=true,
    target_transform::Union{Symbol,AbstractString}=:log10,
    exact_objectives::Union{Nothing,AbstractDict,AbstractVector{<:Pair}}=nothing,
    rescreen::Union{Bool,Nothing}=nothing,
    objective_expressions::Union{Nothing,AbstractVector}=nothing,
    seed::Integer=0)

    budget_rule in (:fixed, :uncertainty) || throw(ArgumentError(
        "the budget rule $budget_rule is unknown, use :fixed or :uncertainty"))
    warmup_batch isa Symbol && warmup_batch !== :auto && throw(ArgumentError(
        "warmup_batch is a count, a share, nothing or :auto; got $warmup_batch"))
    0 <= explore_fraction <= 1 || throw(ArgumentError(
        "explore_fraction is a share of the batch, got $explore_fraction"))
    forward, inverse = get_transform(target_transform)
    exact = Dict{Int,Any}(isnothing(exact_objectives) ? () : exact_objectives)
    all(>=(1), keys(exact)) || throw(ArgumentError(
        "exact_objectives maps objective numbers (from 1) to functions of a chromosome"))
    isnothing(objective_expressions) || tie_objectives!(screen, embedder, objective_expressions)

    # individuals_per_epoch is the readable name of screen_budget
    budget = count_or_share(isnothing(individuals_per_epoch) ? screen_budget : individuals_per_epoch)
    # the defaults are the configuration the Python feasibility study measured as its best
    reference = resolve_count(budget, 100, 10)
    runs = isnothing(warmup_runs) ? max(6 * reference, 60) : Int(warmup_runs)
    batch = warmup_batch === :auto ? max(2 * reference, 20) :
            isnothing(warmup_batch) ? nothing : count_or_share(warmup_batch)

    return SurrogateScreening(embedder, screen, budget, budget_rule,
        count_or_share(min_individuals), Int(min_failures), Float64(explore_fraction), runs,
        batch, warmup_lhs, isnothing(budget_decay) ? nothing : Float64(budget_decay),
        Float64(impute_beta), Int(archive_cap),
        isnothing(offspring_multiplier) ? nothing : Float64(offspring_multiplier),
        characterize_initial, validate_hof, Symbol(target_transform), forward, inverse,
        exact, MersenneTwister(seed),
        Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], Float64[], nothing, nothing,
        Dict{Vector{Int8},Tuple}(), WeakKeyDict{Chromosome,Tuple}(), rescreen, 0, false, 0,
        0, 0, 0, 0, Float64[], Vector{Float64}[], Any[], (bred=0, kept=0, broken=0, ranked=0))
end

function Base.show(io::IO, s::SurrogateScreening)
    print(io, "SurrogateScreening(", archive_size(s), " scored, ", s.evaluated_count,
        " loss calls, ", s.imputed_count, " predictions)")
end

"""
    tie_objectives!(screen, embedder, objective_expressions)

Set the `inputs` of a [`GpScreen`](@ref) so that the process of objective `j` sees the
latent block of expression `objective_expressions[j]` alone ([`expression_blocks`](@ref)),
or the whole latent vector for `0` or `:all`.
"""
function tie_objectives!(screen, embedder, objective_expressions::AbstractVector)
    screen isa GpScreen || throw(ArgumentError(
        "objective_expressions ties the processes of a GpScreen to latent blocks; the " *
        "screen is a $(typeof(screen))"))
    isnothing(screen.inputs) || throw(ArgumentError(
        "give the inputs of the screen or objective_expressions, not both"))
    blocks = expression_blocks(embedder)
    screen.inputs = Any[
        if e === :all || e == 0
            nothing
        elseif e isa Integer && 1 <= e <= length(blocks)
            blocks[e]
        else
            throw(ArgumentError("objective_expressions names expression $e of " *
                                "$(length(blocks)); use 1 to $(length(blocks)), or 0 or :all"))
        end
        for e in objective_expressions]
    return screen
end

"""
    archive_size(s::SurrogateScreening)

Number of scored individuals with a finite loss the processes learn from.
"""
archive_size(s::SurrogateScreening) = length(s.archive_targets)

"""
    is_validated(s::SurrogateScreening, chromosome)

Whether the fitness of `chromosome` is a loss value and not a prediction: the loss has
scored its expression (its karva string), and it does not carry a prediction handed out
before (see [`carries_prediction`](@ref)). A copy scored from the cache counts as
validated, with the penalty on its fitness.
"""
is_validated(s::SurrogateScreening, chromosome::Chromosome) =
    known_expression(s, chromosome) && !carries_prediction(s, chromosome)

"""
    known_expression(s::SurrogateScreening, chromosome)

Whether the loss has scored the expression (the karva string) of `chromosome`.
"""
known_expression(s::SurrogateScreening, chromosome::Chromosome) =
    haskey(s.known, chromosome.expression_raw)

"""
    carries_prediction(s::SurrogateScreening, chromosome)

Whether the fitness of `chromosome` is still the prediction the screening gave it (or a
penalized copy of one): nothing has scored it since.
"""
function carries_prediction(s::SurrogateScreening, chromosome::Chromosome)
    prediction = get(s.predictions, chromosome, nothing)
    return !isnothing(prediction) && isequal(chromosome.fitness, prediction)
end

"""
    mark_prediction!(s::SurrogateScreening, chromosome)

Record that the fitness `chromosome` holds is a prediction.
"""
mark_prediction!(s::SurrogateScreening, chromosome::Chromosome) =
    (s.predictions[chromosome] = chromosome.fitness; chromosome)

# whether the predictions are screened again in a search with `k` objectives
rescreens(s::SurrogateScreening, k::Int) = isnothing(s.rescreen) ? k > 1 : s.rescreen

"""
    rescreen_predictions!(s, population, n, unscored) -> Set{Int}

With `rescreen` (by default with several objectives), drop the prediction of every
individual among the first `n` of `population` that carries one, by setting its fitness
to `unscored`, so that the epoch screens it again with the new individuals; returns the
positions. A prediction is provisional: the process of the next epoch knows more, and an
optimistic prediction should reach the loss rather than lead the population for good.
"""
function rescreen_predictions!(s::SurrogateScreening, population::AbstractVector, n::Int,
    unscored::Tuple)
    positions = Set{Int}()
    rescreens(s, length(unscored)) || return positions
    for i in 1:min(n, length(population))
        carries_prediction(s, population[i]) || continue
        population[i].fitness = unscored
        push!(positions, i)
    end
    return positions
end

"""
    keep_best_scored!(s, population, n) -> population

With several objectives, move every scored individual that holds the best value of an
objective among `population` (sorted by mean fitness) right behind its leader, into the
first `n`, the survivors, where the next generation does not take its place; the others
keep their order, and the last survivors make room. A population survives by its mean
fitness, and a prediction, which the clamp keeps behind the best scored value of every
objective, can still beat those who hold them by its mean; so, as with one objective, no
prediction costs the holder of a best loss its place, and the best scored value of an
objective among the survivors never gets worse. With one objective the holder leads the
sorted population anyway.
"""
function keep_best_scored!(s::SurrogateScreening, population::AbstractVector, n::Int)
    n > 1 || return population
    k = length(first(population).fitness)
    k > 1 || return population
    holders = Int[]
    for j in 1:k
        best = 0
        for (i, c) in enumerate(population)
            v = c.fitness[j]
            (isfinite(v) && is_validated(s, c)) || continue
            (best == 0 || v < population[best].fitness[j]) && (best = i)
        end
        # the leader stays where it is
        best > 1 && !(best in holders) && push!(holders, best)
    end
    sort!(holders)
    holders == 2:length(holders)+1 && return population
    moved = population[holders]
    deleteat!(population, holders)
    for (offset, c) in enumerate(moved)
        insert!(population, 1 + offset, c)
    end
    return population
end

"""
    brood_multiplier(s::SurrogateScreening)

Children an epoch breeds per child the population takes. Breeding more children than the
population takes and letting the process pick among them is the most effective option of
the screening where the epoch scores enough individuals for the process to tell them apart:
the Python package measured the threefold brood as a win at fifteen individuals of a
hundred per epoch (at the same number of loss calls) and as a tie, i.e. pure cost, at five.
So it is `OFFSPRING_MULTIPLIER` (3) from `BROOD_THRESHOLD` (10) individuals per epoch on,
read on a hundred individual epoch, and 1 below; `offspring_multiplier` overrules it.
"""
function brood_multiplier(s::SurrogateScreening)
    isnothing(s.offspring_multiplier) || return max(s.offspring_multiplier, 1.0)
    return resolve_count(s.screen_budget, 100, 10) < BROOD_THRESHOLD ? 1.0 : OFFSPRING_MULTIPLIER
end

"""
    brood_size(s::SurrogateScreening, mating_size)

Children an epoch breeds for `mating_size` places: even, and at least `mating_size`.
"""
function brood_size(s::SurrogateScreening, mating_size::Int)
    bred = round(Int, brood_multiplier(s) * mating_size)
    return max(mating_size, bred - bred % 2)
end

individuals_of(s::SurrogateScreening, proposed::Int) = resolve_count(s.screen_budget, proposed, 10)
floor_of(s::SurrogateScreening, proposed::Int) = resolve_count(s.min_individuals, proposed, 1)
warmup_of(s::SurrogateScreening, proposed::Int) = resolve_count(s.warmup_batch, proposed, nothing)

"""
    epoch_budget(s, proposed)

Individuals the loss may score this epoch: `individuals_per_epoch`, or with a
`budget_decay` the schedule `warmup_batch * exp(-epoch / budget_decay)`, floored at it.
"""
function epoch_budget(s::SurrogateScreening, proposed::Int)
    budget = individuals_of(s, proposed)
    warmup = warmup_of(s, proposed)
    (isnothing(s.budget_decay) || isnothing(warmup)) && return budget
    return max(budget, ceil(Int, warmup * exp(-s.epochs_seen / s.budget_decay)))
end

"""
    embed(s, chromosome)

The latent vector of `chromosome` by the embedder of `s`, or `nothing` if it cannot be
embedded: an embedder that throws, returns `nothing` or a non-finite value, or a vector of
another length than the first one it returned counts as the latter.
"""
function embed(s::SurrogateScreening, chromosome::Chromosome)
    v = try
        s.embedder(chromosome)
    catch
        nothing
    end
    isnothing(v) && return nothing
    v = Vector{Float64}(v)
    all(isfinite, v) || return nothing
    # the processes need latent vectors of one length
    s.latent_dim == 0 && (s.latent_dim = length(v))
    return length(v) == s.latent_dim ? v : nothing
end

"""
    exact_values(s, chromosome, k)

The `k` objectives of `chromosome` known without the loss (`exact_objectives`), `NaN`
where an objective is not known or its function throws or returns a non-finite value.
"""
function exact_values(s::SurrogateScreening, chromosome::Chromosome, k::Int)
    values = fill(NaN, k)
    for (j, f) in s.exact_objectives
        j <= k || throw(ArgumentError(
            "exact_objectives names objective $j of a loss that sets $k objectives"))
        v = try
            Float64(f(chromosome))
        catch
            NaN
        end
        values[j] = isfinite(v) ? v : NaN
    end
    return values
end

"""
    exact_matrix(s, chromosomes, k)

The known objectives of `chromosomes` as a `k × m` matrix in the units of the loss (see
[`exact_values`](@ref)), or `nothing` without `exact_objectives`.
"""
function exact_matrix(s::SurrogateScreening, chromosomes::AbstractVector, k::Int)
    isempty(s.exact_objectives) && return nothing
    exact = Matrix{Float64}(undef, k, length(chromosomes))
    for (i, c) in enumerate(chromosomes)
        exact[:, i] .= exact_values(s, c, k)
    end
    return exact
end

# the known objectives in the transformed units of the processes, NaN kept
transformed(s::SurrogateScreening, exact) = isnothing(exact) ? nothing : s.forward.(exact)

# the number of objectives the archive holds
objective_count(s::SurrogateScreening) = length(first(s.archive_targets))

"""
    feasibility_model(s)

The [`FeasibilityModel`](@ref) of the outcomes of the loss calls, or `nothing` where the
failures say nothing: fewer than `min_failures` of them, or no success at all.
"""
function feasibility_model(s::SurrogateScreening)
    failures = length(s.risk_labels) - round(Int, sum(s.risk_labels))
    (failures < s.min_failures || failures == length(s.risk_labels)) && return nothing
    return FeasibilityModel(reduce(hcat, s.risk_vectors), s.risk_labels)
end

"""
    lhs_pick(rng, F, n)

`n` columns of `F` (`d × c`) spread over the space the columns span: a Latin hypercube of
`n` points in the per-dimension min-max box, each taking the nearest column not yet taken.
"""
function lhs_pick(rng::AbstractRNG, F::AbstractMatrix, n::Int)
    d, c = size(F)
    lo = vec(minimum(F; dims=2))
    span = vec(maximum(F; dims=2)) .- lo
    Z = ifelse.(span .> 0, (F .- lo) ./ ifelse.(span .> 0, span, 1.0), 0.5)
    taken = falses(c)
    picks = Int[]
    strata = [randperm(rng, n) for _ in 1:d]
    for i in 1:n
        target = [(strata[k][i] - 1 + rand(rng)) / n for k in 1:d]
        best, best_dist = 0, Inf
        for j in 1:c
            taken[j] && continue
            dist = sum(abs2, view(Z, :, j) .- target)
            dist < best_dist && ((best, best_dist) = (j, dist))
        end
        best == 0 && break
        taken[best] = true
        push!(picks, best)
    end
    return picks
end

"""
    space_filling(s, X, indices, n)

`n` of the candidate positions `indices` (columns of `X`), picked uniformly at random, or
by [`lhs_pick`](@ref) with `warmup_lhs`.
"""
function space_filling(s::SurrogateScreening, X::AbstractMatrix, indices::AbstractVector{Int}, n::Int)
    n <= 0 && return Int[]
    n >= length(indices) && return collect(indices)
    s.warmup_lhs || return indices[randperm(s.rng, length(indices))[1:n]]
    return indices[lhs_pick(s.rng, X[:, indices], n)]
end

"""
    uncertain_budget(s, X, cap)

Individuals the process cannot rule out, between the floor and `cap`.
"""
function uncertain_budget(s::SurrogateScreening, X::AbstractMatrix, cap::Int; known=nothing)
    isnothing(s.incumbent) && return cap
    plausible = plausible_screen(s.screen, X, s.forward.(s.incumbent); known=known)
    return min(cap, max(floor_of(s, size(X, 2)), count(plausible)))
end

"""
    ScreenPlan

What [`screen_epoch!`](@ref) decided for the individuals of an epoch: the population
indices it was handed (`work`), those it could embed (`entries`, with their latent vectors
as the columns of `vectors`) and those it could not (`broken`), the positions in `entries`
the loss scores (`chosen`, with a `reason` each), the `mode` of the epoch (`:empty`,
`:all`, `:warmup` or `:screened`), and for a screened epoch the predictions of every entry
(`means`, `deviations`, `k × length(entries)`) and its objectives known without the loss
(`exact`, in the units of the loss, `NaN` where unknown; empty without
`exact_objectives`).
"""
struct ScreenPlan
    work::Vector{Int}
    entries::Vector{Int}
    broken::Vector{Int}
    vectors::Matrix{Float64}
    chosen::Vector{Int}
    reasons::Vector{Symbol}
    mode::Symbol
    means::Matrix{Float64}
    deviations::Matrix{Float64}
    exact::Matrix{Float64}
end

"""
    evaluated_indices(plan)

The population indices the loss scores this epoch.
"""
evaluated_indices(plan::ScreenPlan) = plan.entries[plan.chosen]

"""
    cached_indices(plan)

The population indices whose fitness may be cached after the epoch: the scored ones and
the broken ones (scored as a crash); never the predicted ones.
"""
cached_indices(plan::ScreenPlan) = vcat(evaluated_indices(plan), plan.broken)

"""
    screen_epoch!(s, population, work, worst; rescreened=Set{Int}()) -> ScreenPlan

Decide which of the individuals `population[work]` (unscored, one per karva string) the
loss scores this epoch. The individuals that cannot be embedded get the fitness `worst`
at once; the process is fitted here for a screened epoch. The loss is called by the caller
on [`evaluated_indices`](@ref), and [`commit_epoch!`](@ref) then archives the results and
predicts the fitness of the others. `rescreened` holds the positions of the individuals
screened again ([`rescreen_predictions!`](@ref)): they compete for the batch, whose size a
share resolves against the new individuals only.
"""
function screen_epoch!(s::SurrogateScreening, population::AbstractVector,
    work::AbstractVector{Int}, worst::Tuple; rescreened::AbstractSet{Int}=Set{Int}())
    entries = Int[]
    broken = Int[]
    vectors = Vector{Float64}[]
    for i in work
        v = embed(s, population[i])
        if isnothing(v)
            # an expression that cannot be evaluated on the probes would only waste a
            # loss call
            population[i].fitness = worst
            push!(broken, i)
        else
            push!(entries, i)
            push!(vectors, v)
        end
    end
    s.broken_count += length(broken)
    m = length(entries)
    X = m == 0 ? zeros(0, 0) : reduce(hcat, vectors)

    # what the loss scores this epoch; the best individual, scored after the sort, joins it
    s.last_batch = Any[]
    # an epoch of one individual is not an epoch of the schedule
    m > 1 && (s.epochs_seen += 1)

    warming = archive_size(s) < max(s.warmup_runs, 2)
    # a share of the epoch is a share of what it proposes, the new individuals
    fresh = count(i -> !(i in rescreened), entries)
    proposed = fresh > 0 ? fresh : m
    budget = epoch_budget(s, proposed)
    warmup = warmup_of(s, proposed)
    none = zeros(0, 0)

    if m == 0
        return ScreenPlan(collect(work), entries, broken, X, Int[], Symbol[], :empty, none,
            none, none)
    elseif warming && !isnothing(warmup) && m > max(budget, warmup)
        # a capped warmup picks the individuals the process learns most from; the others
        # receive the median of what the batch realized
        chosen = space_filling(s, X, collect(1:m), max(budget, warmup))
        return ScreenPlan(collect(work), entries, broken, X, chosen, fill(:warmup, length(chosen)),
            :warmup, none, none, none)
    elseif warming || m <= budget
        return ScreenPlan(collect(work), entries, broken, X, collect(1:m),
            fill(warming ? :warmup : :acquisition, m), :all, none, none, none)
    end

    fit_screen!(s.screen, reduce(hcat, s.archive_vectors[max(1, end - s.archive_cap + 1):end]),
        reduce(hcat, s.archive_targets[max(1, end - s.archive_cap + 1):end]); rng=s.rng)
    s.screen_ready = true
    # the objectives known without the loss enter the acquisition as they are
    exact = exact_matrix(s, population[entries], objective_count(s))
    known = transformed(s, exact)
    s.budget_rule === :uncertainty && (budget = uncertain_budget(s, X, budget; known=known))
    explore = round(Int, budget * s.explore_fraction)

    # what the loss could not score is worth knowing before the budget is spent again on
    # the same kind of individual
    s.feasibility = feasibility_model(s)
    risk = isnothing(s.feasibility) ? nothing : s.feasibility(X)
    chosen = select_screen(s.screen, X, budget - explore; feasible=risk, known=known)
    reasons = fill(:acquisition, length(chosen))
    # the exploration picks as the warmup does, at random or by Latin hypercube sampling
    # over the latent vectors, rather than where the process already points
    open = setdiff(1:m, chosen)
    explored = space_filling(s, X, open, min(explore, length(open)))
    append!(chosen, explored)
    append!(reasons, fill(:explore, length(explored)))
    means, deviations = predict_screen(s.screen, X; known=known)
    return ScreenPlan(collect(work), entries, broken, X, chosen, reasons, :screened, means,
        deviations, isnothing(exact) ? none : exact)
end

"""
    record_evaluation!(s, chromosome, vector, reason)

Archive one loss call: the outcome for the feasibility model, a finite result for the
processes and the incumbent (the best loss per objective), and the fitness as the known
value of the karva string.
"""
function record_evaluation!(s::SurrogateScreening, chromosome::Chromosome,
    vector::AbstractVector, reason::Symbol)
    fitness = chromosome.fitness
    values = Float64[v for v in fitness]
    usable = all(isfinite, values)
    push!(s.risk_vectors, Vector{Float64}(vector))
    push!(s.risk_labels, usable ? 1.0 : 0.0)
    usable || (s.failed_count += 1)
    if length(s.risk_vectors) > s.archive_cap
        deleteat!(s.risk_vectors, 1:length(s.risk_vectors)-s.archive_cap)
        deleteat!(s.risk_labels, 1:length(s.risk_labels)-s.archive_cap)
    end
    if usable
        push!(s.archive_vectors, Vector{Float64}(vector))
        push!(s.archive_targets, s.forward.(values))
        # the best scored value per objective is the floor of every prediction
        s.incumbent = isnothing(s.incumbent) ? values : min.(s.incumbent, values)
    end
    # a loss that left the individual unscored (NaN) has not scored it
    any(isnan, values) || (s.known[copy(chromosome.expression_raw)] = fitness)
    push!(s.last_batch, (chromosome=chromosome, fitness=fitness, reason=reason))
    s.evaluated_count += 1
    return usable
end

"""
    commit_epoch!(s, population, plan)

After the loss has scored `evaluated_indices(plan)`: archive the results and give every
other embedded individual of the plan its prediction. A screened epoch predicts
`mean + impute_beta * deviation`, a capped warmup epoch the median of the batch per
objective (and leaves the others unscored, to be screened again, if nothing in the batch
was finite); either is made one float step worse than the incumbent where it is not
already worse. Returns the number of predictions.
"""
function commit_epoch!(s::SurrogateScreening, population::AbstractVector, plan::ScreenPlan)
    scored = evaluated_indices(plan)
    for (pos, reason) in zip(plan.chosen, plan.reasons)
        record_evaluation!(s, population[plan.entries[pos]], plan.vectors[:, pos], reason)
    end
    chosen = Set(plan.chosen)
    imputed = 0
    if plan.mode === :screened
        k = size(plan.means, 1)
        for pos in eachindex(plan.entries)
            pos in chosen && continue
            prediction = s.inverse.(plan.means[:, pos] .+ s.impute_beta .* plan.deviations[:, pos])
            exact = isempty(plan.exact) ? nothing : plan.exact[:, pos]
            c = population[plan.entries[pos]]
            c.fitness = impute(s, prediction, exact)
            mark_prediction!(s, c)
            imputed += 1
        end
        realized = [population[i].fitness for i in scored]
        diagnose!(s, plan.means[:, plan.chosen], realized)
    elseif plan.mode === :warmup
        finite = [Float64[v for v in population[i].fitness] for i in scored
                  if all(isfinite, population[i].fitness)]
        if !isempty(finite)
            k = length(first(finite))
            # the median of the batch, per objective, is the prediction of a warmup
            median_fitness = [median(f[j] for f in finite) for j in 1:k]
            for pos in eachindex(plan.entries)
                pos in chosen && continue
                c = population[plan.entries[pos]]
                exact = isempty(s.exact_objectives) ? nothing : exact_values(s, c, k)
                c.fitness = impute(s, median_fitness, exact)
                mark_prediction!(s, c)
                imputed += 1
            end
        end
    end
    s.imputed_count += imputed
    return imputed
end

"""
    impute(s, prediction, exact) -> Tuple

The fitness of an individual the loss has not scored: the known objectives (`exact`, `NaN`
where unknown, or `nothing`) as they are, and every other one predicted and made strictly
worse than the incumbent, not equal to it: a prediction floored at the best scored value
of an objective ties the individual holding it and beats it on the others, i.e. it
dominates a champion it never reproduced. One float step keeps the champion non-dominated
and is free of any scale.
"""
function impute(s::SurrogateScreening, prediction::AbstractVector, exact)
    k = length(prediction)
    values = Vector{Float64}(prediction)
    for j in 1:k
        if !isnothing(exact) && !isnan(exact[j])
            values[j] = exact[j]
        elseif !isnothing(s.incumbent)
            values[j] = max(values[j], nextfloat(s.incumbent[j]))
        end
    end
    return ntuple(j -> values[j], k)
end

# the first objective the processes predict, the one the diagnostic follows
predicted_objective(s::SurrogateScreening, k::Int) =
    something(findfirst(j -> !haskey(s.exact_objectives, j), 1:k), 1)

"""
    diagnose!(s, predicted, realized)

Log the rank correlation of the predictions (`k × n`, transformed) and the realized losses
of a screened batch, per objective the processes predict (`spearman_objectives`), where at
least three of the realized values are finite; `spearman_log` follows the first of them.
"""
function diagnose!(s::SurrogateScreening, predicted::AbstractMatrix, realized::AbstractVector)
    k = size(predicted, 1)
    length(s.spearman_objectives) == k || (s.spearman_objectives = [Float64[] for _ in 1:k])
    followed = predicted_objective(s, k)
    for j in 1:k
        haskey(s.exact_objectives, j) && continue
        values = [Float64(r[j]) for r in realized]
        finite = isfinite.(values)
        count(finite) < 3 && continue
        rho = corspearman(Vector{Float64}(predicted[j, finite]), s.forward.(values[finite]))
        isfinite(rho) || continue
        push!(s.spearman_objectives[j], rho)
        j == followed && push!(s.spearman_log, rho)
    end
    return
end

"""
    record_validation!(s, chromosome) -> Bool

Archive the loss of an individual that was scored outside the screening, e.g. the best
individual of an epoch, which carried a prediction until the loss scored it. Returns
whether it was recorded; an expression already known, or one that cannot be embedded, is
not archived again.
"""
function record_validation!(s::SurrogateScreening, chromosome::Chromosome)
    is_validated(s, chromosome) && return false
    v = embed(s, chromosome)
    if isnothing(v)
        values = Float64[f for f in chromosome.fitness]
        any(isnan, values) || (s.known[copy(chromosome.expression_raw)] = chromosome.fitness)
        s.evaluated_count += 1
        return false
    end
    record_evaluation!(s, chromosome, v, :validation)
    return true
end

"""
    rescore_known(s, key)

The known loss of the karva string `key`, or `nothing`: the memory a copy falls back on
when the fitness cache has dropped the string.
"""
rescore_known(s::SurrogateScreening, key::AbstractVector{Int8}) = get(s.known, key, nothing)

"""
    preselect(s, children, keep; failed=falses(length(children))) -> Vector{Int}

The positions of the `keep` children of an oversampled epoch that enter the population.
The screening has no decoder, it ranks a finite set, and that set is the brood of the
epoch: breeding more children costs nothing but the genetic operators, and the process
picks which enter, which is selection pressure paid for in arithmetic instead of loss
calls. Children marked `failed` (e.g. not homogeneous after the dimension repair) are only
taken where too few others are left; among the others, the ones that cannot be embedded
come last, and the rest are picked like a screened batch, the acquisition taking the bulk
and an exploring remainder, picked as in the warmup, keeping the pick from locking the
search into what the surrogate already believes. Until a process has been fitted, the
first `keep` are kept.
"""
function preselect(s::SurrogateScreening, children::AbstractVector, keep::Int;
    failed::AbstractVector{Bool}=falses(length(children)))
    n = length(children)
    s.last_preselect = (bred=n, kept=keep, broken=0, ranked=0)
    keep >= n && return collect(1:n)
    feasible = findall(!, failed)
    length(feasible) == n && return choose_children(s, children, collect(1:n), keep)
    length(feasible) >= keep && return choose_children(s, children, feasible, keep)
    return vcat(feasible, choose_children(s, children, findall(failed), keep - length(feasible)))
end

function choose_children(s::SurrogateScreening, children::AbstractVector,
    candidates::Vector{Int}, keep::Int)
    keep <= 0 && return Int[]
    keep >= length(candidates) && return candidates
    s.screen_ready || return candidates[1:keep]
    valid = Int[]
    vectors = Vector{Float64}[]
    for c in candidates
        v = embed(s, children[c])
        isnothing(v) && continue
        push!(valid, c)
        push!(vectors, v)
    end
    invalid = setdiff(candidates, valid)
    s.last_preselect = merge(s.last_preselect, (broken=s.last_preselect.broken + length(invalid),))
    # dropping the broken ones already fits the population: nothing left to decide
    length(valid) <= keep && return vcat(valid, invalid[1:keep-length(valid)])
    X = reduce(hcat, vectors)
    explore = round(Int, keep * s.explore_fraction)
    risk = isnothing(s.feasibility) ? nothing : s.feasibility(X)
    known = transformed(s, exact_matrix(s, children[valid], objective_count(s)))
    picks = select_screen(s.screen, X, keep - explore; feasible=risk, known=known)
    open = setdiff(1:length(valid), picks)
    append!(picks, space_filling(s, X, open, min(explore, length(open))))
    s.last_preselect = merge(s.last_preselect, (ranked=s.last_preselect.ranked + length(valid),))
    return valid[picks[1:min(keep, length(picks))]]
end

"""
    characterize(s, population, n) -> Vector{Int}

The indices of `n` individuals of an oversampled initial population, picked by Latin
hypercube sampling (`select_n_samples_lhs`) over their latent vectors, so that the start
population fills the latent space the search is screened in. Individuals that cannot be
embedded are taken only where too few others are left.
"""
function characterize(s::SurrogateScreening, population::AbstractVector, n::Int)
    n >= length(population) && return collect(1:length(population))
    vectors = [embed(s, c) for c in population]
    valid = findall(!isnothing, vectors)
    isempty(valid) && return collect(1:n)
    invalid = findall(isnothing, vectors)
    length(valid) <= n && return vcat(valid, invalid[1:n-length(valid)])
    return valid[select_n_samples_lhs(reduce(hcat, vectors[valid]), n)]
end

end
