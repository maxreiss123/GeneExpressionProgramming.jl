#=
ODEBench (arXiv:2310.05573) through GeneExpressionProgramming.jl -- Route A of PLAN.md.

For each of the 63 systems: corrupt the training trajectory as the ODEFormer harness does
(drop `floor(rho*N)` random points, multiply by `1 + sigma*randn`); estimate derivative
targets by local cubic regression over 7 samples (15 with noise), which copes with the
irregular grid subsampling leaves; fit one `GepRegressor` per state component with
gene-wise linear scaling (so the constant optimiser does not run); integrate the assembled
system with RK4 on the reference grid. Score: variance-weighted R^2 against the clean
reference trajectory, stored unclipped; "accurate" means R^2 > 0.9.

  reconstruction : integrate from the clean initial state of the training trajectory
  generalization : integrate from the second, unseen initial condition

Candidates are the best model of every component, plus one variant per component with its
second-best swapped in; the highest reconstruction R^2 wins. That ranking uses the clean
reference trajectory, i.e. the metric itself; the Python baselines select the same way.

Corruption and metric follow the ODEFormer repository (odebench/solve_and_plot.py,
odeformer/metrics.py) rather than the paper's prose.

    julia --project=. --threads=4 paper/odebench/odebench_gep.jl [--sigma 0.05] [--rho 0.5]
        [--seed 1] [--epochs 150] [--pop 600] [--ids 1,2,3] [--out results/<name>.json]

`--out` is relative to paper/odebench. The figure scripts read
results/gep_sigma*_rho*_seed*.json, not the default name (gep_s<sigma>_r<rho>_seed<k>.json),
so pass e.g. `--out results/gep_sigma0.05_rho0.5_seed1.json`.
=#

include(joinpath(@__DIR__, "..", "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using .GeneExpressionProgramming.GepEntities
using Random
using Statistics
using LinearAlgebra
using JSON
using Printf

const HERE = @__DIR__

# ------------------------------------------------------------------ data ---------
struct SystemData
    id::Int
    dim::Int
    description::String
    truth::Vector{String}
    t::Vector{Float64}                 # clean grid, 512 points
    y::Matrix{Float64}                 # d x 512, clean training trajectory
    t2::Vector{Float64}
    y2::Matrix{Float64}                # the second trajectory (generalization reference)
end

function load_systems()
    raw = JSON.parsefile(joinpath(HERE, "strogatz_extended.json"))
    out = SystemData[]
    for s in raw
        sol1, sol2 = s["solutions"][1][1], s["solutions"][1][2]
        (sol1["success"] && sol2["success"]) || continue
        y1 = reduce(vcat, (permutedims(Float64.(c)) for c in sol1["y"]))
        y2 = reduce(vcat, (permutedims(Float64.(c)) for c in sol2["y"]))
        push!(out, SystemData(s["id"], s["dim"], s["eq_description"],
            String.(s["substituted"][1]),
            Float64.(sol1["t"]), y1, Float64.(sol2["t"]), y2))
    end
    return out
end

"""Corruption as in the reference harness: drop `floor(rho*N)` random indices, then
multiply every observation by `1 + sigma*randn`. The RNG is Julia's, seeded from
`hash((system id, seed))`, so a run is reproducible for a given Julia version; the
reference and the Python baselines use numpy's, so the corrupted datasets are
statistically equivalent rather than bitwise identical."""
function corrupt(sys::SystemData, sigma, rho, seed)
    rng = MersenneTwister(hash((sys.id, seed)))
    n = length(sys.t)
    keep = sort(shuffle(rng, collect(1:n))[1:n-floor(Int, rho * n)])
    t = sys.t[keep]
    y = sys.y[:, keep] .* (1 .+ sigma .* randn(rng, sys.dim, length(keep)))
    return t, y
end

# ------------------------------------------- derivative targets ------------------
"""Derivative of each component at each sample, from a local least-squares polynomial
over the `window` nearest points in time -- the standard replacement for a
Savitzky-Golay filter when subsampling has made the grid irregular."""
function local_poly_derivatives(t::Vector{Float64}, y::Matrix{Float64};
    window::Int=9, degree::Int=3)
    n = length(t)
    d = size(y, 1)
    dy = zeros(d, n)
    half = window ÷ 2
    for i in 1:n
        lo = clamp(i - half, 1, max(1, n - window + 1))
        hi = min(lo + window - 1, n)
        ts = t[lo:hi] .- t[i]
        deg = min(degree, length(ts) - 1)
        V = [ts[j]^p for j in eachindex(ts), p in 0:deg]
        F = qr(V)
        for k in 1:d
            coef = F \ y[k, lo:hi]
            dy[k, i] = coef[2]                     # d/dt at ts = 0
        end
    end
    return dy
end

# ------------------------------------------------- pointwise model ----------------
"""Wrap a fitted chromosome as `x::Vector -> Float64`. The buffer context is built once
on single-sample columns; each call writes the state into those columns in place and
evaluates. Gene-wise scaling weights are applied exactly as the search applied them."""
function make_component_fn(chrom::Chromosome, d::Int)
    ctx = buffer_context(chrom.toolbox, zeros(d, 1))
    ctx === nothing && error("model has no batched counterpart")
    # feature columns are keyed by chromosome symbol id, not feature number; the
    # InputSelector held in the toolbox knows which feature each id selects
    cols = Vector{Vector{Float64}}(undef, d)
    for (id, nd) in chrom.toolbox.nodes
        nd isa GeneExpressionProgramming.InputSelector || continue
        cols[nd.idx] = ctx.nodes[id]
    end
    E = GeneExpressionProgramming.GepEntities
    if isnothing(chrom.scaling_weights)
        expr = chrom.expression_raw
        return function (x::AbstractVector{Float64})
            @inbounds for i in 1:d
                cols[i][1] = x[i]
            end
            # kernels like sin/log throw on non-finite or out-of-domain operands; during
            # integration that is a diverging candidate, not an error
            v = try
                E.ctx_eval(expr, ctx)
            catch
                nothing
            end
            v isa AbstractVector ? Float64(v[1]) : NaN
        end
    end
    raw = E._karva_raw(chrom; split=true)
    genes = [collect(raw[j+1]) for j in eachindex(chrom.scaling_weights)]
    w = chrom.scaling_weights
    return function (x::AbstractVector{Float64})
        @inbounds for i in 1:d
            cols[i][1] = x[i]
        end
        acc = 0.0
        for (j, g) in enumerate(genes)
            v = try
                E.ctx_eval(g, ctx)
            catch
                nothing
            end
            v isa AbstractVector || return NaN
            acc += w[j] * Float64(v[1])
        end
        acc
    end
end

# ------------------------------------------------------ integration --------------
"""RK4 along the reference grid, with substeps capped at `hmax` (the default gives five
per reference interval; the Python baselines score with 10/1024, three). Divergence -- a
non-finite state or |x| beyond `xcap` -- aborts and leaves the rest of the trajectory
NaN, which `r2_vw` scores as a failure."""
function integrate_on_grid(fns::Vector{<:Function}, x0::Vector{Float64},
    tgrid::Vector{Float64}; hmax=10.0 / 2048, xcap=1e8)
    d = length(x0)
    out = fill(NaN, d, length(tgrid))
    x = copy(x0)
    out[:, 1] .= x
    k1 = zeros(d); k2 = zeros(d); k3 = zeros(d); k4 = zeros(d); xt = zeros(d)
    f!(dx, xx) = (for i in 1:d
        dx[i] = fns[i](xx)
    end)
    
    for gi in 2:length(tgrid)
        span = tgrid[gi] - tgrid[gi-1]
        nsub = max(1, ceil(Int, span / hmax))
        h = span / nsub
        for _ in 1:nsub
            f!(k1, x)
            @. xt = x + h / 2 * k1
            f!(k2, xt)
            @. xt = x + h / 2 * k2
            f!(k3, xt)
            @. xt = x + h * k3
            f!(k4, xt)
            @. x = x + h / 6 * (k1 + 2k2 + 2k3 + k4)
            if !all(isfinite, x) || maximum(abs, x) > xcap
                return out
            end
        end
        out[:, gi] .= x
    end
    return out
end

"""Variance-weighted R^2, the sklearn convention the reference evaluation uses:
1 - sum(SS_res) / sum(SS_tot) over components. A non-finite prediction gives -Inf."""
function r2_vw(y_true::Matrix{Float64}, y_pred::Matrix{Float64})
    all(isfinite, y_pred) || return -Inf
    ssres = sum(abs2, y_true .- y_pred)
    sstot = sum(abs2, y_true .- mean(y_true, dims=2))
    sstot <= 0 && return ssres <= 1e-12 ? 1.0 : -Inf
    return 1 - ssres / sstot
end

# ------------------------------------------------------------- fitting -----------
function fit_component(x::Matrix{Float64}, target::Vector{Float64}, d::Int, seed::Int;
    epochs=150, pop=600)
    Random.seed!(seed)
    reg = GepRegressor(d;
        entered_non_terminals=[:+, :-, :*, :/, :sin, :cos, :exp, :sqr],
        gene_count=3, head_len=6, rnd_count=2)
    fit!(reg, epochs, pop, x, target; loss_fun="mse", linear_scaling=true, hof=2)
    return reg.best_models_
end

function run_system(sys::SystemData, sigma, rho, seed; epochs, pop)
    t, y = corrupt(sys, sigma, rho, seed)
    window = sigma > 0 ? 15 : 7
    dy = local_poly_derivatives(t, y; window=window, degree=3)

    t_fit = @elapsed begin
        hof = [fit_component(y, vec(dy[k, :]), sys.dim, seed + 100k;
            epochs=epochs, pop=pop) for k in 1:sys.dim]
    end

    # candidate systems: best model per component, then each second-best swapped in
    combos = [[1 for _ in 1:sys.dim]]
    for k in 1:sys.dim
        length(hof[k]) > 1 || continue
        c = copy(combos[1])
        c[k] = 2
        push!(combos, c)
    end

    best = (r2r=-Inf, r2g=-Inf, exprs=String[])
    t_sel = @elapsed for combo in combos
        fns = Function[]
        ok = true
        for k in 1:sys.dim
            f = try
                make_component_fn(hof[k][combo[k]], sys.dim)
            catch
                ok = false
                break
            end
            push!(fns, f)
        end
        ok || continue
        pred = integrate_on_grid(fns, sys.y[:, 1], sys.t)
        r2r = r2_vw(sys.y, pred)
        isfinite(r2r) || continue
        if r2r > best.r2r
            pred2 = integrate_on_grid(fns, sys.y2[:, 1], sys.t2)
            best = (r2r=r2r,
                r2g=r2_vw(sys.y2, pred2),
                exprs=[equation_string(hof[k][combo[k]]) for k in 1:sys.dim])
        end
    end

    return Dict(
        "id" => sys.id, "dim" => sys.dim, "description" => sys.description,
        "sigma" => sigma, "rho" => rho, "seed" => seed,
        "r2_reconstruction" => best.r2r == -Inf ? nothing : best.r2r,
        "r2_generalization" => best.r2g == -Inf ? nothing : best.r2g,
        "expressions" => best.exprs, "truth" => sys.truth,
        "fit_time_s" => t_fit, "selection_time_s" => t_sel)
end

# ---------------------------------------------------------------- main -----------
function main()
    args = Dict{String,String}("sigma" => "0.0", "rho" => "0.0", "seed" => "1",
        "epochs" => "150", "pop" => "600", "ids" => "", "out" => "")
    i = 1
    while i <= length(ARGS)
        k = lstrip(ARGS[i], '-')
        haskey(args, k) || error("unknown option --$k")
        args[k] = ARGS[i+1]
        i += 2
    end
    sigma = parse(Float64, args["sigma"])
    rho = parse(Float64, args["rho"])
    seed = parse(Int, args["seed"])
    epochs = parse(Int, args["epochs"])
    pop = parse(Int, args["pop"])

    systems = load_systems()
    if !isempty(args["ids"])
        want = Set(parse.(Int, split(args["ids"], ',')))
        systems = [s for s in systems if s.id in want]
    end

    # warm up the JIT off the clock
    run_system(systems[1], sigma, rho, seed; epochs=2, pop=100)

    results = []
    for sys in systems
        r = run_system(sys, sigma, rho, seed; epochs=epochs, pop=pop)
        push!(results, r)
        r2r = something(r["r2_reconstruction"], -Inf)
        r2g = something(r["r2_generalization"], -Inf)
        @printf("id %2d  d=%d  r2_rec %8.4f  r2_gen %8.4f  %5.1fs  %s\n",
            sys.id, sys.dim, max(r2r, -9.9999), max(r2g, -9.9999),
            r["fit_time_s"] + r["selection_time_s"],
            first(sys.description, 42))
        flush(stdout)
    end

    acc(key) = mean([something(r[key], -Inf) > 0.9 for r in results])
    @printf("\nsigma=%.2f rho=%.2f seed=%d : accuracy (R2>0.9)  reconstruction %.3f   generalization %.3f\n",
        sigma, rho, seed, acc("r2_reconstruction"), acc("r2_generalization"))

    out = isempty(args["out"]) ?
          joinpath(HERE, "results", @sprintf("gep_s%.0e_r%.0e_seed%d.json", sigma, rho, seed)) :
          joinpath(HERE, args["out"])
    mkpath(dirname(out))
    open(out, "w") do io
        JSON.print(io, Dict("sigma" => sigma, "rho" => rho, "seed" => seed,
                "epochs" => epochs, "pop" => pop, "results" => results), 1)
    end
    println("wrote $out")
end

main()
