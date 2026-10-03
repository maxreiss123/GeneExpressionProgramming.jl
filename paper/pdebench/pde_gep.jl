#=
GeneExpressionProgramming.jl on the PDE-discovery benchmark: the pointwise route.

Reads the shared conditions file (data/conditions.json, written by pde_common.py): for
each (pde, sigma) cell, the noisy training features/target and the clean test
features/analytic right-hand side that every method gets. Fits u_t = f(u, u_x, ...) with
`GepRegressor` over {+, -, *} and gene-wise linear scaling (so the constant optimiser does
not run) -- the symbolic counterpart of PDE-FIND's fixed linear library -- picks among the
four best models on a validation split (the last 20 % of the training rows), and scores
the winner on the clean test features against the analytic right-hand side (functional
recovery: R^2 > 0.99).

    julia --project=. --threads=4 paper/pdebench/pde_gep.jl [out.json]

The results go to results/gep.json unless another file name is given, e.g.
gep_1thread.json for the single-thread timing in RESULTS.md.
=#

include(joinpath(@__DIR__, "..", "..", "src", "GeneExpressionProgramming.jl"))

using .GeneExpressionProgramming
using .GeneExpressionProgramming.GepEntities
using Random
using Statistics
using JSON
using Printf

const HERE = @__DIR__
const R2_RECOVERED = 0.99

# ---------------------------------------------------------- batched evaluation ---
"""Evaluate a fitted chromosome on a d x n feature matrix in one shot: build the
evaluation context on the matrix itself, then apply the gene-wise scaling weights as the
search did. Returns a length-n vector, all NaN when evaluation fails."""
function eval_batch(chrom::Chromosome, X::Matrix{Float64})
    d, n = size(X)
    ctx = buffer_context(chrom.toolbox, X)
    ctx === nothing && return fill(NaN, n)
    E = GeneExpressionProgramming.GepEntities
    if isnothing(chrom.scaling_weights)
        v = try
            E.ctx_eval(chrom.expression_raw, ctx)
        catch
            nothing
        end
        return v isa AbstractVector ? Float64.(v) : fill(NaN, n)
    end
    raw = E._karva_raw(chrom; split=true)
    genes = [collect(raw[j+1]) for j in eachindex(chrom.scaling_weights)]
    acc = zeros(n)
    for (j, g) in enumerate(genes)
        v = try
            E.ctx_eval(g, ctx)
        catch
            nothing
        end
        v isa AbstractVector || return fill(NaN, n)
        acc .+= chrom.scaling_weights[j] .* Float64.(v)
    end
    return acc
end

function r2(y_true::Vector{Float64}, y_pred::Vector{Float64})
    all(isfinite, y_pred) || return -Inf
    sstot = sum(abs2, y_true .- mean(y_true))
    sstot > 0 || return -Inf
    return 1 - sum(abs2, y_true .- y_pred) / sstot
end

# ------------------------------------------------------------------- fitting -----
function run_condition(cond; epochs=200, pop=1000, seed=1)
    names = String.(cond["names"])
    d = length(names)
    Xtr = reduce(hcat, (Float64.(r) for r in cond["X_train"]))   # d x n (rows were samples)
    ytr = Float64.(cond["y_train"])
    Xte = reduce(hcat, (Float64.(r) for r in cond["X_test"]))
    rhs = Float64.(cond["rhs_test"])

    ntr = floor(Int, 0.8 * length(ytr))
    Xf, yf = Xtr[:, 1:ntr], ytr[1:ntr]
    Xv, yv = Xtr[:, ntr+1:end], ytr[ntr+1:end]

    Random.seed!(seed)
    t_fit = @elapsed begin
        reg = GepRegressor(d;
            entered_non_terminals=[:+, :-, :*],
            gene_count=4, head_len=5, rnd_count=1)
        fit!(reg, epochs, pop, Xf, yf; loss_fun="mse", linear_scaling=true, hof=4)
    end

    best = (r2v=-Inf, chrom=nothing)
    t_sel = @elapsed for chrom in reg.best_models_
        pv = eval_batch(chrom, Xv)
        r = r2(yv, pv)
        isfinite(r) && r > best.r2v && (best = (r2v=r, chrom=chrom))
    end
    best.chrom === nothing && return Dict(
        "pde" => cond["pde"], "sigma" => cond["sigma"], "truth" => cond["truth"],
        "r2_test" => nothing, "recovered" => false, "expression" => nothing,
        "time_s" => t_fit + t_sel)

    pt = eval_batch(best.chrom, Xte)
    r2t = r2(rhs, pt)
    # feature names x1..xd -> u, u_x, ... for a readable equation
    expr = equation_string(best.chrom)
    for i in d:-1:1
        expr = replace(expr, "x$i" => names[i])
    end
    return Dict(
        "pde" => cond["pde"], "sigma" => cond["sigma"], "truth" => cond["truth"],
        "r2_test" => isfinite(r2t) ? r2t : nothing,
        "recovered" => isfinite(r2t) && r2t > R2_RECOVERED,
        "expression" => expr, "time_s" => t_fit + t_sel)
end

# ---------------------------------------------------------------------- main -----
function main()
    conds = JSON.parsefile(joinpath(HERE, "data", "conditions.json"))

    # JIT warm-up off the clock, on a tiny slice of the first condition
    let c = copy(conds[1])
        c["X_train"] = c["X_train"][1:200]
        c["y_train"] = c["y_train"][1:200]
        run_condition(c; epochs=2, pop=100)
    end

    results = []
    for cond in conds
        r = run_condition(cond)
        push!(results, r)
        r2t = something(r["r2_test"], -Inf)
        @printf("%-8s sigma=%-5g R2=%8.4f %s %5.1fs  %s\n",
            r["pde"], r["sigma"], max(r2t, -9.9999),
            r["recovered"] ? "OK" : "no", r["time_s"],
            first(something(r["expression"], "-"), 80))
        flush(stdout)
    end

    out = joinpath(HERE, "results", get(ARGS, 1, "gep.json"))
    mkpath(dirname(out))
    open(out, "w") do io
        JSON.print(io, Dict("method" => "GEP.jl", "results" => results), 1)
    end
    println("wrote $out")
end

main()
