#=
Vectorised route: the PDE-discovery benchmark through GepTensorRegressor.

Same conditions file, split and metric as pde_gep.jl; what changes is the regression
engine. Each derivative feature is one column over all training samples, the fitness is a
loss callback over those columns (the interface examples/Main_streaming_chunks.jl uses),
and the coefficients are not searched at all: `predictT_scaled` solves one least-squares
weight per gene (the tensor path's gene-wise scaling), so evolution searches structure
only.

    u_t(column) = sum_j  w_j * gene_j(u, u_x, ...)     w = argmin ||u_t - G w||

This is the machinery that fits Vec/Tensor-valued equations (the Maxwell case of
paper/site_benchmark); scalar PDE fields are its 1-dimensional special case, run here so
the vectorised route has a measured row next to the pointwise one.

    julia --project=. --threads=4 paper/pdebench/pde_gep_tensor.jl [out.json]

The results go to results/gep_tensor.json unless another file name is given, e.g.
gep_tensor_1thread.json for the single-thread timing in RESULTS.md.
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
const E = GeneExpressionProgramming.GepEntities

# ------------------------------------------------------------------ helpers ------
columns_dict(X::Matrix{Float64}) =
    Dict{Int8,Any}(Int8(i) => X[i, :] for i in 1:size(X, 1))

"""Evaluate a chromosome's genes on `cols` and combine them with the scaling
weights fitted during the search. Mirrors `predictT_scaled`'s usable-gene filter,
so the weight vector lines up gene for gene; returns nothing on any mismatch."""
function eval_scaled(reg, chrom::Chromosome, cols::Dict{Int8,Any}, n::Int)
    isnothing(chrom.scaling_weights) && return nothing
    raw = E._karva_raw(chrom; split=true)
    bases = [calc_stack_batch_tensor(collect(raw[j]), reg.toolbox_.callbacks, cols, nothing)
             for j in 2:length(raw)]
    usable = [b for b in bases if b isa AbstractVector && length(b) == n &&
              eltype(b) <: Float64]
    length(usable) == length(chrom.scaling_weights) || return nothing
    out = zeros(n)
    for (j, b) in enumerate(usable)
        out .+= chrom.scaling_weights[j] .* b
    end
    return out
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
    Xtr = reduce(hcat, (Float64.(r) for r in cond["X_train"]))    # d x n
    ytr = Float64.(cond["y_train"])
    Xte = reduce(hcat, (Float64.(r) for r in cond["X_test"]))
    rhs = Float64.(cond["rhs_test"])

    ntr = floor(Int, 0.8 * length(ytr))
    yf = ytr[1:ntr]
    cols_fit = columns_dict(Xtr[:, 1:ntr])
    cols_val = columns_dict(Xtr[:, ntr+1:end])
    yv = ytr[ntr+1:end]
    cols_test = columns_dict(Xte)

    Random.seed!(seed)
    reg = GepTensorRegressor(d;
        problem_dimension=1,
        gene_count=4,
        head_len=5,
        entered_non_terminals=[:+, :-, :*],
        gene_connections=[:+, :-],
        feature_names=names)
    reg.input_values = cols_fit

    # the loss runs inside the threaded fitness loop: one prediction buffer per thread
    preds = [similar(yf) for _ in 1:thread_slots()]
    function loss(elem, validate::Bool)
        if isnan(mean(elem.fitness)) || validate
            pred = try
                predictT_scaled!(preds[Threads.threadid()], reg, elem, yf)
            catch
                nothing
            end
            if pred isa AbstractVector && allfinite(pred)
                pred .-= yf
                elem.fitness = (sum(abs2, pred) / length(yf),)
            else
                elem.fitness = (1e6,)
            end
        end
    end

    t_fit = @elapsed fit!(reg, epochs, pop, loss; hof=4)

    best = (r2v=-Inf, chrom=nothing)
    t_sel = @elapsed for chrom in reg.best_models_
        pv = eval_scaled(reg, chrom, cols_val, length(yv))
        pv === nothing && continue
        r = r2(yv, pv)
        isfinite(r) && r > best.r2v && (best = (r2v=r, chrom=chrom))
    end
    best.chrom === nothing && return Dict(
        "pde" => cond["pde"], "sigma" => cond["sigma"], "truth" => cond["truth"],
        "r2_test" => nothing, "recovered" => false, "expression" => nothing,
        "time_s" => t_fit + t_sel)

    pt = eval_scaled(reg, best.chrom, cols_test, length(rhs))
    r2t = pt === nothing ? -Inf : r2(rhs, pt)
    return Dict(
        "pde" => cond["pde"], "sigma" => cond["sigma"], "truth" => cond["truth"],
        "r2_test" => isfinite(r2t) ? r2t : nothing,
        "recovered" => isfinite(r2t) && r2t > R2_RECOVERED,
        "expression" => equation_string(best.chrom),
        "time_s" => t_fit + t_sel)
end

# ---------------------------------------------------------------------- main -----
function main()
    conds = JSON.parsefile(joinpath(HERE, "data", "conditions.json"))

    # JIT warm-up off the clock
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

    out = joinpath(HERE, "results", get(ARGS, 1, "gep_tensor.json"))
    mkpath(dirname(out))
    open(out, "w") do io
        JSON.print(io, Dict("method" => "GEP.jl (vector)", "results" => results), 1)
    end
    println("wrote $out")
end

main()
