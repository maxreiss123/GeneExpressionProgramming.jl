#=
Unit-constrained vectorised route: PDE discovery with dimensional homogeneity.

Same conditions, split and metric as pde_gep_tensor.jl, plus physical dimensions. The
fields get real ones -- u [m/s], x [m], t [s] -- under which the textbook forms are
inhomogeneous, so three unit-carrying constants enter as constant-valued feature columns:

    nu2 [m^2/s]     nu3 [m^3/s]     nu4 [m^4/s]

Dimension vectors on the tensor path are [tensor order, kg, m, s, K, mol, A, cd]. Every
terminal carries 1/s, so against the target u_t [m/s^2] the admissible monomials have two
factors, and they are exactly

    u*u_x,   nu2*u_xx,   nu3*u_xxx,   nu4*u_xxxx

The cross term u*u_xxx (dimension 1/(m s^2)), which PDE-FIND's KS fit at sigma = 0.01
includes, cannot be formed. With a target dimension the search repairs individuals by
semantic backpropagation (`correct_genes!`) and scores only homogeneous ones. With `+`
and `-` as the only connectors, that puts the target dimension on every gene, which keeps
the gene-wise least-squares weights dimensionless.

    julia --project=. --threads=4 paper/pdebench/pde_gep_units.jl [out.json]

The results go to results/gep_units.json unless another file name is given, e.g.
gep_units_1thread.json for the single-thread timing in RESULTS.md.
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

# dimension vectors: [order, kg, m, s, K, mol, A, cd]
dimvec(m_, s_) = Float16[0, 0, m_, s_, 0, 0, 0, 0]
const DIM_TARGET = dimvec(1, -2)                      # u_t = m/s^2
const CONST_NAMES = ["nu2", "nu3", "nu4"]
const CONST_DIMS = [dimvec(2, -1), dimvec(3, -1), dimvec(4, -1)]

"""Dimensions of [u, u_x, ..., u_x^m]: u is m/s, each d/dx removes one metre."""
deriv_dims(max_order) = [dimvec(1 - o, -1) for o in 0:max_order]

columns_dict(X::Matrix{Float64}) =
    Dict{Int8,Any}(Int8(i) => X[i, :] for i in 1:size(X, 1))

"""Append the unit-carrying constants as all-ones columns: their numeric value is
structural (the gene-wise weights carry magnitude), their dimension is the point."""
function with_consts(X::Matrix{Float64})
    n = size(X, 2)
    return vcat(X, ones(length(CONST_NAMES), n))
end

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
    base_names = String.(cond["names"])
    names = vcat(base_names, CONST_NAMES)
    d = length(names)
    Xtr = with_consts(reduce(hcat, (Float64.(r) for r in cond["X_train"])))
    ytr = Float64.(cond["y_train"])
    Xte = with_consts(reduce(hcat, (Float64.(r) for r in cond["X_test"])))
    rhs = Float64.(cond["rhs_test"])

    dims = Dict{Symbol,Vector{Float16}}()
    dd = deriv_dims(length(base_names) - 1)
    for (i, v) in enumerate(dd)
        dims[Symbol("x$i")] = v
    end
    for (k, v) in enumerate(CONST_DIMS)
        dims[Symbol("x$(length(base_names)+k)")] = v
    end

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
        feature_names=names,
        considered_dimensions=dims)
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

    t_fit = @elapsed fit!(reg, epochs, pop, loss; hof=4,
        target_dimension=DIM_TARGET, correction_epochs=1)

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

    out = joinpath(HERE, "results", get(ARGS, 1, "gep_units.json"))
    mkpath(dirname(out))
    open(out, "w") do io
        JSON.print(io, Dict("method" => "GEP.jl (vector+units)", "results" => results), 1)
    end
    println("wrote $out")
end

main()
