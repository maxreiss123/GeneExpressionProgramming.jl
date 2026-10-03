#=
Constitutive-relation discovery from DSMC data (arXiv:2507.01466v1, Sec. 4 / Table 4)
with GeneExpressionProgramming.jl.

Data: the authors' own DSMC lid-driven-cavity fields (Kn = 0.005, lid at 50 m/s and
at 337 m/s), prepared by prepare_dsmc_data.py -- 200 points sampled from the central
20%-80% of the domain, velocity gradients by finite differences (the authors' neural
network gradients are not part of their release).

    target     : sigma_ij                       [kg m^-1 s^-2]
    tensors    : S_ij, delta_ij
    scalars    : D_kk, p, mu
    reference  : sigma_ij = -2 mu S_ij + (2/3) mu D_kk delta_ij + p delta_ij
    paper      : -1.906 mu S_ij + 1.000 p delta_ij                       (50 m/s)
                 -1.914 mu S_ij + 0.676 mu D_kk delta_ij + 1.000 p delta_ij (337 m/s)

Unlike the two benchmark cases, this target is *not* satisfied exactly by the data
(DSMC noise, Kn = 0.005 is not the continuum limit), so there is no convergence
threshold: every run uses the full generation budget and is judged by the terms it
finds and by their coefficients, projected onto the three reference terms.

The constants of the best model are tuned every 5th generation if it has improved
(`optimization_epochs=5`). In the recorded results/dsmc_gep.json that optimiser was
inactive, through a bug fixed since, so a rerun can differ.

Usage
    julia --project=. --threads=auto paper/site_benchmark/dsmc_gep.jl \
        [--cases incompressible,compressible] [--seeds 1,2,3] [--epochs 300]
        [--pop 1200] [--head 8] [--genes 3] [--config dhc|nodhc]
        [--out results/dsmc_gep.json]
=#

t_load = @elapsed using GeneExpressionProgramming

using Random
using Statistics
using Printf
using JSON
using OrderedCollections
using LinearAlgebra

const HERE = @__DIR__

# [kg, m, s, K, mol, A, cd]
const DIM_S = Float16[0, 0, -1, 0, 0, 0, 0]        # 1/s
const DIM_ONE = Float16[0, 0, 0, 0, 0, 0, 0]
const DIM_P = Float16[1, -1, -2, 0, 0, 0, 0]       # Pa
const DIM_MU = Float16[1, -1, -1, 0, 0, 0, 0]      # kg/(m s)
const DIM_TARGET = DIM_P

function read_dsmc(case::String, seed::Int)
    path = joinpath(HERE, "data", "dsmc_$(case)_s$(seed).csv")
    lines = readlines(path)
    n = length(lines) - 1
    x = zeros(Float64, 5, n)
    y = zeros(Float64, n)
    comp = zeros(Int, n)
    for (r, line) in enumerate(lines[2:end])
        f = split(line, ',')
        comp[r] = parse(Int, f[2]) * 2 + parse(Int, f[3]) + 1
        x[1, r] = parse(Float64, f[4])   # S_ij
        x[2, r] = parse(Float64, f[5])   # delta_ij
        x[3, r] = parse(Float64, f[6])   # D_kk
        x[4, r] = parse(Float64, f[7])   # p
        x[5, r] = parse(Float64, f[8])   # mu
        y[r] = parse(Float64, f[9])      # sigma_ij
    end
    return x, y, comp
end

"""SITE's fitness, here over the four components of a 2x2 tensor."""
function make_site_loss(y::Vector{Float64}, comp::Vector{Int})
    groups = [findall(==(c), comp) for c in 1:4]
    norms = [sqrt(sum(abs2, y[g])) for g in groups]
    return function (y_true, y_pred)
        total = 0.0
        @inbounds for c in 1:4
            g = groups[c]
            s = 0.0
            for k in g
                d = y_pred[k] - y_true[k]
                s += d * d
            end
            total += norms[c] < 1e-12 ? sqrt(s) : sqrt(s) / norms[c]
        end
        out = total / 4
        return isfinite(out) ? out : typemax(Float64)
    end
end

function build_regressor(; seed::Int, head_len::Int, gene_count::Int, use_dims::Bool)
    Random.seed!(seed)
    dims = Dict{Symbol,Vector{Float16}}(
        :S => DIM_S, :delta => DIM_ONE, :Dkk => DIM_S, :pres => DIM_P, :mu => DIM_MU,
        Symbol(2.0) => DIM_ONE, Symbol(3.0) => DIM_ONE, Symbol(0.5) => DIM_ONE,
    )
    return GepRegressor(5;
        entered_features=[:S, :delta, :Dkk, :pres, :mu],
        entered_non_terminals=[:+, :-, :*, :/],
        gene_connections=[:+, :-],
        entered_terminal_nums=[Symbol(2.0), Symbol(3.0), Symbol(0.5)],
        considered_dimensions=use_dims ? dims : Dict{Symbol,Vector{Float16}}(),
        rnd_count=2, gene_count=gene_count, head_len=head_len)
end

function pretty(model)
    s = string(model)
    for (k, name) in enumerate(["S_ij", "delta_ij", "D_kk", "p", "mu",
        "2.0", "3.0", "0.5", "rnc1", "rnc2"])
        s = replace(s, Regex("\\bx$(k)\\b") => name)
    end
    return s
end

"""Project a model prediction onto (mu S_ij, mu D_kk delta_ij, p delta_ij)."""
function project(x::Matrix{Float64}, y_pred::Vector{Float64})
    A = hcat(x[5, :] .* x[1, :], x[5, :] .* x[3, :] .* x[2, :], x[4, :] .* x[2, :])
    return A \ y_pred
end

function parse_args(argv)
    o = Dict{String,String}("cases" => "incompressible,compressible", "seeds" => "1,2,3",
        "epochs" => "300", "pop" => "1200", "head" => "8", "genes" => "3",
        "config" => "dhc", "out" => "")
    i = 1
    while i <= length(argv)
        key = argv[i][3:end]
        haskey(o, key) || error("unknown option --$key")
        o[key] = argv[i+1]
        i += 2
    end
    return o
end

function main(o)
    use_dims = o["config"] == "dhc"
    epochs = parse(Int, o["epochs"])
    pop = parse(Int, o["pop"])
    cases = split(o["cases"], ',')
    seeds = [parse(Int, s) for s in split(o["seeds"], ',')]

    x0, y0, c0 = read_dsmc(String(cases[1]), seeds[1])
    t_warm = @elapsed begin
        w = build_regressor(seed=0, head_len=parse(Int, o["head"]),
            gene_count=parse(Int, o["genes"]), use_dims=use_dims)
        l = make_site_loss(y0[1:80], c0[1:80])
        fit!(w, 2, 40, x0[:, 1:80], y0[1:80]; loss_fun=l, loss_fun_validation=l,
            target_dimension=use_dims ? DIM_TARGET : nothing,
            optimization_epochs=5, correction_epochs=1, correction_amount=0.3, hof=1)
    end

    runs = Vector{OrderedDict{String,Any}}()
    for case in cases, seed in seeds
        x, y, comp = read_dsmc(String(case), seed)
        loss = make_site_loss(y, comp)
        reg = build_regressor(seed=seed, head_len=parse(Int, o["head"]),
            gene_count=parse(Int, o["genes"]), use_dims=use_dims)
        n_epochs = Ref(0)
        bc = (population, epoch) -> (n_epochs[] = epoch; false)
        GC.gc()
        t = @elapsed fit!(reg, epochs, pop, x, y; loss_fun=loss, loss_fun_validation=loss,
            target_dimension=use_dims ? DIM_TARGET : nothing,
            optimization_epochs=5,          # coefficients here are not round numbers
            correction_epochs=1, correction_amount=0.3, break_condition=bc, hof=1)

        y_pred = vec(reg(x))
        w = project(x, y_pred)
        push!(runs, OrderedDict{String,Any}(
            "case" => String(case), "seed" => seed, "solve_time_s" => t,
            "epochs_run" => n_epochs[], "loss" => loss(y, y_pred),
            "coeff_muS" => w[1], "coeff_muDkk" => w[2], "coeff_p" => w[3],
            "expression" => pretty(reg.best_models_[1])))
        @printf("%-15s seed %d: loss %.3e  ->  %+.3f mu S_ij %+.3f mu D_kk d_ij %+.3f p d_ij  (%.1fs)\n",
            case, seed, runs[end]["loss"], w[1], w[2], w[3], t)
        println("    ", runs[end]["expression"])
    end

    summary = OrderedDict{String,Any}()
    for case in unique(String.(cases))
        rs = [r for r in runs if r["case"] == case]
        summary[case] = OrderedDict(
            "coeff_muS" => OrderedDict("mean" => mean(r["coeff_muS"] for r in rs),
                "std" => std([r["coeff_muS"] for r in rs])),
            "coeff_muDkk" => OrderedDict("mean" => mean(r["coeff_muDkk"] for r in rs),
                "std" => std([r["coeff_muDkk"] for r in rs])),
            "coeff_p" => OrderedDict("mean" => mean(r["coeff_p"] for r in rs),
                "std" => std([r["coeff_p"] for r in rs])),
            "loss" => mean(r["loss"] for r in rs))
    end

    res = OrderedDict{String,Any}(
        "framework" => "GeneExpressionProgramming.jl", "case" => "dsmc_cavity",
        "config" => o["config"], "population" => pop, "max_epochs" => epochs,
        "threads" => Threads.nthreads(), "load_time_s" => t_load, "warmup_time_s" => t_warm,
        "paper_table4" => OrderedDict(
            "incompressible" => "-1.906 mu S_ij + 1.000 p delta_ij",
            "compressible" => "-1.914 mu S_ij + 0.676 mu D_kk delta_ij + 1.000 p delta_ij"),
        "summary" => summary, "runs" => runs)

    if !isempty(o["out"])
        path = isabspath(o["out"]) ? o["out"] : joinpath(HERE, o["out"])
        mkpath(dirname(path))
        open(path, "w") do f
            JSON.print(f, res)
        end
        println("-> ", path)
    end
end

main(parse_args(ARGS))
