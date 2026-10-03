#=
Reynolds-stress-transport benchmark of arXiv:2507.01466v1 (SITE), Sec. 3.2 / Table 3,
run with GeneExpressionProgramming.jl.

    dR_ij/dt = -(2/3) eps delta_ij

in decaying homogeneous isotropic turbulence. R_ij and k are distractors: offered as
terminals, but not part of the target equation.

Data: the authors' repository does not ship the OpenFOAM dataset of Sec. 3.2
(`data/processed_data.mat` is missing), so `export_site_data.py` builds a surrogate from
the paper's description (k0 = 4265.9 m^2/s^2, 100 samples in t = [1e-8, 1e-6] s, dR/dt by
finite differences). Absolute coefficients are therefore only qualitatively comparable
to Table 3; the behaviour under sub-sampling, which Table 3 studies, is comparable.

The identification is repeated for dataset sizes 100/75/50/25 with `--repeats`
sub-sampling seeds, reporting, as Table 3 does, the mean and standard deviation of the
identified coefficient: the least-squares projection of the model's prediction on the
full dataset onto eps*delta_ij.

Usage
    julia --project=. --threads=auto paper/site_benchmark/reynolds_gep.jl \
        [--repeats 25] [--epochs 300] [--pop 800] [--head 6] [--genes 3]
        [--config dhc|nodhc] [--out results/reynolds_gep.json]
=#

t_load = @elapsed using GeneExpressionProgramming

using Random
using Statistics
using Printf
using JSON
using OrderedCollections

const HERE = @__DIR__
const TOL = 1e-6

# [kg, m, s, K, mol, A, cd]
const DIM_R = Float16[0, 2, -2, 0, 0, 0, 0]        # m^2/s^2
const DIM_ONE = Float16[0, 0, 0, 0, 0, 0, 0]
const DIM_EPS = Float16[0, 2, -3, 0, 0, 0, 0]      # m^2/s^3
const DIM_TARGET = DIM_EPS                          # dR/dt

function read_reynolds()
    path = joinpath(HERE, "data", "reynolds_dhit.csv")
    lines = readlines(path)
    n = length(lines) - 1
    x = zeros(Float64, 4, n)
    y = zeros(Float64, n)
    comp = zeros(Int, n)
    sample = zeros(Int, n)
    for (r, line) in enumerate(lines[2:end])
        f = split(line, ',')
        sample[r] = parse(Int, f[1])
        comp[r] = parse(Int, f[2]) * 3 + parse(Int, f[3]) + 1
        x[1, r] = parse(Float64, f[4])   # R_ij
        x[2, r] = parse(Float64, f[5])   # delta_ij
        x[3, r] = parse(Float64, f[6])   # k
        x[4, r] = parse(Float64, f[7])   # epsilon
        y[r] = parse(Float64, f[8])      # dR_ij/dt
    end
    return x, y, comp, sample
end

"""SITE's fitness (mean relative L2 error over the tensor components)."""
function make_site_loss(y::Vector{Float64}, comp::Vector{Int})
    groups = [findall(==(c), comp) for c in 1:9]
    norms = [sqrt(sum(abs2, y[g])) for g in groups]
    return function (y_true, y_pred)
        total = 0.0
        @inbounds for c in 1:9
            g = groups[c]
            isempty(g) && continue
            s = 0.0
            for k in g
                d = y_pred[k] - y_true[k]
                s += d * d
            end
            total += norms[c] < 1e-12 ? sqrt(s) : sqrt(s) / norms[c]
        end
        out = total / 9
        return isfinite(out) ? out : typemax(Float64)
    end
end

function build_regressor(; seed::Int, head_len::Int, gene_count::Int, use_dims::Bool)
    Random.seed!(seed)
    dims = Dict{Symbol,Vector{Float16}}(
        :R => DIM_R, :delta => DIM_ONE, :k => DIM_R, :eps => DIM_EPS,
        Symbol(0.5) => DIM_ONE, Symbol(2.0) => DIM_ONE, Symbol(3.0) => DIM_ONE,
    )
    return GepRegressor(4;
        entered_features=[:R, :delta, :k, :eps],
        entered_non_terminals=[:+, :-, :*, :/],
        gene_connections=[:+, :-],
        entered_terminal_nums=[Symbol(2.0), Symbol(3.0), Symbol(0.5)],
        considered_dimensions=use_dims ? dims : Dict{Symbol,Vector{Float16}}(),
        rnd_count=1, gene_count=gene_count, head_len=head_len)
end

function pretty(model)
    s = string(model)
    for (k, name) in enumerate(["R", "delta", "k", "eps", "2.0", "3.0", "0.5", "rnc"])
        s = replace(s, Regex("\\bx$(k)\\b") => name)
    end
    return s
end

"""Least-squares projection of a prediction onto the ground-truth term eps*delta."""
function effective_coefficient(x::Matrix{Float64}, y_pred::Vector{Float64})
    basis = x[4, :] .* x[2, :]                # eps * delta
    return dot(basis, y_pred) / dot(basis, basis)
end

using LinearAlgebra: dot

function parse_args(argv)
    o = Dict{String,String}("repeats" => "25", "epochs" => "300", "pop" => "800",
        "head" => "6", "genes" => "3", "config" => "dhc", "out" => "")
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
    x_all, y_all, comp_all, sample_all = read_reynolds()
    n_samples = maximum(sample_all) + 1
    use_dims = o["config"] == "dhc"
    epochs = parse(Int, o["epochs"])
    pop = parse(Int, o["pop"])
    repeats = parse(Int, o["repeats"])

    # warm-up (JIT) so that the reported solve times are compilation free
    t_warm = @elapsed begin
        w = build_regressor(seed=0, head_len=parse(Int, o["head"]),
            gene_count=parse(Int, o["genes"]), use_dims=use_dims)
        l = make_site_loss(y_all[1:90], comp_all[1:90])
        fit!(w, 2, 40, x_all[:, 1:90], y_all[1:90]; loss_fun=l, loss_fun_validation=l,
            target_dimension=use_dims ? DIM_TARGET : nothing,
            optimization_epochs=typemax(Int), correction_epochs=1, correction_amount=0.3, hof=1)
    end

    runs = Vector{OrderedDict{String,Any}}()
    for size in (100, 75, 50, 25)
        for rep in 1:repeats
            rng = MersenneTwister(1000 * size + rep)
            keep = size == n_samples ? collect(0:n_samples-1) :
                   sort(randperm(rng, n_samples)[1:size] .- 1)
            mask = [s in Set(keep) for s in sample_all]
            x = x_all[:, mask]
            y = y_all[mask]
            comp = comp_all[mask]
            loss = make_site_loss(y, comp)

            reg = build_regressor(seed=rep, head_len=parse(Int, o["head"]),
                gene_count=parse(Int, o["genes"]), use_dims=use_dims)
            conv = -1
            n_epochs = Ref(0)
            bc = function (population, epoch)
                n_epochs[] = epoch
                if mean(population[1].fitness) < TOL && conv < 0
                    conv = epoch
                    return true
                end
                return false
            end
            GC.gc()
            t = @elapsed fit!(reg, epochs, pop, x, y; loss_fun=loss, loss_fun_validation=loss,
                target_dimension=use_dims ? DIM_TARGET : nothing,
                optimization_epochs=typemax(Int), correction_epochs=1,
                correction_amount=0.3, break_condition=bc, hof=1)

            y_pred_full = vec(reg(x_all))
            coeff = effective_coefficient(x_all, y_pred_full)
            full_loss = make_site_loss(y_all, comp_all)(y_all, y_pred_full)
            push!(runs, OrderedDict{String,Any}(
                "dataset_size" => size, "repeat" => rep, "solve_time_s" => t,
                "epochs_run" => n_epochs[], "converged_epoch" => conv,
                "train_loss" => mean(reg.best_models_[1].fitness),
                "full_data_loss" => full_loss,
                "coefficient" => coeff, "expression" => pretty(reg.best_models_[1])))
            @printf("size %3d rep %2d: coeff %+.4f  loss %.2e  epochs %3d  %.1fs\n",
                size, rep, coeff, full_loss, n_epochs[], t)
        end
    end

    summary = OrderedDict{String,Any}()
    for size in (100, 75, 50, 25)
        cs = [r["coefficient"] for r in runs if r["dataset_size"] == size]
        summary[string(size)] = OrderedDict("mean" => mean(cs), "std" => std(cs),
            "n" => length(cs))
    end

    res = OrderedDict{String,Any}(
        "framework" => "GeneExpressionProgramming.jl", "case" => "reynolds",
        "data" => "reconstructed decaying-HIT surrogate (paper data not published)",
        "config" => o["config"], "population" => pop, "max_epochs" => epochs,
        "threads" => Threads.nthreads(), "load_time_s" => t_load, "warmup_time_s" => t_warm,
        "reference_coefficient" => -2 / 3, "summary" => summary, "runs" => runs)

    println("\nsize   mean coefficient   std")
    for size in (100, 75, 50, 25)
        s = summary[string(size)]
        @printf("%4d   %+.4f          %.1e\n", size, s["mean"], s["std"])
    end

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
