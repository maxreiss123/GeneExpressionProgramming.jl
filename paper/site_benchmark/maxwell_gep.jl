#=
Maxwell-stress-tensor benchmark (Sec. 3.1, Tables 1 and 2) of

    T. Chen, H. Yang, W. Ma, J. Zhang, "Symbolic identification of tensor equations
    in multidimensional physical fields", arXiv:2507.01466v1 (2025)  --  SITE

with the scalar `GepRegressor` of this package.

The data are the paper's 150 samples (export_site_data.py mirrors the authors' generator,
RNG seed included). SITE encodes the target as a tensor expression (host chromosome) with
embedded scalar expressions (plasmids); here the tensor problem is handed over
component-stacked: every (sample, i, j) triple is one scalar row, and the terminals are
the corresponding components of the candidate tensors.

    terminals (tensor components) : EE = E_i E_j , BB = B_i B_j , delta = delta_ij
    terminals (scalars)           : E2 = E_k E_k , B2 = B_k B_k , eps_0 , mu_0
    target                        : T_ij     [kg m^-1 s^-2]

With --config dhc the terminals carry SI dimensions and the target dimension is set, so
the search repairs individuals by semantic backpropagation (SBP) and scores only
homogeneous ones. The loss is a port of SITE's `loss_func` (SITE.py): the mean over the
nine tensor components of the relative L2 error, so the convergence threshold 1e-6 means
the same in both frameworks.

Usage
    julia --project=. --threads=auto paper/site_benchmark/maxwell_gep.jl \
        [--data clean|noise005|noise010|noise020] [--config dhc|nodhc] [--seed 1]
        [--epochs 500] [--pop 1600] [--head 8] [--genes 4] [--scaling false]
        [--constopt false] [--corr 0.3] [--half true] [--sampling 1] [--warmup true]
        [--out results/foo.json]

--scaling fits one least-squares coefficient per gene (`linear_scaling`), --corr is the
fraction of the population SBP may repair per generation, --half offers 0.5 as a
terminal, --sampling is `population_sampling_multiplier`; --buffered is accepted and
ignored.
=#

t_load = @elapsed using GeneExpressionProgramming

using Random
using Statistics
using Printf
using JSON
using OrderedCollections

const HERE = @__DIR__
const TOL = 1e-6                      # SITE: tol = 1e-6
const FEATURE_NAMES = ["EE", "BB", "delta", "E2", "B2"]

# SI base dimensions used by this package: [kg, m, s, K, mol, A, cd]
const DIM_EE = Float16[2, 2, -6, 0, 0, -2, 0]      # (V/m)^2
const DIM_BB = Float16[2, 0, -4, 0, 0, -2, 0]      # T^2
const DIM_ONE = Float16[0, 0, 0, 0, 0, 0, 0]
const DIM_EPS0 = Float16[-1, -3, 4, 0, 0, 2, 0]    # F/m
const DIM_MU0 = Float16[1, 1, -2, 0, 0, -2, 0]     # H/m
const DIM_TARGET = Float16[1, -1, -2, 0, 0, 0, 0]  # Pa

const EPS0 = 8.854e-12
const MU0 = 4pi * 1e-7

# --------------------------------------------------------------------- data --
"""Read a component-stacked CSV written by export_site_data.py."""
function read_maxwell(tag::String)
    path = joinpath(HERE, "data", "maxwell_$(tag).csv")
    lines = readlines(path)
    n = length(lines) - 1
    x = zeros(Float64, 5, n)          # features are column-major samples
    y = zeros(Float64, n)
    comp = zeros(Int, n)              # 1..9 tensor component of each row
    for (r, line) in enumerate(lines[2:end])
        f = split(line, ',')
        i = parse(Int, f[2]); j = parse(Int, f[3])
        comp[r] = i * 3 + j + 1
        x[1, r] = parse(Float64, f[4])   # EE
        x[2, r] = parse(Float64, f[5])   # BB
        x[3, r] = parse(Float64, f[6])   # delta
        x[4, r] = parse(Float64, f[7])   # E2
        x[5, r] = parse(Float64, f[8])   # B2
        y[r] = parse(Float64, f[9])      # T
    end
    return x, y, comp
end

# --------------------------------------------------------------------- loss --
"""
SITE's fitness: mean over the nine tensor components of ||y_pred - y||_2 / ||y||_2
(see `loss_func` in SITE.py), with the absolute error for a component whose target is
zero. `comp` maps every stacked row to its component.
"""
function make_site_loss(y::Vector{Float64}, comp::Vector{Int})
    groups = [findall(==(c), comp) for c in 1:9]
    norms = [sqrt(sum(abs2, y[g])) for g in groups]
    return function site_loss(y_true, y_pred)
        total = 0.0
        @inbounds for c in 1:9
            g = groups[c]
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

# ---------------------------------------------------------------- regressor --
function build_regressor(; seed::Int, head_len::Int, gene_count::Int, use_dims::Bool,
    with_half::Bool=true)
    Random.seed!(seed)
    dims = Dict{Symbol,Vector{Float16}}(
        :EE => DIM_EE, :BB => DIM_BB, :delta => DIM_ONE,
        :E2 => DIM_EE, :B2 => DIM_BB,
        Symbol(EPS0) => DIM_EPS0, Symbol(MU0) => DIM_MU0,
        Symbol(0.5) => DIM_ONE,
    )
    # eps_0 and mu_0 enter as dimensioned symbolic constants -- the counterpart of
    # SITE's plasmid symbol terminals.  0.5 is a plain dimensionless constant; SITE
    # has to build the factor 1/2 either from an RNC or from its linear regression.
    terminals = with_half ? [Symbol(EPS0), Symbol(MU0), Symbol(0.5)] :
                [Symbol(EPS0), Symbol(MU0)]
    return GepRegressor(5;
        entered_features=[:EE, :BB, :delta, :E2, :B2],
        entered_non_terminals=[:+, :-, :*, :/],
        gene_connections=[:+, :-],
        entered_terminal_nums=terminals,
        considered_dimensions=use_dims ? dims : Dict{Symbol,Vector{Float16}}(),
        rnd_count=1,
        gene_count=gene_count,
        head_len=head_len)
end

"""Human readable form of the winning expression."""
function pretty(model; with_half::Bool=true)
    s = string(model)
    # placeholders x1, x2, ... number the terminals in registration order; the current
    # printer already writes the feature names, so this only touches older output
    names = vcat(FEATURE_NAMES, ["eps_0", "mu_0"], with_half ? ["0.5"] : String[], ["rnc"])
    for (k, name) in enumerate(names)
        s = replace(s, Regex("\\bx$(k)\\b") => name)
    end
    return s
end

# --------------------------------------------------------------------- main --
function parse_args(argv)
    o = Dict{String,String}("data" => "clean", "config" => "dhc", "seed" => "1",
        "epochs" => "500", "pop" => "1600", "head" => "8", "genes" => "4",
        "out" => "", "warmup" => "true", "constopt" => "false", "half" => "true",
        "corr" => "0.3", "sampling" => "1", "scaling" => "false", "buffered" => "auto")
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("unexpected argument $a")
        key = a[3:end]
        haskey(o, key) || error("unknown option --$key")
        o[key] = argv[i+1]
        i += 2
    end
    return o
end

function run_case(o)
    tag = o["data"]
    use_dims = o["config"] == "dhc"
    seed = parse(Int, o["seed"])
    epochs = parse(Int, o["epochs"])
    pop = parse(Int, o["pop"])

    x, y, comp = read_maxwell(tag)
    loss = make_site_loss(y, comp)

    # warm-up: compile the whole fit! path, so the timed run below excludes JIT time
    t_warm = 0.0
    if o["warmup"] == "true"
        t_warm = @elapsed begin
            w = build_regressor(seed=seed, head_len=parse(Int, o["head"]),
                gene_count=parse(Int, o["genes"]), use_dims=use_dims,
                with_half=o["half"] == "true")
            fit!(w, 2, 40, x[:, 1:90], y[1:90];
                loss_fun=make_site_loss(y[1:90], comp[1:90]),
                loss_fun_validation=make_site_loss(y[1:90], comp[1:90]),
                target_dimension=use_dims ? DIM_TARGET : nothing,
                optimization_epochs=10_000, correction_epochs=1, correction_amount=0.1,
                hof=1)
        end
    end

    regressor = build_regressor(seed=seed, head_len=parse(Int, o["head"]),
        gene_count=parse(Int, o["genes"]), use_dims=use_dims,
        with_half=o["half"] == "true")

    epoch_loss = Float64[]
    epoch_time = Float64[]
    t0 = time_ns()
    converged_epoch = -1
    break_cond = function (population, epoch)
        best = mean(population[1].fitness)
        push!(epoch_loss, best)
        push!(epoch_time, (time_ns() - t0) / 1e9)
        if best < TOL && converged_epoch < 0
            converged_epoch = epoch
            return true
        end
        return false
    end

    # --constopt true tunes the best model's constants every 10th generation if it has
    # improved, the counterpart of SITE's RNC mutation; not with --scaling true. Off (the
    # default, and in every recorded run), eps_0 and mu_0 keep their physical values.
    const_opt = o["constopt"] == "true"

    GC.gc()
    t_solve = @elapsed fit!(regressor, epochs, pop, x, y;
        loss_fun=loss, loss_fun_validation=loss,
        target_dimension=use_dims ? DIM_TARGET : nothing,
        optimization_epochs=const_opt ? 10 : typemax(Int),
        correction_epochs=1, correction_amount=parse(Float64, o["corr"]),
        break_condition=break_cond,
        population_sampling_multiplier=parse(Int, o["sampling"]),
        linear_scaling=o["scaling"] == "true",
        buffered=o["buffered"] == "auto" ? :auto : o["buffered"] == "true",
        hof=3)

    best = regressor.best_models_[1]
    expr = pretty(best; with_half=o["half"] == "true")
    # whether the winner meets the target as it is scored (gene by gene when scaled);
    # `dimensional_check` below only names the configuration
    winner_homogeneous = !use_dims ? nothing : best.dimension_homogene && (o["scaling"] == "true" ?
        is_gene_wise_homogeneous(best.expression_raw, DIM_TARGET, regressor.token_dto_,
            parse(Int, o["genes"])) :
        is_dimensionally_homogeneous(best.expression_raw, DIM_TARGET, regressor.token_dto_))
    y_pred = vec(regressor(x))
    final_loss = loss(y, y_pred)

    # prediction on the noise-free data, which summarize.py projects onto the four
    # ground-truth tensor terms for the coefficient errors of the paper's Table 2
    x_clean, y_clean, comp_clean = read_maxwell("clean")
    y_pred_clean = vec(regressor(x_clean))
    clean_loss = make_site_loss(y_clean, comp_clean)(y_clean, y_pred_clean)

    res = OrderedDict{String,Any}(
        "framework" => "GeneExpressionProgramming.jl",
        "case" => "maxwell",
        "data" => tag,
        "config" => o["config"],
        "linear_scaling" => o["scaling"] == "true",
        "buffered" => o["buffered"],
        "dimensional_check" => use_dims,
        "winner_homogeneous" => winner_homogeneous,
        "constant_optimisation" => const_opt,
        "half_terminal" => o["half"] == "true",
        "seed" => seed,
        "population" => pop,
        "max_epochs" => epochs,
        "head_len" => parse(Int, o["head"]),
        "correction_amount" => parse(Float64, o["corr"]),
        "population_sampling_multiplier" => parse(Int, o["sampling"]),
        "gene_count" => parse(Int, o["genes"]),
        "threads" => Threads.nthreads(),
        "load_time_s" => t_load,
        "warmup_time_s" => t_warm,
        "solve_time_s" => t_solve,           # <- excludes precompilation and JIT
        "epochs_run" => length(epoch_loss),
        "converged_epoch" => converged_epoch,
        "converged" => converged_epoch > 0,
        "final_loss" => final_loss,
        "clean_data_loss" => clean_loss,
        "y_pred_clean" => y_pred_clean,
        "best_loss" => minimum(epoch_loss),
        "expression" => expr,
        "epoch_loss" => epoch_loss,
        "epoch_time_s" => epoch_time,
    )

    @printf("\n[%s | %s | seed %d]  loss=%.3e  epochs=%d  solve=%.1fs (load %.1fs, warmup %.1fs)\n",
        tag, o["config"], seed, final_loss, length(epoch_loss), t_solve, t_load, t_warm)
    println("  expression: ", expr)

    if !isempty(o["out"])
        path = isabspath(o["out"]) ? o["out"] : joinpath(HERE, o["out"])
        mkpath(dirname(path))
        open(path, "w") do f
            JSON.print(f, res)
        end
        println("  -> ", path)
    end
    return res
end

run_case(parse_args(ARGS))
