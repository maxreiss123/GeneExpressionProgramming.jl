#=
Maxwell benchmark (arXiv:2507.01466v1, Sec. 3.1) through the tensor-native
`GepTensorRegressor`, which evaluates whole batches of tensors into preallocated
per-thread buffers (`allocate_buffers!` / `predictT`) and carries the dimension check into
the tensor operators (`considered_dimensions` + `target_dimension`). A tensor chromosome
under that check is this package's closest counterpart to SITE's host/plasmid encoding.

The file name is historical: SHADOW_ROOT (default: this checkout) selects the source tree
whose src/GeneExpressionProgramming.jl is loaded.

    terminals (scalars) : E_kE_k , B_kB_k , eps_0 , mu_0
    terminals (tensors) : E_iE_j , B_iB_j , delta_ij
    target              : T_ij   [kg m^-1 s^-2]

Loss (mean relative L2 error over the nine tensor components) and convergence threshold
(1e-6) are SITE's, so generations and wall-clock are comparable. --scaling true fits one
least-squares coefficient per gene (`predictT_scaled`); --units true carries SI units
next to the tensor order (see below).

Usage
    julia --project=. --threads=auto paper/site_benchmark/maxwell_tensor_shadow.jl \
        [--data clean|noise005|noise010|noise020] [--config dhc|nodhc] [--seed 1]
        [--epochs 2000] [--pop 1600] [--head 6] [--genes 4] [--corr 0.3]
        [--scaling false] [--units false] [--out results/shadow_dhc_s1.json]
=#

const SHADOW_ROOT = get(ENV, "SHADOW_ROOT", normpath(joinpath(@__DIR__, "..", "..")))

t_load = @elapsed begin
    include(joinpath(SHADOW_ROOT, "src", "GeneExpressionProgramming.jl"))
end
using .GeneExpressionProgramming

using Random
using Statistics
using Printf
using JSON
using OrderedCollections
using Tensors
using LinearAlgebra

const HERE = @__DIR__
const TOL = 1e-6
const EPS0 = 8.854e-12
const MU0 = 4pi * 1e-7

# Dimension vectors on the tensor path: slot 1 is the tensor order, which the tensor unit
# handlers in Sbp.jl read as such (a contraction gives u1[1] + u2[1] - 2; a product needs
# one operand of order 0). By default only the order is set: scalars 0, second-order
# tensors and the target 2 -- the constraint SITE gets from its host/plasmid split, where
# a scalar can only enter through a plasmid.
const DIM_SCALAR = Float16[0, 0, 0, 0, 0, 0, 0]
const DIM_TENSOR2 = Float16[2, 0, 0, 0, 0, 0, 0]
const DIM_TARGET = DIM_TENSOR2

# --units true swaps them for the physical ones: [tensor order, kg, m, s, K, mol, A, cd],
# the scalar path's SI exponents with the order in front, both carried in one vector.
#
#   E = kg m /(s^3 A)  ->  E^2 = kg^2 m^2 /(s^6 A^2)
#   B = kg /(s^2 A)    ->  B^2 = kg^2 /(s^4 A^2)
#   eps_0 = A^2 s^4 /(kg m^3),  mu_0 = kg m /(s^2 A^2),  delta_ij dimensionless
#   T_ij  = kg /(m s^2)
#
# The target is then reached through eps_0 E_iE_j, B_iB_j / mu_0 and their delta_ij
# traces (up to dimensionless factors), while e.g. eps_0 B_iB_j is inadmissible -- a far
# tighter constraint than the order alone.
u(order, kg, m, sec, A) = Float16[order, kg, m, sec, 0, 0, A, 0]
const U_E2 = u(0, 2, 2, -6, -2)
const U_B2 = u(0, 2, 0, -4, -2)
const U_EPS0 = u(0, -1, -3, 4, 2)
const U_MU0 = u(0, 1, 1, -2, -2)
const U_EE = u(2, 2, 2, -6, -2)
const U_BB = u(2, 2, 0, -4, -2)
const U_DELTA = u(2, 0, 0, 0, 0)
const U_TARGET = u(2, 1, -1, -2, 0)

const FEATURE_NAMES = ["E2", "B2", "eps_0", "mu_0", "EE", "BB", "delta"]

"""Read the component-stacked CSV back into per-sample tensors."""
function read_tensors(tag::String)
    lines = readlines(joinpath(HERE, "data", "maxwell_$(tag).csv"))
    n = (length(lines) - 1) ÷ 9
    EE = [zeros(3, 3) for _ in 1:n]
    BB = [zeros(3, 3) for _ in 1:n]
    T = [zeros(3, 3) for _ in 1:n]
    E2 = zeros(n)
    B2 = zeros(n)
    for line in lines[2:end]
        f = split(line, ',')
        p = parse(Int, f[1]) + 1
        i = parse(Int, f[2]) + 1
        j = parse(Int, f[3]) + 1
        EE[p][i, j] = parse(Float64, f[4])
        BB[p][i, j] = parse(Float64, f[5])
        E2[p] = parse(Float64, f[7])
        B2[p] = parse(Float64, f[8])
        T[p][i, j] = parse(Float64, f[9])
    end
    delta = [Tensor{2,3}(Matrix{Float64}(I, 3, 3)) for _ in 1:n]
    x = Any[E2, B2, fill(EPS0, n), fill(MU0, n),
        [Tensor{2,3}(EE[p]) for p in 1:n],
        [Tensor{2,3}(BB[p]) for p in 1:n],
        delta]
    y = [Tensor{2,3}(T[p]) for p in 1:n]
    return x, y
end

"""SITE's fitness for a batch of predicted second-order tensors."""
function make_tensor_loss(targets::Vector{<:Tensor{2,3}})
    n = length(targets)
    norms = [sqrt(sum(t[i, j]^2 for t in targets)) for i in 1:3, j in 1:3]
    return function (preds)
        total = 0.0
        @inbounds for i in 1:3, j in 1:3
            s = 0.0
            for p in 1:n
                d = preds[p][i, j] - targets[p][i, j]
                s += d * d
            end
            total += norms[i, j] < 1e-12 ? sqrt(s) : sqrt(s) / norms[i, j]
        end
        return total / 9
    end
end

function build(; seed::Int, use_dims::Bool, gene_count::Int, head_len::Int,
    with_units::Bool=false)
    Random.seed!(seed)
    dims = with_units ? Dict{Symbol,Vector{Float16}}(
        :x1 => U_E2, :x2 => U_B2, :x3 => U_EPS0, :x4 => U_MU0,
        :x5 => U_EE, :x6 => U_BB, :x7 => U_DELTA,
    ) : Dict{Symbol,Vector{Float16}}(
        :x1 => DIM_SCALAR,    # E_kE_k
        :x2 => DIM_SCALAR,    # B_kB_k
        :x3 => DIM_SCALAR,    # eps_0
        :x4 => DIM_SCALAR,    # mu_0
        :x5 => DIM_TENSOR2,   # E_iE_j
        :x6 => DIM_TENSOR2,   # B_iB_j
        :x7 => DIM_TENSOR2,   # delta_ij
    )
    return GepTensorRegressor(7;
        problem_dimension=3,
        entered_non_terminals=[:+, :-, :*, :/],
        entered_terminal_nums=[0.5],
        gene_connections=[:+, :-],
        gene_count=gene_count,
        head_len=head_len,
        feature_names=FEATURE_NAMES,
        considered_dimensions=use_dims ? dims : Dict{Symbol,Vector{Float16}}())
end

function parse_args(argv)
    o = Dict{String,String}("data" => "clean", "config" => "dhc", "seed" => "1",
        "epochs" => "2000", "pop" => "1600", "head" => "6", "genes" => "4",
        "corr" => "0.3", "scaling" => "false", "units" => "false", "out" => "")
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
    tag = o["data"]
    use_dims = o["config"] == "dhc"
    seed = parse(Int, o["seed"])
    epochs = parse(Int, o["epochs"])
    pop = parse(Int, o["pop"])
    genes = parse(Int, o["genes"])
    head = parse(Int, o["head"])

    x, y = read_tensors(tag)
    lossfn = make_tensor_loss(y)

    n_invalid = Threads.Atomic{Int}(0)

    # SITE gives individuals it rejects a large finite loss (1000); the same value is
    # used here. Tournament selection skips non-finite fitness values, so an infinite
    # penalty would take the (many) type-invalid chromosomes out of the pool altogether.
    const_penalty = 1000.0

    use_scaling = o["scaling"] == "true"
    with_units = o["units"] == "true"
    make_loss(reg, lf, target) = @inline function (elem, validate::Bool)
        if isnan(mean(elem.fitness)) || validate
            try
                # with scaling the genes are the basis and their coefficients are solved
                # for, so the chromosome only has to supply the structure
                pred = use_scaling ? predictT_scaled(reg, elem, target) :
                       predictT(reg, elem.expression_raw)
                if pred isa AbstractVector && length(pred) == length(target) &&
                   eltype(pred) <: Tensor{2,3}
                    l = lf(pred)
                    elem.fitness = (isfinite(l) ? l : const_penalty,)
                else
                    Threads.atomic_add!(n_invalid, 1)
                    elem.fitness = (const_penalty,)
                end
            catch
                Threads.atomic_add!(n_invalid, 1)
                elem.fitness = (const_penalty,)
            end
        end
    end

    # warm-up: compile the whole fit!/predictT path, so the timed run excludes JIT time
    t_warm = @elapsed begin
        xw = [v[1:30] for v in x]
        yw = y[1:30]
        w = build(seed=seed, use_dims=use_dims, gene_count=genes, head_len=head, with_units=with_units)
        allocate_buffers!(w, xw)
        # population 200: a package version this harness once ran against indexed
        # `parents[1:100]` in its genetic operators, so the mating pool (70 % of the
        # population) needed 100 individuals; the current src/ has no such limit
        fit!(w, 2, 200, make_loss(w, make_tensor_loss(yw), yw);
            target_dimension=use_dims ? (with_units ? U_TARGET : DIM_TARGET) : nothing,
            correction_epochs=1, correction_amount=parse(Float64, o["corr"]), hof=1)
    end

    regressor = build(seed=seed, use_dims=use_dims, gene_count=genes, head_len=head, with_units=with_units)
    allocate_buffers!(regressor, x)
    loss_callback = make_loss(regressor, lossfn, y)

    epoch_loss = Float64[]
    epoch_time = Float64[]
    converged = Ref(-1)
    t0 = time_ns()
    bc = function (population, epoch)
        best = mean(population[1].fitness)
        push!(epoch_loss, best)
        push!(epoch_time, (time_ns() - t0) / 1e9)
        if best < TOL && converged[] < 0
            converged[] = epoch
            return true
        end
        return false
    end

    GC.gc()
    t_solve = @elapsed fit!(regressor, epochs, pop, loss_callback;
        target_dimension=use_dims ? (with_units ? U_TARGET : DIM_TARGET) : nothing,
        correction_epochs=1, correction_amount=parse(Float64, o["corr"]),
        break_condition=bc, hof=1)

    best = regressor.best_models_[1]
    # `equation_string` renders what the model actually computes: with fitted gene
    # coefficients that is the weighted sum of the genes, not the connected chromosome.
    karva = try equation_string(best) catch e; "<unprintable: $(e)>" end
    # whether the winner meets the target as it is scored (gene by gene when scaled);
    # `dimensional_check` below only names the configuration
    target_dim = with_units ? U_TARGET : DIM_TARGET
    winner_homogeneous = !use_dims ? nothing : best.dimension_homogene && (use_scaling ?
        is_gene_wise_homogeneous(best.expression_raw, target_dim, regressor.token_dto_, genes) :
        is_dimensionally_homogeneous(best.expression_raw, target_dim, regressor.token_dto_))

    # Prediction of the winner on the clean data, for the coefficient projection, through
    # a second regressor whose buffers are allocated on that data: the three-argument
    # `predictT(reg, rek, x_data)` did not work in the versions this harness was written
    # against (it returns the prediction now).
    x_clean, y_clean = read_tensors("clean")
    clean_reg = build(seed=seed, use_dims=use_dims, gene_count=genes, head_len=head, with_units=with_units)
    allocate_buffers!(clean_reg, x_clean)
    # with scaling the winner's coefficients were fitted on the training data; refit them
    # here and the clean-data projection would report a model that was never trained
    pred_clean = if use_scaling && !isnothing(best.scaling_weights)
        bases = gene_bases(clean_reg, best)
        usable = [b for b in bases if b isa AbstractVector &&
                  length(b) == length(y_clean) && eltype(b) <: Tensor{2,3}]
        length(usable) == length(best.scaling_weights) ?
            [sum(best.scaling_weights[j] * usable[j][i] for j in eachindex(usable))
             for i in eachindex(y_clean)] : nothing
    else
        predictT(clean_reg, best.expression_raw)
    end
    valid_clean = pred_clean isa AbstractVector && eltype(pred_clean) <: Tensor{2,3}
    clean_loss = valid_clean ? make_tensor_loss(y_clean)(pred_clean) : NaN
    y_pred_clean = valid_clean ?
                   [pred_clean[p][i, j] for p in 1:length(pred_clean) for i in 1:3 for j in 1:3] :
                   Float64[]

    res = OrderedDict{String,Any}(
        "framework" => "GepTensorRegressor (batched) @ " * SHADOW_ROOT,
        "case" => "maxwell", "data" => tag, "config" => o["config"],
        "linear_scaling" => o["scaling"] == "true",
        "si_units" => o["units"] == "true",
        "dimensional_check" => use_dims, "winner_homogeneous" => winner_homogeneous,
        "constant_optimisation" => false,
        "seed" => seed, "population" => pop, "max_epochs" => epochs,
        "head_len" => head, "gene_count" => genes,
        "threads" => Threads.nthreads(),
        "load_time_s" => t_load, "warmup_time_s" => t_warm, "solve_time_s" => t_solve,
        "epochs_run" => length(epoch_loss), "converged_epoch" => converged[],
        "converged" => converged[] > 0,
        "final_loss" => minimum(epoch_loss), "best_loss" => minimum(epoch_loss),
        "clean_data_loss" => clean_loss, "y_pred_clean" => y_pred_clean,
        "invalid_evaluations" => n_invalid[],
        "expression" => karva,
        "epoch_loss" => epoch_loss, "epoch_time_s" => epoch_time)

    @printf("\n[shadow tensor | %s | %s | seed %d]  loss=%.3e  epochs=%d  solve=%.1fs (load %.1f, warmup %.1f)\n",
        tag, o["config"], seed, res["final_loss"], length(epoch_loss), t_solve, t_load, t_warm)
    println("  karva: ", karva)

    if !isempty(o["out"])
        path = isabspath(o["out"]) ? o["out"] : joinpath(HERE, o["out"])
        mkpath(dirname(path))
        open(path, "w") do f
            JSON.print(f, res)
        end
        println("  -> ", path)
    end
end

main(parse_args(ARGS))
