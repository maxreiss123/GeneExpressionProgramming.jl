#=
GEP with semantic backpropagation (SBP) on the equations of the PhySO comparison.

    julia --project=. --threads=1 paper/comparisonPhySo/gep_run.jl [key=value ...]

Keys (defaults in brackets): noise [0], stop [floor; none runs the whole budget], front
[false; true records the accuracy-complexity front, see below], runs [all; name:seed pairs,
comma separated, instead of equations and seeds], equations [all of data<tag>/meta.json, comma
separated], seeds [1,...,10], epochs [430], population [1000], head_len [8],
linear_scaling [true], out [paper/comparisonPhySo/results/gep<tag>], worker [1/1]: with
worker=k/K the process runs every K-th of the (equation, seed) runs, starting at the k-th,
so K processes share them. Runs whose JSON exists are skipped. The data comes from
export_data.py; the tag is empty without noise and _noise<level> with it (as common.tag).

Every run builds a fresh `GepRegressor` with the SI units of the features, so the timed
part includes the library SBP draws on, and fits it with the target's units: individuals
whose units miss the target are repaired by SBP, and only homogeneous ones are scored.
Operators + - * / sqr sqrt exp log sin cos, the constant 1 and one random constant, three
genes. With `linear_scaling` (the default) the model is the least-squares combination of
the genes, each held to the target's units; this plays the part of PhySO's free constants.

The run stops when the best individual reaches the normalised training RMSE `stop_nrmse`
of the data's meta.json, the fit at which physo_run.py stops PhySO: 1e-5 without noise,
the noise floor with it (common.stop_nrmse), or after `epochs` epochs. The
population is `population` plus 0.7 * `population` offspring, and each epoch breeds
0.7 * `population` new candidates: 1000 and 430 epochs propose 1700 + 430 * 700 ~ 3.0e5
candidates, as PhySO's 30 epochs of 10 000.

A warm-up run on the first equation (not recorded) compiles the code paths first, so the
wall times exclude Julia's compilation. Writes one JSON per run to `out`.

With front=true the run also keeps, for every model size (the number of symbols of the
expressed chromosome), the best-fitting individual of any epoch, and writes the
non-dominated ones (each fits better than every smaller one) as `front`, with their
normalised training RMSE; `floor_nrmse` is the noise floor of the data (common.stop_nrmse)
whatever `stop` is. judge_front.py judges these fronts.
=#
include(joinpath(@__DIR__, "..", "..", "src", "GeneExpressionProgramming.jl"))
using .GeneExpressionProgramming
using DelimitedFiles
using JSON
using Printf
using Random
using Statistics

const HERE = @__DIR__
const OPS = [:+, :-, :*, :/, :sqr, :sqrt, :exp, :log, :sin, :cos]

function parse_args(args)
    opts = Dict{String,String}()
    for a in args
        k, v = split(a, "="; limit=2)
        opts[k] = v
    end
    return opts
end

load(root, name, file) = readdlm(joinpath(root, name, file), ',', Float64; skipstart=1)

function run_one(eq, seed; root, stop_nrmse, noise, epochs, population, head_len,
                 linear_scaling, out=nothing, front=false, floor_nrmse=stop_nrmse)
    name = eq["name"]
    train, test = load(root, name, "train_s$seed.csv"), load(root, name, "test.csv")
    x_train, y_train = train[:, 1:end-1], train[:, end]
    x_test, y_test = test[:, 1:end-1], test[:, end]
    n = size(x_train, 2)
    dims = Dict{Symbol,Vector{Float16}}(Symbol("x$i") => Float16.(u)
                                        for (i, u) in enumerate(eq["units"]))
    target = Float16.(eq["target_units"])
    # std is the n-1 one, as in PhySO's reward
    threshold = (stop_nrmse * std(y_train))^2
    # called every epoch, so it also counts them
    last_epoch = Ref(0)
    stop(pop, epoch) = (last_epoch[] = epoch; mean(pop[1].fitness) <= threshold)
    # best (training MSE, model) per model size over all epochs; the callback runs every
    # epoch after scoring, before breeding
    archive = Dict{Int,Tuple{Float64,String}}()
    function record(pop, epoch, selected)
        for ind in pop
            f = mean(ind.fitness)
            (isfinite(f) && f < 1e300) || continue
            c = length(ind.expression_raw)
            f < get(archive, c, (Inf, ""))[1] && (archive[c] = (f, equation_string(ind)))
        end
    end

    # the regressor's random generator is seeded from the global one
    Random.seed!(seed)
    t0 = time_ns()
    regressor = GepRegressor(n; considered_dimensions=dims, entered_non_terminals=OPS,
        entered_terminal_nums=[Symbol(1.0)], rnd_count=1, gene_count=3, head_len=head_len,
        max_permutations_lib=10000, rounds=5)
    t_lib = (time_ns() - t0) / 1e9
    fit!(regressor, epochs, population, x_train', y_train; loss_fun="mse",
        target_dimension=target, linear_scaling=linear_scaling, break_condition=stop,
        file_logger_callback=front ? record : nothing)
    wall = (time_ns() - t0) / 1e9

    best = regressor.best_models_[1]
    r2 = get_loss_function("r2_score")
    score(x, y) = try
        p = regressor(x')
        p isa AbstractVector && all(isfinite, p) ? r2(y, p) : NaN
    catch
        NaN
    end
    epochs_run = last_epoch[]
    res = Dict(
        "method" => "GEP-SBP", "equation" => name, "seed" => seed,
        "expression" => equation_string(best),
        "hall_of_fame" => [equation_string(m) for m in regressor.best_models_],
        "variables" => ["x$i" for i in 1:n],
        "homogeneous" => is_gene_wise_homogeneous(best.expression_raw, target,
            regressor.token_dto_, 3),
        "r2_train" => score(x_train, y_train), "r2_test" => score(x_test, y_test),
        "train_mse" => mean(best.fitness),
        "wall_time" => wall, "library_time" => t_lib, "epochs_run" => epochs_run,
        "evaluations" => population + floor(Int, 0.7 * population) +
                         epochs_run * floor(Int, 0.7 * population),
        "epochs_budget" => epochs, "population" => population, "head_len" => head_len,
        "linear_scaling" => linear_scaling, "threads" => Threads.nthreads(),
        "noise" => noise, "stop_nrmse" => stop_nrmse, "floor_nrmse" => floor_nrmse)
    if front
        pareto, best_f = Dict{String,Any}[], Inf
        for c in sort!(collect(keys(archive)))
            f, expr = archive[c]
            f < best_f || continue
            best_f = f
            push!(pareto, Dict("complexity" => c, "nrmse" => sqrt(f) / std(y_train),
                               "expression" => expr))
        end
        res["front"] = pareto
    end
    if !isnothing(out)
        mkpath(out)
        open(joinpath(out, "$(name)_s$(seed).json"), "w") do io
            JSON.print(io, res, 1)
        end
    end
    println("$(name) s$(seed)  r2_test=$(round(res["r2_test"]; digits=6))  " *
            "wall=$(round(wall; digits=1))s  epochs=$(epochs_run)  $(res["expression"])")
    return res
end

function main(args)
    opts = parse_args(args)
    noise = parse(Float64, get(opts, "noise", "0"))
    tag = noise == 0 ? "" : @sprintf("_noise%g", noise)
    root = joinpath(HERE, "data" * tag)
    meta = JSON.parsefile(joinpath(root, "meta.json"))
    floor_of(name, seed) = Float64(meta["stop_nrmse"][name][string(seed)])
    stop_of(name, seed) = get(opts, "stop", "floor") == "none" ? 0.0 : floor_of(name, seed)
    front = parse(Bool, get(opts, "front", "false"))
    eqs = meta["equations"]
    if haskey(opts, "equations")
        wanted = split(opts["equations"], ",")
        eqs = [e for e in eqs if e["name"] in wanted]
    end
    seeds = haskey(opts, "seeds") ? parse.(Int, split(opts["seeds"], ",")) :
            Int.(meta["seeds"])
    kw = (epochs=parse(Int, get(opts, "epochs", "430")),
          population=parse(Int, get(opts, "population", "1000")),
          head_len=parse(Int, get(opts, "head_len", "8")),
          linear_scaling=parse(Bool, get(opts, "linear_scaling", "true")))
    out = get(opts, "out", joinpath(HERE, "results", "gep" * tag))

    k, K = parse.(Int, split(get(opts, "worker", "1/1"), "/"))
    jobs = [(eq, seed) for eq in eqs for seed in seeds]
    if haskey(opts, "runs")
        byname = Dict(e["name"] => e for e in meta["equations"])
        jobs = [(byname[String(a)], parse(Int, b))
                for (a, b) in (split(r, ":") for r in split(opts["runs"], ","))]
        eqs = unique(first.(jobs))
    end
    jobs = jobs[k:K:end]

    # compile the code paths before anything is timed
    run_one(first(eqs), first(seeds); kw..., epochs=3, root=root, noise=noise,
        stop_nrmse=stop_of(first(eqs)["name"], first(seeds)))
    for (eq, seed) in jobs
        isfile(joinpath(out, "$(eq["name"])_s$(seed).json")) && continue
        run_one(eq, seed; kw..., out=out, root=root, noise=noise,
            stop_nrmse=stop_of(eq["name"], seed), front=front,
            floor_nrmse=floor_of(eq["name"], seed))
    end
end

main(ARGS)
