#=
Where GEP-SBP's time per candidate goes (README, "fewer candidates, not cheaper ones").

    julia --project=. --threads=1 paper/comparisonPySR/profile_gep.jl [EQUATION] [EPOCHS] [noise] [noprofile]

Defaults: II.11.3, 100 epochs, noise 0.1 (data of export_data.py), seed
1; the run does not stop early, so every epoch breeds 700 candidates. The same run, with
gep_run.jl's settings, is timed with the units (SBP library seeding and repair) and
without them, after a warm-up that compiles both; then the run with units is profiled and
each sample is put on the innermost function of the package (src/) in its backtrace.
=#
include(joinpath(@__DIR__, "..", "..", "src", "GeneExpressionProgramming.jl"))
using .GeneExpressionProgramming
using DelimitedFiles, JSON, Statistics, Random, Profile, Printf

const OPS = [:+, :-, :*, :/, :sqr, :sqrt, :exp, :log, :sin, :cos]
name = length(ARGS) > 0 ? ARGS[1] : "II.11.3"
epochs = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 100
noise = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.1
const ROOT = joinpath(@__DIR__, noise == 0 ? "data" : @sprintf("data_noise%g", noise))
meta = JSON.parsefile(joinpath(ROOT, "meta.json"))
eq = first(e for e in meta["equations"] if e["name"] == name)
train = readdlm(joinpath(ROOT, name, "train_s1.csv"), ',', Float64; skipstart=1)
x, y = train[:, 1:end-1], train[:, end]
n = size(x, 2)
dims = Dict{Symbol,Vector{Float16}}(Symbol("x$i") => Float16.(u) for (i, u) in enumerate(eq["units"]))
target = Float16.(eq["target_units"])

function run(epochs; units=true)
    Random.seed!(1)
    t0 = time_ns()
    reg = units ?
        GepRegressor(n; considered_dimensions=dims, entered_non_terminals=OPS,
            entered_terminal_nums=[Symbol(1.0)], rnd_count=1, gene_count=3, head_len=8,
            max_permutations_lib=10000, rounds=5) :
        GepRegressor(n; entered_non_terminals=OPS, entered_terminal_nums=[Symbol(1.0)],
            rnd_count=1, gene_count=3, head_len=8)
    tlib = (time_ns() - t0) / 1e9
    if units
        fit!(reg, epochs, 1000, x', y; loss_fun="mse", linear_scaling=true, target_dimension=target)
    else
        fit!(reg, epochs, 1000, x', y; loss_fun="mse", linear_scaling=true)
    end
    return tlib, (time_ns() - t0) / 1e9
end

run(3; units=true); run(3; units=false)          # compile
cand = 1000 + 700 + epochs * 700
for units in (true, false)
    tlib, t = run(epochs; units)
    @printf("%-9s library %.2f s, whole run %.1f s, %d candidates, %.0f candidates/s\n",
        units ? "units" : "no units", tlib, t, cand, cand / t)
end
length(ARGS) > 3 && ARGS[4] == "noprofile" && exit(0)

Profile.clear()
Profile.init(n=10^7, delay=0.002)
@profile run(epochs; units=true)

function attribute(data, lidict)
    byfun, byfile, bt, total = Dict{String,Int}(), Dict{String,Int}(), UInt64[], 0
    for ip in data
        if ip != 0
            push!(bt, ip)
            continue
        end
        isempty(bt) && continue
        total += 1
        key, file = "outside the package", "outside"
        for a in bt, fr in lidict[a]       # innermost frame first
            f = string(fr.file)
            if occursin(joinpath("src", ""), f) && occursin("GeneExpressionProgramming", f)
                key, file = basename(f) * ": " * string(fr.func), basename(f)
                @goto found
            end
        end
        @label found
        byfun[key] = get(byfun, key, 0) + 1
        byfile[file] = get(byfile, file, 0) + 1
        empty!(bt)
    end
    return byfun, byfile, total
end
data = Profile.fetch(include_meta=false)
byfun, byfile, total = attribute(data, Profile.getdict(data))
println("\nsamples: $total; by file (innermost package frame)")
for (k, v) in sort(collect(byfile); by=last, rev=true)
    @printf("  %5.1f %%  %s\n", 100v / total, k)
end
println("by function")
for (k, v) in first(sort(collect(byfun); by=last, rev=true), 25)
    @printf("  %5.1f %%  %s\n", 100v / total, k)
end
