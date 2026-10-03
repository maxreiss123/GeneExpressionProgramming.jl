# Multi-Objective Optimization

Multi-objective optimization balances competing objectives, typically the accuracy of an expression against its simplicity, or several performance criteria, as demonstrated in [1]. In GeneExpressionProgramming.jl a custom loss sets one fitness entry per objective, and parents are selected by NSGA-II (Non-dominated Sorting Genetic Algorithm II) [2]. The non-dominated models among the returned ones form an approximation of the Pareto front, from which you choose the trade-off you need.

## Complete Multi-Objective Example

The two objectives are the training MSE and the size of the model, measured as the length of its karva string.

```julia
using GeneExpressionProgramming
using Random
using Statistics
using Plots

# Set random seed for reproducibility
Random.seed!(123)

# Problem setup
number_features = 2
n_samples = 300
noise_level = 0.1

# Features in [-2, 2], one row per sample
x_data = 4 * (rand(Float64, n_samples, number_features) .- 0.5)

# Target with several terms
y_data = @. x_data[:,1]^3 - 2*x_data[:,1]^2*x_data[:,2] + 
            x_data[:,1]*x_data[:,2]^2 + 0.5*sin(x_data[:,1]) - 
            0.3*cos(x_data[:,2]) + x_data[:,1] + x_data[:,2]
y_data += noise_level * randn(n_samples)

# The search sees only the training part; the validation part is held out
x_train, y_train, x_val, y_val = train_test_split(x_data, y_data; train_ratio=0.8)

# Evolution parameters
epochs = 1000
population_size = 1000

# Two objectives; the connectors are those of the default set that are also entered here
regressor = GepRegressor(number_features; number_of_objectives=2,
                         entered_non_terminals=[:+, :*, :-, :sin, :cos])

# The loss runs inside the threaded fitness loop: one evaluation context per thread,
# built once for the training data
train_ctxs = thread_contexts(regressor.toolbox_, x_train')

function multi_objective_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        y_pred = try
            elem(train_ctxs[Threads.threadid()])
        catch
            nothing
        end
        if y_pred isa AbstractVector && all(isfinite, y_pred)
            mse = mean(abs2, y_train .- y_pred)              # objective 1: training MSE
            model_size = 0.01 * length(elem.expression_raw)  # objective 2: size, scaled
            elem.fitness = (mse, model_size)
        else
            elem.fitness = (Inf, Inf)                        # cannot be evaluated
        end
    end
end

training_time = @elapsed fit!(regressor, epochs, population_size, multi_objective_loss; hof=20)
println("Training completed in $(round(training_time, digits=2)) seconds")

# best_models_ holds the `hof` models ranked by the mean of their objectives; the
# non-dominated ones among them approximate the Pareto front
models = regressor.best_models_
pareto_solutions = [m for m in models if !any(o -> dominates_(o.fitness, m.fitness), models)]
sort!(pareto_solutions, by=m -> length(m.expression_raw))

r2 = get_loss_function("r2_score")
mse(m, x, y) = mean(abs2, y .- m(x'))    # recomputed: a duplicate's fitness carries the penalty
println("Size | Training MSE | Validation MSE | Expression")
for m in pareto_solutions
    println(lpad(length(m.expression_raw), 4), " | ",
            lpad(round(mse(m, x_train, y_train), sigdigits=4), 12), " | ",
            lpad(round(mse(m, x_val, y_val), sigdigits=4), 14), " | ", m)
end

# Three solutions: the simplest, the most accurate on the training data, and one between
n = length(pareto_solutions)
selected = [(pareto_solutions[1], "Simplest"),
            (pareto_solutions[cld(n, 2)], "Balanced"),
            (pareto_solutions[n], "Most Accurate")]

for (m, label) in selected
    val_pred = m(x_val')
    println("$label: $m")
    println("  Validation R²: $(round(r2(y_val, val_pred), digits=4))")
end

# Pareto front: size against training MSE
sizes = [length(m.expression_raw) for m in pareto_solutions]
mses = [mse(m, x_train, y_train) for m in pareto_solutions]
p1 = scatter(sizes, mses,
             xlabel="Size (symbols in the karva string)",
             ylabel="Training MSE",
             title="Pareto Front",
             legend=false)

# Predictions of the selected solutions on the validation data
p2 = plot(layout=(1, 3), size=(1200, 300))
for (i, (m, label)) in enumerate(selected)
    scatter!(p2[i], y_val, m(x_val'),
             xlabel="Actual Values", ylabel="Predicted Values",
             title=label, legend=false, alpha=0.6)
    plot!(p2[i], [minimum(y_val), maximum(y_val)], [minimum(y_val), maximum(y_val)],
          color=:red, linestyle=:dash)
end

savefig(p1, "pareto_front.png")
savefig(p2, "solution_comparison.png")
```

## Understanding the Results

### Pareto Front Analysis

A solution dominates another if it is no worse in every objective and better in at least one; the Pareto front is the set of solutions that no other dominates. Each point on it is a different trade-off between accuracy and size. In the plot, simple but less accurate expressions lie to the upper left and larger, more accurate ones to the lower right; the knee of the front often marks a good trade-off. Choose among the models by their validation error and by how much structure you can interpret.

### Ranking by the Mean

NSGA-II selects the parents by Pareto rank and crowding distance, but survival and `best_models_` rank the population by the mean of each fitness tuple. That is why the example scales the size by 0.01, as `examples/Main_min_example_mo.jl` does: with objectives on very different scales, the larger one would dominate which models survive. The front above is taken from the `hof` best models only; raise `hof` to see more of it.

## Advanced Multi-Objective Techniques

Set `number_of_objectives` to the length of the tuple the loss sets. The snippets below reuse the data of the example.

### Three Objectives

```julia
regressor3 = GepRegressor(number_features; number_of_objectives=3,
                          entered_non_terminals=[:+, :*, :-, :sin, :cos])
ctxs3 = thread_contexts(regressor3.toolbox_, x_train')

function three_objective_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        y_pred = try
            elem(ctxs3[Threads.threadid()])
        catch
            nothing
        end
        if y_pred isa AbstractVector && all(isfinite, y_pred)
            residuals = abs.(y_train .- y_pred)
            elem.fitness = (mean(abs2, residuals),                 # training MSE
                            maximum(residuals),                    # worst-case error
                            0.01 * length(elem.expression_raw))    # size
        else
            elem.fitness = (Inf, Inf, Inf)
        end
    end
end

fit!(regressor3, epochs, population_size, three_objective_loss)
```

### Weighted Objectives

Several criteria can also be folded into one objective, with weights and scales of your choice; the selection is then by tournament.

```julia
regressor_ws = GepRegressor(number_features; entered_non_terminals=[:+, :*, :-, :sin, :cos])
ctxs_ws = thread_contexts(regressor_ws.toolbox_, x_train')

mse_scale = var(y_train)    # your scales
size_scale = 50.0

function weighted_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        y_pred = try
            elem(ctxs_ws[Threads.threadid()])
        catch
            nothing
        end
        if y_pred isa AbstractVector && all(isfinite, y_pred)
            mse = mean(abs2, y_train .- y_pred)
            model_size = length(elem.expression_raw)
            elem.fitness = (0.7 * mse / mse_scale + 0.3 * model_size / size_scale,)
        else
            elem.fitness = (Inf,)
        end
    end
end

fit!(regressor_ws, epochs, population_size, weighted_loss)
```

### Constraint Handling

A hard constraint gives every violating model the worst fitness:

```julia
function constrained_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        # at most 25 symbols in the karva string
        if length(elem.expression_raw) > 25
            elem.fitness = (Inf, Inf)
            return
        end
        y_pred = try
            elem(train_ctxs[Threads.threadid()])
        catch
            nothing
        end
        elem.fitness = y_pred isa AbstractVector && all(isfinite, y_pred) ?
                       (mean(abs2, y_train .- y_pred), 0.01 * length(elem.expression_raw)) :
                       (Inf, Inf)
    end
end

fit!(regressor, epochs, population_size, constrained_loss)
```

`Inf` is a safe penalty: NSGA-II ranks a tuple with more non-finite entries behind one with fewer, and tournament selection (one objective) leaves non-finite fitness values out.

### Several Expressions and an Expensive Loss

One chromosome can carry several expressions: `split_karva(elem, k)` splits its genes into `k` parts (the gene count must be divisible by `k`), `split_predict(elem, ctx, k)` evaluates them in a buffer context, and `split_equations(elem, k)` prints them. A natural objective is then one per expression, e.g. for the right-hand sides of a system of ODEs, the error of each state. Where the loss is a solver in the loop, a `SurrogateScreening` lets it score only a few individuals per epoch and predicts the others with one Gaussian process per objective:

```julia
regressor_sys = GepRegressor(2; entered_features=[:x, :y], entered_non_terminals=[:+, :-, :*],
                             gene_count=4, head_len=4, number_of_objectives=2)

function system_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        # your solver: it evaluates both expressions at every stage, with
        # split_predict(elem, ctx, 2) in this thread's context, and returns one error per state
        elem.fitness = simulate(elem)
    end
end

surrogate = SurrogateScreening(regressor_sys, probes;   # probes: states the solver visits
    expressions=2,                  # one latent block per expression
    objective_expressions=[1, 2],   # the process of objective j sees expression j alone
    individuals_per_epoch=0.15)     # 15 % of the new individuals of an epoch
fit!(regressor_sys, epochs, population_size, system_loss; surrogate=surrogate, hof=10)

best = regressor_sys.best_models_                 # all of them scored by the loss
front = best[calculate_fronts([m.fitness for m in best])[1]]
[split_equations(m, 2) for m in front]
```

A prediction is clamped behind the best scored value of every objective, and since the population survives by the mean of its objectives, the scored individuals that hold these values are kept among the survivors, so the best scored value of an objective never gets worse. `examples/Main_surrogate_multi_objective.jl` runs such a search for the Lotka–Volterra system; with its seed the screened search found both equations after 1,196 solver calls, the unscreened one after 10,168. [Surrogate Screening](surrogate-screening.md) walks through it, and the [API Reference](../api-reference.md#Surrogate-Screening) lists the options.

## Performance Considerations

### Evaluating Inside the Loss

The loss runs inside the threaded fitness loop. Calling `elem(x_train')` would build an evaluation context for every call; the example builds one context per thread once and evaluates in the calling thread's. The result lives in the context's buffers, so use it (or copy it) before the next evaluation in the same context.

### Initial Population

The custom-loss method of `fit!` has no `population_sampling_multiplier` keyword and uses `runGep`'s default of 100: the initial population is picked from 100 × `population_size` random chromosomes, which lengthens the start of a run.

### Convergence Monitoring

The history records, per generation, the objectives of the model ranked first by their mean:

```julia
history = regressor.fitness_history_.train_loss
recorded = [history[i] for i in eachindex(history) if isassigned(history, i)]

p1 = plot([h[1] for h in recorded], xlabel="Generation", ylabel="Training MSE",
          title="Objective 1", legend=false)
p2 = plot([h[2] for h in recorded], xlabel="Generation", ylabel="Size × 0.01",
          title="Objective 2", legend=false)
plot(p1, p2, layout=(1, 2))
```

## References

[1] Waschkowski, F., Zhao, Y., Sandberg R. D., Klewicki J., (2022), Multi-objective CFD-driven development of coupled turbulence closure models. Journal of Computational Physics, vol. 452, 

[2] K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, (2002) "A fast and elitist multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197 

---

*Next: [Physical Dimensionality](physical-dimensions.md)*
