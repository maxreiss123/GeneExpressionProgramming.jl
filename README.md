# GeneExpressionProgramming for symbolic regression
The repository contains an implementation of Gene Expression Programming [1] for symbolic regression. A candidate model is a chromosome, a vector of `Int8` tokens; its expressed part in prefix order, the karva string, is scored by a batched evaluator that runs it over whole data columns in preallocated per-thread buffers, so no expression tree is built. Given the physical dimensions of the features and of the target, semantic backpropagation (SBP) repairs candidates towards dimensional homogeneity [2].

# Features
- Standard GEP symbolic regression
- Multi-objective optimisation (NSGA-II selection)
- Constant optimisation (Nelder-Mead), and optional gene-wise linear scaling
- Vector and tensor regression (scalars, vectors and higher-order tensors via Tensors.jl)
- Physical dimensions: candidates are repaired towards the target's dimension, and only homogeneous ones are scored
- Surrogate screening of expensive losses: Gaussian processes decide which few individuals per epoch the loss scores and predict the loss of the others, for one or several objectives and for chromosomes that carry several expressions
- The constants of a model against an expensive loss (e.g. CFD in the loop): Nelder-Mead whose loss calls a Gaussian process over the constants places, inside a search or on its own
- Multi-threaded evaluation, variation and repair

# How does it work?
A chromosome, a string of symbols, is decoded into an expression, drawn here as a tree:
![Decoding](images/gep_decoding.gif)

Changing the strings with genetic operators over many generations leads to:
![Solve](images/solve_gep.gif)


# How to use it?
- Install the package:
  ```julia
    using Pkg
    
    Pkg.add("GeneExpressionProgramming")
    
    # or to get the latest version
    # Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git") 
    
  ```

  ```julia
  # Min_example 
  using GeneExpressionProgramming
  using Random

  Random.seed!(1)

  # number of epochs (generations) and population size
  epochs = 1000
  population_size = 1000

  # number of features
  number_features = 2

  x_data = randn(Float64, 100, number_features)
  y_data = @. x_data[:,1] * x_data[:,1] + x_data[:,1] * x_data[:,2] - 2 * x_data[:,2] * x_data[:,2]

  # define the regressor
  regressor = GepRegressor(number_features)

  # epochs, population size, the features (one row per feature, hence the transpose), the target and the loss
  fit!(regressor, epochs, population_size, x_data', y_data; loss_fun="mse")

  pred = regressor(x_data') # predictions of the best model, also for new data

  @show regressor.best_models_[1]          # printed as an equation
  @show regressor.best_models_[1].fitness
  ```
- Start Julia with several threads (`julia --threads=auto`): the fitness evaluation, the genetic operators and the unit repair run in parallel.
- `fit!` records the training and validation loss of the best model per epoch in `regressor.fitness_history_` (`train_loss`, `val_loss`).
- The `examples` folder holds runnable scripts; they include the sources directly, so from a clone they run with `julia --project=. --threads=4 examples/<script>.jl`:

  | Script | Shows |
  | --- | --- |
  | `Main_min_example.jl` | the example above, with optional plots when Plots.jl is installed |
  | `Main_min_with_csv.jl` | data read from a CSV file, train/test split |
  | `Main_min_bench.jl` | timing of a small search, with and without compilation |
  | `Main_min_example_mo.jl` | multi-objective search with a custom loss |
  | `Main_physical_dimensions.jl` | the search held to physical units (below) |
  | `Main_tensor_regression.jl` | a vector-valued target from scalar and vector features (below) |
  | `Main_streaming_chunks.jl` | a loss evaluated chunk by chunk, for data larger than memory |
  | `Main_surrogate_screening.jl` | an expensive loss (an ODE solve per call) screened by a Gaussian process (below) |
  | `Main_surrogate_multi_objective.jl` | the same for a system of two ODEs: two expressions per chromosome, one objective per state, the scored Pareto front |
  | `Main_screened_constants.jl` | the constants of a model against an expensive loss (an ODE solve per call), tuned inside a search and calibrated on their own by Nelder-Mead screened by a Gaussian process (below) |
  | `Main_fictive_cfd_in_the_loop.jl` | a turbulence closure searched with a fictive CFD solver in the loop (a 1D stand-in, not an actual CFD run): screened, two objectives, physical units, a loss whose diverged solves give a large error, and a closure calibrated by Nelder-Mead and by the screened swarm (below) |

# How to consider the physical dimensions mentioned within [2]? 
To account for the physical dimensions of the inputs, subexpressions are corrected towards the dimension the output requires by semantic backpropagation. In theory it works as follows:
![Semantic](images/semantic_backprop.gif)

Every feature gets its SI dimension and the search gets the target's. Each generation, the offspring whose dimension misses the target are repaired in place: the required dimension is pushed down the expression, and where a subexpression cannot meet it, it is changed by the cheapest fix available -- swapping a terminal or an operator, splitting the requirement between the operands, or splicing in a precomputed library subexpression of exactly the needed dimension. The library is indexed by dimension: an exact table for the splices, and a kd-tree over the dimensions that one more product, quotient or unary function reaches from the library, which proposes the splits. A repair keeps the gene structure intact (operators stay in the head), and only dimensionally homogeneous individuals are scored.

- For a more concrete example, imagine you want to find $J$ explaining superconductivity as $J=-\rho \frac{q}{m} A$ (Feynman III 21.20)
- $J$ marking the electric current density, $q$ the electric charge, $\rho$ the charge density, $m$ the mass and $A$ the magnetic vector potential

 ```julia
  # Physical-dimension example
  using GeneExpressionProgramming
  using Random
  using CSV
  using DataFrames

  Random.seed!(1)

  # number of epochs (generations) and population size
  epochs = 1000
  population_size = 1000


  # 5 columns: 4 features, then the target (run from the repository root)
  data = Matrix(CSV.read("paper/srsd/feynman-III.21.20\$0.01.txt", DataFrame))
  data = data[all.(x -> !any(isnan, x), eachrow(data)), :]
  num_cols = size(data, 2) #num_cols =5 


  # Perform a simple train test split
  x_train, y_train, x_test, y_test = train_test_split(data[:, 1:num_cols-1], data[:, num_cols]; consider=4)

  #define a target dimension - here ampere per square meter - as SI exponents [kg, m, s, K, mol, A, cd] (the unit order of OpenFOAM) - https://doc.cfd.direct/openfoam/user-guide-v6/basic-file-format
  target_dim = Float16[0, -2, 0, 0, 0, 1, 0] # Aiming for the electric current density (Ampere/m^2)


  # dimensions of the features: the file's columns rho_c_0, q, A_vec, m are named x1 ... x4
  feature_dims = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, -3, 1, 0, 0, 1, 0],   #rho    m^(-3) * s * A 
    :x2 => Float16[0, 0, 1, 0, 0, 1, 0],    #q      s*A
    :x3 => Float16[1, 1, -2, 0, 0, -1, 0],  #A      kg*m*s^(-2)*A^(-1)
    :x4 => Float16[1, 0, 0, 0, 0, 0, 0],    #m      kg
  )


  # the feature dimensions, and the library of dimensionally consistent subexpressions the repair draws on: at most max_permutations_lib new ones per round, up to rounds + 1 symbols long
  regressor = GepRegressor(num_cols-1; considered_dimensions=feature_dims,max_permutations_lib=10000, rounds=7)

  # as above, with test data for the validation loss and the target dimension
  fit!(regressor, epochs, population_size, x_train', y_train; x_test=x_test', y_test=y_test, loss_fun="mse", target_dimension=target_dim)

  pred = regressor(x_test')

  @show regressor.best_models_[1]
  @show regressor.best_models_[1].fitness
  @show regressor.best_models_[1].dimension_homogene
  ```
- `fit!` takes two more knobs for the repair: `correction_epochs` (repair every n-th epoch, default 1) and `correction_amount` (the most individuals repaired per correction epoch, as a fraction of the population, default 1.0). Individuals that are neither homogeneous nor repaired get the worst fitness (`Inf`) instead of being scored.
- `is_dimensionally_homogeneous(expr, target_dim, regressor.token_dto_)` checks any karva string against a target.
- Remark: Template for rerunning the test from the paper is located in the paper directory (`paper/ConstraintViaSBP.jl`)
- Remark: the tutorial folder contains a notebook that runs on Google Colab and introduces the package step by step, up to tuning the constants of a model against a loss of your own


# How can I approximate functions involving vectors or matricies?
- `GepTensorRegressor` takes one column per feature -- scalars, `Tensors.jl` vectors and higher-order tensors side by side -- and a loss callback that scores each chromosome. The candidates are evaluated by the same batched evaluator, into buffers allocated once with `allocate_buffers!`.
- Hint: evaluating tensor-valued expressions costs more than evaluating scalar ones

 ```julia
using GeneExpressionProgramming
using LinearAlgebra
using Random
using Statistics
using Tensors

Random.seed!(789)

#create some testdata - testing simply on a few velocity vectors
n_samples = 300
x1 = fill(2.0, n_samples)
x2 = randn(n_samples)
u1 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]
u2 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]
u3 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]

a = [0.5 * u1[i] + x2[i] * u2[i] + 2.0 * u3[i] for i in 1:n_samples]

#define the regressor: 5 features, 3D tensors
regressor = GepTensorRegressor(5;
    problem_dimension=3,
    gene_count=3,
    head_len=4,
    entered_non_terminals=[:+, :-, :*],
    entered_terminal_nums=[0.5, 2.0],
    gene_connections=[:+, :-],
    feature_names=["x1", "x2", "U1", "U2", "U3"])

#one column per feature, in the order of feature_names; required before fit!
allocate_buffers!(regressor, (x1, x2, u1, u2, u3))

#the loss sets the fitness of a chromosome; predictT evaluates it in the preallocated buffers
#a candidate that cannot be scored gets a large penalty (Inf would do too: tournament selection leaves non-finite fitness values out)
function loss_new(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        pred = try
            predictT(regressor, elem.expression_raw)
        catch
            nothing
        end
        if pred isa AbstractVector && length(pred) == n_samples && eltype(pred) <: Tensor{1,3}
            elem.fitness = (mean(norm(pred[i] - a[i])^2 for i in 1:n_samples),)
        else
            elem.fitness = (1e6,)  # e.g. a scalar where a vector belongs
        end
    end
end

fit!(regressor, 60, 600, loss_new)

#Print the best expression
lsg = regressor.best_models_[1]
println(print_karva_strings(lsg))

#predict on new data: one column per feature
#pred_new = predictT(regressor, lsg.expression_raw, Any[x1_new, x2_new, u1_new, u2_new, u3_new])
```

# Custom losses
`fit!(regressor, epochs, population_size, loss)` hands the chromosomes to `loss(elem, validate::Bool)`, which sets `elem.fitness` to a tuple -- one entry per objective (`GepRegressor(n; number_of_objectives=2)` selects with NSGA-II). The loss runs inside the threaded fitness loop; build one evaluation context per thread once, and evaluate the chromosome in the calling thread's. An error thrown by the loss stops `fit!`, so catch what the evaluation may throw:

```julia
using Statistics

# number_features, x_data, y_data, epochs and population_size as in the first example
regressor = GepRegressor(number_features; number_of_objectives=2)
ctxs = thread_contexts(regressor.toolbox_, x_data')      # x_data': one row per feature

function loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        y_pred = try
            elem(ctxs[Threads.threadid()])
        catch                          # e.g. a DomainError from the log of a negative number
            nothing
        end
        elem.fitness = y_pred isa AbstractVector && all(isfinite, y_pred) ?
                       (mean(abs2, y_data .- y_pred), 0.01 * length(elem.expression_raw)) :
                       (Inf, Inf)
    end
end

fit!(regressor, epochs, population_size, loss)
```

Survivors and `best_models_` are ranked by the mean of the objectives, hence the size is scaled by 0.01: keep the objectives on comparable scales.

# Expensive losses: surrogate screening
When a loss call is expensive -- a solver in the loop, a simulation -- most calls of an epoch are spent on individuals that are visibly not worth one. A `SurrogateScreening` passed to `fit!` as `surrogate` spends the loss on the promising ones only (the Julia counterpart of `gep.SurrogateBatchStrategy` in the Python package):

1. every new individual is *embedded*: its expression is evaluated on a small, fixed probe set, and the (asinh-transformed) outputs are its latent vector, so individuals that behave alike lie close together however they are written;
2. a Gaussian process maps the latent vectors of the individuals scored so far to their (log10) losses, and an acquisition -- by default the optimistic bound `mean - kappa * deviation` -- picks the individuals the loss scores, 10 per epoch by default, a tenth of them at random instead, to explore;
3. every other individual gets the prediction of the process, strictly worse than the best loss scored so far, and competes with it.

Over one direction of the latent space, and for a search with two expressions and two objectives, it works as follows (the Manim source is `images/surrogate_screening.py`):
![Surrogate screening](images/surrogate_screening.gif)

```julia
using Random

# regressor, loss (an expensive loss(elem, validate)), x_data, epochs and population_size
# as in the examples above
probes = x_data'[:, randperm(size(x_data, 1))[1:36]]    # a few dozen states, one row per feature
surrogate = SurrogateScreening(regressor, probes;        # embeds with a SemanticEmbedder
    individuals_per_epoch=0.15)                          # a count, or a share of the new individuals

fit!(regressor, epochs, population_size, loss; surrogate=surrogate)

surrogate.evaluated_count     # loss calls made
surrogate.imputed_count       # predictions handed out instead
surrogate.spearman_log        # rank correlation of prediction and loss, per screened batch
```

With several objectives, and a chromosome that carries several expressions -- e.g. the right-hand sides of a system of ODEs, one objective per equation -- the screening embeds one block per expression and ties the process of every objective to the expression it judges:

```julia
# the loss splits every chromosome into two expressions, f, g = split_predict(elem, ctx, 2),
# and sets one objective per expression; the regressor has gene_count=4 and
# number_of_objectives=2
surrogate = SurrogateScreening(regressor, probes;
    expressions=2,                  # one latent block per expression
    objective_expressions=[1, 2],   # the process of objective j sees expression j alone
    individuals_per_epoch=0.15)

fit!(regressor, epochs, population_size, loss; surrogate=surrogate, hof=10)

best = regressor.best_models_                                      # all scored by the loss
front = best[calculate_fronts([m.fitness for m in best])[1]]      # the non-dominated ones
split_equations.(front, 2)                                         # each as two equations
```

The documentation walks through both cases (`docs/src/examples/surrogate-screening.md`), with what the screening guarantees, its settings and its diagnostics.

- A prediction is never cached: a copy of a predicted individual is screened again, the best individual of an epoch is scored before it is recorded, and `best_models_` holds scored individuals only (`is_validated(surrogate, model)`). With several objectives a prediction is also provisional: the individuals that carry one are screened again every epoch, next to the new ones, so a later process can pick them for a loss call (`rescreen`; with one objective it was a wash on the benchmark below and is off by default).
- During the warmup, until six times the batch (at least 60) individuals have been scored, an epoch scores at most twice the batch (at least 20), picked at random (by Latin hypercube sampling over the latent vectors with `warmup_lhs=true`), and the others get the median loss of that batch.
- Individuals that cannot be evaluated on every probe are scored as a crash without a loss call. Once the loss has failed (returned a non-finite value) 5 times, a model of which individuals the loss can score gates the batch.
- From 10 individuals per epoch on (read on a hundred individual epoch), each epoch breeds three times the children the population takes, and the process picks the ones that enter.
- `SurrogateScreening(regressor, probes; embedding=:genes)` embeds one block per gene (`GeneEmbedder`), the embedding for a least-squares combination of the genes (`linear_scaling`). A `GepTensorRegressor` is embedded with a `TensorEmbedder` on probe columns like those given to `allocate_buffers!` (`SurrogateScreening(regressor, probes; components=3)` for a vector-valued target). Any function `chromosome -> latent vector` (or `nothing`) works as an embedder: `SurrogateScreening(embedder; ...)`.
- Several objectives are screened by a Gaussian process per objective and a random Chebyshev scalarization for the pick (ParEGO), or by the expected hypervolume improvement (`screen=GpScreen(acquisition=:ehvi)`); a prediction is clamped behind the best scored value of every objective, so it dominates none of the individuals holding them, and these survive even where predictions beat them on the mean fitness the population is ranked by. An objective the loss computes from the chromosome alone, such as a size, is computed instead of predicted: `exact_objectives=Dict(2 => c -> 0.01 * length(c.expression_raw))`.
- A chromosome that carries several expressions (split by `split_karva` in the loss; `split_predict(chromosome, ctx, k)` evaluates them, `split_equations(chromosome, k)` prints them) is embedded with one block per expression: `SurrogateScreening(regressor, probes; expressions=2)`. Where each objective judges one of the expressions, `objective_expressions=[1, 2]` lets the process of an objective see the block of its expression alone, whose other blocks would only blur its distances. `Main_surrogate_multi_objective.jl` searches a system of two ODEs this way: one chromosome carries both right-hand sides, and every state is an objective of its own.
- The other knobs (`warmup_runs`, `warmup_batch`, `budget_rule=:uncertainty`, `explore_fraction`, `impute_beta`, `offspring_multiplier`, `target_transform`, `GpScreen(fit=true)`, ...) are documented in `?SurrogateScreening` and `?GpScreen`; the defaults are the ones the Python package measured.
- The screening costs a few hundredths to a few tenths of a second per epoch, growing with the archive of scored individuals (at most 1000) and the objectives: on one thread, 0.04 s with a batch of 10, 0.11 s with 15 % of the new individuals and 0.19 s for the two objectives of `Main_surrogate_multi_objective.jl`, against 0.01 s for the loop without it. It saves time where a loss call costs more than that over the individuals it spares; it is not meant for the plain data method with a cheap loss, although it works there too.

On three benchmark functions (10 seeds each, 200 individuals, 40 epochs, a custom MSE loss; `benchmark/surrogate_screening.jl`, output in `benchmark/surrogate_screening_results.txt`), screening 15 % of the new individuals beat scoring every individual at about 6.5 times fewer loss calls, and a batch of 10 per epoch beat a search without screening at the same number of loss calls on every function. Median MSE of the returned model:

| | loss calls | $x_1^2 + x_1 x_2 - 2x_2^2$ | Nguyen-4 | Nguyen-7 |
| --- | --- | --- | --- | --- |
| every individual scored | 5,165-5,681 | 7.9e-2 (1 exact) | 2.1e-2 | 1.0e-3 |
| screened, 15 % of the new individuals | 751-876 | 7.4e-3 (5 exact) | 8.1e-3 | 8.9e-4 |
| screened, 10 per epoch | 438-472 | 9.4e-2 (1 exact) | 1.4e-2 | 3.4e-3 |
| no screening, population 30, 20 epochs | 415-429 | 2.9e-1 | 9.9e-2 | 7.8e-3 |

(exact: MSE below 1e-10, of 10 seeds. The loss is cheap here, so these numbers speak to loss calls, not to time; `Main_surrogate_screening.jl` runs a search whose loss solves an ODE.)

# Expensive losses: the constants of a model
With a CFD simulation as the cost function, every trial value of the constants of a model costs a simulation. `ScreenedNelderMead` tunes them with few loss calls: Nelder-Mead whose calls a Gaussian process over the constants places (the default), plain Nelder-Mead (`screen=false`, the steps of `Optim.NelderMead()`), or a screened particle swarm in a box before it, for a loss with several minima (`swarm_box`).

```julia
# inside a search: every 5th epoch, if the best model improved, at most 30 loss calls; the
# loss evaluates with elem(ctx), split_predict(elem, ctx, k) or predictT(regressor, elem),
# which apply the tuned constants
fit!(regressor, epochs, population_size, loss;
    constant_optimizer=ScreenedNelderMead(max_evaluations=30), optimization_epochs=5)

# on their own: the constants of one model, or any function of a vector
optimize_constants!(model, loss; method=ScreenedNelderMead(max_evaluations=80))
simplex_search(p -> simulation_error(p), p0; method=ScreenedNelderMead(swarm_box=(lower, upper)))
```

| the loss | use | measured (median of 8 starts, 150 loss calls) |
| --- | --- | --- |
| smooth, and Nelder-Mead needs many steps | `ScreenedNelderMead()` | 1 % of the initial error after 14 to 35 calls on Lotka-Volterra calibrations of 2 to 4 coefficients, Nelder-Mead after 22 to 66 |
| easy, or a narrow, curved valley | `ScreenedNelderMead(screen=false)` | Nelder-Mead converges further within the same calls (logistic growth, Rosenbrock, a fictive CFD closure calibration) |
| several minima in a known range | `ScreenedNelderMead(swarm_box=(lower, upper))` | an oscillator: 1 % after 50 calls instead of 94 to 98, and the global minimum from 8 of 8 starts; 1.2 to 3.4 times the calls on smooth losses |

A run that diverges may return a large error or `Inf`: the Gaussian processes see the logarithm of the loss, so it only marks the region as bad. `examples/Main_fictive_cfd_in_the_loop.jl` puts it together: a turbulence closure searched with a fictive CFD solver in the loop (a 1D momentum balance standing in for CFD), screened, with two objectives and physical units, a quarter of whose runs diverge; with the script's seed it returns `nu_t = 0.00675 y^2 u_tau^2 / nu` after 4,234 fictive CFD runs, and then calibrates van Driest's mixing length within bounds that partly diverge. The documentation (`docs/src/examples/coefficient-tuning.md`) has the details, and `benchmark/screened_constants.jl` and `benchmark/swarm_variants.jl` the measurements.

# Engine for Symbolic Evaluation
- A batched evaluator written for this package: a stack machine walks the karva string once per candidate and applies each operator to whole data columns, writing into buffers allocated once per thread. When all terminals and intermediate values share one type (the scalar case), the operators are compiled into a monomorphic program, which avoids dynamic dispatch.
- The tensor path uses the same evaluator, with buffers per tensor type.


# References
- [1] Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Complex Systems, 13.
- [2] Reissmann, M., Fang, Y., Ooi, A. S. H., & Sandberg, R. D. (2025). Constraining genetic symbolic regression via semantic backpropagation. Genetic Programming and Evolvable Machines, 26(1), 12

# How to cite
Feel free to utilize it for your research, it would be nice __citing us__! Our [paper](https://doi.org/10.1007/s10710-025-09510-z).
```
@article{Reissmann2025,
  author   = {Maximilian Reissmann and Yuan Fang and Andrew S. H. Ooi and Richard D. Sandberg},
  title    = {Constraining Genetic Symbolic Regression via Semantic Backpropagation},
  journal  = {Genetic Programming and Evolvable Machines},
  year     = {2025},
  volume   = {26},
  number   = {1},
  pages    = {12},
  doi      = {10.1007/s10710-025-09510-z},
  url      = {https://doi.org/10.1007/s10710-025-09510-z}
}

```

# Todo 
- [ ] staggered exploration
- [ ] considering Tullio.jl for faster tensor ops
- [ ] LLM-interface
- [ ] Python-interface
- [ ] MOGA-2 implementation - alternative to NSGA-2
