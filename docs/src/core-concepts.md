# Core Concepts

This chapter describes how GeneExpressionProgramming.jl represents, evaluates and evolves models, and what its settings control.

## Gene Expression Programming Fundamentals

### Background

Gene Expression Programming (GEP) was introduced by Cândida Ferreira in 2001 as an evolutionary algorithm that combines ideas of genetic algorithms and genetic programming [1]. It separates the genotype, a fixed-length linear chromosome, from the phenotype, the expression the chromosome encodes, whose size and shape vary. The genetic operators edit the linear chromosome and, except for the fusion operators (off by default), keep every symbol valid for its position, so every chromosome encodes a valid expression.

### Chromosomes and Genes

A chromosome is a vector of `Int8` symbols, laid out as:

- **Connectors**: `gene_count - 1` binary functions (from `gene_connections`) that join the genes
- **Genes**: `gene_count` genes, each a **head** of `head_len` symbols (functions and terminals) and a **tail** of `head_len + 1` symbols (terminals only)
- **Preamble**: optionally `gene_count` further terminals (`preamble_syms`), outside the expression

The tail length follows GEP's rule t = h × (n − 1) + 1 for head length h and maximum arity n; all functions here have arity 1 or 2, so t = h + 1. Even a head of binary functions then finds enough terminals in the tail, so every gene encodes a complete expression.

### Translation

Each gene is read from left to right in prefix order (depth-first, as in prefix GEP), rather than Ferreira's breadth-first K-expression: a function takes the subexpressions that follow it as its operands, and reading stops once every function has its operands. The symbols read form the gene's active part; the rest of the gene is inactive until a change in the head extends the active part. Prefix order keeps every subexpression contiguous in the gene, which is what lets the unit repair replace a subexpression in place.

The connectors, followed by the active part of each gene, form the chromosome's **karva string** (`expression_raw`), which is what the evaluator runs. A chromosome with genes G1, G2, G3 and connectors c1, c2 encodes c1(c2(G1, G2), G3).

### Evaluation

GeneExpressionProgramming.jl never builds an expression tree. The **batched evaluator** walks the karva string as a stack machine, from right to left, and applies every operator to whole data columns at once, writing into buffers allocated once per thread. When all terminals and intermediate values share one type, as in scalar regression, the alphabet is compiled into a monomorphic program (`compile_program`, `run_program!`) that avoids dynamic dispatch; otherwise `calc_stack_batch_tensor` evaluates with dynamic dispatch.

## Evolutionary Operators

### Selection

**Tournament selection** is used for a single objective: each parent is the best of `k` contenders drawn at random, with replacement. Only individuals with a finite fitness take part, and individuals with identical fitness count once; if no fitness is finite, the whole population takes part. The best individual is always among the parents. Larger tournaments raise the selection pressure; `fit!` uses 3 % of the population for `GepRegressor` and 0.3 % for `GepTensorRegressor`, at least 3.

**NSGA-II** is used with more than one objective (a fitness tuple of more than one entry, as `number_of_objectives > 1` sets up): the population is sorted into Pareto fronts, and parents are chosen by tournaments of 3 that prefer the lower front and, within a front, the larger crowding distance. There is no finiteness filter here; instead, a fitness tuple with more non-finite entries (`Inf`, `NaN`) is ranked behind one with fewer (`dominates_`).

### Genetic Operators

All operators edit the linear chromosome. Their probabilities and rates are the entries of `RegressionWrapper.GENE_COMMON_PROBS` (defaults in the [API Reference](api-reference.md#Genetic-Operators)). Crossover and fusion act on a pair of parents; the other operators act on each offspring with its own draw.

**Crossover** between two parents:
- **One-point crossover** (`one_point_cross_over_prob`): in every gene, each offspring keeps one random segment of its own symbols, from a position among the gene's first `head_len + 1` to one among its last `head_len`, and takes all other symbols (connectors and preamble included) from the other parent
- **Two-point crossover** (`two_point_cross_over_prob`): the same with other segments: one offspring keeps two segments of each gene, the other a random suffix of each gene
- **Fusion operators** (`dominant_fusion_prob`, `rezessiv_fusion_prob`, `fusion_prob`): at randomly drawn positions, an offspring takes the larger, the smaller or the mean of the two parents' symbol ids, which need not be valid there (off by default)

**Mutation and rearrangement**:
- **Point mutation** (`mutation_prob`, `mutation_rate`): `round(mutation_rate × length)` positions, drawn with replacement, take the symbol at the same position of a freshly generated chromosome, so each keeps a symbol valid for its place (connector, head or tail)
- **Inversion** (`inversion_prob`): reverses a run of head symbols. As implemented, it reverses positions `gene_count` to `head_len` of the chromosome -- part of the first gene's head -- when the first gene is drawn, and nothing otherwise
- **Insertion** (`insertion_prob`): writes a random terminal into a random head position
- **Root insertion** (`root_insertion_prob`): rotates a gene's head by a random offset, so another head symbol becomes the gene's root
- **Tail rotation** (`reverse_insertion_tail`): rotates a gene's tail, without its first symbol, by a random offset (off by default)
- **Gene transposition** (`gene_transposition_prob`): swaps two segments of `min(5, head_len + 1)` symbols between the tails of two genes, possibly the same gene

**Gene averaging** (`gene_averaging_prob`, `gene_averaging_rate`, `gene_averaging_elite_frac`): pulls an offspring towards the consensus of an elite: the best-ranked individuals, `gene_averaging_elite_frac` times the mating size in number (at least 3). The consensus draws, per position, one of the three symbols most frequent there among the elite, with probability proportional to its count, and each position of the offspring takes it with probability `gene_averaging_rate`.

### Population Dynamics

Each epoch, `m` offspring are bred, `m` being `ceil(mating_size × population_size)` (`mating_size` is 0.7 by default), less one if odd. The population holds `population_size + m` individuals, ranked by the mean of their fitness tuples. The offspring take ranks `population_size - m` to `population_size - 1`; the individuals they displace move behind the first `population_size`, where they are neither scored nor selected but are ranked again in the next epoch, and those previously there are dropped. The best `population_size - m - 1` individuals thus survive unchanged (elitism).

**Duplicate penalty**: fitness is cached by karva string in an LRU cache of 10,000 entries. A new individual whose karva string is cached, or already queued for scoring in the same epoch, is not evaluated (unless its entry has been evicted): it gets the cached fitness multiplied by `penalty` (`fit!`, 2.0 by default). For a non-negative loss this keeps copies of one solution from taking over the population; with a loss that can be negative, a penalty above 1 favours them.

## Multi-Objective Optimization

### Pareto Optimality

For symbolic regression, typical objectives are the prediction error and the size of the expression. A solution dominates another if it is no worse in every objective and better in at least one; the Pareto-optimal solutions are those no other solution dominates. NSGA-II selects parents by Pareto rank.

### Crowding Distance

Within a front, NSGA-II prefers individuals in less crowded regions of the objective space, measured per objective relative to the front's range, so the search spreads along the front instead of converging to one region.

### Ranking by the Mean

Survival and the hall of fame (`best_models_`) use the population ranked by the mean of each fitness tuple, not by Pareto rank. Objectives on very different scales therefore let the larger one dominate that ranking; bring them to comparable scales, and pick the non-dominated models out of `best_models_` with `dominates_` (see [Multi-Objective Optimization](examples/multi-objective.md)).

## Physical Dimensionality and Semantic Backpropagation

### Dimensional Analysis

Semantic backpropagation (SBP) [2] holds evolved expressions to physical units.

**Dimensional representation**: a dimension is a vector of SI exponents, in the order [kg, m, s, K, mol, A, cd] -- mass, length, time, temperature, amount of substance, electric current, luminous intensity (the order OpenFOAM uses). Velocity is `Float16[0, 1, -1, 0, 0, 0, 0]`. On the tensor path the vector carries the tensor order in front of the SI exponents.

**Forward propagation**: the dimension of an expression is computed bottom-up from the dimensions of its terminals, following the rules of dimensional analysis:
- Addition, subtraction, `min`, `max`: the operands must have the same dimension
- Multiplication: dimensions are added
- Division: dimensions are subtracted
- Square root: dimensions are halved; `sqr` doubles them
- Transcendental functions (`exp`, `log`, `sin`, ...): the argument must be dimensionless, and so is the result
- `abs`, `floor`, `ceil`, `round`, `sign`: the dimension is kept
- Power `^`: only between dimensionless operands (use `sqr` and `sqrt` for dimensioned ones)

An expression that breaks a rule is inconsistent; it is *homogeneous* when its dimension equals the target's.

**Backward propagation**: the rules also run top-down: given the dimension a node must have, they give the dimensions its operands must have (for a product, one operand's requirement follows from the other's dimension). This is what lets a requirement be pushed from the root of an expression to the subexpression that violates it.

### The Library

A regressor built with `considered_dimensions` builds a **library** of dimensionally consistent subexpressions from the features, the non-zero constants and the functions: each of `rounds` rounds extends the expressions by one symbol and keeps at most `max_permutations_lib` new ones, so library expressions have up to `rounds + 1` symbols. The library is indexed by dimension: an exact lookup for its own dimensions, and a kd-tree over those and the dimensions one more operator reaches from them (a product or quotient of two library expressions, on the scalar path, or a unary function of one).

### Repair During the Search

With a target dimension (`fit!(...; target_dimension=...)`), every epoch runs two passes before the new individuals are scored:

1. **Check**: every new individual gets the forward check, in parallel. Individuals born homogeneous are flagged and need nothing else.
2. **Repair**: the others are repaired in place (`correct_genes!`), every `correction_epochs` epochs and at most `correction_amount` times the population size of them. The target is pushed down the expression; at each node the cheapest fix that meets the requirement is taken -- swapping a terminal or an operator for one of the right dimension, dropping a unary function, splitting the requirement between the operands of a binary node, or replacing the subexpression with one of the needed dimension from the library. A repair only writes what the genetic operators could have written: operators stay in a gene's head, genes stay within their length, and connectors are only swapped for other connectors. A repaired individual is flagged only once its recompiled expression passes the check of step 1.

Only homogeneous individuals are scored; an individual that is neither born homogeneous nor repaired gets the worst fitness (`Inf`). Part of the initial population (`lib_seed_amount`, half by default) is seeded with library expressions: if some connectors keep the dimension of their operands (`+`, `-`, `min`, `max`), the seeded genes all take the target dimension and are joined by those; otherwise the first gene takes it and the others are dimensionless. An individual is seeded only if the library can fill each of its genes.

Since nearly every offspring passes the repair, neutral variants of the current best -- the same predictions from a different karva string, such as a model multiplied by zero -- could fill the population. Under a target, individuals whose fitness exactly repeats that of a better-ranked one are therefore ranked behind all distinct individuals when the survivors are chosen.

With `linear_scaling=true` the model is the weighted sum of its genes, whose coefficients are dimensionless only if every gene has the target dimension. The check and the repair then go gene by gene (`is_gene_wise_homogeneous`, `correct_genes!` with `gene_wise=true`), and with `+` or `-` among the connectors the connected expression takes the target too. A custom loss that scores the genes' least-squares combination (`predictT_scaled`) asks for the same with `gene_wise_dimension=true`.

## Tensor Operations and Advanced Data Types

### Tensor Regression

For problems involving vector or tensor data, `GepTensorRegressor` evolves expressions over columns of scalars and Tensors.jl vectors and tensors side by side; `problem_dimension` sets the spatial dimension of the tensors. Besides the scalar functions it offers:

**Products**: `*` (products with a scalar), `dot` (single contraction), `dcontract` (double contraction), `otimes` (outer product), `crossp` (cross product of 3D vectors), `hadamard` (element-wise product)

**Invariants and norms**: `tr` (trace), `det` (determinant), `norm`

**Parts and derived tensors**: `symmetric`, `skew`, `dev` (deviatoric part), `vol` (volumetric part), `inv` (inverse), `tdot` and `dott` (both A·Aᵀ), `lap` (contraction of a third-order tensor with the identity)

A chromosome that combines operands of incompatible orders cannot be evaluated (the evaluator returns `NaN`, a column of `Inf`, or throws); the loss callback gives it a penalty (see [Tensor Regression](examples/tensor-regression.md)). The dimension vectors of the tensor path carry the tensor order in front of the SI exponents. With `considered_dimensions`, only the functions that have tensor unit rules can be entered (any other makes the constructor throw): `+`, `-`, `*`, `/`, `inv`, `dot`, `crossp`, `tr`, `det`, `dcontract`, `lap`, `hadamard`, `sqrt`, `norm`, `log`, `exp`, `sin` and `cos`.

### Performance Considerations

Tensor operations cost more than scalar ones. The evaluator applies every operator to a whole column of samples at once, into buffers allocated once per thread and per tensor type (`allocate_buffers!`), so evaluating a candidate does not allocate arrays the size of the data. GPU evaluation is not implemented.

## Loss Functions and Fitness Evaluation

### Standard Loss Functions

Losses are selected by name (`fit!(...; loss_fun="mse")`, or `get_loss_function(name)`):

- `"mse"` Mean Squared Error
- `"mae"` Mean Absolute Error: less sensitive to outliers
- `"rmse"` Root Mean Squared Error: in the units of the target
- `"nrmse"` RMSE divided by the standard deviation of the target, and `"srsme"`, a relative RMSE (each residual divided by the magnitude of the target)
- `"r2_score"`, `"r2_score_f"` (the same, computed on data rescaled by a power of ten for very large or small magnitudes) and `"xi_core"` (a rank correlation modelled on Chatterjee's ξ): scores rather than losses, for reporting -- the search minimises its loss

### Custom Loss Functions

Any function of the targets and the predictions that returns a number to minimise can be the loss:

```julia
# regressor, x_train, y_train, epochs and population_size as in Getting Started
# the largest absolute error
max_abs_error(y_true, y_pred) = maximum(abs.(y_true .- y_pred))

fit!(regressor, epochs, population_size, x_train', y_train; loss_fun=max_abs_error)
```

In the more general form the loss receives the chromosome itself and sets its fitness, so the evaluation is up to the user: `fit!(regressor, epochs, population_size, loss)` with `loss(elem, validate::Bool)`. This is the form for several objectives and for tensor regression.

### Fitness Evaluation Strategies

**Single-objective**: one loss, tournament selection

**Multi-objective**: the loss sets a tuple with one entry per objective; NSGA-II selection

**Weighted sum**: a single-objective loss that combines several criteria with weights of your choice

**Gene-wise linear scaling**: `fit!(...; linear_scaling=true)` scores each chromosome as the least-squares combination of its genes, one coefficient per gene (stored in `scaling_weights`), so evolution searches for the structure and the coefficients are solved for. The constant optimiser does not run with it.

**Surrogate screening**: when a loss call is expensive (a solver in the loop, a simulation), `fit!(...; surrogate=SurrogateScreening(regressor, probes))` lets the loss score only a few individuals per epoch. Each new individual is embedded as the behaviour of its expression on a small probe set, a Gaussian process maps these latent vectors to the losses scored so far, and an optimistic confidence bound picks the individuals worth a loss call; the others receive the prediction of the process, never better than the best loss scored so far, and compete with it. Predictions are never cached, the best individual of every epoch and the returned models are scored by the loss, and an epoch breeds more children than the population takes so that the process can pick among them. Several objectives get one process each; a chromosome that carries several expressions (`split_karva`) is embedded with one block per expression (`expressions = k`), and `objective_expressions` lets the process of an objective see the block of the expression it judges alone. The example [Surrogate Screening](examples/surrogate-screening.md) walks through both cases; the [API Reference](api-reference.md#Surrogate-Screening) lists the options.

**Constants against an expensive loss**: with a custom loss, `fit!(...; constant_optimizer=ScreenedNelderMead())` tunes the constants of the best model against the loss every `optimization_epochs` epochs, by Nelder-Mead whose loss calls a Gaussian process over the constants places where it expects the minimum (`screen=false` for plain Nelder-Mead, with the steps of `Optim.NelderMead()`). `optimize_constants!` tunes the constants of one chromosome, `simplex_search` minimizes any function of a vector, e.g. the coefficients of a closure in a simulation. For a loss with several minima, `swarm_box` lets a particle swarm screened by a process explore a box before. The tuned values, one per occurrence of a constant, are kept in `optimised_constants`, which `elem(ctx)`, `split_predict` and `predictT(regressor, chromosome)` apply. See [Coefficient Tuning](examples/coefficient-tuning.md) and the [API Reference](api-reference.md#Constants-Against-an-Expensive-Loss).

## Algorithm Parameters

The defaults are a starting point; no setting is best for every problem [4].

| Parameter | Where | Default | Effect |
| --- | --- | --- | --- |
| population size, epochs | `fit!` arguments | -- | search effort |
| `head_len` | `GepRegressor` | 6 | maximum size of each gene's expression |
| `gene_count` | `GepRegressor` | 3 | number of subexpressions joined by connectors |
| `entered_non_terminals` | `GepRegressor` | `[:+, :-, :*, :/]` | function set |
| `mutation_rate`, crossover probabilities, ... | `GENE_COMMON_PROBS` | see [API Reference](api-reference.md#Genetic-Operators) | variation |
| `mating_size` | `GENE_COMMON_PROBS` | 0.7 | offspring per epoch, as a fraction of the population |
| `penalty` | `fit!` | 2.0 | factor on the fitness of a duplicate |

Longer heads and more genes allow larger expressions at a higher evaluation cost.

## Performance

**Batched evaluation**: a candidate is evaluated in one pass over its karva string, each operator applied to whole data columns; scalar alphabets run as a compiled monomorphic program.

**Parallelism**: on the threads Julia is started with, the population is generated, scored, varied and repaired in parallel, and the library is built in parallel. Scoring hands the individuals out one at a time to whichever thread is free, so uneven evaluation costs do not leave threads idle. The `"mse"` loss also splits the samples across threads from 100,000 samples on.

**Memory**: chromosomes are vectors of `Int8` symbols, and evaluation writes into buffers allocated once per thread, whose size grows with the data (see `examples/Main_streaming_chunks.jl` for data larger than memory).

**Caching**: fitness is cached per karva string, so a duplicate is not evaluated again while its karva string is in the cache.

### Complexity

**Time**: evaluation usually dominates, at O(E × P × L × N) for E epochs, P individuals, karva strings of average length L and N samples. With a dimensional target, the repair adds a cost per individual that does not grow with N.

**Space**: O(P × S) for a population of chromosomes of S symbols, O(F × N) for F features, and O(T × g × h × N) for the evaluation buffers of T thread slots, g genes and head length h.

## Background

Holland's schema theorem [3] is the classical account of why selection and recombination propagate building blocks of above-average fitness; it was derived for genetic algorithms on fixed-length strings, such as GEP's chromosomes. The no-free-lunch theorems [4] state that no algorithm, and no setting of one, is best on all problems.

## References

[1] Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Complex Systems, 13.

[2] Reissmann, M., Fang, Y., Ooi, A. S. H., & Sandberg, R. D. (2025). Constraining genetic symbolic regression via semantic backpropagation. Genetic Programming and Evolvable Machines, 26(1), 12.

[3] Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence*. University of Michigan Press.

[4] Wolpert, D. H., & Macready, W. G. (1997). No Free Lunch Theorems for Optimization. IEEE Transactions on Evolutionary Computation, 1(1), 67–82.

---

*For the functions and keyword arguments, continue to the [API Reference](api-reference.md).*
