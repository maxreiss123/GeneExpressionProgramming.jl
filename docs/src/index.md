# GeneExpressionProgramming.jl Documentation

GeneExpressionProgramming.jl is a Julia package for symbolic regression by Gene Expression Programming (GEP): it evolves explicit equations that fit data.

## Features

- **Gene Expression Programming**: a model is a chromosome of `Int8` tokens -- genes, each a head and a tail, joined by connectors. Its karva string is scored by a batched evaluator that applies every operator to whole data columns, in buffers allocated once per thread.
- **Multi-objective optimisation**: a custom loss can score several objectives (accuracy, size, ...); parents are then selected by NSGA-II.
- **Physical dimensions**: given the SI dimensions of the features and of the target, semantic backpropagation (SBP) repairs candidates towards the target dimension, and only homogeneous candidates are scored.
- **Tensor regression**: scalar, vector and tensor features (Tensors.jl) side by side, through the same batched evaluator.
- **Constant optimisation and linear scaling** (`fit!` on data): Nelder-Mead tuning of the constants of the best model, or one least-squares coefficient per gene.
- **Surrogate screening of expensive losses**: Gaussian processes over the behaviour of the candidates decide which few individuals per epoch the loss scores, and predict the loss of the others (`SurrogateScreening`), for one or several objectives and for chromosomes that carry several expressions.
- **Constants against an expensive loss**: Nelder-Mead whose loss calls a Gaussian process over the constants places where it expects the minimum (`ScreenedNelderMead`), after a screened particle swarm in a box for a loss with several minima (`swarm_box`), for the constants of the best model inside a search (`constant_optimizer` of `fit!`) or on their own (`optimize_constants!`, `simplex_search`), e.g. with a CFD simulation as the cost function.
- **Threads**: fitness evaluation, genetic operators and repair run on the threads Julia is started with (`julia --threads=auto`).

## Quick Start

```julia
using Pkg
Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git")

using GeneExpressionProgramming
using Random

# Generate sample data
Random.seed!(42)
x_data = randn(100, 2)
y_data = @. x_data[:,1]^2 + x_data[:,2]

# Create and train regressor
regressor = GepRegressor(2)
fit!(regressor, 1000, 1000, x_data', y_data; loss_fun="mse")

# View the discovered expression
println(regressor.best_models_[1])
# e.g. ((x1 * x1) + x2) -- the exact form varies from run to run
```

## Documentation Structure

- **[Installation](installation.md)**: install the package and start Julia with threads
- **[Getting Started](getting-started.md)**: a first symbolic regression, step by step
- **[Core Concepts](core-concepts.md)**: chromosomes, genetic operators, selection, dimensions and the evaluator
- **[API Reference](api-reference.md)**: functions, types and keyword arguments
- **Examples**:
  - **[Basic Regression](examples/basic-regression.md)**: the complete workflow on a known function
  - **[Multi-Objective Optimization](examples/multi-objective.md)**: accuracy against expression size
  - **[Physical Dimensionality](examples/physical-dimensions.md)**: a search held to physical units
  - **[Tensor Regression](examples/tensor-regression.md)**: a vector-valued target from scalar and vector features
  - **[Surrogate Screening](examples/surrogate-screening.md)**: expensive losses (a solver in the loop), with one objective and with several expressions and objectives
  - **[Coefficient Tuning](examples/coefficient-tuning.md)**: the constants of a model against an expensive loss, inside a search and on their own, and a closure model searched with a fictive CFD solver in the loop

## Research Foundation

The package implements the constraint of genetic symbolic regression by semantic backpropagation described in:

> Reissmann, M., Fang, Y., Ooi, A. S. H., & Sandberg, R. D. (2025). Constraining genetic symbolic regression via semantic backpropagation. *Genetic Programming and Evolvable Machines*, 26(1), 12.

It builds on concepts explored and developed in:

> Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Complex Systems, 13.

> K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, (2002) "A fast and elitist multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197 

> Weatheritt, J., Sandberg, R. D. (2016)  A novel evolutionary algorithm applied to algebraic modifications of the RANS stress–strain relationship. Journal of Computational Physics, vol. 325, pp. 22-37

> Waschkowski, F., Zhao, Y., Sandberg R. D., Klewicki J., (2022), Multi-objective CFD-driven development of coupled turbulence closure models. Journal of Computational Physics, vol. 452, 

## Community and Support

Report bugs, request features and ask questions on the [GitHub repository](https://github.com/maxreiss123/GeneExpressionProgramming.jl/issues). Contributions -- fixes, features, documentation and examples -- are welcome.

### Citation

If you use GeneExpressionProgramming.jl in your research, please cite:

```bibtex
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

## Version Information

These pages describe the current code in the [GitHub repository](https://github.com/maxreiss123/GeneExpressionProgramming.jl), which may be ahead of the latest registered release.


## Acknowledgement
 - Tensor-valued features and operations build on [Tensors.jl](https://github.com/Ferrite-FEM/Tensors.jl).
 - Earlier versions evaluated expressions with [DynamicExpressions.jl](https://github.com/SymbolicML/DynamicExpressions.jl); the current evaluator is a batched stack machine written for this package.
---
