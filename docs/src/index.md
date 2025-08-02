# GeneExpressionProgramming.jl Documentation

Welcome to the comprehensive documentation for GeneExpressionProgramming.jl, a powerful Julia package for symbolic regression using Gene Expression Programming (GEP). This documentation provides everything you need to get started with symbolic regression, from basic concepts to advanced applications.

## What is GeneExpressionProgramming.jl?

GeneExpressionProgramming.jl is a state-of-the-art symbolic regression package that combines the power of evolutionary algorithms with unique features like physical dimensionality constraints and tensor regression capabilities. It's designed for researchers, engineers, and data scientists who need to discover mathematical relationships in their data while maintaining physical plausibility and interpretability.

### Key Features

- **Gene Expression Programming**: Advanced evolutionary algorithm for symbolic regression  
- **Multi-Objective Optimization**: Balance accuracy, complexity, and other objectives using NSGA-II  
- **Physical Dimensionality**: Ensure dimensional consistency in evolved expressions  
- **Tensor Regression**: Work with vector and matrix data using Flux.jl backend  
- **High Performance**: Optimized Julia implementation with parallel processing support  
- **Scientific Applications**: Specialized features for physics, engineering, and scientific computing  

## Quick Start

Get up and running in minutes:

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
println(regressor.best_models_[1].compiled_function)
# Output: x1 * x1 + x2
```

## Documentation Structure

###  Getting Started
- **[Installation](installation.md)** - Install the package and dependencies
- **[Getting Started](getting-started.md)** - Your first symbolic regression project
- **[Core Concepts](core-concepts.md)** - Understand the theory behind GEP

### User Guide
- **[API Reference](api-reference.md)** - Complete function and type documentation

### Examples and Tutorials
- **[Basic Regression](examples/basic-regression.md)** - Fundamental symbolic regression workflow
- **[Multi-Objective Optimization](examples/multi-objective.md)** - Balance multiple objectives
- **[Physical Dimensionality](examples/physical-dimensions.md)** - Enforce dimensional consistency
- **[Tensor Regression](examples/tensor-regression.md)** - Work with vector and matrix data



## Why Choose GeneExpressionProgramming.jl?

### Unique Advantages

**Physical Dimensionality**: Unlike other symbolic regression packages, GeneExpressionProgramming.jl can enforce physical unit consistency, ensuring that evolved expressions respect fundamental physical laws. This is crucial for scientific and engineering applications.

**Multi-Objective Optimization**: Built-in support for balancing multiple objectives like accuracy vs. complexity using the proven NSGA-II algorithm. This helps you find interpretable models that don't overfit.

**Tensor Regression**: Native support for vector and matrix data through integration with Flux.jl, enabling discovery of relationships involving geometric and tensor quantities.

**Scientific Rigor**: Developed with features specifically designed for scientific computing and discovery.

### Performance Benefits

- **Julia Performance**: Leverages Julia's speed for fast evolution and expression evaluation
- **Parallel Processing**: Automatic parallelization across available CPU cores
- **Memory Efficiency**: Optimized data structures and algorithms
- **Scalability**: Handles large datasets and complex expressions

### Ease of Use

- **Simple API**: Intuitive interface that's easy to learn
- **Comprehensive Documentation**: Examples and tutorials
- **Active Development**: Regular updates
- **Integration**: Works well with the Julia ML ecosystem

## Research Foundation

GeneExpressionProgramming.jl is based on research in symbolic regression and evolutionary computation. The package implements novel techniques for constraining genetic symbolic regression via semantic backpropagation, as described in:

> Reissmann, M., Fang, Y., Ooi, A. S. H., & Sandberg, R. D. (2025). Constraining genetic symbolic regression via semantic backpropagation. *Genetic Programming and Evolvable Machines*, 26(1), 12.

This research introduces innovative methods for ensuring that evolved expressions respect physical constraints and dimensional consistency, making the package particularly valuable for scientific applications. Moreover, the implementation builds on top of concepts explored and developed in:

> Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Complex Systems, 13.

> K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, (2002) "A fast and elitist multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197 

> Weatheritt, J., Sandberg, R. D. (2016)  A novel evolutionary algorithm applied to algebraic modifications of the RANS stress–strain relationship. Journal of Computational Physics, vol. 325, pp. 22-37

> Waschkowski, F., Zhao, Y., Sandberg R. D., Klewicki J., (2022), Multi-objective CFD-driven development of coupled turbulence closure models. Journal of Computational Physics, vol. 452, 

## Community and Support

### Getting Help

- **Documentation**: Start with this comprehensive documentation
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences


### Contributing

We welcome contributions from the community! Whether you're:
- Fixing bugs or adding features
- Improving documentation
- Sharing examples and tutorials
- Providing feedback and suggestions


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

This documentation covers GeneExpressionProgramming.jl version 0.5.0 and later. The package is actively developed, with regular updates and improvements. Check the [GitHub repository](https://github.com/maxreiss123/GeneExpressionProgramming.jl) for the latest version and release notes.


## Acknowledgement
 - We employ the insane fast [DynamicExpressions.jl](https://github.com/SymbolicML/DynamicExpressions.jl) for evaluating our expressions
---
