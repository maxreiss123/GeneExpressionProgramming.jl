# Core Concepts

This chapter provides a comprehensive overview of the theoretical foundations and key concepts underlying GeneExpressionProgramming.jl. Understanding these concepts will help you make informed decisions about parameter settings, interpret results effectively, and apply the package to complex real-world problems.

## Gene Expression Programming Fundamentals

### Historical Context and Evolution

Gene Expression Programming (GEP) was introduced by Cândida Ferreira in 2001 as an evolutionary algorithm that combines the advantages of both genetic algorithms and genetic programming [1]. GEP addresses several limitations of traditional genetic programming, particularly the issues of closure and structural complexity that can hinder the evolution of mathematical expressions.

The key innovation of GEP lies in its unique representation scheme that separates the genotype (linear chromosome) from the phenotype (expression tree). This separation allows for more efficient genetic operations while maintaining the expressiveness needed for symbolic regression tasks. Unlike genetic programming, which directly manipulates tree structures, GEP operates on fixed-length linear chromosomes that are then translated into expression trees of varying sizes and shapes.

### Genotype-Phenotype Mapping

The fundamental concept in GEP is the translation from linear chromosomes (genotype) to expression trees (phenotype). This process involves several key components:

**Chromosomes**: In GEP, a chromosome consists of one or more genes, each representing a sub-expression. Each gene has a fixed length and is divided into a head and a tail. The head can contain both functions and terminals, while the tail contains only terminals.

**Head and Tail Structure**: The head length (h) determines the maximum complexity of the expression, while the tail length is calculated as t = h × (n-1) + 1, where n is the maximum arity of the functions in the function set. This ensures that any expression can be properly formed regardless of the arrangement of functions and terminals.

**Translation Process**: The linear chromosome is read from left to right, and the expression tree is constructed using a breadth-first approach. Functions require operands according to their arity, and the translation continues until all function nodes have been satisfied.

### Expression Trees and Evaluation

Once translated from the linear chromosome, the expression tree represents the mathematical formula that will be evaluated. The tree structure follows standard mathematical conventions:

- **Internal nodes** represent functions (operators)
- **Leaf nodes** represent terminals (variables or constants)
- **Evaluation** proceeds from the leaves to the root using standard mathematical precedence

The expression tree evaluation in GeneExpressionProgramming.jl is optimized using DynamicExpressions.jl, which provides fast compilation and evaluation of symbolic expressions. This optimization is crucial for the performance of the evolutionary algorithm, as fitness evaluation typically dominates the computational cost.

## Evolutionary Operators

### Selection Mechanisms

GeneExpressionProgramming.jl implements several selection mechanisms to choose parents for reproduction:

**Tournament Selection**: The default selection method where k individuals are randomly chosen from the population, and the best among them is selected as a parent. Tournament size affects selection pressure - larger tournaments increase pressure toward fitter individuals.

**NSGA-II Selection**: For multi-objective optimization, the package implements the Non-dominated Sorting Genetic Algorithm II (NSGA-II), which maintains diversity while progressing toward the Pareto front. This is particularly useful when balancing accuracy against expression complexity.

**Roulette Wheel Selection**: Probabilistic selection based on fitness proportions, though this is less commonly used due to potential issues with fitness scaling.

### Genetic Operators

**Mutation**: GEP uses several mutation operators that work directly on the linear chromosome:
- **Point Mutation**: Randomly changes individual symbols in the chromosome
- **Inversion**: Reverses a sequence of symbols within a gene
- **IS Transposition**: Inserts a sequence from one location to another
- **RIS Transposition**: Similar to IS but with root insertion
- **Gene Transposition**: Moves entire genes within the chromosome

**Crossover**: Recombination operators that combine genetic material from two parents:
- **One-Point Crossover**: Exchanges genetic material at a single crossover point
- **Two-Point Crossover**: Uses two crossover points to exchange a middle segment
- **Gene Crossover**: Exchanges entire genes between parents
- **Uniform Crossover**: Exchanges individual symbols with a certain probability

**Recombination**: Higher-level operators that combine sub-expressions:
- **One-Point Recombination**: Exchanges sub-trees at randomly chosen points
- **Two-Point Recombination**: Exchanges sub-trees between two points

### Population Dynamics

The evolutionary process in GEP follows a generational model where the entire population is replaced in each generation, except for elite individuals that are preserved through elitism. The population dynamics are controlled by several parameters:

**Population Size**: Determines the number of individuals in each generation. Larger populations provide better exploration but require more computational resources.

**Generation Gap**: The proportion of the population replaced in each generation. A generation gap of 1.0 means the entire population (except elites) is replaced.

**Elitism**: The number of best individuals preserved from one generation to the next, ensuring that good solutions are not lost during evolution.

## Multi-Objective Optimization

### Pareto Optimality

In many real-world applications, there are multiple conflicting objectives that need to be optimized simultaneously. For symbolic regression, common objectives include:

- **Accuracy**: Minimizing prediction error (MSE, MAE, etc.)
- **Complexity**: Minimizing expression size or depth
- **Interpretability**: Favoring simpler, more understandable expressions

GeneExpressionProgramming.jl implements multi-objective optimization using NSGA-II, which maintains a diverse set of solutions along the Pareto front. A solution is Pareto optimal if no other solution exists that is better in all objectives simultaneously.

### Crowding Distance

To maintain diversity in the population, NSGA-II uses crowding distance as a secondary selection criterion. Individuals in less crowded regions of the objective space are preferred, encouraging the algorithm to explore the entire Pareto front rather than converging to a single region.

### Objective Function Design

When designing multi-objective problems, consider:

1. **Objective Scaling**: Ensure objectives are on comparable scales
2. **Objective Correlation**: Highly correlated objectives may not provide additional information
3. **Objective Conflict**: Objectives should represent genuine trade-offs

## Physical Dimensionality and Semantic Backpropagation

### Dimensional Analysis

One of the unique features of GeneExpressionProgramming.jl is its support for physical dimensionality constraints through semantic backpropagation [2]. This feature ensures that evolved expressions respect physical units and dimensional homogeneity.

**Dimensional Representation**: Physical dimensions are represented as vectors where each component corresponds to a fundamental unit (mass, length, time, electric current, temperature, amount of substance, luminous intensity).

**Forward Propagation**: During expression evaluation, dimensions are propagated forward through the expression tree according to the rules of dimensional analysis:
- Addition/Subtraction: Operands must have the same dimensions
- Multiplication: Dimensions are added
- Division: Dimensions are subtracted
- Power: Dimensions are multiplied by the exponent

**Backward Propagation**: When dimensional inconsistencies are detected, the algorithm can backpropagate corrections to ensure dimensional validity.

### Semantic Constraints

Semantic backpropagation goes beyond simple dimensional analysis to enforce more complex semantic constraints:

**Type Consistency**: Ensuring that operations are semantically meaningful (e.g., not taking the logarithm of a dimensioned quantity)

**Domain Constraints**: Restricting function domains (e.g., ensuring arguments to sqrt are non-negative)

**Physical Plausibility**: Enforcing constraints based on physical laws and principles

## Tensor Operations and Advanced Data Types

### Tensor Regression

For problems involving vector or matrix data, GeneExpressionProgramming.jl provides tensor regression capabilities through the `GepTensorRegressor`. This functionality is built on top of Flux.jl and supports:

**Vector Operations**: Element-wise operations, dot products, cross products, and norms

**Matrix Operations**: Matrix multiplication, transposition, determinants, and eigenvalue operations

**Tensor Contractions**: General tensor operations for higher-dimensional data

### Performance Considerations

Tensor operations are computationally more expensive than scalar operations, so several optimizations are implemented:

**Lazy Evaluation**: Operations are only computed when needed

**Batch Processing**: Multiple evaluations are batched together for efficiency

**GPU Acceleration**:  not yet implemented

## Loss Functions and Fitness Evaluation

### Standard Loss Functions

GeneExpressionProgramming.jl provides various regression losses:

- Mean Squared Error (MSE): Standard for continuous regression
- Mean Absolute Error (MAE): Robust to outliers
- Root Mean Squared Error (RMSE): Scale-dependent alternative to MSE
- Huber Loss: Combines MSE and MAE properties

### Custom Loss Functions

For specialized applications, you can define custom loss functions:

```julia
function custom_loss(y_true, y_pred)
    # Your custom loss calculation
    return loss_value
end
```

Moreover there is a more general wrapper function, where the user can define the evalulation strategy complete independent.

### Fitness Evaluation Strategies

**Single-Objective**: Direct optimization of a single loss function

**Multi-Objective**: Simultaneous optimization of multiple objectives using Pareto dominance

**Lexicographic**: Hierarchical optimization where objectives are prioritized

**Weighted Sum**: Linear combination of multiple objectives with user-defined weights

## Algorithm Parameters and Tuning

### Population Parameters

**Population Size**: Affects exploration vs. exploitation trade-off
- Small populations (50-200): Fast convergence, limited exploration
- Medium populations (200-1000): Balanced performance
- Large populations (1000+): Extensive exploration, slower convergence

**Number of Generations**: Determines total computational budget
- Early stopping can be used to prevent overfitting
- Monitor convergence to determine appropriate stopping criteria

### Genetic Parameters

**Mutation Rate**: Controls exploration intensity
- Low rates (0.01-0.05): Conservative exploration
- Medium rates (0.05-0.15): Balanced exploration
- High rates (0.15+): Aggressive exploration, risk of disruption

**Crossover Rate**: Controls exploitation of existing solutions
- Typically set between 0.6-0.9 for good performance
- Higher rates emphasize recombination over mutation

### Expression Parameters

**Head Length**: Determines maximum expression complexity
- Shorter heads: Simpler expressions, faster evaluation
- Longer heads: More complex expressions, higher computational cost

**Gene Count**: Number of sub-expressions per chromosome
- Single genes: Simple expressions
- Multiple genes: Complex, modular expressions (e.g different terms within the problem considered)

**Function Set**: Available operations for expression construction
- Basic arithmetic: +, -, *, /
- Transcendental: sin, cos, exp, log
- Specialized: domain-specific functions

## Performance Optimization

### Computational Efficiency

**Expression Compilation**: DynamicExpressions.jl compiles expressions for fast evaluation

**Parallel Evaluation**: Population evaluation can be parallelized across multiple cores

**Memory Management**: Efficient memory usage for large populations and long chromosomes

**Caching**: Intermediate results are cached to avoid redundant computations (cache blow up might remain an issue)

### Scalability Considerations

**Problem Size**: Algorithm complexity scales with:
- Population size
- Expression complexity
- Number of generations
- Dataset size

**Memory Requirements**: Dominated by population storage and expression evaluation

**Parallel Processing**: Multiple levels of parallelization available:
- Population-level: Evaluate individuals in parallel
- Expression-level: Parallel evaluation of sub-expressions
- Data-level: Parallel evaluation across data samples

## Theoretical Foundations

### Convergence Properties

GEP belongs to the class of evolutionary algorithms with the following theoretical properties:

**Schema Theorem**: Building blocks (schemas) with above-average fitness tend to receive exponentially increasing numbers of trials in subsequent generations [3]

**No Free Lunch Theorem**: No single algorithm performs best on all possible problems, emphasizing the importance of problem-specific tuning [4]

**Exploration vs. Exploitation**: The algorithm must balance exploring new regions of the search space with exploiting known good solutions


### Search Space Characteristics

**Representation Space**: The space of all possible linear chromosomes

**Phenotype Space**: The space of all possible expression trees

**Fitness Landscape**: The mapping from expressions to fitness values, which can be:
- Smooth: Gradual fitness changes with small modifications
- Rugged: Many local optima and fitness discontinuities
- Neutral: Large regions with similar fitness values

### Complexity Analysis

**Time Complexity**: O(G × P × L × N) where:
- G: Number of generations
- P: Population size (the one evaluated wihtin each generation)
- L: Average expression complexity (traversing the expression tree)
- N: Dataset size

**Space Complexity**: O(P × L) for population storage plus O(N) for dataset storage (assuming the )

## Advanced Topics

### Hybrid Approaches

GeneExpressionProgramming.jl can be combined with other optimization techniques:

**Local Search**: Gradient-based optimization of numerical constants

**Ensemble Methods**: Combining multiple evolved expressions

**Transfer Learning**: Using knowledge from related problems

### Domain-Specific Extensions

**Time Series**: Specialized operators for temporal data

**Image Processing**: Operators for spatial data and convolution

**Signal Processing**: Frequency domain operations and filtering

**Scientific Computing**: Domain-specific function sets for physics, chemistry, biology


## References

[1] Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Complex Systems, 13.

[2] Reissmann, M., Fang, Y., Ooi, A. S. H., & Sandberg, R. D. (2025). Constraining genetic symbolic regression via semantic backpropagation. Genetic Programming and Evolvable Machines, 26(1), 12.

[3] Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence*. University of Michigan Press.

[4] Wolpert, D. H., & Macready, W. G. (1997). No Free Lunch Theorems for Optimization. IEEE Transactions on Evolutionary Computation, 1(1), 67–82.

---

*This chapter provides the theoretical foundation for understanding GeneExpressionProgramming.jl. For practical applications, continue to the [API Reference](api-reference.md).*

