# VGeneExpressionProgramming for symbolic regression
The repository contains the implementation of the Gene Expression Programming [1], whereby the 'V' refers to the internal representation of the equation as a vector of integers. This representation allows a lower memory footprint, leading to faster processing of the application of the genetic operators. Moreover, the implementation also contains a mechanism for semantic backpropagation, ensuring dimensional homogeneity for physical units [2]. 

# How to use it?
- Install the package:
  ```julia
    using Pkg
    Pkg.add(url="https://github.com/maxreiss123/GEP_SBP_.git")
  ```

  ```julia
  # Min_example 
  using VGeneExpressionProgramming

  Random.seed!(1)

  #Define the iterations for the algorithm and the population size
  epochs = 1000
  population_size = 1000

  #Number of features which needs to be inserted
  number_features = 2

  x_data = randn(Float64, 100, number_features)
  y_data = @. x_data[:,1] * x_data[:,1] + x_data[:,1] * x_data[:,2] - 2 * x_data[:,1] * x_data[:,2]

  #define the 
  regressor = GepRegressor(number_features)
  fit!(regressor, epochs, population_size, x_data, y_data; loss_fun="mse")

  pred = regressor(x_data')

  @show regressor.best_models_[1].compiled_function
  @show regressor.best_models_[1].fitness
  ```


- Remark for your CSV file: Main_min_with_csv.jl in the test folder provides a step-by-step guide on how to initialize the GEP for your own problem
- Remark for your CSV file and utilizing dimensional homogeneity: Main_min_with_csv_and_units.jl in the test folder provides a step-by-step guide
- Remark: the tutorial folder contains notebook, that can be run with google-colab, while showing a step-by-step introduction


# References
- [1] Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Complex Systems, 13.
- [2] Reissmann, M., Fang, Y., Ooi, A., & Sandberg, R. (2024). Constraining genetic symbolic regression via semantic backpropagation. arXiv. https://arxiv.org/abs/2409.07369

 # Acknowledgement
 - The Coefficient optimization is inspired by [https://github.com/MilesCranmer/SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/ConstantOptimization.jl)
 - We employ the insane fast [DynamicExpressions.jl](https://github.com/SymbolicML/DynamicExpressions.jl) for evaluating our expressions

# Todo 
- [ ] Documentation
- [ ] Naming conventions!
- [ ] Re-write postprocessing
- [ ] Improve usability for user interaction
- [ ] Next operations: Tail flip, Connection symbol flip, wrapper class for easy usage, config class for predefinition, staggered exploration
