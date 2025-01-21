# GeneExpressionProgramming for symbolic regression
The repository contains an implementation of the Gene Expression Programming [1], whereby the internal representation of the equation is fully tokenized as a vector of integers. This representation allows a lower memory footprint, leading to faster processing of the application of the genetic operators. Moreover, the implementation also contains a mechanism for semantic backpropagation, ensuring dimensional homogeneity for physical units [2]. 

# How to use it?
- Install the package:
  ```julia
    using Pkg
    
    Pkg.add(url="https://github.com/maxreiss123/GeneExpressionProgramming.jl.git")

  ```

  ```julia
  # Min_example 
  using GeneExpressionProgramming
  using Random

  Random.seed!(1)

  #Define the iterations for the algorithm and the population size
  epochs = 1000
  population_size = 1000

  #Number of features which needs to be inserted
  number_features = 2

  x_data = randn(Float64, 100, number_features)
  y_data = @. x_data[:,1] * x_data[:,1] + x_data[:,1] * x_data[:,2] - 2 * x_data[:,1] * x_data[:,2]

  #define the regressor
  regressor = GepRegressor(number_features)

  #perform the regression by entering epochs, population_size, the feature cols, the target col and the loss function
  fit!(regressor, epochs, population_size, x_data', y_data; loss_fun="mse")

  pred = regressor(x_data') # Can be utilized to perform the prediction for further data

  @show regressor.best_models_[1].compiled_function
  @show regressor.best_models_[1].fitness
  ```

# How to consider the physical dimensions mentioned within [2]? 
- Imagine you want to find $J$ explaining superconductivity as $J=-\rho \frac{q}{m} A$ (Fyneman III 21.20)
- $J$ marking the electric current, $q$ the electric charge, $\rho$ the charge density, $m$ the mass and $A$ the magnetic vector potential

 ```julia
  # Min_example 
  using GeneExpressionProgramming
  using Random

  Random.seed!(1)

  #Define the iterations for the algorithm and the population size
  epochs = 1000
  population_size = 1000


  #By loading the data we end up with 5 cols => 4 for the features and the last one for the target
  data = Matrix(CSV.read("paper/srsd/feynman-III.21.20\$0.01.txt", DataFrame))
  data = data[all.(x -> !any(isnan, x), eachrow(data)), :]
  num_cols = size(data, 2) #num_cols =5 


  # Perform a simple train test split
  x_train, y_train, x_test, y_test = train_test_split(data[:, 1:num_cols-1], data[:, num_cols]; consider=4)

  #define a target dimension - here ampere - (units inspired by openFoam) - https://doc.cfd.direct/openfoam/user-guide-v6/basic-file-format
  target_dim = Float16[0, -2, 0, 0, 0, 1, 0] # Aiming for electric conductivity (Ampere/m^2)


  #define dims for the features
  #header of the reveals rho_c_0,q,A_vec,m -> internally mapt on x_1 ... x_n
  feature_dims = Dict{Symbol,Vector{Float16}}(
    :x1 => Float16[0, -3, 1, 0, 0, 1, 0],   #rho    m^(-3) * s * A 
    :x2 => Float16[0, 0, 1, 0, 0, 1, 0],    #q      s*A
    :x3 => Float16[1, 1, -2, 0, 0, -1, 0],  #A      kg*m*s^(-2)*A^(-1)
    :x4 => Float16[1, 0, 0, 0, 0, 0, 0],    #m      kg
  )


  #define the features, here the numbers of the first two cols - here we add the feature dims and a maximum of permutations per tree high - rounds, referring to the tree high
  regressor = GepRegressor(num_cols-1; considered_dimensions=feature_dims,max_permutations_lib=10000, rounds=7)

   #perform the regression by entering epochs, population_size, the feature cols, the target col and the loss function
  fit!(regressor, epochs, population_size, x_train', y_train; x_test=x_test', y_test=y_test, loss_fun="mse")

  pred = regressor(x_data')

  @show regressor.best_models_[1].compiled_function
  @show regressor.best_models_[1].fitness
  ```
- Remark: Template for rerunning the test from the paper is located in the paper directory
- Remark: the tutorial folder contains notebook, that can be run with google-colab, while showing a step-by-step introduction


# References
- [1] Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Complex Systems, 13.
- [2] Reissmann, M., Fang, Y., Ooi, A., & Sandberg, R. (2024). Constraining genetic symbolic regression via semantic backpropagation. arXiv. https://arxiv.org/abs/2409.07369

 # Acknowledgement
 - The Coefficient optimization is inspired by [https://github.com/MilesCranmer/SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/ConstantOptimization.jl)
 - We employ the insane fast [DynamicExpressions.jl](https://github.com/SymbolicML/DynamicExpressions.jl) for evaluating our expressions

# Todo 
- [ ] Documentation
- [x] Naming conventions!
- [x] Improve usability for user interaction
- [ ] Next operations: Tail flip, Connection symbol flip, wrapper class for easy usage, config class for predefinition, staggered exploration
- [ ] latest enhancements are provided in the branch 'feature/modular_error'
- [ ] Flexible underlying engine -> Currently DynamicExprresions, Flux in the future for GPU support
