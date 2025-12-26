# GeneExpressionProgramming for symbolic regression
The repository contains an implementation of the Gene Expression Programming [1], whereby the internal representation of the equation is fully tokenized as a vector of integers. This representation allows a lower memory footprint, leading to faster processing of the application of the genetic operators. Moreover, the implementation also contains a mechanism for semantic backpropagation, ensuring dimensional homogeneity for physical units [2]. 

# Features
- Standard GEP Symbolic Regression
- Multi-Objective optimization
- Population initialization based on Latin Hypercube Sampling
- Coefficient Optimization
- Matrix/ Tensor optimization
- Phy. Dimensionality Consideration

# How does it work?
It pojects an encoded string of symbols into an expression tree: 
![Decoding](images/gep_decoding.gif)

Performing this over many iterations by changing the string (using genetic operators), leads to:
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
To account for the physical dimension of certain input values, we can correct subtrees according to the expected output using semantic backpropagation. In theory it works as follows:
![Semantic](images/semantic_backprop.gif)







- For a more concrete example, imagine you want to find $J$ explaining superconductivity as $J=-\rho \frac{q}{m} A$ (Fyneman III 21.20)
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
  fit!(regressor, epochs, population_size, x_train', y_train; x_test=x_test', y_test=y_test, loss_fun="mse", target_dimension=target_dim)

  pred = regressor(x_test')

  @show regressor.best_models_[1].compiled_function
  @show regressor.best_models_[1].fitness
  ```
- Remark: Template for rerunning the test from the paper is located in the paper directory
- Remark: the tutorial folder contains notebook, that can be run with google-colab, while showing a step-by-step introduction


# How can I approximate functions involving vectors or matricies?
- To conduct a regression involving higher dimensional objects we swap the underlying evaluation from DynamicExpression.jl to Flux.jl
- Hint: By involving such objects, the performance deteriorates significantly

 ```julia
using GeneExpressionProgramming
using Random
using Tensors

Random.seed!(1)

#Define the iterations for the algorithm and the population size
epochs = 100
population_size = 1000

#Number of features which needs to be inserted
number_features = 5

#define the 
regressor = GepTensorRegressor(number_features,
    gene_count=2, #2 works quite reliable 
    head_len=3;
    feature_names=["x1","x2","U1","U2","U3"]) # 5 works quite reliable

#create some testdata - testing simply on a few velocity vectors
size_test = 1000
u1 = [randn(Tensor{1,3}) for _ in 1:size_test]
u2 = [randn(Tensor{1,3}) for _ in 1:size_test]
u3 = [randn(Tensor{1,3}) for _ in 1:size_test]

x1 = [2.0 for _ in 1:size_test]

x2 = [0.0 for _ in 1:size_test]

a = 0.5 * u1 .+ x2 .* u2 + 2 .* u3

inputs = (x1,x2,u1,u2,u3)


@inline function loss_new(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        model = elem.compiled_function
        a_pred = model(inputs)
        !isfinite(norm(a_pred)) && return (typemax(Float64),)
        size(a_pred) != size(a) && return (typemax(Float64),)
        size(a_pred[1]) != size(a[1]) && return (typemax(Float64),)
        
        loss = norm(a_pred .- a)
        elem.fitness = (loss,)
    end
end
fit!(regressor, epochs, population_size, loss_new)

#Print the best expression
lsg = regressor.best_models_[1]
print_karva_strings(lsg)
```

# Supported `Engines' for Symbolic Evaluation
- DynamicExpressions.jl
- Flux.jl --> should be utilized when performing tensor regression


# References
- [1] Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Complex Systems, 13.
- [2] Reissmann, M., Fang, Y., Ooi, A. S. H., & Sandberg, R. D. (2025). Constraining genetic symbolic regression via semantic backpropagation. Genetic Programming and Evolvable Machines, 26(1), 12

 # Acknowledgement
 - The Coefficient optimization is inspired by [https://github.com/MilesCranmer/SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/ConstantOptimization.jl)
 - We employ the insane fast [DynamicExpressions.jl](https://github.com/SymbolicML/DynamicExpressions.jl) for evaluating our expressions

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
- [x] Documentation - first draft
- [x] Naming conventions!
- [x] Improve usability for user interaction
- [ ] staggered exploration
- [x] nice print flux
- [x] constant node needs to be fixed
- [x] Clean Package Structure
- [ ] considering Tullio.jl for faster tensor ops
- [ ] LLM-interface
- [ ] Python-interface
- [ ] MOGA-2 implementation - alternative to NSGA-2
