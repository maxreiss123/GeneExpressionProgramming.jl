# VGeneExpressionProgramming for symbolic regression
The repository contains the implementation of the Gene Expression Programming[1] in conjunction with the semantic backpropagation approach developed in[2]. Here, the target is to reach dimensional homogeneity for physical dimensions through the cause of the exploration.


# How to use it?
  ```
    julia --project=.
    using Pkg
    Pkg.add(url="https://github.com/maxreiss123/GEP_SBP_.git")
  ```


- Run a test for various Feynman datasets located within 'test/srsd' (from the root folder):
  ```
   julia --project==. paper/ConstraintViaSBP.jl
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
- [ ] Re-write postprocessing
- [ ] Improve usability for user interaction
- [ ] Next operations: Tail flip, Connection symbol flip, wrapper class for easy usage, config class for predefinition, staggered exploration
