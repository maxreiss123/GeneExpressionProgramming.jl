# Abstract
Evolutionary symbolic regression approaches are powerful tools that can approximate an explicit mapping between input features and observation for various problems. However, ensuring that explored expressions maintain consistency with domain-specific constraints remains a crucial challenge. While neural networks are able to employ additional information like conservation laws to achieve more appropriate and robust approximations, the potential remains unrealized within genetic algorithms. This disparity is rooted in the inherent discrete randomness of recombining and mutating to generate new mapping expressions, making it challenging to maintain and preserve inferred constraints or restrictions in the course of the exploration. To address this limitation, we propose an approach centered on semantic backpropagation incorporated into the Gene Expression Programming (GEP), which integrates domain-specific properties in a vector representation as corrective feedback during the evolutionary process. By creating backward rules akin to algorithmic differentiation and leveraging pre-computed subsolutions, the mechanism allows the enforcement of any constraint within an expression tree by determining the misalignment and propagating desired changes back. To illustrate the effectiveness of constraining GEP through semantic backpropagation, we take the constraint of physical dimension as an example. This framework is applied to discovering physical equations from the Feynman lectures. Results have shown not only an increased likelihood of recovering the original equation but also notable robustness in the presence of noisy data.

# VectorizedGeneExpressionProgramming for symbolic regression
The repository contains the implementation of the Gene Expression Programming[1] in conjunction with the semantic backpropagation approach developed in[2]. Here, the target is to reach dimensional homogeneity for physical dimensions through the cause of the exploration.


# How to use it?
- Clone the repository and navigate to the folder:
  ```git clone https://gitlab.unimelb.edu.au/reissmannm/vgep.git```

- Within the folder, install all the required packages:
  ```
    julia --project=.
    using Pgk
    Pkg.instantiate()
  ```

- Alternativly:
  ```
    julia --project=.
    using Pkg
    Pkg.add(url="https://github.com/maxreiss123/GEP_SBP_.git")
  ```


- Run a test for various Feynman datasets located within 'test/srsd' (from the root folder):
  ```
   julia --project==. test/paper_study.jl
  ```

- Remark for your CSV file: Main_min_with_csv.jl in the test folder provides a step-by-step guide on how to initialize the GEP for your own problem
- Remark for your CSV file and utilizing dimensional homogeneity: Main_min_with_csv_and_units.jl in the test folder provides a step-by-step guide 


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
