# Tensor Regression

Hint: This topic is currently under development. The next update will include an improvement in performance and dimension handling. 


Tensor regression is an advanced feature of GeneExpressionProgramming.jl that enables symbolic regression with vector and matrix data. This capability is particularly valuable for applications in computational mechanics, fluid dynamics, computer vision, medical data from (MRI), and other fields where the relationships involve higher-dimensional mathematical objects. The GeneExpressionProgramming.jl provides this capability either by the implementation of the M-GEP[1] or a general exploration. 

## Understanding Tensor Operations

### Mathematical Background

Tensors are generalizations of scalars, vectors, and matrices to higher dimensions:

- **Rank 0 (Scalar)**: Single number
- **Rank 1 (Vector)**: Array of numbers with direction
- **Rank 2 (Matrix)**: 2D array of numbers
- **Rank n (Tensor)**: n-dimensional array

In the context of symbolic regression, tensor operations allow us to discover relationships involving:
- Vector addition and subtraction
- Dot products and cross products
- Matrix multiplication and operations
- Element-wise operations on tensors
- Contraction
- Norms and other tensor properties

### Tensor Operations in GeneExpressionProgramming.jl

The package supports tensor operations through the `GepTensorRegressor`, which uses Flux.jl as the backend for efficient tensor computations. This enables the evolution of expressions that can handle complex mathematical structures while maintaining the interpretability of symbolic expressions.

## Tensor Regression Example

This example demonstrates how to use tensor regression to discover relationships involving vector operations, simulating a scenario from computational fluid dynamics where we want to find a relationship between velocity vectors.

```julia
using GeneExpressionProgramming
using Random
using Tensors
using LinearAlgebra
using Statistics
using Plots

# Set random seed for reproducibility
Random.seed!(789)

println("=== Tensor Regression Example ===")
println("Discovering vector relationships in 3D space")
println("Target: a = 0.5*u1 + x2*u2 + 2*u3")
println("where u1, u2, u3 are 3D vectors and x2 is a scalar")
println()

# Problem setup
number_features = 5  # x1, x2, u1, u2, u3
n_samples = 1000

println("Generating tensor data...")

# Generate scalar features
x1 = [2.0 for _ in 1:n_samples]  # Constant scalar
x2 = randn(n_samples)            # Variable scalar

# Generate 3D vector features using Tensors.jl
u1 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]
u2 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]  
u3 = [Tensor{1,3}(randn(3)) for _ in 1:n_samples]

# Define the true relationship: a = 0.5*u1 + x2*u2 + 2*u3
a_true = [0.5 * u1[i] + x2[i] * u2[i] + 2.0 * u3[i] for i in 1:n_samples]

# Add small amount of noise to make it more realistic
noise_level = 0.01
a_noisy = [a_true[i] + noise_level * Tensor{1,3}(randn(3)) for i in 1:n_samples]

println("Data characteristics:")
println("  Samples: $n_samples")
println("  Scalar features: x1 (constant=2.0), x2 (variable)")
println("  Vector features: u1, u2, u3 (3D vectors)")
println("  Target: a (3D vector)")
println("  Noise level: $noise_level")
println()

# Organize input data
inputs = (x1, x2, u1, u2, u3)

println("Input data structure:")
println("  x1: $(length(x1)) scalars")
println("  x2: $(length(x2)) scalars") 
println("  u1: $(length(u1)) 3D vectors")
println("  u2: $(length(u2)) 3D vectors")
println("  u3: $(length(u3)) 3D vectors")
println()

# Create tensor regressor
regressor = GepTensorRegressor(
    number_features,
    gene_count=2,           # Number of genes (complexity control)
    head_len=3;             # Head length (expression depth control)
    feature_names=["x1", "x2", "U1", "U2", "U3"]  # Names for interpretability
)

println("Tensor regressor configuration:")
println("  Features: $number_features")
println("  Gene count: 2")
println("  Head length: 3")
println()

# Define custom loss function for tensor regression
@inline function tensor_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        try
            # Get the compiled model
            model = elem.compiled_function
            
            # Make predictions
            a_pred = model(inputs)
            
            # Check for valid predictions
            if !isfinite(norm(a_pred)) || size(a_pred) != size(a_noisy)
                elem.fitness = (typemax(Float64),)
                return
            end
            
            # Check individual vector sizes
            if length(a_pred) != length(a_noisy) || 
               (length(a_pred) > 0 && size(a_pred[1]) != size(a_noisy[1]))
                elem.fitness = (typemax(Float64),)
                return
            end
            
            # Calculate loss as norm of difference
            total_error = 0.0
            for i in 1:length(a_noisy)
                error_vec = a_pred[i] - a_noisy[i]
                total_error += norm(error_vec)^2
            end
            
            loss = total_error / length(a_noisy)  # Mean squared error
            elem.fitness = (loss,)
            
        catch e
            # Handle any evaluation errors
            elem.fitness = (typemax(Float64),)
        end
    end
end

# Evolution parameters
epochs = 100  # Reduced for tensor regression (computationally intensive)
population_size = 1000

println("Evolution parameters:")
println("  Epochs: $epochs")
println("  Population size: $population_size")
println("  Loss function: Custom tensor MSE")
println()

println("Starting tensor evolution...")
println("Note: Tensor operations are computationally intensive")
start_time = time()

# Train the tensor regressor
fit!(regressor, epochs, population_size, tensor_loss)

training_time = time() - start_time
println("Training completed in $(round(training_time, digits=2)) seconds")
println()

# Analyze results
println("=== Results Analysis ===")

# Get the best evolved solution
best_solution = regressor.best_models_[1]
best_model = best_solution.compiled_function
best_fitness = best_solution.fitness[1]

println("Best evolved expression:")
print_karva_strings(best_solution)  # Print in Karva notation
println()
println("Best fitness (MSE): $best_fitness")
println()

# Make predictions with the best model
println("Making predictions...")
try
    a_pred = best_model(inputs)
    
    # Calculate comprehensive metrics
    println("Prediction analysis:")
    println("  Predicted vectors: $(length(a_pred))")
    println("  Expected vectors: $(length(a_noisy))")
    
    if length(a_pred) == length(a_noisy)
        # Calculate vector-wise metrics
        vector_errors = [norm(a_pred[i] - a_noisy[i]) for i in 1:length(a_noisy)]
        component_errors = []
        
        for i in 1:length(a_noisy)
            for j in 1:3  # 3D vectors
                push!(component_errors, abs(a_pred[i][j] - a_noisy[i][j]))
            end
        end
        
        println("  Mean vector error: $(round(mean(vector_errors), digits=6))")
        println("  Max vector error: $(round(maximum(vector_errors), digits=6))")
        println("  Mean component error: $(round(mean(component_errors), digits=6))")
        println("  Max component error: $(round(maximum(component_errors), digits=6))")
        
        # Calculate R² equivalent for vectors
        total_variance = sum([norm(a_noisy[i] - mean(a_noisy))^2 for i in 1:length(a_noisy)])
        residual_variance = sum([norm(a_pred[i] - a_noisy[i])^2 for i in 1:length(a_noisy)])
        r2_equivalent = 1 - residual_variance / total_variance
        
        println("  R² equivalent: $(round(r2_equivalent, digits=6))")
        println()
        
        # Detailed component analysis
        println("Component-wise analysis:")
        for comp in 1:3
            true_comp = [a_noisy[i][comp] for i in 1:length(a_noisy)]
            pred_comp = [a_pred[i][comp] for i in 1:length(a_pred)]
            
            comp_correlation = cor(true_comp, pred_comp)
            comp_mse = mean((true_comp .- pred_comp).^2)
            
            println("  Component $comp:")
            println("    Correlation: $(round(comp_correlation, digits=4))")
            println("    MSE: $(round(comp_mse, digits=6))")
        end
        println()
        
        # Visualization
        println("Creating visualizations...")
        
        # 1. Component-wise accuracy plots
        p_components = plot(layout=(1,3), size=(1200, 300))
        
        for comp in 1:3
            true_comp = [a_noisy[i][comp] for i in 1:length(a_noisy)]
            pred_comp = [a_pred[i][comp] for i in 1:length(a_pred)]
            
            scatter!(p_components[comp], true_comp, pred_comp,
                     xlabel="True Component $comp",
                     ylabel="Predicted Component $comp",
                     title="Component $comp Accuracy",
                     alpha=0.6,
                     markersize=2,
                     legend=false)
            
            # Perfect prediction line
            min_val = min(minimum(true_comp), minimum(pred_comp))
            max_val = max(maximum(true_comp), maximum(pred_comp))
            plot!(p_components[comp], [min_val, max_val], [min_val, max_val],
                  color=:red, linestyle=:dash, linewidth=2)
        end
        
        savefig(p_components, "tensor_component_accuracy.png")
        println("Component accuracy plot saved as 'tensor_component_accuracy.png'")
        
        # 2. Vector magnitude comparison
        true_magnitudes = [norm(a_noisy[i]) for i in 1:length(a_noisy)]
        pred_magnitudes = [norm(a_pred[i]) for i in 1:length(a_pred)]
        
        p_magnitude = scatter(true_magnitudes, pred_magnitudes,
                             xlabel="True Vector Magnitude",
                             ylabel="Predicted Vector Magnitude",
                             title="Vector Magnitude Accuracy",
                             alpha=0.6,
                             markersize=3,
                             legend=false)
        
        min_mag = min(minimum(true_magnitudes), minimum(pred_magnitudes))
        max_mag = max(maximum(true_magnitudes), maximum(pred_magnitudes))
        plot!(p_magnitude, [min_mag, max_mag], [min_mag, max_mag],
              color=:red, linestyle=:dash, linewidth=2)
        
        savefig(p_magnitude, "tensor_magnitude_accuracy.png")
        println("Vector magnitude plot saved as 'tensor_magnitude_accuracy.png'")
        
        # 3. Error distribution
        p_error = histogram(vector_errors,
                           xlabel="Vector Error (L2 Norm)",
                           ylabel="Frequency",
                           title="Distribution of Vector Errors",
                           bins=30,
                           alpha=0.7,
                           legend=false)
        
        savefig(p_error, "tensor_error_distribution.png")
        println("Error distribution plot saved as 'tensor_error_distribution.png'")
        
        # 4. 3D visualization of a subset of vectors
        n_viz = min(50, length(a_noisy))  # Visualize subset for clarity
        indices = 1:n_viz
        
        p_3d = plot(layout=(1,2), size=(1000, 400))
        
        # True vectors
        for i in indices
            vec = a_noisy[i]
            plot!(p_3d[1], [0, vec[1]], [0, vec[2]], [0, vec[3]],
                  color=:blue, alpha=0.6, linewidth=1, legend=false)
        end
        plot!(p_3d[1], title="True Vectors (Sample)", xlabel="X", ylabel="Y", zlabel="Z")
        
        # Predicted vectors
        for i in indices
            vec = a_pred[i]
            plot!(p_3d[2], [0, vec[1]], [0, vec[2]], [0, vec[3]],
                  color=:red, alpha=0.6, linewidth=1, legend=false)
        end
        plot!(p_3d[2], title="Predicted Vectors (Sample)", xlabel="X", ylabel="Y", zlabel="Z")
        
        savefig(p_3d, "tensor_3d_visualization.png")
        println("3D vector visualization saved as 'tensor_3d_visualization.png'")
        
    else
        println("Warning: Prediction size mismatch")
        println("Expected: $(length(a_noisy)) vectors")
        println("Got: $(length(a_pred)) vectors")
    end
    
catch e
    println("Error during prediction analysis: $e")
end

println()

# Compare with true relationship
println("=== Comparison with True Relationship ===")
println("True relationship: a = 0.5*u1 + x2*u2 + 2*u3")

# Calculate predictions using true relationship
a_true_clean = [0.5 * u1[i] + x2[i] * u2[i] + 2.0 * u3[i] for i in 1:n_samples]

try
    a_pred = best_model(inputs)
    
    if length(a_pred) == length(a_true_clean)
        # Compare evolved expression with true relationship
        agreement_errors = [norm(a_pred[i] - a_true_clean[i]) for i in 1:length(a_true_clean)]
        mean_agreement_error = mean(agreement_errors)
        
        println("Agreement with true relationship:")
        println("  Mean error: $(round(mean_agreement_error, digits=6))")
        println("  Max error: $(round(maximum(agreement_errors), digits=6))")
        
        # Component-wise agreement
        for comp in 1:3
            true_comp = [a_true_clean[i][comp] for i in 1:length(a_true_clean)]
            pred_comp = [a_pred[i][comp] for i in 1:length(a_pred)]
            agreement_corr = cor(true_comp, pred_comp)
            println("  Component $comp correlation: $(round(agreement_corr, digits=4))")
        end
    end
catch e
    println("Error in true relationship comparison: $e")
end

println()

# Performance analysis
println("=== Performance Analysis ===")
println("Computational considerations:")
println("  Training time: $(round(training_time, digits=2)) seconds")
println("  Time per epoch: $(round(training_time/epochs, digits=2)) seconds")
println("  Population evaluations: $(epochs * population_size)")

# Memory usage estimation
vector_size = 3 * 8  # 3 components × 8 bytes per Float64
total_vector_memory = n_samples * 3 * vector_size  # 3 vector features
println("  Estimated vector memory: $(round(total_vector_memory/1024/1024, digits=2)) MB")

println()
```



### Acceleration
 - [ ] Under developement!

## Best Practices Summary

### 1. Problem Formulation
- Start with simple tensor operations before moving to complex ones
- Ensure tensor dimensions are consistent throughout the problem
- Consider the physical or mathematical meaning of tensor operations

### 2. Data Preparation
- Normalize tensor components to similar scales
- Handle tensor symmetries appropriately (e.g., stress/strain tensors)
- Validate tensor data for physical plausibility

### 3. Algorithm Configuration
- Use smaller populations initially due to computational cost
- Reduce gene count and head length for faster convergence
- Monitor memory usage and use batch processing if needed

### 4. Performance Optimization
- Consider GPU acceleration for large problems
- Use parallel processing when available
- Implement efficient tensor operations

### 5. Result Validation
- Validate results on independent test data
- Check component-wise accuracy
- Verify tensor properties (symmetry, positive definiteness, etc.)
- Compare with known analytical solutions when available

Tensor regression in GeneExpressionProgramming.jl opens up further possibilities for discovering complex mathematical relationships involving higher-dimensional data structures. While computationally more demanding than scalar regression, it provides unique capabilities for applications in physics, engineering, and computer science where tensor operations are fundamental.

## References
 
[1] Weatheritt, J., Sandberg, R. D. (2016)  A novel evolutionary algorithm applied to algebraic modifications of the RANS stress–strain relationship. Journal of Computational Physics, vol. 325, pp. 22-37

---




