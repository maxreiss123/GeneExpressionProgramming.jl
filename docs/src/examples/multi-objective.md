# Multi-Objective Optimization

Multi-objective optimization is a powerful feature of GeneExpressionProgramming.jl that allows you to balance multiple competing objectives simultaneously. This is particularly useful in symbolic regression where you often want to find expressions that are both accurate and simple, or when you need to optimize for multiple performance criteria.

## Understanding Multi-Objective Optimization

In traditional single-objective optimization, the goal is to find the single best solution according to one criterion (e.g., minimizing prediction error). However, real-world problems often involve trade-offs between multiple objectives:

- **Accuracy vs. Complexity**: More complex expressions may fit data better but are harder to interpret
- **Training vs. Validation Performance**: Avoiding overfitting while maintaining good training performance
- **Speed vs. Accuracy**: Faster evaluation vs. more precise predictions
- **Robustness vs. Precision**: Stable performance across different conditions vs. optimal performance in specific scenarios

Multi-objective optimization using NSGA-II (Non-dominated Sorting Genetic Algorithm II) finds a set of Pareto-optimal solutions, allowing you to choose the best trade-off for your specific needs.

## Complete Multi-Objective Example

```julia
using GeneExpressionProgramming
using Random
using Statistics
using Plots

# Set random seed for reproducibility
Random.seed!(123)

println("=== Multi-Objective Optimization Example ===")
println("Objectives: Minimize MSE and Expression Complexity")
println()

# Problem setup
number_features = 2
n_samples = 300
noise_level = 0.1

# Generate more complex synthetic data
println("Generating synthetic data...")
x_data = 4 * (rand(Float64, n_samples, number_features) .- 0.5)  # Range [-2, 2]

# Complex target function with multiple terms
y_data = @. x_data[:,1]^3 - 2*x_data[:,1]^2*x_data[:,2] + 
            x_data[:,1]*x_data[:,2]^2 + 0.5*sin(x_data[:,1]) - 
            0.3*cos(x_data[:,2]) + x_data[:,1] + x_data[:,2]

# Add noise
y_data += noise_level * randn(n_samples)

println("Data generated with $(n_samples) samples")
println("Target function: x₁³ - 2x₁²x₂ + x₁x₂² + 0.5sin(x₁) - 0.3cos(x₂) + x₁ + x₂")
println("Noise level: $(noise_level)")
println()

# Split data for validation
train_ratio = 0.8
n_train = round(Int, n_samples * train_ratio)
train_indices = 1:n_train
val_indices = (n_train+1):n_samples

x_train = x_data[train_indices, :]
y_train = y_data[train_indices]
x_val = x_data[val_indices, :]
y_val = y_data[val_indices]

println("Training samples: $(length(y_train))")
println("Validation samples: $(length(y_val))")
println()

# Evolution parameters
epochs = 1000
population_size = 1000

# Create multi-objective regressor
regressor = GepRegressor(number_features; number_of_objectives=2, entered_non_terminals=[:+,:*,:-,:sin,:cos])

println("Multi-objective evolution parameters:")
println("  Epochs: $epochs")
println("  Population size: $population_size")
println("  Objectives: 2 (MSE + Complexity)")
println()

# Define custom multi-objective loss function
@inline function multi_objective_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        # Get the compiled expression
        model = elem.compiled_function
        
        try
            # Calculate predictions
            y_pred_train = model(x_train')
            y_pred_val = model(x_val')
            
            # Check for invalid predictions
            if !all(isfinite.(y_pred_train)) || !all(isfinite.(y_pred_val))
                elem.fitness = (typemax(Float64), typemax(Float64))
                return
            end
            
            # Objective 1: Mean Squared Error on validation set
            mse_val = mean((y_val .- y_pred_val).^2)
            
            # Objective 2: Expression complexity (number of nodes)
            # This is a proxy for interpretability
            complexity = length(elem.expression_raw)  # Simple complexity measure
            
            # Store both objectives (both to be minimized)
            elem.fitness = (mse_val, Float64(complexity))
            
        catch e
            # Handle evaluation errors
            elem.fitness = (typemax(Float64), typemax(Float64))
        end
    end
end

println("Starting multi-objective evolution...")
start_time = time()

# Train with custom multi-objective loss
fit!(regressor, epochs, population_size, multi_objective_loss, hof=20)

training_time = time() - start_time
println("Training completed in $(round(training_time, digits=2)) seconds")
println()

# Analyze Pareto front
println("=== Pareto Front Analysis ===")
pareto_solutions = regressor.best_models_

println("Found $(length(pareto_solutions)) Pareto-optimal solutions")
println()

# Extract objectives for all solutions
mse_values = [sol.fitness[1] for sol in pareto_solutions]
complexity_values = [sol.fitness[2] for sol in pareto_solutions]

# Sort by complexity for better presentation
sorted_indices = sortperm(complexity_values)
sorted_solutions = pareto_solutions[sorted_indices]
sorted_mse = mse_values[sorted_indices]
sorted_complexity = complexity_values[sorted_indices]

println("Pareto-optimal solutions (sorted by complexity):")
println("Index | Complexity | MSE      | Expression")
println("------|------------|----------|------------------------------------------")

for (i, (sol, mse, comp)) in enumerate(zip(sorted_solutions, sorted_mse, sorted_complexity))
    expr_str = string(sol.compiled_function)
    # Truncate long expressions for display
    if length(expr_str) > 40
        expr_str = expr_str[1:37] * "..."
    end
    println("$(lpad(i, 5)) | $(lpad(round(Int, comp), 10)) | $(lpad(round(mse, digits=6), 8)) | $expr_str")
end
println()

# Select interesting solutions for detailed analysis
simplest_idx = 1  # Simplest expression
most_accurate_idx = length(sorted_solutions)  # Most accurate
middle_idx = div(length(sorted_solutions), 2)  # Middle trade-off

selected_indices = [simplest_idx, middle_idx, most_accurate_idx]
selected_labels = ["Simplest", "Balanced", "Most Accurate"]

println("=== Detailed Analysis of Selected Solutions ===")

detailed_results = []
for (idx, label) in zip(selected_indices, selected_labels)
    sol = sorted_solutions[idx]
    model = sol.compiled_function
    
    # Calculate comprehensive metrics
    train_pred = model(x_train')
    val_pred = model(x_val')
    
    train_mse = mean((y_train .- train_pred).^2)
    val_mse = mean((y_val .- val_pred).^2)
    
    train_r2 = 1 - sum((y_train .- train_pred).^2) / sum((y_train .- mean(y_train)).^2)
    val_r2 = 1 - sum((y_val .- val_pred).^2) / sum((y_val .- mean(y_val)).^2)
    
    complexity = sorted_complexity[idx]
    
    println("$label Solution:")
    println("  Expression: $model")
    println("  Complexity: $(round(Int, complexity))")
    println("  Training MSE: $(round(train_mse, digits=6))")
    println("  Validation MSE: $(round(val_mse, digits=6))")
    println("  Training R²: $(round(train_r2, digits=4))")
    println("  Validation R²: $(round(val_r2, digits=4))")
    println()
    
    push!(detailed_results, (
        label=label,
        model=model,
        complexity=complexity,
        train_mse=train_mse,
        val_mse=val_mse,
        train_r2=train_r2,
        val_r2=val_r2,
        train_pred=train_pred,
        val_pred=val_pred
    ))
end

# Visualization
println("Creating visualizations...")

# 1. Pareto front plot
p1 = scatter(sorted_complexity, sorted_mse,
             xlabel="Expression Complexity",
             ylabel="Validation MSE",
             title="Pareto Front: Accuracy vs Complexity",
             legend=false,
             alpha=0.7,
             color=:blue,
             markersize=4)

# Highlight selected solutions
for (idx, label) in zip(selected_indices, selected_labels)
    scatter!(p1, [sorted_complexity[idx]], [sorted_mse[idx]],
             color=:red, markersize=8, alpha=0.8,
             label=label)
end

# Add annotations for selected points
for (idx, label) in zip(selected_indices, selected_labels)
    annotate!(p1, sorted_complexity[idx], sorted_mse[idx] + 0.05 * maximum(sorted_mse),
              text(label, 8, :center))
end

# 2. Prediction accuracy comparison
p2 = plot(layout=(1,3), size=(1200, 300))

for (i, result) in enumerate(detailed_results)
    # Combine train and validation data for plotting
    all_true = vcat(y_train, y_val)
    all_pred = vcat(result.train_pred, result.val_pred)
    
    scatter!(p2[i], all_true, all_pred,
             xlabel="Actual Values",
             ylabel="Predicted Values",
             title="$(result.label)\nR² = $(round(result.val_r2, digits=3))",
             alpha=0.6,
             legend=false)
    
    # Perfect prediction line
    min_val = min(minimum(all_true), minimum(all_pred))
    max_val = max(maximum(all_true), maximum(all_pred))
    plot!(p2[i], [min_val, max_val], [min_val, max_val],
          color=:red, linestyle=:dash, linewidth=2)
end

# 3. Trade-off analysis
p3 = plot(1:length(sorted_solutions), sorted_mse,
          xlabel="Solution Index (by complexity)",
          ylabel="Validation MSE",
          title="MSE vs Solution Complexity Rank",
          legend=false,
          linewidth=2,
          color=:blue)

# Mark selected solutions
for (idx, label) in zip(selected_indices, selected_labels)
    scatter!(p3, [idx], [sorted_mse[idx]],
             color=:red, markersize=8,
             label=label)
end

# 4. Complexity distribution
p4 = histogram(sorted_complexity,
               xlabel="Expression Complexity",
               ylabel="Number of Solutions",
               title="Distribution of Solution Complexity",
               bins=20,
               alpha=0.7,
               color=:lightblue,
               legend=false)

# Combine all plots
final_plot = plot(p1, p3, p2, p4, layout=(2,2), size=(1000, 800))
savefig(final_plot, "multi_objective_analysis.png")
println("Multi-objective analysis plot saved as 'multi_objective_analysis.png'")

# 5. Detailed comparison of selected solutions
comparison_plot = plot(layout=(1,3), size=(1200, 400))

for (i, result) in enumerate(detailed_results)
    scatter!(comparison_plot[i], y_val, result.val_pred,
             xlabel="Actual Values",
             ylabel="Predicted Values",
             title="$(result.label) (Complexity: $(round(Int, result.complexity)))",
             alpha=0.7,
             legend=false,
             color=[:blue, :green, :orange][i])
    
    # Perfect prediction line
    min_val = min(minimum(y_val), minimum(result.val_pred))
    max_val = max(maximum(y_val), maximum(result.val_pred))
    plot!(comparison_plot[i], [min_val, max_val], [min_val, max_val],
          color=:red, linestyle=:dash, linewidth=2)
end

savefig(comparison_plot, "solution_comparison.png")
println("Solution comparison plot saved as 'solution_comparison.png'")

println()
println("=== Multi-Objective Optimization Complete ===")

# Decision support
println("=== Decision Support ===")
println("Choose a solution based on your priorities:")
println()

for result in detailed_results
    println("$(result.label):")
    println("  - Best for: ")
    if result.label == "Simplest"
        println("Interpretability and fast evaluation")
    elseif result.label == "Balanced"
        println("Good trade-off between accuracy and simplicity")
    else
        println("Maximum prediction accuracy")
    end
    println("  - Validation R²: $(round(result.val_r2, digits=4))")
    println("  - Expression: $(result.model)")
    println()
end

# Recommend based on validation performance
best_val_r2_idx = argmax([r.val_r2 for r in detailed_results])
println("Recommendation: For most applications, consider the '$(detailed_results[best_val_r2_idx].label)' solution")
println("as it provides the best validation performance.")
```

## Understanding the Results

### Pareto Front Analysis

The Pareto front represents the set of solutions where no other solution is better in all objectives simultaneously. Each point on the front represents a different trade-off between accuracy and complexity.

Key insights from the Pareto front:
- **Lower left**: Simple but less accurate expressions
- **Upper right**: Complex but more accurate expressions
- **Elbow points**: Often represent good trade-offs

### Solution Selection Strategies

#### 1. Simplest Solution
- **Use when**: Interpretability is paramount
- **Advantages**: Easy to understand, fast evaluation, less prone to overfitting
- **Disadvantages**: May sacrifice accuracy

#### 2. Balanced Solution
- **Use when**: Need reasonable accuracy with moderate complexity
- **Advantages**: Good compromise between interpretability and performance
- **Disadvantages**: May not excel in either dimension

#### 3. Most Accurate Solution
- **Use when**: Prediction accuracy is the primary concern
- **Advantages**: Best predictive performance
- **Disadvantages**: May be complex and harder to interpret

## Advanced Multi-Objective Techniques

### Custom Objective Functions

```julia
# Example: Three-objective optimization
@inline function three_objective_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        model = elem.compiled_function
        
        try
            y_pred = model(x_train')
            
            # Objective 1: Training MSE
            mse_train = mean((y_train .- y_pred).^2)
            
            # Objective 2: Expression complexity
            complexity = count_nodes(model)  # Custom function to count nodes
            
            # Objective 3: Evaluation time (efficiency)
            eval_time = @elapsed model(x_train')
            
            elem.fitness = (mse_train, Float64(complexity), eval_time)
        catch
            elem.fitness = (typemax(Float64), typemax(Float64), typemax(Float64))
        end
    end
end
```

### Weighted Objectives

```julia
# Convert multi-objective to single-objective with weights
function weighted_objective(mse, complexity; w_mse=0.7, w_complexity=0.3)
    # Normalize objectives to [0, 1] range
    normalized_mse = mse / max_expected_mse
    normalized_complexity = complexity / max_expected_complexity
    
    return w_mse * normalized_mse + w_complexity * normalized_complexity
end
```

### Constraint Handling

```julia
@inline function constrained_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        model = elem.compiled_function
        
        try
            y_pred = model(x_train')
            complexity = length(string(model))
            
            # Hard constraint: complexity must be below threshold
            if complexity > 100
                elem.fitness = (typemax(Float64), typemax(Float64))
                return
            end
            
            mse = mean((y_train .- y_pred).^2)
            elem.fitness = (mse, Float64(complexity))
        catch
            elem.fitness = (typemax(Float64), typemax(Float64))
        end
    end
end
```

## Performance Considerations

### Population Size for Multi-Objective

Multi-objective optimization may requires larger populations than single-objective optimization:

```julia
# Some test scenarios - population sizes
single_objective_pop = 500
multi_objective_pop = 1000  # 2x for 2 objectives
three_objective_pop = 1500  # 3x for 3 objectives
```

### Convergence Monitoring

```julia
function plot_convergence(regressor)
    # Extract fitness history for all objectives
    if hasfield(typeof(regressor), :fitness_history_)
        history = regressor.fitness_history_
        
        # Plot evolution of best solutions for each objective
        generations = 1:length(history)
        
        p1 = plot(generations, [h[1] for h in history],
                  xlabel="Generation",
                  ylabel="Best MSE",
                  title="Objective 1: MSE Evolution",
                  legend=false)
        
        p2 = plot(generations, [h[2] for h in history],
                  xlabel="Generation",
                  ylabel="Best Complexity",
                  title="Objective 2: Complexity Evolution",
                  legend=false)
        
        plot(p1, p2, layout=(1,2))
    end
end
```

## Real-World Applications

### 1. Financial Modeling

```julia
# Objectives: Prediction accuracy + Model stability + Regulatory compliance
@inline function financial_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        model = elem.compiled_function
        
        # Objective 1: Prediction accuracy
        predictions = model(features')
        accuracy = mean((returns .- predictions).^2)
        
        # Objective 2: Model stability (variance across time periods)
        stability = var([model(period_data') for period_data in time_periods])
        
        # Objective 3: Regulatory compliance (expression interpretability)
        compliance_score = interpretability_score(model)
        
        elem.fitness = (accuracy, stability, compliance_score)
    end
end
```

### 2. Engineering Design

```julia
# Objectives: Performance + Cost + Safety margin
@inline function engineering_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        model = elem.compiled_function
        
        # Objective 1: Performance (minimize error)
        performance = mean((target_performance .- model(design_params')).^2)
        
        # Objective 2: Cost (complexity proxy)
        cost = manufacturing_cost(model)
        
        # Objective 3: Safety (maximize safety margin)
        safety_margin = -minimum_safety_factor(model)  # Negative for minimization
        
        elem.fitness = (performance, cost, safety_margin)
    end
end
```

### 3. Scientific Discovery

```julia
# Objectives: Fit quality + Physical plausibility + Simplicity
@inline function scientific_loss(elem, validate::Bool)
    if isnan(mean(elem.fitness)) || validate
        model = elem.compiled_function
        
        # Objective 1: Goodness of fit
        fit_quality = mean((experimental_data .- model(conditions')).^2)
        
        # Objective 2: Physical plausibility
        plausibility_penalty = physical_constraint_violations(model)
        
        # Objective 3: Occam's razor (simplicity)
        simplicity = expression_complexity(model)
        
        elem.fitness = (fit_quality, plausibility_penalty, simplicity)
    end
end
```

## Best Practices for Multi-Objective Optimization

### 1. Objective Design
- Ensure objectives are conflicting (otherwise single-objective is sufficient) - can be figured out by observing correlation of different targets
- Scale objectives to similar ranges
- Consider the number of objectives (2-3 typically work best)

### 2. Population Management
- Use larger populations for multi-objective problems
- Monitor diversity to ensure good Pareto front coverage
- Consider archive strategies for very long runs

### 3. Solution Selection
- Use domain knowledge to guide selection
- Consider elbow points on the Pareto front
- Validate selected solutions on independent test data

### 4. Visualization
- Always visualize the Pareto front
- Use parallel coordinate plots for >2 objectives
- Create decision support tools for stakeholders

### 5. Computational Efficiency
- Profile objective function evaluation
- Consider approximation methods for expensive objectives
- Use parallel evaluation when possible

## Common Pitfalls and Solutions

### Pitfall 1: Dominated Objectives
**Problem**: One objective dominates others, leading to poor trade-offs (can sometimes be observed in NSGA)
**Solution**: Proper objective scaling and normalization

### Pitfall 2: Too Many Objectives
**Problem**: Curse of dimensionality in objective space
**Solution**: Limit to 2-3 objectives or use specialized many-objective algorithms

### Pitfall 3: Poor Diversity
**Problem**: Solutions cluster in one region of the Pareto front
**Solution**: Increase population size or mutation rate

### Pitfall 4: Expensive Evaluation
**Problem**: Multi-objective evaluation is computationally expensive
**Solution**: Use surrogate models, parallel evaluation, or approximation methods

---

*Next: [Physical Dimensionality](physical-dimensions.md)*

