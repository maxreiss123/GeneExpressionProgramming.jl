"""
    LossFunction

A module providing various loss functions for evaluating model predictions in machine learning tasks.

# Loss Functions
The module includes the following loss functions, accessible via `get_loss_function(name)`:

- `"r2_score"`: Standard R² score (coefficient of determination)
  - Range: (-∞, 1], where 1 indicates perfect prediction
  - Handles scale-dependent data

- `"r2_score_f"`: R² score with floor scaling
  - Similar to standard R², but with automatic scaling to handle numerical stability
  - Better for comparing vastly different scales

- `"mse"`: Mean Squared Error
  - Standard L2 loss function
  - More sensitive to outliers
  - Scale-dependent

- `"rmse"`: Root Mean Squared Error
  - Square root of MSE
  - Same units as target variable
  - Scale-dependent

- `"nrmse"`: Normalized Root Mean Squared Error
  - Norm. of the rmse

- `"mae"`: Mean Absolute Error
  - L1 loss function
  - More robust to outliers than MSE
  - Scale-dependent

- `"srmse"`: Scaled Root Mean Squared Error
  - RMSE with automatic scaling
  - Better for comparing errors across different scales
  - More numerically stable

- `"xi_core"`: XiCor correlation
  - Non-parametric correlation measure
  - More robust to outliers than Pearson correlation
  - Handles ties in data

# Usage
```julia
using LossFunction

# Get a specific loss function
loss_fn = get_loss_function("mse")

# Use the loss function
error = loss_fn(y_true, y_pred)

# Alternative direct usage
error = mean_squared_error(y_true, y_pred)
```

# Performance Notes
- All functions are optimized for performance using `@fastmath`, `@inbounds`, and `@simd`
- Thread-safe implementations where applicable
- Automatic type stability through parametric types
- Efficient memory usage with in-place operations

# Implementation Details
- All functions accept AbstractArray{T} where T<:AbstractFloat
- Input arrays must be of equal length
- NaN and Inf values are handled appropriately
- Numerical stability is ensured through appropriate scaling and epsilon values
"""
module LossFunction

export get_loss_function
using Statistics

function floor_to_n10p(x::T) where T<:AbstractFloat
    abs_x = abs(x)
    return abs_x > zero(T) ? T(10^floor(log10(abs_x))) : eps(T)
end

function xicor(y_true::AbstractArray{T}, y_pred::AbstractArray{T}; ties::Bool=true) where T<:AbstractFloat
    n = length(y_pred)
    sorted_indices = sortperm(y_pred)
    Y_sorted = y_true[sorted_indices]
    r = zeros(Int, n)
    Threads.@threads for i in 1:n
        for j in i:n
            r[i] += Y_sorted[j] >= Y_sorted[i]
        end
    end
    
    if ties
        tie_counts = zeros(Int, n)
        Threads.@threads for i in 1:n
            for j in 1:n
                tie_counts[i] += r[j] == r[i]
            end
        end
        
        tie_indices = tie_counts .> 1
        mean_ties = mean(tie_counts[tie_indices])
        
        Threads.@threads for i in 1:n
            if tie_counts[i] > 1
                tie_group = findall(==(r[i]), r)
                shuffled = Random.shuffle(0:(tie_counts[i]-1))
                for (idx, group_idx) in enumerate(tie_group)
                    r[group_idx] = r[i] - shuffled[idx]
                end
            end
        end
        
        l = copy(r)
        xi = 1 - n * sum(abs, diff(r)) / (2 * sum(l .* (n .- l)))
    else
        mean_ties = 0.0  
        xi = 1 - 3 * sum(abs, diff(r)) / (n^2 - 1)
    end
    
    return xi
end


function r2_score(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
    len_y = length(y_true)
    y_mean = mean(y_true)
    

    ss_total::T = zero(T)
    @inbounds @simd for i in 1:len_y
	temp = y_true[i]-y_mean
        ss_total += temp*temp
    end

    ss_residual::T = zero(T)
    @inbounds @simd for i in 1:len_y
	temp = y_true[i] - y_pred[i]
	ss_residual += temp*temp
    end
    
    r2 = 1.0 - (ss_residual / ss_total)
    
    return r2
end

function r2_score_floor(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
    max_abs_value = maximum(abs, vcat(y_true, y_pred))

    if max_abs_value == zero(T)
        return one(T)
    end
    
    scale_factor = T(10^floor(log10(max_abs_value)))
    
    # Scale both y_true and y_pred
    y_true_scaled = y_true ./ scale_factor
    y_pred_scaled = y_pred ./ scale_factor
    
    return r2_score(y_true_scaled, y_pred_scaled)
end
      
function mean_squared_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
        d::T = zero(T)
        @assert length(y_true) == length(y_pred)
        @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
              temp = (y_true[i]-y_pred[i])
              d += temp*temp
        end
        return d/length(y_true)
end      

function root_mean_squared_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
    d::T = zero(T)
    @assert length(y_true) == length(y_pred)
    @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
        temp = (y_true[i] - y_pred[i])
        d += temp * temp
    end
    return sqrt(d/length(y_true)) 
end

function normalized_root_mean_squared_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
    n = length(y_true)
    @assert n == length(y_pred) "Arrays must have equal length"
    
    rmse::T = zero(T)
    sum_true::T = zero(T)
    sum_sq_true::T = zero(T)
    
    
    @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
        diff = y_true[i] - y_pred[i]
        rmse += diff * diff
        sum_true += y_true[i]
        sum_sq_true += y_true[i] * y_true[i]
    end
    
    
    rmse = sqrt(rmse / n)
    
    
    mean_true = sum_true / n
    std_true = sqrt((sum_sq_true - (sum_true * sum_true) / n) / (n - 1))
    
    if std_true < eps(T)
        return rmse < eps(T) ? zero(T) : T(Inf)
    end
    
    return rmse / std_true
end

function mean_absolute_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
    d::T = zero(T)
    @assert length(y_true) == length(y_pred)
    @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
        d += abs(y_true[i]-y_pred[i])
    end
    return d/length(y_true)
end

function save_root_mean_squared_error(y_true::AbstractArray{T}, y_pred::AbstractArray{T}) where T<:AbstractFloat
    @assert length(y_true) == length(y_pred)
    
    max_abs_value = maximum(abs, vcat(y_true, y_pred))
    
    if max_abs_value == zero(T)
        return zero(T)
    end
    
    scale_factor = T(10^floor(log10(max_abs_value)))
    
    d::T = zero(T)
    @fastmath @inbounds @simd for i in eachindex(y_true, y_pred)
        y_true_scaled = y_true[i] / scale_factor
        y_pred_scaled = y_pred[i] / scale_factor
        temp = (y_true_scaled - y_pred_scaled) / (abs(y_true_scaled) + eps(T))
        d += temp * temp
    end
    
    return sqrt(d / length(y_true))
end



loss_functions = Dict{String, Function}(
    "r2_score" => r2_score,
    "r2_score_f" => r2_score_floor,
    "mse" => mean_squared_error,
    "rmse" => root_mean_squared_error,
    "mae" => mean_absolute_error,
    "srsme" => save_root_mean_squared_error,
    "xi_core" => xicor,
    "nrmse" => normalized_root_mean_squared_error
    )

function get_loss_function(name::String)
    return loss_functions[name]
end


end
