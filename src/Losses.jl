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
                temp = (y_true[i]-y_pred[i])
                d += temp*temp
          end
          return abs2(d/length(y_true))
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
    "xi_core" => xicor
    )

function get_loss_function(name::String)
    return loss_functions[name]
end


end