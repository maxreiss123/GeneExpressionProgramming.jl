"""
    SBPUtils

Semantic backpropagation (SBP): keeping evolved expressions dimensionally homogeneous.

A dimension is a `Float16` vector of SI exponents `[kg, m, s, K, mol, A, cd]`; on the
tensor path the tensor order comes first. Every operator has a forward unit rule (the
dimension of its result) and a backward rule (the dimensions its operands need for a
required result). A chromosome whose dimension misses the target is repaired by pushing
the target down its expression with the backward rules and editing the expression where
a requirement cannot be met as it stands.

# Constants
- `STD_DIM_SIZE = 7`: length of a dimension on the scalar path
- `ZERO_DIM`, `EMPTY_DIM`: the dimensionless and the inconsistent (`Inf`) dimension
- `F16_LOWER_BOUND = eps(Float16)`: tolerance when comparing dimensions

# Components
- Data: `TokenLib` (dimensions, forward rules and arities of the alphabet), `LibEntry` (a
  library expression under construction), `TokenDto` (alphabet, backward rules, library
  and its `LibIndex`), `LibIndex` (lookups over the library, and a kd-tree over the
  dimensions the library reaches with one more operator)
- Unit rules, forward and backward: `equal_unit_*`, `mul_unit_*`, `div_unit_*`,
  `zero_unit_*`, `sqr_unit_*`, `sign_unit_*`, `arbitrary_unit_*`; on the tensor path
  `mul_t_unit_*`, `div_t_unit_*`, `contraction_unit_*`, `double_contraction_unit_*`,
  `symmetric_contraction_*`, `crossp_unit_*`, `inv_t_unit_*`, `hadamard_unit_*` and
  `tr_unit_backward`, with the helpers `zero_dim`, `empty_dim`, `split_dim`, `join_dim`
  and `compose_units`
- Library: `create_lib`
- Repair: `correct_genes!` repairs a chromosome in place, the expression as a whole or
  every gene on its own (`gene_wise`)
- Read-only checks: `expression_dimension`, `dimensional_homogeneity_distance`,
  `is_dimensionally_homogeneous`, and gene by gene `gene_dimensions` and
  `is_gene_wise_homogeneous`
- Seeding: `sample_lib_expression`, `random_expression`
- Tree API: `TempComputeTree`, `create_compute_tree`, `calculate_vector_dimension!`,
  `propagate_necessary_changes!` (the repair on a tree), `repair_infeasible_ops!`,
  `flatten_dependents`, `flush!`
- Helpers: `get_feature_dims_json`, `get_target_dim_json` (dimensions from parsed JSON
  case data), `retrieve_coeffs_based_on_similarity` (physical constants whose dimension is
  close to a target)

# Repair
The repair works on the karva string rather than a tree. At each node it tries the
cheapest move that meets the requirement: swap the operator or terminal, drop a unary
operator, propagate the requirement into the operands (for a product or quotient keeping
either operand, or splitting the requirement between both with nearest dimensions from the
kd-tree), and last replace the subtree by an expression built from the library. Connectors
are only swapped or propagated through, and an edited gene stays within the gene length
with its operators in the head, so it remains a valid gene for the genetic operators.
"""
module SBPUtils

const STD_DIM_SIZE = 7
const ZERO_DIM = zeros(Float16, STD_DIM_SIZE)
const EMPTY_DIM = Float16[typemax(Float16) for _ in 1:STD_DIM_SIZE]
const F16_LOWER_BOUND = eps(Float16)

using OrderedCollections
using Random
using StaticArrays
using ..GepUtils: thread_slots
using NearestNeighbors: KDTree, knn

export TokenLib, TokenDto, LibEntry, TempComputeTree
export create_lib, create_compute_tree, propagate_necessary_changes!, calculate_vector_dimension!, flush!, calculate_vector_dimension!, flatten_dependents
export propagate_necessary_changes!, correct_genes!, repair_infeasible_ops!
export equal_unit_forward, mul_unit_forward, div_unit_forward, zero_unit_forward, sqr_unit_backward, sqr_unit_forward, arbitrary_unit_forward
export zero_unit_backward, mul_unit_backward, div_unit_backward, equal_unit_backward
export get_feature_dims_json, get_target_dim_json, retrieve_coeffs_based_on_similarity
export dimensional_homogeneity_distance, is_dimensionally_homogeneous, expression_dimension
export gene_dimensions, is_gene_wise_homogeneous
export sign_unit_forward, sign_unit_backward, inv_t_unit_forward, inv_t_unit_backward
export hadamard_unit_forward, hadamard_unit_backward
export LibIndex, random_expression
export sample_lib_expression
export ZERO_DIM, mul_t_unit_backward, mul_t_unit_forward, div_t_unit_backward, div_t_unit_forward
export zero_dim, empty_dim, split_dim, join_dim, compose_units
export contraction_unit_backward, contraction_unit_forward, crossp_unit_backward, crossp_unit_forward, double_contraction_unit_backward, double_contraction_unit_forward
export symmetric_contraction_backward, symmetric_contraction_forward, arbitrary_unit_backward, tr_unit_backward

# Whether `u` is inconsistent. Any non-finite component counts, not only +Inf: under the
# quotient rule, `x / inconsistent` gives -Inf and `inconsistent / inconsistent` gives NaN.
@inline function has_inf16(u::AbstractVector{Float16})
    @inbounds for x in u
        isfinite(x) || return true
    end
    return false
end

# `(u1, u2)`, with `u1` (`ll_top_up`) or `u2` (`rr_top_up`) replaced by a copy whose tensor
# order (slot `index_`) is `topup`
@inline function ll_top_up(u1::Vector{Float16}, u2::Vector{Float16}; index_::Int=1, topup::Float16=1)
    ll = copy(u1)
    ll[index_] = topup
    return ll, u2
end

@inline function rr_top_up(u1::Vector{Float16}, u2::Vector{Float16}; index_::Int=1, topup::Float16=1)
    rr = copy(u2)
    rr[index_] = topup
    return u1, rr
end


"""
    zero_dim(n)
    empty_dim(n)

The dimensionless and the inconsistent dimension of length `n`. The tensor path needs
these: its dimensions carry the tensor order in an extra slot, so `ZERO_DIM` and
`EMPTY_DIM` have the wrong length there.
"""
@inline zero_dim(n::Int) = zeros(Float16, n)
@inline empty_dim(n::Int) = Float16[typemax(Float16) for _ in 1:n]

"""
    split_dim(u; index_=1)

Split a tensor-path dimension into `(order, si_exponents)`, the order being slot `index_`.
"""
@inline split_dim(u::Vector{Float16}; index_::Int=1) =
    (u[index_], deleteat!(copy(u), index_))

"""
    join_dim(order, units; index_=1)

Inverse of `split_dim`: insert the tensor `order` into `units` at slot `index_`.
"""
@inline function join_dim(order::Real, units::Vector{Float16}; index_::Int=1)
    out = copy(units)
    insert!(out, index_, Float16(order))
    return out
end

"""
    compose_units(u1, u2, order; index_=1, quotient=false)

Dimension of a product-like tensor operator: the SI exponents of `u1` and `u2` add (or
subtract, with `quotient=true`), and slot `index_` is set to the tensor `order`.
"""
@inline function compose_units(u1::Vector{Float16}, u2::Vector{Float16}, order::Real;
    index_::Int=1, quotient::Bool=false)
    res = quotient ? u1 .- u2 : u1 .+ u2
    res[index_] = Float16(order)
    return res
end

# Tensor unit rules: slot `index_` of a dimension holds the tensor order, the other slots
# the SI exponents.
function contraction_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}; index_::Int=1)
    order = u1[index_] + u2[index_] - 2
    if order < 0
        return empty_dim(length(u1))
    end
    return compose_units(u1, u2, order; index_=index_)
end

# Solves for the tensor orders only; the SI units are not split between the operands.
function contraction_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}; index_::Int=1)
    expected_order = expected_dim[index_]

    # the operands already contract to the expected order
    if u1[index_] + u2[index_] - 2 == expected_order
        return u1, u2
    end

    # expected scalar: two vectors
    if expected_order == 0
        ll = copy(expected_dim)
        rr = copy(expected_dim)
        ll[index_] = expected_order + 1
        rr[index_] = expected_order + 1
        return ll, rr
    end

    # expected order 1: orders (1, 2) or (2, 1)
    if expected_order == 1
        if u1[index_] == expected_order
            return rr_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        elseif u2[index_] == expected_order
            return ll_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        elseif u1[index_] == expected_order + 1
            return rr_top_up(u1, u2; index_=index_, topup=expected_order)
        elseif u2[index_] == expected_order + 1
            return ll_top_up(u1, u2; index_=index_, topup=expected_order)
        else
            ll = copy(expected_dim)
            rr = copy(expected_dim)
            if rand() < 0.5
                ll[index_] = expected_order
                rr[index_] = expected_order + 1
            else
                ll[index_] = expected_order + 1
                rr[index_] = expected_order
            end
            return ll, rr
        end
    end

    # expected order 2: orders (2, 2), (3, 1) or (1, 3)
    if expected_order == 2
        if u1[index_] == expected_order
            return rr_top_up(u1, u2; index_=index_, topup=expected_order)
        elseif u2[index_] == expected_order
            return ll_top_up(u1, u2; index_=index_, topup=expected_order)
        elseif u1[index_] == expected_order + 1
            return rr_top_up(u1, u2; index_=index_, topup=expected_order - 1)
        elseif u2[index_] == expected_order + 1
            return ll_top_up(u1, u2; index_=index_, topup=expected_order - 1)
        else
            ll = copy(expected_dim)
            rr = copy(expected_dim)
            if rand() < 0.5
                ll[index_] = expected_order
                rr[index_] = expected_order
            elseif rand() < 0.5
                ll[index_] = expected_order + 1
                rr[index_] = expected_order - 1
            else
                ll[index_] = expected_order - 1
                rr[index_] = expected_order + 1
            end
            return ll, rr
        end
    end

    # expected order 3: orders (3, 2), (2, 3), (4, 1) or (1, 4)
    if expected_order == 3
        if u1[index_] == expected_order
            return rr_top_up(u1, u2; index_=index_, topup=expected_order - 1)
        elseif u2[index_] == expected_order
            return ll_top_up(u1, u2; index_=index_, topup=expected_order - 1)
        elseif u1[index_] == expected_order - 1
            return rr_top_up(u1, u2; index_=index_, topup=expected_order)
        elseif u2[index_] == expected_order - 1
            return ll_top_up(u1, u2; index_=index_, topup=expected_order)
        elseif u1[index_] == expected_order - 2
            return rr_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        elseif u2[index_] == expected_order - 2
            return ll_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        else
            ll = copy(expected_dim)
            rr = copy(expected_dim)
            if rand() < 0.5
                ll[index_] = expected_order
                rr[index_] = expected_order - 1
            elseif rand() < 0.5
                ll[index_] = expected_order - 1
                rr[index_] = expected_order
            elseif rand() < 0.5
                ll[index_] = expected_order - 2
                rr[index_] = expected_order + 1
            else
                ll[index_] = expected_order + 1
                rr[index_] = expected_order - 2
            end
            return ll, rr
        end
    end

    # expected order 4: orders (4, 2), (2, 4) or (3, 3)
    if expected_order == 4
        if u1[index_] == expected_order
            return rr_top_up(u1, u2; index_=index_, topup=expected_order - 2)
        elseif u2[index_] == expected_order
            return ll_top_up(u1, u2; index_=index_, topup=expected_order - 2)
        elseif u1[index_] == expected_order - 1
            return rr_top_up(u1, u2; index_=index_, topup=expected_order - 1)
        elseif u2[index_] == expected_order - 1
            return ll_top_up(u1, u2; index_=index_, topup=expected_order - 1)
        elseif u1[index_] == expected_order - 2
            return rr_top_up(u1, u2; index_=index_, topup=expected_order)
        elseif u2[index_] == expected_order - 2
            return ll_top_up(u1, u2; index_=index_, topup=expected_order)
        else
            ll = copy(expected_dim)
            rr = copy(expected_dim)
            if rand() < 0.5
                ll[index_] = expected_order
                rr[index_] = expected_order - 2
            elseif rand() < 0.5
                ll[index_] = expected_order - 2
                rr[index_] = expected_order
            else
                ll[index_] = expected_order - 1
                rr[index_] = expected_order - 1
            end
            return ll, rr
        end
    end
end

function double_contraction_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}; index_::Int=1)
    order = u1[index_] + u2[index_] - 4
    if order < 0
        return empty_dim(length(u1))
    end
    return compose_units(u1, u2, order; index_=index_)
end

# Solves for the tensor orders only; the SI units are not split between the operands.
function double_contraction_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}; index_::Int=1)
    expected_order = expected_dim[index_]

    # the operands already contract to the expected order
    if u1[index_] + u2[index_] - 4 == expected_order
        return u1, u2
    end

    # expected scalar: two second-order tensors
    if expected_order == 0
        if u1[index_] == expected_order + 2
            return rr_top_up(u1, u2; index_=index_, topup=expected_order + 2)
        elseif u2[index_] == expected_order + 2
            return ll_top_up(u1, u2; index_=index_, topup=expected_order + 2)
        else
            ll = copy(expected_dim)
            rr = copy(expected_dim)
            ll[index_] = expected_order + 2
            rr[index_] = expected_order + 2
            return ll, rr
        end
    end

    # expected order 1: orders (2, 3) or (3, 2)
    if expected_order == 1
        if u1[index_] == expected_order + 1
            return rr_top_up(u1, u2; index_=index_, topup=expected_order + 2)
        elseif u2[index_] == expected_order + 1
            return ll_top_up(u1, u2; index_=index_, topup=expected_order + 2)
        elseif u1[index_] == expected_order + 2
            return rr_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        elseif u2[index_] == expected_order + 2
            return ll_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        else
            ll = copy(expected_dim)
            rr = copy(expected_dim)
            if rand() < 0.5
                ll[index_] = expected_order + 1
                rr[index_] = expected_order + 2
            else
                ll[index_] = expected_order + 2
                rr[index_] = expected_order + 1
            end
            return ll, rr
        end
    end

    # expected order 2: orders (2, 4), (4, 2) or (3, 3)
    if expected_order == 2
        if u1[index_] == expected_order
            return rr_top_up(u1, u2; index_=index_, topup=expected_order + 2)
        elseif u2[index_] == expected_order
            return ll_top_up(u1, u2; index_=index_, topup=expected_order + 2)
        elseif u1[index_] == expected_order + 1
            return rr_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        elseif u2[index_] == expected_order + 1
            return ll_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        else
            ll = copy(expected_dim)
            rr = copy(expected_dim)
            if rand() < 0.5
                ll[index_] = expected_order
                rr[index_] = expected_order + 2
            elseif rand() < 0.5
                ll[index_] = expected_order + 2
                rr[index_] = expected_order
            else
                rand() < 0.5
                ll[index_] = expected_order + 1
                rr[index_] = expected_order + 1
            end
            return ll, rr
        end
    end

    # expected order 3: orders (3, 4) or (4, 3)
    if expected_order == 3
        if u1[index_] == expected_order
            return rr_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        elseif u2[index_] == expected_order
            return ll_top_up(u1, u2; index_=index_, topup=expected_order + 1)
        elseif u1[index_] == expected_order + 1
            return rr_top_up(u1, u2; index_=index_, topup=expected_order)
        elseif u2[index_] == expected_order + 1
            return ll_top_up(u1, u2; index_=index_, topup=expected_order)
        else
            ll = copy(expected_dim)
            rr = copy(expected_dim)
            if rand() < 0.5
                ll[index_] = expected_order
                rr[index_] = expected_order + 1
            else
                ll[index_] = expected_order + 1
                rr[index_] = expected_order
            end
            return ll, rr
        end
    end

    # expected order 4: two fourth-order tensors
    if expected_order == 4
        if u1[index_] == expected_order
            return rr_top_up(u1, u2; index_=index_, topup=expected_order)
        elseif u2[index_] == expected_order
            return ll_top_up(u1, u2; index_=index_, topup=expected_order)
        else
            ll = copy(expected_dim)
            rr = copy(expected_dim)
            ll[index_] = expected_order
            rr[index_] = expected_order
            return ll, rr
        end
    end
end


function symmetric_contraction_forward(u1::Vector{Float16}; index_::Int=1)
    temp = copy(u1)
    temp[index_] = temp[index_] + 2 - 4
    if temp[index_] < 0
        return empty_dim(length(u1))
    end
    return temp
end

function symmetric_contraction_backward(u1::Vector{Float16}; index_::Int=1)
    temp = copy(u1)
    temp[index_] = temp[index_] - 2 + 4
    return temp
end

function arbitrary_unit_forward(u1::Vector{Float16}, u2::Vector{Float16})
    return maximum([u1, u2])
end

function arbitrary_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}; index_::Int=1)
    if u1[index_] > u2[index_]
        return expected_dim, u2
    elseif u1[index_] < u2[index_]
        return u1, expected_dim
    else
        return expected_dim, expected_dim
    end
end


function mul_t_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}; index_::Int=1)
    if u1[index_] == 0 || u2[index_] == 0
        # one operand is a scalar, so the tensor order is whichever is non-zero
        return compose_units(u1, u2, max(u1[index_], u2[index_]); index_=index_)
    else
        return empty_dim(length(u1))
    end

end


function mul_t_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}; index_::Int=1)
    # `a * b` needs a scalar operand, whose units multiply into the result: the other
    # operand takes the expected order and the expected units divided by the scalar's
    n = length(expected_dim)
    if u1[index_] == 0
        other = expected_dim .- u1
        other[index_] = expected_dim[index_]
        return u1, other
    elseif u2[index_] == 0
        other = expected_dim .- u2
        other[index_] = expected_dim[index_]
        return other, u2
    else
        # neither is a scalar, so one of them has to become a dimensionless one
        if rand() < 0.5
            return expected_dim, zero_dim(n)
        else
            return zero_dim(n), expected_dim
        end
    end
end

function div_t_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}; index_::Int=1)
    if u2[index_] == 0
        # the divisor is a scalar: the numerator sets the order, the units divide
        return compose_units(u1, u2, u1[index_]; index_=index_, quotient=true)
    else
        return empty_dim(length(u1))
    end

end


function div_t_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}; index_::Int=1)
    # the divisor becomes a dimensionless scalar and the numerator takes the expected
    # dimension: restrictive, but consistent with the forward rule
    return expected_dim, zero_dim(length(expected_dim))
end


function crossp_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}; index_::Int=1)
    if u1[index_] == u2[index_] && u1[index_] == 1
        return compose_units(u1, u2, 1; index_=index_)
    else
        return empty_dim(length(u1))
    end
end

# Solves for the tensor orders only (both operands must be vectors); an operand whose
# order is already 1 is returned as is.
function crossp_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}; index_::Int=1)
    if expected_dim[index_] > 1
        return empty_dim(length(u1))
    elseif u1[index_] == 1 && u2[index_] == 1
        return u1, u2
    elseif u1[index_] == 1
        temp = copy(u2)
        temp[index_] = 1
        return u1, temp
    elseif u2[index_] == 1
        temp = copy(u1)
        temp[index_] = 1
        return temp, u2
    else
        temp = copy(u1)
        temp[index_] = 1
        return temp, temp
    end
end

function tr_unit_backward(u1::Vector{Float16}; index_::Int=1)
    if u1[index_] > 0
        return empty_dim(length(u1))
    end
    temp = copy(u1)
    temp[1] = 2
    return temp
end

# `inv` of a second-order tensor (the only operand `InversionNode` takes): the order is kept
# and the SI exponents change sign. The rule is its own inverse, so it is also the
# backward rule.
function inv_t_unit_forward(u1::Vector{Float16}; index_::Int=1)
    (has_inf16(u1) || u1[index_] != 2) && return empty_dim(length(u1))
    out = zero(Float16) .- u1
    out[index_] = u1[index_]
    return out
end

inv_t_unit_backward(u1::Vector{Float16}; index_::Int=1) = inv_t_unit_forward(u1; index_=index_)

# `hadamard`, the elementwise product of two tensors of one order (`HadamardNode` takes no
# scalars): the order is kept and the SI exponents add.
function hadamard_unit_forward(u1::Vector{Float16}, u2::Vector{Float16}; index_::Int=1)
    (has_inf16(u1) || has_inf16(u2) || u1[index_] != u2[index_] || u1[index_] < 1) &&
        return empty_dim(length(u1))
    return compose_units(u1, u2, u1[index_]; index_=index_)
end

# An operand that already has the expected order is kept and the other takes the rest of
# the units; with neither, the first takes the expected dimension and the second none.
function hadamard_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16}; index_::Int=1)
    order = expected_dim[index_]
    order < 1 && return empty_dim(length(u1)), empty_dim(length(u2))
    if !has_inf16(u1) && u1[index_] == order
        return u1, compose_units(expected_dim, u1, order; index_=index_, quotient=true)
    elseif !has_inf16(u2) && u2[index_] == order
        return compose_units(expected_dim, u2, order; index_=index_, quotient=true), u2
    end
    other = zero_dim(length(expected_dim))
    other[index_] = order
    return copy(expected_dim), other
end



# Scalar unit rules; they treat every slot of a dimension alike.
function equal_unit_forward(u1::Vector{Float16}, u2::Vector{Float16})
    # `all(u1 .== u2)` without materialising the comparison; lengths that differ keep the
    # broadcast (a length-1 operand broadcasts, any other mismatch throws)
    length(u1) == length(u2) || return all(u1 .== u2) ? u1 : empty_dim(length(u1))
    @inbounds for i in eachindex(u1)
        u1[i] == u2[i] || return empty_dim(length(u1))
    end
    return u1
end

function equal_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16})
    return expected_dim, expected_dim
end


function arbitrary_unit_forward(u1::Vector{Float16})
    return u1
end

function arbitrary_unit_backward(u1::Vector{Float16})
    return u1
end

function mul_unit_forward(u1::Vector{Float16}, u2::Vector{Float16})
    return u1 .+ u2
end

function mul_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16})
    if has_inf16(u2) && has_inf16(u1)
        if 0.5 < rand()
            lr = expected_dim
            rr = zero_dim(length(u1))
        else
            rr = expected_dim
            lr = zero_dim(length(u1))
        end
        return lr, rr
    elseif has_inf16(u2)
        return u1, expected_dim .- u1
    elseif has_inf16(u1)
        return expected_dim .- u2, u2
    else
        if isapprox(u1, u2, atol=F16_LOWER_BOUND)
            lr = expected_dim .- expected_dim .÷ 2
            rl = expected_dim .- lr
            return lr, rl
        elseif isapprox(u1, expected_dim, atol=F16_LOWER_BOUND)
            return u1, expected_dim .- u1
        else
            return expected_dim .- u2, u2
        end
    end
end


function div_unit_forward(u1::Vector{Float16}, u2::Vector{Float16})
    return u1 .- u2
end


function div_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16})
    if has_inf16(u2) && has_inf16(u1)
        if 0.5 < rand()
            lr = expected_dim
            rr = zero_dim(length(u1))
        else
            rr = -expected_dim
            lr = zero_dim(length(u1))
        end
        return lr, rr
    elseif has_inf16(u2)
        return u1, .-expected_dim .+ u1
    elseif has_inf16(u1)
        return expected_dim .+ u2, u2
    else
        if isapprox(u1, u2, atol=F16_LOWER_BOUND)
            lr = expected_dim .- expected_dim .÷ 2
            rl = .-expected_dim .+ lr
            return lr, rl
        elseif isapprox(u1, expected_dim, atol=F16_LOWER_BOUND)
            return u1, .- expected_dim .+ u1
        else
            return expected_dim .+ u2, u2
        end
    end
end


function zero_unit_forward(u1::Vector{Float16})
    @inbounds return all(u1 .== 0) ? zero_dim(length(u1)) : empty_dim(length(u1))
end

function zero_unit_backward(u1::Vector{Float16})
    if any(u1 .!= 0)
        return empty_dim(length(u1))
    end
    return zero_dim(length(u1))
end

# Binary zero-unit rules, for an operator such as `^` whose run-time exponent leaves no
# derivable unit: both operands must be dimensionless.
function zero_unit_forward(u1::Vector{Float16}, u2::Vector{Float16})
    return all(u1 .== 0) && all(u2 .== 0) ? zero_dim(length(u1)) : empty_dim(length(u1))
end

function zero_unit_backward(u1::Vector{Float16}, u2::Vector{Float16}, expected_dim::Vector{Float16})
    any(expected_dim .!= 0) && return empty_dim(length(u1)), empty_dim(length(u2))
    return zero_dim(length(u1)), zero_dim(length(u2))
end

# `sign`: an operand of any consistent dimension, a dimensionless result (sign(λx) = sign(x)).
# The backward rule names one operand dimension that works, the dimensionless one.
function sign_unit_forward(u1::Vector{Float16})
    return has_inf16(u1) ? empty_dim(length(u1)) : zero_dim(length(u1))
end

sign_unit_backward(u1::Vector{Float16}) = zero_unit_backward(u1)

# `Float16` factors keep a `Vector{Float16}`; `2.0 .* u1` would promote to `Vector{Float64}`
function sqr_unit_forward(u1::Vector{Float16})
    return Float16(2) .* u1
end

function sqr_unit_backward(u1::Vector{Float16})
    return Float16(0.5) .* u1
end

"""
    TokenLib(physical_dimension_dict, physical_operation_dict, symbol_arity_mapping)

Dimensions, forward unit rules and arities of the alphabet, keyed by symbol id.

# Fields
- `physical_dimension_dict::Ref{OrderedDict{Int8,Vector{Float16}}}`: dimension of each
  terminal
- `physical_operation_dict::Ref{OrderedDict{Int8,Function}}`: forward unit rule of each
  operator, e.g. `mul_unit_forward`
- `symbol_arity_mapping::Ref{OrderedDict{Int8,Int8}}`: arity of each symbol, 0 for a
  terminal

The constructor takes the three `OrderedDict`s and wraps each in a `Ref`.

# Examples
```julia
dims  = OrderedDict{Int8,Vector{Float16}}(3 => Float16[0, 1, 0, 0, 0, 0, 0],  # x1 [m]
                                          4 => Float16[0, 0, 1, 0, 0, 0, 0])  # x2 [s]
rules = OrderedDict{Int8,Function}(1 => mul_unit_forward, 2 => div_unit_forward)
arity = OrderedDict{Int8,Int8}(1 => 2, 2 => 2, 3 => 0, 4 => 0)
token_lib = TokenLib(dims, rules, arity)
```

See also: [`TokenDto`](@ref), [`LibEntry`](@ref)
"""
mutable struct TokenLib
    physical_dimension_dict::Ref{OrderedDict{Int8,Vector{Float16}}}
    physical_operation_dict::Ref{OrderedDict{Int8,Function}}
    symbol_arity_mapping::Ref{OrderedDict{Int8,Int8}}

    function TokenLib(physical_dimension_dict::OrderedDict{Int8,Vector{Float16}},
        physical_operation_dict::OrderedDict{Int8,Function},
        symbol_arity_mapping::OrderedDict{Int8,Int8})
        new(Ref(physical_dimension_dict), Ref(physical_operation_dict), Ref(symbol_arity_mapping))
    end
end

function get_arity(elem::TokenLib, item::Int8)
    return elem.symbol_arity_mapping[][item]
end

function get_physical_dimension(elem::TokenLib, item::Int8)
    return elem.physical_dimension_dict[][item]
end

function get_physical_operation(elem::TokenLib, item::Int8)
    return elem.physical_operation_dict[][item]
end


"""
    LibEntry(symbol_ref::TokenLib)

An empty library expression, grown by `create_lib` one symbol at a time with
`append!(entry, symbol)`.

Symbols are kept in build order, the reverse of prefix order. A terminal starts the entry;
a unary operator applies to the whole entry (never twice in a row, at most twice per
entry); a further terminal waits for a binary operator, which forms `op(terminal, entry)`.
An operator is appended only if the resulting dimension is consistent; a symbol that fits
none of these cases is ignored.

# Fields
- `elements::Vector{Int8}`: the symbols, in build order
- `physical_dimension::Vector{Float16}`: dimension of the entry, without a pending terminal
- `arity_potential::Int`: `0` empty, `1` complete, `2` a terminal is pending
- `homogene::Bool`: `false` while a terminal is pending or after a unit rule threw
- `tokenLib::TokenLib`: the alphabet

Entries compare and hash by `elements`; `clean!` drops a pending terminal.

See also: [`create_lib`](@ref), [`TokenLib`](@ref)
"""
mutable struct LibEntry
    elements::Vector{Int8}
    physical_dimension::Vector{Float16}
    arity_potential::Int
    homogene::Bool
    tokenLib::TokenLib

    function LibEntry(symbol_ref::TokenLib)
        new(Int8[], Float16[], 0, true, symbol_ref)
    end
end


function Base.:(==)(a::LibEntry, b::LibEntry)
    if length(a.elements) != length(b.elements)
        return false
    end
    @simd for i in eachindex(a.elements)
        @inbounds if a.elements[i] != b.elements[i]
            return false
        end
    end
    return true
end


function Base.hash(entry::LibEntry, h::UInt)
    return hash(entry.elements, h)
end

function Base.length(entry::LibEntry)
    return length(entry.elements)
end

function Base.show(io::IO, entry::LibEntry)
    print(io, "LibEntry(elements=$(entry.elements), physical_dimension=$(entry.physical_dimension), arity_potential=$(entry.arity_potential))")
end


function Base.copy(entry::LibEntry)
    new_entry = LibEntry(entry.tokenLib)
    new_entry.elements = copy(entry.elements)
    new_entry.physical_dimension = copy(entry.physical_dimension)
    new_entry.arity_potential = entry.arity_potential
    new_entry.homogene = entry.homogene
    return new_entry
end


@inline function Base.append!(entry::LibEntry, item::Int8)
    arity = get_arity(entry.tokenLib, item)

    if entry.arity_potential == 0 && arity == 0
        entry.arity_potential += 1
        push!(entry.elements, item)
        if haskey(entry.tokenLib.physical_dimension_dict[], item)
            entry.physical_dimension = convert(Vector{Float16}, get_physical_dimension(entry.tokenLib, item))
            entry.homogene = true
        end

    elseif entry.arity_potential == 1
        if arity == 1 && sanity_check(entry, item)
            try
                operation = get_physical_operation(entry.tokenLib, item)
                physical_dimension = convert(Vector{Float16}, operation(entry.physical_dimension))
                if !isempty(physical_dimension) && !has_inf16(physical_dimension)
                    push!(entry.elements, item)
                    entry.physical_dimension = physical_dimension
                    entry.homogene = true
                end
            catch e
                @warn "Issue in Lib. could not apply unary symbol [append-method]: $e"
                entry.homogene = false
            end
        elseif arity == 0
            push!(entry.elements, item)
            entry.arity_potential += 1
            entry.homogene = false
        end
    elseif entry.arity_potential == 2 && arity == 2
        try
            operation = get_physical_operation(entry.tokenLib, item)
            last_dim = get_physical_dimension(entry.tokenLib, entry.elements[end])
            temp_dim = convert(Vector{Float16}, operation(last_dim, entry.physical_dimension))
            if !isempty(temp_dim) && !has_inf16(temp_dim)
                entry.arity_potential = 1
                push!(entry.elements, item)
                entry.physical_dimension = temp_dim
                entry.homogene = true
            end
        catch e
            operation = get_physical_operation(entry.tokenLib, item)
            @warn "Issue in Lib. could not apply binary symbol [append-method] - $operation : $e, $(catch_backtrace()))" 
            entry.homogene = false
        end
    end
end


function clean!(entry::LibEntry)
    if !entry.homogene && !isempty(entry.elements)
        pop!(entry.elements)
        entry.arity_potential = 1
        entry.homogene = true
    end
end


# Whether unary `item` may be appended: the entry is non-empty, does not end in `item`, and
# holds `item` fewer than `max_occurence` times.
function sanity_check(entry::LibEntry, item::Int8; max_occurence::Int=2)
    if isempty(entry.elements) || entry.elements[end] == item
        return false
    else
        count = 0
        for i in length(entry.elements):-1:1
            if entry.elements[i] == item
                count += 1
            end
            if count >= max_occurence
                return false
            end
        end
    end
    return true
end

"""
    create_lib(tokenLib, features, functions, constants; rounds=25, max_permutations=10000)

Build the library: dimensionally consistent expressions grown from `features` by one
symbol of `functions`, `features` or `constants` per round (see `LibEntry`), so at most
`rounds + 1` symbols long. Each round keeps at most `max_permutations` new entries, chosen
at random; the search stops early when a round adds none.

# Returns
An `OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}}` from
`(dimension, length)` to the expressions of that dimension and length, in build order
(the reverse of prefix order).

See also: [`TokenDto`](@ref)
"""
function create_lib(tokenLib::TokenLib, features::Vector{Int8},
    functions::Vector{Int8},
    constants::Vector{Int8};
    rounds::Int=25, max_permutations::Int=10000)

    lib = Set{LibEntry}()
    for feature in features
        entry = LibEntry(tokenLib)
        append!(entry, feature)
        push!(lib, entry)
    end

    search_space = vcat(functions, features, constants)
    new_entries_local = [Vector{LibEntry}() for _ in 1:thread_slots()]

    @inbounds for round in 1:rounds
        for entries in new_entries_local
            empty!(entries)
        end

        lib_array = collect(lib)
        Threads.@threads :static for i in eachindex(lib_array)
            entry = lib_array[i]
            local_entries = new_entries_local[Threads.threadid()]
            for item in search_space
                new_entry = copy(entry)
                append!(new_entry, item)
                if !(new_entry in lib)
                    push!(local_entries, new_entry)
                end
            end
        end

        new_entries = reduce(vcat, new_entries_local)
        unique!(new_entries)
        if length(new_entries) > max_permutations
            shuffle!(new_entries)
            resize!(new_entries, max_permutations)
        end

        union!(lib, new_entries)

        if isempty(new_entries)
            println("No new entries generated. Stopping early.")
            break
        end
    end

    organized_lib = reorganize_lib(lib)
    sort!(collect(keys(organized_lib)), by=key -> key[2])
    return organized_lib
end

# Group the entries by `(dimension, length)` after dropping pending terminals (`clean!`).
function reorganize_lib(old_lib::Set{LibEntry})
    local_libs = [OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}}() for _ in 1:thread_slots()]

    old_lib_array = collect(old_lib)
    Threads.@threads :static for i in eachindex(old_lib_array)
        entry = old_lib_array[i]
        clean!(entry)
        key = (entry.physical_dimension, length(entry.elements))
        local_lib = local_libs[Threads.threadid()]

        if haskey(local_lib, key)
            push!(local_lib[key], entry.elements)
        else
            local_lib[key] = [entry.elements]
        end
    end

    merged_lib = OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}}()
    for local_lib in local_libs
        for (key, value) in local_lib
            if haskey(merged_lib, key)
                append!(merged_lib[key], value)
            else
                merged_lib[key] = value
            end
        end
    end
    return merged_lib
end

function calculate_distance(k1::AbstractVector{Float16}, k2::AbstractVector{Float16})
    sum_sq = 0.0f0
    @inbounds for i in eachindex(k1, k2)
        diff = Float32(k1[i]) - Float32(k2[i])
        sum_sq += diff * diff
    end
    return sqrt(sum_sq)
end

"""
    canonical_dim(d)

A copy of `d` as a `Vector{Float16}` with every `-0.0` turned into `0.0`. Dimensions are
`Dict` keys and `isequal(-0.0, 0.0)` is false, so a dimension computed by negation, as in
the quotient rules, must be canonical to find its key.
"""
@inline canonical_dim(d::AbstractVector) = Float16[Float16(x) + zero(Float16) for x in d]

"""
    is_feasible(d)

Whether `d` is a consistent dimension: non-empty, with every component finite.
"""
@inline is_feasible(d::AbstractVector{Float16}) = !isempty(d) && all(isfinite, d)

"""
    same_dim(a, b)

Whether `a` and `b` have the same length and agree within `F16_LOWER_BOUND` in every
component; false if either has a non-finite component.
"""
@inline function same_dim(a::AbstractVector{Float16}, b::AbstractVector{Float16})
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a, b)
        x = a[i]
        y = b[i]
        (isfinite(x) && isfinite(y) && abs(Float32(x) - Float32(y)) < Float32(F16_LOWER_BOUND)) ||
            return false
    end
    return true
end

"""
    last_operator_position(arity, expr)
    last_operator_position(ix::LibIndex, expr)

Position of the last operator in `expr`; 0 if it has none.
"""
@inline function last_operator_position(arity::Vector{Int8}, expr::AbstractVector{Int8})
    last = 0
    @inbounds for (p, s) in enumerate(expr)
        arity[Int(s)] > 0 && (last = p)
    end
    return last
end

# Last operator position of `[op; a; b]` from `la = length(a)` and the operands' last
# operator positions `loa`, `lob` (0 if none; `lob = 0` for a unary `op`, which has no `b`)
@inline composed_lastop(la::Int, loa::Int, lob::Int) =
    lob > 0 ? 1 + la + lob : loa > 0 ? 1 + loa : 1

# ---------------------------------------------------------------------------------------
#  Library index
# ---------------------------------------------------------------------------------------

"""
    LibIndex(lib, tokenLib, inverse_operation; max_reach_pairs=1_000_000, n_alternatives=8)

Lookup tables over the library `lib` (from `create_lib`), the alphabet `tokenLib` and the
backward rules `inverse_operation`, built once per `TokenDto`. Expressions are stored in
prefix order, ready to be spliced into a gene.

# Fields
- `dims`, `id`: the distinct library dimensions and their ids; `lengths[k]` and
  `entries[(k, len)]`: the lengths and expressions available for id `k`; `pool[k]`,
  `pool_lastop[k]`: all expressions of id `k` and their last operator positions
- `terminals`: the terminals occurring in the library, by dimension
- `reach_dims`, `reach_id`, `reach_expr`, `reach_lastop`, `reach_alts`: the reach set.
  The library stops at `rounds + 1` symbols; the reach set adds every dimension
  `op(a, b)` of a point operator (product or quotient) over two library dimensions, and
  `u(a)` of a unary operator. For each dimension it keeps the most compact expression
  (earliest last operator, then shortest, as the likeliest to fit a gene's head) and, if
  the library lacks the dimension, up to `n_alternatives` compact constructions
  `(op, a, b)` for `random_expression`. Beyond `max_reach_pairs` pairs, only the most
  compact library dimensions are composed.
- `reach_tree`: kd-tree over `reach_dims` (`nothing` if empty); see `nearest_reach`
- `arity`, `fwd`, `bwd`, `dimof`: arity (`-1` if unknown), forward rule, backward rule and
  terminal dimension, as dense vectors indexed by symbol id; `unary`, `binary`: the
  operators that have a forward rule
"""
struct LibIndex
    dims::Vector{Vector{Float16}}
    id::Dict{Vector{Float16},Int}
    lengths::Vector{Vector{Int}}
    entries::Dict{Tuple{Int,Int},Vector{Vector{Int8}}}
    terminals::Dict{Vector{Float16},Vector{Int8}}
    reach_dims::Vector{Vector{Float16}}
    reach_id::Dict{Vector{Float16},Int}
    reach_expr::Vector{Vector{Int8}}
    reach_lastop::Vector{Int}
    reach_alts::Vector{Vector{NTuple{3,Int}}}
    reach_tree::Any
    pool::Vector{Vector{Vector{Int8}}}          # per library dimension: all its entries
    pool_lastop::Vector{Vector{Int}}            # ... and their last operator positions
    arity::Vector{Int8}
    fwd::Vector{Union{Nothing,Function}}
    bwd::Vector{Union{Nothing,Function}}
    dimof::Vector{Vector{Float16}}
    unary::Vector{Int8}
    binary::Vector{Int8}
end

function LibIndex(lib::AbstractDict, tokenLib::TokenLib, inverse_operation::AbstractDict;
    max_reach_pairs::Int=1_000_000, n_alternatives::Int=8)
    arity_map = tokenLib.symbol_arity_mapping[]
    ops = tokenLib.physical_operation_dict[]
    dimdict = tokenLib.physical_dimension_dict[]

    maxsym = 0
    for s in keys(arity_map)
        maxsym = max(maxsym, Int(s))
    end
    arity = fill(Int8(-1), maxsym)
    fwd = Vector{Union{Nothing,Function}}(nothing, maxsym)
    bwd = Vector{Union{Nothing,Function}}(nothing, maxsym)
    dimof = [Float16[] for _ in 1:maxsym]
    unary = Int8[]
    binary = Int8[]
    for (s, a) in arity_map
        Int(s) >= 1 || continue
        arity[s] = a
        if a == 0
            haskey(dimdict, s) && (dimof[s] = convert(Vector{Float16}, dimdict[s]))
        elseif haskey(ops, s)
            fwd[s] = ops[s]
            bwd[s] = get(inverse_operation, s, nothing)
            push!(a == 1 ? unary : binary, s)
        end
    end
    sort!(unary)
    sort!(binary)

    id = Dict{Vector{Float16},Int}()
    dims = Vector{Vector{Float16}}()
    lengths = Vector{Vector{Int}}()
    entries = Dict{Tuple{Int,Int},Vector{Vector{Int8}}}()
    terminal_syms = Set{Int8}()
    for ((dim, len), exprs) in lib
        d = canonical_dim(dim)
        is_feasible(d) || continue
        k = get(id, d, 0)
        if k == 0
            push!(dims, d)
            push!(lengths, Int[])
            k = length(dims)
            id[d] = k
        end
        bucket = get!(() -> Vector{Vector{Int8}}(), entries, (k, len))
        isempty(bucket) && push!(lengths[k], len)
        for e in exprs
            # library expressions are in build order, the reverse of prefix order
            push!(bucket, reverse(e))
            for s in e
                get(arity_map, s, Int8(-1)) == 0 && push!(terminal_syms, s)
            end
        end
    end
    foreach(sort!, lengths)

    terminals = Dict{Vector{Float16},Vector{Int8}}()
    for s in sort!(collect(terminal_syms))
        d = canonical_dim(dimof[s])
        is_feasible(d) && push!(get!(() -> Int8[], terminals, d), s)
    end

    pool = [Vector{Int8}[] for _ in eachindex(dims)]
    pool_lastop = [Int[] for _ in eachindex(dims)]
    for k in eachindex(dims), l in lengths[k], e in entries[(k, l)]
        push!(pool[k], e)
        push!(pool_lastop[k], last_operator_position(arity, e))
    end

    # the most compact expression per library dimension: earliest last operator, then
    # shortest
    best = Vector{Vector{Int8}}(undef, length(dims))
    best_lo = zeros(Int, length(dims))
    for k in eachindex(dims), l in lengths[k], e in entries[(k, l)]
        lo = last_operator_position(arity, e)
        if !isassigned(best, k) || (lo, length(e)) < (best_lo[k], length(best[k]))
            best[k] = e
            best_lo[k] = lo
        end
    end

    # reach set: the library dimensions, then compositions under the point operators and
    # the unary operators; `reach` maps each dimension to its most compact construction
    # `(lastop, len, op, a, b)`
    reach = Dict{Vector{Float16},Tuple{Int,Int,Int,Int,Int}}()
    # the `n_alternatives` most compact constructions of each dimension outside the
    # library, for `random_expression`: compact ones are the likeliest to fit a gene's head
    alts = Dict{Vector{Float16},Vector{NTuple{5,Int}}}()
    function offer!(d, lo, len, op, a, b)
        cur = get(reach, d, nothing)
        (cur === nothing || (lo, len) < (cur[1], cur[2])) && (reach[d] = (lo, len, op, a, b))
        haskey(id, d) && return          # the library's own entries are the alternatives
        list = get!(() -> NTuple{5,Int}[], alts, d)
        if length(list) < n_alternatives
            push!(list, (lo, len, op, a, b))
        else
            worst = argmax(c -> (c[1], c[2]), list)
            j = findfirst(==(worst), list)
            (lo, len) < (worst[1], worst[2]) && (list[j] = (lo, len, op, a, b))
        end
    end
    for k in eachindex(dims)
        reach[dims[k]] = (best_lo[k], length(best[k]), 0, k, 0)
    end
    point = [op for op in binary if fwd[op] === mul_unit_forward || fwd[op] === div_unit_forward]
    # pairs grow with the square of the library: beyond `max_reach_pairs`, only the most
    # compact dimensions are composed, the likeliest to fit a gene
    order = sortperm(eachindex(dims), by=k -> (best_lo[k], length(best[k])))
    n_compose = isempty(point) ? 0 :
                min(length(dims), floor(Int, sqrt(max_reach_pairs / length(point))))
    composable = order[1:n_compose]
    for op in point
        f = fwd[op]
        for a in composable, b in composable
            d = canonical_dim(f(dims[a], dims[b]))
            offer!(d, composed_lastop(length(best[a]), best_lo[a], best_lo[b]),
                1 + length(best[a]) + length(best[b]), Int(op), a, b)
        end
    end
    for op in unary, a in eachindex(dims)
        d = try
            canonical_dim(fwd[op](dims[a]))
        catch
            continue
        end
        is_feasible(d) || continue
        offer!(d, composed_lastop(length(best[a]), best_lo[a], 0), 1 + length(best[a]),
            Int(op), a, 0)
    end
    reach_dims = Vector{Vector{Float16}}()
    reach_id = Dict{Vector{Float16},Int}()
    reach_expr = Vector{Vector{Int8}}()
    reach_lastop = Int[]
    reach_alts = Vector{Vector{NTuple{3,Int}}}()
    for (d, (lo, _, op, a, b)) in reach
        is_feasible(d) || continue
        push!(reach_dims, d)
        reach_id[d] = length(reach_dims)
        push!(reach_expr, op == 0 ? best[a] :
                          b == 0 ? vcat(Int8(op), best[a]) : vcat(Int8(op), best[a], best[b]))
        push!(reach_lastop, lo)
        push!(reach_alts, NTuple{3,Int}[(c[3], c[4], c[5]) for c in get(alts, d, NTuple{5,Int}[])])
    end
    reach_tree = isempty(reach_dims) ? nothing : KDTree(Float64.(reduce(hcat, reach_dims)))

    return LibIndex(dims, id, lengths, entries, terminals, reach_dims, reach_id, reach_expr,
        reach_lastop, reach_alts, reach_tree, pool, pool_lastop, arity, fwd, bwd, dimof,
        unary, binary)
end

"""
    nearest_reach(ix, d, k)

Reach-set ids of the `k` dimensions nearest to `d` (kd-tree query), nearest first; empty if
`d` is inconsistent or the reach set is empty.
"""
function nearest_reach(ix::LibIndex, d::AbstractVector{Float16}, k::Int)
    (ix.reach_tree === nothing || !is_feasible(d)) && return Int[]
    k = min(k, length(ix.reach_dims))
    idxs, _ = knn(ix.reach_tree, Float64.(d), k, true)
    return idxs
end

# Forward rule of symbol `s` applied to `args`; inconsistent if `s` has none or it throws.
@inline function apply_fwd(ix::LibIndex, s::Int, args::Vararg{Vector{Float16},N}) where {N}
    f = ix.fwd[s]
    f === nothing && return empty_dim(length(first(args)))
    d = try
        f(args...)
    catch
        return empty_dim(length(first(args)))
    end
    return d isa Vector{Float16} ? d : convert(Vector{Float16}, d)
end

"""
    subtree_end(ix, expr, pos)

Index of the last symbol of the prefix subexpression starting at `pos`; 0 if that
subexpression is incomplete or contains an unknown symbol.
"""
function subtree_end(ix::LibIndex, expr::AbstractVector{Int8}, pos::Int)
    need = 1
    @inbounds for j in pos:length(expr)
        s = Int(expr[j])
        (1 <= s <= length(ix.arity) && ix.arity[s] >= 0) || return 0
        need += ix.arity[s] - 1
        need == 0 && return j
    end
    return 0
end

@inline last_operator_position(ix::LibIndex, expr::AbstractVector{Int8}) =
    last_operator_position(ix.arity, expr)

"""
    draw_entry(ix, k, maxl, rng; maxlo=typemax(Int))

A random library expression of dimension id `k` with at most `maxl` symbols and its last
operator at or before position `maxlo`; `nothing` if there is none. The stored vector is
returned, not a copy.
"""
function draw_entry(ix::LibIndex, k::Int, maxl::Int, rng::AbstractRNG; maxlo::Int=typemax(Int))
    p = ix.pool[k]
    lo = ix.pool_lastop[k]
    n = count(j -> length(p[j]) <= maxl && lo[j] <= maxlo, eachindex(p))
    n == 0 && return nothing
    pick = rand(rng, 1:n)
    for j in eachindex(p)
        if length(p[j]) <= maxl && lo[j] <= maxlo
            pick -= 1
            pick == 0 && return p[j]
        end
    end
    return nothing
end

"""
    random_expression(ix, d; max_len, head_len=0, rng=Random.default_rng(), tries=8)
        -> Vector{Int8} or nothing

A random prefix expression of dimension `d` with at most `max_len` symbols and, if
`head_len > 0`, its last operator at or before position `head_len`; `nothing` if none is
found. It draws a library expression of `d` if one fits; otherwise up to `tries` random
constructions `op(a, b)` or `u(a)` from the reach set's alternatives for `d`, composed
from freshly drawn library expressions so that repeated calls differ; last, the reach
set's most compact expression of `d` if it fits. `d` must be canonical (`canonical_dim`).
"""
function random_expression(ix::LibIndex, d::Vector{Float16}; max_len::Int, head_len::Int=0,
    rng::AbstractRNG=Random.default_rng(), tries::Int=8)
    H = head_len > 0 ? head_len : typemax(Int) ÷ 2
    fits(e) = length(e) <= max_len && last_operator_position(ix, e) <= H
    k = get(ix.id, d, 0)
    if k != 0
        e = draw_entry(ix, k, max_len, rng; maxlo=H)
        e === nothing || return copy(e)
    end
    r = get(ix.reach_id, d, 0)
    r == 0 && return nothing
    pairs = ix.reach_alts[r]
    for _ in 1:(isempty(pairs) ? 0 : tries)
        op, a, b = rand(rng, pairs)
        if b == 0
            # [op; a]: a's operators move one place right
            ea = draw_entry(ix, a, max_len - 1, rng; maxlo=H - 1)
            ea === nothing && continue
            return vcat(Int8(op), ea)
        end
        # [op; a; b]: pick b first, then an `a` that keeps the last operator in the head --
        # b's operators land behind all of a, a's one place right of where they were
        eb = draw_entry(ix, b, max_len - 2, rng; maxlo=H - 2)
        eb === nothing && continue
        lob = last_operator_position(ix, eb)
        ea = lob > 0 ?
             draw_entry(ix, a, min(max_len - 1 - length(eb), H - 1 - lob), rng) :
             draw_entry(ix, a, max_len - 1 - length(eb), rng; maxlo=H - 1)
        ea === nothing && continue
        e = vcat(Int8(op), ea, eb)
        fits(e) && return e
    end
    fits(ix.reach_expr[r]) && return copy(ix.reach_expr[r])
    return nothing
end

"""
    TokenDto(tokenLib, point_operations, lib, inverse_operation, gene_count; head_len=-1)

The alphabet, the backward unit rules and the library, with the `LibIndex` built from them.

# Fields
- `tokenLib::TokenLib`: dimensions, forward rules and arities
- `point_operations::Vector{Int8}`: the product and quotient operators, which
  `repair_infeasible_ops!` substitutes
- `lib::Ref{OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}}}`: the library
  from `create_lib`, `(dimension, length) => expressions`
- `inverse_operation::Dict{Int8,Function}`: backward unit rule of each operator
- `gene_count::Int`, `head_len::Int`: chromosome geometry, not read by `SBPUtils`
- `index::LibIndex`: built by the constructor from `lib`, `tokenLib` and
  `inverse_operation`

# Examples
```julia
lib = create_lib(token_lib, Int8[3, 4], Int8[1, 2], Int8[])   # `token_lib` as in `TokenLib`
backward = Dict{Int8,Function}(1 => mul_unit_backward, 2 => div_unit_backward)
dto = TokenDto(token_lib, Int8[1, 2], lib, backward, 3; head_len=6)
```

See also: [`TokenLib`](@ref), [`create_lib`](@ref), [`correct_genes!`](@ref)
"""
mutable struct TokenDto
    tokenLib::TokenLib
    point_operations::Vector{Int8}
    lib::Ref{OrderedDict{Tuple{Vector{Float16},Int},Vector{Vector{Int8}}}}
    inverse_operation::Dict{Int8,Function}
    gene_count::Int
    head_len::Int
    index::LibIndex

    function TokenDto(tokenLib, point_operations, lib, inverse_operation, gene_count; head_len=-1)
        new(tokenLib, point_operations, Ref(lib), inverse_operation, gene_count, head_len,
            LibIndex(lib, tokenLib, inverse_operation))
    end

end

"""
    TempComputeTree(symbol, depend_on, vector_dimension, tokenDto)

Pointer tree over a prefix expression, for the tree-level API: operators are nodes,
terminals are `Int8` leaves. Build one with `create_compute_tree`.

# Fields
- `symbol::Int8`: the node's operator
- `depend_on::Vector{Union{TempComputeTree,Int8}}`: the operands, as subtrees or terminals
- `vector_dimension::Vector{Float16}`: cached dimension of the subtree
- `tokenDto::TokenDto`: alphabet, unit rules and library
- `depend_on_total_number::Int`: number of symbols in the subtree as last counted by
  `flatten_dependents`; the number of operands until then
- `exchange_len::Int`: `-1`; not used by `SBPUtils`
- `modified::Bool`: set on construction and by edits, cleared by
  `calculate_vector_dimension!`

`flatten_dependents(tree)` returns the tree as a prefix expression and `flush!(tree)`
clears the root's cached dimension; `calculate_vector_dimension!` computes the dimension,
and `propagate_necessary_changes!` and `repair_infeasible_ops!` edit the tree.

See also: [`propagate_necessary_changes!`](@ref), [`TokenDto`](@ref)
"""
mutable struct TempComputeTree
    symbol::Int8
    depend_on::Vector{Union{TempComputeTree,Int8}}
    vector_dimension::Vector{Float16}
    tokenDto::TokenDto
    depend_on_total_number::Int
    exchange_len::Int
    modified::Bool

    function TempComputeTree(symbol::Int8,
        depend_on::Vector{T}=Union{TempComputeTree,Int8}[],
        vector_dimension::Vector{Float16}=Float16[],
        tokenDto::TokenDto=nothing) where {T}
        new(symbol,
            convert(Vector{Union{TempComputeTree,Int8}}, depend_on),
            vector_dimension,
            tokenDto,
            length(depend_on),
            -1, true)
    end
end

function flatten_dependents(tree::TempComputeTree)
    ret_val = [tree.symbol]
    for elem in tree.depend_on
        if elem isa TempComputeTree
            append!(ret_val, flatten_dependents(elem))
        else
            push!(ret_val, elem)
        end
    end
    tree.depend_on_total_number = length(ret_val)
    return ret_val
end

function flush!(tree::TempComputeTree)
    tree.vector_dimension = []
end

# `child_dimension!` recomputes a subtree's dimension, `child_dimension` reads its cache
@inline child_dimension!(elem::TempComputeTree, ::TokenLib) = calculate_vector_dimension!(elem)
@inline child_dimension!(elem::Int8, tokenLib::TokenLib) = get_physical_dimension(tokenLib, elem)
@inline child_dimension(elem::TempComputeTree, ::TokenLib) = elem.vector_dimension
@inline child_dimension(elem::Int8, tokenLib::TokenLib) = get_physical_dimension(tokenLib, elem)

# `f` applied to the operands' dimensions, passed positionally: collecting them with `map`
# over the `Union{TempComputeTree,Int8}` vector is type-unstable and allocates per node.
@inline function apply_to_children(f, tree::TempComputeTree, tokenLib::TokenLib, get_dim)
    deps = tree.depend_on
    length(deps) == 1 && return f(get_dim(deps[1], tokenLib))
    length(deps) == 2 && return f(get_dim(deps[1], tokenLib), get_dim(deps[2], tokenLib))
    return f(map(d -> get_dim(d, tokenLib), deps)...)
end

"""
    calculate_vector_dimension!(tree::TempComputeTree)

Recompute and cache the dimension of every node of `tree` and return the root's. The tree
itself is not edited; `repair_infeasible_ops!` is the rewriting counterpart.
"""
function calculate_vector_dimension!(tree::TempComputeTree)
    tokenLib = tree.tokenDto.tokenLib
    function_op = tokenLib.physical_operation_dict[][tree.symbol]
    tree.vector_dimension = apply_to_children(function_op, tree, tokenLib, child_dimension!)
    tree.modified = false
    return tree.vector_dimension
end

"""
    repair_infeasible_ops!(tree::TempComputeTree)

Bottom-up sweep that replaces every binary node with an inconsistent dimension (e.g. `+`
between different units) by a random operator from `tokenDto.point_operations`, whose
dimension composes from consistent operands. Returns the tree's dimension afterwards.

It is separate from `calculate_vector_dimension!` so that computing a dimension never
rewrites the tree. The repair engine (`correct_genes!`, `propagate_necessary_changes!`)
does not call it.
"""
function repair_infeasible_ops!(tree::TempComputeTree)
    tokenLib = tree.tokenDto.tokenLib
    for elem in tree.depend_on
        elem isa TempComputeTree && repair_infeasible_ops!(elem)
    end
    function_op = tokenLib.physical_operation_dict[][tree.symbol]
    tree.vector_dimension = apply_to_children(function_op, tree, tokenLib, child_dimension)
    if length(tree.depend_on) == 2 && !is_feasible(tree.vector_dimension) &&
       !isempty(tree.tokenDto.point_operations)
        tree.symbol = rand(tree.tokenDto.point_operations)
        tree.modified = true
        function_op = tokenLib.physical_operation_dict[][tree.symbol]
        tree.vector_dimension = apply_to_children(function_op, tree, tokenLib, child_dimension)
    end
    return tree.vector_dimension
end


# ---------------------------------------------------------------------------------------
#  Repair engine
#
#  Works on the chromosome's karva string -- the prefix expression the evaluator walks,
#  connectors first -- rather than on a pointer tree. The subtree at position `i` spans
#  `i:ends[i]` and has dimension `dims[i]`; both are recomputed in one right-to-left pass
#  (`refresh!`) after every edit, which is cheap for expressions of a few dozen symbols
#  and keeps every cached dimension exact. `owner[i]` is 0 for a connector and `k` for a
#  symbol of gene `k`, so an edit can be checked against the capacity of its gene.
#
#  Moves, cheapest first, at a node that must take dimension `E`:
#    1. swap its operator (or terminal) for one that yields `E` as it stands;
#    2. drop a unary operator whose operand already has `E`;
#    3. push the requirement into the operands with the backward rule -- for a product or
#       quotient keeping either operand, or splitting `E` between both with dimensions the
#       kd-tree proposes; at a connector, planning the split across the genes below it;
#    4. replace the whole subtree by an expression of dimension `E` from the library or
#       the reach set that fits the gene (never at a connector, which would break the
#       gene structure).
#  A terminal goes from move 1 straight to move 4. A failed move is rolled back before
#  the next is tried.
# ---------------------------------------------------------------------------------------

mutable struct RepairState
    ix::LibIndex
    expr::Vector{Int8}
    owner::Vector{Int}
    dims::Vector{Vector{Float16}}
    ends::Vector{Int}
    glen::Vector{Int}           # active length per gene
    maxlen::Vector{Int}         # per gene: the longest active expression it can hold
    headlim::Vector{Int}        # per gene: the last position an operator may occupy
    conn_syms::Vector{Int8}     # operators allowed at a connector
    rng::AbstractRNG
    work::Int                   # node visits left for the current attempt
    root_plans::Int             # requirements found at the root connector; -1 until known
end

RepairState(ix, expr, owner, glen, maxlen, headlim, conn_syms, rng, work) =
    RepairState(ix, expr, owner, Vector{Float16}[], Int[], glen, maxlen, headlim, conn_syms,
        rng, work, -1)

"""
    refresh!(st::RepairState) -> Bool

Recompute `st.dims` and `st.ends` in one right-to-left pass; return whether `st.expr` is a
single complete prefix expression over known symbols.
"""
function refresh!(st::RepairState)
    ix = st.ix
    expr = st.expr
    n = length(expr)
    resize!(st.dims, n)
    resize!(st.ends, n)
    @inbounds for i in n:-1:1
        s = Int(expr[i])
        (1 <= s <= length(ix.arity)) || return false
        a = ix.arity[s]
        if a == 0
            st.dims[i] = ix.dimof[s]
            st.ends[i] = i
        elseif a == 1
            i < n || return false
            st.dims[i] = apply_fwd(ix, s, st.dims[i+1])
            st.ends[i] = st.ends[i+1]
        elseif a == 2
            i < n || return false
            j = st.ends[i+1] + 1
            j <= n || return false
            st.dims[i] = apply_fwd(ix, s, st.dims[i+1], st.dims[j])
            st.ends[i] = st.ends[j]
        else
            return false
        end
    end
    return n > 0 && st.ends[1] == n
end

@inline snapshot(st::RepairState) = (copy(st.expr), copy(st.owner), copy(st.glen))

@inline function restore!(st::RepairState, snap)
    copy!(st.expr, snap[1])
    copy!(st.owner, snap[2])
    copy!(st.glen, snap[3])
    refresh!(st)
    return nothing
end

function gene_range(st::RepairState, g::Int)
    first = 0
    last = 0
    @inbounds for i in eachindex(st.owner)
        if st.owner[i] == g
            first == 0 && (first = i)
            last = i
        end
    end
    return first:last
end

function gene_fits(st::RepairState, g::Int)
    g == 0 && return true
    r = gene_range(st, g)
    length(r) <= st.maxlen[g] || return false
    lim = st.headlim[g]
    @inbounds for (p, i) in enumerate(r)
        p > lim && st.ix.arity[Int(st.expr[i])] > 0 && return false
    end
    return true
end

"""
    room_at(st, pos)

The number of symbols the subtree at `pos` may occupy without its gene exceeding
`st.maxlen`.
"""
@inline room_at(st::RepairState, pos::Int) =
    (st.ends[pos] - pos + 1) + (st.maxlen[st.owner[pos]] - st.glen[st.owner[pos]])

"""
    requirement_cost(st, pos, d)

Estimated cost of making the subtree at `pos` take dimension `d`: 0 if it has it, 1 if a
terminal swap gives it, 2 if the reach set's expression of `d` fits the room left in its
gene (`room_at`), 3 otherwise (only propagation can tell).
"""
function requirement_cost(st::RepairState, pos::Int, d::Vector{Float16})
    same_dim(st.dims[pos], d) && return 0
    ix = st.ix
    ix.arity[Int(st.expr[pos])] == 0 && haskey(ix.terminals, d) && return 1
    st.owner[pos] == 0 && return 3
    k = get(ix.reach_id, d, 0)
    k != 0 && length(ix.reach_expr[k]) <= room_at(st, pos) && return 2
    return 3
end

"""
    gene_reachable(st, pos, d)

Whether the gene rooted at `pos` can take dimension `d`: it has it, or the reach set's
expression of `d` has its last operator within the gene's head.
"""
function gene_reachable(st::RepairState, pos::Int, d::Vector{Float16})
    same_dim(st.dims[pos], d) && return true
    k = get(st.ix.reach_id, d, 0)
    return k != 0 && st.ix.reach_lastop[k] <= st.headlim[st.owner[pos]]
end

function swap_terminal!(st::RepairState, i::Int, E::Vector{Float16})
    cands = get(st.ix.terminals, E, nothing)
    (cands === nothing || isempty(cands)) && return false
    st.expr[i] = rand(st.rng, cands)
    return refresh!(st)
end

function swap_operator!(st::RepairState, i::Int, E::Vector{Float16})
    ix = st.ix
    s = st.expr[i]
    a = ix.arity[Int(s)]
    syms = a == 2 ? (st.owner[i] == 0 ? st.conn_syms : ix.binary) : ix.unary
    isempty(syms) && return false
    if a == 2
        dl = st.dims[i+1]
        dr = st.dims[st.ends[i+1]+1]
        for op in shuffle(st.rng, syms)
            op == s && continue
            if same_dim(apply_fwd(ix, Int(op), dl, dr), E)
                st.expr[i] = op
                return refresh!(st)
            end
        end
    else
        dc = st.dims[i+1]
        for op in shuffle(st.rng, syms)
            op == s && continue
            if same_dim(apply_fwd(ix, Int(op), dc), E)
                st.expr[i] = op
                return refresh!(st)
            end
        end
    end
    return false
end

function drop_unary!(st::RepairState, i::Int, E::Vector{Float16})
    g = st.owner[i]
    (g == 0 || !same_dim(st.dims[i+1], E)) && return false
    # every later symbol moves one step towards the head, so the gene still fits
    deleteat!(st.expr, i)
    deleteat!(st.owner, i)
    st.glen[g] -= 1
    return refresh!(st)
end

function try_splice!(st::RepairState, i::Int, g::Int, entry::AbstractVector{Int8})
    size = st.ends[i] - i + 1
    snap = snapshot(st)
    stop = st.ends[i]
    splice!(st.expr, i:stop, entry)
    splice!(st.owner, i:stop, fill(g, length(entry)))
    st.glen[g] += length(entry) - size
    refresh!(st) && gene_fits(st, g) && return true
    restore!(st, snap)
    return false
end

function replace_subtree!(st::RepairState, i::Int, E::Vector{Float16})
    g = st.owner[i]
    g == 0 && return false
    ix = st.ix
    size = st.ends[i] - i + 1
    maxl = room_at(st, i)

    # library expressions of `E` first, nearest in length to the subtree they replace, so
    # the gene keeps its layout where the library allows
    k = get(ix.id, E, 0)
    if k != 0
        cands = Int[l for l in ix.lengths[k] if l <= maxl]
        shuffle!(st.rng, cands)
        sort!(cands, by=l -> abs(l - size))
        for l in cands
            bucket = ix.entries[(k, l)]
            for _ in 1:min(3, length(bucket))
                try_splice!(st, i, g, rand(st.rng, bucket)) && return true
            end
        end
    end
    # then random expressions from `random_expression`, so that repeated repairs differ,
    # held to the head room left at this position; last the reach set's most compact one
    head_room = st.headlim[g] - (i - first(gene_range(st, g)))
    for _ in 1:(head_room >= 1 ? 3 : 0)
        e = random_expression(ix, E; max_len=maxl, head_len=head_room, rng=st.rng)
        e === nothing && break
        try_splice!(st, i, g, e) && return true
    end
    r = get(ix.reach_id, E, 0)
    r != 0 && length(ix.reach_expr[r]) <= maxl && try_splice!(st, i, g, ix.reach_expr[r]) &&
        return true
    return false
end

function propagate_unary!(st::RepairState, i::Int, E::Vector{Float16})
    bwd = st.ix.bwd[Int(st.expr[i])]
    bwd === nothing && return false
    # the backward rule is applied to the required `E`: applied to the node's current
    # dimension, it would only re-request what the operand already has
    Ec = try
        canonical_dim(bwd(E))
    catch
        return false
    end
    is_feasible(Ec) || return false
    snap = snapshot(st)
    repair!(st, i + 1, Ec) && same_dim(st.dims[i], E) && return true
    restore!(st, snap)
    return false
end

@inline op_kind(ix::LibIndex, op) =
    (f = ix.fwd[Int(op)]; f === mul_unit_forward ? :mul : f === div_unit_forward ? :div :
                          f === equal_unit_forward ? :equal : :other)

"""
    left_for(kind, E, dr)

The dimension the left operand needs for `op(left, right)` to have dimension `E`, given
the right operand's `dr`, for an operator of kind `:mul`, `:div` or `:equal`; `nothing` if
there is none (`:equal` with `dr` not `E`, or another kind).
"""
@inline function left_for(kind::Symbol, E::Vector{Float16}, dr::Vector{Float16})
    kind === :mul && return canonical_dim(E .- dr)
    kind === :div && return canonical_dim(E .+ dr)
    kind === :equal && return same_dim(dr, E) ? E : nothing
    return nothing
end

"""
    right_for(kind, E, dl)

The dimension the right operand needs for `op(left, right)` to have dimension `E`, given
the left operand's `dl`; the counterpart of `left_for`.
"""
@inline function right_for(kind::Symbol, E::Vector{Float16}, dl::Vector{Float16})
    kind === :mul && return canonical_dim(E .- dl)
    kind === :div && return canonical_dim(dl .- E)
    kind === :equal && return same_dim(dl, E) ? E : nothing
    return nothing
end

"""
    backward_pair(ix, op, dl, dr, E)

The operands' requirements for `op(left, right)` to have dimension `E`, from `op`'s
backward rule, as a pair of consistent dimensions; `nothing` if there is no such pair.
Used for operators other than products, quotients and equal-unit operators, such as the
tensor operators.
"""
function backward_pair(ix::LibIndex, op, dl::Vector{Float16}, dr::Vector{Float16},
    E::Vector{Float16})
    bwd = ix.bwd[Int(op)]
    bwd === nothing && return nothing
    r = try
        bwd(dl, dr, E)
    catch
        return nothing
    end
    (r isa Tuple && length(r) == 2) || return nothing
    a = canonical_dim(r[1])
    b = canonical_dim(r[2])
    return is_feasible(a) && is_feasible(b) ? (a, b) : nothing
end

const Requirement = Tuple{Int8,Vector{Float16},Vector{Float16}}

"""
    gene_requirements(st, i, E) -> Vector{Requirement}

Requirements `(operator, left, right)` to try at the binary gene node `i`, in order:
`(s, E, E)` if the node's operator `s` is an equal-unit operator; then, for every point
operator, those keeping one operand's dimension and giving the other the complement,
ranked by `requirement_cost` plus a penalty for changing the operator (so a `+` between
unlike dimensions can become a product or quotient); then, if `s` is a product or
quotient, splits of `E` over both operands (`split_proposals`) and, when neither operand
is consistent, `(E, 0)` and `(0, E)` (`(0, -E)` for a quotient). Any other operator gets
only its backward rule's requirement. Inconsistent requirements are dropped.
"""
function gene_requirements(st::RepairState, i::Int, E::Vector{Float16})
    ix = st.ix
    s = st.expr[i]
    lpos = i + 1
    rpos = st.ends[lpos] + 1
    dl = st.dims[lpos]
    dr = st.dims[rpos]
    kind = op_kind(ix, s)
    opts = Requirement[]
    if kind === :equal
        push!(opts, (s, E, E))
    elseif kind === :other
        r = backward_pair(ix, s, dl, dr, E)
        r === nothing || push!(opts, (s, r...))
        return opts
    end
    scored = Tuple{Float64,Requirement}[]
    for op in ix.binary
        k = op_kind(ix, op)
        (k === :mul || k === :div) || continue
        penalty = op == s ? 0.0 : 1.5     # changing the operator is one more edit
        if is_feasible(dl)
            er = right_for(k, E, dl)
            push!(scored, (penalty + requirement_cost(st, rpos, er) + rand(st.rng), (op, dl, er)))
        end
        if is_feasible(dr)
            el = left_for(k, E, dr)
            push!(scored, (penalty + requirement_cost(st, lpos, el) + rand(st.rng), (op, el, dr)))
        end
    end
    sort!(scored, by=first)
    for (_, o) in scored
        push!(opts, o)
    end
    kind === :mul || kind === :div || return filter!(o -> is_feasible(o[2]) && is_feasible(o[3]), opts)
    # both operands change: pair kd-tree proposals for one with the complement for the other
    for x in split_proposals(st, lpos, rpos, dl, dr, E, kind)
        push!(opts, (s, x...))
    end
    if !is_feasible(dl) && !is_feasible(dr)
        z = zeros(Float16, length(E))
        push!(opts, (s, E, z))
        push!(opts, (s, z, canonical_dim(kind === :mul ? E : .-E)))
    end
    return filter!(o -> is_feasible(o[2]) && is_feasible(o[3]), opts)
end

"""
    split_proposals(st, lpos, rpos, dl, dr, E, kind; k=12, keep=3)

Splits `(left, right)` of `E` over a product or quotient (`kind`) in which both operands
change: each of the `k` reachable dimensions nearest to one operand's current dimension
(or to `E` if that is inconsistent) is paired with the complement for the other operand.
Returns the `keep` cheapest pairs by summed `requirement_cost`, among those costing at
most 4, with random tie-breaking.
"""
function split_proposals(st::RepairState, lpos::Int, rpos::Int, dl::Vector{Float16},
    dr::Vector{Float16}, E::Vector{Float16}, kind::Symbol; k::Int=12, keep::Int=3)
    ix = st.ix
    out = Tuple{Float64,Vector{Float16},Vector{Float16}}[]
    seen = Set{Vector{Float16}}()
    for id in nearest_reach(ix, is_feasible(dl) ? dl : E, k)
        x = ix.reach_dims[id]
        x in seen && continue
        push!(seen, x)
        y = right_for(kind, E, x)
        c = requirement_cost(st, lpos, x) + requirement_cost(st, rpos, y)
        c <= 4 && push!(out, (c + rand(st.rng), x, y))
    end
    for id in nearest_reach(ix, is_feasible(dr) ? dr : E, k)
        y = ix.reach_dims[id]
        x = left_for(kind, E, y)
        (x === nothing || x in seen) && continue
        push!(seen, x)
        c = requirement_cost(st, lpos, x) + requirement_cost(st, rpos, y)
        c <= 4 && push!(out, (c + rand(st.rng), x, y))
    end
    sort!(out, by=first)
    return [(x, y) for (_, x, y) in out[1:min(keep, length(out))]]
end

"""
    near_ids!(near, st, pos; k=48)

Reach-set ids among the `k` dimensions nearest to the current dimension of the gene rooted
at `pos` (to the dimensionless one if the gene is inconsistent) whose expression fits the
gene's head; a split built from these changes the gene little. Cached in `near` by `pos`,
since the expression does not change during planning.
"""
function near_ids!(near::Dict{Int,Vector{Int}}, st::RepairState, pos::Int; k::Int=48)
    get!(near, pos) do
        d = st.dims[pos]
        centre = is_feasible(d) ? d : zeros(Float16, length(d))
        lim = st.headlim[st.owner[pos]]
        Int[id for id in nearest_reach(st.ix, centre, k) if st.ix.reach_lastop[id] <= lim]
    end
end

"""
    req_ids!(reqnear, st, D; k=32)

Reach-set ids of the `k` dimensions nearest to the requirement `D`, cached in `reqnear`.
As a gene's dimension, they give the splits in which that gene carries (nearly) all of
`D` and its sibling is (nearly) dimensionless.
"""
function req_ids!(reqnear::Dict{Vector{Float16},Vector{Int}}, st::RepairState,
    D::Vector{Float16}; k::Int=32)
    get!(() -> nearest_reach(st.ix, D, k), reqnear, D)
end

@inline fits_gene(st::RepairState, pos::Int, id::Int) =
    st.ix.reach_lastop[id] <= st.headlim[st.owner[pos]]

mutable struct PlanSession
    near::Dict{Int,Vector{Int}}
    reqnear::Dict{Vector{Float16},Vector{Int}}
    budget::Int
end
PlanSession(budget::Int) =
    PlanSession(Dict{Int,Vector{Int}}(), Dict{Vector{Float16},Vector{Int}}(), budget)

"""
    plannable(st, pos, D, ps) -> Bool

Whether the subtree at `pos` can take dimension `D` by choosing connector operators and
giving each gene below it a reachable dimension. A gene answers with `gene_reachable`. A
connector tries every allowed operator. For a product or quotient whose operands are both
genes it solves the split as a two-sum: each candidate for one gene leaves a hash lookup
for the other's complement. With a connector on the left, it tries candidates for the
right gene and recurses into the left operand; other operators pass their backward-rule
or equal-unit requirement down. Candidates come from `near_ids!` and `req_ids!`, cached in
the `PlanSession` `ps`, whose budget bounds the number of checks.
"""
function plannable(st::RepairState, pos::Int, D::Vector{Float16}, ps::PlanSession)
    st.owner[pos] != 0 && return gene_reachable(st, pos, D)
    ix = st.ix
    lpos = pos + 1
    rpos = st.ends[lpos] + 1
    dl = st.dims[lpos]
    dr = st.dims[rpos]
    for op in st.conn_syms
        ps.budget < 0 && return false
        kind = op_kind(ix, op)
        if kind === :other
            ps.budget -= 1
            r = backward_pair(ix, op, dl, dr, D)
            r !== nothing && gene_reachable(st, rpos, r[2]) && plannable(st, lpos, r[1], ps) &&
                return true
        elseif kind === :equal
            ps.budget -= 1
            gene_reachable(st, rpos, D) && plannable(st, lpos, D, ps) && return true
        elseif st.owner[lpos] != 0
            # keep either gene, then a two-sum over the candidates of each
            ps.budget -= 2
            is_feasible(dl) && gene_reachable(st, rpos, right_for(kind, D, dl)) && return true
            is_feasible(dr) && gene_reachable(st, lpos, left_for(kind, D, dr)) && return true
            for ids in (near_ids!(ps.near, st, rpos), req_ids!(ps.reqnear, st, D)), id in ids
                fits_gene(st, rpos, id) || continue
                ps.budget -= 1
                gene_reachable(st, lpos, left_for(kind, D, ix.reach_dims[id])) && return true
            end
            for ids in (near_ids!(ps.near, st, lpos), req_ids!(ps.reqnear, st, D)), id in ids
                fits_gene(st, lpos, id) || continue
                ps.budget -= 1
                gene_reachable(st, rpos, right_for(kind, D, ix.reach_dims[id])) && return true
            end
        else
            if is_feasible(dr)
                ps.budget -= 1
                plannable(st, lpos, left_for(kind, D, dr), ps) && return true
            end
            for ids in (near_ids!(ps.near, st, rpos; k=16), req_ids!(ps.reqnear, st, D; k=16)),
                id in ids
                fits_gene(st, rpos, id) || continue
                ps.budget -= 1
                plannable(st, lpos, left_for(kind, D, ix.reach_dims[id]), ps) && return true
                ps.budget < 0 && return false
            end
        end
    end
    return false
end

"""
    connector_requirements(st, i, E; keep=6) -> Vector{Requirement}

Requirements `(operator, left, right)` at connector `i`, whose right operand is a gene and
whose left operand holds all genes before it, so `E` is split across the genes. The
connector's own operator comes first if it is allowed, then the other allowed operators in
random order.
For a product or quotient, the right gene keeps its dimension, takes the complement of the
left side's, or takes a reachable dimension near its current one or near `E`. A proposal
is kept only if the right gene can take its share (`gene_reachable`) and the left side
too (`plannable`); at most `keep` are returned.
"""
function connector_requirements(st::RepairState, i::Int, E::Vector{Float16}; keep::Int=6)
    ix = st.ix
    lpos = i + 1
    rpos = st.ends[lpos] + 1
    dl = st.dims[lpos]
    dr = st.dims[rpos]
    s = st.expr[i]
    # the connector's own operator first, if it is an allowed one
    others = shuffle(st.rng, [op for op in st.conn_syms if op != s])
    ops = s in st.conn_syms ? vcat(s, others) : others
    out = Requirement[]
    ps = PlanSession(20_000)
    for op in ops
        kind = op_kind(ix, op)
        if kind === :other
            r = backward_pair(ix, op, dl, dr, E)
            r !== nothing && gene_reachable(st, rpos, r[2]) && plannable(st, lpos, r[1], ps) &&
                push!(out, (op, r...))
        elseif kind === :equal
            gene_reachable(st, rpos, E) && plannable(st, lpos, E, ps) && push!(out, (op, E, E))
        else
            cands = Vector{Float16}[]
            is_feasible(dr) && push!(cands, dr)
            is_feasible(dl) && push!(cands, right_for(kind, E, dl))
            for ids in (near_ids!(ps.near, st, rpos; k=24), req_ids!(ps.reqnear, st, E; k=24)),
                id in ids
                fits_gene(st, rpos, id) && push!(cands, ix.reach_dims[id])
            end
            for DR in cands
                gene_reachable(st, rpos, DR) || continue
                DL = left_for(kind, E, DR)
                plannable(st, lpos, DL, ps) || continue
                push!(out, (op, DL, DR))
                (length(out) >= keep || ps.budget < 0) && break
            end
        end
        (length(out) >= keep || ps.budget < 0) && break
    end
    return out
end

function propagate_binary!(st::RepairState, i::Int, E::Vector{Float16})
    opts = st.owner[i] == 0 ? connector_requirements(st, i, E) : gene_requirements(st, i, E)
    i == 1 && st.owner[i] == 0 && (st.root_plans = length(opts))
    s = st.expr[i]
    for (op, el, er) in opts
        snap = snapshot(st)
        if op != s
            st.expr[i] = op
            refresh!(st)
        end
        ok = repair!(st, i + 1, el)
        ok = ok && repair!(st, st.ends[i+1] + 1, er)
        ok && same_dim(st.dims[i], E) && return true
        restore!(st, snap)
        st.work <= 0 && return false
    end
    return false
end

"""
    repair!(st, i, E) -> Bool

Make the subtree at position `i` take dimension `E` by the repair moves, cheapest first,
editing `st.expr` in place; on failure `st.expr` is left as it was. Each call spends one
unit of `st.work` and fails once the budget is exhausted.
"""
function repair!(st::RepairState, i::Int, E::Vector{Float16})
    st.work -= 1
    st.work < 0 && return false
    is_feasible(E) || return false
    same_dim(st.dims[i], E) && return true
    a = st.ix.arity[Int(st.expr[i])]
    a == 0 && return swap_terminal!(st, i, E) || replace_subtree!(st, i, E)
    swap_operator!(st, i, E) && return true
    a == 1 && drop_unary!(st, i, E) && return true
    # propagation into the operands may spend at most half the remaining budget, so the
    # replacement fallback and the node's siblings still get a turn
    left = st.work
    st.work = left ÷ 2
    ok = a == 1 ? propagate_unary!(st, i, E) : propagate_binary!(st, i, E)
    st.work = left - (left ÷ 2 - st.work)
    return ok || replace_subtree!(st, i, E)
end

"""
    expression_dimension(expr, tokenLib) -> Vector{Float16} or nothing

Dimension of the prefix expression `expr`, by a right-to-left stack walk (no tree is
built); `nothing` if `expr` is not one complete expression over known symbols. The result
is not checked for consistency.
"""
function expression_dimension(expr::AbstractVector{Int8}, tokenLib::TokenLib)
    arity = tokenLib.symbol_arity_mapping[]
    ops = tokenLib.physical_operation_dict[]
    dimdict = tokenLib.physical_dimension_dict[]
    stack = Vector{Vector{Float16}}()
    @inbounds for i in length(expr):-1:1
        s = expr[i]
        a = get(arity, s, Int8(-1))
        if a == 0
            push!(stack, dimdict[s])
        elseif a == 1
            isempty(stack) && return nothing
            push!(stack, ops[s](pop!(stack)))
        elseif a == 2
            length(stack) >= 2 || return nothing
            u1 = pop!(stack)
            u2 = pop!(stack)
            push!(stack, ops[s](u1, u2))
        else
            return nothing
        end
    end
    length(stack) == 1 || return nothing
    return convert(Vector{Float16}, stack[1])
end

"""
    propagate_necessary_changes!(tree, expected_dim, distance_to_change=0; cycles=5,
                                 work_limit=800) -> Bool

Repair `tree` in place so that its dimension becomes `expected_dim`, with the repair
engine of `correct_genes!` and the whole tree treated as one gene that may grow by one
symbol: up to `cycles` attempts of at most `work_limit` node visits each. Returns `true`
on success or if the tree already has the dimension; otherwise returns `false` and leaves
the tree unchanged. A repair that reduces the tree to a single terminal is rejected, since
the root of a `TempComputeTree` is an operator. `distance_to_change` is ignored (kept for
compatibility).
"""
function propagate_necessary_changes!(
    tree::TempComputeTree,
    expected_dim::Vector{Float16},
    distance_to_change::Int=0;
    cycles::Int=5,
    work_limit::Int=800
)
    ix = tree.tokenDto.index
    E = canonical_dim(expected_dim)
    expr = flatten_dependents(tree)
    n = length(expr)
    st = RepairState(ix, copy(expr), ones(Int, n), [n], [n + 1], [n + 1], ix.binary,
        Random.default_rng(), work_limit)
    refresh!(st) || return false
    same_dim(st.dims[1], E) && return true
    for _ in 1:cycles
        st.work = work_limit
        if repair!(st, 1, E) && same_dim(st.dims[1], E) && length(st.expr) > 1
            new_tree = create_compute_tree(st.expr, tree.tokenDto)
            if new_tree isa TempComputeTree
                tree.symbol = new_tree.symbol
                tree.depend_on = new_tree.depend_on
                tree.vector_dimension = new_tree.vector_dimension
                tree.depend_on_total_number = new_tree.depend_on_total_number
                tree.modified = true
                return true
            end
        end
        restore!(st, (expr, ones(Int, n), [n]))
    end
    return false
end

"""
    create_compute_tree(expression, tokenDto, initial_state=false)

Build a `TempComputeTree` from a prefix expression and compute its dimension. Returns the
symbol itself for a one-symbol expression, and `nothing` if the expression is empty or an
operator lacks operands. `initial_state` is ignored.
"""
function create_compute_tree(expression::Vector{Int8}, tokenDto::TokenDto, initial_state::Bool=false)
    expression_list = reverse(expression)
    stack = Union{TempComputeTree,Int8}[]

    if length(expression) == 1
        return expression[1]
    end

    for (index, symbol) in enumerate(expression_list)
        arity = get_arity(tokenDto.tokenLib, symbol)

        if arity == 1
            if isempty(stack)
                return nothing
            end
            op1 = pop!(stack)
            computeTree = TempComputeTree(symbol, [op1], Float16[], tokenDto)
            push!(stack, computeTree)
        elseif arity == 2
            if length(stack) < 2
                return nothing
            end
            op1 = pop!(stack)
            op2 = pop!(stack)
            computeTree = TempComputeTree(symbol, [op1, op2], Float16[], tokenDto)
            push!(stack, computeTree)
        else
            push!(stack, symbol)
        end
    end


    if isempty(stack)
        return nothing
    end

    root = stack[end]
    calculate_vector_dimension!(root)

    root.depend_on_total_number = length(flatten_dependents(root))
    return root
end

"""
    correct_genes!(genes, start_indices, expression, target_dimension, token_dto;
                   cycles=5, gene_len=0, head_len=0, connectors=nothing, work_limit=800,
                   rng=Random.default_rng(), gene_wise=false) -> (distance, success)

Repair a chromosome so that its expression takes `target_dimension`, writing the repaired
connectors and genes back into `genes`.

The target is pushed down from the root with the backward rules, each node trying the
cheapest repair move first. Connectors are only swapped or propagated through. A repaired
gene keeps its active part within `gene_len` symbols and its operators within the head (a
gene already beyond these limits is held to its own extent), so it remains a valid gene
for the genetic operators.

With `gene_wise`, every gene is held to the target instead of the expression as a whole,
as a model scored as a least-squares combination of its genes (linear scaling) needs: its
coefficients are dimensionless only if every gene has the target dimension. Each gene is
then repaired on its own, and if `connectors` include an equal-unit operator (`+`, `-`)
every connector becomes one, so that the karva string as a whole takes the target too.

# Arguments
- `genes::Vector{Int8}`: the chromosome's symbols, connectors first; edited in place
- `start_indices::Vector{Int}`: position of each gene's first symbol in `genes`
- `expression::Vector{Int8}`: the chromosome's karva string (`expression_raw`)
- `target_dimension::Vector{Float16}`: dimension the whole expression must take
- `token_dto::TokenDto`: alphabet, unit rules and library

# Keyword Arguments
- `cycles`: repair attempts, each starting from the original expression; the attempts
  stop early when the root connector finds no way to split the target across the genes
- `gene_len`, `head_len`: gene geometry; when 0, `gene_len` is inferred from
  `start_indices` and `head_len` as `max(1, (gene_len - 1) ÷ 2)`
- `connectors`: operators allowed at connectors (non-binary ones are dropped); every
  binary operator when `nothing`
- `work_limit`: node visits per attempt
- `rng`: random number generator
- `gene_wise`: hold every gene to the target (see above)

# Returns
`(distance, success)`: the Euclidean distance between the target and the dimension of the
expression, as repaired on success and as given otherwise (`Inf16` if that dimension is
inconsistent, the input is invalid or an error occurs); with `gene_wise`, the largest
distance of a gene. `genes` is written only on success.

# Examples
```julia
tb = c.toolbox
distance, ok = correct_genes!(c.genes, tb.gen_start_indices, c.expression_raw,
    Float16[1, 1, 0, 0, 0, 0, 0], dto; cycles=10)
ok && compile_expression!(c; force_compile=true)   # recompute c.expression_raw
```

See also: [`is_dimensionally_homogeneous`](@ref), [`propagate_necessary_changes!`](@ref)
"""
function correct_genes!(genes::Vector{Int8}, start_indices::Vector{Int}, expression::Vector{Int8},
    target_dimension::Vector{Float16}, token_dto::TokenDto; cycles::Int=5,
    gene_len::Int=0, head_len::Int=0, connectors::Union{Nothing,AbstractVector{Int8}}=nothing,
    work_limit::Int=800, rng::AbstractRNG=Random.default_rng(), gene_wise::Bool=false)

    gene_wise && return correct_each_gene!(genes, start_indices, expression, target_dimension,
        token_dto; cycles=cycles, gene_len=gene_len, head_len=head_len, connectors=connectors,
        work_limit=work_limit, rng=rng)
    ix = token_dto.index
    target = canonical_dim(target_dimension)
    g = length(start_indices)
    (g >= 1 && is_feasible(target)) || return Inf16, false
    n_conn = g - 1
    length(expression) > n_conn || return Inf16, false
    gl = gene_len > 0 ? gene_len :
         g >= 2 ? start_indices[2] - start_indices[1] : length(genes) - start_indices[1] + 1
    hl = head_len > 0 ? head_len : max(1, (gl - 1) ÷ 2)

    # connectors, then one complete prefix expression per gene
    owner = zeros(Int, length(expression))
    pos = n_conn + 1
    for k in 1:g
        stop = subtree_end(ix, expression, pos)
        stop == 0 && return Inf16, false
        owner[pos:stop] .= k
        pos = stop + 1
    end
    pos == length(expression) + 1 || return Inf16, false

    glen = [count(==(k), owner) for k in 1:g]
    maxlen = [max(gl, glen[k]) for k in 1:g]
    headlim = Vector{Int}(undef, g)
    for k in 1:g
        r = findfirst(==(k), owner):findlast(==(k), owner)
        headlim[k] = max(hl, last_operator_position(ix, view(expression, r)))
    end
    conn_syms = connectors === nothing ? ix.binary :
                Int8[c for c in connectors if 1 <= Int(c) <= length(ix.arity) && ix.arity[c] == 2]

    st = RepairState(ix, copy(expression), copy(owner), copy(glen), maxlen, headlim,
        conn_syms, rng, work_limit)
    try
        refresh!(st) || return Inf16, false
        base = is_feasible(st.dims[1]) ? calculate_distance(st.dims[1], target) : Inf16
        same_dim(st.dims[1], target) && return base, true

        for _ in 1:cycles
            st.work = work_limit
            if repair!(st, 1, target) && same_dim(st.dims[1], target) &&
               all(k -> gene_fits(st, k), 1:g)
                @inbounds for c in 1:n_conn
                    genes[c] = st.expr[c]
                end
                for k in 1:g
                    s0 = start_indices[k]
                    for (j, p) in enumerate(gene_range(st, k))
                        genes[s0+j-1] = st.expr[p]
                    end
                end
                return calculate_distance(st.dims[1], target), true
            end
            # root planning differs between attempts only in the order of the connector
            # operators, so if it finds no split, a retry helps only if its budget ran out
            st.root_plans == 0 && break
            restore!(st, (expression, owner, glen))
        end
        return base, false
    catch e
        @debug "correct_genes! failed" exception = (e, catch_backtrace())
        return Inf16, false
    end
end

# `correct_genes!` with `gene_wise`: every gene is repaired to the target on its own, as a
# one-gene chromosome, and the connectors become equal-unit operators where `connectors`
# offer one. On failure `genes` is restored.
function correct_each_gene!(genes::Vector{Int8}, start_indices::Vector{Int},
    expression::Vector{Int8}, target_dimension::Vector{Float16}, token_dto::TokenDto;
    cycles::Int, gene_len::Int, head_len::Int,
    connectors::Union{Nothing,AbstractVector{Int8}}, work_limit::Int, rng::AbstractRNG)

    ix = token_dto.index
    target = canonical_dim(target_dimension)
    g = length(start_indices)
    (g >= 1 && is_feasible(target)) || return Inf16, false
    gl = gene_len > 0 ? gene_len :
         g >= 2 ? start_indices[2] - start_indices[1] : length(genes) - start_indices[1] + 1

    # the connectors, then one complete prefix expression per gene
    parts = UnitRange{Int}[]
    pos = g
    for _ in 1:g
        stop = subtree_end(ix, expression, pos)
        stop == 0 && return Inf16, false
        push!(parts, pos:stop)
        pos = stop + 1
    end
    pos == length(expression) + 1 || return Inf16, false
    worst = 0.0f0
    for r in parts
        d = try
            expression_dimension(view(expression, r), token_dto.tokenLib)
        catch
            nothing
        end
        worst = d === nothing || !is_feasible(d) ? Inf32 : max(worst, calculate_distance(d, target))
    end

    backup = copy(genes)
    for (k, r) in enumerate(parts)
        ok = try
            correct_genes!(genes, start_indices[k:k], expression[r], target, token_dto;
                cycles=cycles, gene_len=gl, head_len=head_len, work_limit=work_limit,
                rng=rng)[2]
        catch e
            @debug "correct_genes! failed on a gene" exception = (e, catch_backtrace())
            false
        end
        if !ok
            copyto!(genes, backup)
            return Float16(worst), false
        end
    end
    # an equal-unit connector joins genes of one dimension into that dimension
    allowed = connectors === nothing ? ix.binary : connectors
    eq = Int8[c for c in allowed if 1 <= Int(c) <= length(ix.fwd) && ix.fwd[c] === equal_unit_forward]
    if !isempty(eq)
        for c in 1:g-1
            genes[c] in eq || (genes[c] = rand(rng, eq))
        end
    end
    return Float16(0), true
end

"""
    dimensional_homogeneity_distance(expression, target_dimension, token_dto)

Euclidean distance between the dimension of the prefix expression `expression` (e.g. a
karva string) and the target: the read-only counterpart of `correct_genes!`. `Inf16` if
the expression is malformed, its dimension is inconsistent, or a unit rule throws.
"""
function dimensional_homogeneity_distance(expression::AbstractVector{Int8},
    target_dimension::Vector{Float16}, token_dto::TokenDto)
    try
        dim = expression_dimension(expression, token_dto.tokenLib)
        (isnothing(dim) || !is_feasible(dim)) && return Inf16
        return calculate_distance(dim, target_dimension)
    catch
        return Inf16
    end
end

"""
    is_dimensionally_homogeneous(expression, target_dimension, token_dto)

Whether the expression is homogeneous: its `dimensional_homogeneity_distance` to the
target is below `F16_LOWER_BOUND`.
"""
@inline is_dimensionally_homogeneous(expression::AbstractVector{Int8},
    target_dimension::Vector{Float16}, token_dto::TokenDto) =
    dimensional_homogeneity_distance(expression, target_dimension, token_dto) < F16_LOWER_BOUND

"""
    gene_dimensions(expression, gene_count, token_dto) -> Vector{Vector{Float16}} or nothing

The dimension of each gene of the karva string `expression`: `gene_count - 1` connectors,
then one complete prefix expression per gene. `nothing` if `expression` does not have that
form. A dimension is not checked for consistency.
"""
function gene_dimensions(expression::AbstractVector{Int8}, gene_count::Int, token_dto::TokenDto)
    ix = token_dto.index
    pos = gene_count
    out = Vector{Vector{Float16}}()
    for _ in 1:gene_count
        stop = subtree_end(ix, expression, pos)
        stop == 0 && return nothing
        d = expression_dimension(view(expression, pos:stop), token_dto.tokenLib)
        d === nothing && return nothing
        push!(out, d)
        pos = stop + 1
    end
    return pos == length(expression) + 1 ? out : nothing
end

"""
    is_gene_wise_homogeneous(expression, target_dimension, token_dto, gene_count)

Whether every gene of the karva string `expression` has the target dimension, each within
`F16_LOWER_BOUND`: the condition for a model scored as a least-squares combination of its
genes (linear scaling), whose coefficients are dimensionless only then. The connectors are
not looked at.
"""
function is_gene_wise_homogeneous(expression::AbstractVector{Int8},
    target_dimension::Vector{Float16}, token_dto::TokenDto, gene_count::Int)
    dims = try
        gene_dimensions(expression, gene_count, token_dto)
    catch
        nothing
    end
    dims === nothing && return false
    return all(d -> is_feasible(d) && calculate_distance(d, target_dimension) < F16_LOWER_BOUND, dims)
end

"""
    sample_lib_expression(target_dimension, token_dto; max_len, exact_only=false, top_k=25,
                          head_len=0, rng=Random.default_rng()) -> Vector{Int8} or nothing

A random prefix expression that can be written into a gene as it stands, for seeding: at
most `max_len` symbols and, if `head_len > 0`, its last operator at or before position
`head_len`. With `exact_only` its dimension is `target_dimension`; otherwise its dimension
is drawn uniformly from the `top_k` reachable dimensions nearest to the target (kd-tree
query), up to 8 times until an expression fits. `nothing` if none fits. See
`random_expression`.
"""
function sample_lib_expression(target_dimension::Vector{Float16}, token_dto::TokenDto;
    max_len::Int, exact_only::Bool=false, top_k::Int=25, head_len::Int=0,
    rng::AbstractRNG=Random.default_rng())
    ix = token_dto.index
    t = canonical_dim(target_dimension)
    exact_only && return random_expression(ix, t; max_len=max_len, head_len=head_len, rng=rng)
    ids = nearest_reach(ix, t, top_k)
    for _ in 1:(isempty(ids) ? 0 : 8)
        e = random_expression(ix, ix.reach_dims[rand(rng, ids)]; max_len=max_len,
            head_len=head_len, rng=rng)
        e === nothing || return e
    end
    return nothing
end

function get_feature_dims_json(json_data::Dict{String,Any}, features::Vector{String}, case_name::String; dims_identifier::String="dims")
    target_entry = json_data[case_name]
    ret_val = Dict{String,Vector{Float16}}()
    for (index, entry) in enumerate(target_entry[dims_identifier])
        ret_val[features[index]] = convert.(Float16, entry)
    end
    return ret_val
end

function is_close_to_target(target_dim::Vector{Float16}, value_dim::Vector{Float16}, tolerance::Float16=Float16(0.5))
    weighting = [t != 0 ? Float16(0.25) : Float16(2.0) for t in target_dim]
    value = sum(abs.(target_dim .- value_dim) .* weighting)
    return value < tolerance
end


function get_target_dim_json(json_data::Dict{String,Any}, case_name::String; dims_identifier::String="targetdims")
    target_entry = json_data[case_name]
    return convert.(Float16, target_entry[dims_identifier])
end



function retrieve_coeffs_based_on_similarity(target_dim::Vector{Float16},
    physical_constants::Dict{String,Tuple{T,Vector{Float16}}}; tolerance::Float16=Float16(100.0)) where {T<:AbstractFloat}

    ret_val = Dict{String,Vector{Float16}}()
    for (_, (val, dim)) in physical_constants
        if is_close_to_target(target_dim, dim, tolerance)
            ret_val[string(val)] = dim
        end
    end
    return ret_val
end

end
