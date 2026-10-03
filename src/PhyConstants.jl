"""
    PhysicalConstants

Physical constants as `(value, dimension)` pairs: the value in SI units, the dimension a
`Float16` vector of SI exponents in the order [kg, m, s, K, mol, A, cd].
`physical_constants` is a small set, `physical_constants_all` a larger one built from
derived units; the `get_constant*` functions look up `physical_constants` only.
"""
module PhysicalConstants

export physical_constants, physical_constants_all
export get_constant, get_constant_value, get_constant_dims

# SI base units as exponent vectors, then derived units composed from them
const kg = Float16[1, 0, 0, 0, 0, 0, 0]
const m = Float16[0, 1, 0, 0, 0, 0, 0]
const s = Float16[0, 0, 1, 0, 0, 0, 0]
const K = Float16[0, 0, 0, 1, 0, 0, 0]
const mol = Float16[0, 0, 0, 0, 1, 0, 0]
const A = Float16[0, 0, 0, 0, 0, 1, 0]
const cd = Float16[0, 0, 0, 0, 0, 0, 1]

const J = kg + 2m - 2s
const Hz = -s
const W = kg + 2m - 3s
const C = A + s
const V = kg + 2m - 3s - A
const F = -kg - 2m + 4s + 2A
const T = kg - 2s - A
const N = kg + m - 2s
const Pa = kg - m - 2s
const S = -kg - 2m + 3s + 2A
const H = kg + 2m - 2s - 2A
const Bq = -s
const Gy = 2m - 2s
const Sv = 2m - 2s
const kat = mol - s

const physical_constants = Dict{String, Tuple{Float64, Vector{Float16}}}(
    "G" => (6.674e-11, Float16[-1, 3, -2, 0, 0, 0, 0]),      # Gravitational constant
    "c" => (2.998e8, Float16[0, 1, -1, 0, 0, 0, 0]),         # Speed of light
    "epsilon_0" => (8.854e-12, Float16[-1, -3, 4, 0, 0, 2, 0]), # Vacuum permittivity
    "h" => (6.626e-34, Float16[1, 2, -1, 0, 0, 0, 0]),       # Planck constant
    "k_B" => (1.381e-23, Float16[1, 2, -2, -1, 0, 0, 0]),    # Boltzmann constant
    "mu_B" => (9.2740100783e-24, Float16[0, 2, 0, 0, 0, 1, 0]), # Bohr magneton
    "h_bar" => (1.055e-34, Float16[1, 2, -1, 0, 0, 0, 0]),   # Reduced Planck constant
    "m_e" => (9.109e-31, Float16[1, 0, 0, 0, 0, 0, 0]),      # Electron mass
    "g" => (9.807, Float16[0, 1, -2, 0, 0, 0, 0]),           # Gravitational acceleration
    "e" => (1.602e-19, Float16[0, 0, 1, 0, 0, 1, 0]),        # Elementary charge
    "m_p" => (1.67e-27, Float16[1, 0, 0, 0, 0, 0, 0]),       # Proton mass
    "m_n" => (1.67e-27, Float16[1, 0, 0, 0, 0, 0, 0]),       # Neutron mass
    "N_A" => (6.022e23, Float16[0, 0, 0, 0, -1, 0, 0]),      # Avogadro constant
    "R" => (8.31, Float16[1, 2, -2, -1, -1, 0, 0]),          # Gas constant
    "sigma" => (5.67e-8, Float16[1, 0, -3, -4, 0, 0, 0]),    # Stefan-Boltzmann constant
    "Z_0" => (376.73, Float16[1, 2, -3, 0, 0, -2, 0])        # Impedance of free space
)

const physical_constants_all = Dict{String, Tuple{Float64, Vector{Float16}}}(
    "c" => (299792458.0, m - s),
    "c_1" => (3.74e-16, W + 2m),
    "c_2" => (1.43e-2, m + K),
    "e" => (1.60e-19, C),
    "E_h" => (4.35e-18, J),
    "G" => (6.67e-11, 3m - kg - 2s),
    "G_0" => (7.74e-5, S),
    "g_e" => (-2.00, zeros(Float16, 7)),
    "g_p" => (5.58, zeros(Float16, 7)),
    "g_mu" => (-2.00, zeros(Float16, 7)),
    "h" => (6.62e-34, J + s),
    "quantum_of_circulation" => (3.63e-4, 2m - s),
    "h_bar" => (1.05e-34, J + s),
    "k_e" => (8.99e9, N + 2m - 2C),
    "k_B" => (1.38e-23, J - K),
    "m_e" => (9.10e-31, kg),
    "m_n" => (1.67e-27, kg),
    "m_p" => (1.67e-27, kg),
    "m_t" => (3.07e-25, kg),
    "m_u" => (1.66e-27, kg),
    "M_u" => (1.00e-3, kg - mol),
    "m_mu" => (1.88e-28, kg),
    "m_tau" => (3.16e-27, kg),
    "N_A" => (6.02e23, -mol),
    "N_A_h" => (3.99e-10, J + s - mol),
    "R_inf" => (10973731.56, -m),
    "r_e" => (2.81e-15, m),
    "R_K" => (25812.8, kg + 2m - 3s - 2A),
    "R" => (8.31, J - mol - K),
    "Ry" => (2.17e-18, J),
    "V_m_Si" => (1.20e-5, 3m - mol),
    "Z_0" => (376.73, kg + 2m - 3s - 2A),
)

"""
    get_constant(name::String)

The `(value, dimension)` pair stored under `name` in `physical_constants`; any other
name is an error.
"""
function get_constant(name::String)
    if haskey(physical_constants, name)
        return physical_constants[name]
    else
        error("Constant $name not found")
    end
end

"""
    get_constant_value(name::String)

The SI value of `name`, as `get_constant` finds it.
"""
function get_constant_value(name::String)
    return get_constant(name)[1]
end

"""
    get_constant_dims(name::String)

The dimension vector of `name`, as `get_constant` finds it.
"""
function get_constant_dims(name::String)
    return get_constant(name)[2]
end

end