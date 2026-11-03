# Ground-state and first-excited-state F(ϕ) and supercurrent (converged harmonics) for one (Δn, Vg) case
include("setup.jl")

params = merge(base_params, (; ϕ = 0, n = 1, Δ = 0.2, Vg = 2.1, mult=2))
(; hN, ϵ1) = build(; params...)

case = (; Δn = 1, Vg = 1.5)                     # Vg: gate voltage [meV]
cpr_settings = (; N0 = 8, N_max = 128, tol_alias = 1e-3, tol_tail = 1e-3)
FI_plot = plot_free_energy_current(hN, case; n0 = 0, cpr_settings,
    savepath = joinpath(@__DIR__, "figures", "free_energy_current.png"), params...)
