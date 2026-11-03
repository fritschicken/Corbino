# Odd (2e, 6e, …) and even (4e, 8e, …) harmonic weight of the CPR over the (Vg, µ) plane
include("setup.jl")

params = merge(base_params, (; ϕ = 0, n = 1, Δ = 0.2, Vg = 0))
(; hN, ϵ1) = build(; params...)

heatmaps = sweep_vg_mu_harmonic_amplitudes(hN, (; Δn = 1); Vg_values = range(0, 3, 7), µ_values = range(1, 5, 7),
    npoints = 33, max_harmonics = 12, solver, savepath = joinpath(@__DIR__, "figures", "harmonics_Vg_mu.png"), params...)
