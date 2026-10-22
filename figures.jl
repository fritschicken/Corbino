using Quantica, GLMakie, StaticArrays, LinearMaps
using CairoMakie
using ArnoldiMethod
using Arpack
using Printf
using JLD2
using Dates
using ColorSchemes

includet("../arXiv_2506.15502/code.jl")
includet("./project.jl")

base_params = (;
    ħ2ome = 76.2,
    N = 250,
    m = 0.065,
    µ = 2,
    R = 500,
    seed = 1)


solver = ES.LinearAlgebra()


params = merge(base_params,(; R = 500, ϕ = 0, n = 1,
    N = 250, Δ = 0.2 ,Vg = 1.5,μ=0.15))
(; hN, ϵ1) = build(; params...)


#results = detailed_fourier_analysis(hN;npoints = 51,max_harmonics = 14,solver = solver,params...)

#heatmaps = sweep_vg_mu_harmonic_amplitudes(hN, (; Δn = 1); Vg_values = range(0, 3, 7), µ_values = range(1, 5, 7),
#    npoints = 33, max_harmonics = 12, solver, savepath = joinpath(@__DIR__, "figures", "harmonics_Vg_mu.png"), params...)

#flux_params = (; nmax = 4.0, npoints = 250, area_ratio = 0.4)


#spectrum_plot = plot_flux_spectrum(hN; flux_params..., nev = 40,
#    savepath = joinpath(@__DIR__, "figures", "spectrum.png"), params...)

#cases = [(; Δn = 0, Vg = Vg_on),
#         (; Δn = 1, Vg = 0.0),
#         (; Δn = 1, Vg = Vg_on)]
#FI_plot = plot_free_energy_current(hN, cases; n0 = 0, npoints = 51, solver = ES.LinearAlgebra(),
#    savepath = joinpath(@__DIR__, "figures", "free_energy_current.png"), params...)

# ---------------------------------------------------------------------
# Shapiro maps (RCSJ)
# ---------------------------------------------------------------------
cpr_settings = (; N0 = 32, N_max = 512, tol_alias = 1e-3, tol_tail = 1e-2, Ic_tol = 1e-5)
drive = (; Ω = 0.2, idcs = range(-2, 2, 250), iacs = range(0, 2, 250))
bias = (; nper_skip_min = 20, nper_avg = 30)
βc = 0
# Physical device: βc, Ω it would have are only printed per case, the maps use the fixed values above
device = (; R = 1e3, C = 1e-15, f = 100e6, T = 0.02)


Δn_values = (1,0)
µN_values = (0.1,2)
(; N0, N_max, tol_alias, tol_tail, Ic_tol) = cpr_settings
shapiro_results = [begin
        case = (; Δn, Vg = Vg_on, µN)
        println(@sprintf("RCSJ Shapiro map for Δn = %g, µN = %g meV ...", Δn, µN))
        p = (; params..., n0 = 0, n = Δn, Vg = Vg_on, µN)
        s, c, F = converged_cpr(hN, (p, solver), N0, N_max, tol_alias, tol_tail)
        s, c, Ic = normalize_cpr(s, c; Ic_tol)
        print_rcsj_units(@sprintf("  device (Δn = %g, µN = %g)", Δn, µN), Ic; device...)
        (; case, F, s, c, Ic, noise = Ic < Ic_tol, shapiro_map(s, c; drive..., βc, bias...)...)
    end for µN in µN_values, Δn in Δn_values]
run_id = save_shapiro_run(shapiro_results, µN_values, Δn_values; label = "",
    meta = (; params, Vg_on, drive, bias, βc, cpr_settings, device))

shapiro_plot = plot_shapiro_grid(shapiro_results, µN_values, Δn_values; iac_cut = 0.2,
    savepath = joinpath(@__DIR__, "figures", "shapiro_grid_$(run_id).png"))

# Reloading earlier runs (no recomputation):
#   list_shapiro_runs()                                   # table of all saved runs (data/shapiro_runs/index.csv)
#   run = load_shapiro_run("20261021-153012")             # results, µs, Δns, meta (all settings of that run)
#   plot_shapiro_grid(run.results, run.µs, run.Δns; iac_cut = 0.2)
#   both = combine_shapiro_runs(load_shapiro_runs("id1", "id2")...)   # same µN → Δn columns side by side
#   plot_shapiro_grid(both.results, both.µs, both.Δns)
print("done")