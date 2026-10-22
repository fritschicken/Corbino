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


params = merge(base_params,(;ϕ = 0, n = 1, Δ = 0.2 ,Vg = 1.5))
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
cpr_settings = (; N0 = 32, N_max = 512, tol_alias = 1e-3, tol_tail = 1e-3, Ic_tol = 1e-5)
device = (; R = 1e3, C = 1e-15, f = 20e9)       # R [Ω], C [F], drive f [Hz]; T = 0 throughout
Vg_on = 0.0                                     # gate voltage of the Shapiro cases [meV]

# 1: RSJ quartet baseline        (fixed normalized βc, Ω)
# 2: capacitive control          (fixed normalized βc, Ω; Kautz 1996 Fig. 13a parameters)
# 3: physically scaled device    (βc, Ω per case from its own I_c and `device`; currents in µA)
setup = 1
shapiro_setups = Dict(
    1 => (; scaling = :fixed, βc = 0.0, iac_cut = 1.0,
        drive = (; Ω = 0.5, idcs = range(-2, 2, 251), iacs = range(0, 3, 201)),   # 251 → i_dc = 0 on the grid
        bias = (; nper_skip_min = 20, nper_avg = 60)),
    2 => (; scaling = :fixed, βc = 2.78, iac_cut = 17.0,
        drive = (; Ω = 1.8, idcs = range(0, 4, 201), iacs = range(0, 20, 101)),
        bias = (; nper_skip_min = 20, nper_avg = 60)),
    3 => (; scaling = :physical, βc = 0.0, iac_cut = 1.0,                         # βc from I_c; iac_cut in µA
        drive = (; Idcs_uA = range(-3, 3, 301), Iacs_uA = range(0, 5, 201)),      # µA, odd length → I_dc = 0 on the grid
        bias = (; nper_skip_min = 20, nper_avg = 60)))
(; scaling, βc, iac_cut, drive, bias) = shapiro_setups[setup]

Δn_values = (1,0)
µN_values = (0.2,2)
shapiro_results = [begin
        case = (; Δn, Vg = Vg_on, µN)
        label = @sprintf("Δn = %g, µN = %g meV", Δn, µN)
        println("RCSJ Shapiro map for ", label, " ...")
        n0 = 0
        p = (; params..., n0, n = n0 + Δn, Vg = Vg_on, µN)
        (; case, shapiro_case(hN, p, solver; cpr_settings, drive, bias, βc, scaling, device, label)...)
    end for µN in µN_values, Δn in Δn_values]
run_id = save_shapiro_run(shapiro_results, µN_values, Δn_values; label = "setup$(setup)",
    meta = (; params, Vg_on, setup, scaling, drive, bias, βc, cpr_settings, device))

shapiro_plot = plot_shapiro_grid(shapiro_results, µN_values, Δn_values; iac_cut,
    units = scaling == :physical ? :physical : :normalized,
    savepath = joinpath(@__DIR__, "figures", "shapiro_grid_$(run_id).png"))

# Reloading earlier runs (no recomputation):
#   list_shapiro_runs()                                   # table of all saved runs (data/shapiro_runs/index.csv)
#   run = load_shapiro_run("20261021-153012")             # results, µs, Δns, meta (all settings of that run)
#   plot_shapiro_grid(run.results, run.µs, run.Δns; iac_cut = 1.0)   # units = :physical for setup 3 runs
#   both = combine_shapiro_runs(load_shapiro_runs("id1", "id2")...)   # same µN → Δn columns side by side
#   plot_shapiro_grid(both.results, both.µs, both.Δns)
print("done")