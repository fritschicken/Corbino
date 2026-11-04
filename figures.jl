# All figures, one function per figure, each with its own parameters. Usage:
#   include("figures.jl")
#   fig_spectrum()          # or fig_fourier(), fig_free_energy_current(), fig_harmonics_map(), fig_shapiro(), fig_shapiro_parity()
#   replot_shapiro("20261103-170959_parity_setup1")    # replot a saved Shapiro run, no recomputation
include("setup.jl")

figpath(name) = joinpath(@__DIR__, "figures", name)

# BdG spectrum against flux, with the fluxoid windings of the inner and outer ring
function fig_spectrum()
    params = merge(base_params, (; ϕ = 0, n = 1, Δ = 0.2, Vg = 1.5))
    (; hN) = build(; params...)

    flux_params = (; nmax = 1.0, npoints = 250, area_ratio = 0.0)
    return plot_flux_spectrum(hN; flux_params..., nev = 40, savepath = figpath("spectrum.png"), params...)
end

# Free energy, current and harmonic amplitudes against ϕ, even and odd parity
function fig_fourier()
    params = merge(base_params, (; µ = 0.1, ϕ = 0, n = 1, Δ = 0.2, Vg = 1.5, mult = 2))
    (; hN) = build(; params...)

    return detailed_fourier_analysis(hN; npoints = 32, max_harmonics = 8, φsel = π/2, params...)
end

# Ground-state and first-excited-state F(ϕ) and supercurrent (converged harmonics) for one (Δn, Vg) case
function fig_free_energy_current()
    params = merge(base_params, (; ϕ = 0, n = 1, Δ = 0.2, Vg = 2.1, mult = 2))
    (; hN) = build(; params...)

    case = (; Δn = 1, Vg = 1.5)                     # Vg: gate voltage [meV]
    cpr_settings = (; N0 = 8, N_max = 128, tol_alias = 1e-3, tol_tail = 1e-3)
    return plot_free_energy_current(hN, case; n0 = 0, cpr_settings,
        savepath = figpath("free_energy_current.png"), params...)
end

# Odd (2e, 6e, …) and even (4e, 8e, …) harmonic weight of the CPR over the (Vg, µ) plane
function fig_harmonics_map()
    params = merge(base_params, (; ϕ = 0, n = 1, Δ = 0.2, Vg = 0))
    (; hN) = build(; params...)

    return sweep_vg_mu_harmonic_amplitudes(hN, (; Δn = 1); Vg_values = range(0, 3, 7), µ_values = range(1, 5, 7),
        npoints = 33, max_harmonics = 12, solver, savepath = figpath("harmonics_Vg_mu.png"), params...)
end

# Shapiro maps (RCSJ) from the computed CPR, a grid of µN rows × Δn columns.
# Runs are saved in data/shapiro_runs/ and can be replotted with replot_shapiro
function fig_shapiro(; setup = 1)               # Shapiro setup, see shapiro_setups in setup.jl: 1 RSJ baseline, 2 capacitive, 3 physical device
    params = merge(base_params, (; ϕ = 0, n = 1, Δ = 0.2, Vg = 2.5))
    (; hN) = build(; params...)

    cpr_settings = (; N0 = 32, N_max = 512, tol_alias = 1e-3, tol_tail = 1e-3, Ic_tol = 1e-5)
    Vg_on = 1.5                                 # gate voltage of the Shapiro cases [meV]
    (; scaling, βc, iac_cut, drive, bias) = shapiro_setups[setup]

    Δn_values = (1)
    µN_values = (0.1, 2)
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

    return plot_shapiro_grid(shapiro_results, µN_values, Δn_values; iac_cut,
        units = scaling == :physical ? :physical : :normalized,
        savepath = figpath("shapiro_grid_$(run_id).png"))
end

# Shapiro maps (RCSJ) of the even (ground state F₀) and odd (F₀ + εt, one quasiparticle in the tracked level)
# fermion parity of one case, both CPRs from one dense spectrum (as in fig_free_energy_current)
function fig_shapiro_parity(; setup = 1)
    params = merge(base_params, (; ϕ = 0, n = 1, Δ = 0.2, Vg = 2.1, mult = 2))
    (; hN) = build(; params...)

    case = (; Δn = 1, Vg = 1.5)                 # Vg: gate voltage [meV]
    φsel = π/2                                  # the tracked level is the lowest positive one at φsel
    # N_max points are diagonalized at once (no adaptive refinement), harmonics converged on the nested subgrids
    cpr_settings = (; N0 = 8, N_max = 128, tol_alias = 1e-3, tol_tail = 1e-3, Ic_tol = 1e-5)
    (; scaling, βc, drive, bias) = shapiro_setups[setup]

    n0 = 0
    p = (; params..., Base.structdiff(case, NamedTuple{(:Δn,)})..., n0, n = n0 + case.Δn)
    label = @sprintf("Δn = %g, Vg = %g meV", case.Δn, case.Vg)
    println("RCSJ Shapiro maps, even and odd parity, for ", label, " ...")
    shapiro_results = [(; case, r...) for r in
        shapiro_parity_cases(hN, p; cpr_settings, φsel, drive, bias, βc, scaling, device, label)]

    # grid rows = parity (in place of µN), one column named by the gate profile V_g cos(mult θ), θ = angle around the ring
    rows = ["Even parity", "Odd parity"]
    mult = get(params, :mult, 1)
    Δns = [mult == 1 ? "\$V_g \\propto \\cos\\theta\$" : "\$V_g \\propto \\cos($(mult)\\theta)\$"]
    run_id = save_shapiro_run(shapiro_results, rows, Δns; label = "parity_setup$(setup)",
        meta = (; params, case, φsel, setup, scaling, drive, bias, βc, cpr_settings, device))

    return plot_shapiro_grid(shapiro_results, rows, Δns; iac_cut = 2,
        units = scaling == :physical ? :physical : :normalized,
        savepath = figpath("shapiro_parity_$(run_id).png"))
end

# Replot a saved Shapiro run (no recomputation); list_shapiro_runs() shows all saved runs (data/shapiro_runs/index.csv).
# Use units = :physical for setup 3 runs. Two runs side by side (same µN → Δn columns):
#   both = combine_shapiro_runs(load_shapiro_runs("id1", "id2")...); plot_shapiro_grid(both.results, both.µs, both.Δns)
function replot_shapiro(run_id; iac_cut = 1.0, kw...)
    run = load_shapiro_run(run_id)              # results, µs, Δns, meta (all settings of that run)
    return plot_shapiro_grid(run.results, run.µs, run.Δns; iac_cut, kw...)
end
