# Shapiro maps (RCSJ) of the even (ground state F₀) and odd (F₀ + εt, one quasiparticle in the tracked level)
# fermion parity of one case, both CPRs from one dense spectrum (as in fig_free_energy_current.jl).
# Runs are saved in data/shapiro_runs/ and can be replotted (see the end of fig_shapiro.jl)
include("setup.jl")

params = merge(base_params, (; ϕ = 0, n = 1, Δ = 0.2, Vg = 2.1, mult = 2))
(; hN, ϵ1) = build(; params...)

case = (; Δn = 1, Vg = 1.5)                     # Vg: gate voltage [meV]
φsel = π/2                                      # the tracked level is the lowest positive one at φsel
# N_max points are diagonalized at once (no adaptive refinement), harmonics converged on the nested subgrids
cpr_settings = (; N0 = 8, N_max = 128, tol_alias = 1e-3, tol_tail = 1e-3, Ic_tol = 1e-5)

setup = 1                                       # Shapiro setup, see shapiro_setups in setup.jl: 1 RSJ baseline, 2 capacitive, 3 physical device
(; scaling, βc, iac_cut, drive, bias) = shapiro_setups[setup]

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

shapiro_plot = plot_shapiro_grid(shapiro_results, rows, Δns; iac_cut=2,
    units = scaling == :physical ? :physical : :normalized,
    savepath = joinpath(@__DIR__, "figures", "shapiro_parity_$(run_id).png"))
print("done")
