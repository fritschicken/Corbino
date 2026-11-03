# Shapiro maps (RCSJ) from the computed CPR; runs are saved in data/shapiro_runs/ and can be replotted
include("setup.jl")

params = merge(base_params,(;ϕ = 0, n = 1, Δ = 0.2 ,Vg = 2.5))
(; hN, ϵ1) = build(; params...)


# ---------------------------------------------------------------------
# Shapiro maps (RCSJ)
# ---------------------------------------------------------------------
cpr_settings = (; N0 = 32, N_max = 512, tol_alias = 1e-3, tol_tail = 1e-3, Ic_tol = 1e-5)
Vg_on = 1.5                                     # gate voltage of the Shapiro cases [meV]

setup = 1                                       # Shapiro setup, see shapiro_setups in setup.jl: 1 RSJ baseline, 2 capacitive, 3 physical device
(; scaling, βc, iac_cut, drive, bias) = shapiro_setups[setup]

Δn_values = (1)
µN_values = (0.1,2)
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