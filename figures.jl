# Figures of the article. Running the whole file makes all of them in figures/:
#   julia --project -t auto figures.jl          (or include("figures.jl") in the REPL)
# Each figure is also a function that can be called on its own after loading, e.g. fig_bands(CASES[2]) or fig_shapiro_parity().
using Revise                                    # setup.jl loads the project files with includet
include("setup.jl")

# The two systems, one per gate profile V_g cos(mult θ) (θ = position angle around the ring). Same parameters as the saved
# Shapiro runs combined in fig_shapiro_parity (run_id), with the line cut I_ac/I_c of each in `iac_cut`
CASES = [
    (; name = "cos1", Δn = 2,
       params = merge(base_params, (; R = 700, N = 500, µ = 1,n=2 ,Δ = 0.2, Vg = 1.1, mult = 1, λg = Inf)),
       run_id = "20261106-152942_test_setup1_lowOmega", iac_cut = 1.0),     # Ω = 0.05
    (; name = "cos2", Δn = 1,
       params = merge(base_params, (; R = 700, N = 500, µ = 0.3,n=1 ,Δ = 0.2, Vg = 2.0, mult = 2, λg = Inf)),
       run_id = "20261106-150356_test_setup1_lowOmega", iac_cut = 2.0),     # Ω = 0.5
]

# Hamiltonian parameters of a case at ϕ = 0, inner winding n0 = 0 and outer winding n = Δn
case_params(c) = merge(c.params, (; ϕ = 0, n0 = 0, n = c.Δn))

# BdG spectrum against flux: inner ring winding n0 = 0 throughout (area_ratio = 0), outer winding n from 0 to 2
function fig_spectrum()
    params = merge(base_params, CASES[1].params)
    (; hN2) = build(; params...)
    return plot_flux_spectrum(hN2; nmax = 2.0, npoints = 250, area_ratio = 0.0, nev = 40,
        savepath = figpath("spectrum.png"), params...)
end

# F(ϕ) and supercurrent of both cases side by side, even (F₀) and odd (F₀ + εt) fermion parity
function fig_free_energy_current(; cpr_settings = (; N0 = 16, N_max = 256, tol_cpr = 1e-3, nev = 40))
    datas = map(CASES) do c
        (; hN2) = build(; case_params(c)...)
        free_energy_current_data(hN2, (; Δn = c.Δn); n0 = 0, cpr_settings, c.params...)
    end
    titles = [Makie.LaTeXString(gate_label(c.params.mult)) for c in CASES]
    return plot_free_energy_current_cases(datas, titles; panel_labels = true, savepath = figpath("free_energy_current.png"))
end

# Shapiro maps (RCSJ) of both cases side by side (columns), even and odd parity (rows), from the saved runs
# (no recomputation). The runs have different Ω, so each column has its own V/(hf/2e) range
function fig_shapiro_parity(; idc_lims = (-1, 1))
    both = combine_shapiro_runs(load_shapiro_runs((c.run_id for c in CASES)...)...)
    return plot_shapiro_grid(both.results, both.µs, [gate_label(c.params.mult) for c in CASES];
        iac_cut = [c.iac_cut for c in CASES], idc_lims, shared_V = false, panel_labels = true, savepath = figpath("shapiro_parity.png"))
end

# Documentation: bands against ϕ over one period of the odd branch (4π for cos(θ), 2π for cos(2θ)), with the subbands
# holding the lowest positive level at φ0 (red) and at 2π/mult + φ0 (blue), and their |ψ|² along the ring
function fig_bands(c; npoints = 600, nev = 50, φ0 = 0.1)
    p = case_params(c)
    (; hN2) = build(; p...)
    period = 2π * odd_period_factor(p)
    # Half-step shift (npoints a multiple of 4) so no φ is a multiple of π: there the two halves' levels of hN2 are
    # exactly degenerate (θ → -θ symmetry) and bands would merge them into one subband
    phis = range(0, period, length = npoints + 1)[1:npoints] .+ period / (2npoints)
    b = bands(hN2, phis; mapping = φ -> ftuple(; p..., ϕ = φ), solver = ES.ShiftInvert(ES.ArnoldiMethod(; nev), 0))
    js = [lowest_positive_subband(b, φ0), lowest_positive_subband(b, 2π / p.mult + φ0)]
    return plot_bands_density(b, js, [:red, :blue], p.mult; panel_labels = true, savepath = figpath("bands_$(c.name).png"))
end

# Any saved Shapiro run (list_shapiro_runs() shows them all); units = :physical for setup 3 runs
function replot_shapiro(run_id; iac_cut = 1.0, kw...)
    run = load_shapiro_run(run_id)
    return plot_shapiro_grid(run.results, run.µs, run.Δns; iac_cut, kw...)
end

function make_all_figures()
    fig_spectrum()
    fig_free_energy_current()
    fig_shapiro_parity()
    foreach(fig_bands, CASES)
    println("all figures saved in ", figpath(""))
end
make_all_figures()
