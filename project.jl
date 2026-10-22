using Quantica, GLMakie, StaticArrays, LinearMaps
using CairoMakie
using ArnoldiMethod
using Arpack
using Printf
using JLD2
using Dates
using ColorSchemes
using ProgressMeter

includet("../arXiv_2506.15502/code.jl")
includet("./rcsj.jl")

function data_figure2(nev = 8, npoints = 30; params...)
    (; hN, ϵ1) = build(; params...);
    _, psi = lowest_eigenpair(hN(; params...), nev)
    ns = [range(0, 0.49999, npoints+1); range(0.5000001, 1.49999, 2npoints + 1)]
    solver = ES.ShiftInvert(ES.Arpack(nev = 4+nev), 0)
    b = bands(hN, ns; mapping = n -> ftuple(; params..., n), solver)
    Δ = NamedTuple(params).Δ
    params´ = (; params..., Δ0 = Δ, Δ1 = 0.9*Δ)
    nt = NamedTuple(params);
    return (; hN, psi, ns, b, ϵ1, params = nt)
end

function free_energy(h_inst, solver)
    sp = spectrum(h_inst; solver = solver)
    # E_GS = 0.5 * sum of negative BdG eigenvalues
    return 0.5 * sum(filter(<(0), real(energies(sp))))
end

# Free energy and the lowest |ε| of the same spectrum (smallest Andreev level at this φ)
function free_energy_gap(h_inst, solver)
    ε = real(energies(spectrum(h_inst; solver = solver)))
    return 0.5 * sum(filter(<(0), ε)), minimum(abs, ε)
end

function fourier(F_vals, phases, max_harmonics, N)
    # ---------------------------------------------------------
    # Discrete Fourier Decomposition of F(ϕ)
    # F(ϕ) = A0 + ∑ [ A_k cos(kϕ) + B_k sin(kϕ) ]
    # ---------------------------------------------------------
    A0 = sum(F_vals) / N
    A_k = zeros(Float64, max_harmonics)
    B_k = zeros(Float64, max_harmonics)

    for k in 1:max_harmonics
        A_k[k] = (2.0 / N) * sum(F_vals .* cos.(k .* phases))
        B_k[k] = (2.0 / N) * sum(F_vals .* sin.(k .* phases))
    end

    # ---------------------------------------------------------
    # Supercurrent Harmonics: I(ϕ) = 2 * ∂F/∂ϕ (ħ = e = 1)
    # ∂F/∂ϕ = ∑ k * [ -A_k sin(kϕ) + B_k cos(kϕ) ]
    # I(ϕ) = ∑ [ I_k^(s) sin(kϕ) + I_k^(c) cos(kϕ) ]
    # ---------------------------------------------------------
    I_sin = -2.0 .* (1:max_harmonics) .* A_k
    I_cos =  2.0 .* (1:max_harmonics) .* B_k
    I_amp = sqrt.(I_sin.^2 .+ I_cos.^2)

    # Current evaluated from reconstructed Fourier components
    I_recon = zeros(Float64, N)
    for k in 1:max_harmonics
        I_recon .+= I_sin[k] .* sin.(k .* phases) .+ I_cos[k] .* cos.(k .* phases)
    end

    # Numerical central difference derivative of F (for cross-validation)
    Δϕ = 2π / N
    I_numeric = zeros(Float64, N)
    for i in 1:N
        ip = mod1(i + 1, N)
        im = mod1(i - 1, N)
        I_numeric[i] = 2.0 * (F_vals[ip] - F_vals[im]) / (2.0 * Δϕ)
    end
    return I_numeric, I_recon, A0, A_k, B_k, I_sin, I_cos, I_amp
end

function detailed_fourier_analysis(hN;
        npoints = 128, max_harmonics = 12, solver,
        savepath = joinpath(@__DIR__, "figures/fourier_"), params...)

    phases = collect(range(0, 2π, length = npoints + 1)[1:end-1])
    N = length(phases)
    parameter_values = (; params...)

    println("Calculating free energy across $(N) phase points...")
    println("System parameters:")
    show(stdout, MIME"text/plain"(), parameter_values)
    println("\nFourier settings: npoints = $npoints, max_harmonics = $max_harmonics")
    F_vals = zeros(Float64, N)
    for (i, ϕ_val) in enumerate(phases)
        h_at_ϕ = hN(; params..., ϕ = ϕ_val)
        F_vals[i] = free_energy(h_at_ϕ, solver)
    end

    I_numeric, I_recon, A0, A_k, B_k, I_sin, I_cos, I_amp = fourier(F_vals, phases, max_harmonics, N)

    # ---------------------------------------------------------
    # Visualization Dashboard
    # ---------------------------------------------------------
    fig = Figure(size = (1000, 600))

    ax1 = Axis(fig[1, 1],
        xlabel = "Phase difference, ϕ",
        ylabel = "Ground-state free energy [meV]",
        title = "Ground-State Free Energy and Current vs Phase",
        xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]),
        ytickformat = values -> [@sprintf("%.3g", value) for value in values])
    free_energy_line = lines!(ax1, phases, F_vals;
        color = :royalblue, linewidth = 2)

    ax2 = Axis(fig[1, 1],
        ylabel = "Josephson current [meV]",
        yaxisposition = :right,
        xaxisposition = :top,
        backgroundcolor = :transparent,
        ytickformat = values -> [@sprintf("%.3g", value) for value in values])
    hidexdecorations!(ax2)
    hidespines!(ax2, :l)
    linkxaxes!(ax1, ax2)

    numeric_line = lines!(ax2, phases, I_numeric;
        color = :green, linestyle= Linestyle([1,3,5,7,9,11]), linewidth = 2, alpha = 1)
    two_e_line = lines!(ax2, phases, I_sin[1] .* sin.(phases) .+ I_cos[1] .* cos.(phases);
        linestyle = (:dash, :loose), alpha=0.7, linewidth= 2)
    four_e_line = lines!(ax2, phases, I_sin[2] .* sin.(2 .* phases) .+ I_cos[2] .* cos.(2 .* phases);
        color = :red, linestyle = Linestyle([0, 2, 4, 6, 8, 10, 20]),linewidth= 2, alpha=1)

    legend_plots = [free_energy_line, numeric_line, two_e_line, four_e_line]
    legend_labels = [
        "Ground-state free energy",
        "Josephson Current",
        "2e Doublet component",
        "4e Quartet component",
    ]
    Legend(fig[2, 1], legend_plots, legend_labels;
        orientation = :horizontal, nbanks = 2, labelsize = 11)

    ax3 = Axis(fig[1, 2],
        xlabel = "Harmonic order, k",
        ylabel = "Current amplitude, |Iₖ| [meV]",
        yscale = log10,
        title = "Current Harmonic Amplitudes",
        xticks = 1:max_harmonics)
    barplot!(ax3, 1:max_harmonics, max.(I_amp, 1e-12), color = :teal)

    display(fig)

    return (; fig, phases, F_vals, I_numeric, I_recon, A0, A_k, B_k,
        I_sin, I_cos, I_amp, params = parameter_values,
    )
end

function sweep_vg_mu_harmonic_amplitudes(hN, case; Vg_values, µ_values, n0 = 0,
        npoints = 64, max_harmonics = 8, solver, savepath = nothing, params...)
    npoints > 2max_harmonics || throw(ArgumentError("npoints must be larger than 2max_harmonics"))
    phases = 2π .* (0:(npoints - 1)) ./ npoints
    harmonics = 1:max_harmonics
    (; Δn) = case
    overrides = Base.structdiff(case, NamedTuple{(:Δn,)})

    amplitudes = Array{Float64}(undef, length(Vg_values), length(µ_values), max_harmonics)
    for (iµ, µ_value) in enumerate(µ_values), (iVg, Vg_value) in enumerate(Vg_values)
        p = (; params..., overrides..., n0, n = n0 + Δn, Vg = Vg_value, µN = µ_value)
        free_energies = [free_energy(hN(; p..., ϕ = ϕ_value), solver) for ϕ_value in phases]
        A = [(2 / npoints) * sum(free_energies .* cos.(k .* phases)) for k in harmonics]
        B = [(2 / npoints) * sum(free_energies .* sin.(k .* phases)) for k in harmonics]
        I_sin = -2 .* harmonics .* A
        I_cos = 2 .* harmonics .* B
        amplitudes[iVg, iµ, :] .= hypot.(I_sin, I_cos)
    end
    I_odd = dropdims(sum(amplitudes[:, :, isodd.(harmonics)], dims = 3), dims = 3)
    I_even = dropdims(sum(amplitudes[:, :, iseven.(harmonics)], dims = 3), dims = 3)

    common = (; xlabel = L"V_g \text{ [meV]}", ylabel = L"\mu \text{ [meV]}", xlabelsize = 16, ylabelsize = 16)

    colorrange = (0.0, max(maximum(I_odd), maximum(I_even)))
    fig = Figure(size = (1000, 450))
    Label(fig[0, 1:3], @sprintf("Δn = %g", Δn); fontsize = 16)
    ax1 = Axis(fig[1, 1]; title = L"\sum_{k \text{ odd}} |I_k| \;\; (2e, 6e, \ldots)", common...)
    hm = heatmap!(ax1, Vg_values, µ_values, I_odd; colorrange)
    ax2 = Axis(fig[1, 2]; title = L"\sum_{k \text{ even}} |I_k| \;\; (4e, 8e, \ldots)", common...)
    heatmap!(ax2, Vg_values, µ_values, I_even; colorrange)
    hideydecorations!(ax2, ticks = false, ticklabels = false)
    linkaxes!(ax1, ax2)
    Colorbar(fig[1, 3], hm; label = L"|I| \text{ [meV]}")
    display(fig)
    savepath === nothing || save(savepath, fig)

    return (; fig, Vg_values, µ_values, amplitudes, I_odd, I_even)
end

function flux_grid(nmax, npoints, area_ratio; δn = 1e-6)
    jumps = sort(unique(filter(x -> 0 < x < nmax,
        [[k + 0.5 for k in 0:ceil(Int, nmax)]; [(k + 0.5) / area_ratio for k in 0:ceil(Int, nmax * area_ratio)]])))
    edges = [0.0; jumps; nmax]
    segments = [range(edges[k] + (k > 1 ? δn : 0.0), edges[k+1] - (k < length(edges) - 1 ? δn : 0.0),
        length = max(2, round(Int, npoints * (edges[k+1] - edges[k]) / nmax))) for k in 1:length(edges)-1]
    return reduce(vcat, segments), jumps, segments
end

# Fluxoid quanta bars: winding of each pairing term (Δ0 ↔ n0, Δ1 ↔ n) at each flux
function fluxoid_bar!(ax, xs, windings)
    edges = [first(xs); [(xs[i] + xs[i+1]) / 2 for i in 1:length(xs)-1 if windings[i] != windings[i+1]]; last(xs)]
    starts = [1; [i + 1 for i in 1:length(xs)-1 if windings[i] != windings[i+1]]]
    for (k, i) in enumerate(starts)
        w = windings[i]
        poly!(ax, Rect(edges[k], 0, edges[k+1] - edges[k], 1),
            color = (Makie.wong_colors()[mod1(w + 1, 7)], 0.6), strokecolor = :black, strokewidth = 1)
        text!(ax, (edges[k] + edges[k+1]) / 2, 0.5, text = string(w), align = (:center, :center), fontsize = 12)
    end
end

function plot_flux_spectrum(hN; nmax = 2.0, npoints = 150, area_ratio = 0.5, nev = 40,
        solver = ES.ShiftInvert(ES.Arpack(nev = 4 + nev), 0), savepath = nothing, params...)
    Δ = NamedTuple(params).Δ
    ns, jumps, _ = flux_grid(nmax, npoints, area_ratio)
    b = bands(hN, ns; mapping = n -> ftuple(; params..., n, n0 = n * area_ratio), solver)

    fig = Figure(size = (600, 550))
    bar_specs = (("Inner ring", round.(Int, ns .* area_ratio)),
                 ("Outer ring", round.(Int, ns)))
    bar_axes = map(enumerate(bar_specs)) do (row, (label, windings))
        axb = Axis(fig[row, 1], ylabel = label, ylabelrotation = 0,
            title = row == 1 ? L"\Delta_0=\Delta_1" : "", titlealign = :right)
        fluxoid_bar!(axb, ns, windings)
        hidexdecorations!(axb)
        hideydecorations!(axb, label = false)
        hidespines!(axb)
        limits!(axb, 0, nmax, 0, 1)
        rowsize!(fig.layout, row, Fixed(22))
        axb
    end

    # BdG spectrum vs flux (style of figure 2(c) in arXiv_2506.15502/figures.jl)
    ax = Axis(fig[3, 1];
        xlabel = L"\Phi/\Phi_0", ylabel = L"\epsilon/(\Delta_0+\Delta_1)",
        yticks = (range(-2Δ, 2Δ, 3), ["-1", "0", "1"]),
        limits = ((0, nmax), (-2.5Δ, 2.5Δ)), xgridvisible = false,
        xlabelsize = 16, ylabelsize = 16)
    qplot!(b; color = (ψ, ϵ, ϕs) -> ifelse(abs(ϵ) < 0.12, 0, 1), colormap = ColorScheme([colorant"red", colorant"grey"]), hide = :nodes)
    vlines!(ax, jumps; color = :gray, linewidth = 0.5)
    foreach(axb -> linkxaxes!(ax, axb), bar_axes)
    rowgap!(fig.layout, 4)
    display(fig)
    savepath === nothing || save(savepath, fig)
    return (; fig, ns, b, jumps)
end

function plot_free_energy_current(hN, cases; n0 = 0, npoints = 128, solver = ES.LinearAlgebra(),
        maxscale = 1000, savepath = nothing, params...)
    # Periodic grid without the duplicate endpoint (needed for the periodic central difference)
    phases = collect(range(0, 2π, length = npoints + 1)[1:end-1])
    N = length(phases)
    Δϕ = 2π / N

    results = map(cases) do case
        (; Δn, Vg) = case
        overrides = Base.structdiff(case, NamedTuple{(:Δn, :scale)})
        F_vals = [free_energy(hN(; params..., overrides..., n0, n = n0 + Δn, ϕ = ϕ_val), solver) for ϕ_val in phases]
        I_vals = [2.0 * (F_vals[mod1(i + 1, N)] - F_vals[mod1(i - 1, N)]) / (2.0 * Δϕ) for i in 1:N]
        (; case, F_vals, I_vals)
    end

    # Common normalization: largest free-energy variation and largest current over all cases
    Fref = maximum(r -> maximum(r.F_vals) - minimum(r.F_vals), results)
    Iref = maximum(r -> maximum(abs, r.I_vals), results)

    # Magnification of each case: case.scale if given, otherwise the power of 10 that brings
    # its current close to Iref, capped at maxscale so cases that are ~0 do not blow up noise
    function case_scale(r)
        haskey(r.case, :scale) && return r.case.scale
        Imax = maximum(abs, r.I_vals)
        Imax > 0 || return 1
        return min(10.0^floor(log10(Iref / Imax)), maxscale)
    end
    scales = map(case_scale, results)

    common = (; xlabelsize = 16, ylabelsize = 16, xgridvisible = false)
    fig = Figure(size = (600, 550))
    ax1 = Axis(fig[1, 1];
        title = L"\Delta_0=\Delta_1", titlealign = :right,
        ylabel = L"(F - F_\text{min}) / \Delta F_\text{ref}", common...)
    ax2 = Axis(fig[2, 1];
        xlabel = L"\varphi", ylabel = L"I / I_\text{ref}",
        xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]), common...)

    # Close the curves at ϕ = 2π for plotting
    closed(v) = [v; first(v)]
    results = map(results, scales) do r, scale
        (; Δn, Vg) = r.case
        label = @sprintf("Δn = %g, Vg = %g meV", Δn, Vg) * (scale == 1 ? "" : @sprintf(" (×%g)", scale))
        lines!(ax1, [phases; 2π], closed(scale .* (r.F_vals .- minimum(r.F_vals)) ./ Fref); linewidth = 2, label)
        lines!(ax2, [phases; 2π], closed(scale .* r.I_vals ./ Iref); linewidth = 2, label)
        (; r..., label, scale)
    end

    hlines!(ax2, 0; color = :black, linewidth = 0.5)
    linkxaxes!(ax1, ax2)
    xlims!(ax2, 0, 2π)
    hidexdecorations!(ax1, grid = false)
    Legend(fig[3, 1], ax1; orientation = :horizontal, nbanks = 2, labelsize = 11, framevisible = false)
    rowgap!(fig.layout, 5)
    display(fig)
    savepath === nothing || save(savepath, fig)
    return (; fig, phases, results, Fref, Iref)
end

# ---------------------------------------------------------
# Shapiro steps: ground-state CPR of the ring → RSJ/RCSJ dynamics (rcsj.jl), T = 0
# ---------------------------------------------------------

# Number formatted for LaTeX, e.g. 7.4e-11 → 7.4 \times 10^{-11}
function latex_num(x; digits = 3)
    str = @sprintf("%.*g", digits, x)
    m = match(r"^(-?[\d.]+)e([+-])0*(\d+)$", str)
    m === nothing && return str
    return string(m[1], " \\times 10^{", m[2] == "-" ? "-" : "", m[3], "}")
end

# F(φ) and the lowest |ε| (smallest Andreev level) on the periodic grid φ = 2πj/N.
# Needs the full spectrum (ES.LinearAlgebra()), never shift-invert.
function sample_F(hN, N, params; showprogress = true)
    case, solver = params
    prog = Progress(N; desc = @sprintf("CPR N = %-4d ", N), enabled = showprogress)
    FE = [(x = free_energy_gap(hN(; case..., ϕ = 2π * j / N), solver); next!(prog); x) for j in 0:N-1]
    return first.(FE), last.(FE)
end

# N → 2N by inserting the midpoints
function refine_F(hN, F_old, E_old, params; showprogress = true)
    N = length(F_old)
    case, solver = params
    prog = Progress(N; desc = @sprintf("CPR N = %-4d ", 2N), enabled = showprogress)
    FE = [(x = free_energy_gap(hN(; case..., ϕ = 2π * (j + 1/2) / N), solver); next!(prog); x) for j in 0:N-1]
    interleave(old, new) = collect(Iterators.flatten(zip(old, new)))
    return interleave(F_old, first.(FE)), interleave(E_old, last.(FE))
end

function current_harmonics(F)
    N = length(F)
    kmax = div(N-1,2)
    I_sin = Vector{Float64}(undef, kmax)
    I_cos = Vector{Float64}(undef, kmax)
    for k in 1:kmax
        a = b = 0.0
        for j in 0:N-1
            s, c = sincospi(2 * mod(k * j, N) / N)
            a += F[j+1] * c
            b += F[j+1] * s
        end
        I_sin[k] = -4k * a / N
        I_cos[k] =  4k * b / N
    end
    return I_sin, I_cos
end

# Returns the current harmonics (code units), F(φ) and the smallest Andreev level min_φ min|ε| [meV]
function converged_cpr(hN, params, N0, N_max, tol_alias, tol_tail; showprogress = true)
    N0 ≥ 3 || throw(ArgumentError("N0 must be ≥ 3 to resolve at least one harmonic"))
    F, E = sample_F(hN, N0, params; showprogress)
    (s, c) = current_harmonics(F)
    Ic_est = cpr_max(s, c)
    converged = false
    while 2 * length(F) <= N_max
        (s_old, c_old) = (s, c)
        F, E = refine_F(hN, F, E, params; showprogress)
        (s, c) = current_harmonics(F)
        Ic_est = cpr_max(s, c)
        kold = length(s_old)

        alias_change = maximum(k -> abs(s[k] - s_old[k]) + abs(c[k] - c_old[k]), 1:kold)

        tail = maximum(k -> hypot(s[k], c[k]), kold+1:length(s))
        if alias_change < tol_alias * Ic_est && tail < tol_tail * Ic_est
            converged = true
            break
        end
    end
    converged || @warn @sprintf("CPR not converged at N = %d: probably a kink (zero-energy crossing)", length(F))
    kmax = something(findlast(≥(tol_tail * Ic_est), hypot.(s, c)), 1)
    return s[1:kmax], c[1:kmax], F, minimum(E)
end

function normalize_cpr(s, c; Ic_tol = 1e-6)
    Ic = cpr_max(s, c)
    Ic > 0 || throw(ArgumentError("CPR is identically zero"))
    Ic < Ic_tol && @warn @sprintf("I_c = %.2g is numerical noise, the normalized CPR is meaningless", Ic)
    return s ./ Ic, c ./ Ic, Ic
end

# One microscopic case p (all Hamiltonian parameters): converged CPR (computed once), then the Shapiro map.
#   scaling = :fixed     βc, drive.Ω, drive.idcs, drive.iacs are used as given (normalized units)
#   scaling = :physical  βc and Ω follow from this case's own I_c and device = (; R, C, f);
#                        drive = (; Idcs_uA, Iacs_uA) in µA, converted to this case's normalized i
# adiabatic_ratio: warn if hf > adiabatic_ratio × smallest Andreev level (equilibrium-CPR assumption)
function shapiro_case(hN, p, solver; cpr_settings, drive, bias, βc = 0.0, scaling = :fixed, device = nothing,
        adiabatic_ratio = 0.5, label = "")
    (; N0, N_max, tol_alias, tol_tail, Ic_tol) = cpr_settings
    s, c, F, Emin = converged_cpr(hN, (p, solver), N0, N_max, tol_alias, tol_tail)
    s, c, Ic = normalize_cpr(s, c; Ic_tol)
    Ic_A = Ic * I_UNIT_A
    if scaling == :physical
        u = print_rcsj_units(label, Ic; device...)
        (; βc, Ω) = u
        idcs = drive.Idcs_uA ./ (1e6 * Ic_A)
        iacs = drive.Iacs_uA ./ (1e6 * Ic_A)
        println(@sprintf("    smallest Andreev level min|ε| = %.3g meV, hf = %.3g meV", Emin, u.hf_meV))
        u.hf_meV > adiabatic_ratio * Emin && @warn @sprintf(
            "%s: hf = %.3g meV is not well below the smallest Andreev level %.3g meV: the equilibrium (adiabatic) CPR is questionable",
            label, u.hf_meV, Emin)
    elseif scaling == :fixed
        (; Ω, idcs, iacs) = drive
        println(@sprintf("%s: Ic = %.3g (code units) = %.3g nA, βc = %g, Ω = %g", label, Ic, Ic_A * 1e9, βc, Ω))
    else
        throw(ArgumentError("scaling must be :fixed or :physical"))
    end
    m = shapiro_map(s, c; Ω, βc, idcs, iacs, bias...)
    return (; F, s, c, Ic, Ic_A, Emin, noise = Ic < Ic_tol, scaling, m...)
end

#mr claude's plotting IDK what happens here and 
function plot_shapiro_map(results; iac_cuts = (1.0, 2.0), colorrange = (0, 2), savepath = nothing)
    nc = length(results)
    (; Ω, βc) = first(results)
    fig = with_theme(theme_latexfonts()) do
        fig = Figure(size = (420 * nc + 80, 1300))
        Label(fig[0, 1:nc], Makie.LaTeXString(@sprintf("RCSJ, \$\\Omega = %s,\\; \\beta_c = %s\$", latex_num(Ω), latex_num(βc))); fontsize = 18)
        axargs = (; xlabel = L"I_\text{ac}/I_c", ylabel = L"I_\text{dc}/I_c", xlabelsize = 16, ylabelsize = 16)
        hm = hh = nothing
        for (col, r) in enumerate(results)
            (; Δn, Vg) = r.case
            idc_mid = (r.idcs[1:end-1] .+ r.idcs[2:end]) ./ 2
            title = Makie.LaTeXString(@sprintf("\$\\Delta n = %g,\\; V_g = %s\\,\\text{meV},\\; I_c = %s\$", Δn, latex_num(Vg), latex_num(r.Ic)) *
                (r.noise ? " (noise)" : ""))

            ax_up = Axis(fig[1, col]; title, titlesize = 16, axargs...)
            hm = heatmap!(ax_up, r.iacs, idc_mid, r.dVdI_up'; colorrange, colormap = :viridis)
            text!(ax_up, 0.03, 0.97; text = "sweep up", space = :relative, align = (:left, :top), color = :white)
            ax_dn = Axis(fig[2, col]; axargs...)
            heatmap!(ax_dn, r.iacs, idc_mid, r.dVdI_down'; colorrange, colormap = :viridis)
            text!(ax_dn, 0.03, 0.97; text = "sweep down", space = :relative, align = (:left, :top), color = :white)
            ax_h = Axis(fig[3, col]; axargs...)
            hh = heatmap!(ax_h, r.iacs, r.idcs, (r.hyst ./ Ω)'; colormap = :magma)
            for ax in (ax_up, ax_dn)
                vlines!(ax, collect(iac_cuts); color = :white, linestyle = :dash, linewidth = 1)
            end
            linkaxes!(ax_up, ax_dn, ax_h)

            # Line cuts V/Ω vs idc: plateaus at integer (2e) and half-integer (4e / subharmonic) values
            ax_c = Axis(fig[4, col];
                xlabel = L"I_\text{dc}/I_c", ylabel = L"V/\Omega", xlabelsize = 16, ylabelsize = 16,
                yticks = 0:0.5:ceil(maximum(r.V_up) / Ω), ygridvisible = true)
            for (k, iac) in enumerate(iac_cuts)
                j = argmin(abs.(r.iacs .- iac))
                color = Makie.wong_colors()[k]
                label = Makie.LaTeXString(@sprintf("\$I_\\text{ac}/I_c = %s\$", latex_num(r.iacs[j])))
                lines!(ax_c, r.idcs, r.V_up[:, j] ./ Ω; color, linewidth = 2, label)
                lines!(ax_c, r.idcs, r.V_down[:, j] ./ Ω; color, linewidth = 2, linestyle = :dash)
            end
            col == 1 && axislegend(ax_c; position = :lt, labelsize = 13, framevisible = false)
        end
        Colorbar(fig[1:2, nc + 1], hm; label = L"\mathrm{d}V/\mathrm{d}I \; [R]")
        Colorbar(fig[3, nc + 1], hh; label = L"|V_\uparrow - V_\downarrow| / \Omega")
        rowsize!(fig.layout, 4, Relative(0.2))
        fig
    end
    display(fig)
    savepath === nothing || save(savepath, fig)
    return fig
end

# Shapiro grid, µN rows × Δn columns:
#   dV/dI maps of the up (switching) sweep, i_dc horizontal, i_ac vertical
#   hysteresis maps |V_up − V_down| (only if any case is hysteretic)
#   line cuts V/(hf/2e) = ⟨v⟩/Ω against i_dc at iac_cut, all µN of a Δn in one panel
#   step histogram of each cut: i_dc width per ⟨v⟩/Ω bin, so plateaus are peaks
# units = :normalized → currents in units of each case's I_c
# units = :physical   → currents in µA (i × I_c) on shared axes (one sample, one microwave power); iac_cut in µA
function plot_shapiro_grid(results, µs, Δns; iac_cut = 1.0, units = :normalized, colorrange = (0, 2),
        colormap = :Blues, binwidth = 0.02, savepath = nothing)
    µs, Δns = [µs...], [Δns...]
    results = reshape(results, length(µs), length(Δns))
    nr, nc = size(results)
    physical = units == :physical
    scale(r) = physical ? 1e6 * r.Ic_A : 1.0                           # normalized i → plotted current
    jac = map(r -> argmin(abs.(r.iacs .* scale(r) .- iac_cut)), results)  # cut column of each case
    w_up(k) = results[k].V_up[:, jac[k]] ./ results[k].Ω
    w_dn(k) = results[k].V_down[:, jac[k]] ./ results[k].Ω
    colors = Makie.wong_colors()                                      # fixed order: µN index → color
    hysteretic = any(r -> maximum(r.hyst) > 1e-3 * r.Ω, results)
    nmap = hysteretic ? 2nr : nr
    Vmax = ceil(maximum(k -> max(maximum(w_up(k)), maximum(w_dn(k))), eachindex(results)))
    Vmin = min(0.0, floor(minimum(k -> min(minimum(w_up(k)), minimum(w_dn(k))), eachindex(results))))
    # major ticks every 1 (labels thinned for wide ranges), minor grid at 0.5
    Vlab = max(1, ceil(Int, (Vmax - Vmin) / 12))
    vticks = (Vmin:Vmax, [rem(round(Int, k), Vlab) == 0 ? @sprintf("%d", k) : "" for k in Vmin:Vmax])
    tex(str) = Makie.LaTeXString(str)
    Idc_label, Iac_label = physical ? (L"I_\text{dc} \; [\mu\text{A}]", L"I_\text{ac} \; [\mu\text{A}]") :
        (L"I_\text{dc}/I_c", L"I_\text{ac}/I_c")
    V_label = L"V/(hf/2e)"

    fig = with_theme(theme_latexfonts(); fontsize = 15,
            Axis = (; xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true,
                xtickalign = 1, ytickalign = 1, xminorticksvisible = true, yminorticksvisible = true, xminortickalign = 1,
                yminortickalign = 1, titlefont = :regular, titlesize = 15)) do
        fig = Figure(size = (400 * nc + 110, 300 * nmap + 560), figure_padding = (10, 15, 10, 10))
        # Ω and βc differ per case with scaling = :physical: in the header only if common to all cases, else per map
        (; Ω, βc) = first(results)
        common_Ω = all(r -> r.Ω == Ω, results)
        common_βc = all(r -> r.βc == βc, results)
        header = "RCSJ Shapiro response" *
            (common_Ω ? @sprintf(",  \$\\Omega = %s\$", latex_num(Ω)) : "") *
            (common_βc ? @sprintf(",  \$\\beta_c = %s\$", latex_num(βc)) : "")
        Label(fig[0, 1:nc], tex(header); fontsize = 17, tellwidth = false)
        hm = hh = nothing
        map_axes = Axis[]
        cut_axes = Axis[]
        hist_axes = Axis[]
        for (j, Δn) in enumerate(Δns)
            for (i, µ) in enumerate(µs)
                r = results[i, j]
                sc = scale(r)
                idc_mid = (r.idcs[1:end-1] .+ r.idcs[2:end]) ./ 2
                # physical I_c (the normalized one is always 1)
                tag = @sprintf("\$\\mu_N = %s\$ meV,  \$I_c = %s\$ nA", latex_num(µ), latex_num(1e9 * r.Ic_A)) *
                    (common_Ω ? "" : @sprintf(",  \$\\Omega = %s\$", latex_num(r.Ω))) *
                    (common_βc ? "" : @sprintf(",  \$\\beta_c = %s\$", latex_num(r.βc))) * (r.noise ? "  (noise)" : "")
                ax = Axis(fig[i, j]; title = tex(tag), titlealign = :left,
                    xlabel = i == nr && !hysteretic ? Idc_label : "", ylabel = j == 1 ? Iac_label : "",
                    xticklabelsvisible = i == nr && !hysteretic, yticklabelsvisible = j == 1)
                # x = idc, y = iac (dVdI_up is idc × iac, as heatmap! expects)
                hm = heatmap!(ax, idc_mid .* sc, r.iacs .* sc, r.dVdI_up; colorrange, colormap,
                    highclip = Makie.to_colormap(colormap)[end])
                hlines!(ax, r.iacs[jac[i, j]] * sc; color = colors[i], linestyle = :dash, linewidth = 1.5)
                push!(map_axes, ax)
                if hysteretic
                    axh = Axis(fig[nr + i, j]; xlabel = i == nr ? Idc_label : "", ylabel = j == 1 ? Iac_label : "",
                        xticklabelsvisible = i == nr, yticklabelsvisible = j == 1)
                    hh = heatmap!(axh, r.idcs .* sc, r.iacs .* sc, r.hyst ./ r.Ω; colormap = :magma)
                    text!(axh, 0.03, 0.97; text = tex(@sprintf("\$\\mu_N = %s\$ meV", latex_num(µ))), space = :relative,
                        align = (:left, :top), color = :white, fontsize = 12)
                    push!(map_axes, axh)
                end
            end
            Label(fig[1, j, Top()], tex(@sprintf("\$\\Delta n = %g\$", Δn)); fontsize = 17, padding = (0, 0, 28, 0))

            # line cuts: all µN for this Δn in one panel
            axc = Axis(fig[nmap + 1, j]; xlabel = Idc_label, ylabel = j == 1 ? V_label : "",
                yticks = vticks, yminorticks = Vmin:0.5:Vmax, yticklabelsvisible = j == 1,
                ygridvisible = true, ygridcolor = (:black, 0.15), yminorgridvisible = true, yminorgridcolor = (:black, 0.06))
            # step histogram: i_dc width falling in each ⟨v⟩/Ω bin (bins centred on multiples of binwidth)
            axs = Axis(fig[nmap + 2, j]; xlabel = V_label, ylabel = j == 1 ? (physical ? L"\Delta I_\text{dc} \; [\mu\text{A}]" :
                L"\Delta I_\text{dc}/I_c") : "", xticks = vticks, xminorticks = Vmin:0.5:Vmax,
                xgridvisible = true, xgridcolor = (:black, 0.15), xminorgridvisible = true, xminorgridcolor = (:black, 0.06))
            edges = (Vmin - binwidth / 2):binwidth:(Vmax + binwidth / 2)
            for (i, µ) in enumerate(µs)
                r = results[i, j]
                k = LinearIndices(results)[i, j]
                idcs = r.idcs .* scale(r)
                lines!(axc, idcs, w_up(k); color = (colors[i], hysteretic ? 0.5 : 1.0), linewidth = 2.5)
                # down sweep thin and opaque on top of the translucent up sweep, so it stays visible where they overlap
                hysteretic && lines!(axc, idcs, w_dn(k); color = colors[i], linewidth = 1.3, linestyle = :dash)
                for (w, style) in ((w_up(k), :solid), (w_dn(k), :dash))
                    style == :dash && !hysteretic && continue
                    widths = zeros(length(edges) - 1)
                    for x in w
                        b = searchsortedlast(edges, x)
                        1 ≤ b < length(edges) && (widths[b] += step(idcs))
                    end
                    stairs!(axs, collect(edges), [widths; widths[end]]; step = :post, color = colors[i],
                        linewidth = style == :solid ? 1.8 : 1.2, linestyle = style)
                end
            end
            xlims!(axc, extrema(results[1, j].idcs .* scale(results[1, j]))...)
            ylims!(axc, Vmin - 0.1, Vmax + 0.1)
            xlims!(axs, Vmin - 0.1, Vmax + 0.1)
            ylims!(axs, 0, nothing)
            push!(cut_axes, axc)
            push!(hist_axes, axs)
        end
        linkaxes!(map_axes...)
        linkaxes!(cut_axes...)
        linkaxes!(hist_axes...)

        Colorbar(fig[1:nr, nc + 1], hm; label = L"\mathrm{d}V/\mathrm{d}I \; [R]", width = 12, ticksize = 4)
        hysteretic && Colorbar(fig[nr+1:2nr, nc + 1], hh; label = L"|V_\uparrow - V_\downarrow| / (hf/2e)", width = 12, ticksize = 4)
        # legend below: µN identity (only if > 1) plus sweep direction (only if hysteretic)
        entries, labels = Any[], Any[]
        if nr > 1
            append!(entries, [LineElement(color = colors[i], linewidth = 2) for i in 1:nr])
            append!(labels, [tex(@sprintf("\$\\mu_N = %s\$ meV", latex_num(µ))) for µ in µs])
        end
        if hysteretic
            append!(entries, [LineElement(color = (:gray30, 0.5), linewidth = 2.5), LineElement(color = :gray30, linewidth = 1.3, linestyle = :dash)])
            append!(labels, ["sweep up (switching)", "sweep down (retrapping)"])
        end
        iac_cuts = unique(round.([r.iacs[jac[k]] * scale(r) for (k, r) in pairs(results)]; sigdigits = 3))
        cut_text = physical ? @sprintf("line cut at \$I_\\text{ac} = %s\\;\\mu\$A", join(latex_num.(iac_cuts), ", ")) :
            @sprintf("line cut at \$I_\\text{ac}/I_c = %s\$", join(latex_num.(iac_cuts), ", "))
        Label(fig[nmap + 1, 1:nc, Top()], tex(cut_text); padding = (0, 0, 6, 0), fontsize = 14, halign = :left, tellwidth = false)
        isempty(entries) || Legend(fig[nmap + 3, 1:nc], entries, labels; orientation = :horizontal,
            framevisible = false, labelsize = 13, tellheight = true, tellwidth = false)
        rowsize!(fig.layout, nmap + 1, Relative(0.8 / (nmap + 2)))
        rowsize!(fig.layout, nmap + 2, Relative(0.6 / (nmap + 2)))
        rowgap!(fig.layout, 8)
        colgap!(fig.layout, 32)
        rowgap!(fig.layout, nmap, 16)
        fig
    end

    display(fig)
    savepath === nothing || save(savepath, fig; px_per_unit = 2)
    return fig
end

# ---------------------------------------------------------------------
# Shapiro run storage: one JLD2 file per run, named by an ID, plus an index.csv listing all runs
#   id = "yyyymmdd-HHMMSS" or "yyyymmdd-HHMMSS_label"
# ---------------------------------------------------------------------
const SHAPIRO_RUN_DIR = joinpath(@__DIR__, "data", "shapiro_runs")

fmt_values(v) = join(map(x -> @sprintf("%g", x), [v...]), " ")
csv_field(x) = (s = string(x); occursin(r"[,\"\n]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s)

# meta: any NamedTuple of settings to keep with the run (params, drive, bias, βc, cpr_settings, device, ...)
function save_shapiro_run(results, µs, Δns; meta = (;), label = "", dir = SHAPIRO_RUN_DIR)
    mkpath(dir)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    id = isempty(label) ? stamp : string(stamp, "_", replace(label, r"[^\w\-]" => "-"))
    isfile(joinpath(dir, id * ".jld2")) && (id *= "-" * string(rand(UInt16), base = 16))
    r1 = first(results)
    # ASCII keys in the file (Julia folds µ/μ in identifiers, which would break string lookup)
    # per case (they differ with scaling = :physical): code-unit and physical Ic, βc, Ω
    per_case = map(r -> (; Ic = r.Ic, Ic_A = r.Ic_A, βc = r.βc, Ω = r.Ω, Emin = r.Emin), vec(results))
    jldsave(joinpath(dir, id * ".jld2"); id, date = string(now()), label, results, mus = [µs...], dns = [Δns...],
        meta = merge(meta, (; per_case)))
    index = joinpath(dir, "index.csv")
    isfile(index) || write(index, "id,date,label,Δn,µN,Ω,βc,n_idc,n_iac,Ic\n")
    join_g(f) = join(map(r -> @sprintf("%.3g", f(r)), vec(results)), " ")
    row = (id, Dates.format(now(), "yyyy-mm-dd HH:MM"), label, fmt_values(Δns), fmt_values(µs),
        join_g(r -> r.Ω), join_g(r -> r.βc), length(r1.idcs), length(r1.iacs), join_g(r -> r.Ic))
    open(io -> println(io, join(map(csv_field, row), ",")), index, "a")
    println("saved Shapiro run ", id)
    return id
end

function load_shapiro_run(id; dir = SHAPIRO_RUN_DIR)
    file = joinpath(dir, endswith(id, ".jld2") ? id : id * ".jld2")
    isfile(file) || error("no Shapiro run $id in $dir (see list_shapiro_runs())")
    d = load(file)
    return (; id = d["id"], date = d["date"], label = d["label"], results = d["results"],
        µs = d["mus"], Δns = d["dns"], meta = d["meta"])
end

load_shapiro_runs(ids...; dir = SHAPIRO_RUN_DIR) = [load_shapiro_run(id; dir) for id in ids]

function list_shapiro_runs(; dir = SHAPIRO_RUN_DIR)
    index = joinpath(dir, "index.csv")
    isfile(index) ? print(read(index, String)) : println("no Shapiro runs saved in $dir yet")
    return nothing
end

# Combine runs into one grid for plot_shapiro_grid: side by side (new Δn columns) if they share µs,
# or stacked (new µN rows) if they share Δns
function combine_shapiro_runs(runs...)
    length(runs) == 1 && return first(runs)
    r1 = first(runs)
    grids = [reshape(r.results, length(r.µs), length(r.Δns)) for r in runs]
    if all(r -> r.µs == r1.µs, runs)
        return (; results = reduce(hcat, grids), µs = r1.µs, Δns = reduce(vcat, [r.Δns for r in runs]))
    elseif all(r -> r.Δns == r1.Δns, runs)
        return (; results = reduce(vcat, grids), µs = reduce(vcat, [r.µs for r in runs]), Δns = r1.Δns)
    else
        error("runs can only be combined if they share either all µN values or all Δn values")
    end
end
