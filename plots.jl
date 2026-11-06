# All figures' plotting functions. Loaded by project.jl; needs Makie and the functions of project.jl.

# ---------------------------------------------------------------------
# BdG spectrum against flux, with fluxoid winding bars
# ---------------------------------------------------------------------
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

function plot_flux_spectrum(hN2; nmax = 2.0, npoints = 150, area_ratio = 0.5, nev = 40,
        solver = ES.ShiftInvert(ES.ArnoldiMethod(nev = 4 + nev), 0), savepath = nothing, params...)
    Δ = NamedTuple(params).Δ
    ns, jumps, _ = flux_grid(nmax, npoints, area_ratio)
    b = bands(hN2, ns; mapping = n -> ftuple(; params..., n, n0 = n * area_ratio), solver)

    fig = Figure(size = (600, 550))
    bar_specs = (("Inner ring", round.(Int, ns .* area_ratio)),
                 ("Outer ring", round.(Int, ns)))
    bar_axes = map(enumerate(bar_specs)) do (row, (label, windings))
        axb = Axis(fig[row, 1], ylabel = label, ylabelrotation = 0)
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

# ---------------------------------------------------------------------
# Free energy, CPR and Fourier-harmonic figures
# ---------------------------------------------------------------------
# Ground state F₀ (even) and F₀ + εt (odd: one quasiparticle in the level that is the lowest positive one at φsel,
# followed through crossings by spectrum_F0_tracked), not the naive F₀ + min|ε| at each ϕ
function fourier_analysis(hN2;
        npoints = 128, max_harmonics = 12, φsel = π/2,
        savepath = joinpath(@__DIR__, "figures/fourier_"), params...)

    N = npoints
    phases = tracked_phases(N)   # same grid as spectrum_F0_tracked
    parameter_values = (; params...)

    println("Calculating free energy across $(N) phase points...")
    println("System parameters:")
    show(stdout, MIME"text/plain"(), parameter_values)
    println("\nFourier settings: npoints = $npoints, max_harmonics = $max_harmonics, φsel = $φsel")
    F_vals, E_vals, Emin = spectrum_F0_tracked(hN2, N, parameter_values; φsel)

    I_numeric, I_recon, A0, A_k, B_k, I_sin, I_cos, I_amp = fourier(F_vals, phases, max_harmonics, N)
    I_numeric2, I_recon2, A02, A_k2, B_k2, I_sin2, I_cos2, I_amp2 = fourier(F_vals .+ E_vals, phases, max_harmonics, N)

    # ---------------------------------------------------------
    # Visualization Dashboard
    # ---------------------------------------------------------
    even_color, odd_color = :royalblue, :darkorange
    phase_ticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"])
    tickfmt = values -> [@sprintf("%.3g", value) for value in values]

    F_mean = sum(F_vals) / length(F_vals)
    fig = Figure(size = (1100, 650), fontsize = 14)

    ax1 = Axis(fig[1, 1],
        ylabel = "Free energy F [meV]",
        title = "Free energy and Josephson current",
        xticks = phase_ticks,
        ytickformat = tickfmt)
    lines!(ax1, phases, F_vals .- F_mean; color = even_color, linewidth = 2.5)
    lines!(ax1, phases, F_vals .+ E_vals .- F_mean; color = odd_color, linewidth = 2.5)
    hidexdecorations!(ax1, grid = false)

    ax2 = Axis(fig[2, 1],
        xlabel = "Phase difference ϕ",
        ylabel = "Current I [meV]",
        xticks = phase_ticks,
        ytickformat = tickfmt)
    hlines!(ax2, 0; color = (:black, 0.3), linewidth = 1)
    lines!(ax2, phases, I_numeric; color = even_color, linewidth = 2.5)
    lines!(ax2, phases, I_numeric2; color = odd_color, linewidth = 2.5)
    linkxaxes!(ax1, ax2)
    xlims!(ax2, 0, 2π)
    rowgap!(fig.layout, 1, 8)

    ax3 = Axis(fig[1:2, 2],
        xlabel = "Harmonic order k",
        ylabel = "Current amplitude |Iₖ| [meV]",
        yscale = log10,
        title = "Current harmonic amplitudes",
        xticks = 1:max_harmonics,
        xgridvisible = false)
    harmonics = 1:max_harmonics
    amps = max.(vcat(I_amp, I_amp2), 1e-12)
    barplot!(ax3, vcat(harmonics, harmonics), amps;
        dodge = vcat(fill(1, max_harmonics), fill(2, max_harmonics)),
        color = vcat(fill(even_color, max_harmonics), fill(odd_color, max_harmonics)),
        fillto = minimum(amps) / 2, gap = 0.25)

    Legend(fig[3, 1:2],
        [LineElement(color = even_color, linewidth = 3), LineElement(color = odd_color, linewidth = 3)],
        ["Ground state", "Filled soliton"];
        orientation = :horizontal, framevisible = false)
    colsize!(fig.layout, 1, Relative(0.55))

    display(fig)

    return (; fig, phases, F_vals, E_vals, Emin, I_numeric, I_numeric2, I_recon, A0, A_k, B_k,
        I_sin, I_cos, I_amp, I_amp2, params = parameter_values,
    )
end

# Ground state F₀ and first excited state F₁ = F₀ + εt (one quasiparticle in the tracked level), both from the same
# dense spectrum. Currents from the tabulated CPR of each (finite differences, interpolated). Single case (Δn, Vg, …), no normalization.
# The odd branch has period 2πν, ν = odd_period_factor (4π for mult = 1, else 2π); everything is over 0 … 2πν
function free_energy_current_data(hN2, case; n0 = 0, cpr_settings, φsel = π/2, nplot = 400, params...)
    (; N_max) = cpr_settings
    tol = get(cpr_settings, :tol_cpr, 1e-3)

    overrides = Base.structdiff(case, NamedTuple{(:Δn,)})
    p = (; params..., overrides..., n0, n = n0 + case.Δn)
    ν = odd_period_factor(p)
    phases = tracked_phases(N_max, ν)   # same grid as spectrum_F0_tracked
    φ0 = first(phases)
    φplot = range(0, 2π * ν, length = ν * nplot)
    F_vals, εt, _ = spectrum_F0_tracked(hN2, N_max, p; φsel, ν, nev = get(cpr_settings, :nev, 40))
    F1_vals = F_vals .+ εt
    cpr0 = cpr_table_checked(F_vals[1:N_max], 2π, tol; φ0, name = "F₀")      # F₀ is 2π-periodic
    cpr1 = cpr_table_checked(F1_vals, 2π * ν, tol; φ0, name = "F₁")
    I_vals = [cpr_value(φ, cpr0) for φ in φplot]
    I1_vals = [cpr_value(φ, cpr1) for φ in φplot]
    return (; phases, φplot, F_vals, F1_vals, I_vals, I1_vals, cpr0, cpr1, ν)
end

# F on the grid (wrapped periodically on both ends, so it spans 0 … 2πν on the shifted grid) into axF and I from the
# CPR tables into axI; ground state solid, first excited dashed
function free_energy_current_lines!(axF, axI, d; color = Makie.wong_colors()[1])
    closed(v) = [last(v); v; first(v)]
    φF = [last(d.phases) - 2π * d.ν; d.phases; first(d.phases) + 2π * d.ν]
    Fmin = minimum(d.F_vals)
    lines!(axF, φF, closed(d.F_vals .- Fmin); color, linewidth = 2)
    lines!(axF, φF, closed(d.F1_vals .- Fmin); color, linewidth = 2, linestyle = :dash)
    hlines!(axI, 0; color = :black, linewidth = 0.5)
    lines!(axI, d.φplot, d.I_vals; color, linewidth = 2)
    lines!(axI, d.φplot, d.I1_vals; color, linewidth = 2, linestyle = :dash)
    linkxaxes!(axF, axI)
    xlims!(axI, 0, 2π * d.ν)
    hidexdecorations!(axF, grid = false)
end

parity_legend!(pos) = Legend(pos,
    [LineElement(; color = :black, linewidth = 2), LineElement(; color = :black, linewidth = 2, linestyle = :dash)],
    ["Even fermion parity", "Odd fermion parity"]; orientation = :horizontal, labelsize = 11, framevisible = false)

# x ticks of a 0 … 2πν axis: every π/2 up to 2π, every π beyond
φ_ticks(ν) = ν == 1 ? πhalf_ticks(2) : πticks(2ν)

FI_LABELS = (; F = L"F - F_\text{min} \text{ [meV]}", I = L"I \text{ [meV]}")
FI_AXIS = (; xlabelsize = 16, ylabelsize = 16, xgridvisible = false)

function plot_free_energy_current(hN2, case; savepath = nothing, kw...)
    d = free_energy_current_data(hN2, case; kw...)
    fig = Figure(size = (600, 600))
    ax1 = Axis(fig[1, 1]; ylabel = FI_LABELS.F, FI_AXIS...)
    ax2 = Axis(fig[2, 1]; xlabel = L"\varphi", ylabel = FI_LABELS.I, xticks = πhalf_ticks(2d.ν), FI_AXIS...)
    free_energy_current_lines!(ax1, ax2, d)
    parity_legend!(fig[3, 1])
    rowgap!(fig.layout, 5)
    display(fig)
    savepath === nothing || save(savepath, fig)
    return (; fig, d...)
end

# F and I of several cases side by side, one column each (own φ range 0 … 2πν), from free_energy_current_data
function plot_free_energy_current_cases(datas, titles; panel_labels = false, savepath = nothing)
    n = length(datas)
    fig = Figure(size = (380n + 90, 560))
    for (j, (d, title)) in enumerate(zip(datas, titles))
        axF = Axis(fig[1, j]; title, ylabel = j == 1 ? FI_LABELS.F : "", FI_AXIS...)
        axI = Axis(fig[2, j]; xlabel = L"\varphi", ylabel = j == 1 ? FI_LABELS.I : "", xticks = φ_ticks(d.ν), FI_AXIS...)
        free_energy_current_lines!(axF, axI, d; color = Makie.wong_colors()[j])
        if panel_labels
            panel_label!(fig, 1, j, j)
            panel_label!(fig, 2, j, n + j)
        end
    end
    parity_legend!(fig[3, 1:n])
    rowgap!(fig.layout, 5)
    display(fig)
    savepath === nothing || save(savepath, fig; px_per_unit = 2)
    return fig
end

# x ticks at multiples of π/2 from 0 to nπ: "0", "π/2", "π", "3π/2", "2π", …
πhalf_ticks(n) = (m = round(Int, 2n);
    (π/2 .* (0:m), [k == 0 ? "0" : k == 1 ? "π/2" : k == 2 ? "π" : iseven(k) ? "$(k ÷ 2)π" : "$(k)π/2" for k in 0:m]))
# x ticks at multiples of π from 0 to nπ: "0", "π", "2π", …
πticks(n) = (m = round(Int, n); (π .* (0:m), [k == 0 ? "0" : k == 1 ? "π" : "$(k)π" for k in 0:m]))

# Panel label "a)", "b)", … (k = 1, 2, …) in the top-left corner of the layout cell fig[r, c]
panel_label!(fig, r, c, k; fontsize = 18) = Label(fig[r, c, TopLeft()], string('a' + k - 1, ")"); fontsize,
    font = :bold, halign = :right, padding = (0, 8, 4, 0))

# Gate profile V_g cos(mult θ) (θ = position angle around the ring; φ is the phase difference) as a LaTeX string,
# for titles and Shapiro grid labels: cos(θ), cos(2θ), …
gate_label(mult) = mult == 1 ? "\$V_g \\propto \\cos(\\theta)\$" : "\$V_g \\propto \\cos($(mult)\\theta)\$"

# ---------------------------------------------------------------------
# Bands against ϕ with tracked subbands and their |ψ|² along the ring (from test.jl)
# ---------------------------------------------------------------------
# Index of the subband of b that holds the lowest positive level at φ
function lowest_positive_subband(b, φ)
    εmin(x) = minimum(filter(>(0), real.(Quantica.energy.(Quantica.vertices(x)))); init = Inf)
    s = b[(φ, :)]
    return argmin(k -> εmin(s[k]), eachindex(s))
end

# |ψ|² per site from BdG states (norb orbitals per site: 2 = e↑, h↓ for hN2; 4 = e↑, e↓, h↑, h↓ for hN), one column per state
site_density(ψs, norb = 2) = dropdims(sum(reshape(abs2.(ψs), norb, :, size(ψs, 2)), dims = 1), dims = 1)

# |ψ|² per site along subband j of b, read from its band vertices (summed over the degenerate states), sorted by φ
function subband_density(b, j)
    vs = Quantica.vertices(Quantica.subbands(b)[j])
    φ = [Quantica.base_coordinates(v)[1] for v in vs]
    ρ = reduce(hcat, [vec(sum(site_density(Quantica.states(v)), dims = 2)) / Quantica.degeneracy(v) for v in vs])
    p = sortperm(φ)
    return φ[p], ρ[:, p]
end

# Bands of b against ϕ (gray) with the subbands js highlighted in colors, and next to it |ψ|² per site of those subbands
# along ϕ (overlaid, each colormap fading to transparent at zero density). ϕ axis 0 … 4π/mult, energy window ±elim
function plot_bands_density(b, js, colors, mult; elim = 0.5, panel_labels = false, savepath = nothing)
    φmax = 4π / mult
    fig = Figure(size = (1200, 520))
    Label(fig[0, 1:2+length(js)], Makie.LaTeXString(gate_label(mult)); fontsize = 20)
    common = (; xlabel = L"\varphi", xlabelsize = 18, ylabelsize = 18, titlesize = 20, xticks = πticks(round(Int, φmax / π)))

    ax = Axis(fig[1, 1]; ylabel = L"\varepsilon \text{ [meV]}", common...)
    plotbands!(ax, b; color = :gray75, hide = :nodes)
    for (j, c) in zip(js, colors)
        plotbands!(ax, b[[j]]; color = c, size = 3, hide = :nodes)
    end
    limits!(ax, 0, φmax, -elim, elim)

    ax2 = Axis(fig[1, 2]; ylabel = "site", title = L"|\psi|^2", common...)
    for (i, (j, c)) in enumerate(zip(js, colors))
        φ, ρ = subband_density(b, j)
        hm = heatmap!(ax2, φ, 1:size(ρ, 1), permutedims(ρ); colormap = [(c, 0.0), (c, 1.0)])
        Colorbar(fig[1, 2+i], hm)
    end
    xlims!(ax2, 0, φmax)
    colsize!(fig.layout, 1, Relative(0.4))
    panel_labels && (panel_label!(fig, 1, 1, 1); panel_label!(fig, 1, 2, 2))
    display(fig)
    savepath === nothing || save(savepath, fig; px_per_unit = 2)
    return fig
end

# ---------------------------------------------------------------------
# Shapiro grid figure (results of shapiro_case / shapiro_parity_cases)
# ---------------------------------------------------------------------
# Number formatted for LaTeX, e.g. 7.4e-11 → 7.4 \times 10^{-11}
function latex_num(x; digits = 3)
    str = @sprintf("%.*g", digits, x)
    m = match(r"^(-?[\d.]+)e([+-])0*(\d+)$", str)
    m === nothing && return str
    return string(m[1], " \\times 10^{", m[2] == "-" ? "-" : "", m[3], "}")
end

# A Shapiro result restricted to lo ≤ i_dc ≤ hi (normalized units): idcs, V and hysteresis rows, and the dV/dI rows
# between the kept bias points
function crop_idc(r, (lo, hi))
    sel = findfirst(≥(lo), r.idcs):findlast(≤(hi), r.idcs)
    length(sel) ≥ 2 || throw(ArgumentError("idc_lims = ($lo, $hi) keeps fewer than two i_dc points"))
    dsel = first(sel):last(sel)-1
    return merge(r, (; idcs = r.idcs[sel], V_up = r.V_up[sel, :], V_down = r.V_down[sel, :], hyst = r.hyst[sel, :],
        dVdI_up = r.dVdI_up[dsel, :], dVdI_down = r.dVdI_down[dsel, :]))
end

# Shapiro grid, µN rows × Δn columns (rows / columns can instead be named by strings, e.g. ["even", "odd"]):
#   dV/dI maps of the up (switching) sweep, i_dc horizontal, i_ac vertical
#   hysteresis maps |V_up − V_down| (only if any case is hysteretic)
#   line cuts V/(hf/2e) = ⟨v⟩/Ω against i_dc at iac_cut, all µN of a Δn in one panel (opacity cut_alpha, so overlaps show)
# units = :normalized → currents in units of each case's I_c
# units = :physical   → currents in µA (i × I_c) on shared axes (one sample, one microwave power); iac_cut in µA
# idc_lims = (lo, hi) → zoom: only i_dc in [lo, hi] (plotted units) is shown, also in the cuts
# iac_cut: one value, or one per column
# shared_V = false → each column has its own V/(hf/2e) range in the cuts (columns with different Ω)
# panel_labels = true → a), b), … row by row (maps, then cuts)
function plot_shapiro_grid(results, µs, Δns; iac_cut = 1.0, units = :normalized, colorrange = (0, 2),
        colormap = :Blues, cut_alpha = 0.7, idc_lims = nothing, shared_V = true, panel_labels = false, savepath = nothing)
    µs, Δns = [µs...], [Δns...]
    results = reshape(results, length(µs), length(Δns))
    physical = units == :physical
    scale(r) = physical ? 1e6 * r.Ic_A : 1.0                           # normalized i → plotted current
    idc_lims === nothing || (results = map(r -> crop_idc(r, idc_lims ./ scale(r)), results))
    nr, nc = size(results)
    cuts = iac_cut isa Number ? fill(iac_cut, nc) : collect(iac_cut)
    length(cuts) == nc || throw(ArgumentError("iac_cut needs one value or one per column ($nc)"))
    jac = [argmin(abs.(results[i, j].iacs .* scale(results[i, j]) .- cuts[j])) for i in 1:nr, j in 1:nc]  # cut column of each case
    w_up(k) = results[k].V_up[:, jac[k]] ./ results[k].Ω
    w_dn(k) = results[k].V_down[:, jac[k]] ./ results[k].Ω
    colors = Makie.wong_colors()                                      # fixed order: µN index → color
    hysteretic = any(r -> maximum(r.hyst) > 1e-3 * r.Ω, results)
    nmap = hysteretic ? 2nr : nr
    # V/(hf/2e) range of the cases ks: major ticks every 1 (labels thinned for wide ranges), minor grid at 0.5
    function V_range(ks)
        Vmax = ceil(maximum(k -> max(maximum(w_up(k)), maximum(w_dn(k))), ks))
        Vmin = min(0.0, floor(minimum(k -> min(minimum(w_up(k)), minimum(w_dn(k))), ks)))
        Vlab = max(1, ceil(Int, (Vmax - Vmin) / 12))
        return Vmin, Vmax, (Vmin:Vmax, [rem(round(Int, k), Vlab) == 0 ? @sprintf("%d", k) : "" for k in Vmin:Vmax])
    end
    V_ranges = shared_V ? fill(V_range(eachindex(results)), nc) : [V_range(LinearIndices(results)[:, j]) for j in 1:nc]
    tex(str) = Makie.LaTeXString(str)
    Idc_label, Iac_label = physical ? (L"I_\text{dc} \; [\mu\text{A}]", L"I_\text{ac} \; [\mu\text{A}]") :
        (L"I_\text{dc}/I_c", L"I_\text{ac}/I_c")
    V_label = L"V/(hf/2e)"
    rowlabel(µ) = µ isa AbstractString ? µ : @sprintf("\$\\mu_N = %s\$ meV", latex_num(µ))   # rows: µN values or names
    collabel(Δn) = Δn isa AbstractString ? Δn : @sprintf("\$\\Delta n = %g\$", Δn)        # columns: Δn values or names

    fig = with_theme(theme_latexfonts(); fontsize = 15,
            Axis = (; xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true,
                xtickalign = 1, ytickalign = 1, xminorticksvisible = true, yminorticksvisible = true, xminortickalign = 1,
                yminortickalign = 1, titlefont = :regular, titlesize = 15)) do
        fig = Figure(size = (400 * nc + 110, 300 * nmap + 380), figure_padding = (10, 15, 10, 10))
        # Ω and βc differ per case with scaling = :physical: in the header only if common to all cases, else per map
        (; Ω, βc) = first(results)
        common_Ω = all(r -> r.Ω == Ω, results)
        common_βc = all(r -> r.βc == βc, results)
        header = "RCSJ Shapiro response" #*
#            (common_Ω ? @sprintf(",  \$\\Omega = %s\$", latex_num(Ω)) : "") *
#            (common_βc ? @sprintf(",  \$\\beta_c = %s\$", latex_num(βc)) : "")
        Label(fig[0, 1:nc], tex(header); fontsize = 17, tellwidth = false)
        hm = hh = nothing
        map_axes = Axis[]
        cut_axes = Axis[]
        for (j, Δn) in enumerate(Δns)
            for (i, µ) in enumerate(µs)
                r = results[i, j]
                sc = scale(r)
                idc_mid = (r.idcs[1:end-1] .+ r.idcs[2:end]) ./ 2
                # physical I_c (the normalized one is always 1)
                tag = rowlabel(µ) * @sprintf(",  \$I_c = %s\$ nA", latex_num(1e9 * r.Ic_A)) *
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
                    text!(axh, 0.03, 0.97; text = tex(rowlabel(µ)), space = :relative,
                        align = (:left, :top), color = :white, fontsize = 12)
                    push!(map_axes, axh)
                end
            end
            Label(fig[1, j, Top()], tex(collabel(Δn)); fontsize = 17, padding = (0, 0, 28, 0))
            Vmin, Vmax, vticks = V_ranges[j]
            Vlabels = j == 1 || !shared_V

            # line cuts: all µN for this Δn in one panel
            axc = Axis(fig[nmap + 1, j]; xlabel = Idc_label, ylabel = Vlabels ? V_label : "",
                yticks = vticks, yminorticks = Vmin:0.5:Vmax, yticklabelsvisible = Vlabels,
                ygridvisible = true, ygridcolor = (:black, 0.15), yminorgridvisible = true, yminorgridcolor = (:black, 0.06))
            for (i, µ) in enumerate(µs)
                r = results[i, j]
                k = LinearIndices(results)[i, j]
                idcs = r.idcs .* scale(r)
                lines!(axc, idcs, w_up(k); color = (colors[i], hysteretic ? 0.5 : cut_alpha), linewidth = 2.5)
                # down sweep thin and opaque on top of the translucent up sweep, so it stays visible where they overlap
                hysteretic && lines!(axc, idcs, w_dn(k); color = colors[i], linewidth = 1.3, linestyle = :dash)
            end
            xlims!(axc, extrema(results[1, j].idcs .* scale(results[1, j]))...)
            ylims!(axc, Vmin - 0.1, Vmax + 0.1)
            push!(cut_axes, axc)
        end
        linkaxes!(map_axes...)
        shared_V && linkaxes!(cut_axes...)
        if panel_labels
            for row in 1:nmap+1, j in 1:nc
                panel_label!(fig, row, j, (row - 1) * nc + j)
            end
        end

        Colorbar(fig[1:nr, nc + 1], hm; label = L"\mathrm{d}V/\mathrm{d}I \; [R]", width = 12, ticksize = 4)
        hysteretic && Colorbar(fig[nr+1:2nr, nc + 1], hh; label = L"|V_\uparrow - V_\downarrow| / (hf/2e)", width = 12, ticksize = 4)
        # legend below: µN identity (only if > 1) plus sweep direction (only if hysteretic)
        entries, labels = Any[], Any[]
        if nr > 1
            append!(entries, [LineElement(color = colors[i], linewidth = 2) for i in 1:nr])
            append!(labels, [tex(rowlabel(µ)) for µ in µs])
        end
        if hysteretic
            append!(entries, [LineElement(color = (:gray30, 0.5), linewidth = 2.5), LineElement(color = :gray30, linewidth = 1.3, linestyle = :dash)])
            append!(labels, ["sweep up (switching)", "sweep down (retrapping)"])
        end
        # line cut label: one over all columns if they share the cut, else one per column
        cut_text(ks) = (c = unique(round.([results[k].iacs[jac[k]] * scale(results[k]) for k in ks]; sigdigits = 3));
            physical ? @sprintf("line cut at \$I_\\text{ac} = %s\\;\\mu\$A", join(latex_num.(c), ", ")) :
            @sprintf("line cut at \$I_\\text{ac}/I_c = %s\$", join(latex_num.(c), ", ")))
        label_spans = allequal(cuts) ? [1:nc] : [j:j for j in 1:nc]
        for js in label_spans
            Label(fig[nmap + 1, js, Top()], tex(cut_text(vec(LinearIndices(results)[:, js])));
                padding = (0, 0, 6, 0), fontsize = 14, halign = :left, tellwidth = false)
        end
        isempty(entries) || Legend(fig[nmap + 2, 1:nc], entries, labels; orientation = :horizontal,
            framevisible = false, labelsize = 13, tellheight = true, tellwidth = false)
        rowsize!(fig.layout, nmap + 1, Relative(0.8 / (nmap + 1)))
        rowgap!(fig.layout, 8)
        colgap!(fig.layout, 32)
        rowgap!(fig.layout, nmap, 16)
        fig
    end

    display(fig)
    savepath === nothing || save(savepath, fig; px_per_unit = 2)
    return fig
end
