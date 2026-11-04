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

function plot_flux_spectrum(hN; nmax = 2.0, npoints = 150, area_ratio = 0.5, nev = 40,
        solver = ES.ShiftInvert(ES.ArnoldiMethod(nev = 4 + nev), 0), savepath = nothing, params...)
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

# ---------------------------------------------------------------------
# Free energy, CPR and Fourier-harmonic figures
# ---------------------------------------------------------------------
# Ground state F₀ (even) and F₀ + εt (odd: one quasiparticle in the level that is the lowest positive one at φsel,
# followed through crossings by spectrum_F0_tracked), not the naive F₀ + min|ε| at each ϕ
function detailed_fourier_analysis(hN;
        npoints = 128, max_harmonics = 12, φsel = π/2,
        savepath = joinpath(@__DIR__, "figures/fourier_"), params...)

    N = npoints
    phases = 2π .* (0:N-1) ./ N
    parameter_values = (; params...)

    println("Calculating free energy across $(N) phase points...")
    println("System parameters:")
    show(stdout, MIME"text/plain"(), parameter_values)
    println("\nFourier settings: npoints = $npoints, max_harmonics = $max_harmonics, φsel = $φsel")
    F_vals, E_vals, Emin = spectrum_F0_tracked(hN, N, parameter_values; φsel)

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

# Ground state F₀ and first excited state F₁ = F₀ + εt (one quasiparticle in the tracked level), both from the same
# dense spectrum. Currents from the harmonics of each, converged separately by grid doubling. Single case (Δn, Vg, …), no normalization
function plot_free_energy_current(hN, case; n0 = 0, cpr_settings, φsel = π/2, nplot = 400,
        savepath = nothing, params...)
    (; N0, N_max, tol_alias, tol_tail) = cpr_settings
    phases = 2π .* (0:N_max-1) ./ N_max
    φplot = range(0, 2π, length = nplot)

    overrides = Base.structdiff(case, NamedTuple{(:Δn,)})
    p = (; params..., overrides..., n0, n = n0 + case.Δn)
    F_vals, εt, _ = spectrum_F0_tracked(hN, N_max, p; φsel)
    F1_vals = F_vals .+ εt
    h0 = converged_harmonics(F_vals, N0, tol_alias, tol_tail; name = "F₀")
    h1 = converged_harmonics(F1_vals, N0, tol_alias, tol_tail; name = "F₁")
    I_vals = [cpr_value(φ, h0.s, h0.c) for φ in φplot]
    I1_vals = [cpr_value(φ, h1.s, h1.c) for φ in φplot]

    common = (; xlabelsize = 16, ylabelsize = 16, xgridvisible = false)
    fig = Figure(size = (600, 600))
    ax1 = Axis(fig[1, 1]; ylabel = L"F - F_\text{min} \text{ [meV]}", common...)
    ax2 = Axis(fig[2, 1];
        xlabel = L"\varphi", ylabel = L"I \text{ [meV]}",
        xticks = (0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]), common...)

    # F on the grid (closed at ϕ = 2π), I from the harmonics; ground state solid, first excited state dashed
    closed(v) = [v; first(v)]
    color = Makie.wong_colors()[1]
    Fmin = minimum(F_vals)
    lines!(ax1, [phases; 2π], closed(F_vals .- Fmin); color, linewidth = 2)
    lines!(ax1, [phases; 2π], closed(F1_vals .- Fmin); color, linewidth = 2, linestyle = :dash)
    lines!(ax2, φplot, I_vals; color, linewidth = 2)
    lines!(ax2, φplot, I1_vals; color, linewidth = 2, linestyle = :dash)

    hlines!(ax2, 0; color = :black, linewidth = 0.5)
    linkxaxes!(ax1, ax2)
    xlims!(ax2, 0, 2π)
    hidexdecorations!(ax1, grid = false)
    Legend(fig[3, 1], [LineElement(; color = :black, linewidth = 2), LineElement(; color = :black, linewidth = 2, linestyle = :dash)],
        ["Even fermion parity", "Odd fermion parity"]; orientation = :horizontal, labelsize = 11, framevisible = false)
    rowgap!(fig.layout, 5)
    display(fig)
    savepath === nothing || save(savepath, fig)
    return (; fig, phases, φplot, F_vals, F1_vals, I_vals, I1_vals, h0, h1)
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

# Shapiro grid, µN rows × Δn columns (rows / columns can instead be named by strings, e.g. ["even", "odd"]):
#   dV/dI maps of the up (switching) sweep, i_dc horizontal, i_ac vertical
#   hysteresis maps |V_up − V_down| (only if any case is hysteretic)
#   line cuts V/(hf/2e) = ⟨v⟩/Ω against i_dc at iac_cut, all µN of a Δn in one panel (opacity cut_alpha, so overlaps show)
#   step histogram of each cut: i_dc width per ⟨v⟩/Ω bin, so plateaus are peaks
# units = :normalized → currents in units of each case's I_c
# units = :physical   → currents in µA (i × I_c) on shared axes (one sample, one microwave power); iac_cut in µA
function plot_shapiro_grid(results, µs, Δns; iac_cut = 1.0, units = :normalized, colorrange = (0, 2),
        colormap = :Blues, binwidth = 0.02, cut_alpha = 0.7, savepath = nothing)
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
    rowlabel(µ) = µ isa AbstractString ? µ : @sprintf("\$\\mu_N = %s\$ meV", latex_num(µ))   # rows: µN values or names
    collabel(Δn) = Δn isa AbstractString ? Δn : @sprintf("\$\\Delta n = %g\$", Δn)        # columns: Δn values or names

    fig = with_theme(theme_latexfonts(); fontsize = 15,
            Axis = (; xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true,
                xtickalign = 1, ytickalign = 1, xminorticksvisible = true, yminorticksvisible = true, xminortickalign = 1,
                yminortickalign = 1, titlefont = :regular, titlesize = 15)) do
        fig = Figure(size = (400 * nc + 110, 300 * nmap + 560), figure_padding = (10, 15, 10, 10))
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
        hist_axes = Axis[]
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
                lines!(axc, idcs, w_up(k); color = (colors[i], hysteretic ? 0.5 : cut_alpha), linewidth = 2.5)
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
            append!(labels, [tex(rowlabel(µ)) for µ in µs])
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
