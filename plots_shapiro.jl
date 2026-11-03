# Shapiro grid figure (results of shapiro_case / shapiro_parity_cases). Loaded by project.jl; needs Makie and the functions of project.jl.

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
