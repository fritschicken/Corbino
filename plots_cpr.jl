# Free energy, CPR and Fourier-harmonic figures. Loaded by project.jl; needs Makie and the functions of project.jl.

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
