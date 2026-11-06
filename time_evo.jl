include("setup.jl")
using SparseArrays, LinearAlgebra

# Time evolution of one BdG quasiparticle under a phase sweep φ(t) = ω t, ω = 2eV/ħ.
# Units: meV with ħ = 1, so time is in ħ/meV ≈ 0.658 ps and V in mV gives ω = 2V.
# Slow sweeps follow the anticrossings adiabatically (2π), fast ones jump them (Landau–Zener, 4π).

# Crank–Nicolson with the Hamiltonian at the midpoint of each step; dt is capped so that
# dt·|ε| ≪ 1 for the low levels and the phase moves by at most dφmax per step.
# The occupations of the instantaneous levels are recorded nrec times per 2π.
function evolve(hφ, ψ0, ω, solver; params = (;), nper = 2, dtmax = 0.5, dφmax = 2π / 2000, nrec = 400)
    dt = min(dtmax, dφmax / ω)
    nsteps = ceil(Int, nper * 2π / (ω * dt))
    every = max(1, nsteps ÷ (nper * nrec))
    ψ = copy(ψ0)
    φs, εs, occ, fid = Float64[], Vector{Float64}[], Vector{Float64}[], Float64[]
    for s in 0:nsteps-1
        H = hφ((); params..., ϕ = ω * (s + 0.5) * dt)
        ψ = lu(I + (im * dt / 2) * H) \ ((I - (im * dt / 2) * H) * ψ)
        if s % every == 0 || s == nsteps - 1
            φ = ω * (s + 1) * dt
            sp = spectrum(hφ(; params..., ϕ = φ); solver)
            push!(φs, φ)
            push!(εs, real(energies(sp)))
            push!(occ, abs2.(states(sp)' * ψ))
            push!(fid, abs2(ψ0' * ψ))
        end
    end
    return (; φs, εs, occ, fid, dt, nsteps)
end

# Minima of the gaps between consecutive positive levels (k, k+1), located on a coarse grid in φ
# and refined by golden section with dense diagonalization. Exact (symmetry protected) crossings
# go down to machine precision, avoided ones saturate at their gap δE. ΔE′ is the slope difference
# of the diabatic lines and ω_c = π δE² / (2|ΔE′|) the Landau–Zener crossover: P_jump = exp(-ω_c/ω).
function crossing_gaps(hφ, solver; params = (;), l = 400, nlev = 6, gmax = 0.05, φtol = 1e-13)
    φs = range(0, 2π, length = l + 1)[1:l]
    dφ = step(φs)
    ε = [sort(filter(>(0), real(energies(spectrum(hφ(; params..., ϕ = φ); solver)))))[1:nlev] for φ in φs]
    emax = 2maximum(e -> e[end], ε)
    levels(φ) = eigvals(Hermitian(Matrix(hφ((); params..., ϕ = φ))), 0.0, emax)
    out = NamedTuple[]
    for k in 1:nlev-1
        g = [e[k+1] - e[k] for e in ε]
        for i in 1:l
            gm, g0, gp = g[mod1(i - 1, l)], g[i], g[mod1(i + 1, l)]
            (g0 < gm && g0 <= gp && g0 < gmax) || continue
            gap(φ) = (e = levels(φ); e[k+1] - e[k])
            a, b = φs[i] - dφ, φs[i] + dφ
            ρ = (√5 - 1) / 2
            c, d = b - ρ * (b - a), a + ρ * (b - a)
            gc, gd = gap(c), gap(d)
            while b - a > φtol
                if gc < gd
                    b, d, gd = d, c, gc
                    c = b - ρ * (b - a); gc = gap(c)
                else
                    a, c, gc = c, d, gd
                    d = a + ρ * (b - a); gd = gap(d)
                end
            end
            φ0 = (a + b) / 2
            δE = gap(φ0)
            # slopes from points m steps away, beyond the gap, where g² = δE² + (ΔE′ x)²
            m = 4
            ΔE′ = sum(sqrt(max(g[mod1(i + s * m, l)]^2 - δE^2, 0)) for s in (-1, 1)) / (2m * dφ)
            push!(out, (; k, φ = mod(φ0, 2π), ε = levels(φ0)[k], δE, ωc = π * δE^2 / (2ΔE′)))
        end
    end
    return sort(out, by = x -> x.φ)
end

# return probability |⟨ψ0|ψ(φ)⟩|² at the recorded point closest to φ
fidelity_at(r, φ) = r.fid[argmin(abs.(r.φs .- φ))]

function plot_evolution(r, ω)
    (; φs, εs, occ, fid) = r
    πticks(n) = (π .* (0:n), [k == 0 ? "0" : k == 1 ? "π" : "$(k)π" for k in 0:n])
    nπ = ceil(Int, φs[end] / π)
    fig = Figure(size = (1200, 520))
    Label(fig[0, 1:3], "ω = $ω"; fontsize = 18, font = :bold)

    # instantaneous spectrum, colored by the occupation of each level
    ax = Axis(fig[1, 1]; xlabel = L"\phi", ylabel = L"\varepsilon", xticks = πticks(nπ))
    x = reduce(vcat, (fill(φ, length(e)) for (φ, e) in zip(φs, εs)))
    w = reduce(vcat, occ)
    o = sortperm(w)  # occupied levels drawn on top
    sc = scatter!(ax, x[o], reduce(vcat, εs)[o]; color = w[o], colormap = Reverse(:grays), colorrange = (0, 1), markersize = 3)
    ylims!(ax, -0.5, 0.5)
    xlims!(ax, 0, φs[end])
    Colorbar(fig[1, 2], sc; label = "occupation")

    # return probability and weight left in the solver window (leakage)
    ax2 = Axis(fig[1, 3]; xlabel = L"\phi", xticks = πticks(nπ))
    lines!(ax2, φs, fid; linewidth = 2, label = L"|\langle\psi_0|\psi\rangle|^2")
    lines!(ax2, φs, sum.(occ); linewidth = 2, linestyle = :dash, label = "weight in window")
    vlines!(ax2, 2π .* (1:nπ÷2); color = :gray40, linestyle = :dash, linewidth = 1)
    ylims!(ax2, 0, 1.05)
    xlims!(ax2, 0, φs[end])
    axislegend(ax2; position = :lb)
    colsize!(fig.layout, 1, Relative(0.5))
    display(fig)
end

params = merge(base_params, (;R = 600, N = 500, µ = 0.11, ϕ = 0, n = 2, Δ = 0.2, Vg = 2.1, mult=1))

nev = 40
solver = ES.ShiftInvert(ES.ArnoldiMethod(nev = nev), 0)
(; hN2, ϵ1, t) = build(; params...);

# start in the lowest positive level at φ = 0, as in soliton_branches
sp0 = spectrum(hN2(; params..., ϕ = 0.0); solver)
ε0 = real(energies(sp0))
pos = findall(>(0), ε0)
ψ0 = states(sp0)[:, pos[argmin(ε0[pos])]]

# (anti)crossings of the low positive levels: the 4π swap needs the ones the tracked level jumps
# to be exact (δE at machine precision) or to have ω_c ≪ ω, and the ones it follows ω_c ≫ ω
gaps = crossing_gaps(hN2, solver; params)
for c in gaps
    @printf("levels %d-%d  φ = %.4fπ  ε = %.4f  δE = %.2e  ω_c = %.2e%s\n",
        c.k, c.k + 1, c.φ / π, c.ε, c.δE, c.ωc, c.δE < 1e-9 ? "  exact" : "")
end

# scan through the Landau–Zener crossover ω_c ≈ π δE² / (2|ΔE′|) of the anticrossings
ωs = [2e-3, 1e-4]
for ω in ωs
    r = evolve(hN2, ψ0, ω, solver; params, nper = 2)
    println("ω = $ω: $(r.nsteps) steps of dt = $(round(r.dt, sigdigits = 3)), ",
        "P(2π) = $(round(fidelity_at(r, 2π), digits = 3)), P(4π) = $(round(fidelity_at(r, 4π), digits = 3)), ",
        "window weight = $(round(sum(r.occ[end]), digits = 3))")
    plot_evolution(r, ω)
end
