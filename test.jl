include("setup.jl")
function soliton_branches(hφ, φs, solver; params = (;), loops = 4, seed = nothing, edge = 4)
    sps = [spectrum(hφ(; params..., ϕ = φ); solver) for φ in φs]
    εk = [real(energies(sp)) for sp in sps]
    ψk = [states(sp) for sp in sps]
    l = length(φs)

    ε  = εk[1]
    pos = findall(>(0), ε)
    i  = isnothing(seed) ? pos[argmin(ε[pos])] : pos[argmin(ε[pos])+seed]
    v  = ψk[1][:, i]

    n = loops * l
    εs = zeros(n)
    ψs = zeros(ComplexF64, length(v), n)
    for step in 1:n
        k = mod1(step, l)
        ε, ψ = εk[k], ψk[k]
        ov = abs2.(ψ' * v)
        j  = argmax(ov)
        ov[j] < 0.5 && @warn "weak overlap $(ov[j]) at φ = $(φs[k]): refine the grid here"
        (ε[j] ≥ partialsort(ε, length(ε) - edge) || ε[j] ≤ partialsort(ε, edge + 1)) &&
            @warn "tracked level ε = $(ε[j]) at the edge of the solver window at φ = $(φs[k]): increase nev"
        v = ψ[:, j]
        #tracked levels
        ψs[:, step] = v
        εs[step]  = ε[j]

    end
    φall = vcat((φs .+ 2π * m for m in 0:loops-1)...)
    return φall, εs, ψs, εk
end

# |ψ|² per site along subband j of b, read from its band vertices (summed over the degenerate states), sorted by φ
function subband_density(b, j)
    vs = Quantica.vertices(Quantica.subbands(b)[j])
    φ = [Quantica.base_coordinates(v)[1] for v in vs]
    ρ = reduce(hcat, [vec(sum(site_density(Quantica.states(v)), dims = 2)) / Quantica.degeneracy(v) for v in vs])
    p = sortperm(φ)
    return φ[p], ρ[:, p]
end

function plotting(b, js, colors, mult)
    πticks(n) = (π .* (0:n), [k == 0 ? "0" : k == 1 ? "π" : "$(k)π" for k in 0:n])
    fig = Figure(size = (1200, 520))
    Label(fig[0, 1:2+length(js)], rich("V", subscript("g"), " ~ cos(",(mult==1 ? "" : string(mult)),"φ)"); fontsize = 18, font = :bold)

    ax = Axis(fig[1, 1]; xlabel = L"\phi", ylabel = L"\varepsilon", xticks = πticks(2))
    plotbands!(ax, b; color = :gray75, hide = :nodes)
    for (j, c) in zip(js, colors)
        plotbands!(ax, b[[j]]; color = c, size = 3, hide = :nodes)
    end
    lim = 0.5
    ylims!(ax, -lim, lim)
    xlims!(ax, 0, 2π)

    # states of the colored subbands along φ, overlaid (colormaps fade to transparent at zero density)
    ax2 = Axis(fig[1, 2]; xlabel = L"\phi", ylabel = "site", title = L"|\psi|^2", xticks = πticks(2))
    for (i, (j, c)) in enumerate(zip(js, colors))
        φ, ρ = subband_density(b, j)
        hm = heatmap!(ax2, φ, 1:size(ρ, 1), permutedims(ρ); colormap = [(c, 0.0), (c, 1.0)])
        Colorbar(fig[1, 2+i], hm)
    end
    xlims!(ax2, 0, 2π)
    colsize!(fig.layout, 1, Relative(0.4))
    display(fig)
end

site_density(ψs) = dropdims(sum(reshape(abs2.(ψs), 4, :, size(ψs, 2)), dims = 1), dims = 1)

function plot_ring(h, r, l, loops)
    (; es, ψs) = r
    steps = [(k - 1) * l + 1 + c * (l ÷ 8) for k in 1:loops, c in 0:3]
    ρmax = maximum(site_density(ψs[:, vec(steps)]))
    θ = range(0, 2π, length = 400)
    fig = Figure(size = (250 * 4, 270 * loops + 60))
    for k in 1:loops, c in 0:3
        s = steps[k, c + 1]
        w = abs2.(reshape(ψs[:, s], 4, :))  # orbitals (e↑, e↓, h↑, h↓) × sites
        ρe, ρh = vec(sum(w[1:2, :], dims = 1)), vec(sum(w[3:4, :], dims = 1))
        ax = Axis(fig[k, c + 1]; aspect = DataAspect(), title = "φ = $(4(k - 1) + c)π/2,  ε = $(round(es[s], digits = 4))")
        hidedecorations!(ax); hidespines!(ax)
        lines!(ax, params.R .* cos.(θ), params.R .* sin.(θ); color = :gray85)
        plotlattice!(ax, h; hide = :hops, siteoutline = 0,
            siteradius = ranged(ρe .+ ρh, (0, ρmax)), minmaxsiteradius = (0, 6),
            sitecolor = ranged(ρe .- ρh, (-ρmax, ρmax)), sitecolormap = :balance)
    end
    Colorbar(fig[loops + 1, :]; limits = (-ρmax, ρmax), colormap = :balance, vertical = false, flipaxis = false,
        label = "electron − hole weight per site", width = Relative(0.4))
    display(fig)
end

mult = 1

params = merge(base_params, (;R = 700, N = 500, µ = 0.11, ϕ = 0, n = 2, Δ = 0.2, Vg = 2.1, mult=mult))

l = 4000
nev = 40
solver = ES.ShiftInvert(ES.ArnoldiMethod(nev = nev), 0)
(; hN, ϵ1, t) = build(; params...);
phi0 = 0.1
phis = range(0, 4π, length = l+1)[1:l]
b = bands(hN, phis; mapping = φ -> ftuple(;params..., ϕ = φ), solver=solver)
fig = qplot(b;hide=:nodes, color=:black)

# index of the subband holding the lowest positive energy at φ (a subband may cross φ several times)
function lowest_positive_subband(b, φ)
    εmin(x) = minimum((e for e in (real(Quantica.energy(v)) for v in Quantica.vertices(x)) if e > 0); init = Inf)
    s = b[(φ, :)]
    return argmin(k -> εmin(s[k]), eachindex(s))
end
jred = lowest_positive_subband(b, phi0)
qplot!(b[[jred]];color=:red, hide=:nodes)
jblue = lowest_positive_subband(b, 2π + phi0)
qplot!(b[[jblue]];color=:blue, hide=:nodes)
display(fig)
#phiall, es, ψs, εk = soliton_branches(hN, phis, solver; params, loops = 5)
#E = reshape(es, l, :)
#nloops = size(E, 2)
#period = findfirst(m -> isapprox(E[:, 1], E[:, m+1]), 1:nloops-1)
#println("μ: $(params.μ), Vg: $(params.Vg)")
#if isnothing(period)
#    period = nloops
#    println("tracked state not periodic within $(2period)π, plotting all loops")
#else
#    println("tracked state is $(2period)π periodic")
#end
#plotting(b, [jred, jblue], [:red, :blue], mult)
#plot_ring(hN(; params...), (; es, ψs), l, period+1)