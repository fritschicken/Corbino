include("setup.jl")
include("rcsj.jl")

# Tracks one level along φ (repeating φs `loops` times).
# diabatic = false: adiabatic tracking (max overlap with the previous eigenstate), stays on its branch at avoided crossings.
# diabatic = true:  passes through avoided crossings. Each step the previous state is projected onto all levels within
#                   gapmax of its predicted (linearly extrapolated) energy, which keeps its character through crossings
#                   with gaps ≲ gapmax; its energy is Σ|cᵢ|²εᵢ / Σ|cᵢ|². Only applied while |ε| ≤ εmax: keep εmax
#                   below the quasi-continuum, where every level has narrow anticrossings. `mixed[step]` marks steps
#                   where more than one level carries a fraction ≥ tol of the tracked state.
function soliton_branches(hφ, φs, solver; params = (;), loops = 4, seed = nothing, edge = 4,
                          diabatic = false, tol = 0.05, gapmax = Inf, εmax = Inf)
    sps = [spectrum(hφ(; params..., ϕ = φ); solver) for φ in φs]
    εk = [real(energies(sp)) for sp in sps]
    ψk = [states(sp) for sp in sps]
    l = length(φs)

    ε  = εk[1]
    pos = findall(>(0), ε)
    i  = isnothing(seed) ? pos[argmin(ε[pos])] : pos[argmin(ε[pos])+seed]
    v  = ψk[1][:, i]   # reference state

    n = loops * l
    εs = zeros(n)
    ψs = zeros(ComplexF64, length(v), n)
    mixed = falses(n)
    for step in 1:n
        k = mod1(step, l)
        ε, ψ = εk[k], ψk[k]
        c  = ψ' * v
        ov = abs2.(c)
        j  = argmax(ov)
        (ε[j] ≥ partialsort(ε, length(ε) - edge) || ε[j] ≤ partialsort(ε, edge + 1)) &&
            @warn "tracked level ε = $(ε[j]) at the edge of the solver window at φ = $(φs[k]): increase nev"
        if diabatic && step > 2
            εp = 2εs[step-1] - εs[step-2]                       # predicted diabatic energy
            S = union(findall(e -> abs(e - εp) ≤ gapmax, ε), j)  # levels it can mix with
            if length(S) > 1 && abs(εp) ≤ εmax
                # transport the state by projecting it onto those levels (keeps its character through crossings)
                v = normalize(ψ[:, S] * c[S])
                εs[step] = sum(ov[S] .* ε[S]) / sum(ov[S])
                ψs[:, step] = v
                mixed[step] = count(≥(tol * sum(ov[S])), ov[S]) > 1
                continue
            end
        end
        ov[j] < 0.5 && @warn "weak overlap $(ov[j]) at φ = $(φs[k]): refine the grid here"
        v = ψ[:, j]
        ψs[:, step] = v
        εs[step]  = ε[j]
    end
    P = l * (φs[2] - φs[1])                       # span of the (uniform, endpoint-excluded) φ grid
    φall = vcat((φs .+ P * m for m in 0:loops-1)...)
    return φall, εs, ψs, εk, mixed
end

function lowest_positive_subband(b, φ)
    εmin(x) = minimum(filter(>(0), real.(Quantica.energy.(Quantica.vertices(x)))); init = Inf)
    s = b[(φ, :)]
    return argmin(k -> εmin(s[k]), eachindex(s))
end

params = merge(base_params, (; R = 700, N = 500, µ = 0.3,n=1 ,Δ = 0.2, Vg = 2.0, mult = 2, λg = Inf))


l = 400
nev = 50
solver = ES.ShiftInvert(ES.ArnoldiMethod(nev = nev), 0)
(; hN2, ϵ1, t) = build(; params...);
phi0 = 0.3
phis = range(0, 4π, length = l+1)[1:l] .+ 2π/l   # half-step shift, l multiple of 4: φ_j = (2j+1)·2π/l, never a multiple of π
b = bands(hN2, phis; mapping = φ -> ftuple(;params..., ϕ = φ), solver=solver)
jred = lowest_positive_subband(b, phi0)
jblue = lowest_positive_subband(b, 2π/params.mult + phi0)


plot_bands_density(b, [jred, jblue], [:red,:blue], params.mult;elim=0.5)   # plots.jl
#detailed_fourier_analysis(hN2; npoints = 64, params...)

#=
case = (; Δn = 1)                   # Vg: gate voltage [meV]
cpr_settings = (; N0 = 8, N_max = 128, tol_alias = 1e-3, tol_tail = 1e-3)
plot_free_energy_current(hN2, case; n0 = 0, cpr_settings,
    savepath = figpath("free_energy_current.png"), params...)
=#
#=
adiabatic path
_, ea, _, εk = soliton_branches(hN2, phis, solver; params, loops = 1)
phiall, ed, ψs, _, mixed = soliton_branches(hN2, phis, solver; params, loops = 1,
                                            diabatic = true, gapmax = 0.08, εmax = 0.1)
fig = Figure(); ax = Axis(fig[1, 1]; xlabel = "φ", ylabel = "ε")
for k in eachindex(phis)
    scatter!(ax, fill(phis[k], length(εk[k])), εk[k]; color = :gray60, markersize = 3)
end
lines!(ax, phiall, ea; color = (:red, 0.4), linewidth = 5)   # adiabatic
lines!(ax, phiall, ed; color = :blue)                         # diabatic
scatter!(ax, phiall[mixed], ed[mixed]; color = :blue, markersize = 4)
display(fig)
=#

shapiro_setup = 1
(; scaling, βc, iac_cut, drive, bias) = shapiro_setups[shapiro_setup]
case = (; Δn = params.n)
drive = (; Ω = 0.2, idcs = range(-2, 2, 401), iacs = range(0, 3, 301))
bias = (; nper_skip_min = 5, nper_avg = 30)
cpr_settings = (; N0 = 16, N_max = 256, tol_cpr = 1e-3, nev = 40)
shapiro_cpr  = (; cpr_settings..., Ic_tol = 1e-5)
n0 = 0
p_shapiro = (; params..., Base.structdiff(case, NamedTuple{(:Δn,)})..., n0, n = n0 + case.Δn)
shapiro_label = @sprintf("Δn = %g, Vg = %g meV", case.Δn, p_shapiro.Vg)
println("RCSJ Shapiro maps, even and odd parity, for ", shapiro_label, " ...")
shapiro_results = [(; case, r...) for r in
    shapiro_parity_cases(hN2, p_shapiro; cpr_settings = shapiro_cpr, φsel = π/2, drive, bias, βc, scaling, device,
        label = shapiro_label)]
rows = ["Even parity", "Odd parity"]
cols = [params.mult == 1 ? "\$V_g \\propto \\cos\\theta\$" : "\$V_g \\propto \\cos($(params.mult)\\theta)\$"]
run_id = save_shapiro_run(shapiro_results, rows, cols; label = "test_setup$(shapiro_setup)_lowOmega",
    meta = (; params, case, setup = shapiro_setup, scaling, drive, bias, βc, cpr_settings = shapiro_cpr, device))
plot_shapiro_grid(shapiro_results, rows, cols; iac_cut,
    units = scaling == :physical ? :physical : :normalized,
    savepath = figpath("shapiro_parity_$(run_id).png"))
