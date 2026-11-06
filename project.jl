# Physics and computation: free energy, tabulated CPR, Shapiro cases (Fourier harmonics only for detailed_fourier_analysis).
# Packages and code.jl are loaded by setup.jl; this file loads the dynamics, plotting and run-storage files.
includet("./rcsj.jl")            # RSJ/RCSJ dynamics
includet("./plots.jl")           # spectrum, F(ϕ)/CPR/harmonic and Shapiro grid figures
includet("./shapiro_runs.jl")    # saving / loading Shapiro runs

# h_inst is the spin-reduced hN2 (one of the two identical spin blocks, basis (c↑, c↓†)). The full BdG spectrum is
# hN2 ∪ -hN2 and Tr hN2 = 0, so F = ½ Σ_{ε<0, full} ε = Σ_{ε<0, hN2} ε (twice ½ Σ over hN2 alone)
function free_energy(h_inst, solver)
    sp = spectrum(h_inst; solver = solver)
    return sum(filter(<(0), real(energies(sp))))
end

# Free energy (as in free_energy) and the lowest |ε| of the same spectrum (smallest Andreev level at this φ)
function free_energy_gap(h_inst, solver)
    ε = real(energies(spectrum(h_inst; solver = solver)))
    return sum(filter(<(0), ε)), minimum(abs, ε)
end

# F(ϕ) on the periodic grid ϕ_j = phases[1] + 2πj/N: F(ϕ) = A0 + ∑ [ A_k cos(kϕ) + B_k sin(kϕ) ] and the current
# I(ϕ) = 2 ∂F/∂ϕ = ∑ [ I_k^(s) sin(kϕ) + I_k^(c) cos(kϕ) ] (ħ = e = 1), harmonics from current_harmonics,
# so I_k^(s) = -2k A_k and I_k^(c) = 2k B_k
function fourier(F_vals, phases, max_harmonics, N)
    max_harmonics ≤ (N - 1) ÷ 2 || throw(ArgumentError(
        "max_harmonics = $max_harmonics needs at least $(2max_harmonics + 1) phase points (have N = $N)"))
    A0 = sum(F_vals) / N
    I_sin, I_cos = current_harmonics(F_vals; φ0 = first(phases))
    I_sin, I_cos = I_sin[1:max_harmonics], I_cos[1:max_harmonics]
    A_k = -I_sin ./ (2 .* (1:max_harmonics))
    B_k = I_cos ./ (2 .* (1:max_harmonics))
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

# Phase grid of the tracked spectrum: N points per 2π over 2πν, shifted by half a step, φ_k = 2π(k + ½)/N.
# With N even no φ is a multiple of π, where levels of the two gate-separated halves of the ring are exactly
# degenerate (θ → -θ symmetry of hN2) and `bands` would merge them into one subband
tracked_phases(N, ν = 1) = 2π .* ((0:ν*N-1) .+ 1/2) ./ N

# Period of the odd-parity branch (one quasiparticle kept in the tracked level) in units of 2π, ν = 4π/mult / 2π:
# for a mult = 1 gate the tracked level swaps with its partner at φ = 2π, so the branch is 4π-periodic (ν = 2);
# otherwise it is treated as 2π-periodic (ν = 1). The even branch F₀ is always 2π-periodic.
odd_period_factor(p) = get(p, :mult, 1) == 1 ? 2 : 1

# On φ_j = 2π(j + ½)/N, j = 0 … νN-1 (tracked_phases; N even, N points per 2π, period 2πν):
#   F₀ = Σ_{ε<0} ε of hN2 (both spin blocks, see free_energy) and min|ε| from the lower half of the spectrum (eigvals with irange 1:n÷2, no eigenvectors,
#   one φ at a time), on [0, 2π) only and
#   repeated, since H is exactly 2π-periodic in ϕ;
#   εt, the level of the subband that is the lowest positive one at φ = φsel, followed through crossings by the band
#   connectivity of a shift-invert `bands` with the nev levels closest to ε = 0 (as in test.jl).
# A full-spectrum `bands` would store all eigenvectors (dim² per φ, GBs) and drops unconnected vertices (spikes in F₀).
# Returns F₀, εt and the smallest Andreev level min_φ min|ε| [meV]
function spectrum_F0_tracked(hN2, N, case; φsel = π/2, ν = 1, nev = 40)
    iseven(N) || throw(ArgumentError("N = $N must be even, so the shifted grid avoids the degenerate points φ = mπ"))
    Ntot = ν * N
    phases = tracked_phases(N, ν)
    F0 = zeros(N); Emins = zeros(N)
    prog = Progress(N; desc = "F₀ (lower half)      ")
    for k in 1:N
        H = Hermitian(Matrix(hN2(; case..., ϕ = phases[k])(())))
        ε = eigvals(H, 1:size(H, 1)÷2)
        F0[k] = sum(ε)
        Emins[k] = abs(ε[end])
        next!(prog)
    end
    b = bands(hN2, phases; mapping = φ -> ftuple(; case..., ϕ = φ), solver = ES.ShiftInvert(ES.ArnoldiMethod(; nev), 0.0),
              showprogress = true)
    gridindex(v) = round(Int, Quantica.base_coordinates(v)[1] * N / 2π - 1/2) + 1
    s = b[(φsel, :)]
    ε = [isempty(Quantica.vertices(x)) ? Inf : real(Quantica.energy(Quantica.vertices(x)[1])) for x in s]
    j = argmin(k -> ε[k] > 0 ? ε[k] : Inf, eachindex(ε))
    vs = Quantica.vertices(Quantica.subbands(b)[j])
    js = gridindex.(vs)
    sort(js) == 1:Ntot || error("tracked subband does not have one level per φ (levels merged at a degenerate point, " *
                                "or it left the nev = $nev window): try another φsel or a larger nev")
    εt = zeros(Ntot)
    εt[js] = [real(Quantica.energy(v)) for v in vs]
    return repeat(F0, ν), εt, minimum(Emins)
end

# ---------------------------------------------------------
# Shapiro steps: CPR of the ring → RSJ/RCSJ dynamics (rcsj.jl), T = 0
# ---------------------------------------------------------

# F(φ) and the lowest |ε| (smallest Andreev level) on the periodic grid φ = 2πj/N.
# Needs the full spectrum (ES.LinearAlgebra()), never shift-invert.
function sample_F(hN2, N, params; showprogress = true)
    case, solver = params
    prog = Progress(N; desc = @sprintf("CPR N = %-4d ", N), enabled = showprogress)
    FE = [(x = free_energy_gap(hN2(; case..., ϕ = 2π * j / N), solver); next!(prog); x) for j in 0:N-1]
    return first.(FE), last.(FE)
end

# N → 2N by inserting the midpoints
function refine_F(hN2, F_old, E_old, params; showprogress = true)
    N = length(F_old)
    case, solver = params
    prog = Progress(N; desc = @sprintf("CPR N = %-4d ", 2N), enabled = showprogress)
    FE = [(x = free_energy_gap(hN2(; case..., ϕ = 2π * (j + 1/2) / N), solver); next!(prog); x) for j in 0:N-1]
    interleave(old, new) = collect(Iterators.flatten(zip(old, new)))
    return interleave(F_old, first.(FE)), interleave(E_old, last.(FE))
end

# Current harmonics of F on the periodic grid φ_j = φ0 + 2πj/N (φ0 = π/N for tracked_phases)
function current_harmonics(F; φ0 = 0.0)
    N = length(F)
    kmax = div(N-1,2)
    I_sin = Vector{Float64}(undef, kmax)
    I_cos = Vector{Float64}(undef, kmax)
    for k in 1:kmax
        a = b = 0.0
        for j in 0:N-1
            s, c = sincos(2π * mod(k * j, N) / N + k * φ0)
            a += F[j+1] * c
            b += F[j+1] * s
        end
        I_sin[k] = -4k * a / N
        I_cos[k] =  4k * b / N
    end
    return I_sin, I_cos
end

# Tabulated CPR (rcsj.jl TabulatedCPR) from F on the periodic grid φ_j = φ0 + period·j/N, j = 0 … N-1
# (φ0 = 0 for sample_F, π/N for tracked_phases):
# I = 2∂F/∂φ by central differences at the midpoints φ_{j+1/2} (second order there), linearly interpolated.
# Kinks of F (jumps of I, e.g. at zero-energy crossings) stay sharp, where a truncated Fourier series rings.
function cpr_table(F, period = 2π; φ0 = 0.0)
    N = length(F)
    h = period / N
    return TabulatedCPR([2 * (F[mod1(j + 1, N)] - F[j]) / h for j in 1:N], period, φ0 + h / 2)
end

# Resolution change between a table and a coarser one: mean |I − I_old| over the points of t, relative to max|I|
cpr_change(t, t_old) =
    sum(i -> abs(t.I[i] - cpr_value(t.offset + (i - 1) * t.period / length(t.I), t_old)), eachindex(t.I)) /
    (length(t.I) * cpr_max(t))

# Table from all points of F (one period), checked against the table from every other point
function cpr_table_checked(F, period, tol; φ0 = 0.0, name = "CPR")
    iseven(length(F)) || throw(ArgumentError("need an even number of points"))
    t = cpr_table(F, period; φ0)
    change = cpr_change(t, cpr_table(F[1:2:end], period; φ0))
    change > tol && @warn @sprintf("%s table not converged: halving the grid changes I by %.2g of I_c on average (tol_cpr = %g): increase N_max",
        name, change, tol)
    return t
end

# Returns the tabulated CPR (code units), F(φ) and the smallest Andreev level min_φ min|ε| [meV]. The grid is doubled
# N = N0, 2N0, … ≤ N_max (each doubling computes only the new midpoints, refine_F) until the table changes by less
# than tol (mean |ΔI| / I_c, cpr_change)
function converged_cpr(hN2, params, N0, N_max, tol; showprogress = true)
    F, E = sample_F(hN2, N0, params; showprogress)
    t = cpr_table(F)
    converged = false
    while 2 * length(F) <= N_max
        F, E = refine_F(hN2, F, E, params; showprogress)
        t_old, t = t, cpr_table(F)
        if cpr_change(t, t_old) < tol
            converged = true
            break
        end
    end
    converged || @warn @sprintf("CPR table not converged at N = %d (tol_cpr = %g): increase N_max", length(F), tol)
    return t, F, minimum(E)
end

# One microscopic case p (all Hamiltonian parameters): converged CPR (computed once), then the Shapiro map.
#   scaling = :fixed     βc, drive.Ω, drive.idcs, drive.iacs are used as given (normalized units)
#   scaling = :physical  βc and Ω follow from this case's own I_c and device = (; R, C, f);
#                        drive = (; Idcs_uA, Iacs_uA) in µA, converted to this case's normalized i
# adiabatic_ratio: warn if hf > adiabatic_ratio × smallest Andreev level (equilibrium-CPR assumption)
# cpr_settings: N0, N_max, Ic_tol, and optionally tol_cpr (default 1e-3, see converged_cpr)
function shapiro_case(hN2, p, solver; cpr_settings, kw...)
    (; N0, N_max, Ic_tol) = cpr_settings
    t, F, Emin = converged_cpr(hN2, (p, solver), N0, N_max, get(cpr_settings, :tol_cpr, 1e-3))
    return shapiro_from_cpr(t, F, Emin; Ic_tol, kw...)
end

# Even (ground state F₀) and odd (F₀ + εt, one quasiparticle in the tracked level) parity of one microscopic case p,
# both CPRs from the same dense spectrum (spectrum_F0_tracked), then a Shapiro map for each. Returns [even, odd].
# The odd branch has period 2πν, ν = odd_period_factor(p) (4π for mult = 1); F₀ is 2π-periodic (first N_max points)
function shapiro_parity_cases(hN2, p; cpr_settings, φsel = π/2, label = "", kw...)
    (; N_max, Ic_tol) = cpr_settings
    tol = get(cpr_settings, :tol_cpr, 1e-3)
    ν = odd_period_factor(p)
    F0, εt, Emin = spectrum_F0_tracked(hN2, N_max, p; φsel, ν, nev = get(cpr_settings, :nev, 40))
    return map((("even", F0[1:N_max], 1), ("odd", F0 .+ εt, ν))) do (parity, F, νF)
        t = cpr_table_checked(F, 2π * νF, tol; φ0 = π / N_max, name = "F_" * parity)   # tracked_phases grid
        r = shapiro_from_cpr(t, F, Emin; Ic_tol, label = strip(label * " " * parity), kw...)
        (; parity, r...)
    end |> collect
end

# Shapiro map from a tabulated CPR t (code units, period 2π or 4π); F and Emin are only stored / used for the
# adiabaticity check
function shapiro_from_cpr(t::TabulatedCPR, F, Emin; drive, bias, βc = 0.0, scaling = :fixed, device = nothing,
        adiabatic_ratio = 0.5, label = "", Ic_tol = 1e-6)
    Ic = cpr_max(t)
    Ic > 0 || throw(ArgumentError("CPR is identically zero"))
    Ic < Ic_tol && @warn @sprintf("I_c = %.2g is numerical noise, the normalized CPR is meaningless", Ic)
    cpr = TabulatedCPR(t.I ./ Ic, t.period, t.offset)
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
    m = shapiro_map(cpr; Ω, βc, idcs, iacs, bias...)
    return (; F, cpr, Ic, Ic_A, Emin, noise = Ic < Ic_tol, scaling, m...)
end
