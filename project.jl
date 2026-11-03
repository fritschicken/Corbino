# Physics and computation: free energy, Fourier/CPR harmonics, Shapiro cases.
# Packages and code.jl are loaded by setup.jl; this file loads the dynamics, plotting and run-storage files.
includet("./rcsj.jl")            # RSJ/RCSJ dynamics
includet("./plots_cpr.jl")       # F(ϕ), CPR and harmonic figures
includet("./plots_spectrum.jl")  # BdG spectrum against flux
includet("./plots_shapiro.jl")   # Shapiro grid figure
includet("./shapiro_runs.jl")    # saving / loading Shapiro runs

function free_energy(h_inst, solver)
    sp = spectrum(h_inst; solver = solver)
    return 0.5 * sum(filter(<(0), real(energies(sp))))
end

# Free energy and the lowest |ε| of the same spectrum (smallest Andreev level at this φ)
function free_energy_gap(h_inst, solver)
    ε = real(energies(spectrum(h_inst; solver = solver)))
    return 0.5 * sum(filter(<(0), ε)), minimum(abs, ε)
end

# F(ϕ) on the periodic grid ϕ_j = 2πj/N: F(ϕ) = A0 + ∑ [ A_k cos(kϕ) + B_k sin(kϕ) ] and the current
# I(ϕ) = 2 ∂F/∂ϕ = ∑ [ I_k^(s) sin(kϕ) + I_k^(c) cos(kϕ) ] (ħ = e = 1), harmonics from current_harmonics,
# so I_k^(s) = -2k A_k and I_k^(c) = 2k B_k
function fourier(F_vals, phases, max_harmonics, N)
    max_harmonics ≤ (N - 1) ÷ 2 || throw(ArgumentError(
        "max_harmonics = $max_harmonics needs at least $(2max_harmonics + 1) phase points (have N = $N)"))
    A0 = sum(F_vals) / N
    I_sin, I_cos = current_harmonics(F_vals)
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

# Full spectrum on φ_j = 2πj/N from one dense `bands`: ground-state F₀ = ½ Σ_{ε<0} ε and the level εt of the
# subband that is the lowest positive one at φ = φsel, followed through crossings by the band connectivity (as in test.jl),
# and the smallest Andreev level min_φ min|ε| [meV]
function spectrum_F0_tracked(hN, N, case; φsel = π/2)
    phases = 2π .* (0:N-1) ./ N
    b = bands(hN, phases; mapping = φ -> ftuple(; case..., ϕ = φ), solver = ES.LinearAlgebra(), showprogress = true)
    gridindex(v) = round(Int, Quantica.base_coordinates(v)[1] * N / 2π) + 1
    F0 = zeros(N)
    Emin = Inf
    for sb in Quantica.subbands(b), v in Quantica.vertices(sb)
        ε = real(Quantica.energy(v))
        ε < 0 && (F0[gridindex(v)] += 0.5 * ε * Quantica.degeneracy(v))
        Emin = min(Emin, abs(ε))
    end
    s = b[(φsel, :)]
    ε = [isempty(Quantica.vertices(x)) ? Inf : real(Quantica.energy(Quantica.vertices(x)[1])) for x in s]
    j = argmin(k -> ε[k] > 0 ? ε[k] : Inf, eachindex(ε))
    vs = Quantica.vertices(Quantica.subbands(b)[j])
    js = gridindex.(vs)
    sort(js) == 1:N || error("tracked subband does not have one level per φ (levels merged at a degenerate point): try another φsel")
    εt = zeros(N)
    εt[js] = [real(Quantica.energy(v)) for v in vs]
    return F0, εt, Emin
end

# Harmonics converged by doubling N = N0, 2N0, … ≤ N_max until harmonics_converged, then truncated.
# F_on(N) returns F on φ_j = 2πj/N; it is called with N0, 2N0, … in this order, so it may compute the new points only
function doubling_harmonics(F_on, N0, N_max, tol_alias, tol_tail; name = "F")
    N0 ≥ 3 || throw(ArgumentError("N0 must be ≥ 3 to resolve at least one harmonic"))
    N = N0
    (s, c) = current_harmonics(F_on(N))
    Ic_est = cpr_max(s, c)
    converged = false
    while 2N <= N_max
        (s_old, c_old) = (s, c)
        N *= 2
        (s, c) = current_harmonics(F_on(N))
        Ic_est = cpr_max(s, c)
        if harmonics_converged(s, c, s_old, c_old, Ic_est, tol_alias, tol_tail)
            converged = true
            break
        end
    end
    converged || @warn @sprintf("%s harmonics not converged at N = %d: increase N_max (or loosen tol_alias / tol_tail)", name, N)
    (s, c) = truncate_harmonics(s, c, Ic_est, tol_tail)
    return (; s, c, N)
end

# Current harmonics of F given on φ_j = 2πj/Nmax, converged on the nested subgrids N = N0, 2N0, …
function converged_harmonics(F, N0, tol_alias, tol_tail; name = "F")
    Nmax = length(F)
    Nmax % N0 == 0 && ispow2(Nmax ÷ N0) || throw(ArgumentError("need N_max = N0 × 2^m"))
    return doubling_harmonics(N -> F[1:Nmax÷N:end], N0, Nmax, tol_alias, tol_tail; name)
end

# ---------------------------------------------------------
# Shapiro steps: CPR of the ring → RSJ/RCSJ dynamics (rcsj.jl), T = 0
# ---------------------------------------------------------

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

# Returns the current harmonics (code units), F(φ) and the smallest Andreev level min_φ min|ε| [meV];
# each doubling computes only the new midpoints (refine_F)
function converged_cpr(hN, params, N0, N_max, tol_alias, tol_tail; showprogress = true)
    F, E = sample_F(hN, N0, params; showprogress)
    function F_on(N)
        N > length(F) && ((F, E) = refine_F(hN, F, E, params; showprogress))
        return F
    end
    h = doubling_harmonics(F_on, N0, N_max, tol_alias, tol_tail; name = "CPR")
    return h.s, h.c, F, minimum(E)
end

# Doubling stopped: the old harmonics no longer change and the newly resolved ones are negligible
function harmonics_converged(s, c, s_old, c_old, Ic_est, tol_alias, tol_tail)
    kold = length(s_old)
    alias_change = maximum(k -> abs(s[k] - s_old[k]) + abs(c[k] - c_old[k]), 1:kold)
    tail = maximum(k -> hypot(s[k], c[k]), kold+1:length(s))
    return alias_change < tol_alias * Ic_est && tail < tol_tail * Ic_est
end

# Drop the trailing harmonics below tol_tail × Ic_est
function truncate_harmonics(s, c, Ic_est, tol_tail)
    kmax = something(findlast(≥(tol_tail * Ic_est), hypot.(s, c)), 1)
    return s[1:kmax], c[1:kmax]
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
function shapiro_case(hN, p, solver; cpr_settings, kw...)
    (; N0, N_max, tol_alias, tol_tail, Ic_tol) = cpr_settings
    s, c, F, Emin = converged_cpr(hN, (p, solver), N0, N_max, tol_alias, tol_tail)
    return shapiro_from_cpr(s, c, F, Emin; Ic_tol, kw...)
end

# Even (ground state F₀) and odd (F₀ + εt, one quasiparticle in the tracked level) parity of one microscopic case p,
# both CPRs from the same dense spectrum (spectrum_F0_tracked), then a Shapiro map for each. Returns [even, odd]
function shapiro_parity_cases(hN, p; cpr_settings, φsel = π/2, label = "", kw...)
    (; N0, N_max, tol_alias, tol_tail, Ic_tol) = cpr_settings
    F0, εt, Emin = spectrum_F0_tracked(hN, N_max, p; φsel)
    return map((("even", F0), ("odd", F0 .+ εt))) do (parity, F)
        h = converged_harmonics(F, N0, tol_alias, tol_tail; name = "F_" * parity)
        r = shapiro_from_cpr(h.s, h.c, F, Emin; Ic_tol, label = strip(label * " " * parity), kw...)
        (; parity, r...)
    end |> collect
end

# Shapiro map from current harmonics s, c (code units); F and Emin are only stored / used for the adiabaticity check
function shapiro_from_cpr(s, c, F, Emin; drive, bias, βc = 0.0, scaling = :fixed, device = nothing,
        adiabatic_ratio = 0.5, label = "", Ic_tol = 1e-6)
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
