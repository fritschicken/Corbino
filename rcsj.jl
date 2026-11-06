using StaticArrays
using Printf
using ProgressMeter
import OrdinaryDiffEqTsit5 as ODE                 # Tsit5, AutoTsit5, ODEProblem, solve
import OrdinaryDiffEqRosenbrock: Rosenbrock23

# ---------------------------------------------------------
# Shapiro steps: RSJ / RCSJ junction driven by I_dc + I_ac sin(Ωτ), T = 0 (no noise term)
#   βc d²φ/dτ² + dφ/dτ + f(φ) = i_dc + i_ac sin(Ωτ),   f = I_s/I_c,   v = dφ/dτ
# in units τ = 2e I_c R t/ħ, v = V/(I_c R), Ω = ħω/(2e I_c R), so ⟨v⟩/Ω = V/(hf/2e).
# Needs only the normalized CPR (no Quantica), so it can be tested on its own. The CPR is a TabulatedCPR: samples on
# a uniform periodic grid (period 2π, or 4π for an odd-parity branch with a level crossing at 2π), linearly interpolated.
# A table reproduces kinks / jumps of the CPR at the grid resolution, where a truncated Fourier series would ring.
# ---------------------------------------------------------

# CPR samples I[j] at φ_j = offset + (j-1)·period/N, j = 1 … N, on one period; linear interpolation, periodic
struct TabulatedCPR
    I::Vector{Float64}
    period::Float64
    offset::Float64
end

# Table of an analytic CPR f(φ) (tests, model CPRs)
tabulate(f, period = 2π, N = 4096) = TabulatedCPR([f(period * j / N) for j in 0:N-1], period, 0.0)

function cpr_value(φ, t::TabulatedCPR)
    isfinite(φ) || return oftype(float(φ), NaN)         # diverged trial step: NaN lets the ODE solver reject it
    N = length(t.I)
    x = (φ - t.offset) * N / t.period          # position in grid units; no mod on φ, so dI/dφ stays exact under
    i = floor(Int, x)                          # automatic differentiation (stiff solvers); only the index is wrapped
    w = x - i
    return (1 - w) * t.I[mod1(i + 1, N)] + w * t.I[mod1(i + 2, N)]
end

cpr_max(t::TabulatedCPR) = maximum(abs, t.I)

# Junction at rest for i_dc = 0: global minimum of the potential U(φ) = ∫ f over the grid points (trapezoid rule)
function rest_state(t::TabulatedCPR)
    h = t.period / length(t.I)
    U = cumsum([0.0; h .* (t.I[1:end-1] .+ t.I[2:end]) ./ 2])
    return (t.offset + (argmin(U) - 1) * h, 0.0)
end

# ---------------------------------------------------------
# Physical units. Code currents are I = 2∂F/∂φ with F in meV and e = ħ = 1; the physical current is
# (2e/ħ)∂F/∂φ, so 1 code unit = (e/ħ)·(1 meV) = 243.4 nA.
# ---------------------------------------------------------
const e_SI, ħ_SI, h_SI = 1.602176634e-19, 1.054571817e-34, 6.62607015e-34
const Φ0_SI = h_SI / 2e_SI
const I_UNIT_A = (e_SI / ħ_SI) * (1e-3 * e_SI)

# RCSJ parameters of a device (R [Ω], C [F], drive f [Hz]) for a code-unit critical current Ic
function rcsj_units(Ic; R, C, f)
    Ic_A = Ic * I_UNIT_A
    ωc = 2π * Ic_A * R / Φ0_SI                   # characteristic frequency 2e Ic R / ħ
    ωp = sqrt(2π * Ic_A / (Φ0_SI * C))           # plasma frequency
    EJ = Φ0_SI * Ic_A / 2π                       # Josephson energy [J]
    return (; βc = 2π * Ic_A * R^2 * C / Φ0_SI, Ω = Φ0_SI * f / (Ic_A * R),
        Ic_A, IcR = Ic_A * R, fc = ωc / 2π, fp = ωp / 2π, EJ_K = EJ / 1.380649e-23, hf_meV = h_SI * f / e_SI * 1e3)
end

function print_rcsj_units(label, Ic; device...)
    u = rcsj_units(Ic; device...)
    println(@sprintf("%s: Ic = %.3g nA, IcR = %.3g µV, fc = %.3g GHz, fp = %.3g GHz, EJ = %.3g mK",
        label, u.Ic_A * 1e9, u.IcR * 1e6, u.fc * 1e-9, u.fp * 1e-9, u.EJ_K * 1e3))
    println(@sprintf("    → βc = %.3g, Ω = %.3g, hf = %.3g meV", u.βc, u.Ω, u.hf_meV))
    return u
end

# ---------------------------------------------------------
# Dynamics
# ---------------------------------------------------------
function rcsj(u, p, t)
    φ, v = u
    return SVector(v, (p.idc + p.iac * sin(p.Ω * t) - cpr_value(φ, p.cpr) - v) / p.βc)
end

rsj(φ, p, t) = p.idc + p.iac * sin(p.Ω * t) - cpr_value(φ, p.cpr)

# Explicit Tsit5 for βc ≥ βc_stiff; below, the v equation relaxes on the short time βc and AutoTsit5
# switches to Rosenbrock23. dtmax resolves βc and √βc on the explicit path, and the Josephson time 1
# on both paths, independently of the drive period (which is long for small Ω).
function rcsj_solver(βc; βc_stiff = 1.0)
    βc ≥ βc_stiff && return ODE.Tsit5(), 0.1 * min(βc, sqrt(βc), 1)
    return ODE.AutoTsit5(Rosenbrock23()), 0.1 * min(sqrt(βc), 1)
end

# Integrate an integer number of drive periods from τ = 0 (drive phase 0) starting at u_in = (φ, v);
# returns ⟨v⟩ over the last nper_avg periods and the final (φ, v). φ is wrapped (mod the CPR period) only here,
# between solves.
function bias_point(u_in, p; nper_skip_min = 20, nper_avg, reltol = 1e-8, abstol = 1e-8, βc_stiff = 1.0)
    T = 2π / p.Ω
    nper_skip = max(nper_skip_min, ceil(Int, 10 * p.βc * p.Ω / 2π))
    t_skip = nper_skip * T
    t_end = (nper_skip + nper_avg) * T
    φ0, v0 = u_in
    φwrap = mod(φ0, p.cpr.period)
    if iszero(p.βc)
        prob = ODE.ODEProblem(rsj, φwrap, (0.0, t_end), p)
        sol = ODE.solve(prob, ODE.Tsit5(); saveat = [t_skip, t_end], save_start = false, dtmax = 0.1,
            reltol, abstol, maxiters = 10^8)
        v_dc = (sol.u[end] - sol.u[1]) / (t_end - t_skip)
        return v_dc, (sol.u[end], rsj(sol.u[end], p, t_end))
    end
    alg, dtmax = rcsj_solver(p.βc; βc_stiff)
    prob = ODE.ODEProblem(rcsj, SVector(φwrap, v0), (0.0, t_end), p)
    sol = ODE.solve(prob, alg; saveat = [t_skip, t_end], save_start = false, dtmax,
        reltol, abstol, maxiters = 10^8)
    v_dc = (sol.u[end][1] - sol.u[1][1]) / (t_end - t_skip)
    return v_dc, (sol.u[end][1], sol.u[end][2])
end

# One i_ac column. Switching (V_up): outward from i_dc ≈ 0 to +max and to −max, each branch starting from
# rest. Retrapping (V_down): inward from ±max back to i_dc ≈ 0, continuing from the state reached at ±max.
# βc = 0 (first-order RSJ) has a unique ⟨v⟩ per bias, so there V_down = V_up.
function sweep_column(iac, cpr::TabulatedCPR; Ω, βc, idcs, u_rest = rest_state(cpr), bias...)
    n = length(idcs)
    i0 = argmin(abs.(idcs))
    V_up = zeros(n)
    V_down = zeros(n)
    point(u, i) = bias_point(u, (; idc = idcs[i], iac, Ω, βc, cpr); bias...)

    u = u_rest
    for i in i0:n
        V_up[i], u = point(u, i)
    end
    u_top = u
    u_bottom = u_rest
    if i0 > 1
        _, u = point(u_rest, i0)                 # negative branch also starts from rest at i_dc ≈ 0
        for i in i0-1:-1:1
            V_up[i], u = point(u, i)
        end
        u_bottom = u
    end

    if iszero(βc)
        V_down .= V_up
        return V_up, V_down
    end
    u = u_top
    for i in n:-1:i0
        V_down[i], u = point(u, i)
    end
    u = u_bottom
    for i in 1:i0-1
        V_down[i], u = point(u, i)
    end
    return V_up, V_down
end

# Full map, parallel over i_ac columns (each column swept sequentially in i_dc)
function shapiro_map(cpr::TabulatedCPR; Ω = 0.5, βc = 1.0, idcs = range(0, 3, 151), iacs = range(0, 4, 101),
        nper_avg = 30, showprogress = true, bias...)
    u_rest = rest_state(cpr)
    V_up = zeros(length(idcs), length(iacs))
    V_down = zeros(length(idcs), length(iacs))
    prog = Progress(length(iacs); desc = @sprintf("Shapiro map (βc = %.3g, Ω = %.3g) ", βc, Ω), enabled = showprogress)
    Threads.@threads for j in eachindex(iacs)
        V_up[:, j], V_down[:, j] = sweep_column(iacs[j], cpr; Ω, βc, idcs, u_rest, nper_avg, bias...)
        next!(prog)                                      # thread-safe
    end
    dVdI_up = diff(V_up, dims = 1) ./ step(idcs)
    dVdI_down = diff(V_down, dims = 1) ./ step(idcs)
    hyst = abs.(V_up .- V_down)
    return (; idcs, iacs, V_up, V_down, dVdI_up, dVdI_down, hyst, Ω, βc, period = cpr.period)
end

# Chaos diagnostic: φ mod the CPR period once per drive period (a few points = phase locked, a cloud = chaotic),
# together with ⟨v⟩ over the sampled periods
function stroboscopic(cpr::TabulatedCPR; idc, iac, Ω, βc, u0 = rest_state(cpr), nper_skip = 50, nper = 200,
        reltol = 1e-8, abstol = 1e-8, βc_stiff = 1.0)
    T = 2π / Ω
    ts = (nper_skip * T):T:((nper_skip + nper) * T)
    p = (; idc, iac, Ω, βc, cpr)
    if iszero(βc)
        prob = ODE.ODEProblem(rsj, u0[1], (0.0, last(ts)), p)
        sol = ODE.solve(prob, ODE.Tsit5(); saveat = ts, save_start = false, dtmax = 0.1, reltol, abstol, maxiters = 10^8)
        φs = sol.u
    else
        alg, dtmax = rcsj_solver(βc; βc_stiff)
        prob = ODE.ODEProblem(rcsj, SVector(u0[1], u0[2]), (0.0, last(ts)), p)
        sol = ODE.solve(prob, alg; saveat = ts, save_start = false, dtmax, reltol, abstol, maxiters = 10^8)
        φs = first.(sol.u)
    end
    return (; φ = mod.(φs, cpr.period), v_dc = (last(φs) - first(φs)) / (last(ts) - first(ts)))
end
