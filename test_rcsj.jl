# Validation of the RSJ/RCSJ Shapiro solver against known results (CPR = sin φ unless stated).
# Run from this folder:  julia --project -t auto test_rcsj.jl
using Test
include("rcsj.jl")

const SIN = ([1.0], [0.0])
const SIN2 = ([0.0, 1.0], [0.0, 0.0])

# Bessel function of the first kind from its integral representation (avoids a SpecialFunctions dependency)
function besselj_int(n, x; m = 4000)
    f = [cos(n * τ - x * sin(τ)) for τ in range(0, π, m)]
    return (sum(f) - (f[1] + f[end]) / 2) / (m - 1)            # trapezoid of (1/π)∫₀^π
end

# Plateaus of one sweep: runs of ≥ minlen consecutive points whose ⟨v⟩/Ω differs by < tol → (value, idc_from, idc_to)
function plateaus(idcs, w; tol = 1e-3, minlen = 3)
    out = Tuple{Float64, Float64, Float64}[]
    i = 1
    while i < length(w)
        j = i
        while j < length(w) && abs(w[j+1] - w[i]) < tol
            j += 1
        end
        j - i + 1 ≥ minlen && push!(out, (sum(w[i:j]) / (j - i + 1), idcs[i], idcs[j]))
        i = j + 1
    end
    return out
end

# Width (in i_dc) of the plateau at ⟨v⟩/Ω = n
plateau_width(idcs, w, n; tol = 1e-3) = count(x -> abs(x - n) < tol, w) * step(idcs)

@testset "RSJ/RCSJ Shapiro solver" begin

    @testset "1. i_ac = 0, βc = 0: ⟨v⟩ = √(i² − 1)" begin
        idcs = range(0, 3, 61)
        m = shapiro_map(SIN...; Ω = 0.5, βc = 0, idcs, iacs = [0.0], nper_avg = 60, nper_skip_min = 20, showprogress = false)
        exact = [abs(i) > 1 ? sqrt(i^2 - 1) : 0.0 for i in idcs]
        err = maximum(abs, m.V_up[:, 1] .- exact)
        println(@sprintf("  max |⟨v⟩ − √(i²−1)| = %.2e", err))
        @test err < 1e-2
    end

    @testset "2. i_ac = 0, βc = 5: switching ≈ 1, retrapping ≈ 4/(π√βc)" begin
        idcs = range(0, 1.5, 151)
        βc = 5.0
        m = shapiro_map(SIN...; Ω = 0.5, βc, idcs, iacs = [0.0], nper_avg = 60, nper_skip_min = 20, showprogress = false)
        i_sw = idcs[findfirst(>(0.05), m.V_up[:, 1])]
        i_r = idcs[findfirst(>(0.05), m.V_down[:, 1])]
        println(@sprintf("  switching at i_dc = %.3f, retrapping at i_dc = %.3f (4/(π√βc) = %.3f)", i_sw, i_r, 4 / (π * sqrt(βc))))
        @test abs(i_sw - 1) ≤ 0.011
        @test abs(i_r - 4 / (π * sqrt(βc))) < 0.06
    end

    @testset "3. βc = 1e-2 reproduces the βc = 0 map" begin
        grid = (; Ω = 0.5, idcs = range(-2, 2, 81), iacs = range(0, 3, 31), nper_avg = 60, nper_skip_min = 20, showprogress = false)
        m0 = shapiro_map(SIN...; βc = 0, grid...)
        m1 = shapiro_map(SIN...; βc = 1e-2, grid...)
        d = abs.(m1.V_up .- m0.V_up) ./ grid.Ω
        frac = count(>(0.05), d) / length(d)
        println(@sprintf("  ⟨v⟩/Ω: mean |Δ| = %.2e, max |Δ| = %.2e, %.1f %% of points with |Δ| > 0.05; max hysteresis/Ω = %.2e",
            sum(d) / length(d), maximum(d), 100frac, maximum(m1.hyst) / grid.Ω))
        @test frac < 0.03
        @test sum(d) / length(d) < 5e-3
    end

    @testset "4. Ω = 5, βc = 0: step widths ≈ 2|J_n(i_ac/Ω)|" begin
        Ω = 5.0
        idcs = range(-1.5, 12, 1351)
        iacs = [2.5, 5.0, 10.0]
        m = shapiro_map(SIN...; Ω, βc = 0, idcs, iacs, nper_avg = 400, nper_skip_min = 20, showprogress = false)
        for (j, iac) in enumerate(iacs), n in 0:2
            w = plateau_width(idcs, m.V_up[:, j] ./ Ω, n)
            expected = 2 * abs(besselj_int(n, iac / Ω))
            println(@sprintf("  i_ac = %4.1f, n = %d: width %.3f, 2|J_n| = %.3f", iac, n, w, expected))
            @test abs(w - expected) < 0.05 * expected + 3 * step(idcs)
        end
    end

    @testset "5. CPR = sin 2φ, βc = 0: plateaus only at multiples of 1/2" begin
        Ω = 0.5
        idcs = range(0, 2, 201)
        m = shapiro_map(SIN2...; Ω, βc = 0, idcs, iacs = range(0, 3, 31), nper_avg = 60, nper_skip_min = 20, showprogress = false)
        values = [p[1] for j in axes(m.V_up, 2) for p in plateaus(idcs, m.V_up[:, j] ./ Ω; minlen = 4)]
        off = filter(x -> abs(2x - round(2x)) > 0.02, values)
        println(@sprintf("  %d plateaus, distinct values %s, %d off the half-integer grid",
            length(values), string(sort(unique(round.(values; digits = 2)))), length(off)))
        @test isempty(off)
        @test any(x -> abs(x - 0.5) < 0.02, values)
    end

    @testset "6. βc = 100, Ω = 0.3, i_ac = 10: n = ±1 steps cover i_dc = 0" begin
        p = (; Ω = 0.3, βc = 100.0, iac = 10.0, s = SIN[1], c = SIN[2])
        bias = (; nper_avg = 60, nper_skip_min = 20)
        v0, u = bias_point(rest_state(SIN...), (; p..., idc = 0.0); bias...)
        println(@sprintf("  from rest at i_dc = 0: ⟨v⟩/Ω = %.4f", v0 / p.Ω))
        @test abs(v0 / p.Ω - 1) < 1e-2
        # outward sweeps from rest: the n = +1 step extends to both sides of i_dc = 0
        idcs = range(-0.2, 0.2, 41)
        V_up, V_down = sweep_column(p.iac, p.s, p.c; Ω = p.Ω, βc = p.βc, idcs, bias...)
        near = abs.(idcs) .≤ 0.05
        println(@sprintf("  outward sweeps: ⟨v⟩/Ω ∈ [%.4f, %.4f] for |i_dc| ≤ 0.05", extrema(V_up[near] ./ p.Ω)...))
        @test all(x -> abs(x - 1) < 1e-2, V_up[near] ./ p.Ω)
        # mirror image (φ, v, τ) → (−φ, −v, τ + T/2) of the n = +1 orbit is the n = −1 orbit at i_dc = 0
        T = 2π / p.Ω
        sol = ODE.solve(ODE.ODEProblem(rcsj, SVector(mod2pi(u[1]), u[2]), (0.0, T / 2), (; p..., idc = 0.0)), ODE.Tsit5();
            reltol = 1e-10, abstol = 1e-10, dtmax = 0.1)
        vm, _ = bias_point((-sol.u[end][1], -sol.u[end][2]), (; p..., idc = 0.0); bias...)
        println(@sprintf("  mirrored state at i_dc = 0: ⟨v⟩/Ω = %.4f", vm / p.Ω))
        @test abs(vm / p.Ω + 1) < 1e-2
    end

    @testset "7. Kautz 1996 Fig. 13a (βc = 2.78, Ω = 1.8, i_ac = 16.9)" begin
        Ω = 1.8
        idcs = range(0, 4, 201)
        m = shapiro_map(SIN...; Ω, βc = 2.78, idcs, iacs = [16.9], nper_avg = 60, nper_skip_min = 20, showprogress = false)
        w = m.V_up[:, 1] ./ Ω
        for (val, lo, hi) in ((1/3, 0.60, 0.62), (1/2, 0.84, 0.94), (2/3, 1.12, 1.16), (1, 1.30, 2.38), (3/2, 2.66, 2.78))
            inside = findall(i -> lo - 1e-9 ≤ i ≤ hi + 1e-9, idcs)
            got = extrema(w[inside])
            println(@sprintf("  ⟨v⟩/Ω = %.3f expected on i_dc ∈ [%.2f, %.2f]: found [%.4f, %.4f]", val, lo, hi, got...))
            @test all(x -> abs(x - val) < 1e-2, w[inside])
        end
        println("  plateaus found: ", join([@sprintf("%.3f on [%.2f, %.2f]", p...) for p in plateaus(idcs, w) if p[3] > p[2]], ", "))
    end
end
