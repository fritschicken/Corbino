# Periodicity scan at small size (N = R = 500), no plots: julia --project test_periodicity.jl
# Looking for a 2π-periodic ground state F(φ) and a 4π-periodic first excited state F(φ) + ε(φ), where ε is
# the lowest Andreev level followed adiabatically (max overlap) around the loop.
# H(φ + 2π) = H(φ) exactly, so F is always 2π-periodic; the excited branch is 4π-periodic iff the tracked level
# comes back after one loop as its particle-hole partner, ε(2π) = -ε(0) (an odd number of zero crossings).
using Quantica, LinearMaps, ArnoldiMethod, Printf
include("../arXiv_2506.15502/code.jl")

base_params = (; ħ2ome = 76.2, m = 0.065, seed = 1, R = 500, N = 500, n = 1, Δ = 0.2)

# Spectra on the periodic grid φ_k = 2πk/M, k = 0..M-1 (φ = 2π is the same Hamiltonian as φ = 0)
function loop_spectra(hN, params, M, solver)
    φs = range(0, 2π, length = M + 1)[1:M]
    sps = [spectrum(hN(; params..., ϕ = φ); solver) for φ in φs]
    return φs, [real(energies(sp)) for sp in sps], [states(sp) for sp in sps]
end

# Follow the lowest positive level at φ = 0 for `loops` turns. Returns ε along the path (length loops*M + 1),
# the overlap of the final state with the starting one after each loop, and the weakest step overlap.
function track_level(εs, ψs; loops = 2)
    M = length(εs)
    pos = findall(>(0), εs[1])
    i = pos[argmin(εs[1][pos])]
    v0 = v = ψs[1][:, i]
    path = [εs[1][i]]
    back = Float64[]
    minov = 1.0
    for step in 1:loops*M
        k = mod(step, M) + 1
        ov = abs2.(ψs[k]' * v)
        j = argmax(ov)
        minov = min(minov, ov[j])
        v = ψs[k][:, j]
        push!(path, εs[k][j])
        mod(step, M) == 0 && push!(back, abs2(v0' * v))
    end
    return path, back, minov
end

# Classify the tracked branch: :2π (returns after one loop), :4π (returns after two, ε flips sign in between)
function periodicity(hN, params; M = 200, nev = 8, rtol = 1e-3)
    solver = ES.ShiftInvert(ES.ArnoldiMethod(nev = nev), 0)
    φs, εs, ψs = loop_spectra(hN, params, M, solver)
    path, back, minov = track_level(εs, ψs)
    ε0, ε2π, ε4π = path[1], path[M+1], path[2M+1]
    crossings = count(k -> sign(path[k]) != sign(path[k+1]), 1:M)
    same(a, b) = isapprox(a, b; rtol, atol = rtol * abs(ε0))
    kind = same(ε2π, ε0) && back[1] > 0.5 ? Symbol("2π") :
           same(ε2π, -ε0) && same(ε4π, ε0) && back[2] > 0.5 ? Symbol("4π") : :unclear
    # the soliton level should stay the lowest |ε| along the loop, well below the next level
    gap = minimum(k -> partialsort(abs.(εs[k]), 3), 1:M)
    return (; kind, ε0, ε2π, ε4π, crossings, back, minov, εmax = maximum(abs, path[1:M]), gap, φs, path)
end

function scan_periodicity(µs, Vgs; params = base_params, kw...)
    results = []
    @printf("%6s %6s  %-8s %9s %9s %9s %4s %7s %7s %8s %8s\n",
        "µ", "Vg", "kind", "ε(0)", "ε(2π)", "ε(4π)", "#0", "back1", "minov", "max|ε|", "gap")
    for µ in µs, Vg in Vgs
        p = merge(params, (; µ, Vg))
        (; hN) = redirect_stdout(devnull) do  # build @shows 2t, N, kF, hvF
            build(; p...)
        end
        r = periodicity(hN, p; kw...)
        @printf("%6.3f %6.3f  %-8s %9.5f %9.5f %9.5f %4d %7.3f %7.3f %8.5f %8.5f\n",
            µ, Vg, r.kind, r.ε0, r.ε2π, r.ε4π, r.crossings, r.back[1], r.minov, r.εmax, r.gap)
        r.minov < 0.5 && @warn @sprintf("weak overlap %.2f at µ = %g, Vg = %g: increase M", r.minov, µ, Vg)
        push!(results, merge((; µ, Vg), r))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = scan_periodicity(range(0.2, 3.0, step = 0.2), range(0.2, 3.0, step = 0.22), nev = 30)
    hits = filter(r -> r.kind == Symbol("4π") && r.minov > 0.5, results)
    println("\n4π-periodic first excited state (2π-periodic ground state) at:")
    isempty(hits) ? println("  none") :
        foreach(r -> @printf("  µ = %.3f, Vg = %.3f   max|ε| = %.4f, next level ≥ %.4f\n", r.µ, r.Vg, r.εmax, r.gap), hits)
end

results = scan_periodicity(range(0.2, 3.0, step = 0.2), range(0.2, 3.0, step = 0.22), nev = 30)
hits = filter(r -> r.kind == Symbol("4π") && r.minov > 0.5, results)
println("\n4π-periodic first excited state (2π-periodic ground state) at:")
isempty(hits) ? println("  none") :
    foreach(r -> @printf("  µ = %.3f, Vg = %.3f   max|ε| = %.4f, next level ≥ %.4f\n", r.µ, r.Vg, r.εmax, r.gap), hits)
