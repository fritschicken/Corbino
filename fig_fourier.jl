# Free energy, current and harmonic amplitudes against ϕ, even and odd parity
include("setup.jl")

params = merge(base_params, (; µ = 0.1, ϕ = 0, n = 1, Δ = 0.2, Vg = 1.5, mult=2))
(; hN, ϵ1) = build(; params...)

results = detailed_fourier_analysis(hN; npoints = 32, max_harmonics = 8, solver, params...)