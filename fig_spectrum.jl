# BdG spectrum against flux, with the fluxoid windings of the inner and outer ring
include("setup.jl")

params = merge(base_params, (; ϕ = 0, n = 1, Δ = 0.2, Vg = 1.5))
(; hN, ϵ1) = build(; params...)

flux_params = (; nmax = 1.0, npoints = 250, area_ratio = 0.0)
spectrum_plot = plot_flux_spectrum(hN; flux_params..., nev = 40,
    savepath = joinpath(@__DIR__, "figures", "spectrum.png"), params...)
