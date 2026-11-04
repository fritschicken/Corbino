# Shared by the figure scripts: packages, project functions, base parameters, the solver and the Shapiro setups.
# Each script sets its own params on top of base_params and builds hN.
using Quantica, StaticArrays, LinearMaps
using CairoMakie
using ArnoldiMethod
using Printf
using JLD2
using Dates
using ColorSchemes
using ProgressMeter

includet("../arXiv_2506.15502/code.jl")
includet("./project.jl")

base_params = (;
    ħ2ome = 76.2,
    N = 250,
    m = 0.065,
    µ = 0.1,
    R = 500,
    seed = 1)

solver = ES.LinearAlgebra()

# Shapiro (RCSJ) setups, picked in the Shapiro scripts with `setup = 1/2/3`; T = 0 throughout
# 1: RSJ quartet baseline        (fixed normalized βc, Ω)
# 2: capacitive control          (fixed normalized βc, Ω; Kautz 1996 Fig. 13a parameters)
# 3: physically scaled device    (βc, Ω per case from its own I_c and `device`; currents in µA)
device = (; R = 1e3, C = 1e-15, f = 20e9)       # R [Ω], C [F], drive f [Hz], used by setup 3
shapiro_setups = Dict(
    1 => (; scaling = :fixed, βc = 0, iac_cut = 1.0,
        drive = (; Ω = 0.5, idcs = range(-2, 2, 251), iacs = range(0, 3, 201)),   # 251 → i_dc = 0 on the grid
        bias = (; nper_skip_min = 20, nper_avg = 60)),
    2 => (; scaling = :fixed, βc = 2.78, iac_cut = 17.0,
        drive = (; Ω = 1.8, idcs = range(-4, 4, 251), iacs = range(0, 20, 101)),
        bias = (; nper_skip_min = 20, nper_avg = 60)),
    3 => (; scaling = :physical, βc = 0.0, iac_cut = 1.0,                         # βc from I_c; iac_cut in µA
        drive = (; Idcs_uA = range(-3, 3, 301), Iacs_uA = range(0, 5, 201)),      # µA, odd length → I_dc = 0 on the grid
        bias = (; nper_skip_min = 20, nper_avg = 60)))
