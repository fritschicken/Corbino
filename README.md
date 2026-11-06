# Corbino

Numerical study of Corbino Josephson junctions: a ring-shaped normal region with induced superconductivity and fluxoids. It covers the flux spectrum, the free energy and current–phase relation, Fourier harmonics of the CPR, and Shapiro steps from an RCSJ model driven by the computed CPR.

The work builds on the model and code of [arXiv:2506.15502](https://arxiv.org/abs/2506.15502), *Solitonic Andreev spin qubits from Andreev states in Corbino Josephson junctions*.

![Shapiro maps of the even and odd fermion parity](figures/shapiro_parity_20261103-170959_parity_setup1.png)

## Contents

| File | What it does |
|---|---|
| `setup.jl` | Shared by the scripts: packages, `base_params`, `solver`, and the Shapiro setups (`shapiro_setups`, `device`) |
| `project.jl` | Physics and computation: free energy, Fourier/CPR harmonics, even/odd parity CPRs, Shapiro cases; loads all files below |
| `rcsj.jl` | RSJ/RCSJ dynamics at T = 0 (needs only the normalized CPR, not Quantica); physical units |
| `plots.jl` | Plotting functions: `plot_flux_spectrum` (BdG spectrum against flux with fluxoid winding bars); `detailed_fourier_analysis`, `sweep_vg_mu_harmonic_amplitudes`, `plot_free_energy_current` (free energy, CPR and harmonics); `plot_shapiro_grid` (and `latex_num`) |
| `shapiro_runs.jl` | Saving, loading and combining Shapiro runs |
| `figures.jl` | The article figures; running the file makes all of them: `fig_spectrum`, `fig_free_energy_current` (both cases side by side), `fig_shapiro_parity` (both saved parity runs in one grid), `fig_bands` (documentation, one per case), and `replot_shapiro` for any saved run. The two cases (gate profiles cos(θ) and cos(2θ), θ the position angle around the ring) are set in `CASES` |
| `test_rcsj.jl` | Validation of the Shapiro solver against known results: `julia --project -t auto test_rcsj.jl` |
| `test.jl`, `test_periodicity.jl`, `time_evo.jl` | Exploratory: soliton branches, 2π/4π periodicity scan, time evolution under a phase sweep (Landau–Zener) |
| `figures/` | Generated figures |
| `data/shapiro_runs/` | Saved Shapiro runs (JLD2) and their `index.csv` |

Main functions:

- `detailed_fourier_analysis`, `sweep_vg_mu_harmonic_amplitudes`: harmonics of the free energy `F(ϕ)`; the Fourier figure shows the ground state and the filled soliton (`F + E`)
- `plot_flux_spectrum`: BdG spectrum against flux, with fluxoid winding bars
- `spectrum_F0_tracked`: one dense spectrum over `ϕ`, giving the ground-state `F₀`, the tracked level `εt` (lowest positive at `φsel`, followed through the band connectivity) and the smallest Andreev level
- `plot_free_energy_current`: `F(ϕ)` and the supercurrent of one case, even (`F₀`) and odd (`F₀ + εt`) fermion parity, in meV; `plot_free_energy_current_cases` puts several cases side by side (data from `free_energy_current_data`)
- `plot_bands_density`: bands against `ϕ` with highlighted subbands and their `|ψ|²` per site
- `converged_cpr`, `converged_harmonics`: CPR harmonics converged by doubling the number of `ϕ` points (both use `doubling_harmonics`); `converged_cpr` computes the new points at each doubling, `converged_harmonics` subsamples a grid it already has
- `shapiro_case`: CPR of one case, then its Shapiro map; `scaling = :fixed` (given `βc`, `Ω`) or `:physical` (`βc`, `Ω` from the case's own `I_c` and `device = (; R, C, f)`, currents in µA)
- `shapiro_parity_cases`: Shapiro maps of the even and odd parity of one case, both CPRs from the same dense spectrum
- `shapiro_map`, `plot_shapiro_grid`: Shapiro maps of `βc φ̈ + φ̇ + I_s(φ)/I_c = i_dc + i_ac sin(Ωτ)` (`βc = 0` is plain RSJ), swept outward from `i_dc = 0` (switching) and back inward (retrapping); dV/dI and hysteresis maps, line cuts in units of `hf/2e`. Rows and columns are µN and Δn values, or any text labels (e.g. parity, gate profile)
- `stroboscopic` (`rcsj.jl`): `φ mod 2π` once per drive period, to spot chaotic regions
- `save_shapiro_run`, `load_shapiro_run`, `list_shapiro_runs`, `combine_shapiro_runs`: store runs as JLD2 files in `data/shapiro_runs/`, with an `index.csv`, so they can be plotted again without recomputing

## Requirements

The Hamiltonian (`build`, `ES`, etc.) comes from the arXiv:2506.15502 code. Clone that repo **next to** this one:

```
parent/
├── Quartets/            ← this repo
└── arXiv_2506.15502/    ← git clone https://github.com/pablosanjose/arXiv_2506.15502
```

Julia 1.13 (the `Manifest.toml` was generated with it; older versions may need `] resolve`). The exact package versions, including those needed by the arXiv code, are recorded in `Project.toml` and `Manifest.toml`. To install them, run this from this folder:

```
julia --project
] instantiate
```

`Revise` is included because the files are loaded with `includet`.

## Usage

To make all article figures in `figures/`:

```
julia --project -t auto figures.jl
```

or `include("figures.jl")` in the REPL. After that each figure can be redone on its own, e.g. `fig_bands(CASES[2])` or `fig_shapiro_parity()`; `replot_shapiro("20261103-170959_parity_setup1")` replots any saved run.

`base_params`, `solver` and the Shapiro setups are shared in `setup.jl`. New Shapiro runs are made with `shapiro_parity_cases` (even/odd parity of one case) or `shapiro_case` (ground-state CPR), saved with `save_shapiro_run` (see the end of `test.jl`). Each picks one of the setups in `shapiro_setups`:

1. RSJ baseline: `βc = 0`, `Ω = 0.5`
2. Capacitive control: `βc = 2.78`, `Ω = 1.8` (Kautz 1996, Fig. 13a), to tell capacitive half-integer steps from CPR ones
3. Physical scaling: one device (`R`, `C`, `f`), so every case gets its own `βc` and `Ω` from its `I_c`; plots in µA and `hf/2e`

Start Julia with threads (`julia --project -t auto`): the maps run in parallel over `I_ac`. 

To reload saved Shapiro runs without recomputing:

```julia
list_shapiro_runs()
run = load_shapiro_run("20261103-170959_parity_setup1")
plot_shapiro_grid(run.results, run.µs, run.Δns; iac_cut = 1.0)   # units = :physical for setup 3
```
