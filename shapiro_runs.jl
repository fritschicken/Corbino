# ---------------------------------------------------------------------
# Shapiro run storage: one JLD2 file per run, named by an ID, plus an index.csv listing all runs
#   id = "yyyymmdd-HHMMSS" or "yyyymmdd-HHMMSS_label"
# ---------------------------------------------------------------------
const SHAPIRO_RUN_DIR = joinpath(@__DIR__, "data", "shapiro_runs")

fmt_values(v) = join(map(x -> x isa Number ? @sprintf("%g", x) : string(x), [v...]), " ")
csv_field(x) = (s = string(x); occursin(r"[,\"\n]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s)

# meta: any NamedTuple of settings to keep with the run (params, drive, bias, βc, cpr_settings, device, ...)
function save_shapiro_run(results, µs, Δns; meta = (;), label = "", dir = SHAPIRO_RUN_DIR)
    mkpath(dir)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    id = isempty(label) ? stamp : string(stamp, "_", replace(label, r"[^\w\-]" => "-"))
    isfile(joinpath(dir, id * ".jld2")) && (id *= "-" * string(rand(UInt16), base = 16))
    r1 = first(results)
    # ASCII keys in the file (Julia folds µ/μ in identifiers, which would break string lookup)
    # per case (they differ with scaling = :physical): code-unit and physical Ic, βc, Ω
    per_case = map(r -> (; Ic = r.Ic, Ic_A = r.Ic_A, βc = r.βc, Ω = r.Ω, Emin = r.Emin), vec(results))
    jldsave(joinpath(dir, id * ".jld2"); id, date = string(now()), label, results, mus = [µs...], dns = [Δns...],
        meta = merge(meta, (; per_case)))
    index = joinpath(dir, "index.csv")
    isfile(index) || write(index, "id,date,label,Δn,µN,Ω,βc,n_idc,n_iac,Ic\n")
    join_g(f) = join(map(r -> @sprintf("%.3g", f(r)), vec(results)), " ")
    row = (id, Dates.format(now(), "yyyy-mm-dd HH:MM"), label, fmt_values(Δns), fmt_values(µs),
        join_g(r -> r.Ω), join_g(r -> r.βc), length(r1.idcs), length(r1.iacs), join_g(r -> r.Ic))
    open(io -> println(io, join(map(csv_field, row), ",")), index, "a")
    println("saved Shapiro run ", id)
    return id
end

function load_shapiro_run(id; dir = SHAPIRO_RUN_DIR)
    file = joinpath(dir, endswith(id, ".jld2") ? id : id * ".jld2")
    isfile(file) || error("no Shapiro run $id in $dir (see list_shapiro_runs())")
    d = load(file)
    return (; id = d["id"], date = d["date"], label = d["label"], results = d["results"],
        µs = d["mus"], Δns = d["dns"], meta = d["meta"])
end

load_shapiro_runs(ids...; dir = SHAPIRO_RUN_DIR) = [load_shapiro_run(id; dir) for id in ids]

function list_shapiro_runs(; dir = SHAPIRO_RUN_DIR)
    index = joinpath(dir, "index.csv")
    isfile(index) ? print(read(index, String)) : println("no Shapiro runs saved in $dir yet")
    return nothing
end

# Combine runs into one grid for plot_shapiro_grid: side by side (new Δn columns) if they share µs,
# or stacked (new µN rows) if they share Δns
function combine_shapiro_runs(runs...)
    length(runs) == 1 && return first(runs)
    r1 = first(runs)
    grids = [reshape(r.results, length(r.µs), length(r.Δns)) for r in runs]
    if all(r -> r.µs == r1.µs, runs)
        return (; results = reduce(hcat, grids), µs = r1.µs, Δns = reduce(vcat, [r.Δns for r in runs]))
    elseif all(r -> r.Δns == r1.Δns, runs)
        return (; results = reduce(vcat, grids), µs = reduce(vcat, [r.µs for r in runs]), Δns = r1.Δns)
    else
        error("runs can only be combined if they share either all µN values or all Δn values")
    end
end
