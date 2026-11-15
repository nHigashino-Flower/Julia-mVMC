#=
Pf パートン(PartonMode = 2)の対関数 f の同一視を格子図に描く(フレーバーごとに 1 枚の PDF)
--- parton-mode (fork addition) ---

    julia --project=tools tools/plot_parton_pair_idx.jl <in_dir>

`<in_dir>` に `partonpairorbital.def` と `parton_sites.dat`(`tools/parton_pair_idx.jl` の `write_parton_sites`)が要る。
出力は `<in_dir>/partonpair_idx_flavor<a>.pdf`(フレーバー a ごと)。

各サイトに番号を書く。**同じ番号のサイトは同一視されている**: (i, a) の行 f[(i,a), ·] に現れる idx の集合が同じ
(= 共有の並進で写り合う)。番号は全フレーバー通し(`partonpair_idxmap.def` の SITE 行と同じ)なので、
フレーバー対称の def ならフレーバーをまたいで同じ番号が出る。−1 は f の行が空のサイト。
描き方は ~/Code/ModuleExternalFields.jl の plot_lattice_numbers に合わせる(既定の点 + 右横に番号、フォント 8。
ずらす量は最近接距離の 0.2 倍 = 参考の「格子間隔 1 に 0.2」と同じ比)。図のラベルは英語(GR のフォントに日本語が無い)。
=#

using Printf
using Plots
include(joinpath(@__DIR__, "parton_pair_idx.jl"))

function read_sites(path)
    pos = Dict{Int,NTuple{2,Float64}}()
    sub = Dict{Int,Int}()
    period = ((0.0, 0.0), (0.0, 0.0))
    for l in eachline(path)
        t = split(strip(l))
        isempty(t) && continue
        if t[1] == "PERIOD"
            v = parse.(Float64, t[2:5])
            period = ((v[1], v[2]), (v[3], v[4]))
        elseif t[1] == "SITE"
            k = parse(Int, t[2])
            pos[k] = (parse(Float64, t[3]), parse(Float64, t[4]))
            sub[k] = length(t) >= 5 ? parse(Int, t[5]) : 0
        end
    end
    n = length(pos)
    return [pos[k] for k = 0:(n - 1)], [sub[k] for k = 0:(n - 1)], period
end

function main(argv)
    dir = argv[1]
    rows, n_idx = read_pair_def(joinpath(dir, "partonpairorbital.def"))
    pos, sub, period = read_sites(joinpath(dir, "parton_sites.dat"))
    ns = length(pos)
    nf = maximum(max(r[2], r[4]) for r in rows) + 1
    cls = pair_site_classes(pair_idx_lookup(rows), ns, nf)
    xs, ys = first.(pos), last.(pos)
    dmin = minimum(hypot(xs[i] - xs[j], ys[i] - ys[j]) for i = 1:ns for j = 1:ns if i != j)
    off = 0.2 * dmin
    for a = 0:(nf - 1)
        scatter(xs, ys; legend = false, title = "flavor $a", titlefontsize = 10)
        for k = 1:ns
            annotate!(xs[k] + off, ys[k], text(string(cls[(k - 1, a)]), 8))
        end
        out = joinpath(dir, "partonpair_idx_flavor$(a).pdf")
        savefig(out)
        println("図: ", out, "(番号 ", length(unique(cls[(i, a)] for i = 0:(ns - 1))), " 種類)")
    end
end

main(ARGS)
