#=
Pf パートン(PartonMode = 2)の対関数 f の idx 対応表とサイト座標を書く(**コアは読まない**。確認用の副産物)
--- parton-mode (fork addition) ---

どの f_ij が同じパラメータに同一視されているか(def の共有の拘束)を目で確かめるためのもの。
`tools/parton_gen_pair_def.jl` や playground の入力生成スクリプトが def を書くと同時に呼ぶ。図は
`tools/plot_parton_pair_idx.jl`(tools 環境、Plots)が、この 2 ファイルを読んでフレーバーごとの PDF に描く。

- `partonpair_idxmap.def`: 主表は **(サイト i, フレーバー a) → 番号**。行 f[(i,a), ·] に現れる idx の集合が同じ
  (i, a) に同じ番号を振る(全フレーバー通し)= 共有の並進(やフレーバー対称)で同一視されているサイト。
  後ろに参考として、フレーバー対ごとの f[(i,a),(j,b)] の idx の行列(`+idx` / `-idx`、構造的 0 は `.`)も書く
- `parton_sites.dat`: サイト番号 → 実空間の座標(と副格子)、トーラスの周期ベクトル

Plots に依存しない(MVMCOptimizers の環境から呼べる)。
=#

using Printf

"""
    read_pair_def(path) -> (rows, n_idx)

`partonpairorbital.def`(上流 orbitalidxgen と同じ列: site1 flavor1 site2 flavor2 idx [sgn])を読む。
rows は (s1, f1, s2, f2, idx, sgn)(0-based)。
"""
function read_pair_def(path::AbstractString)
    lines = readlines(path)
    n_idx = 0
    for l in lines
        t = split(strip(l))
        length(t) >= 2 && t[1] == "NOrbitalIdx" && (n_idx = parse(Int, t[2]))
    end
    rows = NTuple{6,Int}[]
    for l in lines[6:end]
        t = split(strip(l))
        length(t) in (5, 6) || continue
        v = tryparse.(Int, t)
        any(isnothing, v) && continue
        push!(rows, (v[1], v[2], v[3], v[4], v[5], length(v) == 6 ? v[6] : 1))
    end
    return rows, n_idx
end

"(i, a, j, b) → sgn·idx の辞書(逆向きの行からも反対称性で埋める)。"
function pair_idx_lookup(rows)
    d = Dict{NTuple{4,Int},Tuple{Int,Int}}()
    for (s1, f1, s2, f2, k, sg) in rows
        d[(s1, f1, s2, f2)] = (k, sg)
        d[(s2, f2, s1, f1)] = (k, -sg)
    end
    return d
end

_fmt(e) = e === nothing ? "." : (e[2] > 0 ? "+$(e[1])" : "-$(e[1])")

"""
    pair_site_classes(d, nsite, nflavor) -> Dict((i, a) => 番号)

(i, a) の行に現れる idx の集合(符号は見ない)が同じものに同じ番号を振る。番号は (a, i) の順に初出で 0, 1, 2, …
(全フレーバー通し。フレーバー対称の def ならフレーバーをまたいで同じ番号になる)。行が空の (i, a) は −1。
"""
function pair_site_classes(d, nsite::Int, nflavor::Int)
    rowkey(i, a) = sort(unique([e[1] for b = 0:(nflavor - 1), j = 0:(nsite - 1)
                                for e in (get(d, (i, a, j, b), nothing),) if e !== nothing]))
    num = Dict{Vector{Int},Int}()
    out = Dict{Tuple{Int,Int},Int}()
    for a = 0:(nflavor - 1), i = 0:(nsite - 1)
        k = rowkey(i, a)
        out[(i, a)] = isempty(k) ? -1 : get!(num, k, length(num))
    end
    return out
end

"""
    write_pair_idx_table(dir; nsite, nflavor, def = "partonpairorbital.def", out = "partonpair_idxmap.def")
"""
function write_pair_idx_table(dir::AbstractString; nsite::Int, nflavor::Int,
                              def::AbstractString = "partonpairorbital.def",
                              out::AbstractString = "partonpair_idxmap.def")
    rows, n_idx = read_pair_def(joinpath(dir, def))
    d = pair_idx_lookup(rows)
    cls = pair_site_classes(d, nsite, nflavor)
    open(joinpath(dir, out), "w") do io
        println(io, "# Pf パートンの対関数 f の同一視の対応表($def から生成。コアは読まない)")
        println(io, "# 主表 'SITE i a class': (サイト i, フレーバー a) → 番号。行 f[(i,a), ·] に現れる idx の集合が同じ (i, a) は")
        println(io, "#   同じ番号 = 同一視されている(共有の並進で写り合う。フレーバー対称の def ならフレーバーをまたいでも同じ番号)。")
        println(io, "#   番号は全フレーバー通し。−1 は f の行が空(どの対にも現れない)。図は tools/plot_parton_pair_idx.jl")
        println(io, "# 参考 'flavor a b' ブロック: 行 = i(フレーバー a)、列 = j = 0..nsite-1(フレーバー b)の f[(i,a),(j,b)] の idx。")
        println(io, "#   f = 2·sgn·p_idx。値は sgn·idx を +idx / -idx、def に無い対(構造的 0)は .")
        @printf(io, "nsite %d\nnflavor %d\nn_idx %d\nn_rows %d\nn_class %d\n", nsite, nflavor, n_idx, length(rows),
                length(unique(v for v in values(cls) if v >= 0)))
        println(io, "# SITE i a class")
        for a = 0:(nflavor - 1), i = 0:(nsite - 1)
            @printf(io, "SITE %d %d %d\n", i, a, cls[(i, a)])
        end
        for a = 0:(nflavor - 1), b = a:(nflavor - 1)
            any(r -> (r[2] == a && r[4] == b) || (r[2] == b && r[4] == a), rows) || continue
            println(io, "flavor $a $b")
            for i = 0:(nsite - 1)
                @printf(io, "%4d :", i)
                for j = 0:(nsite - 1)
                    @printf(io, " %6s", _fmt(get(d, (i, a, j, b), nothing)))
                end
                println(io)
            end
        end
    end
    return joinpath(dir, out)
end

"""
    write_parton_sites(dir, positions; sublattice, periods, out = "parton_sites.dat")

サイト番号 → 実空間の座標。periods はトーラスの周期ベクトル 2 本((x, y) のタプル)。
"""
function write_parton_sites(dir::AbstractString, positions::Vector{NTuple{2,Float64}};
                            sublattice::Vector{Int} = zeros(Int, length(positions)),
                            periods::NTuple{2,NTuple{2,Float64}}, out::AbstractString = "parton_sites.dat")
    open(joinpath(dir, out), "w") do io
        println(io, "# サイト番号 → 実空間の座標(tools/plot_parton_pair_idx.jl が読む。コアは読まない)")
        @printf(io, "PERIOD %.12f %.12f %.12f %.12f\n", periods[1]..., periods[2]...)
        println(io, "# site x y sublattice")
        for (k, p) in enumerate(positions)
            @printf(io, "SITE %d %.12f %.12f %d\n", k - 1, p[1], p[2], sublattice[k])
        end
    end
    return joinpath(dir, out)
end

"""
    plot_pair_idx(dir) -> Bool

`tools/plot_parton_pair_idx.jl` を tools 環境で別プロセスとして呼ぶ(この環境に Plots を入れないため)。
失敗しても入力生成は止めず、警告だけ出す。
"""
function plot_pair_idx(dir::AbstractString)
    tools = @__DIR__
    cmd = `$(Base.julia_cmd()) --project=$tools $(joinpath(tools, "plot_parton_pair_idx.jl")) $dir`
    try
        run(pipeline(cmd; stdout = stdout, stderr = devnull))
        return true
    catch e
        @warn "f の idx の図を描けませんでした(手で `julia --project=tools tools/plot_parton_pair_idx.jl $dir`)" exception = e
        return false
    end
end
