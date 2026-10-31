"""
Pfaffian パートンモード(PartonMode = 2)の対軌道 def を作る(DESIGN_parton_pf.md §2.1・§5)

制限は 2 つ(ユーザー提案 2026-10-31):

1. **同フレーバー対(トリプレット的なペアリング)を 0 でロックする**: def に行を書かない(構造的 0)。
   OptFlag 0 の凍結ではパラメータ数(O の蓄積と S の大きさ)が減らないので、行ごと外す
2. **拡大ユニットセルの並進で共有する**: 平均場を保つ並進(`pl_mf_shifts(model, ex, ey)`)で写り合う対を
   同じ idx にまとめる。行は「並進後の向きのまま」書くので、f_{Tα,Tβ} = f_{αβ} が符号なしで成り立つ。
   並進で対 {α, β} が {β, α} に写る軌道は f = 0 が強制されるので外す

使い方(Julia-mVMC-pf の環境で):

    julia --project=MVMCOptimizers.jl tools/parton_gen_pair_def.jl <out_def> cb <nx> <ny> <K> <ex> <ey> [<alpha_pmfpara_def>]
    julia --project=MVMCOptimizers.jl tools/parton_gen_pair_def.jl <out_def> km <Lx> <Ly> <φ> <K> <ex> <ey> [<alpha_pmfpara_def>]

最後の引数に α run の pmfpara.def を渡すと、使った並進が α の idx の写像を保つことを検査する。
"""

using MVMCExpertModeParsers

const _PHYS = joinpath(@__DIR__, "..", "test", "physics")
include(joinpath(_PHYS, "checkerboard_model.jl"))
include(joinpath(_PHYS, "lattice_model.jl"))

"""
    gen_pair_def(path, model, n_flavor, ex, ey; same_flavor = false) -> (n_idx, n_rows, n_forced_zero)

平均場を保つ並進の軌道ごとに 1 つの idx を割り当てた def を書く。
"""
function gen_pair_def(path::AbstractString, model, n_flavor::Int, ex::Int, ey::Int;
                      same_flavor::Bool = false)
    ns = pl_nsite(model)
    perms = [pl_shift_perm(model, tx, ty) for (tx, ty) in pl_mf_shifts(model, ex, ey)]
    n_state = ns * n_flavor
    seen = falses(n_state, n_state)
    rows = NTuple{5,Int}[]       # (site1, flavor1, site2, flavor2, idx)、すべて 0-based
    n_idx = 0
    n_forced_zero = 0
    for α = 0:(n_state - 1), β = (α + 1):(n_state - 1)
        a, i = divrem(α, ns)
        b, j = divrem(β, ns)
        (!same_flavor && a == b) && continue
        seen[α + 1, β + 1] && continue
        orbit = Tuple{Int,Int}[]
        forced_zero = false
        for p in perms
            Tα = a * ns + p[i + 1]
            Tβ = b * ns + p[j + 1]
            lo, hi = minmax(Tα, Tβ)
            seen[lo + 1, hi + 1] = true
            if (Tβ, Tα) in orbit
                forced_zero = true          # 同じ対が逆向きでも現れる: f = −f
            elseif !((Tα, Tβ) in orbit)
                push!(orbit, (Tα, Tβ))
            end
        end
        if forced_zero
            n_forced_zero += 1
            continue
        end
        for (s, t) in orbit
            fa, si = divrem(s, ns)
            fb, sj = divrem(t, ns)
            push!(rows, (si, fa, sj, fb, n_idx))
        end
        n_idx += 1
    end
    open(path, "w") do io
        println(io, "=============================================")
        println(io, "NOrbitalIdx $n_idx")
        println(io, "ComplexType 1")
        println(io, "=============================================")
        println(io, "=============================================")
        for (s1, f1, s2, f2, k) in rows
            println(io, "$s1 $f1 $s2 $f2 $k 1")
        end
    end
    return (n_idx = n_idx, n_rows = length(rows), n_forced_zero = n_forced_zero)
end

"""
    check_against_alpha(model, ex, ey, pmfpara_path) -> Bool

使った並進が α run の pmfpara.def の idx の写像を保つか(非順序の対で引く)。
"""
function check_against_alpha(model, ex::Int, ey::Int, pmfpara_path::AbstractString)
    res, _, _ = MVMCExpertModeParsers.parse_parton_mf_para_def(String(pmfpara_path))
    res.success || error("pmfpara.def を読めません: $(res.error_message)")
    key(i, j, f) = (minmax(i, j)..., f)
    idx = Dict(key(t.site1, t.site2, t.flavor1) => t.idx for t in res.data)
    perms = [pl_shift_perm(model, tx, ty) for (tx, ty) in pl_mf_shifts(model, ex, ey)]
    for p in perms, (k, v) in idx
        i, j, f = k
        get(idx, key(p[i + 1], p[j + 1], f), -1) == v || return false
    end
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = ARGS[1]
    kind = ARGS[2]
    if kind == "cb"
        nx, ny, K, ex, ey = parse.(Int, ARGS[3:7])
        model = CheckerboardLatticeModel(nx, ny)
        rest = ARGS[8:end]
    elseif kind == "km"
        Lx, Ly = parse.(Int, ARGS[3:4])
        φ = eval(Meta.parse(ARGS[5]))
        K, ex, ey = parse.(Int, ARGS[6:8])
        model = KMLatticeModel(Lx, Ly, φ, K)
        rest = ARGS[9:end]
    else
        error("格子は cb か km")
    end
    r = gen_pair_def(out, model, K, ex, ey)
    println("n_idx = $(r.n_idx)  rows = $(r.n_rows)  forced_zero_orbits = $(r.n_forced_zero)")
    if !isempty(rest)
        ok = check_against_alpha(model, ex, ey, rest[1])
        println("alpha の idx 写像を保つ: $ok")
        ok || error("並進が α の pmfpara.def と整合しません(拡大セルの指定を確認すること)")
    end
end
