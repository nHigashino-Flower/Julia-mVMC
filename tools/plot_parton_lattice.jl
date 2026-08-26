#=
パートンアンザッツの格子図(参照実装 `lattice_EF.pdf` 相当)
--- parton-mode (fork addition) ---

    julia --project=tools tools/plot_parton_lattice.jl <dir> [出力.pdf] \
        [--label ef|site] [--shells N]

`<dir>` は `parton_lattice.dat` を含むディレクトリ、または run ディレクトリ
(その場合 `stage*_in` を自動で探す)。既定の出力は
**`<parton_lattice.dat` と同じディレクトリ>/parton_lattice.pdf**
(図は PDF 統一・保存先はその run の出力と同じディレクトリ、という運用規約)。

`parton_lattice.dat` は def 生成時に `write_parton_lattice_meta` が書く副産物で、
**コアは読まない**(namelist.def に載っていない)。tools が模型実装や playground に
依存しないための橋渡し。

## 図の中身

**左パネル — EF クラスと拡大ユニットセル**

サイトを格子座標にプロットし、**EF クラス**(= 平均場の並進群の軌道番号)で
色分け・注記する。参照実装 `plot_lattice_numbers` と同じ読み方。重ねて描くのは:

- **実線の枠** = 拡大ユニットセル(平均場が保つ並進が張るセル)
- **破線の枠** = 物理ハミルトニアンの磁気ユニットセル

**この 2 つの面積比が「拡大セルを物理の何倍に取ったか」**で、パートン構成では
フラックスが φ → φ/F になる分だけ **F 倍**になるのが期待値。タイトルに
並進群の位数(物理 / 平均場 / QP)を出すので、比が F になっていることが読める。

**右パネル — ボンドクラス**

最短 2 シェルのボンドを **共有する α の idx** で色分けする。同じ色のボンドが
1 つの変分パラメータを共有している。**凍結クラス**(平均場の並進が端点を
入れ替えるため α を実数に縛る必要があるボンド)は太い破線で描く。
矢印は QP 射影が張る並進。
=#

using Printf
using Plots

# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------

"`parton_lattice.dat` を読む。"
function read_lattice_meta(path::AbstractString)
    meta = Dict{String,Vector{Int}}()
    model = ""
    sites = NTuple{4,Int}[]                 # (isite, x, y, ef)
    phys = NTuple{2,Int}[]
    mfs = NTuple{2,Int}[]
    qps = NTuple{2,Int}[]
    frozen = Set{Int}()
    bonds = NTuple{4,Int}[]                 # (s1, s2, idx, d2)
    for ln in eachline(path)
        s = strip(ln)
        (isempty(s) || startswith(s, "#")) && continue
        t = split(s)
        k = t[1]
        if k == "model"
            model = t[2]
        elseif k == "SITE"
            push!(sites, (parse(Int, t[2]), parse(Int, t[3]), parse(Int, t[4]),
                          parse(Int, t[5])))
        elseif k == "PHYSSHIFT"
            push!(phys, (parse(Int, t[2]), parse(Int, t[3])))
        elseif k == "MFSHIFT"
            push!(mfs, (parse(Int, t[2]), parse(Int, t[3])))
        elseif k == "QPSHIFT"
            push!(qps, (parse(Int, t[2]), parse(Int, t[3])))
        elseif k == "FROZEN"
            push!(frozen, parse(Int, t[2]))
        elseif k == "BOND"
            push!(bonds, (parse(Int, t[2]), parse(Int, t[3]), parse(Int, t[4]),
                          parse(Int, t[5])))
        else
            meta[k] = [parse(Int, v) for v in t[2:end]]
        end
    end
    isempty(sites) && error("SITE 行がありません: $(path)")
    return (model = model, meta = meta, sites = sites, phys = phys, mfs = mfs,
            qps = qps, frozen = frozen, bonds = bonds)
end

"""
並進の集合から**基本セル**の辺の長さ(グリッド単位)を取り出す。

軸に沿った並進の最小正値を採る。見つからなければ格子いっぱい(= その方向の
並進が 1 本も残っていない = 全破れ)とみなす。
"""
function primitive_cell(shifts::Vector{NTuple{2,Int}}, lx::Int, ly::Int)
    cx = minimum([t[1] for t in shifts if t[1] > 0 && t[2] == 0]; init = lx)
    cy = minimum([t[2] for t in shifts if t[2] > 0 && t[1] == 0]; init = ly)
    return cx, cy
end

# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------

"セル境界の枠を重ねる(周期的に敷き詰める)。"
function draw_cells!(p, cw::Int, ch::Int, lx::Int, ly::Int; color, ls, lw, label)
    first = true
    for x0 = 0:cw:(lx - 1), y0 = 0:ch:(ly - 1)
        xs = [x0 - 0.5, x0 + cw - 0.5, x0 + cw - 0.5, x0 - 0.5, x0 - 0.5]
        ys = [y0 - 0.5, y0 - 0.5, y0 + ch - 0.5, y0 + ch - 0.5, y0 - 0.5]
        plot!(p, xs, ys; color = color, ls = ls, lw = lw, alpha = 0.75,
              label = first ? label : "")
        first = false
    end
    return p
end

function main(argv)
    isempty(argv) && error(
        "使い方: julia --project=tools tools/plot_parton_lattice.jl <dir> " *
        "[出力.pdf] [--label ef|site] [--shells N]")

    dir = argv[1]
    label_mode = :ef
    nshell = 1
    out = nothing
    i = 2
    while i <= length(argv)
        if argv[i] == "--label"
            label_mode = Symbol(argv[i + 1]); i += 2
        elseif argv[i] == "--shells"
            nshell = parse(Int, argv[i + 1]); i += 2
        else
            out = argv[i]; i += 1
        end
    end
    label_mode in (:ef, :site) || error("--label は ef / site。label = $(label_mode)")

    # run ディレクトリを渡されたら stage*_in を探す
    src = joinpath(dir, "parton_lattice.dat")
    if !isfile(src)
        cands = sort([joinpath(dir, d, "parton_lattice.dat") for d in readdir(dir)
                      if isdir(joinpath(dir, d)) &&
                         isfile(joinpath(dir, d, "parton_lattice.dat"))])
        isempty(cands) && error("parton_lattice.dat が見つかりません: $(dir)")
        src = cands[1]
    end
    L = read_lattice_meta(src)
    out === nothing && (out = joinpath(dirname(src), "parton_lattice.pdf"))

    lx, ly = L.meta["grid"]
    ex, ey = L.meta["enlarged_cell"]
    sx, sy = L.meta["cell_step"]
    F = L.meta["nflavor"][1]
    n_idx = L.meta["n_idx"][1]
    n_ef = L.meta["n_ef"][1]

    pcx, pcy = primitive_cell(L.phys, lx, ly)     # 物理の磁気セル(グリッド)
    mcx, mcy = primitive_cell(L.mfs, lx, ly)      # 拡大セル(グリッド)
    ratio = (mcx * mcy) / (pcx * pcy)

    xs = [s[2] for s in L.sites]
    ys = [s[3] for s in L.sites]
    efs = [s[4] for s in L.sites]
    labels = label_mode === :ef ? efs : [s[1] for s in L.sites]

    pal = distinguishable_colors(max(n_ef, 2), [RGB(1, 1, 1)]; dropseed = true)
    cols = [pal[e + 1] for e in efs]

    # --- 左: EF クラスとセル ---
    p1 = scatter(xs, ys; marker_z = nothing, color = cols, ms = 7,
                 markerstrokecolor = :black, markerstrokewidth = 0.6,
                 legend = :outertop, legendfontsize = 7, label = "",
                 xlabel = "x (grid)", ylabel = "y (grid)",
                 title = @sprintf("EF classes: %d   |   enlarged cell %dx%d = %g x physical %dx%d",
                                  n_ef, mcx, mcy, ratio, pcx, pcy),
                 titlefontsize = 9, aspect_ratio = :equal,
                 xlims = (-1, lx), ylims = (-1, ly), grid = true, gridalpha = 0.25)
    draw_cells!(p1, mcx, mcy, lx, ly; color = :black, ls = :solid, lw = 2.4,
                label = @sprintf("enlarged cell %dx%d (mean field)", mcx, mcy))
    draw_cells!(p1, pcx, pcy, lx, ly; color = :steelblue, ls = :dashdot, lw = 1.4,
                label = @sprintf("magnetic cell %dx%d (physical H)", pcx, pcy))
    for (k, s) in enumerate(L.sites)
        annotate!(p1, s[2] + 0.22, s[3] + 0.02, text(string(labels[k]), 7, :left))
    end

    # --- 右: ボンドクラス ---
    shells = sort(unique([b[4] for b in L.bonds]))
    keep = Set(shells[1:min(nshell, length(shells))])
    shown = [b for b in L.bonds if b[4] in keep]
    pos = Dict(s[1] => (s[2], s[3]) for s in L.sites)
    idxs = sort(unique([b[3] for b in shown]))
    bpal = distinguishable_colors(max(length(idxs), 2), [RGB(1, 1, 1)]; dropseed = true)
    ccol = Dict(k => bpal[n] for (n, k) in enumerate(idxs))

    p2 = scatter(xs, ys; color = :gray70, ms = 4, markerstrokewidth = 0,
                 label = "", xlabel = "x (grid)", ylabel = "y (grid)",
                 aspect_ratio = :equal, xlims = (-1, lx), ylims = (-1, ly),
                 legend = :outertop, legendfontsize = 7, grid = true, gridalpha = 0.25,
                 title = @sprintf("bond classes (shells d²=%s): %d idx / %d total, frozen %d",
                                  join(sort(collect(keep)), ","), length(idxs), n_idx,
                                  length(L.frozen)),
                 titlefontsize = 9)
    for (s1, s2, idx, _) in shown
        (x1, y1) = pos[s1]; (x2, y2) = pos[s2]
        # トーラスで巻くボンドは描かない(直線が格子を横断して読めなくなる)
        (abs(x1 - x2) <= max(lx ÷ 2, 1) && abs(y1 - y2) <= max(ly ÷ 2, 1)) || continue
        (abs(x1 - x2) > 2 || abs(y1 - y2) > 2) && continue
        frozen = idx in L.frozen
        plot!(p2, [x1, x2], [y1, y2]; color = ccol[idx], lw = frozen ? 3.0 : 1.6,
              ls = frozen ? :dash : :solid, alpha = 0.9, label = "")
    end
    # QP 並進を矢印で
    for (n, (tx, ty)) in enumerate(L.qps)
        (tx == 0 && ty == 0) && continue
        plot!(p2, [0.0, Float64(tx)], [-0.7, -0.7 + Float64(ty)];
              color = :crimson, lw = 2.5, arrow = :arrow,
              label = @sprintf("QP shift (%d, %d)", tx, ty))
    end

    plt = plot(p1, p2; layout = (1, 2), size = (1500, 720),
               left_margin = 8Plots.mm, bottom_margin = 6Plots.mm,
               top_margin = 4Plots.mm,
               plot_title = @sprintf("%s   Nsite=%d  F=%d   |   translations: physical %d / mean-field %d / QP %d",
                                     L.model, L.meta["nsite"][1], F,
                                     length(L.phys), length(L.mfs), length(L.qps)),
               plot_titlefontsize = 11)
    savefig(plt, out)
    @printf("図: %s   (拡大セル %d×%d = %g × 物理 %d×%d、QP 位数 %d = F %d)\n",
            out, mcx, mcy, ratio, pcx, pcy, length(L.qps), F)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
