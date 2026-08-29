#=
パートン平均場の C4 対称性の可視化(3D サーフェス + Γ-X-M-Y-Γ + 差分マップ)
--- parton-mode (fork addition) ---

    julia --project=tools tools/plot_parton_c4_bands.jl <stage_out_dir> [出力.pdf] \
        [--nx 8] [--ny 8] [--ex 2] [--ey 2] [--nflavor 2] [--ne 32] [--ngrid 48] [--flavor 1]

<stage_out_dir> は `zqp_pmfham_opt.dat` を含むディレクトリ
(例 runs_c4L08/L08_ef4_fsym_c4n2_s1002/stage1_out)。
既定の出力は **<stage_out_dir>/parton_c4_bands.pdf**
(`parton_band_chern.jl` が出す `parton_bands.pdf` / `parton_bands_3d.pdf` とは別名。
あちらの経路図は Γ-X-M-Γ で **Y を通らない**ので C4 破れが見えない)(図は PDF 統一・保存先はその run の
出力と同じディレクトリ、という運用規約)。

パネル:
  1. **3D サーフェス** E_n(kx,ky) 全バンド。参照VMC `mf_parton_band.jl` の
     `Plots.surface` と同じ見え方。ここで 4 回対称の破れが目で見える
  2. **経路バンド図 Γ→X(π,0)→M(π,π)→Y(0,π)→Γ**。`parton_bands.jl:bz_path` は
     Γ-X-M-Γ で **Y を通らない**ので C4 破れが原理的に見えない。この経路なら
     X と Y が非等価であることが直接読める(C4 対称なら E(X) = E(Y))
  3. 占有バンド(最低 n_occ 本)の **C4 差分マップ** E_n(k) − E_n(C4 k)。
     C4 は k → (−ky, kx)(実空間 g の半並進は k を変えず位相にしか効かない)。
     C4 対称なら恒等的に 0

副格子込み C4 が CB の対称性であることの確認や、band_c4 の定量値は
`playground_nozomi/cb_nu12_boson/scripts/c4_band_asym.jl` を参照。
=#

include(joinpath(@__DIR__, "parton_bands.jl"))
using .PartonBands: read_pmfham, fold_hoppings, hk
using LinearAlgebra, Printf

function parse_args(argv)
    length(argv) >= 1 || error(
        "使い方: julia --project=tools tools/plot_parton_c4_bands.jl <stage_out_dir> " *
        "[出力.pdf] [--nx N] [--ny N] [--ex N] [--ey N] [--nflavor N] [--ne N] " *
        "[--ngrid N] [--flavor N]")
    dir = argv[1]
    out = length(argv) >= 2 && !startswith(argv[2], "--") ? argv[2] :
          joinpath(dir, "parton_c4_bands.pdf")
    o = Dict(:nx => 8, :ny => 8, :ex => 2, :ey => 2, :nflavor => 2, :ne => 32,
             :ngrid => 48, :flavor => 1)
    i = (length(argv) >= 2 && !startswith(argv[2], "--")) ? 3 : 2
    while i <= length(argv)
        k = Symbol(lstrip(argv[i], '-'))
        haskey(o, k) || error("未知の引数: $(argv[i])")
        o[k] = parse(Int, argv[i + 1]); i += 2
    end
    return dir, out, o
end

"C4: k → (−ky, kx)。"
c4k(kx, ky) = (-ky, kx)

"Γ→X→M→Y→Γ の経路。X=(π,0) と Y=(0,π) の非等価性が読めるように Y を通す。"
function path_xmy(; nseg::Int = 100)
    corners = [(0.0, 0.0), (pi, 0.0), (pi, pi), (0.0, pi), (0.0, 0.0)]
    labels = ["Γ", "X(π,0)", "M(π,π)", "Y(0,π)", "Γ"]
    ks = Tuple{Float64,Float64}[]; dist = Float64[]
    ticks = Tuple{Float64,String}[(0.0, labels[1])]; d = 0.0
    for s in 1:length(corners)-1
        a, b = corners[s], corners[s+1]
        len = hypot(b[1] - a[1], b[2] - a[2])
        for t in range(0.0, 1.0; length = nseg + 1)
            (s > 1 && t == 0.0) && continue
            push!(ks, (a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])))
            push!(dist, d + t * len)
        end
        d += len; push!(ticks, (d, labels[s+1]))
    end
    return ks, dist, ticks
end

function main(argv)
    dir, out, o = parse_args(argv)
    ENV["GKSwstype"] = "100"
    @eval using Plots
    Base.invokelatest(gr)

    nx, ny, ex, ey = o[:nx], o[:ny], o[:ex], o[:ey]
    nsite = 2 * nx * ny
    H = read_pmfham(joinpath(dir, "zqp_pmfham_opt.dat");
                    nsite = nsite, nflavor = o[:nflavor])
    # フレーバー間の差(indep では 0 でない)を注記用に測る
    fdev = o[:nflavor] >= 2 ? maximum(abs, H[1] - H[2]) / maximum(abs, H[1]) : 0.0
    Hf = H[o[:flavor]]
    h, res = fold_hoppings(Hf, nx, ny, ex, ey)
    n1, n2 = div(nx, ex), div(ny, ey)
    norb = 2 * ex * ey
    n_occ = max(1, div(o[:ne], n1 * n2))       # 占有バンド本数

    # --- パネル 1: 3D サーフェス ---
    ng = o[:ngrid]
    kv = [2pi * a / ng for a = 0:ng-1]
    E = Array{Float64}(undef, ng, ng, norb)
    dev = 0.0
    for (a, kx) in enumerate(kv), (b, ky) in enumerate(kv)
        E[a, b, :] = eigvals(hk(h, kx, ky))
        e2 = eigvals(hk(h, c4k(kx, ky)...))
        dev = max(dev, maximum(abs.(E[a, b, :] .- e2)))
    end
    width = maximum(E) - minimum(E)
    p1 = Base.invokelatest(surface, kv, kv, E[:, :, 1]';
                           xlabel = "kx", ylabel = "ky", zlabel = "E",
                           title = "all bands E_n(k)   band_c4 = $(@sprintf("%.4f", dev))  ($(@sprintf("%.1f%%", 100dev/width)) of width)",
                           titlefontsize = 8, colorbar = false, legend = false)
    for n = 2:norb
        Base.invokelatest(surface!, p1, kv, kv, E[:, :, n]'; colorbar = false, legend = false)
    end

    # --- パネル 2: 経路バンド Γ-X-M-Y-Γ ---
    ks, dist, ticks = path_xmy()
    B = Matrix{Float64}(undef, length(ks), norb)
    for (i, k) in enumerate(ks)
        B[i, :] = eigvals(hk(h, k[1], k[2]))
    end
    p2 = Base.invokelatest(plot, dist, B;
                           legend = false, xlabel = "", ylabel = "E",
                           title = "Γ-X-M-Y-Γ   (C4 symmetric => E(X) = E(Y))",
                           titlefontsize = 8, lw = 1.2,
                           xticks = ([t[1] for t in ticks], [t[2] for t in ticks]))
    for t in ticks
        Base.invokelatest(vline!, p2, [t[1]]; color = :gray, alpha = 0.4, lw = 0.6, label = "")
    end
    # X と Y での占有バンド端の差を注記
    ix = argmin(abs.(dist .- ticks[2][1])); iy = argmin(abs.(dist .- ticks[4][1]))
    dxy = maximum(abs.(B[ix, 1:n_occ] .- B[iy, 1:n_occ]))
    Base.invokelatest(annotate!, p2, dist[end] * 0.5, maximum(B),
                      Base.invokelatest(text, "occupied $(n_occ) bands: max|E(X) - E(Y)| = $(@sprintf("%.4f", dxy))", 7, :top))

    # --- パネル 3: 占有バンドの C4 差分マップ ---
    D = zeros(ng, ng)
    for (a, kx) in enumerate(kv), (b, ky) in enumerate(kv)
        e2 = eigvals(hk(h, c4k(kx, ky)...))
        D[a, b] = maximum(abs.(E[a, b, 1:n_occ] .- e2[1:n_occ]))
    end
    p3 = Base.invokelatest(heatmap, kv, kv, D';
                           xlabel = "kx", ylabel = "ky",
                           title = "occupied $(n_occ) bands: |E(k) - E(C4 k)|   max = $(@sprintf("%.4f", maximum(D)))",
                           titlefontsize = 8, aspect_ratio = 1)

    lay = Base.invokelatest(grid, 1, 3)
    ttl = "$(basename(dirname(dir)))  flavor=$(o[:flavor])  " *
          "translation residual $(@sprintf("%.2e", res))   flavor diff $(@sprintf("%.2e", fdev))"
    p = Base.invokelatest(plot, p1, p2, p3; layout = lay, size = (1500, 480),
                          plot_title = ttl, plot_titlefontsize = 9)
    Base.invokelatest(savefig, p, out)
    @printf("保存: %s\n", out)
    @printf("  band_c4 = %.4f (幅 %.3f の %.1f%%) / 占有バンド |E(X)−E(Y)|max = %.4f / 並進残差 %.2e\n",
            dev, width, 100dev / width, dxy, res)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
