#=
パートン平均場のバンド図 + Chern 数を 1 run 分出す CLI
--- parton-mode (fork addition) ---

    julia --project=tools tools/parton_band_chern.jl <out_dir> \
        --nux 8 [--nuy 8] --ansatz ef4|xexet2|ef9|xexet3 [--grid 32] [--out out.pdf]

<out_dir> は zqp_pmfham_opt.dat / zqp_pmfband_opt.dat / zqp_pmfocc_opt.dat を含む
ディレクトリ(例 runs/L08_ef4_s1001/stage1_out)。

やること(フレーバーごと):
 1. 実空間 H_MF を拡大セルで畳み(並進不変性の残差を検査)、任意 k の H(k) を得る
 2. Γ-X-M-Γ のバンド図を描く(native k 点の固有値を重ね、占有/非占有を色分け)
 3. native グリッドの固有値が zqp_pmfband_opt.dat と一致することを検算
 4. Chern:
    - バンド分解(最低バンドから 1 本ずつ、および占有多様体まとめて)を
      細かいグリッドで(--grid、既定 32)
    - 参照VMC PartonChern.jl と同構成の射影行列経路(native グリッド、
      実際の占有集合を使用)で「系のパートンの Chern」
    xexet2 は占有 = 最低 1 バンド/k なので両者は band1 の C と一致するはず。
    ef4 はバンド数が倍(8 本)で占有 2 バンド/k なので、最低 2 バンドの総 C が
    射影経路の C と一致するはず。
=#

include(joinpath(@__DIR__, "parton_bands.jl"))

using .PartonBands
using LinearAlgebra, Printf

# ---------------------------------------------------------------------------
# 引数
# ---------------------------------------------------------------------------

function parse_args(argv)
    length(argv) >= 1 || error(
        "usage: julia --project=tools tools/parton_band_chern.jl <out_dir> " *
        "--nux N [--nuy N] --ansatz ef4|xexet2|ef9|xexet3 [--grid 32] [--out out.pdf]")
    dir = argv[1]
    nux = nothing; nuy = nothing; ansatz = nothing; grid = 32; png = nothing
    i = 2
    while i <= length(argv)
        a = argv[i]
        if a == "--nux"
            nux = parse(Int, argv[i+1]); i += 2
        elseif a == "--nuy"
            nuy = parse(Int, argv[i+1]); i += 2
        elseif a == "--ansatz"
            ansatz = Symbol(argv[i+1]); i += 2
        elseif a == "--grid"
            grid = parse(Int, argv[i+1]); i += 2
        elseif a == "--out"
            png = argv[i+1]; i += 2
        else
            error("未知の引数: $a")
        end
    end
    nux === nothing && error("--nux は必須です")
    nuy === nothing && (nuy = nux)
    ansatz === nothing && error("--ansatz は必須です (ef4 | xexet2 | ef9 | xexet3)")
    EXEY = Dict(:ef4 => (2, 2), :xexet2 => (2, 1), :ef9 => (3, 3), :xexet3 => (3, 1))
    haskey(EXEY, ansatz) || error("未対応のアンザッツ: $ansatz")
    ex, ey = EXEY[ansatz]
    png === nothing && (png = joinpath(dir, "parton_bands.pdf"))
    return (dir = dir, nux = nux, nuy = nuy, ansatz = ansatz,
            ex = ex, ey = ey, grid = grid, png = png)
end

# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------

function main(argv)
    o = parse_args(argv)
    band = read_pmfband(joinpath(o.dir, "zqp_pmfband_opt.dat"))
    Hs = read_pmfham(joinpath(o.dir, "zqp_pmfham_opt.dat");
                     nsite = band.nsite, nflavor = band.nflavor)
    occs = read_pmfocc(joinpath(o.dir, "zqp_pmfocc_opt.dat"); nflavor = band.nflavor)
    n1, n2 = div(o.nux, o.ex), div(o.nuy, o.ey)
    nk = n1 * n2
    norb = 2 * o.ex * o.ey

    @printf("=== %s : %s (ex,ey)=(%d,%d) ===\n", o.dir, o.ansatz, o.ex, o.ey)
    @printf("Nsite=%d NElec=%d NFlavor=%d | native grid %d×%d (nk=%d), n_orb=%d, 占有 %d バンド/k\n",
            band.nsite, band.nelec, band.nflavor, n1, n2, nk, norb,
            div(band.nelec, nk))

    # n1 または n2 が 2 以下だと fold_hoppings の最小イメージ規約が原理的に破れる
    # (2 セルしかない方向では隣接セルへのホッピング ΔR=+1 と −1 が区別できない)。
    # この場合は H(k) 細グリッド経路(バンド図・バンド分解 Chern)をまるごとスキップし、
    # fold_hoppings を経由しない射影経路(fhs_chern_projector)と Bott 指数だけを出す。
    skip_hk = n1 < 3 || n2 < 3
    if skip_hk
        @info "native grid が $(n1)×$(n2) で fold_hoppings の最小イメージ規約が" *
              "破れるため、H(k) 細グリッド経路(バンド図・バンド分解 Chern)を" *
              "スキップします。射影経路の Chern と Bott 指数のみ出します。"
    end

    ks, dist, ticks = bz_path(nseg = 120)
    results = []                     # フレーバーごとの (folded h, path bands, chern...)

    for f in 1:band.nflavor
        println("\n--- flavor $(f-1) ---")

        # 占有集合の確認(mom 追跡だと最低 Ne 本からずれうる)
        aufbau = collect(0:band.nelec-1)
        is_aufbau = sort(occs[f]) == aufbau
        is_aufbau || @warn "占有集合が最低 $(band.nelec) 本ではありません" occs[f]

        h = nothing; band_C = Float64[]; C_occ = NaN; gap_ok = false
        if !skip_hk
            h, resid = fold_hoppings(Hs[f], o.nux, o.nuy, o.ex, o.ey)
            @printf("並進不変性の残差(畳み込み時の最大ばらつき): %.3e\n", resid)
            resid < 1e-8 || @warn "H_MF が拡大セル並進で不変になっていません" resid

            # 検算: native グリッドの固有値 vs zqp_pmfband_opt.dat
            eig_native = native_grid_eigs(h, n1, n2)
            eig_file = sort(band.eigs[f])
            dev = maximum(abs.(eig_native .- eig_file))
            @printf("native 固有値 vs pmfband: 最大偏差 %.3e\n", dev)
            dev < 1e-8 || @warn "ブロッホ化した H(k) の固有値が pmfband と一致しません" dev

            # Chern: バンド分解(細グリッド)
            n_occ = div(band.nelec, nk)
            println("Chern(H(k) 経路, grid $(o.grid)×$(o.grid)):")
            for b in 1:norb
                r = fhs_chern_bands(h, [b]; n1 = o.grid, n2 = o.grid)
                push!(band_C, r.C)
                marker = b <= n_occ ? " ← 占有" : ""
                @printf("  band %d: C = %+.6f  (min|link| %.3e)%s\n",
                        b, r.C, r.min_abs_link_det, marker)
            end
            r_occ = fhs_chern_bands(h, collect(1:n_occ); n1 = o.grid, n2 = o.grid)
            @printf("  最低 %d バンド多様体: C = %+.6f  (min|link| %.3e)\n",
                    n_occ, r_occ.C, r_occ.min_abs_link_det)
            C_occ = r_occ.C
            gap_ok = dev < 1e-8
        else
            println("Chern(H(k) 経路): スキップ(native $(n1)×$(n2) が小さすぎます)")
        end

        # Chern: 射影行列経路(native グリッド、実占有集合。fold_hoppings を経由しない)
        rp = fhs_chern_projector(Hs[f], occs[f], o.nux, o.nuy, o.ex, o.ey)
        @printf("Chern(射影経路, native %d×%d, 実占有集合): C = %+.6f  (min|link| %.3e, max‖p²−p‖ %.3e)\n",
                rp.grid[1], rp.grid[2], rp.C, rp.min_abs_link_det,
                rp.max_projector_error)
        if !skip_hk
            agree = abs(C_occ - rp.C) < 0.05
            @printf("  → 占有多様体 C(H(k) 経路) と系のパートン C(射影経路): %s (差 %.2e)\n",
                    agree ? "一致" : "不一致!", abs(C_occ - rp.C))
        end

        # Bott index(実空間、k グリッドサイズに依存しない独立経路)
        rb = bott_index(Hs[f], occs[f], o.nux, o.nuy, o.ex, o.ey)
        @printf("Bott(vuvu) = %+.6f  Bott(uvuv) = %+.6f  (‖[Ũ,Ṽ]‖ %.3e)\n",
                rb.B_vuvu, rb.B_uvuv, rb.commutator)

        push!(results, (h = h, band_C = band_C, C_occ = C_occ, C_proj = rp.C,
                        n_occ = div(band.nelec, nk), gap_ok = gap_ok))
    end

    if skip_hk
        println("\nバンド図・3D バンド図はスキップしました(H(k) 経路が使えないため)。")
        return nothing
    end

    # ---------------------------------------------------------------------
    # 作図(フレーバーごとに 1 パネル)
    # ---------------------------------------------------------------------
    @eval begin
        using Plots
        ENV["GKSwstype"] = "100"
        gr()
    end
    Base.invokelatest(_plot, o, band, results, ks, dist, ticks, n1, n2)
    println("\n図: $(o.png)")
    png3d = replace(o.png, r"\.pdf$" => "") * "_3d.pdf"
    Base.invokelatest(_plot3d, o, band, results, png3d)
    println("図(3D): $png3d")
    return nothing
end

"""
kx–ky–E の 3D バンド曲面(フレーバーごとに 1 パネル)。
占有バンドは赤系、非占有は青系。k は換算座標で [−π, π]²。
"""
function _plot3d(o, band, results, png3d)
    nf = band.nflavor
    ng = 48
    kg = range(-pi, pi; length = ng + 1)
    panels = []
    for f in 1:nf
        r = results[f]
        norb = 2 * o.ex * o.ey
        # bandsurf[b][i,j] = E_b(kg[i], kg[j])
        bandsurf = [Matrix{Float64}(undef, ng + 1, ng + 1) for _ in 1:norb]
        for (i, kx) in enumerate(kg), (j, ky) in enumerate(kg)
            e = LinearAlgebra.eigvals(hk(r.h, Float64(kx), Float64(ky)))
            for b in 1:norb
                bandsurf[b][i, j] = e[b]
            end
        end
        p = Plots.plot(; title = @sprintf("flavor %d   C_occ = %+.3f", f - 1, r.C_occ),
                       xlabel = "kx", ylabel = "ky", zlabel = "E",
                       legend = false, camera = (50, 25))
        for b in norb:-1:1        # 上のバンドから描くと手前の占有バンドが隠れない
            Plots.surface!(p, kg, kg, bandsurf[b]';
                           color = b <= r.n_occ ? :reds : :blues,
                           alpha = b <= r.n_occ ? 1.0 : 0.55,
                           colorbar = false)
        end
        push!(panels, p)
    end
    plt = Plots.plot(panels...; layout = (1, nf), size = (820 * nf, 640),
                     left_margin = 6Plots.mm)
    Plots.savefig(plt, png3d)
end

function _plot(o, band, results, ks, dist, ticks, n1, n2)
    nf = band.nflavor
    panels = []
    for f in 1:nf
        r = results[f]
        eb = bands_on_path(r.h, ks)
        norb = size(eb, 2)
        p = Plots.plot(; title = @sprintf("flavor %d   C_occ = %+.3f (parton C = %+.3f)",
                                          f - 1, r.C_occ, r.C_proj),
                       xlabel = "", ylabel = "E", legend = :topright,
                       xticks = ([t[1] for t in ticks], [t[2] for t in ticks]),
                       grid = true, framestyle = :box)
        for b in 1:norb
            lab = b == 1 ? "band C (bottom to top): " *
                  join([@sprintf("%+.2f", c) for c in r.band_C], ", ") : ""
            Plots.plot!(p, dist, eb[:, b];
                        color = b <= r.n_occ ? :crimson : :steelblue,
                        lw = b <= r.n_occ ? 2.2 : 1.2, label = lab)
        end
        # native k 点の固有値を重ねる(この有限系が実際に使う状態)
        nx_pts = Float64[]; ny_pts = Float64[]
        for (i, k) in enumerate(ks)
            # 経路上の点が native グリッドに載っているか
            on1 = abs(rem(k[1] * n1 / (2pi), 1.0, RoundNearest)) < 1e-9
            on2 = abs(rem(k[2] * n2 / (2pi), 1.0, RoundNearest)) < 1e-9
            if on1 && on2
                eigs = eigvals(hk(r.h, k[1], k[2]))
                append!(nx_pts, fill(dist[i], length(eigs)))
                append!(ny_pts, eigs)
            end
        end
        Plots.scatter!(p, nx_pts, ny_pts; color = :black, ms = 2.5,
                       label = "native k ($(n1)×$(n2))")
        push!(panels, p)
    end
    plt = Plots.plot(panels...; layout = (1, nf), size = (720 * nf, 520),
                     left_margin = 8Plots.mm, bottom_margin = 6Plots.mm)
    Plots.savefig(plt, o.png)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
