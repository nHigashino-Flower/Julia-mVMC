#=
パートン段出力の収束診断図(E と min_gap の 4 パネル)
--- parton-mode (fork addition) ---

    julia --project=tools tools/plot_parton_conv.jl <stage_out_dir> [出力.pdf] [--window 100]

<stage_out_dir> は zvo_out.dat と zvo_parton_diag.dat を含むディレクトリ
(例 runs_orientfix/L08_ef4_s1009/stage1_out)。走行中の run にもそのまま使える。
既定の出力は **<stage_out_dir>/parton_conv.pdf**(図は PDF 統一・保存先は
その run の SR 出力と同じディレクトリ、という運用規約)。

パネル:
  1. E vs step(生 + window 移動平均)と min_gap vs step(右軸)
  2. **log10|X(s) − X_tail| vs step** ← 収束の指数的接近を見る本命。
     直線 = 指数緩和。傾きから時定数 τ(e 倍縮むのに要する step 数)を推定して
     注記する。E は移動平均で評価(生は MC ノイズ床で潰れる)、X_tail は
     末尾 window の平均。**末尾 window 分は描かない**(tail の定義窓と重なって
     距離が自己参照になり、最終点はビット差 ~1e-14 の滝になるため)
  3. 変化率 |ΔX|/100step vs step(両対数)。stage_io.jl の収束判定と同じ閾値
     (E: 10·SE、min_gap: 末尾平均の 5%)を破線で重ねる —
     **曲線が破線を割った step が、その条件を満たした瞬間**
  4. min_gap の符号と占有ずれ(n_occ_deviation)。金属/絶縁体の判別

なぜ E だけでは足りないか(2026-08-18 の実測、stage_io.jl の条件 5/6 の根拠):
SR は自然勾配 S⁻¹g で「E をわずかしか下げない柔らかいモード」も普通の速さで
緩和し続ける。緩和の残り振幅 q に対し ΔE = O(q²)、Δmin_gap = O(q) なので、
E は 1 桁早くノイズ床に沈んで「収束」に見えるが、min_gap はまだ動いている。
パネル 2 で両者の log 距離を並べると、この 2 乗の関係(E の傾き ≈ gap の傾き × 2)
がそのまま見える。
=#

using Printf

# ---------------------------------------------------------------------------
# 読み込み(tools は playground に依存しない方針なので readers は自前)
# ---------------------------------------------------------------------------

"数値データ行の指定列を読む(# とヘッダ行はスキップ)。"
function read_col(path::AbstractString, col::Int)
    out = Float64[]
    isfile(path) || return out
    for ln in eachline(path)
        s = strip(ln)
        (isempty(s) || startswith(s, "#") || startswith(s, "=")) && continue
        t = split(s)
        length(t) >= col || continue
        v = tryparse(Float64, t[col])
        v === nothing || push!(out, v)
    end
    return out
end

"window 移動平均(先頭 window-1 は NaN)。"
function moving_avg(x::Vector{Float64}, w::Int)
    out = fill(NaN, length(x))
    s = 0.0
    for i in eachindex(x)
        s += x[i]
        i > w && (s -= x[i-w])
        i >= w && (out[i] = s / w)
    end
    return out
end

"log10 距離の直線フィットで時定数 τ [step/e-fold] を出す。範囲は距離が
最大値の 1/3 〜 ノイズ床の 10 倍の区間。点が少なければ NaN。"
function fit_tau(steps::Vector{Int}, dist::Vector{Float64}, floor_::Float64)
    finite = filter(isfinite, dist)
    isempty(finite) && return NaN
    dmax = maximum(finite)          # 注意: `maximum(...; init=NaN)` は NaN が
    isfinite(dmax) || return NaN    # 全体を汚染するので使わない
    idx = [i for i in eachindex(dist)
           if isfinite(dist[i]) && dist[i] > 0 &&
              dist[i] < dmax / 3 && dist[i] > 10 * floor_]
    length(idx) >= 30 || return NaN
    xs = Float64.(steps[idx]); ys = log10.(dist[idx])
    n = length(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    slope = sum((xs .- mx) .* (ys .- my)) / sum((xs .- mx) .^ 2)
    slope < 0 || return NaN
    return -1 / (slope * log(10))     # log10 → ln 換算
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

function main(argv)
    length(argv) >= 1 || error(
        "usage: julia --project=tools tools/plot_parton_conv.jl <stage_out_dir> " *
        "[out.pdf] [--window 100]")
    dir = argv[1]
    png = nothing
    window = 100
    i = 2
    while i <= length(argv)
        if argv[i] == "--window"
            window = parse(Int, argv[i+1]); i += 2
        else
            png = argv[i]; i += 1
        end
    end
    png === nothing && (png = joinpath(dir, "parton_conv.pdf"))

    e = read_col(joinpath(dir, "zvo_out.dat"), 1)
    gap = read_col(joinpath(dir, "zvo_parton_diag.dat"), 2)
    dev = read_col(joinpath(dir, "zvo_parton_diag.dat"), 10)
    isempty(e) && error("zvo_out.dat が読めません: $dir")
    isempty(gap) && error("zvo_parton_diag.dat が読めません: $dir")
    n = min(length(e), length(gap))
    e = e[1:n]; gap = gap[1:n]
    length(dev) >= n ? (dev = dev[1:n]) : (dev = fill(NaN, n))
    steps = collect(0:n-1)
    n > 2 * window || error("step 数 $n が window の 2 倍 $(2window) 未満です")

    ma = moving_avg(e, window)
    e_tail = sum(@view e[end-window+1:end]) / window
    e_se = let t = @view e[end-window+1:end]
        m = sum(t) / window
        sqrt(sum((t .- m) .^ 2) / (window - 1)) / sqrt(window)
    end
    gap_tail = sum(@view gap[end-window+1:end]) / window

    dE = abs.(ma .- e_tail)                   # 移動平均の tail からの距離
    dG = abs.(gap .- gap_tail)
    # 末尾 window 分は tail の定義窓と重なるため距離が自己参照になる:
    # 最終点は「同じ 100 個の数の一括和 − 走り和」で厳密ゼロのはずがビット差
    # (~1e-14)だけ残り、log 図で滝になる。手前の ~window 点も重なりの分だけ
    # 人工的に小さく出る。距離曲線からは落とす(フィット・図とも)
    dE[end-window+1:end] .= NaN
    dG[end-window+1:end] .= NaN
    # ノイズ床: E は移動平均の統計誤差、gap は末尾 window の標準偏差
    e_floor = e_se
    g_floor = let t = @view gap[end-window+1:end]
        m = sum(t) / window
        max(sqrt(sum((t .- m) .^ 2) / (window - 1)), 1e-12)
    end
    tauE = fit_tau(steps, dE, e_floor)
    tauG = fit_tau(steps, dG, g_floor)

    # 変化率 |ΔX|/100step(100 step 間隔の移動平均差 / gap 差)
    stride = 100
    r_steps = Int[]; rE = Float64[]; rG = Float64[]
    for s in (window + stride):stride:n-1
        push!(r_steps, s)
        push!(rE, abs(ma[s+1] - ma[s+1-stride]))
        push!(rG, abs(gap[s+1] - gap[s+1-stride]))
    end
    thrE = 10 * e_se                           # stage_io 条件 2
    thrG = 0.05 * abs(gap_tail)                # stage_io 条件 6

    @printf("step %d  window %d\n", n, window)
    @printf("E_tail = %.5f ± %.1e   gap_tail = %+.4f\n", e_tail, e_se, gap_tail)
    @printf("τ_E = %s step/e-fold   τ_gap = %s step/e-fold",
            isfinite(tauE) ? @sprintf("%.0f", tauE) : "n/a",
            isfinite(tauG) ? @sprintf("%.0f", tauG) : "n/a")
    if isfinite(tauE) && isfinite(tauG)
        @printf("   (比 τ_E/τ_gap = %.2f、ΔE = O(q²) なら 0.5 が期待値)", tauE / tauG)
    end
    println()

    # -----------------------------------------------------------------------
    ENV["GKSwstype"] = "100"
    @eval using Plots
    Base.invokelatest() do
        gr()
        clean(x) = [v > 0 && isfinite(v) ? v : NaN for v in x]

        p1 = plot(steps, e; alpha = 0.25, color = :gray, label = "E (raw)",
                  xlabel = "step", ylabel = "E", legend = :topright,
                  title = @sprintf("E_tail = %.5f    gap_tail = %+.4f", e_tail, gap_tail))
        plot!(p1, steps, ma; color = :crimson, lw = 2, label = "E ($(window)-step avg)")
        hline!(p1, [e_tail]; color = :crimson, ls = :dash, alpha = 0.5, label = "")
        pg = twinx(p1)
        plot!(pg, steps, gap; color = :steelblue, lw = 2, label = "min_gap",
              ylabel = "min_gap", legend = :bottomright)
        hline!(pg, [0.0]; color = :steelblue, ls = :dot, alpha = 0.5, label = "")

        # 生 E の距離(薄い散布): 1 step ノイズ床 σ の帯がそのまま見える。
        # 主曲線は移動平均: 感度が √window 倍深く、指数緩和の傾き(τ)は
        # 線形フィルタなので厳密に保存される
        dE_raw = abs.(e .- e_tail)
        p2 = scatter(steps, clean(dE_raw); yscale = :log10, color = :salmon,
                     ms = 1.2, msw = 0, alpha = 0.35, label = "|E_raw − E_tail|",
                     xlabel = "step", ylabel = "|X − X_tail|", legend = :bottomleft,
                     title = "log distance to tail (straight line = exponential)")
        plot!(p2, steps, clean(dE); yscale = :log10, color = :crimson, lw = 1.5,
              label = @sprintf("|E_avg − E_tail|  (τ=%s)",
                               isfinite(tauE) ? @sprintf("%.0f", tauE) : "n/a"))
        plot!(p2, steps, clean(dG); yscale = :log10, color = :steelblue, lw = 1.5,
              label = @sprintf("|min_gap − gap_tail|  (τ=%s)",
                               isfinite(tauG) ? @sprintf("%.0f", tauG) : "n/a"))
        hline!(p2, [e_floor]; color = :crimson, ls = :dot, alpha = 0.6,
               label = "E noise floor (SE)")
        hline!(p2, [g_floor]; color = :steelblue, ls = :dot, alpha = 0.6,
               label = "gap noise floor")

        p3 = plot(r_steps, clean(rE); yscale = :log10, color = :crimson, lw = 1.5,
                  marker = :circle, ms = 2.5, label = "|ΔE| / $(stride) step",
                  xlabel = "step", ylabel = "|ΔX| / $(stride) step", legend = :bottomleft,
                  title = "rate vs stage_io thresholds (#2 / #6)")
        plot!(p3, r_steps, clean(rG); yscale = :log10, color = :steelblue, lw = 1.5,
              marker = :circle, ms = 2.5, label = "|Δmin_gap| / $(stride) step")
        hline!(p3, [thrE]; color = :crimson, ls = :dash,
               label = @sprintf("10·SE = %.1e", thrE))
        hline!(p3, [thrG]; color = :steelblue, ls = :dash,
               label = @sprintf("5%% of gap_tail = %.1e", thrG))

        p4 = plot(steps, gap; color = :steelblue, lw = 2, label = "min_gap",
                  xlabel = "step", ylabel = "min_gap", legend = :topleft,
                  title = "min_gap sign / occupation deviation")
        hline!(p4, [0.0]; color = :black, ls = :dash, alpha = 0.6, label = "")
        if any(isfinite, dev)
            pd = twinx(p4)
            plot!(pd, steps, dev; color = :darkorange, lw = 1.2, alpha = 0.8,
                  label = "n_occ_deviation", ylabel = "n_occ_dev",
                  legend = :topright)
        end

        plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1400, 900),
                   left_margin = 8Plots.mm, bottom_margin = 5Plots.mm,
                   right_margin = 12Plots.mm)
        savefig(plt, png)
    end
    println("図: $png")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
