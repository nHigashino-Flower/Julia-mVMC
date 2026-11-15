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

--- PartonMode = 2(Pf)の出力にも対応(2026-11-01) ---

    julia --project=tools tools/plot_parton_conv.jl <out_dir> [出力.pdf] [--window 100] [--dt 0.003]

<out_dir> に zvo_parton_diag.dat が無く zvo_parton_pairdiag.dat がある場合(Pf モード)は、
(zvo_parton_orbdiag.dat がある場合(Φ 直接変分モード、2026-11-10)も同じ図。4 枚目は log10 cond(Φ_f)、8 枚目は捨てたサンプル数で、
構造の判定は「Φ モードは判定しない」と出す)
min_gap の代わりに次を描く。PartonMode = 1 の図と標準出力は従来と同じ(この経路には入らない)。
  1. E vs step(生 + 移動平均)と σ²/E²(右軸、対数)
  2. log10|E_avg − E_tail|(τ_E の推定は Mode 1 と同じ)
  3. 変化率 |ΔE|/100step と 10·SE(stage_io.jl の条件 2)
  4. フレーバーごとの行ブロック特異値比 σ_N/σ_{N+1}(パートン形 f = ΞJΞᵀ なら ∞)と受理率
  5. **dE/dτ vs step(符号つき・線形軸)と ±2·SE の帯**。各点はそれまでの 2·window step のブロック平均の
     直線当てはめ。帯が 0 をまたいだら頭打ち。縦軸は後半 6 割の値に合わせる(序盤の大きな値は枠外)
  6. 同じ量の対数版: |dE/dτ| と 2·SE(全体の桁の推移)
  7. 判定 1 の t 値(±2 の線)
  8. 判定 2: σ_N/σ_{N+1} の window 間の相対変化(5% の線、最大のフレーバー)
  1 step の差分や 100 step 平均どうしの差は終盤で雑音に埋もれる(KM 4×4 の末尾で信号/雑音 3e−4 / 0.3)ので使わない
**頭打ちの判定は虚時間あたりの傾きで行う**。τ = 2·dt·step(SR の更新 −dt·S⁻¹·2(HO − H·O) より)とし、
末尾 2·window を 10 ブロックに分けた平均の直線当てはめで dE/dτ とその t 値を出す。|t| < 2 を頭打ちとする。
step 単位の条件 2 は dt を絞った段ではまだ下がっていても通ってしまう(KM 4×4 の追い込みで実測)ので、
判定に使わない。dt は --dt、無ければ <out_dir>/../modpara.def か <stage>_in/modpara.def から読む。
注意: 頭打ちに見えた区間のあとで急に下がることがある(KM 4×4 乱数スタートで step 5000〜7000 の停滞のあと
500 step で 0.02 下がった)。判定は「その時点で有意に下がっているか」であって、極小に着いた保証ではない。
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
    dt = NaN    # Pf モードの虚時間換算にだけ使う(Mode 1 の経路では読まない)
    i = 2
    while i <= length(argv)
        if argv[i] == "--window"
            window = parse(Int, argv[i+1]); i += 2
        elseif argv[i] == "--dt"
            dt = parse(Float64, argv[i+1]); i += 2
        else
            png = argv[i]; i += 1
        end
    end
    png === nothing && (png = joinpath(dir, "parton_conv.pdf"))
    # Pf モード(PartonMode = 2)の出力: min_gap が無いので専用の図にする。Mode 1 はこの分岐に入らない
    if !isfile(joinpath(dir, "zvo_parton_diag.dat")) && isfile(joinpath(dir, "zvo_parton_pairdiag.dat"))
        return main_pair(dir, png, window, dt)
    end
    # Φ 直接変分モード(PartonOrbParam = 1)の出力: 同じ図で、特異値比の代わりに cond(Φ_f)・捨てたサンプル数を描く
    if !isfile(joinpath(dir, "zvo_parton_diag.dat")) && isfile(joinpath(dir, "zvo_parton_orbdiag.dat"))
        return main_pair(dir, png, window, dt; orb = true)
    end

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

# ---------------------------------------------------------------------------
# Pf モード(PartonMode = 2)
# ---------------------------------------------------------------------------

"""ブロック平均の直線当てはめで傾き [E/step] とその標準誤差を返す。
step ごとの E は相関しているので、ブロック平均にしてから当てはめる。"""
function block_slope(e::AbstractVector{Float64}, nblock::Int)
    b = length(e) ÷ nblock
    xs = Float64[]; ys = Float64[]
    for k in 1:nblock
        r = ((k - 1) * b + 1):(k * b)
        push!(xs, (first(r) + last(r)) / 2); push!(ys, sum(@view e[r]) / b)
    end
    mx = sum(xs) / nblock; my = sum(ys) / nblock
    sxx = sum((xs .- mx) .^ 2)
    slope = sum((xs .- mx) .* (ys .- my)) / sxx
    res = ys .- (my .+ slope .* (xs .- mx))
    return slope, sqrt(sum(res .^ 2) / (nblock - 2) / sxx)
end

"modpara.def の DSROptStepDt を探す(<out>/../modpara.def、<stage>_in/modpara.def の順)。無ければ NaN。"
function find_dt(dir::AbstractString)
    d = rstrip(abspath(dir), '/')
    for mp in (joinpath(dirname(d), "modpara.def"), replace(d, r"_out$" => "_in") * "/modpara.def")
        isfile(mp) || continue
        for ln in eachline(mp)
            t = split(strip(ln))
            length(t) >= 2 && t[1] == "DSROptStepDt" && return parse(Float64, t[2])
        end
    end
    return NaN
end

"""
Pf(`orb = false`)と Φ 直接変分モード(`orb = true`、2026-11-10 追加)の収束図と判定。
Φ モードは σ_N/σ_{N+1} に当たる量が無く、step ごとの Φ も残らない(zvo_var.dat は Jastrow のみ)ので、
**構造の判定はせず「構造: Φ モードは判定しない」と出す**(延長ループは dE/dτ だけで判定する。ユーザー判断 2026-11-10)。
図の 4 枚目は log10 cond(Φ_f)(zvo_parton_orbdiag.dat の 6+F 列目から F 本)と受理率、8 枚目は step ごとの捨てたサンプル数。
"""
function main_pair(dir::AbstractString, png::AbstractString, window::Int, dt::Float64; orb::Bool = false)
    diagfile = joinpath(dir, orb ? "zvo_parton_orbdiag.dat" : "zvo_parton_pairdiag.dat")
    e = read_col(joinpath(dir, "zvo_out.dat"), 1)
    var = read_col(joinpath(dir, "zvo_out.dat"), 4)
    acc = read_col(diagfile, 3)
    isempty(e) && error("zvo_out.dat が読めません: $dir")
    sr = Vector{Float64}[]          # Pf: フレーバーごとの σ_N/σ_{N+1}(7 列目から)
    lcond = Vector{Float64}[]       # Φ: フレーバーごとの log10 cond(Φ_f)
    skipped = Float64[]             # Φ: step ごとの捨てたサンプル数
    if orb
        # 列: step n_skipped accept n_need_recompute n_exact_recompute xmax_pre_1..F log10cond_1..F
        ncol = 0
        for ln in eachline(diagfile)
            t = split(strip(ln))
            (isempty(t) || startswith(t[1], "#")) && continue
            ncol = length(t); break
        end
        nf = (ncol - 5) ÷ 2
        for a = 1:nf
            push!(lcond, read_col(diagfile, 5 + nf + a))
        end
        skipped = read_col(diagfile, 2)
    else
        let k = 7
            while true
                c = read_col(diagfile, k)
                isempty(c) && break
                push!(sr, c); k += 1
            end
        end
    end
    n = length(e)
    steps = collect(0:n-1)
    n > 2 * window || error("step 数 $n が window の 2 倍 $(2window) 未満です")
    isfinite(dt) || (dt = find_dt(dir))

    ma = moving_avg(e, window)
    tail = @view e[end-window+1:end]
    e_tail = sum(tail) / window
    e_se = sqrt(sum((tail .- e_tail) .^ 2) / (window - 1)) / sqrt(window)
    drift = e_tail - sum(@view e[end-2window+1:end-window]) / window
    slope, slope_se = block_slope(e[end-2window+1:end], 10)
    tval = slope / slope_se
    plateau = abs(tval) < 2
    v_tail = isempty(var) ? NaN : sum(@view var[end-window+1:end]) / window
    dE = abs.(ma .- e_tail)
    dE[end-window+1:end] .= NaN     # tail の定義窓と重なる分は落とす(Mode 1 と同じ理由)
    tauE = fit_tau(steps, dE, e_se)

    @printf("step %d  window %d  (%s)\n", n, window, orb ? "Φ モード、PartonOrbParam = 1" : "PartonMode = 2")
    @printf("E_tail = %.5f ± %.1e   σ²/E²_tail = %.2e   drift = %+.1e(条件 2: %s)\n",
            e_tail, e_se, v_tail, drift, abs(drift) <= 10e_se ? "満たす" : "破る")
    if isfinite(dt)
        @printf("dE/dτ = %+.2e(dt = %g、τ = 2·dt·step)  t = %+.1f → %s\n",
                slope / (2dt), dt, tval, plateau ? "頭打ち" : "まだ下がる")
    else
        @printf("dE/dstep = %+.2e(dt 不明)  t = %+.1f → %s\n", slope, tval, plateau ? "頭打ち" : "まだ下がる")
    end
    @printf("τ_E = %s step/e-fold\n", isfinite(tauE) ? @sprintf("%.0f", tauE) : "n/a")

    # 構造の判定: σ_N/σ_{N+1} の末尾 window と直前の window の平均(対数)の相対変化。フレーバーの最大
    # (Mode 1 の min_gap の「末尾平均の 5%」と同じ考え方。E は O(q²) なので構造 O(q) より先に平らに見える)
    sr_rel(k) = maximum(sr) do c
        a = sum(log.(@view c[k-window+1:k])) / window
        b = sum(log.(@view c[k-2window+1:k-window])) / window
        abs(exp(a - b) - 1)
    end
    nsr = isempty(sr) ? 0 : minimum(length, sr)
    if orb
        println("構造: Φ モードは判定しない(dE/dτ だけで判定)")
        isempty(skipped) || @printf("捨てたサンプル: 合計 %d(末尾 window %d)  log10 cond(Φ_f) の末尾: %s\n",
                                   round(Int, sum(skipped)), round(Int, sum(@view skipped[max(1, end - window + 1):end])),
                                   join([@sprintf("%.3f", c[end]) for c in lcond if !isempty(c)], " "))
    elseif nsr >= 2window
        r = sr_rel(nsr)
        @printf("構造: σ_N/σ_{N+1} の window 間の変化 %.1f%%(最大のフレーバー)→ %s\n",
                100r, r < 0.05 ? "落ち着いた" : "まだ動く")
    end

    # 判定量の推移(各点で「それまでの 2·window step」を使う。stride = window/2 ごと)
    jstride = max(window ÷ 2, 1)
    j_steps = Int[]; j_slope = Float64[]; j_se = Float64[]; j_t = Float64[]; j_sr = Float64[]
    for k in (2window):jstride:n
        sl, se_ = block_slope(e[k-2window+1:k], 10)
        push!(j_steps, k - 1); push!(j_slope, sl); push!(j_se, se_); push!(j_t, sl / se_)
        push!(j_sr, nsr >= k && k >= 2window ? sr_rel(k) : NaN)
    end
    tau_scale = isfinite(dt) ? 1 / (2dt) : 1.0      # dE/dτ = (dE/dstep)/(2·dt)

    ENV["GKSwstype"] = "100"
    @eval using Plots
    Base.invokelatest() do
        gr()
        clean(x) = [v > 0 && isfinite(v) ? v : NaN for v in x]

        p1 = plot(steps, e; alpha = 0.25, color = :gray, label = "E (raw)",
                  xlabel = "step", ylabel = "E", legend = :topright,
                  title = @sprintf("E_tail = %.5f ± %.1e", e_tail, e_se))
        plot!(p1, steps, ma; color = :crimson, lw = 2, label = "E ($(window)-step avg)")
        hline!(p1, [e_tail]; color = :crimson, ls = :dash, alpha = 0.5, label = "")
        if !isempty(var)
            pv = twinx(p1)
            plot!(pv, steps[1:length(var)], clean(var); yscale = :log10, color = :steelblue,
                  alpha = 0.6, label = "σ²/E²", ylabel = "σ²/E²", legend = :bottomright)
        end

        p2 = scatter(steps, clean(abs.(e .- e_tail)); yscale = :log10, color = :salmon,
                     ms = 1.2, msw = 0, alpha = 0.35, label = "|E_raw − E_tail|",
                     xlabel = "step", ylabel = "|E − E_tail|", legend = :bottomleft,
                     title = "log distance to tail (straight line = exponential)")
        plot!(p2, steps, clean(dE); yscale = :log10, color = :crimson, lw = 1.5,
              label = @sprintf("|E_avg − E_tail|  (τ=%s)", isfinite(tauE) ? @sprintf("%.0f", tauE) : "n/a"))
        hline!(p2, [e_se]; color = :crimson, ls = :dot, alpha = 0.6, label = "E noise floor (SE)")

        stride = 100
        r_steps = Int[]; rE = Float64[]
        for s in (window + stride):stride:n-1
            push!(r_steps, s); push!(rE, abs(ma[s+1] - ma[s+1-stride]))
        end
        ttl = isfinite(dt) ? @sprintf("dE/dτ = %+.2e (t = %+.1f, %s)", slope / (2dt), tval,
                                      plateau ? "plateau" : "descending") :
                             @sprintf("dE/dstep = %+.2e (t = %+.1f)", slope, tval)
        p3 = plot(r_steps, clean(rE); yscale = :log10, color = :crimson, lw = 1.5,
                  marker = :circle, ms = 2.5, label = "|ΔE| / $(stride) step",
                  xlabel = "step", ylabel = "|ΔE| / $(stride) step", legend = :bottomleft, title = ttl)
        hline!(p3, [10 * e_se]; color = :crimson, ls = :dash, label = @sprintf("10·SE = %.1e", 10e_se))

        if orb
            p4 = plot(; xlabel = "step", ylabel = "log10 cond(Φ_f)", legend = :topleft,
                      title = "log10 cond(Φ_f) / acceptance")
            for (a, c) in enumerate(lcond)
                plot!(p4, 0:(length(c) - 1), c; lw = 1.2, label = "flavor $(a - 1)")
            end
        else
            p4 = plot(; yscale = :log10, xlabel = "step", ylabel = "σ_N / σ_{N+1}", legend = :topleft,
                      title = "flavor row-block singular-value ratio / acceptance")
            for (a, c) in enumerate(sr)
                plot!(p4, 0:(length(c) - 1), clean(c); lw = 1.2, label = "flavor $(a - 1)")
            end
        end
        if !isempty(acc)
            pa = twinx(p4)
            plot!(pa, 0:(length(acc) - 1), acc; color = :darkorange, alpha = 0.6, label = "accept",
                  ylabel = "accept ratio", legend = :topright)
        end

        # 判定 1(符号つき・線形軸): dE/dτ ± 2·SE。帯が 0 をまたいだら頭打ち。
        # 1 step の差分や 100 step 平均どうしの差は終盤で雑音に埋もれる(KM 4×4 の末尾で信号/雑音 3e−4 / 0.3)ので、
        # 区間内の全 step を使うブロック平均の直線当てはめ(block_slope)を使う。縦軸は後半 6 割の値に合わせる
        ulab = isfinite(dt) ? "dE/dτ" : "dE/dstep"
        ys = j_slope .* tau_scale; band = 2 .* j_se .* tau_scale
        late = j_steps .>= 0.4 * n
        lim = any(late) ? 1.2 * maximum(abs.(ys[late]) .+ band[late]) : maximum(abs.(ys) .+ band)
        p5 = plot(j_steps, ys; ribbon = band, fillalpha = 0.25, color = :crimson, lw = 1.5,
                  marker = :circle, ms = 2, mc = :crimson, msw = 0,
                  label = "$(ulab) ± 2·SE  (fit over last $(2window) steps)",
                  xlabel = "step", ylabel = ulab, legend = :bottomright, ylims = (-lim, lim),
                  title = "criterion 1: plateau when the band crosses 0")
        hline!(p5, [0.0]; color = :black, lw = 1, label = "")

        # 判定 1(対数軸): |dE/dτ| と 2·SE。全体の桁の推移を見る
        p6 = plot(j_steps, clean(abs.(ys)); yscale = :log10, color = :crimson, lw = 1.5,
                  marker = :circle, ms = 2, mc = :crimson, msw = 0, label = "|$(ulab)|",
                  xlabel = "step", ylabel = "|$(ulab)|", legend = :bottomleft,
                  title = "criterion 1 (log): plateau when |$(ulab)| < 2·SE")
        plot!(p6, j_steps, clean(band); color = :black, ls = :dash, lw = 1.2, label = "2·SE")
        desc = j_slope .< 0
        any(.!desc) && scatter!(p6, j_steps[.!desc], clean(abs.(ys[.!desc]));
                                color = :steelblue, mc = :steelblue, ms = 3, msw = 0, label = "rising ($(ulab) > 0)")

        # 判定 1 の t 値
        p7 = plot(j_steps, j_t; color = :crimson, lw = 1.5, marker = :circle, ms = 2, mc = :crimson, msw = 0,
                  label = "t = slope / SE", xlabel = "step", ylabel = "t", legend = :bottomright,
                  title = "criterion 1: |t| < 2")
        hline!(p7, [-2.0, 2.0]; color = :black, ls = :dash, alpha = 0.7, label = "t = ±2")

        if orb
            # Φ モード: 構造の判定はしない。代わりに step ごとの捨てたサンプル数(0 が健全)
            p8 = plot(0:(length(skipped) - 1), skipped; color = :darkgreen, lw = 1.2, xlabel = "step",
                      ylabel = "skipped samples", legend = false,
                      title = @sprintf("skipped samples per step (total %d); no structure criterion",
                                       round(Int, sum(skipped))))
        else
            # 判定 2(構造): σ_N/σ_{N+1} の window 間の相対変化
            p8 = plot(; yscale = :log10, xlabel = "step", ylabel = "relative change / window", legend = :bottomleft,
                      title = "criterion 2: σ_N/σ_{N+1} change < 5%")
            any(isfinite, j_sr) && plot!(p8, j_steps, clean(j_sr); color = :darkgreen, lw = 1.5, marker = :circle,
                                         ms = 2, mc = :darkgreen, msw = 0, label = "max over flavors")
            hline!(p8, [0.05]; color = :black, ls = :dash, alpha = 0.7, label = "5%")
        end

        plt = plot(p1, p2, p3, p4, p5, p6, p7, p8; layout = (4, 2), size = (1400, 1700),
                   left_margin = 8Plots.mm, bottom_margin = 5Plots.mm, right_margin = 14Plots.mm)
        savefig(plt, png)
    end
    println("図: $png")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
