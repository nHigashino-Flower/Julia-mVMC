#=
zvo_conv.dat から収束の 4 パネル図を PNG で出す薄いスクリプト。
--- parton-mode (fork addition) ---

本体パッケージからは独立していて、依存は tools/Project.toml に隔離してある。
ローカル実行を想定:

    julia --project=tools tools/plot_conv.jl <zvo_conv.dat> [出力.png]

パネル:
  1. E vs step(線形)
  2. log|E - E_tail| vs step   ← 収束の指数的接近を見る本命
  3. log var vs step
  4. log(var / E²) vs step      ← ゼロ分散外挿の指標

`log E` は作らない(E < 0 なので未定義)。E_tail は最終 NSROptItrSmp ステップの
平均で、zqp_opt.dat がパラメータを平均している区間と同一 — つまり図の漸近値は
採用した最適パラメータの値そのものになる。
=#

using Plots, Printf
ENV["GKSwstype"] = "100"          # headless(画面を開かずファイルに保存)
gr()

length(ARGS) >= 1 || error("usage: julia --project=tools tools/plot_conv.jl <zvo_conv.dat> [out.png]")
src = ARGS[1]
out = length(ARGS) >= 2 ? ARGS[2] : replace(src, r"\.dat$" => "") * ".png"

e_tail = NaN
n_smp = 0
rows = Vector{Vector{Float64}}()
for line in eachline(src)
    s = strip(line)
    if startswith(s, "#")
        m = match(r"E_tail = ([-\d.eE+]+).*?= (\d+) ステップ", s)
        m !== nothing && (e_tail = parse(Float64, m[1]); n_smp = parse(Int, m[2]))
        continue
    end
    isempty(s) && continue
    push!(rows, parse.(Float64, split(s)))
end
isempty(rows) && error("no data rows in $src")

step = [r[1] for r in rows]
E    = [r[2] for r in rows]
var  = [r[3] for r in rows]
dE   = [r[4] for r in rows]
vr   = [r[5] for r in rows]

# log は正の値だけ。tail 区間ではノイズ床に落ちるので 0 が出うる。
poslog(x) = [v > 0 ? log10(v) : NaN for v in x]
tail_from = length(step) - n_smp + 1

p1 = plot(step, E, lw = 1, legend = false, xlabel = "SR step", ylabel = "E",
          title = @sprintf("E  (E_tail = %.6f)", e_tail))
p2 = plot(step, poslog(dE), lw = 1, legend = false, xlabel = "SR step",
          ylabel = "log10 |E - E_tail|", title = "収束(指数的接近)")
p3 = plot(step, poslog(var), lw = 1, legend = false, xlabel = "SR step",
          ylabel = "log10 var", title = "分散")
p4 = plot(step, poslog(vr), lw = 1, legend = false, xlabel = "SR step",
          ylabel = "log10 (var / E^2)", title = "ゼロ分散外挿の指標")

# tail 区間(パラメータ平均に使う区間)を陰影で示す
if n_smp > 0 && tail_from >= 1
    for p in (p1, p2, p3, p4)
        vspan!(p, [step[tail_from], step[end]]; alpha = 0.12, color = :gray, label = "")
    end
end

plot(p1, p2, p3, p4; layout = (2, 2), size = (1000, 700))
savefig(out)
println("wrote ", out)
