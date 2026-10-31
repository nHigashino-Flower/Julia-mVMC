"""
Pfaffian パートンモードの出力(診断・runinfo・行ブロック SVD)
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §1.4・§6。すべて rank 0 だけが書く(呼び出し側が判定する)。
"""

"""
    parton_pair_flavor_singular_values(src) -> Vector{Vector{Float64}}

フレーバー a ごとに、行ブロック f_{a,:} = f[(·, a), :](Ns × K·Ns)の全特異値(降順)を返す。
f = ΞJΞᵀ なら階数はちょうど N で、上位 N 本の左特異ベクトルが V_a = span{φ^a_n} を張る
(DESIGN_parton_pf.md §1.4)。特異値の大きさそのものは J と D_AmpMax のスケールを映すだけ。
"""
function parton_pair_flavor_singular_values(src::PartonPairOrbital)
    ns = src.n_site
    return [svdvals(src.pair_matrix[((a - 1) * ns + 1):(a * ns), :]) for a = 1:src.n_flavor]
end

"""
    parton_pair_sigma_ratio(σ, n_elec) -> Float64

σ_N / σ_{N+1}(det^K 多様体からの離れ具合の診断量)。σ_{N+1} = 0 なら Inf、N が特異値の数以上なら NaN。
"""
function parton_pair_sigma_ratio(σ::AbstractVector{<:Real}, n_elec::Int)
    n_elec < length(σ) || return NaN
    return σ[n_elec] / σ[n_elec + 1]
end

"""
    parton_pair_write_diag(data, pstate, step, dir; n_skipped, xmax_pre, sigma_ratio)

`<head>_parton_pairdiag.dat` に SR ステップごとに 1 行書く(step 0 で "w"、以降 "a")。

- `n_skipped` — |ip| < 1e−30 で捨てたサンプル数(このランク)
- `accept_ratio` / `n_need_recompute` / `n_exact_recompute` — MC の健康診断(α・Φ と同じ定義)
- `xmax_pre` — D_AmpMax で再スケールする**前**の max|p|
- `sigma_ratio_a` — そのステップで使った f の行ブロック SVD の σ_N/σ_{N+1}(フレーバーごと)
"""
function parton_pair_write_diag(
    data::ExpertModeData,
    pstate::PartonOptimizationState,
    step::Int,
    dir;
    n_skipped::Int,
    xmax_pre::Float64,
    sigma_ratio::Vector{Float64},
)
    dir === nothing && return nothing
    path = _parton_out(data, "_parton_pairdiag.dat", dir)
    counter = pstate.config.counter
    trials = counter[1]
    acc = trials > 0 ? counter[2] / trials : 0.0
    open(path, step == 0 ? "w" : "a") do f
        if step == 0
            println(f, "# step  n_skipped  accept_ratio  n_need_recompute  n_exact_recompute  " *
                       "xmax_pre  " * join(("sigma_ratio_$(a - 1)" for a in eachindex(sigma_ratio)), "  "))
        end
        @printf(f, "%6d %8d % .6e %8d %8d % .8e", step, n_skipped, acc, counter[3], counter[4], xmax_pre)
        for r in sigma_ratio
            @printf(f, " % .6e", r)
        end
        println(f)
    end
    return path
end

"""
    parton_pair_append_runinfo(data, dir; init_source)

`zvo_parton_runinfo.dat` の末尾に Pf モードの項目を追記する。`init_source` は初期値の出どころ
(`"random"` か `"InPartonPairOrbital"`)。PartonGaugeFix / PartonFlavorSymFast は Pf モードでは
意味を持たないので「該当なし」と書く。
"""
function parton_pair_append_runinfo(data::ExpertModeData, dir; init_source::AbstractString)
    dir === nothing && return nothing
    mp = data.modpara
    open(_parton_out(data, "_parton_runinfo.dat", dir), "a") do f
        @printf(f, "n_pair_idx %d\n", parton_pair_n_idx(data))
        @printf(f, "n_parton_total %d\n", mp.nflavor * mp.nelec)
        @printf(f, "D_AmpMax %.17g\n", D_AMP_MAX)
        println(f, "pair_init $init_source")
        println(f, "PartonGaugeFix not_applicable")
        println(f, "PartonFlavorSymFast not_applicable")
    end
    return nothing
end
