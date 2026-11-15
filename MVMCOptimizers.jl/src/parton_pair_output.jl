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

- `n_skipped` — 捨てたサンプル数(このランク。ip が 0 か非有限、または E_loc が非有限。`parton_ip_usable`)
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
        @printf(f, "D_AmpMax %.17g\n", mp.parton_pair_amp_max)     # PartonPairAmpMax(既定 4 = 上流の D_AMP_MAX)
        println(f, "pair_init $init_source")
        println(f, "PartonGaugeFix not_applicable")
        println(f, "PartonFlavorSymFast not_applicable")
    end
    return nothing
end

"""
    parton_pair_flavor_orbitals(src) -> Vector{Matrix{ComplexF64}}

フレーバー a ごとに、行ブロック f_{a,:} の上位 N 本の左特異ベクトル U^a(Ns×N)を返す
(DESIGN_parton_pf.md §1.4)。f = ΞJΞᵀ なら span U^a = span Φ^a。U^a の中のユニタリ回転は任意なので、
物理量は射影子 P_a = U^a U^{a†} で扱うこと。
"""
function parton_pair_flavor_orbitals(src::PartonPairOrbital)
    ns = src.n_site
    N = src.n_elec
    return [svd(src.pair_matrix[((a - 1) * ns + 1):(a * ns), :]).U[:, 1:N] for a = 1:src.n_flavor]
end

"""
    parton_pair_write_svd(data, src, dir, tag) -> (sv_path, orb_path)

行ブロック SVD を 2 つのファイルに書く(`tag` は `"init"` か `"opt"`。DESIGN_parton_pf.md §6):

- `<CParaFileHead>_parton_pair_sv_<tag>.dat`: `# flavor index sigma`(0-based)の全特異値と、
  `# sigma_ratio_<a> <σ_N/σ_{N+1}>`
- `<CParaFileHead>_parton_pair_orb_<tag>.dat`: 上位 N 本の左特異ベクトルを **InPmfOrb と同じ形式**
  (`NPmfOrb`、平坦添字 ((f·Ns)+s)·Ne+n、0-based)で書く。Φ モードの入力としてそのまま渡せ、
  射影子 P_a を使う下流(Chern など)の入力にもなる
"""
function parton_pair_write_svd(data::ExpertModeData, src::PartonPairOrbital, dir, tag::AbstractString)
    mp = data.modpara
    ns, N, K = mp.nsite, mp.nelec, mp.nflavor
    σs = parton_pair_flavor_singular_values(src)
    U = parton_pair_flavor_orbitals(src)
    sv_path = _parton_para_out(data, "_parton_pair_sv_$(tag).dat", dir)
    open(sv_path, "w") do f
        println(f, "# NFlavor $K")
        println(f, "# NSite $ns")
        println(f, "# NElec $N")
        for a = 1:K
            @printf(f, "# sigma_ratio_%d % .8e\n", a - 1, parton_pair_sigma_ratio(σs[a], N))
        end
        println(f, "# flavor index sigma")
        for a = 1:K, (k, σ) in enumerate(σs[a])
            @printf(f, "%d %d % .18e\n", a - 1, k - 1, σ)
        end
    end
    orb_path = _parton_para_out(data, "_parton_pair_orb_$(tag).dat", dir)
    n_orb = K * ns * N
    open(orb_path, "w") do f
        println(f, "===============================")
        println(f, "NPmfOrb $n_orb")
        println(f, "===============================")
        println(f, "== idx RePhi ImPhi ==")
        println(f, "===============================")
        for a = 1:K, s = 1:ns, n = 1:N
            idx = ((a - 1) * ns + (s - 1)) * N + (n - 1)
            v = U[a][s, n]
            @printf(f, "%d % .18e % .18e \n", idx, real(v), imag(v))
        end
    end
    return sv_path, orb_path
end
