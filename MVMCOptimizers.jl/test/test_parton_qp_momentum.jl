"""
§8 テスト 11: 運動量射影の向き — 並進固有値による決定論的検証
--- parton-mode (fork addition) ---

## なぜ §8-3 では足りないのか

§8-3(`test_parton_qp.jl`, gather 版 = 実体化版)は、参照側の「実体化版」を
`Φqp[r, n] = sgn[r] * Φ[shifts[qp][r], n]` と、**実装とまったく同じ式**で
定義している。写像の向きを取り違えると両辺が同時にずれるので、この突き合わせは
向きについて検出力がゼロ。n_qp を増やしても、符号 sgn を凝っても上がらない。

## ここで使う参照値

射影 `P = Σ_R w_R T_R` が並進群の要素の線形結合であることだけから、
`w_R = e^{2πi n R/L}`(既約表現)なら **|φ⟩ が何であっても**

    T_S P|φ⟩ = w_S^{±1} · P|φ⟩

が成り立つ(群の性質のみ。平均場の並進不変性は不要)。固有値 `w_S^{±1}` は
`n, S, L` だけで決まり、**実装の式を一切参照しない**ので循環しない。

配置側で書くと、`τ_S: r ↦ r + S`(粒子の順序は保存)に対して

    ip(τ_S x) / ip(x) = e^{∓2πi n S/L}

指数の符号が gather に通す写像の向きそのもの:

- 順写像(`qp_trans[R][j] = j + R`、def の規約)
  → 実装の qp = R 項は `det Φ[r_m + R, n] = ⟨x|T_{-R}|φ⟩`
  → `|ψ⟩ = Σ_R w_R T_{-R}|φ⟩`、`T_S|ψ⟩ = w_S|ψ⟩`
  → `⟨x|T_S|ψ⟩ = ip(τ_{-S} x) = w_S ip(x)` ⇒ **ip(τ_S x)/ip(x) = e^{-2πi n S/L}**
- 逆写像なら同じ計算で **e^{+2πi n S/L}**

`n ≢ -n (mod L)`(= k がゼロでもゾーン境界でもない)を選べば両者は複素共役で
区別できる。L=6, n=1 の `e^{∓iπ/3}` を使う。

## 結果として確定した規約(v3.7)

実装は順写像側で、**射影状態の並進固有値は `T_S|ψ⟩ = e^{+2πi n S/L}|ψ⟩`**。
標準の Bloch 規約 `T_S|ψ_k⟩ = e^{-i k·S}|ψ_k⟩` と照らすと `k = -2πn/L`、つまり

    def の重み exp(2πi K·R) の K は、標準規約の運動量の **符号を反転したラベル**

これは C-mVMC の `tri = xqp[ori+1]` 規約をそのまま引き継いだ結果なので、参照実装
との比較では整合する(バグではない)。ただし時間反転が破れた系(複素ホッピング・
フラックス)では K と -K が別の状態なので、**狙った運動量セクターを指定するときは
この符号を意識する**必要がある。ラベルの取り違えは 3 番目の testset(Φ のバンド
運動量という別経路)で機械的に捕まる。

## 前提

- 符号 `qp_trans_sgn` は全 +1(実 def 経路でも APFlag=0 ならこうなる)。
  反周期境界の符号は別論点なので混ぜない
- 平均場はボンドごとに独立な α で**並進不変でない**。並進不変にすると
  |φ⟩ 自身が運動量固有状態になり、射影が自明化してテストが退化する
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"1D 周期鎖で配置 `sites`(1-based)を +S 並進する。粒子の順序は保存する。"
translate_sites(sites::Vector{Int}, S::Int, L::Int) = [mod1(r + S, L) for r in sites]

"""
    cyclic_qp!(data, L, n; forward=true) -> qp_weight

全並進 L 個を QP として張る。`forward = true` は def の規約どおりの順写像
`qp_trans[R][j] = j + R`、`false` はその逆写像(検出力の確認用)。
重みは `w_R = exp(2πi n R / L)`。
"""
function cyclic_qp!(
    data::MVMCExpertModeParsers.ExpertModeData,
    L::Int,
    n::Int;
    forward::Bool = true,
)
    shifts = [[forward ? mod1(j + R, L) : mod1(j - R, L) for j = 1:L] for R = 0:(L - 1)]
    sgns = [ones(Int, L) for _ = 0:(L - 1)]
    weights = ComplexF64[cis(2π * n * R / L) for R = 0:(L - 1)]
    return set_shift_qp!(data, shifts, sgns, weights)
end

"配置 `sites` での ip を契約1(厳密再計算)から求める。"
function ip_at(
    data::MVMCExpertModeParsers.ExpertModeData,
    mfham,
    qp_weight::Vector{ComplexF64},
    sites::Vector{Int},
)
    mp = data.modpara
    n_qp = length(data.qp_trans)
    cfg = toy_config(data, sites)
    amp = MVMCOptimizers.PartonAmplitudeData(n_qp, mp.nflavor, mp.nelec)
    ws = MVMCOptimizers.PartonSamplingWorkspace(mp.nelec, n_qp * mp.nflavor)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    return MVMCOptimizers.parton_calculate_ip(amp, qp_weight)
end

@testset "§8-11 運動量射影: 並進固有値が e^{-2πinS/L}(順写像規約)" begin
    L, Ne, F, n = 6, 3, 2, 1
    data = per_bond_mf_data(F; n_site = L, n_elec = Ne)
    qp_weight = cyclic_qp!(data, L, n)
    mfham = build_toy_mfham(data; n_idx = L)

    for sites in ([1, 3, 4], [2, 3, 6], [1, 2, 4])
        ip0 = ip_at(data, mfham, qp_weight, sites)
        # 比が意味を持つ前提(ノード上でないこと)を先に確かめる
        @test abs(ip0) > 1e-8
        for S = 0:(L - 1)
            ipS = ip_at(data, mfham, qp_weight, translate_sites(sites, S, L))
            @test ipS / ip0 ≈ cis(-2π * n * S / L) rtol = 1e-10
        end
    end
end

@testset "§8-11 検出力: 写像を逆向きにすると固有値が複素共役に振れる" begin
    # §8-3 に欠けていたのはこの一段。参照値が実装から独立でも、ミューテーションで
    # 落ちることを見ていなければ「検出できるテスト」だとは言えない。
    L, Ne, F, n = 6, 3, 2, 1
    sites = [1, 3, 4]

    λ = map((true, false)) do fwd
        data = per_bond_mf_data(F; n_site = L, n_elec = Ne)
        qp_weight = cyclic_qp!(data, L, n; forward = fwd)
        mfham = build_toy_mfham(data; n_idx = L)
        ip0 = ip_at(data, mfham, qp_weight, sites)
        ip1 = ip_at(data, mfham, qp_weight, translate_sites(sites, 1, L))
        ip1 / ip0
    end

    @test λ[1] ≈ cis(-2π * n / L) rtol = 1e-10
    @test λ[2] ≈ cis(+2π * n / L) rtol = 1e-10
    @test !isapprox(λ[1], λ[2]; rtol = 1e-3)
end

@testset "§8-11 独立経路: 並進不変な平均場で射影が生き残る K が一意に決まる" begin
    # 上の 2 つは配置側の並進 τ_S を経由した検証なので、τ_S の向きの取り違えが
    # あれば結論も一緒に反転する。ここは **Φ のバンド運動量**という完全に別経路の
    # 参照値を使って同じ規約を裏取りする。
    #
    # 平均場を一様(並進不変)にすると |φ⟩ 自身が並進固有状態になり、
    # 固有値は占有軌道のバンド運動量の和 K_0 で決まる:  T_S|φ⟩ = e^{-i K_0 S}|φ⟩。
    # 実装の qp = R 項は ⟨x|T_{-R}|φ⟩ なので
    #     P|φ⟩ = Σ_R w_R T_{-R}|φ⟩ = |φ⟩ · Σ_R e^{i(2πn/L + K_0)R}
    # ⇒ **2πn/L ≡ -K_0 のときだけ非ゼロ**、他の n では厳密にゼロ。
    # K_0 は Φ から直接読むので、qp 経路の式を一切参照しない。
    L, Ne, F = 6, 2, 2
    t = ComplexF64(-1.0, 0.4)
    data = toy_mf_data(; n_site = L, n_elec = Ne, n_flavor = F, t = t)
    set_identity_qp!(data)
    mfham = build_toy_mfham(data)

    # 占有軌道のバンド運動量 q を φ(r+1)/φ(r) = e^{iq} から読む。
    # 比が r によらず一定であること = 縮退で軌道が混ざっていないことの確認も兼ねる。
    Φ = mfham.orbitals[1]
    qs = Float64[]
    for n_orb = 1:Ne
        ratios = [Φ[mod1(r + 1, L), n_orb] / Φ[r, n_orb] for r = 1:L]
        @test all(abs(ρ) ≈ 1 for ρ in ratios)              # 平面波であること
        @test all(ρ ≈ ratios[1] for ρ in ratios)           # 縮退で混ざっていないこと
        push!(qs, angle(ratios[1]))
    end
    # 全フレーバーが同じ Φ を使う(固縛)ので全体の運動量は F 倍
    K_0 = F * sum(qs)
    n_star = mod(round(Int, -K_0 * L / (2π)), L)
    # 符号を取り違えたら別の n に落ちる設定であること。K_0 = 0 や π を選ぶと
    # n* = -n* になって検出力が消える(§8-3 が踏んだのと同じ穴)。
    @test n_star != mod(-n_star, L)

    for n = 0:(L - 1)
        qp_weight = cyclic_qp!(data, L, n)
        # P|φ⟩ = 0 なら任意の配置で厳密にゼロ。非ゼロ側は配置によっては
        # 偶然落ちうるので、複数の配置のうち少なくとも 1 つで有意であればよい。
        ips = [ip_at(data, mfham, qp_weight, s) for s in ([1, 3], [1, 4], [2, 5])]
        if n == n_star
            @test maximum(abs, ips) > 1e-6
        else
            @test maximum(abs, ips) < 1e-10
        end
    end
end
