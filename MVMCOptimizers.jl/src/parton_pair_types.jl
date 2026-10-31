"""
Pfaffian パートンモード(PartonMode = 2)の型とテンプレート
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §1・§4.2。振幅は Pf F(r) で、F は全パートン(K·N 本)の対軌道 f から作る:

    F_qp(x)[I, J] = s_qp(r_I)·s_qp(r_J)·f[(π_qp(r_I), a_I), (π_qp(r_J), a_J)]
    f[α, β]       = 2·sgn[α, β]·p[idx(α, β)]    (def に行がある対。無い対は 0)

- 行 I = (m−1)K + a は「粒子 m のフレーバー a のパートン」(粒子が主・フレーバーが従)。
  m は `cfg.ele_idx` のラベルで、サイト順にはソートしない(第一量子化。det モードと同じ)
- 一粒子状態の合成添字は α = i + (a−1)·Ns(1-based)。0-based では i + a·Ns で、上流の
  site + spin·Ns を K 個へ広げたもの
- 係数 2 と符号の規約は上流の一般スピン Pfaffian と同じ(`slater_update.jl:686-697`)。
  符号は def に書かれた行の向きと符号列のとおり(sgn[β, α] = −sgn[α, β])
"""

"PartonMode の値: Pfaffian パートンモード(DESIGN_parton_pf.md)"
const PARTON_MODE_PFAFFIAN = 2

"1-based の (サイト i, フレーバー a) → 一粒子状態の合成添字(1-based)。"
@inline parton_pair_state(n_site::Int, i::Int, a::Int) = (a - 1) * n_site + i

"Pf(C⁻¹)、C⁻¹ = [[0, −I], [I, 0]](2K×2K)。低ランク更新の比の分母(DESIGN_parton_pf.md §1.5)。"
@inline parton_pair_pf_cinv(k::Int) = iseven(k * (k + 1) ÷ 2) ? 1 : -1

"""
    PartonPairOrbital

対軌道 f の器(「軌道の出どころ」の Pfaffian 版)。

固定部(起動時に 1 回。def の 0-based → 1-based の変換はここだけで行う。本体 DESIGN §3.1):
- `pair_idx[α, β]`: 1-based の idx。0 は def に無い対(構造的な 0。パラメータではない)
- `pair_sgn[α, β]`: ±1(反対称: sgn[β, α] = −sgn[α, β])。def に無い対は 0

パラメータ依存部(SR ステップごとに `parton_pair_load!` が組み直す):
- `pair_matrix[α, β]` = 2·sgn·p。(K·Ns)×(K·Ns) の反対称行列
"""
mutable struct PartonPairOrbital <: PartonOrbitalSource
    n_site::Int
    n_flavor::Int
    n_elec::Int
    n_idx::Int
    pair_idx::Matrix{Int32}
    pair_sgn::Matrix{Int8}
    pair_matrix::Matrix{ComplexF64}
end

"""
    PartonPairOrbital(n_site, n_flavor, n_elec, terms) -> PartonPairOrbital

def の行(`site1, flavor1, site2, flavor2, idx, sign` を持つ要素の並び。すべて 0-based)から
テンプレートを組む。違反は `ArgumentError`(上流のパーサは範囲外を黙って捨てるが、ここでは止める):

- サイト・フレーバーの範囲、対角(α = β)、同じ非順序対の重複(逆向きも含む)
- idx は 0 以上で、0..n_idx−1 の全部が使われていること(連番)
- 符号は ±1
"""
function PartonPairOrbital(n_site::Int, n_flavor::Int, n_elec::Int, terms)
    n_state = n_site * n_flavor
    pair_idx = zeros(Int32, n_state, n_state)
    pair_sgn = zeros(Int8, n_state, n_state)
    isempty(terms) && throw(ArgumentError(
        "PartonPairOrbital has no pair terms: every amplitude would vanish."))
    max_idx = -1
    for (row, t) in enumerate(terms)
        where_ = "PartonPairOrbital row $row ($(t.site1) $(t.flavor1) $(t.site2) $(t.flavor2) $(t.idx) $(t.sign))"
        (0 <= t.site1 < n_site && 0 <= t.site2 < n_site) || throw(ArgumentError(
            "$where_: site out of range 0..$(n_site - 1)."))
        (0 <= t.flavor1 < n_flavor && 0 <= t.flavor2 < n_flavor) || throw(ArgumentError(
            "$where_: flavor out of range 0..$(n_flavor - 1)."))
        t.idx >= 0 || throw(ArgumentError("$where_: idx must be non-negative."))
        (t.sign == 1 || t.sign == -1) || throw(ArgumentError(
            "$where_: sign must be +1 or -1."))
        α = parton_pair_state(n_site, t.site1 + 1, t.flavor1 + 1)
        β = parton_pair_state(n_site, t.site2 + 1, t.flavor2 + 1)
        α == β && throw(ArgumentError(
            "$where_: diagonal pair (the same single-particle state twice) is not allowed."))
        pair_idx[α, β] == 0 || throw(ArgumentError(
            "$where_: this unordered pair is already listed (possibly in the reverse order)."))
        pair_idx[α, β] = t.idx + 1
        pair_idx[β, α] = t.idx + 1
        pair_sgn[α, β] = t.sign
        pair_sgn[β, α] = -t.sign
        max_idx = max(max_idx, t.idx)
    end
    n_idx = max_idx + 1
    used = falses(n_idx)
    for t in terms
        used[t.idx + 1] = true
    end
    all(used) || throw(ArgumentError(
        "PartonPairOrbital idx must be contiguous 0..$(n_idx - 1); unused: " *
        "$(findall(!, used) .- 1)."))
    return PartonPairOrbital(n_site, n_flavor, n_elec, n_idx, pair_idx, pair_sgn,
                             zeros(ComplexF64, n_state, n_state))
end

"""
    parton_pair_load!(src, params)

p(idx ごと、長さ n_idx)から `pair_matrix` を組み直す: f[α, β] = 2·sgn[α, β]·p[idx(α, β)]。
"""
function parton_pair_load!(src::PartonPairOrbital, params::AbstractVector{ComplexF64})
    length(params) == src.n_idx || throw(ArgumentError(
        "pair parameter length $(length(params)) does not match n_idx = $(src.n_idx)."))
    P = src.pair_matrix
    n_state = size(P, 1)
    @inbounds for β = 1:n_state, α = 1:n_state
        k = src.pair_idx[α, β]
        P[α, β] = k == 0 ? zero(ComplexF64) : 2 * src.pair_sgn[α, β] * params[k]
    end
    return src
end

"""
    PartonPairAmplitudeData

全 qp の G = F⁻¹(M×M)と Pf F(生の複素値)。ストライドは `pair_inv_block` だけに封じ込める。

`row_site[m]` は粒子 m の現在のサイト。契約 2・3 は配置(cfg)を受け取らないシグネチャなので
(det モードは軌道の行だけで済むが、Pf は他の粒子のサイトも要る)、配置の写しをここに持つ。
契約 1 が cfg から確定し、契約 3 が受理のたびに更新する。
"""
mutable struct PartonPairAmplitudeData <: PartonAmplitudeStore
    inv_f::Vector{ComplexF64}
    pf_f::Vector{ComplexF64}
    row_site::Vector{Int}
    n_qp::Int
    n_flavor::Int
    n_elec::Int

    function PartonPairAmplitudeData(n_qp::Int, n_flavor::Int, n_elec::Int)
        M = n_flavor * n_elec
        new(zeros(ComplexF64, n_qp * M * M), zeros(ComplexF64, n_qp), zeros(Int, n_elec),
            n_qp, n_flavor, n_elec)
    end
end

"F の次元 M = K·N。"
@inline parton_pair_dim(amp::PartonPairAmplitudeData) = amp.n_flavor * amp.n_elec

"qp の G = F⁻¹ を M×M 行列ビューとして返す。ストライドを書いてよい唯一の場所。"
@inline function pair_inv_block(amp::PartonPairAmplitudeData, qp::Int)
    M = parton_pair_dim(amp)
    off = (qp - 1) * M * M
    return reshape(view(amp.inv_f, (off + 1):(off + M * M)), M, M)
end

"""
    PartonPairSamplingWorkspace

ホットループの作業領域。`gb_blocks` / `x_blocks` / `ratio_blocks` は契約 2 が書き、
直後の契約 3 が読む(qp ごとの GB = [G[:, S], G·D]、X = C⁻¹ + BᵀGB、R_qp)。
"""
mutable struct PartonPairSamplingWorkspace <: PartonWorkspaceStore
    f_scratch::Matrix{ComplexF64}          # M×M: 厳密再計算の組立と Pf(Parlett–Reid が壊す)
    lu_scratch::Matrix{ComplexF64}         # M×M: G = F⁻¹ の LU(F の写し)
    tau::Vector{ComplexF64}                # M: Parlett–Reid の作業領域
    d_buf::Matrix{ComplexF64}              # M×K: 差分 D
    gb_blocks::Vector{Matrix{ComplexF64}}  # qp ごとの GB(M×2K)
    x_blocks::Vector{Matrix{ComplexF64}}   # qp ごとの X(2K×2K)
    ratio_blocks::Vector{ComplexF64}       # qp ごとの R_qp
    x_scratch::Matrix{ComplexF64}          # 2K×2K: 小さい Pf の写し
    tau_small::Vector{ComplexF64}          # 2K: 小さい Pf の作業領域
    y_buf::Matrix{ComplexF64}              # M×2K: GB·X⁻¹

    function PartonPairSamplingWorkspace(n_qp::Int, n_flavor::Int, n_elec::Int)
        M = n_flavor * n_elec
        K2 = 2 * n_flavor
        new(
            zeros(ComplexF64, M, M),
            zeros(ComplexF64, M, M),
            zeros(ComplexF64, M),
            zeros(ComplexF64, M, n_flavor),
            [zeros(ComplexF64, M, K2) for _ = 1:n_qp],
            [zeros(ComplexF64, K2, K2) for _ = 1:n_qp],
            zeros(ComplexF64, n_qp),
            zeros(ComplexF64, K2, K2),
            zeros(ComplexF64, K2),
            zeros(ComplexF64, M, K2),
        )
    end
end
