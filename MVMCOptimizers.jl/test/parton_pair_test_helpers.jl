"""
Pfaffian パートンモード(PartonMode = 2)のテストで共有するトイ入力
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §9。`parton_test_helpers.jl`(QP の実体化 `set_identity_qp!` /
`set_shift_qp!`、`toy_config`)を先に include しておくこと。

一粒子状態の合成添字は α = i + (a−1)·Ns(1-based。0-based なら i + a·Ns)。
def の 1 行は NamedTuple `(site1, flavor1, site2, flavor2, idx, sign)`(すべて 0-based、
パーサの `PartonPairOrbitalTerm` と同じフィールド名)で表す。
"""

using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

"1-based の (サイト, フレーバー) → 合成添字(1-based)。"
_pair_state(n_site::Int, i::Int, a::Int) = (a - 1) * n_site + i

"""
    pair_toy_data(; n_site, n_elec, n_flavor) -> ExpertModeData

PartonMode = 2 の最小の modpara と恒等 QP を持つデータ。def の項と f の値は持たない
(テンプレートは `PartonPairOrbital` を直接組んで渡す)。
"""
function pair_toy_data(; n_site::Int = 5, n_elec::Int = 2, n_flavor::Int = 2)
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite = n_site
    mp.nelec = n_elec
    mp.nflavor = n_flavor
    mp.parton_mode = 2
    mp.two_sz = 0
    mp.complex_flag = 1
    mp.nex_update_path = 6
    set_identity_qp!(data)
    return data
end

"""
    full_pair_terms(n_site, n_flavor) -> Vector{NamedTuple}

全ての非順序対(合成添字の小さい方を先に書く)を 1 行ずつ、別々の idx で列挙する
(共有なしの def)。符号は +1。
"""
function full_pair_terms(n_site::Int, n_flavor::Int)
    terms = NamedTuple{(:site1, :flavor1, :site2, :flavor2, :idx, :sign),
                       NTuple{6,Int}}[]
    idx = 0
    n_state = n_site * n_flavor
    for α = 1:n_state, β = (α + 1):n_state
        i, a = mod1(α, n_site), cld(α, n_site)
        j, b = mod1(β, n_site), cld(β, n_site)
        push!(terms, (site1 = i - 1, flavor1 = a - 1, site2 = j - 1, flavor2 = b - 1,
                      idx = idx, sign = 1))
        idx += 1
    end
    return terms
end

"""
    pair_params_from_matrix(f, terms, n_site) -> Vector{ComplexF64}

反対称な f((K·Ns)²)から、F = 2·sgn·p の規約で p を取り出す(共有なしの def 用。
同じ idx の行が複数あれば平均する)。
"""
function pair_params_from_matrix(f::AbstractMatrix, terms, n_site::Int)
    n_idx = maximum(t.idx for t in terms) + 1
    acc = zeros(ComplexF64, n_idx)
    cnt = zeros(Int, n_idx)
    for t in terms
        α = _pair_state(n_site, t.site1 + 1, t.flavor1 + 1)
        β = _pair_state(n_site, t.site2 + 1, t.flavor2 + 1)
        acc[t.idx + 1] += f[α, β] / (2 * t.sign)
        cnt[t.idx + 1] += 1
    end
    return acc ./ cnt
end

"反対称な複素乱数行列。"
_rand_antisym(rng, n) = (X = randn(rng, ComplexF64, n, n); X - transpose(X))

"""
    random_xi_j(rng, n_site, n_elec, n_flavor) -> (Φ, J, f)

各フレーバーの複素乱数軌道 Φ^a(Ns×N)と反対称な J(M×M)から f = Ξ J Ξᵀ を作る。
Ξ の行は合成添字(フレーバーが主)、列はフレーバー順。
"""
function random_xi_j(rng, n_site::Int, n_elec::Int, n_flavor::Int)
    M = n_flavor * n_elec
    Φ = [randn(rng, ComplexF64, n_site, n_elec) for _ = 1:n_flavor]
    Ξ = zeros(ComplexF64, n_flavor * n_site, M)
    for a = 1:n_flavor
        Ξ[(a - 1) * n_site + 1:a * n_site, (a - 1) * n_elec + 1:a * n_elec] = Φ[a]
    end
    J = _rand_antisym(rng, M)
    return Φ, J, Ξ * J * transpose(Ξ)
end

"PfaPack の LTL 分解による Pf(テストの参照用。本番と同じ部品)。"
function _ref_pfaffian(X::AbstractMatrix)
    n = size(X, 1)
    A = Matrix{ComplexF64}(X)
    ipiv = zeros(Int, n)
    MVMCOptimizers.julia_zsktf2!(A, ipiv)
    return MVMCOptimizers.utu2pfa(n, A, n, ipiv)
end

"""
    build_pair_source(data, terms, params) -> PartonPairOrbital

テンプレートを組んで p を読み込んだ対軌道の器を返す。
"""
function build_pair_source(data, terms, params::AbstractVector{ComplexF64})
    mp = data.modpara
    src = MVMCOptimizers.PartonPairOrbital(mp.nsite, mp.nflavor, mp.nelec, terms)
    MVMCOptimizers.parton_pair_load!(src, params)
    return src
end

"Pf モードの振幅キャッシュと作業領域を確保する。"
function pair_amp_ws(data)
    mp = data.modpara
    n_qp = length(data.qp_trans)
    return (MVMCOptimizers.PartonPairAmplitudeData(n_qp, mp.nflavor, mp.nelec),
            MVMCOptimizers.PartonPairSamplingWorkspace(n_qp, mp.nflavor, mp.nelec))
end

"""
    random_ring_qp!(rng, data, n_qp) -> qp_weight

環の並進 n_qp 個(シフト 0..n_qp−1)を、サイトごとの乱数符号と複素重みつきで実体化する。
代数の検証用(物理的な射影である必要はない)。
"""
function random_ring_qp!(rng, data, n_qp::Int)
    n_site = data.modpara.nsite
    shifts = [[mod(r - 1 + q, n_site) + 1 for r = 1:n_site] for q = 0:(n_qp - 1)]
    sgns = [q == 0 ? ones(Int, n_site) : rand(rng, [-1, 1], n_site) for q = 0:(n_qp - 1)]
    weights = [q == 0 ? ComplexF64(1) : randn(rng, ComplexF64) for q = 0:(n_qp - 1)]
    return set_shift_qp!(data, shifts, sgns, weights)
end
