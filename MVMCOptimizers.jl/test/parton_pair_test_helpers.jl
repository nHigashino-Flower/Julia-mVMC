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
using Printf

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

"""
    _ref_pfaffian(X) -> ComplexF64

テストの参照用の Pf。本番とは独立に書いた素朴な Parlett–Reid(列ピボットつき)で、
pf² = det と包含定理で検算済み(DESIGN_parton_pf.md §1.5)。本番の部品を参照に使うと、
部品の不具合(PfaPack の `julia_zsktf2!` が交互ブロック構造の正則行列でゼロピボットを返す。§3)を
見逃すので、ここでは使わない。
"""
function _ref_pfaffian(X::AbstractMatrix)
    A = Matrix{ComplexF64}(X)
    n = size(A, 1)
    isodd(n) && return zero(ComplexF64)
    p = one(ComplexF64)
    for k = 1:2:(n - 1)
        kp = k + argmax(abs.(A[(k + 1):n, k]))
        if kp != k + 1
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            p = -p
        end
        iszero(A[k + 1, k]) && return zero(ComplexF64)
        p *= A[k, k + 1]
        if k + 2 <= n
            tau = A[k, (k + 2):n] ./ A[k, k + 1]
            A[(k + 2):n, (k + 2):n] .+= tau * transpose(A[(k + 2):n, k + 1]) .-
                                        A[(k + 2):n, k + 1] * transpose(tau)
        end
    end
    return p
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

# --- 実行テストの最小入力(test_parton_pair_run.jl から移した。物理量測定のテストと共有)---

"全対・共有なしの対軌道 def を書く(上流 orbitalidxgen.def と同じ書式)。"
function _write_full_pair_def(path, n_site, n_flavor; opt_lines = true)
    terms = full_pair_terms(n_site, n_flavor)
    open(path, "w") do io
        println(io, "=============================================")
        println(io, "NOrbitalIdx $(length(terms))")
        println(io, "ComplexType 1")
        println(io, "=============================================")
        println(io, "=============================================")
        for t in terms
            println(io, "$(t.site1) $(t.flavor1) $(t.site2) $(t.flavor2) $(t.idx) $(t.sign)")
        end
        opt_lines && for t in terms
            println(io, "$(t.idx) 1")
        end
    end
    return terms
end

"""
    _write_min_pair_input(dir; nsteps, nsmp, in_from, qp) -> namelist

`_write_min_parton_input`(4 サイト環・F=2・Ne=2 の物理項と modpara)から pmftrans / pmfpara を
外し、`PartonMode 2` と全対の def を足した最小入力。`in_from` を渡すとそれを
InPartonPairOrbital にする。`qp = true` なら 2 並進(シフト 0, 1)の qptransidx.def を足す。
"""
function _write_min_pair_input(dir; nsteps::Int = 4, nsmp::Int = 1, in_from = nothing,
                               qp::Bool = false)
    _write_min_parton_input(dir)
    rm(joinpath(dir, "pmftrans.def"))
    rm(joinpath(dir, "pmfpara.def"))
    mod = joinpath(dir, "modpara.def")
    lines = map(readlines(mod)) do l
        startswith(l, "NSROptItrStep") ? "NSROptItrStep $nsteps" :
        startswith(l, "NSROptItrSmp") ? "NSROptItrSmp $nsmp" :
        startswith(l, "PartonMode") ? "PartonMode 2" : l
    end
    qp && push!(lines, "NMPTrans 2")
    write(mod, join(lines, "\n") * "\n")
    _write_full_pair_def(joinpath(dir, "pair.def"), 4, 2)
    entries = "ModPara        modpara.def\nPhysHop        physhop.def\n" *
              "CoulombInter   coulombinter.def\nPartonPairOrbital pair.def\n"
    if in_from !== nothing
        cp(in_from, joinpath(dir, "InPair.def"); force = true)
        entries *= "InPartonPairOrbital InPair.def\n"
    end
    qp && (_write_two_shift_qp(joinpath(dir, "qptransidx.def"), 4);
           entries *= "TransSym       qptransidx.def\n")
    write(joinpath(dir, "namelist.def"), entries)
    return joinpath(dir, "namelist.def")
end

"環のシフト 0, 1 の 2 並進(重み 1 と e^{iπ/3})。"
function _write_two_shift_qp(path, n_site)
    open(path, "w") do io
        println(io, "===============================")
        println(io, "NQPTrans 2")
        println(io, "===============================")
        println(io, "== i_j_TransSym ==")
        println(io, "===============================")
        w = cis(π / 3)
        @printf(io, "0 % .18e % .18e\n", 1.0, 0.0)
        @printf(io, "1 % .18e % .18e\n", real(w), imag(w))
        for j = 0:(n_site - 1)
            @printf(io, "0 %d %d 1\n", j, j)
        end
        for j = 0:(n_site - 1)
            @printf(io, "1 %d %d 1\n", j, mod(j + 1, n_site))
        end
    end
end
