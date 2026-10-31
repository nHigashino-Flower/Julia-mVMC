"""
Φ 直接変分モードの SR 経路の検証 --- parton-mode (fork addition) ---

DESIGN_parton.md §12.11 / §12.9 の Φ-2(力 = 勾配)と Φ-3(S の冗長方向)。
サンプリング誤差を排すため全配置を列挙し、重みは厳密な |P_J·ip|²。蓄積と S・g の
組み立ては上流の `calculate_oo!` / `calculate_oo_store!` / `finalize_oo_store!` /
`build_s_matrix_and_g_vector!` をそのまま通す(test_parton_force_gradient.jl と同じ骨格)。
"""

using Test
using Random
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"全配置を列挙して OO・HO を蓄積し、厳密な ⟨E⟩ を返す。"
function _orb_enumerate(pstate, data, configs; conjugate::Bool, store::Bool = false)
    MVMCOptimizers.parton_orbparam_load_orbitals!(pstate.mfham, data)
    qpw = MVMCOptimizers.parton_qp_weight(data)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    sr = pstate.state.sr_opt
    fill!(sr.sr_opt_oo, 0)
    fill!(sr.sr_opt_ho, 0)
    store && fill!(sr.sr_opt_o_store, 0)

    wc = 0.0
    etot = ComplexF64(0)
    n_stored = 0
    for c in configs
        cfg = _jast_place!(pstate, data, c)
        ip = MVMCOptimizers.parton_calculate_ip(pstate.amp, qpw)
        abs(ip) < 1e-30 && continue
        e = MVMCOptimizers.parton_local_energy(pstate, data, ip)
        w = exp(2 * _ln_pj_ref(cfg, data)) * abs2(ip)
        MVMCOptimizers.parton_fill_sr_opt_o!(
            sr.sr_opt_o, pstate.amp, pstate.mfham, cfg, data, qpw, ip, n_proj;
            conjugate = conjugate)
        if store
            MVMCOptimizers.calculate_oo_store!(
                sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_o_store, sr.sr_opt_o,
                w, e, n_stored, sr.sr_opt_size)
        else
            MVMCOptimizers.calculate_oo!(
                sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_o, w, e, sr.sr_opt_size)
        end
        n_stored += 1
        wc += w
        etot += w * e
    end
    # 本番(parton_main_cal!)と同じく、軌道の器でディスパッチする組み立てを通す
    store && MVMCOptimizers.parton_finalize_oo_store!(
        pstate.mfham, sr.sr_opt_oo, sr.sr_opt_o_store, sr.sr_opt_size, n_stored)
    sr.sr_opt_oo ./= wc
    sr.sr_opt_ho ./= wc
    return etot / wc
end

_orb_energy(pstate, data, configs) = real(_orb_enumerate(pstate, data, configs; conjugate = true))

"全スロット(射影 + Φ、Re・Im)について上流の S と g を組む。"
function _orb_s_and_g(pstate, data)
    n_para = MVMCExpertModeParsers.count_variational_parameters(data)
    idx = Int[]
    for p = 1:n_para
        push!(idx, 2 * (p - 1))
        push!(idx, 2 * (p - 1) + 1)
    end
    n = length(idx)
    S = zeros(Float64, n, n)
    g = zeros(Float64, n)
    sr = pstate.state.sr_opt
    MVMCOptimizers.build_s_matrix_and_g_vector!(
        S, g, idx, sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_size,
        data.modpara.dsr_opt_sta_del, data.modpara.dsr_opt_step_dt)
    return S, g
end

"フィクスチャ: 5 サイト環・F=2・Ne=2 + Jastrow(距離クラス 2 個)、全 10 配置。"
function _orb_force_fixture()
    data = orbparam_toy_data(; n_site = 5, n_elec = 2, n_flavor = 2)
    _add_ring_jastrow!(data, [0.1, -0.2])
    configs = [[i, j] for i = 1:5 for j = (i + 1):5]
    return data, configs
end

"g / (−dt) と中心差分の最大相対誤差(射影は Re のみ、Φ は Re・Im)。"
function _orb_force_max_rel(pstate, data, configs, g)
    dt = data.modpara.dsr_opt_step_dt
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    δ = 1e-6
    max_rel = 0.0
    for p = 1:n_proj
        orig = data.jastrow_terms[p].value
        data.jastrow_terms[p].value = orig + δ
        ep = _orb_energy(pstate, data, configs)
        data.jastrow_terms[p].value = orig - δ
        em = _orb_energy(pstate, data, configs)
        data.jastrow_terms[p].value = orig
        fd = (ep - em) / (2δ)
        max_rel = max(max_rel, abs(g[2 * (p - 1) + 1] / (-dt) - fd) / max(abs(fd), 1e-8))
    end
    for k in eachindex(data.pmforb_params), (part, dir) in ((1, 1), (2, im))
        orig = data.pmforb_params[k]
        data.pmforb_params[k] = orig + dir * δ
        ep = _orb_energy(pstate, data, configs)
        data.pmforb_params[k] = orig - dir * δ
        em = _orb_energy(pstate, data, configs)
        data.pmforb_params[k] = orig
        fd = (ep - em) / (2δ)
        si = 2 * (n_proj + k - 1) + part
        max_rel = max(max_rel, abs(g[si] / (-dt) - fd) / max(abs(fd), 1e-8))
    end
    return max_rel
end

@testset "Φ-2: 力 = 勾配($(store ? "store" : "非 store")経路)" for store in (false, true)
    data, configs = _orb_force_fixture()
    pstate = _orb_pstate(data)
    e0 = _orb_enumerate(pstate, data, configs; conjugate = true, store = store)
    @test isfinite(real(e0))
    @test abs(imag(e0)) < 1e-10
    S, g = _orb_s_and_g(pstate, data)
    @test maximum(abs, g) > 1e-4           # 勾配が実際に立っていること
    @test _orb_force_max_rel(pstate, data, configs, g) < 1e-4
end

@testset "Φ-2: 共役シムを外すと勾配と合わない(回帰ガード)" begin
    data, configs = _orb_force_fixture()
    pstate = _orb_pstate(data)
    _orb_enumerate(pstate, data, configs; conjugate = false)
    _, g = _orb_s_and_g(pstate, data)
    @test _orb_force_max_rel(pstate, data, configs, g) > 1e-2
end

@testset "Φ-2: 共役シムは S を変えない(Jastrow 込み)" begin
    data, configs = _orb_force_fixture()
    pstate = _orb_pstate(data)
    _orb_enumerate(pstate, data, configs; conjugate = true)
    S_c, _ = _orb_s_and_g(pstate, data)
    _orb_enumerate(pstate, data, configs; conjugate = false)
    S_n, _ = _orb_s_and_g(pstate, data)
    @test isapprox(S_c, S_n; atol = 1e-12)
end

"δΦ_f をスロット並び(射影は 0、Φ は Re・Im 交互)の実ベクトルにする。"
function _orb_direction(data, dPhi::Vector{Matrix{ComplexF64}})
    mp = data.modpara
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    n_para = MVMCExpertModeParsers.count_variational_parameters(data)
    v = zeros(Float64, 2 * n_para)
    for f = 1:mp.nflavor, s = 1:mp.nsite, n = 1:mp.nelec
        k = MVMCOptimizers.parton_orbparam_flat_index(f, s, n, mp.nsite, mp.nelec)
        v[2 * (n_proj + k - 1) + 1] = real(dPhi[f][s, n])
        v[2 * (n_proj + k - 1) + 2] = imag(dPhi[f][s, n])
    end
    return v
end

@testset "Φ-3: S の冗長方向" begin
    data, configs = _orb_force_fixture()
    data.modpara.dsr_opt_sta_del = 0.0      # 対角を触らない生の S
    pstate = _orb_pstate(data)
    _orb_enumerate(pstate, data, configs; conjugate = true)
    S, _ = _orb_s_and_g(pstate, data)
    Φ = pstate.mfham.orbitals
    Ne = data.modpara.nelec
    tol = 1e-10 * opnorm(S)

    # (a) GL(Ne) の方向 δΦ_f = Φ_f·B_f は、Im Σ_f Tr B_f = 0 なら O が実定数なので S の零方向
    rng = MersenneTwister(3)
    B = [randn(rng, ComplexF64, Ne, Ne) for _ = 1:2]
    B[1][1, 1] -= im * imag(tr(B[1]) + tr(B[2]))
    v = _orb_direction(data, [Φ[f] * B[f] for f = 1:2])
    @test dot(v, S * v) <= tol * dot(v, v)

    # (b) 全体位相 B_1 = i·I: O = i·Ne の定数。上流の S には Im·Im 項がないので
    #     v'Sv = Ne² の正のペナルティになる(DESIGN §12.1)
    v = _orb_direction(data, [Φ[1] * (im * Matrix{ComplexF64}(I, Ne, Ne)), zero(Φ[2])])
    @test isapprox(dot(v, S * v), Ne^2; rtol = 1e-8)

    # (c) パートンゲージ δΦ_1 = D·Φ_1、δΦ_2 = −D·Φ_2(D は対角): 一次で Ψ は不変
    d = Diagonal(randn(rng, ComplexF64, data.modpara.nsite))
    v = _orb_direction(data, [d * Φ[1], -(d * Φ[2])])
    @test dot(v, S * v) <= tol * dot(v, v)
end

@testset "OO の組み立て: Φ 版(BLAS)は上流の素朴なループと丸め誤差内で一致、α 版はビット一致" begin
    rng = MersenneTwister(11)
    sr_opt_size = 7
    size_2 = 2 * sr_opt_size
    n = 13
    store = randn(rng, ComplexF64, size_2 * (n + 3))    # 余分な列があっても先頭 n 本だけ使う
    ref = zeros(ComplexF64, size_2^2)
    MVMCOptimizers.finalize_oo_store!(ref, store, sr_opt_size, n; nsrcg = false)

    phi = MVMCOptimizers.PartonDirectPhi(2, 1, 1)
    got = fill(ComplexF64(NaN), size_2^2)
    MVMCOptimizers.parton_finalize_oo_store!(phi, got, store, sr_opt_size, n)
    @test isapprox(got, ref; rtol = 1e-13)

    mf = MVMCOptimizers.PartonMFHamiltonian(2, 1, 1, 1)
    got_alpha = zeros(ComplexF64, size_2^2)
    MVMCOptimizers.parton_finalize_oo_store!(mf, got_alpha, store, sr_opt_size, n)
    @test got_alpha == ref

    # サンプル 0 本なら上流と同じく全ゼロ
    MVMCOptimizers.parton_finalize_oo_store!(phi, got, store, sr_opt_size, 0)
    @test all(iszero, got)
end
