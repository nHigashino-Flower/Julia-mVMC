"""
T2: Pfaffian パートンモードの O と SR の配線(DESIGN_parton_pf.md §1.5・§9)
--- parton-mode (fork addition) ---

- O と ln ip の中心差分の一致(Re・Im を別々に。idx の共有・符号・QP・Jastrow 込み)
- 力 = 勾配: 全配置を列挙し、上流の `calculate_oo!` / `calculate_oo_store!` /
  `build_s_matrix_and_g_vector!` を通した g/(−dt) が ∂E/∂Re p・∂E/∂Im p の中心差分に一致する
  (store・非 store の両経路)。共役シムを外すと崩れる。共役シムは S を変えない
- T1-e: 初期点 f = ΞJΞᵀ では、Im Tr(J⁻¹δJ) = 0 の J 方向が S の零方向になる
  (RedCut は対角で判定するので、この方向は落とせない — 記録用)
"""

using Test
using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

const MOG = MVMCOptimizers

"""
    shared_pair_terms(rng, Ns, K, n_idx)

全ての非順序対を n_idx 個の idx へ乱数で割り振り(全 idx を使う)、符号も乱数にした def。
行の向き(小さい合成添字を先に書くか)も乱数にする。
"""
function shared_pair_terms(rng, Ns::Int, K::Int, n_idx::Int)
    base = full_pair_terms(Ns, K)
    n_idx <= length(base) || error("too many idx")
    assign = vcat(collect(0:(n_idx - 1)), rand(rng, 0:(n_idx - 1), length(base) - n_idx))
    shuffle!(rng, assign)
    out = MVMCExpertModeParsers.PartonPairOrbitalTerm[]
    for (t, k) in zip(base, assign)
        sgn = rand(rng, (-1, 1))
        if rand(rng, Bool)
            push!(out, MVMCExpertModeParsers.PartonPairOrbitalTerm(
                t.site1, t.flavor1, t.site2, t.flavor2, k, sgn))
        else
            push!(out, MVMCExpertModeParsers.PartonPairOrbitalTerm(
                t.site2, t.flavor2, t.site1, t.flavor1, k, sgn))
        end
    end
    return out
end

"勾配テスト用の Pf モード入力(環の複素ホップ・最近接クーロン・Jastrow・共有 def・乱数 QP)。"
function pair_grad_data(; n_site = 5, n_elec = 2, n_flavor = 2, n_idx = 12, n_qp = 2,
                        seed = 5, jastrow = [0.1, -0.2])
    rng = MersenneTwister(seed)
    data = pair_toy_data(n_site = n_site, n_elec = n_elec, n_flavor = n_flavor)
    for i = 0:(n_site - 1)
        j = mod(i + 1, n_site)
        push!(data.physhop_terms,
              MVMCExpertModeParsers.PhysHopTerm(i, j, ComplexF64(-1.0, 0.3), true))
        push!(data.coulomb_inter_terms, MVMCExpertModeParsers.CoulombInterTerm(i, j, 0.5))
    end
    n_qp > 1 && random_ring_qp!(rng, data, n_qp)
    jastrow === nothing || _add_ring_jastrow!(data, jastrow)
    data.parton_pair_terms = shared_pair_terms(rng, n_site, n_flavor, n_idx)
    data.parton_pair_params = randn(rng, ComplexF64, n_idx)
    mp = data.modpara
    mp.nvmc_warmup = 50
    mp.nvmc_interval = 1
    mp.nvmc_sample = 200
    mp.parton_block_update_size = 8
    mp.nsr_opt_itr_step = 3
    mp.nsr_opt_itr_smp = 1
    mp.nstore_o = 1
    mp.dsr_opt_step_dt = 0.02
    mp.dsr_opt_red_cut = 1e-8
    mp.dsr_opt_sta_del = 0.02
    MOG.parton_pair_materialize_flags!(data)
    return data
end

"ln ip(現在の p)。"
function _pair_ln_ip(pstate, data, cfg)
    MOG.parton_pair_load!(pstate.mfham, data.parton_pair_params)
    MOG.parton_recompute_amplitude_all!(pstate.amp, pstate.mfham, cfg, data, pstate.workspace)
    return log(MOG.parton_calculate_ip(pstate.amp, MOG.parton_qp_weight(data)))
end

@testset "T2: O = ln ip の中心差分(Re・Im、共有 idx・符号・QP・Jastrow 込み、K = 2, 3)" begin
    for (K, N, Ns, n_idx, n_qp) in ((2, 2, 5, 12, 2), (3, 2, 4, 20, 2), (2, 3, 6, 25, 3))
        data = pair_grad_data(n_site = Ns, n_elec = N, n_flavor = K, n_idx = n_idx,
                              n_qp = n_qp, jastrow = nothing)
        pstate = MOG.parton_pair_build_state(data)
        n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
        qpw = MOG.parton_qp_weight(data)
        rng = MersenneTwister(K * 100 + N)
        δ = 1e-6
        max_rel = 0.0
        for trial = 1:4
            sites = randperm(rng, Ns)[1:N]
            cfg = _jast_place!(pstate, data, sites)
            ip = MOG.parton_calculate_ip(pstate.amp, qpw)
            o = zeros(ComplexF64, length(pstate.state.sr_opt.sr_opt_o))
            MOG.parton_fill_sr_opt_o!(o, pstate.amp, pstate.mfham, cfg, data, qpw, ip, n_proj;
                                      conjugate = false)
            for k = 1:n_idx, (part, dir) in ((1, 1), (2, im))
                orig = data.parton_pair_params[k]
                data.parton_pair_params[k] = orig + dir * δ
                lp = _pair_ln_ip(pstate, data, cfg)
                data.parton_pair_params[k] = orig - dir * δ
                lm = _pair_ln_ip(pstate, data, cfg)
                data.parton_pair_params[k] = orig
                fd = (lp - lm) / (2δ)
                slot = part == 1 ? MOG.o_slot_re(n_proj + k) : MOG.o_slot_im(n_proj + k)
                max_rel = max(max_rel, abs(o[slot] - fd) / max(abs(fd), 1e-6))
            end
            _pair_ln_ip(pstate, data, cfg)
        end
        @test max_rel < 1e-6
    end
end

"全配置を列挙して OO・HO を蓄積し、厳密な ⟨E⟩ を返す(Φ 版の _orb_enumerate と同じ骨格)。"
function _pair_enumerate(pstate, data, configs; conjugate::Bool, store::Bool = false)
    MOG.parton_pair_load!(pstate.mfham, data.parton_pair_params)
    qpw = MOG.parton_qp_weight(data)
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
        ip = MOG.parton_calculate_ip(pstate.amp, qpw)
        abs(ip) < 1e-30 && continue
        e = MOG.parton_local_energy(pstate, data, ip)
        ln_pj = isempty(data.jastrow_terms) ? 0.0 : _ln_pj_ref(cfg, data)
        w = exp(2 * ln_pj) * abs2(ip)
        MOG.parton_fill_sr_opt_o!(sr.sr_opt_o, pstate.amp, pstate.mfham, cfg, data, qpw, ip,
                                  n_proj; conjugate = conjugate)
        if store
            MOG.calculate_oo_store!(sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_o_store, sr.sr_opt_o,
                                    w, e, n_stored, sr.sr_opt_size)
        else
            MOG.calculate_oo!(sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_o, w, e, sr.sr_opt_size)
        end
        n_stored += 1
        wc += w
        etot += w * e
    end
    store && MOG.parton_finalize_oo_store!(pstate.mfham, sr.sr_opt_oo, sr.sr_opt_o_store,
                                           sr.sr_opt_size, n_stored)
    sr.sr_opt_oo ./= wc
    sr.sr_opt_ho ./= wc
    return etot / wc
end

_pair_energy(pstate, data, configs) = real(_pair_enumerate(pstate, data, configs; conjugate = true))

function _pair_s_and_g(pstate, data)
    n_para = MVMCExpertModeParsers.count_variational_parameters(data)
    idx = Int[]
    for p = 1:n_para
        push!(idx, 2 * (p - 1))
        push!(idx, 2 * (p - 1) + 1)
    end
    S = zeros(Float64, length(idx), length(idx))
    g = zeros(Float64, length(idx))
    sr = pstate.state.sr_opt
    MOG.build_s_matrix_and_g_vector!(S, g, idx, sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_size,
                                     data.modpara.dsr_opt_sta_del, data.modpara.dsr_opt_step_dt)
    return S, g
end

"g / (−dt) と中心差分の最大相対誤差(射影は Re のみ、p は Re・Im)。"
function _pair_force_max_rel(pstate, data, configs, g)
    dt = data.modpara.dsr_opt_step_dt
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    δ = 1e-6
    max_rel = 0.0
    for p = 1:n_proj
        orig = data.jastrow_terms[p].value
        data.jastrow_terms[p].value = orig + δ
        ep = _pair_energy(pstate, data, configs)
        data.jastrow_terms[p].value = orig - δ
        em = _pair_energy(pstate, data, configs)
        data.jastrow_terms[p].value = orig
        fd = (ep - em) / (2δ)
        max_rel = max(max_rel, abs(g[2 * (p - 1) + 1] / (-dt) - fd) / max(abs(fd), 1e-8))
    end
    for k in eachindex(data.parton_pair_params), (part, dir) in ((1, 1), (2, im))
        orig = data.parton_pair_params[k]
        data.parton_pair_params[k] = orig + dir * δ
        ep = _pair_energy(pstate, data, configs)
        data.parton_pair_params[k] = orig - dir * δ
        em = _pair_energy(pstate, data, configs)
        data.parton_pair_params[k] = orig
        fd = (ep - em) / (2δ)
        si = 2 * (n_proj + k - 1) + part
        max_rel = max(max_rel, abs(g[si] / (-dt) - fd) / max(abs(fd), 1e-8))
    end
    return max_rel
end

const _PAIR_CONFIGS_5 = [[i, j] for i = 1:5 for j = (i + 1):5]

@testset "T2: 力 = 勾配($(store ? "store" : "非 store")経路)" for store in (false, true)
    data = pair_grad_data()
    pstate = MOG.parton_pair_build_state(data)
    e0 = _pair_enumerate(pstate, data, _PAIR_CONFIGS_5; conjugate = true, store = store)
    @test isfinite(real(e0))
    @test abs(imag(e0)) < 1e-10
    _, g = _pair_s_and_g(pstate, data)
    @test maximum(abs, g) > 1e-4
    @test _pair_force_max_rel(pstate, data, _PAIR_CONFIGS_5, g) < 1e-4
end

@testset "T2: 共役シムを外すと勾配と合わない(回帰ガード)・共役シムは S を変えない" begin
    data = pair_grad_data()
    pstate = MOG.parton_pair_build_state(data)
    _pair_enumerate(pstate, data, _PAIR_CONFIGS_5; conjugate = false)
    S_n, g_n = _pair_s_and_g(pstate, data)
    @test _pair_force_max_rel(pstate, data, _PAIR_CONFIGS_5, g_n) > 1e-2
    _pair_enumerate(pstate, data, _PAIR_CONFIGS_5; conjugate = true)
    S_c, _ = _pair_s_and_g(pstate, data)
    @test isapprox(S_c, S_n; atol = 1e-12)
end

@testset "OO の組み立て: Pf 版(BLAS)は上流の素朴なループと丸め誤差内で一致" begin
    rng = MersenneTwister(11)
    sr_opt_size = 7
    size_2 = 2 * sr_opt_size
    n = 13
    store = randn(rng, ComplexF64, size_2 * (n + 3))
    ref = zeros(ComplexF64, size_2^2)
    MOG.finalize_oo_store!(ref, store, sr_opt_size, n; nsrcg = false)
    src = MOG.PartonPairOrbital(2, 2, 1, full_pair_terms(2, 2))
    got = fill(ComplexF64(NaN), size_2^2)
    MOG.parton_finalize_oo_store!(src, got, store, sr_opt_size, n)
    @test isapprox(got, ref; rtol = 1e-13)
    MOG.parton_finalize_oo_store!(src, got, store, sr_opt_size, 0)
    @test all(iszero, got)
end

@testset "T1-e: 初期点 ΞJΞᵀ の J 方向は S の零方向(RedCut では落ちない)" begin
    K, N, Ns = 2, 2, 5
    data = pair_grad_data(n_site = Ns, n_elec = N, n_flavor = K, n_qp = 1, jastrow = nothing)
    rng = MersenneTwister(77)
    terms = full_pair_terms(Ns, K)
    data.parton_pair_terms = [MVMCExpertModeParsers.PartonPairOrbitalTerm(
        t.site1, t.flavor1, t.site2, t.flavor2, t.idx, t.sign) for t in terms]
    Φ, J, f = random_xi_j(rng, Ns, N, K)
    data.parton_pair_params = pair_params_from_matrix(f, terms, Ns)
    data.parton_pair_opt_flags = Dict{Int,Int}()
    data.modpara.dsr_opt_sta_del = 0.0          # 対角を触らない生の S
    MOG.parton_pair_materialize_flags!(data)
    pstate = MOG.parton_pair_build_state(data)
    _pair_enumerate(pstate, data, _PAIR_CONFIGS_5; conjugate = true)
    S, _ = _pair_s_and_g(pstate, data)

    # δJ を Im Tr(J⁻¹δJ) = 0 に調整する(全体位相の方向は上流の S で正のペナルティになるため)
    M = K * N
    Ξ = zeros(ComplexF64, K * Ns, M)
    for a = 1:K
        Ξ[(a - 1) * Ns + 1:a * Ns, (a - 1) * N + 1:a * N] = Φ[a]
    end
    δJ = _rand_antisym(rng, M)
    c = tr(J \ δJ) / 2
    δJ -= (2im * imag(c) / M) * J
    δp = pair_params_from_matrix(Ξ * δJ * transpose(Ξ), terms, Ns)
    v = zeros(Float64, size(S, 1))
    for k in eachindex(δp)
        v[2 * (k - 1) + 1] = real(δp[k])
        v[2 * (k - 1) + 2] = imag(δp[k])
    end
    @test dot(v, S * v) <= 1e-10 * opnorm(S) * dot(v, v)
    # 座標軸に沿わない(RedCut は対角 S_ii < max·RedCut の成分しか落とさない)
    support = findall(x -> abs(x) > 1e-8 * maximum(abs, v), v)
    @test length(support) > 2
    @test minimum(S[i, i] for i in support) > 1e-6 * maximum(diag(S))
end
