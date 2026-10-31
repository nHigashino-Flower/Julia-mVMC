"""
骨格との結合: 無変更の `parton_make_sample!` / `parton_local_energy` が Pf の器で回る
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §4.4。f = ΞJΞᵀ なら ip_Pf ∝ ip_det なので、同じ seed で回すと
det モード(Φ を直接持つ器)と**同じ乱数列を消費し、同じサンプル列・同じ試行数・受理数**になる
(受理判定は |比|² の比較で、比は丸め誤差の範囲で一致する)。E_loc も丸め誤差で一致する。
T3(SR を含めた一致)のサンプリング部分にあたる。
"""

using Test
using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

const MOS = MVMCOptimizers

"det モードの状態(Φ 直接変分の器)と、その Φ から f = ΞJΞᵀ で作った Pf モードの状態を返す。"
function _det_and_pair_states(rng, data)
    mp = data.modpara
    K, N, Ns = mp.nflavor, mp.nelec, mp.nsite
    ps_det = MOS.parton_orbparam_build_state(data)
    Φ = ps_det.mfham.orbitals
    M = K * N
    Ξ = zeros(ComplexF64, K * Ns, M)
    for a = 1:K
        Ξ[(a - 1) * Ns + 1:a * Ns, (a - 1) * N + 1:a * N] = Φ[a]
    end
    J = _rand_antisym(rng, M)
    f = Ξ * J * transpose(Ξ)
    terms = full_pair_terms(Ns, K)
    src = build_pair_source(data, terms, pair_params_from_matrix(f, terms, Ns))
    n_qp = length(data.qp_trans)
    ps_pf = MOS.PartonOptimizationState(
        ps_det.state,
        MOS.PartonPairAmplitudeData(n_qp, K, N),
        MOS.PartonConfiguration(Ns, N, K, mp.nvmc_sample; n_proj = 0),
        MOS.PartonPairSamplingWorkspace(n_qp, K, N),
        src,
        ps_det.physham,
    )
    return ps_det, ps_pf
end

@testset "骨格: f = ΞJΞᵀ なら det モードと同じサンプル列(QP なし・あり、K = 2, 3)" begin
    for (K, N, Ns, n_qp) in ((2, 2, 6, 1), (2, 3, 7, 3), (3, 2, 6, 2))
        rng = MersenneTwister(1000 * K + N)
        data = orbparam_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
        n_qp > 1 && random_ring_qp!(rng, data, n_qp)
        ps_det, ps_pf = _det_and_pair_states(rng, data)
        @test ps_pf isa MOS.PartonOptimizationState{MOS.PartonPairOrbital}

        for (ps, seed) in ((ps_det, 31), (ps_pf, 31))
            r = MOS.SFMT19937RNG()
            Random.seed!(r, seed)
            MOS.parton_make_sample!(ps, data, r)
        end
        cd, cp = ps_det.config, ps_pf.config
        @test cp.stored_ele_idx == cd.stored_ele_idx
        @test cp.counter[1] == cd.counter[1]              # 試行数(乱数の消費が同じ)
        @test cp.counter[2] == cd.counter[2]              # 受理数
        @test cp.counter[2] > 0

        # 保存済みの全サンプルで E_loc が丸め誤差の範囲で一致する(契約 2 を仮想移動に使う経路)
        qpw = MOS.parton_qp_weight(data)
        max_rel = 0.0
        for s = 1:data.modpara.nvmc_sample
            MOS.parton_restore_sample!(cd, s)
            MOS.parton_restore_sample!(cp, s)
            MOS.parton_recompute_amplitude_all!(ps_det.amp, ps_det.mfham, cd, data, ps_det.workspace)
            MOS.parton_recompute_amplitude_all!(ps_pf.amp, ps_pf.mfham, cp, data, ps_pf.workspace)
            e_det = MOS.parton_local_energy(ps_det, data, MOS.parton_calculate_ip(ps_det.amp, qpw))
            e_pf = MOS.parton_local_energy(ps_pf, data, MOS.parton_calculate_ip(ps_pf.amp, qpw))
            max_rel = max(max_rel, abs(e_pf - e_det) / max(abs(e_det), 1.0))
        end
        @test max_rel < 1e-10

        # 2 回目(burn からの再開)も一致する
        for (ps, seed) in ((ps_det, 32), (ps_pf, 32))
            r = MOS.SFMT19937RNG()
            Random.seed!(r, seed)
            MOS.parton_make_sample!(ps, data, r)
        end
        @test cp.stored_ele_idx == cd.stored_ele_idx
        @test cp.counter[1] == cd.counter[1]
        @test cp.counter[2] == cd.counter[2]
    end
end
