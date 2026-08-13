"""
§8 テスト 2: 契約 2/3(高速更新)を契約 1(厳密再計算)と突き合わせる
--- parton-mode (fork addition) ---

DESIGN_parton.md §8-2 に対応。複素位相つき t で行うのが要点で、
転置積を dot に取り違えても実数だけでは差が出ない。
サンプリング骨格(§4)の不変条件もここで確認する。
"""

using Test
using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

"全提案を受理しながら n 回の固縛移動を行い、amp を高速更新で追従させる。"
function _drive_fast_updates!(amp, cfg, mfham, data, ws, qp_weight, rng, n_moves)
    n_site = data.modpara.nsite
    done = 0
    while done < n_moves
        m = rand(rng, 1:cfg.n_elec)
        r_old = MVMCOptimizers.particle_site(cfg, 1, m)
        r_new = rand(rng, 1:n_site)
        (r_new == r_old || MVMCOptimizers.is_occupied(cfg, r_new)) && continue

        ratio, ip_new =
            MVMCOptimizers.parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m, r_new)
        abs(ratio) < 1e-8 && continue    # ノード近傍は飛ばす(この検証の対象外)

        # 受理: ①配置コミット → ②振幅更新(DESIGN §4 の順序不変条件)
        MVMCOptimizers.parton_update_ele_config!(cfg, m, r_old, r_new)
        st = MVMCOptimizers.parton_update_amplitude!(amp, mfham, data, ws, m, r_new)
        if st === :need_recompute
            MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
        else
            # 契約 2 が予告した ip と、更新後の ip が一致すること
            @test MVMCOptimizers.parton_calculate_ip(amp, qp_weight) ≈ ip_new rtol = 1e-9
        end
        done += 1
    end
    return nothing
end

@testset "契約2: 純粋(状態を変えない)" begin
    data = toy_mf_data()
    qp_weight = set_identity_qp!(data)
    mfham = build_toy_mfham(data)
    cfg = toy_config(data, [1, 3])
    amp = MVMCOptimizers.PartonAmplitudeData(1, 2, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    det_before = copy(amp.det_a)
    inv_before = copy(amp.inv_a)
    idx_before = copy(cfg.ele_idx)

    ratio, ip_new =
        MVMCOptimizers.parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, 1, 2)

    # amp も cfg も触っていない(棄却時の revert が要らない理由)
    @test amp.det_a == det_before
    @test amp.inv_a == inv_before
    @test cfg.ele_idx == idx_before

    # 予告した比を独立に検算: 実際に動かして厳密再計算した ip と比べる
    ip_old = MVMCOptimizers.parton_calculate_ip(amp, qp_weight)
    MVMCOptimizers.parton_update_ele_config!(cfg, 1, 1, 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    ip_exact = MVMCOptimizers.parton_calculate_ip(amp, qp_weight)
    @test ip_new ≈ ip_exact rtol = 1e-12
    @test ratio ≈ ip_exact / ip_old rtol = 1e-12
end

@testset "契約2/3: 200 回の高速更新後に厳密再計算と一致(複素 t)" begin
    data = toy_mf_data()
    qp_weight = set_identity_qp!(data)
    mfham = build_toy_mfham(data)
    cfg = toy_config(data, [1, 3])
    amp = MVMCOptimizers.PartonAmplitudeData(1, 2, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    rng = MersenneTwister(42)
    _drive_fast_updates!(amp, cfg, mfham, data, ws, qp_weight, rng, 200)

    det_fast = copy(amp.det_a)
    inv_fast = copy(amp.inv_a)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    @test maximum(abs, det_fast .- amp.det_a) < 1e-9 * maximum(abs, amp.det_a)
    @test maximum(abs, inv_fast .- amp.inv_a) < 1e-9 * maximum(abs, amp.inv_a)
end

@testset "契約2/3: 6 サイト・F=3・並進 QP でも一致" begin
    n_site, n_elec, n_flavor = 6, 2, 3
    data = toy_mf_data(;
        n_site = n_site,
        n_elec = n_elec,
        n_flavor = n_flavor,
        t = ComplexF64(-1.0, 0.3),
    )
    shift = [collect(2:n_site); 1]
    qp_weight = set_shift_qp!(
        data,
        [collect(1:n_site), shift],
        [ones(Int, n_site), [iseven(r) ? 1 : -1 for r = 1:n_site]],
        ComplexF64[0.5, 0.5],
    )
    mfham = build_toy_mfham(data)
    cfg = toy_config(data, [2, 5])
    amp = MVMCOptimizers.PartonAmplitudeData(2, n_flavor, n_elec)
    ws = MVMCOptimizers.PartonSamplingWorkspace(n_elec, 2 * n_flavor)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    rng = MersenneTwister(7)
    _drive_fast_updates!(amp, cfg, mfham, data, ws, qp_weight, rng, 150)

    det_fast = copy(amp.det_a)
    inv_fast = copy(amp.inv_a)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    @test maximum(abs, det_fast .- amp.det_a) < 1e-9 * maximum(abs, amp.det_a)
    @test maximum(abs, inv_fast .- amp.inv_a) < 1e-9 * maximum(abs, amp.inv_a)
end

@testset "骨格: parton_make_sample! が走り不変条件を保つ" begin
    data = toy_mf_data()
    set_identity_qp!(data)
    mp = data.modpara
    mp.nvmc_warmup = 5
    mp.nvmc_interval = 1
    mp.nvmc_sample = 8
    mp.parton_block_update_size = 4

    mfham = build_toy_mfham(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data; mfham = mfham)
    rng = MersenneTwister(1234)

    MVMCOptimizers.parton_make_sample!(pstate, data, rng)
    cfg = pstate.config

    @test cfg.burn_flag == true
    @test MVMCOptimizers.assert_flavors_locked(cfg) === nothing
    @test cfg.counter[1] > 0        # 試行
    @test cfg.counter[2] > 0        # 受理

    # 保存されたサンプルはどれも正しい固縛配置
    for s = 1:mp.nvmc_sample
        sites = [MVMCOptimizers.stored_particle_site(cfg, s, 1, m) for m = 1:cfg.n_elec]
        @test allunique(sites)
        @test all(r -> 1 <= r <= mp.nsite, sites)
        for f = 2:cfg.n_flavor, m = 1:cfg.n_elec
            @test MVMCOptimizers.stored_particle_site(cfg, s, f, m) == sites[m]
        end
    end

    # burn 再開: 2 回目は burn 配置から始まり、やはり整合が保たれる
    MVMCOptimizers.parton_make_sample!(pstate, data, rng)
    @test MVMCOptimizers.assert_flavors_locked(pstate.config) === nothing
    @test pstate.config.counter[2] > 0
end

@testset "骨格: サンプリング後も振幅が配置と整合している" begin
    data = toy_mf_data()
    qp_weight = set_identity_qp!(data)
    mp = data.modpara
    mp.nvmc_warmup = 4
    mp.nvmc_interval = 1
    mp.nvmc_sample = 5
    mp.parton_block_update_size = 3

    mfham = build_toy_mfham(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data; mfham = mfham)
    MVMCOptimizers.parton_make_sample!(pstate, data, MersenneTwister(99))

    amp_fast = copy(pstate.amp.det_a)
    MVMCOptimizers.parton_recompute_amplitude_all!(
        pstate.amp, mfham, pstate.config, data, pstate.workspace)
    @test maximum(abs, amp_fast .- pstate.amp.det_a) <
          1e-8 * maximum(abs, pstate.amp.det_a)
end
