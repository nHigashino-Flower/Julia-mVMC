"""
§8 テスト 1: 契約 1(gather + det)を全数展開と突き合わせる
--- parton-mode (fork addition) ---

DESIGN_parton.md §8-1 に対応。錨の冪等性もここで確認する。
比較対象はテスト内で素朴に組んだ行列で、契約 1 の実装経路を一切共有しない。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "契約1: 恒等 QP・複素 t での det / A⁻¹ 一致" begin
    data = toy_mf_data()
    qp_weight = set_identity_qp!(data)
    mfham = build_toy_mfham(data)
    cfg = toy_config(data, [1, 3])

    amp = MVMCOptimizers.PartonAmplitudeData(1, 2, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    for f = 1:2
        # 全数展開: A[m, n] = Φ[r_m, n](粒子が行、軌道が列)
        A = [mfham.orbitals[f][r, n] for r in (1, 3), n = 1:2]
        b = MVMCOptimizers.block_index(amp, 1, f)
        @test amp.det_a[b] ≈ det(A) rtol = 1e-13
        @test MVMCOptimizers.inv_block(amp, 1, f) * A ≈ I atol = 1e-12
    end

    ip = MVMCOptimizers.parton_calculate_ip(amp, qp_weight)
    @test ip ≈ qp_weight[1] * prod(amp.det_a) rtol = 1e-13
    @test abs(ip) > 1e-8   # ノードの上ではないこと
end

@testset "契約1: 錨の冪等性" begin
    data = toy_mf_data()
    set_identity_qp!(data)
    mfham = build_toy_mfham(data)
    cfg = toy_config(data, [2, 4])

    amp = MVMCOptimizers.PartonAmplitudeData(1, 2, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    det_before = copy(amp.det_a)
    inv_before = copy(amp.inv_a)

    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    @test amp.det_a == det_before
    @test amp.inv_a == inv_before
end

@testset "契約1: 6 サイト・F=3・非自明な qp_trans と符号" begin
    n_site, n_elec, n_flavor = 6, 2, 3
    data = toy_mf_data(;
        n_site = n_site,
        n_elec = n_elec,
        n_flavor = n_flavor,
        t = ComplexF64(-1.0, 0.3),
    )
    shift = [collect(2:n_site); 1]
    data.qp_trans = [collect(1:n_site), shift]
    data.qp_trans_sgn = [ones(Int, n_site), [iseven(r) ? 1 : -1 for r = 1:n_site]]
    qp_weight = ComplexF64[0.5, 0.5]

    mfham = build_toy_mfham(data)
    sites = [2, 5]
    cfg = toy_config(data, sites)

    amp = MVMCOptimizers.PartonAmplitudeData(2, n_flavor, n_elec)
    ws = MVMCOptimizers.PartonSamplingWorkspace(n_elec, 2 * n_flavor)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    for qp = 1:2, f = 1:n_flavor
        qmap = data.qp_trans[qp]
        qsgn = data.qp_trans_sgn[qp]
        # A[m, n] = sgn_qp(r_m) * Φ[map_qp(r_m), n](写像は順方向)
        A = [qsgn[r] * mfham.orbitals[f][qmap[r], n] for r in sites, n = 1:n_elec]
        b = MVMCOptimizers.block_index(amp, qp, f)
        @test amp.det_a[b] ≈ det(A) rtol = 1e-13
        @test MVMCOptimizers.inv_block(amp, qp, f) * A ≈ I atol = 1e-12
    end

    ip = MVMCOptimizers.parton_calculate_ip(amp, qp_weight)
    expected = sum(
        qp_weight[qp] *
        prod(amp.det_a[MVMCOptimizers.block_index(amp, qp, f)] for f = 1:n_flavor)
        for qp = 1:2
    )
    @test ip ≈ expected rtol = 1e-13
end

@testset "契約1: 特異ブロックは det = 0(ノードはエラーにしない)" begin
    data = toy_mf_data()
    set_identity_qp!(data)
    mfham = build_toy_mfham(data)
    cfg = toy_config(data, [1, 3])

    # 軌道の 2 列を同一にすると、どの配置でも A は特異になる
    for f = 1:2
        mfham.orbitals[f][:, 2] .= mfham.orbitals[f][:, 1]
    end
    amp = MVMCOptimizers.PartonAmplitudeData(1, 2, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    @test all(d -> abs(d) < 1e-12, amp.det_a)
    @test abs(MVMCOptimizers.parton_calculate_ip(amp, ComplexF64[1])) < 1e-12
end
