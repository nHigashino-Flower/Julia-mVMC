"""
§8 テスト 3: QP 写像 gather 版 vs per-QP 軌道の実体化版
--- parton-mode (fork addition) ---

DESIGN_parton.md §1.6 は「per-QP 軌道は実体化しない(行置換+符号なので
gather 時に写像を通す)」と決めている。この最適化が本当に同じ答を出すことを、
実体化版を素朴に組んで全ブロックで突き合わせる。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "QP: gather 版 = 実体化版(6 サイト・F=2・並進 3 種)" begin
    n_site, n_elec, n_flavor = 6, 3, 2
    data = toy_mf_data(;
        n_site = n_site,
        n_elec = n_elec,
        n_flavor = n_flavor,
        t = ComplexF64(-1.0, 0.25),
    )
    shifts = [collect(1:n_site), [collect(2:n_site); 1], [collect(3:n_site); 1; 2]]
    sgns = [
        ones(Int, n_site),
        [isodd(r) ? -1 : 1 for r = 1:n_site],
        [r <= 3 ? 1 : -1 for r = 1:n_site],
    ]
    qp_weight = set_shift_qp!(data, shifts, sgns, ComplexF64[0.5, 0.25, 0.25])

    mfham = build_toy_mfham(data)
    sites = [1, 4, 6]
    cfg = toy_config(data, sites)

    amp = MVMCOptimizers.PartonAmplitudeData(3, n_flavor, n_elec)
    ws = MVMCOptimizers.PartonSamplingWorkspace(n_elec, 3 * n_flavor)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    for qp = 1:3, f = 1:n_flavor
        # 実体化版: Φ_qp[r, n] = sgn_qp(r) * Φ[map_qp(r), n] を先に全サイト分作る
        Φqp = [
            sgns[qp][r] * mfham.orbitals[f][shifts[qp][r], n] for r = 1:n_site,
            n = 1:n_elec
        ]
        A = Φqp[sites, :]
        b = MVMCOptimizers.block_index(amp, qp, f)
        @test amp.det_a[b] ≈ det(A) rtol = 1e-12
        @test MVMCOptimizers.inv_block(amp, qp, f) * A ≈ I atol = 1e-11
    end

    # ip も実体化版から独立に組んで一致すること
    ip_materialised = sum(
        qp_weight[qp] * prod(
            det([
                sgns[qp][r] * mfham.orbitals[f][shifts[qp][r], n] for r in sites,
                n = 1:n_elec
            ]) for f = 1:n_flavor
        ) for qp = 1:3
    )
    @test MVMCOptimizers.parton_calculate_ip(amp, qp_weight) ≈ ip_materialised rtol = 1e-12
end

@testset "QP: 契約2 の比も実体化版と一致する" begin
    n_site, n_elec, n_flavor = 6, 2, 2
    data = toy_mf_data(;
        n_site = n_site,
        n_elec = n_elec,
        n_flavor = n_flavor,
        t = ComplexF64(-1.0, 0.25),
    )
    shifts = [collect(1:n_site), [collect(3:n_site); 1; 2]]
    sgns = [ones(Int, n_site), [isodd(r) ? -1 : 1 for r = 1:n_site]]
    qp_weight = set_shift_qp!(data, shifts, sgns, ComplexF64[0.5, 0.5])

    mfham = build_toy_mfham(data)
    sites = [1, 4]
    cfg = toy_config(data, sites)
    amp = MVMCOptimizers.PartonAmplitudeData(2, n_flavor, n_elec)
    ws = MVMCOptimizers.PartonSamplingWorkspace(n_elec, 2 * n_flavor)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    m, r_old, r_new = 1, 1, 3
    ratio, ip_new =
        MVMCOptimizers.parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m, r_new)

    # 実体化版で移動後の ip を直接組む
    new_sites = [r_new, sites[2]]
    ip_expected = sum(
        qp_weight[qp] * prod(
            det([
                sgns[qp][r] * mfham.orbitals[f][shifts[qp][r], n] for r in new_sites,
                n = 1:n_elec
            ]) for f = 1:n_flavor
        ) for qp = 1:2
    )
    @test ip_new ≈ ip_expected rtol = 1e-11
    @test ratio ≈ ip_expected / MVMCOptimizers.parton_calculate_ip(amp, qp_weight) rtol =
        1e-11
end
