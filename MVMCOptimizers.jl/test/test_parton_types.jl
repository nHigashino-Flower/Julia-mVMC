"""
パートンモードの型・アクセサ・委譲メソッドのテスト
--- parton-mode (fork addition) ---

DESIGN_parton.md §5(構造体カタログ)に対応する。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "パートン数アクセサ" begin
    @test MVMCOptimizers.n_parton_per_flavor(2) == 2
    @test MVMCOptimizers.n_parton_total(2, 3) == 6
    @test MVMCOptimizers.n_phys_particle(2) == 2
    @test MVMCOptimizers.n_site_flavor(4, 3) == 12

    mp = MVMCExpertModeParsers.ModParaParameters(; nsite = 4, nelec = 2, nflavor = 3)
    @test MVMCOptimizers.n_parton_per_flavor(mp) == 2
    @test MVMCOptimizers.n_parton_total(mp) == 6
    @test MVMCOptimizers.n_phys_particle(mp) == 2
    @test MVMCOptimizers.n_site_flavor(mp) == 12

    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite = 4
    data.modpara.nelec = 2
    data.modpara.nflavor = 3
    @test MVMCOptimizers.n_parton_total(data) == 6
    @test MVMCOptimizers.n_site_flavor(data) == 12
end

@testset "PartonAmplitudeData のストライド" begin
    amp = MVMCOptimizers.PartonAmplitudeData(2, 3, 2)   # n_qp=2, n_flavor=3, n_elec=2
    @test amp.n_qp == 2 && amp.n_flavor == 3 && amp.n_elec == 2
    @test length(amp.det_a) == 6
    @test length(amp.inv_a) == 6 * 4

    @test MVMCOptimizers.block_index(amp, 1, 1) == 1
    @test MVMCOptimizers.block_index(amp, 1, 3) == 3
    @test MVMCOptimizers.block_index(amp, 2, 1) == 4
    @test MVMCOptimizers.block_index(amp, 2, 3) == 6

    B = MVMCOptimizers.inv_block(amp, 2, 1)   # block 4
    @test size(B) == (2, 2)
    B[1, 2] = ComplexF64(7, 0)                # view であること(コピーでない)
    @test amp.inv_a[(4 - 1) * 4 + 3] == ComplexF64(7, 0)   # 列優先で (1,2) は 3 番目

    # 全ブロックが重ならないこと
    for qp = 1:2, f = 1:3
        fill!(MVMCOptimizers.inv_block(amp, qp, f), ComplexF64(qp * 10 + f, 0))
    end
    for qp = 1:2, f = 1:3
        @test all(==(ComplexF64(qp * 10 + f, 0)), MVMCOptimizers.inv_block(amp, qp, f))
    end
end

@testset "PartonConfiguration: 配置と固縛" begin
    cfg = MVMCOptimizers.PartonConfiguration(4, 2, 3, 5)  # n_site, n_elec, n_flavor, n_sample
    @test cfg.n_site == 4 && cfg.n_elec == 2 && cfg.n_flavor == 3
    @test cfg.burn_flag == false
    @test length(cfg.ele_idx) == 3 * 2
    @test length(cfg.ele_cfg) == 3 * 4
    @test length(cfg.ele_num) == 3 * 4
    @test length(cfg.stored_ele_idx) == 5 * 3 * 2

    for f = 1:3
        MVMCOptimizers.place_particle!(cfg, f, 1, 1)
        MVMCOptimizers.place_particle!(cfg, f, 2, 3)
    end
    @test MVMCOptimizers.particle_site(cfg, 1, 1) == 1
    @test MVMCOptimizers.particle_site(cfg, 3, 2) == 3
    @test MVMCOptimizers.is_occupied(cfg, 1)
    @test MVMCOptimizers.is_occupied(cfg, 3)
    @test !MVMCOptimizers.is_occupied(cfg, 2)
    @test MVMCOptimizers.assert_flavors_locked(cfg) === nothing

    # 固縛移動: 全フレーバー同時
    for f = 1:3
        MVMCOptimizers.move_particle!(cfg, f, 1, 1, 2)
    end
    @test MVMCOptimizers.particle_site(cfg, 2, 1) == 2
    @test !MVMCOptimizers.is_occupied(cfg, 1)
    @test MVMCOptimizers.is_occupied(cfg, 2)
    @test MVMCOptimizers.assert_flavors_locked(cfg) === nothing

    # 固縛が破れたら検出する(1 フレーバーだけ動かす)
    MVMCOptimizers.move_particle!(cfg, 1, 2, 3, 4)
    @test_throws Exception MVMCOptimizers.assert_flavors_locked(cfg)
end

@testset "PartonSamplingWorkspace の確保" begin
    ws = MVMCOptimizers.PartonSamplingWorkspace(3, 6)   # n_elec=3, n_blocks=n_qp*n_flavor=6
    @test size(ws.a_scratch) == (3, 3)
    @test length(ws.ratio_blocks) == 6
    @test length(ws.u_buf) == 3
    @test length(ws.v_buf) == 3
    @test length(ws.col_buf) == 3
end

@testset "PartonMFHamiltonian の確保" begin
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 3)  # n_site, n_elec, n_flavor, n_idx
    @test mfham.n_idx == 3
    @test length(mfham.h_mf) == 2
    @test size(mfham.h_mf[1]) == (4, 4)
    @test length(mfham.eig_vals) == 2 && length(mfham.eig_vals[1]) == 4
    @test size(mfham.eig_vecs[1]) == (4, 4)
    @test size(mfham.orbitals[1]) == (4, 2)
    @test length(mfham.dorbitals) == 2
    @test length(mfham.dorbitals[1]) == 6                  # 2 * n_idx 個の実自由度
    @test size(mfham.dorbitals[1][1]) == (4, 2)
    @test size(mfham.dh_uo_scratch) == (4, 2)
    @test length(mfham.template) == 3
    @test length(mfham.is_onsite_group) == 3
end

@testset "PartonMFTemplateEntry" begin
    e = MVMCOptimizers.PartonMFTemplateEntry(1, 2, 1, ComplexF64(-1.0, 0.5))
    @test e.site1 == 1 && e.site2 == 2 && e.flavor == 1
    @test e.coeff == ComplexF64(-1.0, 0.5)
    @test isbitstype(MVMCOptimizers.PartonMFTemplateEntry)
end
