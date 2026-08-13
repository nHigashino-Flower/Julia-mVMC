"""
§8 テスト 14: フレーバー対称の det^F 高速路
--- parton-mode (fork addition) ---

DESIGN §9 M3 の前倒し(v3.9)。idx 写像と t^(f) が全フレーバーで一致する入力では
固縛下で A^(f) が全フレーバー同一になるので、ブロックを 1 フレーバー分だけ持ち

    ip = Σ_qp w_qp (det A_qp)^F,  比 = R^F,  O の Tr 部分 = F · Tr[A⁻¹ ∂A]

とできる。**数学的に同一の量を別経路で計算しているだけ**なので、検証は近似では
なく等式で行う:

- det_a / inv_a / ip / 比 / E_loc は**ビット一致**(積の評価順が物理フレーバーの
  ループのまま変わらないため)
- O だけは Tr の総和順が「(f,m) の一続きの走査」から「f=1 の小計 × F 加算」に
  変わるので浮動小数の結合則ぶんだけずれうる。ここは 1e-14 の相対誤差で押さえ、
  実測値を表示する(DESIGN v3.9 の決定ログに記録)

スイッチは `PartonFlavorSymFast`(既定 1 = 自動検出、0 = 強制無効)。
無効時は完全に従来経路(検出そのものが走らない)。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"data から (mfham, amp, ws, pstate) 一式を組む。"
function _detpow_state(data)
    MVMCOptimizers.parton_materialize_flags!(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    MVMCOptimizers.parton_update_orbitals!(
        pstate.mfham, MVMCOptimizers.parton_alpha_from_terms(data), data.modpara.nelec)
    MVMCOptimizers.parton_update_orbital_derivatives!(pstate.mfham, data.modpara.nelec)
    return pstate
end

"対称な dimerized 入力に非自明 QP(2 本)を張る。"
function _detpow_data(; F::Int = 2, sym_fast::Int = 1)
    data = F == 2 ? dimerized_mf_data() : per_bond_mf_data(F; n_site = 4, n_elec = 2)
    n_site = data.modpara.nsite
    shifts = [collect(1:n_site), [collect(2:n_site); 1]]
    sgns = [ones(Int, n_site), ones(Int, n_site)]
    set_shift_qp!(data, shifts, sgns, ComplexF64[0.6, 0.4])
    data.modpara.parton_flavor_sym_fast = sym_fast
    return data
end

@testset "§8-14 検出: 対称入力で ON、非対称・強制無効で OFF" begin
    # 対称(dimerized は idx・t とも全フレーバー共有)
    ps = _detpow_state(_detpow_data())
    @test ps.mfham.flavor_symmetric
    @test ps.amp.n_stored == 1

    # 強制無効
    ps_off = _detpow_state(_detpow_data(; sym_fast = 0))
    @test !ps_off.mfham.flavor_symmetric
    @test ps_off.amp.n_stored == ps_off.amp.n_flavor

    # 非対称 (a): idx をフレーバーで割る(orbit_flavor 相当)
    data = _detpow_data()
    n_idx0 = maximum(t -> t.idx, data.pmfpara_terms) + 1
    for t in data.pmfpara_terms
        t.flavor1 == 1 && (t.idx += n_idx0)
    end
    ps_a = _detpow_state(data)
    @test !ps_a.mfham.flavor_symmetric

    # 非対称 (b): idx は共有のまま t だけフレーバーで変える(immutable なので差し替え)
    data = _detpow_data()
    data.pmftrans_terms = [
        t.flavor1 == 1 ?
        MVMCExpertModeParsers.PartonMFTransTerm(
            t.site1, t.flavor1, t.site2, t.flavor2, t.value * 1.5, t.is_complex) : t
        for t in data.pmftrans_terms
    ]
    ps_b = _detpow_state(data)
    @test !ps_b.mfham.flavor_symmetric
end

@testset "§8-14 等式: 高速路 ON = OFF(F=$F)" for F in (2, 3)
    ps_on = _detpow_state(_detpow_data(; F = F))
    ps_off = _detpow_state(_detpow_data(; F = F, sym_fast = 0))
    data_on = _detpow_data(; F = F)
    data_off = _detpow_data(; F = F, sym_fast = 0)
    mp = data_on.modpara
    Ne, n_qp = mp.nelec, 2
    qpw_on = MVMCOptimizers.parton_qp_weight(data_on)
    qpw_off = MVMCOptimizers.parton_qp_weight(data_off)
    n_idx = ps_on.mfham.n_idx

    # メモリが 1/F になっていること
    @test length(ps_on.amp.inv_a) * F == length(ps_off.amp.inv_a)
    @test length(ps_on.amp.det_a) * F == length(ps_off.amp.det_a)

    sites = [1, 3]
    for (ps, data) in ((ps_on, data_on), (ps_off, data_off))
        cfg = toy_config(data, sites)
        ps.config = cfg
        MVMCOptimizers.parton_recompute_amplitude_all!(
            ps.amp, ps.mfham, cfg, data, ps.workspace)
    end

    # det / inv: アクセサ越しに全 (qp, f) でビット一致
    for qp = 1:n_qp, f = 1:F
        @test ps_on.amp.det_a[MVMCOptimizers.block_index(ps_on.amp, qp, f)] ===
              ps_off.amp.det_a[MVMCOptimizers.block_index(ps_off.amp, qp, f)]
        @test MVMCOptimizers.inv_block(ps_on.amp, qp, f) ==
              MVMCOptimizers.inv_block(ps_off.amp, qp, f)
    end

    # ip / E_loc: ビット一致
    ip_on = MVMCOptimizers.parton_calculate_ip(ps_on.amp, qpw_on)
    ip_off = MVMCOptimizers.parton_calculate_ip(ps_off.amp, qpw_off)
    @test ip_on === ip_off
    e_on = MVMCOptimizers.parton_local_energy(ps_on, data_on, ip_on)
    e_off = MVMCOptimizers.parton_local_energy(ps_off, data_off, ip_off)
    @test e_on === e_off

    # 契約 2 の比: ビット一致
    m, r_new = 1, 2
    r_on, ipn_on = MVMCOptimizers.parton_amplitude_ratio!(
        ps_on.workspace, ps_on.amp, ps_on.mfham, data_on, qpw_on, m, r_new)
    r_off, ipn_off = MVMCOptimizers.parton_amplitude_ratio!(
        ps_off.workspace, ps_off.amp, ps_off.mfham, data_off, qpw_off, m, r_new)
    @test r_on === r_off
    @test ipn_on === ipn_off

    # 契約 3 の高速更新後もブロックが一致(乗法更新の等価性)
    st_on = MVMCOptimizers.parton_update_amplitude!(
        ps_on.amp, ps_on.mfham, data_on, ps_on.workspace, m, r_new)
    st_off = MVMCOptimizers.parton_update_amplitude!(
        ps_off.amp, ps_off.mfham, data_off, ps_off.workspace, m, r_new)
    @test st_on == st_off
    for qp = 1:n_qp, f = 1:F
        @test ps_on.amp.det_a[MVMCOptimizers.block_index(ps_on.amp, qp, f)] ===
              ps_off.amp.det_a[MVMCOptimizers.block_index(ps_off.amp, qp, f)]
        @test MVMCOptimizers.inv_block(ps_on.amp, qp, f) ==
              MVMCOptimizers.inv_block(ps_off.amp, qp, f)
    end

    # O: Tr の総和順だけが変わるので 1e-14 の相対誤差(実測を表示)
    for (ps, data) in ((ps_on, data_on), (ps_off, data_off))
        cfg = toy_config(data, sites)
        ps.config = cfg
        MVMCOptimizers.parton_recompute_amplitude_all!(
            ps.amp, ps.mfham, cfg, data, ps.workspace)
    end
    o_on = zeros(ComplexF64, 2 * n_idx + 2)
    o_off = zeros(ComplexF64, 2 * n_idx + 2)
    MVMCOptimizers.parton_fill_sr_opt_o!(
        o_on, ps_on.amp, ps_on.mfham, ps_on.config, data_on, qpw_on, ip_on, 0)
    MVMCOptimizers.parton_fill_sr_opt_o!(
        o_off, ps_off.amp, ps_off.mfham, ps_off.config, data_off, qpw_off, ip_off, 0)
    dev = maximum(abs, o_on .- o_off) / max(maximum(abs, o_off), 1e-300)
    @test dev < 1e-14
    println("F=$F: O の相対乖離 max = $dev")
end
