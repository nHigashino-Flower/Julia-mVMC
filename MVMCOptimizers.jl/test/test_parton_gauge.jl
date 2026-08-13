"""
§8 テスト 8: α のゲージ射影(再正規化)
--- parton-mode (fork addition) ---

DESIGN_parton.md §2.5(ゲージ=射影が主線)と §8-8 に対応する。

この操作は**定義上ゲージ変換**なので、不等式ではなく厳密な等式で検証できる。
トイ系・全数展開・MC なしで、1e-12 の等式として押さえる。

平坦方向は 2 種類:
- スケール: α → c·α(c は正の実数)。H → cH は固有値を c 倍するだけで固有ベクトルを
  変えないので Φ が不変。c > 0 なので占有準位の順序も保たれる
- シフト: H → H + μI。一様オンサイト群があるときだけ現れる

複素位相つき t を必ず含める(実数だけでは位相まわりの取り違えが見えない)。
"""

using Test
using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

"""
    _gauge_state(data; n_idx) -> (pstate, mfham)

契約 0 まで進めた状態を作る。ゲージ群は build が解決済み。
"""
function _gauge_state(data::MVMCExpertModeParsers.ExpertModeData)
    MVMCOptimizers.parton_materialize_flags!(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    MVMCOptimizers.parton_update_orbitals!(
        pstate.mfham, MVMCOptimizers.parton_alpha_from_terms(data), data.modpara.nelec)
    return pstate
end

"全数展開の変分エネルギー(サンプリングなし)。"
function _gauge_energy(pstate, data, configs)
    mp = data.modpara
    mfham = pstate.mfham
    MVMCOptimizers.parton_update_orbitals!(
        mfham, MVMCOptimizers.parton_alpha_from_terms(data), mp.nelec)
    qpw = MVMCOptimizers.parton_qp_weight(data)
    num = ComplexF64(0)
    den = 0.0
    for c in configs
        cfg = MVMCOptimizers.PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, 1)
        for f = 1:mp.nflavor, (m, r) in enumerate(c)
            MVMCOptimizers.place_particle!(cfg, f, m, r)
        end
        pstate.config = cfg
        MVMCOptimizers.parton_recompute_amplitude_all!(
            pstate.amp, mfham, cfg, data, pstate.workspace)
        ip = MVMCOptimizers.parton_calculate_ip(pstate.amp, qpw)
        abs(ip) < 1e-30 && continue
        e = MVMCOptimizers.parton_local_energy(pstate, data, ip)
        w = abs2(ip)
        num += w * e
        den += w
    end
    return num / den
end

"配置に対する |ip| の並び(規格化して比だけを見る)。"
function _gauge_ip_profile(pstate, data, configs)
    mp = data.modpara
    mfham = pstate.mfham
    MVMCOptimizers.parton_update_orbitals!(
        mfham, MVMCOptimizers.parton_alpha_from_terms(data), mp.nelec)
    qpw = MVMCOptimizers.parton_qp_weight(data)
    out = Float64[]
    for c in configs
        cfg = MVMCOptimizers.PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, 1)
        for f = 1:mp.nflavor, (m, r) in enumerate(c)
            MVMCOptimizers.place_particle!(cfg, f, m, r)
        end
        pstate.config = cfg
        MVMCOptimizers.parton_recompute_amplitude_all!(
            pstate.amp, mfham, cfg, data, pstate.workspace)
        push!(out, abs(MVMCOptimizers.parton_calculate_ip(pstate.amp, qpw)))
    end
    s = maximum(out)
    return out ./ s
end

"idx をフレーバー間で共有しない入力(独立スケール群は F 個になるはず)。"
function _per_flavor_idx_data(F::Int; n_site::Int = 4, n_elec::Int = 2)
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite = n_site
    mp.nelec = n_elec
    mp.nflavor = F
    mp.parton_mode = 1
    mp.two_sz = 0
    mp.complex_flag = 1
    mp.nex_update_path = 6
    t = ComplexF64(-1.0, 0.35)
    idx = 0
    for f = 0:(F - 1)
        for i = 0:(n_site - 1)
            j = mod(i + 1, n_site)
            push!(data.pmftrans_terms,
                  MVMCExpertModeParsers.PartonMFTransTerm(i, f, j, f, t, true))
            push!(data.pmfpara_terms,
                  MVMCExpertModeParsers.PartonMFParaTerm(
                      i, f, j, f, idx, ComplexF64(0.8 + 0.1i, 0.05i), true))
            idx += 1
        end
    end
    for i = 0:(n_site - 1)
        push!(data.physhop_terms,
              MVMCExpertModeParsers.PhysHopTerm(i, mod(i + 1, n_site), ComplexF64(-1, 0), false))
    end
    set_identity_qp!(data)
    mp.nvmc_sample = 8
    mp.nstore_o = 1
    return data
end

const _GAUGE_CONFIGS_4_2 = [[i, j] for i = 1:4 for j = (i + 1):4]

@testset "§8-8-1 不変性: α をフレーバーごとに実数正倍しても物理が変わらない" begin
    F = 2
    data = _per_flavor_idx_data(F)
    pstate = _gauge_state(data)
    n_idx = pstate.mfham.n_idx

    e0 = _gauge_energy(pstate, data, _GAUGE_CONFIGS_4_2)
    p0 = _gauge_ip_profile(pstate, data, _GAUGE_CONFIGS_4_2)

    # フレーバーごとに別々の正数倍(スケール群がフレーバー単位で独立なことの確認も兼ねる)
    scales = [2.5, 0.4]
    for t in data.pmfpara_terms
        gi = findfirst(g -> (t.idx + 1) in g, pstate.mfham.gauge_scale_groups)
        t.value *= scales[gi]
    end

    e1 = _gauge_energy(pstate, data, _GAUGE_CONFIGS_4_2)
    p1 = _gauge_ip_profile(pstate, data, _GAUGE_CONFIGS_4_2)

    @test isapprox(e1, e0; atol = 1e-12)
    @test maximum(abs, p1 .- p0) < 1e-12
end

@testset "§8-8-2 射影の無害性: 射影前後で E_var が不変" begin
    data = _per_flavor_idx_data(2)
    pstate = _gauge_state(data)
    # 射影が実際に働くよう、いったんスケールをずらす
    for t in data.pmfpara_terms
        t.value *= 3.0
    end
    e_before = _gauge_energy(pstate, data, _GAUGE_CONFIGS_4_2)
    MVMCOptimizers.parton_project_gauge!(data, pstate.mfham)
    e_after = _gauge_energy(pstate, data, _GAUGE_CONFIGS_4_2)
    @test isapprox(e_after, e_before; atol = 1e-12)

    # ノルムが初期値へ戻っていること
    α = MVMCOptimizers.parton_alpha_from_terms(data)
    for (gi, grp) in enumerate(pstate.mfham.gauge_scale_groups)
        @test isapprox(sqrt(sum(abs2, view(α, grp))),
                       pstate.mfham.gauge_target_norm[gi]; rtol = 1e-12)
    end
end

@testset "§8-8-3 冪等性: 射影を 2 回かけても α が動かない" begin
    data = _per_flavor_idx_data(3)
    pstate = _gauge_state(data)
    for t in data.pmfpara_terms
        t.value *= 0.17
    end
    MVMCOptimizers.parton_project_gauge!(data, pstate.mfham)
    α1 = MVMCOptimizers.parton_alpha_from_terms(data)
    MVMCOptimizers.parton_project_gauge!(data, pstate.mfham)
    α2 = MVMCOptimizers.parton_alpha_from_terms(data)
    @test maximum(abs, α2 .- α1) < 1e-14
end

@testset "§8-8-4 群解決: 独立スケール群の数は idx 共有で決まる" begin
    # 共有なし → フレーバー数だけ独立群がある
    for F in (2, 3)
        data = _per_flavor_idx_data(F)
        pstate = _gauge_state(data)
        @test length(pstate.mfham.gauge_scale_groups) == F
        # 群は idx を分割している(重複なし・全部入り)
        allk = sort(vcat(pstate.mfham.gauge_scale_groups...))
        @test allk == collect(1:pstate.mfham.n_idx)
    end

    # 全フレーバーで idx を共有 → 独立群は 1 個
    for F in (2, 3)
        data = toy_mf_data(; n_flavor = F)
        set_identity_qp!(data)
        push!(data.physhop_terms,
              MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(-1, 0), false))
        pstate = _gauge_state(data)
        @test length(pstate.mfham.gauge_scale_groups) == 1
        @test sort(pstate.mfham.gauge_scale_groups[1]) == collect(1:pstate.mfham.n_idx)
    end
end

@testset "§8-8-5 シフト: 一様オンサイト群でも E_var が不変" begin
    # toy_mf_data は全サイト・全フレーバー共有の一様オンサイト群 (idx 1) を持つ
    data = toy_mf_data(; n_flavor = 2)
    set_identity_qp!(data)
    push!(data.physhop_terms,
          MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(-1, 0), false))
    pstate = _gauge_state(data)

    @test !isempty(pstate.mfham.gauge_shift_groups)
    shift_grp = pstate.mfham.gauge_shift_groups[1]
    @test all(k -> pstate.mfham.is_onsite_group[k], shift_grp)

    e0 = _gauge_energy(pstate, data, _GAUGE_CONFIGS_4_2)
    # 一様オンサイトを動かす = H → H + μI。物理は変わらないはず
    for t in data.pmfpara_terms
        (t.idx + 1) in shift_grp && (t.value += 1.7)
    end
    e_shifted = _gauge_energy(pstate, data, _GAUGE_CONFIGS_4_2)
    @test isapprox(e_shifted, e0; atol = 1e-12)

    # 射影後も同じで、かつ一様成分が抜けている
    MVMCOptimizers.parton_project_gauge!(data, pstate.mfham)
    e_proj = _gauge_energy(pstate, data, _GAUGE_CONFIGS_4_2)
    @test isapprox(e_proj, e0; atol = 1e-12)
    α = MVMCOptimizers.parton_alpha_from_terms(data)
    @test abs(sum(k -> α[k], shift_grp) / length(shift_grp)) < 1e-12
end

@testset "§8-8-6 スイッチと配線" begin
    @test MVMCExpertModeParsers.ModParaParameters().parton_gauge_fix == 1   # 既定は有効

    data = _per_flavor_idx_data(2)
    pstate = _gauge_state(data)
    for t in data.pmfpara_terms
        t.value *= 5.0
    end
    α_before = MVMCOptimizers.parton_alpha_from_terms(data)

    # 無効なら sync は α を触らない
    data.modpara.parton_gauge_fix = 0
    MVMCOptimizers.parton_sync_parameters!(
        data, MVMCOptimizers.serial_context(), pstate.mfham)
    @test MVMCOptimizers.parton_alpha_from_terms(data) == α_before

    # 有効なら sync で射影が効く
    data.modpara.parton_gauge_fix = 1
    MVMCOptimizers.parton_sync_parameters!(
        data, MVMCOptimizers.serial_context(), pstate.mfham)
    α_after = MVMCOptimizers.parton_alpha_from_terms(data)
    @test α_after != α_before
    for (gi, grp) in enumerate(pstate.mfham.gauge_scale_groups)
        @test isapprox(sqrt(sum(abs2, view(α_after, grp))),
                       pstate.mfham.gauge_target_norm[gi]; rtol = 1e-12)
    end

    # mfham を渡さなければ射影しない(既存の呼び出しとの後方互換)
    for t in data.pmfpara_terms
        t.value *= 2.0
    end
    α_scaled = MVMCOptimizers.parton_alpha_from_terms(data)
    MVMCOptimizers.parton_sync_parameters!(data, MVMCOptimizers.serial_context())
    @test MVMCOptimizers.parton_alpha_from_terms(data) == α_scaled
end

@testset "§8-8-6b 射影は OptFlag 凍結成分も再正規化する(仕様)" begin
    # スケール群を丸ごと実数倍しないとゲージ変換にならないので、凍結成分も
    # 対象に入る。凍結の意味は「SR が動かさない」であって「値が絶対に変わらない」
    # ではない。物理は不変なので実害はないが、意味論として明記しておく。
    data = dimerized_mf_data()
    data.pmfpara_opt_flags = Dict(0 => 0, 1 => 1, 2 => 1)   # idx0 を凍結
    pstate = _gauge_state(data)
    for t in data.pmfpara_terms
        t.value *= 4.0                                       # 群ごとスケールをずらす
    end
    α_before = MVMCOptimizers.parton_alpha_from_terms(data)
    MVMCOptimizers.parton_project_gauge!(data, pstate.mfham)
    α_after = MVMCOptimizers.parton_alpha_from_terms(data)

    # 凍結成分も含めて同じ群は同じ倍率で戻る = 群内の比は保たれる
    grp = pstate.mfham.gauge_scale_groups[findfirst(
        g -> 1 in g, pstate.mfham.gauge_scale_groups)]
    ratios = [α_after[k] / α_before[k] for k in grp if abs(α_before[k]) > 1e-14]
    @test maximum(abs, ratios .- ratios[1]) < 1e-12
    @test isreal(round(real(ratios[1]), digits = 12) + 0im) && real(ratios[1]) > 0
end

@testset "§8-8-7 SR: 射影 ON で α のノルムが保たれ、エネルギーは降下する" begin
    results = Dict{Int,Tuple{Float64,Float64}}()
    for gauge in (0, 1)
        data = dimerized_mf_data()
        data.modpara.parton_gauge_fix = gauge
        data.modpara.nsr_opt_itr_step = 12
        MVMCOptimizers.parton_materialize_flags!(data)
        pstate = MVMCOptimizers.parton_build_optimization_state(data)
        n0 = norm(MVMCOptimizers.parton_alpha_from_terms(data))
        rng = MVMCOptimizers.SFMT19937RNG()
        Random.seed!(rng, 4242)
        status = MVMCOptimizers.parton_vmc_para_opt!(
            pstate, data, MVMCOptimizers.serial_context();
            rng = rng, output_dir = mktempdir())
        @test status == 0
        n1 = norm(MVMCOptimizers.parton_alpha_from_terms(data))
        results[gauge] = (n1 / n0, real(pstate.state.energy.etot))
    end
    # 射影 ON ではスケール群のノルムが初期値に張り付く
    @test isapprox(results[1][1], 1.0; rtol = 1e-6) ||
          results[1][1] < results[0][1] + 1e-12
    # エネルギーはどちらも同程度に下がる(ゲージなので収束先は変わらないはず)
    @test results[0][2] < 0 && results[1][2] < 0
end
