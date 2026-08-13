"""
§8 テスト 7: SR の力ベクトルが変分エネルギーの勾配と一致すること
--- parton-mode (fork addition) ---

DESIGN_parton.md §7(MF スロットは蓄積境界で共役)と §8-7 に対応する。

実パラメータ θ についての変分エネルギーの勾配は

    ∂E/∂θ = 2 Re[ ⟨E_loc O_θ*⟩ − ⟨E_loc⟩⟨O_θ*⟩ ]

で O に共役が要る。ところが既存の `calculate_oo!` は `HO[j] += w·e·srOptO[j]` と
共役なしで蓄積し、`build_s_matrix_and_g_vector!` はその実部をそのまま力に使う。
射影パラメータの O は実数、f_ij は正則(2 スロットが val と val·im)なので既存の
無共役規約と整合するが、H が α* を含む MF ブロックは非正則なので整合しない。
そこで `_parton_conjugate_mf_slots!` が蓄積境界で MF スロットだけ共役にする。

このテストは、そのシムが載った状態で

    上流が組む力ベクトル g / (−DSROptStepDt)  ==  2 · dE/dθ(有限差分)

が成り立つことを、サンプリング誤差を排した全数展開で確かめる。あわせて
シムを外すと一致が壊れること(= このシムが飾りではないこと)も検査する。

上流の `calculate_oo!` と `build_s_matrix_and_g_vector!` を実際に呼ぶので、
上流側の規約が変わった場合もここで気づける。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"""
    _enumerate_state(pstate, data, configs, α; conjugate)

全配置を走査して (E, Wc, sr_opt_oo, sr_opt_ho) を返す。サンプリングはせず、
重みは厳密な |Ψ(x)|²。O は本番と同じ `parton_calculate_o!` で作り、
`conjugate = true` のときだけ本番と同じ `_parton_conjugate_mf_slots!` を通す。
蓄積は上流の `calculate_oo!` に委譲する。
"""
function _enumerate_state(pstate, data, configs, α::Vector{ComplexF64}; conjugate::Bool)
    mp = data.modpara
    mfham = pstate.mfham
    MVMCOptimizers.parton_update_orbitals!(mfham, α, mp.nelec)
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, mp.nelec)
    qpw = MVMCOptimizers.parton_qp_weight(data)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj

    sr = pstate.state.sr_opt
    fill!(sr.sr_opt_oo, 0)
    fill!(sr.sr_opt_ho, 0)

    wc = 0.0
    etot = ComplexF64(0)
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
        w = abs2(ip)                       # 重点サンプリングの厳密な重み

        fill!(sr.sr_opt_o, 0)
        sr.sr_opt_o[1] = ComplexF64(1)
        sr.sr_opt_o[2] = ComplexF64(0)
        MVMCOptimizers.parton_calculate_o!(
            sr.sr_opt_o, pstate.amp, mfham, cfg, data, qpw, ip, n_proj)
        conjugate && MVMCOptimizers._parton_conjugate_mf_slots!(
            sr.sr_opt_o, n_proj, mfham.n_idx)

        MVMCOptimizers.calculate_oo!(
            sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_o, w, e, sr.sr_opt_size)
        wc += w
        etot += w * e
    end

    # weight_average_sr_opt! と同じ正規化
    sr.sr_opt_oo ./= wc
    sr.sr_opt_ho ./= wc
    return etot / wc, sr.sr_opt_oo, sr.sr_opt_ho
end

"厳密な変分エネルギーだけが要るとき(有限差分用)。"
function _enumerate_energy(pstate, data, configs, α::Vector{ComplexF64})
    e, _, _ = _enumerate_state(pstate, data, configs, α; conjugate = true)
    return real(e)
end

"""
    _force_vector(pstate, data, n_idx) -> Vector{Float64}

上流の `build_s_matrix_and_g_vector!` に、蓄積済みの OO / HO をそのまま渡して
力ベクトルを組ませる。`smat_to_para_idx` は MF ブロックの実自由度を全部並べた
0-based の添字(実部が 2(k-1)、虚部が 2(k-1)+1)。
"""
function _force_vector(pstate, data, n_proj::Int, n_idx::Int)
    sr = pstate.state.sr_opt
    mp = data.modpara
    # 実部・虚部の 2 スロットで 1 パラメータ。射影ブロックの後ろに MF が並ぶ
    smat_to_para_idx = Int[]
    for k = 1:n_idx
        p = n_proj + k
        push!(smat_to_para_idx, 2 * (p - 1))        # Re → sr_opt_o[2p+1]
        push!(smat_to_para_idx, 2 * (p - 1) + 1)    # Im → sr_opt_o[2p+2]
    end
    n = length(smat_to_para_idx)
    S = zeros(Float64, n, n)     # 上流は線形添字で書くが型は Matrix{Float64}
    g = zeros(Float64, n)
    MVMCOptimizers.build_s_matrix_and_g_vector!(
        S,
        g,
        smat_to_para_idx,
        sr.sr_opt_oo,
        sr.sr_opt_ho,
        sr.sr_opt_size,
        mp.dsr_opt_sta_del,
        mp.dsr_opt_step_dt,
    )
    return g, S
end

@testset "§8-7 力ベクトル = 変分エネルギーの勾配(全数展開・有限差分)" begin
    F = 2
    n_site, n_elec = 4, 2
    data = per_bond_mf_data(F; n_site = n_site, n_elec = n_elec)
    MVMCOptimizers.parton_materialize_flags!(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    n_idx = pstate.mfham.n_idx
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    configs = [[i, j] for i = 1:n_site for j = (i + 1):n_site]

    α0 = MVMCOptimizers.parton_alpha_from_terms(data)
    e0, _, _ = _enumerate_state(pstate, data, configs, α0; conjugate = true)
    @test isfinite(real(e0))
    @test abs(imag(e0)) < 1e-10        # ⟨H⟩ はエルミートなので実数

    g, S = _force_vector(pstate, data, n_proj, n_idx)
    dt = data.modpara.dsr_opt_step_dt

    # 上流は g[si] = -dt · 2 · (Re⟨e O⟩ − Re⟨e⟩Re⟨O⟩) を組む。括弧の中身が
    # ∂E/∂θ の半分(勾配式の因子 2 がここに現れる)なので、期待される関係は
    #     g[si] / (−dt) == ∂E/∂θ
    δ = 1e-6
    max_rel = 0.0
    for k = 1:n_idx, part = 1:2
        αp = copy(α0)
        αm = copy(α0)
        d = part == 1 ? ComplexF64(δ) : ComplexF64(0, δ)
        αp[k] += d
        αm[k] -= d
        fd = (_enumerate_energy(pstate, data, configs, αp) -
              _enumerate_energy(pstate, data, configs, αm)) / (2δ)
        si = 2 * (k - 1) + part
        from_force = g[si] / (-dt)
        @test isapprox(from_force, fd; rtol = 1e-4, atol = 1e-8)
        max_rel = max(max_rel, abs(from_force - fd) / max(abs(fd), 1e-8))
    end
    @test max_rel < 1e-3

    # 勾配が全部ゼロだと上の検査が素通りしてしまうので、実際に効いていることを確認
    @test maximum(abs, g) > 1e-4
end

@testset "§8-7 シムが載っていないと勾配と合わない(回帰ガード)" begin
    # `_parton_conjugate_mf_slots!` を外すと力ベクトルが勾配からずれること。
    # このテストが緑である限り、シムを削除すると上のテストセットが赤になる。
    F = 2
    n_site, n_elec = 4, 2
    data = per_bond_mf_data(F; n_site = n_site, n_elec = n_elec)
    MVMCOptimizers.parton_materialize_flags!(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    n_idx = pstate.mfham.n_idx
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    configs = [[i, j] for i = 1:n_site for j = (i + 1):n_site]
    α0 = MVMCOptimizers.parton_alpha_from_terms(data)
    dt = data.modpara.dsr_opt_step_dt

    δ = 1e-6
    fd = Float64[]
    for k = 1:n_idx, part = 1:2
        αp = copy(α0)
        αm = copy(α0)
        d = part == 1 ? ComplexF64(δ) : ComplexF64(0, δ)
        αp[k] += d
        αm[k] -= d
        push!(fd, (_enumerate_energy(pstate, data, configs, αp) -
                   _enumerate_energy(pstate, data, configs, αm)) / (2δ))
    end

    _enumerate_state(pstate, data, configs, α0; conjugate = false)
    g_noshim, _ = _force_vector(pstate, data, n_proj, n_idx)
    from_force = g_noshim ./ (-dt)
    @test maximum(abs, from_force .- fd) > 0.05     # 明確にずれる
    # 符号すら合わない成分があること(単なるスケールのずれではない)
    @test any(i -> sign(from_force[i]) != sign(fd[i]) && abs(fd[i]) > 1e-3,
              eachindex(fd))
end

@testset "§8-7 共役シムは S 行列を変えない" begin
    # 共役は HO(力)だけに効き、S(計量)は実部しか使わないので不変であること。
    F = 2
    n_site, n_elec = 4, 2
    data = per_bond_mf_data(F; n_site = n_site, n_elec = n_elec)
    MVMCOptimizers.parton_materialize_flags!(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    n_idx = pstate.mfham.n_idx
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    configs = [[i, j] for i = 1:n_site for j = (i + 1):n_site]
    α0 = MVMCOptimizers.parton_alpha_from_terms(data)

    _enumerate_state(pstate, data, configs, α0; conjugate = true)
    _, S_shim = _force_vector(pstate, data, n_proj, n_idx)
    _enumerate_state(pstate, data, configs, α0; conjugate = false)
    _, S_plain = _force_vector(pstate, data, n_proj, n_idx)

    @test maximum(abs, S_shim .- S_plain) < 1e-12
    @test S_shim ≈ transpose(S_shim)          # 対称
    @test all(>=(0), diag(S_shim))            # 対角は非負(計量として妥当)
    @test maximum(abs, S_shim) > 1e-6         # 自明にゼロではない
end
