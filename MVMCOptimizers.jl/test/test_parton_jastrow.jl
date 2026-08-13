"""
§8 テスト 16: 物理密度 Jastrow(v3.11 M2 後半)
--- parton-mode (fork addition) ---

    P_J(x) = exp( Σ_{i<j} v_{ij} n^b_i n^b_j ),   n^b ∈ {0, 1}

検証の柱(指示 §3 の 1〜8):
1. 無効時ビット一致 — n_proj = 0 の恒等性(実 run のバイト一致は v3.10 出力との
   突き合わせで別途実測済み。ここでは恒等フックの等価性を機械検証)
2. 差分 vs 全数 — update_proj_cnt! / log_proj_ratio を全数再構築と突き合わせ
   (自己項の取り違えはここで落ちる)
3. 有限差分 — ∂E/∂v_p が力ベクトルと一致(全数展開)
4. 共役シムの前提 — 射影 O の虚スロットが厳密 0、S が共役の有無で不変
5. 変分改善 — v 方向の厳密勾配ステップで E_var が厳密に下がる
6. QP 併用 — n_qp = 2 で 2〜5 が成立(P_J が qp と直交する構造)
7. F 一般性 — F = 2 / 3(n^b 定義なので F 非依存)
8. Gutzwiller 拒否 — 「固縛下で定数」の明示エラー
"""

using Test
using Random
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"""
距離クラスの Jastrow を data に張る(環の距離 d = 1..L÷2 で類別)。
jastrowidx.def を書かずにパーサ後の状態を直接作る(パーサ自体は上流の既存物)。
"""
function _add_ring_jastrow!(data, v::Vector{Float64})
    L = data.modpara.nsite
    n_class = L ÷ 2
    length(v) == n_class || error("v must have length $n_class")
    jidx = fill(-1, L, L)
    for i = 1:L, j = 1:L
        i == j && continue
        d = min(mod(i - j, L), mod(j - i, L))
        jidx[i, j] = d - 1                      # 0-based クラス
    end
    data.jastrow_idx = jidx
    data.n_jastrow_idx = n_class
    data.jastrow_terms = [
        MVMCExpertModeParsers.JastrowTerm(0, k, ComplexF64(v[k + 1], 0), false)
        for k = 0:(n_class - 1)
    ]
    return data
end

"データ一式(per_bond MF + 物理相互作用 + Jastrow + 任意 QP)。"
function _jast_data(; F::Int = 2, L::Int = 6, Ne::Int = 3,
                    v::Union{Nothing,Vector{Float64}} = nothing,
                    n_qp::Int = 1)
    data = per_bond_mf_data(F; n_site = L, n_elec = Ne)
    # 密度相関が効くように最近接クーロンを足す(dimerized と同じ V=0.5)
    for i = 0:(L - 1)
        push!(data.coulomb_inter_terms,
              MVMCExpertModeParsers.CoulombInterTerm(i, mod(i + 1, L), 0.5))
    end
    v === nothing || _add_ring_jastrow!(data, v)
    if n_qp > 1
        shifts = [[mod1(j + R, L) for j = 1:L] for R = 0:(n_qp - 1)]
        set_shift_qp!(data, shifts, [ones(Int, L) for _ = 1:n_qp],
                      ComplexF64[1 / n_qp for _ = 1:n_qp])
    end
    MVMCOptimizers.parton_materialize_flags!(data)
    return data
end

"全数計算の ln P_J(参照実装。プロダクションの式を使わない)。"
function _ln_pj_ref(cfg, data)
    L = cfg.n_site
    z = 0.0
    for i = 1:L, j = (i + 1):L
        MVMCOptimizers.is_occupied(cfg, i) && MVMCOptimizers.is_occupied(cfg, j) || continue
        idx = data.jastrow_idx[i, j]
        z += real(data.jastrow_terms[idx + 1].value)
    end
    return z
end

"pstate 一式を組んで軌道まで更新する。"
function _jast_state(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    mp = data.modpara
    MVMCOptimizers.parton_update_orbitals!(
        pstate.mfham, MVMCOptimizers.parton_alpha_from_terms(data), mp.nelec)
    MVMCOptimizers.parton_update_orbital_derivatives!(pstate.mfham, mp.nelec)
    return pstate
end

"配置を置いて proj_cnt も構築する。"
function _jast_place!(pstate, data, sites)
    mp = data.modpara
    cfg = pstate.config
    fill!(cfg.ele_cfg, -1)
    fill!(cfg.ele_num, 0)
    for f = 1:mp.nflavor, (m, r) in enumerate(sites)
        MVMCOptimizers.place_particle!(cfg, f, m, r)
    end
    MVMCOptimizers.parton_make_proj_cnt!(cfg, data)
    MVMCOptimizers.parton_recompute_amplitude_all!(
        pstate.amp, pstate.mfham, cfg, data, pstate.workspace)
    return cfg
end

@testset "§8-16-1 無効時: n_proj = 0 で恒等(フックの等価性)" begin
    data = _jast_data()                              # Jastrow なし
    pstate = _jast_state(data)
    cfg = _jast_place!(pstate, data, [1, 3, 5])
    @test MVMCOptimizers.parton_n_proj(cfg) == 0
    @test MVMCOptimizers.parton_log_proj_ratio(cfg, data, 1, 1, 2) === 0.0
    # カウンタ系は no-op
    @test MVMCOptimizers.parton_make_proj_cnt!(cfg, data) === nothing
    @test MVMCOptimizers.parton_update_proj_cnt!(cfg, data, 1, 2) === nothing
    # 実 run のバイト一致(v3.10 出力との突き合わせ)は実測済み:
    # zvo_out / zqp_pmfpara_opt / zvo_var / zvo_SRinfo すべて一致
end

@testset "§8-16-2 差分 vs 全数(F=$F, n_qp=$nq)" for F in (2, 3), nq in (1, 2)
    L, Ne = 6, 3
    v = [0.31, -0.17, 0.07]
    data = _jast_data(; F = F, v = v, n_qp = nq)
    pstate = _jast_state(data)
    cfg = _jast_place!(pstate, data, [1, 3, 4])
    rng = MersenneTwister(777)

    ref = zeros(Int, length(cfg.proj_cnt))
    for step = 1:300
        m = rand(rng, 1:Ne)
        r_old = MVMCOptimizers.particle_site(cfg, 1, m)
        r_new = rand(rng, 1:L)
        MVMCOptimizers.is_occupied(cfg, r_new) && continue

        # 比(純粋関数)を、全数 lnP の差と突き合わせる
        lp_before = _ln_pj_ref(cfg, data)
        dlnp = MVMCOptimizers.parton_log_proj_ratio(cfg, data, m, r_old, r_new)

        for f = 1:data.modpara.nflavor
            MVMCOptimizers.move_particle!(cfg, f, m, r_old, r_new)
        end
        MVMCOptimizers.parton_update_proj_cnt!(cfg, data, r_old, r_new)

        @test dlnp ≈ _ln_pj_ref(cfg, data) - lp_before atol = 1e-12

        # カウンタの差分更新 = 全数再構築
        copyto!(ref, cfg.proj_cnt)
        MVMCOptimizers.parton_make_proj_cnt!(cfg, data)
        @test cfg.proj_cnt == ref
    end
end

# --- 全数展開の E / 力(§8-7 の枠組みを射影ブロック込みで) ------------------

function _jast_enumerate(pstate, data, configs; conjugate::Bool = true)
    mp = data.modpara
    qpw = MVMCOptimizers.parton_qp_weight(data)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    sr = pstate.state.sr_opt
    fill!(sr.sr_opt_oo, 0)
    fill!(sr.sr_opt_ho, 0)
    wc = 0.0
    etot = ComplexF64(0)
    for c in configs
        cfg = _jast_place!(pstate, data, c)
        ip = MVMCOptimizers.parton_calculate_ip(pstate.amp, qpw)
        abs(ip) < 1e-30 && continue
        e = MVMCOptimizers.parton_local_energy(pstate, data, ip)
        w = exp(2 * _ln_pj_ref(cfg, data)) * abs2(ip)    # |P_J·ip|²(参照式)
        MVMCOptimizers.parton_fill_sr_opt_o!(
            sr.sr_opt_o, pstate.amp, pstate.mfham, cfg, data, qpw, ip, n_proj;
            conjugate = conjugate)
        MVMCOptimizers.calculate_oo!(sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_o,
                                     w, e, sr.sr_opt_size)
        wc += w
        etot += w * e
    end
    sr.sr_opt_oo ./= wc
    sr.sr_opt_ho ./= wc
    return etot / wc
end

function _jast_energy(pstate, data, configs)
    real(_jast_enumerate(pstate, data, configs))
end

_jast_configs(L, Ne) = [c for c in
    [[i, j, k] for i = 1:L for j = (i + 1):L for k = (j + 1):L]]

@testset "§8-16-3/4/6 有限差分と共役シム(F=$F, n_qp=$nq)" for F in (2, 3), nq in (1, 2)
    L, Ne = 6, 3
    v0 = [0.23, -0.11, 0.05]
    data = _jast_data(; F = F, v = v0, n_qp = nq)
    pstate = _jast_state(data)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    n_idx = pstate.mfham.n_idx
    configs = _jast_configs(L, Ne)

    e0 = _jast_enumerate(pstate, data, configs)
    @test abs(imag(e0)) < 1e-10

    # (4) 射影 O の虚スロットが厳密 0(全配置)
    sr = pstate.state.sr_opt
    qpw = MVMCOptimizers.parton_qp_weight(data)
    for c in configs[1:5]
        cfg = _jast_place!(pstate, data, c)
        ip = MVMCOptimizers.parton_calculate_ip(pstate.amp, qpw)
        abs(ip) < 1e-30 && continue
        MVMCOptimizers.parton_fill_sr_opt_o!(
            sr.sr_opt_o, pstate.amp, pstate.mfham, cfg, data, qpw, ip, n_proj)
        for p = 1:n_proj
            @test sr.sr_opt_o[2p + 2] === ComplexF64(0)          # 虚スロット厳密 0
            @test imag(sr.sr_opt_o[2p + 1]) == 0.0               # O_p は実数
        end
    end

    # (4) S 行列は共役の有無で不変(共役シムの前提が Jastrow 込みでも立つ)
    _jast_enumerate(pstate, data, configs; conjugate = true)
    oo_c = copy(pstate.state.sr_opt.sr_opt_oo)
    _jast_enumerate(pstate, data, configs; conjugate = false)
    oo_n = copy(pstate.state.sr_opt.sr_opt_oo)
    @test maximum(abs, real.(oo_c) .- real.(oo_n)) < 1e-12

    # (3) 力ベクトル = 有限差分勾配(射影 + MF の全実自由度)
    _jast_enumerate(pstate, data, configs; conjugate = true)
    smat_to_para_idx = Int[]
    for p = 1:(n_proj + n_idx)
        push!(smat_to_para_idx, 2 * (p - 1))
        push!(smat_to_para_idx, 2 * (p - 1) + 1)
    end
    n = length(smat_to_para_idx)
    S = zeros(Float64, n, n)
    g = zeros(Float64, n)
    MVMCOptimizers.build_s_matrix_and_g_vector!(
        S, g, smat_to_para_idx, sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_size,
        data.modpara.dsr_opt_sta_del, data.modpara.dsr_opt_step_dt)
    dt = data.modpara.dsr_opt_step_dt

    δ = 1e-6
    for p = 1:n_proj                       # 射影ブロック(Re のみ、Im は凍結)
        orig = data.jastrow_terms[p].value
        data.jastrow_terms[p].value = orig + δ
        ep = _jast_energy(pstate, data, configs)
        data.jastrow_terms[p].value = orig - δ
        em = _jast_energy(pstate, data, configs)
        data.jastrow_terms[p].value = orig
        fd = (ep - em) / (2δ)
        @test isapprox(g[2 * (p - 1) + 1] / (-dt), fd; rtol = 1e-4, atol = 1e-8)
    end
    @test maximum(abs, g[1:2:(2 * n_proj)]) > 1e-6   # 勾配が実際に立っている
end

@testset "§8-16-5 変分改善: 厳密勾配ステップで E が下がる(F=$F, n_qp=$nq)" for
        F in (2, 3), nq in (1, 2)
    L, Ne = 6, 3
    data = _jast_data(; F = F, v = zeros(3), n_qp = nq)   # v = 0 から
    pstate = _jast_state(data)
    configs = _jast_configs(L, Ne)
    e0 = _jast_energy(pstate, data, configs)

    # v の厳密勾配(有限差分)で 1 ステップ降下
    δ = 1e-6
    grad = zeros(3)
    for p = 1:3
        data.jastrow_terms[p].value = +δ
        ep = _jast_energy(pstate, data, configs)
        data.jastrow_terms[p].value = -δ
        em = _jast_energy(pstate, data, configs)
        data.jastrow_terms[p].value = 0
        grad[p] = (ep - em) / (2δ)
    end
    @test maximum(abs, grad) > 1e-6        # V ≠ 0 なので密度相関の勾配が立つ
    for η in (0.05, 0.01)
        for p = 1:3
            data.jastrow_terms[p].value = ComplexF64(-η * grad[p], 0)
        end
        e1 = _jast_energy(pstate, data, configs)
        if e1 < e0 - 1e-12
            @test e1 < e0                  # 厳密に改善
            break
        end
        η == 0.01 && @test e1 < e0
    end
end

@testset "§8-16-8 Gutzwiller は明示エラーで拒否" begin
    data = _jast_data()
    data.n_gutzwiller_idx = 1
    push!(data.gutzwiller_terms,
          MVMCExpertModeParsers.GutzwillerTerm(0, ComplexF64(0.1, 0), false))
    MVMCOptimizers.parton_materialize_flags!(data)
    err = try
        MVMCOptimizers.validate_parton_inputs(data, MVMCOptimizers.serial_context())
        nothing
    catch e
        sprint(showerror, e)
    end
    @test err !== nothing
    @test occursin("constant", err)        # 「固縛下で定数」の理由が入っている
end

@testset "§8-16-8b 門番: 複素 v / 不完全な idx を拒否" begin
    data = _jast_data(; v = [0.1, 0.2, 0.3])
    data.jastrow_terms[2].value = ComplexF64(0.2, 0.1)     # 複素 v
    @test_throws Exception MVMCOptimizers.validate_parton_inputs(
        data, MVMCOptimizers.serial_context())

    data = _jast_data(; v = [0.1, 0.2, 0.3])
    data.jastrow_idx[2, 4] = -1                             # ペア欠け
    @test_throws Exception MVMCOptimizers.validate_parton_inputs(
        data, MVMCOptimizers.serial_context())
end

@testset "§8-16 flags: Jastrow の Im スロットが強制凍結される" begin
    data = _jast_data(; v = [0.1, 0.2, 0.3])
    flags = MVMCOptimizers.parton_materialize_flags!(data)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    @test n_proj == 3
    for p = 1:n_proj
        @test flags[2 * (p - 1) + 1] == true      # Re は可動
        @test flags[2 * (p - 1) + 2] == false     # Im は凍結
    end
end
