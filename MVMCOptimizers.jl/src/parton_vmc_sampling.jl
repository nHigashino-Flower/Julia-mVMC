"""
契約 2 / 3 と固縛 Metropolis サンプリングの骨格
--- parton-mode (fork addition) ---

DESIGN_parton.md §1.4(比・更新の固定式)と §4(骨格の規約)に対応する。

C 版からの意図的な差分(DESIGN §4):
- `burn_flag` は counter[11] の間借りではなく PartonConfiguration の Bool
- 交換更新は固縛の下で恒等なので分岐ごと存在しない(NExUpdatePath = 6 を
  門番が保証しているので、更新は固縛ホッピング一択)
- `ratio_floor` を割ったブロックは `:need_recompute` を返す。upstream に無い
  防御で、QP 和では総比が健全でも個別ブロックがノードをかすめ得るため
"""

"""
    parton_qp_weight(data) -> Vector{ComplexF64}

QP 重み(既存の init_qp_weight! が埋めた qp_full_weight)。
"""
function parton_qp_weight(data::ExpertModeData)
    data.qp_weights === nothing &&
        error("QP weights have not been initialised; call init_qp_weight!(data) first.")
    return data.qp_weights.qp_full_weight
end

"""
    parton_build_optimization_state(data; mfham=nothing) -> PartonOptimizationState

パートンモードの状態一式を確保する。SROptData は n_para = NProj + n_idx で
確保する(既存 VMCOptimizationState のコンストラクタが 1 + n_para を渡す規約)。

`state` のスレーター行列や電子配置のフィールドはパートン経路では使わないが、
既存コンストラクタの寸法計算をそのまま通しても害はないので触らない。
"""
function parton_build_optimization_state(
    data::ExpertModeData;
    mfham::Union{PartonMFHamiltonian,Nothing} = nothing,
)
    mp = data.modpara
    n_qp = get_n_qp_full(data)
    n_para = MVMCExpertModeParsers.count_variational_parameters(data)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj

    state = VMCOptimizationState(
        mp.nsite,
        mp.nelec,
        n_proj,
        n_para,
        n_qp,
        mp.nvmc_sample,
        true,     # all_complex: パートンモードは ComplexType = 1 のみ
        false,    # use_fsz
    )

    if mfham === nothing
        mfham = PartonMFHamiltonian(mp.nsite, mp.nelec, mp.nflavor, parton_n_idx(data))
        parton_build_mf_templates!(mfham, data)
    end

    return PartonOptimizationState(
        state,
        PartonAmplitudeData(n_qp, mp.nflavor, mp.nelec),
        PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, mp.nvmc_sample),
        PartonSamplingWorkspace(mp.nelec, n_qp * mp.nflavor),
        mfham,
        parton_build_phys_hamiltonian(data),
    )
end

# =====================================================================
# 配置の生成・保存・復元
# =====================================================================

"""
    parton_make_initial_sample!(cfg, amp, mfham, data, ws, rng; max_trial=100)

非ゼロ振幅になるまで固縛配置をランダムに引き直す。波動関数のノード上は
`det = 0` で比が定義できないので、初期配置を選ぶ責務はここにある
(C 版の MakeInitialSample と同じ分担)。
"""
function parton_make_initial_sample!(
    cfg::PartonConfiguration,
    amp::PartonAmplitudeData,
    mfham::PartonMFHamiltonian,
    data::ExpertModeData,
    ws::PartonSamplingWorkspace;
    rng,
    max_trial::Int = 100,
)
    qp_weight = parton_qp_weight(data)
    pool = collect(1:cfg.n_site)
    for _ = 1:max_trial
        # Fisher-Yates の前半だけ。乱数は既存の rng_mod(C の gen_rand32 準拠)
        for m = 1:cfg.n_elec
            j = m + rng_mod(rng, cfg.n_site - m + 1)
            pool[m], pool[j] = pool[j], pool[m]
        end
        sites = pool[1:cfg.n_elec]
        fill!(cfg.ele_cfg, -1)
        fill!(cfg.ele_num, 0)
        for f = 1:cfg.n_flavor, m = 1:cfg.n_elec
            place_particle!(cfg, f, m, sites[m])
        end
        parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
        abs(parton_calculate_ip(amp, qp_weight)) > 1e-12 && return nothing
    end
    error(
        "Could not find an initial configuration with non-zero amplitude in " *
        "$max_trial trials. The mean-field orbitals may be degenerate or the " *
        "filling may put every configuration on a node.",
    )
end

"サンプル s・フレーバー f・粒子 m の保存済みサイト(1-based)。"
@inline stored_particle_site(cfg::PartonConfiguration, s::Int, f::Int, m::Int) =
    cfg.stored_ele_idx[(s - 1) * cfg.n_flavor * cfg.n_elec + (f - 1) * cfg.n_elec + m]

"""
    parton_store_sample!(cfg, s)

サンプル s として現在の配置を保存する。振幅は保存しない: サンプルごとの
再計算は測定側(契約 4)の分担で、そちらが保存済み配置から錨を打ち直す
(DESIGN §4)。
"""
function parton_store_sample!(cfg::PartonConfiguration, s::Int)
    n = cfg.n_flavor * cfg.n_elec
    copyto!(cfg.stored_ele_idx, (s - 1) * n + 1, cfg.ele_idx, 1, n)
    return nothing
end

"""
    parton_restore_sample!(cfg, s)

保存済みサンプル s を作業配置へ戻す。ele_cfg / ele_num は ele_idx から
組み直す(保存しているのは ele_idx だけ)。
"""
function parton_restore_sample!(cfg::PartonConfiguration, s::Int)
    n = cfg.n_flavor * cfg.n_elec
    copyto!(cfg.ele_idx, 1, cfg.stored_ele_idx, (s - 1) * n + 1, n)
    _rebuild_cfg_from_idx!(cfg)
    return nothing
end

"burn-in 用の退避と復元。ele_idx だけ持ち、残りは復元時に組み直す。"
function parton_copy_to_burn_sample!(cfg::PartonConfiguration)
    copyto!(cfg.burn_ele_idx, cfg.ele_idx)
    return nothing
end

function parton_copy_from_burn_sample!(cfg::PartonConfiguration)
    copyto!(cfg.ele_idx, cfg.burn_ele_idx)
    _rebuild_cfg_from_idx!(cfg)
    return nothing
end

function _rebuild_cfg_from_idx!(cfg::PartonConfiguration)
    fill!(cfg.ele_cfg, -1)
    fill!(cfg.ele_num, 0)
    for f = 1:cfg.n_flavor, m = 1:cfg.n_elec
        r = particle_site(cfg, f, m)
        base = (f - 1) * cfg.n_site
        cfg.ele_cfg[base + r] = m
        cfg.ele_num[base + r] = 1
    end
    return nothing
end

"""
    parton_log_proj_ratio(cfg, m, r_old, r_new) -> Float64

射影因子の対数比のフック。M1 の初点火では射影が無いので恒等 0。
固縛の下で既存の Gutzwiller は自明化する(全占有サイトが常にダブロン)ので、
物理密度 n^b ベースの Jastrow を新設するのは M2 の仕事(DESIGN §9)。
"""
@inline parton_log_proj_ratio(
    ::PartonConfiguration,
    ::Int,
    ::Int,
    ::Int,
)::Float64 = 0.0

# =====================================================================
# 骨格
# =====================================================================

"""
    parton_make_sample!(pstate, data, rng)

固縛 Metropolis サンプリング。契約 0(軌道の更新)は呼び出し前に済んでいること。

C 踏襲: `n_in = NVMCInterval × Nsite`、初回の `n_out = WarmUp + Sample`、
burn からの再開時は `Sample + 1`、`PartonBlockUpdateSize` 回の受理ごとに錨を打ち、
保存するのは末尾 Sample 個。

このループで一番大事なのは受理時の①→②の順序(DESIGN §4)。受理が確定したら
先に配置をコミットし、それから振幅を更新する。こうしておけば契約 3 が途中の
(qp, f) ブロックで `:need_recompute` を返して部分更新のまま抜けても安全で、
直後の全再計算がコミット済みの配置から組み直すので半端な状態は跡形もなく
上書きされる。逆順(振幅 → 配置)だと、再計算が古い配置を読んで受理済みの
移動が消えるという追いにくいバグになる。
"""
function parton_make_sample!(pstate::PartonOptimizationState, data::ExpertModeData, rng)
    amp = pstate.amp
    cfg = pstate.config
    mfham = pstate.mfham
    ws = pstate.workspace
    mp = data.modpara
    n_site = mp.nsite
    qp_weight = parton_qp_weight(data)

    # --- 開始配置: burn 再利用 or 初期生成 --------------------------------
    if cfg.burn_flag
        parton_copy_from_burn_sample!(cfg)
        n_out = mp.nvmc_sample + 1
        parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)   # 最初の錨
    else
        n_out = mp.nvmc_warmup + mp.nvmc_sample
        parton_make_initial_sample!(cfg, amp, mfham, data, ws; rng = rng)  # 錨も打つ
    end
    n_in = mp.nvmc_interval * n_site

    n_accept_anchor = 0
    for out_step = 1:n_out
        for _ = 1:n_in
            cfg.counter[1] += 1                                  # 試行数
            m, r_old, r_new, ok = parton_make_candidate_hopping(rng, cfg, n_site)
            ok || continue         # 占有先 / 同一サイト → 棄却(試行には数える)

            log_pr = parton_log_proj_ratio(cfg, m, r_old, r_new)
            ratio, _ =
                parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m, r_new)
            rng_real2(rng) < exp(2 * log_pr) * abs2(ratio) || continue

            cfg.counter[2] += 1                                  # 受理数
            parton_update_ele_config!(cfg, m, r_old, r_new)      # ① 配置を先に確定
            st = parton_update_amplitude!(amp, mfham, data, ws, m, r_new)  # ② 高速更新
            n_accept_anchor += 1
            if st === :need_recompute || n_accept_anchor >= mp.parton_block_update_size
                parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
                n_accept_anchor = 0
            end
        end
        s = out_step - (n_out - mp.nvmc_sample)
        s >= 1 && parton_store_sample!(cfg, s)
    end

    parton_copy_to_burn_sample!(cfg)
    cfg.burn_flag = true
    assert_flavors_locked(cfg)
    return nothing
end

"""
    parton_make_candidate_hopping(rng, cfg, n_site) -> (m, r_old, r_new, ok)

物理粒子を 1 つ選び、行き先を一様に選ぶ。対称提案なので Metropolis の
補正は要らない。占有判定は固縛によりフレーバー 1 の代表で足りる。
"""
function parton_make_candidate_hopping(rng, cfg::PartonConfiguration, n_site::Int)
    m = 1 + rng_mod(rng, cfg.n_elec)
    r_old = particle_site(cfg, 1, m)
    r_new = 1 + rng_mod(rng, n_site)
    ok = (r_new != r_old) && !is_occupied(cfg, r_new)
    return m, r_old, r_new, ok
end

"""
    parton_update_ele_config!(cfg, m, r_old, r_new)

配置更新: 全フレーバー同時。固縛不変条件を守る唯一の書き込み経路。
"""
function parton_update_ele_config!(
    cfg::PartonConfiguration,
    m::Int,
    r_old::Int,
    r_new::Int,
)
    for f = 1:cfg.n_flavor
        move_particle!(cfg, f, m, r_old, r_new)
    end
    return nothing
end

# =====================================================================
# 契約 2 / 3
# =====================================================================

"""
    parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m, r_new) -> (ratio, ip_new)

契約 2: 固縛移動(粒子 m: r → r′)の振幅比。O(n_qp · F · Ne)。

純粋(書くのは ws だけ)なので棄却時の revert が要らない。各ブロックの R は
`ws.ratio_blocks` に残し、受理時に契約 3 が再利用する。

最内の縮約は共役なしの転置積で、`dot()` は使わない(第一引数を共役するため)。
契約 0′ の随伴とは別物(DESIGN §7)。
"""
function parton_amplitude_ratio!(
    ws::PartonSamplingWorkspace,
    amp::PartonAmplitudeData,
    mfham::PartonMFHamiltonian,
    data::ExpertModeData,
    qp_weight,
    m::Int,
    r_new::Int,
)
    ip_old = zero(ComplexF64)
    ip_new = zero(ComplexF64)
    for qp = 1:amp.n_qp
        rr = data.qp_trans[qp][r_new]
        s = data.qp_trans_sgn[qp][r_new]
        p_old = one(ComplexF64)
        p_new = one(ComplexF64)
        for f = 1:amp.n_flavor
            b = block_index(amp, qp, f)
            Ainv = inv_block(amp, qp, f)
            Φ = mfham.orbitals[f]
            R = zero(ComplexF64)
            @inbounds for n = 1:amp.n_elec
                R += Φ[rr, n] * Ainv[n, m]    # 転置積。dot は使わない(共役が入る)
            end
            R *= s
            ws.ratio_blocks[b] = R
            p_old *= amp.det_a[b]
            p_new *= amp.det_a[b] * R
        end
        ip_old += qp_weight[qp] * p_old
        ip_new += qp_weight[qp] * p_new
    end
    return ip_new / ip_old, ip_new
end

"""
    parton_update_amplitude!(amp, mfham, data, ws, m, r_new; ratio_floor=1e-12) -> Symbol

契約 3: 受理された固縛移動を全 (qp, f) ブロックへ反映する。O(n_qp · F · Ne²)。

直前の契約 2 が同じ (m, r_new) で呼ばれていること(`ws.ratio_blocks` の再利用)が
前提。配置側の更新は `parton_update_ele_config!` の担当で、ここでは触らない。

`|R| < ratio_floor` のブロックが出たら `:need_recompute` を返す。呼び出し側は
コミット済みの配置から錨を打ち直すので、部分更新のまま抜けても安全。
"""
function parton_update_amplitude!(
    amp::PartonAmplitudeData,
    mfham::PartonMFHamiltonian,
    data::ExpertModeData,
    ws::PartonSamplingWorkspace,
    m::Int,
    r_new::Int;
    ratio_floor::Float64 = 1e-12,
)
    Ne = amp.n_elec
    for qp = 1:amp.n_qp, f = 1:amp.n_flavor
        b = block_index(amp, qp, f)
        R = ws.ratio_blocks[b]
        abs(R) < ratio_floor && return :need_recompute   # ノード踏み抜き
        Ainv = inv_block(amp, qp, f)
        rr = data.qp_trans[qp][r_new]
        s = data.qp_trans_sgn[qp][r_new]
        @views ws.u_buf .= s .* mfham.orbitals[f][rr, :]
        mul!(ws.v_buf, transpose(Ainv), ws.u_buf)        # v[j] = Σ_n u[n]·Ainv[n, j]
        @views ws.col_buf .= Ainv[:, m]                  # 旧列 m を退避
        invR = inv(R)
        @inbounds for j = 1:Ne
            j == m && continue
            axpy!(-ws.v_buf[j] * invR, ws.col_buf, @view Ainv[:, j])
        end
        @views Ainv[:, m] .= ws.col_buf .* invR
        amp.det_a[b] *= R
    end
    return :ok
end
