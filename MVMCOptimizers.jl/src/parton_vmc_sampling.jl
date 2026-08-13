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

    # det^F 高速路(v3.9): フレーバー対称ならブロックを 1 フレーバー分だけ持つ。
    # 判定はテンプレート build が済ませている(PartonFlavorSymFast=0 なら false)。
    n_stored = mfham.flavor_symmetric ? 1 : mp.nflavor

    return PartonOptimizationState(
        state,
        PartonAmplitudeData(n_qp, mp.nflavor, mp.nelec; n_stored = n_stored),
        PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, mp.nvmc_sample;
                            n_proj = n_proj),
        PartonSamplingWorkspace(mp.nelec, n_qp * n_stored),
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
    np = parton_n_proj(cfg)
    np > 0 && copyto!(cfg.stored_proj_cnt, (s - 1) * np + 1, cfg.proj_cnt, 1, np)
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
    np = parton_n_proj(cfg)
    np > 0 && copyto!(cfg.proj_cnt, 1, cfg.stored_proj_cnt, (s - 1) * np + 1, np)
    return nothing
end

"""
    parton_restore_sample_from!(dst, src, s)

`src` の保存済みサンプル s を**別の** cfg(スレッドローカル)へ復元する。
§4 層 2 のサンプル並列用: 各スレッドが自分の作業配置に読み出すだけで、
`src` の保存領域は読み取り専用のまま共有される。
"""
function parton_restore_sample_from!(
    dst::PartonConfiguration, src::PartonConfiguration, s::Int)
    n = src.n_flavor * src.n_elec
    copyto!(dst.ele_idx, 1, src.stored_ele_idx, (s - 1) * n + 1, n)
    _rebuild_cfg_from_idx!(dst)
    np = parton_n_proj(dst)
    np > 0 && copyto!(dst.proj_cnt, 1, src.stored_proj_cnt, (s - 1) * np + 1, np)
    return nothing
end

"burn-in 用の退避と復元。ele_idx と proj_cnt を持ち、残りは復元時に組み直す。"
function parton_copy_to_burn_sample!(cfg::PartonConfiguration)
    copyto!(cfg.burn_ele_idx, cfg.ele_idx)
    parton_n_proj(cfg) > 0 && copyto!(cfg.burn_proj_cnt, cfg.proj_cnt)
    return nothing
end

function parton_copy_from_burn_sample!(cfg::PartonConfiguration)
    copyto!(cfg.ele_idx, cfg.burn_ele_idx)
    _rebuild_cfg_from_idx!(cfg)
    parton_n_proj(cfg) > 0 && copyto!(cfg.proj_cnt, cfg.burn_proj_cnt)
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
物理密度 Jastrow(v3.11 M2 後半、DESIGN §1.1 / §2)

    P_J(x) = exp( Σ_{i<j} v_{ij} · n^b_i · n^b_j ),   n^b_i ∈ {0, 1}

- **物理密度 n^b で定義する**。パートン和 Σ_f n^(f) = F·n^b で定義すると同じ v が
  F² 倍の意味になり、F を変えると入力の意味が変わるため。上流(既存 mVMC)は
  変数 x_i = n_i − 1 を使うが、これも同じ理由でそのまま使えない — 差は一体項+定数
  で係数の辻褄合わせは不要(DESIGN §2 に明記。黙って係数は入れない)
- 構造の規約は上流 `make_proj_cnt!` に合わせる: **Σ_{i<j}(1/2 なし・自己項なし)、
  exp の符号は +、v は `real(jastrow_terms[].value)`**、添字は
  `jastrow_idx[ri+1, rj+1]`(0-based 値、対称)
- **P_J は配置のみに依存し、qp に依存しない**(DESIGN §1.4)。したがって
  比・E_loc の式は既存の `exp(log_pr)` フックのまま変わらない
- Jastrow のカウンタ cnt_p は契約 5(O_p = cnt_p、**実数**)とサンプル保存が
  読む。サンプリング中の比は下の純粋関数で O(Ne) で取れるのでカウンタを読まない
"""

"proj_cnt の長さ(= layout.n_proj)。0 なら Jastrow なし。"
@inline parton_n_proj(cfg::PartonConfiguration) = length(cfg.proj_cnt)

"ペア (ri, rj)(1-based サイト)の Jastrow パラメータ v。未定義ペアは門番が拒否済み。"
@inline function _parton_jastrow_v(data::ExpertModeData, ri::Int, rj::Int)
    idx = data.jastrow_idx[ri, rj]              # 0-based の idx 値
    return real(data.jastrow_terms[idx + 1].value)
end

"""
    parton_make_proj_cnt!(cfg, data)

配置から Jastrow カウンタを全数構築する(錨・サンプル復元と同じ役どころ)。
占有サイト(固縛によりフレーバー 1 が代表)のペア走査で O(Ne²)。
"""
function parton_make_proj_cnt!(cfg::PartonConfiguration, data::ExpertModeData)
    n_proj = parton_n_proj(cfg)
    n_proj == 0 && return nothing
    layout = MVMCExpertModeParsers.projection_layout(data)
    fill!(cfg.proj_cnt, 0)
    jidx = data.jastrow_idx
    @inbounds for a = 1:cfg.n_elec
        ri = particle_site(cfg, 1, a)
        for b = (a + 1):cfg.n_elec
            rj = particle_site(cfg, 1, b)
            p = layout.jastrow_offset + jidx[ri, rj] + 1
            cfg.proj_cnt[p] += 1                # n^b_i · n^b_j = 1(両方占有)
        end
    end
    return nothing
end

"""
    parton_log_proj_ratio(cfg, data, m, r_old, r_new) -> Float64

固縛移動 r_old → r_new の Δln P_J。**純粋関数**(状態を変えない。契約 2 と同じ規律)。

    Δ = Σ_{j ∈ 占有, j ≠ r_old} [ v(r_new, j) − v(r_old, j) ]

自己項の扱い: 移動する粒子自身は両側から除く — j = r_old を除外し(移動前の自分)、
j = r_new は占有集合に居ないので自動的に入らない(移動後の自分)。ペア
(r_old, r_new) の寄与は移動前後とも片方が空で 0。§8-16-2 が全数計算と突き合わせる。

Jastrow なし(n_proj = 0)なら厳密に 0.0 を返す(M1 からの恒等フックと同値)。
"""
function parton_log_proj_ratio(
    cfg::PartonConfiguration,
    data::ExpertModeData,
    m::Int,
    r_old::Int,
    r_new::Int,
)::Float64
    parton_n_proj(cfg) == 0 && return 0.0
    z = 0.0
    @inbounds for m2 = 1:cfg.n_elec
        j = particle_site(cfg, 1, m2)
        j == r_old && continue
        z += _parton_jastrow_v(data, r_new, j) - _parton_jastrow_v(data, r_old, j)
    end
    return z
end

"""
    parton_update_proj_cnt!(cfg, data, r_old, r_new)

受理後のカウンタ差分更新。**配置コミット(`parton_update_ele_config!`)の後**に
呼ぶこと: 占有集合は移動後のもので、

    Δcnt: j ∈ 占有(移動後), j ≠ r_new について
          cnt[p(r_new, j)] += 1,  cnt[p(r_old, j)] -= 1

j = r_old は移動後の占有集合に居ないので自動的に除外される(自己項の対)。
"""
function parton_update_proj_cnt!(
    cfg::PartonConfiguration,
    data::ExpertModeData,
    r_old::Int,
    r_new::Int,
)
    parton_n_proj(cfg) == 0 && return nothing
    layout = MVMCExpertModeParsers.projection_layout(data)
    jidx = data.jastrow_idx
    off = layout.jastrow_offset
    @inbounds for m2 = 1:cfg.n_elec
        j = particle_site(cfg, 1, m2)
        j == r_new && continue
        cfg.proj_cnt[off + jidx[r_new, j] + 1] += 1
        cfg.proj_cnt[off + jidx[r_old, j] + 1] -= 1
    end
    return nothing
end

# =====================================================================
# 骨格
# =====================================================================

"""
    parton_n_out(cfg, mp) -> Int
    parton_n_in(mp) -> Int

サンプリング量の式(DESIGN §4 の C 踏襲規約)。**式の家はここ 1 箇所**:
`parton_make_sample!` 本体と `zvo_parton_time.dat` の記録(§3.3.1)の両方が
これを呼ぶ。呼び出しは `parton_make_sample!` が burn_flag を立てる**前**で
あること(サンプリング後に呼ぶと次ステップの値になる)。

- `n_out`: 初回 SR ステップは `NVMCWarmUp + NVMCSample`、burn 再開後は
  `NVMCSample + 1`。ステップごとに値が変わるので time は毎ステップ記録する
- `n_in`: `NVMCInterval × NSite`(サンプル間の内側ステップ数)
"""
@inline parton_n_out(cfg::PartonConfiguration, mp) =
    cfg.burn_flag ? mp.nvmc_sample + 1 : mp.nvmc_warmup + mp.nvmc_sample
@inline parton_n_in(mp) = mp.nvmc_interval * mp.nsite

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
function parton_make_sample!(pstate::PartonOptimizationState, data::ExpertModeData, rng;
                             c_timer::CTimer = CTIMER_DISABLED)
    amp = pstate.amp
    cfg = pstate.config
    mfham = pstate.mfham
    ws = pstate.workspace
    mp = data.modpara
    n_site = mp.nsite
    qp_weight = parton_qp_weight(data)

    # --- 開始配置: burn 再利用 or 初期生成 --------------------------------
    n_out = parton_n_out(cfg, mp)
    if cfg.burn_flag
        parton_copy_from_burn_sample!(cfg)   # proj_cnt も burn から戻る
        ctimer_start!(c_timer, 804)
        parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)   # 最初の錨
        ctimer_stop!(c_timer, 804)
    else
        parton_make_initial_sample!(cfg, amp, mfham, data, ws; rng = rng)  # 錨も打つ
        parton_make_proj_cnt!(cfg, data)     # Jastrow カウンタの錨(全数構築)
    end
    n_in = parton_n_in(mp)

    n_accept_anchor = 0
    for out_step = 1:n_out
        for _ = 1:n_in
            cfg.counter[1] += 1                                  # 試行数
            m, r_old, r_new, ok = parton_make_candidate_hopping(rng, cfg, n_site)
            ok || continue         # 占有先 / 同一サイト → 棄却(試行には数える)

            log_pr = parton_log_proj_ratio(cfg, data, m, r_old, r_new)
            ratio, _ =
                (ctimer_start!(c_timer, 805);
                 _pr = parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m, r_new);
                 ctimer_stop!(c_timer, 805); _pr)
            # 受理判定。Jastrow なし(log_pr = 0)は従来の式そのまま = ビット互換。
            # Jastrow ありは上流(vmc_sampling.jl の w = exp(2(x+Δlog ip)))と同じく
            # log をまとめて 1 回 exp し、オーバーフロー(!isfinite)は棄却に落とす。
            # 乱数の消費はどちらの分岐でも 1 回。
            if log_pr == 0.0
                rng_real2(rng) < abs2(ratio) || continue
            else
                w = exp(2 * (log_pr + log(abs(ratio))))
                isfinite(w) || (w = -1.0)
                rng_real2(rng) < w || continue
            end

            cfg.counter[2] += 1                                  # 受理数
            parton_update_ele_config!(cfg, m, r_old, r_new)      # ① 配置を先に確定
            parton_update_proj_cnt!(cfg, data, r_old, r_new)     # ①′ カウンタ差分
            ctimer_start!(c_timer, 806)
            st = parton_update_amplitude!(amp, mfham, data, ws, m, r_new)  # ② 高速更新
            ctimer_stop!(c_timer, 806)
            n_accept_anchor += 1
            if st === :need_recompute || n_accept_anchor >= mp.parton_block_update_size
                # counter[3] = ratio_floor ヒット回数、counter[4] = 厳密再計算の回数
                st === :need_recompute && (cfg.counter[3] += 1)
                cfg.counter[4] += 1
                ctimer_start!(c_timer, 804)
                parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
                ctimer_stop!(c_timer, 804)
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
        # O(Ne) の縮約は保持ブロックだけ(対称なら f=1 のみ)。
        for f = 1:amp.n_stored
            b = block_index(amp, qp, f)
            Ainv = inv_block(amp, qp, f)
            Φ = mfham.orbitals[f]
            R = zero(ComplexF64)
            @inbounds for n = 1:amp.n_elec
                R += Φ[rr, n] * Ainv[n, m]    # 転置積。dot は使わない(共役が入る)
            end
            ws.ratio_blocks[b] = s * R
        end
        # 積は物理フレーバーのループのまま(対称時は別名ブロックを F 回読む =
        # R^F・det^F)。評価順が従来と同一なので高速路 ON/OFF でビット一致する。
        p_old = one(ComplexF64)
        p_new = one(ComplexF64)
        for f = 1:amp.n_flavor
            b = block_index(amp, qp, f)
            p_old *= amp.det_a[b]
            p_new *= amp.det_a[b] * ws.ratio_blocks[b]
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
    # 書き込みは保持ブロックだけ(対称なら f=1 のみ)。物理フレーバー全部を回すと
    # 別名ブロックへ同じ rank-1 更新が F 回かかって壊れる。
    for qp = 1:amp.n_qp, f = 1:amp.n_stored
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
