"""
契約 4(局所エネルギー)と契約 5(対数微分 O)
--- parton-mode (fork addition) ---

DESIGN_parton.md §1.3(選択則と符号定理)、§1.4(固定式)、§2.4(物理
ハミルトニアン)、§7(O の格納規約)に対応する。
"""

"""
    parton_build_phys_hamiltonian(data) -> PartonPhysHamiltonian

物理ハミルトニアンのテンプレートを起動時に 1 回組む。平均場側の
`parton_build_mf_templates!` と対になる build 段で、physhop と coulombinter の
0-based サイトを 1-based に直すのもここだけ(DESIGN §3.1 の「変換は
テンプレート build の一箇所」を入力ファミリごとに適用したもの)。

行単位の妥当性(自己ループ禁止・逆向き重複・範囲)は門番が済ませているので、
ここは変換に徹する。
"""
function parton_build_phys_hamiltonian(data::ExpertModeData)
    hops = [
        PartonPhysHopEntry(t.site1 + 1, t.site2 + 1, t.value) for t in data.physhop_terms
    ]
    diags = [
        PartonPhysDiagEntry(t.site1 + 1, t.site2 + 1, t.value) for
        t in data.coulomb_inter_terms
    ]
    return PartonPhysHamiltonian(hops, diags)
end

"""
    parton_diag_energy(physham, cfg) -> Float64

対角部 Σ_{(i,j,V)} V n_i n_j。固縛の下では占有数はフレーバーによらないので、
n は物理粒子の占有数 n^b そのもの。対角行 (i == i) は硬芯により V n_i を
与え、化学ポテンシャルを表す(DESIGN §2.4)。
"""
function parton_diag_energy(physham::PartonPhysHamiltonian, cfg::PartonConfiguration)
    e = 0.0
    @inbounds for d in physham.diags
        n_i = cfg.ele_num[d.site1]
        n_j = cfg.ele_num[d.site2]
        e += d.value * n_i * n_j
    end
    return e
end

"""
    parton_local_energy(pstate, data, ip) -> ComplexF64

契約 4: 局所エネルギー E_loc(x) = Σ_x′ ⟨x|H|x′⟩ Ψ(x′)/Ψ(x)。

物理ハミルトニアンは

    H = Σ_{(i,j,t) ∈ physhop} [ t · b†_j b_i + conj(t) · b†_i b_j ]
      + Σ_{(i,j,V) ∈ coulombinter} V n_i n_j

で、physhop は片方向のみ列挙され h.c. はここで供給される。合成粒子演算子
b_i = Π_f f^(f)_i の行列要素は、固縛セクターではフレーバー別 det 比の積に
なる(DESIGN §1.3 の選択則と符号定理)。それはまさに契約 2 が返す比なので、
仮想的な固縛移動として契約 2 を呼ぶだけでよい。統計の符号は行列式が自動で
運ぶので、F が奇数(フェルミオン)でも余分な符号は要らない。

契約 2 は純粋(ws.ratio_blocks しか書かない)なので、ここで何度呼んでも
配置も振幅も動かない。

呼び出し前提: このサンプルについて契約 1 で錨済み・ip 計算済み。
"""
function parton_local_energy(
    pstate::PartonOptimizationState,
    data::ExpertModeData,
    ip::ComplexF64,
)
    amp = pstate.amp
    cfg = pstate.config
    mfham = pstate.mfham
    ws = pstate.workspace
    qp_weight = parton_qp_weight(data)

    e = ComplexF64(parton_diag_energy(pstate.physham, cfg))

    @inbounds for h in pstate.physham.hops
        ri, rj, t = h.site1, h.site2, h.value

        # t · b†_j b_i: 粒子は i から j へ移る(x′ は j の粒子を i へ戻した配置)
        if is_occupied(cfg, rj) && !is_occupied(cfg, ri)
            m = site_particle(cfg, rj)
            ratio, _ = parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m, ri)
            log_pr = parton_log_proj_ratio(cfg, m, rj, ri)
            e += t * exp(log_pr) * ratio
        end

        # conj(t) · b†_i b_j: 逆向き(暗黙の h.c.)
        if is_occupied(cfg, ri) && !is_occupied(cfg, rj)
            m = site_particle(cfg, ri)
            ratio, _ = parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m, rj)
            log_pr = parton_log_proj_ratio(cfg, m, ri, rj)
            e += conj(t) * exp(log_pr) * ratio
        end
    end

    return e
end

# ---------------------------------------------------------------------
# 契約 5
# ---------------------------------------------------------------------

"""
スロット規約: パラメータ p(1-based、フラット並びは [射影 | MF])に対し
sr_opt_o の (2p+1, 2p+2) が (∂lnΨ/∂Re, ∂lnΨ/∂Im) のスロット。

既存コードは imag スロットに `val * im` を書く(vmc_main_cal.jl:2547-48)が、
それは f_ij が α に正則だから許される近道。MF ブロックは H が α* を含むため
非正則なので、両スロットを独立に計算して詰める。この近道をコピーしないこと。
"""
@inline o_slot_re(p::Int) = 2p + 1
@inline o_slot_im(p::Int) = 2p + 2

"""
    parton_calculate_o!(sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, n_proj)

契約 5(MF ブロック): サンプルごとの O を sr_opt_o に格納する。

    O_dof = (1/ip) Σ_qp w_qp (Π_f det^(f)_qp)
                 Σ_f Σ_m sgn_qp(r_m) · Σ_n ∂Φ^(f)_dof[map_qp(r_m), n] · (A^(f)_qp)⁻¹[n, m]

呼び出し前提: このサンプルについて契約 1 で錨済み・ip 計算済み・契約 0′ で
dorbitals 更新済み。射影ブロックの O は既存パターンの担当で [1..2n_proj+2] を
埋める。

最内の縮約は共役なしの転置積(dot 禁止)。コストは O(n_qp · F · n_idx · Ne²)
/ サンプル。
"""
function parton_calculate_o!(
    sr_opt_o::AbstractVector{ComplexF64},
    amp::PartonAmplitudeData,
    mfham::PartonMFHamiltonian,
    cfg::PartonConfiguration,
    data::ExpertModeData,
    qp_weight,
    ip::ComplexF64,
    n_proj::Int,
)
    inv_ip = inv(ip)
    Ne = amp.n_elec
    for k = 1:mfham.n_idx
        o_re = zero(ComplexF64)
        o_im = zero(ComplexF64)
        for qp = 1:amp.n_qp
            qmap = data.qp_trans[qp]
            qsgn = data.qp_trans_sgn[qp]
            prodd = one(ComplexF64)
            for f = 1:amp.n_flavor
                prodd *= amp.det_a[block_index(amp, qp, f)]
            end
            tr_re = zero(ComplexF64)
            tr_im = zero(ComplexF64)
            if amp.n_stored == amp.n_flavor
                # 標準経路。総和順は従来と逐語一致(既定の挙動を 1 bit も
                # 変えないため、下の高速路と共通化しない)。
                for f = 1:amp.n_flavor
                    Ainv = inv_block(amp, qp, f)
                    dPhiR = mfham.dorbitals[f][2k - 1]
                    dPhiI = mfham.dorbitals[f][2k]
                    for m = 1:Ne
                        r = particle_site(cfg, f, m)
                        rr = qmap[r]
                        s = qsgn[r]
                        a_re = zero(ComplexF64)
                        a_im = zero(ComplexF64)
                        @inbounds for n = 1:Ne          # 転置積(共役なし)
                            a_re += dPhiR[rr, n] * Ainv[n, m]
                            a_im += dPhiI[rr, n] * Ainv[n, m]
                        end
                        tr_re += s * a_re
                        tr_im += s * a_im
                    end
                end
            else
                # 対称高速路(v3.9): 固縛により全フレーバーの寄与が同一なので
                # Σ_f Tr → F · Tr。O(Ne²) の縮約が 1/F になる。総和の結合順だけが
                # 変わるので ON/OFF の O の差は丸め(1e-14)以内 — §8-14 が実測を出す。
                Ainv = inv_block(amp, qp, 1)
                dPhiR = mfham.dorbitals[1][2k - 1]
                dPhiI = mfham.dorbitals[1][2k]
                for m = 1:Ne
                    r = particle_site(cfg, 1, m)
                    rr = qmap[r]
                    s = qsgn[r]
                    a_re = zero(ComplexF64)
                    a_im = zero(ComplexF64)
                    @inbounds for n = 1:Ne              # 転置積(共役なし)
                        a_re += dPhiR[rr, n] * Ainv[n, m]
                        a_im += dPhiI[rr, n] * Ainv[n, m]
                    end
                    tr_re += s * a_re
                    tr_im += s * a_im
                end
                tr_re *= amp.n_flavor
                tr_im *= amp.n_flavor
            end
            w = qp_weight[qp] * prodd
            o_re += w * tr_re
            o_im += w * tr_im
        end
        p = n_proj + k                     # フラット並び [射影 | MF] の 1-based 位置
        sr_opt_o[o_slot_re(p)] = o_re * inv_ip
        sr_opt_o[o_slot_im(p)] = o_im * inv_ip   # オンサイト群は 0(∂Φ = 0)
    end
    return nothing
end

"""
    parton_fill_sr_opt_o!(sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, n_proj;
                          conjugate = true)

1 サンプル分の sr_opt_o を、上流アキュムレータへ渡せる形に仕上げる。

契約 5 の `parton_calculate_o!` と蓄積境界の `_parton_conjugate_mf_slots!` は
必ず対で呼ぶ必要があり、片方を忘れても例外は出ず力ベクトルだけが静かに
劣化する。呼び出し側が対を意識しなくて済むよう、ここに束ねてある。

`conjugate = false` は §8-7 の回帰ガード(共役を外すと勾配と合わなくなること
の確認)専用。本番経路では既定の true 以外を使わない。
"""
function parton_fill_sr_opt_o!(
    sr_opt_o::AbstractVector{ComplexF64},
    amp::PartonAmplitudeData,
    mfham::PartonMFHamiltonian,
    cfg::PartonConfiguration,
    data::ExpertModeData,
    qp_weight,
    ip::ComplexF64,
    n_proj::Int;
    conjugate::Bool = true,
    project_gauge::Bool = false,
    alpha::Vector{ComplexF64} = ComplexF64[],
)
    fill!(sr_opt_o, 0)
    sr_opt_o[1] = ComplexF64(1)     # 既存規約: 先頭 2 スロットは (1, 0)
    sr_opt_o[2] = ComplexF64(0)
    parton_calculate_o!(sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, n_proj)
    conjugate && _parton_conjugate_mf_slots!(sr_opt_o, n_proj, mfham.n_idx)
    project_gauge && _parton_project_gauge_from_o!(sr_opt_o, mfham, alpha, n_proj)
    return sr_opt_o
end

"""
    _parton_project_gauge_from_o!(sr_opt_o, mfham, alpha, n_proj)

サンプルごとの O からゲージ方向の成分を除く(DESIGN §2.5)。

厳密演算なら `Σ_slot v_slot O[slot] = 0`(ゲージ方向へ動かしても Ψ が変わらないので
対数微分の射影がゼロ)だが、MC のノイズがこの恒等式を破る。破れたぶんは
正則化 ε 付きの S⁻¹ で 1/ε 倍されて α をゲージ方向へ押し流す。

ここで O 自体から成分を抜いておくと、S も力ベクトルも**構成的に**ゲージ方向を
消す(`S v = 0`, `g·v = 0`)。参照実装(PartonFCI/vmc_chi)は力ベクトルだけを
射影しているが、O を直に落とせば計量まで揃うので上流の `stochastic_opt!` に
一切触らずに済む。

スケール方向は `α → (1+ε)α` なので、実自由度スロット空間での方向ベクトルは
その時点の α そのもの(Re スロットには Re α_k、Im スロットには Im α_k)。
"""
function _parton_project_gauge_from_o!(
    sr_opt_o::AbstractVector{ComplexF64},
    mfham::PartonMFHamiltonian,
    alpha::Vector{ComplexF64},
    n_proj::Int,
)
    for grp in mfham.gauge_scale_groups
        num = zero(ComplexF64)
        den = 0.0
        for k in grp
            p = n_proj + k
            vr, vi = real(alpha[k]), imag(alpha[k])
            num += vr * sr_opt_o[o_slot_re(p)] + vi * sr_opt_o[o_slot_im(p)]
            den += vr * vr + vi * vi
        end
        den > 1e-30 || continue
        c = num / den
        for k in grp
            p = n_proj + k
            sr_opt_o[o_slot_re(p)] -= c * real(alpha[k])
            sr_opt_o[o_slot_im(p)] -= c * imag(alpha[k])
        end
    end
    # シフト方向(H → H + μI)も同様に落とす
    for grp in mfham.gauge_shift_groups
        isempty(grp) && continue
        num = zero(ComplexF64)
        for k in grp
            num += sr_opt_o[o_slot_re(n_proj + k)]
        end
        c = num / length(grp)
        for k in grp
            sr_opt_o[o_slot_re(n_proj + k)] -= c
        end
    end
    return nothing
end

"""
    _parton_conjugate_mf_slots!(sr_opt_o, n_proj, n_idx)

MF ブロックのスロットを複素共役にしてから上流のアキュムレータへ渡す。

実パラメータ θ に対する変分エネルギーの勾配は

    ∂E/∂θ = 2 Re[ ⟨E_loc O_θ*⟩ − ⟨E_loc⟩⟨O_θ*⟩ ]

で、O には共役が要る。ところが既存の `calculate_oo!` / `calculate_oo_store!` は
`HO[j] += w · e · srOptO[j]` と共役なしで蓄積し、`build_s_matrix_and_g_vector!` は
その実部をそのまま力ベクトルに使う。f_ij のように波動関数がパラメータについて
正則な場合は、その規約と 2 スロット(val, val·im)の詰め方が噛み合って正しい
勾配になるが、H が α* を含む MF ブロックは非正則なので噛み合わない。

共役を先に取っておくと、上流が読む量は

    HO[j]     → ⟨e · O_j*⟩        (これが必要な形)
    OO[0][j]  → ⟨O_j*⟩            (実部は ⟨O_j⟩ と同じ)
    OO[i][j]  → ⟨O_i O_j*⟩ 相当   (build が使うのは実部だけで不変)

となり、S 行列は変わらないまま力ベクトルだけが正しくなる。この対応は
有限差分による勾配検証で確認してある(test_parton_ed_convergence.jl)。

契約 5 の `parton_calculate_o!` 自体は DESIGN §1.4 の O をそのまま格納する。
上流の規約に合わせる変換はここ(受け渡し点)に閉じ込める。

**成立に必要な不変条件**: 「S 行列が不変」が言えるのは、共役しない側(MF 以外)の
全スロットの O が**実数**だからである。MF×他ブロックの交差項は片側だけ共役されるので、
相手が複素なら実部が変わってしまう(実測: 相手が `(v, i·v)` の正則型だと交差ブロックだけ
ΔS ≈ 0.17 ずれる。例外も非対称も出ず、静かに間違った計量になる)。
M1 は門番が射影因子を拒否して n_proj = 0 を保証しているのでこの条件は自明に成り立つ。
M2 で物理密度 Jastrow を足すときは「虚スロットに 0 を書く(O は実数)」という
既存 `set_projection_diff!` と同じ規約を守ること。DESIGN §7 に固定条件として記載。

もう 1 つの前提として、SR-CG 経路(`operate_by_s!`)は全スロットが MF なら偶然
共役に不変だが、これは偶然にすぎない。門番の `NSRCG = 0` がこのシムの妥当性を
支えている。
"""
function _parton_conjugate_mf_slots!(
    sr_opt_o::AbstractVector{ComplexF64},
    n_proj::Int,
    n_idx::Int,
)
    @inbounds for k = 1:n_idx
        p = n_proj + k
        sr_opt_o[o_slot_re(p)] = conj(sr_opt_o[o_slot_re(p)])
        sr_opt_o[o_slot_im(p)] = conj(sr_opt_o[o_slot_im(p)])
    end
    return nothing
end

# ---------------------------------------------------------------------
# サンプルループ
# ---------------------------------------------------------------------

"""
    parton_sample_threading_enabled() -> Bool

§4 層 2(測定フェーズのサンプル並列)の opt-in ゲート。既存 threading.jl の
作法どおり `JULIA_MVMC_INNER_THREADS=1` かつ複数スレッド起動時のみ true。
既定は逐次。C-mVMC が OMP で回しているのはまさにこのループで、upstream Julia が
無効化しているのは C の総和順序をビット再現するため — パートン経路は縮約を
サンプル順の逐次パスに分離することでビット一致を保ったまま並列化できる。
"""
@inline parton_sample_threading_enabled() = vmc_inner_threading_requested(true)

"チャンク t(1-based)が受け持つサンプル範囲。連続・決定的。"
@inline _parton_chunk(t::Int, nt::Int, n::Int) =
    (1 + div((t - 1) * n, nt)):div(t * n, nt)

"""
    _parton_thread_ctx!(pstate, data, nt, len_o) -> PartonMainCalThreadContext

スレッド別ワークスペースを遅延確保する(寸法が合えば再利用)。
"""
function _parton_thread_ctx!(
    pstate::PartonOptimizationState, data::ExpertModeData, nt::Int, len_o::Int)
    mp = data.modpara
    amp = pstate.amp
    n_sample = mp.nvmc_sample
    ctx = pstate.thread_ctx
    if ctx isa PartonMainCalThreadContext &&
       length(ctx.cfgs) == nt &&
       size(ctx.o_all) == (len_o, n_sample) &&
       length(ctx.amps[1].inv_a) == length(amp.inv_a)
        return ctx
    end
    ctx = PartonMainCalThreadContext(
        [PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, 1) for _ = 1:nt],
        [PartonAmplitudeData(amp.n_qp, amp.n_flavor, amp.n_elec;
                             n_stored = amp.n_stored) for _ = 1:nt],
        [PartonSamplingWorkspace(amp.n_elec, amp.n_qp * amp.n_stored) for _ = 1:nt],
        [CTimer(false) for _ = 1:nt],
        zeros(ComplexF64, n_sample),
        fill(false, n_sample),
        zeros(ComplexF64, len_o, n_sample),
    )
    pstate.thread_ctx = ctx
    return ctx
end

"""
    _parton_main_cal_samples_threaded!(pstate, data, qp_weight, n_proj,
                                       gauge_proj, α_now, c_timer)

§4 層 2 の並列フェーズ: 保存済みサンプルごとの (E_loc, O) を計算して
サンプル別バッファへ書く。**乱数は一切消費しない**(このフェーズに rng は
存在しない)。各スレッドは自分の cfg / amp / ws だけに書き、サンプル別
バッファへの書き込みは列単位で排他。縮約は呼び出し側の逐次パスが行う。

E_loc / O の値はスレッド割りに依らずサンプルごとに決定的(逐次と同一の
入力から同一の演算列で計算される)なので、後段の逐次縮約と合わせて
**全体がスレッド数に依らずビット一致**する(§8-15 が機械検証)。
"""
function _parton_main_cal_samples_threaded!(
    pstate::PartonOptimizationState,
    data::ExpertModeData,
    qp_weight,
    n_proj::Int,
    gauge_proj::Bool,
    α_now::Vector{ComplexF64},
    c_timer::CTimer,
)
    mp = data.modpara
    sr = pstate.state.sr_opt
    nt = Base.Threads.nthreads()
    len_o = length(sr.sr_opt_o)
    ctx = _parton_thread_ctx!(pstate, data, nt, len_o)
    for t = 1:nt
        ctx.timers[t] = CTimer(ctimer_enabled(c_timer))
    end

    n_sample = mp.nvmc_sample
    Base.Threads.@threads :static for t = 1:nt
        cfg_t = ctx.cfgs[t]
        amp_t = ctx.amps[t]
        ws_t = ctx.wss[t]
        timer_t = ctx.timers[t]
        # E_loc が pstate 越しに amp/cfg/ws を読むので、スレッド分をまとめた
        # 外箱を作る(state / mfham / physham は読み取り共有)
        ps_t = PartonOptimizationState(
            pstate.state, amp_t, cfg_t, ws_t, pstate.mfham, pstate.physham)
        for s in _parton_chunk(t, nt, n_sample)
            parton_restore_sample_from!(cfg_t, pstate.config, s)
            parton_recompute_amplitude_all!(amp_t, pstate.mfham, cfg_t, data, ws_t)
            ip = parton_calculate_ip(amp_t, qp_weight)
            if abs(ip) < 1e-30
                ctx.ok_all[s] = false
                continue
            end
            ctimer_start!(timer_t, 808)
            ctx.e_all[s] = parton_local_energy(ps_t, data, ip)
            ctimer_stop!(timer_t, 808)
            ctimer_start!(timer_t, 809)
            parton_fill_sr_opt_o!(
                view(ctx.o_all, :, s), amp_t, pstate.mfham, cfg_t, data,
                qp_weight, ip, n_proj;
                project_gauge = gauge_proj, alpha = α_now)
            ctimer_stop!(timer_t, 809)
            ctx.ok_all[s] = true
        end
    end
    # スレッド別タイマは合算(808/809 は CPU 秒の総和になる点に注意)
    ctimer_merge_all!(c_timer, ctx.timers)
    return ctx
end

"""
    parton_main_cal!(pstate, data)

契約 4 と 5 をサンプルごとに回してエネルギーと SR 量を蓄積する。

サンプルごとに保存済み配置を復元し、契約 1 で錨を打ち直してから測定する
(DESIGN §4: サンプル毎再計算は測定側の分担)。重み w は既存経路と同じく 1
(重点サンプリング)。

O の蓄積は既存の `calculate_oo!` / `calculate_oo_store!` にそのまま委譲する。
これらは sr_opt_o の中身と (w, e) にしか依存しないので、MF ブロックが
非正則であることは蓄積側に影響しない。
"""
function parton_main_cal!(pstate::PartonOptimizationState, data::ExpertModeData;
                          c_timer::CTimer = CTIMER_DISABLED,
                          force_threaded::Union{Nothing,Bool} = nothing)
    # `force_threaded` は §8-15(並列 = 逐次のビット一致)の検証専用。
    # 本番経路では渡さない(既定 nothing = env と nthreads による自動判定)。
    amp = pstate.amp
    cfg = pstate.config
    mfham = pstate.mfham
    ws = pstate.workspace
    mp = data.modpara
    qp_weight = parton_qp_weight(data)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj

    sr = pstate.state.sr_opt
    energy = pstate.state.energy

    fill!(sr.sr_opt_oo, 0)
    fill!(sr.sr_opt_ho, 0)
    fill!(sr.sr_opt_o, 0)
    energy.wc = 0
    energy.etot = 0
    energy.etot2 = 0
    energy.sztot = 0
    energy.sztot2 = 0

    # ゲージ方向を O から落とすか(PartonGaugeFix と同じスイッチで制御)
    gauge_proj = mp.parton_gauge_fix != 0
    α_now = gauge_proj ? parton_alpha_from_terms(data) : ComplexF64[]

    use_store = mp.nstore_o != 0
    n_skipped = 0
    # 実際に蓄積したサンプル数。ノード上のサンプルを飛ばすと s とずれるので、
    # store の書き込み位置と finalize に渡す個数はこちらを使う。前ステップの O が
    # 残ったスロットを finalize が読むと、OO だけ古い値で汚れて S が静かに壊れる。
    n_stored = 0
    use_store && fill!(sr.sr_opt_o_store, 0)

    threaded = force_threaded === nothing ?
        (parton_sample_threading_enabled() &&
         mp.nvmc_sample >= Base.Threads.nthreads()) : force_threaded
    ctx = threaded ?
        _parton_main_cal_samples_threaded!(
            pstate, data, qp_weight, n_proj, gauge_proj, α_now, c_timer) : nothing

    for s = 1:mp.nvmc_sample
        local e::ComplexF64
        if threaded
            # 並列フェーズが済ませたサンプル別の (E_loc, O) を読むだけ。
            # 縮約はこの逐次ループがサンプル順に行うので、蓄積の演算列は
            # 逐次経路と完全に同一 = ビット一致(スレッド数にも依らない)。
            if !ctx.ok_all[s]
                n_skipped += 1
                continue
            end
            e = ctx.e_all[s]
            copyto!(sr.sr_opt_o, view(ctx.o_all, :, s))
        else
            parton_restore_sample!(cfg, s)
            parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
            ip = parton_calculate_ip(amp, qp_weight)
            if abs(ip) < 1e-30
                n_skipped += 1        # ノード上のサンプル。寄与できないので飛ばす
                continue
            end

            ctimer_start!(c_timer, 808)
            e = parton_local_energy(pstate, data, ip)
            ctimer_stop!(c_timer, 808)

            ctimer_start!(c_timer, 809)
            parton_fill_sr_opt_o!(
                sr.sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, n_proj;
                project_gauge = gauge_proj, alpha = α_now)
            ctimer_stop!(c_timer, 809)
        end
        w = 1.0

        energy.wc += w
        energy.etot += w * e
        energy.etot2 += w * conj(e) * e

        if use_store
            calculate_oo_store!(
                sr.sr_opt_oo,
                sr.sr_opt_ho,
                sr.sr_opt_o_store,
                sr.sr_opt_o,
                w,
                e,
                n_stored,                   # sr_opt_o_store の添字は 0-based
                sr.sr_opt_size,
            )
        else
            calculate_oo!(sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_o, w, e, sr.sr_opt_size)
        end
        n_stored += 1
    end

    # store 経路では calculate_oo_store! は O の保存と HO の蓄積しか行わない。
    # <O†O> は最後にまとめて store から組む(既存 vmc_main_cal! と同じ段取り)。
    # これを落とすと直接ソルバが全ゼロの S を受け取り、SR が NaN を返す。
    if use_store
        ctimer_start!(c_timer, 810)
        finalize_oo_store!(
            sr.sr_opt_oo,
            sr.sr_opt_o_store,
            sr.sr_opt_size,
            n_stored,               # 実際に詰めた個数。wc の母数と揃える
            nsrcg = false,          # 門番が NSRCG = 0 を保証している
        )
        ctimer_stop!(c_timer, 810)
    end

    n_skipped > 0 && @warn "Skipped samples sitting on a node of the wave function" n_skipped
    return nothing
end
