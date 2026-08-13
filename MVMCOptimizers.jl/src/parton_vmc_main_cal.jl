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
)
    fill!(sr_opt_o, 0)
    sr_opt_o[1] = ComplexF64(1)     # 既存規約: 先頭 2 スロットは (1, 0)
    sr_opt_o[2] = ComplexF64(0)
    parton_calculate_o!(sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, n_proj)
    conjugate && _parton_conjugate_mf_slots!(sr_opt_o, n_proj, mfham.n_idx)
    return sr_opt_o
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
    parton_main_cal!(pstate, data)

契約 4 と 5 をサンプルごとに回してエネルギーと SR 量を蓄積する。

サンプルごとに保存済み配置を復元し、契約 1 で錨を打ち直してから測定する
(DESIGN §4: サンプル毎再計算は測定側の分担)。重み w は既存経路と同じく 1
(重点サンプリング)。

O の蓄積は既存の `calculate_oo!` / `calculate_oo_store!` にそのまま委譲する。
これらは sr_opt_o の中身と (w, e) にしか依存しないので、MF ブロックが
非正則であることは蓄積側に影響しない。
"""
function parton_main_cal!(pstate::PartonOptimizationState, data::ExpertModeData)
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

    use_store = mp.nstore_o != 0
    n_skipped = 0
    # 実際に蓄積したサンプル数。ノード上のサンプルを飛ばすと s とずれるので、
    # store の書き込み位置と finalize に渡す個数はこちらを使う。前ステップの O が
    # 残ったスロットを finalize が読むと、OO だけ古い値で汚れて S が静かに壊れる。
    n_stored = 0
    use_store && fill!(sr.sr_opt_o_store, 0)

    for s = 1:mp.nvmc_sample
        parton_restore_sample!(cfg, s)
        parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
        ip = parton_calculate_ip(amp, qp_weight)
        if abs(ip) < 1e-30
            n_skipped += 1        # ノード上のサンプル。寄与できないので飛ばす
            continue
        end

        e = parton_local_energy(pstate, data, ip)
        w = 1.0

        energy.wc += w
        energy.etot += w * e
        energy.etot2 += w * conj(e) * e

        parton_fill_sr_opt_o!(
            sr.sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, n_proj)

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
        finalize_oo_store!(
            sr.sr_opt_oo,
            sr.sr_opt_o_store,
            sr.sr_opt_size,
            n_stored,               # 実際に詰めた個数。wc の母数と揃える
            nsrcg = false,          # 門番が NSRCG = 0 を保証している
        )
    end

    n_skipped > 0 && @warn "Skipped samples sitting on a node of the wave function" n_skipped
    return nothing
end
