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

        fill!(sr.sr_opt_o, 0)
        sr.sr_opt_o[1] = ComplexF64(1)     # 既存規約: 先頭 2 スロットは (1, 0)
        sr.sr_opt_o[2] = ComplexF64(0)
        parton_calculate_o!(sr.sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, n_proj)

        if use_store
            calculate_oo_store!(
                sr.sr_opt_oo,
                sr.sr_opt_ho,
                sr.sr_opt_o_store,
                sr.sr_opt_o,
                w,
                e,
                s - 1,                      # sr_opt_o_store の添字は 0-based
                sr.sr_opt_size,
            )
        else
            calculate_oo!(sr.sr_opt_oo, sr.sr_opt_ho, sr.sr_opt_o, w, e, sr.sr_opt_size)
        end
    end

    # store 経路では calculate_oo_store! は O の保存と HO の蓄積しか行わない。
    # <O†O> は最後にまとめて store から組む(既存 vmc_main_cal! と同じ段取り)。
    # これを落とすと直接ソルバが全ゼロの S を受け取り、SR が NaN を返す。
    if use_store
        finalize_oo_store!(
            sr.sr_opt_oo,
            sr.sr_opt_o_store,
            sr.sr_opt_size,
            mp.nvmc_sample,
            nsrcg = false,          # 門番が NSRCG = 0 を保証している
        )
    end

    n_skipped > 0 && @warn "Skipped samples sitting on a node of the wave function" n_skipped
    return nothing
end
