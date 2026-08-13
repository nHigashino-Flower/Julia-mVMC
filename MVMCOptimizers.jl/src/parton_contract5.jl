# =====================================================================
# 契約0′+契約5: 平均場パラメータ α の対数微分 O
#
# 貼り込み先:
#   parton_update_orbital_derivatives!            → parton_orbital.jl
#   o_slot_re / o_slot_im / parton_calculate_o!   → parton_vmc_main_cal.jl
#
# 前提(PartonMFHamiltonian への追加フィールド):
#   dorbitals     :: Vector{Vector{Matrix{ComplexF64}}}
#                    [f][dof] で dof = 2k-1 (Re α_k) / 2k (Im α_k)、各 n_site×n_elec
#   dh_uo_scratch :: Matrix{ComplexF64}   (n_site × n_elec の作業行列)
# =====================================================================

"""
契約0′: ∂Φ^(f)/∂θ を一次摂動論で構築。SRステップ毎、契約0(対角化)の直後に呼ぶ。

連鎖律の eigen 段:
    W = U_unocc† (∂θH) U_occ,   G[u,n] = W[u,n] / (ε_n − ε_u),   ∂Φ = U_unocc · G

- 占有↔占有の混合は Tr[A⁻¹ ∂A] から厳密に消える(摂動係数が反エルミート・対角ゼロ)
  ので、非占有への漏れだけ構築すればよい。分母は HOMO-LUMO ギャップで下から
  抑えられる(契約0 の min_gap 検知が数値的な保険)。
- ∂θH は α に非正則な H の実自由度ごとの微分:
    ∂H/∂Re α_k = T_k + T_k†      (オンサイト群は T_k のみ・t 実)
    ∂H/∂Im α_k = i (T_k − T_k†)  (オンサイト群の Im は門番凍結: ゼロを格納)
- `Uu' * dHUo` の随伴(共役)は正しい: これはブラ ⟨φ_u| との真の内積。
  「転置積・dot 禁止」の規則は振幅の双線形縮約(契約2/5 の最内ループ)の話で、
  ここには適用されない。取り違えに注意。
"""
function parton_update_orbital_derivatives!(mfham, n_elec::Int)
    n_site = size(mfham.h_mf[1], 1)
    n_un = n_site - n_elec
    for f in eachindex(mfham.h_mf)
        U  = mfham.eig_vecs[f]
        ev = mfham.eig_vals[f]
        Uo = @view U[:, 1:n_elec]
        Uu = @view U[:, (n_elec+1):n_site]
        # 分母 D[u, n] = ε_n − ε_{Ne+u}(u=非占有, n=占有)
        D = [ev[n] - ev[n_elec + u] for u in 1:n_un, n in 1:n_elec]

        for k in 1:mfham.n_idx
            onsite = mfham.is_onsite_group[k]
            for part in 1:2                        # 1 = Re, 2 = Im
                dof = 2 * (k - 1) + part
                dPhi = mfham.dorbitals[f][dof]
                if onsite && part == 2
                    fill!(dPhi, 0)                 # 凍結成分。ゼロを埋めるだけ
                    continue
                end

                # dHUo = (∂θ H^(f)) · U_occ をテンプレートのスパース走査で構築
                dHUo = mfham.dh_uo_scratch
                fill!(dHUo, 0)
                for e in mfham.template[k]
                    e.flavor == f || continue
                    t = e.coeff
                    if onsite                                       # 対角: t は実
                        @views dHUo[e.site1, :] .+= t .* Uo[e.site1, :]
                    elseif part == 1                                # T + T†
                        @views dHUo[e.site1, :] .+= t        .* Uo[e.site2, :]
                        @views dHUo[e.site2, :] .+= conj(t)  .* Uo[e.site1, :]
                    else                                            # i(T − T†)
                        c = im * t
                        @views dHUo[e.site1, :] .+= c        .* Uo[e.site2, :]
                        @views dHUo[e.site2, :] .+= conj(c)  .* Uo[e.site1, :]
                    end
                end

                W = Uu' * dHUo                      # (n_un × n_elec)。随伴で正しい(上記注)
                mul!(dPhi, Uu, W ./ D)              # ∂Φ = U_unocc (W ./ D)
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------
# 契約5(サンプル毎)
# ---------------------------------------------------------------------

"""
スロット規約: パラメータ p(1-based、フラット並びは [射影 | MF])に対し
sr_opt_o の (2p+1, 2p+2) が (∂lnΨ/∂Re, ∂lnΨ/∂Im) のスロット。
既存コードは imag スロットに `val * im` を書く(vmc_main_cal.jl L2547-48)が、
それは f_ij が α に正則だから許される近道。MF ブロックは非正則なので
両スロットを独立に計算して詰める。この近道をコピーしないこと。
"""
@inline o_slot_re(p::Int) = 2p + 1
@inline o_slot_im(p::Int) = 2p + 2

"""
契約5(MFブロック): サンプルごとの O を sr_opt_o に格納。

    O_dof = (1/ip) Σ_qp w_qp (Π_f det^(f)_qp)
                 Σ_f Σ_m sgn_qp(r_m) · Σ_n ∂Φ^(f)_dof[map_qp(r_m), n] · (A^(f)_qp)⁻¹[n, m]

呼び出し前提: このサンプルについて契約1で amp が錨済み・ip 計算済み。
射影ブロックの O は別関数(既存パターン踏襲)が [1..2n_proj+2] を埋める。
最内の縮約は共役なしの転置積(dot 禁止。複素位相つき t のテストで検証)。
コスト: O(n_qp · F · n_idx · Ne²) / サンプル。
"""
function parton_calculate_o!(sr_opt_o, amp, mfham, cfg, data, qp_weight,
                             ip::ComplexF64, n_proj::Int)
    inv_ip = inv(ip)
    Ne = amp.n_elec
    for k in 1:mfham.n_idx
        o_re = zero(ComplexF64)
        o_im = zero(ComplexF64)
        for qp in 1:amp.n_qp
            qmap = data.qp_trans[qp]
            qsgn = data.qp_trans_sgn[qp]
            prodd = one(ComplexF64)
            for f in 1:amp.n_flavor
                prodd *= amp.det_a[block_index(amp, qp, f)]
            end
            tr_re = zero(ComplexF64)
            tr_im = zero(ComplexF64)
            for f in 1:amp.n_flavor
                Ainv  = inv_block(amp, qp, f)
                dPhiR = mfham.dorbitals[f][2k - 1]
                dPhiI = mfham.dorbitals[f][2k]
                for m in 1:Ne
                    r  = particle_site(cfg, f, m)
                    rr = qmap[r]
                    s  = qsgn[r]
                    a_re = zero(ComplexF64)
                    a_im = zero(ComplexF64)
                    @inbounds for n in 1:Ne          # 転置積(共役なし)
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
        p = n_proj + k                               # フラット並び [射影 | MF] の 1-based 位置
        sr_opt_o[o_slot_re(p)] = o_re * inv_ip
        sr_opt_o[o_slot_im(p)] = o_im * inv_ip       # オンサイト群は 0 が入る(dΦ=0)
    end
    return nothing
end

# =====================================================================
# 検証(§8 追補):
#   1. 有限差分: トイ系で ln ip(θ+δ) − ln ip(θ−δ) / 2δ と両スロットを突き合わせ
#      (Re/Im 両方向。これが ForwardDiff 撤回後の契約0′/5 の一次検証)
#   2. 複素位相つき t で実施(転置/随伴の取り違えは実数では見えない)
#   3. sr_opt_oo/ho への蓄積は既存の蓄積関数を流用(sr_opt_o の中身にのみ依存する
#      ことを実装時に一目確認)
# =====================================================================