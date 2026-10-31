"""
Pfaffian パートンモードの O(契約 5)と ⟨O O†⟩ の組み立て
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §1.5:

    O_k = (1/ip) Σ_qp w_qp Pf F_qp · Σ_{I<J} G_qp[J, I] · s_qp(r_I) s_qp(r_J) · 2·sgn[α_I, α_J] · [idx(α_I, α_J) = k]

(∂ln Pf F / ∂F_IJ = G_JI を、F_JI = −F_IJ と連動させた変化について使う。上流の
`slater_elm_diff_fsz!` の `−InvM·cs·sgn`(全順序対の和)と同じ量。)

f は p について正則なので、スロットは (O, i·O) と詰める。そのうえで Φ モードと同じく
**共役シム**を掛ける(上流は E_loc に `conj(z/ip)` を足し、パートンは比をそのまま足すという
規約の差を受け渡し点で吸収する。DESIGN_parton.md §12.2)。T2 が力 = 勾配で検証する。
"""

"""
    parton_pair_calculate_o!(sr_opt_o, amp, src, data, qp_weight, ip, n_proj)

f のブロックの O を `sr_opt_o` に**加算**する(呼び出し側が 0 に初期化しておくこと)。
Re スロットに O_k、Im スロットに i·O_k。コストは O(n_qp·M²/2)/サンプル。
特異な qp(Pf = 0)は寄与しないので飛ばす。
"""
function parton_pair_calculate_o!(
    sr_opt_o::AbstractVector{ComplexF64},
    amp::PartonPairAmplitudeData,
    src::PartonPairOrbital,
    data::ExpertModeData,
    qp_weight,
    ip::ComplexF64,
    n_proj::Int,
)
    K = amp.n_flavor
    N = amp.n_elec
    ns = src.n_site
    inv_ip = inv(ip)
    for qp = 1:amp.n_qp
        pf = amp.pf_f[qp]
        iszero(pf) && continue
        c = qp_weight[qp] * pf * inv_ip
        G = pair_inv_block(amp, qp)
        qmap = data.qp_trans[qp]
        qsgn = data.qp_trans_sgn[qp]
        @inbounds for mj = 1:N
            rj = amp.row_site[mj]
            tj = qmap[rj]
            sj = qsgn[rj]
            for b = 1:K
                J = (mj - 1) * K + b
                β = parton_pair_state(ns, tj, b)
                for mi = 1:mj
                    ri = amp.row_site[mi]
                    ti = qmap[ri]
                    sij = qsgn[ri] * sj
                    for a = 1:K
                        I = (mi - 1) * K + a
                        I < J || continue
                        α = parton_pair_state(ns, ti, a)
                        k = src.pair_idx[α, β]
                        k == 0 && continue
                        sr_opt_o[o_slot_re(n_proj + k)] +=
                            c * G[J, I] * (2 * sij * src.pair_sgn[α, β])
                    end
                end
            end
        end
    end
    @inbounds for k = 1:src.n_idx
        sr_opt_o[o_slot_im(n_proj + k)] = im * sr_opt_o[o_slot_re(n_proj + k)]
    end
    return sr_opt_o
end

"""
    parton_fill_sr_opt_o!(sr_opt_o, amp::PartonPairAmplitudeData, src::PartonPairOrbital, cfg, data, qp_weight, ip, n_proj; conjugate, project_gauge, alpha)

契約 5(Pf 版)。並びは [1, 0 | 射影(Jastrow)| p]。射影は α・Φ と同じく O_p = cnt_p(実数、
虚スロットは 0。共役シムの前提)。`project_gauge` / `alpha` は α 用なので受け取って無視する
(呼び出し元 `parton_main_cal!` が渡す)。`conjugate = false` は回帰ガードのテスト専用。
"""
function parton_fill_sr_opt_o!(
    sr_opt_o::AbstractVector{ComplexF64},
    amp::PartonPairAmplitudeData,
    src::PartonPairOrbital,
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
    @inbounds for p = 1:n_proj
        sr_opt_o[o_slot_re(p)] = ComplexF64(cfg.proj_cnt[p])
    end
    parton_pair_calculate_o!(sr_opt_o, amp, src, data, qp_weight, ip, n_proj)
    conjugate && _parton_conjugate_mf_slots!(sr_opt_o, n_proj, src.n_idx)
    return sr_opt_o
end

"""
    parton_finalize_oo_store!(::PartonPairOrbital, sr_opt_oo, sr_opt_o_store, sr_opt_size, n_stored)

store 経路の ⟨O O†⟩ を BLAS の行列積で組む(Φ 版と同じ本体。f のパラメータ数は大きいので
上流の素朴な 3 重ループは使わない)。並びは上流と同じく OO[(i−1)·size₂ + j] = Σ_s O[i,s]·conj(O[j,s])。
Φ のファイルは別セッションで改装中のため本体を複製している。合流後に共通化する(DESIGN_parton_pf.md §10)。
"""
function parton_finalize_oo_store!(::PartonPairOrbital, sr_opt_oo::Vector{ComplexF64},
                                   sr_opt_o_store::Vector{ComplexF64}, sr_opt_size::Int,
                                   n_stored::Int)
    size_2 = 2 * sr_opt_size
    M = reshape(view(sr_opt_oo, 1:(size_2 * size_2)), size_2, size_2)
    if n_stored == 0
        fill!(M, 0)
        return nothing
    end
    A = reshape(view(sr_opt_o_store, 1:(size_2 * n_stored)), size_2, n_stored)
    mul!(M, A, A')          # ZGEMM
    M .= conj.(M)
    return nothing
end
