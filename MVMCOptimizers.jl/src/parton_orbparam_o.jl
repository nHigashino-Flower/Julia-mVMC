"""
Φ 直接変分モードの O(契約 5 の Φ 版)
--- parton-mode (fork addition) ---

DESIGN_parton.md §12.11 / §12.3.2。パラメータは Φ_f[s,n] の全要素で、ψ はこれについて
正則なので、上流の f_ij と同じく 1 パラメータを 2 スロット (O, i·O) に詰める。
そのうえで α モードと同じ共役シム(`_parton_conjugate_mf_slots!`)を掛ける。
シムが要る理由は非正則性ではなく E_loc の規約差(§12.2):上流の e は
conj(z/ip) で `HO += e·O` と蓄積するが、パートンの E_loc は比をそのまま足すので、
O 側を共役しておかないと力ベクトルが ⟨E_loc O⟩ になってしまう。
"""

"""
    parton_orbparam_calculate_o!(sr_opt_o, amp, phi, cfg, data, qp_weight, ip, n_proj)

1 サンプル分の Φ ブロックの O を書く(**共役前**)。A_qp,f[m,n] = sgn_qp(r_m)·Φ_f[map_qp(r_m), n]
について

    O[f,s,n] = (1/ip) Σ_qp w_qp (Π_g det A_qp,g) Σ_m δ_{s, map_qp(r_m)} sgn_qp(r_m) (A_qp,f)⁻¹[n, m]

を Re スロットに、i·O を Im スロットに書く(k = 平坦位置、p = n_proj + k)。
制限付き(写像が恒等でない、DESIGN §12.12)なら O_θ = Σ_{k ∈ idx} c_k·O[k] を θ のスロット
(p = n_proj + θ の位置)に書く。θ も正則なパラメータなので Im スロットは i·Re のまま。
Φ ブロックのスロットは 0 で初期化済みであること(`parton_fill_sr_opt_o!` が保証する)。

- `Π_g det == 0` の qp は飛ばす。特異ブロックの `inv_a` は更新されていないので、
  0 × 不定値で NaN を作らないため
- Φ モードは det^F 高速路を使わない(n_stored = F)。違えば error

コストは O(n_qp · F · Ne²) / サンプル(と Φ ブロックの書き込み O(n_orb))。
"""
function parton_orbparam_calculate_o!(
    sr_opt_o::AbstractVector{ComplexF64},
    amp::PartonAmplitudeData,
    phi::PartonDirectPhi,
    cfg::PartonConfiguration,
    data::ExpertModeData,
    qp_weight,
    ip::ComplexF64,
    n_proj::Int,
)
    amp.n_stored == amp.n_flavor || error(
        "The direct-orbital mode needs one amplitude block per flavor " *
        "(n_stored = $(amp.n_stored), n_flavor = $(amp.n_flavor)).")
    inv_ip = inv(ip)
    Ne = amp.n_elec
    n_site = size(phi.orbitals[1], 1)
    map = phi.map
    for qp = 1:amp.n_qp
        prodd = one(ComplexF64)
        for f = 1:amp.n_flavor
            prodd *= amp.det_a[block_index(amp, qp, f)]
        end
        iszero(prodd) && continue
        w = qp_weight[qp] * prodd * inv_ip
        qmap = data.qp_trans[qp]
        qsgn = data.qp_trans_sgn[qp]
        for f = 1:amp.n_flavor
            Ainv = inv_block(amp, qp, f)
            for m = 1:Ne
                r = particle_site(cfg, f, m)
                s = qmap[r]
                ws = w * qsgn[r]
                if map.is_identity
                    @inbounds for n = 1:Ne
                        k = parton_orbparam_flat_index(f, s, n, n_site, Ne)
                        sr_opt_o[o_slot_re(n_proj + k)] += ws * Ainv[n, m]
                    end
                else
                    # 制限付き(DESIGN §12.12): O_θ = Σ_{k ∈ idx} c_k·O_Φ,k。
                    # 各寄与を写像経由で θ のスロットへ直接足す(作業領域を持たない)
                    @inbounds for n = 1:Ne
                        k = parton_orbparam_flat_index(f, s, n, n_site, Ne)
                        sr_opt_o[o_slot_re(n_proj + map.idx[k])] += map.coeff[k] * (ws * Ainv[n, m])
                    end
                end
            end
        end
    end
    @inbounds for k = 1:map.n_idx
        p = n_proj + k
        sr_opt_o[o_slot_im(p)] = im * sr_opt_o[o_slot_re(p)]   # 正則: ∂/∂Im Φ = i·∂/∂Φ
    end
    return nothing
end

"""
    parton_fill_sr_opt_o!(sr_opt_o, amp, phi::PartonDirectPhi, cfg, data, qp_weight, ip, n_proj;
                          conjugate = true, project_gauge = false, alpha = ComplexF64[])

1 サンプル分の sr_opt_o を、上流アキュムレータへ渡せる形に仕上げる(Φ モード)。
並びは α 版と同じく [(1, 0) | 射影(Jastrow の cnt、実数)| Φ]。

`project_gauge` と `alpha` は受け取って無視する。呼び出し元(`parton_main_cal!`)が
α 版と共通で、PartonGaugeFix に従って渡してくるため。Φ モードの冗長方向
(GL(Ne) とパートンゲージ)は §12.11 の方針どおり SR の正則化と再スケールに任せる。

`conjugate = false` は回帰ガードのテスト専用。
"""
function parton_fill_sr_opt_o!(
    sr_opt_o::AbstractVector{ComplexF64},
    amp::PartonAmplitudeData,
    phi::PartonDirectPhi,
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
    # 射影ブロック(Jastrow): O_p = cnt_p。実数で虚スロットは 0(共役シムの前提)
    @inbounds for p = 1:n_proj
        sr_opt_o[o_slot_re(p)] = ComplexF64(cfg.proj_cnt[p])
    end
    parton_orbparam_calculate_o!(sr_opt_o, amp, phi, cfg, data, qp_weight, ip, n_proj)
    conjugate && _parton_conjugate_mf_slots!(sr_opt_o, n_proj, phi.map.n_idx)
    return sr_opt_o
end

"""
    parton_finalize_oo_store!(source, sr_opt_oo, sr_opt_o_store, sr_opt_size, n_stored)

store 経路の ⟨O O†⟩ を組む。軌道の器でディスパッチする(DESIGN §12.10 の BLAS 版 finalize)。

- `PartonMFHamiltonian`(α): 上流の `finalize_oo_store!` をそのまま呼ぶ(ビット一致)
- `PartonDirectPhi`(Φ): BLAS の行列積で組む。上流版は (2·SROptSize)² × サンプル数の
  素朴な 3 重ループで、Φ モード(n_para が数百)では実行時間の 9 割を占めた
  (KM 9×4・n_para 486・NVMCSample 2000 で 4.25 s/step)。総和の順序が変わるので
  上流版とは丸め誤差だけ違う

並びは上流と同じく OO[(i−1)·size₂ + j] = Σ_s O[i,s]·conj(O[j,s])。列優先の行列 M として
M[j,i] = conj((A·A†)[j,i])(A = O_store、size₂ × n_stored)。
"""
parton_finalize_oo_store!(::PartonMFHamiltonian, sr_opt_oo::Vector{ComplexF64},
                          sr_opt_o_store::Vector{ComplexF64}, sr_opt_size::Int,
                          n_stored::Int) =
    finalize_oo_store!(sr_opt_oo, sr_opt_o_store, sr_opt_size, n_stored; nsrcg = false)

function parton_finalize_oo_store!(::PartonDirectPhi, sr_opt_oo::Vector{ComplexF64},
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
