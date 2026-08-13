"""
契約1: 現在の固縛配置から全 (qp, f) ブロックの A⁻¹ と det をゼロから再計算。
高速更新(契約3)の誤差蓄積をリセットする錨。O(n_qp · F · Ne³)。
"""
function parton_recompute_amplitude_all!(amp::PartonAmplitudeData,
    # (1) 特異ブロックはエラーにせずdet=0:波動関数のノードは物理なので、初期配置生成側が「非ゼロになるまで引き直す」責務を持ち、走行中に出たら再計算頻度の問題として@warn。
    # (2) detは生の複素数で保持(C版がPfMを生で持つのと同じ)。乗法的更新のドリフトは定期厳密再計算が錨になります。系が大きくなってオーバー/アンダーフローが見えたらlog空間化——§9に一行。
    # (3) lu!のピボット配列など小さな確保は放置:厳密再計算はもともとO(Ne³)なので誤差です(getrf!直叩きの完全ゼロアロケ化も§9)。ワークスペースに必要なのは現時点でa_scratch::Matrix{ComplexF64}(Ne×Ne)一枚だけ。
        mfham::PartonMFHamiltonian, config::PartonConfiguration,
        data::ExpertModeData, ws::PartonSamplingWorkspace)
    for qp in 1:amp.n_qp, f in 1:amp.n_flavor
        gather_a_block!(ws.a_scratch, mfham.orbitals[f], config, f,
                        data.qp_trans[qp], data.qp_trans_sgn[qp])
        F = lu!(ws.a_scratch; check = false)
        if !issuccess(F)                       # ノード上(det=0)
            amp.det_a[block_index(amp, qp, f)] = 0
            continue
        end
        amp.det_a[block_index(amp, qp, f)] = det(F)   # LU から O(Ne)
        Ainv = inv_block(amp, qp, f)
        copyto!(Ainv, I)
        ldiv!(F, Ainv)                          # A⁻¹ をブロックへ直接書く
    end
    return nothing
end

"""行の収集: A[m, n] = sgn · Φ[写像(r_m), n]。粒子=行、軌道=列。"""
function gather_a_block!(dest, Φ, config, f, qmap::Vector{Int}, qsgn::Vector{Int})
    for m in 1:size(dest, 1)
        r  = particle_site(config, f, m)        # PartonConfigurationのアクセサ
        dest[m, :] .= qsgn[r] .* @view Φ[qmap[r], :]
    end
end

"""契約1b: ⟨x|P|φ⟩ = Σ_qp w_qp Π_f det A^(f)_qp"""
function parton_calculate_ip(amp::PartonAmplitudeData, qp_weight)
    ip = zero(ComplexF64)
    for qp in 1:amp.n_qp
        p = one(ComplexF64)
        for f in 1:amp.n_flavor
            p *= amp.det_a[block_index(amp, qp, f)]
        end
        ip += qp_weight[qp] * p
    end
    return ip
end