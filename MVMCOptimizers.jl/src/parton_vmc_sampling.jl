"""
骨格: 固縛Metropolisサンプリング。契約0(軌道)は呼び出し前に済んでいること。
更新は固縛ホッピング一択(NExUpdatePath=6を門番が保証)なので、
既存のget_update_type相当の分岐は存在しない。
"""
function parton_make_sample!(pstate::PartonOptimizationState,
                             data::ExpertModeData, rng)
    amp, cfg, mfham, ws = pstate.parton_amp_data, pstate.parton_config,
                          pstate.parton_mfhamiltonian, pstate.parton_workspace
    mp = data.modpara
    n_site = mp.nsite

    # --- 開始配置: burn再利用 or 初期生成 -------------------------------
    if cfg.burn_flag
        parton_copy_from_burn_sample!(cfg)
        n_out = mp.nvmc_sample + 1
    else
        parton_make_initial_sample!(cfg, amp, mfham, data, ws, rng)
        n_out = mp.nvmc_warmup + mp.nvmc_sample
    end
    parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)   # 最初の錨
    n_in = mp.nvmc_interval * n_site

    n_accept_anchor = 0
    for out_step in 1:n_out
        for in_step in 1:n_in
            cfg.counter[1] += 1                                   # 試行数
            m, r_old, r_new, ok = parton_make_candidate_hopping(rng, cfg, n_site)
            ok || continue                                        # 占有先/同一サイト → 棄却(試行には数える)

            log_pr = parton_log_proj_ratio(cfg, m, r_old, r_new)  # 射影フック(M1初点火は0)
            ratio, _ = parton_amplitude_ratio!(ws, amp, mfham, data,
                                               data.qp_weights, m, r_new)
            rand(rng) < exp(2 * log_pr) * abs2(ratio) || continue

            # このループで一番大事な不変条件は①→②の順序です。 受理が確定したら先に配置をコミットし、それから振幅を更新する。
            #こうしておくと、契約3が途中の (qp,f) ブロックで :need_recompute を返して部分更新のまま抜けても安全です
            # ——直後の全再計算はコミット済みの配置から全ブロックを組み直すので、半端な状態は跡形もなく上書きされる。
            # 逆順(振幅→配置)だと、再計算が古い配置を読んで受理済みの移動が消える、という追いにくいバグになります。
            cfg.counter[2] += 1                                   # 受理数
            parton_update_ele_config!(cfg, m, r_old, r_new)       # ① 配置を先に確定
            st = parton_update_amplitude!(amp, mfham, data, ws, m, r_new)  # ② 高速更新
            n_accept_anchor += 1
            if st === :need_recompute || n_accept_anchor >= mp.nblock_update_size
                parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)  # 錨の打ち直し
                n_accept_anchor = 0
            end
        end
        s = out_step - (n_out - mp.nvmc_sample)
        s >= 1 && parton_store_sample!(cfg, amp, data.qp_weights, s)
    end
    parton_copy_to_burn_sample!(cfg)
    cfg.burn_flag = true
    return nothing
end

"""提案: 物理粒子を選び、行き先を一様に選ぶ。対称提案なのでMetropolis補正は不要。"""
function parton_make_candidate_hopping(rng, cfg, n_site)
    m     = rand(rng, 1:cfg.n_elec)
    r_old = particle_site(cfg, 1, m)          # 固縛によりフレーバー1が代表
    r_new = rand(rng, 1:n_site)
    ok = (r_new != r_old) && !is_occupied(cfg, r_new)   # 占有判定も1系統でよい
    return m, r_old, r_new, ok
end

"""配置更新: 全フレーバー同時。固縛不変条件を守る唯一の書き込み経路。"""
function parton_update_ele_config!(cfg, m, r_old, r_new)
    for f in 1:cfg.n_flavor
        move_particle!(cfg, f, m, r_old, r_new)   # ele_idx / ele_cfg / ele_num の3点更新
    end
    return nothing
end

"""
契約2: 固縛移動(粒子m: r→r′)の振幅比。O(n_qp·F·Ne)。
状態は変更しない(書くのは ws のみ)→ 棄却時の revert が不要。
各ブロックの R は ws.ratio_blocks に残し、受理時に契約3が再利用する。
"""
function parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m::Int, r_new::Int)
    ip_old = zero(ComplexF64); ip_new = zero(ComplexF64)
    for qp in 1:amp.n_qp
        rr = data.qp_trans[qp][r_new]
        s  = data.qp_trans_sgn[qp][r_new]
        p_old = one(ComplexF64); p_new = one(ComplexF64)
        for f in 1:amp.n_flavor
            b    = block_index(amp, qp, f)
            Ainv = inv_block(amp, qp, f)
            Φ    = mfham.orbitals[f]
            R = zero(ComplexF64)
            @inbounds for n in 1:amp.n_elec
                R += Φ[rr, n] * Ainv[n, m]      # ★転置積。dot()は使わない(共役が入る)
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
契約3: 受理された固縛移動を全 (qp,f) ブロックへ反映。O(n_qp·F·Ne²)。
直前の契約2が同じ (m, r_new) で呼ばれていることが前提(ws.ratio_blocks を再利用)。
配置側の更新(ele_idx/ele_cfg/proj_cnt)は parton_update_ele_config! の担当で、ここでは触らない。
"""
function parton_update_amplitude!(amp, mfham, data, ws, m::Int, r_new::Int;
                                  ratio_floor = 1e-12)
    Ne = amp.n_elec
    for qp in 1:amp.n_qp, f in 1:amp.n_flavor
        b = block_index(amp, qp, f)
        R = ws.ratio_blocks[b]
        abs(R) < ratio_floor && return :need_recompute   # ノード踏み抜き→錨を打ち直せ
        Ainv = inv_block(amp, qp, f)
        rr = data.qp_trans[qp][r_new]
        s  = data.qp_trans_sgn[qp][r_new]
        @views ws.u_buf .= s .* mfham.orbitals[f][rr, :]
        mul!(ws.v_buf, transpose(Ainv), ws.u_buf)        # v[j] = Σ_n u[n]·Ainv[n,j]
        @views ws.col_buf .= Ainv[:, m]                  # 旧列mを退避
        invR = inv(R)
        @inbounds for j in 1:Ne
            j == m && continue
            axpy!(-ws.v_buf[j] * invR, ws.col_buf, @view Ainv[:, j])
        end
        @views Ainv[:, m] .= ws.col_buf .* invR
        amp.det_a[b] *= R
    end
    return :ok
end