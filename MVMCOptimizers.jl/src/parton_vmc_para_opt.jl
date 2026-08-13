"""
パートンモードの SR オーケストレータ
--- parton-mode (fork addition) ---

DESIGN_parton.md §3.3 の「配線」行に対応する。既存機構(weight_average /
stochastic_opt! / output_data! / store_opt_data! / bcast_scalar /
reduce_counter!)はそのまま呼び、パートン固有の部分だけを自前で持つ。
"""

"""
    parton_sync_parameters!(data, ctx)

rank0 のパラメータを comm0 で bcast し、全 rank を同じ値に揃える。

既存の `sync_modified_parameter!` は使わない。あちらは bcast のあとに
相関因子のシフト、`D_AmpMax` によるスレーターパラメータのリスケール、
OptTrans の正規化を行うが、平均場パラメータ α にリスケールを掛けると
H(α) が別のハミルトニアンになってしまう(DESIGN の絶対規則: MF パラメータに
D_AmpMax を適用しない)。パートンモードでは相関因子も OptTrans も無いので、
必要なのは bcast だけ。

serial 実行では何もしない(既存の作法と同じく `ctx.is_mpi` で分岐)。
"""
function parton_sync_parameters!(
    data::ExpertModeData,
    ctx::ParallelContext,
    mfham::Union{PartonMFHamiltonian,Nothing} = nothing,
)
    if ctx.is_mpi
        para = pack_parameters(data)
        bcast!(ctx, para; root = 0, which = :comm0)
        unpack_parameters!(data, para)
    end
    # ゲージ射影は bcast の**後**。α から決定論的に決まるので追加通信は要らず、
    # 全ランクが同じ結果になる。
    if mfham !== nothing && data.modpara.parton_gauge_fix != 0
        parton_project_gauge!(data, mfham)
    end
    return nothing
end

"""
    parton_project_gauge!(data, mfham; scale_floor=1e-12)

α をゲージスライスへ引き戻す(DESIGN §2.5)。

α には Ψ を変えない連続自由度がある。厳密演算なら S の固有値も力ベクトルの成分も
ゼロなので SR はそちらへ動かないが、実際には **MC ノイズが力に偽の成分を与え、
正則化 ε 付きの S⁻¹ がそれを 1/ε 倍する**ため α が際限なく漂流する。毎ステップの
同期時に掃き出す。

位置づけは既存 mVMC の `D_AmpMax`(スレーター振幅の全体スケールを上限へ再スケール)と
同じ「更新後にゲージスライスへ引き戻す」機構。OptFlag による成分凍結はゲージ目的では
使わない — 最適解が α_rep = 0 のときスライスに到達できないという失敗モードがある。

- スケール群: ノルムを初期値へ戻す(実数正倍なので Φ は不変)
- シフト群: 一様成分を引く(H → H + μI は Φ を変えない)
- 射影は α にのみ作用する。射影因子(Gutzwiller / Jastrow)には触れない
- 書き戻しはパラメータロケータ経由(α の正準置き場は `pmfpara_terms[].value`)
"""
function parton_project_gauge!(
    data::ExpertModeData,
    mfham::PartonMFHamiltonian;
    scale_floor::Float64 = 1e-12,
)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    para = pack_parameters(data)
    off = n_proj                      # MF ブロックは射影ブロックの後ろ

    changed = false
    for (gi, grp) in enumerate(mfham.gauge_scale_groups)
        target = mfham.gauge_target_norm[gi]
        target > scale_floor || continue
        nrm = sqrt(sum(k -> abs2(para[off + k]), grp))
        if nrm <= scale_floor
            @warn "Gauge scale group has collapsed; skipping the projection" group = gi norm = nrm
            continue
        end
        c = target / nrm
        isapprox(c, 1.0; rtol = 1e-15) && continue
        for k in grp
            para[off + k] *= c        # c は正の実数 → Φ は不変
        end
        changed = true
    end

    for grp in mfham.gauge_shift_groups
        isempty(grp) && continue
        m = sum(k -> para[off + k], grp) / length(grp)
        abs(m) <= scale_floor && continue
        for k in grp
            para[off + k] -= m
        end
        changed = true
    end

    changed && unpack_parameters!(data, para)
    return nothing
end

"""
    parton_materialize_flags!(data) -> Vector{Bool}

OptFlag 配列を実体化する(DESIGN §2.5)。門番より前に呼ぶこと:
`stochastic_opt!` は範囲外のフラグ添字を黙って「凍結」と読むので、配列が
短いまま走ると SR が平均場ブロックを丸ごと無視する。門番はこの関数の結果を
検査する側であって、作る側ではない。

**OptFlag の用途はエルミート性とユーザーの明示的固定に限る**(v3.2)。ゲージ平坦
方向を潰すのは `parton_project_gauge!` の仕事で、OptFlag による成分凍結は使わない
— 最適解が α_rep = 0 のときスライスに到達できないという失敗モードがあるため。

順序:
1. `fill(true, 2 * (n_proj + n_idx))` で全可動に初期化
2. pmfpara.def の末尾フラグ行を適用(ユーザーの明示的固定。実部・虚部の両スロットへ)
3. オンサイト群の虚部を強制凍結(エルミート性。t が実数なので Im は非物理)

オンサイト群の判定は `parton_onsite_idx_set` に切り出してあり、契約 0 の
テンプレート build と同じ結合規則を使う(判定の二重実装を避けるため)。
"""
function parton_materialize_flags!(data::ExpertModeData)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    n_idx = parton_n_idx(data)

    flags = fill(true, 2 * (n_proj + n_idx))

    for (idx, flag) in data.pmfpara_opt_flags
        0 <= idx < n_idx || continue
        base = 2 * (n_proj + idx)
        flags[base + 1] = flag != 0
        flags[base + 2] = flag != 0
    end

    for idx in parton_onsite_idx_set(data)
        flags[2 * (n_proj + idx) + 2] = false   # オンサイト群の Im は非物理
    end

    data.optimization_flags = flags
    return flags
end

"""
    parton_onsite_idx_set(data) -> Set{Int}

オンサイト群に属する idx(0-based)の集合。pmftrans の `site1 == site2` な行を
pmfpara の結合キーで引いて決める。契約 0 のテンプレート build と同じ結合規則。
"""
function parton_onsite_idx_set(data::ExpertModeData)
    idx_of = Dict{NTuple{3,Int},Int}()
    for p in data.pmfpara_terms
        idx_of[(p.site1, p.site2, p.flavor1)] = p.idx
    end
    onsite = Set{Int}()
    for t in data.pmftrans_terms
        t.site1 == t.site2 || continue
        key = (t.site1, t.site2, t.flavor1)
        haskey(idx_of, key) && push!(onsite, idx_of[key])
    end
    return onsite
end

"""
    parton_vmc_para_opt!(pstate, data, ctx; rng, output_dir, c_timer) -> Int

SR ループ本体。各ステップは

    契約0 (H(α) 組立 → 対角化 → Φ)
    契約0′ (摂動論で ∂Φ)
    骨格 + 契約2,3 (サンプリング)
    契約4,5 (E_loc と O)
    weight_average → reduce_counter → output_data
    stochastic_opt! (案 B により MF ブロックも解かれる)
    parton_sync_parameters! (bcast のみ)
    store_opt_data!

の順。`stochastic_opt!` の info は既存の作法どおり bcast してから判定する
(rank-local な info で early return すると comm0 の collective が不整合になり
hang するため)。
"""
function parton_vmc_para_opt!(
    pstate::PartonOptimizationState,
    data::ExpertModeData,
    ctx::ParallelContext;
    rng,
    output_dir::Union{String,Nothing} = nothing,
    c_timer::CTimer = CTIMER_DISABLED,
    write_diagnostics::Bool = true,
)::Int
    validate_parton_inputs(data, ctx)

    mp = data.modpara
    out_rank = is_output_rank(ctx)
    diag_dir = (write_diagnostics && out_rank) ? output_dir : nothing
    t_run0 = time()
    n_elec = mp.nelec
    n_steps = mp.nsr_opt_itr_step
    n_smp = mp.nsr_opt_itr_smp
    mfham = pstate.mfham

    # 既存の SRinfo writer は追記で、ヘッダは「ファイルが無いか空」のときだけ書く。
    # 同じ出力先で回し直したときに前 run の行が残らないよう、開始前に消しておく
    # (既存 writer には触らない — 消すのは呼び出し側の責務)。
    if diag_dir !== nothing
        srinfo_path = _parton_out(data, "_SRinfo.dat", diag_dir)
        isfile(srinfo_path) && rm(srinfo_path)
    end

    ctimer_start!(c_timer, 800)
    for step = 0:(n_steps - 1)
        t_step0 = time()
        α = parton_alpha_from_terms(data)
        ctimer_start!(c_timer, 801)
        parton_update_orbitals!(mfham, α, n_elec)             # 契約 0
        ctimer_stop!(c_timer, 801)
        ctimer_start!(c_timer, 802)
        parton_update_orbital_derivatives!(mfham, n_elec)     # 契約 0′
        ctimer_stop!(c_timer, 802)

        # n_out はサンプリングが burn_flag を立てる前に読む(time の記録用。
        # 初回だけ WarmUp+Sample で長い理由がファイルから追える)
        n_out_step = parton_n_out(pstate.config, mp)
        ctimer_start!(c_timer, 803)
        parton_make_sample!(pstate, data, rng; c_timer = c_timer)   # 骨格 + 契約 2, 3
        ctimer_stop!(c_timer, 803)
        ctimer_start!(c_timer, 807)
        parton_main_cal!(pstate, data; c_timer = c_timer)           # 契約 4, 5
        ctimer_stop!(c_timer, 807)

        weight_average_we!(ctx, pstate.state)
        weight_average_sr_opt!(ctx, pstate.state)
        reduce_counter!(ctx, pstate.config.counter)

        ctimer_start!(c_timer, 813)
        is_output_rank(ctx) &&
            output_data!(data, pstate.state, step; output_dir = output_dir)
        ctimer_stop!(c_timer, 813)

        ctimer_start!(c_timer, 811)
        info = stochastic_opt!(data, pstate.state, c_timer;
                               write_srinfo = diag_dir !== nothing,
                               srinfo_dir = diag_dir, srinfo_iter = step)
        ctimer_stop!(c_timer, 811)
        info = Int(bcast_scalar(ctx, info))
        if info != 0
            is_output_rank(ctx) &&
                @error "Parton SR: stochastic optimization failed" info step
            return info
        end

        norm_pre = sqrt(sum(abs2, parton_alpha_from_terms(data)))
        ctimer_start!(c_timer, 812)
        parton_sync_parameters!(data, ctx, mfham)
        ctimer_stop!(c_timer, 812)
        ctimer_start!(c_timer, 813)
        if diag_dir !== nothing
            norm_post = sqrt(sum(abs2, parton_alpha_from_terms(data)))
            parton_write_diag(data, pstate, step, diag_dir;
                              n_recompute = pstate.config.counter[4],
                              n_need_recompute = pstate.config.counter[3],
                              alpha_norm_pre = norm_pre,
                              alpha_norm_post = norm_post)
            t_now = time()
            parton_write_time(data, step, diag_dir, t_now - t_step0, t_now - t_run0;
                              n_out = n_out_step, n_rank = ctx.size0)
        end
        ctimer_stop!(c_timer, 813)

        if step >= n_steps - n_smp
            store_opt_data!(data, pstate.state, step - (n_steps - n_smp))
        end
    end

    ctimer_stop!(c_timer, 800)

    # 最適化された α を永続化する(zqp_opt.dat)。これを呼ばないと SR の結果が
    # どこにも残らない。data_io.jl の登録点で pmfpara_terms も書かれる。
    is_output_rank(ctx) && output_opt_data!(data; output_dir = output_dir)

    # SR 終了後に 1 回: 平均場ハミルトニアン/バンドと収束テーブル
    if diag_dir !== nothing
        # SR ループ内の最後の parton_update_orbitals! は「最終更新**前**」の α を使って
        # いるので、ダンプ前に最終 α で組み直す。そうしないと α* と H が食い違う。
        parton_update_orbitals!(mfham, parton_alpha_from_terms(data), n_elec)
        parton_write_mfham(data, mfham, diag_dir)
        parton_write_conv(data, diag_dir)
        # CalcTimer は既存 writer が本体セクションを "w" で書いた後に追記する
        write_ctimer_para_opt(c_timer, String(diag_dir);
                              prefix = isempty(data.modpara.c_data_file_head) ?
                                       "zvo" : data.modpara.c_data_file_head)
        parton_write_ctimer(data, c_timer, diag_dir)
    end

    return 0
end
