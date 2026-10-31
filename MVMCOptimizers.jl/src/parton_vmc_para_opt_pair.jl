"""
Pfaffian パートンモード(PartonMode = 2)の状態・SR ループ・ドライバ
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §4.5。SR が動かすのは `data.parton_pair_params`(idx ごとの p)そのもので、
冗長性の処理は上流と同じく全体のスケールだけ(SR 更新のたびに D_AmpMax で max|p| = 4 にそろえる)。
サンプリング骨格・E_loc・測定は α・Φ と共有し、振幅と O は器の型でディスパッチされる。
"""

"""
    parton_pair_build_state(data) -> PartonOptimizationState{PartonPairOrbital}

Pf モードの状態一式を確保する。寸法は α・Φ と同じ規約(`VMCOptimizationState` は 1 + n_para)。
テンプレートを組み、現在の `data.parton_pair_params` を読み込んだ器で返す。
"""
function parton_pair_build_state(data::ExpertModeData)
    mp = data.modpara
    n_qp = get_n_qp_full(data)
    n_para = MVMCExpertModeParsers.count_variational_parameters(data)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    state = VMCOptimizationState(
        mp.nsite, mp.nelec, n_proj, n_para, n_qp, mp.nvmc_sample,
        true,     # all_complex: パートンモードは ComplexType = 1 のみ
        false,    # use_fsz
    )
    return PartonOptimizationState(
        state,
        PartonPairAmplitudeData(n_qp, mp.nflavor, mp.nelec),
        PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, mp.nvmc_sample; n_proj = n_proj),
        PartonPairSamplingWorkspace(n_qp, mp.nflavor, mp.nelec),
        parton_pair_build_source(data),
        parton_build_phys_hamiltonian(data),
    )
end

"""
    parton_pair_accept_update!(data, para_before) -> Bool

SR 更新後のパラメータに非有限値があれば `para_before` に戻して false を返す(α・Φ の NaN ゲートと
同じ趣旨)。D_AmpMax の再スケールより**前**に呼ぶこと(NaN を含む p は max が定義できない)。
判定は bcast 後の値で行うので、全ランクが同じ決定をする。
"""
function parton_pair_accept_update!(data::ExpertModeData, para_before::Vector{ComplexF64})
    any(!isfinite, pack_parameters(data)) || return true
    unpack_parameters!(data, para_before)
    return false
end

"""
    parton_pair_vmc_para_opt!(pstate, data, ctx; rng, output_dir, c_timer, write_diagnostics) -> Int

Pf モードの SR ループ。1 ステップの順序(DESIGN_parton_pf.md §4.5):

1. `data.parton_pair_params` → `pair_matrix`(診断を書く run では行ブロック SVD の σ_N/σ_{N+1} も)
2. サンプリング(骨格 + 契約 2, 3 の Pf 版)→ 測定(契約 4, 5。**逐次**: 第 1 段ではスレッド
   並列の測定経路を使わない。§4.1)
3. weight_average → reduce_counter → output_data
4. stochastic_opt!(info は bcast してから判定)
5. bcast → NaN ゲート → **D_AmpMax**(上流の sync_modified_parameter! と同じ位置)
6. 診断 → store_opt_data!

CalcTimer は α・Φ と共通の 800 番台(801/802 は 0 のまま)。
"""
function parton_pair_vmc_para_opt!(
    pstate::PartonOptimizationState{PartonPairOrbital},
    data::ExpertModeData,
    ctx::ParallelContext;
    rng,
    output_dir::Union{String,Nothing} = nothing,
    c_timer::CTimer = CTIMER_DISABLED,
    write_diagnostics::Bool = true,
)::Int
    validate_parton_pair_inputs(data, ctx)

    mp = data.modpara
    out_rank = is_output_rank(ctx)
    diag_dir = (write_diagnostics && out_rank) ? output_dir : nothing
    t_run0 = time()
    n_steps = mp.nsr_opt_itr_step
    n_smp = mp.nsr_opt_itr_smp
    src = pstate.mfham

    # 既存の SRinfo writer は追記なので、同じ出力先で回し直したときに前 run の行が残らないよう先に消す
    if diag_dir !== nothing
        srinfo_path = _parton_out(data, "_SRinfo.dat", diag_dir)
        isfile(srinfo_path) && rm(srinfo_path)
    end

    ctimer_start!(c_timer, 800)
    for step = 0:(n_steps - 1)
        t_step0 = time()
        parton_pair_load!(src, data.parton_pair_params)
        sigma_ratio = diag_dir === nothing ? Float64[] :
            [parton_pair_sigma_ratio(σ, mp.nelec) for σ in parton_pair_flavor_singular_values(src)]

        n_out_step = parton_n_out(pstate.config, mp)
        ctimer_start!(c_timer, 803)
        parton_make_sample!(pstate, data, rng; c_timer = c_timer)
        ctimer_stop!(c_timer, 803)
        ctimer_start!(c_timer, 807)
        parton_main_cal!(pstate, data; c_timer = c_timer, force_threaded = false)
        ctimer_stop!(c_timer, 807)
        # weight_average がランク間で合算する前に、このランクで捨てた数を読む
        n_skipped = mp.nvmc_sample - round(Int, real(pstate.state.energy.wc))

        weight_average_we!(ctx, pstate.state)
        weight_average_sr_opt!(ctx, pstate.state)
        reduce_counter!(ctx, pstate.config.counter)

        ctimer_start!(c_timer, 813)
        out_rank && output_data!(data, pstate.state, step; output_dir = output_dir)
        ctimer_stop!(c_timer, 813)

        para_before = pack_parameters(data)
        ctimer_start!(c_timer, 811)
        info = stochastic_opt!(data, pstate.state, c_timer;
                               write_srinfo = diag_dir !== nothing,
                               srinfo_dir = diag_dir, srinfo_iter = step)
        ctimer_stop!(c_timer, 811)
        info = Int(bcast_scalar(ctx, info))
        if info != 0
            out_rank && @error "Parton SR (Pfaffian): stochastic optimization failed" info step
            return info
        end

        ctimer_start!(c_timer, 812)
        parton_sync_parameters!(data, ctx)          # bcast のみ
        ctimer_stop!(c_timer, 812)
        if !parton_pair_accept_update!(data, para_before)
            out_rank && @warn "Parton SR (Pfaffian): non-finite parameter update rejected" step
        end
        xmax_pre = parton_pair_rescale!(data)        # 上流の D_AmpMax と同じ位置

        ctimer_start!(c_timer, 813)
        if diag_dir !== nothing
            parton_pair_write_diag(data, pstate, step, diag_dir;
                                   n_skipped = n_skipped, xmax_pre = xmax_pre,
                                   sigma_ratio = sigma_ratio)
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

    # Jastrow の最適値(zqp_opt.dat ほか)。p は data_io の writer に登録していない(列が巨大に
    # なりうるため)ので、専用ファイルに書く。最終 p は結果そのものなので診断の有無に関係なく残す
    out_rank && output_opt_data!(data; output_dir = output_dir)
    parton_pair_load!(src, data.parton_pair_params)  # 最終値(再スケール後)に合わせる
    if out_rank && output_dir !== nothing
        parton_pair_write_params(data, _parton_para_out(data, "_parton_pair_opt.dat", output_dir))
        parton_pair_write_svd(data, src, output_dir, "opt")     # 行ブロック SVD(§6)
    end
    if diag_dir !== nothing
        parton_write_conv(data, diag_dir)
        write_ctimer_para_opt(c_timer, String(diag_dir);
                              prefix = isempty(mp.c_data_file_head) ? "zvo" : mp.c_data_file_head)
        parton_write_ctimer(data, c_timer, diag_dir)
    end
    return 0
end

"""
    parton_pair_run_from_namelist(data, namelist_path, ctx; nsteps, nsmp, output_dir, seed) -> Int

Pf モードのドライバ本体。`parton_run_para_opt_from_namelist` がパース直後に `PartonMode = 2` を見て
ここへ分岐する(α の初期化・フラグ・門番より前)。順序(DESIGN_parton_pf.md §2.3):

modpara の検査 → α・Φ・上流 Orbital の混入の拒否 → OptFlag の実体化 →
InPartonPairOrbital があれば読む / 無ければ乱数(初期値専用の列)→ D_AmpMax → 門番 →
サンプリング RNG → QP → 状態 → `_parton_pair_init.dat` → runinfo → SR ループ。
"""
function parton_pair_run_from_namelist(
    data::ExpertModeData,
    namelist_path::AbstractString,
    ctx::ParallelContext;
    nsteps::Union{Integer,Nothing} = nothing,
    nsmp::Union{Integer,Nothing} = nothing,
    output_dir::AbstractString = tempname(),
    seed::Union{Integer,Nothing} = nothing,
)::Int
    namelist_str = String(namelist_path)
    mp = data.modpara
    # 寸法(NFlavor・NElec)を使う前に modpara を検査する(書き忘れを的外れなエラーにしない)
    validate_parton_modpara(mp; allowed_modes = (PARTON_MODE_PFAFFIAN,))
    parton_pair_reject_other_inputs(namelist_str)
    isempty(data.parton_pair_terms) && error(
        "PartonMode = 2 needs the pair orbitals: add a `PartonPairOrbital` line to $(namelist_str).")

    parton_pair_materialize_flags!(data)
    # 解決済みの基準 seed(ランクのずらしを加える前)。初期値の乱数はこれで専用の列を作るので、
    # 全ランクが同じ値を得て、サンプリング用の列は消費しない(α モードと同じ規律)
    base_seed = resolve_rnd_seed(ctx, mp.rnd_seed, seed) - ctx.group1
    in_path = parton_pair_read_in!(data, namelist_str)
    in_path === nothing && parton_pair_init_random!(data, base_seed)
    parton_pair_rescale!(data)                      # 上流が初期化直後に掛けるのと同じ
    validate_parton_pair_inputs(data, ctx)
    if is_output_rank(ctx)
        parton_warn_threading_config()
        parton_sample_threading_enabled() && @warn "PartonMode = 2 runs the measurement " *
            "phase sequentially for now; JULIA_MVMC_INNER_THREADS is ignored."
    end

    nsteps !== nothing && (mp.nsr_opt_itr_step = Int(nsteps))
    nsmp !== nothing && (mp.nsr_opt_itr_smp = Int(nsmp))
    mp.nsr_opt_itr_step >= mp.nsr_opt_itr_smp || throw(ArgumentError(
        "NSROptItrStep ($(mp.nsr_opt_itr_step)) must be >= NSROptItrSmp " *
        "($(mp.nsr_opt_itr_smp)); a smaller step count would zero-pad the optimisation averages."))

    # サンプリング用の乱数(α・Φ と同じ C 準拠の seed 解決)
    sampling_seed = resolve_rnd_seed(ctx, mp.rnd_seed, seed)
    rng = SFMT19937RNG()
    Random.seed!(rng, sampling_seed)

    parton_ensure_qp!(data)
    pstate = parton_pair_build_state(data)
    mkpath(output_dir)
    out_dir = String(output_dir)
    if is_output_rank(ctx)
        parton_pair_write_params(data, _parton_para_out(data, "_parton_pair_init.dat", out_dir))
        parton_pair_write_svd(data, pstate.mfham, out_dir, "init")   # 行ブロック SVD(§6)
    end

    init_source = in_path === nothing ? "random" : "InPartonPairOrbital"
    t_start = time()
    n_para = MVMCExpertModeParsers.count_variational_parameters(data)
    if is_output_rank(ctx)
        parton_write_runinfo(data, out_dir; namelist_path = namelist_str, base_seed = base_seed,
                             n_idx = 0, n_para = n_para, n_rank = ctx.size0,
                             t_start = t_start, t_end = t_start)
        parton_pair_append_runinfo(data, out_dir; init_source = init_source)
    end

    status = parton_pair_vmc_para_opt!(
        pstate, data, ctx;
        rng = rng,
        c_timer = CTimer(get(ENV, "MVMC_C_TIMER", "1") != "0"),
        output_dir = out_dir,
    )

    if is_output_rank(ctx)
        parton_write_runinfo(data, out_dir; namelist_path = namelist_str, base_seed = base_seed,
                             n_idx = 0, n_para = n_para, n_rank = ctx.size0,
                             t_start = t_start, t_end = time())
        parton_pair_append_runinfo(data, out_dir; init_source = init_source)
    end
    return status
end
