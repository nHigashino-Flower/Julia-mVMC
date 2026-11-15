"""
Φ 直接変分モードの SR ループとドライバ
--- parton-mode (fork addition) ---

DESIGN_parton.md §12.11(上流型の簡素版)。SR が動かすのは `data.pmforb_params`
(Φ の全要素)そのもので、正準化(Löwdin)はしない。冗長性の処理は上流と同じく
スケールだけで、SR 更新のたびにフレーバーごとに max|Φ_f| = PartonOrbAmpMax へ
そろえる(`parton_orbparam_rescale!`)。GL(Ne) のうちスケール以外の方向は SR の
正則化(DSROptStaDel)任せで、その漂流は `<head>_parton_orbdiag.dat` の
log10 cond(Φ_f) に出る。
"""

"""
    parton_orbparam_build_state(data, map) -> PartonOptimizationState{PartonDirectPhi}

Φ モードの状態一式を確保する。寸法は `parton_build_optimization_state` と同じで、
軌道の器だけが `PartonDirectPhi`。det^F 高速路は使わない(n_stored = NFlavor)。
`data.pmforb_params` を読み込み済みで返す。
"""
function parton_orbparam_build_state(data::ExpertModeData, map::PartonOrbIndexMap)
    mp = data.modpara
    n_qp = get_n_qp_full(data)
    n_para = MVMCExpertModeParsers.count_variational_parameters(data)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    state = VMCOptimizationState(
        mp.nsite, mp.nelec, n_proj, n_para, n_qp, mp.nvmc_sample,
        true,     # all_complex: パートンモードは ComplexType = 1 のみ
        false,    # use_fsz
    )
    phi = PartonDirectPhi(mp.nsite, mp.nelec, mp.nflavor, map)
    parton_orbparam_load_orbitals!(phi, data)
    return PartonOptimizationState(
        state,
        PartonAmplitudeData(n_qp, mp.nflavor, mp.nelec; n_stored = mp.nflavor),
        PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, mp.nvmc_sample;
                            n_proj = n_proj),
        PartonSamplingWorkspace(mp.nelec, n_qp * mp.nflavor),
        phi,
        parton_build_phys_hamiltonian(data),
    )
end

# def が無いとき(恒等写像)
parton_orbparam_build_state(data::ExpertModeData) = parton_orbparam_build_state(
    data, parton_orbparam_identity_map(data.modpara.nsite, data.modpara.nelec,
                                       data.modpara.nflavor))

"""
    parton_orbparam_accept_update!(data, para_before) -> Bool

SR 更新後のパラメータに非有限値があれば `para_before` に戻して false を返す
(α モードの NaN ゲートと同じ趣旨)。再スケールより**前**に呼ぶこと
(NaN を含む Φ は再スケールで max が定義できない)。判定は bcast 後の値で行うので
全ランクが同じ決定をする。
"""
function parton_orbparam_accept_update!(data::ExpertModeData, para_before::Vector{ComplexF64})
    any(!isfinite, pack_parameters(data)) || return true
    unpack_parameters!(data, para_before)
    return false
end

"""
    parton_orbparam_write_orbdiag(data, pstate, step, dir; n_skipped, xmax_pre, log10_cond)

`<head>_parton_orbdiag.dat` に SR ステップごとに 1 行書く(rank 0 のみ、step 0 で "w")。
§12.11 の「致命的な問題」の判定材料:

- `n_skipped` — 捨てたサンプル数(このランク。ip が 0 か非有限、または E_loc が非有限。`parton_ip_usable`)。増え続けたら桁落ち
- `accept_ratio` / `n_need_recompute` / `n_exact_recompute` — MC の健康診断
  (α の `_parton_diag.dat` と同じ定義)
- `xmax_pre_g` — 再スケールの組 g ごとの、再スケール**前**の max|θ|(= D·(1 + SR 更新の
  相対的な大きさ))。PartonOrbIdx でフレーバー間の共有が無ければ組 g = フレーバー g
- `log10cond_f` — そのステップで使った Φ_f の log10(σ_max/σ_min)。単調に増えたら
  GL(Ne) のスケール以外の方向へ漂流している
"""
function parton_orbparam_write_orbdiag(
    data::ExpertModeData,
    pstate::PartonOptimizationState,
    step::Int,
    dir;
    n_skipped::Int,
    xmax_pre::Vector{Float64},
    log10_cond::Vector{Float64},
)
    dir === nothing && return nothing
    path = _parton_out(data, "_parton_orbdiag.dat", dir)
    counter = pstate.config.counter
    trials = counter[1]
    accepts = counter[2]
    acc = trials > 0 ? accepts / trials : 0.0
    # xmax は再スケールの組ごと、cond はフレーバーごとに値がある。制限が無ければ
    # 組 = フレーバーなので、ラベルは従来と同じ文字列になる
    n_group = length(xmax_pre)
    n_flavor = length(log10_cond)
    open(path, step == 0 ? "w" : "a") do f
        if step == 0
            println(f, "# step  n_skipped  accept_ratio  n_need_recompute  n_exact_recompute  " *
                       join(("xmax_pre_$g" for g = 1:n_group), "  ") * "  " *
                       join(("log10cond_$fl" for fl = 1:n_flavor), "  "))
        end
        @printf(f, "%6d %8d % .6e %8d %8d", step, n_skipped, acc, counter[3], counter[4])
        for x in xmax_pre
            @printf(f, " % .8e", x)
        end
        for c in log10_cond
            @printf(f, " % .6e", c)
        end
        println(f)
    end
    return path
end

"""
`zvo_parton_runinfo.dat` の末尾に Φ モードの項目を追記する。`n_orb` は Φ の要素数、
`NPmfOrbIdx` は変分パラメータ θ の個数(制限が無ければ n_orb と同じ)、`PartonOrbIdx` は
制限の def のパス(無ければ none)。
"""
function parton_orbparam_append_runinfo(data::ExpertModeData, dir;
                                        map::PartonOrbIndexMap, orbidx_path = nothing)
    dir === nothing && return nothing
    mp = data.modpara
    open(_parton_out(data, "_parton_runinfo.dat", dir), "a") do f
        @printf(f, "PartonOrbParam %d\n", mp.parton_orb_param)
        @printf(f, "PartonOrbAmpMax %.17g\n", mp.parton_orb_amp_max)
        @printf(f, "n_orb %d\n", parton_orbparam_n_orb(mp))
        # α の runinfo writer が同じファイルに "n_idx"(平均場の idx 数)を書くので、
        # θ の個数は def のヘッダと同じ名前 NPmfOrbIdx で書く(キーを重複させない)
        @printf(f, "NPmfOrbIdx %d\n", map.n_idx)
        @printf(f, "PartonOrbIdx %s\n", orbidx_path === nothing ? "none" : orbidx_path)
    end
    return nothing
end

"""
    parton_orbparam_vmc_para_opt!(pstate, data, ctx; rng, output_dir, c_timer, write_diagnostics) -> Int

Φ モードの SR ループ。1 ステップの順序(DESIGN §12.11):

1. `data.pmforb_params` → `phi.orbitals`(条件数は診断を書く run だけ計算)
2. サンプリング(骨格 + 契約 2, 3)→ 測定(契約 4, 5 の Φ 版)
3. weight_average → reduce_counter → output_data
4. stochastic_opt!(info は bcast してから判定)
5. bcast → NaN ゲート → **再スケール**(上流の sync_modified_parameter! と同じ位置)
6. 診断 → store_opt_data!

CalcTimer の番号は α と共通の 800 番台を使う(801/802 は Φ モードでは 0 のまま)。
"""
function parton_orbparam_vmc_para_opt!(
    pstate::PartonOptimizationState{PartonDirectPhi},
    data::ExpertModeData,
    ctx::ParallelContext;
    rng,
    output_dir::Union{String,Nothing} = nothing,
    c_timer::CTimer = CTIMER_DISABLED,
    write_diagnostics::Bool = true,
)::Int
    validate_parton_orbparam_inputs(data, ctx, pstate.mfham.map)

    mp = data.modpara
    out_rank = is_output_rank(ctx)
    diag_dir = (write_diagnostics && out_rank) ? output_dir : nothing
    t_run0 = time()
    n_steps = mp.nsr_opt_itr_step
    n_smp = mp.nsr_opt_itr_smp
    phi = pstate.mfham

    # 既存の SRinfo writer は追記なので、同じ出力先で回し直したときに前 run の行が
    # 残らないよう先に消す(α と同じ)
    if diag_dir !== nothing
        srinfo_path = _parton_out(data, "_SRinfo.dat", diag_dir)
        isfile(srinfo_path) && rm(srinfo_path)
    end

    ctimer_start!(c_timer, 800)
    for step = 0:(n_steps - 1)
        t_step0 = time()
        parton_orbparam_load_orbitals!(phi, data)
        log10_cond = diag_dir === nothing ? Float64[] :
                     [(σ = svdvals(Φ); log10(σ[1] / σ[end])) for Φ in phi.orbitals]

        n_out_step = parton_n_out(pstate.config, mp)
        ctimer_start!(c_timer, 803)
        parton_make_sample!(pstate, data, rng; c_timer = c_timer)
        ctimer_stop!(c_timer, 803)
        ctimer_start!(c_timer, 807)
        parton_main_cal!(pstate, data; c_timer = c_timer)
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
            out_rank && @error "Parton SR (direct orbitals): stochastic optimization failed" info step
            return info
        end

        ctimer_start!(c_timer, 812)
        parton_sync_parameters!(data, ctx)          # bcast のみ
        ctimer_stop!(c_timer, 812)
        if !parton_orbparam_accept_update!(data, para_before)
            out_rank && @warn "Parton SR (direct orbitals): non-finite parameter update rejected" step
        end
        xmax_pre = parton_orbparam_rescale!(data, phi.map)   # 上流の D_AmpMax と同じ位置

        ctimer_start!(c_timer, 813)
        if diag_dir !== nothing
            parton_orbparam_write_orbdiag(data, pstate, step, diag_dir;
                                          n_skipped = n_skipped, xmax_pre = xmax_pre,
                                          log10_cond = log10_cond)
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

    # Jastrow の最適値(zqp_opt.dat ほか)。Φ は data_io の writer に登録していない
    # (列が巨大になるため)ので、専用ファイルに書く
    out_rank && output_opt_data!(data; output_dir = output_dir)
    parton_orbparam_load_orbitals!(phi, data)       # 最終値(再スケール後)に合わせる
    # 最終 Φ は結果そのもの(zqp_opt.dat には載らない)なので、診断の有無に関係なく残す
    out_rank && output_dir !== nothing &&
        parton_orbparam_write_pmforb(data, _parton_para_out(data, "_pmforb_opt.dat", output_dir))
    if diag_dir !== nothing
        parton_write_conv(data, diag_dir)
        write_ctimer_para_opt(c_timer, String(diag_dir);
                              prefix = isempty(mp.c_data_file_head) ? "zvo" : mp.c_data_file_head)
        parton_write_ctimer(data, c_timer, diag_dir)
    end
    return 0
end

"""
    parton_orbparam_run_from_namelist(data, namelist_path, ctx; nsteps, nsmp, output_dir, seed) -> Int

Φ モードのドライバ本体。`parton_run_para_opt_from_namelist` がパース直後に
`PartonOrbParam ≠ 0` を見てここへ分岐する(α の初期化・フラグ・門番より前)。

順序(DESIGN §12.12): modpara の検査 → PartonOrbIdx を読む(無ければ恒等写像)→
θ の乱数初期化(初期化専用の乱数列)→ InPmfOrb があれば上書き → 再スケール(上流が
InitParameter → In* → SyncModifiedParameter と進むのと同じ)→ α の入力が無いことの確認 →
OptFlag の実体化 → 門番 → サンプリング RNG → QP → 状態 → 列フルランクの検査 →
`_pmforb_init.dat` → runinfo → SR ループ。
"""
function parton_orbparam_run_from_namelist(
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
    mp.parton_orb_param == PARTON_ORB_PARAM_THOULESS && error(
        "PartonOrbParam = 2 (Thouless parametrization) is reserved and not implemented " *
        "yet. Use PartonOrbParam = 1 (direct orbitals).")
    # Φ の寸法(NFlavor·NSite·NElec)と再スケールの値を使う前に modpara を検査する。
    # 後に回すと、NFlavor の書き忘れが「idx が範囲外」という的外れなエラーになる
    validate_parton_modpara(mp)
    (isfinite(mp.parton_orb_amp_max) && mp.parton_orb_amp_max > 0) || error(
        "PartonOrbAmpMax must be a positive finite number, got $(mp.parton_orb_amp_max).")

    # 制限の def(DESIGN §12.12)。無ければ恒等写像(各要素が独立、§12.11 と同じ)
    map = parton_orbparam_read_orbidx(data, namelist_str)
    orbidx_path = map === nothing ? nothing :
        _parton_namelist_entry(namelist_str, "PartonOrbIdx")
    map === nothing && (map = parton_orbparam_identity_map(mp.nsite, mp.nelec, mp.nflavor))

    # 初期値: 乱数(初期化専用の乱数列)→ InPmfOrb があれば上書き → 再スケール
    # (上流が InitParameter → In* → SyncModifiedParameter と進むのと同じ順)
    sampling_seed = resolve_rnd_seed(ctx, mp.rnd_seed, seed)
    base_seed = sampling_seed - ctx.group1
    parton_orbparam_init_random!(data, map, base_seed)
    in_pmforb = parton_orbparam_read_in_pmforb!(data, namelist_str, map)
    parton_orbparam_rescale!(data, map)
    parton_orbparam_reject_alpha_files(namelist_str)
    parton_orbparam_materialize_flags!(data, map)
    validate_parton_orbparam_inputs(data, ctx, map)
    # 物理量測定(DESIGN §13)の門番。NVMCCalMode = 0 では Green の def に warn するだけ
    validate_parton_green_inputs(data)
    mp.vmc_calc_mode == 1 && parton_phys_require_fixed_inputs(
        data, namelist_str; orbital_given = in_pmforb !== nothing,
        orbital_hint = "add InPmfOrb (use $(mp.c_para_file_head)_pmforb_opt.dat " *
                       "of the optimization run)")
    is_output_rank(ctx) && parton_warn_threading_config()

    nsteps !== nothing && (mp.nsr_opt_itr_step = Int(nsteps))
    nsmp !== nothing && (mp.nsr_opt_itr_smp = Int(nsmp))
    mp.nsr_opt_itr_step >= mp.nsr_opt_itr_smp || throw(
        ArgumentError(
            "NSROptItrStep ($(mp.nsr_opt_itr_step)) must be >= NSROptItrSmp " *
            "($(mp.nsr_opt_itr_smp)); a smaller step count would zero-pad the " *
            "optimisation averages.",
        ),
    )

    # サンプリング用の乱数(α と同じ C 準拠の seed 解決)。θ の乱数初期化とは別の列
    rng = SFMT19937RNG()
    Random.seed!(rng, sampling_seed)

    parton_ensure_qp!(data)
    pstate = parton_orbparam_build_state(data, map)
    parton_orbparam_check_rank(pstate.mfham)
    mkpath(output_dir)
    out_dir = String(output_dir)
    is_output_rank(ctx) &&
        parton_orbparam_write_pmforb(data, _parton_para_out(data, "_pmforb_init.dat", out_dir))

    t_start = time()
    n_para = MVMCExpertModeParsers.count_variational_parameters(data)
    if is_output_rank(ctx)
        parton_write_runinfo(data, out_dir; namelist_path = namelist_str, base_seed = base_seed,
                             n_idx = 0, n_para = n_para, n_rank = ctx.size0,
                             t_start = t_start, t_end = t_start)
        parton_orbparam_append_runinfo(data, out_dir; map = map, orbidx_path = orbidx_path)
    end

    # 物理量測定(DESIGN §13): Φ は build_state が θ から組み済み。SR ループの代わりに測定へ
    if mp.vmc_calc_mode == 1
        status = parton_vmc_phys_cal!(pstate, data, ctx; rng = rng, output_dir = out_dir)
        if is_output_rank(ctx)
            parton_write_runinfo(data, out_dir; namelist_path = namelist_str,
                                 base_seed = base_seed, n_idx = 0, n_para = n_para,
                                 n_rank = ctx.size0, t_start = t_start, t_end = time())
            parton_orbparam_append_runinfo(data, out_dir; map = map, orbidx_path = orbidx_path)
        end
        return status
    end

    status = parton_orbparam_vmc_para_opt!(
        pstate, data, ctx;
        rng = rng,
        c_timer = CTimer(get(ENV, "MVMC_C_TIMER", "1") != "0"),
        output_dir = out_dir,
    )

    if is_output_rank(ctx)
        parton_write_runinfo(data, out_dir; namelist_path = namelist_str, base_seed = base_seed,
                             n_idx = 0, n_para = n_para, n_rank = ctx.size0,
                             t_start = t_start, t_end = time())
        parton_orbparam_append_runinfo(data, out_dir; map = map, orbidx_path = orbidx_path)
    end
    return status
end

"namelist.def の `key` 行が指すファイルの絶対パス(無ければ nothing)。runinfo の記録用。"
function _parton_namelist_entry(namelist_path::AbstractString, key::AbstractString)
    base_dir = dirname(abspath(String(namelist_path)))
    for (file_type, file_path) in MVMCExpertModeParsers.parse_namelist_content(
            MVMCExpertModeParsers.read_def_file(String(namelist_path)))
        file_type == key && return joinpath(base_dir, file_path)
    end
    return nothing
end
