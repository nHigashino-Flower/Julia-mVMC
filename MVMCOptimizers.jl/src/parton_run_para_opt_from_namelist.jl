"""
パートンモードのドライバ(namelist.def から SR を回す入口)
--- parton-mode (fork addition) ---

既存の `run_para_opt_from_namelist` を縮約したもの。標準経路のパラメータ
初期化(init_parameter! / initial.def / In*.def の重ね合わせ / D_AmpMax の
リスケール)はパートンモードには当てはまらない: α の初期値は pmfpara.def が
持ち、リスケールは H(α) を別のハミルトニアンに変えてしまう。
"""

"""
    parton_run_para_opt_from_namelist(namelist_path; nsteps, nsmp, output_dir, seed) -> Int

namelist.def を読み、パートン平均場 VMC の SR を回す。

処理の順序で外せない点(DESIGN §2.5):
OptFlag 配列の実体化は**門番より前**。門番の仕事はその配列が正しい長さで
可動成分を持つことを検査することで、作ることではない。
"""
function parton_run_para_opt_from_namelist(
    namelist_path::AbstractString;
    nsteps::Union{Integer,Nothing} = nothing,
    nsmp::Union{Integer,Nothing} = nothing,
    output_dir::AbstractString = tempname(),
    seed::Union{Integer,Nothing} = nothing,
)::Int
    namelist_str = String(namelist_path)

    # 1. 入力のパース
    data = MVMCExpertModeParsers.parse_expert_mode_files(namelist_str)

    # 2. MPI コンテキスト(初期化のベースシード解決に要る)
    ctx = build_parallel_context(data.modpara.nsplit_size)

    # 3. α の初期値を確定する(DESIGN §2.3 の順序)。
    #    pmfpara.def の presence 判定 → 未入力 idx を乱数で生成 → InPmfPara.def で上書き。
    #    ベースシードはランクごとのオフセットを加える前の値なので、全ランクが
    #    構成的に同一の α を得る(bcast 不要)。サンプリング用 RNG は消費しない。
    base_seed = resolve_rnd_seed(ctx, data.modpara.rnd_seed, seed) - ctx.group1
    parton_init_alpha!(data, base_seed)
    parton_read_in_pmfpara!(data, namelist_str)

    # 4. OptFlag の実体化(門番より前)
    parton_materialize_flags!(data)

    # 5. 門番
    validate_parton_inputs(data, ctx)

    mp = data.modpara
    nsteps !== nothing && (mp.nsr_opt_itr_step = Int(nsteps))
    nsmp !== nothing && (mp.nsr_opt_itr_smp = Int(nsmp))
    mp.nsr_opt_itr_step >= mp.nsr_opt_itr_smp || throw(
        ArgumentError(
            "NSROptItrStep ($(mp.nsr_opt_itr_step)) must be >= NSROptItrSmp " *
            "($(mp.nsr_opt_itr_smp)); a smaller step count would zero-pad the " *
            "optimisation averages.",
        ),
    )

    # 6. サンプリング用の乱数(既存の C 準拠の seed 解決をそのまま借りる)。
    #    初期化用ストリームとは別物なので、乱数初期化の有無でサンプリング系列は変わらない。
    rng = SFMT19937RNG()
    Random.seed!(rng, resolve_rnd_seed(ctx, mp.rnd_seed, seed))

    # 7. 量子数射影。qptransidx.def が無ければ恒等 QP を実体化する
    parton_ensure_qp!(data)

    # 8. 状態を確保する。テンプレート build がここでゲージ射影の引き戻し先
    #    (gauge_target_norm)を確定するので、α の初期値がすべて決まった後でなければ
    #    ならない(乱数前の値を引き戻し先にしてしまう)。
    pstate = parton_build_optimization_state(data)
    mkpath(output_dir)

    # 9. 確定した初期 α をダンプする。ランタイム乱数を入れると「どの初期値で回したか」が
    #    ファイルに残らなくなるので、再現性の担保として In*.def 互換形式で残す。
    #    次回そのまま InPmfPara.def として渡せば厳密に再現・継続できる。
    is_output_rank(ctx) && parton_write_pmfpara(
        data, joinpath(output_dir, data.modpara.c_para_file_head * "_pmfpara_init.dat"))

    # 10. run メタデータ(出自の追跡用)。base_seed は乱数初期化の再現に要る値。
    t_start = time()
    is_output_rank(ctx) && parton_write_runinfo(
        data, String(output_dir); namelist_path = namelist_str, base_seed = base_seed,
        n_idx = parton_n_idx(data),
        n_para = MVMCExpertModeParsers.count_variational_parameters(data),
        n_rank = ctx.size0, t_start = t_start, t_end = t_start)

    # CalcTimer はパートンモードでは**既定で有効**(v3.5)。既存モードは
    # `MVMC_C_TIMER=1` の opt-in のままで、既定は変えていない。
    # `MVMC_C_TIMER=0` を明示すればパートンでも切れる。
    status = parton_vmc_para_opt!(
        pstate,
        data,
        ctx;
        rng = rng,
        c_timer = CTimer(get(ENV, "MVMC_C_TIMER", "1") != "0"),
        output_dir = String(output_dir),
    )

    # 壁時計を確定させて書き直す
    is_output_rank(ctx) && parton_write_runinfo(
        data, String(output_dir); namelist_path = namelist_str, base_seed = base_seed,
        n_idx = parton_n_idx(data),
        n_para = MVMCExpertModeParsers.count_variational_parameters(data),
        n_rank = ctx.size0, t_start = t_start, t_end = time())
    return status
end

"""
    parton_ensure_qp!(data)

QP 写像と重みを用意する。qptransidx.def が無い入力では恒等写像 1 本
(n_qp = 1、重み 1)を実体化する。射影ありの入力ではパーサが埋めた
qp_trans / qp_trans_sgn をそのまま使う。
"""
function parton_ensure_qp!(data::ExpertModeData)
    n_site = data.modpara.nsite
    if isempty(data.qp_trans)
        data.qp_trans = [collect(1:n_site)]
        data.qp_trans_sgn = [ones(Int, n_site)]
        data.modpara.nmp_trans = 1
        data.para_qp_trans = ComplexF64[1]
    end
    MVMCExpertModeParsers.init_qp_weight!(data)
    return data
end
