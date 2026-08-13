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

    # 2. OptFlag の実体化(門番より前)
    parton_materialize_flags!(data)

    # 3. MPI コンテキスト → 門番
    ctx = build_parallel_context(data.modpara.nsplit_size)
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

    # 4. 乱数(既存の C 準拠の seed 解決をそのまま借りる)
    rng = SFMT19937RNG()
    Random.seed!(rng, resolve_rnd_seed(ctx, mp.rnd_seed, seed))

    # 5. 量子数射影。qptransidx.def が無ければ恒等 QP を実体化する
    parton_ensure_qp!(data)

    # 6. 状態を確保して SR を回す
    pstate = parton_build_optimization_state(data)
    mkpath(output_dir)
    return parton_vmc_para_opt!(
        pstate,
        data,
        ctx;
        rng = rng,
        output_dir = String(output_dir),
    )
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
