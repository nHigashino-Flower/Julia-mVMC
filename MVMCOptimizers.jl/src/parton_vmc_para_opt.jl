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
function parton_sync_parameters!(data::ExpertModeData, ctx::ParallelContext)
    if ctx.is_mpi
        para = pack_parameters(data)
        bcast!(ctx, para; root = 0, which = :comm0)
        unpack_parameters!(data, para)
    end
    return nothing
end

"""
    parton_materialize_flags!(data) -> Vector{Bool}

OptFlag 配列を実体化する(DESIGN §2.5)。門番より前に呼ぶこと:
`stochastic_opt!` は範囲外のフラグ添字を黙って「凍結」と読むので、配列が
短いまま走ると SR が平均場ブロックを丸ごと無視する。門番はこの関数の結果を
検査する側であって、作る側ではない。

順序:
1. `fill(true, 2 * (n_proj + n_idx))` で全可動に初期化
2. pmfpara.def の末尾フラグ行を適用(実部・虚部の両スロットへ)
3. オンサイト群の虚部を強制凍結

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
)::Int
    validate_parton_inputs(data, ctx)

    mp = data.modpara
    n_elec = mp.nelec
    n_steps = mp.nsr_opt_itr_step
    n_smp = mp.nsr_opt_itr_smp
    mfham = pstate.mfham

    for step = 0:(n_steps - 1)
        α = parton_alpha_from_terms(data)
        parton_update_orbitals!(mfham, α, n_elec)             # 契約 0
        parton_update_orbital_derivatives!(mfham, n_elec)     # 契約 0′

        parton_make_sample!(pstate, data, rng)                # 骨格 + 契約 2, 3
        parton_main_cal!(pstate, data)                        # 契約 4, 5

        weight_average_we!(ctx, pstate.state)
        weight_average_sr_opt!(ctx, pstate.state)
        reduce_counter!(ctx, pstate.config.counter)

        is_output_rank(ctx) &&
            output_data!(data, pstate.state, step; output_dir = output_dir)

        info = stochastic_opt!(data, pstate.state, c_timer)
        info = Int(bcast_scalar(ctx, info))
        if info != 0
            is_output_rank(ctx) &&
                @error "Parton SR: stochastic optimization failed" info step
            return info
        end

        parton_sync_parameters!(data, ctx)

        if step >= n_steps - n_smp
            store_opt_data!(data, pstate.state, step - (n_steps - n_smp))
        end
    end

    # 最適化された α を永続化する(zqp_opt.dat)。これを呼ばないと SR の結果が
    # どこにも残らない。data_io.jl の登録点で pmfpara_terms も書かれる。
    is_output_rank(ctx) && output_opt_data!(data; output_dir = output_dir)

    return 0
end
