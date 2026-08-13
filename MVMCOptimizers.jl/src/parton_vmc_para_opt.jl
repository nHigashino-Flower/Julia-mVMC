


function parton_vmc_para_opt!(pstate, data, ctx; c_timer = CTIMER_DISABLED)
    validate_parton_inputs(data, ctx)                       # 門番(フラグ長検査込み)
    for step in 0:(n_sr_opt_itr_step - 1)
        α = parton_alpha_from_terms(data)                   # 正準=pmfpara値
        parton_update_orbitals!(mfham, α, n_elec)           # 契約0
        parton_update_orbital_derivatives!(mfham, α)        # 契約0′
        parton_make_sample!(pstate, data, rng)              # 骨格+契約2,3
        parton_main_cal!(pstate, data)                      # 契約4,5
        weight_average_we!(pstate.state)                    # ↓ここから委譲(1行メソッド)
        weight_average_sr_opt!(pstate.state)
        reduce_counter!(ctx, pstate.parton_config.counter)
        output_data!(data, pstate.state, step)
        info = stochastic_opt!(data, pstate.state, c_timer) # (B)によりMFブロックも解かれる
        info = bcast_scalar(ctx, info)                      # hang防止(既存の作法)
        parton_sync_parameters!(data, ctx)                  # rank0の値をbcast。D_AmpMaxは適用しない
        store_opt_data!(data, pstate.state, step)           # 最終nsmp平均用
    end
    parton_output_opt_params(data, ...)                     # zqp_opt相当の自モード出力
end