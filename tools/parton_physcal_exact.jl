"""
パートンの物理量測定(NVMCCalMode = 1、DESIGN_parton.md §13)の厳密値を全配置列挙で求める
--- parton-mode (fork addition) ---

測定と同じ namelist.def(In ファイルで固定したパラメータ)から波動関数を組み、全配置 C(Nsite, NElec) を
列挙して |Ψ|² 重みの局所値の平均(`MVMCOptimizers.parton_exact_expectations`)を書く。モンテカルロの
測定(サンプリング・蓄積・MPI 集計)の検証用。小さい系(配置数 ≲ 10⁶)専用。

使い方:

    julia --project=MVMCOptimizers.jl tools/parton_physcal_exact.jl <namelist.def> <out_dir>

出力(上流と同じ書式。<head> は CDataFileHead):

- `<head>_cisajs_exact.dat`        `i 0 j 0 Re Im`
- `<head>_cisajscktalt_exact.dat`  `i 0 j 0 k 0 l 0 Re Im`
- `<head>_energy_exact.dat`        `Re Im n_configurations`

状態の組み方はドライバ(`parton_run_para_opt_from_namelist` とモード別の分岐)の初期化と同じ順で、
In ファイルの必須条件も同じ(DESIGN §13.2)。ドライバ側の初期化を変えたらここも合わせること
(食い違えばモンテカルロとの比較が合わなくなるので検出はできる)。
"""

using Printf
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

const MOX = MVMCOptimizers

"""
    parton_exact_state_from_namelist(namelist) -> (data, pstate)

測定と同じ初期化で、固定パラメータの波動関数の状態を組む(逐次、MPI なし)。
"""
function parton_exact_state_from_namelist(namelist::AbstractString)
    nl = String(namelist)
    data = MVMCExpertModeParsers.parse_expert_mode_files(nl)
    mp = data.modpara
    ctx = MOX.serial_context()
    base_seed = MOX.resolve_rnd_seed(ctx, mp.rnd_seed, nothing) - ctx.group1
    MOX.validate_parton_green_inputs(data)

    if mp.parton_mode == MOX.PARTON_MODE_PFAFFIAN
        MOX.validate_parton_modpara(mp; allowed_modes = (MOX.PARTON_MODE_PFAFFIAN,))
        MOX.parton_pair_materialize_flags!(data)
        in_path = MOX.parton_pair_read_in!(data, nl)
        MOX.parton_phys_require_fixed_inputs(data, nl; orbital_given = in_path !== nothing,
                                             orbital_hint = "add InPartonPairOrbital")
        MOX.parton_pair_rescale!(data)
        MOX.parton_ensure_qp!(data)
        return data, MOX.parton_pair_build_state(data)
    elseif mp.parton_orb_param != MOX.PARTON_ORB_PARAM_ALPHA
        MOX.validate_parton_modpara(mp)
        map = MOX.parton_orbparam_read_orbidx(data, nl)
        map === nothing && (map = MOX.parton_orbparam_identity_map(mp.nsite, mp.nelec, mp.nflavor))
        MOX.parton_orbparam_init_random!(data, map, base_seed)
        in_path = MOX.parton_orbparam_read_in_pmforb!(data, nl, map)
        MOX.parton_phys_require_fixed_inputs(data, nl; orbital_given = in_path !== nothing,
                                             orbital_hint = "add InPmfOrb")
        MOX.parton_orbparam_rescale!(data, map)
        MOX.parton_orbparam_materialize_flags!(data, map)
        MOX.parton_ensure_qp!(data)
        return data, MOX.parton_orbparam_build_state(data, map)
    end

    randomized_idx = MOX.parton_init_alpha!(data, base_seed)
    in_pmfpara = MOX.parton_read_in_pmfpara!(data, nl)
    MOX.parton_materialize_flags!(data)
    MOX.validate_parton_inputs(data, ctx)
    MOX.parton_phys_require_fixed_inputs(data, nl;
                                         orbital_given = in_pmfpara || isempty(randomized_idx),
                                         orbital_hint = "add InPmfPara")
    occ_in = MOX.parton_read_in_pmfocc(data, nl)
    occ_in === nothing && error("add InPmfOcc (the α measurement needs the occupation)")
    MOX.parton_ensure_qp!(data)
    pstate = MOX.parton_build_optimization_state(data)
    MOX.parton_update_orbitals!(pstate.mfham, MOX.parton_alpha_from_terms(data), mp.nelec;
                                forced_occ = occ_in)
    return data, pstate
end

"厳密値を上流と同じ書式で書く。"
function parton_write_exact(data, exact, out_dir::AbstractString)
    mkpath(out_dir)
    head = isempty(data.modpara.c_data_file_head) ? "zvo" : data.modpara.c_data_file_head
    open(joinpath(out_dir, "$(head)_cisajs_exact.dat"), "w") do io
        for (t, v) in zip(data.green_one_terms, exact.one_body)
            @printf(io, "%d %d %d %d % .18e  % .18e \n", t.site1, 0, t.site2, 0, real(v), imag(v))
        end
    end
    open(joinpath(out_dir, "$(head)_cisajscktalt_exact.dat"), "w") do io
        for (t, v) in zip(data.green_two_terms, exact.two_body)
            @printf(io, "%d %d %d %d %d %d %d %d % .18e % .18e\n",
                    t.site1, 0, t.site2, 0, t.site3, 0, t.site4, 0, real(v), imag(v))
        end
    end
    open(joinpath(out_dir, "$(head)_energy_exact.dat"), "w") do io
        @printf(io, "% .18e % .18e %d\n", real(exact.energy), imag(exact.energy),
                exact.n_configurations)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 ||
        error("usage: julia --project=MVMCOptimizers.jl tools/parton_physcal_exact.jl <namelist.def> <out_dir>")
    data, pstate = parton_exact_state_from_namelist(ARGS[1])
    t0 = time()
    exact = MOX.parton_exact_expectations(pstate, data)
    parton_write_exact(data, exact, ARGS[2])
    @printf("E_exact = % .12f %+.3e i  (%d configurations, %.1f s) → %s\n",
            real(exact.energy), imag(exact.energy), exact.n_configurations, time() - t0, ARGS[2])
end
