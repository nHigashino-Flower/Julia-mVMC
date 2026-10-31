"""
Φ 直接変分モードの SR ループ・ドライバ・出力のテスト --- parton-mode (fork addition) ---

DESIGN_parton.md §12.11。ドライバ `parton_run_para_opt_from_namelist` はパース直後に
`PartonOrbParam` で分岐する。Φ モードの 1 step は
読み込み → サンプリング → 測定 → SR → bcast → NaN ゲート → 再スケール → 診断。
"""

using Test
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

"""
    _write_min_orbparam_input(dir; nsteps = 5, nsmp = 2, inpmforb_from = nothing) -> namelist

`_write_min_parton_input` と同じ物理項・modpara に `PartonOrbParam 1` を足し、
pmftrans / pmfpara を外した最小入力。InPmfOrb は既定ではトイの α 解の軌道から作り、
`inpmforb_from` を渡すとそのファイルを使う。
"""
function _write_min_orbparam_input(dir; nsteps::Int = 5, nsmp::Int = 2,
                                   inpmforb_from = nothing)
    _write_min_parton_input(dir)
    rm(joinpath(dir, "pmftrans.def"))
    rm(joinpath(dir, "pmfpara.def"))
    mod = joinpath(dir, "modpara.def")
    lines = readlines(mod)
    lines = map(lines) do l
        startswith(l, "NSROptItrStep") ? "NSROptItrStep $nsteps" :
        startswith(l, "NSROptItrSmp") ? "NSROptItrSmp $nsmp" : l
    end
    push!(lines, "PartonOrbParam 1")
    write(mod, join(lines, "\n") * "\n")
    write(joinpath(dir, "namelist.def"),
          "ModPara        modpara.def\nPhysHop        physhop.def\n" *
          "CoulombInter   coulombinter.def\nInPmfOrb       InPmfOrb.def\n")
    if inpmforb_from === nothing
        data = dimerized_mf_data()
        mfham = build_toy_mfham(data; n_idx = 3)
        data.pmforb_params = MVMCOptimizers.parton_orbparam_params_from_orbitals(mfham.orbitals)
        MVMCOptimizers.parton_orbparam_write_pmforb(data, joinpath(dir, "InPmfOrb.def"))
    else
        cp(inpmforb_from, joinpath(dir, "InPmfOrb.def"); force = true)
    end
    return joinpath(dir, "namelist.def")
end

"Φ ファイルを読む(4 サイト・F=2・Ne=2 の最小入力用)。"
function _read_pmforb_file(path)
    data = orbparam_toy_data(; n_site = 4, n_elec = 2, n_flavor = 2)
    dir = dirname(path)
    nl = joinpath(dir, "_read.def")
    write(nl, "InPmfOrb $(basename(path))\n")
    MVMCOptimizers.parton_orbparam_read_in_pmforb!(data, nl)
    rm(nl)
    return data.pmforb_params
end

_data_lines(path) = [l for l in readlines(path) if !isempty(strip(l)) && !startswith(l, "#")]

@testset "E2E: Φ モードが走り切り、出力がそろう" begin
    dir = mktempdir()
    out = joinpath(dir, "out")
    nl = _write_min_orbparam_input(dir)
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0
    rows = _data_lines(joinpath(out, "zvo_out.dat"))
    @test length(rows) == 5
    @test all(isfinite(parse(Float64, split(r)[1])) for r in rows)
    @test isfile(joinpath(out, "zqp_pmforb_init.dat"))
    @test isfile(joinpath(out, "zqp_pmforb_opt.dat"))
    @test length(_data_lines(joinpath(out, "zvo_parton_orbdiag.dat"))) == 5
    @test !isfile(joinpath(out, "zqp_pmfpara_init.dat"))
    @test !isfile(joinpath(out, "zvo_parton_diag.dat"))
    @test occursin("PartonOrbParam 1", read(joinpath(out, "zvo_parton_runinfo.dat"), String))
end

@testset "E2E: init の Φ は各フレーバーで max = PartonOrbAmpMax" begin
    dir = mktempdir()
    out = joinpath(dir, "out")
    nl = _write_min_orbparam_input(dir; nsteps = 2, nsmp = 1)
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0
    params = _read_pmforb_file(joinpath(out, "zqp_pmforb_init.dat"))
    block = 4 * 2
    for f = 1:2
        @test isapprox(maximum(abs, params[((f - 1) * block + 1):(f * block)]), 4.0;
                       rtol = 1e-15)
    end
end

@testset "E2E: opt の Φ を InPmfOrb にして再開できる" begin
    dir1 = mktempdir()
    out1 = joinpath(dir1, "out")
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(
        _write_min_orbparam_input(dir1; nsteps = 3, nsmp = 1); output_dir = out1) == 0
    opt1 = joinpath(out1, "zqp_pmforb_opt.dat")
    dir2 = mktempdir()
    out2 = joinpath(dir2, "out")
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(
        _write_min_orbparam_input(dir2; nsteps = 2, nsmp = 1, inpmforb_from = opt1);
        output_dir = out2) == 0
    @test isapprox(_read_pmforb_file(joinpath(out2, "zqp_pmforb_init.dat")),
                   _read_pmforb_file(opt1); rtol = 1e-14)
end

@testset "NaN の更新は捨てて直前の値に戻す" begin
    data = orbparam_toy_data()
    before = MVMCOptimizers.pack_parameters(data)
    @test MVMCOptimizers.parton_orbparam_accept_update!(data, before) == true
    @test MVMCOptimizers.pack_parameters(data) == before
    data.pmforb_params[2] = ComplexF64(NaN, 0)
    @test MVMCOptimizers.parton_orbparam_accept_update!(data, before) == false
    @test MVMCOptimizers.pack_parameters(data) == before
end

"保存済みサンプルを決定的に用意した Φ 状態(Metropolis は通さない)。"
function _orb_thread_state(; store::Bool)
    data = orbparam_toy_data()
    set_shift_qp!(data, [collect(1:6), [mod1(r + 1, 6) for r = 1:6]],
                  [ones(Int, 6), ones(Int, 6)], ComplexF64[0.6, 0.4])
    data.modpara.nvmc_sample = 24
    data.modpara.nstore_o = store ? 1 : 0
    MVMCOptimizers.parton_orbparam_materialize_flags!(data)
    pstate = MVMCOptimizers.parton_orbparam_build_state(data)
    mp = data.modpara
    rng = MersenneTwister(4242)
    cfg = pstate.config
    for s = 1:mp.nvmc_sample
        sites = sort!(Random.randperm(rng, mp.nsite)[1:mp.nelec])
        fill!(cfg.ele_cfg, -1)
        fill!(cfg.ele_num, 0)
        for f = 1:mp.nflavor, (m, r) in enumerate(sites)
            MVMCOptimizers.place_particle!(cfg, f, m, r)
        end
        MVMCOptimizers.parton_store_sample!(cfg, s)
    end
    return pstate, data
end

@testset "スレッド並列 = 逐次(Φ 状態でビット一致、store=$store)" for store in (true, false)
    ps_seq, data_seq = _orb_thread_state(; store = store)
    ps_thr, data_thr = _orb_thread_state(; store = store)
    @test ps_seq isa MVMCOptimizers.PartonOptimizationState{MVMCOptimizers.PartonDirectPhi}
    MVMCOptimizers.parton_main_cal!(ps_seq, data_seq; force_threaded = false)
    MVMCOptimizers.parton_main_cal!(ps_thr, data_thr; force_threaded = true)
    for (a, b) in ((ps_seq.state.sr_opt.sr_opt_oo, ps_thr.state.sr_opt.sr_opt_oo),
                   (ps_seq.state.sr_opt.sr_opt_ho, ps_thr.state.sr_opt.sr_opt_ho))
        @test a == b
    end
    @test ps_seq.state.energy.etot == ps_thr.state.energy.etot
    @test ps_seq.state.energy.wc == 24
end

@testset "PartonOrbParam = 0 の入力は従来の経路を通る" begin
    dir = mktempdir()
    out = joinpath(dir, "out")
    nl, _ = _write_min_parton_input(dir)
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0
    @test isfile(joinpath(out, "zqp_pmfpara_opt.dat"))
    @test !isfile(joinpath(out, "zvo_parton_orbdiag.dat"))
    @test !isfile(joinpath(out, "zqp_pmforb_init.dat"))
end

# =====================================================================
# α 解 → Φ 入力の変換道具(tools/parton_alpha_to_orbparam.jl)
# =====================================================================

include(joinpath(@__DIR__, "..", "..", "tools", "parton_alpha_to_orbparam.jl"))

"全配置の E_loc(1 配置 = 全フレーバー同じサイト)。"
function _all_eloc(pstate, data)
    mp = data.modpara
    qpw = MVMCOptimizers.parton_qp_weight(data)
    out = ComplexF64[]
    for i = 1:mp.nsite, j = (i + 1):mp.nsite
        cfg = toy_config(data, [i, j])
        pstate.config = cfg
        MVMCOptimizers.parton_recompute_amplitude_all!(
            pstate.amp, pstate.mfham, cfg, data, pstate.workspace)
        ip = MVMCOptimizers.parton_calculate_ip(pstate.amp, qpw)
        push!(out, MVMCOptimizers.parton_local_energy(pstate, data, ip))
    end
    return out
end

@testset "変換道具: α の最終状態と同じ波動関数の Φ 入力を作る" begin
    dir = mktempdir()
    a_in = joinpath(dir, "alpha_in")
    a_out = joinpath(dir, "alpha_out")
    mkpath(a_in)
    nl, _ = _write_min_parton_input(a_in)
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = a_out) == 0

    new_dir = joinpath(dir, "phi_in")
    parton_alpha_to_orbparam(a_in, a_out, new_dir)

    entries = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(joinpath(new_dir, "namelist.def")))
    keys_new = first.(entries)
    @test "InPmfOrb" in keys_new
    @test !any(k -> k in keys_new, ("PartonMFTrans", "PartonMFPara", "InPmfPara", "InPmfOcc"))
    @test all(isfile(joinpath(new_dir, p)) for (_, p) in entries)
    mod = readlines(joinpath(new_dir, "modpara.def"))
    @test "PartonOrbParam 1" in mod
    @test "PartonOccMode 0" in mod
    @test count(l -> startswith(l, "PartonOrbParam"), mod) == 1

    # α の最終状態(α* と占有)と Φ 状態の E_loc が全配置で一致する
    data_a = MVMCExpertModeParsers.parse_expert_mode_files(nl)
    tmp = joinpath(dir, "_ref.def")
    write(tmp, "InPmfPara $(joinpath(a_out, "zqp_pmfpara_opt.dat"))\n" *
               "InPmfOcc $(joinpath(a_out, "zqp_pmfocc_opt.dat"))\n")
    MVMCOptimizers.parton_read_in_pmfpara!(data_a, tmp)
    occ = MVMCOptimizers.parton_read_in_pmfocc(data_a, tmp)
    MVMCOptimizers.parton_ensure_qp!(data_a)
    ps_a = MVMCOptimizers.parton_build_optimization_state(data_a)
    MVMCOptimizers.parton_update_orbitals!(
        ps_a.mfham, MVMCOptimizers.parton_alpha_from_terms(data_a), data_a.modpara.nelec;
        forced_occ = occ)

    data_p = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(new_dir, "namelist.def"))
    MVMCOptimizers.parton_orbparam_read_in_pmforb!(data_p, joinpath(new_dir, "namelist.def"))
    MVMCOptimizers.parton_ensure_qp!(data_p)
    ps_p = MVMCOptimizers.parton_orbparam_build_state(data_p)
    @test isapprox(_all_eloc(ps_p, data_p), _all_eloc(ps_a, data_a); rtol = 1e-10)

    # 変換した入力でそのまま Φ モードが走る
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(
        joinpath(new_dir, "namelist.def"); output_dir = joinpath(dir, "phi_out"), nsteps = 2,
        nsmp = 1) == 0
end

# =====================================================================
# 最終レビューの指摘に対する回帰ガード
# =====================================================================

@testset "最終 Φ は診断を書かない run でも保存される" begin
    dir = mktempdir()
    out = joinpath(dir, "out")
    nl = _write_min_orbparam_input(dir; nsteps = 2, nsmp = 1)
    data = MVMCExpertModeParsers.parse_expert_mode_files(nl)
    MVMCOptimizers.parton_orbparam_read_in_pmforb!(data, nl)
    MVMCOptimizers.parton_orbparam_rescale!(data)
    MVMCOptimizers.parton_orbparam_materialize_flags!(data)
    MVMCOptimizers.parton_ensure_qp!(data)
    pstate = MVMCOptimizers.parton_orbparam_build_state(data)
    rng = MVMCOptimizers.SFMT19937RNG()
    Random.seed!(rng, 11272)
    mkpath(out)
    @test MVMCOptimizers.parton_orbparam_vmc_para_opt!(
        pstate, data, MVMCOptimizers.serial_context();
        rng = rng, output_dir = out, write_diagnostics = false) == 0
    @test isfile(joinpath(out, "zqp_pmforb_opt.dat"))
    @test !isfile(joinpath(out, "zvo_parton_orbdiag.dat"))
end

@testset "NFlavor を書き忘れると NFlavor を名指しして止まる" begin
    dir = mktempdir()
    nl = _write_min_orbparam_input(dir)
    mod = joinpath(dir, "modpara.def")
    write(mod, join(filter(l -> !startswith(l, "NFlavor"), readlines(mod)), "\n") * "\n")
    err = try
        MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = joinpath(dir, "out"))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test err !== nothing && occursin("NFlavor", err.msg)
end

"α の最小入力に環の距離クラス Jastrow(jastrowidx.def)を足す。"
function _add_jastrow_def!(dir, n_site::Int)
    n_class = n_site ÷ 2
    open(joinpath(dir, "jastrowidx.def"), "w") do io
        println(io, "===============================")
        println(io, "NJastrowIdx $n_class")
        println(io, "ComplexType 0")
        println(io, "===============================")
        println(io, "===============================")
        for i = 0:(n_site - 1), j = 0:(n_site - 1)
            i == j && continue
            println(io, "$i $j $(min(mod(i - j, n_site), mod(j - i, n_site)) - 1)")
        end
        for k = 0:(n_class - 1)
            println(io, "$k 1")
        end
    end
    open(joinpath(dir, "namelist.def"), "a") do io
        println(io, "Jastrow        jastrowidx.def")
    end
end

"per-block 出力(idx Re Im)を idx → 値の辞書にする。"
function _read_block_values(path)
    out = Dict{Int,ComplexF64}()
    for l in readlines(path)
        t = split(strip(l))
        (length(t) >= 3 && tryparse(Int, t[1]) !== nothing) || continue
        out[parse(Int, t[1])] = complex(parse(Float64, t[2]), parse(Float64, t[3]))
    end
    return out
end

"jastrow_terms は idx 順に並ぶ(JastrowTerm は idx を持たない)ので位置から idx を取る。"
_jastrow_values(data) = Dict(k - 1 => ComplexF64(t.value) for (k, t) in enumerate(data.jastrow_terms))

@testset "変換道具: Jastrow の最終値を 5 行ヘッダの InJastrow に直して引き継ぐ" begin
    dir = mktempdir()
    a_in = joinpath(dir, "alpha_in")
    a_out = joinpath(dir, "alpha_out")
    mkpath(a_in)
    nl, _ = _write_min_parton_input(a_in)
    _add_jastrow_def!(a_in, 4)
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = a_out) == 0
    raw = joinpath(a_out, "zqp_jastrow_opt.dat")
    expected = _read_block_values(raw)
    @test length(expected) == 2
    @test abs(expected[0]) > 1e-8                  # 最適化されて idx 0 が非零

    new_dir = joinpath(dir, "phi_in")
    parton_alpha_to_orbparam(a_in, a_out, new_dir)
    data_p = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(new_dir, "namelist.def"))
    @test _jastrow_values(data_p) == expected

    # 対照: 4 行ヘッダのまま InJastrow にすると idx 0 が落ちる(この変換が飾りでないこと)
    ctrl = joinpath(dir, "ctrl")
    cp(new_dir, ctrl)
    cp(raw, joinpath(ctrl, "InJastrow.def"); force = true)
    data_c = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(ctrl, "namelist.def"))
    @test _jastrow_values(data_c) != expected
end
