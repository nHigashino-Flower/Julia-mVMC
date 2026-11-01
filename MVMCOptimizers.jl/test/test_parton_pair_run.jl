"""
Pfaffian パートンモードの SR ループ・ドライバ・出力 --- parton-mode (fork addition) ---

DESIGN_parton_pf.md §2.3・§4.5・§6・§9。ドライバ `parton_run_para_opt_from_namelist` は
パース直後に `PartonMode = 2` で分岐する。1 step は
読み込み → サンプリング → 測定 → SR → bcast → NaN ゲート → D_AmpMax → 診断。

- E2E: 乱数初期化で走り切り、出力がそろう
- T1-h: 乱数初期化の run と、その `_init.dat` を InPartonPairOrbital に渡した run とで
  zvo_out がバイト一致(初期値の乱数はサンプリングの乱数列を消費しない)
- T3: PartonMode = 1(α)と、同じ軌道から f = ΞJΞᵀ で作った PartonMode = 2 を同じ seed で
  1 step 回すと、step 0 の E が丸め誤差で一致する(QP なし・あり)
"""

using Test
using Random
using LinearAlgebra
using Printf
using MVMCExpertModeParsers
using MVMCOptimizers

const MOR = MVMCOptimizers

_pair_data_lines(path) =
    [l for l in readlines(path) if !isempty(strip(l)) && !startswith(l, "#")]

"In/zqp の p ファイルを読む(4 サイト・F=2 の全対 def に合わせて)。"
function _read_pair_file(path)
    data = pair_toy_data(n_site = 4, n_elec = 2, n_flavor = 2)
    n = length(full_pair_terms(4, 2))
    data.parton_pair_params = zeros(ComplexF64, n)
    nl = joinpath(dirname(path), "_read_nl.def")
    write(nl, "InPartonPairOrbital $(basename(path))\n")
    MOR.parton_pair_read_in!(data, nl)
    rm(nl)
    return data.parton_pair_params
end

@testset "E2E: Pf モードが乱数初期化で走り切り、出力がそろう" begin
    dir = mktempdir()
    out = joinpath(dir, "out")
    nl = _write_min_pair_input(dir)
    @test MOR.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0
    rows = _pair_data_lines(joinpath(out, "zvo_out.dat"))
    @test length(rows) == 4
    @test all(isfinite(parse(Float64, split(r)[1])) for r in rows)
    for f in ("zqp_parton_pair_init.dat", "zqp_parton_pair_opt.dat", "zvo_parton_pairdiag.dat",
              "zvo_parton_runinfo.dat", "zvo_SRinfo.dat")
        @test isfile(joinpath(out, f))
    end
    @test length(_pair_data_lines(joinpath(out, "zvo_parton_pairdiag.dat"))) == 4
    for f in ("zqp_pmfpara_init.dat", "zvo_parton_diag.dat", "zqp_pmforb_init.dat",
              "zvo_parton_orbdiag.dat")
        @test !isfile(joinpath(out, f))
    end
    info = read(joinpath(out, "zvo_parton_runinfo.dat"), String)
    @test occursin("PartonMode 2", info)
    @test occursin("pair_init random", info)
    # 初期値・最終値とも D_AmpMax で max|p| = 4
    @test maximum(abs, _read_pair_file(joinpath(out, "zqp_parton_pair_init.dat"))) ≈ 4.0 rtol = 1e-15
    @test maximum(abs, _read_pair_file(joinpath(out, "zqp_parton_pair_opt.dat"))) ≈ 4.0 rtol = 1e-15
end

@testset "T1-h: 乱数初期化の run = その _init.dat を読ませた run(zvo_out がバイト一致)" begin
    dir1 = mktempdir()
    out1 = joinpath(dir1, "out")
    @test MOR.parton_run_para_opt_from_namelist(
        _write_min_pair_input(dir1; nsteps = 3); output_dir = out1) == 0
    dir2 = mktempdir()
    out2 = joinpath(dir2, "out")
    @test MOR.parton_run_para_opt_from_namelist(
        _write_min_pair_input(dir2; nsteps = 3,
                              in_from = joinpath(out1, "zqp_parton_pair_init.dat"));
        output_dir = out2) == 0
    @test read(joinpath(out1, "zvo_out.dat")) == read(joinpath(out2, "zvo_out.dat"))
    @test read(joinpath(out1, "zqp_parton_pair_opt.dat")) == read(joinpath(out2, "zqp_parton_pair_opt.dat"))
    @test occursin("pair_init InPartonPairOrbital", read(joinpath(out2, "zvo_parton_runinfo.dat"), String))
end

@testset "E2E: opt を InPartonPairOrbital にして再開できる" begin
    dir1 = mktempdir()
    out1 = joinpath(dir1, "out")
    @test MOR.parton_run_para_opt_from_namelist(
        _write_min_pair_input(dir1; nsteps = 3); output_dir = out1) == 0
    opt1 = joinpath(out1, "zqp_parton_pair_opt.dat")
    dir2 = mktempdir()
    out2 = joinpath(dir2, "out")
    @test MOR.parton_run_para_opt_from_namelist(
        _write_min_pair_input(dir2; nsteps = 2, in_from = opt1); output_dir = out2) == 0
    @test isapprox(_read_pair_file(joinpath(out2, "zqp_parton_pair_init.dat")),
                   _read_pair_file(opt1); rtol = 1e-14)
end

"""
T3 の Pf 側の入力: α モードの最小入力と同じ軌道(初期 α の占有軌道)から f = ΞJΞᵀ を作り、
全対 def の InPartonPairOrbital として書く。
"""
function _write_pair_from_alpha(dir; qp::Bool, seed = 9)
    data = dimerized_mf_data()
    mfham = build_toy_mfham(data; n_idx = 3)
    Φ = mfham.orbitals
    Ns, N, K = data.modpara.nsite, data.modpara.nelec, data.modpara.nflavor
    M = K * N
    Ξ = zeros(ComplexF64, K * Ns, M)
    for a = 1:K
        Ξ[(a - 1) * Ns + 1:a * Ns, (a - 1) * N + 1:a * N] = Φ[a]
    end
    J = _rand_antisym(MersenneTwister(seed), M)
    f = Ξ * J * transpose(Ξ)
    terms = full_pair_terms(Ns, K)
    tmp = pair_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
    tmp.parton_pair_params = pair_params_from_matrix(f, terms, Ns)
    in_path = joinpath(dir, "_alpha_pair_in.dat")
    MOR.parton_pair_write_params(tmp, in_path)
    return _write_min_pair_input(dir; nsteps = 1, nsmp = 1, in_from = in_path, qp = qp)
end

"α の最小入力(QP を足すなら 2 並進)。"
function _write_alpha_input(dir; qp::Bool)
    nl, _ = _write_min_parton_input(dir)
    mod = joinpath(dir, "modpara.def")
    lines = map(readlines(mod)) do l
        startswith(l, "NSROptItrStep") ? "NSROptItrStep 1" :
        startswith(l, "NSROptItrSmp") ? "NSROptItrSmp 1" : l
    end
    push!(lines, "PartonFlavorSymFast 0")
    qp && push!(lines, "NMPTrans 2")
    write(mod, join(lines, "\n") * "\n")
    if qp
        _write_two_shift_qp(joinpath(dir, "qptransidx.def"), 4)
        open(nl, "a") do io
            println(io, "TransSym       qptransidx.def")
        end
    end
    return nl
end

@testset "T3: PartonMode = 1 と f = ΞJΞᵀ の PartonMode = 2 は step 0 の E が一致(QP $(qp ? "あり" : "なし"))" for qp in (false, true)
    da = mktempdir()
    @test MOR.parton_run_para_opt_from_namelist(
        _write_alpha_input(da; qp = qp); output_dir = joinpath(da, "out")) == 0
    dp = mktempdir()
    @test MOR.parton_run_para_opt_from_namelist(
        _write_pair_from_alpha(dp; qp = qp); output_dir = joinpath(dp, "out")) == 0
    ea = parse.(Float64, split(_pair_data_lines(joinpath(da, "out", "zvo_out.dat"))[1]))
    ep = parse.(Float64, split(_pair_data_lines(joinpath(dp, "out", "zvo_out.dat"))[1]))
    @test isapprox(ep[1], ea[1]; rtol = 1e-10)          # ⟨E⟩
    @test isapprox(ep[3], ea[3]; rtol = 1e-8)           # ⟨E²⟩
    # 受理率も一致(同じ乱数列・同じ受理判定)。α の diag では 5 列目、Pf の pairdiag では 3 列目
    acc_a = parse(Float64, split(_pair_data_lines(joinpath(da, "out", "zvo_parton_diag.dat"))[1])[5])
    acc_p = parse(Float64, split(_pair_data_lines(joinpath(dp, "out", "zvo_parton_pairdiag.dat"))[1])[3])
    @test acc_p == acc_a
end

@testset "NaN の更新は捨てて直前の値に戻す" begin
    data = pair_toy_data(n_site = 4, n_elec = 2, n_flavor = 2)
    data.parton_pair_terms = [MVMCExpertModeParsers.PartonPairOrbitalTerm(
        t.site1, t.flavor1, t.site2, t.flavor2, t.idx, t.sign) for t in full_pair_terms(4, 2)]
    data.parton_pair_params = randn(MersenneTwister(1), ComplexF64, length(data.parton_pair_terms))
    before = pack_parameters(data)
    @test MOR.parton_pair_accept_update!(data, before) == true
    data.parton_pair_params[2] = ComplexF64(NaN, 0)
    @test MOR.parton_pair_accept_update!(data, before) == false
    @test pack_parameters(data) == before
end

@testset "門番: PartonMode = 2 に α・Φ の入力が混ざると止まる" begin
    dir = mktempdir()
    nl = _write_min_pair_input(dir)
    open(nl, "a") do io
        println(io, "InPmfOrb  whatever.def")
    end
    @test_throws ErrorException MOR.parton_run_para_opt_from_namelist(nl; output_dir = joinpath(dir, "out"))
end
