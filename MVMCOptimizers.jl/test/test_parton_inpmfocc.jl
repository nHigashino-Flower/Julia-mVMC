"""
§8 テスト 18: `InPmfOcc`(占有集合の読み戻し経路)
--- parton-mode (fork addition) ---

v3.12 の終端の自己完結性検査(§2.5)が `occ_selfcontained = 0` を返した
(REPORT §16-5: 非アウフバウ最適解は本物で、占有の 67〜78% の step が
「下から Ne 個」ではない)。したがって `PartonOccMode = 1` の run は
**α\\* 単独では状態を再現できない**。占有は `zqp_pmfocc_opt.dat` に記録済みなので、
それを読み戻す経路を足す。

作法は `InPmfPara` と同じ: **ファイルの有無がスイッチ**、modpara にキーを足さない、
適用はパートンドライバ(§2.3.1 の確定順序)。読み込んだ占有は**初期占有**として
採用され、以後は `PartonOccMode` に従う(0 なら aufbau で選び直し、1 なら mom が
そこから枝を追う)。

全 step 占有を固定する `explicit`(= `PartonOccMode = 2`)は**実装しない**。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"占有集合を `.def` 族(`zqp_pmfocc_opt.dat` と同一形式)で書く。"
function _write_pmfocc(path, rows)
    open(path, "w") do io
        println(io, "===============================")
        println(io, "NPmfOcc $(length(rows))")
        println(io, "===============================")
        println(io, "== flavor band_index ==")
        println(io, "===============================")
        for (fl, band) in rows
            println(io, "$fl $band")
        end
    end
    return path
end

"namelist.def に InPmfOcc の行を足す。"
function _add_inpmfocc!(dir, fname = "InPmfOcc.def")
    nl = joinpath(dir, "namelist.def")
    open(nl, "a") do io
        println(io, "InPmfOcc       $fname")
    end
    return nl
end

@testset "§8-18-1 InPmfOcc の読み戻し: 形式と往復" begin
    dir = mktempdir()
    nl, _ = _write_min_parton_input(dir)
    data = MVMCExpertModeParsers.parse_expert_mode_files(nl)

    # ファイルが無ければ nothing(既定挙動は変えない)
    @test MVMCOptimizers.parton_read_in_pmfocc(data, nl) === nothing

    # 出力形式(`zqp_pmfocc_opt.dat`)をそのまま渡せること。
    # 4 サイト・NElec 2・2 フレーバー、band は 0-based
    _write_pmfocc(joinpath(dir, "InPmfOcc.def"), [(0, 0), (0, 2), (1, 1), (1, 3)])
    _add_inpmfocc!(dir)
    occ = MVMCOptimizers.parton_read_in_pmfocc(data, nl)
    @test occ == [[1, 3], [2, 4]]        # 1-based・昇順に正規化して返す
end

@testset "§8-18-2 検証: 黙って壊れずエラーで止まる" begin
    mk = function (rows; fname = "InPmfOcc.def", count = nothing)
        dir = mktempdir()
        nl, _ = _write_min_parton_input(dir)
        path = joinpath(dir, fname)
        _write_pmfocc(path, rows)
        if count !== nothing        # ヘッダの件数だけ書き換える
            lines = readlines(path)
            lines[2] = "NPmfOcc $count"
            write(path, join(lines, "\n") * "\n")
        end
        _add_inpmfocc!(dir, fname)
        return MVMCExpertModeParsers.parse_expert_mode_files(nl), nl
    end

    # 行数が n_flavor × Ne と一致しない
    data, nl = mk([(0, 0), (0, 1), (1, 0)])
    @test_throws ErrorException MVMCOptimizers.parton_read_in_pmfocc(data, nl)

    # band_index が範囲外(NSite = 4)
    data, nl = mk([(0, 0), (0, 4), (1, 0), (1, 1)])
    @test_throws ErrorException MVMCOptimizers.parton_read_in_pmfocc(data, nl)

    # flavor が範囲外(NFlavor = 2)
    data, nl = mk([(0, 0), (0, 1), (2, 0), (2, 1)])
    @test_throws ErrorException MVMCOptimizers.parton_read_in_pmfocc(data, nl)

    # 同一フレーバー内で band が重複
    data, nl = mk([(0, 0), (0, 0), (1, 0), (1, 1)])
    @test_throws ErrorException MVMCOptimizers.parton_read_in_pmfocc(data, nl)

    # フレーバーあたりの本数が Ne でない(0 に 3 本、1 に 1 本)
    data, nl = mk([(0, 0), (0, 1), (0, 2), (1, 0)])
    @test_throws ErrorException MVMCOptimizers.parton_read_in_pmfocc(data, nl)

    # ヘッダの件数と実データの行数が食い違う
    data, nl = mk([(0, 0), (0, 1), (1, 0), (1, 1)]; count = 3)
    @test_throws ErrorException MVMCOptimizers.parton_read_in_pmfocc(data, nl)
end

@testset "§8-18-3 forced_occ: 初期占有をそのまま採用する" begin
    α = ComplexF64[0, 1, 2, 3]
    data = onsite_crossing_data()
    mfham = build_crossing_mfham(data; occ_mode = 0)

    # forced_occ を渡すと選択則(aufbau)を上書きする
    MVMCOptimizers.parton_update_orbitals!(mfham, α, 2; forced_occ = [[2, 4]])
    @test mfham.occ[1] == [2, 4]
    # 占有軌道が band 2, 4 になっている(対角系なのでサイト 2, 4)
    @test abs2(mfham.orbitals[1][2, 1]) ≈ 1.0
    @test abs2(mfham.orbitals[1][4, 2]) ≈ 1.0

    # forced_occ 無しで呼び直すと通常の選択則に戻る(aufbau)
    MVMCOptimizers.parton_update_orbitals!(mfham, α, 2)
    @test mfham.occ[1] == [1, 2]

    # mom は forced_occ を最初の参照として枝を追う(§2-d)
    mm = build_crossing_mfham(onsite_crossing_data(); occ_mode = 1)
    MVMCOptimizers.parton_update_orbitals!(mm, α, 2; forced_occ = [[2, 4]])
    @test mm.occ[1] == [2, 4]
    MVMCOptimizers.parton_update_orbitals!(mm, ComplexF64[2.5, 1, 2, 3], 2)
    # サイト 2(band 1)とサイト 4(band 4)を追う
    @test mm.occ[1] == [1, 4]
end

@testset "§8-18-4 往復(本命): (α*, O*) の組で状態が閉じる" begin
    # 1 本目: PartonOccMode = 1 で回して α と占有を出す
    dir1 = mktempdir()
    nl1, _ = _write_min_parton_input(dir1)
    open(joinpath(dir1, "modpara.def"), "a") do io
        println(io, "PartonOccMode 1")
    end
    out1 = mktempdir()
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl1; output_dir = out1) == 0

    # 2 本目: その α と占有を入力にして継続する
    dir2 = mktempdir()
    nl2, _ = _write_min_parton_input(dir2)
    open(joinpath(dir2, "modpara.def"), "a") do io
        println(io, "PartonOccMode 1")
    end
    cp(joinpath(out1, "zqp_pmfpara_opt.dat"), joinpath(dir2, "InPmfPara.def"))
    cp(joinpath(out1, "zqp_pmfocc_opt.dat"), joinpath(dir2, "InPmfOcc.def"))
    open(joinpath(dir2, "namelist.def"), "a") do io
        println(io, "InPmfPara      InPmfPara.def")
        println(io, "InPmfOcc       InPmfOcc.def")
    end
    out2 = mktempdir()
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl2; output_dir = out2) == 0

    # 継続 run の**初期占有**が 1 本目の終端占有と一致する
    @test readlines(joinpath(out2, "zqp_pmfocc_init.dat")) ==
          readlines(joinpath(out1, "zqp_pmfocc_opt.dat"))

    # 初手の Φ が一致する = 状態が閉じている。
    # 1 本目の終端 (α*, O*) で組んだ Φ と、2 本目の初期 Φ を直接比べる
    d1 = MVMCExpertModeParsers.parse_expert_mode_files(nl1)
    MVMCOptimizers.parton_init_alpha!(d1, 1)
    MVMCOptimizers.parton_materialize_flags!(d1)
    MVMCOptimizers.parton_ensure_qp!(d1)
    ps1 = MVMCOptimizers.parton_build_optimization_state(d1)
    # 終端 α を読み込む
    params = MVMCExpertModeParsers.parse_input_parameter_file(
        joinpath(out1, "zqp_pmfpara_opt.dat"))
    for t in d1.pmfpara_terms
        t.value = params[t.idx]
    end
    occ_end = MVMCOptimizers.parton_read_in_pmfocc(d1, nl2)   # nl2 が InPmfOcc を持つ
    MVMCOptimizers.parton_update_orbitals!(
        ps1.mfham, MVMCOptimizers.parton_alpha_from_terms(d1), d1.modpara.nelec;
        forced_occ = occ_end)

    d2 = MVMCExpertModeParsers.parse_expert_mode_files(nl2)
    MVMCOptimizers.parton_init_alpha!(d2, 1)
    MVMCOptimizers.parton_read_in_pmfpara!(d2, nl2)
    MVMCOptimizers.parton_materialize_flags!(d2)
    MVMCOptimizers.parton_ensure_qp!(d2)
    ps2 = MVMCOptimizers.parton_build_optimization_state(d2)
    occ2 = MVMCOptimizers.parton_read_in_pmfocc(d2, nl2)
    MVMCOptimizers.parton_update_orbitals!(
        ps2.mfham, MVMCOptimizers.parton_alpha_from_terms(d2), d2.modpara.nelec;
        forced_occ = occ2)

    for f in eachindex(ps1.mfham.orbitals)
        @test ps1.mfham.orbitals[f] == ps2.mfham.orbitals[f]   # ビット一致
    end
    @test ps1.mfham.occ == ps2.mfham.occ
end

@testset "§8-18-5 非アウフバウ占有の往復(aufbau 初期化では再現できない状態)" begin
    dir = mktempdir()
    nl, _ = _write_min_parton_input(dir)
    data = MVMCExpertModeParsers.parse_expert_mode_files(nl)
    MVMCOptimizers.parton_init_alpha!(data, 1)
    MVMCOptimizers.parton_materialize_flags!(data)
    MVMCOptimizers.parton_ensure_qp!(data)

    # 「下から Ne 個」ではない占有を InPmfOcc で与える
    _write_pmfocc(joinpath(dir, "InPmfOcc.def"), [(0, 0), (0, 2), (1, 0), (1, 2)])
    _add_inpmfocc!(dir)
    occ = MVMCOptimizers.parton_read_in_pmfocc(data, nl)
    @test occ == [[1, 3], [1, 3]]

    ps = MVMCOptimizers.parton_build_optimization_state(data)
    α = MVMCOptimizers.parton_alpha_from_terms(data)
    MVMCOptimizers.parton_update_orbitals!(ps.mfham, α, data.modpara.nelec;
                                           forced_occ = occ)
    @test ps.mfham.occ == [[1, 3], [1, 3]]
    Φ_forced = [copy(o) for o in ps.mfham.orbitals]

    # aufbau で初期化すると別の状態になる = 読み戻しでしか再現できない
    MVMCOptimizers.parton_update_orbitals!(ps.mfham, α, data.modpara.nelec)
    @test ps.mfham.occ == [[1, 2], [1, 2]]
    @test any(Φ_forced[f] != ps.mfham.orbitals[f] for f in eachindex(Φ_forced))
end

@testset "§8-18-6 aufbau との併用: 初期占有は InPmfOcc、以後は選び直される" begin
    dir = mktempdir()
    nl, _ = _write_min_parton_input(dir)
    # PartonOccMode は書かない(既定 0 = aufbau)
    _write_pmfocc(joinpath(dir, "InPmfOcc.def"), [(0, 0), (0, 2), (1, 0), (1, 2)])
    _add_inpmfocc!(dir)
    out = mktempdir()
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0

    # 初期占有は InPmfOcc 由来(非アウフバウ)が記録される
    @test readlines(joinpath(out, "zqp_pmfocc_init.dat"))[6:end] ==
          ["0 0", "0 2", "1 0", "1 2"]
    # 終端は aufbau で選び直されている(仕様どおりの挙動)
    @test readlines(joinpath(out, "zqp_pmfocc_opt.dat"))[6:end] ==
          ["0 0", "0 1", "1 0", "1 1"]
end
