"""
§8 テスト 5: OptFlag で凍結した成分が SR で動かないこと、および配線の疎通
--- parton-mode (fork addition) ---

DESIGN_parton.md §2.5(OptFlag とゲージ固定)と §8-5 に対応。
Φ は H → cH と H → H + μI で不変なので S に厳密なゼロモードが 2 本以上あり、
代表ホッピング振幅とオンサイトをフラグ行で凍結するのが通常形。
"""

using Test
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "flags の実体化はドライバの責務" begin
    data = dimerized_mf_data()
    data.pmfpara_opt_flags = Dict(0 => 0, 1 => 1, 2 => 1)
    flags = MVMCOptimizers.parton_materialize_flags!(data)

    # n_proj = 0, n_idx = 3 → 長さ 6。idx2 はオンサイト群なので Im が強制凍結
    @test flags == [false, false, true, true, true, false]
    @test data.optimization_flags === flags
    @test MVMCOptimizers.parton_onsite_idx_set(data) == Set([2])

    # フラグ行が無ければ全可動(オンサイト Im だけは強制凍結)
    data2 = dimerized_mf_data()
    empty!(data2.pmfpara_opt_flags)
    @test MVMCOptimizers.parton_materialize_flags!(data2) ==
          [true, true, true, true, true, false]

    # 実体化した flags は門番を通る
    ctx = MVMCOptimizers.serial_context()
    @test MVMCOptimizers.validate_parton_inputs(data, ctx) === nothing
end

@testset "SR: 凍結した idx は完全に不動、可動な idx は動く" begin
    # ゲージ射影はスケール群を丸ごと実数倍する(群の一部だけ倍率を変えると
    # ゲージ変換にならない)ので、凍結した idx も再正規化の対象になる。
    # ここは「SR 自身が凍結成分を動かさない」ことの検査なので射影は切る。
    # 射影 ON のときの振る舞いは §8-8 が受け持つ。
    data = dimerized_mf_data()
    data.modpara.parton_gauge_fix = 0
    data.pmfpara_opt_flags = Dict(0 => 0, 1 => 1, 2 => 1)
    MVMCOptimizers.parton_materialize_flags!(data)

    ctx = MVMCOptimizers.serial_context()
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    α_before = MVMCOptimizers.parton_alpha_from_terms(data)

    rng = MVMCOptimizers.SFMT19937RNG()
    Random.seed!(rng, 123)
    @test MVMCOptimizers.parton_vmc_para_opt!(
        pstate, data, ctx; rng = rng, output_dir = mktempdir()) == 0

    α_after = MVMCOptimizers.parton_alpha_from_terms(data)
    @test α_after[1] == α_before[1]     # ゲージ代表として凍結した強ボンド
    @test α_after[2] != α_before[2]     # 物理的な自由度は動いた
    @test α_after[3] == α_before[3]     # 一様オンサイトはゲージ平坦なので不動

    # エネルギーが有限で意味のある値になっていること
    e = pstate.state.energy.etot / pstate.state.energy.wc
    @test isfinite(real(e)) && real(e) < 0
end

@testset "SR: オンサイト群はゲージ平坦(O が厳密にゼロ)" begin
    # 一様オンサイトのシフトは H → H + μI で Φ を変えない。フラグ凍結(虚部)と
    # 冗長方向カット(実部)の二重の保険が効いていることを確認する。
    data = dimerized_mf_data()
    empty!(data.pmfpara_opt_flags)
    MVMCOptimizers.parton_materialize_flags!(data)
    @test data.optimization_flags[6] == false   # オンサイト Im
    @test data.optimization_flags[5] == true    # オンサイト Re はフラグ上は可動

    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 3)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(
        mfham, MVMCOptimizers.parton_alpha_from_terms(data), 2)
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, 2)
    for f = 1:2
        @test maximum(abs, mfham.dorbitals[f][5]) < 1e-10   # オンサイト Re
        @test all(iszero, mfham.dorbitals[f][6])            # オンサイト Im
    end

    ctx = MVMCOptimizers.serial_context()
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    α_before = MVMCOptimizers.parton_alpha_from_terms(data)
    rng = MVMCOptimizers.SFMT19937RNG()
    Random.seed!(rng, 777)
    @test MVMCOptimizers.parton_vmc_para_opt!(
        pstate, data, ctx; rng = rng, output_dir = mktempdir()) == 0

    α_after = MVMCOptimizers.parton_alpha_from_terms(data)
    @test α_after[3] == α_before[3]     # ゲージ平坦なので実部も動かない
end

@testset "SR: 共有 idx の全行が同じ値のままでいる" begin
    data = dimerized_mf_data()
    MVMCOptimizers.parton_materialize_flags!(data)
    ctx = MVMCOptimizers.serial_context()
    pstate = MVMCOptimizers.parton_build_optimization_state(data)

    rng = MVMCOptimizers.SFMT19937RNG()
    Random.seed!(rng, 55)
    MVMCOptimizers.parton_vmc_para_opt!(
        pstate, data, ctx; rng = rng, output_dir = mktempdir())

    by_idx = Dict{Int,ComplexF64}()
    for t in data.pmfpara_terms
        v = get(by_idx, t.idx, nothing)
        if v === nothing
            by_idx[t.idx] = t.value
        else
            @test v == t.value      # 絶対 set は共有 idx を冪等に保つ
        end
    end
end

@testset "SR: F=3 でもループが回る" begin
    data = dimerized_mf_data(; n_flavor = 3)
    MVMCOptimizers.parton_materialize_flags!(data)
    ctx = MVMCOptimizers.serial_context()
    pstate = MVMCOptimizers.parton_build_optimization_state(data)

    rng = MVMCOptimizers.SFMT19937RNG()
    Random.seed!(rng, 31)
    @test MVMCOptimizers.parton_vmc_para_opt!(
        pstate, data, ctx; rng = rng, output_dir = mktempdir()) == 0
    @test MVMCOptimizers.assert_flavors_locked(pstate.config) === nothing
end

@testset "parton_sync_parameters!: serial では何もしない" begin
    data = dimerized_mf_data()
    MVMCOptimizers.parton_materialize_flags!(data)
    before = [t.value for t in data.pmfpara_terms]
    MVMCOptimizers.parton_sync_parameters!(data, MVMCOptimizers.serial_context())
    @test [t.value for t in data.pmfpara_terms] == before

    # D_AmpMax のリスケールを掛けないこと: 既存 sync_modified_parameter! なら
    # スレーター項に |α| = D_AmpMax の正規化が入るが、MF パラメータでは
    # H(α) が別のハミルトニアンになってしまう
    for t in data.pmfpara_terms
        t.value *= 100
    end
    scaled = [t.value for t in data.pmfpara_terms]
    MVMCOptimizers.parton_sync_parameters!(data, MVMCOptimizers.serial_context())
    @test [t.value for t in data.pmfpara_terms] == scaled
end

@testset "ドライバ: namelist から一気通貫で回る" begin
    mktempdir() do dir
        open(joinpath(dir, "pmftrans.def"), "w") do io
            println(io, "====")
            println(io, "NPartonMFTrans 16")
            println(io, "====")
            println(io, "== site1 flavor1 site2 flavor2 Re Im ==")
            println(io, "====")
            for f = 0:1, b = 0:3
                j = mod(b + 1, 4)
                if iseven(b)
                    println(io, "$b $f $j $f -1.0 0.0")     # 強ボンド
                else
                    println(io, "$b $f $j $f -1.0 0.2")     # 弱ボンド(複素)
                end
            end
            for f = 0:1, i = 0:3
                println(io, "$i $f $i $f 1.0 0.0")
            end
        end
        open(joinpath(dir, "pmfpara.def"), "w") do io
            println(io, "====")
            println(io, "NPartonMFParaIdx 3")
            println(io, "ComplexType 1")
            println(io, "====")
            println(io, "====")
            for f = 0:1, b = 0:3
                j = mod(b + 1, 4)
                if iseven(b)
                    println(io, "$b $f $j $f 0 1.0 0.0")
                else
                    println(io, "$b $f $j $f 1 0.7 0.1")
                end
            end
            for f = 0:1, i = 0:3
                println(io, "$i $f $i $f 2 0.0 0.0")
            end
            println(io, "0 0")    # ゲージ代表を凍結
            println(io, "1 1")
            println(io, "2 1")
        end
        open(joinpath(dir, "physhop.def"), "w") do io
            println(io, "====")
            println(io, "NPhysHop 4")
            println(io, "====")
            println(io, "== site1 site2 Re Im ==")
            println(io, "====")
            for i = 0:3
                println(io, "$i $(mod(i + 1, 4)) -1.0 0.0")
            end
        end
        open(joinpath(dir, "coulombinter.def"), "w") do io
            println(io, "====")
            println(io, "NCoulombInter 4")
            println(io, "====")
            println(io, "== CoulombInter ==")
            println(io, "====")
            for i = 0:3
                println(io, "$i $(mod(i + 1, 4)) 0.5")
            end
        end
        open(joinpath(dir, "modpara.def"), "w") do io
            for line in [
                "CDataFileHead zvo",
                "CParaFileHead zqp",
                "NVMCCalMode 0",
                "Nsite 4",
                "NElec 2",
                "PartonMode 1",
                "NFlavor 2",
                "PartonBlockUpdateSize 8",
                "NVMCWarmUp 50",
                "NVMCInterval 1",
                "NVMCSample 200",
                "NSROptItrStep 2",
                "NSROptItrSmp 1",
                "DSROptStepDt 0.02",
                "DSROptStaDel 0.02",
                "DSROptRedCut 1e-8",
                "ComplexType 1",
                "2Sz 0",
                "NExUpdatePath 6",
                "RndSeed 11272",
            ]
                println(io, line)
            end
        end
        write(joinpath(dir, "namelist.def"), """
            ModPara        modpara.def
            PartonMFTrans  pmftrans.def
            PartonMFPara   pmfpara.def
            PhysHop        physhop.def
            CoulombInter   coulombinter.def
            """)

        out = mktempdir()
        status = MVMCOptimizers.parton_run_para_opt_from_namelist(
            joinpath(dir, "namelist.def"); output_dir = out)
        @test status == 0
        @test !isempty(readdir(out))    # SR の出力が書かれていること
    end
end
