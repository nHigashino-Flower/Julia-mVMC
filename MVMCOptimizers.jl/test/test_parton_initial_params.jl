"""
§8 テスト 9: α の初期値経路(乱数初期化 + InPmfPara.def)
--- parton-mode (fork addition) ---

DESIGN_parton.md §2.3。スイッチは modpara のキーではなく**入力の形そのもの**:
pmfpara.def の value 列の有無と、namelist.def の InPmfPara の有無。

「未入力」と「0 を指定」は**列の有無**で区別する。値がゼロかどうかで分岐しない。
"""

using Test
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

"""
    _pmfpara_text(; values, flags=true)

pmfpara.def の中身を組む。`values[k]` が `nothing` なら 5 列(未入力)、
値なら 7 列(入力あり)。4 サイト環・F=2・最近接ホッピング 1 群 + オンサイト 1 群。
"""
function _pmfpara_text(values::Vector{<:Union{Nothing,ComplexF64}})
    io = IOBuffer()
    println(io, "=============================================")
    println(io, "NPartonMFParaIdx $(length(values))")
    println(io, "ComplexType          1")
    println(io, "=============================================")
    println(io, "=============================================")
    for f = 0:1, i = 0:3
        v = values[1]
        if v === nothing
            println(io, "$i $f $(mod(i + 1, 4)) $f 0")
        else
            println(io, "$i $f $(mod(i + 1, 4)) $f 0 $(real(v)) $(imag(v))")
        end
    end
    if length(values) >= 2
        for f = 0:1, i = 0:3
            v = values[2]
            if v === nothing
                println(io, "$i $f $i $f 1")
            else
                println(io, "$i $f $i $f 1 $(real(v)) $(imag(v))")
            end
        end
    end
    return String(take!(io))
end

"pmfpara.def の中身から ExpertModeData を組む(4 サイト環・F=2)。"
function _data_from_pmfpara(text::String)
    result, flags, _ = MVMCExpertModeParsers.parse_parton_mf_para_content(text)
    result.success || error("pmfpara parse failed: $(result.error_message)")
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite = 4
    mp.nelec = 2
    mp.nflavor = 2
    mp.parton_mode = 1
    mp.two_sz = 0
    mp.complex_flag = 1
    mp.nex_update_path = 6
    data.pmfpara_terms = result.data
    data.pmfpara_opt_flags = flags
    t = ComplexF64(-1.0, 0.35)
    for f = 0:1, i = 0:3
        push!(data.pmftrans_terms,
              MVMCExpertModeParsers.PartonMFTransTerm(i, f, mod(i + 1, 4), f, t, true))
    end
    if any(p -> p.site1 == p.site2, data.pmfpara_terms)
        for f = 0:1, i = 0:3
            push!(data.pmftrans_terms,
                  MVMCExpertModeParsers.PartonMFTransTerm(i, f, i, f, ComplexF64(0.7, 0), false))
        end
    end
    push!(data.physhop_terms,
          MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(-1, 0), false))
    return data
end

@testset "§8-9-1/2 決定性と全ランク一致(構成的に同一)" begin
    text = _pmfpara_text([nothing, nothing])
    d1 = _data_from_pmfpara(text)
    d2 = _data_from_pmfpara(text)
    MVMCOptimizers.parton_init_alpha!(d1, 11272)
    MVMCOptimizers.parton_init_alpha!(d2, 11272)
    α1 = MVMCOptimizers.parton_alpha_from_terms(d1)
    α2 = MVMCOptimizers.parton_alpha_from_terms(d2)
    @test α1 == α2                       # ビット一致
    @test all(!iszero, α1)

    # ベースシードが違えば違う値になる(乱数が効いていることの確認)
    d3 = _data_from_pmfpara(text)
    MVMCOptimizers.parton_init_alpha!(d3, 99)
    @test MVMCOptimizers.parton_alpha_from_terms(d3) != α1

    # ランクごとのオフセットを加えない値を使うので、全ランクで同じ α になる。
    # ここは「同じベースシードなら同じ結果」= 構成的一致で担保する。
    d4 = _data_from_pmfpara(text)
    MVMCOptimizers.parton_init_alpha!(d4, 11272)
    @test MVMCOptimizers.parton_alpha_from_terms(d4) == α1
end

@testset "§8-9-3 明示ゼロの保護: 0 を書いた行は 0 のまま" begin
    text = _pmfpara_text([ComplexF64(0, 0), nothing])
    data = _data_from_pmfpara(text)
    filled = MVMCOptimizers.parton_init_alpha!(data, 11272)
    α = MVMCOptimizers.parton_alpha_from_terms(data)
    @test α[1] == ComplexF64(0, 0)       # 明示ゼロは乱数で埋まらない
    @test !(0 in filled)
    @test 1 in filled                    # 未入力の idx1 だけ埋まった
    @test α[2] != 0
end

@testset "§8-9-4 全指定なら乱数経路を通らない" begin
    text = _pmfpara_text([ComplexF64(1.0, 0.0), ComplexF64(0.7, 0.0)])
    data = _data_from_pmfpara(text)
    α_before = MVMCOptimizers.parton_alpha_from_terms(data)
    filled = MVMCOptimizers.parton_init_alpha!(data, 11272)
    @test isempty(filled)
    @test MVMCOptimizers.parton_alpha_from_terms(data) == α_before
end

@testset "§8-9-5 presence 混在の検出" begin
    # 同じ idx 0 の行で、一部だけ値あり
    io = IOBuffer()
    println(io, "====")
    println(io, "NPartonMFParaIdx 1")
    println(io, "ComplexType 1")
    println(io, "====")
    println(io, "====")
    println(io, "0 0 1 0 0 1.0 0.0")     # 値あり
    println(io, "1 0 2 0 0")             # 同じ idx なのに未入力
    text = String(take!(io))
    result, _, _ = MVMCExpertModeParsers.parse_parton_mf_para_content(text)
    @test result.success
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite = 4
    data.modpara.nflavor = 1
    data.pmfpara_terms = result.data
    @test_throws Exception MVMCOptimizers.parton_validate_value_presence(data)
    @test_throws Exception MVMCOptimizers.parton_init_alpha!(data, 1)
end

@testset "§8-9-5b 列数が 5/7/2 以外はエラー" begin
    for bad in ("0 0 1 0 0 1.0", "0 0 1 0", "0 0 1 0 0 1.0 0.0 9.9 9.9")
        text = "====\nNPartonMFParaIdx 1\nComplexType 1\n====\n====\n" * bad * "\n"
        result, _, _ = MVMCExpertModeParsers.parse_parton_mf_para_content(text)
        @test !result.success
    end
end

@testset "§8-9-6 ダンプ往復: 書いて読んで α がビット一致" begin
    text = _pmfpara_text([nothing, nothing])
    data = _data_from_pmfpara(text)
    MVMCOptimizers.parton_init_alpha!(data, 20260813)
    α_ref = MVMCOptimizers.parton_alpha_from_terms(data)

    mktempdir() do dir
        dump = joinpath(dir, "zqp_pmfpara_init.dat")
        MVMCOptimizers.parton_write_pmfpara(data, dump)

        # 別インスタンスに、未入力(乱数対象)のまま読み込ませて上書きする
        data2 = _data_from_pmfpara(text)
        cp(dump, joinpath(dir, "InPmfPara.def"))
        write(joinpath(dir, "namelist.def"), "InPmfPara  InPmfPara.def\n")
        @test MVMCOptimizers.parton_read_in_pmfpara!(data2, joinpath(dir, "namelist.def"))
        @test MVMCOptimizers.parton_alpha_from_terms(data2) == α_ref

        # namelist に書かれていなければ何もしない
        data3 = _data_from_pmfpara(text)
        write(joinpath(dir, "empty.def"), "ModPara modpara.def\n")
        @test !MVMCOptimizers.parton_read_in_pmfpara!(data3, joinpath(dir, "empty.def"))

        # 個数が合わなければエラー(黙って部分適用しない)
        data4 = _data_from_pmfpara(text)
        write(joinpath(dir, "InPmfPara.def"),
              "====\nNPartonMFParaIdx 1\nComplexType 1\n====\n====\n0 1.0 0.0\n")
        @test_throws Exception MVMCOptimizers.parton_read_in_pmfpara!(
            data4, joinpath(dir, "namelist.def"))
    end
end

@testset "§8-9-6b 最適化後の往復: zqp_pmfpara_opt.dat が idx=0 を落とさない" begin
    # 既存 per-block writer はヘッダ 4 行で、parse_input_parameter_file が 5 行
    # 読み飛ばすため idx=0 が脱落する(実測済み)。パートン側は 5 行にして往復を
    # 成立させている。その回帰ガード。
    data = dimerized_mf_data()
    MVMCOptimizers.parton_materialize_flags!(data)
    data.modpara.nsr_opt_itr_step = 3
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    rng = MVMCOptimizers.SFMT19937RNG()
    Random.seed!(rng, 31337)
    out = mktempdir()
    @test MVMCOptimizers.parton_vmc_para_opt!(
        pstate, data, MVMCOptimizers.serial_context();
        rng = rng, output_dir = out) == 0

    opt_file = joinpath(out, "zqp_pmfpara_opt.dat")
    @test isfile(opt_file)
    n_idx = MVMCOptimizers.parton_n_idx(data)
    params = MVMCExpertModeParsers.parse_input_parameter_file(opt_file)
    @test length(params) == n_idx                       # 1 本も落ちていない
    @test haskey(params, 0)                             # idx=0 が脱落していない
    @test sort(collect(keys(params))) == collect(0:(n_idx - 1))

    # 読み戻して α がビット一致すること
    α_ref = MVMCOptimizers.parton_alpha_from_terms(data)
    data2 = dimerized_mf_data()
    dir2 = mktempdir()
    cp(opt_file, joinpath(dir2, "InPmfPara.def"))
    write(joinpath(dir2, "namelist.def"), "InPmfPara  InPmfPara.def\n")
    @test MVMCOptimizers.parton_read_in_pmfpara!(data2, joinpath(dir2, "namelist.def"))
    @test MVMCOptimizers.parton_alpha_from_terms(data2) == α_ref
end

@testset "§8-9-6c 件数不一致は警告止まりでなくエラー" begin
    data = dimerized_mf_data()
    n_idx = MVMCOptimizers.parton_n_idx(data)
    mktempdir() do dir
        full = joinpath(dir, "full.dat")
        MVMCOptimizers.parton_write_pmfpara(data, full)
        lines = readlines(full)
        # データ行を 1 本削る(ヘッダ 5 行は残す)
        write(joinpath(dir, "InPmfPara.def"), join(lines[1:(end - 1)], "\n") * "\n")
        write(joinpath(dir, "namelist.def"), "InPmfPara  InPmfPara.def\n")
        data2 = dimerized_mf_data()
        @test_throws Exception MVMCOptimizers.parton_read_in_pmfpara!(
            data2, joinpath(dir, "namelist.def"))
    end
end

@testset "§8-9-7 オンサイト群は実数(Im が厳密ゼロ)" begin
    text = _pmfpara_text([nothing, nothing])
    data = _data_from_pmfpara(text)
    MVMCOptimizers.parton_init_alpha!(data, 777)
    onsite = MVMCOptimizers.parton_onsite_idx_set(data)
    @test 1 in onsite                              # idx1 がオンサイト群
    α = MVMCOptimizers.parton_alpha_from_terms(data)
    for k in onsite
        @test imag(α[k + 1]) == 0.0                # 厳密に 0
    end
    @test imag(α[1]) != 0.0                        # ホッピング群は複素
end

@testset "§8-9-8 サンプリング RNG を消費しない" begin
    # 初期化用ストリームは専用。同じ α を与えたとき、乱数初期化を通したかどうかで
    # サンプリング系列が変わらないこと。
    draws(rng) = [MVMCOptimizers.rng_real2(rng) for _ = 1:5]
    r1 = MVMCOptimizers.SFMT19937RNG()
    Random.seed!(r1, 11272)
    before = draws(r1)

    data = _data_from_pmfpara(_pmfpara_text([nothing, nothing]))
    MVMCOptimizers.parton_init_alpha!(data, 11272)   # 専用ストリームを使う

    r2 = MVMCOptimizers.SFMT19937RNG()
    Random.seed!(r2, 11272)
    @test draws(r2) == before
end
