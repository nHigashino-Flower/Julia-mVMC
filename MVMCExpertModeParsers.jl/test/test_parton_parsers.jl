"""
パートンモード用パーサのテスト --- parton-mode (fork addition) ---

DESIGN_parton.md §2(入力契約)に対応する。
"""

using Test
using MVMCExpertModeParsers

@testset "modpara: パートンモードのキー" begin
    content = """
    --------------------
    CDataFileHead  zvo
    CParaFileHead  zqp
    --------------------
    NVMCCalMode    0
    Nsite          4
    NElec          2
    PartonMode     1
    NFlavor        3
    PartonBlockUpdateSize 8
    """
    result = MVMCExpertModeParsers.parse_modpara_content(content)
    @test result.success
    p = result.data
    @test p.parton_mode == 1
    @test p.nflavor == 3
    @test p.parton_block_update_size == 8
    @test p.nelec == 2
end

@testset "modpara: パートンキーの既定値" begin
    p = MVMCExpertModeParsers.ModParaParameters()
    @test p.parton_mode == 0
    @test p.nflavor == 0
    @test p.parton_block_update_size == 16
end

@testset "modpara: NParticle / NPartonPerFlavor は NElec の別名" begin
    base = "Nsite 4\n"
    r1 = MVMCExpertModeParsers.parse_modpara_content(base * "NParticle 3\n")
    @test r1.success
    @test r1.data.nelec == 3

    r2 = MVMCExpertModeParsers.parse_modpara_content(base * "NPartonPerFlavor 3\n")
    @test r2.success
    @test r2.data.nelec == 3

    # 同値なら重複してよい
    r3 = MVMCExpertModeParsers.parse_modpara_content(base * "NElec 3\nNParticle 3\n")
    @test r3.success
    @test r3.data.nelec == 3

    # 食い違いはパースエラー(綴りの取り違えを黙って通さない)
    r4 = MVMCExpertModeParsers.parse_modpara_content(base * "NElec 3\nNParticle 2\n")
    @test !r4.success
end

@testset "modpara: 標準モードでは既定値のまま" begin
    result = MVMCExpertModeParsers.parse_modpara_content("Nsite 4\nNElec 2\n")
    @test result.success
    @test result.data.parton_mode == 0
    @test result.data.nflavor == 0
end

@testset "pmftrans.def パーサ(6 列)" begin
    content = """
    ====================
    NPartonMFTrans 3
    ====================
    == site1 flavor1 site2 flavor2 Re Im ==
    ====================
    0 0 1 0  -1.0  0.5
    1 1 2 1  -1.0  0.0
    0 0 0 0   0.3  0.0
    """
    r = MVMCExpertModeParsers.parse_parton_mf_trans_content(content)
    @test r.success
    ts = r.data
    @test length(ts) == 3
    @test ts[1].site1 == 0 && ts[1].flavor1 == 0 && ts[1].site2 == 1 && ts[1].flavor2 == 0
    @test ts[1].value == ComplexF64(-1.0, 0.5)
    @test ts[1].is_complex
    @test ts[2].flavor1 == 1 && ts[2].flavor2 == 1
    @test ts[3].site1 == 0 && ts[3].site2 == 0
    @test !ts[3].is_complex

    # 宣言個数と実行数の不一致はエラー
    bad = replace(content, "NPartonMFTrans 3" => "NPartonMFTrans 4")
    @test !MVMCExpertModeParsers.parse_parton_mf_trans_content(bad).success
end

@testset "pmfpara.def パーサ(7 列+末尾フラグ行)" begin
    content = """
    =============================================
    NPartonMFParaIdx  2
    ComplexType       1
    =============================================
    =============================================
    0 0 1 0  0  -1.0  0.0
    1 1 2 1  1  -1.0  0.25
    0 0 0 0  0  -1.0  0.0
    0 1
    1 0
    """
    result, flags, declared = MVMCExpertModeParsers.parse_parton_mf_para_content(content)
    @test result.success
    ts = result.data
    @test length(ts) == 3
    @test ts[1].idx == 0
    @test ts[1].value == ComplexF64(-1.0, 0.0)
    @test ts[1].is_complex
    @test ts[2].idx == 1
    @test ts[2].value == ComplexF64(-1.0, 0.25)
    @test ts[3].idx == 0   # idx 共有(value の一致検証は build / 門番の担当)
    @test flags == Dict(0 => 1, 1 => 0)
    @test declared == 2
end

@testset "physhop.def パーサ" begin
    content = """
    ==================
    NPhysHop 2
    ==================
    == site1 site2 Re Im ==
    ==================
    0 1  -1.0  0.2
    1 2  -1.0  0.0
    """
    r = MVMCExpertModeParsers.parse_physhop_content(content)
    @test r.success
    @test length(r.data) == 2
    @test r.data[1].site1 == 0 && r.data[1].site2 == 1
    @test r.data[1].value == ComplexF64(-1.0, 0.2)
    @test r.data[1].is_complex
    @test !r.data[2].is_complex

    bad = replace(content, "NPhysHop 2" => "NPhysHop 3")
    @test !MVMCExpertModeParsers.parse_physhop_content(bad).success
end

@testset "namelist 経由での取り込み(parse_file_by_type!)" begin
    mktempdir() do dir
        write(joinpath(dir, "pmftrans.def"),
              "====\nNPartonMFTrans 2\n====\n====\n====\n0 0 1 0 -1.0 0.0\n0 1 1 1 -1.0 0.0\n")
        write(joinpath(dir, "pmfpara.def"),
              "====\nNPartonMFParaIdx 1\nComplexType 1\n====\n====\n" *
              "0 0 1 0 0 -1.0 0.0\n0 1 1 1 0 -1.0 0.0\n0 1\n")
        write(joinpath(dir, "physhop.def"),
              "====\nNPhysHop 1\n====\n====\n====\n0 1 -1.0 0.0\n")
        write(joinpath(dir, "modpara.def"),
              "Nsite 4\nNElec 2\nPartonMode 1\nNFlavor 2\n")
        write(joinpath(dir, "namelist.def"), """
            ModPara        modpara.def
            PartonMFTrans  pmftrans.def
            PartonMFPara   pmfpara.def
            PhysHop        physhop.def
            """)
        data = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(dir, "namelist.def"))
        @test length(data.pmftrans_terms) == 2
        @test length(data.pmfpara_terms) == 2
        @test length(data.physhop_terms) == 1
        @test data.pmfpara_opt_flags == Dict(0 => 1)
        @test data.modpara.parton_mode == 1
        @test data.modpara.nflavor == 2
    end
end
