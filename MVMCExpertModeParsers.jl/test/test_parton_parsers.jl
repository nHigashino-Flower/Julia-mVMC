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
    NBlockUpdateSize 8
    """
    result = MVMCExpertModeParsers.parse_modpara_content(content)
    @test result.success
    p = result.data
    @test p.parton_mode == 1
    @test p.nflavor == 3
    @test p.nblock_update_size == 8
    @test p.nelec == 2
end

@testset "modpara: パートンキーの既定値" begin
    p = MVMCExpertModeParsers.ModParaParameters()
    @test p.parton_mode == 0
    @test p.nflavor == 0
    @test p.nblock_update_size == 16
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
