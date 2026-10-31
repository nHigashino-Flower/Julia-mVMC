"""
Pfaffian パートンモード(PartonMode = 2)の対軌道 def のパーサ
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §2.1。書式は上流の orbitalidxgen.def(OrbitalGeneral)と同じで、
spin の列をフレーバーと読む。パーサは書式だけを厳格に見る(範囲・対角・重複・連番は
テンプレート build の担当)。
"""

using Test
using MVMCExpertModeParsers

const _PAIR_HEADER = """
=============================================
NOrbitalIdx          3
ComplexType          1
=============================================
=============================================
"""

@testset "PartonPairOrbital: 5 列・6 列の行と末尾の OptFlag 行" begin
    content = _PAIR_HEADER * """
        0  0      1  1      0
        1  0      0  1      1  -1
        2  1      0  0      2   1
    0 1
    1 0
    2 1
    """
    result, flags, declared = MVMCExpertModeParsers.parse_parton_pair_orbital_content(content)
    @test result.success
    terms = result.data
    @test length(terms) == 3
    @test (terms[1].site1, terms[1].flavor1, terms[1].site2, terms[1].flavor2) == (0, 0, 1, 1)
    @test terms[1].idx == 0 && terms[1].sign == 1            # 符号列の省略は +1
    @test terms[2].sign == -1
    @test (terms[3].site1, terms[3].flavor1, terms[3].site2, terms[3].flavor2, terms[3].idx) ==
          (2, 1, 0, 0, 2)
    @test flags == Dict(0 => 1, 1 => 0, 2 => 1)
    @test declared == 3
end

@testset "PartonPairOrbital: 書式の違反はエラー" begin
    bad_rows = (
        "0 0 1 1",              # 4 列
        "0 0 1 1 0 1 7",        # 7 列
        "0 0 1 x 0",            # 整数でない
        "0 0 1 1 0 2",          # 符号が ±1 でない
        "0 0 1 1 -1",           # idx が負
    )
    for row in bad_rows
        result, _, _ = MVMCExpertModeParsers.parse_parton_pair_orbital_content(
            _PAIR_HEADER * row * "\n")
        @test !result.success
    end
    # OptFlag 行が壊れている(数字で始まらない行はヘッダとして読み飛ばす家風なので、数字で始める)
    result, _, _ = MVMCExpertModeParsers.parse_parton_pair_orbital_content(
        _PAIR_HEADER * "0 0 1 1 0\n1 0 0 1 1\n2 1 0 0 2\n1 y\n")
    @test !result.success
    # 同じ idx の OptFlag が 2 回
    result, _, _ = MVMCExpertModeParsers.parse_parton_pair_orbital_content(
        _PAIR_HEADER * "0 0 1 1 0\n1 0 0 1 1\n2 1 0 0 2\n1 1\n1 0\n")
    @test !result.success
    # 宣言数と使われている idx の数が食い違う
    result, _, _ = MVMCExpertModeParsers.parse_parton_pair_orbital_content(
        _PAIR_HEADER * "0 0 1 1 0\n1 0 0 1 1\n")
    @test !result.success
    # 複素パラメータのみ(ComplexType 0 は未対応)
    real_header = replace(_PAIR_HEADER, "ComplexType          1" => "ComplexType          0")
    result, _, _ = MVMCExpertModeParsers.parse_parton_pair_orbital_content(
        real_header * "0 0 1 1 0\n1 0 0 1 1\n2 1 0 0 2\n")
    @test !result.success
end

@testset "PartonPairOrbital: 上流の orbitalidxgen.def がそのまま読める(K = 2)" begin
    path = joinpath(@__DIR__, "..", "..", "examples", "inputs", "heisenberg_chain_fsz",
                    "orbitalidxgen.def")
    result, flags, declared = MVMCExpertModeParsers.parse_parton_pair_orbital_def(path)
    @test result.success
    terms = result.data
    @test declared == maximum(t.idx for t in terms) + 1
    @test all(t -> t.flavor1 in (0, 1) && t.flavor2 in (0, 1), terms)
    @test all(t -> t.sign in (1, -1), terms)
    @test length(flags) == declared
end

@testset "PartonPairOrbital: parse_expert_mode_files への登録とパラメータの数" begin
    mktempdir() do dir
        write(joinpath(dir, "modpara.def"), """
        --------------------
        Model_Parameters   0
        --------------------
        NVMCCalMode    0
        --------------------
        CDataFileHead  zvo
        CParaFileHead  zqp
        --------------------
        Nsite          3
        NElec          2
        PartonMode     2
        NFlavor        2
        """)
        write(joinpath(dir, "pair.def"), _PAIR_HEADER * """
        0 0 1 1 0
        1 0 0 1 1
        2 1 0 0 2
        0 1
        1 1
        2 0
        """)
        write(joinpath(dir, "namelist.def"), """
        ModPara            modpara.def
        PartonPairOrbital  pair.def
        """)
        data = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(dir, "namelist.def"))
        @test length(data.parton_pair_terms) == 3
        @test data.parton_pair_opt_flags == Dict(0 => 1, 1 => 1, 2 => 0)
        @test data.parton_pair_params == zeros(ComplexF64, 3)
        # SR の並びに f のブロックが数えられる(射影なし)
        @test MVMCExpertModeParsers.count_variational_parameters(data) == 3
    end
end

@testset "PartonPairOrbital: 既定は空" begin
    data = MVMCExpertModeParsers.ExpertModeData()
    @test isempty(data.parton_pair_terms)
    @test isempty(data.parton_pair_opt_flags)
    @test isempty(data.parton_pair_params)
end
