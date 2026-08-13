"""
SR パラメータロケータの PMF ブロックのテスト --- parton-mode (fork addition) ---

DESIGN_parton.md §3.1 の登録点 5(案 B: ロケータ拡張、RBM 前例)に対応する。
共有 idx は ORBITAL パターン同様「全行訪問・絶対 set は冪等・δ は各行の自値+δ」。
"""

using Test
using MVMCExpertModeParsers
using MVMCOptimizers

"idx 0 を 2 行で共有(フレーバー跨ぎ)、idx 1 は 1 行のトイ入力。"
function _toy_pmf_data()
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite = 4
    data.modpara.nelec = 2
    data.modpara.nflavor = 2
    data.modpara.parton_mode = 1
    push!(
        data.pmfpara_terms,
        MVMCExpertModeParsers.PartonMFParaTerm(0, 0, 1, 0, 0, ComplexF64(-1.0, 0.0), true),
        MVMCExpertModeParsers.PartonMFParaTerm(0, 1, 1, 1, 0, ComplexF64(-1.0, 0.0), true),
        MVMCExpertModeParsers.PartonMFParaTerm(0, 0, 0, 0, 1, ComplexF64(0.3, 0.0), true),
    )
    return data
end

@testset "count_variational_parameters が n_pmf を数える" begin
    data = _toy_pmf_data()
    # 射影 0 + RBM 0 + orbital 0 + opttrans 0 + MF 2
    @test MVMCExpertModeParsers.count_variational_parameters(data) == 2
end

@testset "pack/unpack roundtrip(共有 idx の冪等性)" begin
    data = _toy_pmf_data()
    para = MVMCOptimizers.pack_parameters(data)
    @test length(para) == 2
    @test para[1] == ComplexF64(-1.0, 0.0)
    @test para[2] == ComplexF64(0.3, 0.0)

    para2 = copy(para)
    para2[1] = ComplexF64(2.0, -0.5)
    MVMCOptimizers.unpack_parameters!(data, para2)
    @test data.pmfpara_terms[1].value == ComplexF64(2.0, -0.5)
    @test data.pmfpara_terms[2].value == ComplexF64(2.0, -0.5)  # 共有 idx は全行更新
    @test data.pmfpara_terms[3].value == ComplexF64(0.3, 0.0)

    # roundtrip
    @test MVMCOptimizers.pack_parameters(data) == para2
end

@testset "get / set / delta(_at 版のロケータ)" begin
    data = _toy_pmf_data()
    @test MVMCOptimizers.get_parameter_value(data, 1) == ComplexF64(-1.0, 0.0)
    @test MVMCOptimizers.get_parameter_value(data, 2) == ComplexF64(0.3, 0.0)

    MVMCOptimizers.set_parameter_value!(data, 2, ComplexF64(9.0, 1.0))
    @test data.pmfpara_terms[3].value == ComplexF64(9.0, 1.0)

    # 絶対 set は共有 idx の全行に同値を書く(冪等)
    MVMCOptimizers.set_parameter_value!(data, 1, ComplexF64(5.0, 0.0))
    @test data.pmfpara_terms[1].value == ComplexF64(5.0, 0.0)
    @test data.pmfpara_terms[2].value == ComplexF64(5.0, 0.0)

    # δ 加算は各行の自値 + δ(共有行は同値なので結果も同値)
    MVMCOptimizers._add_parameter_delta_direct!(data, 1, ComplexF64(0.5, 0.5))
    @test data.pmfpara_terms[1].value == ComplexF64(5.5, 0.5)
    @test data.pmfpara_terms[2].value == ComplexF64(5.5, 0.5)
end

@testset "標準モード(pmfpara 空)では挙動不変" begin
    data = MVMCExpertModeParsers.ExpertModeData()
    @test MVMCExpertModeParsers.count_variational_parameters(data) == 0
    @test MVMCOptimizers.pack_parameters(data) == ComplexF64[]
    @test MVMCOptimizers.get_parameter_value(data, 1) == ComplexF64(0)
end
