"""
パートンモードの Term 構造体と ExpertModeData 登録点のテスト
--- parton-mode (fork addition) ---

DESIGN_parton.md §2.2/§2.3/§2.4(入力契約)と §3.1(登録点)に対応する。
"""

using Test
using MVMCExpertModeParsers

@testset "parton Term 構造体" begin
    tr = MVMCExpertModeParsers.PartonMFTransTerm(0, 1, 2, 1, ComplexF64(-1.0, 0.5), true)
    @test tr.site1 == 0 && tr.flavor1 == 1 && tr.site2 == 2 && tr.flavor2 == 1
    @test tr.value == ComplexF64(-1.0, 0.5)
    @test isbitstype(MVMCExpertModeParsers.PartonMFTransTerm)

    pa = MVMCExpertModeParsers.PartonMFParaTerm(0, 1, 2, 1, 3, ComplexF64(1.0, 0.0), true)
    @test pa.idx == 3
    pa.value = ComplexF64(2.0, 0.0)   # α の正準置き場なので可変であること
    @test pa.value == ComplexF64(2.0, 0.0)

    ph = MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(-1.0, 0.2), true)
    @test ph.site1 == 0 && ph.site2 == 1
    @test isbitstype(MVMCExpertModeParsers.PhysHopTerm)
end

@testset "ExpertModeData のパートンフィールド" begin
    data = MVMCExpertModeParsers.ExpertModeData()
    @test data.pmftrans_terms isa Vector{MVMCExpertModeParsers.PartonMFTransTerm}
    @test data.pmfpara_terms isa Vector{MVMCExpertModeParsers.PartonMFParaTerm}
    @test data.physhop_terms isa Vector{MVMCExpertModeParsers.PhysHopTerm}
    @test data.pmfpara_opt_flags isa Dict{Int,Int}
    @test isempty(data.pmftrans_terms)
    @test isempty(data.pmfpara_terms)
    @test isempty(data.physhop_terms)
    @test isempty(data.pmfpara_opt_flags)

    # pmfpara_idx_matrix は DESIGN §2.3 で廃止(結合は build 内ローカル Dict)
    @test !hasfield(MVMCExpertModeParsers.ExpertModeData, :pmfpara_idx_matrix)
end
