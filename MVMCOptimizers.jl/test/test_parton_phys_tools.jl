"""
物理量測定用の def 生成の道具 `tools/parton_gen_green_def.jl`(DESIGN_parton.md §13.2)
--- parton-mode (fork addition) ---

生成した greenone / greentwo が上流のパーサで読め、全 (i, j) の 1 体と全 (i, i, j, j) の密度相関に
なっていること。
"""

using Test
using MVMCExpertModeParsers
using MVMCOptimizers

# 道具は parton_phys_test_helpers.jl が include 済み(parton_write_green_defs)

@testset "§13.2 greenone / greentwo の生成(Nsite = 4)" begin
    dir = mktempdir()
    parton_write_green_defs(dir, 4)

    one = MVMCExpertModeParsers.parse_green_one_def(joinpath(dir, "greenone.def"))
    @test one.success
    @test length(one.data) == 16
    @test all(t -> t.spin1 == :up && t.spin2 == :up, one.data)
    @test Set((t.site1, t.site2) for t in one.data) == Set((i, j) for i = 0:3 for j = 0:3)

    two = MVMCExpertModeParsers.parse_green_two_def(joinpath(dir, "greentwo.def"))
    @test two.success
    @test length(two.data) == 16
    @test all(t -> all(==(:up), (t.spin1, t.spin2, t.spin3, t.spin4)), two.data)
    @test all(t -> MVMCOptimizers.parton_green2_class(t.site1, t.site2, t.site3, t.site4) ==
                   :density, two.data)
    @test Set((t.site1, t.site3) for t in two.data) == Set((i, j) for i = 0:3 for j = 0:3)
end
