"""
物理検証テスト(P 層)のランナー
--- parton-mode (fork addition) ---

DESIGN_parton.md §8 の P 層。外部 ED 結果との突き合わせ・変分上界・overlap・
MC 収束を見る。既定の `Pkg.test()` からは**分離**してある(実行時間と外部データ
依存のため)。走らせるときはワークスペース root から:

    julia --project=@. test/physics/runtests.jl

外部 ED データが無い環境ではスキップする(CI では走らない想定)。
"""

using Test

const PHYSICS_DIR = @__DIR__

include(joinpath(PHYSICS_DIR, "ed_reference.jl"))

if !isdir(ED_CASE_BOSON_NU12) || !isdir(ED_CASE_FERMION_NU13)
    @warn """外部 ED データが見つからないので P 層をスキップします。
             boson:   $ED_CASE_BOSON_NU12
             fermion: $ED_CASE_FERMION_NU13"""
else
    @testset "Parton physics validation (P layer)" begin
        @testset "P0 ED reference" begin
            include(joinpath(PHYSICS_DIR, "test_p0_ed_reference.jl"))
        end
    end
end
