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
include(joinpath(PHYSICS_DIR, "checkerboard_model.jl"))
include(joinpath(PHYSICS_DIR, "parton_fixture.jl"))

@testset "Parton physics validation (P layer)" begin
    # P2 は外部 ED データに依存しない(格子の対称性と射影の代数だけを見る)ので、
    # ED ダンプが無い環境でも走らせる。
    @testset "P2 QP construction" begin
        include(joinpath(PHYSICS_DIR, "test_p2_qp_translation.jl"))
    end

    # fixture の向き正準化(2026-08-18)。ED 非依存なので常に走らせる。
    @testset "fixture orientation" begin
        include(joinpath(PHYSICS_DIR, "test_fixture_orientation.jl"))
    end

    # アンザッツ変種(flavor_groups / graph = :full)。ED 非依存。
    @testset "ansatz variants" begin
        include(joinpath(PHYSICS_DIR, "test_ansatz_variants.jl"))
    end

    if !isdir(ED_CASE_BOSON_NU12) || !isdir(ED_CASE_FERMION_NU13)
        @warn """外部 ED データが見つからないので P0/P1 をスキップします。
                 boson:   $ED_CASE_BOSON_NU12
                 fermion: $ED_CASE_FERMION_NU13"""
    else
        @testset "P0 ED reference" begin
            include(joinpath(PHYSICS_DIR, "test_p0_ed_reference.jl"))
        end
        @testset "P1 model conventions" begin
            include(joinpath(PHYSICS_DIR, "test_p1_onebody.jl"))
        end
    end
end
