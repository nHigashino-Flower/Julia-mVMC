"""
P0: 外部 ED 結果の読み取りと台帳化
--- parton-mode (fork addition) ---

DESIGN_parton.md §8 P 層の P0。メタデータ(格子・境界条件・統計・準位)を
読み出し、準縮退多様体の最低値 E_min を抽出する。

期待値はすべて外部 ED のログに書かれている値そのもの。テストを緑にするために
値を動かさないこと(P 層の絶対規則)。
"""

using Test

@testset "P0-a ν=1/2 ボソン(4×4 ユニットセル・32 サイト・8 粒子)" begin
    ref = read_ed_reference(ED_CASE_BOSON_NU12)
    print(ed_ledger(ref))

    @test ref.nx == 4
    @test ref.ny == 4
    @test ref.nsite == 32
    @test ref.nsite == 2 * ref.nx * ref.ny
    @test ref.nelec == 8
    @test ref.statistics == "Boson"
    @test ref.boundary == "periodic"
    # 最低 Chern バンドの ν = N / (Nx*Ny) = 8/16 = 1/2
    @test ref.nelec // (ref.nx * ref.ny) == 1 // 2

    @test ref.t == 1.0
    @test isapprox(ref.t1, 0.2928932188134525; atol = 1e-15)
    @test isapprox(ref.t2, -0.2928932188134525; atol = 1e-15)
    @test isapprox(ref.t3, 0.20710678118654754; atol = 1e-15)
    @test isapprox(ref.psi, 0.7853981633974483; atol = 1e-15)   # π/4
    @test ref.phi == 0.0 && ref.xi == 0.0 && ref.eta == 0.0     # ゲージ因子 J = 1
    @test ref.u == 0.0 && ref.v == 0.0                          # 硬芯のみ
    @test ref.random_potential_max == 0.0                       # r=0 → 乱雑ポテンシャルなし

    e_min, spread, gap = ed_ground_manifold(ref, 2)             # 2 重準縮退
    @test e_min == -16.304913354429445
    @test isapprox(spread, 0.005501242337331; atol = 1e-12)     # 多様体内の分裂
    @test gap > 10 * spread                                     # 多様体は外と分離している
end

@testset "P0-b ν=1/3 フェルミオン(5×3 ユニットセル・30 サイト・5 粒子)" begin
    ref = read_ed_reference(ED_CASE_FERMION_NU13)
    print(ed_ledger(ref))

    @test ref.nx == 5
    @test ref.ny == 3
    @test ref.nsite == 30
    @test ref.nsite == 2 * ref.nx * ref.ny
    @test ref.nelec == 5
    @test ref.statistics == "Fermion"
    @test ref.boundary == "periodic"
    # ν = N / (Nx*Ny) = 5/15 = 1/3
    @test ref.nelec // (ref.nx * ref.ny) == 1 // 3

    @test ref.t == 1.0
    @test isapprox(ref.t1, 0.2928932188134525; atol = 1e-15)
    @test isapprox(ref.t2, -0.2928932188134525; atol = 1e-15)
    @test isapprox(ref.t3, 0.20710678118654754; atol = 1e-15)
    @test isapprox(ref.psi, 0.7853981633974483; atol = 1e-15)
    @test ref.phi == 0.0 && ref.xi == 0.0 && ref.eta == 0.0
    @test ref.u == 1.0 && ref.v == 0.0                          # NN 斥力のみ

    # r=1.0e8 の乱雑ポテンシャルは縮退を割るためだけの摂動で 1e-8 のオーダー。
    # VMC 側では入れないので、その分だけ E_min がずれうることを台帳に記録する。
    @test ref.random_potential_max < 1e-7

    e_min, spread, gap = ed_ground_manifold(ref, 3)             # 3 重準縮退
    @test e_min == -10.104174670830064
    @test isapprox(spread, 0.021663284299546; atol = 1e-12)
    @test gap > 0                                               # 4 番目は多様体の外
end

if !isdir(ED_CASE_FERMION_NU13_6X3)
    @warn "6×3 の ED データが無いので P0-d をスキップします: $ED_CASE_FERMION_NU13_6X3"
else
    @testset "P0-d ν=1/3 フェルミオン(6×3 ユニットセル・36 サイト・6 粒子)" begin
        ref = read_ed_reference(ED_CASE_FERMION_NU13_6X3)
        print(ed_ledger(ref))

        @test ref.nx == 6
        @test ref.ny == 3
        @test ref.nsite == 36
        @test ref.nsite == 2 * ref.nx * ref.ny
        @test ref.nelec == 6
        @test ref.statistics == "Fermion"
        @test ref.boundary == "periodic"
        @test ref.nelec // (ref.nx * ref.ny) == 1 // 3

        @test ref.t == 1.0
        @test isapprox(ref.t1, 0.2928932188134525; atol = 1e-15)
        @test isapprox(ref.t2, -0.2928932188134525; atol = 1e-15)
        @test isapprox(ref.t3, 0.20710678118654754; atol = 1e-15)
        @test isapprox(ref.psi, 0.7853981633974483; atol = 1e-15)
        @test ref.phi == 0.0 && ref.xi == 0.0 && ref.eta == 0.0
        @test ref.u == 1.0 && ref.v == 0.0                          # NN 斥力のみ

        # 乱雑ポテンシャルなし(4×4 ボゾン参照と同じ条件)
        @test ref.random_potential_max == 0.0

        e_min, spread, gap = ed_ground_manifold(ref, 3)             # 3 重準縮退
        @test isapprox(e_min, -12.126195092720709; atol = 1e-12)
        @test isapprox(spread, 0.025446002588616068; atol = 1e-12)
        @test gap > spread                                          # 多様体の外が離れている
    end
end

@testset "P0-c パーサの頑健性" begin
    ref = read_ed_reference(ED_CASE_FERMION_NU13)
    # ディレクトリ指定とファイル直接指定で同じ結果
    @test read_ed_reference(ref.path).energies == ref.energies
    # 準位は昇順
    @test issorted(ref.energies)
    @test length(ref.energies) >= 10
    # 存在しないパスは明示的に失敗する
    @test_throws Exception read_ed_reference(joinpath(ED_CASE_FERMION_NU13, "nope"))
end
