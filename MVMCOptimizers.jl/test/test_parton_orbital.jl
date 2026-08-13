"""
契約 0(テンプレート構築・対角化)と契約 0′(摂動論による ∂Φ)のテスト
--- parton-mode (fork addition) ---

DESIGN_parton.md §1.2 / §1.5 / §3.3 に対応する。契約 0′ の数値的正しさ本体は
§8 テスト 4(有限差分、test_parton_contract45.jl)が検証する。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "契約0: テンプレート構築と H の組み立て" begin
    data = toy_mf_data()
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)

    @test mfham.n_idx == 2
    @test mfham.is_onsite_group == [false, true]
    # ホッピング群は 4 ボンド × 2 フレーバー、オンサイト群は 4 サイト × 2 フレーバー
    @test length(mfham.template[1]) == 8
    @test length(mfham.template[2]) == 8
    # テンプレートは 1-based(0-based からの変換はここだけ)
    @test all(e -> 1 <= e.site1 <= 4 && 1 <= e.site2 <= 4 && 1 <= e.flavor <= 2,
              mfham.template[1])

    α = MVMCOptimizers.parton_alpha_from_terms(data)
    @test α == [ComplexF64(1, 0), ComplexF64(1, 0)]

    MVMCOptimizers.parton_update_orbitals!(mfham, α, 2)
    for f = 1:2
        H = mfham.h_mf[f]
        @test H ≈ H'                              # h.c. の暗黙付与でエルミート
        @test H[1, 2] ≈ ComplexF64(-1.0, 0.4)     # α = 1 なので t がそのまま
        @test H[2, 1] ≈ ComplexF64(-1.0, -0.4)    # h.c. 側
        @test H[1, 1] ≈ ComplexF64(0.7, 0.0)      # オンサイトは h.c. なし直接加算
        @test mfham.orbitals[f] ≈ eigen(Hermitian(H)).vectors[:, 1:2]
        @test size(mfham.eig_vecs[f]) == (4, 4)   # 摂動論用に非占有まで保持
    end
    @test mfham.min_gap > 0
    @test isfinite(mfham.min_gap)
end

@testset "契約0: α を変えると H が線形に応答する" begin
    data = toy_mf_data()
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)

    MVMCOptimizers.parton_update_orbitals!(mfham, ComplexF64[2, 1], 2)
    @test mfham.h_mf[1][1, 2] ≈ 2 * ComplexF64(-1.0, 0.4)
    @test mfham.h_mf[1][1, 1] ≈ ComplexF64(0.7, 0.0)

    # α が複素になるとホッピングは α·t、h.c. 側は conj(α·t)
    MVMCOptimizers.parton_update_orbitals!(mfham, ComplexF64[im, 1], 2)
    @test mfham.h_mf[1][1, 2] ≈ im * ComplexF64(-1.0, 0.4)
    @test mfham.h_mf[1][2, 1] ≈ conj(im * ComplexF64(-1.0, 0.4))
    @test mfham.h_mf[1] ≈ mfham.h_mf[1]'
end

@testset "契約0: 入力違反の検出" begin
    # 逆向き重複(片方向のみ列挙が規約)
    data = toy_mf_data()
    push!(
        data.pmftrans_terms,
        MVMCExpertModeParsers.PartonMFTransTerm(1, 0, 0, 0, ComplexF64(-1, -0.4), true),
    )
    push!(
        data.pmfpara_terms,
        MVMCExpertModeParsers.PartonMFParaTerm(1, 0, 0, 0, 0, ComplexF64(1, 0), true),
    )
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham, data)

    # idx が連番でない(欠番)
    data = toy_mf_data()
    for t in data.pmfpara_terms
        t.idx == 1 && (t.idx = 5)
    end
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 6)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham, data)

    # オンサイトの t が複素(実数必須)
    data = toy_mf_data()
    k = findfirst(t -> t.site1 == t.site2, data.pmftrans_terms)
    tt = data.pmftrans_terms[k]
    data.pmftrans_terms[k] = MVMCExpertModeParsers.PartonMFTransTerm(
        tt.site1, tt.flavor1, tt.site2, tt.flavor2, ComplexF64(0.7, 0.1), true)
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham, data)

    # 同一 idx グループにオンサイトとホッピングが混在
    data = toy_mf_data()
    for t in data.pmfpara_terms
        t.site1 == t.site2 && (t.idx = 0)
    end
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 1)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham, data)

    # pmfpara にあって pmftrans に無い(双方向完全性)
    data = toy_mf_data()
    push!(
        data.pmfpara_terms,
        MVMCExpertModeParsers.PartonMFParaTerm(0, 0, 2, 0, 0, ComplexF64(1, 0), true),
    )
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham, data)

    # pmftrans にあって pmfpara に無い(固定項は OptFlag 凍結で表現する規約)
    data = toy_mf_data()
    push!(
        data.pmftrans_terms,
        MVMCExpertModeParsers.PartonMFTransTerm(0, 0, 2, 0, ComplexF64(-0.5, 0), false),
    )
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham, data)

    # 共有 idx なのに初期値が食い違う
    data = toy_mf_data()
    data.pmfpara_terms[1].value = ComplexF64(2, 0)
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham, data)

    # あるフレーバーに項が 1 つも無い(空 H の eigen は縮退したゴミを返す)
    data = toy_mf_data(; n_flavor = 3)
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 3, 2)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)   # F=3 でも全部埋まる
    filter!(t -> t.flavor1 != 2, data.pmftrans_terms)
    filter!(t -> t.flavor1 != 2, data.pmfpara_terms)
    data.modpara.nflavor = 3
    mfham2 = MVMCOptimizers.PartonMFHamiltonian(4, 2, 3, 2)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham2, data)
end

@testset "契約0: mfham の n_idx が入力と食い違えば弾く" begin
    data = toy_mf_data()
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 3)   # 実際は 2
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham, data)
end

@testset "契約0′: ∂Φ の構造(オンサイト Im はゼロ)" begin
    data = toy_mf_data()
    mfham = build_toy_mfham(data)
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, 2)

    for f = 1:2
        @test any(!iszero, mfham.dorbitals[f][1])   # idx0(ホッピング)の Re
        @test any(!iszero, mfham.dorbitals[f][2])   # idx0 の Im
        @test all(iszero, mfham.dorbitals[f][4])    # idx1(オンサイト)の Im は凍結
        @test size(mfham.dorbitals[f][1]) == (4, 2)
    end

    # 一様オンサイト α のシフトは H → H + μI なので軌道を変えない(ゲージ平坦方向)
    for f = 1:2
        @test maximum(abs, mfham.dorbitals[f][3]) < 1e-10
    end
end
