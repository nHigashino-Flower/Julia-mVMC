"""
§8 テスト 17: SR 安定化の応急処置(v3.11、参照 chi-VMC 互換の最小セット)
--- parton-mode (fork addition) ---

REPORT §14 で特定した機構(フェルミ準位の level crossing で ∂Φ ∝ 1/gap →
sDiagMax ∝ 1/gap² → RedCut の相対閾値が健全な方向を大量カット → SR が漂流)への
下限ガード。根治(gap トラスト領域など)は別途。

1. 契約 0′ の gap_tol clamp: |ε_n − ε_u| < gap_tol の摂動項を 0 に落とす
   (参照 `vmc_chi_grad.jl` と同じ式・同じ既定値 1e-8)。**厳密な縮退でも
   Inf/NaN を出さない**
2. NaN/Inf ゲート: SR 更新後のパラメータに非有限が混入したら、その iter の
   更新を捨てて直前の値へ戻す(参照の「本 iter スキップ」と同義)。
   ここでは部品(pack が検出できる / unpack が復元できる)を機械検証する
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "§8-17-1 gap clamp: 厳密な縮退でも ∂Φ が有限" begin
    data = per_bond_mf_data(2; n_site = 6, n_elec = 3)
    mp = data.modpara
    n_idx = maximum(t -> t.idx, data.pmfpara_terms) + 1
    mfham = MVMCOptimizers.PartonMFHamiltonian(mp.nsite, mp.nelec, mp.nflavor, n_idx)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(
        mfham, MVMCOptimizers.parton_alpha_from_terms(data), mp.nelec)

    # フェルミ準位を人工的に厳密縮退させる(HOMO = LUMO)。clamp が無ければ
    # W / 0 で Inf/NaN が出る状況。
    for f = 1:mp.nflavor
        mfham.eig_vals[f][mp.nelec + 1] = mfham.eig_vals[f][mp.nelec]
    end
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, mp.nelec)
    for f = 1:mp.nflavor, dof in eachindex(mfham.dorbitals[f])
        @test all(isfinite, mfham.dorbitals[f][dof])
    end

    # 縮退ペア以外の寄与は clamp の影響を受けない: gap_tol を極小にした結果と
    # 比べ、差は「縮退ペアの寄与」のみ(= 有限差になる。Inf にはならないことは
    # 上で確認済みなので、ここでは clamp が選択的であることだけ見る)
    ref = [[copy(m) for m in v] for v in mfham.dorbitals]
    # 縮退を解消すれば tol に依らず一致する
    for f = 1:mp.nflavor
        mfham.eig_vals[f][mp.nelec + 1] = mfham.eig_vals[f][mp.nelec] + 0.5
    end
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, mp.nelec; gap_tol = 1e-8)
    a = [[copy(m) for m in v] for v in mfham.dorbitals]
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, mp.nelec; gap_tol = 1e-30)
    for f in eachindex(a), dof in eachindex(a[f])
        @test a[f][dof] == mfham.dorbitals[f][dof]   # 健全な gap では clamp は不発
    end
end

@testset "§8-17-2 NaN/Inf ゲートの部品: pack が検出し unpack が復元する" begin
    data = per_bond_mf_data(2; n_site = 4, n_elec = 2)
    MVMCOptimizers.parton_materialize_flags!(data)
    before = MVMCOptimizers.pack_parameters(data)
    @test all(isfinite, before)

    # SR 解の NaN 混入を模擬(α の 1 成分を汚す)
    poisoned = copy(before)
    poisoned[2] = ComplexF64(NaN, 0)
    MVMCOptimizers.unpack_parameters!(data, poisoned)
    @test any(!isfinite, MVMCOptimizers.pack_parameters(data))   # 検出できる

    # 復元(ゲートの実体はこの 1 手)
    MVMCOptimizers.unpack_parameters!(data, before)
    after = MVMCOptimizers.pack_parameters(data)
    @test after == before                                        # ビット単位で戻る
end
