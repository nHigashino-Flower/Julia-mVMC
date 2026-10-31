"""
Φ 直接変分モードの型のテスト --- parton-mode (fork addition) ---

DESIGN_parton.md §12.11(上流型の簡素版)と §12.5(型)。振幅系(契約 1〜4)は
型注釈を `PartonOrbitalSource` に広げただけなので、同じ軌道を渡せば α モードと
ビット一致しなければならない(§12.9 の Φ-0b)。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "平坦添字: 全単射で、load と from_orbitals が往復する" begin
    Ns, Ne, F = 5, 2, 3
    ks = [MVMCOptimizers.parton_orbparam_flat_index(f, s, n, Ns, Ne)
          for f = 1:F, s = 1:Ns, n = 1:Ne]
    @test sort(vec(ks)) == collect(1:F * Ns * Ne)
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite, data.modpara.nelec, data.modpara.nflavor = Ns, Ne, F
    @test MVMCOptimizers.parton_orbparam_n_orb(data.modpara) == F * Ns * Ne
    data.pmforb_params = ComplexF64.(1:F * Ns * Ne) .* (1 + 0.5im)
    phi = MVMCOptimizers.PartonDirectPhi(Ns, Ne, F)
    MVMCOptimizers.parton_orbparam_load_orbitals!(phi, data)
    @test phi.orbitals[2][3, 1] ==
          data.pmforb_params[MVMCOptimizers.parton_orbparam_flat_index(2, 3, 1, Ns, Ne)]
    @test MVMCOptimizers.parton_orbparam_params_from_orbitals(phi.orbitals) ==
          data.pmforb_params

    # 長さが合わなければ写さずに止める
    data.pmforb_params = data.pmforb_params[1:(end - 1)]
    @test_throws ErrorException MVMCOptimizers.parton_orbparam_load_orbitals!(phi, data)
end

@testset "Φ-0b: 同じ軌道なら振幅系が α とビット一致" begin
    data = dimerized_mf_data()
    data.modpara.parton_flavor_sym_fast = 0      # 両者とも n_stored = F で比べる
    mp = data.modpara
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    mfham = pstate.mfham
    MVMCOptimizers.parton_update_orbitals!(
        mfham, MVMCOptimizers.parton_alpha_from_terms(data), mp.nelec)

    phi = MVMCOptimizers.PartonDirectPhi(mp.nsite, mp.nelec, mp.nflavor)
    phi.orbitals = deepcopy(mfham.orbitals)
    pstate_phi = MVMCOptimizers.PartonOptimizationState(
        pstate.state, deepcopy(pstate.amp), deepcopy(pstate.config),
        deepcopy(pstate.workspace), phi, pstate.physham)
    @test pstate_phi isa MVMCOptimizers.PartonOptimizationState{MVMCOptimizers.PartonDirectPhi}

    qpw = MVMCOptimizers.parton_qp_weight(data)
    for sites in ([1, 2], [1, 3], [2, 4], [3, 4])
        for (ps, src) in ((pstate, mfham), (pstate_phi, phi))
            cfg = toy_config(data, sites)
            ps.config = cfg
            MVMCOptimizers.parton_recompute_amplitude_all!(
                ps.amp, src, cfg, data, ps.workspace)
        end
        @test pstate_phi.amp.det_a == pstate.amp.det_a
        @test pstate_phi.amp.inv_a == pstate.amp.inv_a

        ip = MVMCOptimizers.parton_calculate_ip(pstate.amp, qpw)
        ip_phi = MVMCOptimizers.parton_calculate_ip(pstate_phi.amp, qpw)
        @test ip_phi == ip
        @test MVMCOptimizers.parton_local_energy(pstate_phi, data, ip_phi) ==
              MVMCOptimizers.parton_local_energy(pstate, data, ip)

        # 比と更新: 粒子 1 を空きサイトへ
        r_new = first(setdiff(1:mp.nsite, sites))
        a = MVMCOptimizers.parton_amplitude_ratio!(
            pstate.workspace, pstate.amp, mfham, data, qpw, 1, r_new)
        b = MVMCOptimizers.parton_amplitude_ratio!(
            pstate_phi.workspace, pstate_phi.amp, phi, data, qpw, 1, r_new)
        @test a == b
        MVMCOptimizers.parton_update_amplitude!(
            pstate.amp, mfham, data, pstate.workspace, 1, r_new)
        MVMCOptimizers.parton_update_amplitude!(
            pstate_phi.amp, phi, data, pstate_phi.workspace, 1, r_new)
        @test pstate_phi.amp.det_a == pstate.amp.det_a
        @test pstate_phi.amp.inv_a == pstate.amp.inv_a
    end
end
