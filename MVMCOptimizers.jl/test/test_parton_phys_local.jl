"""
DESIGN_parton.md §13.5-1 / §13.5-2: 物理量の局所値
--- parton-mode (fork addition) ---

1. 全配置列挙で |Ψ|² 重みの局所値の平均(`parton_exact_expectations`)が、Fock 空間で b†・b を
   素直に作用させた密行列の期待値と一致する(α F=2・α F=3・Φ F=2・Pf F=2・Pf F=3)
2. 配置ごとに E_loc = Σ_physhop [t·G1(j,i) + t*·G1(i,j)] + Σ_coulomb V·G2(i,i,j,j)
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "§13 分類" begin
    cls = MVMCOptimizers.parton_green2_class
    @test cls(1, 1, 2, 2) == :density
    @test cls(1, 2, 3, 3) == :hop_then_density
    @test cls(1, 1, 2, 3) == :density_then_hop
    @test cls(1, 2, 2, 3) == :same_particle
    @test cls(1, 2, 2, 1) == :same_particle
    @test cls(1, 2, 3, 4) == :two_particle
    @test cls(1, 2, 3, 1) == :two_particle
end

const PHYS_CASES = phys_toy_cases()

@testset "§13.5-1 全配置列挙 vs 密行列: $(c.label)" for c in PHYS_CASES
    data, pstate = c.data, c.pstate
    exact = MVMCOptimizers.parton_exact_expectations(pstate, data)
    configs, psi = dense_wavefunction(data, pstate)
    @test exact.n_configurations == length(configs)

    ref1 = [dense_expectation(configs, psi, [(t.site1 + 1, t.site2 + 1)], c.fermionic)
            for t in data.green_one_terms]
    @test isapprox(exact.one_body, ref1; atol = 1e-12, rtol = 1e-10)
    # 非対角(i ≠ j)に十分な大きさの値があること(偶然 0 どうしの比較にしない)
    offdiag = [ref1[n] for (n, t) in enumerate(data.green_one_terms) if t.site1 != t.site2]
    @test maximum(abs, offdiag) > 0.05
    # 識別力: 虚部が十分あるので ⟨b†_i b_j⟩ と ⟨b†_j b_i⟩(= 共役)の取り違えを見分けられる
    @test maximum(abs ∘ imag, offdiag) > 0.01
    # 識別力: フェルミオンのケースでは、符号を落とした(ボソンとして作用させた)参照と食い違う
    if c.fermionic
        ref1_bosonic = [dense_expectation(configs, psi, [(t.site1 + 1, t.site2 + 1)], false)
                        for t in data.green_one_terms]
        @test maximum(abs, ref1_bosonic .- ref1) > 1e-3
    end

    ref2 = [dense_expectation(configs, psi,
                              [(t.site1 + 1, t.site2 + 1), (t.site3 + 1, t.site4 + 1)],
                              c.fermionic)
            for t in data.green_two_terms]
    @test isapprox(exact.two_body, ref2; atol = 1e-12, rtol = 1e-10)
    classes = Set(MVMCOptimizers.parton_green2_class(t.site1, t.site2, t.site3, t.site4)
                  for t in data.green_two_terms)
    @test classes == Set([:density, :hop_then_density, :density_then_hop, :same_particle])
    # 類ごとに 0 でない値が入っていること
    for cl in classes
        vals = [abs(ref2[n]) for (n, t) in enumerate(data.green_two_terms)
                if MVMCOptimizers.parton_green2_class(t.site1, t.site2, t.site3, t.site4) == cl]
        @test maximum(vals) > 1e-3
    end
end

@testset "§13.5-2 配置ごとの恒等式: $(c.label)" for c in PHYS_CASES
    data, pstate = c.data, c.pstate
    @test any(h -> imag(h.value) != 0, pstate.physham.hops)   # t は複素
    qp_weight = MVMCOptimizers.parton_qp_weight(data)
    for sites in phys_combinations(data.modpara.nsite, data.modpara.nelec)
        ip = phys_place_sorted!(pstate, data, sites)
        abs(ip) < 1e-30 && continue
        e = MVMCOptimizers.parton_local_energy(pstate, data, ip)
        e_green = zero(ComplexF64)
        for h in pstate.physham.hops
            i, j, t = h.site1, h.site2, h.value
            e_green += t * MVMCOptimizers.parton_local_green1(pstate, data, qp_weight, j, i)
            e_green += conj(t) * MVMCOptimizers.parton_local_green1(pstate, data, qp_weight, i, j)
        end
        for d in pstate.physham.diags
            i, j = d.site1, d.site2
            e_green += d.value * MVMCOptimizers.parton_local_green2(pstate, data, qp_weight, i, i, j, j)
        end
        @test isapprox(e, e_green; atol = 1e-12)
    end
end

@testset "§13 two_particle は局所値で error" begin
    c = PHYS_CASES[1]
    phys_place_sorted!(c.pstate, c.data, [1, 2, 4])
    qp_weight = MVMCOptimizers.parton_qp_weight(c.data)
    @test_throws ErrorException MVMCOptimizers.parton_local_green2(
        c.pstate, c.data, qp_weight, 1, 3, 2, 5)
end
