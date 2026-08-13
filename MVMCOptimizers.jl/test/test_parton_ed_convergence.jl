"""
§8 テスト 6: トイ系の SR を ED と突き合わせる(F=2 ボソン / F=3 フェルミオン)
--- parton-mode (fork addition) ---

DESIGN_parton.md §8-6 と §1.3(選択則と符号定理)に対応する。

符号定理の主張は「合成粒子の統計は行列式が自動で運ぶので、局所エネルギーの
行列要素に余分な符号は要らない」。F が偶なら合成粒子はボソン、奇ならフェルミオン
なので、同じ実装から F 依存で異なる統計が出てくるはずである。ここではそれを
2 段構えで確かめる。

1. 厳密な機械検証: 固定した Ψ について、全数展開した VMC 推定量が、
   F 偶なら硬芯ボソンの ED 行列、F 奇なら占有順序基底(JW 符号つき)の ED 行列の
   レイリー商と機械精度で一致すること。E_loc に符号を書いていないのに統計が
   出てくることの直接確認。
2. SR の収束: F=3 はフェルミオン基底エネルギーに収束し(この変分族は飽和する)、
   F=2 はフェルミオン基底エネルギーを下回る(フェルミオンの状態では不可能)一方で
   ボソン基底エネルギーは下回らない(変分下限)。

ED はテスト内に素朴に書く(パッケージ追加なし)。
"""

using Test
using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

"""
    ed_hamiltonian(n_site, n_part, bonds, t, V; statistics) -> (configs, H)

硬芯粒子 n_part 個の厳密対角化行列。基底は占有サイトの昇順リスト。

    H = Σ_{(i,j) ∈ bonds} [ t · b†_j b_i + conj(t) · b†_i b_j ] + Σ V n_i n_j

`statistics = :boson` は符号なし、`:fermion` は占有順序基底での JW 符号
(移動区間に挟まれた占有サイトの数の偶奇)をつける。
"""
function ed_hamiltonian(n_site::Int, n_part::Int, bonds, t::ComplexF64, V::Float64;
                        statistics::Symbol)
    configs = sort([collect(c) for c in
                    Iterators.filter(c -> length(c) == n_part,
                                     (Set(x) for x in
                                      Iterators.product(ntuple(_ -> 1:n_site, n_part)...)))] |>
                   unique .|> sort)
    pos = Dict(c => k for (k, c) in enumerate(configs))
    n = length(configs)
    H = zeros(ComplexF64, n, n)
    for (k, c) in enumerate(configs)
        for (i, j) in bonds
            H[k, k] += V * ((i in c) ? 1 : 0) * ((j in c) ? 1 : 0)
        end
        for (i, j) in bonds
            for (a, b, amp) in ((i, j, t), (j, i, conj(t)))
                # b†_b b_a: a が占有・b が空きのときだけ効く
                ((a in c) && !(b in c)) || continue
                cnew = sort(replace(copy(c), a => b))
                sgn = 1.0
                if statistics === :fermion
                    lo, hi = minmax(a, b)
                    sgn = (-1.0)^count(s -> lo < s < hi, c)
                end
                H[pos[cnew], k] += sgn * amp
            end
        end
    end
    @assert norm(H - H') < 1e-12 "ED Hamiltonian is not Hermitian"
    return configs, Hermitian(H)
end

const _ED_N_SITE = 4
const _ED_N_ELEC = 2
const _ED_BONDS = [(i, mod1(i + 1, _ED_N_SITE)) for i = 1:_ED_N_SITE]
const _ED_T = ComplexF64(-1.0, 0.0)

@testset "§8-6 準備: ED はボソンとフェルミオンで別物" begin
    configs, Hb =
        ed_hamiltonian(_ED_N_SITE, _ED_N_ELEC, _ED_BONDS, _ED_T, 0.0; statistics = :boson)
    _, Hf =
        ed_hamiltonian(_ED_N_SITE, _ED_N_ELEC, _ED_BONDS, _ED_T, 0.0; statistics = :fermion)
    @test length(configs) == 6
    e_b = eigmin(Hb)
    e_f = eigmin(Hf)
    @test e_b ≈ -2 * sqrt(2) rtol = 1e-12    # 硬芯ボソン 2 個 on 4 サイト環
    @test e_f ≈ -2.0 rtol = 1e-12            # スピンレスフェルミオン(k = 0, ±π/2)
    @test e_b < e_f - 0.5                     # 十分離れているので判別に使える
end

@testset "§8-6a 符号定理の機械検証: F 偶=ボソン / F 奇=フェルミオン" begin
    configs, Hb =
        ed_hamiltonian(_ED_N_SITE, _ED_N_ELEC, _ED_BONDS, _ED_T, 0.0; statistics = :boson)
    _, Hf =
        ed_hamiltonian(_ED_N_SITE, _ED_N_ELEC, _ED_BONDS, _ED_T, 0.0; statistics = :fermion)

    for F in (2, 3, 4, 5)
        data = per_bond_mf_data(F)
        MVMCOptimizers.parton_materialize_flags!(data)
        pstate = MVMCOptimizers.parton_build_optimization_state(data)
        mfham = pstate.mfham
        MVMCOptimizers.parton_update_orbitals!(
            mfham, MVMCOptimizers.parton_alpha_from_terms(data), _ED_N_ELEC)
        qpw = MVMCOptimizers.parton_qp_weight(data)

        # 全数展開: Ψ(x) と E_loc(x) を全配置について求める(サンプリングなし)
        ψ = ComplexF64[]
        eloc = ComplexF64[]
        for c in configs
            cfg = MVMCOptimizers.PartonConfiguration(_ED_N_SITE, _ED_N_ELEC, F, 1)
            for f = 1:F, (m, r) in enumerate(c)
                MVMCOptimizers.place_particle!(cfg, f, m, r)
            end
            pstate.config = cfg
            MVMCOptimizers.parton_recompute_amplitude_all!(
                pstate.amp, mfham, cfg, data, pstate.workspace)
            ip = MVMCOptimizers.parton_calculate_ip(pstate.amp, qpw)
            push!(ψ, ip)
            push!(eloc, MVMCOptimizers.parton_local_energy(pstate, data, ip))
        end
        w = abs2.(ψ)
        estimator = sum(w .* eloc) / sum(w)

        rayleigh_boson = dot(ψ, Hb * ψ) / dot(ψ, ψ)
        rayleigh_fermion = dot(ψ, Hf * ψ) / dot(ψ, ψ)
        @test !isapprox(rayleigh_boson, rayleigh_fermion; atol = 1e-6)  # 判別可能

        if iseven(F)
            @test isapprox(estimator, rayleigh_boson; atol = 1e-10)
            @test !isapprox(estimator, rayleigh_fermion; atol = 1e-6)
        else
            @test isapprox(estimator, rayleigh_fermion; atol = 1e-10)
            @test !isapprox(estimator, rayleigh_boson; atol = 1e-6)
        end
    end
end

@testset "§8-6b SR の収束: F=3 はフェルミオン ED、F=2 はボソン側" begin
    _, Hb =
        ed_hamiltonian(_ED_N_SITE, _ED_N_ELEC, _ED_BONDS, _ED_T, 0.0; statistics = :boson)
    _, Hf =
        ed_hamiltonian(_ED_N_SITE, _ED_N_ELEC, _ED_BONDS, _ED_T, 0.0; statistics = :fermion)
    e_boson = eigmin(Hb)
    e_fermion = eigmin(Hf)

    function run_sr(F, seed)
        data = per_bond_mf_data(F)
        MVMCOptimizers.parton_materialize_flags!(data)
        pstate = MVMCOptimizers.parton_build_optimization_state(data)
        rng = MVMCOptimizers.SFMT19937RNG()
        Random.seed!(rng, seed)
        status = MVMCOptimizers.parton_vmc_para_opt!(
            pstate, data, MVMCOptimizers.serial_context();
            rng = rng, output_dir = mktempdir())
        # weight_average_we! が既に Wc で割っているので etot がそのまま平均値
        return status, real(pstate.state.energy.etot)
    end

    # F=3(フェルミオン): この変分族はフェルミオン基底状態を厳密に表現できる
    for seed in (4321, 99)
        status, e = run_sr(3, seed)
        @test status == 0
        @test e >= e_fermion - 1e-6          # 変分下限を破らない
        @test isapprox(e, e_fermion; rtol = 1e-2)
    end

    # F=2(ボソン): フェルミオン基底エネルギーより下へ行ける(フェルミオンの状態
    # では不可能)。一方でボソン基底エネルギーは下回らない。
    for seed in (4321, 99)
        status, e = run_sr(2, seed)
        @test status == 0
        @test e >= e_boson - 1e-6            # 変分下限を破らない
        @test e < e_fermion - 0.3            # フェルミオンセクターではありえない
    end
end
