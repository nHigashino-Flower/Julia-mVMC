"""
Bloch def の生成道具のテスト --- parton-mode (fork addition) ---

DESIGN_parton.md §12.12。`tools/parton_bloch_orbidx.jl` が `parton_lattice.dat`
(SITE・MFSHIFT・grid)と modpara から、Φ_f[s,(k,b)] = e^{ik·a(s)}·θ_{(f,σ,k,b)} の
PartonOrbIdx を作る。拡大セルの並進で射影子 P が不変になることが本体の性質。
"""

using Test
using Random
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

include(joinpath(@__DIR__, "..", "..", "tools", "parton_bloch_orbidx.jl"))

"""
grid 4×2、サイト i = x + 4y、σ = x mod 2、拡大セル 2×1(MFSHIFT は (0,0)(2,0)(0,1)(2,1))の
stage_in を書く。modpara は Nsite 8・NElec `n_elec`・NFlavor 2。
"""
function _bloch_stage(; n_elec::Int = 4)
    dir = mktempdir()
    open(joinpath(dir, "parton_lattice.dat"), "w") do io
        println(io, "# テスト用")
        println(io, "grid 4 2")
        for y = 0:1, x = 0:3
            println(io, "SITE $(x + 4y) $x $y $(x % 2)")
        end
        for (tx, ty) in ((0, 0), (2, 0), (0, 1), (2, 1))
            println(io, "MFSHIFT $tx $ty")
        end
    end
    write(joinpath(dir, "modpara.def"),
          "Nsite 8\nNElec $n_elec\nNFlavor 2\nPartonMode 1\nPartonOrbParam 1\n")
    return dir
end

"サイトの置換(座標を (tx, ty) だけずらす)。1-based。"
_bloch_perm(tx, ty) = [((x + tx) % 4) + 4 * ((y + ty) % 2) + 1 for y = 0:1 for x = 0:3]

"写像と θ から Φ を組み、フレーバーごとの射影子を返す。"
function _bloch_projectors(map, θ; n_site = 8, n_elec = 4, n_flavor = 2)
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite, data.modpara.nelec, data.modpara.nflavor = n_site, n_elec, n_flavor
    data.pmforb_params = θ
    phi = MVMCOptimizers.PartonDirectPhi(n_site, n_elec, n_flavor, map)
    MVMCOptimizers.parton_orbparam_load_orbitals!(phi, data)
    return [Φ * ((Φ' * Φ) \ Φ') for Φ in phi.orbitals]
end

@testset "Bloch: k の個数 = N_cell、θ の個数 = F·N_σ·Ne" begin
    bs = parton_bloch_structure(_bloch_stage())
    @test bs.kpoints == [(0, 0), (0, 1), (1, 0), (1, 1)]
    @test bs.map.n_idx == 2 * 2 * 4               # 全要素なら 2·8·4 = 64
    @test all(length(c) == 4 for c in bs.columns)  # 各 k に 1 本
    @test bs.sigma == [x % 2 for y = 0:1 for x = 0:3]
end

@testset "Bloch: 乱数 θ から組んだ Φ の射影子は拡大セルの並進で不変" begin
    bs = parton_bloch_structure(_bloch_stage())
    Ps = _bloch_projectors(bs.map, randn(MersenneTwister(1), ComplexF64, bs.map.n_idx))
    for (tx, ty) in bs.shifts, P in Ps
        p = _bloch_perm(tx, ty)
        @test maximum(abs, P[p, p] - P) < 1e-12
    end
    # 対照: 制限なしの乱数 Φ では成り立たない
    idmap = MVMCOptimizers.parton_orbparam_identity_map(8, 4, 2)
    Pr = _bloch_projectors(idmap, randn(MersenneTwister(2), ComplexF64, 64))
    p = _bloch_perm(2, 0)
    @test maximum(abs, Pr[1][p, p] - Pr[1]) > 1e-3
end

@testset "Bloch: def を書いて読むと同じ写像" begin
    stage = _bloch_stage()
    bs = parton_bloch_structure(stage)
    out = joinpath(stage, "pmforbidx.def")
    @test parton_bloch_write(stage, out) == out
    nl = joinpath(stage, "namelist.def")
    write(nl, "PartonOrbIdx pmforbidx.def\n")
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite, data.modpara.nelec, data.modpara.nflavor = 8, 4, 2
    got = MVMCOptimizers.parton_orbparam_read_orbidx(data, nl)
    @test got.idx == bs.map.idx
    @test got.coeff == bs.map.coeff
end

@testset "Bloch: 占有ファイル(k ごとの本数指定)" begin
    stage = _bloch_stage()
    occ = joinpath(stage, "occ.dat")
    # フレーバー 0 は k(0,0) と k(1,0) に 2 本ずつ((2,0) は (0,0) と同値)、フレーバー 1 は全 k に 1 本
    write(occ, "0 2 0 2\n0 1 0 2\n1 0 0 1\n1 0 1 1\n1 1 0 1\n1 1 1 1\n")
    bs = parton_bloch_structure(stage; occupation_file = occ)
    k_of(c) = [kk for (kk, _) in c]
    @test k_of(bs.columns[1]) == [1, 1, 3, 3]
    @test k_of(bs.columns[2]) == [1, 2, 3, 4]
    Ps = _bloch_projectors(bs.map, randn(MersenneTwister(3), ComplexF64, bs.map.n_idx))
    p = _bloch_perm(0, 1)
    @test maximum(abs, Ps[1][p, p] - Ps[1]) < 1e-12
end

@testset "Bloch: 壊れた占有は error($case)" for case in (:unknown_k, :wrong_total, :indivisible)
    stage = _bloch_stage(; n_elec = case == :indivisible ? 3 : 4)
    occ = nothing
    if case == :unknown_k
        occ = joinpath(stage, "occ.dat")
        write(occ, "0 7 0 4\n1 0 0 4\n")
    elseif case == :wrong_total
        occ = joinpath(stage, "occ.dat")
        write(occ, "0 0 0 3\n1 0 0 4\n")
    end
    @test_throws ErrorException parton_bloch_structure(stage; occupation_file = occ)
end

# =====================================================================
# α 解 → Bloch の θ(射影子 P を k ごとに対角化)
# =====================================================================

"4 サイト環(dimerized_mf_data)用の stage_in。拡大セル 2 サイト、MFSHIFT (0,0)(2,0)。"
function _bloch_ring_stage()
    dir = mktempdir()
    open(joinpath(dir, "parton_lattice.dat"), "w") do io
        println(io, "grid 4 1")
        for i = 0:3
            println(io, "SITE $i $i 0 $(i % 2)")
        end
        println(io, "MFSHIFT 0 0")
        println(io, "MFSHIFT 2 0")
    end
    write(joinpath(dir, "modpara.def"), "Nsite 4\nNElec 2\nNFlavor 2\nPartonMode 1\n")
    return dir
end

_bloch_P(Φ) = Φ * ((Φ' * Φ) \ Φ')

@testset "α → Bloch: 射影子が一致する" begin
    bs = parton_bloch_structure(_bloch_ring_stage())
    mfham = build_toy_mfham(dimerized_mf_data(); n_idx = 3)
    θ = parton_bloch_theta_from_orbitals(bs, mfham.orbitals)
    @test length(θ) == bs.map.n_idx == 2 * 2 * 2
    Ps = _bloch_projectors(bs.map, θ; n_site = 4, n_elec = 2, n_flavor = 2)
    for f = 1:2
        @test maximum(abs, Ps[f] - _bloch_P(mfham.orbitals[f])) < 1e-10
    end
end

@testset "α → Bloch: 並進で不変でない軌道は error" begin
    bs = parton_bloch_structure(_bloch_ring_stage())
    mfham = build_toy_mfham(dimerized_mf_data(); n_idx = 3)
    gauge = Diagonal(cis.([0.0, 0.3, 1.1, -0.7]))      # サイトごとの位相で並進を壊す
    broken = [gauge * Φ for Φ in mfham.orbitals]
    err = try
        parton_bloch_theta_from_orbitals(bs, broken)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test err !== nothing && occursin("eigen", err.msg)
end

"_bloch_stage に、x 方向の最近接ホップの physhop.def を足す(折り返しのボンドだけ逆向きに格納)。"
function _bloch_stage_with_physhop(; broken::Bool = false)
    stage = _bloch_stage()
    t = ComplexF64(-1.0, 0.3)
    open(joinpath(stage, "physhop.def"), "w") do io
        for _ = 1:5
            println(io, "=====")
        end
        for y = 0:1, x = 0:3
            i = x + 4y
            j = ((x + 1) % 4) + 4y
            tt = (broken && i == 1) ? 2t : t
            if x == 3
                println(io, "$j $i $(real(tt)) $(-imag(tt))")    # 逆向き(共役)で格納
            else
                println(io, "$i $j $(real(tt)) $(imag(tt))")
            end
        end
    end
    return stage
end

@testset "Bloch: 逆向きに格納されたボンドを不変性の検査で取りこぼさない" begin
    @test_logs min_level = Base.CoreLogging.Warn parton_bloch_structure(_bloch_stage_with_physhop())
    @test_logs (:warn, r"not invariant") match_mode = :any parton_bloch_structure(
        _bloch_stage_with_physhop(; broken = true))
end
