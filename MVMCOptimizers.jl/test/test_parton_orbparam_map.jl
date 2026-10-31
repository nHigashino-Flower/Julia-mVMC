"""
制限付き Φ(PartonOrbIdx)の写像のテスト --- parton-mode (fork addition) ---

DESIGN_parton.md §12.12。Φ_f[s,n] = c·θ_idx で、SR が動かすのは θ(`data.pmforb_params`)。
恒等写像(def なし)は §12.11 の挙動とビット一致でなければならない。
再スケールは「同じ idx を使うフレーバーを結んだ連結成分」ごとに行う。
"""

using Test
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

const _FLAT = MVMCOptimizers.parton_orbparam_flat_index

"寸法だけを持つ data(Φ のテスト用)。"
function _map_data(n_site, n_elec, n_flavor; θ = ComplexF64[])
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite, mp.nelec, mp.nflavor = n_site, n_elec, n_flavor
    data.pmforb_params = θ
    return data
end

"""
4 サイト・Ne 2・F 2。サイト 0,1 が自前の θ(4 個)を持ち、サイト 2 はサイト 0 を符号 −1、
サイト 3 はサイト 1 を位相 im で共有する。フレーバー 1 はフレーバー 0 と同じ θ・同じ係数。
"""
function _shared_map()
    Ns, Ne, F = 4, 2, 2
    idx = zeros(Int, F * Ns * Ne)
    coeff = ones(ComplexF64, F * Ns * Ne)
    for f = 1:F, s = 1:Ns, n = 1:Ne
        k = _FLAT(f, s, n, Ns, Ne)
        base = s <= 2 ? s : s - 2
        idx[k] = (base - 1) * Ne + n
        coeff[k] = s == 3 ? -1 : (s == 4 ? im : 1)
    end
    return MVMCOptimizers.PartonOrbIndexMap(idx, coeff, fill(true, 4), Ns, Ne, F)
end

"2 サイト・Ne 1・F 3。フレーバー 0 と 2 が θ を共有し、フレーバー 1 は別。"
function _three_flavor_map()
    idx = [1, 2, 3, 4, 1, 2]           # 平坦位置 ((f−1)·2 + (s−1))·1 + 1 の順
    return MVMCOptimizers.PartonOrbIndexMap(idx, ones(ComplexF64, 6), fill(true, 4), 2, 1, 3)
end

@testset "写像: 恒等写像は既存と同じ組み立て・同じ再スケール" begin
    data = orbparam_toy_data()          # 6 サイト・Ne 2・F 2
    mp = data.modpara
    map = MVMCOptimizers.parton_orbparam_identity_map(mp.nsite, mp.nelec, mp.nflavor)
    @test map.is_identity
    @test map.n_idx == 24
    @test map.scale_groups == [collect(1:12), collect(13:24)]
    @test all(map.opt_flags)

    phi_a = MVMCOptimizers.PartonDirectPhi(mp.nsite, mp.nelec, mp.nflavor)
    phi_b = MVMCOptimizers.PartonDirectPhi(mp.nsite, mp.nelec, mp.nflavor, map)
    MVMCOptimizers.parton_orbparam_load_orbitals!(phi_a, data)
    MVMCOptimizers.parton_orbparam_load_orbitals!(phi_b, data)
    @test phi_a.orbitals == phi_b.orbitals

    d1 = deepcopy(data)
    d2 = deepcopy(data)
    x1 = MVMCOptimizers.parton_orbparam_rescale!(d1)
    x2 = MVMCOptimizers.parton_orbparam_rescale!(d2, map)
    @test x1 == x2
    @test d1.pmforb_params == d2.pmforb_params
end

@testset "写像: サイト・フレーバーの同一視と係数で Φ が組み立つ" begin
    map = _shared_map()
    @test !map.is_identity
    @test map.n_idx == 4
    @test length(map.scale_groups) == 1             # フレーバーが連結
    data = _map_data(4, 2, 2; θ = randn(MersenneTwister(1), ComplexF64, 4))
    phi = MVMCOptimizers.PartonDirectPhi(4, 2, 2, map)
    MVMCOptimizers.parton_orbparam_load_orbitals!(phi, data)
    Φ1, Φ2 = phi.orbitals
    @test Φ2 == Φ1
    for n = 1:2
        @test Φ1[3, n] == -Φ1[1, n]
        @test Φ1[4, n] == im * Φ1[2, n]
        @test Φ1[1, n] == data.pmforb_params[n]
    end
    # 長さが n_idx と違えば写さずに止める
    data.pmforb_params = data.pmforb_params[1:3]
    @test_throws ErrorException MVMCOptimizers.parton_orbparam_load_orbitals!(phi, data)
end

@testset "写像: 共有の無いフレーバーは別の組、共有があれば 1 組" begin
    map = _three_flavor_map()
    @test map.n_idx == 4
    @test map.scale_groups == [[1, 2], [3, 4]]
end

@testset "写像: 組ごとの再スケールは共有を壊さない" begin
    map = _three_flavor_map()
    data = _map_data(2, 1, 3; θ = ComplexF64[0.01, 0.02im, 10, -3])
    data.modpara.parton_orb_amp_max = 4.0
    xmax = MVMCOptimizers.parton_orbparam_rescale!(data, map)
    @test xmax == [0.02, 10.0]
    for g in map.scale_groups
        @test isapprox(maximum(abs, data.pmforb_params[g]), 4.0; rtol = 1e-15)
    end
    phi = MVMCOptimizers.PartonDirectPhi(2, 1, 3, map)
    MVMCOptimizers.parton_orbparam_load_orbitals!(phi, data)
    @test phi.orbitals[3] == phi.orbitals[1]
end

# =====================================================================
# PartonOrbIdx の読み書き
# =====================================================================

"dir に `PartonOrbIdx pmforbidx.def` を書いた namelist を作る。"
function _orbidx_namelist(dir; with_entry::Bool = true)
    path = joinpath(dir, "namelist.def")
    write(path, with_entry ? "PartonOrbIdx pmforbidx.def\n" : "ModPara modpara.def\n")
    return path
end

@testset "PartonOrbIdx: 書いて読むと同じ写像(係数 1・−1・複素)" begin
    map = _shared_map()
    dir = mktempdir()
    MVMCOptimizers.parton_orbparam_write_orbidx(joinpath(dir, "pmforbidx.def"), map, 4, 2, 2)
    got = MVMCOptimizers.parton_orbparam_read_orbidx(_map_data(4, 2, 2), _orbidx_namelist(dir))
    @test got isa MVMCOptimizers.PartonOrbIndexMap
    @test got.idx == map.idx
    @test got.coeff == map.coeff
    @test got.opt_flags == map.opt_flags
    @test got.scale_groups == map.scale_groups
    @test !got.is_identity
end

@testset "PartonOrbIdx: namelist に無ければ nothing" begin
    dir = mktempdir()
    @test MVMCOptimizers.parton_orbparam_read_orbidx(
        _map_data(4, 2, 2), _orbidx_namelist(dir; with_entry = false)) === nothing
end

@testset "PartonOrbIdx: 壊れた入力は error($case)" for case in
        (:header, :site_range, :idx_range, :missing_element, :duplicate_element, :bad_sign,
         :non_unit_coeff, :missing_optflag, :bad_optflag, :unused_idx)
    dir = mktempdir()
    file = joinpath(dir, "pmforbidx.def")
    MVMCOptimizers.parton_orbparam_write_orbidx(file, _shared_map(), 4, 2, 2)
    lines = readlines(file)          # 5 行ヘッダ + 要素 16 行(6〜21)+ OptFlag 4 行(22〜25)
    if case == :header
        lines[2] = "NFoo 4"
    elseif case == :site_range
        lines[6] = "4 0 0 0"
    elseif case == :idx_range
        lines[6] = "0 0 0 4"
    elseif case == :missing_element
        deleteat!(lines, 6)
    elseif case == :duplicate_element
        lines[7] = lines[6]
    elseif case == :bad_sign
        lines[6] = "0 0 0 0 2"
    elseif case == :non_unit_coeff
        lines[6] = "0 0 0 0 0.5 0.0"
    elseif case == :missing_optflag
        lines = lines[1:(end - 1)]
    elseif case == :bad_optflag
        lines[end] = "3 2"
    elseif case == :unused_idx
        lines[2] = "NPmfOrbIdx 5"
        push!(lines, "4 1")
    end
    write(file, join(lines, "\n") * "\n")
    err = try
        MVMCOptimizers.parton_orbparam_read_orbidx(_map_data(4, 2, 2), _orbidx_namelist(dir))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test err !== nothing && occursin("PartonOrbIdx", err.msg)
end
