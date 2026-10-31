"""
Φ 直接変分モードの入出力と再スケール(D_AmpMax 相当)のテスト
--- parton-mode (fork addition) ---

DESIGN_parton.md §12.11。InPmfOrb.def は厳格に読む(件数・範囲・重複・有限性・
列フルランクのどれかが崩れたら error。黙って部分適用しない)。再スケールは上流の
`sync_modified_parameter!` と同じく**毎回** max|Φ_f| = PartonOrbAmpMax にそろえる
(縮めるだけでなく拡大もする)。上流と違ってフレーバーごとに行う。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"dir に InPmfOrb.def を参照する namelist を書いてパスを返す(with_entry = false なら行なし)。"
function _io_namelist(dir; with_entry::Bool = true)
    path = joinpath(dir, "namelist.def")
    write(path, with_entry ? "InPmfOrb InPmfOrb.def\n" : "ModPara modpara.def\n")
    return path
end

"ip だけを返す(再スケールの不変性の確認用)。"
function _io_ip(data, sites)
    pstate = _orb_pstate(data)
    cfg = toy_config(data, sites)
    pstate.config = cfg
    MVMCOptimizers.parton_recompute_amplitude_all!(
        pstate.amp, pstate.mfham, cfg, data, pstate.workspace)
    ip = MVMCOptimizers.parton_calculate_ip(pstate.amp, MVMCOptimizers.parton_qp_weight(data))
    return ip, MVMCOptimizers.parton_local_energy(pstate, data, ip)
end

"例外を取り出す(投げなければ nothing)。"
_io_error(f) = try
    f()
    nothing
catch e
    e
end

@testset "InPmfOrb: 書いて読むとビット一致" begin
    data = orbparam_toy_data()
    dir = mktempdir()
    MVMCOptimizers.parton_orbparam_write_pmforb(data, joinpath(dir, "InPmfOrb.def"))
    other = orbparam_toy_data(; seed = 99)
    @test other.pmforb_params != data.pmforb_params
    path = MVMCOptimizers.parton_orbparam_read_in_pmforb!(other, _io_namelist(dir))
    @test path == joinpath(dir, "InPmfOrb.def")
    @test other.pmforb_params == data.pmforb_params
end

@testset "InPmfOrb: 壊れた入力は error($case)" for case in
        (:count, :idx_range, :duplicate, :nan, :rank_deficient, :missing_entry)
    data = orbparam_toy_data()
    n = length(data.pmforb_params)
    dir = mktempdir()
    file = joinpath(dir, "InPmfOrb.def")
    if case == :rank_deficient
        # フレーバー 2 の第 2 列 = 2 × 第 1 列
        mp = data.modpara
        for s = 1:mp.nsite
            k1 = MVMCOptimizers.parton_orbparam_flat_index(2, s, 1, mp.nsite, mp.nelec)
            k2 = MVMCOptimizers.parton_orbparam_flat_index(2, s, 2, mp.nsite, mp.nelec)
            data.pmforb_params[k2] = 2 * data.pmforb_params[k1]
        end
    end
    MVMCOptimizers.parton_orbparam_write_pmforb(data, file)
    lines = readlines(file)                    # 5 行ヘッダ + n 行
    if case == :count
        lines = lines[1:(end - 1)]
    elseif case == :idx_range
        lines[6] = replace(lines[6], r"^0 " => "$n ")
    elseif case == :duplicate
        lines[7] = replace(lines[7], r"^1 " => "0 ")
    elseif case == :nan
        lines[8] = "2 NaN 0.0"
    end
    write(file, join(lines, "\n") * "\n")
    nl = _io_namelist(dir; with_entry = case != :missing_entry)
    err = _io_error(() -> MVMCOptimizers.parton_orbparam_read_in_pmforb!(
        orbparam_toy_data(; seed = 99), nl))
    @test err isa ErrorException
    @test err !== nothing && occursin("InPmfOrb", err.msg)
end

@testset "再スケール: フレーバーごとに max = D、拡大もする" begin
    data = orbparam_toy_data()
    mp = data.modpara
    mp.parton_orb_amp_max = 4.0
    block = mp.nsite * mp.nelec
    r1, r2 = 1:block, (block + 1):(2 * block)
    data.pmforb_params[r1] .*= 0.01 / maximum(abs, data.pmforb_params[r1])   # max 0.01
    data.pmforb_params[r2] .*= 10.0 / maximum(abs, data.pmforb_params[r2])   # max 10
    pre = [maximum(abs, data.pmforb_params[r]) for r in (r1, r2)]
    xmax = MVMCOptimizers.parton_orbparam_rescale!(data)
    @test xmax == pre
    for r in (r1, r2)
        @test isapprox(maximum(abs, data.pmforb_params[r]), 4.0; rtol = 1e-15)
    end
end

@testset "再スケール: 全要素 0 のフレーバーは error" begin
    data = orbparam_toy_data()
    block = data.modpara.nsite * data.modpara.nelec
    data.pmforb_params[(block + 1):(2 * block)] .= 0
    @test_throws ErrorException MVMCOptimizers.parton_orbparam_rescale!(data)
end

@testset "再スケールで状態は変わらない" begin
    data = orbparam_toy_data()
    ip0, e0 = _io_ip(data, [1, 4])
    xmax = MVMCOptimizers.parton_orbparam_rescale!(data)
    ip1, e1 = _io_ip(data, [1, 4])
    ratios = data.modpara.parton_orb_amp_max ./ xmax
    @test isapprox(ip1 / ip0, prod(ratios .^ data.modpara.nelec); rtol = 1e-12)
    @test isapprox(e1, e0; rtol = 1e-12)
end
