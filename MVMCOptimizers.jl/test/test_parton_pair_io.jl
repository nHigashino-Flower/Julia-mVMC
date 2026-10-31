"""
Pfaffian パートンモードの入出力・ロケータ・初期値・D_AmpMax・門番
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §2.3(初期値)・§4.6(D_AmpMax)・§6(出力)・§7(門番)・§9(T1-g / T1-h)。
"""

using Test
using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

const MOI = MVMCOptimizers

"PartonPairOrbitalTerm の並び(全対・共有なし)。"
_pair_terms_typed(Ns, K) =
    [MVMCExpertModeParsers.PartonPairOrbitalTerm(t.site1, t.flavor1, t.site2, t.flavor2,
                                                 t.idx, t.sign) for t in full_pair_terms(Ns, K)]

"門番を通る最小の Pf モード入力(環の複素ホップ、全対の def、p は複素乱数、フラグ実体化済み)。"
function pair_ok_data(; n_site::Int = 5, n_elec::Int = 2, n_flavor::Int = 2, seed::Int = 3)
    data = pair_toy_data(n_site = n_site, n_elec = n_elec, n_flavor = n_flavor)
    for i = 0:(n_site - 1)
        push!(data.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(
            i, mod(i + 1, n_site), ComplexF64(-1.0, 0.3), true))
    end
    data.parton_pair_terms = _pair_terms_typed(n_site, n_flavor)
    data.parton_pair_params = randn(MersenneTwister(seed), ComplexF64, length(data.parton_pair_terms))
    data.modpara.dsr_opt_sta_del = 0.1
    MOI.parton_pair_materialize_flags!(data)
    return data
end

@testset "ロケータ: f のブロックは射影の直後(Φ・α が空のとき)" begin
    data = pair_ok_data()
    n_pair = length(data.parton_pair_params)
    @test MVMCExpertModeParsers.count_variational_parameters(data) == n_pair
    @test pack_parameters(data) == data.parton_pair_params
    set_parameter_value!(data, 3, ComplexF64(7, -1))
    @test data.parton_pair_params[3] == ComplexF64(7, -1)
    @test get_parameter_value(data, 3) == ComplexF64(7, -1)
    p = pack_parameters(data) .* 2
    unpack_parameters!(data, p)
    @test data.parton_pair_params == p

    # Jastrow があれば射影の後ろ
    _add_ring_jastrow!(data, [0.1, -0.2])
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    @test n_proj > 0
    @test MVMCExpertModeParsers.count_variational_parameters(data) == n_proj + n_pair
    @test pack_parameters(data)[(n_proj + 1):end] == data.parton_pair_params
end

@testset "OptFlag の実体化: 長さ 2(n_proj + n_pair)、def の OptFlag 行が Re・Im の両方に効く" begin
    data = pair_ok_data()
    data.parton_pair_opt_flags = Dict(0 => 0, 2 => 1)
    flags = MOI.parton_pair_materialize_flags!(data)
    n_pair = length(data.parton_pair_params)
    @test length(flags) == 2 * n_pair
    @test flags[1] == false && flags[2] == false          # idx 0 は凍結(Re・Im とも)
    @test all(flags[3:end])                               # 指定なし・flag 1 は可動
end

@testset "D_AmpMax: max|p| を 4 にそろえる(拡大もする)" begin
    data = pair_ok_data()
    data.parton_pair_params .*= 1e-3
    before = maximum(abs, data.parton_pair_params)
    xmax = MOI.parton_pair_rescale!(data)
    @test xmax == before
    @test maximum(abs, data.parton_pair_params) ≈ MOI.D_AMP_MAX rtol = 1e-15
    data.parton_pair_params .= 0
    @test_throws ErrorException MOI.parton_pair_rescale!(data)
end

@testset "T1-h 乱数初期化: 上流と同じ分布・専用の乱数列・凍結は 0" begin
    data = pair_ok_data()
    data.parton_pair_opt_flags = Dict(1 => 0)
    MOI.parton_pair_materialize_flags!(data)
    MOI.parton_pair_init_random!(data, 12345)
    p1 = copy(data.parton_pair_params)
    @test p1[2] == 0                                       # 凍結は 0
    # 同じ seed なら同じ値、違う seed なら違う値
    MOI.parton_pair_init_random!(data, 12345)
    @test data.parton_pair_params == p1
    MOI.parton_pair_init_random!(data, 12346)
    @test data.parton_pair_params != p1
    # 分布の式: 可動成分ごとに [2(r₁−½) + 2i(r₂−½)]/√2(r は rng_real2、Re → Im の順)
    rng = MOI.SFMT19937RNG()
    Random.seed!(rng, 12345)
    expected = zeros(ComplexF64, length(p1))
    for k in eachindex(expected)
        k == 2 && continue
        r1 = MOI.rng_real2(rng)
        r2 = MOI.rng_real2(rng)
        expected[k] = ComplexF64(2 * (r1 - 0.5), 2 * (r2 - 0.5)) / sqrt(2.0)
    end
    @test p1 == expected
end

"namelist と In ファイルを書いて読む。"
function _with_in_file(f, body::AbstractString; key = "InPartonPairOrbital")
    mktempdir() do dir
        write(joinpath(dir, "in.def"), body)
        write(joinpath(dir, "namelist.def"), "$key  in.def\n")
        f(joinpath(dir, "namelist.def"))
    end
end

@testset "T1-g 入出力: zqp_parton_pair の往復がビット一致" begin
    data = pair_ok_data()
    mktempdir() do dir
        path = MOI.parton_pair_write_params(data, joinpath(dir, "zqp_parton_pair_opt.dat"))
        lines = readlines(path)
        @test lines[2] == "NOrbitalIdx $(length(data.parton_pair_params))"
        write(joinpath(dir, "namelist.def"), "InPartonPairOrbital  zqp_parton_pair_opt.dat\n")
        orig = copy(data.parton_pair_params)
        data.parton_pair_params .= 0
        @test MOI.parton_pair_read_in!(data, joinpath(dir, "namelist.def")) == path
        @test data.parton_pair_params == orig
    end
    # namelist に無ければ nothing を返して何もしない(初期値は乱数になる)
    mktempdir() do dir
        write(joinpath(dir, "namelist.def"), "ModPara modpara.def\n")
        before = copy(data.parton_pair_params)
        @test MOI.parton_pair_read_in!(data, joinpath(dir, "namelist.def")) === nothing
        @test data.parton_pair_params == before
    end
end

@testset "T1-g InPartonPairOrbital の厳格な検査" begin
    data = pair_ok_data(n_site = 3, n_elec = 2, n_flavor = 2)
    n = length(data.parton_pair_params)
    head = "======\nNOrbitalIdx $n\n======\n======\n======\n"
    rows(ks) = join(("$k 0.5 -0.25" for k in ks), "\n") * "\n"
    bad = Dict(
        "件数不足" => head * rows(0:(n - 2)),
        "宣言と行数の食い違い" => replace(head, "NOrbitalIdx $n" => "NOrbitalIdx $(n + 1)") * rows(0:(n - 1)),
        "範囲外" => head * rows(1:n),
        "重複" => head * rows(vcat(0, 0:(n - 2))),
        "非有限" => head * rows(0:(n - 2)) * "$(n - 1) NaN 0.0\n",
        "列不足" => head * rows(0:(n - 2)) * "$(n - 1) 0.5\n",
        "キーワード違い" => replace(head, "NOrbitalIdx" => "NPmfOrb") * rows(0:(n - 1)),
    )
    for (name, body) in bad
        _with_in_file(body) do nl
            @test_throws ErrorException MOI.parton_pair_read_in!(data, nl)
        end
    end
    _with_in_file(head * rows(0:(n - 1))) do nl
        @test MOI.parton_pair_read_in!(data, nl) !== nothing
        @test all(==(ComplexF64(0.5, -0.25)), data.parton_pair_params)
    end
end

@testset "門番: 正常入力は通り、違反は個別に検出する" begin
    ctx = MOI.serial_context()
    @test MOI.validate_parton_pair_inputs(pair_ok_data(), ctx) === nothing
    violations = Dict(
        "PartonMode=1" => d -> d.modpara.parton_mode = 1,
        "KN 奇数" => d -> (d.modpara.nelec = 3; d.modpara.nflavor = 3),
        "NFlavor=1" => d -> (d.modpara.nflavor = 1; d.modpara.nelec = 2),
        "PartonOrbParam≠0" => d -> d.modpara.parton_orb_param = 1,
        "PartonOccMode≠0" => d -> d.modpara.parton_occ_mode = 1,
        "NExUpdatePath≠6" => d -> d.modpara.nex_update_path = 1,
        "def なし" => d -> (empty!(d.parton_pair_terms); empty!(d.parton_pair_params)),
        "p の長さ違い" => d -> push!(d.parton_pair_params, 0),
        "p が非有限" => d -> (d.parton_pair_params[1] = NaN),
        "def の範囲外" => d -> (d.parton_pair_terms[1] =
            MVMCExpertModeParsers.PartonPairOrbitalTerm(99, 0, 1, 1, 0, 1)),
        "pmfpara が混ざる" => d -> push!(d.pmfpara_terms,
            MVMCExpertModeParsers.PartonMFParaTerm(0, 0, 1, 0, 0, ComplexF64(1), true)),
        "Φ が混ざる" => d -> (d.pmforb_params = ComplexF64[1.0]),
        "既存スピン Orbital" => d -> push!(d.orbital_terms,
            MVMCExpertModeParsers.OrbitalTerm(0, 1, 0, ComplexF64(0), true, 1)),
        "physhop なし" => d -> empty!(d.physhop_terms),
        "フラグの長さ違い" => d -> push!(d.optimization_flags, true),
        "f が全凍結" => d -> (d.optimization_flags .= false),
    )
    for (name, mutate!) in violations
        data = pair_ok_data()
        mutate!(data)
        @test_throws Exception MOI.validate_parton_pair_inputs(data, ctx)
    end
    # KN 奇数は理由がわかるメッセージで落ちる
    data = pair_ok_data(n_site = 6, n_elec = 3, n_flavor = 3)
    err = try MOI.validate_parton_pair_inputs(data, ctx); nothing catch e; e end
    @test err isa ErrorException && occursin("odd", err.msg)
end

@testset "門番: namelist に α・Φ・上流 Orbital の入力があれば拒否" begin
    for key in ("PartonMFTrans", "PartonMFPara", "InPmfPara", "InPmfOcc", "InPmfOrb",
                "Orbital", "OrbitalAntiParallel", "OrbitalParallel", "OrbitalGeneral",
                "InOrbital", "InOrbitalGeneral")
        mktempdir() do dir
            write(joinpath(dir, "namelist.def"), "$key  x.def\n")
            @test_throws ErrorException MOI.parton_pair_reject_other_inputs(
                joinpath(dir, "namelist.def"))
        end
    end
    mktempdir() do dir
        write(joinpath(dir, "namelist.def"), "PartonPairOrbital pair.def\nPhysHop h.def\n")
        @test MOI.parton_pair_reject_other_inputs(joinpath(dir, "namelist.def")) === nothing
    end
end

@testset "テンプレートの器を data から組む" begin
    data = pair_ok_data()
    src = MOI.parton_pair_build_source(data)
    @test src isa MOI.PartonPairOrbital
    @test src.n_idx == length(data.parton_pair_params)
    t = data.parton_pair_terms[4]
    α = MOI.parton_pair_state(5, t.site1 + 1, t.flavor1 + 1)
    β = MOI.parton_pair_state(5, t.site2 + 1, t.flavor2 + 1)
    @test src.pair_matrix[α, β] == 2 * t.sign * data.parton_pair_params[t.idx + 1]
end
