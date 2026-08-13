"""
パートンモードの門番(入力契約の執行)のテスト --- parton-mode (fork addition) ---

DESIGN_parton.md §2(入力契約)と §3.3 の「門番」行に対応する。
"""

using Test
using MVMCExpertModeParsers
using MVMCOptimizers

"門番を素通りする最小の正常入力(4 サイト鎖・F=2・最近接 1 変分)。"
function _parton_ok_data()
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite = 4
    mp.nelec = 2
    mp.nflavor = 2
    mp.parton_mode = 1
    mp.two_sz = 0
    mp.complex_flag = 1
    mp.nex_update_path = 6
    mp.ncond = -1
    mp.nlocspin = 0
    mp.vmc_calc_mode = 0
    mp.nsrcg = 0
    mp.lanczos_mode = 0
    mp.nsp_gauss_leg = 1
    mp.nsp_stot = 0
    mp.n_orbital_idx = 0
    mp.nneuron = 0
    mp.nsplit_size = 1

    push!(
        data.pmftrans_terms,
        MVMCExpertModeParsers.PartonMFTransTerm(0, 0, 1, 0, ComplexF64(-1, 0), false),
        MVMCExpertModeParsers.PartonMFTransTerm(0, 1, 1, 1, ComplexF64(-1, 0), false),
    )
    push!(
        data.pmfpara_terms,
        MVMCExpertModeParsers.PartonMFParaTerm(0, 0, 1, 0, 0, ComplexF64(-1, 0), true),
        MVMCExpertModeParsers.PartonMFParaTerm(0, 1, 1, 1, 0, ComplexF64(-1, 0), true),
    )
    push!(data.physhop_terms,
          MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(-1, 0), false))

    # n_proj = 0, n_pmf = 1 → flags 長は 2
    data.optimization_flags = fill(true, 2)
    return data
end

@testset "門番: 正常入力は素通りする" begin
    data = _parton_ok_data()
    ctx = MVMCOptimizers.serial_context()
    @test MVMCOptimizers.validate_parton_inputs(data, ctx) === nothing
end

@testset "門番: modpara の違反を個別に検出" begin
    ctx = MVMCOptimizers.serial_context()
    violations = Dict(
        "PartonMode=0" => d -> d.modpara.parton_mode = 0,
        "PartonMode=2(予約)" => d -> d.modpara.parton_mode = 2,
        "NFlavor 欠落" => d -> d.modpara.nflavor = 0,
        "2Sz=-1(FSZ の罠)" => d -> d.modpara.two_sz = -1,
        "ComplexType=0" => d -> d.modpara.complex_flag = 0,
        "NExUpdatePath≠6" => d -> d.modpara.nex_update_path = 1,
        "NElec > NSite" => d -> d.modpara.nelec = 5,
        "NElec=0" => d -> d.modpara.nelec = 0,
        "NCond≠-1" => d -> d.modpara.ncond = 4,
        "NLocSpin≠0" => d -> d.modpara.nlocspin = 2,
        "NVMCCalMode=1" => d -> d.modpara.vmc_calc_mode = 1,
        "NSRCG≠0" => d -> d.modpara.nsrcg = 1,
        "NLanczosMode≠0" => d -> d.modpara.lanczos_mode = 1,
        "NSPGaussLeg≠1" => d -> d.modpara.nsp_gauss_leg = 2,
        "NOrbitalIdx≠0" => d -> d.modpara.n_orbital_idx = 3,
        "NNeuron≠0" => d -> d.modpara.nneuron = 2,
        "NSplitSize≠1" => d -> d.modpara.nsplit_size = 2,
    )
    for (name, mutate!) in violations
        data = _parton_ok_data()
        mutate!(data)
        @test_throws Exception MVMCOptimizers.validate_parton_inputs(data, ctx)
    end
end

@testset "門番: 入力ファイルの違反を個別に検出" begin
    ctx = MVMCOptimizers.serial_context()
    violations = Dict(
        "pmfpara 欠落" => d -> empty!(d.pmfpara_terms),
        "pmftrans 欠落" => d -> empty!(d.pmftrans_terms),
        "physhop 欠落" => d -> empty!(d.physhop_terms),
        "physhop 逆向き重複" => d -> push!(
            d.physhop_terms,
            MVMCExpertModeParsers.PhysHopTerm(1, 0, ComplexF64(1, 0), false),
        ),
        "physhop 同一重複" => d -> push!(
            d.physhop_terms,
            MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(1, 0), false),
        ),
        "physhop 対角" => d -> push!(
            d.physhop_terms,
            MVMCExpertModeParsers.PhysHopTerm(2, 2, ComplexF64(1, 0), false),
        ),
        "physhop 範囲外" => d -> push!(
            d.physhop_terms,
            MVMCExpertModeParsers.PhysHopTerm(0, 9, ComplexF64(1, 0), false),
        ),
        "pmfpara フレーバー混成" => d -> (d.pmfpara_terms[1].flavor2 = 1),
        "pmfpara サイト範囲外" => d -> (d.pmfpara_terms[1].site2 = 9),
        "pmfpara フレーバー範囲外" => d -> begin
            d.pmfpara_terms[1].flavor1 = 5
            d.pmfpara_terms[1].flavor2 = 5
        end,
        "pmftrans 逆向き重複" => d -> push!(
            d.pmftrans_terms,
            MVMCExpertModeParsers.PartonMFTransTerm(1, 0, 0, 0, ComplexF64(-1, 0), false),
        ),
        "CoulombIntra 使用" => d -> push!(
            d.coulomb_intra_terms,
            MVMCExpertModeParsers.CoulombIntraTerm(0, 1.0),
        ),
        "Orbital 併存" => d -> push!(
            d.orbital_terms,
            MVMCExpertModeParsers.OrbitalTerm(0, 1, 0, ComplexF64(1, 0), false, 1),
        ),
    )
    for (name, mutate!) in violations
        data = _parton_ok_data()
        mutate!(data)
        @test_throws Exception MVMCOptimizers.validate_parton_inputs(data, ctx)
    end
end

@testset "門番: OptFlag 配列の長さと可動成分" begin
    ctx = MVMCOptimizers.serial_context()

    # 長さ不足: SR が MF ブロックを黙って無視する最悪の沈黙故障を弾く
    data = _parton_ok_data()
    data.optimization_flags = fill(true, 1)
    @test_throws Exception MVMCOptimizers.validate_parton_inputs(data, ctx)

    # 長すぎるのもエラー(NPara の食い違い)
    data = _parton_ok_data()
    data.optimization_flags = fill(true, 4)
    @test_throws Exception MVMCOptimizers.validate_parton_inputs(data, ctx)

    # MF 成分が全凍結: 最適化するものが無い
    data = _parton_ok_data()
    data.optimization_flags = fill(false, 2)
    @test_throws Exception MVMCOptimizers.validate_parton_inputs(data, ctx)

    # 実部だけ可動なら通る(オンサイト群の Im 凍結が通常形)
    data = _parton_ok_data()
    data.optimization_flags = [true, false]
    @test MVMCOptimizers.validate_parton_inputs(data, ctx) === nothing
end
