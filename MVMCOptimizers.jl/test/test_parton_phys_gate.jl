"""
DESIGN_parton.md §13.3: 物理量測定の門番
--- parton-mode (fork addition) ---
"""

using Test
using MVMCExpertModeParsers
using MVMCOptimizers

"NVMCCalMode = 1 で全 1 体と密度相関を持つ正常な入力(toy_mf_data、4 サイト)。"
function _phys_gate_data()
    data = toy_mf_data()
    data.modpara.vmc_calc_mode = 1
    data.modpara.n_data_qty_smp = 1
    L = data.modpara.nsite
    data.green_one_terms = [MVMCExpertModeParsers.GreenOneTerm(i, j, :up, :up)
                            for i = 0:(L - 1) for j = 0:(L - 1)]
    data.green_two_terms = [MVMCExpertModeParsers.GreenTwoTerm(i, i, j, j, :up, :up, :up, :up)
                            for i = 0:(L - 1) for j = 0:(L - 1)]
    return data
end

@testset "§13.3 NVMCCalMode と NDataQtySmp" begin
    mp = _phys_gate_data().modpara
    @test MVMCOptimizers.validate_parton_modpara(mp) === nothing
    mp.n_data_qty_smp = 0
    @test_throws ErrorException MVMCOptimizers.validate_parton_modpara(mp)
    mp.n_data_qty_smp = 1
    mp.vmc_calc_mode = 2
    @test_throws ErrorException MVMCOptimizers.validate_parton_modpara(mp)
end

@testset "§13.3 Green の入力" begin
    gate = MVMCOptimizers.validate_parton_green_inputs
    @test gate(_phys_gate_data()) === nothing

    data = _phys_gate_data()
    data.green_one_terms[2] = MVMCExpertModeParsers.GreenOneTerm(0, 1, :down, :down)
    err = try gate(data); nothing catch e; e end
    @test err isa ErrorException && occursin("spin", err.msg)

    data = _phys_gate_data()
    data.green_two_terms[3] = MVMCExpertModeParsers.GreenTwoTerm(0, 0, 1, 1, :up, :up, :down, :down)
    @test_throws ErrorException gate(data)

    data = _phys_gate_data()
    push!(data.green_one_terms, MVMCExpertModeParsers.GreenOneTerm(0, data.modpara.nsite, :up, :up))
    @test_throws ErrorException gate(data)

    data = _phys_gate_data()
    push!(data.green_two_terms, MVMCExpertModeParsers.GreenTwoTerm(0, 0, -1, -1, :up, :up, :up, :up))
    @test_throws ErrorException gate(data)

    data = _phys_gate_data()
    push!(data.green_two_ex_terms, MVMCExpertModeParsers.GreenTwoExTerm(0, 0, 1, 0, 1, 0, 2, 0))
    err = try gate(data); nothing catch e; e end
    @test err isa ErrorException && occursin("TwoBodyGEx", err.msg)

    data = _phys_gate_data()
    push!(data.green_two_terms, MVMCExpertModeParsers.GreenTwoTerm(0, 1, 2, 3, :up, :up, :up, :up))
    err = try gate(data); nothing catch e; e end
    @test err isa ErrorException
    @test occursin("two_particle", err.msg) && occursin("(0, 1, 2, 3)", err.msg)
    @test occursin("term $(length(data.green_two_terms))", err.msg)
end

@testset "§13.3 NVMCCalMode = 0 で Green の def があれば warn" begin
    data = _phys_gate_data()
    data.modpara.vmc_calc_mode = 0
    @test_logs (:warn, r"OneBodyG") MVMCOptimizers.validate_parton_green_inputs(data)
    empty!(data.green_one_terms)
    empty!(data.green_two_terms)
    @test_logs MVMCOptimizers.validate_parton_green_inputs(data)
end

@testset "§13.2 固定パラメータの必須入力" begin
    req = MVMCOptimizers.parton_phys_require_fixed_inputs
    dir = mktempdir()
    nl = joinpath(dir, "namelist.def")
    write(nl, "ModPara modpara.def\n")
    data = _phys_gate_data()
    @test req(data, nl; orbital_given = true, orbital_hint = "unused") === nothing

    err = try req(data, nl; orbital_given = false,
                  orbital_hint = "add InPmfOrb (use zqp_pmforb_opt.dat)"); nothing catch e; e end
    @test err isa ErrorException && occursin("add InPmfOrb (use zqp_pmforb_opt.dat)", err.msg)

    _add_ring_jastrow!(data, [0.1, 0.2])
    err = try req(data, nl; orbital_given = true, orbital_hint = ""); nothing catch e; e end
    @test err isa ErrorException && occursin("InJastrow", err.msg)

    write(nl, "ModPara modpara.def\nInJastrow InJastrow.def\n")
    # 最終レビュー Important 1: 行があってもファイルが無い / 4 行ヘッダで idx 0 が落ちる → error
    err = try req(data, nl; orbital_given = true, orbital_hint = ""); nothing catch e; e end
    @test err isa ErrorException && occursin("not found", err.msg)
    head4 = "===============================\nNJastrowIdx 2\n===============================\n" *
            "===============================\n"
    write(joinpath(dir, "InJastrow.def"), head4 * "0 0.1 0.0\n1 0.2 0.0\n")   # zqp_jastrow_opt.dat の形
    err = try req(data, nl; orbital_given = true, orbital_hint = ""); nothing catch e; e end
    @test err isa ErrorException && occursin("5-line header", err.msg)
    write(joinpath(dir, "InJastrow.def"),
          head4 * "===============================\n0 0.1 0.0\n1 0.2 0.0\n")    # 正しい 5 行ヘッダ
    @test req(data, nl; orbital_given = true, orbital_hint = "") === nothing
end
