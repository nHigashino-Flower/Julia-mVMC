"""
DESIGN_parton.md §13.2 / §13.4 / §13.5-3,4: α・Φ・Pf の入口で NVMCCalMode = 1 を測定へ分岐する
--- parton-mode (fork addition) ---

どのモードも「短い最適化 → その `_opt.dat` を In にして測定」の流れを通す。
"""

using Test
using MVMCExpertModeParsers
using MVMCOptimizers

const PHYS_RUN = MVMCOptimizers.parton_run_para_opt_from_namelist

"モードごとの最適化入力を作り、最適化を回して (namelist, In ファイルの対) を返す。"
function _phys_optimized_input(mode::Symbol, dir)
    nl = mode === :alpha ? first(_write_min_parton_input(dir)) :
         mode === :orbparam ? _write_min_orbparam_input(dir) :
         _write_min_pair_input(dir)
    opt_out = joinpath(dir, "opt_out")
    @test PHYS_RUN(nl; output_dir = opt_out) == 0
    in_files = mode === :alpha ?
        ["InPmfPara" => joinpath(opt_out, "zqp_pmfpara_opt.dat"),
         "InPmfOcc" => joinpath(opt_out, "zqp_pmfocc_opt.dat")] :
        mode === :orbparam ? ["InPmfOrb" => joinpath(opt_out, "zqp_pmforb_opt.dat")] :
        ["InPartonPairOrbital" => joinpath(opt_out, "zqp_parton_pair_opt.dat")]
    return nl, in_files
end

@testset "§13.4 最適化 → In 化 → 測定: $(mode)" for mode in (:alpha, :orbparam, :pair)
    dir = mktempdir()
    nl, in_files = _phys_optimized_input(mode, dir)
    # Φ の最小入力は InPmfOrb を最初から持つので、最適化の出力で置き換える
    mode === :orbparam && write(nl, replace(read(nl, String), r"InPmfOrb\s+InPmfOrb.def\n" => ""))
    nl = phys_convert_to_measurement!(dir, in_files; n_site = 4, n_data_qty_smp = 2)
    out = joinpath(dir, "phys_out")
    @test PHYS_RUN(nl; output_dir = out) == 0
    @test !isfile(joinpath(out, "zvo_SRinfo.dat"))            # SR ループに入っていない
    for (n_sum, e, e_green) in phys_check_outputs(nl, out, 2)
        @test isapprox(n_sum, 2; atol = 1e-10)                  # NElec = 2
        @test isapprox(e, e_green; atol = 1e-10)
        @test abs(e) > 0.1
    end
end

"pmfpara.def の値列を落とす(乱数で埋まる idx を作る)。"
function _phys_strip_pmfpara_values!(dir)
    path = joinpath(dir, "pmfpara.def")
    lines = map(readlines(path)) do l
        tok = split(l)
        length(tok) == 7 ? join(tok[1:5], " ") : l
    end
    write(path, join(lines, "\n") * "\n")
end

"err が ErrorException で、メッセージに needle を含むか。"
_phys_err_has(f, needle) = (err = try f(); nothing catch e; e end;
                            err isa ErrorException && occursin(needle, err.msg))

@testset "§13.2 固定パラメータが欠けると error" begin
    # α: pmfpara.def に値が無く InPmfPara も無い → 乱数の idx が残る
    dir = mktempdir()
    nl, in_files = _phys_optimized_input(:alpha, dir)
    _phys_strip_pmfpara_values!(dir)
    nl = phys_convert_to_measurement!(dir, filter(p -> p.first == "InPmfOcc", in_files); n_site = 4)
    @test _phys_err_has(() -> PHYS_RUN(nl; output_dir = joinpath(dir, "o")), "InPmfPara")

    # Review Focus 1: α で InPmfOcc が無い(MOM の枝の解を aufbau で組み直す事故を防ぐ)
    dir = mktempdir()
    nl, in_files = _phys_optimized_input(:alpha, dir)
    nl = phys_convert_to_measurement!(dir, filter(p -> p.first == "InPmfPara", in_files); n_site = 4)
    @test _phys_err_has(() -> PHYS_RUN(nl; output_dir = joinpath(dir, "o")), "InPmfOcc")

    # Φ: InPmfOrb が無い
    dir = mktempdir()
    nl, _ = _phys_optimized_input(:orbparam, dir)
    write(nl, replace(read(nl, String), r"InPmfOrb\s+InPmfOrb.def\n" => ""))
    nl = phys_convert_to_measurement!(dir, Pair{String,String}[]; n_site = 4)
    @test _phys_err_has(() -> PHYS_RUN(nl; output_dir = joinpath(dir, "o")), "InPmfOrb")

    # Pf: InPartonPairOrbital が無い
    dir = mktempdir()
    nl, _ = _phys_optimized_input(:pair, dir)
    nl = phys_convert_to_measurement!(dir, Pair{String,String}[]; n_site = 4)
    @test _phys_err_has(() -> PHYS_RUN(nl; output_dir = joinpath(dir, "o")), "InPartonPairOrbital")

    # Jastrow の項があり InJastrow が無い
    dir = mktempdir()
    nl, in_files = _phys_optimized_input(:alpha, dir)
    _add_jastrow_def!(dir, 4)
    nl = phys_convert_to_measurement!(dir, in_files; n_site = 4)
    @test _phys_err_has(() -> PHYS_RUN(nl; output_dir = joinpath(dir, "o")), "InJastrow")
end

@testset "Review Focus 5: NVMCCalMode = 0 では Green の def を読まず、出力はバイト一致" begin
    dir1, dir2 = mktempdir(), mktempdir()
    nl1, _ = _write_min_parton_input(dir1)
    nl2, _ = _write_min_parton_input(dir2)
    phys_write_green_defs(dir2, 4)
    open(nl2, "a") do io
        println(io, "OneBodyG       greenone.def")
        println(io, "TwoBodyG       greentwo.def")
    end
    @test PHYS_RUN(nl1; output_dir = joinpath(dir1, "o")) == 0
    status = @test_logs (:warn, r"OneBodyG") match_mode = :any PHYS_RUN(nl2; output_dir = joinpath(dir2, "o"))
    @test status == 0
    @test read(joinpath(dir1, "o", "zvo_out.dat")) == read(joinpath(dir2, "o", "zvo_out.dat"))
    @test !isfile(joinpath(dir2, "o", "zvo_cisajs_001.dat"))
end
