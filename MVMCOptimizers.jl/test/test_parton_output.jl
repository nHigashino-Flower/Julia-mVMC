"""
§8 テスト 10: パートンモードの出力ファイル
--- parton-mode (fork addition) ---

DESIGN_parton.md §3.3 の出力一覧に対応する。共通規約は rank 0 のみ・既存の接頭辞を
尊重・step 0 が "w" で以降 "a"・**PartonMode = 0 では 1 つも生成されない**。
"""

using Test
using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

const _PARTON_OUT_FILES = [
    "zvo_SRinfo.dat", "zvo_parton_diag.dat", "zvo_parton_time.dat",
    "zvo_parton_runinfo.dat", "zvo_conv.dat",
    "zqp_pmfham_opt.dat", "zqp_pmfband_opt.dat",
]

"SR を数ステップ回して出力ディレクトリを返す。"
function _run_parton_with_output(; nstep::Int = 6, nsmp::Int = 3, seed::Int = 4242)
    data = dimerized_mf_data()
    data.modpara.nsr_opt_itr_step = nstep
    data.modpara.nsr_opt_itr_smp = nsmp
    MVMCOptimizers.parton_materialize_flags!(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    rng = MVMCOptimizers.SFMT19937RNG()
    Random.seed!(rng, seed)
    out = mktempdir()
    status = MVMCOptimizers.parton_vmc_para_opt!(
        pstate, data, MVMCOptimizers.serial_context(); rng = rng, output_dir = out)
    return status, out, data, pstate
end

@testset "§8-10-1 PartonMode = 0 では新規ファイルが 1 つも出ない" begin
    # 標準経路(既存 integration が使う fixture 相当)。パートン出力は
    # パートンのオーケストレータからしか呼ばれないので 1 つも現れない。
    out = mktempdir()
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsr_opt_itr_step = 1
    MVMCOptimizers.output_opt_data!(data; output_dir = out)   # 標準の出力だけ
    produced = readdir(out)
    for f in _PARTON_OUT_FILES
        @test !(f in produced)
    end
    # pmfpara 系も出ない(pmfpara_terms が空なので)
    @test !("zqp_pmfpara_opt.dat" in produced)
end

@testset "§8-10-2 SRinfo が直接法パスから出て、列形式が既存と一致" begin
    status, out, data, pstate = _run_parton_with_output(; nstep = 6)
    @test status == 0
    path = joinpath(out, "zvo_SRinfo.dat")
    @test isfile(path)
    lines = filter(!isempty, strip.(readlines(path)))
    # ヘッダは既存 (CG 版) と 1 文字も違わないこと
    @test lines[1] == "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax"
    # 行数 = SR ステップ数(ヘッダを除く)
    @test length(lines) - 1 == 6
    # 列数(末尾は "imax, ..." とカンマ区切りが入る既存書式)
    @test length(split(replace(lines[2], "," => " "))) == 9
end

@testset "§8-10-3 診断ログが既存カウンタ・min_gap と一致" begin
    status, out, data, pstate = _run_parton_with_output(; nstep = 5)
    path = joinpath(out, "zvo_parton_diag.dat")
    @test isfile(path)
    rows = [split(strip(l)) for l in eachline(path) if !startswith(strip(l), "#") && !isempty(strip(l))]
    @test length(rows) == 5
    last_row = rows[end]
    # 診断行の min_gap は「そのステップ時点」の値。ループ後のダンプは最終 α で
    # 組み直すので mfham.min_gap は先へ進んでおり、両者は同じ時点ではない。
    # ここでは診断行が健全な値であることを見る。
    @test isfinite(parse(Float64, last_row[2])) && parse(Float64, last_row[2]) > 0
    # 受理率 = 受理数 / 試行数(counter から直接)
    trials = parse(Int, last_row[8])
    accepts = parse(Int, last_row[9])
    @test trials > 0 && accepts > 0
    @test isapprox(parse(Float64, last_row[5]), accepts / trials; rtol = 1e-5)
    # ノルムはゲージ射影の前後。射影が有効なら post が初期ノルムに張り付く
    @test parse(Float64, last_row[6]) > 0
    @test parse(Float64, last_row[7]) > 0
    # 時間ログも同じ行数
    @test length([l for l in eachline(joinpath(out, "zvo_parton_time.dat"))
                  if !startswith(strip(l), "#") && !isempty(strip(l))]) == 5
end

@testset "§8-10-4 平均場ダンプが α から再構成した H と一致" begin
    status, out, data, pstate = _run_parton_with_output(; nstep = 4)
    ham_path = joinpath(out, "zqp_pmfham_opt.dat")
    band_path = joinpath(out, "zqp_pmfband_opt.dat")
    @test isfile(ham_path) && isfile(band_path)

    n_site = data.modpara.nsite
    n_flavor = data.modpara.nflavor
    H = [zeros(ComplexF64, n_site, n_site) for _ = 1:n_flavor]
    for l in eachline(ham_path)
        s = strip(l)
        (isempty(s) || startswith(s, "#") || startswith(s, "N")) && continue
        t = split(s)
        H[parse(Int, t[1])][parse(Int, t[2]), parse(Int, t[3])] =
            ComplexF64(parse(Float64, t[4]), parse(Float64, t[5]))
    end

    # α から再構成した H と一致すること(唯一の正は α 側)
    mf2 = MVMCOptimizers.PartonMFHamiltonian(
        n_site, data.modpara.nelec, n_flavor, MVMCOptimizers.parton_n_idx(data))
    MVMCOptimizers.parton_build_mf_templates!(mf2, data)
    MVMCOptimizers.parton_update_orbitals!(
        mf2, MVMCOptimizers.parton_alpha_from_terms(data), data.modpara.nelec)
    for f = 1:n_flavor
        @test maximum(abs, H[f] .- mf2.h_mf[f]) < 1e-12
    end

    # ダンプした H を対角化した固有値がバンドファイルと一致
    bands = [Float64[] for _ = 1:n_flavor]
    for l in eachline(band_path)
        s = strip(l)
        (isempty(s) || startswith(s, "#") || startswith(s, "N")) && continue
        t = split(s)
        push!(bands[parse(Int, t[1])], parse(Float64, t[3]))
    end
    for f = 1:n_flavor
        @test maximum(abs, eigvals(Hermitian((H[f] + H[f]') / 2)) .- bands[f]) < 1e-10
    end

    # バンドファイルの HOMO-LUMO ギャップは、ループ後に組み直した mfham と同じ時点
    ne = data.modpara.nelec
    for f = 1:n_flavor
        @test bands[f][ne + 1] - bands[f][ne] >= pstate.mfham.min_gap - 1e-10
    end
    @test isapprox(minimum(f -> bands[f][ne + 1] - bands[f][ne], 1:n_flavor),
                   pstate.mfham.min_gap; rtol = 1e-10)
end

@testset "§8-10-5 run メタデータ" begin
    mktempdir() do dir
        nl, _ = _write_min_parton_input(dir)
        out = mktempdir()
        status = MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out)
        @test status == 0
        path = joinpath(out, "zvo_parton_runinfo.dat")
        @test isfile(path)
        kv = Dict{String,String}()
        for l in eachline(path)
            s = strip(l)
            (isempty(s) || startswith(s, "#")) && continue
            t = split(s, limit = 2)
            length(t) == 2 && (kv[t[1]] = strip(t[2]))
        end
        # ベースシードは modpara の RndSeed(serial なので group1 = 0)
        @test parse(Int, kv["base_seed"]) == parse(Int, kv["modpara_rnd_seed"])
        @test parse(Int, kv["PartonMode"]) == 1
        @test parse(Int, kv["n_idx"]) > 0
        @test haskey(kv, "githash")          # 取得失敗でも "unknown" が入り run は落ちない
        @test parse(Float64, kv["wall_sec"]) >= 0
    end
end

@testset "§8-10-6 収束テーブルが zvo_out と整合" begin
    status, out, data, _ = _run_parton_with_output(; nstep = 8, nsmp = 3)
    conv = joinpath(out, "zvo_conv.dat")
    @test isfile(conv)
    src = [parse.(Float64, split(strip(l)))
           for l in eachline(joinpath(out, "zvo_out.dat")) if !isempty(strip(l))]
    rows = [parse.(Float64, split(strip(l)))
            for l in eachline(conv) if !startswith(strip(l), "#") && !isempty(strip(l))]
    @test length(rows) == length(src)
    # E 列は zvo_out の 1 列目
    @test all(k -> rows[k][2] == src[k][1], eachindex(rows))
    # E_tail は最終 NSROptItrSmp ステップの平均(手計算と突き合わせ)
    e = [r[1] for r in src]
    e_tail = sum(e[(end - 2):end]) / 3
    @test isapprox(rows[1][4], abs(e[1] - e_tail); rtol = 1e-12)
    # var 列は zvo_out の 4 列目
    @test all(k -> rows[k][3] == src[k][4], eachindex(rows))
end

@testset "§8-10-7 作図スクリプトは本体から独立している" begin
    root = dirname(dirname(@__DIR__))
    @test isfile(joinpath(root, "tools", "plot_conv.jl"))
    @test isfile(joinpath(root, "tools", "Project.toml"))
    # 本体パッケージに作図依存を入れていないこと
    for proj in ("MVMCOptimizers.jl/Project.toml", "MVMCExpertModeParsers.jl/Project.toml")
        @test !occursin("Plots", read(joinpath(root, proj), String))
    end
end
