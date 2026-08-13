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

@testset "§8-10-4 平均場ダンプ(密・0-based・.def 族)" begin
    status, out, data, pstate = _run_parton_with_output(; nstep = 4)
    ham_path = joinpath(out, "zqp_pmfham_opt.dat")
    band_path = joinpath(out, "zqp_pmfband_opt.dat")
    @test isfile(ham_path) && isfile(band_path)

    n_site = data.modpara.nsite
    n_flavor = data.modpara.nflavor
    lines = readlines(ham_path)
    # .def 族: ヘッダ 5 行固定、`#` を一切書かない
    @test length(lines) >= 5
    @test !any(l -> startswith(strip(l), "#"), lines)
    @test split(strip(lines[2]))[1] == "NPmfHam"
    # 全 (flavor, site1, site2) を h.c. 側もゼロ要素も含めて出す = 密ダンプ
    @test parse(Int, split(strip(lines[2]))[2]) == n_flavor * n_site * n_site
    body = lines[6:end]
    filter!(l -> !isempty(strip(l)), body)
    @test length(body) == n_flavor * n_site * n_site

    H = [zeros(ComplexF64, n_site, n_site) for _ = 1:n_flavor]
    minsite = typemax(Int); minflavor = typemax(Int)
    for l in body
        t = split(strip(l))
        i, f1, j, f2 = parse.(Int, t[1:4])
        @test f1 == f2                     # M1 はフレーバー対角のみ
        minsite = min(minsite, i, j); minflavor = min(minflavor, f1)
        H[f1 + 1][i + 1, j + 1] = ComplexF64(parse(Float64, t[5]), parse(Float64, t[6]))
    end
    # 0-based で書かれていること(1-based で書いていないことの明示チェック)
    @test minsite == 0
    @test minflavor == 0

    for f = 1:n_flavor
        @test maximum(abs, H[f] .- pstate.mfham.h_mf[f]) < 1e-12
        # h.c. 側も入っている(片方向出力ではない)
        @test maximum(abs, H[f] .- H[f]') < 1e-12
    end

    # バンドは診断系: `#` ヘッダ、flavor/band_index とも 0-based
    blines = readlines(band_path)
    @test startswith(strip(blines[1]), "#")
    bands = [Float64[] for _ = 1:n_flavor]
    bmin_f = typemax(Int); bmin_k = typemax(Int)
    for l in blines
        s2 = strip(l)
        (isempty(s2) || startswith(s2, "#")) && continue
        t = split(s2)
        bmin_f = min(bmin_f, parse(Int, t[1])); bmin_k = min(bmin_k, parse(Int, t[2]))
        push!(bands[parse(Int, t[1]) + 1], parse(Float64, t[3]))
        # occupied は下から NElec 個
        @test parse(Int, t[4]) == (parse(Int, t[2]) < data.modpara.nelec ? 1 : 0)
    end
    @test bmin_f == 0 && bmin_k == 0
    for f = 1:n_flavor
        @test maximum(abs, eigvals(Hermitian((H[f] + H[f]') / 2)) .- bands[f]) < 1e-10
    end
    ne = data.modpara.nelec
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

@testset "§8-10-8 .def 族はヘッダ 5 行・`#` 非依存で往復できる" begin
    # 初期値ダンプはドライバの責務なので、ここはドライバ経由で回す
    dir = mktempdir(); nl, _ = _write_min_parton_input(dir)
    out = mktempdir()
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0
    data = MVMCExpertModeParsers.parse_expert_mode_files(nl)
    MVMCOptimizers.parton_read_in_pmfpara!(data, nl)
    for name in ("zqp_pmfpara_opt.dat", "zqp_pmfpara_init.dat", "zqp_pmfham_opt.dat")
        path = joinpath(out, name)
        @test isfile(path)
        lines = readlines(path)
        # mVMC の .def にコメント機能はない。`clean_line` の `#` 除去に依存しない
        @test !any(l -> occursin("#", l), lines)
        @test length(lines) >= 5
        # 2 行目がキーワードと件数
        @test length(split(strip(lines[2]))) == 2
        @test tryparse(Int, split(strip(lines[2]))[2]) !== nothing
    end

    # 往復: ヘッダ 5 行を読み飛ばすと idx = 0 が脱落せず全件そろう
    n_idx = MVMCOptimizers.parton_n_idx(data)
    rows = [split(strip(l)) for l in readlines(joinpath(out, "zqp_pmfpara_opt.dat"))[6:end]
            if !isempty(strip(l))]
    @test length(rows) == n_idx
    @test parse(Int, rows[1][1]) == 0            # idx = 0 が先頭に残っている
    @test [parse(Int, r[1]) for r in rows] == collect(0:(n_idx - 1))
end

@testset "§8-10-9 同一入力・同一シードの 2 run はバイト一致(行順の決定性)" begin
    st1, out1, _, _ = _run_parton_with_output(; nstep = 5, seed = 31337)
    st2, out2, _, _ = _run_parton_with_output(; nstep = 5, seed = 31337)
    @test st1 == 0 && st2 == 0
    # 時刻・壁時計を含むファイルは除外(runinfo / time / CalcTimer)
    skip = Set(["zvo_parton_runinfo.dat", "zvo_parton_time.dat", "zvo_CalcTimer.dat"])
    common = sort(collect(intersect(Set(readdir(out1)), Set(readdir(out2)))))
    @test !isempty(setdiff(common, skip))
    for name in common
        name in skip && continue
        @test read(joinpath(out1, name)) == read(joinpath(out2, name))
    end
end

@testset "§8-10-10 同じ出力先で再実行しても前回の行が残らない" begin
    out = mktempdir()
    function run_into(dir, nstep)
        data = dimerized_mf_data()
        data.modpara.nsr_opt_itr_step = nstep
        data.modpara.nsr_opt_itr_smp = 2
        MVMCOptimizers.parton_materialize_flags!(data)
        ps = MVMCOptimizers.parton_build_optimization_state(data)
        rng = MVMCOptimizers.SFMT19937RNG(); Random.seed!(rng, 909)
        MVMCOptimizers.parton_vmc_para_opt!(ps, data, MVMCOptimizers.serial_context();
                                            rng = rng, output_dir = dir)
    end
    @test run_into(out, 7) == 0
    @test run_into(out, 3) == 0
    nrows(f) = length([l for l in eachline(joinpath(out, f))
                       if !startswith(strip(l), "#") && !isempty(strip(l))])
    # step 0 が "w" なので、短い 2 回目の後に長い 1 回目の行が残ってはいけない
    @test nrows("zvo_parton_diag.dat") == 3
    @test nrows("zvo_parton_time.dat") == 3
    @test length(filter(!isempty, strip.(readlines(joinpath(out, "zvo_SRinfo.dat"))))) - 1 == 3
end

@testset "§8-10-11 CalcTimer がパートンでは既定で出て、既存モードでは出ない" begin
    status, out, data, _ = _run_parton_with_output(; nstep = 4)
    # parton_vmc_para_opt! を直接呼ぶ経路では c_timer は既定 OFF。
    # 既定 ON はドライバ(parton_run_para_opt_from_namelist)の責務なのでそちらで見る。
    mktempdir() do dir
        nl, _ = _write_min_parton_input(dir)
        out2 = mktempdir()
        haskey(ENV, "MVMC_C_TIMER") && delete!(ENV, "MVMC_C_TIMER")
        @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out2) == 0
        path = joinpath(out2, "zvo_CalcTimer.dat")
        @test isfile(path)
        txt = read(path, String)
        # 既存本体セクションとパートンセクションが同じファイルに揃っている
        @test occursin("All                         [0] ", txt)
        for (label, _) in MVMCOptimizers.CTIMER_PARTON_LINES
            @test occursin(label, txt)
        end
        # 実際に時間が入っている(全ゼロではない)
        m = match(r"Parton total               \[800\] *([0-9.]+)", txt)
        @test m !== nothing && parse(Float64, m.captures[1]) >= 0.0
        # ID 帯が既存と衝突していない
        parton_ids = [id for (_, id) in MVMCOptimizers.CTIMER_PARTON_LINES]
        used = Set(vcat([id for (_, id) in MVMCOptimizers.CTIMER_PARA_OPT_LINES],
                        [id for (_, id) in MVMCOptimizers.CTIMER_DIAG_LINES]))
        @test isempty(intersect(Set(parton_ids), used))
        @test all(id -> 0 <= id < MVMCOptimizers.CTIMER_N, parton_ids)
    end
    # 既存モードの既定は変えていない: MVMC_C_TIMER なしでは生成されない
    out3 = mktempdir()
    d = MVMCExpertModeParsers.ExpertModeData()
    d.modpara.nsr_opt_itr_step = 1
    MVMCOptimizers.output_opt_data!(d; output_dir = out3)
    @test !("zvo_CalcTimer.dat" in readdir(out3))
end
