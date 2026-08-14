"""
§8 テスト 17-3 以降: 占有追跡(MOM)による SR 漂流の根治
--- parton-mode (fork addition) ---

REPORT §15 で確定した機構への対処。主犯は「下から Ne 個を占有する」アウフバウ
規則による枝の乗り換えで、RedCut は症状だった(撤廃しても軌跡が変わらない)。
gap ~ 5e-6 に対し SR のステップ長が 0.0017 = 混成の効く境界層の幅の 300 倍
以上あるため、離散ステップから見ると Φ(α) は事実上不連続になる。

`PartonOccMode`:

- 0 = `aufbau`(既定): 下から Ne 個。**現行と完全に同一**
- 1 = `mom`: 前ステップの占有部分空間と重なり最大の Ne 本を選ぶ
- 2 以上は予約(明示的に拒否)

占有集合は導出量ではなく**状態の一部**として持ち、出力にも含める
(`<CParaFileHead>_pmfocc_{init,opt}.dat`)。これにより `(α*, O*)` の組で状態が
完全に決まり、履歴は出力の中に閉じ込められる。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "§8-17-3 PartonOccMode: 既定 aufbau / mom は 1 / 2 以上は拒否" begin
    # 既定は 0 = aufbau(既定挙動を変えないことの明示)
    @test MVMCExpertModeParsers.ModParaParameters().parton_occ_mode ==
          MVMCExpertModeParsers.DEFAULT_PARTON_OCC_MODE
    @test MVMCExpertModeParsers.DEFAULT_PARTON_OCC_MODE == 0

    # modpara.def から読める
    mktempdir() do dir
        path = joinpath(dir, "modpara.def")
        open(path, "w") do f
            println(f, "CDataFileHead zvo")
            println(f, "Nsite 4")
            println(f, "NElec 2")
            println(f, "PartonMode 1")
            println(f, "NFlavor 2")
            println(f, "PartonOccMode 1")
        end
        res = MVMCExpertModeParsers.parse_modpara_def(path)
        @test res.success
        @test res.data.parton_occ_mode == 1
    end

    # 門番: 2 以上は予約なので明示的に拒否する
    data = per_bond_mf_data(2; n_site = 4, n_elec = 2)
    data.modpara.parton_occ_mode = 2
    @test_throws ErrorException MVMCOptimizers.validate_parton_modpara(data.modpara)

    # 0 と 1 は通る
    for m in (0, 1)
        data.modpara.parton_occ_mode = m
        @test MVMCOptimizers.validate_parton_modpara(data.modpara) === nothing
    end
end

@testset "§8-17-4 占有選択: 部分空間の重なりで選ぶ" begin
    sel = MVMCOptimizers.parton_select_occupation

    # 参照なし(初回)は aufbau に落ちる
    U = Matrix{ComplexF64}(I, 4, 4)
    ev = [1.0, 2.0, 3.0, 4.0]
    @test sel(U, ev, nothing, 2) == [1, 2]

    # 参照が「下から 2 個」でない部分空間なら、そちらが選ばれる
    Φref = ComplexF64.(U[:, [2, 4]])
    @test sel(U, ev, Φref, 2) == [2, 4]

    # 返り値は昇順に整列している(添字順に依存しない)
    @test sel(U, ev, ComplexF64.(U[:, [4, 1]]), 2) == [1, 4]

    # 部分空間の重なりで測る: 参照が固有ベクトルの線形結合でも、その部分空間を張る
    # 2 本(ここでは 1 と 2)が選ばれる。個別軌道の一対一対応では拾えないケース。
    c = 1 / sqrt(2)
    Φmix = ComplexF64[c c; c -c; 0 0; 0 0]
    @test sel(U, ev, Φmix, 2) == [1, 2]

    # 位相の違いは重なりに効かない(|·|² で測るため)
    Φphase = ComplexF64.(U[:, [2, 4]]) .* cis(0.7)
    @test sel(U, ev, Φphase, 2) == [2, 4]

    # 同点のタイブレークは固有値昇順 → 添字昇順(決定論的)。
    # 参照が 4 本すべてに均等に重なると全スコアが等しくなる。
    Φflat = ComplexF64[0.5 0.5; 0.5 -0.5; 0.5 0.5; 0.5 -0.5]
    @test sel(U, ev, Φflat, 2) == [1, 2]

    # 固有値が縮退していてもタイブレークは添字昇順で決まる(再現性の要)
    ev_deg = [1.0, 1.0, 1.0, 1.0]
    @test sel(U, ev_deg, Φflat, 2) == [1, 2]
end

@testset "§8-17-5 契約0: 占有集合 O が状態に入り、MOM が枝を追う" begin
    # α = (0, 1, 2, 3): 準位は昇順のまま。占有はサイト 1, 2(band 1, 2)
    α1 = ComplexF64[0, 1, 2, 3]
    # α = (2.5, 1, 2, 3): サイト 1 が band 1 → band 3 へ動く(厳密な交差)。
    #   aufbau → band [1,2] = サイト 2, 3  … 占有からサイト 1 が抜ける = 枝の乗り換え
    #   mom    → band [1,3] = サイト 2, 1  … 前の占有部分空間を保つ
    α2 = ComplexF64[2.5, 1, 2, 3]

    @testset "aufbau(既定)は従来どおり下から Ne 個" begin
        data = onsite_crossing_data()
        mfham = build_crossing_mfham(data; occ_mode = 0)
        MVMCOptimizers.parton_update_orbitals!(mfham, α1, 2)
        @test mfham.occ[1] == [1, 2]
        MVMCOptimizers.parton_update_orbitals!(mfham, α2, 2)
        @test mfham.occ[1] == [1, 2]                       # band 番号としては不変
        # 占有軌道が実際に「サイト 2, 3」へ乗り換わっている(枝の不連続)
        @test abs2(mfham.orbitals[1][1, 1]) + abs2(mfham.orbitals[1][1, 2]) < 1e-20
    end

    @testset "mom は占有部分空間を追う" begin
        data = onsite_crossing_data()
        mfham = build_crossing_mfham(data; occ_mode = 1)
        MVMCOptimizers.parton_update_orbitals!(mfham, α1, 2)
        @test mfham.occ[1] == [1, 2]                       # 初回は aufbau に落ちる
        MVMCOptimizers.parton_update_orbitals!(mfham, α2, 2)
        @test mfham.occ[1] == [1, 3]
        # サイト 1 が占有に残っている(部分空間が保たれた)
        @test abs2(mfham.orbitals[1][1, 1]) + abs2(mfham.orbitals[1][1, 2]) ≈ 1.0
    end

    @testset "健全域では mom と aufbau が一致する" begin
        # 交差を起こさない α の列では、mom の選択は aufbau と同じでなければならない
        # (既定挙動を変えないことの担保。診断で確認済みの性質を恒久化する)
        data_a = onsite_crossing_data()
        data_m = onsite_crossing_data()
        ma = build_crossing_mfham(data_a; occ_mode = 0)
        mm = build_crossing_mfham(data_m; occ_mode = 1)
        for α in (ComplexF64[0, 1, 2, 3], ComplexF64[0.1, 1.2, 2.1, 3.3],
                  ComplexF64[0.2, 1.1, 2.4, 3.1])
            MVMCOptimizers.parton_update_orbitals!(ma, α, 2)
            MVMCOptimizers.parton_update_orbitals!(mm, α, 2)
            @test ma.occ[1] == mm.occ[1]
            @test ma.orbitals[1] == mm.orbitals[1]          # ビット一致
        end
    end
end

@testset "§8-17-6 min_gap は符号つき(占有↔非占有のエネルギー差)" begin
    α1 = ComplexF64[0, 1, 2, 3]
    α2 = ComplexF64[2.5, 1, 2, 3]

    # aufbau では従来と同じ「HOMO-LUMO」
    data = onsite_crossing_data()
    mfham = build_crossing_mfham(data; occ_mode = 0)
    MVMCOptimizers.parton_update_orbitals!(mfham, α1, 2)
    @test mfham.min_gap ≈ 1.0                              # ε_3 − ε_2 = 2 − 1

    # mom で非アウフバウ占有になると負を取りうる
    data_m = onsite_crossing_data()
    mm = build_crossing_mfham(data_m; occ_mode = 1)
    MVMCOptimizers.parton_update_orbitals!(mm, α1, 2)
    MVMCOptimizers.parton_update_orbitals!(mm, α2, 2)
    @test mm.occ[1] == [1, 3]
    # 占有 = {1.0, 2.5}、非占有 = {2.0, 3.0} → min(非占有) − max(占有) = 2.0 − 2.5
    @test mm.min_gap ≈ -0.5
end

@testset "§8-17-8 占有集合の出力(_pmfocc)と pmfband の occupied 列" begin
    α1 = ComplexF64[0, 1, 2, 3]
    α2 = ComplexF64[2.5, 1, 2, 3]
    data = onsite_crossing_data()
    mfham = build_crossing_mfham(data; occ_mode = 1)
    MVMCOptimizers.parton_update_orbitals!(mfham, α1, 2)
    MVMCOptimizers.parton_update_orbitals!(mfham, α2, 2)
    @test mfham.occ[1] == [1, 3]

    mktempdir() do dir
        path = joinpath(dir, "zqp_pmfocc_opt.dat")
        @test MVMCOptimizers.parton_write_pmfocc(data, mfham, path) == path
        lines = readlines(path)
        # .def 族(DESIGN §3.3.1 (a)): 5 行ヘッダ、`#` は書かない
        @test length(lines) == 5 + 1 * 2
        @test lines[2] == "NPmfOcc 2"
        @test !any(startswith(strip(l), "#") for l in lines)
        # データ行は (flavor, band_index) の辞書順、どちらも 0-based
        @test lines[6:end] == ["0 0", "0 2"]
    end

    # pmfband の occupied 列は「下から NElec 個」ではなく**実際の占有**
    mktempdir() do dir
        MVMCOptimizers.parton_write_mfham(data, mfham, dir)
        band = readlines(joinpath(dir, "zqp_pmfband_opt.dat"))
        rows = [split(l) for l in band if !startswith(l, "#")]
        occupied = [parse(Int, r[4]) for r in rows]
        @test occupied == [1, 0, 1, 0]        # band 0 と 2 が占有
    end
end

@testset "§8-17-10 終端の自己完結性検査(explicit が要るかを決める測定)" begin
    check = MVMCOptimizers.parton_check_occupation_selfcontained
    α1 = ComplexF64[0, 1, 2, 3]
    α2 = ComplexF64[2.5, 1, 2, 3]

    # アウフバウ占有 + 健全な gap → 成立(α* だけで状態が決まる)
    data = onsite_crossing_data()
    ma = build_crossing_mfham(data; occ_mode = 0)
    MVMCOptimizers.parton_update_orbitals!(ma, α1, 2)
    r = check(ma, 2)
    @test r.selfcontained
    @test r.gap_ok && r.aufbau_ok
    @test r.n_deviation == 0

    # 非アウフバウ占有 → 不成立(占有ファイルを添えないと再現できない)
    data_m = onsite_crossing_data()
    mm = build_crossing_mfham(data_m; occ_mode = 1)
    MVMCOptimizers.parton_update_orbitals!(mm, α1, 2)
    MVMCOptimizers.parton_update_orbitals!(mm, α2, 2)
    r = check(mm, 2)
    @test !r.selfcontained
    @test !r.aufbau_ok
    @test r.n_deviation == 1              # band 2 が抜けて band 3 が入った

    # 占有/非占有が数値的に縮退 → gap 条件で不成立
    data_d = onsite_crossing_data()
    md = build_crossing_mfham(data_d; occ_mode = 0)
    MVMCOptimizers.parton_update_orbitals!(md, ComplexF64[0, 1, 1 + 1e-13, 3], 2)
    r = check(md, 2)
    @test !r.gap_ok
    @test !r.selfcontained
end

@testset "§8-17-12 決定性: 同一入力・同一シードで占有履歴がビット一致" begin
    # タイブレークが非決定的だと、同じ入力でも占有が run ごとに変わりうる。
    # (スコア降順のみで並べると sortperm の実装依存になる)
    dir = mktempdir()
    nl, _ = _write_min_parton_input(dir)
    open(joinpath(dir, "modpara.def"), "a") do io
        println(io, "PartonOccMode 1")
    end
    outs = [mktempdir() for _ = 1:2]
    for out in outs
        @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0
    end
    for name in ("zqp_pmfocc_init.dat", "zqp_pmfocc_opt.dat", "zqp_pmfpara_opt.dat",
                 "zvo_out.dat", "zvo_var.dat")
        @test read(joinpath(outs[1], name)) == read(joinpath(outs[2], name))
    end
end

@testset "§8-17-13 高速路との整合: フレーバー対称なら占有も一致" begin
    # PartonFlavorSymFast は「H^(f) が全フレーバーで同一」を前提に f=1 の結果を配る。
    # 占有も同一になるはずで、そうでなければ前提が崩れているので落とす。
    data = dimerized_mf_data()          # フレーバー対称な入力
    data.modpara.parton_occ_mode = 1
    mp = data.modpara
    mfham = MVMCOptimizers.PartonMFHamiltonian(
        mp.nsite, mp.nelec, mp.nflavor, MVMCOptimizers.parton_n_idx(data))
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    @test mfham.flavor_symmetric
    MVMCOptimizers.parton_update_orbitals!(
        mfham, MVMCOptimizers.parton_alpha_from_terms(data), mp.nelec)
    @test all(occ -> occ == mfham.occ[1], mfham.occ)

    # 前提が崩れた状態(f=2 だけ違う占有)を人工的に作ると検出して落ちる
    mfham.occ[2] = [1, 3]
    @test_throws ErrorException MVMCOptimizers.parton_update_orbitals!(
        mfham, MVMCOptimizers.parton_alpha_from_terms(data), mp.nelec)
end

@testset "§8-17-10b 検査結果が runinfo に記録される" begin
    dir = mktempdir()
    nl, _ = _write_min_parton_input(dir)
    out = mktempdir()
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0
    info = read(joinpath(out, "zvo_parton_runinfo.dat"), String)
    # 「α* 単独で状態が決まるか」は run ごとに記録する(explicit の要否を決める材料)
    @test occursin("occ_selfcontained", info)
    @test occursin("occ_gap_ok", info)
    @test occursin("occ_aufbau_ok", info)
    @test occursin("occ_n_deviation", info)
    @test occursin("occ_min_gap", info)
    @test occursin("PartonOccMode", info)
end

@testset "§8-17-11 診断列: n_occ_deviation と principal_angle_max" begin
    dir = mktempdir()
    nl, _ = _write_min_parton_input(dir)
    out = mktempdir()
    @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0
    header = first(l for l in eachline(joinpath(out, "zvo_parton_diag.dat"))
                   if startswith(l, "#"))
    @test occursin("n_occ_deviation", header)
    @test occursin("principal_angle_max", header)
    cols = split(header)[2:end]
    rows = [split(l) for l in eachline(joinpath(out, "zvo_parton_diag.dat"))
            if !startswith(l, "#")]
    @test all(length(r) == length(cols) for r in rows)
    i_dev = findfirst(==("n_occ_deviation"), cols)
    i_ang = findfirst(==("principal_angle_max"), cols)
    # 既定は aufbau なので乗り換えは 0、主角は有限値
    @test all(parse(Int, r[i_dev]) == 0 for r in rows)
    @test all(isfinite(parse(Float64, r[i_ang])) for r in rows)
end

@testset "§8-17-9 ドライバが占有集合を出力し、band 出力と整合する" begin
    for occ_mode in (0, 1)
        dir = mktempdir()
        nl, _ = _write_min_parton_input(dir)
        # PartonOccMode を modpara に足す(既定 0 の run は行そのものを書かない)
        occ_mode == 1 && open(joinpath(dir, "modpara.def"), "a") do io
            println(io, "PartonOccMode 1")
        end
        out = mktempdir()
        @test MVMCOptimizers.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0

        init_path = joinpath(out, "zqp_pmfocc_init.dat")
        opt_path  = joinpath(out, "zqp_pmfocc_opt.dat")
        @test isfile(init_path)
        @test isfile(opt_path)

        # 4 サイト・NElec 2・2 フレーバー → 占有は 4 行
        for p in (init_path, opt_path)
            lines = readlines(p)
            @test lines[2] == "NPmfOcc 4"
            @test length(lines) == 9
            # (flavor, band) 辞書順で、band は昇順
            rows = [parse.(Int, split(l)) for l in lines[6:end]]
            @test issorted(rows)
        end

        # 初期占有は aufbau(確定順序: 乱数 → 上書き → 門番 → 初期占有)
        @test readlines(init_path)[6:end] == ["0 0", "0 1", "1 0", "1 1"]

        # pmfband の occupied 列と整合すること(唯一の正は _pmfocc 側)
        occ_rows = [parse.(Int, split(l)) for l in readlines(opt_path)[6:end]]
        occ_set = Set((r[1], r[2]) for r in occ_rows)
        for l in readlines(joinpath(out, "zqp_pmfband_opt.dat"))
            startswith(l, "#") && continue
            c = split(l)
            key = (parse(Int, c[1]), parse(Int, c[2]))
            @test (parse(Int, c[4]) == 1) == (key in occ_set)
        end
    end
end

@testset "§8-17-7a テンプレート build が modpara の PartonOccMode を写す" begin
    for m in (0, 1)
        data = onsite_crossing_data()
        data.modpara.parton_occ_mode = m
        mp = data.modpara
        mfham = MVMCOptimizers.PartonMFHamiltonian(
            mp.nsite, mp.nelec, mp.nflavor, MVMCOptimizers.parton_n_idx(data))
        MVMCOptimizers.parton_build_mf_templates!(mfham, data)
        @test mfham.occ_mode == m
    end
end

@testset "§8-17-7 契約0′: 非アウフバウ占有でも占有↔非占有の分割が正しい" begin
    # ∂Φ は「占有軌道の非占有方向への漏れ」なので、占有部分空間への射影は
    # 厳密にゼロでなければならない。占有集合を取り違えると(1:Ne を仮定すると)
    # 実際の占有 band への漏れが混ざってこの恒等式が破れる。
    n_site, n_elec = 4, 2
    α1 = ComplexF64[0, 1, 2, 3, 1]        # 末尾はホッピング振幅
    α2 = ComplexF64[2.5, 1, 2, 3, 1]

    data = crossing_hop_data(; n_site = n_site, n_elec = n_elec)
    mfham = build_crossing_mfham(data; occ_mode = 1)
    MVMCOptimizers.parton_update_orbitals!(mfham, α1, n_elec)
    MVMCOptimizers.parton_update_orbitals!(mfham, α2, n_elec)
    @test mfham.occ[1] == [1, 3]           # 非アウフバウ占有になっている

    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, n_elec)
    Φ = mfham.orbitals[1]
    leak = maximum(norm(Φ' * dΦ) for dΦ in mfham.dorbitals[1])
    @test leak < 1e-12

    # ∂Φ は非占有部分空間に完全に収まる(占有 + 非占有で全空間なので、
    # 非占有への射影で元に戻れば占有成分がないことの裏取りになる)
    unocc = setdiff(1:n_site, mfham.occ[1])
    Uu = mfham.eig_vecs[1][:, unocc]
    for dΦ in mfham.dorbitals[1]
        @test norm(Uu * (Uu' * dΦ) - dΦ) < 1e-12
    end

    # aufbau 経路は従来どおり(回帰): 占有 1:Ne でも同じ恒等式が成り立つ
    data_a = crossing_hop_data(; n_site = n_site, n_elec = n_elec)
    ma = build_crossing_mfham(data_a; occ_mode = 0)
    MVMCOptimizers.parton_update_orbitals!(ma, α2, n_elec)
    MVMCOptimizers.parton_update_orbital_derivatives!(ma, n_elec)
    @test ma.occ[1] == [1, 2]
    @test maximum(norm(ma.orbitals[1]' * dΦ) for dΦ in ma.dorbitals[1]) < 1e-12
end
