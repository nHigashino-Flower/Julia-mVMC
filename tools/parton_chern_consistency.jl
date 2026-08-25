#=
パートン Chern 数の 3 経路整合性検査
--- parton-mode (fork addition) ---

    julia --project=tools tools/parton_chern_consistency.jl <out_dir> \
        --nux 8 [--nuy 8] --ansatz ef4|xexet2 [--grids 4,8,16,24,32,48]

同じ状態の Chern 数を、独立な 3 つの経路で出して突き合わせる。
一致すれば数値の信頼度が上がり、食い違えば「どの前提が崩れているか」が分かる。

| 経路 | 使うもの | k 空間 | 前提 |
|---|---|---|---|
| A. 平滑(細 k) | h(ΔR) → H(k) を任意 k で対角化 | 任意グリッド | H_MF が拡大セル並進で共変 |
| B. native k    | 実占有集合の P を Bloch 射影 p(k)=bPb† | 系サイズの k 点のみ | P が並進不変(‖p²−p‖ で検証) |
| C. Bott index  | 実空間 P と位置位相演算子      | **使わない**   | ギャップが開いている |

A は熱力学極限のバンド構造の Chern(格子の k 点数に縛られない)。
B は有限系が実際に持つ k 点だけを使い、**実際の占有集合**を反映する
(mom 追跡でアウフバウからずれていてもそのまま扱える)。
C は k 空間を一切使わない実空間の指標で、A/B と独立性が高い。

A はグリッドを変えて収束も見る(`--grids`)。粗いグリッドでの FHS は
バンド交差を跨ぐと値が飛ぶので、細かくして安定する値を採る。
=#

include(joinpath(@__DIR__, "parton_bands.jl"))

using .PartonBands
using LinearAlgebra, Printf

function parse_args(argv)
    length(argv) >= 1 || error(
        "usage: julia --project=tools tools/parton_chern_consistency.jl <out_dir> " *
        "--nux N [--nuy N] --ansatz ef4|xexet2 [--grids 4,8,16,24,32,48]")
    dir = argv[1]
    nux = nothing; nuy = nothing; ansatz = nothing
    grids = [4, 8, 16, 24, 32, 48]
    i = 2
    while i <= length(argv)
        a = argv[i]
        if a == "--nux";        nux = parse(Int, argv[i+1]); i += 2
        elseif a == "--nuy";    nuy = parse(Int, argv[i+1]); i += 2
        elseif a == "--ansatz"; ansatz = Symbol(argv[i+1]); i += 2
        elseif a == "--grids";  grids = [parse(Int, s) for s in split(argv[i+1], ",")]; i += 2
        else error("未知の引数: $a")
        end
    end
    nux === nothing && error("--nux は必須です")
    nuy === nothing && (nuy = nux)
    ansatz === nothing && error("--ansatz は必須です")
    EXEY = Dict(:ef4 => (2, 2), :xexet2 => (2, 1))
    haskey(EXEY, ansatz) || error("未対応のアンザッツ: $ansatz")
    ex, ey = EXEY[ansatz]
    return (dir = dir, nux = nux, nuy = nuy, ansatz = ansatz,
            ex = ex, ey = ey, grids = grids)
end

function main(argv)
    o = parse_args(argv)
    band = read_pmfband(joinpath(o.dir, "zqp_pmfband_opt.dat"))
    Hs = read_pmfham(joinpath(o.dir, "zqp_pmfham_opt.dat");
                     nsite = band.nsite, nflavor = band.nflavor)
    occs = read_pmfocc(joinpath(o.dir, "zqp_pmfocc_opt.dat"); nflavor = band.nflavor)
    n1, n2 = div(o.nux, o.ex), div(o.nuy, o.ey)
    nk = n1 * n2
    n_occ = div(band.nelec, nk)

    @printf("=== %s : %s (ex,ey)=(%d,%d) ===\n", o.dir, o.ansatz, o.ex, o.ey)
    @printf("Nsite=%d NElec=%d | native k %d×%d (nk=%d), n_orb=%d, 占有 %d バンド/k\n\n",
            band.nsite, band.nelec, n1, n2, nk, 2*o.ex*o.ey, n_occ)

    verdicts = String[]
    for f in 1:band.nflavor
        println("─"^68)
        println("flavor $(f-1)")
        println("─"^68)
        H = Hs[f]
        # --- 状態の前提を先に検査 ---
        h, resid = fold_hoppings(H, o.nux, o.nuy, o.ex, o.ey)
        e = eigvals(Hermitian(H))
        gap = e[band.nelec + 1] - e[band.nelec]
        is_aufbau = sort(occs[f]) == collect(0:band.nelec-1)
        @printf("前提: 並進共変の残差 %.2e / HOMO-LUMO ギャップ %+.6f / 占有はアウフバウ %s\n",
                resid, gap, is_aufbau ? "yes" : "**no**")
        resid < 1e-8 || @warn "H_MF が拡大セル並進で共変ではありません(経路 A が無効)" resid
        gap > 0 || @warn "ギャップが閉じています(Chern 数は定義できません)" gap

        # --- A: 平滑(細 k グリッド)---
        println("\n[A] 平滑 k グリッド(H(k) を任意 k で対角化、最低 $n_occ バンド多様体)")
        @printf("  %8s %14s %14s\n", "grid", "C", "min|link det|")
        A_vals = Float64[]
        for g in o.grids
            r = fhs_chern_bands(h, collect(1:n_occ); n1 = g, n2 = g)
            push!(A_vals, r.C)
            @printf("  %4d×%-3d %14.6f %14.3e%s\n", g, g, r.C, r.min_abs_link_det,
                    g == maximum(o.grids) ? "   ← 採用" : "")
        end
        C_A = A_vals[end]
        stable = length(A_vals) >= 3 && all(abs(v - C_A) < 1e-6 for v in A_vals[end-2:end])
        @printf("  → 収束: %s(細かい側 3 点が一致 %s)\n",
                stable ? "OK" : "**未収束**", stable ? "yes" : "no")
        # バンド分解(参考)
        norb = 2 * o.ex * o.ey
        gmax = maximum(o.grids)
        bandC = [fhs_chern_bands(h, [b]; n1 = gmax, n2 = gmax).C for b in 1:norb]
        @printf("  バンド分解 C(下から): %s\n",
                join([@sprintf("%+.3f", c) for c in bandC], ", "))
        @printf("  最低 %d バンドの和 = %+.6f(多様体としての C = %+.6f)\n",
                n_occ, sum(bandC[1:n_occ]), C_A)

        # --- B: native k グリッド ---
        println("\n[B] native k グリッド($(n1)×$(n2)、実占有集合の射影行列)")
        rB = fhs_chern_projector(H, occs[f], o.nux, o.nuy, o.ex, o.ey)
        @printf("  C = %+.6f   min|link det| = %.3e   max‖p²−p‖ = %.3e %s\n",
                rB.C, rB.min_abs_link_det, rB.max_projector_error,
                rB.max_projector_error < 0.05 ? "(参照VMC 閾値 0.05 を満たす)" :
                                                "(**閾値 0.05 超 → C_valid = false**)")
        # 同じ native グリッドで経路 A を評価(グリッド差の切り分け用)
        rBa = fhs_chern_bands(h, collect(1:n_occ); n1 = n1, n2 = n2)
        @printf("  参考: 同じ %d×%d で経路 A を評価すると C = %+.6f\n", n1, n2, rBa.C)

        # --- C: Bott index ---
        println("\n[C] Bott index(実空間、k 空間を使わない)")
        rC = bott_index(H, occs[f], o.nux, o.nuy, o.ex, o.ey)
        @printf("  B_vuvu = %+.6f   B_uvuv = %+.6f\n", rC.B_vuvu, rC.B_uvuv)
        @printf("  診断: ‖[Ũ,Ṽ]‖/√dim = %.3e   min σ(Ũ) = %.3e   min σ(Ṽ) = %.3e\n",
                rC.commutator, rC.min_singular_x, rC.min_singular_y)

        # --- 突き合わせ ---
        println("\n[整合性]")
        @printf("  A(平滑 %d×%d) = %+.4f   B(native %d×%d) = %+.4f   C(Bott) = %+.4f\n",
                gmax, gmax, C_A, n1, n2, rB.C, rC.B_vuvu)
        dAB = abs(C_A - rB.C); dAC = abs(C_A - rC.B_vuvu); dBC = abs(rB.C - rC.B_vuvu)
        ok = dAB < 0.05 && dAC < 0.05 && dBC < 0.05
        @printf("  差: |A−B| = %.3e   |A−C| = %.3e   |B−C| = %.3e   → %s\n",
                dAB, dAC, dBC, ok ? "**3 経路一致**" : "**不一致**")
        push!(verdicts, @sprintf("flavor %d: A=%+.3f B=%+.3f C=%+.3f %s",
                                 f-1, C_A, rB.C, rC.B_vuvu, ok ? "一致" : "不一致"))
        println()
    end
    println("="^68)
    println("まとめ")
    for v in verdicts; println("  ", v); end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
