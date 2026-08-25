"""
P2: 運動量射影の QP 構成 — 並進が cb 模型の対称性であることの機械検証
--- parton-mode (fork addition) ---

DESIGN_parton.md §8 P 層。M2 で運動量射影を CB 模型に載せる前提を固める。

## なぜ「平均場」ではなく「物理ハミルトニアン」の対称性を見るのか

射影の目的は**平均場が破った対称性を回復する**ことなので、H_MF が並進不変で
ある必要はない(むしろ破れているから射影に意味がある)。変分的に意味を持つ条件は
`T_R` が **H_phys の対称性**であること — そうでない操作で射影すると、変分空間を
対称性でない方向に制限するだけでエネルギーは下がらない。

したがってここで検査するのは:

1. `cb_translations` が巡回群として閉じた全単射の集合であること
2. **cb 模型の 1 体項が並進不変**であること(= 射影が変分的に正当)
3. 拡大セルで縮約した**平均場は基本セル並進で不変でない**こと
   (= 射影が自明でない。ここが潰れていると射影しても状態が変わらない)
4. 生成した `qptransidx.def` が往復し、門番(`validate_parton_qp`)を通ること

## 格子の規約

サイトは倍密グリッド `Lx = 2nx, Ly = 2ny` 上の偶奇の揃った点(§checkerboard_model)。
基本セル並進は `(dx, dy) = (2·ucx, 2·ucy)` — 偶数変位なので偶奇が保たれ、
副格子を混ぜない。参照実装 `CheckerBoard.jl:264` の `build_QNPTransSiteList`
(`dx, dy = 2*ucx, 2*ucy`)と同じ取り方。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "P2-1 並進写像の基本性質(巡回群・全単射)" begin
    nx, ny = 4, 4
    nsite = 2 * nx * ny
    maps, ucs = cb_translations(nx, ny)

    @test length(maps) == nx * ny
    @test length(ucs) == nx * ny
    # 恒等が (0,0) として 1 本目に来る(重み exp(2πi K·0) = 1 と対応)
    @test ucs[1] == (0, 0)
    @test maps[1] == collect(0:(nsite - 1))
    # 各写像が全単射
    for (k, m) in enumerate(maps)
        @test sort(m) == collect(0:(nsite - 1))
    end
    # 群として閉じている: 合成 T_a∘T_b が再び集合の要素
    key = Dict(m => k for (k, m) in enumerate(maps))
    for a = 1:length(maps), b = 1:length(maps)
        composed = [maps[a][maps[b][j + 1] + 1] for j = 0:(nsite - 1)]
        @test haskey(key, composed)
    end
end

@testset "P2-2 cb 模型の 1 体項が基本セル並進で不変(射影の前提)" begin
    nx, ny = 4, 4
    p = CheckerboardParams()
    H = cb_onebody(nx, ny, p)
    maps, _ = cb_translations(nx, ny)

    for (k, m) in enumerate(maps)
        # 行と列をそれぞれ写した H[T(i), T(j)] が元と一致すること
        Ht = [H[m[i] + 1, m[j] + 1] for i = 1:size(H, 1), j = 1:size(H, 2)]
        @test Ht ≈ H atol = 1e-13
    end
end

@testset "P2-3 拡大セルで縮約した平均場は基本セル並進で不変でない" begin
    # 射影が自明でないことの確認。ここが不変だと T_R|φ⟩ = |φ⟩ となり、
    # 射影しても状態が変わらない = エネルギーも下がらない。
    nx, ny, F = 4, 4, 2
    fx = parton_fixture(nx, ny, F, 2, 1)          # 拡大セル (2,1)
    nsite = 2 * nx * ny

    # α を idx ごとに違う値にして H_MF を組む(乱数初期化の実運用に対応)。
    # α = 1 一律だと H_MF ∝ H_phys になって不変になってしまうので、
    # ここで差をつけないとテストが退化する。
    α = [1.0 + 0.37k for k = 0:(fx.n_idx - 1)]
    Hmf = zeros(ComplexF64, nsite, nsite)
    for (t, q) in zip(fx.pmftrans, fx.pmfpara)
        t[1] == q[1] && t[3] == q[3] ||
            error("pmftrans と pmfpara の行が対応していない")
        q[2] == 0 || continue                      # flavor 0 のブロックだけ見る
        Hmf[t[1] + 1, t[3] + 1] += α[q[5] + 1] * t[5]
        t[1] == t[3] || (Hmf[t[3] + 1, t[1] + 1] += conj(α[q[5] + 1] * t[5]))
    end

    maps, ucs = cb_translations(nx, ny)
    broken = 0
    for (k, m) in enumerate(maps)
        Ht = [Hmf[m[i] + 1, m[j] + 1] for i = 1:nsite, j = 1:nsite]
        isapprox(Ht, Hmf; atol = 1e-13) || (broken += 1)
    end
    # 拡大セル (2,1) なので x 方向の奇数セル並進で破れる。全部が不変なら
    # 射影は何もしないので、少なくとも 1 つは破れていなければならない。
    @test broken > 0
end

@testset "P2-4 qptransidx.def が往復し、門番を通る" begin
    nx, ny, F = 4, 4, 2
    nsite = 2 * nx * ny
    kx, ky = 0, 0

    mktempdir() do dir
        nl, fx = write_parton_def_files(dir, nx, ny, F, 2, 1;
            n_elec = 8, nsr_step = 1, nsr_smp = 1, qp_momentum = (kx, ky))
        @test isfile(joinpath(dir, "qptransidx.def"))

        data = MVMCExpertModeParsers.parse_expert_mode_files(nl)
        @test length(data.qp_trans) == nx * ny
        @test data.modpara.nmp_trans == nx * ny
        @test length(data.para_qp_trans) == nx * ny

        # 書いた写像がそのまま読めていること(パーサは 0-based で保持)
        maps, ucs = cb_translations(nx, ny)
        @test data.qp_trans == maps
        # 重みは exp(2πi(Kx·ucx + Ky·ucy))
        for (k, (ucx, ucy)) in enumerate(ucs)
            @test data.para_qp_trans[k] ≈ cis(2π * (kx * ucx / nx + ky * ucy / ny)) atol =
                1e-12
        end

        # 門番(NMPTrans と qptransidx.def の整合)を通る
        MVMCOptimizers.parton_materialize_flags!(data)
        @test MVMCOptimizers.validate_parton_inputs(data, MVMCOptimizers.serial_context()) ===
              nothing
    end
end

"サイト集合の全列挙(1-based、昇順)。"
function _all_configs(nsite::Int, ne::Int)
    out, cur = Vector{Int}[], Int[]
    function rec(start)
        length(cur) == ne && (push!(out, copy(cur)); return)
        for r = start:nsite
            push!(cur, r)
            rec(r + 1)
            pop!(cur)
        end
    end
    rec(1)
    return out
end

"CB 模型の全並進 QP を載せ、重み exp(2πi(nkx·ucx/nx + nky·ucy/ny)) を張る。"
function _set_cb_qp!(data, nx::Int, ny::Int, nkx::Int, nky::Int)
    maps, ucs = cb_translations(nx, ny)
    data.qp_trans = [m .+ 1 for m in maps]        # parton_ensure_qp! と同じ 1-based
    data.qp_trans_sgn = [ones(Int, length(m)) for m in maps]
    data.modpara.nsp_gauss_leg = 1
    data.modpara.nsp_stot = 0
    data.modpara.nmp_trans = length(maps)
    data.para_qp_trans =
        ComplexF64[cis(2π * (nkx * ucx / nx + nky * ucy / ny)) for (ucx, ucy) in ucs]
    MVMCExpertModeParsers.init_qp_weight!(data)
    return data
end

"射影なし(恒等 1 本)に戻す。"
function _set_identity_qp!(data, nsite::Int)
    data.qp_trans = [collect(1:nsite)]
    data.qp_trans_sgn = [ones(Int, nsite)]
    data.modpara.nmp_trans = 1
    data.para_qp_trans = ComplexF64[1]
    MVMCExpertModeParsers.init_qp_weight!(data)
    return data
end

"全数展開(サンプリングなし)で (E_var, ⟨ψ|ψ⟩) を返す。"
function _exact_energy_and_norm(data, configs)
    mp = data.modpara
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    MVMCOptimizers.parton_update_orbitals!(
        pstate.mfham, MVMCOptimizers.parton_alpha_from_terms(data), mp.nelec)
    qpw = MVMCOptimizers.parton_qp_weight(data)
    num, den = ComplexF64(0), 0.0
    for c in configs
        cfg = MVMCOptimizers.PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, 1)
        for f = 1:mp.nflavor, (m, r) in enumerate(c)
            MVMCOptimizers.place_particle!(cfg, f, m, r)
        end
        pstate.config = cfg
        MVMCOptimizers.parton_recompute_amplitude_all!(
            pstate.amp, pstate.mfham, cfg, data, pstate.workspace)
        ip = MVMCOptimizers.parton_calculate_ip(pstate.amp, qpw)
        abs(ip) < 1e-30 && continue
        num += abs2(ip) * MVMCOptimizers.parton_local_energy(pstate, data, ip)
        den += abs2(ip)
    end
    return num / den, den
end

@testset "P2-5 射影の変分的性質(全数展開・厳密)" begin
    # 運動量射影 P_k は H_phys が並進不変なら射影演算子で、Σ_k P_k = 1。したがって
    #     E_noproj = Σ_k w_k E_k   (w_k = ⟨φ|P_k|φ⟩/⟨φ|φ⟩ ≥ 0、Σ_k w_k = 1)
    # という**凸結合**になり、min_k E_k ≤ E_noproj が厳密に従う(= 射影は変分的に
    # 必ず得をする)。等式・不等式ともサンプリング誤差なしで機械精度で立つので、
    # 写像の向き・重みの位相・n_qp の配線のどれか 1 つでも狂うと破れる。
    #
    # 充填率は射影の数学的性質と無関係なので、全数展開が軽くなる NElec = 2 で見る。
    nx, ny, F, ne = 4, 4, 2, 2
    nsite = 2 * nx * ny
    configs = _all_configs(nsite, ne)
    @test length(configs) == binomial(nsite, ne)

    mktempdir() do dir
        nl, fx = write_parton_def_files(dir, nx, ny, F, 2, 1;
            n_elec = ne, nsr_step = 1, nsr_smp = 1)
        data = MVMCExpertModeParsers.parse_expert_mode_files(nl)
        # α を idx ごとに違えて平均場の並進対称性を破る。一律 α だと H_MF ∝ H_phys
        # となり |φ⟩ 自身が並進固有状態になって射影が自明化する(P2-3 と同じ理由)。
        for t in data.pmfpara_terms
            t.value = ComplexF64(1.0 + 0.13 * t.idx, 0.07 * t.idx)
        end
        MVMCOptimizers.parton_materialize_flags!(data)

        _set_identity_qp!(data, nsite)
        e_ref, norm_ref = _exact_energy_and_norm(data, configs)
        @test abs(imag(e_ref)) < 1e-9

        n_trans = nx * ny
        es, ws = ComplexF64[], Float64[]
        for nky = 0:(ny - 1), nkx = 0:(nx - 1)
            _set_cb_qp!(data, nx, ny, nkx, nky)
            e, nrm = _exact_energy_and_norm(data, configs)
            push!(es, e)
            # 実装の重みは規格化されていない(P_impl = n_trans · P_k)ので、
            # ⟨ψ_k|ψ_k⟩ = n_trans² ⟨φ|P_k|φ⟩ の分を割り戻す
            push!(ws, nrm / (n_trans^2 * norm_ref))
        end

        @test all(w > -1e-12 for w in ws)            # 重みは非負
        @test sum(ws) ≈ 1 rtol = 1e-9                # 完全性 Σ_k P_k = 1
        # 重みが実質ゼロのセクターは e_k = 0/0 で無意味なので、e の検査は
        # w > 0 のセクターに限る。fixture の向き正準化(2026-08-18)で H_MF が
        # 拡大セル並進を厳密に保つようになり、|φ⟩ が保存並進の固有状態になった
        # ため、非整合な k の重みが機械精度で 0 になる(修正前は向きの破れが
        # 全セクターに漏れていたので空セクターが無く、この場合分けが不要だった)
        live = [k for k in eachindex(ws) if ws[k] > 1e-10]
        @test 2 <= length(live)                      # 凸結合の検出力に 2 セクター以上要る
        @test all(abs(imag(es[k])) < 1e-8 for k in live)   # 生きたセクターの e は実
        @test sum(ws[k] * es[k] for k in live) ≈ e_ref rtol = 1e-8       # 凸結合
        @test minimum(real(es[k]) for k in live) <= real(e_ref) + 1e-10  # 変分的に得
        # 等号しか出ないなら射影が自明ということ(テストの検出力の確認)
        @test minimum(real(es[k]) for k in live) < real(e_ref) - 1e-6
    end
end

@testset "P2-6 参照実装準拠の QP 構成(make_QNPidx と同じ本数・同じ並進)" begin
    # 参照実装 `make_QNPidx` をそのまま実行して得た対応:
    #   Nux=Nuy=4 (Nsite=32): K(puc)=1 → 1 本、K=2 → 2 本 [(0,0),(1,0)]、
    #                          K=4 → 4 本 [(0,0),(1,0),(2,0),(3,0)]
    # 並進は **x 方向のみ**。y 方向はアンザッツが破っていないので入らない。
    nx, ny = 4, 4
    nsite = 2 * nx * ny
    full, fucs = cb_translations(nx, ny)

    for kext in (1, 2, 4)
        maps, ucs = cb_qp_translations(nx, ny, kext)
        @test length(maps) == kext                          # 本数 = K
        @test ucs == [(u, 0) for u = 0:(kext - 1)]          # x 方向のみ
        for m in maps
            @test sort(m) == collect(0:(nsite - 1))         # 全単射
        end
        # 中身が全並進側の対応する要素と一致すること
        for (k, uc) in enumerate(ucs)
            j = findfirst(==(uc), fucs)
            @test maps[k] == full[j]
        end
    end

    # nx が kext で割り切れないときは落とす(剰余類が定義できない)
    @test_throws Exception cb_qp_translations(nx, ny, 3)
end

@testset "P2-7 参照準拠 QP でも凸結合と変分下界が成り立つ" begin
    # P2-5 と同じ厳密関係を、実運用の QP 構成(x 方向 kext 本)で確かめる。
    # kext 本の巡回群 Z_kext 上でも P_k は射影演算子なので、
    #   E_noproj = Σ_k w_k E_k、min_k E_k ≤ E_noproj
    # が成り立つ。ここが崩れたら QP 構成か重みの位相が壊れている。
    nx, ny, F, ne, kext = 4, 4, 2, 2, 2
    nsite = 2 * nx * ny
    configs = _all_configs(nsite, ne)

    mktempdir() do dir
        nl, fx = write_parton_def_files(dir, nx, ny, F, kext, 1;
            n_elec = ne, nsr_step = 1, nsr_smp = 1)
        data = MVMCExpertModeParsers.parse_expert_mode_files(nl)
        for t in data.pmfpara_terms
            t.value = ComplexF64(1.0 + 0.13 * t.idx, 0.07 * t.idx)
        end
        MVMCOptimizers.parton_materialize_flags!(data)

        _set_identity_qp!(data, nsite)
        e_ref, norm_ref = _exact_energy_and_norm(data, configs)

        maps, ucs = cb_qp_translations(nx, ny, kext)
        es, ws = ComplexF64[], Float64[]
        for nkx = 0:(kext - 1)
            data.qp_trans = [m .+ 1 for m in maps]
            data.qp_trans_sgn = [ones(Int, nsite) for _ in maps]
            data.modpara.nmp_trans = kext
            data.para_qp_trans = ComplexF64[cis(2π * nkx * ucx / kext) for (ucx, _) in ucs]
            MVMCExpertModeParsers.init_qp_weight!(data)
            e, nrm = _exact_energy_and_norm(data, configs)
            push!(es, e)
            push!(ws, nrm / (kext^2 * norm_ref))
        end

        @test all(abs(imag(e)) < 1e-8 for e in es)
        @test sum(ws) ≈ 1 rtol = 1e-9
        @test sum(w * e for (w, e) in zip(ws, es)) ≈ e_ref rtol = 1e-8
        @test minimum(real, es) <= real(e_ref) + 1e-10
        @test minimum(real, es) < real(e_ref) - 1e-6
    end
end
