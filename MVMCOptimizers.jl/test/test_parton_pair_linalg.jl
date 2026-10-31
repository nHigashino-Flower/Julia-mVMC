"""
T1: Pfaffian パートンモードの線形代数(DESIGN_parton_pf.md §1, §9)
--- parton-mode (fork addition) ---

- T1-a 包含定理: f = ΞJΞᵀ なら Pf F_qp(x) = s_{N,K}·Pf(J)·Π_a det A^(a)_qp(x)。
  比例定数が配置にも qp にもよらないことを、既存の det モード(Φ 直接変分の器)の振幅と比べて確かめる
- T1-c 比と更新: 契約 2 の比・契約 3 の G 更新が厳密再計算と一致する。契約 2 は純粋。
  粒子の交換で (−1)^K 倍。Pf(C⁻¹) = (−1)^{K(K+1)/2}
- T1-d 契約 1 vs 3: 複素 f・QP ありで多数回の高速更新の後に厳密再計算と一致する
- テンプレートの検査(対角・重複・範囲・idx の連番・符号)
"""

using Test
using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

const MO = MVMCOptimizers

"det モード(Φ を直接持つ器)の ip。T1-a の比較相手。"
function _det_mode_ip(data, Φ, cfg, qp_weight)
    mp = data.modpara
    n_qp = length(data.qp_trans)
    phi = MO.PartonDirectPhi(mp.nsite, mp.nelec, mp.nflavor)
    for a = 1:mp.nflavor
        phi.orbitals[a] .= Φ[a]
    end
    amp = MO.PartonAmplitudeData(n_qp, mp.nflavor, mp.nelec)
    ws = MO.PartonSamplingWorkspace(mp.nelec, n_qp * mp.nflavor)
    MO.parton_recompute_amplitude_all!(amp, phi, cfg, data, ws)
    return MO.parton_calculate_ip(amp, qp_weight)
end

"Pf モードの ip(契約 1 で錨を打ってから)。"
function _pair_mode_ip(data, src, cfg, qp_weight)
    amp, ws = pair_amp_ws(data)
    MO.parton_recompute_amplitude_all!(amp, src, cfg, data, ws)
    return MO.parton_calculate_ip(amp, qp_weight), amp, ws
end

@testset "T1-a 包含定理(det モードの振幅との比が一定)" begin
    rng = MersenneTwister(20261031)
    for (K, N, Ns, n_qp) in ((2, 2, 5, 1), (2, 3, 6, 3), (3, 2, 5, 2), (4, 2, 4, 2))
        data = pair_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
        qp_weight = n_qp == 1 ? set_identity_qp!(data) : random_ring_qp!(rng, data, n_qp)
        Φ, J, f = random_xi_j(rng, Ns, N, K)
        terms = full_pair_terms(Ns, K)
        src = build_pair_source(data, terms, pair_params_from_matrix(f, terms, Ns))
        # 規約 F = 2·sgn·p なので pair_matrix は f そのものに戻る
        @test src.pair_matrix ≈ f rtol = 1e-14

        s_nk = (-1)^(binomial(K, 2) * binomial(N, 2))
        expected = s_nk * _ref_pfaffian(J)
        for trial = 1:12
            sites = randperm(rng, Ns)[1:N]     # ラベル順はソートしない
            cfg = toy_config(data, sites)
            ip_pf, _, _ = _pair_mode_ip(data, src, cfg, qp_weight)
            ip_det = _det_mode_ip(data, Φ, cfg, qp_weight)
            @test ip_pf / ip_det ≈ expected rtol = 1e-10
        end
    end
end

@testset "T1-c 粒子の交換で (−1)^K 倍" begin
    rng = MersenneTwister(11)
    for (K, N, Ns) in ((2, 2, 5), (3, 2, 5), (4, 2, 4), (3, 4, 6))
        data = pair_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
        qp_weight = set_identity_qp!(data)
        terms = full_pair_terms(Ns, K)
        src = build_pair_source(data, terms, randn(rng, ComplexF64, length(terms)))
        sites = randperm(rng, Ns)[1:N]
        swapped = copy(sites)
        swapped[1], swapped[2] = swapped[2], swapped[1]
        ip1, _, _ = _pair_mode_ip(data, src, toy_config(data, sites), qp_weight)
        ip2, _, _ = _pair_mode_ip(data, src, toy_config(data, swapped), qp_weight)
        @test ip2 / ip1 ≈ (-1)^K rtol = 1e-11
    end
end

@testset "T1-c Pf(C⁻¹) の符号" begin
    for K = 1:5
        Cinv = [zeros(K, K) -Matrix(I, K, K); Matrix(I, K, K) zeros(K, K)]
        @test MO.parton_pair_pf_cinv(K) == round(Int, real(_ref_pfaffian(Cinv)))
        @test MO.parton_pair_pf_cinv(K) == (-1)^(K * (K + 1) ÷ 2)
    end
end

"移動 (m, r_new) を受理した配置のコピー(ラベルは保つ)。"
function _moved_config(data, cfg, m, r_new)
    sites = [MO.particle_site(cfg, 1, mm) for mm = 1:cfg.n_elec]
    sites[m] = r_new
    return toy_config(data, sites)
end

@testset "T1-c 契約 2 の比・契約 3 の更新 = 厳密再計算" begin
    rng = MersenneTwister(4242)
    for (K, N, Ns, n_qp) in ((2, 2, 5, 1), (2, 3, 6, 3), (3, 2, 5, 2), (4, 2, 4, 2))
        data = pair_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
        qp_weight = n_qp == 1 ? set_identity_qp!(data) : random_ring_qp!(rng, data, n_qp)
        terms = full_pair_terms(Ns, K)
        src = build_pair_source(data, terms, randn(rng, ComplexF64, length(terms)))
        n_checked = 0
        for trial = 1:30
            sites = randperm(rng, Ns)[1:N]
            cfg = toy_config(data, sites)
            ip_old, amp, ws = _pair_mode_ip(data, src, cfg, qp_weight)
            m = rand(rng, 1:N)
            empty = setdiff(1:Ns, sites)
            r_new = rand(rng, empty)

            # 契約 2 は純粋: amp を変えない
            inv_before = copy(amp.inv_f)
            pf_before = copy(amp.pf_f)
            row_before = copy(amp.row_site)
            ratio, ip_new = MO.parton_amplitude_ratio!(ws, amp, src, data, qp_weight, m, r_new)
            @test amp.inv_f == inv_before
            @test amp.pf_f == pf_before
            @test amp.row_site == row_before

            cfg_new = _moved_config(data, cfg, m, r_new)
            ip_ref, amp_ref, _ = _pair_mode_ip(data, src, cfg_new, qp_weight)
            @test ratio ≈ ip_ref / ip_old rtol = 1e-10
            @test ip_new ≈ ip_ref rtol = 1e-10

            # 受理: ① 配置 → ② 振幅(骨格と同じ順序)
            r_old = MO.particle_site(cfg, 1, m)
            MO.parton_update_ele_config!(cfg, m, r_old, r_new)
            st = MO.parton_update_amplitude!(amp, src, data, ws, m, r_new)
            st === :ok || continue
            @test amp.pf_f ≈ amp_ref.pf_f rtol = 1e-10
            @test norm(amp.inv_f - amp_ref.inv_f) <= 1e-10 * norm(amp_ref.inv_f)
            @test amp.row_site == amp_ref.row_site
            @test MO.parton_calculate_ip(amp, qp_weight) ≈ ip_new rtol = 1e-10
            n_checked += 1
        end
        @test n_checked >= 20
    end
end

@testset "T1-d 多数回の高速更新の後 = 厳密再計算(複素 f・QP あり)" begin
    rng = MersenneTwister(777)
    for (K, N, Ns, n_qp) in ((2, 3, 7, 3), (3, 2, 6, 2), (4, 2, 5, 2), (2, 4, 9, 1))
        data = pair_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
        qp_weight = n_qp == 1 ? set_identity_qp!(data) : random_ring_qp!(rng, data, n_qp)
        terms = full_pair_terms(Ns, K)
        src = build_pair_source(data, terms, randn(rng, ComplexF64, length(terms)))
        cfg = toy_config(data, randperm(rng, Ns)[1:N])
        _, amp, ws = _pair_mode_ip(data, src, cfg, qp_weight)
        n_moves = 0
        while n_moves < 200
            m = rand(rng, 1:N)
            r_old = MO.particle_site(cfg, 1, m)
            r_new = rand(rng, 1:Ns)
            (r_new == r_old || MO.is_occupied(cfg, r_new)) && continue
            ratio, _ = MO.parton_amplitude_ratio!(ws, amp, src, data, qp_weight, m, r_new)
            abs(ratio) < 1e-6 && continue      # ノード近傍はこの検証の対象外
            MO.parton_update_ele_config!(cfg, m, r_old, r_new)
            st = MO.parton_update_amplitude!(amp, src, data, ws, m, r_new)
            st === :need_recompute && MO.parton_recompute_amplitude_all!(amp, src, cfg, data, ws)
            n_moves += 1
        end
        ip_fast = MO.parton_calculate_ip(amp, qp_weight)
        pf_fast = copy(amp.pf_f)
        inv_fast = copy(amp.inv_f)
        # 高速更新の後も G は厳密に反対称(射影を外すと対称成分が等比的に増幅されて壊れる)
        for q = 1:n_qp
            G = MO.pair_inv_block(amp, q)
            @test G == -transpose(G)
        end
        MO.parton_recompute_amplitude_all!(amp, src, cfg, data, ws)
        @test MO.parton_calculate_ip(amp, qp_weight) ≈ ip_fast rtol = 1e-8
        @test pf_fast ≈ amp.pf_f rtol = 1e-8
        @test norm(inv_fast - amp.inv_f) <= 1e-8 * norm(amp.inv_f)
        # 錨の冪等性
        pf_once = copy(amp.pf_f)
        MO.parton_recompute_amplitude_all!(amp, src, cfg, data, ws)
        @test amp.pf_f == pf_once
    end
end

@testset "契約 1: G·F = I・特異な配置は Pf = 0" begin
    rng = MersenneTwister(99)
    data = pair_toy_data(n_site = 5, n_elec = 2, n_flavor = 2)
    qp_weight = set_identity_qp!(data)
    terms = full_pair_terms(5, 2)
    src = build_pair_source(data, terms, randn(rng, ComplexF64, length(terms)))
    cfg = toy_config(data, [2, 5])
    _, amp, ws = _pair_mode_ip(data, src, cfg, qp_weight)
    # F を組み直して G·F = I を見る(行 I = (m−1)K + a、x_I = (r_m, a))
    M = 4
    x = [_pair_state(5, r, a) for r in (2, 5) for a = 1:2]
    F = src.pair_matrix[x, x]
    G = reshape(amp.inv_f[1:M * M], M, M)
    @test norm(G * F - I) < 1e-12
    @test amp.pf_f[1] ≈ _ref_pfaffian(F) rtol = 1e-13

    # 全成分 0 の f では全配置が特異: Pf = 0 とし、G は有限のまま(det モードと同じ扱い)
    src0 = build_pair_source(data, terms, zeros(ComplexF64, length(terms)))
    ip0, amp0, ws0 = _pair_mode_ip(data, src0, toy_config(data, [1, 2]), qp_weight)
    @test ip0 == 0
    @test all(isfinite, amp0.inv_f)
    # 特異な qp を抱えたまま受理しようとしたら再計算を求める
    MO.parton_amplitude_ratio!(ws0, amp0, src0, data, qp_weight, 1, 3)
    @test MO.parton_update_amplitude!(amp0, src0, data, ws0, 1, 3) === :need_recompute
end

@testset "交互ブロック構造の正則行列でも Pf を取りこぼさない(契約 1 と 2K×2K の Pf)" begin
    # PfaPack の julia_zsktf2! は、この形の正則行列でゼロピボット(info > 0)を返す
    # (DESIGN_parton_pf.md §3)。同フレーバーだけを組んだ f や、X ≈ C⁻¹ の移動で起きる
    a, b = 0.5 + 0.1im, -0.3 + 0.2im
    X = ComplexF64[0 0 a 0; 0 0 0 b; -a 0 0 0; 0 -b 0 0]
    @test _ref_pfaffian(X) ≈ -a * b
    ws = MO.PartonPairSamplingWorkspace(1, 2, 2)
    @test MO._parton_pair_small_pfaffian!(ws, X) ≈ -a * b rtol = 1e-14
    for K = 1:4
        Cinv = ComplexF64[zeros(K, K) -Matrix(I, K, K); Matrix(I, K, K) zeros(K, K)]
        wsK = MO.PartonPairSamplingWorkspace(1, K, 2)
        @test MO._parton_pair_small_pfaffian!(wsK, Cinv) == MO.parton_pair_pf_cinv(K)
    end

    # 同フレーバーだけを組んだ f(F がフレーバーごとのブロックの交互配置になる)
    rng = MersenneTwister(31)
    K, N, Ns = 2, 2, 5
    data = pair_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
    qp_weight = set_identity_qp!(data)
    terms = [t for t in full_pair_terms(Ns, K) if t.flavor1 == t.flavor2]
    terms = [(site1 = t.site1, flavor1 = t.flavor1, site2 = t.site2, flavor2 = t.flavor2,
              idx = k - 1, sign = 1) for (k, t) in enumerate(terms)]
    src = build_pair_source(data, terms, randn(rng, ComplexF64, length(terms)))
    for sites in ([1, 2], [2, 4], [3, 5])
        ip, amp, ws2 = _pair_mode_ip(data, src, toy_config(data, sites), qp_weight)
        x = [_pair_state(Ns, r, a) for r in sites for a = 1:K]
        F = src.pair_matrix[x, x]
        @test ip ≈ _ref_pfaffian(F) rtol = 1e-12
        @test norm(reshape(amp.inv_f, 4, 4) * F - I) < 1e-12
    end
end

@testset "テンプレートの検査" begin
    nt(s1, f1, s2, f2, idx, sgn) =
        (site1 = s1, flavor1 = f1, site2 = s2, flavor2 = f2, idx = idx, sign = sgn)
    ok = [nt(0, 0, 1, 1, 0, 1), nt(1, 0, 2, 1, 1, -1)]
    src = MO.PartonPairOrbital(3, 2, 2, ok)
    @test src.n_idx == 2
    α, β = _pair_state(3, 1, 1), _pair_state(3, 2, 2)
    @test src.pair_idx[α, β] == 1 && src.pair_idx[β, α] == 1
    @test src.pair_sgn[α, β] == 1 && src.pair_sgn[β, α] == -1
    MO.parton_pair_load!(src, ComplexF64[0.5 + 0.25im, 2.0])
    @test src.pair_matrix[α, β] == 2 * (0.5 + 0.25im)
    @test src.pair_matrix[β, α] == -2 * (0.5 + 0.25im)
    γ, δ = _pair_state(3, 2, 1), _pair_state(3, 3, 2)
    @test src.pair_matrix[γ, δ] == -4.0                 # 符号列 −1 は 2·sgn·p = −4
    @test src.pair_matrix[_pair_state(3, 3, 1), _pair_state(3, 1, 2)] == 0   # def に無い対は 0

    @test_throws ArgumentError MO.PartonPairOrbital(3, 2, 2, [nt(0, 0, 0, 0, 0, 1)])   # 対角
    @test_throws ArgumentError MO.PartonPairOrbital(3, 2, 2,
        [nt(0, 0, 1, 1, 0, 1), nt(1, 1, 0, 0, 1, 1)])                                   # 逆向きの重複
    @test_throws ArgumentError MO.PartonPairOrbital(3, 2, 2, [nt(3, 0, 1, 1, 0, 1)])   # サイト範囲外
    @test_throws ArgumentError MO.PartonPairOrbital(3, 2, 2, [nt(0, 2, 1, 1, 0, 1)])   # フレーバー範囲外
    @test_throws ArgumentError MO.PartonPairOrbital(3, 2, 2,
        [nt(0, 0, 1, 1, 0, 1), nt(1, 0, 2, 1, 2, 1)])                                   # idx が連番でない
    @test_throws ArgumentError MO.PartonPairOrbital(3, 2, 2, [nt(0, 0, 1, 1, 0, 2)])   # 符号が ±1 でない
    @test_throws ArgumentError MO.PartonPairOrbital(3, 2, 2, [nt(0, 0, 1, 1, -1, 1)])  # idx 負
    @test_throws ArgumentError MO.parton_pair_load!(src, ComplexF64[1.0])               # p の長さ違い
end
