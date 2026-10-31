"""
Pfaffian パートンモードの行ブロック SVD 出力と、α 解からの変換道具
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §1.4・§5・§6・§9(T1-b)。

- T1-b: f = ΞJΞᵀ なら行ブロック f_{a,:} の上位 N 本の左特異ベクトルの射影子 = Φ^a の射影子、
  σ_N/σ_{N+1} は桁違いに大きい
- SVD の出力ファイル: `zqp_parton_pair_orb_*.dat` は InPmfOrb と同じ形式(Φ モードの読み手で読める)
- 変換道具 `tools/parton_alpha_to_pair.jl`: α の最終状態と同じ波動関数(全配置の E_loc が一致)の
  Pf 入力を作る。ペアリングは異フレーバー組 / 隣どうし
"""

using Test
using LinearAlgebra
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

const MOT = MVMCOptimizers

"行列 X の列空間への射影子。"
_proj(X) = (Q = Matrix(qr(X).Q)[:, 1:size(X, 2)]; Q * Q')

@testset "T1-b: 行ブロック SVD の射影子 = Φ^a の射影子(K = 2, 3)" begin
    rng = MersenneTwister(2026)
    for (K, N, Ns) in ((2, 2, 5), (3, 2, 5), (2, 3, 7))
        data = pair_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
        Φ, J, f = random_xi_j(rng, Ns, N, K)
        terms = full_pair_terms(Ns, K)
        src = build_pair_source(data, terms, pair_params_from_matrix(f, terms, Ns))
        σs = MOT.parton_pair_flavor_singular_values(src)
        U = MOT.parton_pair_flavor_orbitals(src)
        for a = 1:K
            @test size(U[a]) == (Ns, N)
            @test norm(U[a] * U[a]' - _proj(Φ[a])) < 1e-10
            @test MOT.parton_pair_sigma_ratio(σs[a], N) > 1e10      # 階数はちょうど N
        end
        # 一般の f では比は有限(det^K 多様体から離れている)
        src2 = build_pair_source(data, terms, randn(rng, ComplexF64, length(terms)))
        @test all(σ -> MOT.parton_pair_sigma_ratio(σ, N) < 1e6,
                  MOT.parton_pair_flavor_singular_values(src2))
    end
end

@testset "SVD の出力: 特異値ファイルと、InPmfOrb 形式の軌道ファイル" begin
    rng = MersenneTwister(5)
    K, N, Ns = 2, 2, 5
    data = pair_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
    Φ, J, f = random_xi_j(rng, Ns, N, K)
    terms = full_pair_terms(Ns, K)
    data.parton_pair_params = pair_params_from_matrix(f, terms, Ns)
    src = build_pair_source(data, terms, data.parton_pair_params)
    mktempdir() do dir
        sv_path, orb_path = MOT.parton_pair_write_svd(data, src, dir, "init")
        @test basename(sv_path) == "zqp_parton_pair_sv_init.dat"
        @test basename(orb_path) == "zqp_parton_pair_orb_init.dat"
        rows = [split(l) for l in readlines(sv_path) if !startswith(l, "#") && !isempty(strip(l))]
        @test length(rows) == K * Ns                     # フレーバーごとに min(Ns, K·Ns) = Ns 本
        @test occursin("sigma_ratio_0", read(sv_path, String))
        # Φ モードの読み手でそのまま読め、射影子が Φ^a と一致する
        phi_data = orbparam_toy_data(n_site = Ns, n_elec = N, n_flavor = K)
        write(joinpath(dir, "nl.def"), "InPmfOrb $(basename(orb_path))\n")
        MOT.parton_orbparam_read_in_pmforb!(phi_data, joinpath(dir, "nl.def"))
        phi = MOT.PartonDirectPhi(Ns, N, K)
        MOT.parton_orbparam_load_orbitals!(phi, phi_data)
        for a = 1:K
            @test norm(_proj(phi.orbitals[a]) - _proj(Φ[a])) < 1e-10
        end
    end
end

include(joinpath(@__DIR__, "..", "..", "tools", "parton_alpha_to_pair.jl"))

"全配置の E_loc(1 配置 = 全フレーバー同じサイト)。"
function _pair_all_eloc(pstate, data)
    mp = data.modpara
    qpw = MOT.parton_qp_weight(data)
    out = ComplexF64[]
    for i = 1:mp.nsite, j = (i + 1):mp.nsite
        cfg = toy_config(data, [i, j])
        pstate.config = cfg
        MOT.parton_recompute_amplitude_all!(pstate.amp, pstate.mfham, cfg, data, pstate.workspace)
        ip = MOT.parton_calculate_ip(pstate.amp, qpw)
        push!(out, MOT.parton_local_energy(pstate, data, ip))
    end
    return out
end

@testset "変換道具: α の最終状態と同じ波動関数の Pf 入力を作る(ペアリング $pairing)" for pairing in (:interflavor, :adjacent)
    dir = mktempdir()
    a_in = joinpath(dir, "alpha_in")
    a_out = joinpath(dir, "alpha_out")
    mkpath(a_in)
    nl, _ = _write_min_parton_input(a_in)
    @test MOT.parton_run_para_opt_from_namelist(nl; output_dir = a_out) == 0

    new_dir = joinpath(dir, "pair_in")
    res = parton_alpha_to_pair(a_in, a_out, new_dir; pairing = pairing)
    @test res.residual < 1e-12                       # 全対の def なので射影は厳密
    entries = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(joinpath(new_dir, "namelist.def")))
    keys_new = first.(entries)
    @test "PartonPairOrbital" in keys_new && "InPartonPairOrbital" in keys_new
    @test !any(k -> k in keys_new, ("PartonMFTrans", "PartonMFPara", "InPmfPara", "InPmfOcc"))
    @test occursin("PartonMode 2", read(joinpath(new_dir, "modpara.def"), String))

    # α の最終状態(α* と占有)と、変換した Pf 入力とで全配置の E_loc が一致する
    data_a = MVMCExpertModeParsers.parse_expert_mode_files(nl)
    tmp = joinpath(dir, "_nl.def")
    write(tmp, "InPmfPara $(abspath(joinpath(a_out, "zqp_pmfpara_opt.dat")))\n" *
               "InPmfOcc $(abspath(joinpath(a_out, "zqp_pmfocc_opt.dat")))\n")
    MOT.parton_read_in_pmfpara!(data_a, tmp)
    occ = MOT.parton_read_in_pmfocc(data_a, tmp)
    data_a.modpara.parton_flavor_sym_fast = 0
    MOT.parton_materialize_flags!(data_a)
    ps_a = MOT.parton_build_optimization_state(data_a)
    MOT.parton_update_orbitals!(ps_a.mfham, MOT.parton_alpha_from_terms(data_a),
                                data_a.modpara.nelec; forced_occ = occ)
    MOT.parton_ensure_qp!(data_a)

    data_p = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(new_dir, "namelist.def"))
    MOT.parton_pair_read_in!(data_p, joinpath(new_dir, "namelist.def"))
    MOT.parton_pair_materialize_flags!(data_p)
    MOT.parton_ensure_qp!(data_p)
    ps_p = MOT.parton_pair_build_state(data_p)

    e_a = _pair_all_eloc(ps_a, data_a)
    e_p = _pair_all_eloc(ps_p, data_p)
    @test maximum(abs.(e_p .- e_a)) < 1e-10 * maximum(abs.(e_a))
end

@testset "変換道具: 異フレーバー組のペアリングは同フレーバーを組まない(K = 3)" begin
    J = pair_pairing_matrix(3, 4, :interflavor)
    @test J == -transpose(J)
    @test abs(_ref_pfaffian(J)) == 1
    for c1 = 1:12, c2 = 1:12
        iszero(J[c1, c2]) && continue
        @test cld(c1, 4) != cld(c2, 4)                  # 列 (a−1)N + n のフレーバー a が違う
    end
    Jadj = pair_pairing_matrix(3, 4, :adjacent)
    @test Jadj[1, 2] == 1 && Jadj[2, 1] == -1
    @test_throws ErrorException pair_pairing_matrix(1, 4, :interflavor)
end
