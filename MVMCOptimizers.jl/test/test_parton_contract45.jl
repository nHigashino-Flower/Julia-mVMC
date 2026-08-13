"""
§8 テスト 4: 契約 0′/5 を有限差分で検証、および契約 4(局所エネルギー)
--- parton-mode (fork addition) ---

DESIGN_parton.md §8-4 に対応。ForwardDiff を撤回した(LAPACK zheev が Dual を
受けない)ぶん、一次摂動論で作った ∂Φ の正しさは有限差分が担保する。
複素位相つき t は必須で、転置積と随伴の取り違えは実数だけでは見えない。

固有ベクトルの位相ゲージについて: eigen が返す位相は α ごとに任意なので、
素朴に ln ip(α+δ) を取ると位相の跳びが混ざる。摂動論は占有↔占有の混合を
落とす(= ⟨φ_n|∂φ_n⟩ = 0 の平行移動ゲージ)ので、有限差分側も基準軌道へ
位相を合わせてから比較する。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"""
    _orbitals_at(data, α, n_idx; ref=nothing) -> Vector{Matrix{ComplexF64}}

α における占有軌道。`ref` を渡すと各列の位相を基準軌道へ合わせる
(平行移動ゲージ = 摂動論が採る規約)。
"""
function _orbitals_at(data, α::Vector{ComplexF64}, n_idx::Int; ref = nothing)
    mp = data.modpara
    mfham = MVMCOptimizers.PartonMFHamiltonian(mp.nsite, mp.nelec, mp.nflavor, n_idx)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(mfham, α, mp.nelec)
    orbs = [copy(Φ) for Φ in mfham.orbitals]
    if ref !== nothing
        for f in eachindex(orbs), n = 1:mp.nelec
            ov = dot(@view(ref[f][:, n]), @view(orbs[f][:, n]))
            abs(ov) > 1e-10 && (@views orbs[f][:, n] .*= conj(ov) / abs(ov))
        end
    end
    return orbs
end

"占有軌道と配置サイトから ip(恒等 QP)を直接組む。"
function _ip_at(orbs, sites::Vector{Int})
    n_elec = length(sites)
    p = one(ComplexF64)
    for Φ in orbs
        p *= det([Φ[r, n] for r in sites, n = 1:n_elec])
    end
    return p
end

"""
    _fd_ln_ip(data, α0, k, part, δ, n_idx, sites, ref) -> ComplexF64

d(ln ip)/dθ の中心差分。`log(ip₊) - log(ip₋)` ではなく `log(ip₊ / ip₋)` を取る:
ip が負の実軸近傍にあると前者は arg が +π と −π に分かれて 2πi の跳びが出る。
比なら 1 の近傍なので分枝が一意に決まる。
"""
function _fd_ln_ip(data, α0, k::Int, part::Int, δ::Float64, n_idx::Int, sites, ref)
    d = part == 1 ? ComplexF64(δ) : ComplexF64(0, δ)
    αp = copy(α0)
    αm = copy(α0)
    αp[k] += d
    αm[k] -= d
    ip_p = _ip_at(_orbitals_at(data, αp, n_idx; ref = ref), sites)
    ip_m = _ip_at(_orbitals_at(data, αm, n_idx; ref = ref), sites)
    return log(ip_p / ip_m) / (2δ)
end

@testset "契約0′/5: 有限差分 vs 両スロット(Re/Im 独立)" begin
    data = toy_mf_data()
    qp_weight = set_identity_qp!(data)
    n_idx = 2
    sites = [1, 3]

    mfham = build_toy_mfham(data)
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, data.modpara.nelec)
    cfg = toy_config(data, sites)
    amp = MVMCOptimizers.PartonAmplitudeData(1, 2, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    ip = MVMCOptimizers.parton_calculate_ip(amp, qp_weight)

    n_para = n_idx                      # n_proj = 0
    sr_opt_o = zeros(ComplexF64, 2 * (1 + n_para))
    MVMCOptimizers.parton_calculate_o!(
        sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, 0)

    α0 = MVMCOptimizers.parton_alpha_from_terms(data)
    ref = _orbitals_at(data, α0, n_idx)
    δ = 1e-6
    for k = 1:n_idx, part = 1:2
        mfham.is_onsite_group[k] && part == 2 && continue   # 凍結成分は FD 対象外
        fd = _fd_ln_ip(data, α0, k, part, δ, n_idx, sites, ref)
        slot = part == 1 ? MVMCOptimizers.o_slot_re(k) : MVMCOptimizers.o_slot_im(k)
        @test isapprox(sr_opt_o[slot], fd; rtol = 1e-4, atol = 1e-7)
    end

    # ホッピング群の Re/Im スロットは独立な値(val * im の近道ではない)
    @test sr_opt_o[MVMCOptimizers.o_slot_im(1)] !=
          sr_opt_o[MVMCOptimizers.o_slot_re(1)] * im

    # オンサイト群の Im スロットは厳密にゼロ(門番が凍結する成分)
    k_onsite = findfirst(mfham.is_onsite_group)
    @test sr_opt_o[MVMCOptimizers.o_slot_im(k_onsite)] == 0

    # 先頭 2 スロットは触っていない
    @test sr_opt_o[1] == 0 && sr_opt_o[2] == 0
end

@testset "契約0′/5: 有限差分(6 サイト・F=3・別配置)" begin
    n_site, n_elec, n_flavor = 6, 2, 3
    data = toy_mf_data(;
        n_site = n_site,
        n_elec = n_elec,
        n_flavor = n_flavor,
        t = ComplexF64(-1.0, 0.35),
        mu = 0.45,
    )
    qp_weight = set_identity_qp!(data)
    n_idx = 2
    sites = [2, 5]

    mfham = build_toy_mfham(data)
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, n_elec)
    cfg = toy_config(data, sites)
    amp = MVMCOptimizers.PartonAmplitudeData(1, n_flavor, n_elec)
    ws = MVMCOptimizers.PartonSamplingWorkspace(n_elec, n_flavor)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    ip = MVMCOptimizers.parton_calculate_ip(amp, qp_weight)

    sr_opt_o = zeros(ComplexF64, 2 * (1 + n_idx))
    MVMCOptimizers.parton_calculate_o!(
        sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, 0)

    α0 = MVMCOptimizers.parton_alpha_from_terms(data)
    ref = _orbitals_at(data, α0, n_idx)
    δ = 1e-6
    for k = 1:n_idx, part = 1:2
        mfham.is_onsite_group[k] && part == 2 && continue
        fd = _fd_ln_ip(data, α0, k, part, δ, n_idx, sites, ref)
        slot = part == 1 ? MVMCOptimizers.o_slot_re(k) : MVMCOptimizers.o_slot_im(k)
        @test isapprox(sr_opt_o[slot], fd; rtol = 1e-4, atol = 1e-7)
    end
end

@testset "契約5: O のスロットが射影ブロックの後ろに来る" begin
    data = toy_mf_data()
    qp_weight = set_identity_qp!(data)
    mfham = build_toy_mfham(data)
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, 2)
    cfg = toy_config(data, [1, 3])
    amp = MVMCOptimizers.PartonAmplitudeData(1, 2, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    ip = MVMCOptimizers.parton_calculate_ip(amp, qp_weight)

    n_proj = 3
    sr_opt_o = zeros(ComplexF64, 2 * (1 + n_proj + 2))
    MVMCOptimizers.parton_calculate_o!(
        sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, n_proj)

    @test all(iszero, sr_opt_o[1:(2 * n_proj + 2)])       # 射影ブロックは触らない
    @test sr_opt_o[MVMCOptimizers.o_slot_re(n_proj + 1)] != 0
    @test sr_opt_o[MVMCOptimizers.o_slot_im(n_proj + 1)] != 0
    @test sr_opt_o[MVMCOptimizers.o_slot_im(n_proj + 2)] == 0   # オンサイト Im
end

@testset "契約4: 局所エネルギーを素朴な実装と突き合わせる" begin
    data = toy_mf_data()
    qp_weight = set_identity_qp!(data)
    tp = ComplexF64(-1.0, 0.15)
    for i = 0:3
        push!(
            data.physhop_terms,
            MVMCExpertModeParsers.PhysHopTerm(i, mod(i + 1, 4), tp, true),
        )
    end
    push!(
        data.coulomb_inter_terms,
        MVMCExpertModeParsers.CoulombInterTerm(0, 1, 0.5),   # V n_0 n_1
        MVMCExpertModeParsers.CoulombInterTerm(2, 2, -1.3),  # 対角 = 化学ポテンシャル
    )

    mfham = build_toy_mfham(data)
    sites = [1, 3]
    pstate = MVMCOptimizers.parton_build_optimization_state(data; mfham = mfham)
    cfg = pstate.config
    for f = 1:2, (m, r) in enumerate(sites)
        MVMCOptimizers.place_particle!(cfg, f, m, r)
    end
    amp, ws = pstate.amp, pstate.workspace
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    ip = MVMCOptimizers.parton_calculate_ip(amp, qp_weight)

    e = MVMCOptimizers.parton_local_energy(pstate, data, ip)

    # 独立実装: 契約 2 を使わず、移動後の ip を軌道から直接組んで比を取る
    orbs = mfham.orbitals
    ip_of(ss) = prod(det([Φ[r, n] for r in ss, n = 1:2]) for Φ in orbs)
    occ = falses(4)
    for r in sites
        occ[r] = true
    end
    e_naive = ComplexF64(0)
    e_naive += 0.5 * (occ[1] ? 1 : 0) * (occ[2] ? 1 : 0)     # V n_0 n_1(1-based で 1,2)
    e_naive += -1.3 * (occ[3] ? 1 : 0) * (occ[3] ? 1 : 0)    # 対角(1-based で 3)
    for i = 1:4
        j = mod1(i + 1, 4)
        if occ[j] && !occ[i]
            moved = sort(replace(copy(sites), j => i))
            e_naive += tp * ip_of(moved) / ip
        end
        if occ[i] && !occ[j]
            moved = sort(replace(copy(sites), i => j))
            e_naive += conj(tp) * ip_of(moved) / ip
        end
    end
    @test isapprox(e, e_naive; rtol = 1e-10, atol = 1e-13)
    @test abs(e) > 0.1   # フィクスチャが偶然の相殺で 0 になっていないこと
end

@testset "契約4: 対角部だけの系" begin
    data = toy_mf_data()
    set_identity_qp!(data)
    push!(data.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(0, 0), false))
    push!(
        data.coulomb_inter_terms,
        MVMCExpertModeParsers.CoulombInterTerm(0, 2, 2.0),   # 1-based で (1, 3)
    )
    mfham = build_toy_mfham(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data; mfham = mfham)
    for f = 1:2, (m, r) in enumerate([1, 3])
        MVMCOptimizers.place_particle!(pstate.config, f, m, r)
    end
    @test MVMCOptimizers.parton_diag_energy(pstate.physham, pstate.config) ≈ 2.0
end

@testset "契約4/5: parton_main_cal! がサンプルを回して蓄積する" begin
    data = toy_mf_data()
    set_identity_qp!(data)
    mp = data.modpara
    mp.nvmc_warmup = 5
    mp.nvmc_interval = 1
    mp.nvmc_sample = 12
    mp.nblock_update_size = 4
    mp.nstore_o = 1
    push!(data.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(-1, 0.1), true))
    push!(data.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(1, 2, ComplexF64(-1, 0.1), true))
    push!(data.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(2, 3, ComplexF64(-1, 0.1), true))

    mfham = build_toy_mfham(data)
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, mp.nelec)
    pstate = MVMCOptimizers.parton_build_optimization_state(data; mfham = mfham)
    MVMCOptimizers.parton_make_sample!(pstate, data, MersenneTwister(2024))
    MVMCOptimizers.parton_main_cal!(pstate, data)

    energy = pstate.state.energy
    @test real(energy.wc) ≈ mp.nvmc_sample
    @test isfinite(real(energy.etot)) && isfinite(imag(energy.etot))
    @test real(energy.etot2) > 0
    # <H> はエルミートなので実数(統計誤差の範囲で虚部は消える…わけではないが、
    # 個々の E_loc は複素でも和は有限)
    @test isfinite(abs(energy.etot / energy.wc))

    sr = pstate.state.sr_opt
    @test any(!iszero, sr.sr_opt_ho)
    @test any(!iszero, sr.sr_opt_o_store)
end
