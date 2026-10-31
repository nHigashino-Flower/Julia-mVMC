"""
Φ 直接変分モードの O(契約 5 の Φ 版)のテスト --- parton-mode (fork addition) ---

DESIGN_parton.md §12.11 / §12.9 の Φ-1。Φ_f[s,n] は正則なパラメータなので、
Re スロットに O = ∂ln ip/∂Φ、Im スロットに i·O が入る(共役シムの前)。
有限差分は ln ip の中心差分で、分岐を避けるため log(ip₊/ip₋) の形で取る。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"data.pmforb_params の Φ で振幅を組み直し、(ip, amp, phi) を返す。"
function _orb_amplitude(data, cfg)
    mp = data.modpara
    n_qp = MVMCOptimizers.get_n_qp_full(data)
    amp = MVMCOptimizers.PartonAmplitudeData(n_qp, mp.nflavor, mp.nelec;
                                             n_stored = mp.nflavor)
    ws = MVMCOptimizers.PartonSamplingWorkspace(mp.nelec, n_qp * mp.nflavor)
    phi = MVMCOptimizers.PartonDirectPhi(mp.nsite, mp.nelec, mp.nflavor)
    MVMCOptimizers.parton_orbparam_load_orbitals!(phi, data)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, phi, cfg, data, ws)
    ip = MVMCOptimizers.parton_calculate_ip(amp, MVMCOptimizers.parton_qp_weight(data))
    return ip, amp, phi
end

"1 配置分の sr_opt_o(スロット並び)を本番と同じ関数で作って返す。"
function _orb_slots(data, cfg; conjugate::Bool)
    ip, amp, phi = _orb_amplitude(data, cfg)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    n_para = MVMCExpertModeParsers.count_variational_parameters(data)
    o = zeros(ComplexF64, 2 * (1 + n_para))
    MVMCOptimizers.parton_fill_sr_opt_o!(
        o, amp, phi, cfg, data, MVMCOptimizers.parton_qp_weight(data), ip, n_proj;
        conjugate = conjugate)
    return o
end

"Φ_k を dir 方向(1 = 実部、im = 虚部)に ±δ 動かしたときの ∂ln ip。"
function _orb_fd_ln_ip(data, cfg, k::Int, dir; δ = 1e-6)
    p0 = copy(data.pmforb_params)
    data.pmforb_params[k] = p0[k] + dir * δ
    ip_p, _, _ = _orb_amplitude(data, cfg)
    data.pmforb_params[k] = p0[k] - dir * δ
    ip_m, _, _ = _orb_amplitude(data, cfg)
    data.pmforb_params .= p0
    return log(ip_p / ip_m) / (2δ)
end

function _orb_check_fd(data, cfg)
    n_orb = length(data.pmforb_params)
    o = _orb_slots(data, cfg; conjugate = false)
    n_bad = 0
    for k = 1:n_orb
        fd_re = _orb_fd_ln_ip(data, cfg, k, 1)
        fd_im = _orb_fd_ln_ip(data, cfg, k, im)
        ok_re = isapprox(o[MVMCOptimizers.o_slot_re(k)], fd_re; rtol = 1e-6, atol = 1e-9)
        ok_im = isapprox(o[MVMCOptimizers.o_slot_im(k)], fd_im; rtol = 1e-6, atol = 1e-9)
        n_bad += !(ok_re && ok_im)
    end
    return n_bad, o
end

@testset "Φ-1: O = ∂ln ip/∂Φ(有限差分、恒等 QP)" begin
    data = orbparam_toy_data()
    for sites in ([1, 4], [2, 3], [5, 6])
        n_bad, o = _orb_check_fd(data, toy_config(data, sites))
        @test n_bad == 0
        @test maximum(abs, o[3:end]) > 1e-3        # 全部ゼロで素通りしないこと
    end
end

@testset "Φ-1: O = ∂ln ip/∂Φ(有限差分、並進 QP 2 本)" begin
    data = orbparam_toy_data()
    set_shift_qp!(data, [collect(1:6), [mod1(r + 1, 6) for r = 1:6]],
                  [ones(Int, 6), ones(Int, 6)], ComplexF64[1, 0.5 + 0.2im])
    @test MVMCOptimizers.get_n_qp_full(data) == 2
    for sites in ([1, 4], [2, 3])
        n_bad, _ = _orb_check_fd(data, toy_config(data, sites))
        @test n_bad == 0
    end
end

@testset "Euler の恒等式: Σ_k Φ_k O_k = NFlavor·NElec" begin
    # 各 det A_qp,f は Φ_f の Ne 次斉次なので Σ_f Σ Φ_f ∂ip/∂Φ_f = F·Ne·ip
    for (F, Ne) in ((2, 2), (3, 2))
        data = orbparam_toy_data(; n_flavor = F, n_elec = Ne)
        set_shift_qp!(data, [collect(1:6), [mod1(r + 1, 6) for r = 1:6]],
                      [ones(Int, 6), ones(Int, 6)], ComplexF64[1, 0.5 + 0.2im])
        cfg = toy_config(data, [2, 5])
        o = _orb_slots(data, cfg; conjugate = false)
        s = sum(data.pmforb_params[k] * o[MVMCOptimizers.o_slot_re(k)]
                for k in eachindex(data.pmforb_params))
        @test isapprox(s, F * Ne; atol = 1e-10)
    end
end

@testset "Im スロット = i × Re スロット(共役前)/ 共役後は両方とも conj" begin
    data = orbparam_toy_data()
    cfg = toy_config(data, [1, 3])
    raw = _orb_slots(data, cfg; conjugate = false)
    shim = _orb_slots(data, cfg; conjugate = true)
    @test raw[1] == 1 && raw[2] == 0             # 既存規約: 先頭 2 スロットは (1, 0)
    for k in eachindex(data.pmforb_params)
        re, imk = MVMCOptimizers.o_slot_re(k), MVMCOptimizers.o_slot_im(k)
        @test raw[imk] == im * raw[re]
        @test shim[re] == conj(raw[re])
        @test shim[imk] == conj(raw[imk])
    end
end

@testset "Jastrow スロットは cnt の実数のまま、Φ は p = n_proj+1 から" begin
    data = orbparam_toy_data()
    _add_ring_jastrow!(data, [0.1, -0.2, 0.05])
    mp = data.modpara
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    @test n_proj == 3
    cfg = MVMCOptimizers.PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, 1;
                                             n_proj = n_proj)
    for f = 1:mp.nflavor, (m, r) in enumerate([1, 3])
        MVMCOptimizers.place_particle!(cfg, f, m, r)
    end
    MVMCOptimizers.parton_make_proj_cnt!(cfg, data)

    o = _orb_slots(data, cfg; conjugate = true)
    for p = 1:n_proj
        @test o[MVMCOptimizers.o_slot_re(p)] == cfg.proj_cnt[p]
        @test o[MVMCOptimizers.o_slot_im(p)] == 0
    end
    # Φ ブロックは射影の後ろ: Euler の恒等式がオフセット n_proj で成り立つ
    s = sum(data.pmforb_params[k] * conj(o[MVMCOptimizers.o_slot_re(n_proj + k)])
            for k in eachindex(data.pmforb_params))
    @test isapprox(s, mp.nflavor * mp.nelec; atol = 1e-10)
end
