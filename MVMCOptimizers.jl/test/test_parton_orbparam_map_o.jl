"""
制限付き Φ の O・OptFlag・門番のテスト --- parton-mode (fork addition) ---

DESIGN_parton.md §12.12。O_θ = Σ_{k ∈ idx} c_k·O_Φ,k。θ は正則なパラメータなので、
共役前の Re スロットに ∂ln ip/∂θ、Im スロットに i·∂ln ip/∂θ が入る。
"""

using Test
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"写像付きの Φ で振幅を組み、(ip, amp, phi) を返す。"
function _map_amplitude(data, map, cfg)
    mp = data.modpara
    n_qp = MVMCOptimizers.get_n_qp_full(data)
    amp = MVMCOptimizers.PartonAmplitudeData(n_qp, mp.nflavor, mp.nelec; n_stored = mp.nflavor)
    ws = MVMCOptimizers.PartonSamplingWorkspace(mp.nelec, n_qp * mp.nflavor)
    phi = MVMCOptimizers.PartonDirectPhi(mp.nsite, mp.nelec, mp.nflavor, map)
    MVMCOptimizers.parton_orbparam_load_orbitals!(phi, data)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, phi, cfg, data, ws)
    return MVMCOptimizers.parton_calculate_ip(amp, MVMCOptimizers.parton_qp_weight(data)), amp, phi
end

function _map_slots(data, map, cfg)
    ip, amp, phi = _map_amplitude(data, map, cfg)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    o = zeros(ComplexF64, 2 * (1 + MVMCExpertModeParsers.count_variational_parameters(data)))
    MVMCOptimizers.parton_fill_sr_opt_o!(o, amp, phi, cfg, data,
                                         MVMCOptimizers.parton_qp_weight(data), ip, n_proj;
                                         conjugate = false)
    return o
end

function _map_fd(data, map, cfg, j, dir; δ = 1e-6)
    θ0 = copy(data.pmforb_params)
    data.pmforb_params[j] = θ0[j] + dir * δ
    ip_p, _, _ = _map_amplitude(data, map, cfg)
    data.pmforb_params[j] = θ0[j] - dir * δ
    ip_m, _, _ = _map_amplitude(data, map, cfg)
    data.pmforb_params .= θ0
    return log(ip_p / ip_m) / (2δ)
end

@testset "写像付き O = ∂ln ip/∂θ(有限差分、qp2 = $qp2)" for qp2 in (false, true)
    data, map = orbparam_map_fixture(; qp2 = qp2)
    # サイト s と s+3 の行は比例する(ω 倍)ので、両方を含む配置はノード。避けて選ぶ
    for sites in ([1, 2], [2, 3], [1, 6])
        cfg = toy_config(data, sites)
        o = _map_slots(data, map, cfg)
        for j = 1:map.n_idx
            @test isapprox(o[MVMCOptimizers.o_slot_re(j)], _map_fd(data, map, cfg, j, 1);
                           rtol = 1e-6, atol = 1e-9)
            @test isapprox(o[MVMCOptimizers.o_slot_im(j)], _map_fd(data, map, cfg, j, im);
                           rtol = 1e-6, atol = 1e-9)
        end
        @test maximum(abs, o[3:end]) > 1e-3
    end
end

@testset "写像付き Euler の恒等式: Σ_j θ_j O_j = NFlavor·NElec" begin
    data, map = orbparam_map_fixture(; qp2 = true)
    o = _map_slots(data, map, toy_config(data, [1, 2]))
    s = sum(data.pmforb_params[j] * o[MVMCOptimizers.o_slot_re(j)] for j = 1:map.n_idx)
    @test isapprox(s, 2 * 2; atol = 1e-10)
end

@testset "写像付き OptFlag の実体化と門番" begin
    data, map0 = orbparam_map_fixture(; jastrow = true)
    flags_in = fill(true, map0.n_idx)
    flags_in[2] = false
    map = MVMCOptimizers.PartonOrbIndexMap(map0.idx, map0.coeff, flags_in, 6, 2, 2)
    mp = data.modpara
    mp.ncond = -1; mp.nlocspin = 0; mp.vmc_calc_mode = 0; mp.nsrcg = 0
    mp.lanczos_mode = 0; mp.nsp_gauss_leg = 1; mp.nsp_stot = 0; mp.n_orbital_idx = 0
    mp.nneuron = 0; mp.nsplit_size = 1
    flags = MVMCOptimizers.parton_orbparam_materialize_flags!(data, map)
    n_proj = 3
    @test length(flags) == 2 * (n_proj + map.n_idx)
    @test flags[2 * (n_proj + 1) + 1] == false && flags[2 * (n_proj + 1) + 2] == false
    @test flags[2 * n_proj + 1] == true && flags[2 * n_proj + 2] == true
    ctx = MVMCOptimizers.serial_context()
    @test MVMCOptimizers.validate_parton_orbparam_inputs(data, ctx, map) === nothing
    data.pmforb_params = data.pmforb_params[1:(end - 1)]
    @test_throws ErrorException MVMCOptimizers.validate_parton_orbparam_inputs(data, ctx, map)
end
