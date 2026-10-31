"""
Pfaffian パートンモードの 1 sweep の時間を測る(DESIGN_parton_pf.md §9 の速度の報告)
--- parton-mode (fork addition) ---

    julia --project=MVMCOptimizers.jl tools/parton_pair_bench.jl

1 sweep = NSite 回の提案(NVMCInterval = 1)。次の 3 つを比べる:

- `pf_fast`: 本番の骨格 `parton_make_sample!`(契約 2 の 2K×2K の Pf と契約 3 の Woodbury 更新)
- `pf_full`: 提案ごとに移動後の配置で契約 1(Pf と LU、O(M³))を回す場合(高速更新が無いときの比較基準)
- `det`: 同じ寸法の det モード(Φ を直接持つ器)の骨格(参考)

f と Φ は複素乱数、QP なし、Jastrow なし。乱数の f は det^K 多様体の外なので受理率は物理系と違う。
"""

using LinearAlgebra
using Printf
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

const MOB = MVMCOptimizers

function _bench_data(n_site, n_elec, n_flavor; n_sample = 200)
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite, mp.nelec, mp.nflavor = n_site, n_elec, n_flavor
    mp.parton_mode = 2
    mp.two_sz = 0
    mp.complex_flag = 1
    mp.nex_update_path = 6
    mp.nvmc_warmup = 20
    mp.nvmc_interval = 1
    mp.nvmc_sample = n_sample
    mp.parton_block_update_size = 16
    data.qp_trans = [collect(1:n_site)]
    data.qp_trans_sgn = [ones(Int, n_site)]
    mp.nsp_gauss_leg = 1
    mp.nsp_stot = 0
    mp.nmp_trans = 1
    data.para_qp_trans = ComplexF64[1]
    MVMCExpertModeParsers.init_qp_weight!(data)
    n_state = n_site * n_flavor
    for α = 1:n_state, β = (α + 1):n_state
        push!(data.parton_pair_terms, MVMCExpertModeParsers.PartonPairOrbitalTerm(
            mod1(α, n_site) - 1, cld(α, n_site) - 1, mod1(β, n_site) - 1, cld(β, n_site) - 1,
            length(data.parton_pair_terms), 1))
    end
    data.parton_pair_params = randn(MersenneTwister(1), ComplexF64, length(data.parton_pair_terms))
    return data
end

"本番の骨格で 1 sweep あたりの秒数(burn からの再開 = n_out = Sample + 1 回の sweep)。"
function _sweep_time(pstate, data; seed = 7)
    rng = MOB.SFMT19937RNG()
    Random.seed!(rng, seed)
    MOB.parton_make_sample!(pstate, data, rng)            # 初回(コンパイルと warm-up)
    n_out = MOB.parton_n_out(pstate.config, data.modpara)
    t = @elapsed MOB.parton_make_sample!(pstate, data, rng)
    acc = pstate.config.counter[2] / pstate.config.counter[1]
    return t / n_out, acc
end

"提案ごとに契約 1 を回した場合の 1 sweep あたりの秒数(NSite 回の全再計算で見積もる)。"
function _full_time(pstate, data; n_rep = 200)
    cfg = pstate.config
    amp, ws, src = pstate.amp, pstate.workspace, pstate.mfham
    MOB.parton_recompute_amplitude_all!(amp, src, cfg, data, ws)
    t = @elapsed for _ = 1:n_rep
        MOB.parton_recompute_amplitude_all!(amp, src, cfg, data, ws)
    end
    return t / n_rep * data.modpara.nsite
end

function bench(; cases = ((16, 8, 2, "CB 4x4 nu=1/2 相当"), (36, 12, 3, "KM 9x4 nu=1/3 相当")))
    BLAS.set_num_threads(1)
    @printf("%-22s %4s %4s %4s %12s %12s %10s %12s %8s\n",
            "system", "Ns", "N", "K", "pf_fast[ms]", "pf_full[ms]", "full/fast", "det[ms]", "acc_pf")
    for (Ns, N, K, name) in cases
        data = _bench_data(Ns, N, K)
        MOB.parton_pair_materialize_flags!(data)
        ps = MOB.parton_pair_build_state(data)
        t_fast, acc = _sweep_time(ps, data)
        t_full = _full_time(ps, data)
        # det モード(同じ寸法・乱数の Φ)
        data.pmforb_params = randn(MersenneTwister(2), ComplexF64, K * Ns * N)
        ps_det = MOB.parton_orbparam_build_state(data)
        t_det, _ = _sweep_time(ps_det, data)
        @printf("%-22s %4d %4d %4d %12.3f %12.3f %10.1f %12.3f %8.3f\n",
                name, Ns, N, K, 1e3 * t_fast, 1e3 * t_full, t_full / t_fast, 1e3 * t_det, acc)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    bench()
end
