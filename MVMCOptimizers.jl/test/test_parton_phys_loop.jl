"""
DESIGN_parton.md §13.4 / §13.5-3: 測定ループと出力
--- parton-mode (fork addition) ---

`parton_phys_test_helpers.jl` の `phys_toy_cases` を使う。モンテカルロでも厳密に成り立つ量
(Σ⟨n_i⟩ = N と、同じサンプルで組んだ E の恒等式)で検査する。
"""

using Test
using Random
using MVMCExpertModeParsers
using MVMCOptimizers

"phys_toy_cases から label のケースを取り出す(毎回作り直すので状態は汚れていない)。"
_phys_case(label) = only(filter(c -> c.label == label, phys_toy_cases()))

"平均済みの Green から E を組む(DESIGN §13.5-2 の式)。"
function _phys_energy_from_green(pstate, data, g1::AbstractVector, g2::AbstractVector)
    one_pos = Dict((t.site1 + 1, t.site2 + 1) => n for (n, t) in enumerate(data.green_one_terms))
    two_pos = Dict((t.site1 + 1, t.site2 + 1, t.site3 + 1, t.site4 + 1) => n
                   for (n, t) in enumerate(data.green_two_terms))
    e = zero(ComplexF64)
    for h in pstate.physham.hops
        i, j, t = h.site1, h.site2, h.value
        e += t * g1[one_pos[(j, i)]] + conj(t) * g1[one_pos[(i, j)]]
    end
    for d in pstate.physham.diags
        e += d.value * g2[two_pos[(d.site1, d.site1, d.site2, d.site2)]]
    end
    return e
end

"対角 ⟨n_i⟩ の和。"
_phys_density_sum(data, g1) =
    sum(g1[n] for (n, t) in enumerate(data.green_one_terms) if t.site1 == t.site2)

@testset "§13.4 総和則と恒等式(平均): $(label)" for label in ("alpha_F2", "pair_F2")
    c = _phys_case(label)
    data, pstate = c.data, c.pstate
    mp = data.modpara
    mp.nvmc_warmup = 50
    mp.nvmc_sample = 200
    ctx = MVMCOptimizers.serial_context()
    MVMCOptimizers.initialize_phys_quantities!(pstate.state, data)
    MVMCOptimizers.parton_make_sample!(pstate, data, MersenneTwister(7))
    n_skipped = MVMCOptimizers.parton_phys_main_cal!(pstate, data)
    @test n_skipped == 0
    MVMCOptimizers.weight_average_we!(ctx, pstate.state)
    MVMCOptimizers.weight_average_green_func!(ctx, pstate.state)

    energy = pstate.state.energy
    phys = pstate.state.phys_quantities
    @test real(energy.wc) == mp.nvmc_sample
    @test isapprox(_phys_density_sum(data, phys.phys_cis_ajs), mp.nelec; atol = 1e-12)
    e_green = _phys_energy_from_green(pstate, data, phys.phys_cis_ajs, phys.phys_cis_ajs_ckt_alt_dc)
    @test isapprox(energy.etot, e_green; atol = 1e-12)
end

@testset "Review Focus 2: ノード上のサンプルは飛ばし、有効数で正規化する" begin
    c = _phys_case("orbparam_F2")
    data, pstate = c.data, c.pstate
    mp = data.modpara
    cfg = pstate.config
    r_node = 2
    for Φ in pstate.mfham.orbitals
        Φ[r_node, :] .= 0                     # サイト r_node を含む配置は全フレーバーで det = 0
    end
    # サンプルを手で詰める: 6 本のうち 2 本が r_node を含む
    samples = [[1, 3, 5], [1, 2, 4], [3, 4, 6], [2, 5, 6], [1, 4, 5], [3, 5, 6]]
    mp.nvmc_sample = length(samples)
    for (s, sites) in enumerate(samples)
        phys_place_sorted!(pstate, data, sites)
        MVMCOptimizers.parton_store_sample!(cfg, s)
    end
    MVMCOptimizers.initialize_phys_quantities!(pstate.state, data)
    n_skipped = MVMCOptimizers.parton_phys_main_cal!(pstate, data)
    @test n_skipped == 2
    @test real(pstate.state.energy.wc) == 4
    ctx = MVMCOptimizers.serial_context()
    MVMCOptimizers.weight_average_we!(ctx, pstate.state)
    MVMCOptimizers.weight_average_green_func!(ctx, pstate.state)
    @test isapprox(_phys_density_sum(data, pstate.state.phys_quantities.phys_cis_ajs), mp.nelec;
                   atol = 1e-12)
end

"上流書式の zvo_cisajs を読む(ri si rj sj Re Im)。"
function _read_cisajs(path)
    vals = ComplexF64[]
    sites = NTuple{2,Int}[]
    for line in eachline(path)
        tok = split(line)
        length(tok) == 6 || continue
        push!(sites, (parse(Int, tok[1]), parse(Int, tok[3])))
        push!(vals, complex(parse(Float64, tok[5]), parse(Float64, tok[6])))
    end
    return sites, vals
end

@testset "Review Focus 3 と出力: NDataQtySmp = 2" begin
    c = _phys_case("alpha_F2")
    data, pstate = c.data, c.pstate
    mp = data.modpara
    mp.nvmc_warmup = 50
    mp.nvmc_sample = 100
    mp.n_data_qty_smp = 2
    mp.n_data_idx_start = 1
    dir = mktempdir()
    status = MVMCOptimizers.parton_vmc_phys_cal!(pstate, data, MVMCOptimizers.serial_context();
                                                 rng = MersenneTwister(11), output_dir = dir)
    @test status == 0
    for k in (1, 2)
        path = joinpath(dir, "zvo_cisajs_$(lpad(k, 3, '0')).dat")
        @test isfile(path)
        sites, vals = _read_cisajs(path)
        @test length(vals) == length(data.green_one_terms)
        # ismp ごとに 0 に戻っていれば各ファイルの和は N(持ち越すと 2N になる)
        @test isapprox(sum(vals[n] for (n, (i, j)) in enumerate(sites) if i == j), mp.nelec;
                       atol = 1e-10)
        two = joinpath(dir, "zvo_cisajscktalt_$(lpad(k, 3, '0')).dat")
        @test isfile(two)
        @test count(!isempty ∘ strip, eachline(two)) == length(data.green_two_terms)
    end
    @test count(!isempty ∘ strip, eachline(joinpath(dir, "zvo_out.dat"))) == 2
end
