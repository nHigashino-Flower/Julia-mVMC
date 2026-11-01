"""
物理量測定(NVMCCalMode = 1、DESIGN_parton.md §13)のテストで共有する補助関数
--- parton-mode (fork addition) ---

`parton_test_helpers.jl` と `parton_pair_test_helpers.jl` を先に include しておくこと。

参照値は「昇順配置の Fock 基底で b†・b を素直に作用させた密行列の期待値」。プロダクションの
振幅比(契約 2)も E_loc も使わない。Ψ(x) だけは本番の振幅(錨 + ip)から取る — Ψ の定義そのもの
だからで、検証の対象は「局所値の作り方(どの配置へ動かし、どの比を取り、符号・共役をどう扱うか)」。
"""

using Random
using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

"""
    phys_all_green_terms!(data)

greenone に全 (i, j)、greentwo に §13.1 の扱える 4 類の全項を入れる(0-based、スピン 0 = :up)。
`two_particle` 類は入れない(門番が止める類なので局所値の比較対象にしない)。
"""
function phys_all_green_terms!(data)
    L = data.modpara.nsite
    data.green_one_terms = [
        MVMCExpertModeParsers.GreenOneTerm(i, j, :up, :up) for i = 0:(L - 1) for j = 0:(L - 1)
    ]
    two = MVMCExpertModeParsers.GreenTwoTerm[]
    for i = 0:(L - 1), j = 0:(L - 1), k = 0:(L - 1), l = 0:(L - 1)
        MVMCOptimizers.parton_green2_class(i, j, k, l) == :two_particle && continue
        push!(two, MVMCExpertModeParsers.GreenTwoTerm(i, j, k, l, :up, :up, :up, :up))
    end
    data.green_two_terms = two
    return data
end

"環の最近接に複素ホップと実クーロンを張る(対角クーロン 1 本で化学ポテンシャルも入れる)。"
function phys_ring_hamiltonian!(data; t = ComplexF64(-1.0, 0.3), v = 0.7)
    L = data.modpara.nsite
    for i = 0:(L - 1)
        j = mod(i + 1, L)
        push!(data.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(i, j, t, true))
        push!(data.coulomb_inter_terms, MVMCExpertModeParsers.CoulombInterTerm(i, j, v))
    end
    push!(data.coulomb_inter_terms, MVMCExpertModeParsers.CoulombInterTerm(1, 1, -0.4))
    return data
end

"α モードの状態(per_bond MF。軌道は α から対角化して作る)。"
function _phys_alpha_case(F::Int; L::Int, Ne::Int, jastrow::Bool, n_qp::Int, rng)
    data = per_bond_mf_data(F; n_site = L, n_elec = Ne)
    empty!(data.physhop_terms)                  # per_bond のホップは実数なので複素に張り替える
    empty!(data.coulomb_inter_terms)
    phys_ring_hamiltonian!(data)
    # per_bond の α は一様なので、idx ごとの乱数で崩して縮退と偶然の相殺を避ける
    # (同じ idx を共有するセルは同じ値でなければならない)
    n_idx = maximum(t.idx for t in data.pmfpara_terms) + 1
    shift = 0.3 .* randn(rng, ComplexF64, n_idx)
    for t in data.pmfpara_terms
        t.value = t.value + shift[t.idx + 1]
    end
    jastrow && _add_ring_jastrow!(data, [0.35, -0.2, 0.15][1:(L ÷ 2)])
    if n_qp > 1
        random_ring_qp!(rng, data, n_qp)
    end
    MVMCOptimizers.parton_materialize_flags!(data)
    phys_all_green_terms!(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    MVMCOptimizers.parton_update_orbitals!(
        pstate.mfham, MVMCOptimizers.parton_alpha_from_terms(data), data.modpara.nelec)
    return data, pstate
end

"Pf モードの状態(全対を別々の idx にした def と乱数の p)。"
function _phys_pair_case(F::Int; L::Int, Ne::Int, n_qp::Int, rng)
    data = pair_toy_data(; n_site = L, n_elec = Ne, n_flavor = F)
    mp = data.modpara
    mp.nvmc_warmup = 50                         # 測定ループのテスト(§13.4)でサンプリングに使う
    mp.nvmc_interval = 1
    mp.nvmc_sample = 200
    mp.parton_block_update_size = 4
    phys_ring_hamiltonian!(data)
    n_qp > 1 && random_ring_qp!(rng, data, n_qp)
    phys_all_green_terms!(data)
    terms = full_pair_terms(L, F)
    src = build_pair_source(data, terms, randn(rng, ComplexF64, length(terms)))
    nq = length(data.qp_trans)
    state = MVMCOptimizers.VMCOptimizationState(mp.nsite, mp.nelec, 0, 0, nq,
                                                max(mp.nvmc_sample, 1), true, false)
    pstate = MVMCOptimizers.PartonOptimizationState(
        state,
        MVMCOptimizers.PartonPairAmplitudeData(nq, F, Ne),
        MVMCOptimizers.PartonConfiguration(L, Ne, F, max(mp.nvmc_sample, 1); n_proj = 0),
        MVMCOptimizers.PartonPairSamplingWorkspace(nq, F, Ne),
        src,
        MVMCOptimizers.parton_build_phys_hamiltonian(data),
    )
    return data, pstate
end

"""
    phys_toy_cases() -> Vector{NamedTuple{(:label, :pstate, :data, :fermionic)}}

DESIGN §13.5-1 の 5 ケース。各 data には全 Green 項(`phys_all_green_terms!`)が入っている。
"""
function phys_toy_cases()
    rng = MersenneTwister(20261101)
    cases = NamedTuple{(:label, :pstate, :data, :fermionic)}[]

    data, ps = _phys_alpha_case(2; L = 6, Ne = 3, jastrow = true, n_qp = 2, rng = rng)
    push!(cases, (label = "alpha_F2", pstate = ps, data = data, fermionic = false))

    data, ps = _phys_alpha_case(3; L = 6, Ne = 3, jastrow = false, n_qp = 2, rng = rng)
    push!(cases, (label = "alpha_F3", pstate = ps, data = data, fermionic = true))

    data = orbparam_toy_data(; n_site = 6, n_elec = 3, n_flavor = 2)
    phys_all_green_terms!(data)
    push!(data.coulomb_inter_terms, MVMCExpertModeParsers.CoulombInterTerm(1, 1, -0.4))
    ps = MVMCOptimizers.parton_orbparam_build_state(data)
    push!(cases, (label = "orbparam_F2", pstate = ps, data = data, fermionic = false))

    data, ps = _phys_pair_case(2; L = 5, Ne = 2, n_qp = 2, rng = rng)
    push!(cases, (label = "pair_F2", pstate = ps, data = data, fermionic = false))

    data, ps = _phys_pair_case(3; L = 5, Ne = 2, n_qp = 1, rng = rng)
    push!(cases, (label = "pair_F3", pstate = ps, data = data, fermionic = true))
    return cases
end

"昇順のサイト集合 `sites`(1-based)を全フレーバーへ固縛配置し、錨を打って ip を返す。"
function phys_place_sorted!(pstate, data, sites::Vector{Int})
    cfg = pstate.config
    for f = 1:cfg.n_flavor, (m, r) in enumerate(sites)
        cfg.ele_idx[(f - 1) * cfg.n_elec + m] = r
    end
    MVMCOptimizers._rebuild_cfg_from_idx!(cfg)
    MVMCOptimizers.parton_make_proj_cnt!(cfg, data)
    MVMCOptimizers.parton_recompute_amplitude_all!(
        pstate.amp, pstate.mfham, cfg, data, pstate.workspace)
    return MVMCOptimizers.parton_calculate_ip(pstate.amp, MVMCOptimizers.parton_qp_weight(data))
end

"辞書式の全組み合わせ(1-based、昇順)。"
function phys_combinations(n::Int, k::Int)
    out = Vector{Int}[]
    c = collect(1:k)
    while true
        push!(out, copy(c))
        i = k
        while i >= 1 && c[i] == n - k + i
            i -= 1
        end
        i == 0 && return out
        c[i] += 1
        for j = (i + 1):k
            c[j] = c[j - 1] + 1
        end
    end
end

"""
    dense_wavefunction(data, pstate) -> (configs, psi)

昇順配置の Fock 基底 |x⟩ = b†_{r1} … b†_{rN}|0⟩ での Ψ(x) = ip(x)·P_J(x) を全列挙する。
"""
function dense_wavefunction(data, pstate)
    mp = data.modpara
    configs = phys_combinations(mp.nsite, mp.nelec)
    psi = ComplexF64[]
    for sites in configs
        ip = phys_place_sorted!(pstate, data, sites)
        lnpj = MVMCOptimizers.parton_n_proj(pstate.config) == 0 ? 0.0 :
               _ln_pj_ref(pstate.config, data)
        push!(psi, ip * exp(lnpj))
    end
    return configs, psi
end

"""
    _dense_hop(x, create, annihilate, fermionic) -> Union{Nothing, Tuple{Int, Vector{Int}}}

b†_create b_annihilate を昇順配置 x に作用させる(硬芯)。フェルミオンの符号は、作用するサイトより
小さい占有サイトの数の偶奇(b を先に、b† を後に)。0 になるなら nothing。
"""
function _dense_hop(x::Vector{Int}, create::Int, annihilate::Int, fermionic::Bool)
    annihilate in x || return nothing
    s = fermionic && isodd(count(<(annihilate), x)) ? -1 : 1
    x1 = filter(!=(annihilate), x)
    create in x1 && return nothing
    fermionic && isodd(count(<(create), x1)) && (s = -s)
    return s, sort!(push!(x1, create))
end

"""
    dense_expectation(configs, psi, ops, fermionic) -> ComplexF64

O = b†_{ops[1][1]} b_{ops[1][2]} b†_{ops[2][1]} b_{ops[2][2]} …(1-based)の期待値
⟨Ψ|O|Ψ⟩ / ⟨Ψ|Ψ⟩。O を右端の因子から |x′⟩ に作用させる。
"""
function dense_expectation(configs, psi, ops::Vector{Tuple{Int,Int}}, fermionic::Bool)
    index = Dict(c => n for (n, c) in enumerate(configs))
    num = zero(ComplexF64)
    for (n, x) in enumerate(configs)
        s, y = 1, x
        for (c, a) in reverse(ops)
            r = _dense_hop(y, c, a, fermionic)
            r === nothing && (s = 0; break)
            s *= r[1]
            y = r[2]
        end
        s == 0 && continue
        num += conj(psi[index[y]]) * s * psi[n]
    end
    return num / sum(abs2, psi)
end

"上流書式の zvo_cisajs を読む(ri si rj sj Re Im)。"
function phys_read_cisajs(path)
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

"上流書式の zvo_cisajscktalt を読む(ri si rj sj rk sk rl sl Re Im)。"
function phys_read_cisajscktalt(path)
    sites = NTuple{4,Int}[]
    vals = ComplexF64[]
    for line in eachline(path)
        tok = split(line)
        length(tok) == 10 || continue
        push!(sites, Tuple(parse(Int, tok[n]) for n in (1, 3, 5, 7)))
        push!(vals, complex(parse(Float64, tok[9]), parse(Float64, tok[10])))
    end
    return sites, vals
end

"greenone(全 (i, j))と greentwo(全 (i, i, j, j))を上流の書式で dir に書く。"
function phys_write_green_defs(dir, n_site::Int)
    open(joinpath(dir, "greenone.def"), "w") do io
        println(io, "===============================")
        println(io, "NCisAjs $(n_site^2)")
        println(io, "===============================")
        println(io, "======== Green functions ======")
        println(io, "===============================")
        for i = 0:(n_site - 1), j = 0:(n_site - 1)
            println(io, "$i 0 $j 0")
        end
    end
    open(joinpath(dir, "greentwo.def"), "w") do io
        println(io, "===============================")
        println(io, "NCisAjsCktAltDC $(n_site^2)")
        println(io, "===============================")
        println(io, "======== Green functions ======")
        println(io, "===============================")
        for i = 0:(n_site - 1), j = 0:(n_site - 1)
            println(io, "$i 0 $i 0 $j 0 $j 0")
        end
    end
end

"""
    phys_convert_to_measurement!(dir, in_files; n_site, n_data_qty_smp = 2, nvmc_sample = 100)

最適化の入力ディレクトリを測定用に書き換える: modpara を `NVMCCalMode 1`・`NDataQtySmp`・
`NDataIdxStart 1`・`NVMCSample` にし、namelist に OneBodyG / TwoBodyG と `in_files`
(namelist のキー => コピー元のファイル)を足す。
"""
function phys_convert_to_measurement!(dir, in_files; n_site::Int, n_data_qty_smp::Int = 2,
                                      nvmc_sample::Int = 100)
    mod = joinpath(dir, "modpara.def")
    lines = filter(l -> !startswith(l, "NVMCCalMode") && !startswith(l, "NVMCSample"),
                   readlines(mod))
    append!(lines, ["NVMCCalMode 1", "NDataQtySmp $n_data_qty_smp", "NDataIdxStart 1",
                    "NVMCSample $nvmc_sample"])
    write(mod, join(lines, "\n") * "\n")
    phys_write_green_defs(dir, n_site)
    open(joinpath(dir, "namelist.def"), "a") do io
        println(io, "OneBodyG       greenone.def")
        println(io, "TwoBodyG       greentwo.def")
        for (key, src) in in_files
            name = key * ".def"
            cp(src, joinpath(dir, name); force = true)
            println(io, "$key $name")
        end
    end
    return joinpath(dir, "namelist.def")
end

"""
    phys_check_outputs(namelist, out_dir, n_data_qty_smp) -> Vector{Tuple{ComplexF64,ComplexF64,ComplexF64}}

測定の出力を読み、ismp ごとに (Σ⟨n_i⟩, zvo_out の E, Green から組んだ E) を返す。
"""
function phys_check_outputs(namelist, out_dir, n_data_qty_smp::Int)
    data = MVMCExpertModeParsers.parse_expert_mode_files(namelist)
    e_lines = [split(l) for l in eachline(joinpath(out_dir, "zvo_out.dat")) if !isempty(strip(l))]
    out = Tuple{ComplexF64,ComplexF64,ComplexF64}[]
    for k = 1:n_data_qty_smp
        s1, g1 = phys_read_cisajs(joinpath(out_dir, "zvo_cisajs_$(lpad(k, 3, '0')).dat"))
        s2, g2 = phys_read_cisajscktalt(joinpath(out_dir, "zvo_cisajscktalt_$(lpad(k, 3, '0')).dat"))
        one = Dict(s => g1[n] for (n, s) in enumerate(s1))
        two = Dict(s => g2[n] for (n, s) in enumerate(s2))
        e_green = zero(ComplexF64)
        for h in data.physhop_terms
            e_green += h.value * one[(h.site2, h.site1)] + conj(h.value) * one[(h.site1, h.site2)]
        end
        for d in data.coulomb_inter_terms
            e_green += d.value * two[(d.site1, d.site1, d.site2, d.site2)]
        end
        n_sum = sum(g1[n] for (n, (i, j)) in enumerate(s1) if i == j)
        e = complex(parse(Float64, e_lines[k][1]), parse(Float64, e_lines[k][2]))
        push!(out, (n_sum, e, e_green))
    end
    return out
end
