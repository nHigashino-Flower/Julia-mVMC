"""
物理量測定(NVMCCalMode = 1)の局所値・全配置列挙
--- parton-mode (fork addition) ---

DESIGN_parton.md §13。α・Φ・Pf の 3 モードに共通(振幅比は契約 2 の型ディスパッチ)。

局所値は O_loc(x) = Σ_x′ ⟨x|O|x′⟩ Ψ(x′)/Ψ(x)。⟨x|O = (O†|x⟩)† なので、O† を |x⟩ に右から順に
作用させた先の配置 x′ について Ψ(x′)/Ψ(x) を取る。比は E_loc のホップ項と同じく契約 2 の振幅比 ×
exp(Δln P_J)。統計の符号は粒子ラベルつきの det / Pf が運ぶので、ここで符号は足さない。

サイト番号は**すべて 1-based**(def の 0-based からの変換は呼び出し側の入口で 1 回だけ)。
"""

"""
    parton_green2_class(ri, rj, rk, rl) -> Symbol

2 体 ⟨b†_i b_j b†_k b_l⟩ の類(DESIGN §13.1 の表)。添字の等号だけで決まるので、0-based でも
1-based でも同じ結果になる。

- `:density`           i = j かつ k = l
- `:hop_then_density`  i ≠ j かつ k = l
- `:density_then_hop`  i = j かつ k ≠ l
- `:same_particle`     i ≠ j、k ≠ l、k = j(同じ粒子が i → j → l と動く)
- `:two_particle`      i ≠ j、k ≠ l、k ≠ j(別々の 2 粒子が動く。未対応)
"""
function parton_green2_class(ri::Int, rj::Int, rk::Int, rl::Int)
    if ri == rj
        return rk == rl ? :density : :density_then_hop
    end
    rk == rl && return :hop_then_density
    return rk == rj ? :same_particle : :two_particle
end

"""
    _parton_move_ratio(pstate, data, qp_weight, r_from, r_to) -> ComplexF64

サイト r_from の粒子を空サイト r_to へ移した配置との比 Ψ(x[r_from→r_to]) / Ψ(x)
(Jastrow 比込み)。契約 2 は純粋なので、何度呼んでも配置も振幅も動かない。
"""
function _parton_move_ratio(pstate::PartonOptimizationState, data::ExpertModeData,
                            qp_weight, r_from::Int, r_to::Int)
    cfg = pstate.config
    m = site_particle(cfg, r_from)
    ratio, _ = parton_amplitude_ratio!(pstate.workspace, pstate.amp, pstate.mfham, data,
                                       qp_weight, m, r_to)
    return ratio * exp(parton_log_proj_ratio(cfg, data, m, r_from, r_to))
end

"""
    parton_local_green1(pstate, data, qp_weight, ri, rj) -> ComplexF64

⟨b†_i b_j⟩ の局所値(1-based、DESIGN §13.1)。i = j なら n_i(x)、i 占有・j 空なら
Ψ(x[i→j])/Ψ(x)、それ以外は 0。呼び出し前提: 錨済み(E_loc と同じ)。
"""
function parton_local_green1(pstate::PartonOptimizationState, data::ExpertModeData,
                             qp_weight, ri::Int, rj::Int)
    cfg = pstate.config
    ri == rj && return ComplexF64(is_occupied(cfg, ri))
    (is_occupied(cfg, ri) && !is_occupied(cfg, rj)) || return zero(ComplexF64)
    return _parton_move_ratio(pstate, data, qp_weight, ri, rj)
end

"""
    parton_local_green2(pstate, data, qp_weight, ri, rj, rk, rl) -> ComplexF64

⟨b†_i b_j b†_k b_l⟩ の局所値(1-based、DESIGN §13.1 の 2 体の表)。O† = b†_l b_k b†_j b_i を
段 A(b†_j b_i)、段 B(b†_l b_k)の順に |x⟩ へ作用させる。`:two_particle` 類は error
(門番 `validate_parton_green_inputs` が def の段階で止めるので、本番では来ない)。
"""
function parton_local_green2(pstate::PartonOptimizationState, data::ExpertModeData,
                             qp_weight, ri::Int, rj::Int, rk::Int, rl::Int)
    cfg = pstate.config
    occ(r) = is_occupied(cfg, r)
    class = parton_green2_class(ri, rj, rk, rl)
    if class === :density
        return ComplexF64(occ(ri) && occ(rk))
    elseif class === :density_then_hop
        occ(ri) || return zero(ComplexF64)
        return parton_local_green1(pstate, data, qp_weight, rk, rl)
    elseif class === :hop_then_density
        (occ(ri) && !occ(rj)) || return zero(ComplexF64)
        # 段 A の後の配置 x[i→j] での n_k
        n_k_after = rk == rj ? true : (rk == ri ? false : occ(rk))
        n_k_after || return zero(ComplexF64)
        return _parton_move_ratio(pstate, data, qp_weight, ri, rj)
    elseif class === :same_particle
        (occ(ri) && !occ(rj)) || return zero(ComplexF64)
        rl == ri && return one(ComplexF64)          # i → j → i で元の配置に戻る
        occ(rl) && return zero(ComplexF64)
        return _parton_move_ratio(pstate, data, qp_weight, ri, rl)
    end
    error("parton_local_green2: two-particle term (i, j, k, l) = ($(ri - 1), $(rj - 1), " *
          "$(rk - 1), $(rl - 1)) (0-based) moves two distinct particles; it is not supported " *
          "(DESIGN_parton.md §13.1).")
end

"""
    _parton_green_site_lists(data) -> (one_body::Vector{NTuple{2,Int}}, two_body::Vector{NTuple{4,Int}})

greenone / greentwo の 0-based サイトを 1-based に直した添字表(変換はここ 1 箇所)。
"""
function _parton_green_site_lists(data::ExpertModeData)
    one_body = [(t.site1 + 1, t.site2 + 1) for t in data.green_one_terms]
    two_body = [(t.site1 + 1, t.site2 + 1, t.site3 + 1, t.site4 + 1) for t in data.green_two_terms]
    return one_body, two_body
end

"昇順の組み合わせ c(長さ k、値 1..n)を辞書式で次へ進める。最後なら false。"
function _parton_next_combination!(c::Vector{Int}, n::Int)
    k = length(c)
    i = k
    while i >= 1 && c[i] == n - k + i
        i -= 1
    end
    i == 0 && return false
    c[i] += 1
    for j = (i + 1):k
        c[j] = c[j - 1] + 1
    end
    return true
end

"""
    parton_exact_expectations(pstate, data; max_configurations = 2_000_000)
        -> (; one_body, two_body, energy, n_configurations)

全配置 C(Nsite, NElec) を列挙し、|Ψ(x)|² = |ip(x)|²·exp(2 ln P_J(x)) で重みづけた局所値の平均を返す
(小さい系の厳密値。モンテカルロの検証用)。`one_body` は `data.green_one_terms` の順、
`two_body` は `data.green_two_terms` の順、`energy` は E_loc の平均。測定ループと同じく
|ip| < 1e−30 の配置は飛ばす。配置数が上限を超えたら error。pstate の配置と振幅は書き換わる。
"""
function parton_exact_expectations(pstate::PartonOptimizationState, data::ExpertModeData;
                                   max_configurations::Int = 2_000_000)
    mp = data.modpara
    n_site, n_elec = mp.nsite, mp.nelec
    n_total = binomial(big(n_site), big(n_elec))
    n_total <= max_configurations || error(
        "parton_exact_expectations: C($n_site, $n_elec) = $n_total configurations exceed " *
        "max_configurations = $max_configurations.")

    cfg = pstate.config
    qp_weight = parton_qp_weight(data)
    one_sites, two_sites = _parton_green_site_lists(data)
    one_body = zeros(ComplexF64, length(one_sites))
    two_body = zeros(ComplexF64, length(two_sites))
    energy = zero(ComplexF64)
    norm = 0.0
    n_configurations = 0

    sites = collect(1:n_elec)
    while true
        n_configurations += 1
        for f = 1:cfg.n_flavor, (m, r) in enumerate(sites)
            cfg.ele_idx[(f - 1) * cfg.n_elec + m] = r
        end
        _rebuild_cfg_from_idx!(cfg)
        parton_make_proj_cnt!(cfg, data)
        parton_recompute_amplitude_all!(pstate.amp, pstate.mfham, cfg, data, pstate.workspace)
        ip = parton_calculate_ip(pstate.amp, qp_weight)
        if abs(ip) >= 1e-30
            ln_pj = 0.0
            if parton_n_proj(cfg) > 0
                for a = 1:n_elec, b = (a + 1):n_elec
                    ln_pj += _parton_jastrow_v(data, sites[a], sites[b])
                end
            end
            w = abs2(ip) * exp(2 * ln_pj)
            norm += w
            energy += w * parton_local_energy(pstate, data, ip)
            for (n, (ri, rj)) in enumerate(one_sites)
                one_body[n] += w * parton_local_green1(pstate, data, qp_weight, ri, rj)
            end
            for (n, (ri, rj, rk, rl)) in enumerate(two_sites)
                two_body[n] += w * parton_local_green2(pstate, data, qp_weight, ri, rj, rk, rl)
            end
        end
        _parton_next_combination!(sites, n_site) || break
    end

    norm > 0 || error("parton_exact_expectations: every configuration sits on a node (Ψ = 0).")
    return (one_body = one_body ./ norm, two_body = two_body ./ norm, energy = energy / norm,
            n_configurations = n_configurations)
end

# =====================================================================
# 測定ループ(DESIGN §13.4)
# =====================================================================

"""
    parton_phys_main_cal!(pstate, data) -> Int

保存済みサンプルを順に復元して錨を打ち直し、E_loc と全 Green の局所値を蓄積する
(`parton_main_cal!` の逐次経路から O と SR を抜いたもの)。ノード上(|ip| < 1e−30)のサンプルは
飛ばし、その数を返す。w = 1(重点サンプリング)。energy の wc / etot / etot2 は 0 から蓄積し直す。
`pstate.state.phys_quantities` は呼び出し側で `initialize_phys_quantities!` 済み・0 に戻し済みの前提。
"""
function parton_phys_main_cal!(pstate::PartonOptimizationState, data::ExpertModeData)
    cfg = pstate.config
    mp = data.modpara
    qp_weight = parton_qp_weight(data)
    phys = pstate.state.phys_quantities
    phys === nothing && error("parton_phys_main_cal!: call initialize_phys_quantities! first.")
    one_sites, two_sites = _parton_green_site_lists(data)

    energy = pstate.state.energy
    energy.wc = 0
    energy.etot = 0
    energy.etot2 = 0
    energy.sztot = 0
    energy.sztot2 = 0

    n_skipped = 0
    for s = 1:mp.nvmc_sample
        parton_restore_sample!(cfg, s)
        parton_recompute_amplitude_all!(pstate.amp, pstate.mfham, cfg, data, pstate.workspace)
        ip = parton_calculate_ip(pstate.amp, qp_weight)
        if abs(ip) < 1e-30
            n_skipped += 1            # ノード上のサンプル。寄与できないので飛ばす
            continue
        end
        w = 1.0
        e = parton_local_energy(pstate, data, ip)
        energy.wc += w
        energy.etot += w * e
        energy.etot2 += w * conj(e) * e
        for (n, (ri, rj)) in enumerate(one_sites)
            accumulate_phys_cis_ajs!(phys, n, w,
                                     parton_local_green1(pstate, data, qp_weight, ri, rj))
        end
        for (n, (ri, rj, rk, rl)) in enumerate(two_sites)
            accumulate_phys_cis_ajs_ckt_alt_dc!(
                phys, n, w, parton_local_green2(pstate, data, qp_weight, ri, rj, rk, rl))
        end
    end
    return n_skipped
end

"""
    parton_vmc_phys_cal!(pstate, data, ctx; rng, output_dir) -> Int

物理量測定の本体(DESIGN §13.4)。`ismp = 0 … NDataQtySmp−1` ごとに: アキュムレータを 0 に戻す →
サンプリング → 測定ループ → `weight_average_we!` → `weight_average_green_func!`(上流。rank 0 へ
集めて全体の Wc で割る)→ カウンタの集計 → 出力(上流の `output_data_phys!`。番号は
ismp + NDataIdxStart)。振幅の源(α の軌道・Φ・Pf の p)は呼び出し側で固定値に組み済みの前提。
"""
function parton_vmc_phys_cal!(pstate::PartonOptimizationState, data::ExpertModeData,
                              ctx::ParallelContext; rng, output_dir::String)
    mp = data.modpara
    out_rank = is_output_rank(ctx)
    state = pstate.state
    initialize_phys_quantities!(state, data)
    mkpath(output_dir)
    for ismp = 0:(mp.n_data_qty_smp - 1)
        out_rank && println("Start: Calculate parton physical quantities. Sampling: " *
                            "$ismp / $(mp.n_data_qty_smp)")
        reset_phys_quantities!(state)
        parton_make_sample!(pstate, data, rng)
        n_skipped = parton_phys_main_cal!(pstate, data)
        n_skipped > 0 && @warn "Skipped samples sitting on a node of the wave function" n_skipped ismp
        weight_average_we!(ctx, state)
        # 全体の Wc(allreduce 済みなので全ランクで同じ値)が 0 なら平均できない。0 埋めの結果を
        # 書かずに止める(L が大きいと |ip| < 1e-30 の閾値に全サンプルが掛かりうる)
        abs(state.energy.wc) < 1e-15 && error(
            "Parton measurement, bin $ismp: every sample on every rank sat on a node of the wave " *
            "function (|ip| < 1e-30), so there is nothing to average. No output is written for " *
            "this bin.")
        weight_average_green_func!(ctx, state)
        reduce_counter!(ctx, pstate.config.counter)
        out_rank && output_data_phys!(data, state, ismp; output_dir = output_dir)
        out_rank && println("End  : Calculate parton physical quantities. Sampling: " *
                            "$ismp / $(mp.n_data_qty_smp)")
    end
    return 0
end

# =====================================================================
# 門番(DESIGN §13.2 / §13.3)
# =====================================================================

"""
    validate_parton_green_inputs(data) -> Nothing

greenone / greentwo / greentwoex の門番(DESIGN §13.3)。

- NVMCCalMode = 1: サイトは [0, Nsite)、スピンはすべて 0(パートンの物理粒子にスピンは無い)、
  TwoBodyGEx は無いこと、greentwo に別々の 2 粒子を動かす項(`two_particle` 類)が無いこと。
  違反は error で、greentwo なら何番目の項(1-based)と 0-based の 4 添字を示す
- NVMCCalMode = 0: OneBodyG / TwoBodyG があれば warn(最適化では読まない)
"""
function validate_parton_green_inputs(data::ExpertModeData)
    mp = data.modpara
    if mp.vmc_calc_mode != 1
        if !isempty(data.green_one_terms) || !isempty(data.green_two_terms) ||
           !isempty(data.green_two_ex_terms)
            @warn "OneBodyG / TwoBodyG are not read in parameter optimization (NVMCCalMode = 0); " *
                  "set NVMCCalMode = 1 to measure them (DESIGN_parton.md §13)."
        end
        return nothing
    end

    isempty(data.green_two_ex_terms) || error(
        "Parton physical measurement does not support TwoBodyGEx (factored two-body Green, " *
        "used by upstream Lanczos); remove it from namelist.def and use TwoBodyG instead.")
    n_site = mp.nsite
    in_range(r) = 0 <= r < n_site
    for (n, t) in enumerate(data.green_one_terms)
        (in_range(t.site1) && in_range(t.site2)) || error(
            "OneBodyG term $n: site out of range [0, $n_site): (i, j) = ($(t.site1), $(t.site2)).")
        (t.spin1 === :up && t.spin2 === :up) || error(
            "OneBodyG term $n: the parton particle has no spin, so every spin column must be 0; " *
            "got spins ($(t.spin1), $(t.spin2)).")
    end
    for (n, t) in enumerate(data.green_two_terms)
        sites = (t.site1, t.site2, t.site3, t.site4)
        all(in_range, sites) || error(
            "TwoBodyG term $n: site out of range [0, $n_site): (i, j, k, l) = $sites.")
        all(==(:up), (t.spin1, t.spin2, t.spin3, t.spin4)) || error(
            "TwoBodyG term $n: the parton particle has no spin, so every spin column must be 0; " *
            "got spins $((t.spin1, t.spin2, t.spin3, t.spin4)).")
        parton_green2_class(sites...) === :two_particle && error(
            "TwoBodyG term $n: (i, j, k, l) = $sites is two_particle (it moves two distinct " *
            "particles), which parton measurement does not support. Supported: i = j, k = l, " *
            "or k = j (DESIGN_parton.md §13.1).")
    end
    return nothing
end

"""
    parton_phys_require_fixed_inputs(data, namelist_path; orbital_given, orbital_hint) -> Nothing

測定では乱数で埋めた値を使わない(DESIGN §13.2)。`orbital_given = false`(軌道のパラメータが
In ファイルで与えられていない)なら `orbital_hint` を添えて error。項のある射影族(Jastrow)の
In ファイルが namelist に無ければ error。
"""
function parton_phys_require_fixed_inputs(data::ExpertModeData, namelist_path::AbstractString;
                                          orbital_given::Bool, orbital_hint::AbstractString)
    orbital_given || error(
        "NVMCCalMode = 1 measures fixed parameters, but the orbital parameters were not given " *
        "by an input file (they would be random): $orbital_hint.")
    isempty(data.jastrow_terms) && return nothing
    path = _parton_namelist_entry(namelist_path, "InJastrow")
    path === nothing && error(
        "NVMCCalMode = 1 measures fixed parameters, but namelist.def has Jastrow terms " *
        "and no InJastrow line (the Jastrow factors would be zero). Add InJastrow.")
    # 上流のパーサは、ファイルが無ければ warn して読み飛ばし、4 行ヘッダなら idx 0 を黙って落とす。
    # どちらも最適化とは別の状態を黙って測ることになるので、ここで読み直して件数まで確かめる
    isfile(path) || error("InJastrow file listed in namelist.def but not found: $path")
    params = MVMCExpertModeParsers.parse_input_parameter_file(path)
    n = length(data.jastrow_terms)
    (length(params) == n && all(k -> 0 <= k < n, keys(params))) || error(
        "InJastrow: $path holds $(length(params)) Jastrow parameters but jastrowidx.def declares " *
        "$n (idx 0 … $(n - 1)). The optimizer's zqp_jastrow_opt.dat has a 4-line header, so read " *
        "as an In file its idx 0 is dropped: add one separator line to give it the 5-line header " *
        "the In format expects.")
    return nothing
end
