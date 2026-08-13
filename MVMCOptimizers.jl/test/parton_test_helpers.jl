"""
パートンモードのテストで共有するトイ入力
--- parton-mode (fork addition) ---

複素位相つき t を必ず含める(DESIGN §8): 転置積と随伴の取り違えは実数では
見えないので、実数だけのフィクスチャは検証にならない。
"""

using MVMCExpertModeParsers
using MVMCOptimizers

"""
    toy_mf_data(; n_site=4, n_elec=2, n_flavor=2, t=..., mu=0.7)

周期鎖のトイ入力。変分グループは 2 つ:
- idx 0: 最近接ホッピング群(全フレーバー共有、複素 t)
- idx 1: オンサイト群(全サイト・全フレーバー共有、実 t)

α の初期値はどちらも 1 なので、H の非対角成分がそのまま t になる。
"""
function toy_mf_data(;
    n_site::Int = 4,
    n_elec::Int = 2,
    n_flavor::Int = 2,
    t::ComplexF64 = ComplexF64(-1.0, 0.4),
    mu::Float64 = 0.7,
)
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite = n_site
    mp.nelec = n_elec
    mp.nflavor = n_flavor
    mp.parton_mode = 1
    mp.two_sz = 0
    mp.complex_flag = 1
    mp.nex_update_path = 6

    for f = 0:(n_flavor - 1), i = 0:(n_site - 1)
        j = mod(i + 1, n_site)
        push!(
            data.pmftrans_terms,
            MVMCExpertModeParsers.PartonMFTransTerm(i, f, j, f, t, imag(t) != 0),
        )
        push!(
            data.pmfpara_terms,
            MVMCExpertModeParsers.PartonMFParaTerm(i, f, j, f, 0, ComplexF64(1, 0), true),
        )
    end
    for f = 0:(n_flavor - 1), i = 0:(n_site - 1)
        push!(
            data.pmftrans_terms,
            MVMCExpertModeParsers.PartonMFTransTerm(i, f, i, f, ComplexF64(mu, 0), false),
        )
        push!(
            data.pmfpara_terms,
            MVMCExpertModeParsers.PartonMFParaTerm(i, f, i, f, 1, ComplexF64(1, 0), true),
        )
    end
    return data
end

"""
    set_identity_qp!(data) -> Vector{ComplexF64}

恒等 QP(射影なし、n_qp = 1、重み 1)を実体化して重みベクトルを返す。
`data.qp_weights` も既存の init_qp_weight! で埋めるので、
`parton_qp_weight(data)` から同じ値が読める。
"""
function set_identity_qp!(data::MVMCExpertModeParsers.ExpertModeData)
    n_site = data.modpara.nsite
    data.qp_trans = [collect(1:n_site)]
    data.qp_trans_sgn = [ones(Int, n_site)]
    data.modpara.nsp_gauss_leg = 1
    data.modpara.nsp_stot = 0
    data.modpara.nmp_trans = 1
    data.para_qp_trans = ComplexF64[1]
    MVMCExpertModeParsers.init_qp_weight!(data)
    return data.qp_weights.qp_full_weight
end

"""
    set_shift_qp!(data, shifts, sgns, weights) -> Vector{ComplexF64}

非自明な並進 QP を実体化する。`shifts[qp][r]` は 1-based の写像、
`sgns[qp][r]` はサイトごとの符号。
"""
function set_shift_qp!(
    data::MVMCExpertModeParsers.ExpertModeData,
    shifts::Vector{Vector{Int}},
    sgns::Vector{Vector{Int}},
    weights::Vector{ComplexF64},
)
    data.qp_trans = shifts
    data.qp_trans_sgn = sgns
    data.modpara.nsp_gauss_leg = 1
    data.modpara.nsp_stot = 0
    data.modpara.nmp_trans = length(shifts)
    data.para_qp_trans = copy(weights)
    MVMCExpertModeParsers.init_qp_weight!(data)
    return data.qp_weights.qp_full_weight
end

"""
    build_toy_mfham(data; n_idx=2)

契約 0 のテンプレート構築と対角化まで一気に行い、mfham を返す。
"""
function build_toy_mfham(data::MVMCExpertModeParsers.ExpertModeData; n_idx::Int = 2)
    mp = data.modpara
    mfham = MVMCOptimizers.PartonMFHamiltonian(mp.nsite, mp.nelec, mp.nflavor, n_idx)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(
        mfham,
        MVMCOptimizers.parton_alpha_from_terms(data),
        mp.nelec,
    )
    return mfham
end

"""
    dimerized_mf_data(; n_flavor=2, ...)

SR を回すためのトイ入力。`toy_mf_data` は変分自由度が「一様ホッピング 1 個 +
一様オンサイト 1 個」しかなく、そのどちらもゲージ平坦方向(Φ は H → cH と
H → H + μI で不変)なので S 行列が全ゼロになって SR が解けない。こちらは
ボンドを強弱 2 群に分けてあり、その比が物理的な変分自由度になる。

- idx 0: 強ボンド (0,1), (2,3)。ゲージ代表としてフラグ行で凍結する想定
- idx 1: 弱ボンド (1,2), (3,0)。複素位相つきの物理的な自由度
- idx 2: 一様オンサイト。ゲージ平坦なので動かないのが正解

物理ハミルトニアンは一様な最近接ホップ + 最近接クーロン。
"""
function dimerized_mf_data(;
    n_flavor::Int = 2,
    n_site::Int = 4,
    n_elec::Int = 2,
    t_strong::ComplexF64 = ComplexF64(-1.0, 0.0),
    t_weak::ComplexF64 = ComplexF64(-1.0, 0.2),
    α_weak::ComplexF64 = ComplexF64(0.7, 0.1),
    t_phys::ComplexF64 = ComplexF64(-1.0, 0.0),
    v_phys::Float64 = 0.5,
)
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite = n_site
    mp.nelec = n_elec
    mp.nflavor = n_flavor
    mp.parton_mode = 1
    mp.two_sz = 0
    mp.complex_flag = 1
    mp.nex_update_path = 6

    bonds = [(i, mod(i + 1, n_site)) for i = 0:(n_site - 1)]
    for f = 0:(n_flavor - 1)
        for (b, (i, j)) in enumerate(bonds)
            strong = isodd(b)
            t = strong ? t_strong : t_weak
            idx = strong ? 0 : 1
            α = strong ? ComplexF64(1, 0) : α_weak
            push!(
                data.pmftrans_terms,
                MVMCExpertModeParsers.PartonMFTransTerm(i, f, j, f, t, imag(t) != 0),
            )
            push!(
                data.pmfpara_terms,
                MVMCExpertModeParsers.PartonMFParaTerm(i, f, j, f, idx, α, true),
            )
        end
        for i = 0:(n_site - 1)
            push!(
                data.pmftrans_terms,
                MVMCExpertModeParsers.PartonMFTransTerm(
                    i, f, i, f, ComplexF64(1, 0), false),
            )
            push!(
                data.pmfpara_terms,
                MVMCExpertModeParsers.PartonMFParaTerm(
                    i, f, i, f, 2, ComplexF64(0, 0), true),
            )
        end
    end

    for (i, j) in bonds
        push!(
            data.physhop_terms,
            MVMCExpertModeParsers.PhysHopTerm(i, j, t_phys, imag(t_phys) != 0),
        )
        push!(data.coulomb_inter_terms, MVMCExpertModeParsers.CoulombInterTerm(i, j, v_phys))
    end

    set_identity_qp!(data)
    mp.nvmc_warmup = 50
    mp.nvmc_interval = 1
    mp.nvmc_sample = 300
    mp.nblock_update_size = 8
    mp.nsr_opt_itr_step = 3
    mp.nsr_opt_itr_smp = 1
    mp.nstore_o = 1
    mp.dsr_opt_step_dt = 0.02
    mp.dsr_opt_red_cut = 1e-8
    mp.dsr_opt_sta_del = 0.02
    return data
end

"サイト集合 `sites`(1-based)を全フレーバーへ固縛配置した cfg を返す。"
function toy_config(
    data::MVMCExpertModeParsers.ExpertModeData,
    sites::Vector{Int};
    n_sample::Int = 1,
)
    mp = data.modpara
    cfg = MVMCOptimizers.PartonConfiguration(mp.nsite, mp.nelec, mp.nflavor, n_sample)
    for f = 1:mp.nflavor, (m, r) in enumerate(sites)
        MVMCOptimizers.place_particle!(cfg, f, m, r)
    end
    return cfg
end

"""
    per_bond_mf_data(F; n_site=4, n_elec=2)

ボンドごとに独立な α を持つ MF(変分自由度を確保するため)+ 一様な物理ホップの環。
`toy_mf_data` や `dimerized_mf_data` より自由度が多く、SR が実際に降下する。
ボンド 1 の α をゲージ代表として OptFlag で凍結する前提。
"""
function per_bond_mf_data(F::Int; n_site::Int = 4, n_elec::Int = 2)
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite = n_site
    mp.nelec = n_elec
    mp.nflavor = F
    mp.parton_mode = 1
    mp.two_sz = 0
    mp.complex_flag = 1
    mp.nex_update_path = 6

    bonds0 = [(i, mod(i + 1, n_site)) for i = 0:(n_site - 1)]
    for f = 0:(F - 1)
        for (b, (i, j)) in enumerate(bonds0)
            push!(
                data.pmftrans_terms,
                MVMCExpertModeParsers.PartonMFTransTerm(i, f, j, f, ComplexF64(-1, 0), false),
            )
            # ボンド 1 はゲージ代表として 1 に固定。残りは非対称な初期値
            α = b == 1 ? ComplexF64(1, 0) : ComplexF64(0.85 + 0.07b, 0.13b)
            push!(
                data.pmfpara_terms,
                MVMCExpertModeParsers.PartonMFParaTerm(i, f, j, f, b - 1, α, true),
            )
        end
    end
    for (i, j) in bonds0
        push!(
            data.physhop_terms,
            MVMCExpertModeParsers.PhysHopTerm(i, j, ComplexF64(-1, 0), false),
        )
    end

    mp.nsp_gauss_leg = 1
    mp.nsp_stot = 0
    mp.nmp_trans = 1
    data.para_qp_trans = ComplexF64[1]
    data.qp_trans = [collect(1:n_site)]
    data.qp_trans_sgn = [ones(Int, n_site)]
    MVMCExpertModeParsers.init_qp_weight!(data)

    mp.nvmc_warmup = 200
    mp.nvmc_interval = 1
    mp.nvmc_sample = 2000
    mp.nblock_update_size = 8
    mp.nstore_o = 1
    mp.dsr_opt_step_dt = 0.05
    mp.dsr_opt_sta_del = 0.02
    mp.dsr_opt_red_cut = 1e-8
    mp.nsr_opt_itr_step = 120
    mp.nsr_opt_itr_smp = 1
    data.pmfpara_opt_flags = Dict(0 => 0)   # ゲージ代表を凍結
    return data
end

