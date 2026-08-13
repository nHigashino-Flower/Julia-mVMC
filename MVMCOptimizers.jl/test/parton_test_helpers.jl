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

"恒等 QP(射影なし)を実体化する。n_qp = 1、重みは 1。"
function set_identity_qp!(data::MVMCExpertModeParsers.ExpertModeData)
    n_site = data.modpara.nsite
    data.qp_trans = [collect(1:n_site)]
    data.qp_trans_sgn = [ones(Int, n_site)]
    return ComplexF64[1]
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
