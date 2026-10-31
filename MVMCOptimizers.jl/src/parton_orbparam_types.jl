"""
Φ 直接変分モードの型と添字
--- parton-mode (fork addition) ---

DESIGN_parton.md §12.11(上流型の簡素版)。上流 mVMC が Pfaffian の f_ij を
そのまま変分するのと同じ形で、Slater 行列式の軌道行列 Φ_f[s,n] の全要素を
SR の変分パラメータにする(`PartonOrbParam = 1`)。

**SR が読み書きする正準の置き場は `data.pmforb_params`**(平坦な複素ベクトル)。
状態側の `PartonDirectPhi.orbitals` は毎ステップそこから写す導出量で、
振幅系(契約 1〜4)はこちらだけを読む。

平坦添字は 0-based で ((f·Ns)+s)·Ne+n。1-based の位置 k はこれに 1 を足したもの
(`parton_orbparam_flat_index`)。InPmfOrb.def の idx 列は 0-based で書く。
"""

# PartonOrbParam の値(DESIGN §12.1 / §12.11)
const PARTON_ORB_PARAM_ALPHA = 0      # α(既定): Φ は H_MF(α) の固有ベクトル
const PARTON_ORB_PARAM_DIRECT = 1     # Φ の全要素を直接変分(上流型、第 1 段で実装)
const PARTON_ORB_PARAM_THOULESS = 2   # Thouless 型(予約。未実装)

"""
    PartonDirectPhi <: PartonOrbitalSource

Φ 直接変分モードの軌道の器。振幅系が読むのは `.orbitals` だけ。

α 専用のフィールド(`n_idx`・`min_gap`・`gauge_*`・`dorbitals` など)は持たない。
α 専用のコードに誤って渡ると、フィールドが無いという例外で止まる(安全側)。
"""
mutable struct PartonDirectPhi <: PartonOrbitalSource
    orbitals::Vector{Matrix{ComplexF64}}    # フレーバーごとに Ns×Ne
end

PartonDirectPhi(n_site::Int, n_elec::Int, n_flavor::Int) =
    PartonDirectPhi([zeros(ComplexF64, n_site, n_elec) for _ = 1:n_flavor])

"""
    parton_orbparam_flat_index(f, s, n, n_site, n_elec) -> Int

Φ_f[s, n] の平坦位置。引数も返り値も 1-based(= 0-based 添字 ((f·Ns)+s)·Ne+n に 1 を足したもの)。
"""
@inline parton_orbparam_flat_index(f::Int, s::Int, n::Int, n_site::Int, n_elec::Int) =
    ((f - 1) * n_site + (s - 1)) * n_elec + n

"Φ の要素数 NFlavor·NSite·NElec。"
parton_orbparam_n_orb(mp::ModParaParameters) = mp.nflavor * mp.nsite * mp.nelec

"""
    parton_orbparam_load_orbitals!(phi, data) -> phi

`data.pmforb_params`(SR の値)を `phi.orbitals` へ写す。長さが
NFlavor·NSite·NElec と合わなければ error(黙って一部だけ写さない)。
"""
function parton_orbparam_load_orbitals!(phi::PartonDirectPhi, data::ExpertModeData)
    mp = data.modpara
    n_site, n_elec, n_flavor = mp.nsite, mp.nelec, mp.nflavor
    n_orb = parton_orbparam_n_orb(mp)
    length(data.pmforb_params) == n_orb || error(
        "pmforb_params has $(length(data.pmforb_params)) entries but " *
        "NFlavor × NSite × NElec = $n_orb are required.")
    length(phi.orbitals) == n_flavor || error(
        "PartonDirectPhi holds $(length(phi.orbitals)) flavors but NFlavor = $n_flavor.")
    for f = 1:n_flavor
        Φ = phi.orbitals[f]
        size(Φ) == (n_site, n_elec) || error(
            "PartonDirectPhi flavor $f has size $(size(Φ)); expected ($n_site, $n_elec).")
        for s = 1:n_site, n = 1:n_elec
            Φ[s, n] = data.pmforb_params[parton_orbparam_flat_index(f, s, n, n_site, n_elec)]
        end
    end
    return phi
end

"""
    parton_orbparam_params_from_orbitals(orbitals) -> Vector{ComplexF64}

`parton_orbparam_load_orbitals!` の逆。α 解の軌道から Φ モードの入力を作るときに使う。
"""
function parton_orbparam_params_from_orbitals(orbitals::Vector{Matrix{ComplexF64}})
    n_flavor = length(orbitals)
    n_site, n_elec = size(orbitals[1])
    params = zeros(ComplexF64, n_flavor * n_site * n_elec)
    for f = 1:n_flavor
        size(orbitals[f]) == (n_site, n_elec) || error(
            "flavor $f has size $(size(orbitals[f])); expected ($n_site, $n_elec).")
        for s = 1:n_site, n = 1:n_elec
            params[parton_orbparam_flat_index(f, s, n, n_site, n_elec)] = orbitals[f][s, n]
        end
    end
    return params
end
