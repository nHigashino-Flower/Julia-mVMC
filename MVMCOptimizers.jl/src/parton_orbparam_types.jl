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
    PartonOrbIndexMap

Φ の要素から変分パラメータ θ への写像(DESIGN §12.12、Pfaffian の `orbitalidx.def` 相当)。
Φ_f[s,n] = coeff[k]·θ[idx[k]](k は平坦位置)。

- `idx`・`coeff`: 長さ n_orb = NFlavor·NSite·NElec。idx は θ の 1-based 位置
- `opt_flags`: 長さ n_idx。θ ごとの OptFlag
- `scale_groups`: 再スケールの組。同じ θ を使うフレーバーを結んだ連結成分ごとに、
  その組のフレーバーが使う θ の位置(昇順)。組の並びは最小フレーバー番号順
- `is_identity`: 各要素が自分の θ を係数 1 で持つ写像(def なし)。このときは
  組み立てと O で既存の経路をそのまま通す(§12.11 とビット一致)
"""
struct PartonOrbIndexMap
    n_idx::Int
    idx::Vector{Int}
    coeff::Vector{ComplexF64}
    opt_flags::Vector{Bool}
    scale_groups::Vector{Vector{Int}}
    is_identity::Bool
end

function PartonOrbIndexMap(idx::Vector{Int}, coeff::Vector{ComplexF64},
                           opt_flags::AbstractVector{Bool}, n_site::Int, n_elec::Int,
                           n_flavor::Int)
    opt_flags = Vector{Bool}(opt_flags)     # BitVector も受ける
    n_orb = n_flavor * n_site * n_elec
    length(idx) == n_orb || error("index map has $(length(idx)) entries; expected $n_orb.")
    length(coeff) == n_orb || error("coefficient list has $(length(coeff)) entries; expected $n_orb.")
    n_idx = length(opt_flags)
    all(i -> 1 <= i <= n_idx, idx) || error("index map refers to θ outside 1:$n_idx.")
    block = n_site * n_elec

    # 同じ θ を使うフレーバーを結ぶ(union-find)
    parent = collect(1:n_flavor)
    root(f) = (while parent[f] != f; f = parent[f]; end; f)
    owner = zeros(Int, n_idx)               # その θ を最初に使ったフレーバー
    for f = 1:n_flavor, k = ((f - 1) * block + 1):(f * block)
        j = idx[k]
        if owner[j] == 0
            owner[j] = f
        else
            a, b = root(owner[j]), root(f)
            a == b || (parent[max(a, b)] = min(a, b))
        end
    end
    roots = unique(root(f) for f = 1:n_flavor)          # 最小フレーバー番号順
    groups = [sort!(unique(idx[k] for f = 1:n_flavor if root(f) == r
                           for k = ((f - 1) * block + 1):(f * block))) for r in roots]
    is_identity = idx == collect(1:n_orb) && all(==(1), coeff)
    return PartonOrbIndexMap(n_idx, idx, coeff, opt_flags, groups, is_identity)
end

"""
    parton_orbparam_identity_map(n_site, n_elec, n_flavor) -> PartonOrbIndexMap

def が無いときの写像。各要素が自分の θ を係数 1 で持ち、OptFlag はすべて可動、
再スケールの組はフレーバーごとの連続ブロック。
"""
function parton_orbparam_identity_map(n_site::Int, n_elec::Int, n_flavor::Int)
    n_orb = n_flavor * n_site * n_elec
    block = n_site * n_elec
    groups = [collect(((f - 1) * block + 1):(f * block)) for f = 1:n_flavor]
    return PartonOrbIndexMap(n_orb, collect(1:n_orb), ones(ComplexF64, n_orb),
                             fill(true, n_orb), groups, true)
end

"""
    PartonDirectPhi <: PartonOrbitalSource

Φ 直接変分モードの軌道の器。振幅系が読むのは `.orbitals` だけ。

α 専用のフィールド(`n_idx`・`min_gap`・`gauge_*`・`dorbitals` など)は持たない。
α 専用のコードに誤って渡ると、フィールドが無いという例外で止まる(安全側)。
"""
mutable struct PartonDirectPhi <: PartonOrbitalSource
    orbitals::Vector{Matrix{ComplexF64}}    # フレーバーごとに Ns×Ne
    map::PartonOrbIndexMap                  # Φ の要素 → θ(def が無ければ恒等写像)
end

PartonDirectPhi(n_site::Int, n_elec::Int, n_flavor::Int,
                map::PartonOrbIndexMap = parton_orbparam_identity_map(n_site, n_elec, n_flavor)) =
    PartonDirectPhi([zeros(ComplexF64, n_site, n_elec) for _ = 1:n_flavor], map)

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

`data.pmforb_params`(θ)から `phi.orbitals` を組み立てる。Φ_k = coeff[k]·θ[idx[k]]。
恒等写像のときは θ をそのまま写す(既存の経路)。θ の長さが写像の n_idx と
合わなければ error(黙って一部だけ写さない)。
"""
function parton_orbparam_load_orbitals!(phi::PartonDirectPhi, data::ExpertModeData)
    mp = data.modpara
    n_site, n_elec, n_flavor = mp.nsite, mp.nelec, mp.nflavor
    map = phi.map
    length(data.pmforb_params) == map.n_idx || error(
        "pmforb_params has $(length(data.pmforb_params)) entries but the orbital index " *
        "map has $(map.n_idx) parameters" *
        (map.is_identity ? " (= NFlavor × NSite × NElec)." : " (NPmfOrbIdx in PartonOrbIdx)."))
    length(phi.orbitals) == n_flavor || error(
        "PartonDirectPhi holds $(length(phi.orbitals)) flavors but NFlavor = $n_flavor.")
    for f = 1:n_flavor
        Φ = phi.orbitals[f]
        size(Φ) == (n_site, n_elec) || error(
            "PartonDirectPhi flavor $f has size $(size(Φ)); expected ($n_site, $n_elec).")
        if map.is_identity
            for s = 1:n_site, n = 1:n_elec
                Φ[s, n] = data.pmforb_params[parton_orbparam_flat_index(f, s, n, n_site, n_elec)]
            end
        else
            for s = 1:n_site, n = 1:n_elec
                k = parton_orbparam_flat_index(f, s, n, n_site, n_elec)
                Φ[s, n] = map.coeff[k] * data.pmforb_params[map.idx[k]]
            end
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
