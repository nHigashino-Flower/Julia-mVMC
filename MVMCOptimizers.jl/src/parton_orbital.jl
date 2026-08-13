"""
契約 0 / 0′: 平均場ハミルトニアンの組み立て・対角化・軌道微分
--- parton-mode (fork addition) ---

DESIGN_parton.md §1.2(H は α について線形)と §1.5(連鎖律の eigen 段を
一次摂動論で処理)に対応する。
"""

"""
    parton_build_mf_templates!(mfham, data) -> mfham

契約 0(起動時 1 回): pmftrans(固定係数 t)と pmfpara(α と idx 写像)を
結合キー (site1, site2, flavor) で突き合わせ、∂H/∂α_k の定数テンプレートを組む。

ここは結合を持つ唯一の場所なので、結合を要する検証もここが家:
双方向の完全性、逆向きの重複列挙、1 つの idx グループ内でのオンサイトと
ホッピングの混在、オンサイト t の実数性、idx の 0-based 連番性、共有 idx の
初期値一致、各フレーバーに項が存在すること。行単位で決まる検査(範囲、
フレーバー混成、physhop の規約)は門番の担当。

0-based から 1-based への変換はこの関数の一箇所だけで行う(DESIGN §3.1)。

注記: `site1 == site2 ? :onsite : :hopping` という分類が正しいのは
flavor1 == flavor2 を門番が保証しているから。混成を許すと「サイト対角・
フレーバー非対角」の項が現れ、それは対角要素ではなく h.c. の要る非対角要素に
なる。将来ゲートを外すときは分類条件が
`site1 == site2 && flavor1 == flavor2` に変わる。
"""
function parton_build_mf_templates!(mfham::PartonMFHamiltonian, data::ExpertModeData)
    # (i, j, f) → idx の結合表。ローカル・使い捨て(0-based のまま)。
    idx_of = Dict{NTuple{3,Int},Int}()
    value_of = Dict{Int,ComplexF64}()
    for p in data.pmfpara_terms
        key = (p.site1, p.site2, p.flavor1)
        haskey(idx_of, key) &&
            error("pmfpara.def: duplicate entry for (site1, site2, flavor) = $key.")
        idx_of[key] = p.idx

        prev = get(value_of, p.idx, nothing)
        if prev === nothing
            value_of[p.idx] = p.value
        elseif prev != p.value
            error(
                "pmfpara.def: idx $(p.idx) is shared by several cells but their " *
                "initial values disagree ($prev vs $(p.value)). Cells sharing an " *
                "idx share one variational parameter, so they must start equal.",
            )
        end
    end

    isempty(idx_of) && error("pmfpara.def: no mean-field parameter cells.")
    n_idx = maximum(values(idx_of)) + 1
    sort!(collect(keys(value_of))) == collect(0:(n_idx - 1)) || error(
        "pmfpara.def: idx values must be a contiguous 0-based sequence " *
        "0..$(n_idx - 1), got $(sort!(collect(keys(value_of)))).",
    )
    mfham.n_idx == n_idx || error(
        "PartonMFHamiltonian was allocated for n_idx = $(mfham.n_idx) but " *
        "pmfpara.def declares $n_idx parameters. The derivative buffers are " *
        "sized by n_idx, so the two must agree.",
    )

    template = [PartonMFTemplateEntry[] for _ = 1:n_idx]
    group_kind = fill(:unset, n_idx)
    matched = Set{NTuple{3,Int}}()

    for t in data.pmftrans_terms
        key = (t.site1, t.site2, t.flavor1)
        key in matched && error("pmftrans.def: duplicate entry for $key.")
        if t.site1 != t.site2 && (t.site2, t.site1, t.flavor1) in matched
            error(
                "pmftrans.def: both directions of $key are listed. List each " *
                "bond once — the Hermitian conjugate is added implicitly.",
            )
        end
        push!(matched, key)

        haskey(idx_of, key) || error(
            "pmftrans.def: $key has no matching pmfpara idx. A fixed (non-" *
            "variational) coupling is expressed as an OptFlag-frozen α, not as " *
            "a missing pmfpara row.",
        )
        k = idx_of[key] + 1                    # 0-based → 1-based はこの一行だけ

        kind = t.site1 == t.site2 ? :onsite : :hopping
        if kind === :onsite && abs(imag(t.value)) > 1e-12
            error(
                "pmftrans.def: the onsite coefficient at site $(t.site1) " *
                "(flavor $(t.flavor1)) must be real, got $(t.value). Onsite " *
                "terms are added without a Hermitian conjugate, so a complex " *
                "coefficient would break the Hermiticity of H.",
            )
        end
        if group_kind[k] === :unset
            group_kind[k] = kind
        elseif group_kind[k] !== kind
            error(
                "pmftrans.def: idx $(k - 1) mixes onsite and hopping terms in " *
                "one group. They differ in whether the Hermitian conjugate is " *
                "added, so they cannot share a variational parameter.",
            )
        end

        push!(
            template[k],
            PartonMFTemplateEntry(t.site1 + 1, t.site2 + 1, t.flavor1 + 1, t.value),
        )
    end

    length(matched) == length(idx_of) || error(
        "pmfpara.def: $(length(idx_of) - length(matched)) cell(s) have no " *
        "matching pmftrans coefficient. The coupling must be complete in both " *
        "directions.",
    )

    # 空の H は eigen が縮退したゴミを返すので、ここで気づく。
    n_flavor = data.modpara.nflavor
    for f = 1:n_flavor
        any(k -> any(e -> e.flavor == f, template[k]), 1:n_idx) || error(
            "No mean-field term acts on flavor $(f - 1). Every flavor needs at " *
            "least one term: diagonalising an all-zero Hamiltonian yields an " *
            "arbitrary degenerate basis.",
        )
    end

    for k = 1:n_idx
        mfham.template[k] = template[k]
        mfham.is_onsite_group[k] = group_kind[k] === :onsite
    end

    # --- フレーバー対称の検出(v3.9 det^F 高速路)------------------------------
    # idx 写像と係数 t^(f) が全フレーバーで一致するなら、任意の α に対して
    # H^(f) が全フレーバーで同一 → Φ / ∂Φ / A ブロックも同一。
    # 判定は「(k, site1, site2, coeff) の集合がフレーバー間で一致するか」。
    # onsite/hopping 分類は群単位(is_onsite_group)なので自動的に共通。
    # 同じ物理を h.c. の向きを変えて書いた入力((s1,s2,t) vs (s2,s1,conj t))は
    # 保守的に非対称と判定する — 正しさは変わらず、高速路が効かないだけ。
    mfham.flavor_symmetric =
        data.modpara.parton_flavor_sym_fast != 0 &&
        _parton_templates_flavor_symmetric(template, n_flavor, n_idx)

    parton_resolve_gauge_groups!(mfham, data)
    return mfham
end

"テンプレートの (k, site1, site2, coeff) 集合が全フレーバーで一致するか。"
function _parton_templates_flavor_symmetric(
    template::Vector{Vector{PartonMFTemplateEntry}}, n_flavor::Int, n_idx::Int)
    n_flavor >= 2 || return false          # F=1 に高速化の意味はない
    sig(f) = sort!([
        (k, e.site1, e.site2, real(e.coeff), imag(e.coeff))
        for k = 1:n_idx for e in template[k] if e.flavor == f
    ])
    ref = sig(1)
    return all(sig(f) == ref for f = 2:n_flavor)
end

"""
    parton_resolve_gauge_groups!(mfham, data)

α のゲージ平坦方向を解決して `mfham` に保持する(起動時 1 回、テンプレート構築の直後)。

DESIGN §2.5 の平坦方向は 2 種類:

**スケール**: `α → c·α` で `c` は**正の実数**。`H^(f) → c H^(f)` は固有値だけを
`c` 倍し(`c > 0` なので順序も保つ)固有ベクトルを変えないので Φ が不変。
位相 `e^{iθ}` は不変ではない — H が h.c. を含むため `α → e^{iθ}α` は
`H → e^{iθ}T + e^{-iθ}T†` となり別のハミルトニアンになる。よって 1 群あたり実 1 次元。

Φ^(f) は H^(f) にしか依存しないので、素朴にはフレーバーごとに独立な `c_f` がある。
ただし 1 つの idx が複数フレーバーに跨って項を持つと、その idx を通じて `c_f` が
連動する。したがって独立なスケール群は「フレーバーを節点、共有 idx を辺とするグラフの
連結成分」で決まる。共有なしなら F 個、全共有なら 1 個。**個数を仮定しないこと**。

**シフト**: `H^(f) → H^(f) + μ I`。オンサイト idx が全サイトを等係数で覆っているときだけ
現れる。再正規化(スケール射影)では潰れないので別扱い。
"""
function parton_resolve_gauge_groups!(mfham::PartonMFHamiltonian, data::ExpertModeData)
    n_idx = mfham.n_idx
    n_flavor = data.modpara.nflavor

    # --- スケール群: フレーバーを Union-Find で束ねる ---
    parent = collect(1:n_flavor)
    find(x) = parent[x] == x ? x : (parent[x] = find(parent[x]))
    union!(a, b) = (ra = find(a); rb = find(b); ra != rb && (parent[ra] = rb))

    flavors_of_idx = [Set{Int}() for _ = 1:n_idx]
    for k = 1:n_idx, e in mfham.template[k]
        push!(flavors_of_idx[k], e.flavor)
    end
    for k = 1:n_idx
        fs = collect(flavors_of_idx[k])
        for i = 2:length(fs)
            union!(fs[1], fs[i])
        end
    end

    root_to_group = Dict{Int,Vector{Int}}()
    for k = 1:n_idx
        isempty(flavors_of_idx[k]) && continue
        r = find(first(flavors_of_idx[k]))
        push!(get!(root_to_group, r, Int[]), k)
    end
    scale_groups = [sort(g) for g in values(root_to_group)]
    sort!(scale_groups; by = first)

    # --- シフト群: 各フレーバーで、オンサイト idx が全サイトを等係数で覆うか ---
    n_site = size(mfham.h_mf[1], 1)
    shift_groups = Vector{Int}[]
    for f = 1:n_flavor
        onsite_idx = [k for k = 1:n_idx if mfham.is_onsite_group[k] &&
                      any(e -> e.flavor == f, mfham.template[k])]
        isempty(onsite_idx) && continue
        covered = Int[]
        coeffs = Float64[]
        for k in onsite_idx, e in mfham.template[k]
            e.flavor == f || continue
            push!(covered, e.site1)
            push!(coeffs, real(e.coeff))
        end
        # 全サイトをちょうど 1 回ずつ、等しい係数で覆っているときだけシフト方向になる
        sort(covered) == collect(1:n_site) || continue
        all(c -> isapprox(c, coeffs[1]; rtol = 1e-12), coeffs) || continue
        abs(coeffs[1]) > 1e-12 || continue
        push!(shift_groups, sort(onsite_idx))
    end
    unique!(shift_groups)

    # --- 引き戻し先のノルムは初期 α に固定(再現性のため) ---
    α0 = parton_alpha_from_terms(data)
    target = [sqrt(sum(abs2, view(α0, g))) for g in scale_groups]

    mfham.gauge_scale_groups = scale_groups
    mfham.gauge_shift_groups = shift_groups
    mfham.gauge_target_norm = target
    return mfham
end

"""
    parton_alpha_from_terms(data) -> Vector{ComplexF64}

α の正準置き場は `pmfpara_terms[].value`(パラメータロケータの読み書き先)
なので、契約 0 に渡す前に毎ステップここで idx 順のベクトルへ詰め直す。
共有 idx の値一致は契約 0 の build が検証済みなので、上書きしても同値。
"""
function parton_alpha_from_terms(data::ExpertModeData)
    n_idx = parton_n_idx(data)
    α = zeros(ComplexF64, n_idx)
    for t in data.pmfpara_terms
        α[t.idx + 1] = t.value
    end
    return α
end

"""
    parton_update_orbitals!(mfham, alpha, n_elec; gap_tol=1e-8) -> mfham

契約 0(SR ステップ毎): H(α) を組んでフレーバーごとに対角化し、占有軌道 Φ を
更新する。h.c. はホッピング群にだけ暗黙付与し、オンサイト群は直接加算
(t は実数であることを build が検証済み)。

`eigen(Hermitian(...))` は LAPACK の zheev 経路を通る意図の宣言でもある。
非占有ベクトルも `eig_vecs` に残すのは契約 0′ の摂動論が使うため。
`min_gap` は HOMO-LUMO ギャップで、摂動論の分母を下から抑える保険。
"""
function parton_update_orbitals!(
    mfham::PartonMFHamiltonian,
    alpha::Vector{ComplexF64},
    n_elec::Int;
    gap_tol::Float64 = 1e-8,
)
    foreach(H -> fill!(H, 0), mfham.h_mf)
    for k = 1:mfham.n_idx
        onsite = mfham.is_onsite_group[k]
        for e in mfham.template[k]
            v = alpha[k] * e.coeff
            H = mfham.h_mf[e.flavor]
            H[e.site1, e.site2] += v
            onsite || (H[e.site2, e.site1] += conj(v))   # h.c. はホッピングのみ
        end
    end

    mfham.min_gap = Inf
    n_site = size(mfham.h_mf[1], 1)
    for f in eachindex(mfham.h_mf)
        if mfham.flavor_symmetric && f > 1
            # 対称なら H^(f) は f=1 と同一。eigen を繰り返す代わりにコピーする。
            # 同一入力の LAPACK は決定的なので値は変わらない(コピーはそれを
            # 定義で保証する)。バンド出力などが全フレーバー分を読むので、
            # f > 1 も空にはしない。
            mfham.eig_vals[f] .= mfham.eig_vals[1]
            mfham.eig_vecs[f] .= mfham.eig_vecs[1]
            mfham.orbitals[f] .= mfham.orbitals[1]
            continue
        end
        F = eigen(Hermitian(mfham.h_mf[f]))
        mfham.eig_vals[f] .= F.values
        mfham.eig_vecs[f] .= F.vectors
        mfham.orbitals[f] .= @view F.vectors[:, 1:n_elec]
        if n_elec < n_site
            mfham.min_gap =
                min(mfham.min_gap, F.values[n_elec + 1] - F.values[n_elec])
        end
    end
    mfham.min_gap < gap_tol &&
        @warn "MF spectrum nearly degenerate at the Fermi level" min_gap = mfham.min_gap
    return mfham
end

"""
    parton_update_orbital_derivatives!(mfham, n_elec)

契約 0′(SR ステップ毎、契約 0 の直後): ∂Φ^(f)/∂θ を一次摂動論で構築する。

連鎖律の eigen 段(DESIGN §1.5):

    W = U_unocc† (∂θH) U_occ,   G[u, n] = W[u, n] / (ε_n − ε_u),   ∂Φ = U_unocc · G

占有↔占有の混合は Tr[A⁻¹ ∂A] から厳密に消える(摂動係数が反エルミートで
対角がゼロ)ので、非占有への漏れだけ構築すればよい。分母は HOMO-LUMO ギャップで
下から抑えられる。

∂θH は α に非正則な H の実自由度ごとの微分:

    ∂H/∂Re α_k = T_k + T_k†      (オンサイト群は T_k のみ、t は実)
    ∂H/∂Im α_k = i (T_k − T_k†)  (オンサイト群の Im は凍結なのでゼロを格納)

`Uu' * dHUo` の随伴(共役)はここでは正しい。これはブラ ⟨φ_u| との真の内積で、
「転置積を使い dot は禁止」の規則(DESIGN §7)は振幅の双線形縮約(契約 2/3/5 の
最内ループ)の話。この 2 種の縮約を「統一」しないこと。
"""
function parton_update_orbital_derivatives!(mfham::PartonMFHamiltonian, n_elec::Int;
                                            gap_tol::Float64 = 1e-8)
    # gap_tol(v3.11 応急処置、参照 chi-VMC `vmc_chi_grad.jl` と同じ式・同じ既定値):
    # |ε_n − ε_u| < gap_tol の摂動項は 0 に落とす。フェルミ準位の level crossing
    # 近傍で ∂Φ ∝ 1/gap が発散して SR を壊す(REPORT §14)のを Inf/NaN の手前で
    # 止める。根治(gap トラスト領域など)は別途 — これは参照実証済みの下限ガード。
    n_site = size(mfham.h_mf[1], 1)
    n_un = n_site - n_elec
    # W は dh_uo_scratch(n_site × Ne)の先頭 n_un 行を間借りする。専用バッファを
    # 足さないのは、旧経路の dHUo(この scratch の本来の用途)が rank-1 化で
    # 不要になり、まるごと空いたため。
    W = view(mfham.dh_uo_scratch, 1:n_un, :)
    for f in eachindex(mfham.h_mf)
        if mfham.flavor_symmetric && f > 1
            # 対称なら ∂Φ^(f) も f=1 と同一(H・テンプレートとも同一)。
            # コピーは gemm(O(NSite·n_un·Ne))より n_un 倍安い。
            for dof in eachindex(mfham.dorbitals[f])
                mfham.dorbitals[f][dof] .= mfham.dorbitals[1][dof]
            end
            continue
        end
        U = mfham.eig_vecs[f]
        ev = mfham.eig_vals[f]
        Uo = @view U[:, 1:n_elec]
        Uu = @view U[:, (n_elec + 1):n_site]

        for k = 1:mfham.n_idx
            onsite = mfham.is_onsite_group[k]
            for part = 1:2                        # 1 = Re, 2 = Im
                dof = 2 * (k - 1) + part
                dPhi = mfham.dorbitals[f][dof]
                if onsite && part == 2
                    fill!(dPhi, 0)                # 門番が凍結する成分
                    continue
                end

                # W = Uu'(∂θH)Uo を、ボンドごとの rank-1 更新で直接積み上げる。
                # 旧経路の「dHUo を疎に組む → 密 gemm」は gemm の時点で疎性が
                # 消えて O(n_un·n_site·Ne) だった。rank-1 なら O(b_k·n_un·Ne)。
                #
                #   ∂H/∂Re α = T + T†   → t·(s1,s2) + conj(t)·(s2,s1)
                #   ∂H/∂Im α = i(T − T†) → c·(s1,s2) + conj(c)·(s2,s1), c = i·t
                #   onsite(t 実)       → t·(s1,s1)
                #
                # (a, s1, s2) は W += a · Uu'[:, s1] ⊗ Uo[s2, :] の意。
                # conj(c) = −i·conj(t) が −T† 側を正しく運ぶ(旧経路と同一の式)。
                fill!(W, 0)
                for e in mfham.template[k]
                    e.flavor == f || continue
                    t = e.coeff
                    if onsite                                    # 対角: t は実
                        _parton_w_rank1!(W, Uu, Uo, t, e.site1, e.site1, n_un, n_elec)
                    else
                        c = part == 1 ? t : im * t
                        _parton_w_rank1!(W, Uu, Uo, c, e.site1, e.site2, n_un, n_elec)
                        _parton_w_rank1!(W, Uu, Uo, conj(c), e.site2, e.site1,
                                         n_un, n_elec)
                    end
                end

                # 分母 D[u, n] = ε_n − ε_{Ne+u} をインプレースで割る(バッファ不要)。
                # |D| < gap_tol は 0(参照互換の clamp。上の docstring 参照)
                @inbounds for n = 1:n_elec, u = 1:n_un
                    d = ev[n] - ev[n_elec + u]
                    W[u, n] = abs(d) < gap_tol ? zero(ComplexF64) : W[u, n] / d
                end
                mul!(dPhi, Uu, W)                 # ∂Φ = U_unocc (W ./ D)
            end
        end
    end
    return nothing
end

"""
    _parton_w_rank1!(W, Uu, Uo, a, s1, s2, n_un, n_elec)

`W += a · Uu'[:, s1] ⊗ Uo[s2, :]`(rank-1 更新)。`Uu'[:, s1][u] = conj(Uu[s1, u])`
の随伴(共役)は契約 0′ では正しい(DESIGN §1.5 / §7 — ブラとの真の内積。
振幅側の転置積規則と混同しないこと)。
"""
@inline function _parton_w_rank1!(W, Uu, Uo, a::ComplexF64, s1::Int, s2::Int,
                                  n_un::Int, n_elec::Int)
    @inbounds for n = 1:n_elec
        an = a * Uo[s2, n]
        @simd for u = 1:n_un
            W[u, n] += an * conj(Uu[s1, u])
        end
    end
    return nothing
end

