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
function parton_update_orbital_derivatives!(mfham::PartonMFHamiltonian, n_elec::Int)
    n_site = size(mfham.h_mf[1], 1)
    n_un = n_site - n_elec
    for f in eachindex(mfham.h_mf)
        U = mfham.eig_vecs[f]
        ev = mfham.eig_vals[f]
        Uo = @view U[:, 1:n_elec]
        Uu = @view U[:, (n_elec + 1):n_site]
        # 分母 D[u, n] = ε_n − ε_{Ne+u}(u = 非占有, n = 占有)
        D = [ev[n] - ev[n_elec + u] for u = 1:n_un, n = 1:n_elec]

        for k = 1:mfham.n_idx
            onsite = mfham.is_onsite_group[k]
            for part = 1:2                        # 1 = Re, 2 = Im
                dof = 2 * (k - 1) + part
                dPhi = mfham.dorbitals[f][dof]
                if onsite && part == 2
                    fill!(dPhi, 0)                # 門番が凍結する成分
                    continue
                end

                # dHUo = (∂θ H^(f)) · U_occ をテンプレートのスパース走査で構築
                dHUo = mfham.dh_uo_scratch
                fill!(dHUo, 0)
                for e in mfham.template[k]
                    e.flavor == f || continue
                    t = e.coeff
                    if onsite                                    # 対角: t は実
                        @views dHUo[e.site1, :] .+= t .* Uo[e.site1, :]
                    elseif part == 1                             # T + T†
                        @views dHUo[e.site1, :] .+= t .* Uo[e.site2, :]
                        @views dHUo[e.site2, :] .+= conj(t) .* Uo[e.site1, :]
                    else                                         # i (T − T†)
                        c = im * t
                        @views dHUo[e.site1, :] .+= c .* Uo[e.site2, :]
                        @views dHUo[e.site2, :] .+= conj(c) .* Uo[e.site1, :]
                    end
                end

                W = Uu' * dHUo                    # (n_un × n_elec)。随伴で正しい
                mul!(dPhi, Uu, W ./ D)            # ∂Φ = U_unocc (W ./ D)
            end
        end
    end
    return nothing
end
