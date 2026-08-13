
#コメントは自分の理解のために書いてる、後で消すかも
function parton_build_mf_templates!(mfham, data::ExpertModeData)
    #　注記：現在のkind = site1 == site2 ? :onsite : :hoppingという分類はflavor1==flavor2の門番があるから正しい式です(混成を許すと「サイト対角・フレーバー非対角」の項が現れ、
    #　     それは対角要素ではなくh.c.が要る非対角要素になる)。将来ゲートを外すときは分類条件がsite1==site2 && flavor1==flavor2に変わる
    # (i,j,f) → idx の結合表。ローカル・使い捨て(0-basedのまま), i,j はサイトインデックス, f はフレーバーインデックス
    idx_of = Dict{NTuple{3,Int},Int}()
    for p in data.pmfpara_terms
        key = (p.site1, p.site2, p.flavor1)
        haskey(idx_of, key) && error("pmfpara: duplicate entry $key")
        idx_of[key] = p.idx
    end
    n_idx = maximum(values(idx_of)) + 1        # 0..n_idx-1 の連番性もここで検査

    template   = [PartonMFTemplateEntry[] for _ in 1:n_idx]
    group_kind = fill(:unset, n_idx)           # :onsite / :hopping
    matched    = Set{NTuple{3,Int}}()

    for t in data.pmftrans_terms  # 結合を要する検証(混在禁止・逆向き重複・完全性)はここが家
        key = (t.site1, t.site2, t.flavor1)
        key in matched && error("pmftrans: duplicate $key")
        t.site1 != t.site2 && (t.site2, t.site1, t.flavor1) in matched &&
            error("pmftrans: both directions of $key listed — list once, h.c. is implicit") 
        push!(matched, key)

        haskey(idx_of, key) ||
            error("pmftrans: $key has no pmfpara idx — fixed terms are expressed as OptFlag-frozen α")
        k = idx_of[key] + 1                    # ★ 0-based→1-based 変換はこの一行だけ

        kind = t.site1 == t.site2 ? :onsite : :hopping
        kind === :onsite && abs(imag(t.value)) > 1e-12 && error("pmftrans: onsite t must be real")
        group_kind[k] === :unset ? (group_kind[k] = kind) :
            group_kind[k] === kind || error("idx $(k-1): onsite and hopping mixed in one group")

        push!(template[k], PartonMFTemplateEntry(t.site1+1, t.site2+1, t.flavor+1, t.value))
    end
    length(matched) == length(idx_of) ||
        error("pmfpara: some idx cells have no matching pmftrans term")   # 逆方向の完全性
    # 各フレーバーに項が1つもない、も検査対象(空Hのeigenは縮退ゴミを返す)

    mfham.template        = template
    mfham.is_onsite_group = group_kind .== :onsite
    mfham.n_idx           = n_idx
    return mfham
end



function parton_update_orbitals!(mfham, alpha::Vector{ComplexF64}, n_elec::Int;
                                 gap_tol = 1e-8)
    foreach(H -> fill!(H, 0), mfham.h_mf)
    for k in 1:mfham.n_idx
        onsite = mfham.is_onsite_group[k]
        for e in mfham.template[k]
            v = alpha[k] * e.coeff
            H = mfham.h_mf[e.flavor]
            H[e.site1, e.site2] += v
            onsite || (H[e.site2, e.site1] += conj(v))   # h.c.はホッピングのみ
        end
    end
    mfham.min_gap = Inf
    for f in eachindex(mfham.h_mf)
        F = eigen(Hermitian(mfham.h_mf[f]))              # zheev経路・意図の宣言
        mfham.eig_vals[f] = F.values
        mfham.eig_vecs[f] = F.vectors                    # 摂動論用に全固有対を保持
        mfham.orbitals[f] .= @view F.vectors[:, 1:n_elec]
        mfham.min_gap = min(mfham.min_gap, F.values[n_elec+1] - F.values[n_elec])
    end
    mfham.min_gap < gap_tol &&
        @warn "MF spectrum nearly degenerate at Fermi level" min_gap = mfham.min_gap
    return mfham
end