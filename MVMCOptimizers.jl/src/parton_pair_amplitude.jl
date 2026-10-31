"""
Pfaffian パートンモードの振幅: 契約 1(厳密再計算)・ip・契約 2(比)・契約 3(更新)
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §1.5・§4.3。既存の det 版と同じ関数名に Pf 版のメソッドを足すだけで、
骨格(`parton_make_sample!` / `parton_local_energy` / `parton_main_cal!`)からは
`pstate.amp` の型でディスパッチされる。

粒子 m の移動で変わる行・列の集合は S = {(m−1)K + a : a = 1..K}。qp ごとに

    D[J, a] = Δ[S_a, J](J ∉ S)、Δ[S_a, J]/2(J ∈ S)       ⇒  Δ = E Dᵀ − D Eᵀ
    GB      = [G[:, S], G·D]                                  (M×2K、コストの主項 O(K·M²))
    X       = C⁻¹ + Bᵀ G B = C⁻¹ + [GB[S, :]; Dᵀ·GB]           (2K×2K、反対称)
    R_qp    = Pf(X) / Pf(C⁻¹)
    G′      = G + GB · X⁻¹ · GBᵀ                               (Bᵀ G = −(GB)ᵀ を使う)

振幅の縮約は転置積で書き、`dot()` は使わない(第一引数を共役するため。本体 DESIGN §7)。
"""

"""
    _parton_pair_gather!(F, src, row_site, qmap, qsgn, n_flavor, n_elec)

F_qp[I, J] = s(r_I)·s(r_J)·f[(π(r_I), a_I), (π(r_J), a_J)] を組む(I = (m−1)K + a)。
"""
function _parton_pair_gather!(F::AbstractMatrix{ComplexF64}, src::PartonPairOrbital,
                              row_site::Vector{Int}, qmap::Vector{Int}, qsgn::Vector{Int},
                              n_flavor::Int, n_elec::Int)
    ns = src.n_site
    P = src.pair_matrix
    @inbounds for mj = 1:n_elec
        rj = row_site[mj]
        tj = qmap[rj]
        sj = qsgn[rj]
        for b = 1:n_flavor
            J = (mj - 1) * n_flavor + b
            βs = parton_pair_state(ns, tj, b)
            for mi = 1:n_elec
                ri = row_site[mi]
                ti = qmap[ri]
                sij = qsgn[ri] * sj
                for a = 1:n_flavor
                    F[(mi - 1) * n_flavor + a, J] = sij * P[parton_pair_state(ns, ti, a), βs]
                end
            end
        end
    end
    return F
end

"""
    _parton_pair_pfaffian!(A, tau) -> ComplexF64

反対称行列 A の Pf を Parlett–Reid(列ピボットつき)で計算する。A は壊す。`tau` は長さ n 以上の
作業領域。割り当てなし。

PfaPack の `julia_zsktf2!` は使わない: フレーバーごとのブロックが交互に並ぶ形の**正則な**行列で
ゼロピボット(info > 0)を返すことを P1 で確かめた(DESIGN_parton_pf.md §3)。同フレーバーだけを
組んだ f(F の行が粒子が主・フレーバーが従なので交互配置になる)や、契約 2 の X ≈ C⁻¹ がまさにこの形で、
そのまま使うと正則な配置を Pf = 0 と誤り、受理すべき移動を黙って棄却する。
"""
function _parton_pair_pfaffian!(A::AbstractMatrix{ComplexF64}, tau::AbstractVector{ComplexF64})
    n = size(A, 1)
    isodd(n) && return zero(ComplexF64)
    pf = one(ComplexF64)
    @inbounds for k = 1:2:(n - 1)
        # ピボット: 第 k 列の k+1 行目以降で絶対値が最大の行を k+1 へ(行と列を同時に入れ替える)
        kp = k + 1
        amax = abs(A[k + 1, k])
        for i = (k + 2):n
            v = abs(A[i, k])
            if v > amax
                amax = v
                kp = i
            end
        end
        if kp != k + 1
            for j = 1:n
                A[k + 1, j], A[kp, j] = A[kp, j], A[k + 1, j]
            end
            for i = 1:n
                A[i, k + 1], A[i, kp] = A[i, kp], A[i, k + 1]
            end
            pf = -pf
        end
        piv = A[k, k + 1]
        iszero(piv) && return zero(ComplexF64)
        pf *= piv
        k + 2 <= n || continue
        # 反対称ランク 2 更新: A[i, j] += τ_i·A[j, k+1] − A[i, k+1]·τ_j(i, j ≥ k+2)、τ_i = A[k, i]/piv
        for i = (k + 2):n
            tau[i] = A[k, i] / piv
        end
        for j = (k + 2):n
            tj = tau[j]
            ajk = A[j, k + 1]
            for i = (k + 2):n
                A[i, j] += tau[i] * ajk - A[i, k + 1] * tj
            end
        end
    end
    return pf
end

"""
    parton_recompute_amplitude_all!(amp::PartonPairAmplitudeData, src::PartonPairOrbital, cfg, data, ws)

契約 1(Pf 版): 全 qp の F を組み、Pf(Parlett–Reid、`_parton_pair_pfaffian!`)と G = F⁻¹(LU)を
得る。G は最後に厳密に反対称へ射影する(契約 3 の更新式の前提。`_parton_pair_antisymmetrize!`)。
特異(Pf が 0 か非有限、または LU が特異)なら Pf = 0・G = 0 とする(det モードの「特異 = det 0 =
ノード」と同じ扱い。G を 0 にしておけば比が NaN にならない)。`row_site` もここで cfg から確定する。

PfaPack の LTL 部品(`julia_zsktf2!` → `utu2pfa` → `utu2inv!`)は、交互ブロック構造の正則行列で
ゼロピボットを返すので使わない(`_parton_pair_pfaffian!` の docstring)。
"""
function parton_recompute_amplitude_all!(
    amp::PartonPairAmplitudeData,
    src::PartonPairOrbital,
    cfg::PartonConfiguration,
    data::ExpertModeData,
    ws::PartonPairSamplingWorkspace,
)
    K = amp.n_flavor
    N = amp.n_elec
    @inbounds for m = 1:N
        amp.row_site[m] = particle_site(cfg, 1, m)
    end
    F = ws.f_scratch
    for qp = 1:amp.n_qp
        _parton_pair_gather!(F, src, amp.row_site, data.qp_trans[qp], data.qp_trans_sgn[qp], K, N)
        G = pair_inv_block(amp, qp)
        copyto!(ws.lu_scratch, F)                          # LU 用の写し(Pf の計算が F を壊す)
        pf = _parton_pair_pfaffian!(F, ws.tau)
        lu = (iszero(pf) || !isfinite(pf)) ? nothing : lu!(ws.lu_scratch; check = false)
        if lu === nothing || !issuccess(lu)
            amp.pf_f[qp] = zero(ComplexF64)
            fill!(G, zero(ComplexF64))
            continue
        end
        copyto!(G, LinearAlgebra.inv!(lu))
        _parton_pair_antisymmetrize!(G)
        amp.pf_f[qp] = pf
    end
    return nothing
end

"ip = Σ_qp w_qp · Pf F_qp"
function parton_calculate_ip(amp::PartonPairAmplitudeData, qp_weight)
    ip = zero(ComplexF64)
    for qp = 1:amp.n_qp
        ip += qp_weight[qp] * amp.pf_f[qp]
    end
    return ip
end

"正方行列 A を反対称部分 (A − Aᵀ)/2 で置き換える(A[j,i] = −A[i,j] が浮動小数点で厳密に成り立つ)。"
function _parton_pair_antisymmetrize!(A::AbstractMatrix{ComplexF64})
    n = size(A, 1)
    @inbounds for j = 1:n
        A[j, j] = zero(ComplexF64)
        for i = (j + 1):n
            g = (A[i, j] - A[j, i]) / 2
            A[i, j] = g
            A[j, i] = -g
        end
    end
    return A
end

"2K×2K の反対称行列 X の Pf(作業領域の写しで Parlett–Reid。X は壊さない)。"
function _parton_pair_small_pfaffian!(ws::PartonPairSamplingWorkspace, X::Matrix{ComplexF64})
    A = ws.x_scratch
    copyto!(A, X)
    return _parton_pair_pfaffian!(A, ws.tau_small)
end

"""
    parton_amplitude_ratio!(ws::PartonPairSamplingWorkspace, amp::PartonPairAmplitudeData, src, data, qp_weight, m, r_new) -> (ratio, ip_new)

契約 2(Pf 版): 粒子 m を r_new へ動かしたときの振幅比。O(n_qp·K·M²)。

**純粋**(ws だけに書く)。E_loc の仮想移動で何度呼ばれても配置も振幅も動かない。
qp ごとの GB・X・R_qp を ws に残し、受理時に契約 3 が再利用する。
"""
function parton_amplitude_ratio!(
    ws::PartonPairSamplingWorkspace,
    amp::PartonPairAmplitudeData,
    src::PartonPairOrbital,
    data::ExpertModeData,
    qp_weight,
    m::Int,
    r_new::Int,
)
    K = amp.n_flavor
    N = amp.n_elec
    ns = src.n_site
    P = src.pair_matrix
    D = ws.d_buf
    r_old = amp.row_site[m]
    row0 = (m - 1) * K
    pf_cinv = parton_pair_pf_cinv(K)
    ip_old = zero(ComplexF64)
    ip_new = zero(ComplexF64)
    for qp = 1:amp.n_qp
        qmap = data.qp_trans[qp]
        qsgn = data.qp_trans_sgn[qp]
        tn = qmap[r_new]
        sn = qsgn[r_new]
        to = qmap[r_old]
        so = qsgn[r_old]

        # D: 変わる行(S)の新旧の差。S × S の部分は 1/2(Δ = E Dᵀ − D Eᵀ が二重に数えるため)。
        # S × S では行・列とも同じサイトなので符号は s² = 1 で消える
        @inbounds for mj = 1:N
            if mj == m
                for b = 1:K
                    J = row0 + b
                    βn = parton_pair_state(ns, tn, b)
                    βo = parton_pair_state(ns, to, b)
                    for a = 1:K
                        D[J, a] = (P[parton_pair_state(ns, tn, a), βn] -
                                   P[parton_pair_state(ns, to, a), βo]) / 2
                    end
                end
            else
                rj = amp.row_site[mj]
                tj = qmap[rj]
                sj = qsgn[rj]
                for b = 1:K
                    J = (mj - 1) * K + b
                    βs = parton_pair_state(ns, tj, b)
                    for a = 1:K
                        D[J, a] = sj * (sn * P[parton_pair_state(ns, tn, a), βs] -
                                        so * P[parton_pair_state(ns, to, a), βs])
                    end
                end
            end
        end

        G = pair_inv_block(amp, qp)
        GB = ws.gb_blocks[qp]
        @inbounds for a = 1:K
            copyto!(view(GB, :, a), view(G, :, row0 + a))      # G·E = G[:, S]
        end
        mul!(view(GB, :, (K + 1):(2K)), G, D)                  # G·D(主項)

        X = ws.x_blocks[qp]
        @inbounds for c = 1:(2K), a = 1:K
            X[a, c] = GB[row0 + a, c]                           # Eᵀ·GB = GB[S, :]
        end
        mul!(view(X, (K + 1):(2K), :), transpose(D), GB)       # Dᵀ·GB(転置積。随伴ではない)
        @inbounds for a = 1:K                                   # + C⁻¹ = [[0, −I], [I, 0]]
            X[a, K + a] -= 1
            X[K + a, a] += 1
        end

        R = _parton_pair_small_pfaffian!(ws, X) * pf_cinv
        ws.ratio_blocks[qp] = R
        w = qp_weight[qp]
        ip_old += w * amp.pf_f[qp]
        ip_new += w * amp.pf_f[qp] * R
    end
    return ip_new / ip_old, ip_new
end

"""
    parton_update_amplitude!(amp::PartonPairAmplitudeData, src, data, ws, m, r_new; ratio_floor=1e-12) -> Symbol

契約 3(Pf 版): 受理した移動を全 qp へ反映する。G′ = G + GB·X⁻¹·GBᵀ、Pf *= R_qp。O(n_qp·K·M²)。

直前の契約 2 が同じ (m, r_new) で呼ばれていること(ws の GB・X・R の再利用)が前提。
配置側の更新は `parton_update_ele_config!` の担当。

**更新のたびに G を厳密に反対称へ射影する**(`_parton_pair_antisymmetrize!`、O(M²))。
更新式は BᵀG = −(GB)ᵀ(G の反対称性)を使っているので、丸め誤差で G に乗った対称成分は
X を通って U·sym(X⁻¹)·Uᵀ として戻り、**受理のたびに等比的に増幅される**(実測: 1 回ごとに
約 10 倍、40 回で G の相対誤差 1e6・Pf が NaN)。射影すると 1e−14 前後で止まる(T1-d)。
上流の 1 電子更新 `InvM[i][j] += v1[i]v2[j] − v1[j]v2[i]` が構造的に反対称な形をとるのと同じ理由。

次のどれかなら `:need_recompute` を返す(呼び出し側がコミット済みの配置から錨を打ち直すので、
途中まで更新したまま抜けても安全):
- `|R_qp| < ratio_floor`(ノード踏み抜き。det モードと同じ)
- X の LU が特異(|R_qp| が下限を超えていても、ほぼ特異な f では起きる。例外は投げない)
- 錨で特異だった qp がある(Pf = 0)。乗法更新では 0 から戻れないので、移動後に正則に
  なっていても取りこぼさないよう、そのつど組み直す(det モードは 0 のまま進む)
"""
function parton_update_amplitude!(
    amp::PartonPairAmplitudeData,
    src::PartonPairOrbital,
    data::ExpertModeData,
    ws::PartonPairSamplingWorkspace,
    m::Int,
    r_new::Int;
    ratio_floor::Float64 = 1e-12,
)
    for qp = 1:amp.n_qp
        R = ws.ratio_blocks[qp]
        (iszero(amp.pf_f[qp]) || abs(R) < ratio_floor) && return :need_recompute
        X = ws.x_blocks[qp]
        # |R| が下限を超えていても X が数値的に特異になりうる(ほぼ特異な f の配置)。lu! の既定の
        # 検査は SingularException を投げて run を止めるので、検査を外して再計算を求める
        Xlu = lu!(X; check = false)                           # 2K×2K(X は壊してよい)
        issuccess(Xlu) || return :need_recompute
        Xinv = LinearAlgebra.inv!(Xlu)
        mul!(ws.y_buf, ws.gb_blocks[qp], Xinv)
        G = pair_inv_block(amp, qp)
        mul!(G, ws.y_buf, transpose(ws.gb_blocks[qp]), true, true)   # G += Y·GBᵀ
        _parton_pair_antisymmetrize!(G)
        amp.pf_f[qp] *= R
    end
    amp.row_site[m] = r_new
    return :ok
end
