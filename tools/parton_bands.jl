#=
パートン平均場のバンド解析ライブラリ
--- parton-mode (fork addition) ---

`zqp_pmfham_opt.dat`(フレーバーごとの実空間 H_MF、DESIGN §3.3.1 の密な行形式)を
読み、CheckerBoard の拡大セル (ex, ey) でブロッホ化してバンドと Chern 数を出す。
本体パッケージからは独立(依存は tools/Project.toml に隔離)。

規約(参照VMC ~/VMC/src/Quantity/Topology/{Topology,PartonChern}.jl と同一):
- サイト → 倍密グリッド: `cb_site_to_xy`(test/physics/checkerboard_model.jl と同一)
- 基本セル: (ucx, ucy, bi) = (x÷2, y÷2, y mod 2)。セルあたり 2 サイト(副格子 A/B)
- 拡大セル (ex, ey): スーパーセル座標 (UCX, UCY) = (ucx÷ex, ucy÷ey)、
  内部軌道 α = (mod(ucy,ey)·ex + mod(ucx,ex))·2 + bi(y-major。参照VMC
  `uc_rectangular` と同一)。軌道数 n_orb = 2·ex·ey
- Bloch 変換の符号: H(k)_{αβ} = (1/N) Σ_{ij} e^{i k·(R_i − R_j)} H_{ij}
  (参照VMC `bloch_basis_dagger_site` の p(k) = b P b† と同じ符号。
  ホッピング表現では H(k) = Σ_{ΔR} h(ΔR) e^{−i k·ΔR}、ΔR = R_j − R_i)

2 経路の Chern を持つ:
- `fhs_chern_bands`: H(k) を任意グリッドで対角化して最低 n_occ バンド多様体の
  FHS Chern。グリッドを細かくして収束を確かめられる。バンドを 1 本ずつ渡せば
  バンド分解 Chern も出る
- `fhs_chern_projector`: 実空間 H の占有固有ベクトルから P = Φ Φ† を作り、
  Bloch 射影 p(k) = b P b† の上位 n_occ フレームで FHS(参照VMC
  `fhs_chern_parton` と同じ構成)。系サイズの native グリッド限定だが、
  **実際の占有集合**(mom 追跡で最低バンドからずれていても)をそのまま反映する
=#

module PartonBands

using LinearAlgebra

export read_pmfham, read_pmfband, read_pmfocc,
       site_alpha, fold_hoppings, hk, bands_on_path, bz_path,
       fhs_chern_bands, fhs_chern_projector, native_grid_eigs,
       bott_index, polar_unitary

# ---------------------------------------------------------------------------
# CheckerBoard 幾何(参照VMC / test/physics と同一規約)
# ---------------------------------------------------------------------------

"サイト番号(0-based)→ 倍密グリッド座標 (x, y)。"
function cb_site_to_xy(isite::Int, nx::Int)
    y = div(isite, nx)
    col = isite - nx * y
    x = iseven(y) ? 2 * col : 2 * col + 1
    return x, y
end

"サイト → (ucx, ucy, bi)。基本セル座標と副格子。"
function site_uc(isite::Int, nx::Int)
    x, y = cb_site_to_xy(isite, nx)
    return div(x, 2), div(y, 2), mod(y, 2)
end

"""
    site_alpha(isite, nx, ex, ey) -> (UCX, UCY, α)

サイト → スーパーセル座標 (UCX, UCY) と内部軌道 α(0-based)。
α の採番は参照VMC `uc_rectangular` と同一(y-major、副格子が最下位)。
"""
function site_alpha(isite::Int, nx::Int, ex::Int, ey::Int)
    ucx, ucy, bi = site_uc(isite, nx)
    cell = mod(ucy, ey) * ex + mod(ucx, ex)
    return div(ucx, ex), div(ucy, ey), cell * 2 + bi
end

# ---------------------------------------------------------------------------
# 出力ファイルの読み込み
# ---------------------------------------------------------------------------

"""
    read_pmfham(path; nsite, nflavor) -> Vector{Matrix{ComplexF64}}

`zqp_pmfham_opt.dat`(5 行ヘッダ + `site1 flavor1 site2 flavor2 ReH ImH`、
全組・0-based・辞書順)からフレーバーごとの Nsite×Nsite 行列を返す。
"""
function read_pmfham(path::AbstractString; nsite::Int, nflavor::Int)
    lines = readlines(path)
    length(lines) >= 5 || error("ヘッダが短すぎます: $path")
    m = match(r"NPmfHam\s+(\d+)", lines[2])
    m === nothing && error("2 行目に NPmfHam がありません: $path")
    ndecl = parse(Int, m.captures[1])
    H = [zeros(ComplexF64, nsite, nsite) for _ in 1:nflavor]
    nread = 0
    for ln in lines[6:end]
        isempty(strip(ln)) && continue
        t = split(ln)
        s1 = parse(Int, t[1]); f1 = parse(Int, t[2])
        s2 = parse(Int, t[3]); f2 = parse(Int, t[4])
        f1 == f2 || error("フレーバー非対角成分は想定外です: $ln")
        H[f1+1][s1+1, s2+1] = complex(parse(Float64, t[5]), parse(Float64, t[6]))
        nread += 1
    end
    nread == ndecl || error("NPmfHam=$ndecl に対し $nread 行しか読めませんでした")
    for f in 1:nflavor
        herr = norm(H[f] - H[f]') / max(norm(H[f]), 1e-300)
        herr < 1e-10 || error("flavor $(f-1) の H がエルミートでありません (相対残差 $herr)")
    end
    return H
end

"""
    read_pmfband(path) -> (nflavor, nsite, nelec, eigs, occ)

`zqp_pmfband_opt.dat`(`# key value` ヘッダ + `flavor band_index eigenvalue occupied`)。
eigs[f] は band_index 順(= 昇順)の固有値、occ[f] は占有フラグ(Bool)。
"""
function read_pmfband(path::AbstractString)
    nflavor = nsite = nelec = 0
    eigs = Vector{Vector{Float64}}()
    occ = Vector{Vector{Bool}}()
    for ln in eachline(path)
        s = strip(ln)
        isempty(s) && continue
        if startswith(s, "#")
            m = match(r"NFlavor\s+(\d+)\s+NSite\s+(\d+)\s+NElec\s+(\d+)", s)
            if m !== nothing
                nflavor = parse(Int, m.captures[1])
                nsite = parse(Int, m.captures[2])
                nelec = parse(Int, m.captures[3])
                eigs = [Float64[] for _ in 1:nflavor]
                occ = [Bool[] for _ in 1:nflavor]
            end
            continue
        end
        t = split(s)
        f = parse(Int, t[1]) + 1
        push!(eigs[f], parse(Float64, t[3]))
        push!(occ[f], parse(Int, t[4]) == 1)
    end
    nflavor > 0 || error("ヘッダ行(NFlavor ...)が見つかりません: $path")
    return (nflavor = nflavor, nsite = nsite, nelec = nelec, eigs = eigs, occ = occ)
end

"""
    read_pmfocc(path; nflavor) -> Vector{Vector{Int}}

`zqp_pmfocc_opt.dat`(5 行ヘッダ + `flavor band_index`)。フレーバーごとの
占有バンド番号(0-based)の列。
"""
function read_pmfocc(path::AbstractString; nflavor::Int)
    lines = readlines(path)
    occ = [Int[] for _ in 1:nflavor]
    for ln in lines[6:end]
        isempty(strip(ln)) && continue
        t = split(ln)
        push!(occ[parse(Int, t[1])+1], parse(Int, t[2]))
    end
    return occ
end

# ---------------------------------------------------------------------------
# ホッピングへの畳み込みと H(k)
# ---------------------------------------------------------------------------

"""
    fold_hoppings(H, nx, ny, ex, ey) -> (h, residual)

実空間 H(Nsite×Nsite)をスーパーセル並進で畳み、
h::Dict{(ΔR1,ΔR2) => n_orb×n_orb 行列} を返す。ΔR = R_j − R_i(mod グリッド)。

pmfpara の idx_mode = :orbit_flavor は拡大セル並進で結ばれるボンドに同じ α を
割り当てるので、H_MF はこの並進で厳密に不変のはず。residual は
「同じ (ΔR, α, β) に落ちる成分の最大ばらつき」で、丸め誤差程度でなければ
規約の取り違えを疑う。
"""
function fold_hoppings(H::Matrix{ComplexF64}, nx::Int, ny::Int, ex::Int, ey::Int)
    nsite = size(H, 1)
    mod(nx, ex) == 0 && mod(ny, ey) == 0 ||
        error("(Nux,Nuy)=($nx,$ny) が (ex,ey)=($ex,$ey) で割り切れません")
    n1, n2 = div(nx, ex), div(ny, ey)
    norb = 2 * ex * ey
    ncell = n1 * n2
    ucx = Vector{Int}(undef, nsite); ucy = Vector{Int}(undef, nsite)
    al = Vector{Int}(undef, nsite)
    for s in 0:nsite-1
        ucx[s+1], ucy[s+1], al[s+1] = site_alpha(s, nx, ex, ey)
    end
    acc = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    cnt = Dict{Tuple{Int,Int},Matrix{Int}}()
    residual = 0.0
    first_val = Dict{Tuple{Int,Int,Int,Int},ComplexF64}()
    # ΔR は**最小イメージ**((−N/2, N/2] の代表)に取る。`mod` で [0,N) に
    # ラップすると、格子 k 点では e^{−ikΔR} が ΔR と ΔR±N で同一なので害が無いが、
    # **任意の k で H(k) を組むと位相が別物になる**(トーラスの巻き込みが
    # 連続的な k 依存性を壊す)。実測: 素の CB 模型・下バンド完全充填で
    # 正解 C = −1 に対しラップ版は (1,1) で +1、(2,1) で −3、(2,2) で −7 を返した。
    minimg(d, N) = (r = mod(d, N); r > div(N, 2) ? r - N : r)
    for i in 1:nsite, j in 1:nsite
        dr = (minimg(ucx[j] - ucx[i], n1), minimg(ucy[j] - ucy[i], n2))
        a, b = al[i] + 1, al[j] + 1
        v = H[i, j]
        key4 = (dr[1], dr[2], a, b)
        if haskey(first_val, key4)
            residual = max(residual, abs(v - first_val[key4]))
        else
            first_val[key4] = v
        end
        m = get!(acc, dr) do; zeros(ComplexF64, norb, norb); end
        c = get!(cnt, dr) do; zeros(Int, norb, norb); end
        m[a, b] += v
        c[a, b] += 1
    end
    for (dr, c) in cnt
        all(c .== ncell) || error("畳み込みの重複数が不均一です at ΔR=$dr")
        acc[dr] ./= ncell
    end
    # 最小イメージが一意であること(= ホッピングがトーラスの半周に届いていない)。
    # 届いていると ΔR = N/2 が ±で縮退し、H(k) が一意に決まらない。
    for dr in keys(acc)
        (iseven(n1) && abs(dr[1]) == div(n1, 2) && norm(acc[dr]) > 1e-12) &&
            error("ホッピングが x 方向トーラスの半周 (ΔR=$(dr[1])) に届いています")
        (iseven(n2) && abs(dr[2]) == div(n2, 2) && norm(acc[dr]) > 1e-12) &&
            error("ホッピングが y 方向トーラスの半周 (ΔR=$(dr[2])) に届いています")
    end
    return acc, residual
end

"""
    hk(h, kx, ky) -> Matrix{ComplexF64}

H(k) = Σ_{ΔR} h(ΔR) e^{−i k·ΔR}。k は換算座標(スーパーセル逆格子単位、
第一 BZ が kx, ky ∈ [0, 2π))。
"""
function hk(h::Dict{Tuple{Int,Int},Matrix{ComplexF64}}, kx::Float64, ky::Float64)
    norb = size(first(values(h)), 1)
    out = zeros(ComplexF64, norb, norb)
    for (dr, m) in h
        out .+= m .* exp(-im * (kx * dr[1] + ky * dr[2]))
    end
    return Hermitian(0.5 * (out + out'))
end

# ---------------------------------------------------------------------------
# バンド経路
# ---------------------------------------------------------------------------

"""
    bz_path(; nseg) -> (ks, dist, ticks)

Γ(0,0) → X(π,0) → M(π,π) → Γ の折れ線。換算座標。
戻り値: k 点列、経路弧長(横軸)、(位置, ラベル) の目盛り。
"""
function bz_path(; nseg::Int = 120)
    corners = [(0.0, 0.0), (pi, 0.0), (pi, pi), (0.0, 0.0)]
    labels = ["Γ", "X", "M", "Γ"]
    ks = Tuple{Float64,Float64}[]
    dist = Float64[]
    ticks = Tuple{Float64,String}[(0.0, labels[1])]
    d = 0.0
    for seg in 1:length(corners)-1
        a, b = corners[seg], corners[seg+1]
        len = hypot(b[1] - a[1], b[2] - a[2])
        for t in range(0.0, 1.0; length = nseg + 1)
            (seg > 1 && t == 0.0) && continue   # 角の重複を除く
            push!(ks, (a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])))
            push!(dist, d + t * len)
        end
        d += len
        push!(ticks, (d, labels[seg+1]))
    end
    return ks, dist, ticks
end

"経路上の全バンド。戻り値は (n_k × n_orb) 行列(行 = k 点、列 = バンド昇順)。"
function bands_on_path(h::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
                       ks::Vector{Tuple{Float64,Float64}})
    norb = size(first(values(h)), 1)
    out = Matrix{Float64}(undef, length(ks), norb)
    for (i, k) in enumerate(ks)
        out[i, :] = eigvals(hk(h, k[1], k[2]))
    end
    return out
end

"native グリッド(n1×n2)の全固有値を昇順で返す(pmfband との突き合わせ用)。"
function native_grid_eigs(h::Dict{Tuple{Int,Int},Matrix{ComplexF64}}, n1::Int, n2::Int)
    eigs = Float64[]
    for ix in 0:n1-1, iy in 0:n2-1
        append!(eigs, eigvals(hk(h, 2pi * ix / n1, 2pi * iy / n2)))
    end
    return sort!(eigs)
end

# ---------------------------------------------------------------------------
# FHS Chern(バンド経路: 任意グリッド)
# ---------------------------------------------------------------------------

"""
    fhs_chern_bands(h, bands; n1, n2) -> NamedTuple

H(k) を n1×n2 グリッドで対角化し、`bands`(1-based のバンド番号の集合、
例 [1] や [1,2])が張る多様体の FHS Chern。バンド交差でフレームが飛ぶと
リンク行列式が小さくなるので min |det| を併記する(0 に近ければ
その多様体は孤立しておらず、値は信用できない)。
"""
function fhs_chern_bands(h::Dict{Tuple{Int,Int},Matrix{ComplexF64}},
                         bands::Vector{Int}; n1::Int = 24, n2::Int = 24)
    frames = Matrix{Matrix{ComplexF64}}(undef, n1, n2)
    for ix in 0:n1-1, iy in 0:n2-1
        F = eigen(hk(h, 2pi * ix / n1, 2pi * iy / n2))
        frames[ix+1, iy+1] = F.vectors[:, bands]
    end
    return _fhs_from_frames(frames)
end

function _fhs_from_frames(frames::Matrix{Matrix{ComplexF64}})
    n1, n2 = size(frames)
    ux = zeros(ComplexF64, n1, n2); uy = zeros(ComplexF64, n1, n2)
    min_link = 1.0
    for ix in 1:n1, iy in 1:n2
        detx = det(frames[ix, iy]' * frames[mod1(ix + 1, n1), iy])
        dety = det(frames[ix, iy]' * frames[ix, mod1(iy + 1, n2)])
        min_link = min(min_link, abs(detx), abs(dety))
        ux[ix, iy] = detx / max(abs(detx), 1e-300)
        uy[ix, iy] = dety / max(abs(dety), 1e-300)
    end
    flux = 0.0
    for ix in 1:n1, iy in 1:n2
        plaq = ux[ix, iy] * uy[mod1(ix + 1, n1), iy] /
               (ux[ix, mod1(iy + 1, n2)] * uy[ix, iy])
        flux += angle(plaq)
    end
    return (C = flux / (2pi), min_abs_link_det = min_link, grid = (n1, n2))
end

# ---------------------------------------------------------------------------
# FHS Chern(射影行列経路: native グリッド、参照VMC PartonChern.jl と同構成)
# ---------------------------------------------------------------------------

"""
    fhs_chern_projector(H, occ_bands, nx, ny, ex, ey) -> NamedTuple

実空間 H の占有固有ベクトル(占有集合 `occ_bands`、0-based バンド番号)から
P = Φ Φ† を作り、Bloch 射影 p(k) = b P b† の上位 n_occ 固有ベクトルの
フレームで FHS。native な n1×n2 グリッド限定。占有数が k 点数で割り切れる
必要がある(バンド絶縁体でなければ Chern は定義できない)。
max_projector_error は p(k)² − p(k) の最大ノルム(k 分解できない占有集合だと
大きくなる)。
"""
function fhs_chern_projector(H::Matrix{ComplexF64}, occ_bands::Vector{Int},
                             nx::Int, ny::Int, ex::Int, ey::Int)
    nsite = size(H, 1)
    n1, n2 = div(nx, ex), div(ny, ey)
    nk = n1 * n2
    ne = length(occ_bands)
    mod(ne, nk) == 0 || error("占有数 $ne が k 点数 $nk で割り切れません")
    n_occ = div(ne, nk)
    F = eigen(Hermitian(H))
    Φ = F.vectors[:, occ_bands .+ 1]
    P = Φ * Φ'
    # b(k): (n_orb × Nsite)。参照VMC bloch_basis_dagger_site のサイト版
    norb = 2 * ex * ey
    ucx = Vector{Int}(undef, nsite); ucy = Vector{Int}(undef, nsite)
    al = Vector{Int}(undef, nsite)
    for s in 0:nsite-1
        ucx[s+1], ucy[s+1], al[s+1] = site_alpha(s, nx, ex, ey)
    end
    frames = Matrix{Matrix{ComplexF64}}(undef, n1, n2)
    max_perr = 0.0
    for ix in 0:n1-1, iy in 0:n2-1
        kx, ky = 2pi * ix / n1, 2pi * iy / n2
        b = zeros(ComplexF64, norb, nsite)
        for s in 0:nsite-1
            b[al[s+1]+1, s+1] = exp(im * (kx * ucx[s+1] + ky * ucy[s+1])) / sqrt(nk)
        end
        pk = b * P * b'
        pk = Hermitian(0.5 * (pk + pk'))
        max_perr = max(max_perr, norm(pk * pk - pk))
        E = eigen(pk)
        frames[ix+1, iy+1] = E.vectors[:, end-n_occ+1:end]   # 固有値 ≈ 1 の上位 n_occ
    end
    r = _fhs_from_frames(frames)
    return (C = r.C, min_abs_link_det = r.min_abs_link_det, grid = r.grid,
            n_occ_per_k = n_occ, max_projector_error = max_perr)
end

# ---------------------------------------------------------------------------
# Bott index(実空間、k 空間を使わない独立経路)
# ---------------------------------------------------------------------------

"極分解 mat = U·P のユニタリ因子(参照VMC `polar_unitary` と同一)。最小特異値も返す。"
function polar_unitary(mat::Matrix{ComplexF64})
    F = svd(mat)
    return F.U * F.Vt, minimum(F.S)
end

"""
    bott_index(H, occ_bands, nx, ny, ex, ey) -> NamedTuple

実空間 Bott index。占有射影 P に対しスーパーセル座標の位置位相演算子
X = exp(2πi·UCX/N1)、Y = exp(2πi·UCY/N2) を作り、
Ũ = P X P + (1−P)、Ṽ = P Y P + (1−P) を極分解して
B = −(1/2π) Σ arg eig(Ṽ Ũ Ṽ† Ũ†)。

**符号規約**: 参照VMC `bott_index_parton`(`PartonChern.jl:166`)に合わせ
**−sum** を採る(Nambu 版 `bott_index_uc` とは逆符号)。これで FHS の C と
同符号になる。

診断量: `commutator` は ‖ŨṼ − ṼŨ‖/√dim(ギャップが閉じていると大きくなる)、
`min_singular_*` は極分解の安定性(0 に近いと信用できない)。
"""
function bott_index(H::Matrix{ComplexF64}, occ_bands::Vector{Int},
                    nx::Int, ny::Int, ex::Int, ey::Int)
    nsite = size(H, 1)
    n1, n2 = div(nx, ex), div(ny, ey)
    F = eigen(Hermitian(H))
    Φ = F.vectors[:, occ_bands .+ 1]
    P = Φ * Φ'
    ucx = Vector{Float64}(undef, nsite); ucy = Vector{Float64}(undef, nsite)
    for s in 0:nsite-1
        UCX, UCY, _ = site_alpha(s, nx, ex, ey)
        ucx[s+1] = UCX; ucy[s+1] = UCY
    end
    xop = Diagonal(exp.(2im * pi .* ucx ./ n1))
    yop = Diagonal(exp.(2im * pi .* ucy ./ n2))
    q = Matrix{ComplexF64}(I, nsite, nsite) - P
    u = P * xop * P + q
    v = P * yop * P + q
    comm = norm(u * v - v * u) / sqrt(nsite)
    upol, su = polar_unitary(Matrix(u))
    vpol, sv = polar_unitary(Matrix(v))
    b_vuvu = -sum(angle.(eigvals(vpol * upol * vpol' * upol'))) / (2pi)
    b_uvuv = -sum(angle.(eigvals(upol * vpol * upol' * vpol'))) / (2pi)
    return (B_vuvu = b_vuvu, B_uvuv = b_uvuv, commutator = comm,
            min_singular_x = su, min_singular_y = sv)
end

end # module PartonBands
