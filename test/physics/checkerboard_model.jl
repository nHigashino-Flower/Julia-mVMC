"""
checkerboard 模型の格子・ボンド・ホッピング係数(外部 ED 実装の規約の移植)
--- parton-mode (fork addition) ---

DESIGN_parton.md §8 P 層。フィクスチャ(pmftrans / physhop / coulombinter)を
組むには ED と同じ t_ij が要るので、`ModuleParentHamCB.jl` の規約をここに移植する。

移植が正しいことは P1(`test_p1_onebody.jl`)が外部 ED 実装から吸い出した
1 粒子ハミルトニアンとの 1e-10 一致で保証する。ここを自分で「それらしく」
埋めないこと — 規約差は静かに間違ったエネルギーを出す。

## 格子とサイト番号(ED: trans_isite_cb)

倍密グリッド Lx = 2Nx, Ly = 2Ny の上で、(x, y) の偶奇が揃う点だけがサイト:

    y が偶数 → x も偶数(副格子 A)
    y が奇数 → x も奇数(副格子 B)

    isite = Nx*y + x/2         (y 偶)
    isite = Nx*y + (x-1)/2     (y 奇)

## ホッピング(ED: transtion_value、φ=ξ=η=0 なのでゲージ因子 J = 1)

始点 (x0, y0) から変位 (dx, dy) への有向ホップ。副格子指標は**始点**で取る:

    bi = +1 (y0 偶) / -1 (y0 奇)

    |dx|=|dy|=1 : z = (1 + dx*dy*bi)/2 ∈ {0,1};  t * exp(i*psi*z) * exp(-i*psi*(1-z))
    (|dx|,|dy|) が (2,0) か (0,2) : z = (1+bi)/2;
                (t1*|dy|/2 + t2*|dx|/2)*z + (t1*|dx|/2 + t2*|dy|/2)*(1-z)
    |dx|=|dy|=2 : t3

ED のハミルトニアンは H = Σ t_ij c†_j c_i(符号反転なし)。mVMC の trans.def が
既定で -1 を掛ける規約とは別物なので、新設形式の pmftrans / physhop には
ここで得た t_ij を**そのまま**入れる。
"""

"checkerboard 模型のホッピング係数一式。ED ログの値と 1 対 1 に対応する。"
struct CheckerboardParams
    t::Float64
    t1::Float64
    t2::Float64
    t3::Float64
    psi::Float64
end

CheckerboardParams(; t = 1.0, t1 = 0.2928932188134525, t2 = -0.2928932188134525,
                   t3 = 0.20710678118654754, psi = 0.7853981633974483) =
    CheckerboardParams(t, t1, t2, t3, psi)

"サイト番号(0-based)→ 倍密グリッド座標 (x, y)。"
function cb_site_to_xy(isite::Int, nx::Int)
    y = div(isite, nx)
    col = isite - nx * y
    x = iseven(y) ? 2 * col : 2 * col + 1
    return x, y
end

"倍密グリッド座標 (x, y) → サイト番号(0-based)。偶奇が揃っていること。"
function cb_xy_to_site(x::Int, y::Int, nx::Int)
    (x % 2) == (y % 2) || error("($x, $y) is not a checkerboard site")
    return iseven(y) ? nx * y + div(x, 2) : nx * y + div(x - 1, 2)
end

"隣接の変位ベクトル: d² = 2(最近接)、4(2 次)、8(3 次)。"
const CB_HOPS = [
    (dx, dy) for dx = -2:2, dy = -2:2 if (dx^2 + dy^2) in (2, 4, 8)
]

"""
    cb_hopping(p, dx, dy, y0) -> ComplexF64

始点の行 y0 から変位 (dx, dy) へのホッピング係数(ED の transtion_value)。
"""
function cb_hopping(p::CheckerboardParams, dx::Int, dy::Int, y0::Int)
    bi = iseven(y0) ? 1 : -1
    d = dx^2 + dy^2
    if d == 2
        z = div(1 + dx * dy * bi, 2)
        (z == 0 || z == 1) || error("z must be 0 or 1, got $z")
        return p.t * exp(im * p.psi * z) * exp(-im * p.psi * (1 - z))
    elseif d == 4
        z = div(1 + bi, 2)
        return ComplexF64(
            (p.t1 * abs(dy) / 2 + p.t2 * abs(dx) / 2) * z +
            (p.t1 * abs(dx) / 2 + p.t2 * abs(dy) / 2) * (1 - z),
        )
    elseif d == 8
        return ComplexF64(p.t3)
    end
    error("(dx, dy) = ($dx, $dy) is not a checkerboard bond")
end

"""
    cb_directed_hops(nx, ny) -> Vector{NTuple{5,Int}}

全ての有向ホップ `(i, j, dx, dy, d²)`(0-based サイト、周期境界で巻く)。
各サイトから 12 本ずつ出るので長さは 12 * Nsite。
"""
function cb_directed_hops(nx::Int, ny::Int)
    lx, ly = 2 * nx, 2 * ny
    nsite = 2 * nx * ny
    out = NTuple{5,Int}[]
    for i = 0:(nsite - 1)
        x0, y0 = cb_site_to_xy(i, nx)
        for (dx, dy) in CB_HOPS
            x1 = mod(x0 + dx, lx)
            y1 = mod(y0 + dy, ly)
            j = cb_xy_to_site(x1, y1, nx)
            push!(out, (i, j, dx, dy, dx^2 + dy^2))
        end
    end
    return out
end

"""
    cb_onebody(nx, ny, p) -> Matrix{ComplexF64}

1 粒子セクターのハミルトニアン H[j+1, i+1] = t(i→j)。P1 の照合対象。
"""
function cb_onebody(nx::Int, ny::Int, p::CheckerboardParams = CheckerboardParams())
    nsite = 2 * nx * ny
    H = zeros(ComplexF64, nsite, nsite)
    for (i, j, dx, dy, _) in cb_directed_hops(nx, ny)
        _, y0 = cb_site_to_xy(i, nx)
        H[j + 1, i + 1] += cb_hopping(p, dx, dy, y0)
    end
    return H
end

"""
    cb_undirected_bonds(nx, ny) -> Vector{NTuple{5,Int}}

無向ボンドを 1 本ずつ `(i, j, dx, dy, d²)` で返す(i < j となる向きを採用し、
変位は i → j のもの)。physhop / coulombinter は片方向列挙なのでこれを使う。
"""
function cb_undirected_bonds(nx::Int, ny::Int)
    seen = Set{NTuple{2,Int}}()
    out = NTuple{5,Int}[]
    for (i, j, dx, dy, d2) in cb_directed_hops(nx, ny)
        i == j && error("self loop at site $i: lattice too small for this hop set")
        key = minmax(i, j)
        key in seen && continue
        push!(seen, key)
        if i < j
            push!(out, (i, j, dx, dy, d2))
        else
            push!(out, (j, i, -dx, -dy, d2))
        end
    end
    return out
end

"""
    cb_translations(nx, ny) -> (maps, ucs)

基本セル並進が誘導するサイト置換。`maps[k][j+1]` が並進後のサイト、
`ucs[k] = (ucx, ucy)`(いずれも 0-based)。恒等 (0,0) が 1 本目に来る。

変位は倍密グリッド上で `(dx, dy) = (2·ucx, 2·ucy)`。**偶数変位**なので (x, y) の
偶奇が保たれ、副格子を混ぜない — 奇数変位は副格子を入れ替えるので `cb_hopping`
の `bi`(始点の行の偶奇)が反転し、一般には対称性にならない。参照実装
`CheckerBoard.jl` の `build_QNPTransSiteList`(`dx, dy = 2*ucx, 2*ucy`)と同じ
取り方。これが cb 模型の 1 体項の対称性であることは P2-2 で機械検証する。
"""
function cb_translations(nx::Int, ny::Int)
    lx, ly = 2 * nx, 2 * ny
    nsite = 2 * nx * ny
    maps = Vector{Int}[]
    ucs = Tuple{Int,Int}[]
    for ucy = 0:(ny - 1), ucx = 0:(nx - 1)
        m = Vector{Int}(undef, nsite)
        for j = 0:(nsite - 1)
            x, y = cb_site_to_xy(j, nx)
            m[j + 1] = cb_xy_to_site(mod(x + 2 * ucx, lx), mod(y + 2 * ucy, ly), nx)
        end
        push!(maps, m)
        push!(ucs, (ucx, ucy))
    end
    return maps, ucs
end

"""
    cb_qp_translations(nx, ny, kext) -> (maps, ucs)

**参照実装 `make_QNPidx` 準拠**の QP 並進(パートン v2 の実際の QP 構成)。

アンザッツが x 方向に `kext` セルの拡大周期を持つとき、破られている並進の剰余類
`Z_kext` の代表 `(ucx, 0), ucx = 0 … kext-1` の **kext 本だけ**を採る。

`cb_translations`(全並進 nx·ny 本)との違いは意図的:

- y 方向の並進はアンザッツが破っていないので入れない。保たれている対称性まで
  射影すると、変分空間を余計に狭めるだけでエネルギーは下がらない
- x 方向も `T_x^kext` は保たれているので、剰余類の代表 kext 本で足りる

参照実装で確認した対応(`make_QNPidx` をそのまま実行して照合):
`Nux=Nuy=4, Nsite=32` のとき `K(puc)=2 → NQPTrans=2, uclist=[(0,0),(1,0)]`、
`K=4 → NQPTrans=4, uclist=[(0,0),(1,0),(2,0),(3,0)]`。
"""
function cb_qp_translations(nx::Int, ny::Int, kext::Int)
    nx % kext == 0 ||
        error("nx = $nx must be divisible by the enlarged-cell factor kext = $kext")
    lx, ly = 2 * nx, 2 * ny
    nsite = 2 * nx * ny
    maps = Vector{Int}[]
    ucs = Tuple{Int,Int}[]
    for ucx = 0:(kext - 1)
        m = Vector{Int}(undef, nsite)
        for j = 0:(nsite - 1)
            x, y = cb_site_to_xy(j, nx)
            m[j + 1] = cb_xy_to_site(mod(x + 2 * ucx, lx), y, nx)
        end
        push!(maps, m)
        push!(ucs, (ucx, 0))
    end
    return maps, ucs
end
