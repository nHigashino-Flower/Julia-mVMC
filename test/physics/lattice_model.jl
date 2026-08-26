"""
パートン平均場フィクスチャの模型インターフェース
--- parton-mode (fork addition) ---

`parton_fixture.jl` は長らく checkerboard 模型にべた書きだった。Kapit-Mueller (KM)
模型を足すにあたり、**模型に依存する部分だけ**をここに切り出す。fixture 本体
(向きの正準化・idx 振り分け・自己交換ペアの検出・def の書き出し)は 1 本のまま
共有する — 同じ罠を模型ごとに踏み直さないための構造上の要請。

## 契約(新しい模型を足すときに実装するもの)

| 関数 | 意味 |
|---|---|
| `pl_nsite(m)` | サイト数 |
| `pl_grid(m)` | 座標の周期 `(lx, ly)`。cb は倍密グリッド `(2nx, 2ny)`、KM は `(Lx, Ly)` |
| `pl_cell_step(m)` | 基本セル 1 歩がグリッド上で何マスか。cb `(2, 2)` / KM `(1, 1)` |
| `pl_site_to_xy(m, i)` / `pl_xy_to_site(m, x, y)` | サイト番号 ⇄ グリッド座標(後者は mod を取る) |
| `pl_bonds(m)` | 模型グラフの無向ボンド `(i, j, dx, dy, d²)`(`i < j` 正規化) |
| `pl_all_pairs(m)` | 全無向サイト対(`graph = :full` 用) |
| `pl_hopping(m, i, j, dx, dy)` | **物理**ホッピング係数(有向 i → j) |
| `pl_mf_hopping(m, i, j, dx, dy, F)` | **平均場**ホッピング係数(有向 i → j) |
| `pl_nn_pairs(m)` | 物理ハミルトニアンの密度型を載せるペア |
| `pl_cell_class(m, i, ex, ey)` | 拡大セルの軌道ラベル |
| `pl_physical_shifts(m)` | **物理 H** を保つ並進の `(tx, ty)`(グリッド座標) |
| `pl_mf_shifts(m, ex, ey)` | **平均場**を保つ並進の `(tx, ty)` |

QP 射影に使う並進は「物理の並進群 ÷ 平均場の並進群」の剰余代表として
fixture 側で自動的に決まる(`pl_qp_shifts`)。

## なぜ `pl_physical_shifts` を分けるか

checkerboard は正味フラックスが 0 なので**全並進**が物理 H の対称性だが、
KM は磁気並進しか持たない(Landau ゲージでは `⟨T_x^q, T_y⟩`、q = 1/φ)。
「全並進 ÷ 拡大セル」という cb の既定を KM に流用すると、H を保たない並進まで
射影に入れてしまう。ここを模型の責務にしておけば取り違えない。

## `pl_hopping` が変位ではなく端点を受け取る理由

cb の係数は始点の**行の偶奇**、KM の係数は始点の **x 座標**(Landau ゲージ)で
決まる。どちらも始点のサイト番号から引けるので、端点を渡す形に統一する。
KM はさらに「同じ端点対に落ちる全変位の和」なので、`(dx, dy)` は
クラス分けの代表としてしか使わない。
"""

abstract type PartonLatticeModel end

_pl_todo(name, m) = error("$(typeof(m)) は $name を実装していません。")

pl_nsite(m::PartonLatticeModel) = _pl_todo("pl_nsite", m)
pl_grid(m::PartonLatticeModel) = _pl_todo("pl_grid", m)
pl_cell_step(m::PartonLatticeModel) = _pl_todo("pl_cell_step", m)
pl_site_to_xy(m::PartonLatticeModel, ::Int) = _pl_todo("pl_site_to_xy", m)
pl_xy_to_site(m::PartonLatticeModel, ::Int, ::Int) = _pl_todo("pl_xy_to_site", m)
pl_bonds(m::PartonLatticeModel) = _pl_todo("pl_bonds", m)
pl_all_pairs(m::PartonLatticeModel) = _pl_todo("pl_all_pairs", m)
pl_hopping(m::PartonLatticeModel, ::Int, ::Int, ::Int, ::Int) = _pl_todo("pl_hopping", m)
pl_mf_hopping(m::PartonLatticeModel, ::Int, ::Int, ::Int, ::Int, ::Int) =
    _pl_todo("pl_mf_hopping", m)
pl_nn_pairs(m::PartonLatticeModel) = _pl_todo("pl_nn_pairs", m)
# オンサイト項(トーラス自己像)。既定 0 — checkerboard には無い。
pl_onsite(::PartonLatticeModel, ::Int) = ComplexF64(0)
pl_mf_onsite(::PartonLatticeModel, ::Int, ::Int) = ComplexF64(0)
pl_cell_class(m::PartonLatticeModel, ::Int, ::Int, ::Int) = _pl_todo("pl_cell_class", m)
pl_physical_shifts(m::PartonLatticeModel) = _pl_todo("pl_physical_shifts", m)
pl_mf_shifts(m::PartonLatticeModel, ::Int, ::Int) = _pl_todo("pl_mf_shifts", m)

"""
    pl_shift_perm(m, tx, ty) -> Vector{Int}

グリッド変位 `(tx, ty)` が誘導するサイト置換。`perm[i + 1]` が移動先(0-based)。
"""
function pl_shift_perm(m::PartonLatticeModel, tx::Int, ty::Int)
    return [begin
                x, y = pl_site_to_xy(m, i)
                pl_xy_to_site(m, x + tx, y + ty)
            end for i = 0:(pl_nsite(m) - 1)]
end

"""
    pl_qp_shifts(m, ex, ey) -> Vector{Tuple{Int,Int}}

QP 射影に使う並進 = **物理の並進群 ÷ 平均場の並進群**の剰余代表(グリッド変位)。

平均場が破っている並進だけを張る。保たれている並進まで射影に入れても変分空間を
狭めるだけでエネルギーは下がらず、破っていない並進を落とすと運動量固有状態に
ならない。代表は各剰余類の中で `(tx, ty)` が最小のものを採る(決定的)。
"""
function pl_qp_shifts(m::PartonLatticeModel, ex::Int, ey::Int)
    phys = pl_physical_shifts(m)
    mf = Set(pl_mf_shifts(m, ex, ey))
    lx, ly = pl_grid(m)
    (0, 0) in mf || error("平均場の並進群に恒等が入っていません($(typeof(m)))")
    reps = Tuple{Int,Int}[]
    covered = Set{Tuple{Int,Int}}()
    for t in sort(phys)
        t in covered && continue
        push!(reps, t)
        for s in mf
            push!(covered, (mod(t[1] + s[1], lx), mod(t[2] + s[2], ly)))
        end
    end
    length(covered) == length(phys) || error(
        "剰余類の被覆が合いません: 被覆 $(length(covered)) / 物理並進 $(length(phys))。" *
        "平均場の並進群が物理の部分群になっているか確認すること")
    return reps
end

"""
    pl_swapped_pairs(m, ex, ey) -> Set{Tuple{Int,Int}}

**平均場の並進が両端点を入れ替えてしまう無向ペア**(キーは `minmax(i, j)`)。

そのようなペアは、並進 T が結合 {i,j} をそれ自身へ「向きを反転して」写すので、
H_MF が T で不変であるためには `α·t = conj(α·t)`、すなわち **α が実数**でなければ
ならない。コアの虚部強制凍結は `site1 == site2`(オンサイト)しか見ないので
(`parton_orbital.jl:85, 128`)、放置すると α は複素のまま SR に最適化され
**並進対称性が静かに破れる**。v3.14 の向き正準化バグ、PSG_NOTES §2.2 の
半分位相の枝と同じ族の罠。

- checkerboard の模型グラフ(d² ∈ {2,4,8})は対蹠変位を含まないので無傷。
  `graph = :full` は全サイト対を含むので必ず踏む
- KM は **Ly = 4 のとき**変位 (0, ±2) が対蹠になって踏む(4×4 で 8 ペア、
  9×4 で 18 ペア)。Ly = 5, 6 なら 0
"""
function pl_swapped_pairs(m::PartonLatticeModel, ex::Int, ey::Int)
    lx, ly = pl_grid(m)
    nsite = pl_nsite(m)
    shifts = [t for t in pl_mf_shifts(m, ex, ey) if t != (0, 0)]
    perms = [pl_shift_perm(m, tx, ty) for (tx, ty) in shifts]
    out = Set{Tuple{Int,Int}}()
    for i = 0:(nsite - 1), j = (i + 1):(nsite - 1)
        for p in perms
            if p[i + 1] == j && p[j + 1] == i
                push!(out, (i, j))
                break
            end
        end
    end
    return out
end

# ===========================================================================
# checkerboard 模型(既存 `checkerboard_model.jl` の関数へ委譲する)
# ===========================================================================

"""
    CheckerboardLatticeModel(nx, ny; p)

倍密グリッド `(2nx, 2ny)` 上の checkerboard。既存の `cb_*` 関数をそのまま呼ぶので、
`parton_fixture` を一般化しても係数はビット一致する(P 層 442 テストが守る)。
"""
struct CheckerboardLatticeModel <: PartonLatticeModel
    nx::Int
    ny::Int
    p::CheckerboardParams
end
CheckerboardLatticeModel(nx::Int, ny::Int; p::CheckerboardParams = CheckerboardParams()) =
    CheckerboardLatticeModel(nx, ny, p)

pl_nsite(m::CheckerboardLatticeModel) = 2 * m.nx * m.ny
pl_grid(m::CheckerboardLatticeModel) = (2 * m.nx, 2 * m.ny)
pl_cell_step(::CheckerboardLatticeModel) = (2, 2)
pl_site_to_xy(m::CheckerboardLatticeModel, i::Int) = cb_site_to_xy(i, m.nx)
function pl_xy_to_site(m::CheckerboardLatticeModel, x::Int, y::Int)
    lx, ly = pl_grid(m)
    return cb_xy_to_site(mod(x, lx), mod(y, ly), m.nx)
end
pl_bonds(m::CheckerboardLatticeModel) = cb_undirected_bonds(m.nx, m.ny)
pl_all_pairs(m::CheckerboardLatticeModel) = cb_all_pairs(m.nx, m.ny)

function pl_hopping(m::CheckerboardLatticeModel, i::Int, ::Int, dx::Int, dy::Int)
    _, y0 = cb_site_to_xy(i, m.nx)
    return cb_hopping(m.p, dx, dy, y0)
end
# checkerboard は正味フラックス 0 なので、平均場も物理と同じ t_ij から出発する
# (パートンごとにフラックスを割るという構造が無い)。
pl_mf_hopping(m::CheckerboardLatticeModel, i::Int, j::Int, dx::Int, dy::Int, ::Int) =
    pl_hopping(m, i, j, dx, dy)

pl_nn_pairs(m::CheckerboardLatticeModel) =
    [(i, j) for (i, j, _, _, d2) in cb_undirected_bonds(m.nx, m.ny) if d2 == 2]

pl_cell_class(m::CheckerboardLatticeModel, i::Int, ex::Int, ey::Int) =
    enlarged_cell_class(i, m.nx, m.ny, ex, ey)

# 正味フラックス 0 なので**全並進**が物理 H の対称性(P2-2 で機械検証済み)。
pl_physical_shifts(m::CheckerboardLatticeModel) =
    [(2 * ucx, 2 * ucy) for ucy = 0:(m.ny - 1) for ucx = 0:(m.nx - 1)]

# 平均場は拡大セル (ex, ey) の並進だけを保つ。
pl_mf_shifts(m::CheckerboardLatticeModel, ex::Int, ey::Int) =
    [(2 * ex * a, 2 * ey * b) for b = 0:(div(m.ny, ey) - 1) for a = 0:(div(m.nx, ex) - 1)]

# ===========================================================================
# Kapit-Mueller 模型
# ===========================================================================

"""
    KMLatticeModel(Lx, Ly, φ, F; hopmax)

正方格子 `Lx × Ly`(1 セル 1 サイト、`isite = Lx·y + x`)の Kapit-Mueller 模型。

```
t(z) = W(z)·G(z)
W(z) = exp(−(π/2)(1 − φ)|z|²)·(−1)^(x + y + xy)
G(z) = exp(iπφ · y · (2·x_start + x))          Landau ゲージ
```

- 平均場は各パートンが `φ_f = φ/F` を見る。`G(z; φ/F)^F = G(z; φ)` が**厳密に**
  成立するのは Landau ゲージだけで、これにより F 枚の平均場の積が物理の φ を再現する
- **同じ端点対に落ちる全変位の振幅を足す**。最短 1 本だけ残す規約(元 ED の
  `select_minimum_bonds`)は対蹠変位で並進対称性を壊す(9×4・φ=1/3 で T_y が落ちる)
- `hopmax` は打ち切り距離。**8.0 で厳密 KM に収束**(1 体バンド幅が厳密に 0)。
  2.0(第 3 近接)では ν=1/2 が FCI にならない(多重項幅 > ギャップ)

`φ·Lx ∈ ℤ` が必要。満たさないと巻き付き位相が残り H が非エルミートになる。
"""
struct KMLatticeModel <: PartonLatticeModel
    Lx::Int
    Ly::Int
    φ::Float64
    F::Int
    hopmax::Float64
    function KMLatticeModel(Lx::Int, Ly::Int, φ::Real, F::Int; hopmax::Real = 8.0)
        q = 1 / φ
        for (name, f) in (("物理 φ", float(φ)), ("平均場 φ/F", float(φ) / F))
            v = f * Lx
            abs(v - round(v)) < 1e-10 || error(
                "$name·Lx が整数ではありません($v)。Landau ゲージでは巻き付き位相が " *
                "残り H が非エルミートになります。Lx は F/φ = $(F * q) の倍数にすること")
        end
        hopmax >= 1.0 || error("hopmax は 1.0 以上(最近接が入らない): $hopmax")
        return new(Lx, Ly, float(φ), F, float(hopmax))
    end
end

pl_nsite(m::KMLatticeModel) = m.Lx * m.Ly
pl_grid(m::KMLatticeModel) = (m.Lx, m.Ly)
pl_cell_step(::KMLatticeModel) = (1, 1)
pl_site_to_xy(m::KMLatticeModel, i::Int) = (mod(i, m.Lx), div(i, m.Lx))
pl_xy_to_site(m::KMLatticeModel, x::Int, y::Int) = m.Lx * mod(y, m.Ly) + mod(x, m.Lx)

"KM の Gauss 包絡 W(z)。"
km_envelope(dx::Int, dy::Int, φ::Float64) =
    exp(-(π / 2) * (1 - φ) * (dx^2 + dy^2)) * (-1)^(dx + dy + dx * dy)

"Landau ゲージの位相因子。`x_start` はホップ元の x 座標。"
km_gauge(x_start::Int, dx::Int, dy::Int, φ::Float64) =
    exp(π * im * φ * dy * (2 * x_start + dx))

"""
    km_pair_hopping(m, i, j, φ) -> ComplexF64

`i → j` のホッピング振幅(= c†_j c_i の係数)。**同じ端点対に落ちる全変位を足す**。
"""
function km_pair_hopping(m::KMLatticeModel, i::Int, j::Int, φ::Float64)
    xi, yi = pl_site_to_xy(m, i)
    dmax = floor(Int, m.hopmax)
    v = ComplexF64(0)
    for dy = -dmax:dmax, dx = -dmax:dmax
        d2 = dx^2 + dy^2
        (0 < d2 <= m.hopmax^2) || continue
        pl_xy_to_site(m, xi + dx, yi + dy) == j || continue
        v += km_envelope(dx, dy, φ) * km_gauge(xi, dx, dy, φ)
    end
    return v
end

pl_hopping(m::KMLatticeModel, i::Int, j::Int, ::Int, ::Int) = km_pair_hopping(m, i, j, m.φ)

"""
オンサイト項。トーラス上でサイト i から変位 `(±Lx·a, ±Ly·b)` で自分自身に戻る像の和。
`hopmax` が格子サイズを超えると必ず出る(4×4・φ=1/2・hopmax=8 で 1.4e-5、
平均場 φ/4 で 2.6e-8)。**並進対称なので全サイト同じ値**の定数シフトだが、ED 側は
これを含むので落とすとエネルギーが合わない。エルミート性から実数になる。
"""
pl_onsite(m::KMLatticeModel, i::Int) = km_pair_hopping(m, i, i, m.φ)
pl_mf_onsite(m::KMLatticeModel, i::Int, F::Int) = km_pair_hopping(m, i, i, m.φ / F)
pl_mf_hopping(m::KMLatticeModel, i::Int, j::Int, ::Int, ::Int, F::Int) =
    km_pair_hopping(m, i, j, m.φ / F)

"""
無向ボンド。トーラス上で同じ端点対に落ちる変位は**代表 1 本**に併合する
(係数は `km_pair_hopping` が全変位を足すので、変分自由度は失われない)。
代表は d² 昇順 →(dx, dy) 昇順で決まるので、並進コピーは必ず同じ代表へ落ちる。
"""
function pl_bonds(m::KMLatticeModel)
    dmax = floor(Int, m.hopmax)
    disps = sort([(dx^2 + dy^2, dx, dy) for dy = -dmax:dmax, dx = -dmax:dmax
                  if 0 < dx^2 + dy^2 <= m.hopmax^2])
    seen = Set{Tuple{Int,Int}}()
    out = NTuple{5,Int}[]
    for (d2, dx, dy) in disps, i = 0:(pl_nsite(m) - 1)
        x0, y0 = pl_site_to_xy(m, i)
        j = pl_xy_to_site(m, x0 + dx, y0 + dy)
        i == j && continue
        key = minmax(i, j)
        key in seen && continue
        push!(seen, key)
        # 無向ボンドは i < j 側を採り、変位もその向きに合わせる
        i < j ? push!(out, (i, j, dx, dy, d2)) : push!(out, (j, i, -dx, -dy, d2))
    end
    return out
end

"全無向サイト対。hopmax を格子いっぱいに広げた `pl_bonds` と同じ列挙順。"
function pl_all_pairs(m::KMLatticeModel)
    nsite = pl_nsite(m)
    full = KMLatticeModel(m.Lx, m.Ly, m.φ, m.F;
                          hopmax = sqrt((m.Lx ÷ 2)^2 + (m.Ly ÷ 2)^2) + 1e-9)
    out = pl_bonds(full)
    length(out) == div(nsite * (nsite - 1), 2) || error(
        "全サイト対の列挙が不足: $(length(out)) / $(div(nsite * (nsite - 1), 2))")
    return out
end

"物理ハミルトニアンの密度型は最近接(d² = 1)に載せる。"
function pl_nn_pairs(m::KMLatticeModel)
    out = Tuple{Int,Int}[]
    for i = 0:(pl_nsite(m) - 1)
        x, y = pl_site_to_xy(m, i)
        for (dx, dy) in ((1, 0), (0, 1))
            j = pl_xy_to_site(m, x + dx, y + dy)
            i == j && continue
            push!(out, minmax(i, j))
        end
    end
    return sort(unique(out))
end

"""
拡大セルの軌道ラベル。KM の平均場(φ/F、Landau ゲージ)が保つ並進は `⟨T_y⟩` だけで、
x 方向は全破れなので、既定の拡大セルは `(ex, ey) = (Lx, 1)` になる。
このときラベルは `(x, 0)` = 「始点の x 座標」で、クラス = T_y 軌道。
"""
function pl_cell_class(m::KMLatticeModel, i::Int, ex::Int, ey::Int)
    x, y = pl_site_to_xy(m, i)
    return (mod(x, ex), mod(y, ey))
end

"""
物理 H を保つ並進 = `⟨T_x^q, T_y⟩`(q = 1/φ)。Landau ゲージでは x 方向の並進が
`exp(2πi φ · dy · a)` の位相を生むので、`φ·a ∈ ℤ` すなわち a が q の倍数のときだけ
素の置換として H を保つ。y 方向は位相が座標に依らないので常に保つ。
"""
function pl_physical_shifts(m::KMLatticeModel)
    q = round(Int, 1 / m.φ)
    return [(a, b) for b = 0:(m.Ly - 1) for a = 0:(m.Lx - 1) if a % q == 0]
end

"平均場(φ/F)を保つ並進。拡大セル `(ex, ey)` の倍数。"
pl_mf_shifts(m::KMLatticeModel, ex::Int, ey::Int) =
    [(ex * a, ey * b) for b = 0:(div(m.Ly, ey) - 1) for a = 0:(div(m.Lx, ex) - 1)]

"""
    km_default_cell(m) -> (ex, ey)

KM の既定の拡大セル。平均場のフラックスは `φ/F` なので x 方向は全破れ(`ex = Lx`)、
y 方向は全保存(`ey = 1`)。このとき QP 群の位数は `Lx/q = F` になり、
期待されるトーラス位相縮退と一致する。
"""
km_default_cell(m::KMLatticeModel) = (m.Lx, 1)
