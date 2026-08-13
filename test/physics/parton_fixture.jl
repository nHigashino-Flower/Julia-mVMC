"""
checkerboard 模型のパートン ansatz フィクスチャ生成
--- parton-mode (fork addition) ---

DESIGN_parton.md §8 P 層。P1 で検証済みの `checkerboard_model.jl` の t_ij から、
pmftrans / pmfpara / physhop / coulombinter を組む。

## 支給された仕様(発明しない)

- パートン境界条件: 全フレーバー周期(θ_f = 0)。Σθ_f = θ_phys = 0 を満たす
- ν=1/2 (F=2, ボソン b = f0 f1):
  - pmftrans = cb 模型のホッピング項をそのまま(全フレーバー同一)
  - pmfpara  = x 方向に 2 倍した拡大ユニットセル (2,0) の並進対称性
  - physhop  = cb 模型を b = f0 f1 でパートン展開したもの = t_ij そのまま
  - coulombinter = U = V = 0 なので無し
- ν=1/3 (F=3, フェルミオン f = f0 f1 f2):
  - pmftrans = 非対角ホッピングは t_ij + U、対角(オンサイト)は U のみ。係数は 1
  - pmfpara  = y 方向に 3 倍した拡大ユニットセル (0,3) の並進対称性
  - physhop  = cb 模型のパートン展開 = t_ij そのまま
  - coulombinter = 最近接ボンドに U(素の値。下の「規格化について」を参照)

## 規格化について

支給仕様には「フレーバーの分だけ余分に足した分を帳消しに規格化」とあったが、
M1 実装ではその補正は**不要**。`parton_diag_energy` が読む `cfg.ele_num[site]` は
固縛下でフレーバー 1 の占有数 = 物理粒子の占有数そのものなので、パートン数で
数えて U を F 倍する事故が起きない。よって coulombinter には素の U を入れる。

## α の初期値とゲージ

α は全 idx で 1.0(pmftrans が t_ij を丸ごと持つので、H_MF は非相互作用 cb 模型
そのもの = 最低 Chern バンドが埋まる)。凍結は入れず、冗長方向カットに任せる。
"""

"""
    enlarged_cell_class(isite, nx, ny, ex, ey) -> Int

サイトを拡大ユニットセルの並進で分類したときのクラス番号。
基本セルは (ux, uy) = (x ÷ 2, y) の格子点に副格子 bi ∈ {A, B} が 1 つずつ乗る。
拡大セルは基本セルを (ex, ey) 倍したものなので、クラスは
`(ux mod ex, uy mod ey, bi)` で決まる。
"""
function enlarged_cell_class(isite::Int, nx::Int, ny::Int, ex::Int, ey::Int)
    x, y = cb_site_to_xy(isite, nx)
    bi = iseven(y) ? 0 : 1          # 副格子 A / B
    uy = div(y, 2)                  # 基本セルの y 座標(倍密グリッドの 2 行で 1 セル)
    ux = div(x, 2)                  # 基本セルの x 座標
    return (mod(ux, ex), mod(uy, ey), bi)
end

"""
    bond_class(i, j, dx, dy, nx, ny, ex, ey)

ボンドを拡大セルの並進で分類する鍵。同じ鍵のボンドが 1 つの変分パラメータ idx を
共有する。

無向ボンドを `i < j` で正規化してあるので、始点だけで分類すると並進で端点の大小が
入れ替わったときに同じ軌道が 2 つのクラスへ割れる。両端点から見た鍵を作り、
小さい方を代表に採ることで軌道と 1 対 1 にする。
"""
function bond_class(i::Int, j::Int, dx::Int, dy::Int,
                    nx::Int, ny::Int, ex::Int, ey::Int)
    ki = (enlarged_cell_class(i, nx, ny, ex, ey), dx, dy)
    kj = (enlarged_cell_class(j, nx, ny, ex, ey), -dx, -dy)
    return min(ki, kj)
end

"""
    parton_fixture(nx, ny, nflavor, ex, ey; u_mf, p)

パートン ansatz の def テーブル一式を組んで返す。

返り値は NamedTuple:
- `pmftrans` : `(site1, flavor1, site2, flavor2, value)` の並び(0-based、片方向)
- `pmfpara`  : `(site1, flavor1, site2, flavor2, idx, value)` の並び
- `physhop`  : `(site1, site2, value)` の並び(0-based、片方向)
- `n_idx`    : 変分パラメータ数

`u_mf` は平均場に載せる U(ν=1/3 では 1.0、ν=1/2 では 0.0)。
`u_mf != 0` のときだけオンサイト項(対角 = U)が入る。
"""
function parton_fixture(nx::Int, ny::Int, nflavor::Int, ex::Int, ey::Int;
                        u_mf::Float64 = 0.0,
                        u_bonds::Symbol = :nn,
                        p::CheckerboardParams = CheckerboardParams())
    nsite = 2 * nx * ny
    bonds = cb_undirected_bonds(nx, ny)

    # --- 平均場: ボンドを拡大セルの並進クラスへ落として idx を振る ---
    class_of_idx = Dict{Any,Int}()
    pmftrans = NTuple{5,ComplexF64}[]
    pmfpara = Tuple{Int,Int,Int,Int,Int,ComplexF64}[]

    for (i, j, dx, dy, d2) in bonds
        _, y0 = cb_site_to_xy(i, nx)
        t = cb_hopping(p, dx, dy, y0)         # b†_j b_i の係数(i → j のホップ)
        # 非対角の平均場係数 = t_ij + U(支給仕様、係数 1)。U を載せるボンドは
        # `u_bonds` で選ぶ(:nn なら最近接のみ、:all なら全ボンド)。
        add_u = (u_mf != 0.0) && (u_bonds === :all || d2 == 2)
        coeff = t + (add_u ? u_mf : 0.0)
        key = bond_class(i, j, dx, dy, nx, ny, ex, ey)
        idx = get!(class_of_idx, key, length(class_of_idx))
        # pmftrans は `H[site1, site2] += α·value` と読まれる = c†_{site1} c_{site2}
        # の係数。こちらの t は c†_j c_i の係数なので (site1, site2) = (j, i) で出す。
        # physhop の (site1, site2, value) は b†_{site2} b_{site1} の係数なので
        # 向きが逆になる — 2 つの新設形式で添字の向きが違うことに注意。
        for f = 0:(nflavor - 1)
            push!(pmftrans, (ComplexF64(j), ComplexF64(f), ComplexF64(i),
                             ComplexF64(f), coeff))
            push!(pmfpara, (j, f, i, f, idx, ComplexF64(1, 0)))
        end
    end

    # --- 平均場: オンサイト(対角 = U のみ)。u_mf = 0 なら入れない ---
    if u_mf != 0.0
        for i = 0:(nsite - 1)
            key = (enlarged_cell_class(i, nx, ny, ex, ey), 0, 0)
            idx = get!(class_of_idx, key, length(class_of_idx))
            for f = 0:(nflavor - 1)
                push!(pmftrans, (ComplexF64(i), ComplexF64(f), ComplexF64(i),
                                 ComplexF64(f), ComplexF64(u_mf)))
                push!(pmfpara, (i, f, i, f, idx, ComplexF64(1, 0)))
            end
        end
    end

    # --- 物理ハミルトニアンの合成ホップ: t_ij をそのまま(片方向) ---
    physhop = Tuple{Int,Int,ComplexF64}[]
    for (i, j, dx, dy, _) in bonds
        _, y0 = cb_site_to_xy(i, nx)
        push!(physhop, (i, j, cb_hopping(p, dx, dy, y0)))
    end

    return (
        pmftrans = [(Int(real(a)), Int(real(b)), Int(real(c)), Int(real(d)), v)
                    for (a, b, c, d, v) in pmftrans],
        pmfpara = pmfpara,
        physhop = physhop,
        n_idx = length(class_of_idx),
        nsite = nsite,
    )
end

"""
    physical_coulomb(nx, ny, u) -> Vector{Tuple{Int,Int,Float64}}

物理ハミルトニアンの密度型: 最近接ボンド(d² = 2)に U。素の値を入れる
(上の「規格化について」を参照)。V = 0 なので 2 次近接は入れない。
"""
function physical_coulomb(nx::Int, ny::Int, u::Float64)
    u == 0.0 && return Tuple{Int,Int,Float64}[]
    return [(i, j, u) for (i, j, _, _, d2) in cb_undirected_bonds(nx, ny) if d2 == 2]
end
