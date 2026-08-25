using Printf

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

**係数の向きも代表に揃えること**(2026-08-18 のバグ修正)。クラスだけ min で
1 対 1 化して係数 t をリスト向きのまま書くと、大小が入れ替わったコピーは
conj(t) 側で載り、複素 α で拡大セル並進が破れる。`parton_fixture` のボンド
ループが正準化を行う(このヘルパは鍵の定義の記録として残す)。
"""
function bond_class(i::Int, j::Int, dx::Int, dy::Int,
                    nx::Int, ny::Int, ex::Int, ey::Int)
    ki = (enlarged_cell_class(i, nx, ny, ex, ey), dx, dy)
    kj = (enlarged_cell_class(j, nx, ny, ex, ey), -dx, -dy)
    return min(ki, kj)
end


"""
    _idx_key(mode, orbit_key, bond) / _flavor_key(mode, key, f, groups)

変分パラメータ idx の粒度を決める(候補 C の検証用)。参照実装 PartonFCI の
`symmetry_mode` に対応させてある:

- `:orbit`        拡大セル軌道で縮約 + フレーバー共有 (= SU(K)_UNIFIED 相当)
- `:orbit_flavor` 拡大セル軌道で縮約 + フレーバー独立
- `:bond_flavor`  ボンドごと + フレーバー独立 (= SU(K)_INDEPENDENT 相当)

`:orbit` が既定で、P1〜P3 のこれまでの結果はすべてこれ。
"""
_idx_key(mode::Symbol, orbit_key, bond) =
    mode === :bond_flavor ? (:bond, bond) : orbit_key

"""
`groups` が与えられたらフレーバー f は `groups[f+1]` 番の群に属し、同じ群の
フレーバーが 1 つの idx を共有する(`fill(0,F)` = 全共有 = `:orbit` 相当、
`collect(0:F-1)` = 全独立 = `:orbit_flavor` 相当、`[0,0,1]` = 2+1)。
`nothing` なら従来の `idx_mode` 規約。
"""
_flavor_key(mode::Symbol, key, f::Int, groups::Union{Nothing,Vector{Int}}) =
    groups === nothing ? (mode === :orbit ? key : (key, f)) : (key, groups[f + 1])

"""
    cb_psg_extra_bonds(nx, ny, shells) -> Vector{NTuple{5,Int}}

PSG 拡張用の追加ボンド(d² ∈ shells)を無向で 1 本ずつ返す。トーラス上の
巻き付きで同じサイト対に落ちる変位は先着 1 本に併合する(H の行列要素は
サイト対ごとに 1 つなので変分自由度は失われない)。既存の NN/NNN/NNNN
(d² = 2, 4, 8)と重なるサイト対も除外する。変位の列挙順を固定することで、
並進コピーが必ず同じ変位クラスへ落ちる(idx 共有の射影的並進対称性を保つ)。
"""
function cb_psg_extra_bonds(nx::Int, ny::Int, shells::Vector{Int})
    lx, ly = 2 * nx, 2 * ny
    nsite = 2 * nx * ny
    seen = Set{Tuple{Int,Int}}()
    for (i, j, _, _, _) in cb_undirected_bonds(nx, ny)
        push!(seen, minmax(i, j))
    end
    out = NTuple{5,Int}[]
    for d2 in sort(unique(shells))
        d2 in (2, 4, 8) && error("d² = $(d2) は既存シェル。shells には追加分だけを渡す")
        disps = sort([(dx, dy) for dx = -(lx ÷ 2):(lx ÷ 2), dy = -(ly ÷ 2):(ly ÷ 2)
                      if dx^2 + dy^2 == d2 && mod(dx, 2) == mod(dy, 2)])
        for (dx, dy) in disps, i = 0:(nsite - 1)
            x0, y0 = cb_site_to_xy(i, nx)
            j = cb_xy_to_site(mod(x0 + dx, lx), mod(y0 + dy, ly), nx)
            i == j && continue
            key = minmax(i, j)
            key in seen && continue
            push!(seen, key)
            push!(out, (i, j, dx, dy, d2))
        end
    end
    return out
end

"""
    cb_all_pairs(nx, ny) -> Vector{NTuple{5,Int}}

トーラス上の**全**無向サイト対を 1 本ずつ返す(`(i, j, dx, dy, d2)`)。
`graph = :full` 用。変位の列挙順を d² 昇順 → (dx, dy) 昇順 → サイト昇順に
固定してあるので、並進コピーは必ず同じ変位クラスへ落ちる(idx 共有が
射影的並進対称性を保つ条件)。巻き付きで同じサイト対に落ちる変位は先着 1 本。
"""
function cb_all_pairs(nx::Int, ny::Int)
    lx, ly = 2 * nx, 2 * ny
    nsite = 2 * nx * ny
    seen = Set{Tuple{Int,Int}}()
    disps = Tuple{Int,Int,Int}[]
    for dx = -(lx ÷ 2):(lx ÷ 2), dy = -(ly ÷ 2):(ly ÷ 2)
        (dx == 0 && dy == 0) && continue
        mod(dx, 2) == mod(dy, 2) || continue        # CB の副格子条件
        push!(disps, (dx^2 + dy^2, dx, dy))
    end
    sort!(disps)
    out = NTuple{5,Int}[]
    for (d2, dx, dy) in disps, i = 0:(nsite - 1)
        x0, y0 = cb_site_to_xy(i, nx)
        j = cb_xy_to_site(mod(x0 + dx, lx), mod(y0 + dy, ly), nx)
        i == j && continue
        key = minmax(i, j)
        key in seen && continue
        push!(seen, key)
        push!(out, (i, j, dx, dy, d2))
    end
    length(out) == div(nsite * (nsite - 1), 2) || error(
        "全サイト対の列挙が不足: $(length(out)) / $(div(nsite*(nsite-1), 2))")
    return out
end

"""
    cb_translation_swapped_pairs(nx, ny, ex, ey) -> Set{Tuple{Int,Int}}

**拡大セル並進が両端点を入れ替えてしまう無向ペア**を返す(キーは `minmax(i, j)`)。

このようなペアは、並進 T が結合 {i,j} をそれ自身へ「向きを反転して」写すので、
H_MF が T で不変であるためには `α·t = conj(α·t)` すなわち **α が実数**でなければ
ならない。ところがコアの虚部強制凍結は `site1 == site2`(オンサイト)だけを見て
判定する(`parton_orbital.jl:85, 128`)ので、この種のボンドの α は複素のまま
SR に最適化され、**拡大セル並進が静かに破れる**。v3.14 の向き正準化バグ、
PSG_NOTES §2.2 の半分位相の枝と同じ族の罠(3 度目)。

`graph = :full` はトーラス上の全サイト対を含むので必ずこれを踏む。
模型グラフ(`graph = :model`、d² ∈ {2,4,8})は対蹠変位を含まないので無傷。
"""
function cb_translation_swapped_pairs(nx::Int, ny::Int, ex::Int, ey::Int)
    lx, ly = 2 * nx, 2 * ny
    nsite = 2 * nx * ny
    trans = [(a * ex, b * ey) for a = 0:(div(nx, ex) - 1) for b = 0:(div(ny, ey) - 1)
             if !(a == 0 && b == 0)]
    out = Set{Tuple{Int,Int}}()
    for i = 0:(nsite - 1), j = (i + 1):(nsite - 1)
        xi, yi = cb_site_to_xy(i, nx)
        xj, yj = cb_site_to_xy(j, nx)
        for (tx, ty) in trans
            gi = cb_xy_to_site(mod(xi + 2tx, lx), mod(yi + 2ty, ly), nx)
            gj = cb_xy_to_site(mod(xj + 2tx, lx), mod(yj + 2ty, ly), nx)
            if gi == j && gj == i
                push!(out, (i, j))
                break
            end
        end
    end
    return out
end

"""
    parton_fixture(nx, ny, nflavor, ex, ey; u_mf, p)

パートン ansatz の def テーブル一式を組んで返す。

返り値は NamedTuple:
- `pmftrans` : `(site1, flavor1, site2, flavor2, value)` の並び(0-based、片方向)
- `pmfpara`  : `(site1, flavor1, site2, flavor2, idx, value)` の並び
- `physhop`  : `(site1, site2, value)` の並び(0-based、片方向)
- `n_idx`    : 変分パラメータ数
- `psg_idx`  : PSG 拡張項の idx 集合(α = 0 で始める行の目印)
- `n_swapped_dropped` : `graph = :full` で除外した「並進が端点を入れ替えるペア」の数
  (`graph = :model` では常に 0。詳細は下の `## graph = :full` 節)

`u_mf` は平均場に載せる U(ν=1/3 では 1.0、ν=1/2 では 0.0)。
`u_mf != 0` のときだけオンサイト項(対角 = U)が入る。

## PSG 拡張(2026-08-20、Lu–Ran 射影構成の同一 PSG クラス内での完備化)

- `psg_onsite = true`  : 拡大セルクラスごとのオンサイト項(係数 1、α 実)を足す
- `psg_shells = [...]` : 追加ホッピングシェル(d² のリスト、係数 1、α 複素)を足す

いずれも idx は既存と同じ拡大セル軌道クラス(× idx_mode の粒度)で振るので、
射影的並進(磁気並進)対称性は保たれる。物理ハミルトニアン(physhop)には
一切触れない。α の初期値は 0(pmfpara.def に明示)で、初期状態 = 既存
アンザッツから SR が新しい方向を育てる。

## graph = :full(2026-08-25、FCI アンザッツ比較キャンペーン)

`graph::Symbol = :model`(既定、模型の t_ij による距離制限)/ `:full`
(全サイト対 + オンサイト、係数はすべて 1、α に全部持たせる)。`:full` は
`u_mf` / `psg_onsite` / `psg_shells` と併用不可(全サイト対を自前で
並べるため、二重定義になる)。`psg_idx` は空(全 idx が乱数初期化)。
物理ハミルトニアン(physhop)は `:full` でも模型の t_ij のまま変わらない。

**拡大セル並進が端点を入れ替えるペアは除外する**(2026-08-25、v2 修正)。
そのようなペアは α が実数でないと並進共変にできないが、コアの虚部強制
凍結は `site1 == site2`(オンサイト)しか見ないため、複素のまま SR に
最適化され並進対称性が静かに破れる(詳細は `cb_translation_swapped_pairs`
の docstring)。4×4(ex=ey=2)で 48/496 ペア、6×3(ex=ey=3)で 18/630
ペアが該当し除外される。除外数は返り値の `n_swapped_dropped` で確認できる
(`graph = :model` では常に 0)。
"""
function parton_fixture(nx::Int, ny::Int, nflavor::Int, ex::Int, ey::Int;
                        u_mf::Float64 = 0.0,
                        u_bonds::Symbol = :nn,
                        idx_mode::Symbol = :orbit,
                        flavor_groups::Union{Nothing,Vector{Int}} = nothing,
                        psg_onsite::Bool = false,
                        psg_shells::Vector{Int} = Int[],
                        graph::Symbol = :model,
                        p::CheckerboardParams = CheckerboardParams())
    psg_onsite && u_mf != 0.0 &&
        error("psg_onsite と u_mf != 0 は同じ (i,i) 行を二重に作るので併用不可")
    graph in (:model, :full) || error("graph は :model / :full。graph = $(graph)")
    if graph === :full
        (u_mf != 0.0 || psg_onsite || !isempty(psg_shells)) && error(
            "graph = :full は全サイト対を自前で並べるので u_mf / psg_onsite / " *
            "psg_shells とは併用しません")
    end
    if flavor_groups !== nothing
        length(flavor_groups) == nflavor || error(
            "flavor_groups の長さ $(length(flavor_groups)) が NFlavor = $(nflavor) と違います")
        all(0 .<= flavor_groups .< nflavor) || error(
            "flavor_groups の値は 0..$(nflavor-1): $(flavor_groups)")
        idx_mode === :bond_flavor && error(
            "flavor_groups と idx_mode = :bond_flavor は併用しません")
    end
    nsite = 2 * nx * ny
    bonds = cb_undirected_bonds(nx, ny)
    # graph = :full は平均場のグラフだけを全サイト対へ広げる。物理 H(physhop)は
    # 模型の t_ij のまま(下の physhop ループが `bonds` を使う)。ただし拡大セル
    # 並進が端点を入れ替えるペアは α が実数でないと並進共変にできず、コアは
    # その虚部を凍結しない(オンサイトしか見ない)ので**除外する**
    # (`cb_translation_swapped_pairs` の docstring 参照)。
    n_swapped_dropped = 0
    mf_bonds = if graph === :full
        swapped = cb_translation_swapped_pairs(nx, ny, ex, ey)
        kept = [b for b in cb_all_pairs(nx, ny) if !(minmax(b[1], b[2]) in swapped)]
        n_swapped_dropped = length(swapped)
        kept
    else
        bonds
    end

    # --- 平均場: ボンドを拡大セルの並進クラスへ落として idx を振る ---
    class_of_idx = Dict{Any,Int}()
    pmftrans = NTuple{5,ComplexF64}[]
    pmfpara = Tuple{Int,Int,Int,Int,Int,ComplexF64}[]

    for (i, j, dx, dy, d2) in mf_bonds
        # --- 向きの正準化(2026-08-18 のバグ修正)---------------------------
        # `bond_class` はクラス代表を min(ki, kj) で選ぶが、係数 t を i < j の
        # リスト向きのまま書くと、並進コピーで端点の大小が入れ替わったボンド
        # (8×8 ef4 で 1536 行中 624 行)が t の共役側で載る。組み立ては
        # 「リストされた向きで α·t + h.c.」なので、同一 idx に α·t と α·conj(t)
        # が同居し、**α が複素だと拡大セル並進が破れる**(y 方向残差 0.6〜0.8 を
        # 実測)。α = 1(実数)では h.c. と合流して同一の H になるため、初期
        # ハミルトニアンだけを見る検査では発見できない。
        # 係数もクラス代表と同じ向き(min キーを供給する端点を始点)で書く。
        ki = (enlarged_cell_class(i, nx, ny, ex, ey), dx, dy)
        kj = (enlarged_cell_class(j, nx, ny, ex, ey), -dx, -dy)
        if ki <= kj
            a, b, da, db = i, j, dx, dy      # ホップ a → b(向きは i → j のまま)
        else
            a, b, da, db = j, i, -dx, -dy    # 反転: j → i を正準に採る
        end
        _, y0 = cb_site_to_xy(a, nx)
        if graph === :full
            coeff = ComplexF64(1)      # 全 1。t の距離制限を外し α に全部持たせる
        else
            t = cb_hopping(p, da, db, y0)         # b†_b b_a の係数(a → b のホップ)
            # 内部検査: 反転はエルミート共役に一致するはず(モデル側の規約が
            # 破れたらここで気づく)
            _, y0i = cb_site_to_xy(i, nx)
            abs(t - (ki <= kj ? cb_hopping(p, dx, dy, y0i) :
                                conj(cb_hopping(p, dx, dy, y0i)))) < 1e-13 ||
                error("cb_hopping is not Hermitian for bond ($i, $j, $dx, $dy)")
            # 非対角の平均場係数 = t_ij + U(支給仕様、係数 1)。U を載せるボンドは
            # `u_bonds` で選ぶ(:nn なら最近接のみ、:all なら全ボンド)。
            add_u = (u_mf != 0.0) && (u_bonds === :all || d2 == 2)
            coeff = t + (add_u ? u_mf : 0.0)
        end
        key = _idx_key(idx_mode, min(ki, kj), (i, j))
        # pmftrans は `H[site1, site2] += α·value` と読まれる = c†_{site1} c_{site2}
        # の係数。こちらの t は c†_b c_a の係数なので (site1, site2) = (b, a) で出す。
        # physhop の (site1, site2, value) は b†_{site2} b_{site1} の係数なので
        # 向きが逆になる — 2 つの新設形式で添字の向きが違うことに注意。
        for f = 0:(nflavor - 1)
            idx = get!(class_of_idx, _flavor_key(idx_mode, key, f, flavor_groups),
                       length(class_of_idx))
            push!(pmftrans, (ComplexF64(b), ComplexF64(f), ComplexF64(a),
                             ComplexF64(f), coeff))
            push!(pmfpara, (b, f, a, f, idx, ComplexF64(1, 0)))
        end
    end

    # --- 平均場: オンサイト。u_mf != 0(ν=1/3 の Hartree)か graph = :full ---
    if u_mf != 0.0 || graph === :full
        onsite_coeff = graph === :full ? ComplexF64(1) : ComplexF64(u_mf)
        for i = 0:(nsite - 1)
            key = _idx_key(idx_mode, (enlarged_cell_class(i, nx, ny, ex, ey), 0, 0),
                           (i, i))
            for f = 0:(nflavor - 1)
                idx = get!(class_of_idx, _flavor_key(idx_mode, key, f, flavor_groups),
                           length(class_of_idx))
                push!(pmftrans, (ComplexF64(i), ComplexF64(f), ComplexF64(i),
                                 ComplexF64(f), onsite_coeff))
                push!(pmfpara, (i, f, i, f, idx, ComplexF64(1, 0)))
            end
        end
    end

    # --- PSG 拡張(2026-08-20)。既存クラス割り当ての後に足すので idx は後詰め ---
    psg_idx = Set{Int}()

    # (a) オンサイト(係数 1、α は乱数規約によりオンサイト群 = 実のみ)
    if psg_onsite
        for i = 0:(nsite - 1)
            key = _idx_key(idx_mode, (enlarged_cell_class(i, nx, ny, ex, ey), 0, 0),
                           (i, i))
            for f = 0:(nflavor - 1)
                idx = get!(class_of_idx, _flavor_key(idx_mode, key, f, flavor_groups),
                           length(class_of_idx))
                push!(psg_idx, idx)
                push!(pmftrans, (ComplexF64(i), ComplexF64(f), ComplexF64(i),
                                 ComplexF64(f), ComplexF64(1)))
                push!(pmfpara, (i, f, i, f, idx, ComplexF64(0, 0)))
            end
        end
    end

    # (b) 追加シェル(係数 1、α 複素)。正準化はメインのボンドループと同一
    #     (t = 1 は実なので係数の向きは効かないが、クラス代表との整合は保つ)
    for (i, j, dx, dy, _) in cb_psg_extra_bonds(nx, ny, psg_shells)
        ki = (enlarged_cell_class(i, nx, ny, ex, ey), dx, dy)
        kj = (enlarged_cell_class(j, nx, ny, ex, ey), -dx, -dy)
        a, b = ki <= kj ? (i, j) : (j, i)
        key = _idx_key(idx_mode, min(ki, kj), (i, j))
        for f = 0:(nflavor - 1)
            idx = get!(class_of_idx, _flavor_key(idx_mode, key, f, flavor_groups),
                       length(class_of_idx))
            push!(psg_idx, idx)
            push!(pmftrans, (ComplexF64(b), ComplexF64(f), ComplexF64(a),
                             ComplexF64(f), ComplexF64(1)))
            push!(pmfpara, (b, f, a, f, idx, ComplexF64(0, 0)))
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
        psg_idx = psg_idx,
        n_swapped_dropped = n_swapped_dropped,
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

"""
    write_parton_def_files(dir, nx, ny, F, ex, ey; u_mf, u_phys, sr...)

パートンモードの入力一式を `dir` に書き出す。

**pmfpara.def は value 列を空けた 5 列**で出す(DESIGN §2.3.1)。未入力なので
α は乱数で初期化され、SR が π-flux 構造を探しに行く。明示指定したい場合は
7 列(… idx Re Im)で書くか `InPmfPara.def` を与える。

末尾の OptFlag 行は**全 idx を明示的に**書く(既定は 1 = 可動)。`opt_flags` に
`Dict(idx => 0)` を渡せばその idx を凍結できる。ゲージ平坦方向は v3.2 以降
`parton_project_gauge!` が潰すので、ゲージ目的でここを 0 にする必要はない。
"""
function write_parton_def_files(dir::AbstractString, nx::Int, ny::Int, F::Int,
                                ex::Int, ey::Int;
                                u_mf::Float64 = 0.0,
                                u_phys::Float64 = 0.0,
                                n_elec::Int,
                                nvmc_sample::Int = 1500,
                                nvmc_warmup::Int = 500,
                                nsr_step::Int = 2300,
                                nsr_smp::Int = 100,
                                dt::Float64 = 0.005,
                                sta_del::Float64 = 0.01,
                                red_cut::Float64 = 1e-6,   # v3.11: 参照 chi-VMC の「必須」値。gap 崩壊域で 1e-8 との差が出る(REPORT §14)
                                block_update::Int = 16,
                                seed::Int = 11272,
                                idx_mode::Symbol = :orbit,
                                flavor_groups::Union{Nothing,Vector{Int}} = nothing,
                                psg_onsite::Bool = false,
                                psg_shells::Vector{Int} = Int[],
                                graph::Symbol = :model,
                                opt_flags::Union{Nothing,Dict{Int,Int}} = nothing,
                                qp_momentum::Union{Nothing,Tuple{Int,Int}} = nothing,
                                qp_xext::Union{Nothing,Tuple{Int,Int}} = nothing,
                                jastrow_full_trans::Bool = false)
    mkpath(dir)
    fx = parton_fixture(nx, ny, F, ex, ey; u_mf = u_mf, idx_mode = idx_mode,
                        flavor_groups = flavor_groups,
                        psg_onsite = psg_onsite, psg_shells = psg_shells,
                        graph = graph)
    cb = physical_coulomb(nx, ny, u_phys)

    open(joinpath(dir, "pmftrans.def"), "w") do io
        println(io, "===============================")
        println(io, "NPartonMFTrans $(length(fx.pmftrans))")
        println(io, "===============================")
        println(io, "== site1 flavor1 site2 flavor2 Re Im ==")
        println(io, "===============================")
        for (a, b, c, d, v) in fx.pmftrans
            @printf(io, "%d %d %d %d % .18e % .18e\n", a, b, c, d, real(v), imag(v))
        end
    end

    # value 列を出さない = 5 列 = 未入力 → 乱数初期化
    open(joinpath(dir, "pmfpara.def"), "w") do io
        println(io, "===============================")
        println(io, "NPartonMFParaIdx $(fx.n_idx)")
        println(io, "ComplexType          1")
        println(io, "===============================")
        println(io, "===============================")
        # PSG 拡張の idx は α = 0 を 7 列で明示する(初期状態 = 既存アンザッツ)。
        # 既存 idx は 5 列(未入力)のまま乱数初期化。同一 idx 内で presence が
        # 揃っていることはパーサが検証する(DESIGN §2.3.1)。
        for (a, b, c, d, i, _) in fx.pmfpara
            if i in fx.psg_idx
                @printf(io, "%d %d %d %d %d % .18e % .18e\n", a, b, c, d, i, 0.0, 0.0)
            else
                @printf(io, "%d %d %d %d %d\n", a, b, c, d, i)
            end
        end
        # 末尾フラグ行は全 idx を明示的に書く(既定は 1 = 可動)。
        # v3.2 以降 OptFlag はゲージ目的では使わず、用途はエルミート性
        # (オンサイト群 Im の強制凍結。コード側が自動でやるのでここには書かない)と
        # ユーザーの明示的固定に限られる。値を省略せず並べておくことで、
        # 「何も固定していない」ことがファイル自身から読み取れるようにする。
        flags = opt_flags === nothing ? Dict{Int,Int}() : opt_flags
        for i = 0:(fx.n_idx - 1)
            @printf(io, "%d %d\n", i, get(flags, i, 1))
        end
    end

    open(joinpath(dir, "physhop.def"), "w") do io
        println(io, "===============================")
        println(io, "NPhysHop $(length(fx.physhop))")
        println(io, "===============================")
        println(io, "== site1 site2 Re Im ==")
        println(io, "===============================")
        for (a, b, v) in fx.physhop
            @printf(io, "%d %d % .18e % .18e\n", a, b, real(v), imag(v))
        end
    end

    entries = ["ModPara        modpara.def",
               "PartonMFTrans  pmftrans.def",
               "PartonMFPara   pmfpara.def",
               "PhysHop        physhop.def"]

    # transJastrow(v3.11 M2 後半)。無順序ペア {i, j} を**全基本セル並進**の軌道で
    # 類別した jastrowidx.def を書く。参照実装 `symmetrize_jastrow_idx` の
    # trans_maps = 全並進の場合と同じ同値類(P_J が全並進と可換 = 運動量射影の外側に
    # かけても |ψ⟩ が運動量固有状態のままでいられる条件)。初期値 v = 0(明示の
    # InJastrow は書かない — v = 0 は Jastrow なしと同じ状態から SR が降下を始める)。
    if jastrow_full_trans
        maps, _ = cb_translations(nx, ny)
        class_of = Dict{Tuple{Int,Int},Int}()
        pair_class = Dict{Tuple{Int,Int},Int}()
        for i = 0:(fx.nsite - 1), j = 0:(fx.nsite - 1)
            i == j && continue
            key0 = minmax(i, j)
            haskey(pair_class, key0) && continue
            # 軌道の正準代表 = 並進像の (min, max) の最小
            rep = key0
            for m in maps
                key = minmax(m[i + 1], m[j + 1])
                key < rep && (rep = key)
            end
            cls = get!(class_of, rep, length(class_of))
            for m in maps
                pair_class[minmax(m[i + 1], m[j + 1])] = cls
            end
        end
        n_jast = length(class_of)
        open(joinpath(dir, "jastrowidx.def"), "w") do io
            println(io, "===============================")
            println(io, "NJastrowIdx $n_jast")
            println(io, "ComplexType 0")
            println(io, "===============================")
            println(io, "===============================")
            for i = 0:(fx.nsite - 1), j = 0:(fx.nsite - 1)
                i == j && continue
                @printf(io, "%d %d %d\n", i, j, pair_class[minmax(i, j)])
            end
            for k = 0:(n_jast - 1)
                println(io, "$k 1")
            end
        end
        push!(entries, "Jastrow        jastrowidx.def")
    end

    # 運動量射影(M2)。`qp_momentum = (nkx, nky)` で k = 2π(nkx/nx, nky/ny) を指定する。
    # 重み exp(2πi(Kx·ucx + Ky·ucy)) と写像は参照実装 build_TranslationalOperatorparams
    # と同じ列規約 — データ行は `Opidx jsite isite sgn` で **col2 が元サイト、
    # col3 が並進後**。NMPTrans は modpara 側にも書く: 別経路なので片方だけだと
    # 射影の項が黙って落ちる(門番 validate_parton_qp が捕まえる)。
    n_mp_trans = 1
    if qp_momentum !== nothing || qp_xext !== nothing
        qp_momentum === nothing || qp_xext === nothing ||
            error("qp_momentum(全並進)と qp_xext(参照実装準拠)は同時に指定できない")
        # `qp_xext = (kext, nkx)` が実運用の構成 — 参照実装 make_QNPidx と同じく
        # x 方向 kext 本だけを張る。`qp_momentum` は全並進 nx·ny 本で、射影の
        # 数学的性質(凸結合・完全性)を検証するための構成。
        maps, ucs = qp_xext === nothing ? cb_translations(nx, ny) :
                    cb_qp_translations(nx, ny, qp_xext[1])
        nkx, nky = qp_xext === nothing ? qp_momentum : (qp_xext[2], 0)
        n_kx, n_ky = qp_xext === nothing ? (nx, ny) : (qp_xext[1], 1)
        n_mp_trans = length(maps)
        open(joinpath(dir, "qptransidx.def"), "w") do io
            println(io, "===============================")
            println(io, "NQPTrans $(length(maps))")
            println(io, "===============================")
            println(io, "== i_j_TransSym ==")
            println(io, "===============================")
            for (k, (ucx, ucy)) in enumerate(ucs)
                w = cis(2π * (nkx * ucx / n_kx + nky * ucy / n_ky))
                @printf(io, "%d % .18e % .18e\n", k - 1, real(w), imag(w))
            end
            for (k, m) in enumerate(maps)
                for j = 0:(fx.nsite - 1)
                    @printf(io, "%d %d %d 1\n", k - 1, j, m[j + 1])
                end
            end
        end
        push!(entries, "TransSym       qptransidx.def")
    end
    if !isempty(cb)
        open(joinpath(dir, "coulombinter.def"), "w") do io
            println(io, "===============================")
            println(io, "NCoulombInter $(length(cb))")
            println(io, "===============================")
            println(io, "== CoulombInter ==")
            println(io, "===============================")
            for (a, b, v) in cb
                @printf(io, "%d %d % .18e\n", a, b, v)
            end
        end
        push!(entries, "CoulombInter   coulombinter.def")
    end

    open(joinpath(dir, "modpara.def"), "w") do io
        for line in [
            "CDataFileHead zvo", "CParaFileHead zqp", "NVMCCalMode 0",
            "Nsite $(fx.nsite)", "NElec $n_elec",
            "PartonMode 1", "NFlavor $F",
            "PartonBlockUpdateSize $block_update",
            "NVMCWarmUp $nvmc_warmup", "NVMCInterval 1", "NVMCSample $nvmc_sample",
            "NSROptItrStep $nsr_step", "NSROptItrSmp $nsr_smp",
            "DSROptStepDt $dt", "DSROptStaDel $sta_del", "DSROptRedCut $red_cut",
            "ComplexType 1", "2Sz 0", "NExUpdatePath 6", "RndSeed $seed",
            "NMPTrans $n_mp_trans",
        ]
            println(io, line)
        end
    end

    write(joinpath(dir, "namelist.def"), join(entries, "\n") * "\n")
    return joinpath(dir, "namelist.def"), fx
end
