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
function parton_fixture(model::PartonLatticeModel, nflavor::Int, ex::Int, ey::Int;
                        u_mf::Float64 = 0.0,
                        u_bonds::Symbol = :nn,
                        idx_mode::Symbol = :orbit,
                        flavor_groups::Union{Nothing,Vector{Int}} = nothing,
                        psg_onsite::Bool = false,
                        psg_shells::Vector{Int} = Int[],
                        graph::Symbol = :model)
    psg_onsite && u_mf != 0.0 &&
        error("psg_onsite と u_mf != 0 は同じ (i,i) 行を二重に作るので併用不可")
    graph in (:model, :full) || error("graph は :model / :full。graph = $(graph)")
    if graph === :full
        (u_mf != 0.0 || psg_onsite || !isempty(psg_shells)) && error(
            "graph = :full は全サイト対を自前で並べるので u_mf / psg_onsite / " *
            "psg_shells とは併用しません")
    end
    isempty(psg_shells) || model isa CheckerboardLatticeModel || error(
        "psg_shells は checkerboard 専用です(cb_psg_extra_bonds の d² 規約に依存)")
    if flavor_groups !== nothing
        length(flavor_groups) == nflavor || error(
            "flavor_groups の長さ $(length(flavor_groups)) が NFlavor = $(nflavor) と違います")
        all(0 .<= flavor_groups .< nflavor) || error(
            "flavor_groups の値は 0..$(nflavor-1): $(flavor_groups)")
        idx_mode === :bond_flavor && error(
            "flavor_groups と idx_mode = :bond_flavor は併用しません")
    end
    nsite = pl_nsite(model)
    bonds = pl_bonds(model)
    # 平均場が保つ並進の置換。idx クラスはこの群の**軌道そのもの**で定義する。
    mf_perms = [pl_shift_perm(model, tx, ty) for (tx, ty) in pl_mf_shifts(model, ex, ey)]
    # 平均場の並進が両端点を入れ替えるペア(`pl_swapped_pairs` の docstring 参照)。
    # α が実数でないと並進共変にできないが、コアの虚部強制凍結はオンサイトしか
    # 見ないので、**そのクラスの α を丸ごと凍結**して扱う(係数は正しい値のまま
    # 残るので平均場は動かない)。graph = :full は係数が全部 1 なので、従来どおり
    # ペアごと除外する方が変分空間を歪めない。
    swapped = pl_swapped_pairs(model, ex, ey)
    n_swapped_dropped = 0
    mf_bonds = if graph === :full
        kept = [b for b in pl_all_pairs(model) if !(minmax(b[1], b[2]) in swapped)]
        n_swapped_dropped = length(swapped)
        kept
    else
        bonds
    end

    # --- 平均場: ボンドを拡大セルの並進クラスへ落として idx を振る ---
    class_of_idx = Dict{Any,Int}()
    pmftrans = NTuple{5,ComplexF64}[]
    pmfpara = Tuple{Int,Int,Int,Int,Int,ComplexF64}[]
    frozen_idx = Set{Int}()      # 自己交換クラス(α を実数に縛る代わりに凍結する)

    for (i, j, dx, dy, d2) in mf_bonds
        # --- クラス代表と向きの正準化 ----------------------------------------
        # クラス鍵は**有向ボンドの軌道**(平均場の並進群で写した先の集合)の最小元。
        #
        # 旧実装は `(拡大セルクラス, dx, dy)` を鍵にしていたが、トーラス上で同じ
        # 端点対に複数の変位が落ちる場合(4×4 の (±2,±2) は四重縮退)、並進で
        # 関係するペアに**別の代表変位が割り当たり**、同じ軌道が 2 つのクラスへ
        # 割れて並進共変性が静かに破れる(4×4 KM で残差 0.0018 を実測)。
        # 軌道そのものを鍵にすればこの縮退に依らない。cb の模型グラフでは
        # 変位が一意なので分割は旧実装と同一(P 層テストがビット一致を守る)。
        #
        # 向きも代表に揃える(2026-08-18 のバグ修正)。係数 t を i < j のリスト
        # 向きのまま書くと、並進コピーで端点の大小が入れ替わったボンドが t の
        # 共役側で載り、同一 idx に α·t と α·conj(t) が同居して **α が複素だと
        # 並進が破れる**。α = 1(実数)では h.c. と合流するので初期 H だけを見る
        # 検査では発見できない。
        orbit_ij = Set((p[i + 1], p[j + 1]) for p in mf_perms)
        orbit_ji = Set((p[j + 1], p[i + 1]) for p in mf_perms)
        rep_ij, rep_ji = minimum(orbit_ij), minimum(orbit_ji)
        # 軌道が逆向きも含む = 並進がこのボンドを向き反転で自分自身へ写す
        # → α·t = conj(α·t) が必要 = α は実数(`pl_swapped_pairs` の docstring)
        self_conj = rep_ij in orbit_ji
        if rep_ij <= rep_ji
            a, b, da, db = i, j, dx, dy      # ホップ a → b(向きは i → j のまま)
            orbit_key = rep_ij
        else
            a, b, da, db = j, i, -dx, -dy    # 反転: j → i を正準に採る
            orbit_key = rep_ji
        end
        if graph === :full
            coeff = ComplexF64(1)      # 全 1。t の距離制限を外し α に全部持たせる
        else
            t = pl_mf_hopping(model, a, b, da, db, nflavor)   # b†_b b_a の係数
            # 内部検査: 反転はエルミート共役に一致するはず(モデル側の規約が
            # 破れたらここで気づく)
            tij = pl_mf_hopping(model, i, j, dx, dy, nflavor)
            abs(t - (rep_ij <= rep_ji ? tij : conj(tij))) < 1e-12 ||
                error("pl_mf_hopping is not Hermitian for bond ($i, $j, $dx, $dy)")
            # 非対角の平均場係数 = t + U(支給仕様、係数 1)。U を載せるボンドは
            # `u_bonds` で選ぶ(:nn なら最近接のみ、:all なら全ボンド)。
            nn_d2 = minimum(bb[5] for bb in bonds)
            add_u = (u_mf != 0.0) && (u_bonds === :all || d2 == nn_d2)
            coeff = t + (add_u ? u_mf : 0.0)
        end
        key = _idx_key(idx_mode, orbit_key, (i, j))
        is_swapped = self_conj
        # pmftrans は `H[site1, site2] += α·value` と読まれる = c†_{site1} c_{site2}
        # の係数。こちらの t は c†_b c_a の係数なので (site1, site2) = (b, a) で出す。
        # physhop の (site1, site2, value) は b†_{site2} b_{site1} の係数なので
        # 向きが逆になる — 2 つの新設形式で添字の向きが違うことに注意。
        for f = 0:(nflavor - 1)
            idx = get!(class_of_idx, _flavor_key(idx_mode, key, f, flavor_groups),
                       length(class_of_idx))
            is_swapped && push!(frozen_idx, idx)
            push!(pmftrans, (ComplexF64(b), ComplexF64(f), ComplexF64(a),
                             ComplexF64(f), coeff))
            push!(pmfpara, (b, f, a, f, idx, ComplexF64(1, 0)))
        end
    end

    # --- 平均場: オンサイト。u_mf(ν=1/3 の Hartree)/ 模型由来(KM の自己像)/
    #     graph = :full のいずれかがあるとき。
    # オンサイトは並進対称群の軌道ごとに値が違いうる(KM の自己像は Landau 位相が
    # x に依存するので x 周期 q で変わる)。全サイトを見て判定すること。
    mf_onsites = [pl_mf_onsite(model, i, nflavor) for i = 0:(nsite - 1)]
    all(abs(imag(v)) < 1e-12 for v in mf_onsites) || error(
        "平均場のオンサイト項が複素です(pmftrans のオンサイトは実数必須)")
    has_model_onsite = maximum(abs, mf_onsites; init = 0.0) > 1e-14
    if u_mf != 0.0 || has_model_onsite || graph === :full
        for i = 0:(nsite - 1)
            onsite_coeff = graph === :full ? ComplexF64(1) :
                           ComplexF64(u_mf + real(mf_onsites[i + 1]))
            key = _idx_key(idx_mode, (:onsite, minimum(p[i + 1] for p in mf_perms)), (i, i))
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
            key = _idx_key(idx_mode, (:onsite, minimum(p[i + 1] for p in mf_perms)), (i, i))
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
    if !isempty(psg_shells)
        for (i, j, dx, dy, _) in cb_psg_extra_bonds(model.nx, model.ny, psg_shells)
            orbit_ij = Set((p[i + 1], p[j + 1]) for p in mf_perms)
            orbit_ji = Set((p[j + 1], p[i + 1]) for p in mf_perms)
            rep_ij, rep_ji = minimum(orbit_ij), minimum(orbit_ji)
            a, b = rep_ij <= rep_ji ? (i, j) : (j, i)
            key = _idx_key(idx_mode, min(rep_ij, rep_ji), (i, j))
            for f = 0:(nflavor - 1)
                idx = get!(class_of_idx, _flavor_key(idx_mode, key, f, flavor_groups),
                           length(class_of_idx))
                push!(psg_idx, idx)
                push!(pmftrans, (ComplexF64(b), ComplexF64(f), ComplexF64(a),
                                 ComplexF64(f), ComplexF64(1)))
                push!(pmfpara, (b, f, a, f, idx, ComplexF64(0, 0)))
            end
        end
    end

    # --- 物理ハミルトニアンの合成ホップ: t_ij をそのまま(片方向) ---
    physhop = Tuple{Int,Int,ComplexF64}[]
    for (i, j, dx, dy, _) in bonds
        push!(physhop, (i, j, pl_hopping(model, i, j, dx, dy)))
    end
    # 物理のオンサイト項(KM のトーラス自己像)。physhop.def は site1 == site2 を
    # 禁じている(暗黙 h.c. + 両向き評価で二重計上する)ので、DESIGN §2.4 のとおり
    # **coulombinter の対角行**(硬芯なら V n_i² = V n_i)として渡す。
    phys_onsite = [(i, real(pl_onsite(model, i))) for i = 0:(nsite - 1)]
    all(abs(imag(pl_onsite(model, i))) < 1e-12 for i = 0:(nsite - 1)) ||
        error("物理のオンサイト項が複素です(エルミート性が壊れています)")

    # --- 図示・診断用の幾何メタ(コアは読まない。tools/plot_parton_lattice.jl 用)---
    # サイトの EF クラス = 平均場の並進群の軌道。0..K-1 に詰め直す。
    site_orbit_rep = [minimum(p[i + 1] for p in mf_perms) for i = 0:(nsite - 1)]
    ef_of_rep = Dict{Int,Int}()
    site_ef = Vector{Int}(undef, nsite)
    for i = 0:(nsite - 1)
        site_ef[i + 1] = get!(ef_of_rep, site_orbit_rep[i + 1], length(ef_of_rep))
    end

    return (
        pmftrans = [(Int(real(a)), Int(real(b)), Int(real(c)), Int(real(d)), v)
                    for (a, b, c, d, v) in pmftrans],
        pmfpara = pmfpara,
        physhop = physhop,
        phys_onsite = phys_onsite,
        n_idx = length(class_of_idx),
        nsite = nsite,
        psg_idx = psg_idx,
        frozen_idx = frozen_idx,
        n_swapped_dropped = n_swapped_dropped,
        model = model,
        site_ef = site_ef,
        n_ef = length(ef_of_rep),
        ex = ex,
        ey = ey,
    )
end

"""
    parton_fixture(nx, ny, nflavor, ex, ey; ...)

checkerboard 向けの後方互換ラッパ。既存の呼び出しはこちらに落ちる。
"""
parton_fixture(nx::Int, ny::Int, nflavor::Int, ex::Int, ey::Int;
               p::CheckerboardParams = CheckerboardParams(), kwargs...) =
    parton_fixture(CheckerboardLatticeModel(nx, ny; p = p), nflavor, ex, ey; kwargs...)

"""
    physical_coulomb(model, u) -> Vector{Tuple{Int,Int,Float64}}

物理ハミルトニアンの密度型: 模型の最近接ペアに U。素の値を入れる
(上の「規格化について」を参照)。V = 0 なので 2 次近接は入れない。
"""
function physical_coulomb(model::PartonLatticeModel, u::Float64)
    out = Tuple{Int,Int,Float64}[]
    u == 0.0 || append!(out, [(i, j, u) for (i, j) in pl_nn_pairs(model)])
    # 模型由来のオンサイト項は対角行 (i, i, ε₀) として載せる(DESIGN §2.4)
    for i = 0:(pl_nsite(model) - 1)
        e0 = real(pl_onsite(model, i))
        abs(e0) > 1e-14 && push!(out, (i, i, e0))
    end
    return out
end

physical_coulomb(nx::Int, ny::Int, u::Float64) =
    physical_coulomb(CheckerboardLatticeModel(nx, ny), u)

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
function write_parton_def_files(dir::AbstractString, model::PartonLatticeModel, F::Int,
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
                                alpha_init::Symbol = :random,
                                opt_flags::Union{Nothing,Dict{Int,Int}} = nothing,
                                qp_momentum::Union{Nothing,Tuple{Int,Int}} = nothing,
                                qp_xext::Union{Nothing,Tuple{Int,Int}} = nothing,
                                qp_auto::Union{Nothing,Tuple{Float64,Float64}} = nothing,
                                jastrow_full_trans::Bool = false)
    mkpath(dir)
    fx = parton_fixture(model, F, ex, ey; u_mf = u_mf, idx_mode = idx_mode,
                        flavor_groups = flavor_groups,
                        psg_onsite = psg_onsite, psg_shells = psg_shells,
                        graph = graph)
    cb = physical_coulomb(model, u_phys)

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
        alpha_init in (:random, :one) ||
            error("alpha_init は :random / :one。alpha_init = $(alpha_init)")
        for (a, b, c, d, i, _) in fx.pmfpara
            if i in fx.psg_idx
                @printf(io, "%d %d %d %d %d % .18e % .18e\n", a, b, c, d, i, 0.0, 0.0)
            elseif alpha_init === :one
                # α = 1 明示。pmftrans が平均場 t をそのまま持つので、初期状態が
                # そのまま「正しい位相の平均場」になる(KM のパートン構成では
                # 最低 C=1 バンドがちょうど充填され、占有集合が一意に決まる)。
                @printf(io, "%d %d %d %d %d % .18e % .18e\n", a, b, c, d, i, 1.0, 0.0)
            else
                @printf(io, "%d %d %d %d %d\n", a, b, c, d, i)
            end
        end
        # 末尾フラグ行は全 idx を明示的に書く(既定は 1 = 可動)。
        # v3.2 以降 OptFlag はゲージ目的では使わず、用途はエルミート性
        # (オンサイト群 Im の強制凍結。コード側が自動でやるのでここには書かない)と
        # ユーザーの明示的固定に限られる。値を省略せず並べておくことで、
        # 「何も固定していない」ことがファイル自身から読み取れるようにする。
        # 自己交換クラス(平均場の並進が端点を入れ替えるペア)は α が実数でないと
        # 並進共変にできない。コアはオンサイトしか虚部を凍結しないので、ここで
        # **クラスごと凍結**する(係数は正しい値のままなので平均場は動かない)。
        # 詳細は lattice_model.jl の `pl_swapped_pairs`。
        flags = copy(opt_flags === nothing ? Dict{Int,Int}() : opt_flags)
        for i in fx.frozen_idx
            haskey(flags, i) || (flags[i] = 0)
        end
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
        # **物理 H を保つ並進**の軌道で類別する(P_J が物理の並進と可換 = 運動量
        # 射影の外側にかけても |ψ⟩ が運動量固有状態のままでいられる条件)。
        # checkerboard は正味フラックス 0 なので全並進、KM は磁気並進 ⟨T_x^q, T_y⟩。
        maps = [pl_shift_perm(model, tx, ty) for (tx, ty) in pl_physical_shifts(model)]
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
    if qp_momentum !== nothing || qp_xext !== nothing || qp_auto !== nothing
        count(!isnothing, (qp_momentum, qp_xext, qp_auto)) == 1 ||
            error("qp_momentum / qp_xext / qp_auto は同時に 1 つだけ指定すること")
        # `qp_auto = (kx, ky)` が模型一般の構成 — 「物理の並進群 ÷ 平均場の並進群」の
        # 剰余代表を張る(`pl_qp_shifts`)。重みは k·(セル座標)。
        # `qp_xext = (kext, nkx)` は checkerboard の参照実装 make_QNPidx 準拠、
        # `qp_momentum` は全並進 nx·ny 本(射影の数学的性質の検証用)。
        sx, sy = pl_cell_step(model)
        lx, ly = pl_grid(model)
        if qp_auto !== nothing
            shifts = pl_qp_shifts(model, ex, ey)
            maps = [pl_shift_perm(model, tx, ty) for (tx, ty) in shifts]
            ucs = [(div(tx, sx), div(ty, sy)) for (tx, ty) in shifts]
            nkx, nky = qp_auto
            n_kx, n_ky = div(lx, sx), div(ly, sy)
        elseif qp_xext === nothing
            shifts = pl_physical_shifts(model)
            maps = [pl_shift_perm(model, tx, ty) for (tx, ty) in shifts]
            ucs = [(div(tx, sx), div(ty, sy)) for (tx, ty) in shifts]
            nkx, nky = qp_momentum
            n_kx, n_ky = div(lx, sx), div(ly, sy)
        else
            maps, ucs = cb_qp_translations(model.nx, model.ny, qp_xext[1])
            nkx, nky = (qp_xext[2], 0)
            n_kx, n_ky = (qp_xext[1], 1)
        end
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
    write_parton_lattice_meta(joinpath(dir, "parton_lattice.dat"), model, fx, F, ex, ey)
    return joinpath(dir, "namelist.def"), fx
end

"""
    write_parton_lattice_meta(path, model, fx, F, ex, ey)

**アンザッツの幾何をそのまま書き出した副産物**。`namelist.def` に載せないので
コアは一切読まない。`tools/plot_parton_lattice.jl` がこれだけを見て
参照実装の `lattice_EF.pdf` 相当の図を描く(tools が playground や模型実装に
依存しないようにするための橋渡し)。

含めるもの:

| キー | 意味 |
|---|---|
| `SITE isite x y ef` | サイトの格子座標と **EF クラス**(= 平均場の並進群の軌道) |
| `PHYSSHIFT tx ty` | **物理 H** を保つ並進(グリッド変位) |
| `MFSHIFT tx ty` | **平均場**を保つ並進 = 拡大セルの並進 |
| `QPSHIFT tx ty` | QP 射影が張る並進(物理 ÷ 平均場 の剰余代表) |
| `BOND s1 s2 idx d2` | 平均場のボンドと共有 idx(フレーバー 0 のみ、正準向き) |

拡大セルが物理の磁気セルの何倍かは `PHYSSHIFT` と `MFSHIFT` の本数比で読める。
"""
function write_parton_lattice_meta(path::AbstractString, model::PartonLatticeModel,
                                   fx, F::Int, ex::Int, ey::Int)
    lx, ly = pl_grid(model)
    sx, sy = pl_cell_step(model)
    phys = pl_physical_shifts(model)
    mfs = pl_mf_shifts(model, ex, ey)
    qps = pl_qp_shifts(model, ex, ey)
    open(path, "w") do io
        println(io, "# パートン格子とアンザッツの幾何。**コアは読まない**(namelist.def に載せない)。")
        println(io, "# tools/plot_parton_lattice.jl が読んで lattice_EF 相当の図を描く。")
        @printf(io, "model %s\n", string(nameof(typeof(model))))
        @printf(io, "nsite %d\n", fx.nsite)
        @printf(io, "grid %d %d\n", lx, ly)
        @printf(io, "cell_step %d %d\n", sx, sy)
        @printf(io, "enlarged_cell %d %d\n", ex, ey)
        @printf(io, "nflavor %d\n", F)
        @printf(io, "n_idx %d\n", fx.n_idx)
        @printf(io, "n_ef %d\n", fx.n_ef)
        @printf(io, "n_frozen %d\n", length(fx.frozen_idx))
        for i = 0:(fx.nsite - 1)
            x, y = pl_site_to_xy(model, i)
            @printf(io, "SITE %d %d %d %d\n", i, x, y, fx.site_ef[i + 1])
        end
        for (tx, ty) in phys; @printf(io, "PHYSSHIFT %d %d\n", tx, ty); end
        for (tx, ty) in mfs;  @printf(io, "MFSHIFT %d %d\n", tx, ty);  end
        for (tx, ty) in qps;  @printf(io, "QPSHIFT %d %d\n", tx, ty);  end
        for k in sort(collect(fx.frozen_idx)); @printf(io, "FROZEN %d\n", k); end
        for ((s1, f1, s2, _, _), (_, _, _, _, idx, _)) in zip(fx.pmftrans, fx.pmfpara)
            f1 == 0 || continue
            s1 == s2 && continue
            x1, y1 = pl_site_to_xy(model, s1)
            x2, y2 = pl_site_to_xy(model, s2)
            dx = mod(x1 - x2 + lx ÷ 2, lx) - lx ÷ 2
            dy = mod(y1 - y2 + ly ÷ 2, ly) - ly ÷ 2
            @printf(io, "BOND %d %d %d %d\n", s2, s1, idx, dx^2 + dy^2)
        end
    end
    return path
end

"""
    write_parton_def_files(dir, nx, ny, F, ex, ey; ...)

checkerboard 向けの後方互換ラッパ。既存の呼び出しはこちらに落ちる。
"""
write_parton_def_files(dir::AbstractString, nx::Int, ny::Int, F::Int, ex::Int, ey::Int;
                       p::CheckerboardParams = CheckerboardParams(), kwargs...) =
    write_parton_def_files(dir, CheckerboardLatticeModel(nx, ny; p = p), F, ex, ey; kwargs...)
