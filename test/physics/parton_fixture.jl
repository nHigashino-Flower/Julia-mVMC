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
"""
function bond_class(i::Int, j::Int, dx::Int, dy::Int,
                    nx::Int, ny::Int, ex::Int, ey::Int)
    ki = (enlarged_cell_class(i, nx, ny, ex, ey), dx, dy)
    kj = (enlarged_cell_class(j, nx, ny, ex, ey), -dx, -dy)
    return min(ki, kj)
end


"""
    _idx_key(mode, orbit_key, bond) / _flavor_key(mode, key, f)

変分パラメータ idx の粒度を決める(候補 C の検証用)。参照実装 PartonFCI の
`symmetry_mode` に対応させてある:

- `:orbit`        拡大セル軌道で縮約 + フレーバー共有 (= SU(K)_UNIFIED 相当)
- `:orbit_flavor` 拡大セル軌道で縮約 + フレーバー独立
- `:bond_flavor`  ボンドごと + フレーバー独立 (= SU(K)_INDEPENDENT 相当)

`:orbit` が既定で、P1〜P3 のこれまでの結果はすべてこれ。
"""
_idx_key(mode::Symbol, orbit_key, bond) =
    mode === :bond_flavor ? (:bond, bond) : orbit_key

_flavor_key(mode::Symbol, key, f::Int) = mode === :orbit ? key : (key, f)

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
                        idx_mode::Symbol = :orbit,
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
        key = _idx_key(idx_mode, bond_class(i, j, dx, dy, nx, ny, ex, ey), (i, j))
        # pmftrans は `H[site1, site2] += α·value` と読まれる = c†_{site1} c_{site2}
        # の係数。こちらの t は c†_j c_i の係数なので (site1, site2) = (j, i) で出す。
        # physhop の (site1, site2, value) は b†_{site2} b_{site1} の係数なので
        # 向きが逆になる — 2 つの新設形式で添字の向きが違うことに注意。
        for f = 0:(nflavor - 1)
            idx = get!(class_of_idx, _flavor_key(idx_mode, key, f),
                       length(class_of_idx))
            push!(pmftrans, (ComplexF64(j), ComplexF64(f), ComplexF64(i),
                             ComplexF64(f), coeff))
            push!(pmfpara, (j, f, i, f, idx, ComplexF64(1, 0)))
        end
    end

    # --- 平均場: オンサイト(対角 = U のみ)。u_mf = 0 なら入れない ---
    if u_mf != 0.0
        for i = 0:(nsite - 1)
            key = _idx_key(idx_mode, (enlarged_cell_class(i, nx, ny, ex, ey), 0, 0),
                           (i, i))
            for f = 0:(nflavor - 1)
                idx = get!(class_of_idx, _flavor_key(idx_mode, key, f),
                           length(class_of_idx))
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
                                red_cut::Float64 = 1e-8,
                                block_update::Int = 16,
                                seed::Int = 11272,
                                idx_mode::Symbol = :orbit,
                                opt_flags::Union{Nothing,Dict{Int,Int}} = nothing,
                                qp_momentum::Union{Nothing,Tuple{Int,Int}} = nothing,
                                qp_xext::Union{Nothing,Tuple{Int,Int}} = nothing)
    mkpath(dir)
    fx = parton_fixture(nx, ny, F, ex, ey; u_mf = u_mf, idx_mode = idx_mode)
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
        for (a, b, c, d, i, _) in fx.pmfpara
            @printf(io, "%d %d %d %d %d\n", a, b, c, d, i)
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
