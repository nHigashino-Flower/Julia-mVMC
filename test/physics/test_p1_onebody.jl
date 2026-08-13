"""
P1: 模型照合 — 移植した checkerboard 規約を外部 ED 実装と突き合わせる
--- parton-mode (fork addition) ---

DESIGN_parton.md §8 P 層の P1(規約ゲート)。

フィクスチャ(pmftrans / physhop / coulombinter)は移植した t_ij から組むので、
移植が 1 箇所でも狂うと、E_VMC ≥ E_min のような上界の検査は通ったまま
静かに間違った物理を計算してしまう。ここで止める。

参照値は外部 ED 実装(`ModuleParentHamCB.jl`)そのものから吸い出した 1 粒子
ハミルトニアンで、`test/physics/ed_dump/` にコミットしてある(生成は
`dump_ed_onebody.jl` を Julia 1.8 + @v1.8 環境で 1 回だけ実行)。

不一致は規約差 = エスカレーション対象。許容値を緩めて緑にしないこと。
"""

using Test
using LinearAlgebra

"ダンプファイルを読む。"
function read_ed_onebody_dump(path::AbstractString)
    meta = Dict{String,String}()
    nsite = 0
    H = zeros(ComplexF64, 0, 0)
    bonds = NTuple{4,Int}[]
    eigvals_ref = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        tok = split(s)
        if tok[1] == "ONEBODY"
            r, c = parse(Int, tok[2]) + 1, parse(Int, tok[3]) + 1
            H[r, c] = ComplexF64(parse(Float64, tok[4]), parse(Float64, tok[5]))
        elseif tok[1] == "BOND"
            push!(bonds, (parse(Int, tok[2]), parse(Int, tok[3]),
                          parse(Int, tok[4]), parse(Int, tok[5])))
        elseif tok[1] == "EIGVAL"
            push!(eigvals_ref, parse(Float64, tok[2]))
        elseif tok[1] in ("NONZERO", "NBOND", "NEIGVAL")
            meta[tok[1]] = tok[2]
        else
            meta[tok[1]] = tok[2]
            if tok[1] == "Nsite"
                nsite = parse(Int, tok[2])
                H = zeros(ComplexF64, nsite, nsite)
            end
        end
    end
    return meta, H, bonds, eigvals_ref
end

const ED_DUMP_DIR = joinpath(@__DIR__, "ed_dump")

@testset "P1 移植した checkerboard 規約 vs 外部 ED($tag)" for
        (tag, nx, ny) in (("boson_nu12_4x4", 4, 4), ("fermion_nu13_5x3", 5, 3))
    meta, H_ed, bonds_ed, ev_ed = read_ed_onebody_dump(
        joinpath(ED_DUMP_DIR, "ed_onebody_$(tag).dat"))

    @test parse(Int, meta["Nx"]) == nx
    @test parse(Int, meta["Ny"]) == ny
    nsite = parse(Int, meta["Nsite"])
    @test nsite == 2 * nx * ny

    p = CheckerboardParams(;
        t = parse(Float64, meta["t"]),
        t1 = parse(Float64, meta["t1"]),
        t2 = parse(Float64, meta["t2"]),
        t3 = parse(Float64, meta["t3"]),
        psi = parse(Float64, meta["psi"]),
    )

    # --- 1 粒子ハミルトニアンの全要素一致(規約ゲートの本体) ---
    H_port = cb_onebody(nx, ny, p)
    @test size(H_port) == size(H_ed)
    @test maximum(abs, H_port .- H_ed) < 1e-10

    # エルミート性(ED 側は残差 0 だった)
    @test maximum(abs, H_port .- H_port') < 1e-13

    # --- スペクトル一致 ---
    ev_port = eigvals(Hermitian((H_port + H_port') / 2))
    @test length(ev_port) == length(ev_ed)
    @test maximum(abs, ev_port .- ev_ed) < 1e-10

    # --- ボンドの本数と種別 ---
    bonds_port = cb_undirected_bonds(nx, ny)
    @test length(bonds_port) == length(bonds_ed)
    @test length(bonds_port) == 6 * nsite            # 1 サイトあたり 12 本 / 2
    pairs_port = Set(minmax(b[1], b[2]) for b in bonds_port)
    pairs_ed = Set(minmax(b[1], b[2]) for b in bonds_ed)
    @test pairs_port == pairs_ed

    # 距離の内訳: 最近接 / 2 次 / 3 次 が同数ずつ(各 4 本 × Nsite / 2)
    for d2 in (2, 4, 8)
        @test count(b -> b[5] == d2, bonds_port) == 2 * nsite
    end

    # --- 最低 Chern バンドが分離していること(ansatz の前提) ---
    # 占有すべきは下から Nsite/2 本(= Nx*Ny 本)。そのバンドとの間にギャップ。
    n_band = nx * ny
    gap = ev_port[n_band + 1] - ev_port[n_band]
    @test gap > 0.1
end
