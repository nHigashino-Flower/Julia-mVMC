#=
Chern 3 経路(平滑 k / native k / Bott)の検証
--- parton-mode (fork addition) ---

    julia --project=tools tools/parton_chern_validate.jl

**答えの分かっている系**で 3 経路を突き合わせ、符号規約と実装を固定する。

ケース 1(基準): 素の CheckerBoard 模型で**下 Chern バンドを完全充填**
  (Ne = Nux·Nuy = 64、ν = 1)。この模型の下バンドの Chern 数は ±1 と分かっている。
  拡大セルを (2,2) に取っても (2,1) に取っても、折り畳まれた副バンドの総和は
  同じ値でなければならない(= 折り畳みに対する不変性の検査も兼ねる)。

ケース 2: 旧 run の最適 α を正準向きに組み直した共変 H(ν = 1/2、Ne = 32)。
  ギャップが開いた実データで 3 経路が一致するかを見る。
=#

include(joinpath(@__DIR__, "parton_bands.jl"))
include(joinpath(@__DIR__, "..", "test", "physics", "checkerboard_model.jl"))

using .PartonBands
using LinearAlgebra, Printf
const B = PartonBands

"3 経路をまとめて評価する。"
function three_routes(H::Matrix{ComplexF64}, occ::Vector{Int},
                      nx::Int, ny::Int, ex::Int, ey::Int; grid::Int = 32)
    n1, n2 = div(nx, ex), div(ny, ey)
    n_occ = div(length(occ), n1 * n2)
    h, resid = B.fold_hoppings(H, nx, ny, ex, ey)
    e = eigvals(Hermitian(H))
    gap = e[length(occ) + 1] - e[length(occ)]
    A = B.fhs_chern_bands(h, collect(1:n_occ); n1 = grid, n2 = grid)
    Bp = B.fhs_chern_projector(H, occ, nx, ny, ex, ey)
    C = B.bott_index(H, occ, nx, ny, ex, ey)
    return (resid = resid, gap = gap, n_occ = n_occ,
            A = A.C, A_link = A.min_abs_link_det,
            Bc = Bp.C, B_perr = Bp.max_projector_error, B_grid = (n1, n2),
            Cb = C.B_vuvu, Cb2 = C.B_uvuv, comm = C.commutator)
end

function report(label, r; grid)
    @printf("  %-22s 残差 %.1e  gap %+.4f  占有 %d バンド/k\n",
            label, r.resid, r.gap, r.n_occ)
    @printf("      A 平滑(%d×%d)   C = %+.6f   (min|link| %.2e)\n", grid, grid, r.A, r.A_link)
    @printf("      B native(%d×%d)  C = %+.6f   (max‖p²−p‖ %.2e %s)\n",
            r.B_grid[1], r.B_grid[2], r.Bc, r.B_perr,
            r.B_perr < 0.05 ? "OK" : "**閾値超**")
    @printf("      C Bott          B_vuvu = %+.6f  B_uvuv = %+.6f  (‖[Ũ,Ṽ]‖ %.2e)\n",
            r.Cb, r.Cb2, r.comm)
    d = maximum(abs.([r.A - r.Bc, r.A - r.Cb, r.Bc - r.Cb]))
    @printf("      → 最大差 %.2e : %s\n\n", d, d < 0.05 ? "**3 経路一致**" : "**不一致**")
    return d < 0.05
end

# =========================================================================
println("="^72)
println("ケース 1: 素の CheckerBoard 模型、下 Chern バンドを完全充填(ν = 1)")
println("          → 既知の答え |C| = 1。折り畳み方に依らないはず")
println("="^72)
nx = ny = 8
nsite = 2 * nx * ny
Hbare = Matrix{ComplexF64}(cb_onebody(nx, ny))
ne_full = nx * ny                       # = 64、下バンドの全状態
occ_full = collect(0:ne_full-1)
e = eigvals(Hermitian(Hbare))
@printf("下バンド [%.4f, %.4f] / 上バンド [%.4f, %.4f] / バンド間ギャップ %.4f\n\n",
        e[1], e[ne_full], e[ne_full+1], e[end], e[ne_full+1] - e[ne_full])
ok1 = true
for (nm, ex, ey) in (("ef4 (2,2)", 2, 2), ("xexet2 (2,1)", 2, 1), ("基本セル (1,1)", 1, 1))
    r = three_routes(Hbare, occ_full, nx, ny, ex, ey; grid = 32)
    global ok1 &= report(nm, r; grid = 32)
end

# =========================================================================
println("="^72)
println("ケース 2: 旧 run の α を正準向きに組み直した共変 H(ν = 1/2、Ne = 32)")
println("="^72)
cls(s) = (b = B.site_uc(s, nx); (mod(b[1],2), mod(b[2],2), b[3]))
function disp(i, j)
    xi, yi = B.cb_site_to_xy(i, nx); xj, yj = B.cb_site_to_xy(j, nx)
    dx = mod(xj-xi+nx, 2nx); dx > nx && (dx -= 2nx)
    dy = mod(yj-yi+ny, 2ny); dy > ny && (dy -= 2ny)
    (dx, dy)
end
function build_canonical(dir)
    rows = Tuple{Int,Int,Int,Int}[]
    for ln in readlines(joinpath(dir,"stage1_in","pmfpara.def"))[6:end]
        t = split(ln); length(t)==5 || continue
        push!(rows, (parse(Int,t[1]), parse(Int,t[3]), parse(Int,t[2]), parse(Int,t[5])))
    end
    tval = Dict{NTuple{3,Int},ComplexF64}()
    for ln in readlines(joinpath(dir,"stage1_in","pmftrans.def"))[6:end]
        t = split(ln); length(t)==6 || continue
        tval[(parse(Int,t[1]),parse(Int,t[3]),parse(Int,t[2]))] =
            complex(parse(Float64,t[5]), parse(Float64,t[6]))
    end
    vals = Dict{Int,ComplexF64}()
    for ln in readlines(joinpath(dir,"stage1_out","zqp_pmfpara_opt.dat"))[6:end]
        t = split(ln); length(t)>=3 || continue
        vals[parse(Int,t[1])] = complex(parse(Float64,t[2]), parse(Float64,t[3]))
    end
    H = [zeros(ComplexF64,nsite,nsite) for _ in 1:2]
    for (s1,s2,f,k) in rows
        tv = tval[(s1,s2,f)]; α = vals[k]
        d21 = disp(s2,s1)
        flip = (cls(s2), d21...) > (cls(s1), (.-d21)...)
        a,b,v = flip ? (s2,s1,α*conj(tv)) : (s1,s2,α*tv)
        H[f+1][a+1,b+1] += v
        a != b && (H[f+1][b+1,a+1] += conj(v))
    end
    H
end
ok2 = true
for seed in ("1008", "1004", "1003")
    dir = "playground_nozomi/cb_nu12_boson/runs/L08_ef4_s$(seed)"
    isdir(dir) || continue
    H = build_canonical(dir)
    for f in 1:2
        r = three_routes(H[f], collect(0:31), nx, ny, 2, 2; grid = 32)
        global ok2 &= report("s$seed flavor $(f-1)", r; grid = 32)
    end
end

println("="^72)
@printf("ケース 1(既知の答え): %s\n", ok1 ? "3 経路一致" : "**不一致あり**")
@printf("ケース 2(実データ)  : %s\n", ok2 ? "3 経路一致" : "**不一致あり**")
