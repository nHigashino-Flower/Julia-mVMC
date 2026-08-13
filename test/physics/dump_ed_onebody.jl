#=
外部 ED 実装から 1 粒子ハミルトニアン・ボンド列・スペクトルを吸い出す生成スクリプト。

- 外部ファイルは include するだけで一切編集しない
- Julia 1.8 + @v1.8 環境で 1 回だけ手動実行し、出力を test/physics/ にコミットする
  (以後のテストはこのダンプを参照値として使う)

使い方:
  /opt/julia-1.8.5/bin/julia --project=@v1.8 dump_ed_onebody.jl <出力ディレクトリ>
=#

include("/home/nozomihigashino/K-S-Model/ParentCode/ModuleParentHamCB.jl")
using .ED_Hamiltnian_CB
using LinearAlgebra, SparseArrays, Printf

const T   = 1.0
const T1  = 0.2928932188134525
const T2  = -0.2928932188134525
const T3  = 0.20710678118654754
const PSI = 0.7853981633974483

"1 粒子セクターの Nsite×Nsite ハミルトニアンを ED 実装から得る。"
function onebody_from_ed(Nx, Ny, statistics)
    Nsite = 2 * Nx * Ny
    sp1 = SystemParams1(Nx, Ny, 1, "periodic", statistics)
    sp2 = SystemParams2(; t = T, t1 = T1, t2 = T2, t3 = T3, ψ = PSI,
                        R = zeros(Float64, Nsite), EFlist = zeros(Float64, Nsite))
    ip = IntrctParams(; U = 0.0, V = 0.0)
    H = Matrix(make_CB_hamiltonian(sp1, sp2, ip))
    # basis_list は整数昇順 = 1 粒子なら site 0,1,2,... の順
    @assert size(H) == (Nsite, Nsite)
    return H, SetBHFL(sp1)
end

function dump_case(outdir, tag, Nx, Ny, statistics)
    Nsite = 2 * Nx * Ny
    H, bhfl = onebody_from_ed(Nx, Ny, statistics)
    herm = maximum(abs.(H - H'))
    ev = eigvals(Hermitian((H + H') / 2))

    open(joinpath(outdir, "ed_onebody_$(tag).dat"), "w") do io
        println(io, "# 外部 ED 実装(ModuleParentHamCB.jl)から生成。手で編集しないこと。")
        println(io, "# generator: dump_ed_onebody.jl  julia $(VERSION)")
        @printf(io, "Nx %d\nNy %d\nNsite %d\nstatistics %s\nbc periodic\n",
                Nx, Ny, Nsite, statistics)
        @printf(io, "t %.17g\nt1 %.17g\nt2 %.17g\nt3 %.17g\npsi %.17g\n", T, T1, T2, T3, PSI)
        @printf(io, "hermiticity_residual %.3e\n", herm)
        # 1 粒子ハミルトニアンの非ゼロ要素(行 col row = t、いずれも 0-based サイト)
        println(io, "# ONEBODY  site_row site_col Re Im   (H[row,col], 0-based)")
        n_nz = 0
        for c = 1:Nsite, r = 1:Nsite
            abs(H[r, c]) > 1e-14 || continue
            n_nz += 1
        end
        println(io, "NONZERO $n_nz")
        for c = 1:Nsite, r = 1:Nsite
            abs(H[r, c]) > 1e-14 || continue
            @printf(io, "ONEBODY %d %d %.17g %.17g\n", r - 1, c - 1, real(H[r, c]), imag(H[r, c]))
        end
        # ボンド列(片方向。ED の generate_bonds_cb がそのまま返すもの)
        println(io, "# BOND  bond1 bond2 hop_x hop_y   (0-based サイト、hop は bond1→bond2)")
        println(io, "NBOND $(length(bhfl.bonds_1))")
        for i in eachindex(bhfl.bonds_1)
            @printf(io, "BOND %d %d %d %d\n",
                    bhfl.bonds_1[i], bhfl.bonds_2[i], bhfl.hopping_x[i], bhfl.hopping_y[i])
        end
        println(io, "# EIGVAL  1 粒子スペクトル(昇順)")
        println(io, "NEIGVAL $(length(ev))")
        for e in ev
            @printf(io, "EIGVAL %.17g\n", e)
        end
    end
    @printf("%-10s Nsite=%2d  nnz=%d  bonds=%d  herm_res=%.2e  E1min=%.15g\n",
            tag, Nsite, count(x -> abs(x) > 1e-14, H), length(bhfl.bonds_1), herm, ev[1])
end

outdir = length(ARGS) >= 1 ? ARGS[1] : "."
mkpath(outdir)
dump_case(outdir, "boson_nu12_4x4", 4, 4, "Boson")
dump_case(outdir, "fermion_nu13_5x3", 5, 3, "Fermion")
println("wrote to ", outdir)
