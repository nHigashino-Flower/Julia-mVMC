#=
Kapit-Mueller 模型の 1 粒子ハミルトニアン・スペクトルを ED 実装から吸い出す生成スクリプト。

- 参照は ED リポジトリの新モジュール
  `/home/nozomihigashino/ED/Code/test/ExactDiagonalization/`(module `ExactDiagonalization`)
- そちらは JLD2 / Arpack の都合で julia 1.8.5 でしか走らないので、ここで 1 回だけ
  手動実行し、出力を `test/physics/ed_dump/` にコミットする。以後のテスト
  (`test_km_model.jl`)はこのダンプを参照値として使う

使い方:
  /opt/julia-1.8.5/bin/julia dump_ed_onebody_km.jl <出力ディレクトリ>

**mVMC 側(`lattice_model.jl` の `KMLatticeModel`)と ED 側(`coupling.jl` の `tij`)は
独立に書かれている**ので、この照合は 2 実装のクロスチェックとして機能する。
規約差は静かに間違ったエネルギーを出すので、ここを「それらしく」埋めないこと。
=#

include("/home/nozomihigashino/ED/Code/test/ExactDiagonalization/src/ExactDiagonalization.jl")
using .ExactDiagonalization
using LinearAlgebra, Printf

# (tag, Lx, Ly, φ, F, N, 統計, 全行列をダンプするか)
const CASES = [
    ("km_nu12_4x4", 4, 4, 1 // 2, 2, 4, "HardcoreBoson", true),
    ("km_nu12_4x6", 4, 6, 1 // 2, 2, 6, "HardcoreBoson", false),
    ("km_nu13_9x4", 9, 4, 1 // 3, 3, 4, "Fermion", true),
    ("km_nu13_9x5", 9, 5, 1 // 3, 3, 5, "Fermion", false),
]
const HOPMAXES = [2.0, 8.0]

onebody(lat, model) = [ExactDiagonalization.tij(model, lat, i, j)
                       for i = 0:n_site(lat)-1, j = 0:n_site(lat)-1]

function dump_case(outdir, tag, Lx, Ly, φ, F, N, stat, full)
    lat = KapitMuellerLattice(Lx, Ly)
    open(joinpath(outdir, "ed_onebody_$(tag).dat"), "w") do io
        println(io, "# 外部 ED 実装(ExactDiagonalization/coupling.jl の tij)から生成。手で編集しないこと。")
        println(io, "# generator: dump_ed_onebody_km.jl  julia $(VERSION)")
        @printf(io, "Lx %d\nLy %d\nNsite %d\nphi %.17g\nF %d\nNelec %d\nstatistics %s\n",
                Lx, Ly, n_site(lat), float(φ), F, N, stat)
        println(io, "gauge landau")
        println(io, "bonds displacement_sum")
        for hm in HOPMAXES
            for (kind, ff) in (("phys", float(φ)), ("mf", float(φ) / F))
                m = KapitMuellerModel(ff; hopmax = hm)
                H = onebody(lat, m)
                ev = eigvals(Hermitian((H + H') / 2))
                @printf(io, "BLOCK %s hopmax %.17g flux %.17g\n", kind, hm, ff)
                @printf(io, "HERMRES %.3e\n", maximum(abs.(H - H')))
                if full
                    nz = count(x -> abs(x) > 1e-14, H)
                    println(io, "NONZERO $nz")
                    for c = 1:n_site(lat), r = 1:n_site(lat)
                        abs(H[r, c]) > 1e-14 || continue
                        @printf(io, "ONEBODY %d %d %.17g %.17g\n",
                                r - 1, c - 1, real(H[r, c]), imag(H[r, c]))
                    end
                else
                    println(io, "NONZERO 0")
                end
                println(io, "NEIGVAL $(length(ev))")
                for e in ev
                    @printf(io, "EIGVAL %.17g\n", e)
                end
                println(io, "ENDBLOCK")
            end
        end
    end
    @printf("%-14s Lx=%d Ly=%d Nsite=%2d  全行列=%s\n", tag, Lx, Ly, n_site(lat), full)
end

outdir = length(ARGS) >= 1 ? ARGS[1] : "."
mkpath(outdir)
for (tag, Lx, Ly, φ, F, N, stat, full) in CASES
    dump_case(outdir, tag, Lx, Ly, φ, F, N, stat, full)
end
println("wrote to ", outdir)
