"""
パートンモード共通の SR ソルバ(DESIGN_parton.md §14)--- parton-mode (fork addition) ---

- SR-CG(NSRCG = 1)と直接法(NSRCG = 0)が α・Φ・Pf の 3 モードで同じ更新を与える
  (同じ乱数列で 2 step。step 0 の E はバイト一致、更新後のパラメータと step 1 の E は CG の打ち切りの範囲で一致)
- ⟨O O†⟩ の BLAS 版(ZHERK + 写し)が上流の 3 重ループと一致する
"""

using Test
using LinearAlgebra
using MVMCOptimizers

const MSR = MVMCOptimizers

_sr_numbers(path) = [parse(Float64, t) for l in readlines(path) if !startswith(l, "#")
                     for t in split(replace(l, "," => " ")) if tryparse(Float64, t) !== nothing]

"modpara.def の step 数を 2 にし、追加の行を足す。"
function _sr_patch_modpara(nl, extra)
    mod = joinpath(dirname(nl), "modpara.def")
    lines = [startswith(l, "NSROptItrStep") ? "NSROptItrStep 2" :
             startswith(l, "NSROptItrSmp") ? "NSROptItrSmp 1" : l for l in readlines(mod)]
    write(mod, join(vcat(lines, extra), "\n") * "\n")
    return nl
end

function _sr_run_pair(make_input, opt_file)
    outs = String[]
    for extra in (String[], ["NSRCG 1", "DSROptCGTol 1e-13"])
        dir = mktempdir()
        nl = _sr_patch_modpara(make_input(dir), extra)
        out = joinpath(dir, "out")
        @test MSR.parton_run_para_opt_from_namelist(nl; output_dir = out) == 0
        push!(outs, out)
    end
    e = [_sr_numbers(joinpath(o, "zvo_out.dat")) for o in outs]
    p = [_sr_numbers(joinpath(o, opt_file)) for o in outs]
    return outs, e, p
end

@testset "SR-CG = 直接法(α・Φ・Pf、同じ乱数列で 2 step)" begin
    cases = (
        ("α", dir -> _write_min_parton_input(dir)[1], "zqp_pmfpara_opt.dat"),
        ("Φ", dir -> _write_min_orbparam_input(dir), "zqp_pmforb_opt.dat"),
        ("Pf", dir -> _write_min_pair_input(dir), "zqp_parton_pair_opt.dat"),
    )
    for (name, make_input, opt_file) in cases
        outs, e, p = _sr_run_pair(make_input, opt_file)
        n_col = length(e[1]) ÷ 2
        # step 0 は同じ状態・同じ乱数列なのでバイト一致
        @test e[1][1:n_col] == e[2][1:n_col]
        # step 1 は更新後の状態。CG の打ち切りの範囲で一致
        @test e[1][n_col + 1] ≈ e[2][n_col + 1] rtol = 1e-10
        @test length(p[1]) == length(p[2])
        @test maximum(abs, p[1] .- p[2]) < 1e-8 * max(1.0, maximum(abs, p[1]))
        @test isfile(joinpath(outs[2], "zvo_SRinfo.dat"))
        @info "SR-CG vs 直接法" name ΔE1 = e[1][n_col + 1] - e[2][n_col + 1] Δp = maximum(abs, p[1] .- p[2])
    end
end

@testset "⟨O O†⟩ の BLAS 版 = 上流の 3 重ループ" begin
    sr_opt_size = 7
    n = 5
    size_2 = 2 * sr_opt_size
    store = ComplexF64[sin(0.3k) + im * cos(0.7k) for k = 1:(size_2 * n)]
    ref = zeros(ComplexF64, size_2 * (size_2 + 2))
    MSR.finalize_oo_store!(ref, store, sr_opt_size, n; nsrcg = false)
    got = fill(ComplexF64(9), size_2 * (size_2 + 2))
    MSR.parton_oo_from_store_blas!(got, store, sr_opt_size, n)
    @test got[1:(size_2^2)] ≈ ref[1:(size_2^2)] rtol = 1e-13
    MSR.parton_oo_from_store_blas!(got, store, sr_opt_size, 0)
    @test all(iszero, got[1:(size_2^2)])
end
