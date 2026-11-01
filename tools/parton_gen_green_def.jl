"""
パートンの物理量測定(NVMCCalMode = 1、DESIGN_parton.md §13)用の greenone.def / greentwo.def を作る
--- parton-mode (fork addition) ---

- greenone.def: 全 (i, j)(i = j を含む Nsite² 行)の ⟨b†_i b_j⟩。n(k) と ODLRO 用
- greentwo.def: 全 (i, i, j, j)(Nsite² 行)の ⟨n_i n_j⟩。S(q) と密度相関用

書式は上流と同じ(5 行ヘッダ、スピン列はすべて 0)。2 体はこの密度相関だけを書く
(1 粒子移動の他の類は必要になったら手で足す。別々の 2 粒子を動かす項は未対応)。

使い方:

    julia tools/parton_gen_green_def.jl <Nsite> <out_dir>

namelist.def には次の 2 行を足す:

    OneBodyG       greenone.def
    TwoBodyG       greentwo.def
"""

"""
    parton_write_green_defs(out_dir, n_site) -> Nothing

`out_dir` に greenone.def(全 (i, j))と greentwo.def(全 (i, i, j, j))を書く。サイトは 0-based。
"""
function parton_write_green_defs(out_dir::AbstractString, n_site::Integer)
    n_site > 0 || throw(ArgumentError("n_site must be positive, got $n_site"))
    mkpath(out_dir)
    header(io, key, n) = begin
        println(io, "===============================")
        println(io, "$key $n")
        println(io, "===============================")
        println(io, "======== Green functions ======")
        println(io, "===============================")
    end
    open(joinpath(out_dir, "greenone.def"), "w") do io
        header(io, "NCisAjs", n_site^2)
        for i = 0:(n_site - 1), j = 0:(n_site - 1)
            println(io, "$i 0 $j 0")
        end
    end
    open(joinpath(out_dir, "greentwo.def"), "w") do io
        header(io, "NCisAjsCktAltDC", n_site^2)
        for i = 0:(n_site - 1), j = 0:(n_site - 1)
            println(io, "$i 0 $i 0 $j 0 $j 0")
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: julia tools/parton_gen_green_def.jl <Nsite> <out_dir>")
    n_site = parse(Int, ARGS[1])
    parton_write_green_defs(ARGS[2], n_site)
    println("wrote greenone.def / greentwo.def ($(n_site^2) lines each) to $(ARGS[2])")
end
