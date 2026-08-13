"""
外部 ED 結果(K-S-Model / ED_Hamiltnian_CB)の読み取り
--- parton-mode (fork addition) ---

DESIGN_parton.md §8 の P 層(物理検証)P0 に対応する。

読むのは `result_eigen_periodic.txt`(ファイル名は接頭辞つきの場合がある)。
中身は ED 実装 `ModuleParentHamCB.jl` が書いたログで、以下を含む:

    SystemParams1 :: SystemParams1(Nx, Ny, nelec, BC, statistics, Nsite, ...)
    SystemParams2 :: SystemParams2(t, t1, t2, t3, ψ, ϕ, ξ, η, R, ...)
    IntrctParams :: IntrctParams(U, V)
    n-th energy :: <値>

巨大な `.jld2`(波動関数)はここでは読まない。P2 の overlap で必要になったとき
だけ、別の読み取り関数で扱う(ボソン系は 6.3 GB あるので不用意に触らない)。

このファイルはパーサだけを提供し、期待値の判定は行わない。
"""

using Printf

"外部 ED 1 ケース分のメタデータと固有値。"
struct EDReference
    path::String
    nx::Int
    ny::Int
    nelec::Int
    boundary::String
    statistics::String
    nsite::Int
    t::Float64
    t1::Float64
    t2::Float64
    t3::Float64
    psi::Float64
    phi::Float64
    xi::Float64
    eta::Float64
    u::Float64
    v::Float64
    random_potential_max::Float64
    energies::Vector{Float64}
end

"`Foo :: Foo(a, b, c)` 形式の行から括弧の中身を取り出す。"
function _ed_struct_args(text::AbstractString, name::AbstractString)
    m = match(Regex("^$(name) :: $(name)\\(", "m"), text)
    m === nothing && error("ED log does not contain a $name line")
    start = m.offset + length(m.match)
    depth = 1
    i = start
    while i <= lastindex(text) && depth > 0
        c = text[i]
        c == '(' && (depth += 1)
        c == ')' && (depth -= 1)
        depth == 0 && break
        i = nextind(text, i)
    end
    depth == 0 || error("unbalanced parentheses in the $name line")
    return text[start:prevind(text, i)]
end

"""
括弧の中身をトップレベルのカンマで分割する(角括弧の中のカンマは無視)。
`[0.0, 0.0]` のようなベクタ引数があるのでナイーブな split では足りない。
"""
function _ed_split_args(args::AbstractString)
    parts = String[]
    depth = 0
    buf = IOBuffer()
    for c in args
        if c in ('[', '(')
            depth += 1
        elseif c in (']', ')')
            depth -= 1
        end
        if c == ',' && depth == 0
            push!(parts, strip(String(take!(buf))))
        else
            print(buf, c)
        end
    end
    push!(parts, strip(String(take!(buf))))
    return parts
end

_ed_unquote(s::AbstractString) = strip(strip(s), '"')

"`[a, b, c]` を Float64 のベクタに。空や `[0.0]` も許す。"
function _ed_parse_float_vector(s::AbstractString)
    body = strip(strip(s), ['[', ']'])
    isempty(strip(body)) && return Float64[]
    return [parse(Float64, strip(x)) for x in split(body, ',')]
end

"""
    read_ed_reference(path) -> EDReference

ED のログを読む。`path` はファイルそのものでも、ディレクトリでもよい
(ディレクトリなら `*result_eigen_periodic.txt` を 1 件だけ探す)。
"""
function read_ed_reference(path::AbstractString)
    file = path
    if isdir(path)
        cands = filter(f -> endswith(f, "result_eigen_periodic.txt"), readdir(path))
        length(cands) == 1 || error(
            "expected exactly one *result_eigen_periodic.txt under $path, found $cands")
        file = joinpath(path, cands[1])
    end
    isfile(file) || error("ED reference file not found: $file")
    text = read(file, String)

    sp1 = _ed_split_args(_ed_struct_args(text, "SystemParams1"))
    length(sp1) >= 6 || error("SystemParams1 has fewer fields than expected: $sp1")
    sp2 = _ed_split_args(_ed_struct_args(text, "SystemParams2"))
    length(sp2) >= 9 || error("SystemParams2 has fewer fields than expected")
    ip = _ed_split_args(_ed_struct_args(text, "IntrctParams"))
    length(ip) >= 2 || error("IntrctParams has fewer fields than expected")

    r = _ed_parse_float_vector(sp2[9])

    energies = Float64[]
    for m in eachmatch(r"^(\d+)-th energy :: (\S+)$"m, text)
        push!(energies, parse(Float64, m.captures[2]))
    end
    isempty(energies) && error("no '<n>-th energy' lines found in $file")

    return EDReference(
        file,
        parse(Int, sp1[1]),
        parse(Int, sp1[2]),
        parse(Int, sp1[3]),
        _ed_unquote(sp1[4]),
        _ed_unquote(sp1[5]),
        parse(Int, sp1[6]),
        parse(Float64, sp2[1]),
        parse(Float64, sp2[2]),
        parse(Float64, sp2[3]),
        parse(Float64, sp2[4]),
        parse(Float64, sp2[5]),
        parse(Float64, sp2[6]),
        parse(Float64, sp2[7]),
        parse(Float64, sp2[8]),
        parse(Float64, ip[1]),
        parse(Float64, ip[2]),
        isempty(r) ? 0.0 : maximum(abs, r),
        energies,
    )
end

"""
    ed_ground_manifold(ref, degeneracy) -> (E_min, gap_to_next)

準縮退多様体の最低値と、多様体の外側(degeneracy+1 番目)までのギャップ。
ν=1/2 は 2 重、ν=1/3 は 3 重(DESIGN §8 P 層)。
"""
function ed_ground_manifold(ref::EDReference, degeneracy::Int)
    length(ref.energies) > degeneracy || error(
        "ED log has only $(length(ref.energies)) levels; need more than $degeneracy " *
        "to measure the gap out of the quasi-degenerate manifold")
    e_min = ref.energies[1]
    spread = ref.energies[degeneracy] - e_min
    gap = ref.energies[degeneracy + 1] - ref.energies[degeneracy]
    return e_min, spread, gap
end

"台帳(人が読む形)。テストの失敗時と P4 のベンチマーク表で使う。"
function ed_ledger(ref::EDReference)
    io = IOBuffer()
    println(io, "ED reference: ", ref.path)
    @printf(io, "  lattice     : Nx=%d Ny=%d  Nsite=%d (=2*Nx*Ny)  BC=%s\n",
            ref.nx, ref.ny, ref.nsite, ref.boundary)
    @printf(io, "  particles   : N=%d  statistics=%s  filling=N/(Nx*Ny)=%s\n",
            ref.nelec, ref.statistics, string(ref.nelec // (ref.nx * ref.ny)))
    @printf(io, "  hoppings    : t=%.10g t1=%.10g t2=%.10g t3=%.10g psi=%.10g\n",
            ref.t, ref.t1, ref.t2, ref.t3, ref.psi)
    @printf(io, "  gauge       : phi=%.10g xi=%.10g eta=%.10g\n", ref.phi, ref.xi, ref.eta)
    @printf(io, "  interaction : U=%.10g (NN)  V=%.10g (2nd NN)\n", ref.u, ref.v)
    @printf(io, "  random pot. : max|R|=%.3g\n", ref.random_potential_max)
    @printf(io, "  levels      : %d recorded, lowest=%.15g\n",
            length(ref.energies), ref.energies[1])
    return String(take!(io))
end

# 支給された 2 ケース。パスは外部データなのでここに集約する。
const ED_ROOT = "/home/nozomihigashino/ED/Data/Checkerboard"
const ED_CASE_BOSON_NU12 = joinpath(
    ED_ROOT,
    "Boson/t=1.0-t1=0.293-t2=-0.293-t3=0.207-ψ=0.785",
    "Nx=4-Ny=4-N=8-q=2-r=0.0/n=0/U=0.0-V=0.0/Psite-Vp=0-0.0",
)
const ED_CASE_FERMION_NU13 = joinpath(
    ED_ROOT,
    "Fermion/t=1.0-t1=0.293-t2=-0.293-t3=0.207-ψ=0.785",
    "Nx=5-Ny=3-N=5-q=3-r=1.0e8/n=0/U=1.0-V=0.0/Psite-Vp=0-0.0",
)
