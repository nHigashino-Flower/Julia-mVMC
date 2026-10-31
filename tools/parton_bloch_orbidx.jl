"""
拡大ユニットセルの Bloch 形の PartonOrbIdx を生成する
--- parton-mode (fork addition) ---

使い方:

    julia --project=MVMCOptimizers.jl tools/parton_bloch_orbidx.jl <stage_in> <out_def> [occupation_file]

- `<stage_in>`: `parton_lattice.dat`(def 生成時に書かれる格子の幾何)と `modpara.def` がある場所
- `<out_def>`: 書き出す PartonOrbIdx(namelist に `PartonOrbIdx <file>` で載せる)
- `[occupation_file]`: どの k に何本の軌道を置くか。行は `flavor m1 m2 nbands`
  (k = 2π(m1/Lx, m2/Ly)。同じ指標を与える (m1, m2) なら代表でなくてよい)。
  省略すると全 k に NElec / N_cell 本ずつ(割り切れなければ error)

Φ_f[s,(k,b)] = e^{ik·a(s)}·θ_{(f,σ(s),k,b)}(DESIGN_parton.md §12.12):

- σ(s): 拡大セル内の位置(`SITE i x y σ` の 4 列目)
- a(s): σ の代表元(その σ を持つ最小のサイト)から s へ移す並進。`MFSHIFT` の要素から選ぶ
- k: 拡大セルの並進群(`MFSHIFT` の全体)の指標。N_cell 個
- 変分パラメータはフレーバーあたり N_σ × NElec 個(全要素なら NSite × NElec)

並進は格子座標 (x, y) → ((x + tx) mod Lx, (y + ty) mod Ly) の素の並進だけを扱う。
ゲージ変換を伴う磁気並進と、パートン境界条件のひねりは対象外。
"""

using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

const _BLOCH_PHASE_TOL = 1e-9   # 指標を同一視するときの位相の許容

"`parton_lattice.dat` から grid・SITE・MFSHIFT を読む。"
function _bloch_read_lattice(path::AbstractString)
    isfile(path) || error("parton_lattice.dat not found: $path")
    grid = nothing
    sites = NTuple{4,Int}[]
    shifts = NTuple{2,Int}[]
    for ln in eachline(path)
        s = strip(ln)
        (isempty(s) || startswith(s, "#")) && continue
        t = split(s)
        if t[1] == "grid"
            grid = (parse(Int, t[2]), parse(Int, t[3]))
        elseif t[1] == "SITE"
            push!(sites, (parse(Int, t[2]), parse(Int, t[3]), parse(Int, t[4]), parse(Int, t[5])))
        elseif t[1] == "MFSHIFT"
            push!(shifts, (parse(Int, t[2]), parse(Int, t[3])))
        end
    end
    grid === nothing && error("$path has no 'grid' line.")
    isempty(sites) && error("$path has no SITE lines.")
    isempty(shifts) && error("$path has no MFSHIFT lines.")
    return grid, sites, shifts
end

"拡大セルの並進 a(格子座標の変位)に対する k = (m1, m2) の指標 e^{2πi(m1·tx/Lx + m2·ty/Ly)}。"
_bloch_phase(m::NTuple{2,Int}, a::NTuple{2,Int}, grid::NTuple{2,Int}) =
    cis(2π * (m[1] * a[1] / grid[1] + m[2] * a[2] / grid[2]))

"(m1, m2) が与える指標が、すでに集めた代表のどれと等しいか(無ければ nothing)。"
function _bloch_find_k(kpoints, m, shifts, grid)
    χ = [_bloch_phase(m, a, grid) for a in shifts]
    for (j, km) in enumerate(kpoints)
        maximum(abs(χ[i] - _bloch_phase(km, shifts[i], grid)) for i in eachindex(shifts)) <
            _BLOCH_PHASE_TOL && return j
    end
    return nothing
end

"""
    parton_bloch_structure(stage_in; occupation_file = nothing) -> NamedTuple

Bloch 形の写像と、その組み立てに使った幾何(σ・並進・k・軌道列)を返す。フィールド:
`n_site`・`n_elec`・`n_flavor`・`grid`・`sigma`(サイトごとの σ、0-based)・`shifts`・
`shift_of_site`(代表元から s へ移す shift の 1-based 位置)・`kpoints`(指標の代表 (m1, m2))・
`columns`(フレーバーごとの軌道列 = (k の位置, band))・`map`(`PartonOrbIndexMap`)。
"""
function parton_bloch_structure(stage_in::AbstractString; occupation_file = nothing)
    grid, sites, shifts = _bloch_read_lattice(joinpath(stage_in, "parton_lattice.dat"))
    res = MVMCExpertModeParsers.parse_modpara_content(read(joinpath(stage_in, "modpara.def"), String))
    res.success || error("could not parse $(joinpath(stage_in, "modpara.def")): $(res.error_message)")
    mp = res.data
    n_site, n_elec, n_flavor = mp.nsite, mp.nelec, mp.nflavor
    Lx, Ly = grid
    length(sites) == n_site || error(
        "parton_lattice.dat has $(length(sites)) SITE lines but Nsite = $n_site.")
    sort!(sites)
    [s[1] for s in sites] == collect(0:(n_site - 1)) || error(
        "SITE lines must cover sites 0..$(n_site - 1) exactly once.")

    pos_to_site = Dict((x, y) => i + 1 for (i, x, y, _) in sites)
    sigma = [σ for (_, _, _, σ) in sites]
    n_sigma = maximum(sigma) + 1
    sort(unique(sigma)) == collect(0:(n_sigma - 1)) || error(
        "the cell positions σ (4th column of SITE) must be 0..$(n_sigma - 1).")
    translate(s, a) = get(pos_to_site, (mod(sites[s][2] + a[1], Lx), mod(sites[s][3] + a[2], Ly))) do
        error("MFSHIFT $a moves site $(s - 1) outside the lattice.")
    end
    rep = [findfirst(==(σ), sigma) for σ = 0:(n_sigma - 1)]
    shift_of_site = zeros(Int, n_site)
    for s = 1:n_site
        cands = [j for (j, a) in enumerate(shifts) if translate(rep[sigma[s] + 1], a) == s]
        length(cands) == 1 || error(
            "site $(s - 1) is reached from its cell representative by $(length(cands)) " *
            "MFSHIFT translations; expected exactly one (are SITE σ and MFSHIFT consistent?).")
        shift_of_site[s] = cands[1]
    end

    # 指標の代表: m1 外側・m2 内側に走査し、新しい指標だけを残す
    kpoints = NTuple{2,Int}[]
    for m1 = 0:(Lx - 1), m2 = 0:(Ly - 1)
        _bloch_find_k(kpoints, (m1, m2), shifts, grid) === nothing && push!(kpoints, (m1, m2))
    end
    n_k = length(kpoints)
    n_k == length(shifts) || error(
        "found $n_k distinct characters for $(length(shifts)) MFSHIFT translations; " *
        "MFSHIFT must list a group of translations.")

    # 占有: フレーバーごと・k ごとの本数
    nb = zeros(Int, n_flavor, n_k)
    if occupation_file === nothing
        n_elec % n_k == 0 || error(
            "NElec = $n_elec is not divisible by the number of k points $n_k; give an " *
            "occupation file (flavor m1 m2 nbands).")
        nb .= n_elec ÷ n_k
    else
        for (i, ln) in enumerate(eachline(occupation_file))
            t = split(strip(ln))
            (isempty(t) || startswith(t[1], "#")) && continue
            v = [tryparse(Int, x) for x in t]
            (length(v) == 4 && !any(isnothing, v)) || error(
                "occupation $occupation_file line $i: expected 'flavor m1 m2 nbands'.")
            f, m1, m2, n = v
            0 <= f < n_flavor || error("occupation line $i: flavor $f is out of range.")
            (0 <= m1 < Lx && 0 <= m2 < Ly) || error(
                "occupation line $i: (m1, m2) = ($m1, $m2) is outside [0,$Lx)×[0,$Ly).")
            n >= 1 || error("occupation line $i: nbands must be >= 1.")
            j = _bloch_find_k(kpoints, (m1, m2), shifts, grid)
            j === nothing && error("occupation line $i: (m1, m2) = ($m1, $m2) is not a known k.")
            nb[f + 1, j] += n
        end
        for f = 1:n_flavor
            sum(nb[f, :]) == n_elec || error(
                "occupation: flavor $(f - 1) has $(sum(nb[f, :])) orbitals but NElec = $n_elec.")
        end
    end
    columns = [[(j, b) for j = 1:n_k for b = 1:nb[f, j]] for f = 1:n_flavor]

    # 写像: θ は (f, 軌道列, σ) の辞書順(σ が最内)
    n_orb = n_flavor * n_site * n_elec
    idx = zeros(Int, n_orb)
    coeff = zeros(ComplexF64, n_orb)
    for f = 1:n_flavor, s = 1:n_site, n = 1:n_elec
        k = MVMCOptimizers.parton_orbparam_flat_index(f, s, n, n_site, n_elec)
        j, _ = columns[f][n]
        idx[k] = ((f - 1) * n_elec + (n - 1)) * n_sigma + sigma[s] + 1
        coeff[k] = _bloch_phase(kpoints[j], shifts[shift_of_site[s]], grid)
    end
    map = MVMCOptimizers.PartonOrbIndexMap(idx, coeff, fill(true, n_flavor * n_elec * n_sigma),
                                           n_site, n_elec, n_flavor)

    _bloch_warn_physhop(stage_in, n_site, shifts, translate)
    return (; n_site, n_elec, n_flavor, grid, sigma, shifts, shift_of_site, kpoints, columns, map)
end

"physhop.def のホッピングが MFSHIFT の並進で不変でなければ警告する(error にはしない)。"
function _bloch_warn_physhop(stage_in, n_site, shifts, translate)
    path = joinpath(stage_in, "physhop.def")
    isfile(path) || return nothing
    hop = Dict{Tuple{Int,Int},ComplexF64}()
    for (i, ln) in enumerate(eachline(path))
        i <= 5 && continue
        t = split(strip(ln))
        length(t) >= 4 || continue
        v = tryparse.(Float64, t[1:4])
        any(isnothing, v) && continue
        hop[(Int(v[1]) + 1, Int(v[2]) + 1)] = ComplexF64(v[3], v[4])
    end
    for a in shifts, ((i, j), t) in hop
        # physhop.def は各ボンドを片方の向きでしか持たない。並進後のボンドが逆向きに
        # 格納されていれば、その共役と比べる
        ti, tj = translate(i, a), translate(j, a)
        t2 = haskey(hop, (ti, tj)) ? hop[(ti, tj)] :
             haskey(hop, (tj, ti)) ? conj(hop[(tj, ti)]) : nothing
        if t2 === nothing || abs(t2 - t) > 1e-10
            @warn "physhop.def is not invariant under MFSHIFT $a (e.g. bond ($(i - 1), $(j - 1))); " *
                  "the Bloch-restricted ansatz is still valid but may be a poor variational state."
            return nothing
        end
    end
    return nothing
end

"""
    parton_bloch_theta_from_orbitals(bs, orbitals) -> Vector{ComplexF64}

α 解などの軌道 Φ_f から、同じ状態を表す Bloch 形の θ を作る(DESIGN §12.12)。
最小二乗の近似は使わない:

1. フレーバーごとに射影子 P = Φ(Φ†Φ)⁻¹Φ†
2. k ごとに基底 v_{σ,k}[s] = e^{ik·a(s)}/√N_cell(σ(s) = σ のサイトだけ)を並べた B_k で
   P(k) = B_k† P B_k(N_σ × N_σ)を作り、エルミート固有値分解する
3. 固有値 > 1 − 1e-8 の固有ベクトルが、その k の占有軌道 u_{k,b}。本数が `bs.columns` の
   その k の本数と違えば error(Φ が拡大セルの並進で不変でないか、占有の指定が違う)

θ のスケールは任意(Φ モードのドライバが再スケールする)。
"""
function parton_bloch_theta_from_orbitals(bs, orbitals::Vector{Matrix{ComplexF64}})
    n_site, n_elec, n_flavor = bs.n_site, bs.n_elec, bs.n_flavor
    n_sigma = maximum(bs.sigma) + 1
    n_cell = length(bs.shifts)
    length(orbitals) == n_flavor || error("expected $n_flavor flavors of orbitals.")
    θ = zeros(ComplexF64, n_flavor * n_elec * n_sigma)
    for f = 1:n_flavor
        Φ = orbitals[f]
        size(Φ) == (n_site, n_elec) || error("flavor $(f - 1): orbitals must be $n_site × $n_elec.")
        P = Φ * ((Φ' * Φ) \ Φ')
        for (j, km) in enumerate(bs.kpoints)
            B = zeros(ComplexF64, n_site, n_sigma)
            for s = 1:n_site
                B[s, bs.sigma[s] + 1] =
                    _bloch_phase(km, bs.shifts[bs.shift_of_site[s]], bs.grid) / sqrt(n_cell)
            end
            ev = eigen(Hermitian(B' * P * B))
            occupied = findall(>(1 - 1e-8), ev.values)
            cols = [n for (n, (jj, _)) in enumerate(bs.columns[f]) if jj == j]
            length(occupied) == length(cols) || error(
                "flavor $(f - 1), k = $km: P(k) has $(length(occupied)) eigenvalues ≈ 1 but " *
                "the occupation asks for $(length(cols)) orbitals (eigenvalues = " *
                "$(round.(ev.values; digits = 6))). The orbitals are not invariant under the " *
                "MFSHIFT translations, or the occupation does not match.")
            for (b, n) in enumerate(cols), σ = 1:n_sigma
                θ[((f - 1) * n_elec + (n - 1)) * n_sigma + σ] = ev.vectors[σ, occupied[b]]
            end
        end
    end
    return θ
end

"""
    parton_bloch_write(stage_in, out_path; occupation_file = nothing) -> String

Bloch 形の PartonOrbIdx を `out_path` に書く。
"""
function parton_bloch_write(stage_in::AbstractString, out_path::AbstractString;
                            occupation_file = nothing)
    bs = parton_bloch_structure(stage_in; occupation_file = occupation_file)
    return MVMCOptimizers.parton_orbparam_write_orbidx(out_path, bs.map, bs.n_site, bs.n_elec,
                                                       bs.n_flavor)
end

if abspath(PROGRAM_FILE) == @__FILE__
    2 <= length(ARGS) <= 3 || error(
        "使い方: julia --project=MVMCOptimizers.jl tools/parton_bloch_orbidx.jl " *
        "<stage_in> <out_def> [occupation_file]")
    bs = parton_bloch_structure(ARGS[1]; occupation_file = length(ARGS) == 3 ? ARGS[3] : nothing)
    MVMCOptimizers.parton_orbparam_write_orbidx(ARGS[2], bs.map, bs.n_site, bs.n_elec, bs.n_flavor)
    println("$(ARGS[2]): N_cell = $(length(bs.shifts)), n_idx = $(bs.map.n_idx) " *
            "(全要素なら $(bs.n_flavor * bs.n_site * bs.n_elec))")
end
