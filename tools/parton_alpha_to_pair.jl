"""
α モード(PartonMode = 1)の解から Pfaffian パートンモード(PartonMode = 2)の入力一式を作る
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §5。本番の初期化は乱数(§2.3)で、この道具は T3(PartonMode = 1 との一致)と
T4 の継続最適化(E_Pf ≤ E_det の裏取り)のためにある。

使い方:

    julia --project=MVMCOptimizers.jl tools/parton_alpha_to_pair.jl \\
        <alpha_stage_in> <alpha_stage_out> <new_dir> [interflavor|adjacent|auto] [<pair_def>]

- `<alpha_stage_in>`: α run の入力ディレクトリ(namelist.def がある場所)
- `<alpha_stage_out>`: その出力(`<CParaFileHead>_pmfpara_opt.dat`、あれば `_pmfocc_opt.dat`・
  `_jastrow_opt.dat`)
- `<new_dir>`: Pf モードの入力を書く先
- ペアリング(既定 `auto`): def が異フレーバーの対しか持たなければ `interflavor`、それ以外は `adjacent`
- `<pair_def>`: 対軌道の def(外部スクリプトで作った、対称性で縮約したもの)。省略すると
  全対・共有なしの def を書く(このとき射影は厳密)

処理:

1. α* と占有から各フレーバーの占有軌道 Φ^a を組み直し、Ξ = diag(Φ^1, …, Φ^K) を作る
2. ペアリング J(Pf J = ±1)を選び、f = ΞJΞᵀ を作る
3. def の idx 構造へ射影する: p_k = 平均_{行 ∈ k} f[α, β] / (2·sgn)。残差 ‖f − f_proj‖/‖f‖ を
   `parton_alpha_to_pair.log` に書く(0 でなければ Ψ_Pf ∝ Ψ_det は近似。並進で共有した def と素朴な J の
   組み合わせではこうなる。§5)
4. InPartonPairOrbital.def を書く(D_AmpMax はドライバが掛ける)。α 用の def と行を外して namelist と
   modpara(`PartonMode 2`)を書く。`_jastrow_opt.dat` があれば 5 行ヘッダの InJastrow.def に直す
"""

using LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

const _PAIR_DROPPED_KEYS = ("PartonMFTrans", "PartonMFPara", "InPmfPara", "InPmfOcc", "InPmfOrb",
                            "InJastrow", "PartonPairOrbital", "InPartonPairOrbital")
const _PAIR_DROPPED_MODPARA = ("PartonMode", "PartonOrbParam", "PartonOccMode")

"""
    pair_pairing_matrix(n_flavor, n_elec, pairing) -> Matrix{ComplexF64}

M×M(M = K·N)の反対称なペアリング J。列は (a−1)·N + n(フレーバー順)。各組で J[c1, c2] = 1、
J[c2, c1] = −1 なので Pf J = ±1。

- `:adjacent`: 列の隣どうし (2j−1, 2j)
- `:interflavor`: 異フレーバーの列どうしの完全マッチング(同フレーバーを組まない)。残りの多い
  フレーバーから順に組むので、K ≥ 2 なら必ずとれる
"""
function pair_pairing_matrix(n_flavor::Int, n_elec::Int, pairing::Symbol)
    M = n_flavor * n_elec
    iseven(M) || error("NFlavor × NElec = $M is odd: no pairing exists.")
    J = zeros(ComplexF64, M, M)
    if pairing == :adjacent
        for j = 1:2:M
            J[j, j + 1] = 1
            J[j + 1, j] = -1
        end
    elseif pairing == :interflavor
        n_flavor >= 2 || error("interflavor pairing needs NFlavor >= 2, got $n_flavor.")
        remaining = [collect(((a - 1) * n_elec + 1):(a * n_elec)) for a = 1:n_flavor]
        while any(!isempty, remaining)
            order = sortperm([(-length(r), a) for (a, r) in enumerate(remaining)])
            a, b = order[1], order[2]
            (isempty(remaining[a]) || isempty(remaining[b])) &&
                error("interflavor pairing failed (internal error)")
            c1 = popfirst!(remaining[a])
            c2 = popfirst!(remaining[b])
            J[c1, c2] = 1
            J[c2, c1] = -1
        end
    else
        error("unknown pairing :$pairing (use :interflavor, :adjacent or :auto)")
    end
    return J
end

"全対・共有なしの def を書き、その行を返す。"
function _pair_write_full_def(path::AbstractString, n_site::Int, n_flavor::Int)
    terms = MVMCExpertModeParsers.PartonPairOrbitalTerm[]
    n_state = n_site * n_flavor
    idx = 0
    for α = 1:n_state, β = (α + 1):n_state
        push!(terms, MVMCExpertModeParsers.PartonPairOrbitalTerm(
            mod1(α, n_site) - 1, cld(α, n_site) - 1, mod1(β, n_site) - 1, cld(β, n_site) - 1, idx, 1))
        idx += 1
    end
    open(path, "w") do io
        println(io, "=============================================")
        println(io, "NOrbitalIdx $(length(terms))")
        println(io, "ComplexType 1")
        println(io, "=============================================")
        println(io, "=============================================")
        for t in terms
            println(io, "$(t.site1) $(t.flavor1) $(t.site2) $(t.flavor2) $(t.idx) $(t.sign)")
        end
    end
    return terms
end

"Jastrow の per-block 出力(4 行ヘッダ)を InJastrow.def(5 行ヘッダ)に直す。"
function _pair_jastrow_to_in(src::AbstractString, dst::AbstractString)
    lines = readlines(src)
    first_data = findfirst(lines) do l
        tok = split(strip(l))
        length(tok) >= 2 && !occursin("=", l) && tryparse(Int, tok[1]) !== nothing
    end
    first_data === nothing && error("Could not find the data rows of $src")
    open(dst, "w") do io
        for k = 1:(first_data - 1)
            println(io, lines[k])
        end
        for _ = first_data:5
            println(io, "===============================")
        end
        for k = first_data:length(lines)
            println(io, lines[k])
        end
    end
    return dst
end

"""
    parton_alpha_to_pair(alpha_in, alpha_out, new_dir; pairing = :auto, pair_def = nothing)
        -> (namelist, residual, pairing)

Pf モードの入力一式を `new_dir` に書く。`residual` は射影の残差 ‖f − f_proj‖/‖f‖。
"""
function parton_alpha_to_pair(alpha_in::AbstractString, alpha_out::AbstractString,
                              new_dir::AbstractString; pairing::Symbol = :auto,
                              pair_def::Union{AbstractString,Nothing} = nothing)
    mkpath(new_dir)
    nl_in = joinpath(alpha_in, "namelist.def")
    data = MVMCExpertModeParsers.parse_expert_mode_files(nl_in)
    mp = data.modpara
    Ns, N, K = mp.nsite, mp.nelec, mp.nflavor
    head = isempty(mp.c_para_file_head) ? "zqp" : mp.c_para_file_head

    # 1. α* と占有から軌道を組む(既存の読み手を使うため、絶対パスの一時 namelist を書く)
    alpha_file = abspath(joinpath(alpha_out, head * "_pmfpara_opt.dat"))
    occ_file = abspath(joinpath(alpha_out, head * "_pmfocc_opt.dat"))
    isfile(alpha_file) || error("α の最終値が見つかりません: $alpha_file")
    tmp = joinpath(new_dir, "_alpha_to_pair_namelist.def")
    write(tmp, "InPmfPara $alpha_file\n" * (isfile(occ_file) ? "InPmfOcc $occ_file\n" : ""))
    local Φ
    try
        MVMCOptimizers.parton_read_in_pmfpara!(data, tmp)
        occ = MVMCOptimizers.parton_read_in_pmfocc(data, tmp)
        mfham = MVMCOptimizers.PartonMFHamiltonian(Ns, N, K, MVMCOptimizers.parton_n_idx(data))
        MVMCOptimizers.parton_build_mf_templates!(mfham, data)
        MVMCOptimizers.parton_update_orbitals!(
            mfham, MVMCOptimizers.parton_alpha_from_terms(data), N; forced_occ = occ)
        Φ = deepcopy(mfham.orbitals)
    finally
        rm(tmp; force = true)
    end

    # 2. def(外部のもの / 全対)とペアリング
    pair_def_name = "partonpairorbital.def"
    if pair_def === nothing
        terms = _pair_write_full_def(joinpath(new_dir, pair_def_name), Ns, K)
    else
        cp(pair_def, joinpath(new_dir, pair_def_name); force = true)
        res, _, _ = MVMCExpertModeParsers.parse_parton_pair_orbital_def(String(pair_def))
        res.success || error("Failed to parse $pair_def: $(res.error_message)")
        terms = res.data
    end
    if pairing == :auto
        pairing = all(t -> t.flavor1 != t.flavor2, terms) ? :interflavor : :adjacent
    end
    J = pair_pairing_matrix(K, N, pairing)

    M = K * N
    Ξ = zeros(ComplexF64, K * Ns, M)
    for a = 1:K
        Ξ[((a - 1) * Ns + 1):(a * Ns), ((a - 1) * N + 1):(a * N)] = Φ[a]
    end
    f = Ξ * J * transpose(Ξ)

    # 3. idx 構造へ射影(同じ idx の成分を平均。符号と係数 2 を外す)
    n_idx = maximum(t.idx for t in terms) + 1
    acc = zeros(ComplexF64, n_idx)
    cnt = zeros(Int, n_idx)
    for t in terms
        α = MVMCOptimizers.parton_pair_state(Ns, t.site1 + 1, t.flavor1 + 1)
        β = MVMCOptimizers.parton_pair_state(Ns, t.site2 + 1, t.flavor2 + 1)
        acc[t.idx + 1] += f[α, β] / (2 * t.sign)
        cnt[t.idx + 1] += 1
    end
    params = acc ./ cnt
    src = MVMCOptimizers.PartonPairOrbital(Ns, K, N, terms)
    MVMCOptimizers.parton_pair_load!(src, params)
    residual = norm(f - src.pair_matrix) / norm(f)

    data.parton_pair_params = params
    MVMCOptimizers.parton_pair_write_params(data, joinpath(new_dir, "InPartonPairOrbital.def"))

    # 4. def の複製・namelist・Jastrow・modpara
    entries = MVMCExpertModeParsers.parse_namelist_content(MVMCExpertModeParsers.read_def_file(nl_in))
    kept = Tuple{String,String}[]
    for (key, path) in entries
        key in _PAIR_DROPPED_KEYS && continue
        dst = joinpath(new_dir, path)
        mkpath(dirname(dst))
        cp(joinpath(alpha_in, path), dst; force = true)
        push!(kept, (key, path))
    end
    push!(kept, ("PartonPairOrbital", pair_def_name), ("InPartonPairOrbital", "InPartonPairOrbital.def"))
    jastrow_file = joinpath(alpha_out, head * "_jastrow_opt.dat")
    if isfile(jastrow_file)
        _pair_jastrow_to_in(jastrow_file, joinpath(new_dir, "InJastrow.def"))
        push!(kept, ("InJastrow", "InJastrow.def"))
    end
    open(joinpath(new_dir, "namelist.def"), "w") do io
        for (key, path) in kept
            println(io, rpad(key, 20), path)
        end
    end
    mod_path = joinpath(new_dir, only(p for (k, p) in kept if k == "ModPara"))
    lines = filter(l -> !any(k -> startswith(strip(l), k), _PAIR_DROPPED_MODPARA), readlines(mod_path))
    push!(lines, "PartonMode 2")
    write(mod_path, join(lines, "\n") * "\n")

    open(joinpath(new_dir, "parton_alpha_to_pair.log"), "w") do io
        println(io, "alpha_stage_in  $(abspath(alpha_in))")
        println(io, "alpha_stage_out $(abspath(alpha_out))")
        println(io, "pair_def        $(pair_def === nothing ? "full (written here)" : abspath(pair_def))")
        println(io, "pairing         $pairing")
        println(io, "n_pair_idx      $n_idx")
        println(io, "projection_residual $(residual)")
    end
    return (namelist = joinpath(new_dir, "namelist.def"), residual = residual, pairing = pairing)
end

if abspath(PROGRAM_FILE) == @__FILE__
    3 <= length(ARGS) <= 5 || error(
        "使い方: julia --project=MVMCOptimizers.jl tools/parton_alpha_to_pair.jl " *
        "<alpha_stage_in> <alpha_stage_out> <new_dir> [interflavor|adjacent|auto] [<pair_def>]")
    pairing = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :auto
    pair_def = length(ARGS) >= 5 ? ARGS[5] : nothing
    res = parton_alpha_to_pair(ARGS[1], ARGS[2], ARGS[3]; pairing = pairing, pair_def = pair_def)
    println(res.namelist)
    println("projection_residual = $(res.residual)  pairing = $(res.pairing)")
end
