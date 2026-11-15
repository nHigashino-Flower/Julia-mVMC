"""
Pfaffian パートンモードの入力・初期値・D_AmpMax・OptFlag・門番
--- parton-mode (fork addition) ---

DESIGN_parton_pf.md §2(入力)・§2.3(初期値)・§4.6(D_AmpMax)・§7(門番)。

流れは上流と同じ: def(PartonPairOrbital)が変分の構造(どの対を変分するか・idx の同一視・
符号・凍結)を決め、初期値は InPartonPairOrbital があればその値、無ければ seed に従った乱数。
初期化の直後と SR の各ステップの後に D_AmpMax で max|p| = 4 にそろえる。
"""

"対軌道のパラメータ数(idx の数)。"
parton_pair_n_idx(data::ExpertModeData) = length(data.parton_pair_params)

"""
    parton_pair_build_source(data) -> PartonPairOrbital

def の行からテンプレートを組み(意味の検査はここ)、現在の p を読み込んだ器を返す。
"""
function parton_pair_build_source(data::ExpertModeData)
    mp = data.modpara
    src = PartonPairOrbital(mp.nsite, mp.nflavor, mp.nelec, data.parton_pair_terms)
    src.n_idx == length(data.parton_pair_params) || error(
        "PartonPairOrbital uses $(src.n_idx) parameters but $(length(data.parton_pair_params)) " *
        "pair values are held. The def and the values are out of sync.")
    parton_pair_load!(src, data.parton_pair_params)
    return src
end

"""
    parton_pair_write_params(data, path) -> String

p を InPartonPairOrbital と同じ形式で書く(`_init` と `_opt` で共有する唯一の writer)。
ヘッダは 5 行で、キーワードは上流の InOrbital と同じ `NOrbitalIdx`:

    ===============================
    NOrbitalIdx <n>
    ===============================
    == idx RePair ImPair ==
    ===============================
    <idx 0-based> <Re %.18e> <Im %.18e>

上流の `zqp_orbital_opt.dat` はヘッダが 4 行で行番号を idx として書くので往復できないが
(DESIGN_parton_pf.md §3)、こちらは `parton_pair_read_in!` でビット一致に読み戻せる。
"""
function parton_pair_write_params(data::ExpertModeData, path::AbstractString)
    params = data.parton_pair_params
    dir = dirname(abspath(String(path)))
    isempty(dir) || mkpath(dir)
    open(String(path), "w") do f
        println(f, "===============================")
        println(f, "NOrbitalIdx $(length(params))")
        println(f, "===============================")
        println(f, "== idx RePair ImPair ==")
        println(f, "===============================")
        for k in eachindex(params)
            @printf(f, "%d % .18e % .18e \n", k - 1, real(params[k]), imag(params[k]))
        end
    end
    return String(path)
end

"namelist.def の (キーワード, パス) の並び(パスは namelist の場所からの相対を解決済み)。"
function _parton_pair_namelist_entries(namelist_path::AbstractString)
    base_dir = dirname(abspath(String(namelist_path)))
    file_list = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(String(namelist_path)))
    return [(String(t), joinpath(base_dir, p)) for (t, p) in file_list]
end

"""
    parton_pair_read_in!(data, namelist_path) -> Union{String, Nothing}

namelist.def に `InPartonPairOrbital` があればそのファイルから p を読み、`data.parton_pair_params`
に置いてパスを返す。無ければ何もせず `nothing`(初期値は乱数になる。DESIGN_parton_pf.md §2.3)。

上流の汎用リーダ(読めない値を 0 にし、NaN を通し、件数の不一致を警告で済ませる)は使わず、
次をすべて error で検査する(部分適用はしない):

- ファイルが存在する。ヘッダ 2 行目が `NOrbitalIdx <n>` で、n = 行数 = def の idx の数
- 各行は `idx Re Im`、idx ∈ [0, n)、重複なし、値は有限
"""
function parton_pair_read_in!(data::ExpertModeData, namelist_path::AbstractString)
    path = nothing
    for (file_type, file_path) in _parton_pair_namelist_entries(namelist_path)
        file_type == "InPartonPairOrbital" && (path = file_path)
    end
    path === nothing && return nothing
    isfile(path) || error("InPartonPairOrbital file listed in namelist.def but not found: $path")

    n_idx = parton_pair_n_idx(data)
    lines = readlines(path)
    length(lines) >= 5 || error(
        "InPartonPairOrbital: $path is too short to hold the 5-line header.")
    head = split(strip(lines[2]))
    (length(head) >= 2 && head[1] == "NOrbitalIdx") || error(
        "InPartonPairOrbital: $path line 2 must read 'NOrbitalIdx <n>', got '$(lines[2])'.")
    declared = tryparse(Int, head[2])
    declared === nothing && error(
        "InPartonPairOrbital: $path line 2: invalid count '$(head[2])'.")

    params = fill(ComplexF64(NaN, NaN), n_idx)
    seen = falses(n_idx)
    n_rows = 0
    for (i, line) in enumerate(lines)
        i <= 5 && continue
        tok = split(strip(line))
        isempty(tok) && continue
        length(tok) >= 3 || error(
            "InPartonPairOrbital: $path line $i: expected 'idx Re Im', got '$line'.")
        idx = tryparse(Int, tok[1])
        re = tryparse(Float64, tok[2])
        imv = tryparse(Float64, tok[3])
        (idx === nothing || re === nothing || imv === nothing) && error(
            "InPartonPairOrbital: $path line $i: could not parse '$line'.")
        0 <= idx < n_idx || error(
            "InPartonPairOrbital: $path line $i: idx $idx is out of range [0, $(n_idx - 1)].")
        seen[idx + 1] && error("InPartonPairOrbital: $path line $i: idx $idx appears twice.")
        (isfinite(re) && isfinite(imv)) || error(
            "InPartonPairOrbital: $path line $i: the value of idx $idx is not finite.")
        seen[idx + 1] = true
        params[idx + 1] = ComplexF64(re, imv)
        n_rows += 1
    end
    declared == n_rows || error(
        "InPartonPairOrbital: $path declares NOrbitalIdx = $declared but holds $n_rows data rows.")
    n_rows == n_idx || error(
        "InPartonPairOrbital: $path holds $n_rows entries but PartonPairOrbital uses " *
        "$n_idx parameters. Refusing to apply it partially.")
    data.parton_pair_params = params
    return path
end

"""
    parton_pair_materialize_flags!(data) -> Vector{Bool}

Pf モードの OptFlag 配列を実体化する(乱数初期化と門番より前に呼ぶこと)。`stochastic_opt!` は
範囲外のフラグを黙って「凍結」と読むので、長さを 2(n_proj + n_pair) にそろえる。

- 射影ブロック(Jastrow): α・Φ モードと同じ。Re は jastrowidx.def の opt フラグを引き継ぎ、
  Im は強制凍結(v は実数。共役シムの前提)
- f のブロック: def の末尾の OptFlag 行(`idx flag`)を Re・Im の両方に当てる(上流の
  `set_orbital_opt_flags!` と同じく、複素パラメータの両スロットに同じ値)。行の無い idx は可動
"""
function parton_pair_materialize_flags!(data::ExpertModeData)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    n_pair = parton_pair_n_idx(data)
    old = data.optimization_flags
    flags = fill(true, 2 * (n_proj + n_pair))
    for p = 1:n_proj
        re = 2 * (p - 1) + 1
        flags[re] = re <= length(old) ? old[re] : true
        flags[re + 1] = false
    end
    for k = 1:n_pair
        movable = get(data.parton_pair_opt_flags, k - 1, 1) != 0
        flags[2 * (n_proj + k) - 1] = movable
        flags[2 * (n_proj + k)] = movable
    end
    data.optimization_flags = flags
    return flags
end

"""
    parton_pair_init_random!(data, base_seed)

p を乱数で初期化する(InPartonPairOrbital が無いとき。DESIGN_parton_pf.md §2.3)。

- **分布は上流と同じ**(パーサ `parameter_init.jl:205-220`): 可動成分は
  p = [2(r₁−½) + 2i(r₂−½)]/√2、OptFlag = 0 の成分は 0。可動かどうかは Re スロットのフラグで決める
  (上流と同じ)。先に `parton_pair_materialize_flags!` を呼んでおくこと
- **乱数列は初期値専用**(α モードの `parton_init_alpha!` と同じ規律): 解決済みの基準 seed
  (ランクのずらしを加える前の値)で専用の SFMT を作る。サンプリング用の乱数列は消費しないので、
  初期値を乱数で作ったかファイルから読んだかでサンプリングの系列は変わらない。全ランクが
  構成的に同じ値を得るので bcast も要らない。上流は 1 本の列から引くので、値そのものは上流と一致しない
"""
function parton_pair_init_random!(data::ExpertModeData, base_seed::Integer)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    n_pair = parton_pair_n_idx(data)
    flags = data.optimization_flags
    length(flags) == 2 * (n_proj + n_pair) || error(
        "parton_pair_init_random! needs materialized OptFlags (length 2(n_proj + n_pair) = " *
        "$(2 * (n_proj + n_pair))), got $(length(flags)). Call parton_pair_materialize_flags! first.")
    rng = SFMT19937RNG()
    Random.seed!(rng, Int(base_seed))
    params = zeros(ComplexF64, n_pair)
    for k = 1:n_pair
        flags[2 * (n_proj + k) - 1] || continue
        r1 = rng_real2(rng)
        r2 = rng_real2(rng)
        params[k] = ComplexF64(2 * (r1 - 0.5), 2 * (r2 - 0.5)) / sqrt(2.0)
    end
    data.parton_pair_params = params
    return params
end

"""
    parton_pair_rescale!(data) -> Float64

上流の D_AmpMax と同じ再スケール(`parameter_sync.jl:138-151`): p ← (D / max_k |p_k|)·p。
D は modpara の `PartonPairAmpMax`(既定 4.0 = 上流の D_AMP_MAX。2026-11-05 にキー化)。
OptFlag = 0 の成分も含めて全成分に掛け、**max が D 未満でも拡大する**。倍率は正の実数なので
全体位相は固定しない。Pf(λF) = λ^{M/2} Pf F なので物理は変わらず、変わるのは |ip| の絶対値だけ。

返り値は再スケール**前**の max|p|(診断用)。全成分が 0 なら error(0 で割らない)。
"""
function parton_pair_rescale!(data::ExpertModeData)
    params = data.parton_pair_params
    x = isempty(params) ? 0.0 : maximum(abs, params)
    D = data.modpara.parton_pair_amp_max
    x > 0 || error(
        "The pair parameters are identically zero; cannot rescale them to PartonPairAmpMax = $D.")
    ratio = D / x
    for k in eachindex(params)
        params[k] *= ratio
    end
    return x
end

"PartonMode = 2 と混ぜてはいけない namelist のキーワード(α・Φ・上流の Orbital 系)。"
const _PARTON_PAIR_FOREIGN_KEYS = (
    # PartonOrbIdx(Φ の制限)は Φ のドライバが namelist から直接読むので、ここで弾かないと黙って無視される
    "PartonMFTrans", "PartonMFPara", "InPmfPara", "InPmfOcc", "InPmfOrb", "PartonOrbIdx",
    "Orbital", "OrbitalAntiParallel", "OrbitalParallel", "OrbitalGeneral",
    "InOrbital", "InOrbitalAntiParallel", "InOrbitalParallel", "InOrbitalGeneral",
)

"""
    parton_pair_reject_other_inputs(namelist_path)

namelist.def に α・Φ モードの入力や上流の Orbital 系があれば error(入力の取り違えとみなす)。
上流の Orbital 系は対軌道を 2 スピン前提で読むので、Pf モードの対軌道とは別物
(DESIGN_parton_pf.md §7「既存スピン Orbital との同時指定」)。
"""
function parton_pair_reject_other_inputs(namelist_path::AbstractString)
    for (file_type, _) in _parton_pair_namelist_entries(namelist_path)
        file_type in _PARTON_PAIR_FOREIGN_KEYS && error(
            "PartonMode = 2 (Pfaffian parton mode) does not use `$file_type`; remove it from " *
            "$(namelist_path). The pair orbitals come from PartonPairOrbital " *
            "(and InPartonPairOrbital for initial values).")
    end
    return nothing
end

"""
    validate_parton_pair_inputs(data, ctx)

Pfaffian パートンモード(PartonMode = 2)の門番。非破壊。α・Φ と共通の検査(modpara・物理項・
QP・並列)を再利用し、Pf 固有の検査を足す(DESIGN_parton_pf.md §7)。

`DSROptStaDel = 0` は止めない(初期点 f = ΞJΞᵀ の S の零空間は RedCut では落ちないが、
StaDel = 0 で走らせる運用はしない、というユーザー判断。§1.3)。
"""
function validate_parton_pair_inputs(data::ExpertModeData, ctx::ParallelContext)
    mp = data.modpara
    validate_parton_modpara(mp; allowed_modes = (PARTON_MODE_PFAFFIAN,))
    mp.nflavor >= 2 || error(
        "PartonMode = 2 needs NFlavor >= 2, got NFlavor = $(mp.nflavor).")
    iseven(mp.nflavor * mp.nelec) || error(
        "PartonMode = 2: the total parton number NFlavor × NElec = $(mp.nflavor * mp.nelec) " *
        "is odd, so every Pfaffian vanishes. An extension with an unpaired orbital is " *
        "planned; until then use an even NFlavor × NElec.")
    mp.parton_orb_param == PARTON_ORB_PARAM_ALPHA || error(
        "PartonOrbParam = $(mp.parton_orb_param) has no meaning in PartonMode = 2; set it " *
        "to 0 or remove it.")
    # 再スケールの値(2026-11-05)。Φ 用の PartonOrbAmpMax は Pf では読まないので、書いてあれば止める
    (isfinite(mp.parton_pair_amp_max) && mp.parton_pair_amp_max > 0) || error(
        "PartonPairAmpMax must be a positive finite number, got $(mp.parton_pair_amp_max).")
    mp.parton_orb_amp_max == MVMCExpertModeParsers.DEFAULT_PARTON_ORB_AMP_MAX || error(
        "PartonOrbAmpMax = $(mp.parton_orb_amp_max) has no meaning in PartonMode = 2 (it rescales " *
        "the direct orbitals of PartonMode = 1); use PartonPairAmpMax for the pair orbitals.")
    mp.parton_occ_mode == PARTON_OCC_AUFBAU || error(
        "PartonOccMode = $(mp.parton_occ_mode) has no meaning in PartonMode = 2 (there is " *
        "no occupation to choose); set it to 0 or remove it.")
    (isempty(data.pmfpara_terms) && isempty(data.pmftrans_terms)) || error(
        "PartonMode = 2 does not use pmfpara.def / pmftrans.def; remove PartonMFPara and " *
        "PartonMFTrans from namelist.def.")
    isempty(data.pmforb_params) || error(
        "PartonMode = 2 does not use the direct orbitals (InPmfOrb); remove them.")
    isempty(data.parton_pair_terms) && error(
        "PartonMode = 2 needs the pair orbitals: add a `PartonPairOrbital` line to namelist.def.")

    validate_parton_data(data; require_mf_inputs = false)
    validate_parton_qp(data)
    validate_parton_parallel(ctx, mp)

    # 意味の検査(範囲・対角・重複・連番)はテンプレート build が行う
    parton_pair_build_source(data)
    all(isfinite, data.parton_pair_params) || error("The pair parameters contain non-finite values.")

    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    n_pair = parton_pair_n_idx(data)
    expected = 2 * (n_proj + n_pair)
    length(data.optimization_flags) == expected || error(
        "optimization_flags has length $(length(data.optimization_flags)) but PartonMode = 2 " *
        "needs exactly $expected (2 slots x (NProj = $n_proj + n_pair = $n_pair)). " *
        "stochastic_opt! reads an out-of-range flag as 'frozen' without warning. Call " *
        "parton_pair_materialize_flags! before this gate.")
    any(i -> data.optimization_flags[i], (2 * n_proj + 1):expected) || error(
        "Every pair-orbital slot in optimization_flags is frozen, so SR has nothing to optimize.")
    return nothing
end
