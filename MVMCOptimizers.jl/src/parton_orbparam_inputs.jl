"""
Φ 直接変分モードの入力(PartonOrbIdx・InPmfOrb)・初期値・再スケール・OptFlag・門番
--- parton-mode (fork addition) ---

DESIGN_parton.md §12.11(上流型)と §12.12(制限付き Φ)。SR が動かすのは θ
(`data.pmforb_params`)で、Φ_f[s,n] = c·θ_idx。PartonOrbIdx が無ければ恒等写像。

- **PartonOrbIdx**(任意): Pfaffian の orbitalidx.def と同じ規約の制限。厳格に読む
- **初期値**: 乱数(Pfaffian と同じ分布、初期化専用の乱数列)→ **InPmfOrb**(任意、θ)で上書き。
  InPmfOrb の読み手は厳格で、件数・範囲・重複・有限性のどれかが崩れたら error
  (汎用の `parse_input_parameter_file` は読めない値を 0 にし、件数不一致を警告で済ませる
  ので使わない。§12.2)。列フルランクは組み立てた Φ について `parton_orbparam_check_rank` が検査する
- **再スケール**は上流 `sync_modified_parameter!` の D_AmpMax と同じ形で、**毎回**
  max|θ| = PartonOrbAmpMax にそろえる。上流は Slater 全体で 1 つのスケールだが、
  ここは行列式がフレーバーごとにあり、それぞれに定数倍の自由度があるので、
  θ を共有するフレーバーの組(制限が無ければフレーバー)ごとに行う
"""

const _PARTON_ORBPARAM_RANK_TOL = 1e-12   # 列フルランクの判定(σ_min / σ_max)

"""
    parton_orbparam_write_pmforb(data, path) -> String

Φ を InPmfOrb.def と同じ形式で書く(初期値のダンプ `_pmforb_init.dat` と最終値
`_pmforb_opt.dat` で共有する唯一の writer)。書式は `parton_write_pmfpara` に合わせる:

    ===============================
    NPmfOrb <n>
    ===============================
    == idx RePhi ImPhi ==
    ===============================
    <idx 0-based> <Re %.18e> <Im %.18e>

idx は θ の番号(0-based)。PartonOrbIdx が無ければ平坦添字 ((f·Ns)+s)·Ne+n と同じ。
"""
function parton_orbparam_write_pmforb(data::ExpertModeData, path::AbstractString)
    params = data.pmforb_params
    dir = dirname(abspath(String(path)))
    isempty(dir) || mkpath(dir)
    open(String(path), "w") do f
        println(f, "===============================")
        println(f, "NPmfOrb $(length(params))")
        println(f, "===============================")
        println(f, "== idx RePhi ImPhi ==")
        println(f, "===============================")
        for k in eachindex(params)
            @printf(f, "%d % .18e % .18e \n", k - 1, real(params[k]), imag(params[k]))
        end
    end
    return String(path)
end

"""
    parton_orbparam_read_in_pmforb!(data, namelist_path, map) -> Union{Nothing,String}

namelist.def の `InPmfOrb` が指すファイルから θ を読み、`data.pmforb_params` を上書きする。
返り値は読んだファイルのパス。**InPmfOrb が無ければ nothing を返し、何もしない**
(乱数初期値のまま。Pfaffian の InOrbital と同じ、DESIGN §12.12)。

検査(どれも error。部分適用はしない):

- ファイルが存在する
- ヘッダ 2 行目が `NPmfOrb <n>` で、n = 行数 = 写像の n_idx(恒等写像なら NFlavor·NSite·NElec)
- idx ∈ [0, n_idx)、重複なし、値は有限

列フルランクの検査はここではしない(θ から組み立てた Φ について
`parton_orbparam_check_rank` が行う)。
"""
function parton_orbparam_read_in_pmforb!(data::ExpertModeData, namelist_path::AbstractString,
                                         map::PartonOrbIndexMap)
    base_dir = dirname(abspath(String(namelist_path)))
    file_list = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(String(namelist_path)))
    path = nothing
    for (file_type, file_path) in file_list
        file_type == "InPmfOrb" && (path = joinpath(base_dir, file_path))
    end
    path === nothing && return nothing
    isfile(path) || error("InPmfOrb file listed in namelist.def but not found: $path")

    n_theta = map.n_idx
    lines = readlines(path)
    length(lines) >= 5 || error("InPmfOrb: $path is too short to hold the 5-line header.")
    head = split(strip(lines[2]))
    (length(head) >= 2 && head[1] == "NPmfOrb") || error(
        "InPmfOrb: $path line 2 must read 'NPmfOrb <n>', got '$(lines[2])'.")
    declared = tryparse(Int, head[2])
    declared === nothing && error("InPmfOrb: $path line 2: invalid count '$(head[2])'.")

    params = fill(ComplexF64(NaN, NaN), n_theta)
    seen = falses(n_theta)
    n_rows = 0
    for (i, line) in enumerate(lines)
        i <= 5 && continue
        tok = split(strip(line))
        isempty(tok) && continue
        length(tok) >= 3 || error(
            "InPmfOrb: $path line $i: expected 'idx Re Im', got '$line'.")
        idx = tryparse(Int, tok[1])
        re = tryparse(Float64, tok[2])
        imv = tryparse(Float64, tok[3])
        (idx === nothing || re === nothing || imv === nothing) && error(
            "InPmfOrb: $path line $i: could not parse '$line'.")
        0 <= idx < n_theta || error(
            "InPmfOrb: $path line $i: idx $idx is out of range [0, $(n_theta - 1)].")
        seen[idx + 1] && error("InPmfOrb: $path line $i: idx $idx appears twice.")
        (isfinite(re) && isfinite(imv)) || error(
            "InPmfOrb: $path line $i: the value of idx $idx is not finite.")
        seen[idx + 1] = true
        params[idx + 1] = ComplexF64(re, imv)
        n_rows += 1
    end
    declared == n_rows || error(
        "InPmfOrb: $path declares NPmfOrb = $declared but holds $n_rows data rows.")
    n_rows == n_theta || error(
        "InPmfOrb: $path holds $n_rows entries but " *
        (map.is_identity ? "NFlavor × NSite × NElec" : "NPmfOrbIdx in PartonOrbIdx") *
        " = $n_theta are required. Refusing to apply it partially.")

    data.pmforb_params = params
    return path
end

# def が無いとき(恒等写像)
parton_orbparam_read_in_pmforb!(data::ExpertModeData, namelist_path::AbstractString) =
    parton_orbparam_read_in_pmforb!(
        data, namelist_path,
        parton_orbparam_identity_map(data.modpara.nsite, data.modpara.nelec, data.modpara.nflavor))

"""
    parton_orbparam_init_random!(data, map, base_seed) -> Nothing

θ を乱数で初期化する(DESIGN §12.12、Pfaffian の InitParameter と同じ分布)。idx 昇順に、
OptFlag が可動なら (U[−1,1) + i·U[−1,1))/√2(実部を先に引く)、凍結なら 0。

乱数は**初期化専用の列**(`SFMT19937RNG` を base_seed で初期化)。α の
`parton_init_alpha!` と同じ規律で、サンプリングの乱数列は消費しない。base_seed は
ランクのオフセットを足す前の値なので、全ランクが同じ θ を得る(bcast 不要)。
"""
function parton_orbparam_init_random!(data::ExpertModeData, map::PartonOrbIndexMap,
                                      base_seed::Integer)
    rng = SFMT19937RNG()
    Random.seed!(rng, Int(base_seed))
    θ = zeros(ComplexF64, map.n_idx)
    for j = 1:map.n_idx
        map.opt_flags[j] || continue
        re = 2 * rng_real2(rng) - 1
        imv = 2 * rng_real2(rng) - 1
        θ[j] = ComplexF64(re, imv) / sqrt(2.0)
    end
    data.pmforb_params = θ
    return nothing
end

"""
    parton_orbparam_check_rank(phi) -> Nothing

組み立てた Φ_f がフレーバーごとに列フルランク(σ_min / σ_max > 1e-12)かを検査する。
ランク落ちの Φ では行列式が恒等的に 0 になり、初期配置が見つからないという
分かりにくい失敗になるので、初期化の直後にここで止める。
"""
function parton_orbparam_check_rank(phi::PartonDirectPhi)
    for (f, Φ) in enumerate(phi.orbitals)
        σ = svdvals(Φ)
        (σ[1] > 0 && σ[end] / σ[1] > _PARTON_ORBPARAM_RANK_TOL) || error(
            "The orbitals of flavor $(f - 1) are not of full column rank (σ_min / σ_max = " *
            "$(σ[1] > 0 ? σ[end] / σ[1] : 0.0)); every determinant would vanish. Check " *
            "whether PartonOrbIdx shares too many elements, or the values in InPmfOrb.")
    end
    return nothing
end
"""
    parton_orbparam_rescale!(data, map) -> Vector{Float64}

上流の D_AmpMax と同じ形の再スケール(DESIGN §12.11 / §12.12)。再スケールの組
(同じ θ を使うフレーバーの連結成分)ごとに θ ← (D / max|θ|)·θ とする
(D = `PartonOrbAmpMax`)。係数は |c| = 1 なので max|θ| = その組の max|Φ|。
**max が D 未満でも拡大する**(上流 `parameter_init.jl:274-276` の規約)。
倍率は正の実数なので全体位相は固定しない。

返り値は組ごとの再スケール**前**の max|θ|(漂流の監視用)。共有が無ければ組 = フレーバー。
全要素が 0 の組があれば error(0 で割らない)。

Φ_f → c·Φ_f は det を c^Ne 倍するだけなので状態は変わらない。O も SR の更新も
c によらない(O は 1/c 倍、S は 1/c² 倍、更新量は c 倍になり相対変化は不変)。
"""
function parton_orbparam_rescale!(data::ExpertModeData, map::PartonOrbIndexMap)
    D = data.modpara.parton_orb_amp_max
    length(data.pmforb_params) == map.n_idx || error(
        "pmforb_params has $(length(data.pmforb_params)) entries but the orbital index " *
        "map has $(map.n_idx) parameters.")
    xmax = zeros(Float64, length(map.scale_groups))
    for (g, group) in enumerate(map.scale_groups)
        x = maximum(abs, view(data.pmforb_params, group))
        x > 0 || error(
            "The orbitals of rescale group $g are identically zero; cannot rescale them " *
            "to PartonOrbAmpMax = $D (with no PartonOrbIdx sharing, group g is flavor g − 1).")
        ratio = D / x
        for k in group
            data.pmforb_params[k] *= ratio
        end
        xmax[g] = x
    end
    return xmax
end

# def が無いときの再スケール(恒等写像。組はフレーバーごとで、§12.11 と同じ演算順)
parton_orbparam_rescale!(data::ExpertModeData) = parton_orbparam_rescale!(
    data, parton_orbparam_identity_map(data.modpara.nsite, data.modpara.nelec,
                                       data.modpara.nflavor))

"""
    parton_orbparam_materialize_flags!(data, map) -> Vector{Bool}

Φ モードの OptFlag 配列を実体化する(門番より前に呼ぶこと)。`stochastic_opt!` は
範囲外のフラグを黙って「凍結」と読むので、長さを 2(n_proj + n_orb) にそろえる。

- 射影ブロック(Jastrow): α モードの `parton_materialize_flags!` と同じ。Re は
  jastrowidx.def の opt フラグを引き継ぎ、Im は強制凍結(v は実数。共役シムの前提)
- Φ ブロック(長さ = 写像の n_idx): θ ごとの OptFlag を Re・Im の両方に入れる。
  def が無ければ全部可動
"""
function parton_orbparam_materialize_flags!(data::ExpertModeData, map::PartonOrbIndexMap)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    old = data.optimization_flags
    flags = fill(true, 2 * (n_proj + map.n_idx))
    for p = 1:n_proj
        re = 2 * (p - 1) + 1
        flags[re] = re <= length(old) ? old[re] : true
        flags[re + 1] = false
    end
    for j = 1:map.n_idx                     # θ ごとの OptFlag(Re・Im とも)
        base = 2 * (n_proj + j - 1)
        flags[base + 1] = map.opt_flags[j]
        flags[base + 2] = map.opt_flags[j]
    end
    data.optimization_flags = flags
    return flags
end

# def が無いとき(恒等写像。Φ の全要素が可動)
parton_orbparam_materialize_flags!(data::ExpertModeData) = parton_orbparam_materialize_flags!(
    data, parton_orbparam_identity_map(data.modpara.nsite, data.modpara.nelec,
                                       data.modpara.nflavor))

"""
    parton_orbparam_reject_alpha_files(namelist_path)

namelist.def に α モードの入力(`PartonMFTrans` / `PartonMFPara` / `InPmfPara` /
`InPmfOcc`)があれば error。Φ モードに α は存在しないので、混ざっていたら入力の
取り違えとみなす(DESIGN §12.8)。
"""
function parton_orbparam_reject_alpha_files(namelist_path::AbstractString)
    file_list = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(String(namelist_path)))
    alpha_keys = ("PartonMFTrans", "PartonMFPara", "InPmfPara", "InPmfOcc")
    for (file_type, _) in file_list
        file_type in alpha_keys && error(
            "PartonOrbParam = 1 (direct orbitals) does not use `$file_type`; remove it " *
            "from $(namelist_path). The orbitals come from InPmfOrb only.")
    end
    return nothing
end

"""
    validate_parton_orbparam_inputs(data, ctx)

Φ 直接変分モード(PartonOrbParam = 1)の門番。非破壊。α モードと共通の検査
(modpara・フレーバー整合・物理項・QP・並列)を再利用し、Φ 固有の検査を足す
(DESIGN §12.8 / §12.11)。
"""
function validate_parton_orbparam_inputs(data::ExpertModeData, ctx::ParallelContext,
                                          map::PartonOrbIndexMap)
    mp = data.modpara
    validate_parton_modpara(mp)

    mp.parton_orb_param == PARTON_ORB_PARAM_THOULESS && error(
        "PartonOrbParam = 2 (Thouless parametrization) is reserved and not implemented " *
        "yet. Use PartonOrbParam = 1 (direct orbitals).")
    mp.parton_orb_param == PARTON_ORB_PARAM_DIRECT || error(
        "validate_parton_orbparam_inputs needs PartonOrbParam = 1, got " *
        "$(mp.parton_orb_param).")
    (isempty(data.pmfpara_terms) && isempty(data.pmftrans_terms)) || error(
        "PartonOrbParam = 1 (direct orbitals) does not use pmfpara.def / pmftrans.def; " *
        "remove PartonMFPara and PartonMFTrans from namelist.def.")

    validate_parton_flavor_consistency(data)
    validate_parton_data(data; require_mf_inputs = false)
    validate_parton_qp(data)
    validate_parton_parallel(ctx, mp)

    mp.parton_occ_mode == PARTON_OCC_AUFBAU || error(
        "PartonOccMode = $(mp.parton_occ_mode) has no meaning in the direct-orbital " *
        "mode (there is no occupation to choose); set it to 0 or remove it.")
    mp.dsr_opt_sta_del > 0 || error(
        "PartonOrbParam = 1 needs DSROptStaDel > 0 (got $(mp.dsr_opt_sta_del)): the " *
        "GL(NElec) redundancy of each determinant is an exact null space of the S " *
        "matrix, so the unshifted Cholesky solve breaks down.")
    (isfinite(mp.parton_orb_amp_max) && mp.parton_orb_amp_max > 0) || error(
        "PartonOrbAmpMax must be a positive finite number, got $(mp.parton_orb_amp_max).")

    n_theta = map.n_idx                     # 恒等写像なら NFlavor·NSite·NElec
    length(data.pmforb_params) == n_theta || error(
        "The orbital parameters hold $(length(data.pmforb_params)) entries but " *
        (map.is_identity ? "NFlavor × NSite × NElec" : "NPmfOrbIdx in PartonOrbIdx") *
        " = $n_theta are required. Check InPmfOrb against modpara.def / PartonOrbIdx.")
    all(isfinite, data.pmforb_params) || error("The orbitals contain non-finite values.")

    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    expected = 2 * (n_proj + n_theta)
    length(data.optimization_flags) == expected || error(
        "optimization_flags has length $(length(data.optimization_flags)) but the " *
        "direct-orbital mode needs exactly $expected (2 slots x (NProj = $n_proj + " *
        "n_theta = $n_theta)). stochastic_opt! reads an out-of-range flag as 'frozen' " *
        "without warning. Call parton_orbparam_materialize_flags! before this gate.")
    any(i -> data.optimization_flags[i], (2 * n_proj + 1):expected) || error(
        "Every orbital slot in optimization_flags is frozen, so SR has nothing to " *
        "optimize.")
    return nothing
end

# def が無いとき(恒等写像)
validate_parton_orbparam_inputs(data::ExpertModeData, ctx::ParallelContext) =
    validate_parton_orbparam_inputs(
        data, ctx, parton_orbparam_identity_map(data.modpara.nsite, data.modpara.nelec,
                                                data.modpara.nflavor))

# =====================================================================
# PartonOrbIdx(Φ の制限。Pfaffian の orbitalidx.def 相当、DESIGN §12.12)
# =====================================================================

const _PARTON_ORBIDX_UNIT_TOL = 1e-10   # 係数の |c| = 1 の許容

"""
    parton_orbparam_write_orbidx(path, map, n_site, n_elec, n_flavor) -> String

写像を PartonOrbIdx の書式で書く。5 行ヘッダ(`NPmfOrbIdx N` / `ComplexType 1`)のあと、
全要素を `site flavor orbital idx [c]`(0-based、フレーバー外側・サイト・軌道の順)で書く。
c は 1 なら省略、−1 なら `-1`(Pfaffian 互換)、それ以外は `Re Im`。
続けて `idx OptFlag` を N 行。
"""
function parton_orbparam_write_orbidx(path::AbstractString, map::PartonOrbIndexMap,
                                      n_site::Int, n_elec::Int, n_flavor::Int)
    dir = dirname(abspath(String(path)))
    isempty(dir) || mkpath(dir)
    open(String(path), "w") do io
        println(io, "===============================")
        println(io, "NPmfOrbIdx $(map.n_idx)")
        println(io, "ComplexType 1")
        println(io, "===============================")
        println(io, "===============================")
        for f = 1:n_flavor, s = 1:n_site, n = 1:n_elec
            k = parton_orbparam_flat_index(f, s, n, n_site, n_elec)
            c = map.coeff[k]
            @printf(io, "%d %d %d %d", s - 1, f - 1, n - 1, map.idx[k] - 1)
            if c == -1
                print(io, " -1")
            elseif c != 1
                @printf(io, " % .18e % .18e", real(c), imag(c))
            end
            println(io)
        end
        for j = 1:map.n_idx
            @printf(io, "%d %d\n", j - 1, map.opt_flags[j] ? 1 : 0)
        end
    end
    return String(path)
end

"""
    parton_orbparam_read_orbidx(data, namelist_path) -> Union{Nothing,PartonOrbIndexMap}

namelist.def の `PartonOrbIdx` が指す def を読んで写像を返す。キーが無ければ nothing
(制限なし = 恒等写像で走る)。**検査はすべて error で、部分適用はしない**:

- ヘッダ 2 行目が `NPmfOrbIdx N`(N ≥ 1)
- 要素行は 4〜6 列。site・flavor・orbital・idx が範囲内。5 列なら符号 ±1、
  6 列なら |Re + i·Im| = 1(許容 1e-10)
- 全 (site, flavor, orbital) がちょうど 1 回(Pfaffian と同じく全要素の記載が必須)
- `idx OptFlag` が各 idx ちょうど 1 回、値は 0 か 1
- どの要素からも使われない idx が無い
"""
function parton_orbparam_read_orbidx(data::ExpertModeData, namelist_path::AbstractString)
    base_dir = dirname(abspath(String(namelist_path)))
    file_list = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(String(namelist_path)))
    path = nothing
    for (file_type, file_path) in file_list
        file_type == "PartonOrbIdx" && (path = joinpath(base_dir, file_path))
    end
    path === nothing && return nothing
    isfile(path) || error("PartonOrbIdx file listed in namelist.def but not found: $path")

    mp = data.modpara
    n_site, n_elec, n_flavor = mp.nsite, mp.nelec, mp.nflavor
    n_orb = n_flavor * n_site * n_elec
    bad(i, msg) = error("PartonOrbIdx: $path line $i: $msg")

    lines = readlines(path)
    length(lines) >= 5 || error("PartonOrbIdx: $path is too short to hold the 5-line header.")
    head = split(strip(lines[2]))
    (length(head) >= 2 && head[1] == "NPmfOrbIdx") ||
        bad(2, "expected 'NPmfOrbIdx <N>', got '$(lines[2])'.")
    n_idx = something(tryparse(Int, head[2]), -1)
    n_idx >= 1 || bad(2, "invalid parameter count '$(head[2])'.")

    idx = zeros(Int, n_orb)
    coeff = ones(ComplexF64, n_orb)
    opt = fill(-1, n_idx)
    n_elements = 0
    in_flags = false
    for (i, line) in enumerate(lines)
        i <= 5 && continue
        tok = split(strip(line))
        isempty(tok) && continue
        if length(tok) == 2
            in_flags = true
            j = tryparse(Int, tok[1])
            flag = tryparse(Int, tok[2])
            (j === nothing || flag === nothing) && bad(i, "could not parse '$line'.")
            0 <= j < n_idx || bad(i, "idx $j is out of range [0, $(n_idx - 1)].")
            flag in (0, 1) || bad(i, "OptFlag must be 0 or 1, got $flag.")
            opt[j + 1] == -1 || bad(i, "OptFlag of idx $j appears twice.")
            opt[j + 1] = flag
            continue
        end
        in_flags && bad(i, "element lines must come before the 'idx OptFlag' lines.")
        4 <= length(tok) <= 6 || bad(i, "expected 'site flavor orbital idx [c]', got '$line'.")
        v = [tryparse(Int, t) for t in tok[1:4]]
        any(isnothing, v) && bad(i, "could not parse '$line'.")
        s, f, n, j = v
        0 <= s < n_site || bad(i, "site $s is out of range [0, $(n_site - 1)].")
        0 <= f < n_flavor || bad(i, "flavor $f is out of range [0, $(n_flavor - 1)].")
        0 <= n < n_elec || bad(i, "orbital $n is out of range [0, $(n_elec - 1)].")
        0 <= j < n_idx || bad(i, "idx $j is out of range [0, $(n_idx - 1)].")
        c = ComplexF64(1)
        if length(tok) == 5
            sg = tryparse(Int, tok[5])
            sg in (1, -1) || bad(i, "the sign must be 1 or -1, got '$(tok[5])'.")
            c = ComplexF64(sg)
        elseif length(tok) == 6
            re = tryparse(Float64, tok[5])
            imv = tryparse(Float64, tok[6])
            (re === nothing || imv === nothing) && bad(i, "could not parse the coefficient.")
            c = ComplexF64(re, imv)
            abs(abs(c) - 1) <= _PARTON_ORBIDX_UNIT_TOL ||
                bad(i, "the coefficient must have |c| = 1, got |c| = $(abs(c)).")
        end
        k = parton_orbparam_flat_index(f + 1, s + 1, n + 1, n_site, n_elec)
        idx[k] == 0 || bad(i, "element (site $s, flavor $f, orbital $n) appears twice.")
        idx[k] = j + 1
        coeff[k] = c
        n_elements += 1
    end
    n_elements == n_orb || error(
        "PartonOrbIdx: $path lists $n_elements elements but NFlavor × NSite × NElec = " *
        "$n_orb are required (every element must be listed, as in orbitalidx.def).")
    missing_flag = findfirst(==(-1), opt)
    missing_flag === nothing || error(
        "PartonOrbIdx: $path has no OptFlag line for idx $(missing_flag - 1).")
    used = falses(n_idx)
    for j in idx
        used[j] = true
    end
    unused = findfirst(!, used)
    unused === nothing || error(
        "PartonOrbIdx: $path declares idx $(unused - 1) but no element uses it.")
    return PartonOrbIndexMap(idx, coeff, opt .== 1, n_site, n_elec, n_flavor)
end
