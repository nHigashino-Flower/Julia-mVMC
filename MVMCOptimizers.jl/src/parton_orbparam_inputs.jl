"""
Φ 直接変分モードの入力(InPmfOrb)・再スケール・OptFlag・門番
--- parton-mode (fork addition) ---

DESIGN_parton.md §12.11(上流型の簡素版)。

- InPmfOrb.def は Φ の唯一の初期値経路(Φ の初期化に乱数は使わない)。読み手は
  厳格で、件数・範囲・重複・有限性・列フルランクのどれかが崩れたら error にする
  (汎用の `parse_input_parameter_file` は読めない値を 0 にし、件数不一致を警告で
  済ませるので使わない。§12.2)
- 再スケールは上流 `sync_modified_parameter!` の D_AmpMax と同じ形で、**毎回**
  max|Φ_f| = PartonOrbAmpMax にそろえる。上流は Slater 全体で 1 つのスケールだが、
  ここは行列式がフレーバーごとにあり、それぞれに定数倍の自由度があるので
  フレーバーごとに行う
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

idx は平坦添字 ((f·Ns)+s)·Ne+n(0-based)。
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
    parton_orbparam_read_in_pmforb!(data, namelist_path) -> String

namelist.def の `InPmfOrb` が指すファイルから Φ を読み、`data.pmforb_params` に置く。
返り値は読んだファイルのパス。Φ モードでは InPmfOrb は**必須**。

検査(どれも error。部分適用はしない):

- namelist に `InPmfOrb` がある、ファイルが存在する
- ヘッダ 2 行目が `NPmfOrb <n>` で、n = 行数 = NFlavor·NSite·NElec
- idx ∈ [0, n)、重複なし、値は有限
- フレーバーごとに Φ_f が列フルランク(σ_min / σ_max > 1e-12)。
  ランク落ちの Φ では行列式が恒等的に 0 で、初期配置が見つからないという
  分かりにくい失敗になる
"""
function parton_orbparam_read_in_pmforb!(data::ExpertModeData, namelist_path::AbstractString)
    base_dir = dirname(abspath(String(namelist_path)))
    file_list = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(String(namelist_path)))
    path = nothing
    for (file_type, file_path) in file_list
        file_type == "InPmfOrb" && (path = joinpath(base_dir, file_path))
    end
    path === nothing && error(
        "PartonOrbParam = 1 needs the initial orbitals: add an `InPmfOrb` line to " *
        "$(namelist_path). The orbitals are never initialised at random.")
    isfile(path) || error("InPmfOrb file listed in namelist.def but not found: $path")

    mp = data.modpara
    n_orb = parton_orbparam_n_orb(mp)
    lines = readlines(path)
    length(lines) >= 5 || error("InPmfOrb: $path is too short to hold the 5-line header.")
    head = split(strip(lines[2]))
    (length(head) >= 2 && head[1] == "NPmfOrb") || error(
        "InPmfOrb: $path line 2 must read 'NPmfOrb <n>', got '$(lines[2])'.")
    declared = tryparse(Int, head[2])
    declared === nothing && error("InPmfOrb: $path line 2: invalid count '$(head[2])'.")

    params = fill(ComplexF64(NaN, NaN), n_orb)
    seen = falses(n_orb)
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
        0 <= idx < n_orb || error(
            "InPmfOrb: $path line $i: idx $idx is out of range [0, $(n_orb - 1)].")
        seen[idx + 1] && error("InPmfOrb: $path line $i: idx $idx appears twice.")
        (isfinite(re) && isfinite(imv)) || error(
            "InPmfOrb: $path line $i: the value of idx $idx is not finite.")
        seen[idx + 1] = true
        params[idx + 1] = ComplexF64(re, imv)
        n_rows += 1
    end
    declared == n_rows || error(
        "InPmfOrb: $path declares NPmfOrb = $declared but holds $n_rows data rows.")
    n_rows == n_orb || error(
        "InPmfOrb: $path holds $n_rows entries but NFlavor × NSite × NElec = $n_orb " *
        "are required. Refusing to apply it partially.")

    for f = 1:mp.nflavor
        Φ = [params[parton_orbparam_flat_index(f, s, n, mp.nsite, mp.nelec)]
             for s = 1:mp.nsite, n = 1:mp.nelec]
        σ = svdvals(Φ)
        (σ[1] > 0 && σ[end] / σ[1] > _PARTON_ORBPARAM_RANK_TOL) || error(
            "InPmfOrb: $path: the orbitals of flavor $(f - 1) are not of full column " *
            "rank (σ_min / σ_max = $(σ[1] > 0 ? σ[end] / σ[1] : 0.0)). A rank-deficient " *
            "Φ makes every determinant vanish.")
    end

    data.pmforb_params = params
    return path
end

"""
    parton_orbparam_rescale!(data) -> Vector{Float64}

上流の D_AmpMax と同じ形の再スケール(DESIGN §12.11)。フレーバーごとに
Φ_f ← (D / max_{s,n}|Φ_f[s,n]|)·Φ_f とする(D = `PartonOrbAmpMax`)。
**max が D 未満でも拡大する**(上流 `parameter_init.jl:274-276` の規約)。
倍率は正の実数なので全体位相は固定しない。

返り値はフレーバーごとの再スケール**前**の max|Φ_f|(漂流の監視用)。
全要素が 0 のフレーバーがあれば error(0 で割らない)。

Φ_f → c·Φ_f は det を c^Ne 倍するだけなので状態は変わらない。O も SR の更新も
c によらない(O は 1/c 倍、S は 1/c² 倍、更新量は c 倍になり相対変化は不変)。
"""
function parton_orbparam_rescale!(data::ExpertModeData)
    mp = data.modpara
    D = mp.parton_orb_amp_max
    block = mp.nsite * mp.nelec
    length(data.pmforb_params) == mp.nflavor * block || error(
        "pmforb_params has $(length(data.pmforb_params)) entries but " *
        "NFlavor × NSite × NElec = $(mp.nflavor * block) are required.")
    xmax = zeros(Float64, mp.nflavor)
    for f = 1:mp.nflavor
        r = ((f - 1) * block + 1):(f * block)
        x = maximum(abs, view(data.pmforb_params, r))
        x > 0 || error(
            "The orbitals of flavor $(f - 1) are identically zero; cannot rescale them " *
            "to PartonOrbAmpMax = $D.")
        ratio = D / x
        for k in r
            data.pmforb_params[k] *= ratio
        end
        xmax[f] = x
    end
    return xmax
end

"""
    parton_orbparam_materialize_flags!(data) -> Vector{Bool}

Φ モードの OptFlag 配列を実体化する(門番より前に呼ぶこと)。`stochastic_opt!` は
範囲外のフラグを黙って「凍結」と読むので、長さを 2(n_proj + n_orb) にそろえる。

- 射影ブロック(Jastrow): α モードの `parton_materialize_flags!` と同じ。Re は
  jastrowidx.def の opt フラグを引き継ぎ、Im は強制凍結(v は実数。共役シムの前提)
- Φ ブロック: 全部可動(Re・Im とも)
"""
function parton_orbparam_materialize_flags!(data::ExpertModeData)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    n_orb = length(data.pmforb_params)
    old = data.optimization_flags
    flags = fill(true, 2 * (n_proj + n_orb))
    for p = 1:n_proj
        re = 2 * (p - 1) + 1
        flags[re] = re <= length(old) ? old[re] : true
        flags[re + 1] = false
    end
    data.optimization_flags = flags
    return flags
end

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
function validate_parton_orbparam_inputs(data::ExpertModeData, ctx::ParallelContext)
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

    n_orb = parton_orbparam_n_orb(mp)
    length(data.pmforb_params) == n_orb || error(
        "The orbitals hold $(length(data.pmforb_params)) entries but NFlavor × NSite × " *
        "NElec = $n_orb are required. Check InPmfOrb against modpara.def.")
    all(isfinite, data.pmforb_params) || error("The orbitals contain non-finite values.")

    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    expected = 2 * (n_proj + n_orb)
    length(data.optimization_flags) == expected || error(
        "optimization_flags has length $(length(data.optimization_flags)) but the " *
        "direct-orbital mode needs exactly $expected (2 slots x (NProj = $n_proj + " *
        "n_orb = $n_orb)). stochastic_opt! reads an out-of-range flag as 'frozen' " *
        "without warning. Call parton_orbparam_materialize_flags! before this gate.")
    any(i -> data.optimization_flags[i], (2 * n_proj + 1):expected) || error(
        "Every orbital slot in optimization_flags is frozen, so SR has nothing to " *
        "optimize.")
    return nothing
end
