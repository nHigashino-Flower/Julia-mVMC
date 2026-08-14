"""
α の初期値経路(乱数初期化 + InPmfPara.def によるウォームスタート)
--- parton-mode (fork addition) ---

DESIGN_parton.md §2.3。modpara に初期化モードのキーは足さない。スイッチは 2 つとも
**入力の形そのもの**:

    pmfpara.def の各行:
       value 列が未入力 (5 列) → 乱数で初期化
       value 列が入力あり (7 列) → その値を採用(0 でも 0 を採用。乱数で埋めない)
            ↓ 上書き(namelist.def に InPmfPara が書かれているときだけ)
    InPmfPara.def            → ウォームスタート / リスタート

「未入力」と「0 を指定」を**列の有無で区別する**。値がゼロかどうかで分岐する判定は
採らない — 意図的に α = 0 で始めたい結合を黙って乱数で埋めないため。既存 mVMC の
`InOrbital.def` の有無がモード指定になる作法にも揃う。

## 確定順序(DESIGN §2.3・§3.3)

    pmfpara.def 読み込み(presence 判定)
      → 未入力 idx を乱数で生成      … parton_init_alpha!
      → InPmfPara.def で上書き(あれば) … parton_read_in_pmfpara!
      → 門番検証                      … validate_parton_inputs
      → gauge_target_norm 確定        … parton_build_mf_templates! の中
      → SR ループ

ゲージ射影の引き戻し先は初期 α のノルムなので、**乱数と上書きが済んでから**
テンプレート build を呼ぶこと。順序を崩すと引き戻し先が乱数前の値になる。
"""

"""
    parton_validate_value_presence(data)

同一 idx を共有する行は value の presence が揃っていること。一部だけ値ありは
エラー(既存の「同一 idx の value 一致検証」の自然な拡張)。片方だけ乱数で
埋まると同じパラメータに 2 つの初期値が生じてしまう。
"""
function parton_validate_value_presence(data::ExpertModeData)
    seen = Dict{Int,Bool}()
    for (k, t) in enumerate(data.pmfpara_terms)
        prev = get(seen, t.idx, nothing)
        if prev === nothing
            seen[t.idx] = t.has_value
        elseif prev != t.has_value
            error(
                "pmfpara.def term $k: rows sharing idx $(t.idx) disagree on whether a " *
                "value column is present. Cells sharing an idx share one variational " *
                "parameter, so they must either all give a value or all omit it — " *
                "otherwise the same parameter would get two different initial values.",
            )
        end
    end
    return nothing
end

"""
    parton_init_alpha!(data, base_seed; amplitude=1.0)

value 列が未入力の idx にだけ初期値を生成する。

- **オンサイト群は実数のみ**。虚部はエルミート性(t が実数)により門番が凍結するので、
  乱数でも 0 を入れる。ホッピング群は複素
- 振幅は既定 1.0。**群全体が乱数なら全体スケールはゲージ自由度なので物理に効かない**が、
  同一ゲージ群内に指定値と乱数が混在する場合は**相対スケールが効く**ので、その入力では
  振幅の選び方が結果に影響する。混在させるときは指定値のスケールに合わせること
- RNG は**専用ストリーム**。解決済みのベースシード(ランクごとのオフセットを加える前の
  値)で初期化するので、全ランクが構成的に同一の α を得る(bcast 不要)。
  **サンプリング用 RNG を消費しない** — 消費すると同一入力でもサンプリング系列が
  変わり、既存の再現性が壊れる

返り値は乱数で埋めた idx の集合(0-based)。
"""
function parton_init_alpha!(data::ExpertModeData, base_seed::Integer;
                            amplitude::Float64 = 1.0)
    parton_validate_value_presence(data)

    n_idx = parton_n_idx(data)
    n_idx == 0 && return Set{Int}()

    # idx ごとの「未入力か」と「オンサイト群か」を集める
    needs_random = falses(n_idx)
    for t in data.pmfpara_terms
        t.has_value || (needs_random[t.idx + 1] = true)
    end
    any(needs_random) || return Set{Int}()

    onsite = parton_onsite_idx_set(data)

    # 専用ストリーム。サンプリング用 RNG には触れない。
    rng = SFMT19937RNG()
    Random.seed!(rng, Int(base_seed))

    α = zeros(ComplexF64, n_idx)
    for k = 1:n_idx
        needs_random[k] || continue
        if (k - 1) in onsite
            # オンサイトは実数のみ(Im は非物理でどのみち凍結される)
            α[k] = ComplexF64(amplitude * (2 * rng_real2(rng) - 1), 0.0)
        else
            α[k] = ComplexF64(amplitude * (2 * rng_real2(rng) - 1),
                              amplitude * (2 * rng_real2(rng) - 1))
        end
    end

    filled = Set{Int}()
    for t in data.pmfpara_terms
        t.has_value && continue
        t.value = α[t.idx + 1]
        push!(filled, t.idx)
    end
    return filled
end

"""
    parton_read_in_pmfpara!(data, namelist_path) -> Bool

`namelist.def` に `InPmfPara` が書かれているときだけ、そのファイルの値で
`pmfpara_terms[].value` を直接上書きする(α の正準置き場は 1 つのまま。
新しいフィールドは作らない)。

形式は既存 In*.def と同じ 5 行ヘッダ + `idx Re Im`。パラメータ数が n_idx と
一致しない、あるいは idx が範囲外なら**エラー**(黙って部分適用しない)。

返り値は上書きを行ったかどうか。
"""
function parton_read_in_pmfpara!(data::ExpertModeData, namelist_path::AbstractString)
    base_dir = dirname(abspath(String(namelist_path)))
    file_list = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(String(namelist_path)))

    path = nothing
    for (file_type, file_path) in file_list
        file_type == "InPmfPara" && (path = joinpath(base_dir, file_path))
    end
    path === nothing && return false
    isfile(path) || error("InPmfPara file listed in namelist.def but not found: $path")

    params = MVMCExpertModeParsers.parse_input_parameter_file(path)
    n_idx = parton_n_idx(data)
    length(params) == n_idx || error(
        "InPmfPara: $path holds $(length(params)) parameters but pmfpara.def declares " *
        "$n_idx. Refusing to apply it partially — a warm start must cover every " *
        "mean-field parameter.",
    )
    for idx in keys(params)
        0 <= idx < n_idx || error(
            "InPmfPara: $path contains idx $idx which is out of range [0, $(n_idx - 1)].")
    end

    for t in data.pmfpara_terms
        t.value = params[t.idx]
    end
    return true
end

"""
    parton_write_pmfpara(data, path)

α を In*.def 互換形式で書き出す。**初期値のダンプ(§2.3.1)と最適化後の per-block
出力(`zqp_pmfpara_opt.dat`)で共有する唯一の writer**で、形式を二重管理しない。

書式(既存 per-block writer と同じ体裁だが**ヘッダは 5 行**):

    ===============================
    NPmfParaIdx <n_idx>
    ===============================
    == idx ReAlpha ImAlpha ==
    ===============================
    <idx 0-based> <Re %.18e> <Im %.18e>

読み手は既存の汎用実装 `parse_input_parameter_file`(`idx Re Im` の 3 列を読む)を
そのまま使う。新規パーサは書かない。

**ヘッダ 5 行にする理由(実測済み)**: `parse_input_parameter_file` は
`data_start = 6`(ヘッダ 5 行読み飛ばし)で実装されているのに、既存 3 ブロックの
writer が出すヘッダは 4 行しかない。そのまま往復させると **idx = 0 の行が脱落**し、
しかも件数不一致は `@warn` 止まりなので警告 1 行で素通りする(3 個書いて
`[1, 2]` が返ることを実測で確認)。パートン側は 5 行にして自分の往復を成立させる。
**既存 3 ブロックの writer には触らない** — C-parity の出力比較に影響しうるため。
既存側の不一致は upstream への報告候補として DESIGN §11 に記録してある。
"""
function parton_write_pmfpara(data::ExpertModeData, path::AbstractString)
    n_idx = parton_n_idx(data)
    α = parton_alpha_from_terms(data)
    dir = dirname(abspath(String(path)))
    isempty(dir) || mkpath(dir)
    open(String(path), "w") do f
        println(f, "===============================")
        println(f, "NPmfParaIdx $n_idx")
        println(f, "===============================")
        println(f, "== idx ReAlpha ImAlpha ==")
        println(f, "===============================")
        for k = 1:n_idx
            @printf(f, "%d % .18e % .18e \n", k - 1, real(α[k]), imag(α[k]))
        end
    end
    return String(path)
end

"""
    parton_write_pmfocc(data, mfham, path) -> String

占有集合 `O` を `.def` 族(DESIGN §3.3.1 系統 (a))で書く。

    ===============================
    NPmfOcc <n_flavor × Ne>
    ===============================
    == flavor band_index ==
    ===============================
    <flavor 0-based> <band_index 0-based>

**占有集合は導出量ではなく状態の一部**(REPORT §15)。`PartonOccMode = 1` では
α → Φ が履歴依存になるので α だけでは状態を再現できない。`(α*, O*)` の組で
状態が完全に決まるように、α と同じタイミングで必ず書く。

行順は `(flavor, band_index)` の辞書順に固定する(run 間で `diff` が取れること)。
`aufbau` の run でも書く — そちらは α だけで再現できるが、ファイルの有無で
読み手の経路が分岐すると壊れやすいため。
"""
function parton_write_pmfocc(
    data::ExpertModeData,
    mfham::PartonMFHamiltonian,
    path::AbstractString,
)
    n_occ = sum(length, mfham.occ)
    dir = dirname(abspath(String(path)))
    isempty(dir) || mkpath(dir)
    open(String(path), "w") do f
        println(f, "===============================")
        println(f, "NPmfOcc $n_occ")
        println(f, "===============================")
        println(f, "== flavor band_index ==")
        println(f, "===============================")
        for fl in eachindex(mfham.occ)
            for band in mfham.occ[fl]           # occ は昇順(契約 0 が保証)
                @printf(f, "%d %d\n", fl - 1, band - 1)
            end
        end
    end
    return String(path)
end

"""
    parton_read_in_pmfocc(data, namelist_path) -> Union{Nothing,Vector{Vector{Int}}}

`namelist.def` に `InPmfOcc` が書かれているときだけ、占有集合を読んで
**フレーバーごとの 1-based・昇順 band index** として返す。書かれていなければ
`nothing`(= 初期占有は aufbau。既定挙動は変わらない)。

形式は `zqp_pmfocc_opt.dat` と**同一**(`.def` 族、5 行ヘッダ、キーワード
`NPmfOcc`、データ行 `flavor band_index`、どちらも 0-based)。出力した
`_pmfocc_opt.dat` / `_pmfocc_init.dat` をそのまま渡せる。

`parse_input_parameter_file` は `idx Re Im` の 3 列専用でこの形式には使えないので、
既存の汎用ヘルパ(`read_def_file` / `clean_line` / `split_def_line`)を組み合わせた
最小のリーダをここに置く。**新規パーサファイルは作らない**(DESIGN §3.1)。

検証は**読んだ直後に全部**行い、ひとつでも破れたらエラーで止める(部分適用しない):

- ヘッダの件数と実データの行数が一致
- 行数が `n_flavor × Ne` と一致
- `flavor ∈ [0, NFlavor)`、`band_index ∈ [0, NSite)`
- 各フレーバーちょうど Ne 本、かつ重複なし

なぜ必要か(REPORT §16-5): `PartonOccMode = 1` の run は終端が非アウフバウ占有に
なるため、α\\* 単独では状態を再現できない。ウォームスタート / 鎖方式・PhysCal で
最適化済み状態から始めるには占有も渡す必要がある。
"""
function parton_read_in_pmfocc(data::ExpertModeData, namelist_path::AbstractString)
    base_dir = dirname(abspath(String(namelist_path)))
    file_list = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(String(namelist_path)))

    path = nothing
    for (file_type, file_path) in file_list
        file_type == "InPmfOcc" && (path = joinpath(base_dir, file_path))
    end
    path === nothing && return nothing
    isfile(path) || error("InPmfOcc file listed in namelist.def but not found: $path")

    lines = split(MVMCExpertModeParsers.read_def_file(String(path)), '\n')
    length(lines) >= 5 || error("InPmfOcc: $path is too short to hold the 5-line header.")

    declared = -1
    header = MVMCExpertModeParsers.split_def_line(
        MVMCExpertModeParsers.clean_line(lines[2]))
    length(header) >= 2 && (declared = MVMCExpertModeParsers.safe_parse_int(header[2], -1))

    # 読み取りは形式(トークン数と整数パース)だけを見る。意味の検証(範囲・本数・
    # 重複・件数)は門番 `validate_parton_occupation` が一手に引き受ける(DESIGN §2)。
    rows = Tuple{Int,Int}[]
    for i = 6:length(lines)
        line = MVMCExpertModeParsers.clean_line(lines[i])
        isempty(line) && continue
        tok = MVMCExpertModeParsers.split_def_line(line)
        length(tok) >= 2 || error(
            "InPmfOcc: $path line $i: expected 'flavor band_index', got '$line'.")
        fl = _parse_int_strict_local(tok[1], path, i, "flavor")
        band = _parse_int_strict_local(tok[2], path, i, "band_index")
        push!(rows, (fl, band))
    end

    validate_parton_occupation(rows, declared, data.modpara, path)

    n_flavor = data.modpara.nflavor
    occ = [Int[] for _ = 1:n_flavor]
    for (fl, band) in rows
        push!(occ[fl + 1], band + 1)        # 0-based → 1-based はここで 1 回だけ
    end
    foreach(sort!, occ)
    return occ
end

"`InPmfOcc` の整数フィールドを厳密にパースする(位置つきエラー)。"
function _parse_int_strict_local(tok::AbstractString, path, line_num::Int, field::String)
    v = tryparse(Int, tok)
    v === nothing && error("InPmfOcc: $path line $line_num: invalid $field '$tok'.")
    return v
end
