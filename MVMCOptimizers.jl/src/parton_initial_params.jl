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
    ===============================
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
        println(f, "===============================")
        println(f, "===============================")
        for k = 1:n_idx
            @printf(f, "%d % .18e % .18e \n", k - 1, real(α[k]), imag(α[k]))
        end
    end
    return String(path)
end
