"""
PartonMFPara Parser  --- parton-mode (fork addition) ---

pmfpara.def のパーサ: パートン平均場の変分パラメータ α と idx 写像。
α の正準置き場は `PartonMFParaTerm.value` で、SR のパラメータロケータが
直接読み書きする(DESIGN §1.2)。

形式(0-based サイト・フレーバー・idx、複素値 2 列):

    =============================================
    NPartonMFParaIdx  2
    ComplexType       1
    =============================================
    =============================================
    0 0 1 0  0  -1.0  0.0      <- site1 flavor1 site2 flavor2 idx Re Im
    1 1 2 1  1  -1.0  0.0
    0 1                        <- 末尾フラグ行: idx OptFlag
    1 0

規約(DESIGN §2.3):
- idx は 0-based の連番。同一 idx の複数セル(フレーバー跨ぎ可)は α の共有で、
  value の一致検証と連番性の検査はテンプレート build / 門番が行う
- 末尾フラグ行はゲージ固定・固定項の表現(欠落ではなく OptFlag 凍結)
パーサ自身は家風どおり「忠実な読み手」に徹する。
"""

"""
    parse_parton_mf_para_def(filepath::String)
        -> Tuple{ParseResult{Vector{PartonMFParaTerm}}, Dict{Int,Int}, Int}

pmfpara.def をファイルパスから読む。返り値は (項のパース結果, OptFlag 辞書
(0-based idx -> flag), ヘッダ宣言のパラメータ数 NPartonMFParaIdx)。
"""
function parse_parton_mf_para_def(
    filepath::String,
)::Tuple{ParseResult{Vector{PartonMFParaTerm}},Dict{Int,Int},Int}
    try
        content = read_def_file(filepath)
        return parse_parton_mf_para_content(content)
    catch e
        return (
            ParseResult{Vector{PartonMFParaTerm}}(
                false,
                nothing,
                "Error reading file: $e",
                0,
            ),
            Dict{Int,Int}(),
            0,
        )
    end
end

"""
    parse_parton_mf_para_content(content::String)
        -> Tuple{ParseResult{Vector{PartonMFParaTerm}}, Dict{Int,Int}, Int}

pmfpara.def の中身を文字列から読む。7 トークン行が α セル、2 トークン行が
末尾の OptFlag 行。
"""
function parse_parton_mf_para_content(
    content::String,
)::Tuple{ParseResult{Vector{PartonMFParaTerm}},Dict{Int,Int},Int}
    context = ParsingContext("pmfpara.def")
    terms = PartonMFParaTerm[]
    opt_flags = Dict{Int,Int}()
    declared = -1
    is_complex_flag = false

    lines = split(content, '\n')
    for (line_num, line) in enumerate(lines)
        context.line_number = line_num
        cl = clean_line(line)
        isempty(cl) && continue

        tokens = split_def_line(cl)
        isempty(tokens) && continue

        # 数字で始まらない行はヘッダ。NPartonMFParaIdx と ComplexType だけ拾う。
        if tryparse(Int, String(tokens[1])) === nothing
            if length(tokens) >= 2
                key = String(tokens[1])
                if startswith(key, "NPartonMFPara")
                    v = tryparse(Int, String(tokens[2]))
                    v !== nothing && (declared = v)
                elseif key == "ComplexType"
                    v = tryparse(Int, String(tokens[2]))
                    v !== nothing && (is_complex_flag = v != 0)
                end
            end
            continue
        end

        if length(tokens) == 2
            # 末尾フラグ行: idx OptFlag
            idx = tryparse(Int, String(tokens[1]))
            flag = tryparse(Int, String(tokens[2]))
            if idx === nothing || flag === nothing || idx < 0
                push!(
                    context.errors,
                    "Line $line_num: invalid PartonMFPara OptFlag row (need: idx flag)",
                )
            else
                opt_flags[idx] = flag
            end
            continue
        end

        length(tokens) < 7 && continue
        try
            term = parse_parton_mf_para_term(tokens, context, is_complex_flag)
            term !== nothing && push!(terms, term)
        catch e
            push!(
                context.errors,
                "Line $line_num: Error parsing parton mean-field parameter term: $e",
            )
        end
    end

    success = isempty(context.errors)
    result = ParseResult{Vector{PartonMFParaTerm}}(
        success,
        success ? terms : nothing,
        join(context.errors, "; "),
        context.line_number,
    )

    # ヘッダの宣言値が一意なパラメータ数。ヘッダが無い場合のみ max(idx)+1 で代替。
    declared_count = if declared >= 0
        declared
    elseif !isempty(terms)
        maximum(t.idx for t in terms) + 1
    else
        0
    end
    return (result, opt_flags, declared_count)
end

"""
    parse_parton_mf_para_term(tokens, context, is_complex_flag) -> Union{PartonMFParaTerm, Nothing}

1 行分: `site1 flavor1 site2 flavor2 idx Re Im`(0-based のまま格納)。
"""
function parse_parton_mf_para_term(
    tokens::Vector{<:AbstractString},
    context::ParsingContext,
    is_complex_flag::Bool = false,
)::Union{PartonMFParaTerm,Nothing}
    site1 = tryparse(Int, String(tokens[1]))
    flavor1 = tryparse(Int, String(tokens[2]))
    site2 = tryparse(Int, String(tokens[3]))
    flavor2 = tryparse(Int, String(tokens[4]))
    idx = tryparse(Int, String(tokens[5]))
    re = tryparse(Float64, String(tokens[6]))
    imv = tryparse(Float64, String(tokens[7]))

    if any(isnothing, (site1, flavor1, site2, flavor2, idx, re, imv))
        push!(
            context.errors,
            "Line $(context.line_number): invalid PartonMFPara row " *
            "(need: site1 flavor1 site2 flavor2 idx Re Im)",
        )
        return nothing
    end
    if idx < 0
        push!(
            context.errors,
            "Line $(context.line_number): PartonMFPara idx must be >= 0, got $idx",
        )
        return nothing
    end

    return PartonMFParaTerm(
        site1,
        flavor1,
        site2,
        flavor2,
        idx,
        ComplexF64(re, imv),
        is_complex_flag,
    )
end
