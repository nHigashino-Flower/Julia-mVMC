"""
PartonMFTrans Parser  --- parton-mode (fork addition) ---

pmftrans.def のパーサ: パートン平均場ハミルトニアンの固定係数 t。
変分パラメータ α(pmfpara.def)と結合キー (site1, flavor1, site2, flavor2) で
対応し、H^(f)_ij(α) = α_{idx(i,j,f)} · t^(f)_ij を成す(DESIGN §1.2)。

形式(0-based サイト・フレーバー、複素値 2 列):

    ====================
    NPartonMFTrans 4
    ====================
    == site1 flavor1 site2 flavor2 Re Im ==
    ====================
    0 0 1 0  -1.0  0.0
    ...

規約(検査は結合を要するのでランタイム側が家 — DESIGN §2.2):
- 片方向のみ列挙し h.c. は暗黙付与。逆向きの重複列挙はエラー(門番/build)
- オンサイト行(site1 == site2)は h.c. なしの直接加算で、t は実数必須(build)
- flavor1 == flavor2 の要求(M1)は門番
パーサ自身は家風どおり「忠実な読み手」: ヘッダ行は静かにスキップし、
宣言個数 NPartonMFTrans と実行数の不一致のみエラーに積む。
"""

"""
    parse_parton_mf_trans_def(filepath::String) -> ParseResult{Vector{PartonMFTransTerm}}

pmftrans.def をファイルパスから読む。
"""
function parse_parton_mf_trans_def(
    filepath::String,
)::ParseResult{Vector{PartonMFTransTerm}}
    try
        content = read_def_file(filepath)
        return parse_parton_mf_trans_content(content)
    catch e
        return ParseResult{Vector{PartonMFTransTerm}}(
            false,
            nothing,
            "Error reading file: $e",
            0,
        )
    end
end

"""
    parse_parton_mf_trans_content(content::String) -> ParseResult{Vector{PartonMFTransTerm}}

pmftrans.def の中身を文字列から読む。
"""
function parse_parton_mf_trans_content(
    content::String,
)::ParseResult{Vector{PartonMFTransTerm}}
    context = ParsingContext("pmftrans.def")
    terms = PartonMFTransTerm[]
    declared = -1

    lines = split(content, '\n')
    for (line_num, line) in enumerate(lines)
        context.line_number = line_num
        cl = clean_line(line)
        isempty(cl) && continue

        tokens = split_def_line(cl)
        isempty(tokens) && continue

        # 数字で始まらない行はヘッダ(家風: 静かにスキップ)。NPartonMFTrans だけ拾う。
        if tryparse(Int, String(tokens[1])) === nothing
            if length(tokens) >= 2 && String(tokens[1]) == "NPartonMFTrans"
                v = tryparse(Int, String(tokens[2]))
                v !== nothing && (declared = v)
            end
            continue
        end

        length(tokens) < 6 && continue
        try
            term = parse_parton_mf_trans_term(tokens, context)
            term !== nothing && push!(terms, term)
        catch e
            push!(
                context.errors,
                "Line $line_num: Error parsing parton mean-field transfer term: $e",
            )
        end
    end

    if declared >= 0 && declared != length(terms)
        push!(
            context.errors,
            "NPartonMFTrans=$declared declared but $(length(terms)) terms parsed",
        )
    end

    success = isempty(context.errors)
    return ParseResult{Vector{PartonMFTransTerm}}(
        success,
        success ? terms : nothing,
        join(context.errors, "; "),
        context.line_number,
    )
end

"""
    parse_parton_mf_trans_term(tokens, context) -> Union{PartonMFTransTerm, Nothing}

1 行分: `site1 flavor1 site2 flavor2 Re Im`(0-based のまま格納。1-based 変換は
ランタイムのテンプレート build の一箇所だけ)。
"""
function parse_parton_mf_trans_term(
    tokens::Vector{<:AbstractString},
    context::ParsingContext,
)::Union{PartonMFTransTerm,Nothing}
    site1 = tryparse(Int, String(tokens[1]))
    flavor1 = tryparse(Int, String(tokens[2]))
    site2 = tryparse(Int, String(tokens[3]))
    flavor2 = tryparse(Int, String(tokens[4]))
    re = tryparse(Float64, String(tokens[5]))
    imv = tryparse(Float64, String(tokens[6]))

    if any(isnothing, (site1, flavor1, site2, flavor2, re, imv))
        push!(
            context.errors,
            "Line $(context.line_number): invalid PartonMFTrans row " *
            "(need: site1 flavor1 site2 flavor2 Re Im)",
        )
        return nothing
    end

    return PartonMFTransTerm(
        site1,
        flavor1,
        site2,
        flavor2,
        ComplexF64(re, imv),
        imv != 0.0,
    )
end
