"""
PartonPairOrbital Parser  --- parton-mode (fork addition) ---

Pfaffian パートンモード(PartonMode = 2)の対軌道 f の idx 構造(DESIGN_parton_pf.md §2.1)。
書式は上流の orbitalidxgen.def(OrbitalGeneral)と同じで、spin の列をフレーバーと読む。
K = 2 なら上流の def がそのまま通る(Julia 移植の OrbitalGeneral パーサは 6 列を
`site1 site2 idx sgn` と誤読するので使わない)。

    =============================================
    NOrbitalIdx          3
    ComplexType          1
    =============================================
    =============================================
    0 0  1 1  0          <- site1 flavor1 site2 flavor2 idx        (符号の省略は +1)
    1 0  0 1  1 -1       <- site1 flavor1 site2 flavor2 idx sign
    0 1                  <- 末尾の OptFlag 行: idx flag
    1 0

パーサは**書式だけ**を厳格に見て、違反はエラーにする(上流は範囲外を黙って捨てる):

- データ行は 5 列か 6 列の整数。符号は ±1、idx は 0 以上
- 末尾の OptFlag 行は 2 列の整数
- ヘッダの `NOrbitalIdx`(宣言数)= 使われている idx の数(max idx + 1)
- `ComplexType` は 1 だけ(実数の f は未対応)

範囲・対角・重複・idx の連番はテンプレート build(`PartonPairOrbital` の構築)が検査する
(パースの時点では NSite / NFlavor が確定していないことがあるため)。
"""

"""
    parse_parton_pair_orbital_def(filepath)
        -> Tuple{ParseResult{Vector{PartonPairOrbitalTerm}}, Dict{Int,Int}, Int}

返り値は (行のパース結果, OptFlag 辞書(0-based idx -> flag), 宣言数 NOrbitalIdx)。
"""
function parse_parton_pair_orbital_def(
    filepath::String,
)::Tuple{ParseResult{Vector{PartonPairOrbitalTerm}},Dict{Int,Int},Int}
    try
        content = read_def_file(filepath)
        return parse_parton_pair_orbital_content(content)
    catch e
        return (
            ParseResult{Vector{PartonPairOrbitalTerm}}(false, nothing, "Error reading file: $e", 0),
            Dict{Int,Int}(),
            0,
        )
    end
end

"""
    parse_parton_pair_orbital_content(content)
        -> Tuple{ParseResult{Vector{PartonPairOrbitalTerm}}, Dict{Int,Int}, Int}

文字列から読む。数字で始まらない行はヘッダ(`NOrbital*` と `ComplexType` だけ拾う)。
"""
function parse_parton_pair_orbital_content(
    content::String,
)::Tuple{ParseResult{Vector{PartonPairOrbitalTerm}},Dict{Int,Int},Int}
    context = ParsingContext("PartonPairOrbital")
    terms = PartonPairOrbitalTerm[]
    opt_flags = Dict{Int,Int}()
    declared = -1
    complex_type = -1

    for (line_num, line) in enumerate(split(content, '\n'))
        context.line_number = line_num
        cl = clean_line(line)
        isempty(cl) && continue
        tokens = split_def_line(cl)
        isempty(tokens) && continue

        if tryparse(Int, String(tokens[1])) === nothing
            if length(tokens) >= 2
                key = String(tokens[1])
                v = tryparse(Int, String(tokens[2]))
                if startswith(key, "NOrbital")
                    v !== nothing && (declared = v)
                elseif key == "ComplexType"
                    v !== nothing && (complex_type = v)
                end
            end
            continue
        end

        ints = [tryparse(Int, String(t)) for t in tokens]
        if any(isnothing, ints)
            push!(context.errors, "Line $line_num: PartonPairOrbital rows must be integers: '$cl'")
            continue
        end
        if length(ints) == 2
            idx, flag = ints
            if idx < 0
                push!(context.errors, "Line $line_num: OptFlag row idx must be >= 0, got $idx")
            elseif haskey(opt_flags, idx)
                push!(context.errors, "Line $line_num: OptFlag for idx $idx is given twice")
            else
                opt_flags[idx] = flag
            end
            continue
        end
        if !(length(ints) in (5, 6))
            push!(context.errors,
                  "Line $line_num: PartonPairOrbital row has $(length(ints)) columns; expected " *
                  "5 (site1 flavor1 site2 flavor2 idx) or 6 (… idx sign), or 2 for an OptFlag row")
            continue
        end
        s1, f1, s2, f2, idx = ints[1:5]
        sign = length(ints) == 6 ? ints[6] : 1
        if idx < 0
            push!(context.errors, "Line $line_num: idx must be >= 0, got $idx")
            continue
        end
        if !(sign == 1 || sign == -1)
            push!(context.errors, "Line $line_num: sign must be +1 or -1, got $sign")
            continue
        end
        push!(terms, PartonPairOrbitalTerm(s1, f1, s2, f2, idx, sign))
    end

    n_used = isempty(terms) ? 0 : maximum(t.idx for t in terms) + 1
    if declared >= 0 && declared != n_used
        push!(context.errors,
              "PartonPairOrbital declares NOrbitalIdx = $declared but the rows use " *
              "$n_used parameters (max idx + 1)")
    end
    if complex_type != -1 && complex_type != 1
        push!(context.errors,
              "PartonPairOrbital needs ComplexType 1 (complex pair parameters); got " *
              "ComplexType $complex_type. Real pair orbitals are not supported yet.")
    end

    success = isempty(context.errors)
    result = ParseResult{Vector{PartonPairOrbitalTerm}}(
        success, success ? terms : nothing, join(context.errors, "; "), context.line_number)
    return (result, opt_flags, declared >= 0 ? declared : n_used)
end
