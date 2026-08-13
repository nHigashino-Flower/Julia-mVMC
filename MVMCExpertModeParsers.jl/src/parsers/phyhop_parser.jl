"""
PhysHop Parser  --- parton-mode (fork addition) ---

physhop.def のパーサ: 物理ハミルトニアンの合成粒子ホッピング項
t_ij b†_j b_i(b_i = Π_f f^(f)_i)。

形式(0-based サイト・複素値2列・片方向列挙+暗黙 h.c.):
    ==================
    NPhysHop 4
    ==================
    == site1 site2 Re Im ==
    ==================
    0 1  -1.0  0.0
    1 2  -1.0  0.0
    ...

規約:
- 同一結合の両方向 (i,j) と (j,i) の列挙は入力エラー(h.c. はランタイムが供給し、
  E_loc が t 側・t* 側の両方向を評価する)。検査は門番(parton_unsupported_inputs.jl)。
- site1 == site2 は不可(化学ポテンシャル μ n_i は coulombinter の対角行
  V n_i n_i = V n_i(硬芯)で表現する)。検査は門番。
- パーサ自身は家風どおり「忠実な読み手」: ヘッダ行は静かにスキップし、
  宣言個数 NPhysHop と実行数の不一致のみエラーに積む。

貼り込み先: MVMCExpertModeParsers.jl/src/parsers/physhop_parser.jl
(ファイル末尾に登録点3箇所+門番のスニペットをコメントで添付)
"""

"""
    parse_physhop_def(filepath::String) -> ParseResult{Vector{PhysHopTerm}}

physhop.def をファイルパスから読む。
"""
function parse_physhop_def(filepath::String)::ParseResult{Vector{PhysHopTerm}}
    try
        content = read_def_file(filepath)
        return parse_physhop_content(content)
    catch e
        return ParseResult{Vector{PhysHopTerm}}(false, nothing, "Error reading file: $e", 0)
    end
end

"""
    parse_physhop_content(content::String) -> ParseResult{Vector{PhysHopTerm}}

physhop.def の中身を文字列から読む。
"""
function parse_physhop_content(content::String)::ParseResult{Vector{PhysHopTerm}}
    context = ParsingContext("physhop.def")
    terms = PhysHopTerm[]
    declared = -1

    lines = split(content, '\n')
    for (line_num, line) in enumerate(lines)
        context.line_number = line_num
        cl = clean_line(line)
        isempty(cl) && continue

        tokens = split_def_line(cl)
        isempty(tokens) && continue

        # 数字で始まらない行はヘッダ(家風: 静かにスキップ)。NPhysHop だけ拾う。
        if tryparse(Int, String(tokens[1])) === nothing
            if length(tokens) >= 2 && String(tokens[1]) == "NPhysHop"
                v = tryparse(Int, String(tokens[2]))
                v !== nothing && (declared = v)
            end
            continue
        end

        length(tokens) < 4 && continue
        try
            term = parse_physhop_term(tokens, context)
            term !== nothing && push!(terms, term)
        catch e
            push!(context.errors, "Line $line_num: Error parsing PhysHop term: $e")
        end
    end

    if declared >= 0 && declared != length(terms)
        push!(context.errors,
              "NPhysHop=$declared declared but $(length(terms)) terms parsed")
    end

    success = isempty(context.errors)
    return ParseResult{Vector{PhysHopTerm}}(
        success,
        success ? terms : nothing,
        join(context.errors, "; "),
        context.line_number,
    )
end

"""
    parse_physhop_term(tokens, context) -> Union{PhysHopTerm, Nothing}

1 行分: `site1 site2 Re Im`(0-based のまま格納。1-based 変換はランタイム側の一箇所)。
"""
function parse_physhop_term(
    tokens::Vector{<:AbstractString},
    context::ParsingContext,
)::Union{PhysHopTerm,Nothing}
    site1 = tryparse(Int, String(tokens[1]))
    site2 = tryparse(Int, String(tokens[2]))
    re    = tryparse(Float64, String(tokens[3]))
    imv   = tryparse(Float64, String(tokens[4]))
    if site1 === nothing || site2 === nothing || re === nothing || imv === nothing
        push!(context.errors,
              "Line $(context.line_number): invalid PhysHop row (need: site1 site2 Re Im)")
        return nothing
    end
    return PhysHopTerm(site1, site2, ComplexF64(re, imv), imv != 0.0)
end

# =====================================================================
# 登録スニペット(各既存ファイルの分岐末尾へ、fork マーカー付きで)
# =====================================================================
#
# ① types/expert_types.jl — Term 構造体群の末尾(パートングループ内):
#
#     # --- parton-mode (fork addition) ---
#     struct PhysHopTerm            # 物理ハミルトニアンの定義 = 不変(値は変分でない)
#         site1      :: Int
#         site2      :: Int
#         value      :: ComplexF64
#         is_complex :: Bool
#     end
#
#   ExpertModeData 本体の一番下にフィールド追加+内部コンストラクタ new(...) 末尾に既定値:
#
#     physhop_terms :: Vector{PhysHopTerm}     # フィールド
#     PhysHopTerm[],                           # new(...) 末尾
#
# ② utils/constants.jl — MVMC_KEYWORDS 表の末尾:
#
#     # --- parton-mode (fork addition) ---
#     "PhysHop" => "physhop.def",
#
# ③ 入口ファイル parse_file_by_type! の elseif 連鎖末尾:
#
#     # --- parton-mode (fork addition) ---
#     elseif file_type == "PhysHop"
#         result = parse_physhop_def(file_path)
#         result.success && (data.physhop_terms = result.data)
#         # 失敗時は家風どおり警告経由で続行(致命化しない)。
#         # 「読めているべきものが読めているか」は門番が検査する(下記④)
#
# ④ parton_unsupported_inputs.jl — validate_parton_data に追記:
#
#     # PhysHop: 存在・範囲・自己ループ禁止・逆向き重複禁止
#     isempty(data.physhop_terms) && error(
#         "No physical hopping terms parsed: physhop.def missing or failed. " *
#         "Check parser warnings and the PhysHop entry in namelist.def.")
#     seen = Set{NTuple{2,Int}}()
#     for (k, t) in enumerate(data.physhop_terms)
#         0 <= t.site1 < n_site && 0 <= t.site2 < n_site || error(
#             "physhop.def term $k: site out of range")
#         t.site1 != t.site2 || error(
#             "physhop.def term $k: site1 == site2 is not allowed " *
#             "(chemical potential goes to coulombinter diagonal rows)")
#         (t.site1, t.site2) in seen && error("physhop.def: duplicate bond at term $k")
#         (t.site2, t.site1) in seen && error(
#             "physhop.def term $k: both directions listed — list once, h.c. is implicit")
#         push!(seen, (t.site1, t.site2))
#     end
# =====================================================================