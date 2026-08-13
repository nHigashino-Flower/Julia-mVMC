"""
PartonMFTrans.def Parser

Parser for pmftrans.def files containing parton mean-field transfer (hopping) and interaction terms as cofficients coupled to parton mean-field parameters.
"""

"""
    parse_parton_mf_trans_def(filepath::String) -> ParseResult{Vector{PartonMFTransTerm}}

Parse pmftrans.def file from file path.
"""
function parse_parton_mf_trans_def(filepath::String)::ParseResult{Vector{TransferTerm}}
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

Parse pmftrans.def content from string.
"""
function parse_parton_mf_trans_content(content::String)::ParseResult{Vector{PartonMFTransTerm}}
    context = ParsingContext("pmftrans.def")
    terms = PartonMFTransTerm[]

    lines = split(content, '\n')

    for (line_num, line) in enumerate(lines)
        context.line_number = line_num
        clean_line_str = clean_line(line)

        if isempty(clean_line_str)
            continue
        end

        tokens = split_def_line(clean_line_str)
        if length(tokens) < 3
            continue
        end

        try
            term = parse_parton_mf_trans_term(tokens, context)
            if term !== nothing
                push!(terms, term)
            end
        catch e
            push!(context.errors, "Line $line_num: Error parsing parton mean-field transfer term: $e")
        end
    end

    success = length(context.errors) == 0
    return ParseResult{Vector{PartonMFTransTerm}}(
        success,
        success ? terms : nothing,
        join(context.errors, "; "),
        context.line_number,
    )
end

"""
    parse_parton_mf_trans_term(tokens::Vector{String}, context::ParsingContext) -> Union{PartonMFTransTerm, Nothing}

Parse a single parton mean-field transfer term from tokens.
Format: site1 site2 real_value imag_value
"""
function parse_parton_mf_trans_term(
    tokens::Vector{String},
    context::ParsingContext,
)::Union{PartonMFTransTerm,Nothing}
    if length(tokens) < 4
        push!(
            context.warnings,
            "Insufficient tokens for partom mean-field transfer term (need 4: site1 site2 real imag)",
        )
        return nothing
    end

    # Parse site indices and spin indices
    # Format: site1 site2 real imag
    site1 = safe_parse_int(tokens[1], -1)
    site2 = safe_parse_int(tokens[2], -1)

    if site1 < 0 || site2 < 0
        push!(context.errors, "Invalid site indices: $site1, $site2")
        return nothing
    end


    # Parse complex value (real and imaginary parts)
    real_val = safe_parse_float(tokens[3], 0.0)
    imag_val = safe_parse_float(tokens[4], 0.0)
    value = ComplexF64(real_val, imag_val)

    return PartonMFTransTerm(site1, site2, value)
end
