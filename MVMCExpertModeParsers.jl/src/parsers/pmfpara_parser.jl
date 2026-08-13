"""
PartonMFPara Parser

Parser for pmfpara.def files containing parton mean-field parameter terms.
"""

"""
    parse_parton_mf_params_def(filepath::String) -> Tuple{ParseResult{Vector{PartonMFParaTerm}}, Dict{Int, Int}, Int}

Parse orbital.def file from file path.
Returns parton mean-field parameter terms, OptFlag dictionary (idx -> opt_flag), and the header
declared parameter count (`NPartonMFParaIdx`). The declared count
is the authoritative parameter count even when some indices are unreferenced.
"""
function parse_parton_mf_params_def(
    filepath::String,
)::Tuple{ParseResult{Vector{PartonMFParaTerm}},Dict{Int,Int},Int}
    try
        content = read_def_file(filepath)
        return parse_parton_mf_params_content(content)
    catch e
        return (
            ParseResult{Vector{PartonMFParaTerm}}(false, nothing, "Error reading file: $e", 0),
            Dict{Int,Int}(),
            0,
        )
    end
end

"""
    parse_parton_mf_params_content(content::String) -> Tuple{ParseResult{Vector{PartonMFParaTerm}}, Dict{Int, Int}, Int}

Parse pmfpara.def content from string.
Returns parton mean-field parameter terms, OptFlag dictionary (idx -> opt_flag), and the header
declared parameter count (0 for the headerless legacy format, where it falls
back to `max(idx)+1`).
"""
function parse_parton_mf_params_content(
    content::String,
)::Tuple{ParseResult{Vector{PatonMFParaTerm}},Dict{Int,Int},Int}
    context = ParsingContext("pmfpara.def")
    terms = OrbitalTerm[]

    lines = split(content, '\n')

    # Skip header lines (first 5 lines) for pmfpara.def format
    # Format: =============================================
    #         NPartonMFParaIdx          N
    #         ComplexType          flag
    #         =============================================
    #         =============================================
    # Then data lines: site1 flavor1 site2 flavor2 idx (NPartonMFParaIdx lines), then idx opt_flag (NPara lines)
    IGNORE_LINES_IN_DEF = 5
    start_line = 1

    # Read NPartonMFParaIdx from header (line 2: "NPartonMFParaIdx N")
    n_partonmfpara_idx = 0
    has_header = false
    if length(lines) > 1
        header_line = clean_line(lines[2])
        tokens = split_def_line(header_line)
        # Accept any NPartonMFPara* header keyword (NPartonMFParaIdx,
        if length(tokens) >= 2 && startswith(tokens[1], "NPartonMFPara")
            n_partonmfpara_idx = safe_parse_int(tokens[2], 0)
            has_header = true
        end
    end

    # Read ComplexType from header (line 3: "ComplexType flag")
    complex_type = 0
    if has_header && length(lines) > 2
        complex_type_line = clean_line(lines[3])
        tokens = split_def_line(complex_type_line)
        if length(tokens) >= 2 && tokens[1] == "ComplexType"
            complex_type = safe_parse_int(tokens[2], 0)
        end
    end
    # Convert to boolean: 0 = false (real), != 0 = true (complex)
    is_complex_flag = (complex_type != 0)

    # Check if this looks like pmfpara.def format (has header)
    if length(lines) > IGNORE_LINES_IN_DEF && has_header
        first_data_line = clean_line(lines[IGNORE_LINES_IN_DEF+1])
        if !isempty(first_data_line)
            tokens = split_def_line(first_data_line)
            # If first data line has at least 5 tokens (site1 flavor1 site2 flavor2 idx), skip header
            if length(tokens) >= 5
                start_line = IGNORE_LINES_IN_DEF + 1
            end
        end
    end

    # If no header, process all lines normally (orbital.def format)
    if !has_header
        start_line = 1
    end

    # Parse all PartonMFParaIdx entries (5-column lines: site1 flavor1 site2 flavor2 idx)
    # Note: pmfpara.def has N_site * N_site * Nflavor * Nflavor PartonMFParaIdx entries (64 for 4 sites and 2 flavors),
    # followed by NPartonMFParaIdx OptFlag entries (idx opt_flag format, 2 columns).
    # The NPartonMFParaIdx value is the number of unique parton mean-field parameters, NOT the number of PartonMFParaIdx entries.
    processing_partonmfpara_idx = true  # Flag to track if we're still in PartonMFParaIdx section
    opt_flags = Dict{Int,Int}()  # Store OptFlag: idx -> opt_flag

    for line_num = start_line:length(lines)
        line = lines[line_num]
        context.line_number = line_num
        clean_line_str = clean_line(line)

        if isempty(clean_line_str)
            continue
        end

        tokens = split_def_line(clean_line_str)

        # 2-column lines are OptFlag (idx opt_flag format)
        # Once we see 2-column lines after 5-column lines, we're done with PartonMFParaIdx
        if length(tokens) == 2
            if processing_partonmfpara_idx && !isempty(terms)
                # We've finished PartonMFParaIdx section, now in OptFlag section
                processing_partonmfpara_idx = false
            end

            # Parse OptFlag: idx opt_flag
            if !processing_partonmfpara_idx
                try
                    idx = safe_parse_int(tokens[1], -1)
                    opt_flag = safe_parse_int(tokens[2], 0)
                    if idx >= 0
                        opt_flags[idx] = opt_flag
                    end
                catch e
                    push!(context.errors, "Line $line_num: Error parsing OptFlag: $e")
                end
            end
            continue
        end

        if length(tokens) < 5
            continue
        end

        # Only process 5-column lines in PartonMFParaIdx section
        if !processing_partonmfpara_idx
            continue
        end

        try
            # Pass is_complex_flag to parse_orbital_term
            term = parse_parton_mf_params_term(tokens, context, is_complex_flag)
            if term !== nothing
                push!(terms, term)
            end
        catch e
            push!(context.errors, "Line $line_num: Error parsing parton mean-field params term: $e")
        end
    end

    # Note: NPartonMFParaIdx is the number of unique parton mean-field parameters, not the number of PartonMFParaIdx entries.
    # All PartonMFParaIdx entries (site pairs) should be parsed, even if there are more than NPartonMFParaIdx.

    success = length(context.errors) == 0
    result = ParseResult{Vector{PartonMFParaTerm}}(
        success,
        success ? terms : nothing,
        join(context.errors, "; "),
        context.line_number,
    )
    # Declared parameter count from the header.so it is the
    # authoritative count even when some indices are unreferenced by site pairs.
    # Fall back to max(idx)+1 only for the headerless legacy pmfpara.def format.
    declared_count = if has_header
        n_partonmfpara_idx
    elseif !isempty(terms)
        maximum(t.idx for t in terms) + 1
    else
        0
    end
    return (result, opt_flags, declared_count)
end

"""
    parse_parton_mf_params_term(tokens::Vector{String}, context::ParsingContext, is_complex_flag::Bool) -> Union{PartonMFParaTerm, Nothing}

Parse a single parton mean-field parameter term from tokens.
For pmfpara.def format, tokens[5] is the parton mean-field parameter index (idx), not a value.
The is_complex_flag comes from the ComplexType header in the file.
"""
function parse_parton_mf_params_term(
    tokens::Vector{String},
    context::ParsingContext,
    is_complex_flag::Bool = false,
)::Union{PartonMFParaTerm,Nothing}
    if length(tokens) < 5
        push!(context.warnings, "Insufficient tokens for parton mean-field parameter term")
        return nothing
    end

    # Parse site and flavor indices
    site1 = safe_parse_int(tokens[1], -1)
    flavor1 = safe_parse_int(tokens[2], -1)
    site2 = safe_parse_int(tokens[3], -1)
    flavor2 = safe_parse_int(tokens[4], -1)
    
    if site1 < 0 || site2 < 0
        push!(context.errors, "Invalid site indices: $site1, $site2")
        return nothing
    end

    if flavor1 < 0 || flavor2 < 0
        push!(context.errors, "Invalid flavor indices: $flavor1, $flavor2")
        return nothing
    end

    # Parse fifth token 
    # For pmfpara.def: this is the parton mean-field parameter index (idx), not a value
    # For pmfpara.def (legacy format): this might be a value
    # Try to parse as integer first (pmfpara.def format)
    idx = safe_parse_int(tokens[5], -1)

    if idx >= 0
        # This is pmfpara.def format: site1 flavor1 site2 flavor2 idx
        # Value is not specified here, initialize to 0.0
        # The actual value will be set from InPartonMFPara.def or during initialization
        value = ComplexF64(0.0, 0.0)
        # Store idx in PartonMFParaTerm
        return PartonMFParaTerm(site1, flavor1, site2, flavor2, idx, value, is_complex_flag)
    else
        # Try to parse as a value (legacy pmfpara.def format)
        value = safe_parse_complex(tokens[5])
        # For legacy format, check if value has imaginary part
        # But prefer the is_complex_flag if provided
        if is_complex_flag
            # Use flag from header
        elseif imag(value) != 0.0
            # Fallback to checking value if flag not provided
            is_complex_flag = true
        end
        # Use is_complex_flag from ComplexType header
        return PartonMFParaTerm(site1, flavor1, site2, flavor2, 0, value, is_complex_flag)
    end
end
