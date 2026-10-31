"""
α モードの解から Φ 直接変分モード(PartonOrbParam = 1)の入力一式を作る
--- parton-mode (fork addition) ---

使い方:

    julia --project=MVMCOptimizers.jl tools/parton_alpha_to_orbparam.jl \\
        <alpha_stage_in> <alpha_stage_out> <new_dir>

- `<alpha_stage_in>`: α run の入力ディレクトリ(namelist.def がある場所)
- `<alpha_stage_out>`: その出力ディレクトリ(`<CParaFileHead>_pmfpara_opt.dat`、
  あれば `_pmfocc_opt.dat` と `_jastrow_opt.dat`)
- `<new_dir>`: Φ モードの入力を書く先

処理(DESIGN_parton.md §12.11 / §12.1「α 解からの開始」):

1. α* と占有から軌道を組み直し(MOM の run は占有を読まないと Φ が変わる)、
   `InPmfOrb.def` に書く。値は α の軌道そのもの(再スケールは Φ ドライバが行う)
2. `pmftrans.def` / `pmfpara.def` 以外の def を複製する。namelist からは
   `PartonMFTrans` / `PartonMFPara` / `InPmfPara` / `InPmfOcc` / `InJastrow` の行を除き、
   `InPmfOrb` を足す
3. `_jastrow_opt.dat` があれば 4 行ヘッダを 5 行に直して `InJastrow.def` にする
   (汎用リーダは 5 行を読み飛ばすので、そのままだと idx 0 が落ちる。DESIGN §11)
4. modpara は `PartonOrbParam 1`・`PartonOccMode 0` に置き換え、他の行はそのまま残す。
   `DSROptStaDel <= 0` なら error(Φ モードの門番と同じ条件)
"""

using MVMCExpertModeParsers
using MVMCOptimizers

const _ORBPARAM_DROPPED_KEYS = ("PartonMFTrans", "PartonMFPara", "InPmfPara", "InPmfOcc", "InJastrow")

"Jastrow の per-block 出力(4 行ヘッダ)を InJastrow.def(5 行ヘッダ)に直す。"
function _orbparam_jastrow_to_in(src::AbstractString, dst::AbstractString)
    lines = readlines(src)
    first_data = findfirst(lines) do l
        tok = split(strip(l))
        length(tok) >= 2 && !occursin("=", l) && tryparse(Int, tok[1]) !== nothing
    end
    first_data === nothing && error("Could not find the data rows of $src")
    open(dst, "w") do io
        for k = 1:(first_data - 1)
            println(io, lines[k])
        end
        for _ = first_data:5                      # ヘッダを 5 行にそろえる
            println(io, "===============================")
        end
        for k = first_data:length(lines)
            println(io, lines[k])
        end
    end
    return dst
end

"""
    parton_alpha_to_orbparam(alpha_in, alpha_out, new_dir) -> String

Φ モードの入力一式を `new_dir` に書き、namelist.def のパスを返す。
"""
function parton_alpha_to_orbparam(alpha_in::AbstractString, alpha_out::AbstractString,
                                  new_dir::AbstractString)
    mkpath(new_dir)
    nl_in = joinpath(alpha_in, "namelist.def")
    data = MVMCExpertModeParsers.parse_expert_mode_files(nl_in)
    mp = data.modpara
    head = isempty(mp.c_para_file_head) ? "zqp" : mp.c_para_file_head

    # 1. α* と占有から軌道を組む(既存の読み手を使うため、絶対パスの一時 namelist を書く)
    alpha_file = abspath(joinpath(alpha_out, head * "_pmfpara_opt.dat"))
    occ_file = abspath(joinpath(alpha_out, head * "_pmfocc_opt.dat"))
    isfile(alpha_file) || error("α の最終値が見つかりません: $alpha_file")
    tmp = joinpath(new_dir, "_alpha_to_orbparam_namelist.def")
    write(tmp, "InPmfPara $alpha_file\n" * (isfile(occ_file) ? "InPmfOcc $occ_file\n" : ""))
    try
        MVMCOptimizers.parton_read_in_pmfpara!(data, tmp)
        occ = MVMCOptimizers.parton_read_in_pmfocc(data, tmp)
        mfham = MVMCOptimizers.PartonMFHamiltonian(
            mp.nsite, mp.nelec, mp.nflavor, MVMCOptimizers.parton_n_idx(data))
        MVMCOptimizers.parton_build_mf_templates!(mfham, data)
        MVMCOptimizers.parton_update_orbitals!(
            mfham, MVMCOptimizers.parton_alpha_from_terms(data), mp.nelec; forced_occ = occ)
        data.pmforb_params = MVMCOptimizers.parton_orbparam_params_from_orbitals(mfham.orbitals)
    finally
        rm(tmp; force = true)
    end
    MVMCOptimizers.parton_orbparam_write_pmforb(data, joinpath(new_dir, "InPmfOrb.def"))

    # 2. def の複製と namelist
    entries = MVMCExpertModeParsers.parse_namelist_content(
        MVMCExpertModeParsers.read_def_file(nl_in))
    kept = Tuple{String,String}[]
    for (key, path) in entries
        key in _ORBPARAM_DROPPED_KEYS && continue
        dst = joinpath(new_dir, path)
        mkpath(dirname(dst))
        cp(joinpath(alpha_in, path), dst; force = true)
        push!(kept, (key, path))
    end
    push!(kept, ("InPmfOrb", "InPmfOrb.def"))

    # 3. Jastrow の最終値
    jastrow_file = joinpath(alpha_out, head * "_jastrow_opt.dat")
    if isfile(jastrow_file)
        _orbparam_jastrow_to_in(jastrow_file, joinpath(new_dir, "InJastrow.def"))
        push!(kept, ("InJastrow", "InJastrow.def"))
    end
    open(joinpath(new_dir, "namelist.def"), "w") do io
        for (key, path) in kept
            println(io, rpad(key, 15), path)
        end
    end

    # 4. modpara(ModPara の行が指すファイルを書き換える)
    mod_path = joinpath(new_dir, only(p for (k, p) in kept if k == "ModPara"))
    mp.dsr_opt_sta_del > 0 || error(
        "DSROptStaDel = $(mp.dsr_opt_sta_del): the direct-orbital mode needs a positive " *
        "value (the GL(NElec) redundancy is an exact null space of S).")
    lines = filter(l -> !startswith(strip(l), "PartonOrbParam") &&
                        !startswith(strip(l), "PartonOccMode"), readlines(mod_path))
    push!(lines, "PartonOrbParam 1", "PartonOccMode 0")
    write(mod_path, join(lines, "\n") * "\n")
    return joinpath(new_dir, "namelist.def")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || error(
        "使い方: julia --project=MVMCOptimizers.jl tools/parton_alpha_to_orbparam.jl " *
        "<alpha_stage_in> <alpha_stage_out> <new_dir>")
    println(parton_alpha_to_orbparam(ARGS[1], ARGS[2], ARGS[3]))
end
