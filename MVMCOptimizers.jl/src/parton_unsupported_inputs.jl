"""
Input contract for the parton mean-field VMC mode (`PartonMode = 1`).
--- parton-mode (fork addition) ---

This file is the parton-mode counterpart of `unsupported_inputs.jl`. It is a
fork addition: it is never called from the standard mVMC path, and the standard
path never calls into it, so `PartonMode = 0` runs are bit-identical to
upstream.

Layering follows the existing convention. The parser
(`MVMCExpertModeParsers.jl`) stays a faithful reader of the input format; the
runtime owns the "what is actually supported" contract. Two deliberate
differences from `unsupported_inputs.jl`:

1. It also checks *physical* consistency (particle counts, filling), not just
   feature support. Upstream's physical checks live in the parser package's
   `validation.jl`, which is not wired into the production path, so the parton
   mode cannot rely on them.
2. It checks that inputs the parton mode *requires* were actually parsed.
   Unknown modpara keys are only warnings, and per-file parse failures are
   swallowed by `parse_expert_mode_files`, so a typo can silently leave a
   default in place. Anything this mode depends on is asserted loudly here.

Checks that need the *coupling* between pmftrans and pmfpara (bidirectional
completeness, onsite/hopping mixing within one idx group, the reality of onsite
t, idx contiguity) live in `parton_build_mf_templates!` instead: that function
already builds the joining dictionary, and duplicating it here would mean two
implementations of the same rule. This file checks what can be decided per row.

Particle numbers are not stored redundantly. `NElec` (partons per flavor) is
the only stored field; `NParticle` and `NPartonPerFlavor` are parser-level
aliases that write into it, and the derived names (`n_phys_particle`,
`n_parton_total`, `n_parton_per_flavor`) are accessor functions in
`parton_types.jl`.

Call order from `parton_run_para_opt_from_namelist` / `parton_vmc_para_opt!`:
the driver materialises `optimization_flags` *before* calling
`validate_parton_inputs`, because one of the checks here is that the flag array
covers the mean-field block (DESIGN §2.5).
"""

const PARTON_MODE_OFF = 0
const PARTON_MODE_MEAN_FIELD = 1

# 占有集合の選び方(REPORT §15、DESIGN §1.1)。
# `aufbau` は「下から Ne 個」= v3.11 までの唯一の規則。`mom` は前ステップの占有
# 部分空間との重なりで選ぶ(枝を連続に追う)。値 2 以降は予約。
const PARTON_OCC_AUFBAU = 0
const PARTON_OCC_MOM = 1

"""
    is_parton_mode(modpara::ModParaParameters) -> Bool

True when the input selects a parton mode. `PartonMode` is the *only* switch:
`NFlavor` carries data, never mode selection, so that a missing or misspelled
`NFlavor` line fails loudly instead of silently reverting to standard mVMC.
"""
is_parton_mode(modpara::ModParaParameters) = modpara.parton_mode != PARTON_MODE_OFF

"""
    parton_n_idx(data::ExpertModeData) -> Int

Number of distinct mean-field variational parameters (`max(idx) + 1`). Same
formula the parameter locator uses, so the two cannot drift apart.
"""
parton_n_idx(data::ExpertModeData) =
    isempty(data.pmfpara_terms) ? 0 : maximum(t.idx for t in data.pmfpara_terms) + 1

"""
    validate_parton_modpara(modpara::ModParaParameters)

Reject ModPara inputs the parton mean-field mode does not support, and assert
the settings it silently depends on. Non-mutating.

Rejected or required:

- `PartonMode`: must be `1`. `0` means this entry point was reached by mistake;
  `>= 2` is reserved for future parton ansatze.
- `NFlavor`: must be `> 0`. `0` usually means the `NFlavor` line is missing or
  misspelled — unknown modpara keys are only warnings.
- `2Sz = 0`: the default is `-1`, which the reused machinery reads as
  Sz-non-conserving (FSZ). The parton mode fixes the per-flavor particle
  numbers, so this must be pinned explicitly.
- `NCond = -1`, `NLocSpin = 0`: the upstream relation
  `NElec = (NLocSpin + NCond) / 2` is meaningless once flavors replace spin.
- `ComplexType = 1`, `NSRCG = 0`, `NLanczosMode = 0`, `NVMCCalMode = 0`:
  the single variant implemented in M1.
- `NExUpdatePath = 6`: the gauge constraint that every flavor's partons hop
  with the same coordinate (flavor lock).
- `NSPGaussLeg = 1`, `NSPStot = 0`: spin projection assumes SU(2) and does not
  carry over to general flavors. Momentum projection (qptrans) is unaffected.
- `NOrbitalIdx = 0`, `NNeuron = 0`: the pair-orbital (f_ij) and RBM parameter
  blocks do not coexist with the parton mean-field parameter block.
"""
function validate_parton_modpara(modpara::ModParaParameters)
    # --- mode switch -------------------------------------------------------
    if modpara.parton_mode == PARTON_MODE_OFF
        error(
            "The parton entry point requires PartonMode = 1, got " *
            "PartonMode = 0. Use run_para_opt_from_namelist for standard mVMC.",
        )
    end
    if modpara.parton_mode != PARTON_MODE_MEAN_FIELD
        error(
            "PartonMode = $(modpara.parton_mode) is not implemented: only " *
            "PartonMode = 1 (parton mean-field VMC) exists. Values >= 2 are " *
            "reserved for future parton ansatze.",
        )
    end

    # --- occupation rule ---------------------------------------------------
    if modpara.parton_occ_mode != PARTON_OCC_AUFBAU &&
       modpara.parton_occ_mode != PARTON_OCC_MOM
        error(
            "PartonOccMode = $(modpara.parton_occ_mode) is not implemented: " *
            "only 0 (aufbau, default) and 1 (mom, occupation tracking) exist. " *
            "Values >= 2 are reserved.",
        )
    end

    # --- flavors -----------------------------------------------------------
    if modpara.nflavor <= 0
        error(
            "PartonMode = 1 requires NFlavor > 0, got NFlavor = " *
            "$(modpara.nflavor). If you did set it, check the spelling in " *
            "modpara.def: unknown keywords are only reported as warnings, so " *
            "a typo leaves NFlavor at its default.",
        )
    end

    # --- physical consistency ---------------------------------------------
    if modpara.nsite <= 0
        throw(ArgumentError("NSite must be >= 1; got NSite = $(modpara.nsite)."))
    end
    if modpara.nelec <= 0
        error(
            "Parton mode requires the particle number, got NElec = " *
            "$(modpara.nelec). Give it in modpara.def as NElec, NParticle or " *
            "NPartonPerFlavor (all three write the same field). Note that " *
            "this is the count per flavor, not the total: the total parton " *
            "number is NFlavor * NElec.",
        )
    end
    if modpara.nelec > modpara.nsite
        error(
            "NElec = $(modpara.nelec) exceeds NSite = $(modpara.nsite): NElec " *
            "counts partons per flavor, and each flavor fills at most one " *
            "single-particle state per site. Check whether the intended value " *
            "was the total parton number (NFlavor * NElec = " *
            "$(modpara.nflavor * modpara.nelec)).",
        )
    end
    if modpara.nex_update_path != 6
        error(
            "Parton mode requires NExUpdatePath = 6, got " *
            "$(modpara.nex_update_path). Only that update path enforces the " *
            "gauge constraint that every flavor's partons hop with the same " *
            "coordinate, which is what makes the flavor lock (and hence the " *
            "product-of-determinants ansatz) well defined.",
        )
    end

    # --- quantities whose defaults would silently mislead ------------------
    if modpara.two_sz != 0
        error(
            "Parton mode requires 2Sz = 0, got 2Sz = $(modpara.two_sz). " *
            "The default -1 selects the Sz-non-conserving (FSZ) path, which " *
            "is incompatible with fixed per-flavor particle numbers. Write " *
            "the 2Sz line explicitly in modpara.def.",
        )
    end
    if modpara.ncond != -1
        error(
            "Parton mode requires NCond to be unset (-1), got NCond = " *
            "$(modpara.ncond). The upstream relation " *
            "NElec = (NLocSpin + NCond) / 2 assumes spin-1/2 electrons and " *
            "does not carry over to flavors; give NElec or NParticle instead.",
        )
    end
    if modpara.nlocspin != 0
        error(
            "Parton mode does not support local spins, got NLocSpin = " *
            "$(modpara.nlocspin). Remove the LocSpin file from namelist.def.",
        )
    end

    # --- single implemented variant (M1) -----------------------------------
    if modpara.vmc_calc_mode != 0
        error(
            "Parton mode implements parameter optimization only " *
            "(NVMCCalMode = 0), got NVMCCalMode = $(modpara.vmc_calc_mode). " *
            "Physical measurement for the parton ansatz is not ported yet.",
        )
    end
    if modpara.complex_flag != 1
        error(
            "Parton mode implements the complex path only (ComplexType = 1), " *
            "got ComplexType = $(modpara.complex_flag). The real-valued " *
            "variant of the determinant engine is not implemented.",
        )
    end
    if modpara.nsrcg != 0
        error(
            "Parton mode implements the direct SR solver only (NSRCG = 0), " *
            "got NSRCG = $(modpara.nsrcg). The mean-field parameter count is " *
            "small, so the S matrix is cheap to build explicitly and SR-CG " *
            "brings no benefit.",
        )
    end
    if modpara.lanczos_mode != 0
        error(
            "Parton mode does not support Lanczos steps (NLanczosMode = 0 " *
            "required), got NLanczosMode = $(modpara.lanczos_mode).",
        )
    end

    # --- projections -------------------------------------------------------
    if modpara.nsp_gauss_leg != 1 || modpara.nsp_stot != 0
        error(
            "Parton mode requires spin projection to be off " *
            "(NSPGaussLeg = 1, NSPStot = 0), got NSPGaussLeg = " *
            "$(modpara.nsp_gauss_leg), NSPStot = $(modpara.nsp_stot). " *
            "The Gauss-Legendre spin projection assumes SU(2) and does not " *
            "generalize to flavors. Momentum projection (qptransidx.def) is " *
            "supported.",
        )
    end

    # --- non-coexisting parameter blocks -----------------------------------
    if modpara.n_orbital_idx != 0
        error(
            "Parton mode does not coexist with the pair-orbital block, got " *
            "NOrbitalIdx = $(modpara.n_orbital_idx). Remove the Orbital " *
            "entries from namelist.def: the variational parameters are the " *
            "mean-field couplings, not f_ij.",
        )
    end
    if modpara.nneuron != 0
        error(
            "Parton mode does not support RBM parameters, got NNeuron = " *
            "$(modpara.nneuron).",
        )
    end

    return nothing
end

"""
    validate_parton_flavor_consistency(data::ExpertModeData)

M1 restriction: the ansatz is a product of per-flavor determinants, so a term
may not mix flavors. `PartonMFTransTerm` and `PartonMFParaTerm` keep the general
(flavor1, flavor2) form for a future hybridising ansatz; this gate is what makes
`parton_build_mf_templates!` free to fold the pair down to a single flavor.
"""
function validate_parton_flavor_consistency(data::ExpertModeData)
    for (name, terms) in
        (("pmftrans", data.pmftrans_terms), ("pmfpara", data.pmfpara_terms))
        for (k, t) in enumerate(terms)
            t.flavor1 == t.flavor2 || error(
                "$name.def term $k: flavor mixing (flavor1 = $(t.flavor1), " *
                "flavor2 = $(t.flavor2)) is not supported: the parton ansatz " *
                "is a product of per-flavor determinants. Reserved for a " *
                "future extension.",
            )
        end
    end
    return nothing
end

"""
    validate_parton_data(data::ExpertModeData)

Validate parton-mode settings that need parsed data, not just ModPara.

Non-mutating. In particular this asserts that the mean-field input files were
actually read: `parse_expert_mode_files` continues past per-file parse failures
(matching the C implementation), so a malformed or missing pmfpara.def would
otherwise surface much later as an empty Hamiltonian.

Momentum projection (qptransidx.def) is supported and deliberately not
restricted — it multiplies the determinant product and reuses the existing
quantum-projection machinery unchanged. Gutzwiller and Jastrow are rejected in
M1: the parton local-energy path does not compute their logarithmic
derivatives, so SR would silently leave them frozen (see the NProj check
below).
"""
function validate_parton_data(data::ExpertModeData)
    n_site = data.modpara.nsite
    n_flavor = data.modpara.nflavor

    # --- required inputs ---------------------------------------------------
    if isempty(data.pmfpara_terms)
        error(
            "No mean-field parameter terms were parsed: pmfpara.def is missing, " *
            "empty, or failed to parse. Parse failures of individual .def files " *
            "are not fatal (C-compatible behaviour), so check the parser " *
            "warnings and the PartonMFPara entry in namelist.def.",
        )
    end
    if isempty(data.pmftrans_terms)
        error(
            "No mean-field transfer terms were parsed: pmftrans.def is missing, " *
            "empty, or failed to parse. Check the parser warnings and the " *
            "PartonMFTrans entry in namelist.def.",
        )
    end
    if isempty(data.physhop_terms)
        error(
            "No physical hopping terms were parsed: physhop.def is missing, " *
            "empty, or failed to parse. Check the parser warnings and the " *
            "PhysHop entry in namelist.def. Without it the local energy has no " *
            "off-diagonal part.",
        )
    end

    # --- non-coexisting blocks --------------------------------------------
    if !isempty(data.orbital_terms)
        error(
            "Orbital terms are present ($(length(data.orbital_terms)) terms) " *
            "but the parton ansatz replaces the pair orbital entirely. " *
            "Remove Orbital/OrbitalAntiParallel/OrbitalParallel/OrbitalGeneral " *
            "from namelist.def.",
        )
    end
    if !isempty(data.coulomb_intra_terms)
        error(
            "CoulombIntra is not meaningful in parton mode: the flavor lock " *
            "makes every occupied site carry one parton per flavor, so the " *
            "on-site repulsion is a constant. Express a chemical potential as " *
            "a diagonal row of coulombinter.def instead (V n_i n_i = V n_i " *
            "under the hard-core constraint).",
        )
    end
    if !isempty(data.doublon_holon_2site_indices) ||
       !isempty(data.doublon_holon_4site_indices)
        error(
            "Doublon-holon projection is not supported in parton mode yet. " *
            "Remove the DH2/DH4 entries from namelist.def.",
        )
    end

    # M1 は射影因子なし。二つの理由でここで止める。
    # 1. parton_main_cal! は射影ブロックの O を計算しない(MF ブロックだけ埋める)ので、
    #    射影パラメータがあると O がゼロのまま SR に渡り、黙って最適化されない。
    # 射影因子(v3.11 M2 後半): 物理密度 Jastrow のみ受け入れる。
    # 蓄積境界の共役シム(DESIGN §7)は「MF 以外のスロットの O が実数」を前提に
    # S 行列の不変性を得ている — Jastrow は O = cnt(実数)なので前提を満たす。
    layout = MVMCExpertModeParsers.projection_layout(data)
    if layout.n_gutzwiller != 0
        error(
            "Gutzwiller factors are meaningless in parton mode: under flavor " *
            "locking every occupied site is always a full multiplet, so the " *
            "doublon count is the constant NElec and the factor cancels from " *
            "every ratio. Remove the Gutzwiller entry from namelist.def " *
            "(a density-density Jastrow does not trivialise and is supported).",
        )
    end
    if layout.n_dh2 != 0 || layout.n_dh4 != 0
        error(
            "Doublon-holon factors are not supported in parton mode " *
            "(NDoublonHolon2site = $(layout.n_dh2), 4site = $(layout.n_dh4)). " *
            "They are defined through spin-resolved doublons, which flavor " *
            "locking freezes. Only the density-density Jastrow is supported.",
        )
    end
    if layout.n_jastrow > 0
        # v は実数(DESIGN §7: 射影 O は実数、Im スロットは 0)。
        for (k, t) in enumerate(data.jastrow_terms)
            abs(imag(t.value)) <= 1e-12 || error(
                "jastrow parameter $k has a complex value $(t.value). Parton " *
                "mode requires real Jastrow parameters: the accumulation-" *
                "boundary conjugation (DESIGN section 7) assumes every non-" *
                "mean-field O is real, which holds only for real v.",
            )
        end
        # 全ての非対角ペアに idx が張られていること(未指定 = -1 が残っていると
        # v = 0 ではなく添字事故として静かに壊れる)
        jidx = data.jastrow_idx
        size(jidx) == (n_site, n_site) || error(
            "jastrowidx.def: index matrix has size $(size(jidx)), expected " *
            "($n_site, $n_site).",
        )
        n_jast = layout.n_jastrow
        for i = 1:n_site, j = 1:n_site
            i == j && continue
            0 <= jidx[i, j] < n_jast || error(
                "jastrowidx.def: pair ($(i - 1), $(j - 1)) has no valid index " *
                "(got $(jidx[i, j]), expected 0..$(n_jast - 1)). Every " *
                "off-diagonal pair must be listed — an implicit v = 0 pair is " *
                "expressed as an explicit idx with value 0.",
            )
            jidx[i, j] == jidx[j, i] || error(
                "jastrowidx.def: index for pair ($(i - 1), $(j - 1)) is not " *
                "symmetric ($(jidx[i, j]) vs $(jidx[j, i])).",
            )
        end
    end
    if !isempty(data.opt_trans) || data.n_qp_opt_trans > 1
        error(
            "OptTrans (variational quantum-number projection) is not " *
            "supported in parton mode: it would require updating the QP " *
            "weights every SR step against the mean-field orbitals.",
        )
    end

    # --- per-row range checks ---------------------------------------------
    for (k, t) in enumerate(data.pmfpara_terms)
        _check_site_range("pmfpara.def", k, t.site1, t.site2, n_site)
        _check_flavor_range("pmfpara.def", k, t.flavor1, t.flavor2, n_flavor)
    end
    for (k, t) in enumerate(data.pmftrans_terms)
        _check_site_range("pmftrans.def", k, t.site1, t.site2, n_site)
        _check_flavor_range("pmftrans.def", k, t.flavor1, t.flavor2, n_flavor)
    end

    # --- pmftrans: one direction only (h.c. is implicit) -------------------
    seen_mf = Set{NTuple{3,Int}}()
    for (k, t) in enumerate(data.pmftrans_terms)
        key = (t.site1, t.site2, t.flavor1)
        key in seen_mf &&
            error("pmftrans.def term $k: duplicate entry for $key.")
        if t.site1 != t.site2 && (t.site2, t.site1, t.flavor1) in seen_mf
            error(
                "pmftrans.def term $k: both directions of the bond " *
                "($(t.site1), $(t.site2)) are listed. List each bond once — " *
                "the Hermitian conjugate is supplied implicitly.",
            )
        end
        push!(seen_mf, key)
    end

    # --- physhop: one direction only, no self loops ------------------------
    seen_hop = Set{NTuple{2,Int}}()
    for (k, t) in enumerate(data.physhop_terms)
        _check_site_range("physhop.def", k, t.site1, t.site2, n_site)
        t.site1 != t.site2 || error(
            "physhop.def term $k: site1 == site2 is not allowed. A chemical " *
            "potential belongs on the diagonal of coulombinter.def; a diagonal " *
            "hop would be double counted by the implicit h.c. and the " *
            "both-directions evaluation in the local energy.",
        )
        (t.site1, t.site2) in seen_hop &&
            error("physhop.def term $k: duplicate bond ($(t.site1), $(t.site2)).")
        (t.site2, t.site1) in seen_hop && error(
            "physhop.def term $k: both directions of the bond " *
            "($(t.site1), $(t.site2)) are listed. List each bond once — the " *
            "Hermitian conjugate is supplied implicitly.",
        )
        push!(seen_hop, (t.site1, t.site2))
    end

    return nothing
end

function _check_site_range(file::String, k::Int, site1::Int, site2::Int, n_site::Int)
    for (label, site) in (("site1", site1), ("site2", site2))
        0 <= site < n_site || error(
            "$file term $k: $label = $site is out of range [0, $(n_site - 1)].",
        )
    end
    return nothing
end

function _check_flavor_range(
    file::String,
    k::Int,
    flavor1::Int,
    flavor2::Int,
    n_flavor::Int,
)
    for (label, flavor) in (("flavor1", flavor1), ("flavor2", flavor2))
        0 <= flavor < n_flavor || error(
            "$file term $k: $label = $flavor is out of range " *
            "[0, $(n_flavor - 1)] for NFlavor = $n_flavor.",
        )
    end
    return nothing
end

"""
    validate_parton_opt_flags(data::ExpertModeData)

Check the OptFlag array against the parameter count (DESIGN §2.5).

`stochastic_opt!` treats an out-of-range flag index as "frozen" and says
nothing, so an array that is too short means SR silently ignores the entire
mean-field block — the worst kind of silent failure this mode can have. The
driver materialises the array before the gate runs, so by this point the length
must be exactly `2 * NPara`.
"""
function validate_parton_opt_flags(data::ExpertModeData)
    n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
    n_idx = parton_n_idx(data)
    expected = 2 * (n_proj + n_idx)

    if length(data.optimization_flags) != expected
        error(
            "optimization_flags has length $(length(data.optimization_flags)) " *
            "but the parton mode needs exactly $expected " *
            "(2 slots x (NProj = $n_proj + NPartonMFParaIdx = $n_idx)). " *
            "stochastic_opt! reads an out-of-range flag as 'frozen' without " *
            "warning, so a short array would make SR ignore the mean-field " *
            "parameters silently. The driver must materialise the flags before " *
            "this gate runs.",
        )
    end

    if data.modpara.parton_gauge_fix == 0
        @warn """PartonGaugeFix = 0: ゲージ射影を切っています。α のスケール方向と
                 一様オンサイトシフトは Ψ を変えないので S が厳密に特異になりえます。
                 MC ノイズが力ベクトルに与える偽の成分を正則化 ε 付きの S⁻¹ が
                 1/ε 倍するため、α が漂流するか SR が NaN で落ちることがあります
                 (DESIGN §2.5)。"""
    end

    mf_slots = (2 * n_proj + 1):expected
    any(i -> data.optimization_flags[i], mf_slots) || error(
        "Every mean-field slot in optimization_flags is frozen, so SR has " *
        "nothing to optimize. At least one real or imaginary component of the " *
        "mean-field parameters must have flag = 1. Gauge fixing is meant to " *
        "freeze one representative amplitude, not the whole block.",
    )

    return nothing
end

"""
    validate_parton_parallel(ctx::ParallelContext, modpara::ModParaParameters)

Validate parallel settings for the parton mode.

Non-mutating. `NSplitSize > 1` splits VMC samples inside each comm1 group
upstream; the parton sampling path has not been checked against that split yet,
so M1 restricts it. Plain multi-process runs (`NSplitSize = 1`) are unaffected.
"""
function validate_parton_parallel(ctx::ParallelContext, modpara::ModParaParameters)
    if modpara.nsplit_size != 1
        error(
            "Parton mode requires NSplitSize = 1, got NSplitSize = " *
            "$(modpara.nsplit_size). Sample splitting across comm1 groups is " *
            "not validated for the parton sampling path yet.",
        )
    end
    return nothing
end

"""
    validate_parton_qp(data::ExpertModeData)

Check that the momentum-projection count agrees between `modpara.def` and
`qptransidx.def`.

`NMPTrans` arrives through `modpara.def` while `qp_trans`, `qp_trans_sgn` and
`para_qp_trans` arrive through `qptransidx.def` -- two independent paths. The
amplitude side sizes itself from `get_n_qp_full` (= NSPGaussLeg x NMPTrans x
NQPOptTrans) whereas `gather_a_block!` indexes `data.qp_trans[qp]`, so writing
only one of the two makes projection terms disappear:

- too few mappings: `qp_trans[qp]` raises a BoundsError (still noticeable)
- too many mappings: the trailing translations are silently ignored
- a short `para_qp_trans`: `init_qp_weight!` leaves those entries at zero, so
  the terms vanish as **weight-zero contributions** without any error

`NMPTrans < 0` (C-mVMC's APFlag, i.e. anti-periodic boundaries) is rejected as
well: `init_qp_weight!` takes `abs` and builds N weights while `get_n_qp_full`
clamps with `max(1, .)` and collapses to a single QP, so the projection would
silently degrade to the identity. Site-dependent signs can be carried by
`qp_trans_sgn`, so the parton mode does not need that path.

This runs *before* `parton_ensure_qp!` (identity fallback and the 0-to-1-based
normalisation), so the values inspected here are exactly what the input files
declared. Non-mutating.
"""
function validate_parton_qp(data::ExpertModeData)
    mp = data.modpara
    n_mp = mp.nmp_trans

    if n_mp < 0
        error(
            "Parton mode does not support the anti-periodic quantum-projection " *
            "flag (NMPTrans < 0), got NMPTrans = $n_mp. InitQPWeight uses " *
            "abs(NMPTrans) while the QP count uses max(1, NMPTrans), so the " *
            "projection would silently collapse to a single identity term. " *
            "Use positive NMPTrans and carry the signs in the fourth column of " *
            "qptransidx.def (QPTransSgn) instead.",
        )
    end

    n_map = length(data.qp_trans)
    if n_map == 0
        n_mp <= 1 || error(
            "modpara.def declares NMPTrans = $n_mp but no QPTrans mappings were " *
            "parsed. Momentum projection needs qptransidx.def listed in " *
            "namelist.def (keyword TransSym); without it the run would fall " *
            "back to a single identity projection and silently drop the " *
            "projection entirely.",
        )
        return nothing
    end

    n_mp == n_map || error(
        "Quantum-projection count mismatch: modpara.def declares NMPTrans = " *
        "$n_mp but qptransidx.def provides $n_map translation mappings. These " *
        "come from two different files, and a mismatch silently drops or " *
        "duplicates projection terms instead of failing.",
    )
    length(data.qp_trans_sgn) == n_map || error(
        "qptransidx.def: $n_map translation mappings but " *
        "$(length(data.qp_trans_sgn)) sign arrays.",
    )
    length(data.para_qp_trans) == n_map || error(
        "qptransidx.def: $n_map translation mappings but " *
        "$(length(data.para_qp_trans)) ParaQPTrans weights. Missing weights are " *
        "left at zero by InitQPWeight, so those projection terms would vanish " *
        "without any error.",
    )
    for (qp, m) in enumerate(data.qp_trans)
        length(m) == mp.nsite || error(
            "qptransidx.def: mapping $qp has $(length(m)) entries, expected " *
            "Nsite = $(mp.nsite).",
        )
    end
    return nothing
end

"""
    validate_parton_inputs(data::ExpertModeData, ctx::ParallelContext)

Convenience aggregator: run every parton-mode validator. Non-mutating -- no
input is derived or rewritten here, so calling it twice is harmless.
"""
function validate_parton_inputs(data::ExpertModeData, ctx::ParallelContext)
    validate_parton_modpara(data.modpara)
    validate_parton_flavor_consistency(data)
    validate_parton_data(data)
    validate_parton_qp(data)
    validate_parton_opt_flags(data)
    validate_parton_parallel(ctx, data.modpara)
    return nothing
end
