"""
Input contract for the parton mean-field VMC mode (`PartonMode = 1`).

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

Particle numbers are not stored redundantly. `NElec` (partons per flavor) is
the only stored field; `NParticle` and `NPartonPerFlavor` are parser-level
aliases that write into it, and the derived names (`n_phys_particle`,
`n_parton_total`, `n_parton_per_flavor`) are accessor functions in
`parton_types.jl`. There is therefore nothing to resolve here — only to check.

Call order from `parton_run_para_opt_from_namelist` / `parton_vmc_para_opt!`:

    validate_parton_modpara(data.modpara)            # non-mutating
    validate_parton_data(data)                       # non-mutating
    validate_parton_parallel(ctx, data.modpara)      # non-mutating

Milestone scope (M1): NFlavor = 2 via the flavor<->spin mapping, complex
sz-conserved path, direct SR solver, parameter optimization only.
"""

# --- parton-mode (fork addition) ---

const PARTON_MODE_OFF = 0
const PARTON_MODE_MEAN_FIELD = 1

"""
    is_parton_mode(modpara::ModParaParameters) -> Bool

True when the input selects a parton mode. `PartonMode` is the *only* switch:
`NFlavor` carries data, never mode selection, so that a missing or misspelled
`NFlavor` line fails loudly instead of silently reverting to standard mVMC.
"""
is_parton_mode(modpara::ModParaParameters) = modpara.parton_mode != PARTON_MODE_OFF

"""
    validate_parton_modpara(modpara::ModParaParameters)

Reject ModPara inputs the parton mean-field mode does not support, and assert
the settings it silently depends on.

Non-mutating. `NElec` is the single stored particle-number field; everything
else (`n_phys_particle`, `n_parton_total`) is derived by accessor.

Rejected or required:

- `PartonMode`: must be `1`. `0` means this entry point was reached by mistake;
  `>= 2` is reserved for future parton ansatze.
- `NFlavor`: must be `> 2`. `0` usually
  means the `NFlavor` line is missing or misspelled — unknown modpara keys are
  only warnings.
- `2Sz = 0`: the default is `-1`, which the reused machinery reads as
  Sz-non-conserving (FSZ). The parton mode fixes the per-flavor particle
  numbers, so this must be pinned explicitly.
- `NCond = -1`, `NLocSpin = 0`: the upstream relation
  `NElec = (NLocSpin + NCond) / 2` is meaningless once flavors replace spin.
- `ComplexType = 1`, `NSRCG = 0`, `NLanczosMode = 0`, `NVMCCalMode = 0`:
  the single variant implemented in M1.
- `NExUpdatePath = 6` which corresponds to gauge constraints where each distinct flavor partons 
    must hop with the same coordinate.
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
    if modpara.partonv_vmc_calc_mode != PARTON_MODE_MEAN_FIELD
        error(
            "PartonMode = $(modpara.parton_mode) is not implemented: only " *
            "PartonMode = 1 (parton mean-field VMC) exists. Values >= 2 are " *
            "reserved for future parton ansatze.",
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
    if nex_update_path != 6
        error(
            "NExUpdatePath = $(NExUpdatePath) does not protect parton gauge constraints."*
            "PartonVMC must obey gauge constraint where each distinct flavor partons 
            must hop with the same coordinate, corresponding with NExUpdatePath = 6."
        )


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
    validate_parton_data(data::ExpertModeData)

Validate parton-mode settings that need parsed data, not just ModPara.

Non-mutating. In particular this asserts that the mean-field parameter file was
actually read: `parse_expert_mode_files` continues past per-file parse failures
(matching the C implementation), so a malformed or missing pmfpara.def would
otherwise surface much later as an empty Hamiltonian.

Gutzwiller, Jastrow and momentum projection (qptransidx.def) are supported and
deliberately not restricted here — they multiply the determinant product and
reuse the existing projection machinery unchanged.
"""
function validate_parton_data(data::ExpertModeData)
    if isempty(data.pmfpara_terms)
        error(
            "No mean-field parameter terms were parsed: pmfpara.def is missing, empty, " *
            "or failed to parse. Parse failures of individual .def files are " *
            "not fatal (C-compatible behaviour), so check the parser warnings " *
            "and the PartonMFPara entry in namelist.def.",
        )
    end
    if isempty(data.pmftrans_terms)
        error(
            "No mean-field transfer and interaction terms were parsed: pmftrans.def is missing, empty, " *
            "or failed to parse. Parse failures of individual .def files are " *
            "not fatal (C-compatible behaviour), so check the parser warnings " *
            "and the PartonMFTrans entry in namelist.def.",
        )
    end
    if !isempty(data.orbital_terms)
        error(
            "Orbital terms are present ($(length(data.orbital_terms)) terms) " *
            "but the parton ansatz replaces the pair orbital entirely. " *
            "Remove Orbital/OrbitalAntiParallel/OrbitalParallel/OrbitalGeneral " *
            "from namelist.def.",
        )
    end
    if !isempty(data.doublon_holon_2site_indices) ||
       !isempty(data.doublon_holon_4site_indices)
        error(
            "Doublon-holon projection is not supported in parton mode yet. " *
            "Remove the DH2/DH4 entries from namelist.def.",
        )
    end
    if !isempty(data.opt_trans) || data.n_qp_opt_trans > 1
        error(
            "OptTrans (variational quantum-number projection) is not " *
            "supported in parton mode: it would require updating the QP " *
            "weights every SR step against the mean-field orbitals.",
        )
    end

    n_site = data.modpara.nsite
    for (k, t) in enumerate(data.pmfpara_terms)
        for (label, site) in (("site1", t.site1), ("site2", t.site2))
            if site < 0 || site >= n_site
                error(
                    "mfparam.def term $k: $label = $site is out of range " *
                    "[0, $(n_site - 1)].",
                )
            end
        end
        for (label, flavor) in (("flavor1", t.flavor1), ("flavor2", t.flavor2))
            if flavor < 0 || flavor >= data.modpara.nflavor
                error(
                    "mfparam.def term $k: $label = $flavor is out of range " *
                    "[0, $(data.modpara.nflavor - 1)] for NFlavor = " *
                    "$(data.modpara.nflavor).",
                )
            end
        end
    end

    for (k, t) in enumerate(data.pmftrans_terms)
        for (label, site) in (("site1", t.site1), ("site2", t.site2))
            if site < 0 || site >= n_site
                error(
                    "mfparam.def term $k: $label = $site is out of range " *
                    "[0, $(n_site - 1)].",
                )
            end
        end
    end

    return nothing
end

function validate_parton_flavor_consistency(data::ExpertModeData)
    for (name, terms) in (("pmftrans", data.pmftrans_terms),
        ("pmfpara",  data.pmfpara_terms))
        for (k, t) in enumerate(terms)
            t.flavor1 == t.flavor2 || error(
                                            "$name.def term $k: flavor-mixing (flavor1=$(t.flavor1), " *
                                            "flavor2=$(t.flavor2)) is not supported: the parton ansatz " *
                                            "is a product of per-flavor determinants. Reserved for a " *
                                            "future extension."
                                            )
            # site/flavor の範囲チェックも両テーブル共通化してここで
        end
    end
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
    validate_parton_inputs(data::ExpertModeData, ctx::ParallelContext)

Convenience aggregator: run every parton-mode validator. Non-mutating -- no
input is derived or rewritten here, so calling it twice is harmless.
"""
function validate_parton_inputs(data::ExpertModeData, ctx::ParallelContext)
    validate_parton_modpara(data.modpara)
    validate_parton_flavor_consistency(data)
    validate_parton_data(data)
    validate_parton_parallel(ctx, data.modpara)
    return nothing
end