"""
MVMCOptimizers.jl

A Julia package for VMC parameter optimization using Stochastic Reconfiguration method.
"""

module MVMCOptimizers


using LinearAlgebra
using LinearAlgebra.LAPACK: potrf!, potrs!

using Printf
using Random

using SFMT

using LoopVectorization: @turbo

# Import ExpertModeData and update_qp_weight! from MVMCExpertModeParsers
using MVMCExpertModeParsers: MVMCExpertModeParsers
using MVMCExpertModeParsers: ExpertModeData
using MVMCExpertModeParsers: ModParaParameters
using MVMCExpertModeParsers: init_qp_weight!
using MVMCExpertModeParsers: initialize_parameters!, init_parameter!, read_input_parameters!
# sync_modified_parameter! is defined in parameter_sync.jl

# PfaPack provides generic LTL / utu2 / Pfaffian routines.
# calculate_m_all_* and PfaPackWorkspace are defined locally (GPL-bound helpers)
# in workspace.jl and calculate_m_all.jl, included below.
using PfaPack: julia_zsktf2!, utu2pfa, cimpl_utu2inv!

# C-compatible lightweight timer (CTimer). No dependencies beyond Base/Printf;
# included first so later files (vmc_para_opt.jl, etc.) can reference it.
include("c_timer.jl")

# MPI infrastructure for v0.4 (ParallelContext, split_loop, communication
# wrappers, seed resolution). Serial runs never touch MPI APIs.
include("parallel.jl")

# Thread-local Pfaffian workspaces (GPL helpers, formerly in MVMCPfaPack).
# Must be included before types.jl because some types reference
# ThreadedPfaPackWorkspace.
include("workspace.jl")

# Include type definitions
include("types.jl")

# Threading helpers for C OpenMP-equivalent inner loops and local accumulation.
include("threading.jl")

# calculate_m_all helpers (GPL helpers, formerly in MVMCPfaPack).
include("calculate_m_all.jl")

# Aliases preserve the historical _pfapack! suffix used by vmc_sampling.jl
# wrappers (same generic name `calculate_m_all_fcmp!` is also overloaded there
# with a (data, state) signature).
const calculate_m_all_fcmp_pfapack! = calculate_m_all_fcmp!
const calculate_m_all_real_pfapack! = calculate_m_all_real!

# Include optimization functions
include("qp_weight_update.jl")
include("slater_update.jl")
include("vmc_sampling.jl")
include("vmc_main_cal.jl")
include("weight_average.jl")
include("counter.jl")
include("stochastic_opt.jl")
include("parameter_sync.jl")
include("data_io.jl")
include("initial_params.jl")
# Runtime compatibility contract: reject unsupported global and entry-point
# ModPara combinations. Must precede the entry points that call it below.
include("unsupported_inputs.jl")
include("vmc_para_opt.jl")
include("vmc_phys_cal.jl")
include("run_para_opt_from_namelist.jl")
include("run_phys_cal_from_namelist.jl")

# --- parton-mode (fork addition) ---
# パートンモードの新規実装。標準経路(PartonMode = 0)からは一切呼ばれない。
# include はこの順(types が先頭必須)。
include("parton_types.jl")
include("parton_orbparam_types.jl")   # Φ 直接変分(DESIGN §12.11)
include("parton_pair_types.jl")       # Pfaffian モード(PartonMode = 2、DESIGN_parton_pf.md)
include("parton_unsupported_inputs.jl")
include("parton_orbparam_inputs.jl")  # Φ 直接変分の入力・再スケール・門番
include("parton_pair_inputs.jl")      # Pfaffian モードの入力・初期値・D_AmpMax・門番
include("parton_orbital.jl")
include("parton_initial_params.jl")
include("parton_calculate_m_all.jl")
include("parton_vmc_sampling.jl")
include("parton_pair_amplitude.jl")   # Pfaffian モードの契約 1〜3
include("parton_vmc_main_cal.jl")
include("parton_orbparam_o.jl")       # Φ 直接変分の O(DESIGN §12.11)
include("parton_pair_o.jl")           # Pfaffian モードの O と ⟨O O†⟩
include("parton_output.jl")
include("parton_vmc_para_opt.jl")
include("parton_vmc_para_opt_orbparam.jl")   # Φ 直接変分の SR ループとドライバ
include("parton_pair_output.jl")             # Pfaffian モードの出力(診断・runinfo・SVD)
include("parton_vmc_para_opt_pair.jl")       # Pfaffian モードの状態・SR ループ・ドライバ
include("parton_phys_cal.jl")                 # 物理量測定(NVMCCalMode = 1、DESIGN §13)
include("parton_run_para_opt_from_namelist.jl")

# Export main functions
export ParallelContext, serial_context, build_parallel_context, is_output_rank
export bcast!, bcast_scalar, allreduce_sum!, reduce_sum_to_root!, barrier,
       reduce_counter!, abort_parallel, split_loop, split_range, qp_split_range,
       allreduce_sum_scalar, allreduce_max_scalar, resolve_rnd_seed
export count_total_parameters, pack_parameters, unpack_parameters!,
       set_parameter_value!, get_parameter_value
export vmc_para_opt!
export vmc_phys_cal!
export run_para_opt_from_namelist
export run_phys_cal_from_namelist
export read_initial_def!
export read_opt_para_file!

# --- parton-mode (fork addition) ---
export parton_run_para_opt_from_namelist, parton_vmc_para_opt!, validate_parton_inputs

end # module MVMCOptimizers
