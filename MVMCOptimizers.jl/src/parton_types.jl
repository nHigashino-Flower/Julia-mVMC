""" 
    Accessor for Paton-based VMC ('PartonVMCCalMode = 1')

   Defining Number of Physical particle (NParticle), Parton per flavor (NPartonPerFlavor) and Total Parton (NPartonTot) 
   Nelec is regarded as number of parton per flavor. In addition, we assume the 'NParticle == NPartonPerFlavor' system.
   so, when 'PartonVMCCalMode = 1', we assume that,

   NPartonPerFlavor = NElec
   NParticle = NElec
   NPartonTot = NFlavor*NElec
   NSite2 = NFlavor*NSite
""" 

# call from n_flavor, n_elec and n_site
n_parton_per_flavor(n_elec::Int) = n_elec
n_parton_total(n_elec::Int, n_flavor::Int) = n_flavor * n_elec
n_phys_particle(n_elec::Int) = n_elec
n_site_flavor(n_site::Int, n_flavor::Int) = n_flavor*n_site

# call from ModParaParameters (Multi Dispacth)
n_parton_per_flavor(modpara::ModParaParameters) = n_parton_per_flavor(modpara.nelec)
n_parton_total(modpara::ModParaParameters) = n_parton_total(modpara.nelec, modpara.nflavor)
n_phys_particle(modpara::ModParaParameters) = n_phys_particle(modpara.nelec)
n_site_flavor(modpara::ModParaParameters) = n_site_flavor(modpara.nsite, modpara.nflavor) 

# call from ExpertModeData (Multi Dispacth)
n_parton_per_flavor(data::ExpertModeData) = n_parton_per_flavor(data.modpara)
n_parton_total(data::ExpertModeData)      = n_parton_total(data.modpara)
n_phys_particle(data::ExpertModeData)     = n_phys_particle(data.modpara)
n_site_flavor(data.ExpertModeData) = n_site_flavor(data.modpara)

"""

Data types for Parton-based VMC optimization based on types.jl

"""

mutable struct PartonOptimizationState
    state    :: VMCOptimizationState   # 再利用機構への窓口
    parton_amp_data :: PartonAmplitudeData       # 振幅エンジンの状態
    parton_config ::  PartonConfiguration
    parton_workspace :: PartonSamplingWorkspace
    parton_mfhamiltonian :: PartonMFHamiltonian
end

stochastic_opt!(d, st::PartonOptimizationState, t=CTIMER_DISABLED) = stochastic_opt!(d, st.vmc, t)
weight_average_we!(st::PartonOptimizationState)                    = weight_average_we!(st.vmc)
weight_average_sr_opt!(st::PartonOptimizationState)                = weight_average_sr_opt!(st.vmc)
output_data!(d, st::PartonOptimizationState, step; kw...)          = output_data!(d, st.vmc, step; kw...)
reduce_counter!(ctx, st::PartonOptimizationState)                  = reduce_counter!(ctx, st.vmc)

"""
    PartonMFHamiltonian

Parton mean-field related data for VMC calclation
"""

mutable struct PartonMFHamiltonian
    # ---- 固定部(起動時に1回組む)----
    n_idx           :: Int                          # フレーバー解決後の変分グループ数
    template        :: Vector{Vector{NTuple{4,Any}}} # k → [(ri, rj, f, t)] ※型は実装時に具体化
    is_onsite_group :: Vector{Bool}                 # k → Im凍結・h.c.なし加算の対象か
    # ---- α依存部(SRステップ毎に契約0が更新)----
    h_mf     :: Vector{Matrix{ComplexF64}}  # f → H^(f)(α)     (n_site × n_site)
    eig_vals :: Vector{Vector{Float64}}     # f → ε_n(全固有値。摂動論の分母用)
    eig_vecs :: Vector{Matrix{ComplexF64}}  # f → 全固有ベクトル(非占有も保持)
    orbitals :: Vector{Matrix{ComplexF64}}  # f → Φ^(f) = 占有ブロック (n_site × n_elec)
    min_gap  :: Float64                     # 縮退検知(DESIGN §8)
end

"""
    ElectronConfiguration

Electron configuration for VMC sampling.
"""
mutable struct PartonConfiguration
    ele_idx::Vector{Int}      # Electron indices [sample][mi+si*Ne]
    ele_cfg::Vector{Int}      # Electron configuration [sample][ri+si*Nsite]
    ele_num::Vector{Int}      # Electron number [sample][ri+si*Nsite]
    ele_proj_cnt::Vector{Int} # Projection count [sample][proj]

    # Temporary arrays for sampling (single sample)
    tmp_ele_idx::Vector{Int}      # Temporary electron indices [mi+si*Ne]
    tmp_ele_cfg::Vector{Int}      # Temporary electron configuration [ri+si*Nsite]
    tmp_ele_num::Vector{Int}      # Temporary electron number [ri+si*Nsite]
    tmp_ele_proj_cnt::Vector{Int} # Temporary projection count [proj]

    # Burn-in sample storage
    burn_ele_idx::Vector{Int}
    burn_ele_cfg::Vector{Int}
    burn_ele_num::Vector{Int}
    burn_ele_proj_cnt::Vector{Int}

    # Counters for statistics
    counter::Vector{Int}  # Various counters (hopping attempts, accepts, etc.)

    function PartonConfiguration(
        n_sample::Int,
        n_site::Int,
        n_elec::Int,
        n_proj::Int,
        n_flavor::Int
    )
        n_parton_tot = n_parton_total(n_elec, n_flavor)
        n_sitef = n_site_flavor(n_site, n_flavor)

       
        new(
            zeros(Int, n_sample * n_parton_tot),
            zeros(Int, n_sample * n_sitef),
            zeros(Int, n_sample * n_sitef),
            zeros(Int, n_sample * n_proj),
            # Temporary arrays
            zeros(Int, n_parton_tot),
            zeros(Int, n_sitef),
            zeros(Int, n_sitef),
            zeros(Int, n_proj),
            # Burn-in arrays
            zeros(Int, n_parton_tot + n_sitef + n_sitef + n_proj),  # Combined storage
            zeros(Int, n_sitef),
            zeros(Int, n_sitef),
            zeros(Int, n_proj),
            # Counters
            zeros(Int, 10),
        )
    end
end

"""
    SlaterMatrixData

Slater matrix and inverse matrix data.
"""
mutable struct PartonAmplitudeData
    slater_elm::Vector{ComplexF64}      # Slater matrix elements [QPidx][ri+si*Nsite][rj+sj*Nsite]
    inv_m::Vector{ComplexF64}           # Inverse matrix [QPidx][mi+si*Ne][mj+sj*Ne]
    pf_m::Vector{ComplexF64}            # Pfaffian [QPidx]

    # Real versions (for real TBC)
    slater_elm_real::Vector{Float64}
    inv_m_real::Vector{Float64}
    pf_m_real::Vector{Float64}

    function SlaterMatrixData(n_qp_full::Int, n_site::Int, n_elec::Int, all_complex::Bool)
        # Validate inputs
        n_qp_full = max(1, n_qp_full)
        n_site = max(1, n_site)
        n_elec = max(1, n_elec)

        n_sitef = 2 * n_site
        n_parton_tot = 2 * n_elec

        if all_complex
            new(
                zeros(ComplexF64, n_qp_full * n_sitef * n_sitef),
                zeros(ComplexF64, n_qp_full * (n_parton_tot * n_parton_tot + 1)),
                zeros(ComplexF64, n_qp_full),
                Float64[],  # Real versions not used
                Float64[],
                Float64[],
            )
        else
            new(
                zeros(ComplexF64, n_qp_full * n_sitef * n_sitef),
                zeros(ComplexF64, n_qp_full * (n_parton_tot * n_parton_tot + 1)),
                zeros(ComplexF64, n_qp_full),
                zeros(Float64, n_qp_full * n_sitef * n_sitef),
                zeros(Float64, n_qp_full * (n_parton_tot * n_parton_tot + 1)),
                zeros(Float64, n_qp_full),
            )
        end
    end
end

"""
    SamplingWorkspace

Pre-allocated workspace arrays for VMC sampling to avoid repeated allocations.
This significantly reduces memory allocation overhead in hot loops.
"""
mutable struct PartonSamplingWorkspace
    # For calculate_m_all_real!
    inv_m_real_temp::Array{Float64,3}
    pf_m_real_temp::Vector{Float64}

    # For calculate_m_all! (complex version)
    inv_m_temp::Array{ComplexF64,3}
    pf_m_temp::Vector{ComplexF64}

    # For vmc_make_sample_real! / vmc_make_sample!
    proj_cnt_new::Vector{Int}
    pf_m_new_real::Vector{Float64}
    pf_m_new::Vector{ComplexF64}

    # Cached arrays (computed once)
    loc_spn::Vector{Int}

    # Workspace for PfaPack (Pfaffian calculations)
    # Use ThreadedPfaPackWorkspace for parallel execution (one workspace per thread)
    pfapack_workspace::ThreadedPfaPackWorkspace

    # Cached VMCMainCal local accumulator. Kept as Any because VMCThreadAccumulator
    # is defined later in threading.jl.
    main_cal_accumulator::Any

    function SamplingWorkspace(n_parton_tot::Int, n_qp_full::Int, n_proj::Int, n_site::Int)
        new(
            zeros(Float64, n_parton_tot, n_parton_tot, n_qp_full),
            zeros(Float64, n_qp_full),
            zeros(ComplexF64, n_parton_tot, n_parton_tot, n_qp_full),
            zeros(ComplexF64, n_qp_full),
            zeros(Int, n_proj),
            zeros(Float64, n_qp_full),
            zeros(ComplexF64, n_qp_full),
            zeros(Int, n_site),  # loc_spn will be initialized later
            ThreadedPfaPackWorkspace(n_parton_tot),  # Thread-local workspaces for parallel Pfaffian calculations
            nothing,
        )
    end
end

"""
    VMCOptimizationState

State data for VMC optimization.
"""
mutable struct VMCOptimizationState
    energy::EnergyData
    slater_matrix::SlaterMatrixData
    electron_config::ElectronConfiguration
    sr_opt::SROptData
    opt_data::Vector{OptDataPoint}
    workspace::SamplingWorkspace
    phys_quantities::Union{PhysicalQuantities,Nothing}  # For VMCPhysCal mode

    function VMCOptimizationState(
        n_site::Int,
        n_elec::Int,
        n_proj::Int,
        n_para::Int,
        n_qp_full::Int,
        n_vmc_sample::Int,
        all_complex::Bool,
        use_fsz::Bool,
    )
        n_parton_tot = 2 * n_elec
        new(
            EnergyData(),
            SlaterMatrixData(n_qp_full, n_site, n_elec, all_complex),
            ElectronConfiguration(n_vmc_sample, n_site, n_elec, n_proj, use_fsz),
            SROptData(1 + n_para, n_vmc_sample, all_complex),
            OptDataPoint[],
            SamplingWorkspace(n_parton_tot, n_qp_full, n_proj, n_site),
            nothing,  # Initialize as nothing, will be set when needed
        )
    end
end
