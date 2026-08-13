"""
パートン平均場 VMC モードの型・アクセサ・委譲メソッド
--- parton-mode (fork addition) ---

DESIGN_parton.md §5(構造体カタログ)に対応する。標準経路(`PartonMode = 0`)
からは一切参照されない。

粒子数の関係(`PartonMode = 1`):

    NPartonPerFlavor = NElec        # NElec はフレーバーあたりのパートン数
    NParticle        = NElec        # 物理粒子数(固縛により一致)
    NPartonTot       = NFlavor * NElec
    NSiteFlavor      = NFlavor * NSite

添字文字の約束(DESIGN §5): サイトは `ri, rj` / フレーバーは `fi, fj` /
パックした配置空間の添字は `rfi = ri + fi * n_site`。軌道の行空間(サイトのみ)
と配置空間(サイト⊗フレーバー)は別の空間なので混ぜない。
"""

# =====================================================================
# 派生量アクセサ(冗長保持はしない。NElec だけが格納された量)
# =====================================================================

n_parton_per_flavor(n_elec::Int) = n_elec
n_parton_total(n_elec::Int, n_flavor::Int) = n_flavor * n_elec
n_phys_particle(n_elec::Int) = n_elec
n_site_flavor(n_site::Int, n_flavor::Int) = n_flavor * n_site

n_parton_per_flavor(modpara::ModParaParameters) = n_parton_per_flavor(modpara.nelec)
n_parton_total(modpara::ModParaParameters) = n_parton_total(modpara.nelec, modpara.nflavor)
n_phys_particle(modpara::ModParaParameters) = n_phys_particle(modpara.nelec)
n_site_flavor(modpara::ModParaParameters) = n_site_flavor(modpara.nsite, modpara.nflavor)

n_parton_per_flavor(data::ExpertModeData) = n_parton_per_flavor(data.modpara)
n_parton_total(data::ExpertModeData) = n_parton_total(data.modpara)
n_phys_particle(data::ExpertModeData) = n_phys_particle(data.modpara)
n_site_flavor(data::ExpertModeData) = n_site_flavor(data.modpara)

# =====================================================================
# 意味層: 平均場ハミルトニアン
# =====================================================================

"""
    PartonMFTemplateEntry

∂H/∂α_k の定数テンプレートの 1 成分。`H^(f)_ij(α) = α_k · t^(f)_ij` の
(i, j, f, t) を 1-based で保持する(0-based からの変換は
`parton_build_mf_templates!` の一箇所だけで行う)。
"""
struct PartonMFTemplateEntry
    site1::Int
    site2::Int
    flavor::Int
    coeff::ComplexF64
end

"""
    PartonMFHamiltonian

平均場ハミルトニアンと、そこから作られる軌道・軌道微分。

固定部(起動時に契約 0 の `parton_build_mf_templates!` が 1 回組む):
- `n_idx`: フレーバーを解決したあとの変分グループ数
- `template[k]`: グループ k に属する (i, j, f, t) の並び
- `is_onsite_group[k]`: h.c. なし直接加算・Im 凍結の対象か

α 依存部(SR ステップごとに契約 0/0′ が更新):
- `h_mf[f]`, `eig_vals[f]`, `eig_vecs[f]`(摂動論の分母に非占有まで要る)
- `orbitals[f]`: 占有ブロック Φ^(f)(n_site × n_elec)
- `dorbitals[f][dof]`: ∂Φ^(f)/∂θ。dof = 2k-1 が Re α_k、2k が Im α_k
- `dh_uo_scratch`: 契約 0′ の作業行列(n_site × n_elec)
- `min_gap`: HOMO-LUMO ギャップ(縮退検知)
"""
mutable struct PartonMFHamiltonian
    n_idx::Int
    template::Vector{Vector{PartonMFTemplateEntry}}
    is_onsite_group::Vector{Bool}

    h_mf::Vector{Matrix{ComplexF64}}
    eig_vals::Vector{Vector{Float64}}
    eig_vecs::Vector{Matrix{ComplexF64}}
    orbitals::Vector{Matrix{ComplexF64}}
    dorbitals::Vector{Vector{Matrix{ComplexF64}}}
    dh_uo_scratch::Matrix{ComplexF64}
    min_gap::Float64

    # ゲージ平坦方向(起動時に契約 0 の build が解決する。DESIGN §2.5)
    # - gauge_scale_groups[g]: 同時に実数正倍できる idx 集合。独立なスケール群の数は
    #   idx のフレーバー共有パターンで決まるので、フレーバー数を仮定しないこと
    # - gauge_shift_groups[g]: 一様オンサイトシフト(H → H + μI)を成す idx 集合
    # - gauge_target_norm[g]: 各スケール群の初期ノルム(射影の引き戻し先)
    gauge_scale_groups::Vector{Vector{Int}}
    gauge_shift_groups::Vector{Vector{Int}}
    gauge_target_norm::Vector{Float64}

    function PartonMFHamiltonian(n_site::Int, n_elec::Int, n_flavor::Int, n_idx::Int)
        n_dof = 2 * n_idx
        new(
            n_idx,
            [PartonMFTemplateEntry[] for _ = 1:n_idx],
            fill(false, n_idx),
            [zeros(ComplexF64, n_site, n_site) for _ = 1:n_flavor],
            [zeros(Float64, n_site) for _ = 1:n_flavor],
            [zeros(ComplexF64, n_site, n_site) for _ = 1:n_flavor],
            [zeros(ComplexF64, n_site, n_elec) for _ = 1:n_flavor],
            [[zeros(ComplexF64, n_site, n_elec) for _ = 1:n_dof] for _ = 1:n_flavor],
            zeros(ComplexF64, n_site, n_elec),
            Inf,
            Vector{Int}[],
            Vector{Int}[],
            Float64[],
        )
    end
end

# =====================================================================
# 意味層: 物理ハミルトニアン(局所エネルギー用)
# =====================================================================

"""
    PartonPhysHopEntry

physhop.def の 1 行を 1-based に直したもの。片方向のみ保持し、h.c. は
局所エネルギーが t 側・t* 側の両方向を評価することで供給する。
"""
struct PartonPhysHopEntry
    site1::Int
    site2::Int
    value::ComplexF64
end

"""
    PartonPhysDiagEntry

coulombinter.def の 1 行を 1-based に直したもの(V n_i n_j)。対角行
(site1 == site2)は硬芯により V n_i を意味し、化学ポテンシャルを表す。
"""
struct PartonPhysDiagEntry
    site1::Int
    site2::Int
    value::Float64
end

"""
    PartonPhysHamiltonian

局所エネルギーが読む物理ハミルトニアンのテンプレート。平均場側の
テンプレートと同じく起動時に 1 回だけ組み、0-based からの変換もそこで済ませる。
"""
struct PartonPhysHamiltonian
    hops::Vector{PartonPhysHopEntry}
    diags::Vector{PartonPhysDiagEntry}
end

# =====================================================================
# 速度層: 配置
# =====================================================================

"""
    PartonConfiguration

固縛パートン配置。既存の `ElectronConfiguration` は内部コンストラクタが
2 フレーバー(スピン)の寸法を焼き付けているので、一般 F 用に自前で持つ。

- `ele_idx[(f-1)*n_elec + m]`: フレーバー f の粒子 m が居るサイト(1-based)
- `ele_cfg[(f-1)*n_site + r]`: サイト r に居る粒子番号、空きは -1
- `ele_num[(f-1)*n_site + r]`: サイト r の占有数(0 か 1)
- `stored_ele_idx`: サンプル s の配置(`(s-1)*n_flavor*n_elec + ...`)
- `burn_flag`: burn-in 済みか。C 版のように counter[11] を間借りせず Bool で持つ
- `counter[1]` = 試行数、`counter[2]` = 受理数

固縛(全フレーバーが常に同一サイト集合を占有)は `assert_flavors_locked` で
錨と同じタイミングに検査する。
"""
mutable struct PartonConfiguration
    ele_idx::Vector{Int}
    ele_cfg::Vector{Int}
    ele_num::Vector{Int}

    burn_ele_idx::Vector{Int}
    burn_flag::Bool

    stored_ele_idx::Vector{Int}

    counter::Vector{Int}

    n_site::Int
    n_elec::Int
    n_flavor::Int

    function PartonConfiguration(n_site::Int, n_elec::Int, n_flavor::Int, n_sample::Int)
        n_tot = n_parton_total(n_elec, n_flavor)
        n_sf = n_site_flavor(n_site, n_flavor)
        new(
            zeros(Int, n_tot),
            fill(-1, n_sf),
            zeros(Int, n_sf),
            zeros(Int, n_tot),
            false,
            zeros(Int, n_sample * n_tot),
            zeros(Int, 10),
            n_site,
            n_elec,
            n_flavor,
        )
    end
end

"フレーバー f の粒子 m が居るサイト(1-based)。"
@inline particle_site(cfg::PartonConfiguration, f::Int, m::Int) =
    cfg.ele_idx[(f - 1) * cfg.n_elec + m]

"""
サイト r が占有されているか。固縛なので全フレーバーで同じ答になり、
フレーバー 1 を代表として読む。
"""
@inline is_occupied(cfg::PartonConfiguration, r::Int) = cfg.ele_num[r] != 0

"サイト r に居る粒子の番号(空きなら -1)。固縛によりフレーバー 1 が代表。"
@inline site_particle(cfg::PartonConfiguration, r::Int) = cfg.ele_cfg[r]

"""
    place_particle!(cfg, f, m, r)

初期配置用: フレーバー f の粒子 m をサイト r に置く(移動元なし)。
"""
function place_particle!(cfg::PartonConfiguration, f::Int, m::Int, r::Int)
    base = (f - 1) * cfg.n_site
    cfg.ele_idx[(f - 1) * cfg.n_elec + m] = r
    cfg.ele_cfg[base + r] = m
    cfg.ele_num[base + r] = 1
    return nothing
end

"""
    move_particle!(cfg, f, m, r_old, r_new)

フレーバー f の粒子 m を r_old から r_new へ動かす。ele_idx / ele_cfg /
ele_num の 3 点を同時に更新する唯一の経路で、固縛を守るために
`parton_update_ele_config!` から全フレーバー分まとめて呼ばれる。
"""
function move_particle!(cfg::PartonConfiguration, f::Int, m::Int, r_old::Int, r_new::Int)
    base = (f - 1) * cfg.n_site
    cfg.ele_idx[(f - 1) * cfg.n_elec + m] = r_new
    cfg.ele_cfg[base + r_old] = -1
    cfg.ele_num[base + r_old] = 0
    cfg.ele_cfg[base + r_new] = m
    cfg.ele_num[base + r_new] = 1
    return nothing
end

"""
    assert_flavors_locked(cfg)

固縛不変条件の検査: 全フレーバーが同一サイト集合を、同一の粒子番号で占有して
いること。破れていれば error を投げる。
"""
function assert_flavors_locked(cfg::PartonConfiguration)
    for f = 2:cfg.n_flavor, m = 1:cfg.n_elec
        r1 = particle_site(cfg, 1, m)
        rf = particle_site(cfg, f, m)
        rf == r1 || error(
            "flavor lock violated: particle $m is on site $r1 for flavor 1 but on " *
            "site $rf for flavor $f",
        )
    end
    for f = 1:cfg.n_flavor, r = 1:cfg.n_site
        n1 = cfg.ele_num[r]
        nf = cfg.ele_num[(f - 1) * cfg.n_site + r]
        nf == n1 || error(
            "flavor lock violated: site $r has occupation $n1 for flavor 1 but " *
            "$nf for flavor $f",
        )
    end
    return nothing
end

# =====================================================================
# 速度層: 振幅
# =====================================================================

"""
    PartonAmplitudeData

全 (qp, f) ブロックの A⁻¹ と det A。フラット配列+手動ストライドの家風に従い、
ストライド計算は `block_index` と `inv_block` の 2 関数だけに封じ込める。
`det_a` は生の複素値(乗法更新+定期的な厳密再計算が錨。DESIGN §7)。
"""
mutable struct PartonAmplitudeData
    inv_a::Vector{ComplexF64}
    det_a::Vector{ComplexF64}

    n_qp::Int
    n_flavor::Int
    n_elec::Int

    function PartonAmplitudeData(n_qp::Int, n_flavor::Int, n_elec::Int)
        n_block = n_qp * n_flavor
        new(
            zeros(ComplexF64, n_block * n_elec * n_elec),
            zeros(ComplexF64, n_block),
            n_qp,
            n_flavor,
            n_elec,
        )
    end
end

"(qp, f) → ブロック番号(1-based)。ストライドを書いてよい場所その 1。"
@inline block_index(amp::PartonAmplitudeData, qp::Int, f::Int) =
    (qp - 1) * amp.n_flavor + f

"(qp, f) の A⁻¹ を Ne×Ne 行列ビューとして返す。ストライドを書いてよい場所その 2。"
@inline function inv_block(amp::PartonAmplitudeData, qp::Int, f::Int)
    ne = amp.n_elec
    b = block_index(amp, qp, f)
    off = (b - 1) * ne * ne
    return reshape(view(amp.inv_a, (off + 1):(off + ne * ne)), ne, ne)
end

"""
    PartonSamplingWorkspace

ホットループでの確保をゼロにするための作業領域。`ratio_blocks` は契約 2 が
書き、直後の契約 3 が読む((qp, f) ブロックごとの R)。
"""
mutable struct PartonSamplingWorkspace
    a_scratch::Matrix{ComplexF64}
    ratio_blocks::Vector{ComplexF64}
    u_buf::Vector{ComplexF64}
    v_buf::Vector{ComplexF64}
    col_buf::Vector{ComplexF64}

    function PartonSamplingWorkspace(n_elec::Int, n_block::Int)
        new(
            zeros(ComplexF64, n_elec, n_elec),
            zeros(ComplexF64, n_block),
            zeros(ComplexF64, n_elec),
            zeros(ComplexF64, n_elec),
            zeros(ComplexF64, n_elec),
        )
    end
end

# =====================================================================
# 外箱と委譲
# =====================================================================

"""
    PartonOptimizationState

既存機構への窓口(`state`)とパートン固有の 4 部品をまとめた外箱。
`state` は EnergyData / SROptData だけを実使用し、スレーター行列や
電子配置のフィールドは触らない(パートン側が自前で持つ)。
"""
mutable struct PartonOptimizationState
    state::VMCOptimizationState
    amp::PartonAmplitudeData
    config::PartonConfiguration
    workspace::PartonSamplingWorkspace
    mfham::PartonMFHamiltonian
    physham::PartonPhysHamiltonian
end

# 既存関数への委譲。新しい振る舞いは足さず、`state` へ橋渡しするだけ。
weight_average_we!(ctx::ParallelContext, st::PartonOptimizationState,
                   t::CTimer = CTIMER_DISABLED) = weight_average_we!(ctx, st.state, t)
weight_average_sr_opt!(ctx::ParallelContext, st::PartonOptimizationState,
                       t::CTimer = CTIMER_DISABLED) =
    weight_average_sr_opt!(ctx, st.state, t)
stochastic_opt!(d::ExpertModeData, st::PartonOptimizationState,
                t::CTimer = CTIMER_DISABLED) = stochastic_opt!(d, st.state, t)
output_data!(d::ExpertModeData, st::PartonOptimizationState, step::Int; kw...) =
    output_data!(d, st.state, step; kw...)
store_opt_data!(d::ExpertModeData, st::PartonOptimizationState, sample_idx::Int) =
    store_opt_data!(d, st.state, sample_idx)
reduce_counter!(ctx::ParallelContext, st::PartonOptimizationState) =
    reduce_counter!(ctx, st.config.counter)
