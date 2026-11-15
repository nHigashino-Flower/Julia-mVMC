"""
パートンモード(α・Φ・Pf 共通)の SR ソルバの入口 --- parton-mode (fork addition) ---

DESIGN_parton.md §14。3 つの SR ループ(`parton_vmc_para_opt*.jl`)はここの 2 関数だけを呼ぶ。

- `parton_weight_average_sr_opt!`: ⟨O O†⟩・⟨H O⟩ のランク間の合算と規格化
- `parton_stochastic_opt!`: 連立方程式 S x = g を解いてパラメータを更新

modpara の `NSRCG` で解法を選ぶ。

| NSRCG | 解法 | ⟨O O†⟩ |
|---|---|---|
| 0(既定) | 直接法(上流 `stochastic_opt!`、Cholesky) | 全体を組む(パラメータ数の 2 乗) |
| 1 | SR-CG(S を作らない前処理つき共役勾配) | 先頭行と対角だけ |

S の定義は両者で同じ: S_ij = (1/W) Σ_s [Re O_i Re O_j + Im O_i Im O_j] − ⟨Re O_i⟩⟨Re O_j⟩
(+ 対角に DSROptStaDel)。SR-CG はサンプルごとの O(共役シムを掛けた後の store)から
S·x を直接作るので、厳密演算では直接法と同じ更新になる(違いは CG の打ち切り DSROptCGTol だけ)。

MPI での直接法は、解くのをランク 0 だけにした(以前は全ランクが同じ方程式を同時に解き、
ランク 0 以外の結果は直後の bcast で捨てていた)。ランク 0 の計算列は変わらないので結果は同じ。
解いている間だけ BLAS のスレッド数を環境変数 `MVMC_SR_BLAS_THREADS` の値に上げられる
(未設定なら変えない。サンプリング中は 1 のままにしたいので、解く間だけ切り替える)。
"""

"SR-CG を使うか(modpara の NSRCG ≠ 0)。"
parton_sr_use_cg(data::ExpertModeData) = data.modpara.nsrcg != 0

"""
    parton_weight_average_sr_opt!(ctx, state, data)

⟨O O†⟩・⟨H O⟩ をランク間で合算して global Wc で割る。

- serial: 上流の `weight_average_sr_opt!` そのもの
- MPI・直接法: 合算はランク 0 へ集めるだけ(S を組むのはランク 0 だけなので、全ランクへ
  配り直さない)。ランク 0 以外の `sr_opt_oo`・`sr_opt_ho` はそのステップでは使わない
- MPI・SR-CG: 各ランクが CG を同じ手順で進めるので、⟨O⟩ の行と対角(先頭 2 行ぶん)と
  ⟨H O⟩ だけを全ランクへ合算する。行列の残りは触らない
"""
function parton_weight_average_sr_opt!(
    ctx::ParallelContext,
    state::VMCOptimizationState,
    data::ExpertModeData,
)
    use_cg = parton_sr_use_cg(data)
    sr = state.sr_opt
    if abs(state.energy.wc) < 1e-15
        is_output_rank(ctx) && @warn "Weight Wc is too small: $(state.energy.wc)"
        return nothing
    end
    inv_w = 1.0 / state.energy.wc
    if use_cg
        head = view(sr.sr_opt_oo, 1:(4 * sr.sr_opt_size))
        allreduce_sum!(ctx, head; which = :comm0)
        allreduce_sum!(ctx, sr.sr_opt_ho; which = :comm0)
        head .*= inv_w
        sr.sr_opt_ho .*= inv_w
    elseif ctx.is_mpi
        reduce_sum_to_root!(ctx, sr.sr_opt_oo; root = 0, which = :comm0)
        reduce_sum_to_root!(ctx, sr.sr_opt_ho; root = 0, which = :comm0)
        if is_output_rank(ctx)
            sr.sr_opt_oo .*= inv_w
            sr.sr_opt_ho .*= inv_w
        end
    else
        weight_average_sr_opt!(state)
    end
    return nothing
end

"解く間だけ使う BLAS のスレッド数(環境変数 MVMC_SR_BLAS_THREADS。未設定・不正なら 0 = 変えない)。"
function parton_sr_blas_threads()
    n = tryparse(Int, get(ENV, "MVMC_SR_BLAS_THREADS", ""))
    return (n === nothing || n < 1) ? 0 : n
end

"""
    parton_stochastic_opt!(data, state, c_timer, ctx; write_srinfo, srinfo_dir, srinfo_iter) -> info

S x = g を解いてパラメータを更新する(ランク 0 の `data` だけが正しく更新される。
呼び出し側が直後に `parton_sync_parameters!` で配る)。全ランクが呼ぶこと。
"""
function parton_stochastic_opt!(
    data::ExpertModeData,
    state::VMCOptimizationState,
    c_timer::CTimer,
    ctx::ParallelContext;
    write_srinfo::Bool = false,
    srinfo_dir::Union{String,Nothing} = nothing,
    srinfo_iter::Int = 0,
)::Int
    if parton_sr_use_cg(data)
        return parton_stochastic_opt_cg!(data, state, c_timer, ctx;
                                         srinfo_dir = write_srinfo ? srinfo_dir : nothing)
    end
    # 直接法。MPI ではランク 0 だけが解く
    (ctx.is_mpi && !is_output_rank(ctx)) && return 0
    n_threads = parton_sr_blas_threads()
    n_before = LinearAlgebra.BLAS.get_num_threads()
    n_threads > 0 && LinearAlgebra.BLAS.set_num_threads(n_threads)
    try
        return stochastic_opt!(data, state, c_timer;
                               write_srinfo = write_srinfo,
                               srinfo_dir = srinfo_dir, srinfo_iter = srinfo_iter)
    finally
        n_threads > 0 && LinearAlgebra.BLAS.set_num_threads(n_before)
    end
end

"""
    parton_stochastic_opt_cg!(data, state, c_timer, ctx; srinfo_dir) -> info

SR-CG。解く方程式・可動パラメータの選び方・収束の基準(‖r‖² < DSROptCGTol²·n²)・反復の上限・
SRinfo の書式は上流 `stochastic_opt_cg!` と同じ。反復の中身だけ次の 2 点が違う。

- S·d は、サンプルごとの O の列 a_c(実部の列と虚部の列)を 1 本ずつ読んで
  z += a_c (a_c·d) と足す。上流は行列全体を 1 反復に 4 回なめるので、パラメータが多いと
  メモリの転送で頭打ちになる(LiWuC 3×4・7830 次元で 18 ms/反復)
- 対角 (1 + DSROptStaDel)·S_ii で前処理する。S_ii は RedCut の幅(既定で 8 桁)まで散らばるので、
  前処理なしでは反復が多い

各ランクは自分のサンプルの列だけを持ち、S·d をランク間で合算する。全ランクが同じ手順で進む
(探索方向は反復ごとにランク 0 から配る)。捨てたサンプルの列は `parton_main_cal!` が 0 で
埋めているので和に寄与しない(母数は global Wc)。
SRinfo の最後の列は直接法では step、SR-CG では CG の反復回数。
"""
function parton_stochastic_opt_cg!(
    data::ExpertModeData,
    state::VMCOptimizationState,
    c_timer::CTimer,
    ctx::ParallelContext;
    srinfo_dir::Union{String,Nothing} = nothing,
)::Int
    mp = data.modpara
    n_para_full = 2 * _parameter_count_breakdown(data).n_para
    sr = state.sr_opt
    size_2 = 2 * sr.sr_opt_size
    n_sample = mp.nvmc_sample

    # --- 可動パラメータの選別(直接法と同じ規則。⟨O⟩ は先頭行、⟨|O|²⟩ は 2 行目)
    ctimer_start!(c_timer, 50)
    o_mean_all = view(sr.sr_opt_oo, 1:size_2)
    o_abs2_all = view(sr.sr_opt_oo, (size_2 + 1):(2 * size_2))
    s_diag_all = [real(o_abs2_all[pi + 3]) - real(o_mean_all[pi + 3])^2 for pi = 0:(n_para_full - 1)]
    s_diag_max = maximum(s_diag_all)
    s_diag_min = minimum(s_diag_all)
    diag_cut_threshold = s_diag_max * mp.dsr_opt_red_cut
    smat_to_para_idx = Int[]
    opt_num = 0
    cut_num = 0
    for pi = 0:(n_para_full - 1)
        if get_opt_flag_for_parameter(data, pi) != 1
            opt_num += 1
        elseif s_diag_all[pi + 1] < diag_cut_threshold
            cut_num += 1
        else
            push!(smat_to_para_idx, pi)
        end
    end
    n = length(smat_to_para_idx)
    ctimer_stop!(c_timer, 50)
    if n == 0
        is_output_rank(ctx) && @warn "Parton SR-CG: no parameters to optimize (all fixed or cut)"
        return 0
    end

    # --- 準備: このランクのサンプルの列(実部 n_sample 本 + 虚部 n_sample 本)、⟨O⟩、対角、右辺
    ctimer_start!(c_timer, 51)
    ctimer_start!(c_timer, 815)
    columns = Matrix{Float64}(undef, n, 2 * n_sample)
    @inbounds for s = 1:n_sample
        offset = (s - 1) * size_2 + 3
        for (si, pi) in enumerate(smat_to_para_idx)
            o = sr.sr_opt_o_store[offset + pi]
            columns[si, s] = real(o)
            columns[si, n_sample + s] = imag(o)
        end
    end
    o_mean = [real(o_mean_all[pi + 3]) for pi in smat_to_para_idx]
    s_diag = [s_diag_all[pi + 1] for pi in smat_to_para_idx]
    energy = real(sr.sr_opt_ho[1])
    g = [-2.0 * mp.dsr_opt_step_dt * (real(sr.sr_opt_ho[pi + 3]) - energy * real(o_mean_all[pi + 3]))
         for pi in smat_to_para_idx]
    inv_w = 1.0 / real(state.energy.wc)
    sta_del = mp.dsr_opt_sta_del
    precond = [1.0 / ((1.0 + sta_del) * d) for d in s_diag]
    ctimer_stop!(c_timer, 815)

    # z = S·d(S = (1/W) Σ_c a_c a_cᵀ − ⟨O⟩⟨O⟩ᵀ + sta_del·diag)
    function apply_s!(z, d)
        ctimer_start!(c_timer, 817)
        bcast!(ctx, d; root = 0, which = :comm0)
        ctimer_stop!(c_timer, 817)
        ctimer_start!(c_timer, 816)
        fill!(z, 0.0)
        @inbounds for c = 1:(2 * n_sample)
            y = 0.0
            @simd for i = 1:n
                y += columns[i, c] * d[i]
            end
            @simd for i = 1:n
                z[i] += y * columns[i, c]
            end
        end
        ctimer_stop!(c_timer, 816)
        ctimer_start!(c_timer, 817)
        allreduce_sum!(ctx, z; which = :comm0)
        ctimer_stop!(c_timer, 817)
        coef = dot(o_mean, d)
        @inbounds @simd for i = 1:n
            z[i] = inv_w * z[i] - coef * o_mean[i] + sta_del * s_diag[i] * d[i]
        end
        return z
    end

    # --- 前処理つき共役勾配
    ctimer_start!(c_timer, 57)
    x = zeros(n)
    r = copy(g)
    p = precond .* r
    d = copy(p)
    q = zeros(n)
    rp = dot(r, p)
    threshold = mp.dsr_opt_cg_tol^2 * Float64(n)^2
    max_iter = mp.nsr_opt_cg_max_iter > 0 ? mp.nsr_opt_cg_max_iter : n
    iter = 0
    while iter < max_iter && dot(r, r) >= threshold
        iter += 1
        apply_s!(q, d)
        dq = dot(d, q)
        abs(dq) < 1e-300 && break
        step = rp / dq
        x .+= step .* d
        if iter % 20 == 0          # 丸めの蓄積を払う(上流と同じ間隔で r = g − S x を取り直す)
            apply_s!(q, x)
            r .= g .- q
        else
            r .-= step .* q
        end
        ctimer_start!(c_timer, 818)
        p .= precond .* r
        ctimer_stop!(c_timer, 818)
        rp_new = dot(r, p)
        d .= p .+ (rp_new / rp) .* d
        rp = rp_new
    end
    ctimer_stop!(c_timer, 57)
    ctimer_stop!(c_timer, 51)

    ctimer_start!(c_timer, 52)
    info = Int(bcast_scalar(ctx, all(isfinite, x) ? 0 : 1))
    if info == 0 && is_output_rank(ctx)
        _write_cg_srinfo!(data, n_para_full, n, opt_num, cut_num, s_diag_max, s_diag_min,
                          smat_to_para_idx, x, iter, srinfo_dir)
        for (si, pi) in enumerate(smat_to_para_idx)
            if pi % 2 == 0
                update_parameter_value(data, div(pi, 2) + 1, x[si], 0.0)
            else
                update_parameter_value(data, div(pi - 1, 2) + 1, 0.0, x[si])
            end
        end
    end
    ctimer_stop!(c_timer, 52)
    return info
end

"""
    parton_oo_from_store_blas!(sr_opt_oo, sr_opt_o_store, sr_opt_size, n_stored)

store から ⟨O O†⟩ の全体を BLAS で組む(Φ・Pf 共通。直接法のときだけ使う)。
並びは上流と同じ OO[(i−1)·size₂ + j] = Σ_s O[i,s]·conj(O[j,s])。列優先の行列 M として
M[j,i] = conj((A·A†)[j,i])(A = O_store、size₂ × n_stored)。
A·A† はエルミートなので ZHERK で上三角だけ作り(ZGEMM の半分の手間)、残りは写す。
"""
function parton_oo_from_store_blas!(sr_opt_oo::Vector{ComplexF64},
                                    sr_opt_o_store::Vector{ComplexF64}, sr_opt_size::Int,
                                    n_stored::Int)
    size_2 = 2 * sr_opt_size
    M = reshape(view(sr_opt_oo, 1:(size_2 * size_2)), size_2, size_2)
    if n_stored == 0
        fill!(M, 0)
        return nothing
    end
    A = reshape(view(sr_opt_o_store, 1:(size_2 * n_stored)), size_2, n_stored)
    LinearAlgebra.BLAS.herk!('U', 'N', 1.0, A, 0.0, M)      # 上三角に (A·A†)[j,i]、j ≤ i
    @inbounds for i = 2:size_2, j = 1:(i - 1)
        h = M[j, i]
        M[j, i] = conj(h)
        M[i, j] = h
    end
    return nothing
end
