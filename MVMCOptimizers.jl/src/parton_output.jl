"""
パートンモードの診断・解析・再現性のための出力
--- parton-mode (fork addition) ---

DESIGN_parton.md §3.3 の出力一覧に対応する。共通規約:

- 書き出しは **rank 0 のみ**(呼び出し側が `is_output_rank(ctx)` で判定する)
- 接頭辞は `modpara.c_data_file_head`(既定 `zvo`)/ `c_para_file_head`(既定 `zqp`)
- ステップ毎に追記する系は step 0 が `"w"`、以降 `"a"`(既存の流儀)
- **`PartonMode = 0` ではここのどれも呼ばれない**(パートン経路専用)
- 診断値は既存の状態(`PartonConfiguration.counter` / `PartonMFHamiltonian.min_gap`)から
  読むだけで、新しい数値計算は足さない
"""

"接頭辞つきの出力パスを作る。`_output_path` は data_io.jl の既存ヘルパ。"
_parton_out(data::ExpertModeData, suffix::String, dir) =
    _output_path((isempty(data.modpara.c_data_file_head) ? "zvo" :
                  data.modpara.c_data_file_head) * suffix, dir)
_parton_para_out(data::ExpertModeData, suffix::String, dir) =
    _output_path((isempty(data.modpara.c_para_file_head) ? "zqp" :
                  data.modpara.c_para_file_head) * suffix, dir)

# =====================================================================
# C. パートン固有の健全性ログ
# =====================================================================

"""
    parton_write_diag(data, pstate, step, dir; n_recompute, n_need_recompute,
                      alpha_norm_pre, alpha_norm_post)

`<head>_parton_diag.dat` に SR ステップ毎 1 行。新機構が健康かを一目で見る計器で、
upstream に対応物はない。

列の意味:
- `min_gap` — 全フレーバーの HOMO-LUMO ギャップの最小値。閉じると契約 0′ の分母が
  小さくなり O_k が発散する(DESIGN §8 のリスク項目)
- `n_need_recompute` — `ratio_floor` を割って `:need_recompute` になった回数。
  DESIGN §7 の防御が何回発動したか
- `n_exact_recompute` — 厳密再計算(錨)の総発動回数。頻度が妥当かを見る
- `accept_ratio` — 受理数 / 試行数。MC の基本的な健康診断
- `alpha_norm_pre/post` — ゲージ射影**直前/直後**の α のノルム。平坦方向への
  ドリフトの実測と、射影が効いているかの確認。射影が無効なら両者は同値になる
"""
function parton_write_diag(
    data::ExpertModeData,
    pstate::PartonOptimizationState,
    step::Int,
    dir;
    n_recompute::Int = 0,
    n_need_recompute::Int = 0,
    alpha_norm_pre::Float64 = 0.0,
    alpha_norm_post::Float64 = 0.0,
)
    dir === nothing && return nothing
    path = _parton_out(data, "_parton_diag.dat", dir)
    cfg = pstate.config
    trials = cfg.counter[1]
    accepts = cfg.counter[2]
    acc = trials > 0 ? accepts / trials : 0.0
    open(path, step == 0 ? "w" : "a") do f
        if step == 0
            println(f, "# step  min_gap  n_need_recompute  n_exact_recompute  " *
                       "accept_ratio  alpha_norm_pre  alpha_norm_post  n_trial  n_accept")
        end
        @printf(f, "%6d % .8e %8d %8d % .6e % .8e % .8e %10d %10d\n",
                step, pstate.mfham.min_gap, n_need_recompute, n_recompute,
                acc, alpha_norm_pre, alpha_norm_post, trials, accepts)
    end
    return path
end

# =====================================================================
# B. ステップ毎の壁時計
# =====================================================================

"""
    parton_write_time(data, step, dir, elapsed_step, elapsed_total)

`<head>_parton_time.dat` にステップ毎の壁時計。

C 版 mVMC に対応する出力は `zvo_CalcTimer.dat`(区間別の累積)だけで、
**ステップ毎の壁時計を持つファイルは存在しない**ため、パートン専用として定義する
(既存の名前と衝突しないよう `_parton_time.dat` とした)。区間別の累積が要るときは
既存の `MVMC_C_TIMER=1` で `zvo_CalcTimer.dat` を出せる。
"""
function parton_write_time(data::ExpertModeData, step::Int, dir,
                           elapsed_step::Float64, elapsed_total::Float64)
    dir === nothing && return nothing
    path = _parton_out(data, "_parton_time.dat", dir)
    open(path, step == 0 ? "w" : "a") do f
        step == 0 && println(f, "# step  sec_this_step  sec_total")
        @printf(f, "%6d % .6e % .6e\n", step, elapsed_step, elapsed_total)
    end
    return path
end

# =====================================================================
# D. 最適化後の平均場ハミルトニアン / バンド
# =====================================================================

"""
    parton_write_mfham(data, mfham, dir; with_vectors=false)

SR ループ終了後に 1 回。研究の出口(バンド構造・Chern 数解析)へ直結する成果物。

- `<para>_pmfham_opt.dat` — 各フレーバーの H^(f)(α*)(サイト×サイト、複素)
- `<para>_pmfband_opt.dat` — 各フレーバーの固有値。占有/非占有の境界と
  HOMO-LUMO ギャップも記録
- 固有ベクトルは大きくなるので既定 OFF(`with_vectors = true` で
  `<para>_pmfvec_opt.dat` に出す)

**唯一の正は α 側**(`<para>_pmfpara_opt.dat`)。このファイルは α から H を組み直す
手間を省くための便宜で、α と齟齬が出たら α を信じること。
"""
function parton_write_mfham(data::ExpertModeData, mfham::PartonMFHamiltonian, dir;
                            with_vectors::Bool = false)
    dir === nothing && return nothing
    n_elec = data.modpara.nelec
    n_flavor = length(mfham.h_mf)
    n_site = size(mfham.h_mf[1], 1)

    ham_path = _parton_para_out(data, "_pmfham_opt.dat", dir)
    open(ham_path, "w") do f
        println(f, "# 最適化後の平均場ハミルトニアン H^(f)(alpha*)")
        println(f, "# 唯一の正は alpha 側(*_pmfpara_opt.dat)。これは再構成の手間を省く便宜。")
        println(f, "# flavor(1-based)  site1(1-based)  site2  Re  Im")
        @printf(f, "NFlavor %d\nNSite %d\n", n_flavor, n_site)
        for fl = 1:n_flavor, i = 1:n_site, j = 1:n_site
            v = mfham.h_mf[fl][i, j]
            abs(v) > 1e-14 || continue
            @printf(f, "%d %d %d % .18e % .18e\n", fl, i, j, real(v), imag(v))
        end
    end

    band_path = _parton_para_out(data, "_pmfband_opt.dat", dir)
    open(band_path, "w") do f
        println(f, "# 最適化後の平均場バンド(固有値、昇順)")
        println(f, "# flavor  level(1-based)  eigenvalue  occupied(1=占有)")
        @printf(f, "NFlavor %d\nNSite %d\nNElec %d\n", n_flavor, n_site, n_elec)
        for fl = 1:n_flavor
            ev = mfham.eig_vals[fl]
            gap = n_elec < n_site ? ev[n_elec + 1] - ev[n_elec] : NaN
            @printf(f, "# flavor %d  HOMO=% .10e  LUMO=% .10e  gap=% .10e\n",
                    fl, ev[n_elec], n_elec < n_site ? ev[n_elec + 1] : NaN, gap)
            for k = 1:n_site
                @printf(f, "%d %d % .18e %d\n", fl, k, ev[k], k <= n_elec ? 1 : 0)
            end
        end
    end

    vec_path = nothing
    if with_vectors
        vec_path = _parton_para_out(data, "_pmfvec_opt.dat", dir)
        open(vec_path, "w") do f
            println(f, "# 平均場固有ベクトル  flavor  level  site  Re  Im")
            @printf(f, "NFlavor %d\nNSite %d\n", n_flavor, n_site)
            for fl = 1:n_flavor, k = 1:n_site, i = 1:n_site
                v = mfham.eig_vecs[fl][i, k]
                @printf(f, "%d %d %d % .18e % .18e\n", fl, k, i, real(v), imag(v))
            end
        end
    end
    return (ham_path, band_path, vec_path)
end

# =====================================================================
# E. run メタデータ
# =====================================================================

"""
    parton_write_runinfo(data, dir; namelist_path, base_seed, n_idx, ...)

`<head>_parton_runinfo.dat` に 1 回書き。鎖方式で大量の run を繋ぐ運用のため、
後から出自を辿れるようにする。**git ハッシュの取得に失敗しても run は落とさない**
(`unknown` を書く)。
"""
function parton_write_runinfo(
    data::ExpertModeData,
    dir;
    namelist_path::AbstractString = "",
    base_seed::Integer = 0,
    n_idx::Int = 0,
    n_para::Int = 0,
    n_rank::Int = 1,
    t_start::Float64 = 0.0,
    t_end::Float64 = 0.0,
)
    dir === nothing && return nothing
    path = _parton_out(data, "_parton_runinfo.dat", dir)
    mp = data.modpara

    githash = try
        d = isempty(namelist_path) ? pwd() : dirname(abspath(String(namelist_path)))
        strip(read(`git -C $d rev-parse HEAD`, String))
    catch
        "unknown"
    end

    open(path, "w") do f
        println(f, "# パートン run のメタデータ(再現・追跡用)")
        println(f, "githash $githash")
        println(f, "namelist $(namelist_path)")
        @printf(f, "base_seed %d\n", base_seed)   # 乱数初期化の再現に必要な値
        @printf(f, "modpara_rnd_seed %d\n", mp.rnd_seed)
        @printf(f, "n_rank %d\nn_thread %d\n", n_rank, Threads.nthreads())
        @printf(f, "PartonMode %d\nNFlavor %d\nNElec %d\nNSite %d\n",
                mp.parton_mode, mp.nflavor, mp.nelec, mp.nsite)
        @printf(f, "n_idx %d\nn_para %d\n", n_idx, n_para)
        @printf(f, "NVMCSample %d\nNVMCWarmUp %d\nNVMCInterval %d\n",
                mp.nvmc_sample, mp.nvmc_warmup, mp.nvmc_interval)
        @printf(f, "NSROptItrStep %d\nNSROptItrSmp %d\n",
                mp.nsr_opt_itr_step, mp.nsr_opt_itr_smp)
        @printf(f, "DSROptStepDt %.17g\nDSROptStaDel %.17g\nDSROptRedCut %.17g\n",
                mp.dsr_opt_step_dt, mp.dsr_opt_sta_del, mp.dsr_opt_red_cut)
        @printf(f, "PartonGaugeFix %d\nPartonBlockUpdateSize %d\n",
                mp.parton_gauge_fix, mp.parton_block_update_size)
        @printf(f, "t_start %.6f\nt_end %.6f\nwall_sec %.6f\n",
                t_start, t_end, t_end - t_start)
        # 入力 .def 一式(namelist に挙がっているもの)の名前とサイズ・ハッシュ
        if !isempty(namelist_path) && isfile(namelist_path)
            base = dirname(abspath(String(namelist_path)))
            println(f, "# input file_type file_name bytes hash")
            for (ft, fp) in MVMCExpertModeParsers.parse_namelist_content(
                    MVMCExpertModeParsers.read_def_file(String(namelist_path)))
                full = joinpath(base, fp)
                h = isfile(full) ? string(hash(read(full, String)), base = 16) : "missing"
                sz = isfile(full) ? filesize(full) : 0
                @printf(f, "input %s %s %d %s\n", ft, fp, sz, h)
            end
        end
    end
    return path
end

# =====================================================================
# F-1. 収束テーブル
# =====================================================================

"""
    parton_write_conv(data, dir)

SR ループ終了後に `<head>_out.dat` を読んで `<head>_conv.dat` を出す。列は

    step   E   var   |E - E_tail|   var/E^2

`E_tail` は**最終 NSROptItrSmp ステップの E の平均**。この区間は `zqp_opt.dat` が
パラメータを平均している区間と同一なので、「図の漸近値 = 採用した最適パラメータの値」が
保証される。tail 区間の中では `|E - E_tail|` がノイズ床に落ちるが、これは想定どおりで
異常ではない。

**log は取らない**(0 や負値の扱いは作図側の責任。ここは生の量だけを出す)。
"""
function parton_write_conv(data::ExpertModeData, dir)
    dir === nothing && return nothing
    src = _parton_out(data, "_out.dat", dir)
    isfile(src) || return nothing

    rows = [parse.(Float64, split(strip(l)))
            for l in eachline(src) if !isempty(strip(l))]
    isempty(rows) && return nothing
    e = [r[1] for r in rows]
    var = [length(r) >= 4 ? r[4] : NaN for r in rows]

    n_smp = max(1, min(data.modpara.nsr_opt_itr_smp, length(e)))
    e_tail = sum(@view e[(end - n_smp + 1):end]) / n_smp

    path = _parton_out(data, "_conv.dat", dir)
    open(path, "w") do f
        println(f, "# 収束テーブル。作図は tools/plot_conv.jl(本体に作図依存を入れない)")
        @printf(f, "# E_tail = %.17g  (最終 NSROptItrSmp = %d ステップの平均)\n",
                e_tail, n_smp)
        println(f, "# step  E  var  |E - E_tail|  var/E^2")
        for (k, ek) in enumerate(e)
            @printf(f, "%6d % .17g % .17g % .17g % .17g\n",
                    k, ek, var[k], abs(ek - e_tail),
                    ek == 0 ? NaN : var[k] / (ek * ek))
        end
    end
    return path
end
