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

"`.def` 族の区切り行。mVMC の .def はコメント機能を持たないので `#` は使わない。"
const PARTON_DEF_RULE = "==============================="

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
    parton_write_time(data, step, dir, elapsed_step, elapsed_total;
                      n_out, n_rank)

`<head>_parton_time.dat` にステップ毎の壁時計と**サンプリング量**(v3.10)。

## C 版との対応(v3.10 調査)

C 版の per-step 出力 `<head>_time_<idx>.dat`(`vmcclock.c` の `OutputTime`)は
**受理率 3 種(hop/ex/lsf)+ 試行カウンタ + ctime 文字列**で、経過秒も
サンプリング量も持たない。その役割(受理率・試行数)はパートンでは
`zvo_parton_diag.dat` が per-step で担っており、hop/ex/lsf の 3 種更新は
パートンに存在しない(固縛ホップ 1 種のみ)ため、列構成は踏襲しない。
このファイルは「**時間と仕事量**」に専念する(役割分担は §3.3.1)。

## 列(§4 の規約から導出。時間だけでは何を何回やった時間か分からないため)

- `n_out`: そのステップの外側ループ数。**初回は WarmUp+Sample、burn 再開後は
  Sample+1** とステップごとに変わる(だから毎ステップ記録する)。値は
  `parton_n_out`(式の家)から取ったものを受け取る
- `n_in = NVMCInterval × NSite`: サンプル間の内側ステップ数
- `n_sample_total = NVMCSample × n_rank`: **統計量**(誤差評価の母数。
  `weight_average_we!` が comm0 で allreduce するため rank 合算が実効値)
- `n_update_total = n_out × n_in × n_rank`: **仕事量**(1 更新あたりの時間の分母)

書式は診断系(§3.3.1 系統 (b)、SRinfo と同じ流儀): `#` ヘッダ・空白区切り・
`% .6e` / `%d`、step 0 で `"w"`(前 run の行を残さない)、rank 0 のみ。
列名ヘッダの前に `# key value` で並列構成(n_mpi_rank / n_julia_thread /
nvmc_sample_total)を置き、**このファイル単体でスケーリング解析ができる**ようにする。
"""
function parton_write_time(data::ExpertModeData, step::Int, dir,
                           elapsed_step::Float64, elapsed_total::Float64;
                           n_out::Int, n_rank::Int)
    dir === nothing && return nothing
    path = _parton_out(data, "_parton_time.dat", dir)
    mp = data.modpara
    n_in = parton_n_in(mp)
    n_sample_total = mp.nvmc_sample * n_rank
    n_update_total = n_out * n_in * n_rank
    open(path, step == 0 ? "w" : "a") do f
        if step == 0
            @printf(f, "# n_mpi_rank %d\n", n_rank)
            @printf(f, "# n_julia_thread %d\n", Threads.nthreads())
            @printf(f, "# nvmc_sample_total %d\n", n_sample_total)
            println(f, "# step  step_sec  cumulative_sec  n_out  n_in  " *
                       "n_sample_total  n_update_total")
        end
        @printf(f, "%6d % .6e % .6e %8d %6d %10d %14d\n",
                step, elapsed_step, elapsed_total,
                n_out, n_in, n_sample_total, n_update_total)
    end
    return path
end

# =====================================================================
# D. 最適化後の平均場ハミルトニアン / バンド
# =====================================================================

"""
    parton_write_mfham(data, mfham, dir; with_vectors=false)

SR ループ終了後に 1 回。研究の出口(バンド構造・Chern 数解析)へ直結する成果物。

- `<para>_pmfham_opt.dat` — **`.def` 族**。`site1 flavor1 site2 flavor2 Re Im` の
  行形式(サイト・フレーバーとも 0-based)で、**全 (flavor, site1, site2) の組を
  h.c. 側もゼロ要素も含めて**出力する(密ダンプ)
- `<para>_pmfband_opt.dat` — **診断系**。各フレーバーの固有値
- 固有ベクトルは大きくなるので既定 OFF(`with_vectors = true` で
  `<para>_pmfvec_opt.dat` に出す。こちらも診断系)

**唯一の正は α 側**(`<para>_pmfpara_opt.dat`)。このファイルは α から H を組み直す
手間を省くための便宜で、α と齟齬が出たら α を信じること。

## `.def` 族の形式規約(v3.5)

`clean_line` が `#` と `//` を除去するのは **Julia 移植で足された拡張**で、mVMC の
.def 形式にコメント機能はない。.def 族の出力に `#` を**書かないし読まない**。
読み手はヘッダ 5 行を固定でスキップする(`parse_input_parameter_file` の
`data_start = 6` と整合)。既存 per-block writer は 4 行ヘッダで読み手と食い違うが、
**既存側には触らない**(DESIGN §11 に upstream 報告候補として記録済み)。

## pmftrans.def との関係

行の**形式**は pmftrans.def のデータ行と同じだが、**内容は 1 対 1 ではない**。
pmftrans は片方向のみ列挙して h.c. を暗黙付与する規約なのに対し、こちらは
H^(f) の全要素をそのまま並べる(h.c. 側も入る)。したがって
**このファイルをそのまま pmftrans.def として再投入することはできない**
(逆向き重複としてテンプレート build が弾く)。キーワードを `NPmfHam` にしてあるのは
そのため — pmftrans パーサの `NPartonMFTrans` とは別物であることを名前で示す。

## 行順(決定論)

`(flavor, site1, site2)` の辞書順で全組を走査する。run 間で `diff` が取れることが
再現性・回帰テストの前提。
"""
function parton_write_mfham(data::ExpertModeData, mfham::PartonMFHamiltonian, dir;
                            with_vectors::Bool = false)
    dir === nothing && return nothing
    n_elec = data.modpara.nelec
    n_flavor = length(mfham.h_mf)
    n_site = size(mfham.h_mf[1], 1)

    ham_path = _parton_para_out(data, "_pmfham_opt.dat", dir)
    open(ham_path, "w") do f
        println(f, PARTON_DEF_RULE)
        println(f, "NPmfHam $(n_flavor * n_site * n_site)")
        println(f, PARTON_DEF_RULE)
        println(f, "== site1 flavor1 site2 flavor2 ReH ImH ==")
        println(f, PARTON_DEF_RULE)
        for fl = 0:(n_flavor - 1), i = 0:(n_site - 1), j = 0:(n_site - 1)
            v = mfham.h_mf[fl + 1][i + 1, j + 1]   # 0-based → 1-based はここだけ
            @printf(f, "%d %d %d %d % .18e % .18e\n", i, fl, j, fl, real(v), imag(v))
        end
    end

    band_path = _parton_para_out(data, "_pmfband_opt.dat", dir)
    open(band_path, "w") do f
        @printf(f, "# NFlavor %d  NSite %d  NElec %d\n", n_flavor, n_site, n_elec)
        for fl = 1:n_flavor
            ev = mfham.eig_vals[fl]
            gap = n_elec < n_site ? ev[n_elec + 1] - ev[n_elec] : NaN
            @printf(f, "# gap flavor %d % .18e\n", fl - 1, gap)
        end
        println(f, "# flavor band_index eigenvalue occupied")
        for fl = 1:n_flavor
            ev = mfham.eig_vals[fl]
            for k = 1:n_site
                @printf(f, "%d %d % .18e %d\n", fl - 1, k - 1, ev[k], k <= n_elec ? 1 : 0)
            end
        end
    end

    vec_path = nothing
    if with_vectors
        vec_path = _parton_para_out(data, "_pmfvec_opt.dat", dir)
        open(vec_path, "w") do f
            println(f, "# flavor band_index site Re Im")
            for fl = 1:n_flavor, k = 1:n_site, i = 1:n_site
                v = mfham.eig_vecs[fl][i, k]
                @printf(f, "%d %d %d % .18e % .18e\n", fl - 1, k - 1, i - 1,
                        real(v), imag(v))
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
        # 並列構成(v3.10)。時間の数字は「何ランク・何スレッドで出たか」が
        # 分からないと解釈できない。既存キー n_rank / n_thread はより明示的な
        # 名前に置き換えた(情報の二重管理をしないため旧キーは残さない)。
        @printf(f, "n_mpi_rank %d\n", n_rank)
        @printf(f, "n_julia_thread %d\n", Threads.nthreads())
        @printf(f, "blas_num_threads %d\n", LinearAlgebra.BLAS.get_num_threads())
        # JULIA_MVMC_INNER_THREADS の解決値(env とスレッド数の両方を満たすか)
        @printf(f, "inner_threads_enabled %d\n",
                parton_sample_threading_enabled() ? 1 : 0)
        @printf(f, "nsplit_size %d\n", mp.nsplit_size)
        # 実効サンプル数 = NVMCSample × ランク数(weight_average_we! が comm0 で
        # allreduce して合算後の Wc で正規化するため)。統計誤差の母数はこちら。
        # 注意: NSplitSize > 1 を将来解禁すると comm1 グループ内でサンプルが
        # 分割されるため、この式は「comm0 の全 rank 合算」のままでよいかを
        # 再確認すること(現状は門番が NSplitSize = 1 を保証)。
        @printf(f, "nvmc_sample_per_rank %d\n", mp.nvmc_sample)
        @printf(f, "nvmc_sample_total %d\n", mp.nvmc_sample * n_rank)
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

# =====================================================================
# G. CalcTimer(パートンセクション)
# =====================================================================

"""
パートンモードの CalcTimer セクション。

## ID 帯の選定(v3.5)

既存で使用中の ID は **0–72 / 600–603 / 920–966**(リポジトリ全体を grep して確認)。
C 版 mVMC の `OutputTimerParaOpt` は 0–99 を主要フェーズに、600 番台を lspinflip の
下位に使う体系で、920 番台以降は Julia 移植が足した診断。**800 番台は完全に空き**で、
C 版の番号体系(0–99 / 600 番台)からも Julia 移植の診断(900 番台)からも離れているので
将来の衝突が最も起きにくい。`CTIMER_N = 1000` なので上限内。

ラベル書式は既存 `CTIMER_PARA_OPT_LINES` と同じ `"  Label  [ID] "` + `%12.5f`。
タイマは C と同じく **inclusive**(親を止めずに子を回す)。
"""
const CTIMER_PARTON_LINES = Tuple{String,Int}[
    ("Parton total               [800] ", 800),
    ("  contract0 H+eigen        [801] ", 801),
    ("  contract0' dPhi          [802] ", 802),
    ("  sampling                 [803] ", 803),
    ("    contract1 recompute    [804] ", 804),
    ("    contract2 ratio        [805] ", 805),
    ("    contract3 update       [806] ", 806),
    ("  main_cal                 [807] ", 807),
    ("    contract4 E_loc        [808] ", 808),
    ("    contract5 O            [809] ", 809),
    ("    OO accumulate          [810] ", 810),
    ("  SR                       [811] ", 811),
    ("  sync                     [812] ", 812),
    ("  output                   [813] ", 813),
]

"""
    parton_write_ctimer(data, timer, dir)

`<head>_CalcTimer.dat` に**追記**する。既存の `write_ctimer_para_opt` が
先に本体セクションを `"w"` で書いた後に呼ぶこと。既存 writer には触らず、
同じファイルにパートンセクションを足すためにこの形にしてある。
"""
function parton_write_ctimer(data::ExpertModeData, timer::CTimer, dir)
    dir === nothing && return nothing
    path = _parton_out(data, "_CalcTimer.dat", dir)
    open(path, "a") do f
        for (label, id) in CTIMER_PARTON_LINES
            print(f, label, @sprintf("%12.5f\n", ctimer_seconds(timer, id)))
        end
    end
    return path
end
