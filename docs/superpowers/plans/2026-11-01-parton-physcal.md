# パートン物理量測定(NVMCCalMode = 1)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** α・Φ・Pf の 3 モードで、固定パラメータの射影後波動関数について ⟨b†_i b_j⟩ と 1 粒子移動で済む ⟨b†_i b_j b†_k b_l⟩ を測り、上流互換の `zvo_cisajs_*.dat` / `zvo_cisajscktalt_*.dat` に書く。

**Architecture:** 局所値は E_loc のホップ項と同じく契約 2 の振幅比(型ディスパッチで 3 モード共通)× Jastrow 比で作る。器・MPI 集計・出力は上流の `PhysicalQuantities` / `weight_average_green_func!` / `output_data_phys!` をそのまま使う。入口は既存の 3 ドライバで、`NVMCCalMode = 1` のとき SR ループの代わりに測定ループへ進む。

**Tech Stack:** Julia 1.10、MVMCOptimizers.jl / MVMCExpertModeParsers.jl、MPI.jl(付属の mpiexec)

**Spec:** `DESIGN_parton.md` §13(局所値の表・2 体の 4 類・必須入力・門番・出力・検証)。各 Task はこの節を前提にする。

## Global Constraints

- julia は `~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia`(以下 `julia`)。`/opt/julia-1.8.5` は使わない
- **NVMCCalMode = 0 の経路はビット一致**(既存テスト全緑のまま。最適化の出力を変えない)
- 新しい modpara キーは作らない。上流のファイル(`green_func_calc.jl`・`vmc_phys_cal.jl`・`data_io.jl` ほか)は**呼ぶだけで変更しない**
- コメント・docstring・エラーメッセージの説明は日本語(エラーメッセージ本文は既存の門番に合わせて英語)。新しい名前は略さない
- 新規ファイルの先頭 docstring に `--- parton-mode (fork addition) ---` を入れる(既存のパートンファイルと同じ)
- サイト番号: def と出力は 0-based、局所値関数の引数は 1-based。**変換は測定ループと全配置列挙の入口の 1 箇所ずつ**(DESIGN §3.1 規則 8)
- **コミットはユーザーの明示的な許可を得てから**(CLAUDE.md)。各 Task の Commit ステップはユーザーに確認してから実行する
- テストの単独実行は `cd MVMCOptimizers.jl && julia --project=. -e 'using Test; include("test/parton_test_helpers.jl"); include("test/parton_pair_test_helpers.jl"); include("test/parton_phys_test_helpers.jl"); include("test/<file>.jl")'`。全体は `julia --project=. test/runtests.jl`

## Review Focus

1. **α で MOM の枝(非アウフバウ占有)の解を `InPmfOcc` なしで測る** → 黙って別の状態を測ってはいけない。error で `_pmfocc_opt.dat` を In にするよう示す(Task 4 にテスト)
2. **ノード上のサンプル(|ip| < 1e−30)を飛ばしたとき**、正規化は有効サンプル数(wc)で行い Σ⟨n_i⟩ = N が保たれる(Task 2 にテスト)
3. **NDataQtySmp ≥ 2** で ismp ごとにアキュムレータが 0 に戻る(前の ismp を持ち越すと Σ⟨n_i⟩ = 2N になる)(Task 2 にテスト)
4. **MPI 多ランク**で rank 0 へ集めてから全体の Wc で割る(単体テストは逐次のみ。Task 6 の MPI 16 の実系検証で押さえる)
5. **NVMCCalMode = 0 なのに greenone/greentwo を namelist に書いた**場合、warn を出し、最適化の出力は Green の def が無いときとバイト一致(Task 4 にテスト)

---

### Task 1: 局所値と全配置列挙

**Files:**
- Create: `MVMCOptimizers.jl/src/parton_phys_cal.jl`(include は `MVMCOptimizers.jl/src/MVMCOptimizers.jl` のパートン群の末尾)
- Create: `MVMCOptimizers.jl/test/parton_phys_test_helpers.jl`
- Create: `MVMCOptimizers.jl/test/test_parton_phys_local.jl`
- Modify: `MVMCOptimizers.jl/test/runtests.jl`(パートン群の末尾に helpers と test を include)

**Interfaces:**
- Consumes: `parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m, r_new) -> (ratio, ip_new)`、`parton_log_proj_ratio(cfg, data, m, r_old, r_new)`、`is_occupied(cfg, r)`、`site_particle(cfg, r)`、`place_particle!`、`parton_make_proj_cnt!`、`parton_recompute_amplitude_all!`、`parton_calculate_ip`、`parton_qp_weight(data)`、`parton_local_energy(pstate, data, ip)`
- Produces:
  - `parton_green2_class(ri::Int, rj::Int, rk::Int, rl::Int) -> Symbol`(`:density` / `:hop_then_density` / `:density_then_hop` / `:same_particle` / `:two_particle`。DESIGN §13.1 の表の条件そのもの。0-based でも 1-based でも同じ結果)
  - `parton_local_green1(pstate::PartonOptimizationState, data::ExpertModeData, qp_weight, ri::Int, rj::Int) -> ComplexF64`(1-based、§13.1 の 1 体の表)
  - `parton_local_green2(pstate::PartonOptimizationState, data::ExpertModeData, qp_weight, ri::Int, rj::Int, rk::Int, rl::Int) -> ComplexF64`(1-based、§13.1 の 2 体の表。`:two_particle` なら error)
  - `parton_exact_expectations(pstate::PartonOptimizationState, data::ExpertModeData; max_configurations::Int = 2_000_000) -> NamedTuple{(:one_body, :two_body, :energy, :n_configurations)}`(全配置 C(Nsite, NElec) を辞書式に列挙し、各配置に全フレーバー固縛で置いて錨を打ち、重み |ip · exp(ln P_J)|² で局所値を平均する。`one_body` は `data.green_one_terms` の順、`two_body` は `data.green_two_terms` の順、`energy` は E_loc の平均。|Ψ(x)| = 0 の配置は飛ばす。配置数が上限を超えたら error。ln P_J は占有ペアの `_parton_jastrow_v` の和。組み合わせ列挙は依存を増やさず自前で書く)
  - テスト補助 `phys_toy_cases() -> Vector{NamedTuple{(:label, :pstate, :data, :fermionic)}}`(下の 5 ケース。各 data に greenone = 全 (i, j)、greentwo = 全 (i,i,j,j) + 各類の代表を足す)
  - テスト補助 `dense_expectation(data, pstate, ops::Vector{Tuple{Int,Int}}, fermionic::Bool) -> ComplexF64`(ops は 1-based の (生成サイト, 消滅サイト) の列で、O = b†_{ops[1][1]} b_{ops[1][2]} b†_{ops[2][1]} … の左から並べた積。昇順配置の Fock 基底 |x⟩ = b†_{r1} … b†_{rN}|0⟩ で Ψ(x) を全列挙し、O を右端の因子から素直に作用させる。硬芯: 占有サイトへの b† と空サイトの b は 0。フェルミオンなら作用サイトより小さい占有サイト数の偶奇で符号。**プロダクションの比の関数は使わない**)

5 ケース(既存フィクスチャから作る):

| label | 元 | F | 追加 |
|---|---|---|---|
| `alpha_F2` | `dimerized_mf_data` | 2 | QP 2 本(`set_shift_qp!`、重みは複素)+ Jastrow(`_add_ring_jastrow!`) |
| `alpha_F3` | `per_bond_mf_data(3)` | 3(フェルミオン) | QP 2 本 |
| `orbparam_F2` | `orbparam_toy_data` | 2 | なし |
| `pair_F2` | `pair_toy_data(n_flavor = 2)` | 2 | QP 2 本(`random_ring_qp!`) |
| `pair_F3` | `pair_toy_data(n_flavor = 3, n_elec = 2)` | 3(フェルミオン) | なし |

- [ ] **Step 1: 失敗するテストを書く** — `test_parton_phys_local.jl` に 4 つの testset:
  - `"§13 分類"`: `parton_green2_class(1,1,2,2) == :density`、`(1,2,3,3) == :hop_then_density`、`(1,1,2,3) == :density_then_hop`、`(1,2,2,3) == :same_particle`、`(1,2,2,1) == :same_particle`、`(1,2,3,4) == :two_particle`、`(1,2,3,1) == :two_particle`
  - `"§13.5-1 全配置列挙 vs 密行列"`: 5 ケースそれぞれで `parton_exact_expectations` の全 one_body・two_body が `dense_expectation` と `isapprox(...; atol = 1e-12, rtol = 1e-10)`。ケースごとに `maximum(abs, one_body) > 0.05`(偶然 0 の比較にしない)、two_body に 4 類がすべて含まれることも `@test`
  - `"§13.5-2 配置ごとの恒等式"`: 5 ケースの全配置 x で `parton_local_energy == Σ_physhop [t·G1(j,i) + conj(t)·G1(i,j)] + Σ_coulomb V·G2(i,i,j,j)` を `atol = 1e-12`(物理ホップの t は複素であることを確認)
  - `"§13 two_particle は局所値で error"`: `@test_throws ErrorException parton_local_green2(..., 1,2,3,4)`
- [ ] **Step 2: 失敗を確認** — 上の単独実行コマンドで `test_parton_phys_local.jl`。Expected: `UndefVarError: parton_green2_class`
- [ ] **Step 3: `src/parton_phys_cal.jl` に上の 4 関数を実装する** — G1 の比は E_loc のホップ項(`parton_vmc_main_cal.jl:83-97`)と同じ呼び方。`qp_weight` は呼び出し側で 1 回だけ作る
- [ ] **Step 4: テストが通ることを確認** — Expected: 4 testset とも PASS。**F = 3 のケースだけ落ちたら**、E_loc 側の符号(既存の F 奇数の扱い)も疑って原因を切り分けてから報告する(実装を密行列に合わせて曲げない)
- [ ] **Step 5: Commit**(ユーザーの許可後)— `feat(parton-phys): 物理量の局所値(1 体・1 粒子移動の 2 体)と全配置列挙`

### Task 2: 測定ループと出力

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_phys_cal.jl`
- Create: `MVMCOptimizers.jl/test/test_parton_phys_loop.jl`(runtests に追加)

**Interfaces:**
- Consumes: Task 1 の局所値関数。上流の `initialize_phys_quantities!(state, data)`、`reset_phys_quantities!(state)`、`accumulate_phys_cis_ajs!(phys, idx, w, val)`、`accumulate_phys_cis_ajs_ckt_alt_dc!(phys, idx, w, val)`、`weight_average_we!(ctx, state)`、`weight_average_green_func!(ctx, state)`、`reduce_counter!(ctx, counter)`、`output_data_phys!(data, state, ismp; output_dir)`。パートンの `parton_make_sample!`、`parton_restore_sample!`、`parton_store_sample!`
- Produces:
  - `parton_phys_main_cal!(pstate::PartonOptimizationState, data::ExpertModeData) -> Int`(戻り値 = 飛ばしたサンプル数。energy の wc/etot/etot2 を 0 から蓄積し、サンプルごとに E_loc と全 Green を蓄積する。`state.phys_quantities` は呼び出し側で初期化済みの前提。逐次)
  - `parton_vmc_phys_cal!(pstate::PartonOptimizationState, data::ExpertModeData, ctx::ParallelContext; rng, output_dir::String) -> Int`(DESIGN §13.4 の順。`initialize_phys_quantities!` はここで 1 回。0 を返す)

- [ ] **Step 1: 失敗するテストを書く** — `test_parton_phys_loop.jl`:
  - `"§13.4 総和則と恒等式(平均)"`: `alpha_F2` と `pair_F2` で `parton_make_sample!` → `parton_phys_main_cal!` → `weight_average_we!(serial_context(), state)` → `weight_average_green_func!(serial_context(), state)` のあと、`real(wc) == NVMCSample`、Σ_i 対角 ⟨n_i⟩ が N に `atol = 1e-12`、etot が §13.5-2 の式を Green の平均から組んだ値に `atol = 1e-12`
  - `"Review Focus 2: ノード上のサンプル"`: `orbparam_F2` で Φ の 1 行(サイト r)を 0 にし、`parton_store_sample!` で手で詰めたサンプル列の一部を r 占有の配置にする → 戻り値(飛ばした数)> 0、`wc` = 有効数、Σ⟨n_i⟩ = N(`atol = 1e-12`)
  - `"Review Focus 3 と出力"`: `alpha_F2` で `NDataQtySmp = 2` にして `parton_vmc_phys_cal!(...; output_dir = mktempdir())` → `zvo_cisajs_001.dat` と `_002.dat` がともにあり、各ファイルの対角の和 = N(`atol = 1e-10`)、`zvo_cisajscktalt_001.dat` の行数 = greentwo の項数、`zvo_out.dat` が 2 行
- [ ] **Step 2: 失敗を確認** — Expected: `UndefVarError: parton_phys_main_cal!`
- [ ] **Step 3: 2 関数を実装する** — 構造は `parton_main_cal!`(`parton_vmc_main_cal.jl:531-`)の逐次経路から O と SR を抜いたもの。Green の 0-based → 1-based はループ前に 1 回だけ変換した添字表を作る
- [ ] **Step 4: テストが通ることを確認** — Expected: PASS
- [ ] **Step 5: Commit**(ユーザーの許可後)— `feat(parton-phys): 測定ループと上流互換の出力`

### Task 3: 門番

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_unsupported_inputs.jl`(`validate_parton_modpara` の NVMCCalMode 検査と docstring)
- Modify: `MVMCOptimizers.jl/src/parton_phys_cal.jl`
- Modify: `MVMCOptimizers.jl/test/test_parton_gatekeeper.jl:70`(拒否の例を `NVMCCalMode=2` に変える)
- Create: `MVMCOptimizers.jl/test/test_parton_phys_gate.jl`(runtests に追加)

**Interfaces:**
- Produces:
  - `validate_parton_modpara` は `vmc_calc_mode ∈ (0, 1)` を受け、1 のとき `n_data_qty_smp ≥ 1` を要求する
  - `validate_parton_green_inputs(data::ExpertModeData) -> Nothing`(DESIGN §13.3。NVMCCalMode = 1: サイト範囲・スピン 0・TwoBodyGEx なし・`two_particle` なしを error で検査し、メッセージに greentwo の何番目の項か(1-based)と 4 つの添字を入れる。NVMCCalMode = 0: green_one/green_two が空でなければ `@warn "OneBodyG / TwoBodyG are not read in parameter optimization (NVMCCalMode = 0)"`)
  - `parton_phys_require_fixed_inputs(data::ExpertModeData, namelist_path::AbstractString; orbital_given::Bool, orbital_hint::String) -> Nothing`(DESIGN §13.2 の表。`orbital_given = false` なら `orbital_hint`(例: "add InPmfOrb (use <head>_pmforb_opt.dat)")を含めて error。射影族は `jastrow_terms` が空でなく namelist に `InJastrow` が無ければ error。パートンが受け付ける他の射影族があれば同じ規則で足す)

- [ ] **Step 1: 失敗するテストを書く** — `test_parton_phys_gate.jl` で、`toy_mf_data` に `vmc_calc_mode = 1` を入れたものを起点に: スピン 1 の greenone → error(メッセージに "spin")、サイト = Nsite → error、TwoBodyGEx が 1 項 → error、greentwo に (0,1,2,3) → error でメッセージに "two_particle" と "(0, 1, 2, 3)"、`n_data_qty_smp = 0` → error、正常な入力 → `nothing`。NVMCCalMode = 0 + greenone あり → `@test_logs (:warn, r"OneBodyG") validate_parton_green_inputs(data)`。`parton_phys_require_fixed_inputs` は一時ディレクトリの namelist で、Jastrow 項あり・InJastrow なし → error(メッセージに "InJastrow")、`orbital_given = false` → hint の文字列を含む error
- [ ] **Step 2: 失敗を確認** — Expected: `UndefVarError: validate_parton_green_inputs`、および gatekeeper の既存テストは NVMCCalMode=1 がまだ拒否されるので変更前は通る
- [ ] **Step 3: 実装する**
- [ ] **Step 4: 通ることを確認** — `test_parton_phys_gate.jl` と `test_parton_gatekeeper.jl`。Expected: PASS
- [ ] **Step 5: Commit**(ユーザーの許可後)— `feat(parton-phys): 物理量測定の門番(NVMCCalMode を 0/1 に開放)`

### Task 4: 3 ドライバの分岐と実行テスト

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_run_para_opt_from_namelist.jl`(α)
- Modify: `MVMCOptimizers.jl/src/parton_vmc_para_opt_orbparam.jl`(`parton_orbparam_run_from_namelist`、Φ)
- Modify: `MVMCOptimizers.jl/src/parton_vmc_para_opt_pair.jl`(`parton_pair_run_from_namelist`、Pf)
- Modify: `MVMCOptimizers.jl/test/parton_test_helpers.jl` / `parton_pair_test_helpers.jl`(`test_parton_orbparam_run.jl` の `_write_min_orbparam_input`・`_add_jastrow_def!` と `test_parton_pair_run.jl` の `_write_min_pair_input` を helpers へ移す。中身は変えない)
- Create: `MVMCOptimizers.jl/test/test_parton_phys_run.jl`(runtests に追加)

**Interfaces:**
- Consumes: Task 2 の `parton_vmc_phys_cal!`、Task 3 の 2 関数。`parton_init_alpha!` の戻り値(乱数で埋めた idx の集合)、`parton_read_in_pmfpara!`(戻り値は読んだか)、`parton_orbparam_read_in_pmforb!` / `parton_pair_read_in!`(戻り値はパスか nothing)、`parton_read_in_pmfocc`、`parton_check_occupation_selfcontained`
- Produces: 3 ドライバとも、パース直後(modpara 検査の後)に `validate_parton_green_inputs(data)`、`NVMCCalMode = 1` なら初期化の後に `parton_phys_require_fixed_inputs(...)` を呼び、振幅の源を固定値で組んでから `parton_vmc_phys_cal!` を呼んで SR ループを飛ばす。`_init.dat` 系と runinfo の書き出しは従来どおり。α は `InPmfOcc` が無く占有が自己完結でなければ error("add InPmfOcc (use <head>_pmfocc_opt.dat)")。Φ・Pf は SR ループの各 step 冒頭で振幅の源を組む関数(Pf は `parton_pair_load!`、Φ は同じ役の関数)を測定の前に 1 回呼ぶ

- [ ] **Step 1: 失敗するテストを書く** — `test_parton_phys_run.jl`。α・Φ・Pf のそれぞれで: 最小入力で短い最適化(NSROptItrStep 3)を回す → その `_pmfpara_opt.dat` + `_pmfocc_opt.dat`(α)/ `_pmforb_opt.dat`(Φ)/ `_parton_pair_opt.dat`(Pf)を In として namelist に足し、全 (i,j) の greenone と全 (i,i,j,j) の greentwo を足し、`NVMCCalMode 1`・`NDataQtySmp 2` にして同じ入口で実行 → status 0、`zvo_cisajs_001.dat`・`_002.dat` があり対角和 = N(`atol = 1e-10`)、各 ismp の zvo_out.dat の E が Green の出力から組んだ値に `atol = 1e-10`。さらに:
  - In を外すと error(α: `InPmfPara` が無く乱数 idx が残る / Φ: `InPmfOrb` なし / Pf: `InPartonPairOrbital` なし)
  - Review Focus 1: α で `onsite_crossing_data` を使い非アウフバウ占有を `InPmfOcc` で与えた最適化の出力から、`InPmfOcc` を外して測定 → error に "InPmfOcc"
  - Review Focus 5: α の最小入力で NVMCCalMode = 0 の最適化を greenone あり / なしで 2 回 → `read(zvo_out.dat)` がバイト一致し、ありの方で warn
- [ ] **Step 2: 失敗を確認** — Expected: 測定 run が SR ループに入り Green のファイルが無い(または門番で落ちる)
- [ ] **Step 3: 3 ドライバを改修し、helpers の移動を行う**
- [ ] **Step 4: 全テスト** — `julia --project=. test/runtests.jl`。Expected: 全緑(既存の Parton 件数 + 新規)。P 層 `test/physics` も回す(1292 件全緑)
- [ ] **Step 5: Commit**(ユーザーの許可後)— `feat(parton-phys): α・Φ・Pf の入口で NVMCCalMode = 1 を測定ループへ分岐`

### Task 5: def 生成の道具と文書

**Files:**
- Create: `tools/parton_gen_green_def.jl`
- Create: `MVMCOptimizers.jl/test/test_parton_phys_tools.jl`(runtests に追加。`test_parton_pair_tools.jl` と同じ流儀で道具を実行する)
- Modify: `docs/parton_run.md`(節「物理量測定(NVMCCalMode = 1)」)、`docs/parton_orbparam.md`・`docs/parton_pair.md`(その節への 1 行の案内)、`.claude/CLAUDE.md`(現在の状況と TODO)

**Interfaces:**
- Produces: `julia tools/parton_gen_green_def.jl <Nsite> <out_dir>` が `<out_dir>/greenone.def`(全 (i, j)、i = j を含む Nsite² 行、スピン 0)と `<out_dir>/greentwo.def`(全 (i, i, j, j)、Nsite² 行)を上流の 5 行ヘッダ(2 行目 `NCisAjs <n>` / `NCisAjsCktAltDC <n>`)つきで書く

- [ ] **Step 1: 失敗するテストを書く** — Nsite = 4 で道具を実行 → `parse_green_one_def` / `parse_green_two_def` が成功し、項数 16 / 16、スピンがすべて `:up`、greentwo がすべて `parton_green2_class == :density`
- [ ] **Step 2: 失敗を確認** — Expected: 道具が無い
- [ ] **Step 3: 道具を書き、文書を更新する** — 文書には: namelist に足す行(OneBodyG / TwoBodyG / In ファイル)、モード別の必須入力の表(DESIGN §13.2)、出力ファイルと列、2 体で扱える 4 類と未対応の類、MPI での実行例(`run_case_mpi.jl` の流用)
- [ ] **Step 4: 通ることを確認** — Expected: PASS
- [ ] **Step 5: Commit**(ユーザーの許可後)— `feat(tools): パートン測定用の greenone/greentwo 生成と運用文書`

### Task 6: 実系での検証(KM ν=1/2 4×4、α と Pf、MPI 16)

**Files:**
- Create: `tools/parton_physcal_exact.jl`(`<namelist> <out_dir>`。ドライバと同じ初期化で状態を組み、`parton_exact_expectations` の結果を `<head>_cisajs_exact.dat` / `<head>_cisajscktalt_exact.dat` / `<head>_energy_exact.dat` に上流と同じ書式で書く)
- Create(git 管理外): `playground_nozomi/projects/kapit_mueller/km/physcal/`(入力・出力・比較スクリプト `compare_exact.jl`・`REPORT_physcal.md`)

**Interfaces:**
- Consumes: Task 1 の `parton_exact_expectations`、Task 4 の入口、Task 5 の生成道具

- [ ] **Step 1: 入力を作る** — α は `playground_nozomi/projects/kapit_mueller/km/runs/L4x4_nu12_sym_k0_hm8p0_s1001` の段 1 の最適解(`_pmfpara_opt` / `_pmfocc_opt` / Jastrow を段 2 と同じ引き継ぎ方で In 化)、Pf は `playground_nozomi/dev/parton_pf/T4/` の KM 4×4 の最良 run の `_parton_pair_opt.dat`。どちらも `NVMCCalMode 1`・`NDataQtySmp 5`・NVMCSample はランクあたり 2000(合計 32000/ismp)、greenone/greentwo は Task 5 の道具で全対
- [ ] **Step 2: 厳密値** — `julia --project=MVMCOptimizers.jl tools/parton_physcal_exact.jl <namelist> <out>`(C(16,4) = 1820 配置)
- [ ] **Step 3: 測定** — MPI 16(`JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`、MPI.jl 付属の mpiexec)
- [ ] **Step 4: 比較** — 合格条件: 全成分で |MC 平均 − 厳密| < 4σ(σ = 5 個の ismp の標準誤差)、各 ismp で Σ⟨n_i⟩ = 4(`1e-10`)、E の再構成(`1e-10`)、E_MC と厳密 E(−4)の一致。外れた成分は表で REPORT に残す
- [ ] **Step 5: 記録** — `REPORT_physcal.md` に結果を書き、`.claude/CLAUDE.md` の TODO を更新(次の一手 = C4 を課した 8×8 の解で物理量の C4 対称性を測る)。`tools/parton_physcal_exact.jl` の Commit(ユーザーの許可後)

---

## 後続(この計画の範囲外)

- 2 粒子を動かす 2 体項(ペア相関など)。Slater はフレーバーごとの 2×2 det、Pf は小 Pf の拡張
- n(k)・S(q) の後処理(`parton_lattice.dat` の座標を使う)と、C4 射影の効果の測定(TODO「パートンモードの物理量計算」の本題)
- 測定ループのスレッド並列
