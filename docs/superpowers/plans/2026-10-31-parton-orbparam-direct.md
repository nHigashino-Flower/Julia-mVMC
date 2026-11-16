# Φ 直接変分モード(上流型・第 1 段)実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `PartonOrbParam = 1` のとき、パートン Slater 行列式の軌道行列 Φ_f の全要素を SR の変分パラメータとし、SR 更新のたびにフレーバーごとに max|Φ_f| = `PartonOrbAmpMax` へそろえる(上流の D_AmpMax と同じ形)モードを足す。

**Architecture:** Φ は `ExpertModeData.pmforb_params` に平坦に置き、SR ロケータの新種別 `_PARAM_PMFORB` で既存の `stochastic_opt!` に乗せる。振幅系(契約 1〜4)は型注釈を抽象型 `PartonOrbitalSource` に広げるだけで本体を変えずに流用する。O(契約 5)・入出力・門番・SR ループ・ドライバの Φ 版は新規ファイルに置き、既存ドライバからはパース直後の 1 分岐で入る。α モード(`PartonOrbParam = 0`、既定)はビット一致。

**Tech Stack:** Julia 1.10、Test 標準ライブラリ、MVMCExpertModeParsers.jl / MVMCOptimizers.jl

**Spec:** `DESIGN_parton.md` §12.11(**優先**)と §12.0〜12.10(参照。§12.11 と食い違う箇所は §12.11 に従う)

## Global Constraints

- `PartonOrbParam = 0`(既定)の挙動はビット一致。全出力のバイト比較で確認する(Task 9)
- マルコフ連鎖の乱数消費順を変えない。Φ の初期化に乱数を使わない(InPmfOrb 必須)
- upstream ファイルの編集は登録点だけにし、各箇所に `# --- parton-mode (fork addition) ---` を一字一句同じに付ける(DESIGN §3.1)
- 構造体フィールドの追加は末尾+コンストラクタ末尾
- 0-based → 1-based の変換は InPmfOrb のリーダの中だけで行う
- O の縮約は転置積(共役なし)。`dot` や `'` を使わない(DESIGN §7)
- コメントは日本語、門番の error 文は既存に合わせて英語
- 新しい名前は略さない。接頭辞は `parton_orbparam_` / `PartonOrbParam`
- **git commit はユーザーの明示的な指示があるまでしない**(各タスクはテスト緑の確認で終える)
- julia は `JL=~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia`。`/opt/julia-1.8.5` は使わない
- テストの単独実行: `cd MVMCOptimizers.jl && $JL --project=. -e 'using Test; include("test/parton_test_helpers.jl"); include("test/<file>.jl")'`
- 新しいテストファイルは `MVMCOptimizers.jl/test/runtests.jl` のパートン節(`test_parton_force_gradient.jl` の後)に `include` を足す

## Review Focus

1. **InPmfOrb の Φ が列ランク落ち**(ゼロ列・列の重複)→ 起動時に error。行列式が恒等的に 0 で初期配置が見つからない、という分かりにくい失敗にしない(Task 6 にテスト)
2. **InPmfOrb の要素数と modpara の NSite・NElec・NFlavor が食い違う** → 件数の error を出す。OptFlag の長さの不一致で黙って凍結させない(Task 6・7 にテスト)
3. **全要素が 0 のフレーバー**を再スケールしようとする → 0 で割らずに error(Task 6 にテスト)
4. **`PartonOrbParam` のキー名を書き間違える** → 未知キーは黙って無視され mode 0 に落ちる。α の門番の「No mean-field parameter terms」に、Φ モードのつもりならキー名を確かめるよう一文足す(Task 7 にテスト)
5. **SR の解が NaN を含む** → 再スケールより**前**に判定して直前の Φ に戻し、その step は MC だけ進める(Task 8 にテスト)

---

### Task 1: パーサ(modpara キー・Φ の置き場・パラメータ数)

**Files:**
- Modify: `MVMCExpertModeParsers.jl/src/utils/constants.jl`(`DEFAULT_PARTON_OCC_MODE` の後に定数 2 つ、`MVMC_KEYWORDS` の `"InPmfOcc"` の後に `"InPmfOrb" => "InPmfOrb.def"`)
- Modify: `MVMCExpertModeParsers.jl/src/types/expert_types.jl`(`ModParaParameters` の 3 箇所の末尾 / `ExpertModeData` の末尾フィールド+コンストラクタ末尾)
- Modify: `MVMCExpertModeParsers.jl/src/parsers/modpara_parser.jl`(`PartonOccMode` の elseif の後)
- Modify: `MVMCExpertModeParsers.jl/src/utils/read_input_parameters.jl:163-171`(`count_variational_parameters`)
- Test: `MVMCExpertModeParsers.jl/test/test_parton_parsers.jl`(末尾に追加)

**Interfaces:**
- Produces: `DEFAULT_PARTON_ORB_PARAM = 0`、`DEFAULT_PARTON_ORB_AMP_MAX = 4.0`、`ModParaParameters.parton_orb_param::Int`、`ModParaParameters.parton_orb_amp_max::Float64`、`ExpertModeData.pmforb_params::Vector{ComplexF64}`(既定 `ComplexF64[]`)。`count_variational_parameters` は末尾に `length(data.pmforb_params)` を足す

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "modpara: Φ 直接変分のキー" begin
    r = MVMCExpertModeParsers.parse_modpara_content(
        "Nsite 4\nPartonOrbParam 1\nPartonOrbAmpMax 2.5\n")
    @test r.success
    @test r.data.parton_orb_param == 1
    @test r.data.parton_orb_amp_max == 2.5
    p = MVMCExpertModeParsers.ModParaParameters()
    @test p.parton_orb_param == 0
    @test p.parton_orb_amp_max == 4.0
end

@testset "count_variational_parameters が Φ の要素数を数える" begin
    data = MVMCExpertModeParsers.ExpertModeData()
    @test isempty(data.pmforb_params)
    @test MVMCExpertModeParsers.count_variational_parameters(data) == 0
    data.pmforb_params = zeros(ComplexF64, 2 * 4 * 2)
    @test MVMCExpertModeParsers.count_variational_parameters(data) == 16
end
```

- [ ] **Step 2: 失敗を確認する**

Run: `cd MVMCExpertModeParsers.jl && $JL --project=. -e 'using Test; include("test/test_parton_parsers.jl")'`
Expected: FAIL(`type ModParaParameters has no field parton_orb_param`)

- [ ] **Step 3: 実装する**(Files の各登録点。キーの読み取りは `safe_parse_int` / `safe_parse_float`)

- [ ] **Step 4: 通過を確認する**

Run: Step 2 と同じ。続けて `cd MVMCExpertModeParsers.jl && $JL --project=. -e 'using Pkg; Pkg.test()'`
Expected: すべて PASS

---

### Task 2: SR ロケータの Φ ブロック

**Files:**
- Modify: `MVMCOptimizers.jl/src/stochastic_opt.jl`(`@enum _ParameterKind` の `_PARAM_PMF` の後に `_PARAM_PMFORB`、`_foreach_parameter_location` と `_foreach_parameter_location_at` の末尾、`_parameter_location_value` と `_set_parameter_location_value!` の elseif 末尾)
- Modify: `MVMCOptimizers.jl/test/parton_test_helpers.jl`(`test_parton_jastrow.jl` から `_add_ring_jastrow!`・`_ln_pj_ref`・`_jast_place!` を**本体を変えずに**移す。Task 4・5 も使う)
- Modify: `MVMCOptimizers.jl/test/test_parton_jastrow.jl`(上の 3 関数の定義を削除)
- Test: `MVMCOptimizers.jl/test/test_parton_locator.jl`(末尾に追加)

**Interfaces:**
- Consumes: Task 1 の `pmforb_params`
- Produces: Φ ブロックはパラメータ並びの**末尾**。開始位置は `counts.n_para - length(data.pmforb_params) + 1`。`item_idx` は `pmforb_params` の 1-based 位置。共有 idx はない

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "Φ ブロック: pack/unpack/get/set/δ が往復する" begin
    data = MVMCExpertModeParsers.ExpertModeData()
    data.pmforb_params = ComplexF64[1+2im, 3, 4im, 5, 6, 7]
    para = MVMCOptimizers.pack_parameters(data)
    @test para == data.pmforb_params
    para[4] = -1 + 0.5im
    MVMCOptimizers.unpack_parameters!(data, para)
    @test data.pmforb_params[4] == -1 + 0.5im
    @test MVMCOptimizers.get_parameter_value(data, 6) == 7
    MVMCOptimizers.set_parameter_value!(data, 1, ComplexF64(9))
    @test data.pmforb_params[1] == 9
    MVMCOptimizers._add_parameter_delta_direct!(data, 2, ComplexF64(0.5, 0.5))
    @test data.pmforb_params[2] == ComplexF64(3.5, 0.5)
end

@testset "Φ ブロックは射影(Jastrow)ブロックの後ろ" begin
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite = 4
    _add_ring_jastrow!(data, [0.1, 0.2])          # 環 L=4 → 距離クラス 2 個
    data.pmforb_params = ComplexF64[10, 20, 30, 40]
    para = MVMCOptimizers.pack_parameters(data)
    @test length(para) == 6
    @test para[1:2] == ComplexF64[0.1, 0.2]
    @test para[3:6] == data.pmforb_params
end
```

- [ ] **Step 2: 失敗を確認する**(Global Constraints の単独実行コマンドで `test_parton_locator.jl`)。Expected: FAIL(Φ が pack に現れない)

- [ ] **Step 3: 実装する**。ヘルパの移動もこのステップで行う

- [ ] **Step 4: 通過を確認する**。`test_parton_locator.jl` と `test_parton_jastrow.jl` の両方を単独実行して PASS

---

### Task 3: 型(抽象型・PartonDirectPhi・状態の型パラメータ化・署名の拡大)

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_types.jl`(`PartonMFHamiltonian` の前に `abstract type PartonOrbitalSource end`、`PartonMFHamiltonian <: PartonOrbitalSource`、`PartonOptimizationState{O<:PartonOrbitalSource}` にしてフィールドを `mfham::O`、6 引数コンストラクタの注釈を `mfham::PartonOrbitalSource` に広げる)
- Modify: `MVMCOptimizers.jl/src/parton_calculate_m_all.jl:9`、`MVMCOptimizers.jl/src/parton_vmc_sampling.jl` の `parton_make_initial_sample!`(:89)・`parton_amplitude_ratio!`(:478)・`parton_update_amplitude!`(:528)の `mfham::PartonMFHamiltonian` → `mfham::PartonOrbitalSource`(本体は変えない)
- Create: `MVMCOptimizers.jl/src/parton_orbparam_types.jl`
- Modify: `MVMCOptimizers.jl/src/MVMCOptimizers.jl`(`parton_types.jl` の直後に include)
- Test: Create `MVMCOptimizers.jl/test/test_parton_orbparam_types.jl`

**Interfaces:**
- Produces:
  - `mutable struct PartonDirectPhi <: PartonOrbitalSource; orbitals::Vector{Matrix{ComplexF64}}; end`(各 Ns×Ne)。docstring に「振幅系が読むのは `.orbitals` だけ。α 専用フィールド(`n_idx`・`min_gap`・`gauge_*`)は持たないので、α 専用コードに誤って渡ると例外で止まる」と書く
  - `PartonDirectPhi(n_site::Int, n_elec::Int, n_flavor::Int)`(ゼロ行列)
  - `parton_orbparam_flat_index(f::Int, s::Int, n::Int, n_site::Int, n_elec::Int) -> Int` = `((f-1)*n_site + (s-1))*n_elec + n`(引数も返り値も 1-based)
  - `parton_orbparam_n_orb(mp::ModParaParameters) -> Int` = `nflavor * nsite * nelec`
  - `parton_orbparam_load_orbitals!(phi::PartonDirectPhi, data::ExpertModeData) -> PartonDirectPhi`(`pmforb_params` を写す。長さが `n_orb` と違えば error)
  - `parton_orbparam_params_from_orbitals(orbitals::Vector{Matrix{ComplexF64}}) -> Vector{ComplexF64}`(load の逆)

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "平坦添字: 全単射で、load と from_orbitals が往復する" begin
    Ns, Ne, F = 5, 2, 3
    ks = [MVMCOptimizers.parton_orbparam_flat_index(f, s, n, Ns, Ne)
          for f = 1:F, s = 1:Ns, n = 1:Ne]
    @test sort(vec(ks)) == collect(1:F * Ns * Ne)
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite, data.modpara.nelec, data.modpara.nflavor = Ns, Ne, F
    data.pmforb_params = ComplexF64.(1:F * Ns * Ne) .* (1 + 0.5im)
    phi = MVMCOptimizers.PartonDirectPhi(Ns, Ne, F)
    MVMCOptimizers.parton_orbparam_load_orbitals!(phi, data)
    @test phi.orbitals[2][3, 1] ==
          data.pmforb_params[MVMCOptimizers.parton_orbparam_flat_index(2, 3, 1, Ns, Ne)]
    @test MVMCOptimizers.parton_orbparam_params_from_orbitals(phi.orbitals) == data.pmforb_params
end

@testset "Φ-0b: 同じ軌道なら振幅系が α とビット一致" begin
    data = dimerized_mf_data()
    mfham = build_toy_mfham(data; n_idx = 3)
    phi = MVMCOptimizers.PartonDirectPhi(data.modpara.nsite, data.modpara.nelec,
                                         data.modpara.nflavor)
    phi.orbitals = deepcopy(mfham.orbitals)
    # 全配置で: parton_recompute_amplitude_all! 後の amp.det_a と amp.inv_a が == 、
    #           parton_amplitude_ratio!(粒子 1 を空きサイトへ)の (ratio, ip_new) が == 、
    #           parton_update_amplitude! 後の det_a が == 、
    #           PartonOptimizationState の 6 引数コンストラクタで phi 版を作り
    #           parton_local_energy が == 。typeof(state_phi) == PartonOptimizationState{PartonDirectPhi}
end
```

(2 つ目の testset の本体は上のコメントの 4 項目を `@test a == b` で書く。比較は `isapprox` ではなく `==`。)

- [ ] **Step 2: 失敗を確認する**。Expected: FAIL(`PartonDirectPhi not defined`)

- [ ] **Step 3: 実装する**

- [ ] **Step 4: 通過を確認する**。続けて `cd MVMCOptimizers.jl && $JL --project=. -e 'using Pkg; Pkg.test()'` が全緑(状態の型パラメータ化で既存テストが壊れていないこと)

---

### Task 4: O の Φ 版(契約 5)

**Files:**
- Create: `MVMCOptimizers.jl/src/parton_orbparam_o.jl`
- Modify: `MVMCOptimizers.jl/src/MVMCOptimizers.jl`(`parton_vmc_main_cal.jl` の直後に include)
- Modify: `MVMCOptimizers.jl/test/parton_test_helpers.jl`(フィクスチャ `orbparam_toy_data` を追加)
- Test: Create `MVMCOptimizers.jl/test/test_parton_orbparam_o.jl`

**Interfaces:**
- Consumes: Task 3 の `PartonDirectPhi`・`parton_orbparam_flat_index`。既存の `o_slot_re` / `o_slot_im` / `block_index` / `inv_block` / `particle_site` / `_parton_conjugate_mf_slots!`
- Produces:
  - `parton_orbparam_calculate_o!(sr_opt_o, amp, phi::PartonDirectPhi, cfg, data, qp_weight, ip::ComplexF64, n_proj::Int)`。k = 1..n_orb、p = n_proj + k について、Re スロットに O_k、Im スロットに i·O_k を書く(**共役前**)
  - `parton_fill_sr_opt_o!(sr_opt_o, amp, phi::PartonDirectPhi, cfg, data, qp_weight, ip, n_proj; conjugate=true, project_gauge=false, alpha=ComplexF64[])`:α 版と同じ前処理((1,0) と Jastrow の cnt)→ O → `conjugate` なら `_parton_conjugate_mf_slots!(sr_opt_o, n_proj, n_orb)`。`project_gauge` と `alpha` は受け取って無視する(呼び出し元が α 版と共通のため。docstring に明記)
  - テスト用 `orbparam_toy_data(; n_site=6, n_elec=2, n_flavor=2, seed=7) -> ExpertModeData`:`dimerized_mf_data` と同じ物理項(環の最近接ホップ `-1+0.3im`、最近接クーロン 0.5)・恒等 QP・同じ modpara 値に加えて `parton_orb_param = 1`。pmftrans/pmfpara は空。`pmforb_params = randn(MersenneTwister(seed), ComplexF64, n_flavor*n_site*n_elec)`

O の式(Φ_f[s,n] についての対数微分。A_qp,f[m,n] = sgn_qp(r_m)·Φ_f[map_qp(r_m), n]):

    O[f,s,n] = (1/ip) Σ_qp w_qp (Π_g det A_qp,g) Σ_m δ_{s, map_qp(r_m)} sgn_qp(r_m) (A_qp,f)⁻¹[n, m]

- (qp, f, m, n) の 4 重ループで Re スロットに加算し、最後に Im スロットを i·Re で埋める
- `Π_g det == 0` の qp は飛ばす(特異ブロックの `inv_a` は更新されていないので、0 × 不定値で NaN を作らない)
- `amp.n_stored != amp.n_flavor` なら error(Φ モードは高速路を使わない)

- [ ] **Step 1: 失敗するテストを書く**(以下の testset。いずれも `orbparam_toy_data()`、配置は `toy_config(data, [1, 4])` ほか 2 配置。FD は `δ = 1e-6` の中心差分で `log(ip₊/ip₋)/(2δ)`、比較は `isapprox(; rtol = 1e-6, atol = 1e-9)`)

```julia
@testset "Φ-1: O = ∂ln ip/∂Φ(有限差分、恒等 QP)"           # 全 k、Re 方向と Im 方向(Φ ± iδ e_k)。conjugate = false のスロットと比べる
@testset "Φ-1: O = ∂ln ip/∂Φ(有限差分、並進 QP 2 本)"      # set_shift_qp!(data, [collect(1:6), [mod1(r+1,6) for r=1:6]], [ones(Int,6), ones(Int,6)], ComplexF64[1, 0.5+0.2im])
@testset "Euler の恒等式: Σ_k Φ_k O_k = NFlavor·NElec"     # atol 1e-10(各 det は Φ_f の Ne 次斉次)
@testset "Im スロット = i × Re スロット(共役前)/ 共役後は両方とも conj"
@testset "Jastrow スロットは cnt の実数のまま、Φ は p = n_proj+1 から"  # _add_ring_jastrow!(data, [0.1, -0.2, 0.05]) と parton_make_proj_cnt!
```

- [ ] **Step 2: 失敗を確認する**。Expected: FAIL(`parton_orbparam_calculate_o! not defined`)
- [ ] **Step 3: 実装する**
- [ ] **Step 4: 通過を確認する**

---

### Task 5: SR 経路の検証(力 = 勾配、S の冗長方向)

**Files:**
- Test: Create `MVMCOptimizers.jl/test/test_parton_orbparam_force_gradient.jl`
- (失敗したら修正は `parton_orbparam_o.jl` に入れる)

**Interfaces:**
- Consumes: Task 4 の `parton_fill_sr_opt_o!`(Φ メソッド)。上流の `calculate_oo!` / `calculate_oo_store!` / `finalize_oo_store!` / `build_s_matrix_and_g_vector!`。列挙の骨格は `test_parton_force_gradient.jl` の `_enumerate_state` と `_force_vector` を写し、重みは `exp(2·_ln_pj_ref(cfg, data))·abs2(ip)`(`test_parton_jastrow.jl` と同じ)

フィクスチャ: `orbparam_toy_data(; n_site = 5, n_elec = 2, n_flavor = 2)` + `_add_ring_jastrow!(data, [0.1, -0.2])`、全 C(5,2) = 10 配置。`smat_to_para_idx` は Jastrow の Re スロットと Φ の Re・Im スロット全部。

- [ ] **Step 1: テストを書く**

```julia
@testset "Φ-2: 力 = 勾配($(store ? "store" : "非 store")経路)" for store in (false, true)
    # g[si] / (−DSROptStepDt) を ∂E/∂Re Φ_k・∂E/∂Im Φ_k・∂E/∂v_p の中心差分(δ = 1e-6)と比べる
    # isapprox(; rtol = 1e-4, atol = 1e-8)、かつ maximum(abs, g) > 1e-4
end
@testset "Φ-2: 共役シムを外すと勾配と合わない(回帰ガード)"   # conjugate = false で最大相対誤差 > 1e-2
@testset "Φ-2: 共役シムは S を変えない(Jastrow 込み)"        # conjugate true/false の S が isapprox(atol = 1e-12)
@testset "Φ-3: S の冗長方向" begin
    # data.modpara.dsr_opt_sta_del = 0.0 にして S を組む。δΦ_f = Φ_f·B_f をスロット並び(Re, Im 交互)の実ベクトル v にする
    # (a) B_f を複素乱数にとり、Im Σ_f Tr B_f = 0 となるよう B_1[1,1] の虚部を調整 → v'Sv ≤ 1e-10 · opnorm(S) · v'v
    # (b) B_1 = im·I、B_2 = 0(全体位相)→ v'Sv ≈ NElec^2(rtol 1e-8)。上流の S に Im·Im 項がないための正のペナルティ(§12.1)
    # (c) パートンゲージ δΦ_1 = Diagonal(d)·Φ_1、δΦ_2 = −Diagonal(d)·Φ_2(d は複素乱数)→ v'Sv ≤ 1e-10 · opnorm(S) · v'v
end
```

- [ ] **Step 2: 実行する**。Expected: PASS(Task 4 が正しければ最初から通る。FAIL なら原因を `parton_orbparam_o.jl` 側で直し、テストの許容値は緩めない)

---

### Task 6: InPmfOrb の読み書きと再スケール(D_AmpMax)

**Files:**
- Create: `MVMCOptimizers.jl/src/parton_orbparam_inputs.jl`
- Modify: `MVMCOptimizers.jl/src/MVMCOptimizers.jl`(`parton_unsupported_inputs.jl` の直後に include)
- Test: Create `MVMCOptimizers.jl/test/test_parton_orbparam_io.jl`

**Interfaces:**
- Produces:
  - `parton_orbparam_read_in_pmforb!(data, namelist_path) -> String`(読んだパス)。namelist に `InPmfOrb` が無ければ error(Φ モードでは必須)。書式は 5 行ヘッダ(2 行目 `NPmfOrb <n>`)+ `idx Re Im` 行。idx は 0-based の平坦添字 `((f·Ns)+s)·Ne+n`。**検査はすべて error**: 宣言件数 = 行数 = `parton_orbparam_n_orb`、idx ∈ [0, n)、重複なし、有限、フレーバーごとに `σ_min/σ_max > 1e-12`(列フルランク)。namelist の探し方は `parton_read_in_pmfocc` と同じ
  - `parton_orbparam_write_pmforb(data, path) -> String`。ヘッダは `===============================` / `NPmfOrb <n>` / `===============================` / `== idx RePhi ImPhi ==` / `===============================`、行は `parton_write_pmfpara` と同じ `"%d % .18e % .18e \n"`
  - `parton_orbparam_rescale!(data) -> Vector{Float64}`(フレーバーごとの再スケール前の max|Φ_f|)。各フレーバーで `ratio = parton_orb_amp_max / xmax_f` を掛ける。**max が D 未満でも拡大する**(上流 `parameter_init.jl:274-276` の規約)。`xmax_f == 0` なら error

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "InPmfOrb: 書いて読むとビット一致"                     # write → namelist に "InPmfOrb InPmfOrb.def" → read → == 元の pmforb_params
@testset "InPmfOrb: 壊れた入力は error" for case in
    (:count, :idx_range, :duplicate, :nan, :rank_deficient, :missing_entry)
    # :rank_deficient はフレーバー 2 の第 2 列 = 2 × 第 1 列。:missing_entry は namelist に InPmfOrb 行なし
    # @test_throws ErrorException で、メッセージに "InPmfOrb" を含む
end
@testset "再スケール: フレーバーごとに max = D、拡大もする" begin
    # data.modpara.parton_orb_amp_max = 4.0。フレーバー 1 は全要素 0.01 程度、フレーバー 2 は 10 程度
    # 返り値 == 再スケール前の各フレーバーの max
    # 再スケール後 maximum(abs, Φ_f) ≈ 4.0(rtol 1e-15)が両フレーバーで成立
end
@testset "再スケール: 全要素 0 のフレーバーは error"
@testset "再スケールで状態は変わらない" begin
    # 1 配置で ip'/ip == Π_f ratio_f^NElec(rtol 1e-12)、E_loc は isapprox(rtol 1e-12)
end
```

- [ ] **Step 2: 失敗を確認する**。Expected: FAIL(関数未定義)
- [ ] **Step 3: 実装する**
- [ ] **Step 4: 通過を確認する**

---

### Task 7: OptFlag と門番

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_orbparam_inputs.jl`
- Modify: `MVMCOptimizers.jl/src/parton_unsupported_inputs.jl`(`validate_parton_data` に kwarg `require_mf_inputs::Bool = true` を足し、pmfpara・pmftrans の必須検査だけをその条件の内側へ入れる。pmfpara 必須検査の error 文の末尾に「direct-orbital mode のつもりなら `PartonOrbParam 1` のつづりを確認。未知の modpara キーは黙って無視される」の一文を足す。`validate_parton_modpara` に `parton_orb_param ∉ {0, 1, 2}` で error。`validate_parton_inputs`(α の集約)の先頭で `parton_orb_param != 0` なら error)
- Test: `MVMCOptimizers.jl/test/test_parton_gatekeeper.jl`(末尾に追加)

**Interfaces:**
- Produces:
  - `parton_orbparam_materialize_flags!(data) -> Vector{Bool}`:長さ `2(n_proj + n_orb)`。射影ブロックは `parton_materialize_flags!` と同じ(Re は引き継ぎ、Im は強制凍結)、Φ ブロックは全部 true
  - `parton_orbparam_reject_alpha_files(namelist_path)`:namelist に `PartonMFTrans` / `PartonMFPara` / `InPmfPara` / `InPmfOcc` があれば error
  - `validate_parton_orbparam_inputs(data, ctx)`:`validate_parton_modpara`・`validate_parton_flavor_consistency`・`validate_parton_data(data; require_mf_inputs = false)`・`validate_parton_qp`・`validate_parton_parallel` を呼び、加えて次を検査する

| 条件 | 動作 |
|---|---|
| `parton_orb_param == 2` | error(Thouless 型は予約、未実装) |
| `parton_orb_param != 1` | error |
| pmfpara_terms または pmftrans_terms が空でない | error(α を混ぜない) |
| `parton_occ_mode != 0` | error |
| `dsr_opt_sta_del <= 0` | error(GL(Ne) の零空間で Cholesky が壊れる) |
| `parton_orb_amp_max` が正の有限値でない | error |
| `length(pmforb_params) != parton_orbparam_n_orb` | error |
| `length(optimization_flags) != 2(n_proj + n_orb)` | error |
| Φ のスロットが全部凍結 | error |

- [ ] **Step 1: 失敗するテストを書く**:上の表の各行で `@test_throws`。加えて、正常入力(`orbparam_toy_data()` + `parton_ensure_qp!` + `parton_orbparam_materialize_flags!`)は通ること、α の `validate_parton_inputs` が `parton_orb_param = 1` を拒否すること、`PartonOrbParam` を書き間違えた入力(pmfpara 空)の α 側 error 文に `"PartonOrbParam"` が含まれること、`parton_orbparam_reject_alpha_files` が 4 種のキーそれぞれで error になること
- [ ] **Step 2: 失敗を確認する**
- [ ] **Step 3: 実装する**
- [ ] **Step 4: 通過を確認する**。`test_parton_gatekeeper.jl` 全体が PASS(既存の α の検査が変わっていないこと)

---

### Task 8: SR ループ・ドライバ・出力

**Files:**
- Create: `MVMCOptimizers.jl/src/parton_vmc_para_opt_orbparam.jl`
- Modify: `MVMCOptimizers.jl/src/parton_run_para_opt_from_namelist.jl`(手順 2 の `ctx` 構築の直後、α 初期化より前に 1 分岐)
- Modify: `MVMCOptimizers.jl/src/MVMCOptimizers.jl`(`parton_run_para_opt_from_namelist.jl` の直前に include)
- Test: Create `MVMCOptimizers.jl/test/test_parton_orbparam_run.jl`

**Interfaces:**
- Consumes: Task 3〜7 の全関数。既存の `parton_make_sample!`・`parton_main_cal!`・`weight_average_we!`・`weight_average_sr_opt!`・`reduce_counter!`・`output_data!`・`stochastic_opt!`・`bcast_scalar`・`parton_sync_parameters!(data, ctx)`(mfham なし = bcast のみ)・`store_opt_data!`・`output_opt_data!`・`parton_write_time`・`parton_write_conv`・`write_ctimer_para_opt`・`parton_write_ctimer`・`parton_write_runinfo`・`parton_ensure_qp!`・`resolve_rnd_seed`
- Produces:
  - `parton_orbparam_build_state(data) -> PartonOptimizationState{PartonDirectPhi}`:`parton_build_optimization_state` と同じ寸法で、`n_stored = nflavor` 固定、`PartonDirectPhi` に `pmforb_params` を load 済み
  - `parton_orbparam_accept_update!(data, para_before) -> Bool`:`pack_parameters(data)` に非有限があれば `para_before` に戻して false
  - `parton_orbparam_write_orbdiag(data, step, dir; n_skipped, xmax_pre, log10_cond)`:`<CDataFileHead>_parton_orbdiag.dat` に 1 行追記。列は `step n_skipped xmax_pre_1..F log10cond_1..F`。先頭に列名のコメント行(step 0 で "w")
  - `parton_orbparam_vmc_para_opt!(pstate, data, ctx; rng, output_dir = nothing, c_timer = CTIMER_DISABLED, write_diagnostics = true) -> Int`
  - `parton_orbparam_run_from_namelist(data, namelist_path, ctx; nsteps, nsmp, output_dir, seed) -> Int`

1 step の順序(DESIGN §12.11):

1. `parton_orbparam_load_orbitals!(pstate.mfham, data)`。log10 cond(Φ_f) を `svdvals` で計算する(診断を書く run のみ)
2. `parton_make_sample!` → `parton_main_cal!`。直後に `n_skipped = nvmc_sample − round(Int, energy.wc)`(weight_average より前に読む)
3. `weight_average_we!` / `weight_average_sr_opt!` / `reduce_counter!` → `output_data!`
4. `para_before = pack_parameters(data)` → `stochastic_opt!`(SRinfo 書き出しは α と同じ kwargs)→ info を bcast し、非 0 なら return
5. `parton_sync_parameters!(data, ctx)` → `parton_orbparam_accept_update!(data, para_before)`(false なら警告)
6. `xmax_pre = parton_orbparam_rescale!(data)`
7. orbdiag と `parton_write_time` を書く → `store_opt_data!`

ループの後: `output_opt_data!`(Jastrow)、Φ を load し直して `_pmforb_opt.dat`(`_parton_para_out(data, "_pmforb_opt.dat", dir)`)、`parton_write_conv`、CalcTimer は既存 writer をそのまま使う(801/802 は 0 になる。新しいタイマー番号は作らない)。

`parton_orbparam_run_from_namelist` の順序: InPmfOrb を読む → `parton_orbparam_rescale!` → `parton_orbparam_reject_alpha_files` → `parton_orbparam_materialize_flags!` → `validate_parton_orbparam_inputs` → `parton_warn_threading_config` → nsteps・nsmp の上書きと `NSROptItrStep >= NSROptItrSmp` の検査(α ドライバと同じ文面)→ サンプリング RNG → `parton_ensure_qp!` → 状態の構築 → `_pmforb_init.dat` → runinfo(既存 writer を `n_idx = 0` で呼んだ後、`PartonOrbParam`・`PartonOrbAmpMax`・`n_orb` の 3 行を "a" で追記)→ ループ → runinfo を t_end 付きで書き直して 3 行を再追記。

- [ ] **Step 1: 失敗するテストを書く**

テスト内ヘルパ `_write_min_orbparam_input(dir) -> namelist_path`:`_write_min_parton_input` と同じ physhop・coulombinter・modpara(`PartonOrbParam 1` を追加、`NSROptItrStep 5`、`NSROptItrSmp 2`)を書き、pmftrans・pmfpara は書かない。InPmfOrb.def は `build_toy_mfham(dimerized_mf_data(); n_idx = 3).orbitals` を `parton_orbparam_params_from_orbitals` で詰めて `parton_orbparam_write_pmforb` で書く。

```julia
@testset "E2E: Φ モードが走り切り、出力がそろう"
    # status == 0。zvo_out.dat が 5 行で全 E が有限。zqp_pmforb_init.dat / zqp_pmforb_opt.dat /
    # zvo_parton_orbdiag.dat(データ 5 行)がある。zqp_pmfpara_init.dat は無い。runinfo に "PartonOrbParam 1"
@testset "E2E: init の Φ は各フレーバーで max = PartonOrbAmpMax"   # rtol 1e-15
@testset "E2E: opt の Φ を InPmfOrb にして再開できる"              # 2 本目の init が 1 本目の opt と isapprox(rtol 1e-14)
@testset "NaN の更新は捨てて直前の値に戻す"                        # parton_orbparam_accept_update! の単体テスト
@testset "スレッド並列 = 逐次(Φ 状態でビット一致)"                # parton_main_cal!(…; force_threaded = true / false) の sr_opt_oo・sr_opt_ho・energy.etot が ==
@testset "PartonOrbParam = 0 の入力は従来の経路を通る"            # _write_min_parton_input の run が zqp_pmfpara_opt.dat を出し、orbdiag を出さない
```

- [ ] **Step 2: 失敗を確認する**
- [ ] **Step 3: 実装する**
- [ ] **Step 4: 通過を確認する**。続けて `cd MVMCOptimizers.jl && $JL --project=. -e 'using Pkg; Pkg.test()'` が全緑

---

### Task 9: mode 0 のビット一致(Φ-0)

**Files:** なし(検証のみ。スクリプトはスクラッチに置く)

- [ ] **Step 1: 全スイートを回す**

Run:
```bash
cd MVMCExpertModeParsers.jl && $JL --project=. -e 'using Pkg; Pkg.test()'
cd ../MVMCOptimizers.jl && $JL --project=. -e 'using Pkg; Pkg.test()'
cd .. && $JL --project=@. test/physics/runtests.jl
```
Expected: すべて PASS(P 層は 1292 件が基準)

- [ ] **Step 2: 変更前と全出力をバイト比較する**(手順は `REPORT_nu12_stuck.md` §16-2)

```bash
git worktree add --detach <scratch>/baseline HEAD
# SFMT.jl / PfaPack.jl はサブモジュールなので baseline 側へシンボリックリンクを張る
```
入力は `playground_nozomi/projects/kapit_mueller/km/runs/L4x4_nu12_sym_k0_hm8p0_s1001/stage1_in` をスクラッチへ複製し、modpara の `NSROptItrStep 20`・`NSROptItrSmp 5` に書き換えたもの。両方の worktree で
`$JL --project=<tree>/MVMCOptimizers.jl -e 'using MVMCOptimizers; exit(parton_run_para_opt_from_namelist(ARGS[1]; output_dir = ARGS[2]))' <in>/namelist.def <out>`
を回し、`cmp` で比べる。

Expected: `zvo_out.dat`・`zvo_var.dat`・`zvo_SRinfo.dat`・`zvo_conv.dat`・`zqp_opt.dat`・`zqp_jastrow_opt.dat`・`zqp_pmfpara_{init,opt}.dat`・`zqp_pmfocc_{init,opt}.dat`・`zqp_pmfham_opt.dat`・`zqp_pmfband_opt.dat`・`zvo_parton_diag.dat` が**すべてバイト一致**。`zvo_parton_runinfo.dat` は githash・namelist パス・時刻の行だけが違う。計時ファイルは比較しない。

- [ ] **Step 3: worktree を畳む**。サブモジュールを含むので、ディレクトリを消してから `git worktree prune`

---

### Task 10: α 解 → Φ 入力の変換道具と Φ-9(厳密解の保存)

**Files:**
- Create: `tools/parton_alpha_to_orbparam.jl`
- Test: `MVMCOptimizers.jl/test/test_parton_orbparam_types.jl`(末尾に追加)

**Interfaces:**
- Consumes: `parton_read_in_pmfpara!`・`parton_read_in_pmfocc`・`PartonMFHamiltonian`・`parton_build_mf_templates!`・`parton_update_orbitals!(…; forced_occ)`・`parton_orbparam_params_from_orbitals`・`parton_orbparam_write_pmforb`
- Produces: `julia --project=MVMCOptimizers.jl tools/parton_alpha_to_orbparam.jl <alpha_stage_in> <alpha_stage_out> <new_dir>`

道具の処理:

1. `<alpha_stage_in>/namelist.def` の各ファイルパスを絶対パスにし、`InPmfPara <stage_out>/zqp_pmfpara_opt.dat` と `InPmfOcc <stage_out>/zqp_pmfocc_opt.dat` を足した一時 namelist を `<new_dir>` に書く(既存の読み手をそのまま使うため。終わったら消す)
2. パース → α と占有を読む → `forced_occ` 付きで軌道を作る → `InPmfOrb.def` を書く
3. `<new_dir>` に `pmftrans.def` と `pmfpara.def` 以外の def を複製する。namelist からは `PartonMFTrans` / `PartonMFPara` / `InPmfPara` / `InPmfOcc` の行を除き、`InPmfOrb InPmfOrb.def` を足す
4. `zqp_jastrow_opt.dat` があれば、4 行ヘッダを 5 行ヘッダに直して `InJastrow.def` にし、namelist に `InJastrow InJastrow.def` を足す(`parse_input_parameter_file` は 5 行を読み飛ばすため、そのままだと idx 0 が落ちる。DESIGN §11 の既知の不一致)
5. modpara は `PartonOrbParam 1`・`PartonOccMode 0` に置き換え、その他の行はそのまま残す。`DSROptStaDel <= 0` なら error

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "α の軌道から作った Φ 状態は α と同じ E_loc"
    # dimerized_mf_data → build_toy_mfham → params_from_orbitals → rescale → Φ 状態
    # 全配置で parton_local_energy が α 版と isapprox(rtol 1e-12)(スケールは E_loc に効かない)
```

- [ ] **Step 2: 失敗を確認する → Step 3: 道具を実装する → Step 4: 通過を確認する**

- [ ] **Step 5: Φ-9 を手で回す**。`playground_nozomi/projects/kapit_mueller/km/runs/L4x4_nu12_sym_k0_hm8p0_s1001/stage1_{in,out}` をスクラッチへ変換し、`NSROptItrStep 50`・`NSROptItrSmp 10` で 1 プロセス実行。
Expected: `zvo_out.dat` の全行で |E + 4| < 1e-8、分散(4 列目)< 1e-12。orbdiag の log10 cond が step を通じて一定

---

### Task 11: 試走(Φ-10)と「致命的な問題」の判定

**Files:**
- Create: `playground_nozomi/projects/kapit_mueller/km/REPORT_orbparam.md`(git 管理外)
- Modify: `.claude/CLAUDE.md`(現在の状況に結果を 1 節)

- [ ] **Step 1: コストを実測する**。`playground_nozomi/projects/kapit_mueller/km/runs/L9x4_nu13_indep_k0_hm8p0_s1001/stage1_{in,out}`(α の最良 −3.93989)を Task 10 の道具で変換し、`NSROptItrStep 20` で 1 プロセス実行。`zvo_parton_time.dat` から 1 step の秒数を読み、本走の step 数と並列数を決める(推測で決めない)
- [ ] **Step 2: 本走**(Step 1 で決めた条件。並列は MPI.jl 付属の mpiexec、BLAS は 1 スレッド)
- [ ] **Step 3: §12.11 の目安で判定し、REPORT に書く**:
  - E が α の −3.93989 以下に来るか(ED −3.941102316835)
  - log10 cond(Φ_f) が単調に増えるか
  - n_skipped が増え続けるか、SR が NaN で止まるか
  - max|Φ_f|(再スケール前)の推移
- [ ] **Step 4: CLAUDE.md の現在の状況を更新する**。致命的な問題があれば §12 本体の正準化(Löwdin)へ進むかどうかを AskUserQuestion でユーザーに諮る
