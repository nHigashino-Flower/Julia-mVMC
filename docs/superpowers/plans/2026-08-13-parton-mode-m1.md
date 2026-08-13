# パートン平均場 VMC モード M1 統合 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `DESIGN_parton.md`(唯一の正)に従い、パートン平均場 VMC モード(`PartonMode = 1`)の統合・整合修正・テスト(§8 の 0〜6)を完成させ、M1 完了条件を満たす。

**Architecture:** 新規コードは `parton_*` ファイルに閉じ、既存ファイルは DESIGN §3.1 の登録点のみをマーカー付きで編集する。SR 配線は案 B(パラメータロケータ拡張、RBM 前例)。契約 0〜5 は納品済みコードの整合修正+欠け埋めで完成させる。

**Tech Stack:** Julia 1.11.9(`~/.local/opt/julia-1.11.9/bin/julia`)、LinearAlgebra(eigen/lu)、Test。パッケージ追加なし(ED はテスト内で素朴に全数展開)。

## Global Constraints(絶対規則 — 違反したらその変更は破棄)

1. **PartonMode=0 のビット互換**: 既存テストが T0 ベースライン(scratchpad/T0_baseline.md: パーサ全緑・オプティマイザ 793/793・integration 全 13 テストセット緑)と同一結果であり続けること
2. 既存ファイルの編集は登録点のみ。マーカー `# --- parton-mode (fork addition) ---` を**一字一句同一**で付ける(現状の `# Parton-mode (fork addition)` 等の揺れは全て正規化する)
3. 構造体フィールド追加は末尾のみ(位置引数コンストラクタのズレ防止)
4. 振幅の双線形縮約(契約 2/3/5 の最内)に `dot()` を使わない(転置積)。契約 0′ の `Uu' * dHUo` の随伴は正しい — この 2 つを「統一」しない(DESIGN §7)
5. O 格納で `imag スロット = val * im` の近道(vmc_main_cal.jl:2547-48)をコピーしない。独立計算した 2 値を (2p+1, 2p+2) に格納
6. MF パラメータに D_AmpMax リスケールを適用しない
7. 受理時は①配置コミット→②振幅更新の順(DESIGN §4)。逆順への「整理」禁止
8. 0-based→1-based 変換は `parton_build_mf_templates!` の一箇所のみ
9. フラット配列+手動ストライドの家風を抽象化しない(ストライドはアクセサ 2 関数 `block_index`/`inv_block` のみ)
10. 新規関数は `parton_` 接頭辞。呼ぶだけの既存関数をラップ・改名しない
11. テストが赤いとき期待値の側を書き換えて緑にしない(期待値は DESIGN §1 の固定式)。式と実装が矛盾し実装誤りを排除しても解消しない場合は停止して人間に質問
12. コミットは parton-mode ブランチにタスク単位で行う(プロジェクト指示による明示的許可)。`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` を付ける

**テスト実行環境**(全タスク共通):

```bash
export PATH=~/.local/opt/julia-1.11.9/bin:$PATH
cd /home/nozomihigashino/Julia-mVMC/.claude/worktrees/parton-mode
# パーサ層:      (cd MVMCExpertModeParsers.jl && julia --project=@. -e 'using Pkg; Pkg.test()')
# オプティマイザ層: (cd MVMCOptimizers.jl && julia --project=@. -e 'using Pkg; Pkg.test()')
# integration:   julia --project=@. test/integration/runtests.jl
```

## 調査で確定した実名(T1 照合結果)

| DESIGN 上の名前 | 実名 | 場所 |
|---|---|---|
| NVMCWarmUp | `modpara.nvmc_warmup` | expert_types.jl:81 |
| NVMCInterval / NVMCSample | `nvmc_interval` / `nvmc_sample` | expert_types.jl:82-83 |
| NBlockUpdateSize 相当 | **存在しない → 新設** `nblock_update_size`(デフォルト 16) | 本計画 Task 1 |
| NSROptItrStep / Smp | `nsr_opt_itr_step` / `nsr_opt_itr_smp` | expert_types.jl:78-79 |
| 2Sz / ComplexType / NCond / NLocSpin | `two_sz` / `complex_flag` / `ncond` / `nlocspin` | expert_types.jl |
| NVMCCalMode / NSRCG / NLanczosMode | `vmc_calc_mode` / `nsrcg` / `lanczos_mode` | expert_types.jl |
| NSPGaussLeg / NSPStot / NMPTrans | `nsp_gauss_leg` / `nsp_stot` / `nmp_trans` | expert_types.jl |
| NOrbitalIdx / NNeuron / NExUpdatePath / NSplitSize | `n_orbital_idx` / `nneuron` / `nex_update_path` / `nsplit_size` | expert_types.jl |
| QP 写像 | `data.qp_trans[qp]::Vector{Int}` / `data.qp_trans_sgn[qp]` (1-based site→site) | expert_types.jl:786-788 |
| QP 重み | `data.qp_weights.qp_full_weight::Vector{ComplexF64}` | qp_weight.jl:191 |
| n_qp | `get_n_qp_full(data)` | vmc_para_opt.jl:412 |
| O スロット | `sr_opt_o[1]=1, [2]=0`; パラメータ p → (2p+1, 2p+2) | vmc_main_cal.jl:3193 |
| OO/HO 蓄積 | `calculate_oo!` / `calculate_oo_store!`(sr_opt_o の中身+w,e にのみ依存 — §10 確認済) | vmc_main_cal.jl:3392,3499 |
| サンプル重み | `w = 1.0`(重点サンプリング) | vmc_main_cal.jl:4018 |
| flags 実体化 | `data.optimization_flags::Vector{Bool}`(1 パラメータ 2 スロット) | read_input_parameters.jl:405 |
| ロケータ | `_foreach_parameter_location`(f(para_idx, loc))/ `_at`(f(loc))/ `_parameter_location_value` / `_set_parameter_location_value!` | stochastic_opt.jl:78,130,196,217 |
| pack/unpack/sync | `pack_parameters` / `unpack_parameters!` / `sync_modified_parameter!` | parameter_sync.jl |

**DESIGN §10 未決事項の解決(§11 へ追記する内容)**:
- `nblock_update_size` は Julia 移植に存在しない → ModParaParameters 末尾へ新設(キー `NBlockUpdateSize`、デフォルト 16)
- pmfpara.def の value 列は家風(trans.def / physhop.def)に合わせ **Re Im の 2 トークン**。データ行 = `site1 flavor1 site2 flavor2 idx Re Im`(7 列)、末尾フラグ行 = `idx flag`(2 列)
- pmftrans.def のデータ行 = `site1 flavor1 site2 flavor2 Re Im`(6 列)
- 登録点リスト(§3.1)に modpara_parser.jl の elseif を追加(§2.1 の PartonMode/NFlavor キーの実装に必然)
- validation.jl への編集は登録点外 → **revert**(検査は門番 parton_unsupported_inputs.jl が家)
- `calculate_oo!`/`calculate_oo_store!` が sr_opt_o の中身にのみ依存することを確認済(消し込み)
- store_opt_data!/output_data! はロケータ経由(pack_parameters)で値を取るため MF も自動記録(消し込み)

---

### Task 1: modpara 登録点(PartonMode / NFlavor / NElec 別名 / NBlockUpdateSize)

**Files:**
- Modify: `MVMCExpertModeParsers.jl/src/types/expert_types.jl:136,183-185,228-230`(ModParaParameters)
- Modify: `MVMCExpertModeParsers.jl/src/utils/constants.jl:165-166`
- Modify: `MVMCExpertModeParsers.jl/src/parsers/modpara_parser.jl:198-203`
- Test: `MVMCExpertModeParsers.jl/test/test_parton_parsers.jl`(新規、runtests.jl へ include 追加)

**Interfaces:**
- Produces: `modpara.parton_mode::Int`(0)、`modpara.nflavor::Int`(0)、`modpara.nblock_update_size::Int`(16)。modpara.def キー `PartonMode` / `NFlavor` / `NParticle`・`NPartonPerFlavor`(nelec 別名、食い違いはパースエラー)/ `NBlockUpdateSize`

- [ ] **Step 1: 失敗するテストを書く**

`MVMCExpertModeParsers.jl/test/test_parton_parsers.jl` を新規作成:

```julia
# パートンモード用パーサのテスト --- parton-mode (fork addition) ---
using Test
using MVMCExpertModeParsers

@testset "modpara parton keys" begin
    content = """
    --------------------
    Model_Parameters   0
    --------------------
    VMC_Cal_Parameters
    --------------------
    CDataFileHead  zvo
    CParaFileHead  zqp
    --------------------
    NVMCCalMode    0
    NDataIdxStart  1
    NDataQtySmp    1
    Nsite          4
    NElec          2
    PartonMode     1
    NFlavor        3
    NBlockUpdateSize 8
    """
    result = MVMCExpertModeParsers.parse_modpara_content(content)
    @test result.success
    p = result.data
    @test p.parton_mode == 1
    @test p.nflavor == 3
    @test p.nblock_update_size == 8
    @test p.nelec == 2
end

@testset "modpara parton defaults" begin
    p = MVMCExpertModeParsers.ModParaParameters()
    @test p.parton_mode == 0
    @test p.nflavor == 0
    @test p.nblock_update_size == 16
end

@testset "NParticle / NPartonPerFlavor は NElec の別名" begin
    base = "Nsite 4\n"
    r1 = MVMCExpertModeParsers.parse_modpara_content(base * "NParticle 3\n")
    @test r1.success && r1.data.nelec == 3
    r2 = MVMCExpertModeParsers.parse_modpara_content(base * "NElec 3\nNPartonPerFlavor 3\n")
    @test r2.success && r2.data.nelec == 3
    # 食い違いはパースエラー
    r3 = MVMCExpertModeParsers.parse_modpara_content(base * "NElec 3\nNParticle 2\n")
    @test !r3.success
end
```

注: `parse_modpara_content` の実シグネチャ(引数・返り値)は modpara_parser.jl 冒頭を確認し、既存テスト(test_parsers.jl)の呼び方に合わせて調整する。runtests.jl の include 一覧末尾に `include("test_parton_parsers.jl")` を追加。

- [ ] **Step 2: 落ちることを確認**

Run: `cd MVMCExpertModeParsers.jl && julia --project=@. -e 'using Pkg; Pkg.test()'`
Expected: FAIL(現状は expert_types.jl:184-185 がカンマ欠落の構文エラーでプリコンパイル自体が失敗する)

- [ ] **Step 3: 実装**

`expert_types.jl` — 3 箇所(全てマーカー付き):

(a) フィールド宣言末尾(L136 `n_orbital_idx::Int` の直後):

```julia
    # --- parton-mode (fork addition) ---
    parton_mode::Int          # PartonMode: 0=既存 mVMC / 1=パートン平均場 VMC
    nflavor::Int              # NFlavor: フレーバー数(PartonMode=1 で必須)
    nblock_update_size::Int   # NBlockUpdateSize: 受理 N 回ごとに厳密再計算の錨
```

(b) kwargs(L182 `n_orbital_idx::Int = 0,` の直後。現在の壊れた 2 行を置換):

```julia
        # --- parton-mode (fork addition) ---
        parton_mode::Int = DEFAULT_PARTON_MODE,
        nflavor::Int = DEFAULT_NFLAVOR,
        nblock_update_size::Int = DEFAULT_NBLOCK_UPDATE_SIZE,
```

(c) `new(...)` 末尾(L227 `n_orbital_idx,` の直後。現在の `parton_vmc_calc_mode, nflavor,` を置換):

```julia
            # --- parton-mode (fork addition) ---
            parton_mode,
            nflavor,
            nblock_update_size,
```

`constants.jl`(L165-166 の 2 行を置換、マーカー正規化):

```julia
# --- parton-mode (fork addition) ---
const DEFAULT_PARTON_MODE = 0
const DEFAULT_NFLAVOR = 0
const DEFAULT_NBLOCK_UPDATE_SIZE = 16
```

`modpara_parser.jl`(L198-203 の現在のブロックを置換):

```julia
    # --- parton-mode (fork addition) ---
    elseif name == "PartonMode"
        params.parton_mode = safe_parse_int(value, DEFAULT_PARTON_MODE)
    elseif name == "NFlavor"
        params.nflavor = safe_parse_int(value, DEFAULT_NFLAVOR)
    elseif name == "NBlockUpdateSize"
        params.nblock_update_size = safe_parse_int(value, DEFAULT_NBLOCK_UPDATE_SIZE)
    elseif name == "NParticle" || name == "NPartonPerFlavor"
        v = safe_parse_int(value, 0)
        if params.nelec != 0 && params.nelec != v
            push!(context.errors,
                  "$name = $v conflicts with previously set particle number $(params.nelec) " *
                  "(NElec / NParticle / NPartonPerFlavor all write the same field)")
        else
            params.nelec = v
        end
```

注: `parse_modpara_parameter!` が `context` を受けているか確認(受けていなければ errors の積み先は既存の警告機構に合わせ、「食い違い」はエラーにできる経路を選ぶ)。`nelec` を後から `NElec` が上書きして食い違いを隠さないか、既存の `NElec` 分岐にも同じ一致検証が要るかを確認し、必要なら `NElec` 分岐はそのまま(標準経路のビット互換を優先)とし別名側でのみ検証する。

- [ ] **Step 4: テストが通ることを確認**

Run: `cd MVMCExpertModeParsers.jl && julia --project=@. -e 'using Pkg; Pkg.test()'`
Expected: PASS(既存テスト含め全緑)

- [ ] **Step 5: コミット**

```bash
git add MVMCExpertModeParsers.jl
git commit -m "feat(parton): modpara に PartonMode/NFlavor/NBlockUpdateSize 登録点を追加"
```

---

### Task 2: Term 構造体群と ExpertModeData 登録点+validation.jl の revert

**Files:**
- Modify: `MVMCExpertModeParsers.jl/src/types/expert_types.jl:706-740,831-835,895-899`
- Modify: `MVMCExpertModeParsers.jl/src/utils/validation.jl`(**revert** — 登録点外)
- Test: `MVMCExpertModeParsers.jl/test/test_parton_parsers.jl` に追記

**Interfaces:**
- Produces: `PartonMFTransTerm`(immutable: site1, flavor1, site2, flavor2, value::ComplexF64, is_complex::Bool — 0-based のまま保持)、`PartonMFParaTerm`(mutable: site1, flavor1, site2, flavor2, idx, value::ComplexF64, is_complex::Bool)、`PhysHopTerm`(immutable: site1, site2, value::ComplexF64, is_complex::Bool)
- Produces: `data.pmftrans_terms::Vector{PartonMFTransTerm}`、`data.pmfpara_terms::Vector{PartonMFParaTerm}`、`data.physhop_terms::Vector{PhysHopTerm}`、`data.pmfpara_opt_flags::Dict{Int,Int}`(0-based idx → flag。ドライバが flags 実体化に使う)
- **Removes**: `pmfpara_idx_matrix`(DESIGN §2.3 で廃止)

- [ ] **Step 1: 失敗するテストを書く**(test_parton_parsers.jl に追記)

```julia
@testset "parton Term structs / ExpertModeData fields" begin
    data = MVMCExpertModeParsers.ExpertModeData()
    @test data.pmftrans_terms isa Vector{MVMCExpertModeParsers.PartonMFTransTerm}
    @test data.pmfpara_terms isa Vector{MVMCExpertModeParsers.PartonMFParaTerm}
    @test data.physhop_terms isa Vector{MVMCExpertModeParsers.PhysHopTerm}
    @test data.pmfpara_opt_flags isa Dict{Int,Int}
    @test isempty(data.pmftrans_terms) && isempty(data.pmfpara_terms) && isempty(data.physhop_terms)
    @test !hasfield(MVMCExpertModeParsers.ExpertModeData, :pmfpara_idx_matrix)
    @test isbitstype(MVMCExpertModeParsers.PartonMFTransTerm)
end
```

- [ ] **Step 2: 落ちることを確認**(physhop_terms / pmfpara_opt_flags 不在、pmfpara_idx_matrix 存在で FAIL)

- [ ] **Step 3: 実装**

expert_types.jl L706-740 の構造体群を整理(PhysHopTerm は既にある。PartonMFTransTerm はフィールド順を `site1, flavor1, site2, flavor2, value, is_complex` に統一 — 入力列順と同じ)。ExpertModeData 末尾フィールド(L831-835)を置換:

```julia
    # --- parton-mode (fork addition) ---
    pmftrans_terms::Vector{PartonMFTransTerm}
    pmfpara_terms::Vector{PartonMFParaTerm}
    physhop_terms::Vector{PhysHopTerm}
    pmfpara_opt_flags::Dict{Int,Int}   # 0-based idx → OptFlag(pmfpara.def 末尾フラグ行)
```

コンストラクタ `new(...)` 末尾(L895-899)を置換:

```julia
            # --- parton-mode (fork addition) ---
            PartonMFTransTerm[],
            PartonMFParaTerm[],
            PhysHopTerm[],
            Dict{Int,Int}(),
```

validation.jl を revert: `git checkout -- MVMCExpertModeParsers.jl/src/utils/validation.jl`(検査は Task 6 の門番が家。DESIGN の「validation.jl は本番未接続のため頼らない」に従う)

- [ ] **Step 4: テストが通ることを確認**(パーサパッケージ全緑)

- [ ] **Step 5: コミット** `feat(parton): Term 構造体群と ExpertModeData 登録点を確定`

---

### Task 3: パーサ 3 本の整合(pmftrans 6 列化・pmfpara 7 列化・physhop 登録)

**Files:**
- Modify: `MVMCExpertModeParsers.jl/src/parsers/pmftrans_parser.jl`(全面改修: 6 列)
- Modify: `MVMCExpertModeParsers.jl/src/parsers/pmfpara_parser.jl`(7 列+関数名整合)
- Rename: `MVMCExpertModeParsers.jl/src/parsers/phyhop_parser.jl` → `physhop_parser.jl`(末尾の登録スニペットコメント 4 点は各所へ反映後に削除)
- Modify: `MVMCExpertModeParsers.jl/src/MVMCExpertModeParsers.jl:52-53,769-786`(include 3 行+parse_file_by_type! elseif 3 種)
- Modify: `MVMCExpertModeParsers.jl/src/utils/constants.jl:148-149`(キーワード表: PhysHop 追加+マーカー正規化)
- Test: `test_parton_parsers.jl` に追記

**Interfaces:**
- Produces: `parse_parton_mf_trans_def(path) -> ParseResult{Vector{PartonMFTransTerm}}`(行 = `site1 flavor1 site2 flavor2 Re Im`)
- Produces: `parse_parton_mf_para_def(path) -> Tuple{ParseResult{Vector{PartonMFParaTerm}}, Dict{Int,Int}, Int}`(データ行 = `site1 flavor1 site2 flavor2 idx Re Im`、フラグ行 = `idx flag`、ヘッダ `NPartonMFParaIdx`)
- Produces: `parse_physhop_def(path) -> ParseResult{Vector{PhysHopTerm}}`(納品済み・変更なし)
- parse_file_by_type! が `PartonMFTrans` / `PartonMFPara` / `PhysHop` を処理し、pmfpara は `data.pmfpara_terms` と `data.pmfpara_opt_flags` の両方へ格納。idx の 0-based 連番性(`Set(idx) == 0:max`)と declared_count 一致は**門番(Task 6)と build(契約 0)が検査**するのでパーサでは検査しない(家風: 忠実な読み手)

- [ ] **Step 1: 失敗するテストを書く**(test_parton_parsers.jl に追記)

```julia
@testset "pmftrans parser (6 列)" begin
    content = """
    ====================
    NPartonMFTrans 3
    ====================
    == site1 flavor1 site2 flavor2 Re Im ==
    ====================
    0 0 1 0  -1.0  0.5
    1 1 2 1  -1.0  0.0
    0 0 0 0   0.3  0.0
    """
    r = MVMCExpertModeParsers.parse_parton_mf_trans_content(content)
    @test r.success
    ts = r.data
    @test length(ts) == 3
    @test ts[1].site1 == 0 && ts[1].flavor1 == 0 && ts[1].site2 == 1 && ts[1].flavor2 == 0
    @test ts[1].value == ComplexF64(-1.0, 0.5) && ts[1].is_complex
    @test ts[3].site1 == ts[3].site2 == 0 && !ts[3].is_complex
end

@testset "pmfpara parser (7 列+フラグ行)" begin
    content = """
    =============================================
    NPartonMFParaIdx  2
    ComplexType       1
    =============================================
    =============================================
    0 0 1 0  0  -1.0  0.0
    1 1 2 1  1  -1.0  0.0
    0 0 0 0  0  -1.0  0.0
    0 1
    1 0
    """
    result, flags, declared = MVMCExpertModeParsers.parse_parton_mf_para_content(content)
    @test result.success
    ts = result.data
    @test length(ts) == 3
    @test ts[1].idx == 0 && ts[1].value == ComplexF64(-1.0, 0.0)
    @test ts[3].idx == 0            # idx 共有(value 一致は門番が検証)
    @test flags == Dict(0 => 1, 1 => 0)
    @test declared == 2
end

@testset "physhop parser" begin
    content = """
    ==================
    NPhysHop 2
    ==================
    == site1 site2 Re Im ==
    ==================
    0 1  -1.0  0.2
    1 2  -1.0  0.0
    """
    r = MVMCExpertModeParsers.parse_physhop_content(content)
    @test r.success
    @test length(r.data) == 2
    @test r.data[1].value == ComplexF64(-1.0, 0.2)
    # 宣言数不一致はエラー
    bad = replace(content, "NPhysHop 2" => "NPhysHop 3")
    @test !MVMCExpertModeParsers.parse_physhop_content(bad).success
end

@testset "parse_file_by_type! 経由(namelist 統合)" begin
    mktempdir() do dir
        write(joinpath(dir, "pmftrans.def"), "0 0 1 0 -1.0 0.0\n")
        write(joinpath(dir, "pmfpara.def"),
              "====\nNPartonMFParaIdx 1\nComplexType 1\n====\n====\n0 0 1 0 0 -1.0 0.0\n0 1\n")
        write(joinpath(dir, "physhop.def"),
              "====\nNPhysHop 1\n====\n====\n====\n0 1 -1.0 0.0\n")
        write(joinpath(dir, "modpara.def"), "Nsite 4\nNElec 2\nPartonMode 1\nNFlavor 2\n")
        write(joinpath(dir, "namelist.def"), """
            ModPara        modpara.def
            PartonMFTrans  pmftrans.def
            PartonMFPara   pmfpara.def
            PhysHop        physhop.def
            """)
        data = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(dir, "namelist.def"))
        @test length(data.pmftrans_terms) == 1
        @test length(data.pmfpara_terms) == 1
        @test length(data.physhop_terms) == 1
        @test data.pmfpara_opt_flags == Dict(0 => 1)
        @test data.modpara.parton_mode == 1
    end
end
```

- [ ] **Step 2: 落ちることを確認**(pmftrans は 6 列非対応・構造体引数不一致、pmfpara は関数名不一致等で FAIL)

- [ ] **Step 3: 実装**

pmftrans_parser.jl 改修の要点: `parse_parton_mf_trans_term` を 6 トークン読みに(`site1 flavor1 site2 flavor2 Re Im`、flavor も 0-based)、返り値型注釈の `Vector{TransferTerm}` を `Vector{PartonMFTransTerm}` に修正、構築を `PartonMFTransTerm(site1, flavor1, site2, flavor2, ComplexF64(re, imv), imv != 0.0)` に。非数値開始行は物理ホップパーサと同じ規約で静かにスキップ(`tryparse(Int, tokens[1]) === nothing → continue`)。

pmfpara_parser.jl 改修の要点: 公開名を `parse_parton_mf_para_def` / `parse_parton_mf_para_content` に統一(登録側の呼称と一致)、`PatonMFParaTerm` 型注釈 typo と `terms = OrbitalTerm[]` を修正、データ行を 7 トークン(`... idx Re Im`)で読み `value = ComplexF64(re, imv)`、5〜6 トークンの legacy 分岐は削除(DESIGN に legacy なし)、2 トークン行はフラグ行。

physhop: `git mv phyhop_parser.jl physhop_parser.jl`。末尾コメントの登録スニペット①〜③を反映(①は Task 2 で済、②constants.jl キーワード表に `"PhysHop" => "physhop.def",`+マーカー、③parse_file_by_type! に PhysHop elseif)、④は Task 6 の門番へ。反映後、末尾スニペットコメント(L115-166)を削除。

入口ファイル MVMCExpertModeParsers.jl: include 3 行(マーカー正規化):

```julia
# --- parton-mode (fork addition) ---
include("parsers/pmftrans_parser.jl")
include("parsers/pmfpara_parser.jl")
include("parsers/physhop_parser.jl")
```

parse_file_by_type! の elseif(既存 2 種を修正+PhysHop 追加。「ongoing...」コメント削除):

```julia
    # --- parton-mode (fork addition) ---
    elseif file_type == "PartonMFTrans"
        result = parse_parton_mf_trans_def(file_path)
        if result.success
            data.pmftrans_terms = result.data
        else
            error("Failed to parse PartonMFTrans file '$file_path': $(result.error_message)")
        end
    elseif file_type == "PartonMFPara"
        result, opt_flags, _declared = parse_parton_mf_para_def(file_path)
        if result.success
            data.pmfpara_terms = result.data
            data.pmfpara_opt_flags = opt_flags
        else
            error("Failed to parse PartonMFPara file '$file_path': $(result.error_message)")
        end
    elseif file_type == "PhysHop"
        result = parse_physhop_def(file_path)
        if result.success
            data.physhop_terms = result.data
        else
            error("Failed to parse PhysHop file '$file_path': $(result.error_message)")
        end
```

構造体・パーサで公開が要る名前(PartonMFTransTerm 等)が export されているか確認し、既存 Term 群の export 方式に合わせる(export されていなければ完全修飾でテストする — 既存の家風を変えない)。

- [ ] **Step 4: テストが通ることを確認**(パーサパッケージ全緑)

- [ ] **Step 5: コミット** `feat(parton): pmftrans/pmfpara/physhop パーサと登録点を DESIGN §2 に整合`

---

### Task 4: count_variational_parameters と パラメータロケータ(案 B)

**Files:**
- Modify: `MVMCExpertModeParsers.jl/src/utils/read_input_parameters.jl:162-185`
- Modify: `MVMCOptimizers.jl/src/stochastic_opt.jl:78-151`(bulk 版末尾)、`:130-196`(_at 版末尾 — 現在の誤配置ブロックを置換)、`:217-238`(set 末尾)
- Test: `MVMCOptimizers.jl/test/test_parton_locator.jl`(新規、runtests.jl へ include)

**Interfaces:**
- Consumes: `data.pmfpara_terms`(Task 2/3)
- Produces: `count_variational_parameters(data)` が `n_pmf = max(idx)+1` を**実際に**加算(現在は `return` 後の死文)。ロケータ 4 関数が `_PARAM_PMF` を ORBITAL の完全な鏡写しで扱う(共有 idx = 全行訪問・絶対 set 冪等・δ は各行の自値+δ)。標準モード(pmfpara_terms 空)では全登録点で挙動不変

- [ ] **Step 1: 失敗するテストを書く**

`MVMCOptimizers.jl/test/test_parton_locator.jl`:

```julia
# パラメータロケータ PMF ブロックのテスト --- parton-mode (fork addition) ---
using Test
using MVMCExpertModeParsers
using MVMCOptimizers

function _toy_pmf_data(; n_idx = 2)
    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara.nsite = 4
    data.modpara.nelec = 2
    data.modpara.nflavor = 2
    data.modpara.parton_mode = 1
    # idx 0 を 2 行で共有(フレーバー跨ぎ)、idx 1 は 1 行
    push!(data.pmfpara_terms,
          MVMCExpertModeParsers.PartonMFParaTerm(0, 0, 1, 0, 0, ComplexF64(-1.0, 0.0), true),
          MVMCExpertModeParsers.PartonMFParaTerm(0, 1, 1, 1, 0, ComplexF64(-1.0, 0.0), true),
          MVMCExpertModeParsers.PartonMFParaTerm(0, 0, 0, 0, 1, ComplexF64(0.3, 0.0), true))
    return data
end

@testset "count_variational_parameters が n_pmf を数える" begin
    data = _toy_pmf_data()
    @test MVMCExpertModeParsers.count_variational_parameters(data) == 2   # 射影 0 + MF 2
end

@testset "pack/unpack roundtrip(共有 idx の冪等性)" begin
    data = _toy_pmf_data()
    para = MVMCOptimizers.pack_parameters(data)
    @test length(para) == 2
    @test para[1] == ComplexF64(-1.0, 0.0)
    @test para[2] == ComplexF64(0.3, 0.0)
    para2 = copy(para); para2[1] = ComplexF64(2.0, -0.5)
    MVMCOptimizers.unpack_parameters!(data, para2)
    @test data.pmfpara_terms[1].value == ComplexF64(2.0, -0.5)
    @test data.pmfpara_terms[2].value == ComplexF64(2.0, -0.5)   # 共有 idx の全行が更新
    @test data.pmfpara_terms[3].value == ComplexF64(0.3, 0.0)
end

@testset "get/set/delta(_at 版)" begin
    data = _toy_pmf_data()
    @test MVMCOptimizers.get_parameter_value(data, 1) == ComplexF64(-1.0, 0.0)
    MVMCOptimizers.set_parameter_value!(data, 2, ComplexF64(9.0, 1.0))
    @test data.pmfpara_terms[3].value == ComplexF64(9.0, 1.0)
    MVMCOptimizers._add_parameter_delta_direct!(data, 1, ComplexF64(0.5, 0.5))
    @test data.pmfpara_terms[1].value == ComplexF64(-0.5, 0.5)
    @test data.pmfpara_terms[2].value == ComplexF64(-0.5, 0.5)
end

@testset "標準モードでは挙動不変" begin
    data = MVMCExpertModeParsers.ExpertModeData()
    @test MVMCExpertModeParsers.count_variational_parameters(data) == 0
    @test MVMCOptimizers.pack_parameters(data) == ComplexF64[]
end
```

- [ ] **Step 2: 落ちることを確認**(count は死文で 0、pack は PMF 行を訪問しない)

- [ ] **Step 3: 実装**

read_input_parameters.jl(L162-185 を修正。`+n_pmf` の死文を式の中へ):

```julia
function count_variational_parameters(data::ExpertModeData)::Int
    n_rbm = count_rbm_parameters(data)
    # --- parton-mode (fork addition) ---
    n_pmf = isempty(data.pmfpara_terms) ? 0 : maximum(t.idx for t in data.pmfpara_terms) + 1
    return projection_layout(data).n_proj +
           n_rbm +
           count_orbital_parameters(data) +
           count_opt_trans_parameters(data) +
           n_pmf   # --- parton-mode (fork addition) ---
end
```

stochastic_opt.jl:

(a) `_foreach_parameter_location`(bulk 版)の `return nothing` 直前(L150 付近)に追加:

```julia
    # --- parton-mode (fork addition) ---
    pmf_start = counts.n_proj + counts.n_rbm + counts.n_orbital_idx + counts.n_opt_trans + 1
    for term_idx in eachindex(data.pmfpara_terms)
        para_idx = pmf_start + data.pmfpara_terms[term_idx].idx     # idx は 0-based
        pmf_start <= para_idx <= n_para &&
            f(para_idx, _ParameterLocation(_PARAM_PMF, 0, term_idx))
    end
```

(b) `_foreach_parameter_location_at` の現在の誤配置ブロック(L184-192)を削除し、OPTTRANS の elseif 連鎖の後(`return nothing` 直前)に置換:

```julia
    # --- parton-mode (fork addition) ---
    pmf_start = counts.n_proj + counts.n_rbm + counts.n_orbital_idx + counts.n_opt_trans + 1
    if pmf_start <= para_idx <= counts.n_para
        pmf_idx = para_idx - pmf_start                              # 0-based
        for term_idx in eachindex(data.pmfpara_terms)
            data.pmfpara_terms[term_idx].idx == pmf_idx &&
                f(_ParameterLocation(_PARAM_PMF, 0, term_idx))     # 共有 idx は全行訪問
        end
    end
```

注: `_at` 版の `f` は `loc` のみを受ける(bulk 版と違う)。既存の OPTTRANS ブロックが `elseif` 連鎖なので、PMF 判定は連鎖の外の独立 if(PMF は最後のブロックで、上の elseif がどれも成立しない para_idx 域)。ORBITAL/OPTTRANS の elseif 連鎖と範囲が重ならないことを目視確認する。

(c) `_set_parameter_location_value!` の OPTTRANS elseif の後(L234)に追加:

```julia
    elseif loc.kind == _PARAM_PMF   # --- parton-mode (fork addition) ---
        data.pmfpara_terms[loc.item_idx].value = value
```

(d) `_parameter_location_value` の PMF elseif は実装済み(L212-213)— マーカーが正規形であることを確認のみ。

- [ ] **Step 4: テストが通ることを確認**(オプティマイザパッケージ全緑 793+新規)

- [ ] **Step 5: 回帰確認**: integration テスト実行 → T0 と同一の全緑

- [ ] **Step 6: コミット** `feat(parton): SR パラメータロケータ案 B(_PARAM_PMF)を完成`

---

### Task 5: parton_types.jl の書き直し(型+アクセサ+委譲)

**Files:**
- Rewrite: `MVMCOptimizers.jl/src/parton_types.jl`(全面)
- Modify: `MVMCOptimizers.jl/src/MVMCOptimizers.jl:80-89`(include 群 — マーカー正規化のみ。順序は現状どおり既存 include の後)
- Test: `MVMCOptimizers.jl/test/test_parton_types.jl`(新規)

**Interfaces:**
- Produces(DESIGN §5 のカタログどおり):
  - アクセサ: `n_parton_per_flavor` / `n_parton_total` / `n_phys_particle` / `n_site_flavor`(Int 版・ModParaParameters 版・ExpertModeData 版)
  - `PartonMFTemplateEntry`(immutable isbits: `site1::Int, site2::Int, flavor::Int, coeff::ComplexF64` — 1-based)
  - `PartonMFHamiltonian`(mutable: `n_idx::Int, template::Vector{Vector{PartonMFTemplateEntry}}, is_onsite_group::Vector{Bool}, h_mf::Vector{Matrix{ComplexF64}}, eig_vals::Vector{Vector{Float64}}, eig_vecs::Vector{Matrix{ComplexF64}}, orbitals::Vector{Matrix{ComplexF64}}, dorbitals::Vector{Vector{Matrix{ComplexF64}}}, dh_uo_scratch::Matrix{ComplexF64}, min_gap::Float64` + コンストラクタ `PartonMFHamiltonian(n_site, n_elec, n_flavor, n_idx)`)
  - `PartonConfiguration`(mutable: `ele_idx::Vector{Int}`(F·Ne)、`ele_cfg::Vector{Int}`(F·Nsite、粒子番号 or -1)、`ele_num::Vector{Int}`(F·Nsite、0/1)、`burn_ele_idx::Vector{Int}`、`burn_flag::Bool`、`counter::Vector{Int}`(10)、`n_site::Int, n_elec::Int, n_flavor::Int` + コンストラクタ)— 既存 ElectronConfiguration は 2 フレーバー寸法焼き付けのため自前(DESIGN §5)。M1 は測定を parton_main_cal! がサンプル保存済み配置から直接行うため、per-sample 保存は `stored_ele_idx::Vector{Int}`(n_sample·F·Ne)を持つ
  - `PartonAmplitudeData`(mutable: `inv_a::Vector{ComplexF64}`(n_qp·F·Ne²)、`det_a::Vector{ComplexF64}`(n_qp·F)、`n_qp::Int, n_flavor::Int, n_elec::Int` + コンストラクタ)
  - ストライドアクセサ(この 2 つだけ、他の場所にストライド計算を書かない):
    `block_index(amp, qp, f) = (qp - 1) * amp.n_flavor + f`、
    `inv_block(amp, qp, f) = reshape(view(amp.inv_a, ((block_index(amp,qp,f)-1)*amp.n_elec^2 + 1):(block_index(amp,qp,f)*amp.n_elec^2)), amp.n_elec, amp.n_elec)`
  - `PartonSamplingWorkspace`(mutable: `a_scratch::Matrix{ComplexF64}`(Ne×Ne)、`ratio_blocks::Vector{ComplexF64}`(n_qp·F)、`u_buf::Vector{ComplexF64}`、`v_buf::Vector{ComplexF64}`、`col_buf::Vector{ComplexF64}`(各 Ne)+ コンストラクタ)
  - `PartonOptimizationState`(mutable: `state::VMCOptimizationState, amp::PartonAmplitudeData, config::PartonConfiguration, workspace::PartonSamplingWorkspace, mfham::PartonMFHamiltonian`)+ 委譲 1 行メソッド(`weight_average_we!` / `weight_average_sr_opt!` / `stochastic_opt!` / `output_data!` / `store_opt_data!` / `reduce_counter!` — すべて `st.state` へ委譲。**呼ぶだけの既存関数はラップも改名もしない**規則に従い、多重ディスパッチによる同名メソッド追加のみ)
  - 配置アクセサ: `particle_site(cfg, f, m) = cfg.ele_idx[(f-1)*cfg.n_elec + m] `(1-based サイト)、`is_occupied(cfg, r) = cfg.ele_num[r] != 0`(フレーバー1 で代表)、`move_particle!(cfg, f, m, r_old, r_new)`(ele_idx/ele_cfg/ele_num の 3 点更新)、`assert_flavors_locked(cfg)`(全フレーバーの占有サイト集合が一致することを検査)
- 削除: 納品版にあった `VMCOptimizationState` の重複定義(types.jl の既存を使う)

- [ ] **Step 1: 失敗するテストを書く**

`MVMCOptimizers.jl/test/test_parton_types.jl`:

```julia
# parton 型とアクセサのテスト --- parton-mode (fork addition) ---
using Test
using MVMCOptimizers
using MVMCExpertModeParsers

@testset "パートン数アクセサ" begin
    @test MVMCOptimizers.n_parton_total(2, 3) == 6
    @test MVMCOptimizers.n_site_flavor(4, 3) == 12
    mp = MVMCExpertModeParsers.ModParaParameters(; nsite = 4, nelec = 2)
    mp.nflavor = 3
    @test MVMCOptimizers.n_parton_total(mp) == 6
end

@testset "PartonAmplitudeData ストライド" begin
    amp = MVMCOptimizers.PartonAmplitudeData(2, 3, 2)   # n_qp=2, n_flavor=3, n_elec=2
    @test length(amp.det_a) == 6
    @test length(amp.inv_a) == 6 * 4
    @test MVMCOptimizers.block_index(amp, 1, 1) == 1
    @test MVMCOptimizers.block_index(amp, 2, 3) == 6
    B = MVMCOptimizers.inv_block(amp, 2, 1)
    @test size(B) == (2, 2)
    B[1, 2] = 7.0 + 0im                       # view であること(コピーでない)
    @test amp.inv_a[(4 - 1) * 4 + 3] == 7.0 + 0im   # block 4, 列優先 (1,2)→3 番目
end

@testset "PartonConfiguration と固縛" begin
    cfg = MVMCOptimizers.PartonConfiguration(4, 2, 3, 1)  # n_site=4, n_elec=2, n_flavor=3, n_sample=1
    # 粒子 1→サイト1, 粒子 2→サイト3 を全フレーバーへ
    for f in 1:3
        MVMCOptimizers.move_particle!(cfg, f, 1, 0, 1)   # r_old=0 は「未配置から」を表す規約にするか、
        MVMCOptimizers.move_particle!(cfg, f, 2, 0, 3)   # 初期化専用関数にするかは実装時に決めて統一
    end
    @test MVMCOptimizers.particle_site(cfg, 2, 1) == 1
    @test MVMCOptimizers.is_occupied(cfg, 3)
    @test !MVMCOptimizers.is_occupied(cfg, 2)
    @test MVMCOptimizers.assert_flavors_locked(cfg) === nothing
    @test cfg.burn_flag == false
end

@testset "PartonMFHamiltonian 確保" begin
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 3)  # n_site, n_elec, n_flavor, n_idx
    @test length(mfham.h_mf) == 2 && size(mfham.h_mf[1]) == (4, 4)
    @test size(mfham.orbitals[1]) == (4, 2)
    @test length(mfham.dorbitals[1]) == 6            # 2 * n_idx dof
    @test size(mfham.dh_uo_scratch) == (4, 2)
end
```

注: `move_particle!` の未配置初期化の扱い(r_old=0 規約 vs 専用 `place_particle!`)は実装時にどちらかへ統一し、テストを実装に合わせて**構造は変えず**表記だけ直してよい(期待する不変条件 — 3 配列の整合と固縛 — は変えない)。

- [ ] **Step 2: 落ちることを確認**(現状の parton_types.jl は構文エラー(L30)で precompile 失敗)

- [ ] **Step 3: 実装**(上記 Interfaces の仕様どおり全面書き直し。ele_cfg は「サイト→粒子番号 or -1」、ele_num は「サイト→0/1」の C 家風を踏襲。counter[1]=試行、counter[2]=受理。stored_ele_idx はサンプル s の粒子配置を `stored_ele_idx[(s-1)*F*Ne + (f-1)*Ne + m]` で保存)

- [ ] **Step 4: テストが通ることを確認**(オプティマイザパッケージ全緑 — 他の parton ファイルがまだ壊れて precompile を止める場合は、include 順を保ったまま**未完成ファイルの中身を一時的に最小化せず**、このタスクの間だけ MVMCOptimizers.jl の include を parton_types.jl+parton_unsupported_inputs.jl までに絞ってよい(絞った状態をコミットしない。Task 7-9 で全て復帰する)

- [ ] **Step 5: コミット** `feat(parton): parton_types.jl を DESIGN §5 カタログに書き直し`

---

### Task 6: 門番(parton_unsupported_inputs.jl)の実名化+追加検査

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_unsupported_inputs.jl`
- Test: `MVMCOptimizers.jl/test/test_parton_gatekeeper.jl`(新規)

**Interfaces:**
- Consumes: Task 1-3 のフィールド実名(`parton_mode`, `nflavor`, `nelec`, `two_sz`, `ncond`, `nlocspin`, `vmc_calc_mode`, `complex_flag`, `nsrcg`, `lanczos_mode`, `nsp_gauss_leg`, `nsp_stot`, `n_orbital_idx`, `nneuron`, `nex_update_path`, `nsplit_size`)
- Produces: `validate_parton_inputs(data, ctx)`(集約)。修正点:
  - L91 `modpara.partonv_vmc_calc_mode` → `modpara.parton_mode`
  - L66 docstring「NFlavor must be > 2」→「> 0」(コードは正しい)
  - L131 `nex_update_path` 未修飾+`end` 欠落の構文エラー修正: `if modpara.nex_update_path != 6 ... end`
  - CoulombIntra 拒否を追加(`isempty(data.coulomb_intra_terms)` — 実フィールド名は expert_types.jl で確認して合わせる)
  - validate_parton_data に physhop 検査(納品スニペット④: 存在・範囲・site1≠site2・逆向き重複)と pmftrans の逆向き重複+オンサイト t 実数+双方向完全性の**入力レベル**検査(結合検証の本体は契約 0 の build が家 — 門番は「読めているべきものが読めているか」と行単位の不正のみ)
  - flags 長検査: `length(data.optimization_flags) == 2 * (projection_layout(data).n_proj + n_pmf)` と「MF 成分に flag=1 が最低 1 つ」(DESIGN §2.5。呼び出し時点でドライバが実体化済み前提)
  - `validate_parton_flavor_consistency` は現状維持(flavor1==flavor2 要求)
- 全域のマーカーを正規形に統一

- [ ] **Step 1: 失敗するテストを書く**

```julia
# 門番のテスト --- parton-mode (fork addition) ---
using Test
using MVMCExpertModeParsers
using MVMCOptimizers

function _parton_ok_data()
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite = 4; mp.nelec = 2; mp.parton_mode = 1; mp.nflavor = 2
    mp.two_sz = 0; mp.complex_flag = 1; mp.nex_update_path = 6
    push!(data.pmftrans_terms,
          MVMCExpertModeParsers.PartonMFTransTerm(0, 0, 1, 0, ComplexF64(-1, 0), false),
          MVMCExpertModeParsers.PartonMFTransTerm(0, 1, 1, 1, ComplexF64(-1, 0), false))
    push!(data.pmfpara_terms,
          MVMCExpertModeParsers.PartonMFParaTerm(0, 0, 1, 0, 0, ComplexF64(-1, 0), true),
          MVMCExpertModeParsers.PartonMFParaTerm(0, 1, 1, 1, 0, ComplexF64(-1, 0), true))
    push!(data.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(-1, 0), false))
    data.optimization_flags = fill(true, 2 * 1)   # n_proj=0, n_pmf=1
    return data
end

@testset "門番: 正常系は通る" begin
    data = _parton_ok_data()
    ctx = MVMCOptimizers.serial_context()
    @test MVMCOptimizers.validate_parton_inputs(data, ctx) === nothing
end

@testset "門番: 各違反を個別に検出" begin
    ctx = MVMCOptimizers.serial_context()
    for mutate! in (
        d -> d.modpara.parton_mode = 2,                    # 予約値
        d -> d.modpara.nflavor = 0,                        # NFlavor 欠落
        d -> d.modpara.two_sz = -1,                        # FSZ の罠
        d -> d.modpara.complex_flag = 0,
        d -> d.modpara.nex_update_path = 1,
        d -> d.modpara.nelec = 5,                          # NElec > NSite
        d -> empty!(d.pmfpara_terms),
        d -> empty!(d.physhop_terms),
        d -> push!(d.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(1, 0, ComplexF64(1, 0), false)),  # 逆向き重複
        d -> push!(d.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(2, 2, ComplexF64(1, 0), false)),  # 対角
        d -> (d.pmfpara_terms[1] = MVMCExpertModeParsers.PartonMFParaTerm(0, 0, 1, 1, 0, ComplexF64(-1, 0), true)),  # flavor 混成
        d -> d.optimization_flags = fill(true, 1),         # flags 長不足
        d -> d.optimization_flags = fill(false, 2),        # 全凍結
    )
        data = _parton_ok_data()
        mutate!(data)
        @test_throws Exception MVMCOptimizers.validate_parton_inputs(data, ctx)
    end
end
```

注: PartonMFParaTerm が mutable なので要素置換でなくフィールド代入でもよい。CoulombIntra 拒否のテストは実フィールド名確認後に 1 ケース追加する。

- [ ] **Step 2: 落ちることを確認**(現状は構文エラー+実名不一致)

- [ ] **Step 3: 実装**(Interfaces の修正点を全て適用)

- [ ] **Step 4: テストが通ることを確認**

- [ ] **Step 5: コミット** `feat(parton): 門番を実名に整合し flags 長・physhop・混成検査を追加`

---

### Task 7: 契約 0/0′(parton_orbital.jl)— §8 テストの前提

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_orbital.jl`(納品コードの修正+contract5 前半の移設)
- Modify: `MVMCOptimizers.jl/src/parton_contract5.jl` → 前半(L1-75)を parton_orbital.jl へ移し、ファイル自体は Task 9 完了後に削除
- Test: `MVMCOptimizers.jl/test/test_parton_orbital.jl`(新規)

**Interfaces:**
- Consumes: `PartonMFHamiltonian`, `PartonMFTemplateEntry`(Task 5)、`data.pmftrans_terms` / `data.pmfpara_terms`
- Produces:
  - `parton_build_mf_templates!(mfham, data)`(起動時 1 回。0→1based はここだけ。納品コードのバグ修正: L35 `t.flavor+1` → `t.flavor1+1`。idx 連番性検査を明示: `sort(unique(values(idx_of))) == 0:n_idx-1` でなければ error。同一 idx 内の value 一致検証(DESIGN §2.3)と「各フレーバーに項が 1 つもない」検査も追加)
  - `parton_alpha_from_terms(data) -> Vector{ComplexF64}`(**新規**: idx ごとに value を詰める。長さ n_idx)

    ```julia
    function parton_alpha_from_terms(data::ExpertModeData)
        n_idx = isempty(data.pmfpara_terms) ? 0 :
                maximum(t.idx for t in data.pmfpara_terms) + 1
        α = zeros(ComplexF64, n_idx)
        for t in data.pmfpara_terms
            α[t.idx + 1] = t.value     # 共有 idx は一致検証済み(build)なので上書きで同値
        end
        return α
    end
    ```
  - `parton_update_orbitals!(mfham, alpha, n_elec; gap_tol=1e-8)`(納品どおり。h.c. はホッピングのみ、Hermitian eigen、min_gap)
  - `parton_update_orbital_derivatives!(mfham, n_elec)`(contract5 L30-75 をそのまま移設 — `Uu' * dHUo` の随伴は正しい。変更しない)

- [ ] **Step 1: 失敗するテストを書く**

```julia
# 契約0/0′ のテスト --- parton-mode (fork addition) ---
using Test, LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

# 4 サイト鎖・F=2・複素位相つき t の共通 fixture
function _toy_mf_data()
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara
    mp.nsite = 4; mp.nelec = 2; mp.nflavor = 2; mp.parton_mode = 1
    t = ComplexF64(-1.0, 0.4)
    for f in 0:1, i in 0:3
        j = mod(i + 1, 4)
        push!(data.pmftrans_terms,
              MVMCExpertModeParsers.PartonMFTransTerm(i, f, j, f, t, true))
        push!(data.pmfpara_terms,
              MVMCExpertModeParsers.PartonMFParaTerm(i, f, j, f, 0, ComplexF64(1.0, 0.0), true))
    end
    # オンサイト群 idx=1(全サイト・全フレーバー共有)
    for f in 0:1, i in 0:3
        push!(data.pmftrans_terms,
              MVMCExpertModeParsers.PartonMFTransTerm(i, f, i, f, ComplexF64(0.7, 0.0), false))
        push!(data.pmfpara_terms,
              MVMCExpertModeParsers.PartonMFParaTerm(i, f, i, f, 1, ComplexF64(1.0, 0.0), true))
    end
    return data
end

@testset "契約0: build+update の基本性質" begin
    data = _toy_mf_data()
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    @test mfham.n_idx == 2
    @test mfham.is_onsite_group == [false, true]
    α = MVMCOptimizers.parton_alpha_from_terms(data)
    @test α == [ComplexF64(1, 0), ComplexF64(1, 0)]
    MVMCOptimizers.parton_update_orbitals!(mfham, α, 2)
    for f in 1:2
        H = mfham.h_mf[f]
        @test H ≈ H'                            # エルミート性(h.c. 暗黙付与の検証)
        @test H[1, 2] ≈ ComplexF64(-1.0, 0.4)   # α=1 なので t がそのまま
        @test real(H[1, 1]) ≈ 0.7 && imag(H[1, 1]) ≈ 0 atol = 1e-14
        @test mfham.orbitals[f] ≈ eigen(Hermitian(H)).vectors[:, 1:2]
    end
    @test mfham.min_gap > 0
end

@testset "契約0: 入力違反の検出" begin
    # 逆向き重複
    data = _toy_mf_data()
    push!(data.pmftrans_terms,
          MVMCExpertModeParsers.PartonMFTransTerm(1, 0, 0, 0, ComplexF64(-1.0, -0.4), true))
    push!(data.pmfpara_terms,
          MVMCExpertModeParsers.PartonMFParaTerm(1, 0, 0, 0, 0, ComplexF64(1.0, 0.0), true))
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    # idx 欠番(連番でない)
    data2 = _toy_mf_data()
    for t in data2.pmfpara_terms
        t.idx == 1 && (t.idx = 5)
    end
    mfham2 = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 6)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham2, data2)
    # オンサイト t が複素
    data3 = _toy_mf_data()
    data3.pmftrans_terms[9] = MVMCExpertModeParsers.PartonMFTransTerm(0, 0, 0, 0, ComplexF64(0.7, 0.1), true)
    mfham3 = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    @test_throws Exception MVMCOptimizers.parton_build_mf_templates!(mfham3, data3)
end

@testset "契約0′: 摂動論 ∂Φ の内部整合(オンサイト Im=0)" begin
    data = _toy_mf_data()
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(mfham, MVMCOptimizers.parton_alpha_from_terms(data), 2)
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, 2)
    @test all(iszero, mfham.dorbitals[1][2 * 2])      # idx=2(オンサイト)の Im dof はゼロ
    @test any(!iszero, mfham.dorbitals[1][1])         # ホッピング Re dof は非ゼロ
end
```

(契約 0′ の数値正しさの本検証は §8 テスト 4 の有限差分 — Task 11)

- [ ] **Step 2: 落ちることを確認**

- [ ] **Step 3: 実装**(Interfaces のとおり。移設後、contract5 前半(L1-75)は parton_contract5.jl から削除)

- [ ] **Step 4: テストが通ることを確認**

- [ ] **Step 5: コミット** `feat(parton): 契約0/0′(テンプレ構築・対角化・摂動論 ∂Φ)を完成`

---

### Task 8: §8 テスト 1 — 契約 1(gather+det の全数展開一致・錨の冪等性)

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_calculate_m_all.jl`(納品コードの整合確認: `lu!` は `LinearAlgebra.lu!`、`issuccess`、`det(F)` は LU から。壊れていれば修正)
- Create: `MVMCOptimizers.jl/src/parton_qp.jl` は**作らない** — 恒等 QP の実体化は driver/テスト側で `data.qp_trans = [collect(1:n_site)]`, `data.qp_trans_sgn = [ones(Int, n_site)]`, `data.qp_weights` 相当の重み `[1.0+0im]` を渡す(契約シグネチャは qp_weight ベクタを引数で受けるため ExpertModeData 改変不要)
- Test: `MVMCOptimizers.jl/test/test_parton_contract1.jl`(新規)

**Interfaces:**
- Consumes: Task 5 の型・アクセサ、Task 7 の契約 0
- Produces: `parton_recompute_amplitude_all!(amp, mfham, config, data, ws)` / `gather_a_block!(dest, Φ, config, f, qmap, qsgn)` / `parton_calculate_ip(amp, qp_weight)` が動作(納品コードほぼそのまま。`data.qp_trans[qp]` を読むため、テスト・ドライバは恒等写像を必ず実体化してから呼ぶ)

- [ ] **Step 1: 失敗するテストを書く**

```julia
# §8 テスト1: 契約1 vs 全数展開 --- parton-mode (fork addition) ---
using Test, LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

# _toy_mf_data() は test_parton_orbital.jl と同じ fixture を共通ヘルパへ
# (test/parton_test_helpers.jl を作り両方から include する)

@testset "契約1: det の直接構成一致(複素 t・恒等 QP)" begin
    data = _toy_mf_data()
    data.qp_trans = [collect(1:4)]
    data.qp_trans_sgn = [ones(Int, 4)]
    qp_weight = [ComplexF64(1, 0)]
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(mfham, MVMCOptimizers.parton_alpha_from_terms(data), 2)

    cfg = MVMCOptimizers.PartonConfiguration(4, 2, 2, 1)
    # 粒子1→サイト1、粒子2→サイト3(全フレーバー固縛)
    for f in 1:2
        MVMCOptimizers.move_particle!(cfg, f, 1, 0, 1)
        MVMCOptimizers.move_particle!(cfg, f, 2, 0, 3)
    end
    amp = MVMCOptimizers.PartonAmplitudeData(1, 2, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 1 * 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    # 全数展開: A[m,n] = Φ[r_m, n] を素朴に組んで det
    for f in 1:2
        A = [mfham.orbitals[f][r, n] for r in (1, 3), n in 1:2]
        b = MVMCOptimizers.block_index(amp, 1, f)
        @test amp.det_a[b] ≈ det(A) rtol = 1e-13
        @test MVMCOptimizers.inv_block(amp, 1, f) * A ≈ I atol = 1e-12
    end
    ip1 = MVMCOptimizers.parton_calculate_ip(amp, qp_weight)
    @test ip1 ≈ qp_weight[1] * prod(amp.det_a) rtol = 1e-13

    # 錨の冪等性: もう一度呼んでも同一
    det_before = copy(amp.det_a); inv_before = copy(amp.inv_a)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    @test amp.det_a == det_before
    @test amp.inv_a == inv_before
end

@testset "契約1: 6 サイト・F=3・非自明 qp_trans_sgn" begin
    # 6 サイト鎖 F=3、qp_trans = 1 サイト巡回シフト+sgn 交互で gather の写像・符号を検証
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara; mp.nsite = 6; mp.nelec = 2; mp.nflavor = 3; mp.parton_mode = 1
    t = ComplexF64(-1.0, 0.3)
    for f in 0:2, i in 0:5
        push!(data.pmftrans_terms, MVMCExpertModeParsers.PartonMFTransTerm(i, f, mod(i+1,6), f, t, true))
        push!(data.pmfpara_terms,  MVMCExpertModeParsers.PartonMFParaTerm(i, f, mod(i+1,6), f, 0, ComplexF64(1,0), true))
    end
    shift = [collect(2:6); 1]
    data.qp_trans = [collect(1:6), shift]
    data.qp_trans_sgn = [ones(Int, 6), [iseven(r) ? 1 : -1 for r in 1:6]]
    qp_weight = ComplexF64[0.5, 0.5]
    mfham = MVMCOptimizers.PartonMFHamiltonian(6, 2, 3, 1)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(mfham, MVMCOptimizers.parton_alpha_from_terms(data), 2)
    cfg = MVMCOptimizers.PartonConfiguration(6, 2, 3, 1)
    for f in 1:3
        MVMCOptimizers.move_particle!(cfg, f, 1, 0, 2)
        MVMCOptimizers.move_particle!(cfg, f, 2, 0, 5)
    end
    amp = MVMCOptimizers.PartonAmplitudeData(2, 3, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 2 * 3)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    for qp in 1:2, f in 1:3
        qmap = data.qp_trans[qp]; qsgn = data.qp_trans_sgn[qp]
        A = [qsgn[r] * mfham.orbitals[f][qmap[r], n] for r in (2, 5), n in 1:2]
        @test amp.det_a[MVMCOptimizers.block_index(amp, qp, f)] ≈ det(A) rtol = 1e-13
    end
end
```

- [ ] **Step 2: 落ちることを確認** → **Step 3: 最小修正**(納品コードのバグがあればここで直る)→ **Step 4: 全緑確認** → **Step 5: コミット** `test(parton): §8-1 契約1 全数展開一致・錨の冪等性`

---

### Task 9: §8 テスト 2 — 契約 2/3+骨格の欠け埋め(高速更新 vs 厳密再計算)

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_vmc_sampling.jl`(納品コード整合+欠け埋め)
- Delete: `MVMCOptimizers.jl/src/parton_contract5.jl`(前半は Task 7、後半は Task 11 で移設完了後)
- Test: `MVMCOptimizers.jl/test/test_parton_contract23.jl`(新規)

**Interfaces:**
- Consumes: Task 5/7/8 の全て
- Produces(欠けている関数を新規実装):
  - `parton_make_initial_sample!(cfg, amp, mfham, data, ws, rng)`: 非ゼロ振幅になるまで固縛配置をランダム生成(全 det ≠ 0 まで最大 100 回引き直し。C の MakeInitialSample の責務)
  - `parton_copy_to_burn_sample!(cfg)` / `parton_copy_from_burn_sample!(cfg)`: `burn_ele_idx` との copyto!(復元時は ele_cfg/ele_num を ele_idx から再構成)
  - `parton_log_proj_ratio(cfg, m, r_old, r_new) = 0.0`(M1: 射影なしの恒等 0。M2 フック)
  - `parton_store_sample!(cfg, amp, qp_weight, s)`: `stored_ele_idx` へ配置を保存(振幅はサンプル毎再計算が測定側の分担 — DESIGN §4)
  - `parton_make_sample!(pstate, data, rng)` 本体の整合(納品コードの `pstate.parton_amp_data` 等のフィールド名を Task 5 の実名 `amp/config/workspace/mfham` に合わせる。`data.qp_weights` の実体は `data.qp_weights.qp_full_weight` を渡す)
  - 契約 2 `parton_amplitude_ratio!` / 契約 3 `parton_update_amplitude!` は納品どおり(転置積・ratio_floor・:need_recompute)

- [ ] **Step 1: 失敗するテストを書く**

```julia
# §8 テスト2: 契約3 vs 契約1(機械精度)+骨格 --- parton-mode (fork addition) ---
using Test, LinearAlgebra, Random
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "契約2/3: 多数回高速更新後の厳密再計算一致(複素 t)" begin
    data = _toy_mf_data()             # 4 サイト・F=2・複素 t(共通ヘルパ)
    data.qp_trans = [collect(1:4)]; data.qp_trans_sgn = [ones(Int, 4)]
    qp_weight = [ComplexF64(1, 0)]
    mfham = MVMCOptimizers.PartonMFHamiltonian(4, 2, 2, 2)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(mfham, MVMCOptimizers.parton_alpha_from_terms(data), 2)
    cfg = MVMCOptimizers.PartonConfiguration(4, 2, 2, 1)
    for f in 1:2
        MVMCOptimizers.move_particle!(cfg, f, 1, 0, 1)
        MVMCOptimizers.move_particle!(cfg, f, 2, 0, 3)
    end
    amp = MVMCOptimizers.PartonAmplitudeData(1, 2, 2)
    ws = MVMCOptimizers.PartonSamplingWorkspace(2, 2)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    rng = MersenneTwister(42)
    n_updates = 0
    while n_updates < 200
        m = rand(rng, 1:2)
        r_old = MVMCOptimizers.particle_site(cfg, 1, m)
        r_new = rand(rng, 1:4)
        (r_new == r_old || MVMCOptimizers.is_occupied(cfg, r_new)) && continue
        ratio, ip_new = MVMCOptimizers.parton_amplitude_ratio!(ws, amp, mfham, data, qp_weight, m, r_new)
        # 契約2 は純粋: 棄却しても amp は不変(ここでは全提案を受理)
        for f in 1:2
            MVMCOptimizers.move_particle!(cfg, f, m, r_old, r_new)
        end
        st = MVMCOptimizers.parton_update_amplitude!(amp, mfham, data, ws, m, r_new)
        st === :need_recompute && MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
        # 比の検証: 更新後 ip == 契約2 が予告した ip_new
        @test MVMCOptimizers.parton_calculate_ip(amp, qp_weight) ≈ ip_new rtol = 1e-10
        n_updates += 1
    end
    # 200 回の乗法更新後、厳密再計算と機械精度一致
    det_fast = copy(amp.det_a); inv_fast = copy(amp.inv_a)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)
    @test maximum(abs.(det_fast .- amp.det_a)) < 1e-9 * maximum(abs.(amp.det_a))
    @test maximum(abs.(inv_fast .- amp.inv_a)) < 1e-9
    # デバッグ恒等式 v[m] == R(DESIGN §7)は契約2 実装内の @assert でなくここで検証:
    # ratio_blocks[b] は直前の契約2 の R — 受理後の det 比と一致している
end

@testset "骨格: parton_make_sample! が走り固縛を保つ" begin
    data = _toy_mf_data()
    mp = data.modpara
    mp.nvmc_warmup = 5; mp.nvmc_interval = 1; mp.nvmc_sample = 8
    mp.nblock_update_size = 4
    data.qp_trans = [collect(1:4)]; data.qp_trans_sgn = [ones(Int, 4)]
    # qp_weights: 実体は init_qp_weight!(data) が作るが、恒等 QP のテストでは直接
    # qp_full_weight を持つ最小構造を渡す(実装時に parton_make_sample! が読む形に合わせる)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)   # Task 12 で作るビルダを先行利用
    rng = MersenneTwister(7)
    MVMCOptimizers.parton_make_sample!(pstate, data, rng)
    cfg = pstate.config
    @test cfg.burn_flag == true
    @test MVMCOptimizers.assert_flavors_locked(cfg) === nothing
    @test cfg.counter[1] > 0 && cfg.counter[2] > 0        # 試行・受理が数えられている
    # 保存されたサンプルが全て正しい固縛配置(粒子数 Ne、重複なし)
    for s in 1:mp.nvmc_sample
        sites = [pstate.config.stored_ele_idx[(s-1)*2*2 + m] for m in 1:2]
        @test allunique(sites) && all(1 .<= sites .<= 4)
    end
end
```

注: `parton_build_optimization_state` が Task 12 の場合は依存が逆転するので、このタスクでは骨格テストを `PartonOptimizationState` を手組みして行う(ビルダ関数は Task 12 で導入し、このテストを差し替える)。実装順の都合はテストの意図(骨格が走る・固縛不変・カウンタ)を変えない範囲で調整可。

- [ ] **Step 2: 落ちることを確認** → **Step 3: 実装** → **Step 4: 全緑** → **Step 5: コミット** `feat(parton): 契約2/3+サンプリング骨格を完成(§8-2 緑)`

---

### Task 10: §8 テスト 3 — QP: 写像 gather 版 vs per-QP 軌道実体化版

**Files:**
- Test: `MVMCOptimizers.jl/test/test_parton_qp.jl`(新規。実装変更なし — 検証のみ)

**Interfaces:**
- Consumes: Task 8 の契約 1(qp_trans 対応済み)

- [ ] **Step 1: テストを書く**

```julia
# §8 テスト3: gather 写像 vs 実体化 per-QP 軌道 --- parton-mode (fork addition) ---
using Test, LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "QP: gather 版 = 実体化版(全ブロック)" begin
    # 6 サイト・F=2・並進 3 種(恒等・+1 シフト・+2 シフト)+交互符号
    data = MVMCExpertModeParsers.ExpertModeData()
    mp = data.modpara; mp.nsite = 6; mp.nelec = 3; mp.nflavor = 2; mp.parton_mode = 1
    t = ComplexF64(-1.0, 0.25)
    for f in 0:1, i in 0:5
        push!(data.pmftrans_terms, MVMCExpertModeParsers.PartonMFTransTerm(i, f, mod(i+1,6), f, t, true))
        push!(data.pmfpara_terms,  MVMCExpertModeParsers.PartonMFParaTerm(i, f, mod(i+1,6), f, 0, ComplexF64(1,0), true))
    end
    shifts = [collect(1:6), [collect(2:6); 1], [collect(3:6); 1; 2]]
    sgns = [ones(Int, 6), [(-1)^r for r in 1:6], ones(Int, 6)]
    data.qp_trans = shifts; data.qp_trans_sgn = sgns
    qp_weight = ComplexF64[0.5, 0.25, 0.25]

    mfham = MVMCOptimizers.PartonMFHamiltonian(6, 3, 2, 1)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(mfham, MVMCOptimizers.parton_alpha_from_terms(data), 3)
    cfg = MVMCOptimizers.PartonConfiguration(6, 3, 2, 1)
    for f in 1:2, (m, r) in enumerate((1, 4, 6))
        MVMCOptimizers.move_particle!(cfg, f, m, 0, r)
    end
    amp = MVMCOptimizers.PartonAmplitudeData(3, 2, 3)
    ws = MVMCOptimizers.PartonSamplingWorkspace(3, 6)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data, ws)

    # 実体化版: Φ_qp[r, n] = sgn[r] * Φ[map[r], n] を先に作ってから det
    for qp in 1:3, f in 1:2
        Φqp = [sgns[qp][r] * mfham.orbitals[f][shifts[qp][r], n] for r in 1:6, n in 1:3]
        A = Φqp[[1, 4, 6], :]
        @test amp.det_a[MVMCOptimizers.block_index(amp, qp, f)] ≈ det(A) rtol = 1e-13
    end
end
```

- [ ] **Step 2: 全緑確認**(実装済みなら即緑のはず。赤なら gather の写像方向のバグ — DESIGN §1.1 の `qp_trans 順方向` 規約と突き合わせて実装側を直す)

- [ ] **Step 3: コミット** `test(parton): §8-3 QP gather vs 実体化の全ブロック一致`

---

### Task 11: §8 テスト 4 — 契約 4/5(parton_vmc_main_cal.jl)+有限差分

**Files:**
- Create: `MVMCOptimizers.jl/src/parton_vmc_main_cal.jl`(現在 0 行)
- Modify: `parton_contract5.jl` 後半(L88-145)を移設 → その後 parton_contract5.jl を削除
- Test: `MVMCOptimizers.jl/test/test_parton_contract45.jl`(新規)

**Interfaces:**
- Consumes: 契約 0-3、`calculate_oo!`/`calculate_oo_store!`(既存)、`set_projection_diff!` 相当のスロット規約
- Produces:
  - `o_slot_re(p) = 2p + 1` / `o_slot_im(p) = 2p + 2`(移設)
  - `parton_calculate_o!(sr_opt_o, amp, mfham, cfg, data, qp_weight, ip, n_proj)`(移設。`particle_site` は Task 5 のアクセサに一致済みか確認)
  - `parton_diag_energy(cfg, data) -> Float64`(**新規**: `Σ_{(i,j,V)} V n_i n_j` を coulombinter 項から。n はフレーバー 1 の物理粒子占有 = 固縛下の n^b。実フィールド名は expert_types.jl の CoulombInterTerm を確認して合わせる)
  - `parton_main_cal!(pstate, data)`(**新規**: サンプルループ。各サンプル s について:
    1. `stored_ele_idx` から cfg.ele_idx/ele_cfg/ele_num を復元
    2. 契約 1 で錨(サンプル毎再計算 — DESIGN §4)→ `ip = parton_calculate_ip(...)`
    3. E_loc: 対角=`parton_diag_energy` + 合成ホップ: 各 physhop 項 (i,j,t) について **t 側・t* 側の両方向**を評価 — 占有 i・空き j なら契約 2 の仮想呼び出しで `t * (射影比=1) * ip(x_{i→j})/ip(x)` を加算(移動せず ratio だけ読む。ws.ratio_blocks を汚すが直後に別項で上書きされるため契約 2 が純粋なら安全)
    4. `sr_opt_o[1]=1, [2]=0`; 射影 O は M1 では n_proj=0 なのでスキップ; `parton_calculate_o!` で MF ブロック
    5. `w = 1.0`; `accumulate_energy!` 相当は `pstate.state.energy` へ直接(`energy.wc += w; energy.etot += w*e; energy.etot2 += w*conj(e)*e` — 既存 EnergyData のフィールドへ。既存 accumulate_energy! は VMCEnergyAccumulator 用なので使えない。スレッド化はしない)
    6. `calculate_oo_store!`(nstore≠0)or `calculate_oo!` へ委譲(sr_opt_size = pstate.state.sr_opt.sr_opt_size)
  - `parton_ln_ip(data, cfg_sites; ...) 相当のテスト用ヘルパは作らない** — テストは契約 0→1→ip を直接連ねて ln ip(θ) を評価する

- [ ] **Step 1: 失敗するテストを書く**

```julia
# §8 テスト4: 契約0′/5 の有限差分検証(複素 t 必須) --- parton-mode (fork addition) ---
using Test, LinearAlgebra
using MVMCExpertModeParsers
using MVMCOptimizers

# θ の実自由度 dof(1-based: 2k-1=Re α_k, 2k=Im α_k)に δ を加えて ln ip を返す
function _ln_ip_at(data, sites::Vector{Int}, n_elec, n_flavor, n_idx; dof = 0, δ = 0.0)
    α = MVMCOptimizers.parton_alpha_from_terms(data)
    if dof > 0
        k = (dof + 1) ÷ 2
        α[k] += isodd(dof) ? δ : δ * im
    end
    n_site = data.modpara.nsite
    mfham = MVMCOptimizers.PartonMFHamiltonian(n_site, n_elec, n_flavor, n_idx)
    MVMCOptimizers.parton_build_mf_templates!(mfham, data)
    MVMCOptimizers.parton_update_orbitals!(mfham, α, n_elec)
    cfg = MVMCOptimizers.PartonConfiguration(n_site, n_elec, n_flavor, 1)
    for f in 1:n_flavor, (m, r) in enumerate(sites)
        MVMCOptimizers.move_particle!(cfg, f, m, 0, r)
    end
    amp = MVMCOptimizers.PartonAmplitudeData(1, n_flavor, n_elec)
    ws = MVMCOptimizers.PartonSamplingWorkspace(n_elec, n_flavor)
    MVMCOptimizers.parton_recompute_amplitude_all!(amp, mfham, cfg, data,
        (data.qp_trans = [collect(1:n_site)]; data.qp_trans_sgn = [ones(Int, n_site)]; ws))
    return log(MVMCOptimizers.parton_calculate_ip(amp, [ComplexF64(1, 0)])), mfham, cfg, amp
end

@testset "契約0′/5: 有限差分 vs 両スロット(Re/Im 独立)" begin
    data = _toy_mf_data()                     # 4 サイト・F=2・複素 t・n_idx=2
    sites = [1, 3]
    lnip0, mfham, cfg, amp = _ln_ip_at(data, sites, 2, 2, 2)
    MVMCOptimizers.parton_update_orbital_derivatives!(mfham, 2)
    ip = exp(lnip0)
    n_para = 2                                 # n_proj=0 + n_idx=2
    sr_opt_o = zeros(ComplexF64, 2 * (1 + n_para))
    MVMCOptimizers.parton_calculate_o!(sr_opt_o, amp, mfham, cfg, data,
                                        [ComplexF64(1, 0)], ip, 0)
    δ = 1e-6
    for k in 1:2, part in 1:2
        mfham.is_onsite_group[k] && part == 2 && continue   # 凍結成分は FD 対象外
        dof = 2 * (k - 1) + part
        lp, _, _, _ = _ln_ip_at(data, sites, 2, 2, 2; dof = dof, δ = +δ)
        lm, _, _, _ = _ln_ip_at(data, sites, 2, 2, 2; dof = dof, δ = -δ)
        fd = (lp - lm) / (2δ)                  # 複素数(ln ip の実自由度微分)
        slot = part == 1 ? MVMCOptimizers.o_slot_re(k) : MVMCOptimizers.o_slot_im(k)
        @test sr_opt_o[slot] ≈ fd rtol = 1e-5 atol = 1e-8
    end
    # オンサイト群の Im スロットは厳密にゼロ
    k_onsite = findfirst(mfham.is_onsite_group)
    @test sr_opt_o[MVMCOptimizers.o_slot_im(k_onsite)] == 0
end

@testset "契約4: E_loc の全数展開一致(4 サイト・F=2)" begin
    data = _toy_mf_data()
    # physhop: 0-1, 1-2, 2-3, 3-0 の複素ホップ+coulombinter の対角 μ 行
    tp = ComplexF64(-1.0, 0.15)
    for i in 0:3
        push!(data.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(i, mod(i+1,4), tp, true))
    end
    # CoulombInterTerm の実コンストラクタに合わせて μ n_i(対角)と V n_1 n_2 を追加
    # (フィールド名は expert_types.jl で確認して記述)
    # E_loc(x) = Σ V n_i n_j + Σ_hop [ t ip(x_{i→j}) + conj(t) ip(x_{j→i}) ] / ip(x)
    # を、parton_main_cal! 内部の計算経路と、テスト内の素朴な独立実装
    # (全 hop を移動→契約1 全再計算→ip 比)で突き合わせる
    # 独立実装は契約2 を使わないこと(同じバグを共有しない)
    # ... (実装時に parton_main_cal! から E_loc 計算を小関数 parton_local_energy に
    #      切り出してテスト可能にする。粒度は関数 1 個の追加まで)
end
```

- [ ] **Step 2: 落ちることを確認** → **Step 3: 実装** → **Step 4: 全緑** → **Step 5: コミット** `feat(parton): 契約4/5(E_loc・O)+§8-4 有限差分緑`

---

### Task 12: 配線 — オーケストレータ・ドライバ・sync(§8 テスト 5: OptFlag 凍結)

**Files:**
- Rewrite: `MVMCOptimizers.jl/src/parton_vmc_para_opt.jl`(骨組 21 行 → 完成)
- Create: `MVMCOptimizers.jl/src/parton_run_para_opt_from_namelist.jl`(現在 0 行)
- Modify: `MVMCOptimizers.jl/src/MVMCOptimizers.jl`(export 行: `export parton_run_para_opt_from_namelist, parton_vmc_para_opt!, validate_parton_inputs` — 借用 using は既存リストに全部含まれるか確認)
- Test: `MVMCOptimizers.jl/test/test_parton_optflag.jl`(新規)

**Interfaces:**
- Produces:
  - `parton_build_optimization_state(data; n_qp, n_vmc_sample) -> PartonOptimizationState`(VMCOptimizationState は `SROptData(1 + n_para, ...)` の n_para = n_proj + n_idx で確保。energy/sr_opt のみ実使用 — DESIGN §5)
  - `parton_sync_parameters!(data, ctx)`: rank0 の `pack_parameters(data)` を `bcast!` し全 rank で `unpack_parameters!`。**D_AmpMax・shift・normalize は呼ばない**(既存 sync_modified_parameter! を使わない理由をコメントで明記)
  - `parton_vmc_para_opt!(pstate, data, ctx; rng, output_dir, c_timer)`: 納品骨組の未定義変数を実体化した完成形:

    ```julia
    function parton_vmc_para_opt!(pstate::PartonOptimizationState, data::ExpertModeData,
                                  ctx::ParallelContext; rng, output_dir = nothing,
                                  c_timer = CTIMER_DISABLED)
        validate_parton_inputs(data, ctx)
        mp = data.modpara
        n_elec = mp.nelec
        mfham = pstate.mfham
        for step in 0:(mp.nsr_opt_itr_step - 1)
            α = parton_alpha_from_terms(data)
            parton_update_orbitals!(mfham, α, n_elec)          # 契約0
            parton_update_orbital_derivatives!(mfham, n_elec)  # 契約0′
            parton_make_sample!(pstate, data, rng)             # 骨格+契約2,3
            parton_main_cal!(pstate, data)                     # 契約4,5
            weight_average_we!(ctx, pstate.state, CTIMER_DISABLED)
            weight_average_sr_opt!(ctx, pstate.state, CTIMER_DISABLED)
            reduce_counter!(ctx, pstate.config.counter)        # counter 直渡し
            is_output_rank(ctx) && output_data!(data, pstate.state, step;
                                                output_dir = output_dir)
            info = stochastic_opt!(data, pstate.state, c_timer)
            info = Int(bcast_scalar(ctx, info))
            info != 0 && return info
            parton_sync_parameters!(data, ctx)                 # bcast のみ
            n_keep = mp.nsr_opt_itr_step - mp.nsr_opt_itr_smp
            step >= n_keep && store_opt_data!(data, pstate.state, step - n_keep)
        end
        return 0
    end
    ```

    (weight_average_we!/sr_opt! と stochastic_opt! の実シグネチャ(ctx 有無・timer 位置)、store_opt_data! のインデックス規約は既存 vmc_para_opt.jl:280-350 の呼び方をそのまま写す — 上のコードは調査済みの呼称に基づくが、貼る前に該当行を再確認)
  - `parton_run_para_opt_from_namelist(namelist_path; nsteps, nsmp, output_dir, seed)`: 既存ドライバ(run_para_opt_from_namelist.jl:62-215)の縮約版:
    1. parse_expert_mode_files
    2. **flags 実体化(門番より前 — DESIGN §2.5)**:

       ```julia
       n_pmf = isempty(data.pmfpara_terms) ? 0 :
               maximum(t.idx for t in data.pmfpara_terms) + 1
       n_proj = MVMCExpertModeParsers.projection_layout(data).n_proj
       data.optimization_flags = fill(true, 2 * (n_proj + n_pmf))
       for (idx, flag) in data.pmfpara_opt_flags          # フラグ行適用(実虚両スロット)
           base = 2 * (n_proj + idx)
           data.optimization_flags[base + 1] = flag != 0
           data.optimization_flags[base + 2] = flag != 0
       end
       # オンサイト群の Im を強制凍結(is_onsite 判定は pmftrans の site1==site2 から)
       onsite_idx = Set(t.idx for p in (data.pmfpara_terms,) for t in p
                        if any(tt -> tt.site1 == tt.site2 &&
                                     tt.site1 == t.site1 && tt.flavor1 == t.flavor1,
                               data.pmftrans_terms))
       for idx in onsite_idx
           data.optimization_flags[2 * (n_proj + idx) + 2] = false
       end
       ```
       (onsite 判定は実装時に build と同じ結合 Dict を使う小関数 `parton_onsite_idx_set(data)` に切り出して契約 0 と共有 — 判定ロジックの二重実装を避ける)
    3. ctx 構築 → validate_parton_inputs(data, ctx)(門番)
    4. RNG(resolve_rnd_seed 流用)
    5. qp 恒等実体化(qptransidx.def なしのとき): `isempty(data.qp_trans) && (data.qp_trans = [collect(1:nsite)]; data.qp_trans_sgn = [ones(Int, nsite)])` + `init_qp_weight!(data)`
    6. pstate 構築 → parton_vmc_para_opt! 呼び出し
  - export 3 点

- [ ] **Step 1: 失敗するテストを書く**(§8 テスト 5)

```julia
# §8 テスト5: OptFlag 凍結成分が SR で動かない --- parton-mode (fork addition) ---
using Test, Random
using MVMCExpertModeParsers
using MVMCOptimizers

@testset "OptFlag: ゲージ固定 idx と オンサイト Im が SR 1 step で不動" begin
    data = _toy_mf_data()                 # idx0=ホッピング, idx1=オンサイト
    mp = data.modpara
    mp.nvmc_warmup = 20; mp.nvmc_interval = 1; mp.nvmc_sample = 50
    mp.nsr_opt_itr_step = 2; mp.nsr_opt_itr_smp = 1
    mp.nblock_update_size = 8; mp.complex_flag = 1; mp.two_sz = 0
    mp.nex_update_path = 6; mp.nstore_o = 1
    push!(data.physhop_terms, MVMCExpertModeParsers.PhysHopTerm(0, 1, ComplexF64(-1, 0), false))
    # flags: idx0 を凍結(ゲージ代表)、idx1 は Re のみ可(Im は onsite 凍結)
    data.pmfpara_opt_flags = Dict(0 => 0, 1 => 1)
    # ドライバの flags 実体化ロジックを直接呼ぶ(小関数化しておく):
    MVMCOptimizers.parton_materialize_flags!(data)
    @test data.optimization_flags == [false, false, true, false]

    ctx = MVMCOptimizers.serial_context()
    data.qp_trans = [collect(1:4)]; data.qp_trans_sgn = [ones(Int, 4)]
    MVMCExpertModeParsers.init_qp_weight!(data)
    pstate = MVMCOptimizers.parton_build_optimization_state(data)
    α_before = MVMCOptimizers.parton_alpha_from_terms(data)
    rng = MVMCOptimizers.SFMT19937RNG(); Random.seed!(rng, 123)
    status = MVMCOptimizers.parton_vmc_para_opt!(pstate, data, ctx;
                                                 rng = rng, output_dir = mktempdir())
    @test status == 0
    α_after = MVMCOptimizers.parton_alpha_from_terms(data)
    @test α_after[1] == α_before[1]                        # 凍結 idx0 完全不動
    @test imag(α_after[2]) == imag(α_before[2]) == 0.0     # オンサイト Im 不動
    @test α_after[2] != α_before[2]                        # 可動成分は動いた
end
```

- [ ] **Step 2: 落ちることを確認** → **Step 3: 実装**(flags 実体化は `parton_materialize_flags!(data)` として門番と独立にテスト可能な小関数に)→ **Step 4: 全緑+回帰(integration T0 一致)** → **Step 5: コミット** `feat(parton): 配線完成(オーケストレータ・ドライバ・sync)+§8-5 緑`

---

### Task 13: §8 テスト 6 — 結合テスト: トイ系 SR → ED 基底エネルギー(F=2 と F=3)

**Files:**
- Test: `MVMCOptimizers.jl/test/test_parton_ed_convergence.jl`(新規)

**Interfaces:**
- Consumes: 全タスク
- テスト設計(パッケージ追加なし・テスト内 ED):
  - 系: 4 サイト鎖・物理粒子 2 個・ハードコア(固縛射影と等価)・H_phys = Σ physhop t b†b + h.c.(t は複素位相つき)+ V n_i n_j 最近接
  - ED: 基底 = C(4,2)=6 次元の占有配置。H を 6×6 に素朴に組み、eigmin を厳密解とする。**符号**: F 奇(フェルミオン的合成粒子)は 1 次元鎖・最近接ホップなら交換が起きず符号は自明だが、ED 側は一般に「合成粒子の統計」で組む — F=2 はハードコアボソン(正符号)、F=3 はスピンレスフェルミオン(occupation ordering の符号)として **2 種類の ED を別々に書く**(符号定理 §1.3 の機械検証)
  - VMC 側: pmftrans/pmfpara は最近接ホッピング 1 変分 α(+ゲージ固定のオンサイト 1 個を凍結初期値で)。SR を 50-100 step 回し、最後の 10 step の平均エネルギーが ED 基底エネルギーに `rtol=2e-2` 程度で到達(サンプル数 500-1000。乱数固定で決定的)
  - F=2 と F=3 の両方で実施

- [ ] **Step 1: テストを書く**(ED 実装込み。乱数 seed・サンプル数・step 数・許容誤差はテスト内定数にし、落ちたら「期待値でなく実装を疑う」— 特に §7 の縮約規約と契約 4 の両方向評価)

```julia
# §8 テスト6: SR → ED 収束(F=2 ボソン / F=3 フェルミオン) --- parton-mode (fork addition) ---
using Test, LinearAlgebra, Random
using MVMCExpertModeParsers
using MVMCOptimizers

"4 サイト・粒子 2・ハードコアの ED。statistics = :boson / :fermion"
function _ed_ground_energy(t::ComplexF64, V::Float64; statistics::Symbol)
    sites = 4
    configs = [(i, j) for i in 1:sites for j in (i+1):sites]   # 占有サイト対(昇順)
    n = length(configs)
    H = zeros(ComplexF64, n, n)
    pos = Dict(c => k for (k, c) in enumerate(configs))
    for (k, (i, j)) in enumerate(configs)
        # 対角: V n_a n_b(最近接のみ)
        for (a, b) in ((i, j),)
            abs(a - b) == 1 || abs(a - b) == sites - 1 || continue
            H[k, k] += V
        end
        # ホップ: 各占有サイト r から r±1 へ(周期境界)。t b†_j b_i + h.c.
        for (which, r) in ((1, i), (2, j))
            for dst in (mod1(r + 1, sites), mod1(r - 1, sites))
                dst in (i, j) && continue
                new_pair = which == 1 ? minmax(dst, j) : minmax(i, dst)
                kk = pos[new_pair]
                hop_t = dst == mod1(r + 1, sites) ? t : conj(t)
                sign = 1.0
                if statistics === :fermion
                    # 昇順基底での並べ替え符号(通過する占有数)
                    other = which == 1 ? j : i
                    sign = (min(r, dst) < other < max(r, dst)) ? -1.0 : 1.0
                end
                H[kk, k] += sign * hop_t
            end
        end
    end
    @assert H ≈ H'
    return eigmin(Hermitian(Matrix(H)))
end

function _run_parton_sr(F::Int; nsteps = 80, nsample = 800, seed = 11)
    # fixture 構築(最近接 pmftrans idx=0 可動+オンサイト idx=1 凍結ゲージ固定、
    # physhop t 複素、coulombinter V)→ parton_run_para_opt_from_namelist 相当を
    # in-memory で実行し、最後の 10 step の etot/wc 平均を返す
    # ...(Task 12 の部品を組み合わせる。ファイル I/O を経ずに data を直接組んでよい)
end

@testset "§8-6 ED 収束" begin
    t = ComplexF64(-1.0, 0.3); V = 0.5
    for (F, stat) in ((2, :boson), (3, :fermion))
        e_ed = _ed_ground_energy(t, V; statistics = stat)
        e_vmc = _run_parton_sr(F)
        @test isapprox(e_vmc, e_ed; rtol = 2e-2)
    end
end
```

(_run_parton_sr の中身は Task 12 の部品の直結。変分空間が ED 基底状態を張れるか — 単一スレーター×固縛で 4 サイト 2 粒子のこの H に十分か — は SR が到達する最良値で判断し、`rtol` を初回実行の実測で 5e-2 まで緩めることは許す(それ以上悪ければ実装バグを疑う)。**期待値 e_ed の側は動かさない**)

- [ ] **Step 2: F=2 で緑にする → F=3 で緑にする**(F=3 が赤で F=2 が緑なら符号定理まわり=契約 2 の det 積か ED の符号のどちらかを systematic-debugging で切り分け)

- [ ] **Step 3: コミット** `test(parton): §8-6 SR→ED 収束(F=2/F=3)緑 — M1 契約完了`

---

### Task 14: T5 監査+回帰+DESIGN 更新+仕上げ

**Files:**
- Modify: `DESIGN_parton.md`(§10 消し込み・§11 追記)
- Audit only: 全 upstream ファイル

- [ ] **Step 1: マーカー監査**

```bash
grep -rn "# --- parton-mode (fork addition) ---" --include="*.jl" \
  MVMCExpertModeParsers.jl/src MVMCOptimizers.jl/src | grep -v "/parton_\|/pmf\|/physhop"
```

出力が次の登録点と 1:1 で一致すること(DESIGN §3.1 + modpara 系):
constants.jl(キーワード表+デフォルト定数)/ MVMCExpertModeParsers.jl(include+parse_file_by_type!)/ expert_types.jl(ModParaParameters 3 箇所+Term 群+ExpertModeData 2 箇所)/ modpara_parser.jl / read_input_parameters.jl(count)/ stochastic_opt.jl(enum+bulk+_at+value+set の 5 箇所)/ MVMCOptimizers.jl(include+export)。
変則マーカー(`Parton-mode` 等)が 0 件であること:

```bash
grep -rni "parton" --include="*.jl" MVMCExpertModeParsers.jl/src MVMCOptimizers.jl/src \
  | grep -v "/parton_\|/pmf\|/physhop" | grep -v "# --- parton-mode (fork addition) ---" \
  | grep -vi "PartonMF\|PhysHop\|parton_mode\|nflavor\|_PARAM_PMF\|pmfpara\|pmftrans\|physhop\|parton_"
```

- [ ] **Step 2: upstream diff 監査**: `git diff main --stat` で既存ファイルの変更が登録点ファイルのみであること。`git diff main -- MVMCExpertModeParsers.jl/src/utils/validation.jl` が空であること

- [ ] **Step 3: 回帰全緑**: 3 系統のテストを全部回し、T0 と一致(integration の数値列も含め)

- [ ] **Step 4: DESIGN_parton.md 更新**: §10 のチェックボックス消し込み(実名照合済・sync 詳細確定・oo/ho 依存確認済・store_opt_data 経路確認済)、§11 に v3.1 行を追記(nblock_update_size 新設 / pmfpara 7 列・pmftrans 6 列の確定 / modpara_parser を登録点に追加 / validation.jl revert の記録)

- [ ] **Step 5: 最終コミット** `docs(parton): DESIGN §10 消し込み・§11 追記(M1 完了)` → verification-before-completion → requesting-code-review → finishing-a-development-branch

---

## Self-Review 済みの注意点

- **契約 2 の純粋性と契約 4 の仮想呼び出し**: parton_amplitude_ratio! は ws.ratio_blocks を書くが amp は不変。契約 4 が同関数を測定に流用するとき、直後のホップ項評価が ratio_blocks を上書きしても、受理経路(契約 3)はサンプリング中しか使わないため安全。ただし parton_make_sample! 内で「棄却→次の提案」で ratio_blocks が古い値を持つ状態で契約 3 を呼ばない(契約 3 は直前の契約 2 と同じ (m, r_new) が前提 — 納品コードは受理直後に呼ぶので満たされる)
- **_at 版と bulk 版で f のシグネチャが違う**(loc のみ vs (para_idx, loc))— Task 4 のコードはこれを反映済み。混同すると全 setter が silent no-op になる
- **`n_keep`(store_opt_data! の開始 step)**: 既存 vmc_para_opt.jl:350 は `step - (n_steps - n_smp)` を渡す — Task 12 のコードはこれに合わせた。貼る前に該当行を再確認
- **エスカレーション条件**(即停止して人間に質問): DESIGN §1 の固定式とテスト結果の矛盾が実装誤り排除後も残る / 登録点リスト外の既存編集が必要に見える / 既存テストが赤くなり原因が自分の変更(revert してから報告)
