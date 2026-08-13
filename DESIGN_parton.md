# DESIGN: パートン平均場モード for Julia-mVMC

- ステータス: **M1 完了**(契約 0〜5 実装・§8 のテスト 0〜8 全緑・既存回帰全緑)
- 改訂: v3.2 (2026-08-13) — ゲージ平坦方向を同期時の射影で潰す方式へ(§2.5 主線変更)
- ベース: tmisawa/Julia-mVMC v0.5.0(fork、ブランチ `parton-mode`)

## 0. 目的と位置づけ

パートン構成に基づく変分計算を Julia-mVMC 上の**別モード**(`PartonMode = 1`)として実装する。

- 変分波動関数: フレーバーごとのスレーター行列式の積(+射影・量子数射影)
- 変分パラメータ: 平均場ハミルトニアン H_MF の成分 α(f_ij / Orbital とは**非共存**)
- **一般 NFlavor ≥ 2 対応**。物理粒子は F 個パートンの合成(F 偶=ボソン、F 奇=フェルミオン)で、
  一般の 2F 個の生成消滅演算子の積(合成粒子演算子)を扱う
- モード OFF(`PartonMode = 0`)時は既存経路・出力ともビット単位で従来と同一(既存 C-parity テスト全緑を維持)
- 既存機構(骨格設計・射影・SR ソルバ・MPI・I/O・タイマ)は最大限再利用

## 1. 変分波動関数と固定式

### 1.1 ansatz

```
Ψ(x) = P_J P_G ⟨x|φ⟩,   ⟨x|φ⟩ = ip(x) = Σ_qp w_qp Π_f det A^(f)_qp(x)
A^(f)_qp[m, n] = sgn_qp(r_m) · Φ^(f)[map_qp(r_m), n]     (行=粒子, 列=軌道, Ne×Ne)
```

- Φ^(f): H_MF^(f)(α) の下から Ne 個の固有ベクトル(占有軌道)
- 配置は**フレーバー固縛**: 全フレーバーが常に同一サイト集合を占有し、同時に移動する
  (硬い Gutzwiller 射影と厳密等価。移動集合で拘束を保つ = 物理的セクター内サンプリング)
- 量子数射影: 運動量射影(qptrans)対応。スピン射影は SU(2) 前提のため OFF(NSPGaussLeg=1)
- **gather に通す写像は `qp_trans`(順方向)+ `qp_trans_sgn`(サイトごと)で確定**
  (slater_update.jl の `tri = xqp[ori+1]` の規約に一致。OptTrans 層は門番で拒否済み)

### 1.2 平均場ハミルトニアン(α について線形)

```
H^(f)_ij(α) = α_{idx(i,j,f)} · t^(f)_ij   (+ h.c. はホッピングのみ暗黙付与)
⇔ H(α) = Σ_k α_k T_k (+h.c.),  ∂H/∂α_k = 定数テンプレート
```

- 結合キーは (site1, flavor1, site2, flavor2)。pmftrans(固定係数 t)× pmfpara(α と idx 写像)
- α: **オンサイト群 = 実数**(Im を OptFlag で強制凍結)、**ホッピング群 = 複素**
  (Re/Im を独立な 2 実変分パラメータとして扱う。mVMC の複素パラメータ規約と同じ)
- **α の正準置き場は `pmfpara_terms[].value`**(パラメータロケータの読み書き先=家風)。
  契約0 は毎ステップ `parton_alpha_from_terms(data)` で Vector{ComplexF64} に詰める
- 微分(**非正則性**: H は α* を含むため Ψ は α に非正則):
  `∂H/∂Re α_k = T_k + T_k†`, `∂H/∂Im α_k = i(T_k − T_k†)` — 独立に計算。i 倍の近道は禁止

### 1.3 定理(固縛セクターにおける 2F 演算子積)

- **選択則**: 固縛ブラ・ケット間で非ゼロの 2F 積は
  **合成ホップ b†_j b_i(全フレーバー同一始点・同一終点)と密度型のみ**。混在サイトの積は厳密にゼロ。
- **符号定理**: b† = (b)† のダガー順序反転により
  `b†_j b_i = Π_f (f†_j f_i)^(f)` が**余分な符号なしで**成立(任意の F。双線形は偶で可換)。
  ⇒ 局所推定量はフレーバー別 det 比の積 = 契約 2 の仮想呼び出し。統計符号は行列式が自動で運ぶ。

### 1.4 比・更新・測定の固定式

固縛移動(粒子 m: r→r′)。u[n] = sgn_qp(r′)·Φ^(f)[map_qp(r′), n]

```
R          = det A′/det A = Σ_n u[n]·A⁻¹[n, m]                    (契約2, O(Ne)/ブロック)
v[j]       = Σ_n u[n]·A⁻¹[n, j]                                    (v[m] = R)
A′⁻¹[:,m]  = A⁻¹[:,m]/R                                            (契約3, O(Ne²)/ブロック)
A′⁻¹[:,j]  = A⁻¹[:,j] − (v[j]/R)·A⁻¹[:,m]   (j ≠ m)
det A′     = R·det A
受理確率    = min(1, exp(2·log射影比)·|ip′/ip|²)
E_loc(x)   = Σ_diag V n_i n_j + Σ_hop t_ij·(射影比)·ip(x_{i→j})/ip(x)   (仮想固縛移動=契約2流用)
O_k        = ∂lnP/∂θ_k + (1/ip)·Σ_qp w_qp (Π_f det^(f)_qp)·Σ_f Tr[(A^(f)_qp)⁻¹ ∂A^(f)_qp/∂θ_k]
```

### 1.5 契約0′の連鎖律(一次摂動論)

θ_k(実自由度)→ H → eigen → Φ → A → ln ip の eigen 段:

```
W = U_unocc† (∂θH) U_occ,   G[u,n] = W[u,n]/(ε_n − ε_u),   ∂Φ = U_unocc · G
```

- 占有↔占有の混合は Tr[A⁻¹∂A] から**厳密に消える**(摂動係数が反エルミート・対角ゼロ)。
  非占有への漏れのみ構築。分母は HOMO-LUMO ギャップで下から抑制(契約0 の min_gap が保険)
- **バックエンドは摂動論で確定**(ForwardDiff は撤回: LAPACK zheev が Dual を受けないため。
  汎用固有ソルバ経由の AD は重く依存も増える)。**検証は有限差分**(§8)
- `Uu' * dHUo` の**随伴はここでは正しい**(ブラとの真の内積)。転置積・dot 禁止則(§7)は
  振幅の双線形縮約の話であり、この 2 種の縮約を取り違えないこと

### 1.6 QP 並進のコスト構造

per-QP 軌道は実体化しない(行置換+符号なので gather 時に写像を通す)。
- 実体化=なし / 厳密再計算 = n_qp·F·O(Ne³) / 移動毎 = n_qp·F·O(Ne)〜O(Ne²)
  (**サンプリングの n_qp 倍・F 倍は残る。消えるのは per-QP 行列の構築・保持のみ**)
- 有効条件: 射影群が置換で書ける操作のみ。フレーバー回転射影の導入時は実体化に戻す(§10)

## 2. 入力契約

### 2.1 modpara.def(パートンモード)

| キー | 意味 |
|---|---|
| `PartonMode` | 0=既存 mVMC(デフォルト)/ 1=パートン平均場 VMC / ≥2 は予約(明示拒否) |
| `NFlavor` | フレーバー数。PartonMode=1 で必須(0 なら綴り間違いを疑うエラー) |
| `NElec` | **フレーバーあたりパートン数 = 物理粒子数**(互換の錨: 既存機構の配列寸法単位) |

- `NParticle` / `NPartonPerFlavor` は NElec の**別名**(同一フィールドへ書き、食い違いはパースエラー)。
  派生量はアクセサ(`n_parton_total` 等)。冗長保持はしない
- **注意: 別名は `NElec` より後に書くこと**(v3.1 で明記)。食い違い検出は別名側の分岐にしか
  無く、上流の `NElec` 分岐は登録点ではないので無条件代入のまま残してある。したがって
  `NParticle 3` → `NElec 2` の順だと **`NElec` が黙って勝つ**。逆順(`NElec` が先)なら
  パースエラーになる。上流分岐を触らないという規律を優先した結果の非対称性
- **`PartonBlockUpdateSize`**: 受理 N 回ごとに振幅を厳密再計算する錨の周期(既定 16)。
  C-mVMC の `NBlockUpdateSize`(スレーター行列のブロック更新閾値)とは**意味が違う**ので、
  同名を避けてフォーク固有の接頭辞を付けてある
- 必須設定(門番が検査): `2Sz=0`(デフォルト −1=FSZ の罠)、`ComplexType=1`, `NVMCCalMode=0`,
  `NSRCG=0`, `NLanczosMode=0`, `NSPGaussLeg=1`, `NSPStot=0`, `NCond=-1`, `NLocSpin=0`,
  `NOrbitalIdx=0`, `NNeuron=0`, `NExUpdatePath=6`, `NSplitSize=1`(M1)

### 2.2 pmftrans.def(固定係数 t)

- 列: `site1 flavor1 site2 flavor2 value`。`PartonMFTransTerm`(immutable)は一般形で保持し、
  **初期実装は門番で flavor1==flavor2 を要求**(混成は将来の別 ansatz)
- **片方向のみ列挙 + h.c. 暗黙付与**(複素 α とエルミート性の帳尻。trans.def と意図的に異なる)。
  逆向き重複はエラー。オンサイトは h.c. なし直接加算、t 実数必須

### 2.3 pmfpara.def(変分パラメータ α と idx 写像)

- 列: `site1 flavor1 site2 flavor2 idx value` + 末尾フラグ行(idx flag、ゲージ固定・固定項用)
- idx は 0-based・連番。同一 idx の複数セル(フレーバー跨ぎ可)= **α の共有**。value 重複は一致検証
- **結合完全性は双方向エラー**(trans↔idx)。固定項は欠落でなく **OptFlag 凍結**で表現
- `pmfpara_idx_matrix` は**廃止**(結合は build 内ローカル Dict)

### 2.4 物理ハミルトニアン(E_loc 用)【確定】

- 密度型: 既存 `CoulombInterTerm`((i,j,V))流用。μ n_i は **coulombinter の対角行**
  (硬芯: V n_i² = V n_i)。CoulombIntra は拒否
- 合成ホップ: **新規 `physhop.def`**。5 行ヘッダ+`NPhysHop 個数`、行 = `site1 site2 Re Im`
  (0-based)。**片方向列挙+暗黙 h.c.**(E_loc が t 側・t* 側の両方向を評価)。
  **site1 == site2 禁止**(対角は暗黙 h.c.+両方向評価で二重計上の温床)。逆向き重複拒否
- 選択則(§1.3)に還元できない項は門番で拒否/警告

### 2.5 OptFlag とゲージ固定

- `optimization_flags` は 1 パラメータあたり実虚 2 スロットの平坦配列。
  `stochastic_opt!` は flag≠1 を固定扱い(確認済: L480, L1026)。
  **範囲外アクセスは「凍結」を黙って返す**実装なので、配列長不足=SR が α を黙って無視、が最悪の沈黙故障
- **実体化の順序**: パートンドライバが門番より前に flags を実体化
  (`fill(true, 2n_para)` → フラグ行適用 → オンサイト Im 強制凍結)→ 門番が
  `length == 2*(n_proj + n_idx)` と「MF 成分に flag=1 が最低 1 つ」を検査
- **ゲージ平坦方向は同期時の射影で潰す(v3.2 で主線を変更)**。Φ は α→cα(c は
  **正の実数**)と H→H+μI で不変。厳密演算なら S の固有値も力の成分もゼロなので SR は
  そちらへ動かないが、実際には **MC ノイズが力に偽の成分を与え、正則化 ε 付きの S⁻¹ が
  それを 1/ε 倍する**ため α が際限なく漂流する。毎ステップの同期時に
  `parton_project_gauge!` が掃き出す。位置づけは既存 mVMC の `D_AmpMax` と同じ
  「更新後にゲージスライスへ引き戻す」機構
  - **位相 e^{iθ} は不変ではない**(H が h.c. を含むので α→e^{iθ}α は
    H→e^{iθ}T + e^{-iθ}T† という別のハミルトニアンになる)。1 群あたり実 1 次元
  - Φ^(f) は H^(f) にしか依存しないので素朴にはフレーバーごとに独立な c_f があるが、
    1 つの idx が複数フレーバーに跨ると c_f が連動する。**独立なスケール群は
    「フレーバーを節点・共有 idx を辺とするグラフの連結成分」**で決まる(共有なし → F 個、
    全共有 → 1 個)。個数をハードコードしないこと
  - シフト H→H+μI は再正規化では潰れない。オンサイト idx が全サイトを等係数で覆うときだけ
    現れ、その群の一様成分を引いて潰す
  - スイッチは modpara の `PartonGaugeFix`(既定 1)。射影は bcast の**後**に適用する
    (α から決定論的に決まるので追加通信は不要)
- **OptFlag はゲージ目的では使わない**(v3.2 で降格)。用途はエルミート性
  (オンサイト Im の強制凍結)とユーザーの明示的固定に限る。凍結には
  「最適解が α_rep = 0 のときスライスに到達できない」という失敗モードがある。
  なお射影はスケール群を**丸ごと**実数倍しないとゲージ変換にならないので、凍結成分も
  再正規化の対象になる。凍結の意味は「SR が動かさない」であって「値が絶対に変わらない」
  ではない(物理は不変なので実害はない)
- **警告: 一様 MF は S が厳密に特異になり SR が解けない**(v3.1、実測で確認)。
  変分グループが「一様ホッピング 1 個 + 一様オンサイト 1 個」だけの入力では、実自由度 4 個が
  **すべて**ゲージ平坦になる(Re α_hop = 全体スケール、α_onsite = 一様シフト、
  Im α_hop も並進対称なリングでは軌道を変えない)。O が全成分ゼロ → S も g も全ゼロ →
  `DSROptRedCut` の閾値は `max(S_diag) × RedCut` なので冗長方向カットも働かず、
  Cholesky が NaN を返して `stochastic_opt!` が info≠0 で落ちる。
  - 対処: 物理的な変分方向を最低 1 本作る。ボンドを強弱 2 群に分ける(二量体化)か、
    ボンドごとに独立な idx を与える。テストのフィクスチャは
    `dimerized_mf_data` / `per_bond_mf_data` がこれに当たる
  - 症状の見分け方: SR が初手から info=1(NaN)/ エネルギーがステップ間で振動して降下しない /
    `min_gap` は健全なのに O が軒並み 1e-15。入力の変分自由度を疑うこと

## 3. アーキテクチャ

### 3.1 原則と登録点

- **新規コードは新規ファイルへ。既存ファイル編集は登録点のみ**。全箇所
  `# --- parton-mode (fork addition) ---` マーカー(一字一句同一。grep 監査用)
- 構造体フィールド追加は**必ず末尾**+コンストラクタ末尾。Dict/elseif も末尾+マーカー
- 0-based→1-based 変換は **`parton_build_*` の中だけ**(v3.1 で明確化)。
  起動時 1 回のテンプレート build 段に閉じ込めるという趣旨で、入力ファミリごとに 1 関数:
  - `parton_build_mf_templates!` — pmftrans / pmfpara(平均場)
  - `parton_build_phys_hamiltonian` — physhop / coulombinter(物理ハミルトニアン)

  ホットループ・アクセサ・E_loc・契約 2/3/5 に ±1 演算を書かない。新しい入力ファイルを
  足すときも `parton_build_*` を 1 本増やしてそこで変換する
- 登録点の全リスト(M1 実装で確定。編集された upstream ファイルはこの 9 つだけ):
  1. `utils/constants.jl` — MVMC_KEYWORDS 表(PartonMFTrans/PartonMFPara/PhysHop の行)
     + デフォルト定数(DEFAULT_PARTON_MODE / DEFAULT_NFLAVOR / DEFAULT_PARTON_BLOCK_UPDATE_SIZE)
  2. 入口ファイル — include 3 行 + `parse_file_by_type!` の elseif(同 3 種)
  3. `types/expert_types.jl` — ModParaParameters 末尾フィールド+kwargs+`new(...)` の 3 箇所 /
     Term 構造体群 / ExpertModeData 末尾フィールド+コンストラクタ既定値
  4. `utils/read_input_parameters.jl` — `count_variational_parameters` に n_pmf 加算
  5. **パラメータロケータ(案 B・RBM 前例)**: `stochastic_opt.jl` の
     `@enum _ParameterKind`(_PARAM_PMF 追加)/ `_foreach_parameter_location` と `_at` 版
     (末尾に PMF ループ)/ `_parameter_location_value` / `_set_parameter_location_value!`
     (elseif 末尾)。共有 idx は ORBITAL パターン同様「全行訪問・絶対 set 冪等・δ は各行の自値+δ」。
     標準モードでは pmfpara_terms が空なので全登録点で挙動不変
  6. `MVMCOptimizers.jl` 入口 — include 行 + export 行 + 借用 using リスト
  7. `parsers/modpara_parser.jl` — `parse_modpara_parameter!` の elseif
     (PartonMode / NFlavor / PartonBlockUpdateSize / NElec 別名)。§2.1 のキーを読むのに必然で、
     v3 のリストから漏れていた分を v3.1 で追加
  8. `data_io.jl` — `store_opt_data!` / `output_data!`(zvo_var)/ `output_opt_data!`(zqp_opt)
     に pmfpara_terms のループを末尾追加。これが無いと最適化された α が永続化されない。
     列・行は既存ブロックの後ろに付くだけなので標準入力での出力バイト列は不変
  9. `parameter_sync.jl` — `_duplicate_checked_sections` に pmfpara_terms を追加
     (共有 idx を持つセクションの診断網羅性。この関数は手動検査用ヘルパ)
- `utils/validation.jl` は登録点では**ない**(本番経路に未接続)。入力検査は門番が一手に引き受ける

### 3.2 ファイル構成(include はこの順、types が先頭必須)

```
MVMCExpertModeParsers.jl/src/parsers/
  pmfpara_parser.jl, pmftrans_parser.jl(実装済み), physhop_parser.jl(納品済み)
MVMCOptimizers.jl/src/
  parton_types.jl                  # 全 parton 型 + アクセサ + 委譲メソッド
  parton_unsupported_inputs.jl     # 門番(納品済みドラフト。フィールド実名へ要調整)
  parton_orbital.jl                # 契約0/0′(テンプレ構築・対角化・∂Φ ← parton_contract5.jl 前半)
  parton_calculate_m_all.jl        # 契約1
  parton_vmc_sampling.jl           # 契約2,3 + 骨格
  parton_vmc_main_cal.jl           # 契約4,5(← parton_contract5.jl 後半)
  parton_vmc_para_opt.jl           # オーケストレータ
  parton_run_para_opt_from_namelist.jl   # ドライバ
```

### 3.3 契約×関数カタログ

| 契約 | 関数 | 要点 |
|---|---|---|
| 0 | `parton_build_mf_templates!` | 起動時1回。結合(ローカル Dict)・双方向完全性・混在禁止・逆向き重複・t_ii 実数の検証。0→1based はここだけ |
| 0 | `parton_update_orbitals!` | ステップ毎。H(α) 組立(h.c. はホッピングのみ)→ Hermitian eigen → 全固有対+Φ → min_gap 検知 |
| 0′ | `parton_update_orbital_derivatives!` | ステップ毎。§1.5 の摂動論で dorbitals[f][dof] を構築。オンサイト Im はゼロ埋め |
| 1 | `parton_recompute_amplitude_all!` / `gather_a_block!` / `parton_calculate_ip` | 全 (qp,f) の LU 厳密再計算。特異は det=0(ノード)。gather は qp_trans 順方向 |
| 2 | `parton_amplitude_ratio!` | **純粋(ws のみ)**→棄却 revert 不要。R を ws.ratio_blocks へ。転置積(dot 禁止) |
| 3 | `parton_update_amplitude!` | SM 列更新。`|R|<ratio_floor` → `:need_recompute`(upstream にない防御) |
| 骨格 | `parton_make_sample!` ほか | §4。受理時は**配置コミット→振幅更新**の順(最重要不変条件) |
| 4 | `parton_main_cal!` / `parton_diag_energy` | サンプル毎: 契約1で錨→対角=占有数→合成ホップ=契約2 仮想呼び出し(両方向) |
| 5 | `parton_calculate_o!` + `o_slot_re/im` | §1.4 の O。スロット (2p+1, 2p+2) に**独立値**を格納(既存の `val*im` 近道は正則性前提=流用禁止) |
| 門番 | `validate_parton_inputs` ほか | §2 の執行+物理整合+flags 長検査(validation.jl は本番未接続のため頼らない) |
| 配線 | `parton_vmc_para_opt!` / ドライバ | 委譲: weight_average / stochastic_opt! / output_data! / bcast_scalar / reduce_counter!(counter 直渡し)。`parton_sync_parameters!` は bcast + ゲージ射影(D_AmpMax は不適用)。射影は α にのみ作用し、射影因子には触れない |

## 4. サンプリング骨格の規約

- C 踏襲: `n_in = NVMCInterval×Nsite`、初回 `n_out = WarmUp+Sample`、burn 再開時 `Sample+1`、
  `PartonBlockUpdateSize` 受理毎に錨、保存は末尾 Sample 個、サンプル毎再計算は測定側の分担
- 意図的差分: `burn_flag` は counter[11] 間借りでなく **PartonConfiguration の Bool**。
  交換更新は固縛下で恒等のため分岐ごと廃止
- **受理時の順序不変条件**: ①配置コミット → ②振幅更新。② の `:need_recompute` 部分更新は
  直後の契約 1(コミット済み配置から全再計算)が上書きするので安全。逆順は禁止

## 5. 構造体カタログ

**再利用**: EnergyData / SROptData(n_para=n_proj+n_idx で確保)/ 射影一式 / qp_trans 系 /
stochastic_opt!(案 B 後は MF ブロックも解く)/ weight_average / parallel / c_timer / data_io。
**純追加**: ExpertModeData(pmftrans_terms, pmfpara_terms, physhop_terms)。
**新規**(parton_types.jl):

| 構造体 | 層 | 中身 |
|---|---|---|
| `PartonMFTemplateEntry` | 意味層 | immutable isbits: site1, site2, flavor(1本に畳む), coeff。1-based |
| `PartonMFHamiltonian` | 意味層(Vector{Matrix} 可) | template, is_onsite_group, n_idx / h_mf, eig_vals, **eig_vecs(全固有対)**, orbitals, **dorbitals[f][dof]**, **dh_uo_scratch**, min_gap |
| `PartonAmplitudeData` | 速度層(フラット) | inv_a, det_a(生の複素 det)。[qp][f] 順。ストライドは block_index/inv_block に封じ込め |
| `PartonConfiguration` | 速度層 | 自前定義(既存は内部コンストラクタが 2 フレーバー寸法を焼き付け)。ele_idx=F·Ne, ele_cfg/num=F·Nsite, burn_flag::Bool, counter |
| `PartonSamplingWorkspace` | 速度層 | a_scratch(Ne²), ratio_blocks(n_qp·F), u_buf/v_buf/col_buf(Ne) |
| `PartonOptimizationState` | 外箱 | state::VMCOptimizationState(実使用 energy/sr_opt のみ)+ parton 4 部品+委譲 1 行メソッド |

添字文字: サイト `ri,rj` / フレーバー `fi,fj` / パック `rfi = ri + fi·n_site`。
軌道の行空間(サイトのみ)と配置空間(サイト⊗フレーバー)は別空間・別名。
固縛不変条件は `assert_flavors_locked` を錨と同タイミングで検査。

## 6. 命名規約

- 新規実装関数: `parton_` 接頭辞。呼ぶだけの既存関数はラップも改名もしない
- 関数名は表現中立(amplitude 系)。フィールド名は表現に正直で可(det_a)。抽象型は第二実装まで作らない
- フィールドに構造体名の反復接頭辞は付けない(h_mf の mf は物理記号なので可)。
  例外は出自混在コンテナ(PartonOptimizationState)
- 型 `PartonXxx` / 定数 `PARTON_XXX` / 破壊的 `!`

## 7. 数値ポリシー

- det は生の複素値(乗法更新+定期錨)。特異=det 0=ノード。log 空間化は §10
- `ratio_floor`(1e-12): QP 和では総比が健全でも個別ブロックがノードをかすめ得る →
  `:need_recompute` プロトコル(upstream は無防備。意図的追加)
- **縮約の使い分け(重要)**: 振幅の双線形縮約(契約 2/3/5 の最内)は**転置積**、`dot()` 禁止
  (第一引数を共役するため)。契約 0′ の `Uu' * dHUo` は**随伴が正しい**(ブラとの内積)
- **O 格納**: パラメータ p(1-based, [射影|MF])→ (2p+1, 2p+2) に (∂Re, ∂Im) を**独立格納**。
  既存の `sr_opt_o[imag_idx] = val*im`(vmc_main_cal.jl L2547-48)は f_ij 正則性の近道=流用禁止
- **MF スロットは蓄積境界で共役**(v3.1 で追加): `parton_calculate_o!` は §1.4 の O をそのまま
  格納し、上流アキュムレータへ渡す直前に `_parton_conjugate_mf_slots!` が MF スロットだけ
  複素共役にする。
  - 理由: 実パラメータ θ の勾配は `∂E/∂θ = 2 Re[⟨E_loc O*⟩ − ⟨E_loc⟩⟨O*⟩]` で O に共役が要るが、
    `calculate_oo!` / `calculate_oo_store!` は `HO[j] += w·e·srOptO[j]` と**共役なし**で蓄積し、
    `build_s_matrix_and_g_vector!` はその実部をそのまま力にする。射影 O は実数、f_ij は正則
    (2 スロットが val と val·im)なのでこの無共役規約と整合するが、**H が α* を含む MF ブロックは
    非正則**なので整合しない(有限差分と比べると符号すら合わない)。
  - S 行列は不変: 上流は実部しか使わず、`Re⟨O*⟩ = Re⟨O⟩`、`Re⟨O_i O_j*⟩ = Re⟨O_j O_i*⟩` なので
    計量は共役の有無に依らない。効くのは力ベクトルだけ。
  - 上流には手を入れない。適応は受け渡し点 1 箇所に閉じ込める(登録点を増やさないため)。
  - 恒久検証は §8-7。`g / (−DSROptStepDt)` が有限差分の `∂E/∂θ` に一致すること、シムを外すと
    一致が壊れること、S が共役の有無で変わらないことを全数展開で確認する
- ギャップ検知は HOMO-LUMO のみ(占有内縮退は無害)。twist 境界で偶発縮退を割る
- ホットループ内アロケーションゼロ。eigen/lu の小確保は頻度が低く許容
- デバッグ恒等式: v[m]==R / gather vs 実体化 / 錨の冪等性

## 8. 検証戦略(テストはこの順で)

0. 回帰ベースライン: PartonMode=0 で既存テスト全緑(着手前後に実施)
1. 契約1: トイ系(4〜6 サイト)で gather+det を全数展開と突き合わせ/錨の冪等性
2. 契約3 vs 契約1: 多数回高速更新後の厳密再計算一致(機械精度)。**複素位相つき t で**
3. QP: 写像 gather 版 vs per-QP 軌道実体化版の全ブロック一致
4. **契約0′/5: 有限差分** — ln ip(θ±δ) の数値微分と両スロット(Re/Im 独立)の一致。複素 t 必須
5. OptFlag: 凍結成分(オンサイト Im・ゲージ固定)が SR で動かないこと
6. 結合: トイ系 SR → ED 基底エネルギー収束。**F=2(ボソン)と F=3(フェルミオン)両方**(符号定理の機械検証)
7. **力ベクトル vs 勾配**(v3.1 追加、恒久): 全数展開(サンプリング誤差なし)で、上流の
   `build_s_matrix_and_g_vector!` が組む力ベクトルが変分エネルギーの有限差分勾配と一致すること。
   期待される関係は `g / (−DSROptStepDt) == ∂E/∂θ`(上流の g に入っている因子 2 が勾配式の
   因子 2 に対応する)。あわせて §7 の共役シムを外すと一致が壊れること(シムが飾りでないこと)、
   および S 行列が共役の有無で変わらないことを確認する。
   実装: `MVMCOptimizers.jl/test/test_parton_force_gradient.jl`
8. **ゲージ射影**(v3.2 追加、恒久): 射影は定義上ゲージ変換なので**厳密な等式**で検証する。
   トイ系・全数展開・MC なしで、(1) α をフレーバーごとに実数正倍しても E_var と |ip| 比が
   1e-12 で不変、(2) 射影の前後で E_var が不変、(3) 射影は冪等、(4) 独立スケール群の数が
   idx 共有パターンどおり(共有なし → F 個、全共有 → 1 個)、(5) 一様オンサイト群の
   シフトでも E_var 不変、(6) スイッチと配線、(7) SR で射影 ON ならノルムが保たれる。
   実装: `MVMCOptimizers.jl/test/test_parton_gauge.jl`

## 9. マイルストーン

- **M1**(完了): 一般 F 構造での ParaOpt 初点火(射影なし・n_qp=1・複素 1 変種・直接 SR)。
  Done: §8 の 0〜8 全緑(F=2/F=3 の ED 一致含む)。
  テストは `MVMCOptimizers.jl/test/test_parton_*.jl` と
  `MVMCExpertModeParsers.jl/test/test_parton_*.jl`
- **M2**: 運動量射影 ON + 相関因子。既存 Gutzwiller は固縛で自明化(全占有サイトが常にダブロン)
  → **物理密度 n^b ベースの Jastrow を RBM 前例に倣い新設**
- **M3**: PhysCal、det^F 高速路(フレーバー対称入力の検出時)、log 空間 det、NSplitSize>1、
  eigen ワークスペース化、命名・負債整理(返済トリガー: M1 全緑)

## 10. リスクと未決事項

- [ ] 物理密度 Jastrow の設計(M2)。それまで parton_log_proj_ratio は恒等 0
- [x] modpara フィールド実名の照合 — `nvmc_warmup` はそのまま存在。`parton_block_update_size` は
  Julia 移植に相当物が無かったので新設(modpara キー `PartonBlockUpdateSize`、既定 16)
- [x] parton_sync_parameters! の詳細 — `pack_parameters` → `bcast!` → `unpack_parameters!` の
  3 行。既存 `sync_modified_parameter!` は使わない(D_AmpMax リスケールが入るため)。
  store_opt_data! / 出力はロケータ経由で値を取るので MF も自動で記録される
- [x] sr_opt_oo/ho への蓄積関数が sr_opt_o の中身にのみ依存すること — 確認済
  (`calculate_oo!` / `calculate_oo_store!` は sr_opt_o と (w, e) しか読まない)。
  ただし store 経路では `finalize_oo_store!` を最後に呼ばないと OO が空のままになる
- リスク: フェルミ準位縮退 / ゲージ平坦方向 / 非正則スロット詰め / NExUpdatePath=6 の
  upstream 衝突 / フレーバー回転射影導入時の QP gather 前提崩壊 /
  時間反転対パートン(t 共役対+α 実共有)は現形式で表現不可

## 11. 決定ログ(要約)

- v3.2 (2026-08-13): **ゲージ平坦方向を同期時の射影で潰す方式に変更**(§2.5)。
  OptFlag による成分凍結は主線から降格し、用途をエルミート性とユーザーの明示的固定に限定。
  独立スケール群は idx のフレーバー共有パターン(連結成分)で決まり、個数は仮定しない。
  位相回転はゲージではない(H が h.c. を含むため)。modpara に `PartonGaugeFix`(既定 1)を
  追加、登録点は expert_types / constants / modpara_parser。§8 に 8-8 を追加
- 2026-08-03: シナリオ B(ペア部置換)/ 別モード / 新規は新ファイル+登録点最小
- 〜v2: NElec=フレーバーあたり / α 実虚規約+OptFlag / 片方向+暗黙 h.c. / idx フレーバー次元 /
  idx_matrix 廃止 / PartonConfiguration 自前 / 委譲ラッパ / ratio_floor / 2F 定理 / 一般 F 直行
- v3 (2026-08-13): **SR 配線=案 B(ロケータ拡張、RBM 前例)** / **α 正準=pmfpara Term 値** /
  **契約0′=一次摂動論(ForwardDiff 撤回)・検証=有限差分** / **O スロット (2p+1,2p+2) 独立格納** /
  **qp_trans 順方向で確定** / **physhop 確定(site1 site2 Re Im・site1≠site2・片方向)** /
  flags 実体化はドライバで門番より前
- v3.1 (2026-08-13, M1 統合で確定):
  - **入力形式**: pmftrans.def = `site1 flavor1 site2 flavor2 Re Im`(6 列)、
    pmfpara.def = `site1 flavor1 site2 flavor2 idx Re Im`(7 列)+ 末尾フラグ行 `idx flag`。
    値は trans.def と同じ Re/Im の 2 トークンに揃えた
  - **PartonBlockUpdateSize を新設**(Julia 移植に相当フィールドが無かった。既定 16)。
    C-mVMC には別の意味の `NBlockUpdateSize` が既にあるので、フォーク固有の接頭辞を付けて
    名前衝突を避けた(上流が C 版を移植したとき elseif 連鎖で先に来た側が黙って勝つのを防ぐ)
  - **登録点に modpara_parser.jl を追加**(§3.1-7)。validation.jl は登録点ではない
  - **SR 力ベクトルの共役**(規約は §7、恒久検証は §8-7): 実パラメータの勾配は
    2 Re[⟨E_loc O*⟩ − ⟨E_loc⟩⟨O*⟩] で O に共役が要るが、既存 `calculate_oo!` 系は HO を共役なしで
    蓄積する。f_ij のような正則パラメータでは 2 スロットの詰め方と噛み合って正しくなるが、
    非正則な MF ブロックでは噛み合わず符号が狂う。上流には手を入れず、アキュムレータへ渡す
    時点で MF スロットを共役にする(`_parton_conjugate_mf_slots!`)。S 行列は実部しか使わない
    ので不変
  - **規則「0→1based はテンプレート build の一箇所」を `parton_build_*` の中だけ、に書き換え**
    (§3.1)。物理ハミルトニアンにも変換が要るため入力ファミリごとに 1 関数とする
  - **一様 MF = 厳密特異 S の警告を §2.5 に追加**。実測で、一様ホッピング + 一様オンサイトだけの
    入力は実自由度が全部ゲージ平坦になり SR が NaN で落ちる
  - **§8 に 8-7(力ベクトル vs 有限差分勾配)を恒久テストとして追加**
  - コードレビュー(共役シム / 登録点 diff 全件)を受けた修正:
    - **射影因子を M1 では門番で拒否**。`parton_main_cal!` は射影ブロックの O を計算しない
      ので、あると SR が黙って凍結したまま最適化されない。加えて §7 の共役シムは
      「MF 以外のスロットの O が実数」を前提に S の不変性を得ており、複素な射影ブロックが
      共存すると交差項の実部が変わる。M2 で Jastrow を足すときは
      「虚スロットに 0 を書く」既存規約を守ること
    - **store 経路のサンプルスキップ対策**。ノード上のサンプルを飛ばすと store の書き込み
      位置がずれ、前ステップの O が残ったスロットを `finalize_oo_store!` が読んで OO だけ
      汚れる。実際に詰めた個数を数えて書き込み位置と `sample_size` の両方に使う
    - **`output_opt_data!` をドライバから呼ぶ**+`data_io.jl` を登録点に追加。これが無いと
      最適化された α がどこにも永続化されない(zqp_opt.dat / zvo_var.dat とも MF 列が空)
    - **`parton_fill_sr_opt_o!`** に契約 5 と共役シムを束ねた。対で呼ぶ必要があるのに
      片方を忘れても例外が出ないため
    - `_duplicate_checked_sections` に pmfpara_terms を追加(診断の網羅性)
  - 既知の非対応(M3 送り): `MVMC_KEYWORDS` 表は現状どこからも参照されないデッドデータ
    (実際の namelist ディスパッチは `parse_file_by_type!` の文字列連鎖)。
    パートン 3 ファイルは `_is_required_if_present_file_type` に未登録で、破損時のエラーは
    門番まで遅延する
  - **finalize_oo_store!** の呼び出しが必須(store 経路では OO はサンプルループ後にまとめて組む)
  - **物理ハミルトニアンのテンプレート build** を追加(`parton_build_phys_hamiltonian`)。
    physhop / coulombinter の 0→1based 変換はそこ 1 箇所に閉じる(平均場側の build と対)
  - **符号定理の機械検証が済んだ**: 全数展開した VMC 推定量が、F 偶なら硬芯ボソン ED、
    F 奇なら JW 符号つき ED のレイリー商と 1e-10 で一致(F = 2,3,4,5)。E_loc に符号を
    書いていないのに統計が F の偶奇で出る。SR は F=3 でフェルミオン基底エネルギーに厳密収束
  - **ゲージ平坦方向の実測**: 一様ホッピング + 一様オンサイトだけの MF は全自由度がゲージ平坦で
    S が全ゼロになり SR が解けない。テストは強弱 2 群(またはボンドごと)に分けた入力を使う