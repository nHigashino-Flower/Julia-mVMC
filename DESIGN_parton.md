# DESIGN: パートン平均場モード for Julia-mVMC

- ステータス: **M2 前半完了**(運動量射影の配線・検証まで。§8 の 0〜15 と P 層 P0〜P2 全緑・
  既存回帰全緑。残り: 物理密度 Jastrow)
- 改訂: v3.9 (2026-08-13) — 性能: 契約0′ rank-1 / det^F 高速路 / 測定フェーズのサンプル並列(§7 性能ポリシー)
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
  (slater_update.jl の `tri = xqp[ori+1]` の規約に一致。OptTrans 層は門番で拒否済み)。
  **v3.7 で運動量固有値により機械検証済み**(§8-11)。付随して確定した符号規約:
  重み `w_R = exp(2πi K·R)` を与えると射影状態の並進固有値は `T_S|ψ⟩ = e^{+2πi K·S}|ψ⟩`。
  標準の Bloch 規約 `T_S|ψ_k⟩ = e^{-i k·S}|ψ_k⟩` では **k = −2πK** にあたるので、
  **def の K は標準規約の運動量の符号を反転したラベル**。C-mVMC の規約をそのまま
  引き継いだ結果で参照実装との比較では整合するが、時間反転が破れた系(複素ホッピング・
  フラックス)では K と −K が別状態なので、セクターを指定するときは符号を意識する

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
- **`PartonFlavorSymFast`**(v3.9): 1=フレーバー対称の自動検出 + det^F 高速路(既定)/
  0=強制無効(完全に従来経路)。対称判定はテンプレート build が行う(§7)
- 必須設定(門番が検査): `2Sz=0`(デフォルト −1=FSZ の罠)、`ComplexType=1`, `NVMCCalMode=0`,
  `NSRCG=0`, `NLanczosMode=0`, `NSPGaussLeg=1`, `NSPStot=0`, `NCond=-1`, `NLocSpin=0`,
  `NOrbitalIdx=0`, `NNeuron=0`, `NExUpdatePath=6`, `NSplitSize=1`(M1)

### 2.2 pmftrans.def(固定係数 t)

- 列: `site1 flavor1 site2 flavor2 value`。`PartonMFTransTerm`(immutable)は一般形で保持し、
  **初期実装は門番で flavor1==flavor2 を要求**(混成は将来の別 ansatz)
- **片方向のみ列挙 + h.c. 暗黙付与**(複素 α とエルミート性の帳尻。trans.def と意図的に異なる)。
  逆向き重複はエラー。オンサイトは h.c. なし直接加算、t 実数必須

### 2.3 pmfpara.def(変分パラメータ α と idx 写像)

- 列: `site1 flavor1 site2 flavor2 idx [Re Im]` + 末尾フラグ行(idx flag、明示的固定用)
- idx は 0-based・連番。同一 idx の複数セル(フレーバー跨ぎ可)= **α の共有**。value 重複は一致検証
- **結合完全性は双方向エラー**(trans↔idx)。固定項は欠落でなく **OptFlag 凍結**で表現
- `pmfpara_idx_matrix` は**廃止**(結合は build 内ローカル Dict)

### 2.3.1 α の初期値経路(v3.3 追加)

modpara に初期化モードのキーは**追加しない**。スイッチは 2 つとも入力の形そのもの:

```
pmfpara.def の各行:
   value 列が未入力 (5 列)   → 乱数で初期化
   value 列が入力あり (7 列) → その値を採用(0 でも 0 を採用。乱数で埋めない)
        ↓ 上書き(namelist.def に InPmfPara が書かれているときだけ)
InPmfPara.def              → ウォームスタート / リスタート
```

- **「未入力」と「0 を指定」を列の有無で区別する**。値がゼロかどうかで分岐する判定は
  採らない — 意図的に α = 0 で始めたい結合を黙って乱数で埋めないため。
  `PartonMFParaTerm.has_value` が presence を運ぶ
- 列数は 2(フラグ行)/ 5(未入力)/ 7(入力あり)のいずれか。それ以外はパースエラー
- **idx 単位の整合検証**: 同一 idx を共有する行は presence が揃っていること。一部だけ
  値ありはエラー(同じパラメータに 2 つの初期値が生じるため)
- 乱数は**オンサイト群が実数のみ**(Im はエルミート性で凍結される)、ホッピング群は複素。
  振幅は既定 1.0。群全体が乱数なら全体スケールはゲージ自由度なので物理に効かないが、
  **同一ゲージ群内に指定値と乱数が混在する場合は相対スケールが効く**
- RNG は**専用ストリーム**。解決済みのベースシード(ランクごとのオフセットを加える前の値)
  で初期化するので全ランクが構成的に同一の α を得る(bcast 不要)。
  **サンプリング用 RNG を消費しない**(消費すると同一入力でもサンプリング系列が変わる)
- **出力は 2 本、writer は 1 つ**(形式を二重管理しない):
  - `<CParaFileHead>_pmfpara_opt.dat` — 既存 per-block 出力
    (`zqp_gutzwiller_opt.dat` 等)に揃えた最適化後の α。`output_opt_data!` が書く
  - `<CParaFileHead>_pmfpara_init.dat` — 初期値確定直後のダンプ。ランタイム乱数を
    入れると「どの初期値で回したか」が残らなくなるので再現性の担保として出す

  どちらも `InPmfPara.def` としてそのまま読み戻せる。書式:

  ```
  ===============================
  NPmfParaIdx <n_idx>
  ===============================
  ===============================
  ===============================
  <idx 0-based> <Re %.18e> <Im %.18e>
  ```

  読み手は既存の汎用実装 `parse_input_parameter_file` をそのまま使う(新規パーサ不要)
- **ヘッダは 5 行**にする。`parse_input_parameter_file` は `data_start = 6`
  (5 行読み飛ばし)なのに、既存 3 ブロックの writer が出すヘッダは **4 行**しかない。
  そのまま往復させると **idx = 0 の行が脱落**し、しかも件数不一致は `@warn` 止まりなので
  警告 1 行で素通りする(3 個書いて `[1, 2]` が返ることを実測で確認)。パートン側は
  5 行にして自分の往復を成立させ、**既存 3 ブロックの writer には触らない**
  (C-parity の出力比較に影響しうるため)。既存側の不一致は §11 に upstream への
  報告候補として記録
- `zqp_opt.dat`(添字列もヘッダも無い位置依存形式)も In*.def とは往復できないが、
  こちらも標準経路の出力なので触らない

**確定順序**(ゲージ射影と順序依存がある):

```
pmfpara.def 読み込み(presence 判定)
  → 未入力 idx を乱数で生成
  → InPmfPara.def で上書き(あれば)
  → OptFlag 実体化 → 門番検証
  → gauge_target_norm 確定(テンプレート build の中)
  → SR ループ
```

ゲージ射影の引き戻し先は初期 α のノルムなので、**乱数と上書きが済んでから** build を
呼ぶこと。InPmfPara の適用は `parse_expert_mode_files` の中(既存 In*.def と同じ位置)
ではなく**パートンドライバ**が行う — 中でやると乱数初期化より前になってしまうため

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
  10. `utils/constants.jl` の `MVMC_KEYWORDS` に `InPmfPara`(v3.3)。ただし
      InPmfPara の**適用**は `read_input_parameters!` ではなくパートンドライバが行う
      (§2.3.1 の確定順序: 乱数初期化より後でなければならないため)
  11. `data_io.jl` の `output_opt_data!` に per-block 出力
      `<CParaFileHead>_pmfpara_opt.dat`(v3.3)。既存 3 ブロックの writer は触らない
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
| 出力 | `parton_write_*`(§3.3.1) | 既存 writer を再利用。**読むだけ**で新規の数値計算はしない。rank 0 のみ。step 0 が "w"、以降 "a" |
| 配線 | `parton_vmc_para_opt!` / ドライバ | 委譲: weight_average / stochastic_opt! / output_data! / bcast_scalar / reduce_counter!(counter 直渡し)。`parton_sync_parameters!` は bcast + ゲージ射影(D_AmpMax は不適用)。射影は α にのみ作用し、射影因子には触れない |

### 3.3.1 出力ファイル一覧(v3.5)

すべて rank 0 のみ、`_output_path` 経由。`PartonMode = 0` では 1 つも生成されない。
数値は `%.18e`。1 回書きは `"w"`、ステップ毎追記は step 0 が `"w"` で以降 `"a"`
(同じ出力先で回し直したときに前 run の行が残らないこと。既存 SRinfo writer は
「ファイルが無いか空」のときだけヘッダを出す追記実装なので、**開始前に消すのは
呼び出し側の責務**)。既存ファイルの列形式は一切変更していない。

出力は 2 系統に分ける。

#### (a) `.def` 族 — 入力として読み戻せる出力

`clean_line` が `#` と `//` を除去するのは **Julia 移植で足された拡張**であり、
mVMC の .def 形式にコメント機能はない。**.def 族には `#` を書かないし読まない。**
読み手はヘッダ 5 行を固定でスキップする(`parse_input_parameter_file` の
`data_start = 6` と整合)。ヘッダは

```
===============================
<キーワード> <件数>
===============================
== <列の説明> ==
===============================
<データ行…>
```

| ファイル | キーワード | データ行 |
|---|---|---|
| `zqp_pmfpara_opt.dat` / `zqp_pmfpara_init.dat` | `NPmfParaIdx` | `idx ReAlpha ImAlpha`(idx は 0-based) |
| `zqp_pmfham_opt.dat` | `NPmfHam` | `site1 flavor1 site2 flavor2 ReH ImH`(**サイト・フレーバーとも 0-based**) |

`zqp_pmfham_opt.dat` は **全 (flavor, site1, site2) の組を h.c. 側もゼロ要素も含めて**
出す密ダンプで、行順は `(flavor, site1, site2)` の辞書順に固定(run 間で `diff` が
取れることが再現性・回帰テストの前提)。行の**形式**は `pmftrans.def` のデータ行と
同じだが**内容は 1 対 1 ではない** — pmftrans は片方向のみ列挙して h.c. を暗黙付与
する規約なので、**このファイルをそのまま pmftrans.def として再投入することはできない**
(逆向き重複としてテンプレート build が弾く)。キーワードを `NPmfHam` にしてあるのは
pmftrans の `NPartonMFTrans` とは別物であることを名前で示すため。

既存 per-block writer(`zqp_orbital_opt.dat` 等)はヘッダ **4 行**で読み手と
食い違うが(idx = 0 が脱落する)、**既存側には触らない**。§11 に upstream 報告候補。

#### (b) 診断・解析用の出力

先頭に `#` のヘッダ行(列名)。既存 SRinfo(`#Npara Msize optCut diagCut …`)の
前例に倣う。こちらは `#` で正しい。

| ファイル | 内容 |
|---|---|
| `zvo_SRinfo.dat` | Npara/Msize/optCut/diagCut/sDiagMax/sDiagMin/absRmax/imax。**直接法パス**から既存 writer をそのまま呼ぶ(ヘッダ・列は CG 版と 1 文字も違わない) |
| `zvo_parton_diag.dat` | step, min_gap, 受理率, α ノルム(ゲージ射影の前/後), 試行数, 受理数, 再計算数。既存カウンタと `PartonMFHamiltonian` の保持値を読むだけ |
| `zvo_parton_time.dat` | step ごとの所要時間と累積時間(wall-clock) |
| `zqp_pmfband_opt.dat` | `flavor band_index eigenvalue occupied`(**flavor・band_index とも 0-based**)。`occupied` は下から `NElec` 個が 1。`# key value` 形式で NFlavor / NSite / NElec と各フレーバーの HOMO-LUMO ギャップを併記 |
| `zqp_pmfvec_opt.dat` | 固有ベクトル。既定 OFF(`with_vectors = true` のときだけ) |
| `zvo_parton_runinfo.dat` | `# key value` 形式。base_seed / PartonMode / n_idx / githash / wall_sec ほか。githash 取得失敗時は `"unknown"` で run は落ちない |
| `zvo_conv.dat` | step, E, var, \|E − E_tail\|。`zvo_out.dat` の列を読み直したもの(再計算しない) |
| `zvo_CalcTimer.dat` | §3.3.2 |

`zqp_pmfham_opt.dat` の唯一の正は **α 側**(`zqp_pmfpara_opt.dat`)。H のダンプは
SR ループ後に**最終 α で組み直してから**書く(ループ内の最後の
`parton_update_orbitals!` は最終更新前の α で走っている)。

作図は本体から切り離す: `tools/plot_conv.jl` + `tools/Project.toml`(Plots はここだけ)。

### 3.3.2 CalcTimer(v3.5)

既存 `c_timer.jl` の枠組みをそのまま使う(**新しいタイマ機構を作らない**)。
既存 `write_ctimer_para_opt` が `zvo_CalcTimer.dat` を `"w"` で書いた後、
`parton_write_ctimer` が同じファイルにパートンセクションを**追記**する
(既存 writer に触らずにセクションを足すための形)。

**パートンモードでは既定で有効。** 既存モードは `MVMC_C_TIMER=1` の opt-in のままで
既定を変えていない。パートンでも `MVMC_C_TIMER=0` を明示すれば切れる。

**ID 帯は 800–813。** 使用中の ID はリポジトリ全体で 0–72 / 600–603 / 920–966。
C 版 `OutputTimerParaOpt` は 0–99 を主要フェーズ・600 番台を lspinflip 下位に使う
体系で、920 番台以降は Julia 移植が足した診断。800 番台は完全に空きで、C 版の
番号体系からも Julia 移植の診断からも離れており将来の衝突が最も起きにくい
(`CTIMER_N = 1000` の上限内)。

| ID | 区間 |
|---|---|
| 800 | Parton total(SR ループ全体) |
| 801 / 802 | 契約0(H 構築+対角化)/ 契約0′(∂Φ) |
| 803 | サンプリング骨格 |
| 804 / 805 / 806 | 契約1(厳密再計算)/ 契約2(比)/ 契約3(更新) |
| 807 | main_cal |
| 808 / 809 / 810 | 契約4(E_loc)/ 契約5(O)/ OO 蓄積 |
| 811 / 812 / 813 | SR / 同期 / 出力 |

タイマは C と同じく **inclusive**(親を止めずに子を回す)。

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

### 性能ポリシー(v3.9)

**大原則: 最適化は厳密に同じ値を出す。** 数式の変更は含まない。合格条件は
§8 の等式テスト群(8-2 / 8-4 / 8-7 / 8-8 / 8-14 / 8-15)が緑のままであること。

- **契約 0′ は rank-1 蓄積**: `W = Uu'(∂θH)Uo` をボンドごとに
  `W += a · Uu'[:, s1] ⊗ Uo[s2, :]` で直接積み上げる(`_parton_w_rank1!`)。
  「dHUo を疎に組む → 密 gemm」は gemm の時点で疎性が消えて
  O(n_un·NSite·Ne) だった。rank-1 なら O(b_k·n_un·Ne)。W は空いた
  `dh_uo_scratch` の先頭 n_un 行を間借り、分母除算はインプレース(バッファ増なし)
- **det^F 高速路**(§2.1 `PartonFlavorSymFast`): idx 写像と t^(f) が全フレーバーで
  一致(= 任意の α で H^(f) が同一)なら、eigen/∂Φ は f=1 だけ計算して残りへ
  コピー、振幅ブロックは 1 フレーバー分だけ保持(`PartonAmplitudeData.n_stored`)。
  **書き込みループは 1:n_stored、読み出しの積・和は 1:n_flavor のまま別名ブロックを
  読む** — 積の評価順が変わらないので det/inv/ip/比/E_loc は ON/OFF でビット一致。
  唯一 O の Tr だけ Σ_f Tr → F·Tr と総和順が変わる(実測 ≤1.3e-16、E2E の E_var
  軌道で ≤4.1e-14/100 step)。ストライドは block_index / inv_block の 2 関数に
  閉じたまま(§5)
- **スレッド化は opt-in**(`JULIA_MVMC_INNER_THREADS=1`、既存 threading.jl の作法)。
  既定は逐次で従来とビット同一。対象は**測定フェーズのサンプル並列のみ**
  (§4 層 2): `parton_main_cal!` は保存済み配置を舐めるだけで**乱数を消費しない**
  ので並列化が正当。マルコフ連鎖(`parton_make_sample!`)は本質的に逐次
  (乱数消費順 = 再現性)で、絶対に触らない。
  **ビット一致の構成**: 並列フェーズはサンプル別の (E_loc, O) を書くだけ、
  縮約(energy / OO / HO / store)はサンプル順の逐次パスが従来と同一の演算列で
  行う。スレッド数にも依存しない(E2E で zvo_out / SRinfo / pmfpara_opt / var の
  全ファイルが逐次とバイト一致、8 スレッドで main_cal 2.3 倍)。
  注: Julia の Task 生成は task-local `default_rng()` を設計上進める(fork 仕様)が、
  パートン本番経路は明示 rng オブジェクトのみを使い default_rng を使わないので
  再現性に影響しない(§8-15 が機械検証)
- **縮約の総和順は既定経路では 1 bit も変えない**: 契約 5 の標準経路(非対称)は
  高速路と共通化せず逐語のまま残す。共通化すると (f, m) の走査順が変わり、
  既存 run の再現がビットレベルで壊れる

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
   **v3.8 拡張**: 同じ恒等式を **n_qp > 1**(6 サイト環・Z_6 全並進・k=1 の複素重み)
   でも立てる。契約 5 の「∂Φ も QP 写像を通して gather する」経路(`qmap[r]`)は
   n_qp = 1 では恒等写像になるため、従来の検証では素通りしていた。
   トイ系のサイズに注意 — 4 サイト・Ne=2 だと k=1 セクターが 1 次元しかなく、
   Ψ が α によらず同一状態になって勾配も力も恒等的にゼロ(テストが退化)。
   実装: `MVMCOptimizers.jl/test/test_parton_force_gradient.jl`
8. **ゲージ射影**(v3.2 追加、恒久): 射影は定義上ゲージ変換なので**厳密な等式**で検証する。
   トイ系・全数展開・MC なしで、(1) α をフレーバーごとに実数正倍しても E_var と |ip| 比が
   1e-12 で不変、(2) 射影の前後で E_var が不変、(3) 射影は冪等、(4) 独立スケール群の数が
   idx 共有パターンどおり(共有なし → F 個、全共有 → 1 個)、(5) 一様オンサイト群の
   シフトでも E_var 不変、(6) スイッチと配線、(7) SR で射影 ON ならノルムが保たれる。
   実装: `MVMCOptimizers.jl/test/test_parton_gauge.jl`
9. **α の初期値経路**(v3.3 追加、恒久): 決定性(同一ベースシードでビット一致)、
   全ランク構成的一致、明示ゼロの保護、全指定時に乱数経路を通らないこと、presence 混在の
   検出、列数の検証、ダンプ往復(書いて `InPmfPara.def` として読み戻すとビット一致)、
   オンサイト群の Im が厳密ゼロ、サンプリング RNG を消費しないこと。
   加えて (6b) 最適化後の `zqp_pmfpara_opt.dat` を読み戻して α がビット一致し
   **idx = 0 が脱落していない**こと(既存 writer のヘッダ 4 行問題の回帰ガード)、
   (6c) 行を 1 本削った `InPmfPara.def` は警告止まりでなくエラーで停止すること。
   実装: `MVMCOptimizers.jl/test/test_parton_initial_params.jl`
10. **出力ファイル**(v3.4 追加、v3.5 拡張、恒久): (1) `PartonMode = 0` で新規ファイルが
   1 つも出ないこと、(2) SRinfo が直接法パスから出て既存 CG 版とヘッダ・列数が一致、
   (3) 診断ログの行数・受理率が既存カウンタと整合、(4) 平均場ダンプが**密**(件数 =
   NFlavor×NSite²)で `mfham.h_mf` と 1e-12 一致・h.c. 側も入っている・**0-based**で
   書かれている・`#` 行を含まない、バンドが対角化結果と 1e-10 一致し `occupied` が
   下から NElec 個、(5) run メタデータのベースシードが modpara と一致、(6) 収束
   テーブルが `zvo_out.dat` と行単位で整合、(7) 作図スクリプトが本体依存に入って
   いないこと、(8) `.def` 族がヘッダ 5 行・`#` 非依存で往復でき **idx = 0 が
   脱落しない**こと、(9) 同一入力・同一シードの 2 run が**バイト一致**すること
   (行順の決定性。時刻を含む runinfo / time / CalcTimer は除外)、(10) 同じ出力先で
   再実行しても前 run の行が残らないこと、(11) CalcTimer がパートンでは既定で生成され
   パートンセクションが全て出ており **ID が既存と衝突しない**こと・既存モードでは
   `MVMC_C_TIMER` なしで生成されないこと。
   実装: `MVMCOptimizers.jl/test/test_parton_output.jl`
11. **運動量射影の向き**(v3.7 追加、恒久): §8-3(gather 版 = 実体化版)は参照側を
   実装と同じ式 `Φ[shifts[qp][r], n]` で定義していて**循環しており、写像の向きに
   ついては検出力がゼロ**。n_qp や符号を増やしても上がらない。向きは**群の性質だけで
   決まる並進固有値**でしか決められない。射影 `P = Σ_R w_R T_R` は `w_R = e^{2πinR/L}`
   なら |φ⟩ によらず `T_S P|φ⟩ = w_S^{±1} P|φ⟩` を満たすので、配置側の並進
   `τ_S: r ↦ r+S` に対する `ip(τ_S x)/ip(x)` が `n, S, L` だけから決まる。
   (1) 順写像規約で `e^{-2πinS/L}` に一致すること(並進不変**でない**平均場・複数配置・
   全 S)、(2) **写像を逆向きにすると複素共役に振れる**こと(= 検出力の証明。§8-3 に
   欠けていたのはこの一段)、(3) 独立経路の裏取り: 並進不変な平均場では |φ⟩ 自身が
   並進固有状態なので、射影が非ゼロで生き残る K が一意に決まり、その K は **Φ の
   バンド運動量**から qp 経路と無関係に予言できる。あわせて `n* ≠ −n*` (K がゼロでも
   ゾーン境界でもない)を assert して、パラメータ変更で検出力が消えたら落ちるようにする。
   実装: `MVMCOptimizers.jl/test/test_parton_qp_momentum.jl`
12. **QP の本数整合(門番)**(v3.8 追加、恒久): `NMPTrans` は modpara.def、
   `qp_trans` / `qp_trans_sgn` / `para_qp_trans` は qptransidx.def と**別経路**で
   入る。振幅側は `get_n_qp_full`(= NSPGaussLeg × NMPTrans × NQPOptTrans)で
   寸法を決め、gather は `data.qp_trans[qp]` を引くので、片方だけ書き忘れると
   射影の項が黙って落ちる — とくに `para_qp_trans` が短いと `init_qp_weight!` が
   0 のまま残し、**重み 0 の項**としてエラーなく消える。`validate_parton_qp` で
   本数・重み配列長・写像長を突き合わせる。`NMPTrans < 0`(C の APFlag =
   反周期境界)も拒否する: `init_qp_weight!` は `abs` を取って N 本作るのに
   `get_n_qp_full` は `max(1, ·)` で 1 に潰すため、射影が黙って恒等に縮退する。
   符号は `qp_trans_sgn` で運べるのでこの経路は要らない。
   実装: `MVMCOptimizers.jl/src/parton_unsupported_inputs.jl`、
   テスト: `MVMCOptimizers.jl/test/test_parton_gatekeeper.jl`
13. **P 層 P2: QP 構成**(v3.8 追加、恒久): 射影が変分的に正当なのは `T_R` が
   **H_phys の対称性**のときで、H_MF の並進不変性は不要(むしろ破れているから
   射影に意味がある)。(1) 並進写像が巡回群として閉じた全単射、(2) cb 模型の
   1 体項が全並進で不変、(3) 拡大セルで縮約した平均場は基本セル並進で**不変で
   ない**(射影が自明でないこと)、(4) `qptransidx.def` が往復し門番を通る、
   (5) **射影の変分的性質**: `Σ_k w_k E_k = E_noproj`(凸結合)と
   `min_k E_k ≤ E_noproj` が全数展開で機械精度で立つこと — 写像の向き・重みの
   位相・n_qp の配線のどれかが狂うと破れる、(6) 参照実装 `make_QNPidx` と同じ
   本数・同じ並進であること、(7) 参照準拠 QP でも (5) が成り立つこと。
   実装: `test/physics/test_p2_qp_translation.jl`
14. **det^F 高速路の等式**(v3.9 追加、恒久): 高速路 ON / OFF で det_a・inv_a・
   ip・比・E_loc が**ビット一致**(積の評価順を変えない設計の検証)、O は総和順の
   変更ぶんだけ許容(実測 F=2: 0.0、F=3: 1.3e-16、閾値 1e-14)。対称でない入力
   (フレーバーで idx を割る / t を変える)で自動 OFF、`PartonFlavorSymFast = 0`
   で強制 OFF。F=2 / F=3、n_qp = 2、契約 3 の高速更新後のブロック一致まで。
   実装: `MVMCOptimizers.jl/test/test_parton_detpow.jl`
15. **測定フェーズのサンプル並列**(v3.9 追加、恒久): `force_threaded` で並列経路を
   強制し、逐次と energy / OO / HO / store の**全要素ビット一致**(store / 非 store
   両経路、スレッド文脈の再利用経路も)。明示 rng オブジェクトの列が不変であること。
   `julia -t N` でも同じテストがそのまま実スレッド検証になる。
   実装: `MVMCOptimizers.jl/test/test_parton_threading.jl`

   **QP 構成の 2 系統に注意**(v3.8 で判明): 参照実装には
   `build_QNPTransSiteList`(全並進 Nu = Nsite/2 本)と `make_QNPidx`(x 方向
   kext 本)の 2 つがあり、**パートン v2 が使うのは後者**。アンザッツが x 方向に
   kext セル周期を持つとき、破れている並進の剰余類 `Z_kext` の代表だけを張る。
   y 方向はアンザッツが破っていないので、そこで射影しても状態は(規格化を除いて)
   変わらない — 得が無くコストだけ増えるから入れない、という**実務的な**理由。
   **原理的には全並進で射影しても不利にはならない**(上の 5 より
   `min_k E_k ≤ E_noproj` が常に成立する)。
   参照実装をそのまま実行して確認した対応(Nux=Nuy=4, Nsite=32):
   `puc(K)=2 → NQPTrans=2, uclist=[(0,0),(1,0)]`、`K=4 → 4 本`。

## 9. マイルストーン

- **M1**(完了): 一般 F 構造での ParaOpt 初点火(射影なし・n_qp=1・複素 1 変種・直接 SR)。
  Done: §8 の 0〜9 全緑(F=2/F=3 の ED 一致含む)。
  テストは `MVMCOptimizers.jl/test/test_parton_*.jl` と
  `MVMCExpertModeParsers.jl/test/test_parton_*.jl`
- **M2**(前半完了): 運動量射影 ON + 相関因子。
  前半 Done(v3.6〜v3.8): qp 写像の base 正規化・ParaQPTrans 虚部・gather の向きの
  決定論的検証(§8-11)・QP 門番(§8-12)・CB 模型の QP 構成と変分的性質(§8-13)。
  実測: `:orbit` では射影が全シードで改善し Kx=0 が最良。`:orbit_flavor` +
  ランダム初期値では射影 ON が劣化するが、4 系統の切り分けで実装バグではなく
  最適化ランドスケープの問題と確定(REPORT §12-4。ウォームスタートなら留まる)。
  残り: 既存 Gutzwiller は固縛で自明化(全占有サイトが常にダブロン)
  → **物理密度 n^b ベースの Jastrow を RBM 前例に倣い新設**
- **M3**: PhysCal、log 空間 det、NSplitSize>1、
  eigen ワークスペース化、命名・負債整理(返済トリガー: M1 全緑)

## 10. リスクと未決事項

- [ ] 物理密度 Jastrow の設計(M2)。それまで parton_log_proj_ratio は恒等 0
- [ ] **`NSplitSize > 1` を門番が拒否している理由は「未検証」**(v3.9 明記)。
  comm1 グループ内のサンプル分割はパートンの保存配置・振幅の持ち方と噛み合うかを
  確かめていないだけで、原理的な障害は特定していない。解禁は M3 の検証作業
  (プレーンな複数ランク MPI = 層 1 は動作確認済み: 2 ランクで E_var が
  単一ランクと統計誤差内一致、サンプル総数 n_rank 倍)
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

- v3.9 (2026-08-13, 性能): **数式不変の最適化一式**(§7 性能ポリシー / §8-14 / §8-15)。
  まず CalcTimer で実測してから実装した(32 サイト・1500 サンプル・200 step:
  main_cal 73〜81% ≫ sampling 17〜26% ≫ 契約0′ 0.6〜1.2% — 想定表の
  「契約0′ が重い」はこのサイズでは外れ。優先順位は実測に従い、§1/§2 は指示どおり
  無条件実施)。
  (1) **契約 0′ を rank-1 蓄積へ**。gemm 版との一致は全 dof・全フレーバーで
  1.2e-16、ホットループのアロケーションは 0 bytes(実測)。
  (2) **det^F 高速路**(`PartonFlavorSymFast`、既定 ON・自動検出)。
  検出は「テンプレートの (k, s1, s2, t) 集合がフレーバー間で一致」— h.c. の向きを
  変えて書いた等価入力は保守的に非対称と判定(正しさは不変、高速路が効かないだけ)。
  対称ケース実測 1.51 倍(:orbit n_qp=2: 契約1 2.0× / 契約3 1.9× / 契約5 2.1×)、
  メモリ 1/F。E2E は step 0 ビット一致・100 step 軌道で ≤4.1e-14。
  (3) **測定フェーズのサンプル並列**(層 2)。8 スレッドで main_cal 2.3 倍・
  全出力ファイルが逐次と**バイト一致**(縮約をサンプル順の逐次パスに分離する構成)。
  MPI(層 1)は 2 ランクで統計一致を確認。
  **層 3(サンプリング中の内側ループ)は実装しない(実測に基づく決定)**:
  契約 1/2/3 の (qp, f) ブロックは典型 4 個で、既存 threading.jl の発火閾値
  (max(64, nthreads))を大きく下回り、実装しても発火しない。契約 5 の k ループは
  main_cal にしか現れず層 2 が包含する(@threads の入れ子は不可)。n_qp·F が
  数十を超える系が現れたら再訪。
  **不採用(指示 §5 の記録)**:
  - *W = Φ·InvM トリック*(比を O(1) で引く): W 構築 O(NSite·Ne²) に対し
    ボンド毎ドット積は O(N_bond·Ne) ≈ O(4·NSite·Ne) で、Ne > 4 ではドット積の
    方が安い。受理毎の W 更新 O(NSite·Ne) も提案数 ≫ 受理数で不利
  - *測定のサンプリングへの統合*(main_cal のサンプル毎の錨を消す): 保留。
    サンプリング中にも周期錨が入るので節約は錨コストの 3 割程度で、
    PhysCal(M3)が保存配置を再測定する道を塞ぐ
  `PartonBlockUpdateSize` の実測(§3): 16/32/64/128 で錨直前の乖離は
  1e-13(det 相対)/ 1e-11(A⁻¹)級で **B に対する増加傾向なし**(累積誤差より
  1 手の丸めが支配的)。錨 [804] は線形に減る(0.23→0.03s)が全体比が小さく、
  wall は 4.2→4.0s。既定 16 は据え置き(変更は指示待ち。推奨値の議論は
  コミットメッセージ / 報告参照)
- v3.8 (2026-08-13, M2): **QP の門番と CB 模型の QP 構成**(§8-12 / §8-13)。
  (1) 門番 `validate_parton_qp` を追加。`NMPTrans`(modpara.def)と qptransidx.def
  由来の配列は別経路なので、片方だけ書き忘れると重み 0 の項が黙って落ちる。
  `NMPTrans < 0`(APFlag)は `init_qp_weight!` が `abs`、`get_n_qp_full` が
  `max(1, ·)` と**扱いが食い違って**射影が恒等に縮退するため拒否する。
  (2) **QP 構成に 2 系統あることが判明**。参照実装の `build_QNPTransSiteList`
  (全並進 Nu = Nsite/2 本)ではなく、パートン v2 が使うのは `make_QNPidx`
  (**x 方向 kext 本のみ**)。アンザッツが破っている並進の剰余類だけを張るのが
  実務的に得(保たれている y 方向で射影しても状態は変わらず、コストだけ増える)。
  **原理的には全並進でも不利にならない** — 当初「変分空間を狭めるので悪化する」と
  書いたのは誤りで、短い run(800 step)の未収束を機構と取り違えたもの。
  参照実装をそのまま実行して本数を照合した
  (Nux=Nuy=4: K=2 → 2 本、K=4 → 4 本)。`cb_qp_translations` として実装し、
  fixture の `qp_xext = (kext, nkx)` から `qptransidx.def` を書く。
  (3) 射影の**変分的性質を厳密な等式で検証**: `Σ_k w_k E_k = E_noproj` と
  `min_k E_k ≤ E_noproj` が全数展開で機械精度で立つ。これは写像の向き・重みの
  位相・n_qp の配線が同時に正しくないと成立しないので、配線の総合ゲートになる
- v3.7 (2026-08-13, M2): **gather の向きを決定論的に検証し、符号規約を確定**(§1.1 / §8-11)。
  実装は順写像側で**正しい**(参照実装 `build_def.jl:678` の `build_TranslationalOperatorparams`
  が `(Opidx, jsite, isite, 1)` = col2 が元サイト・col3 が並進後 `isite = jsite + R` で書き、
  パーサ `build_qp_trans_mappings!` が `qp_trans[mpidx][j] = itmp` と読むので順写像。
  gather はそれを `Φ[qmap[r_m], n]` に通す)。**プロダクションコードの変更はなし**。
  付随して確定した符号規約: def の重み `exp(2πi K·R)` に対し `T_S|ψ⟩ = e^{+2πi K·S}|ψ⟩`
  なので、標準 Bloch 規約では **k = −2πK**。C-mVMC から引き継いだラベル規約であって
  バグではないが、時間反転が破れた系では K と −K が別状態になるので明記する。
  **方法論の教訓**: §8-3 が「実装と同じ式で参照を定義する」循環に落ちていたのが
  この項目が M1 の全緑をすり抜けた理由。参照値が実装から独立でも、**ミューテーションで
  落ちることを見ていなければ「検出できるテスト」とは言えない** — 検出力の証明を
  テスト自身に同居させる(§8-11 の 2 番目・3 番目の testset)
- v3.6 (2026-08-13, M2 前半の着手): **運動量射影の前提に 2 つの実バグを発見・修正**。
  どちらも n_qp = 1 では表に出ず M1 の全緑をすり抜けていた。
  (1) `qptransidx.def` から読んだ `data.qp_trans` は**値が 0-based**(upstream
  `slater_update.jl:642` の `xqp[ri+1]` 規約)なのに、パートンの契約 1/2/3/5 は
  返り値を Φ の行番号(1-based)として使っていた。恒等フォールバックが
  `collect(1:n_site)` と 1-based を入れていたため偶然一致し、実 def を与えると
  全行が 1 ずれる(site 0 に写る行は `Φ[0,:]` で BoundsError)。
  **§3.1 規則 8「0→1 変換は一箇所」を qp 写像にも適用**し、`parton_ensure_qp!` で
  1 回だけ正規化する(`parton_normalize_qp_trans!`、冪等、全単射検査つき)。
  使用箇所 5 箇所に `+1` を撒く案は採らない — 追加時に忘れると静かにずれる。
  (2) `ParaQPTrans` の**虚部が読まれていなかった**(第 2 トークンのみ)。運動量射影の
  重みは `exp(2πi k·R)` なので、k ≠ 0 では位相が丸ごと落ち、しかもエラーにならず
  「重み 0 の項」として静かに消える。3 列目があれば虚部として読む(2 列の既存
  ファイルは虚部 0 で従来どおり)。共有パーサへの修正なので登録点マーカーを付けた。
  **upstream への報告候補**
- v3.5 (2026-08-13): **出力形式の 2 系統化 + CalcTimer**(§3.3.1 / §3.3.2)。
  `.def` 族は 5 行ヘッダ・`#` 非依存(mVMC の .def にコメント機能はなく、`clean_line` の
  `#` 除去は Julia 移植の拡張。これに依存した形式を .def 族に持ち込まない)。
  `zqp_pmfham_opt.dat` は**密な行形式**(全 flavor×site×site、h.c. もゼロも含む、
  0-based、辞書順)。行の形式は pmftrans.def と同じだが**内容は 1 対 1 ではなく
  再投入はできない**ためキーワードを `NPmfHam` として名前で区別する。
  CalcTimer はパートンで既定 ON・ID 帯 800–813(調査の上、C 版と Julia 移植の
  どちらの帯からも離れた空き帯を選んだ)。既存 writer は一切書き換えていない。
  SRinfo の再実行残留は**呼び出し側で開始前に消す**ことで解決(既存 writer は
  「無いか空のときだけヘッダ」の追記実装なので、writer 側は触らない)
- v3.4 (2026-08-13): **出力ファイル整備**(§3.3.1)。診断は既存状態を読むだけで、
  収集のための新しい数値計算は足さない。平均場ダンプは SR ループ後に**最終 α で
  組み直してから**書く(ループ内の最後の `parton_update_orbitals!` は最終更新前の α で
  走っており、そのままでは α* と H が食い違う)。診断行の `min_gap` は「そのステップ
  時点」の値なので、ループ後の `mfham.min_gap` とは時点が異なる — 比較するなら
  バンドファイル側と突き合わせる。作図は `tools/` に隔離し本体に Plots を入れない。
  **upstream への PR 候補**: 直接法ソルバに SRinfo 出力がないのは CG 版との非対称で、
  `stochastic_opt!` に kwarg を足すだけで既存 writer を再利用できる
- v3.3 (2026-08-13): **α の初期値経路**(§2.3.1)。modpara にキーを足さず、pmfpara.def の
  value 列の有無と namelist.def の InPmfPara の有無だけで切り替える。presence は列の有無で
  判定し、値がゼロかどうかでは分岐しない。乱数は専用 RNG ストリーム・ベースシード基準で
  全ランク構成的一致。出力は `_pmfpara_opt.dat`(既存 per-block に揃えた最適化後)と
  `_pmfpara_init.dat`(初期値ダンプ)の 2 本で writer は共有。
  **upstream への報告候補**: 既存 per-block writer(`zqp_gutzwiller_opt.dat` 等)は
  ヘッダ 4 行だが `parse_input_parameter_file` は 5 行読み飛ばすので、往復させると
  idx = 0 が脱落し件数不一致も `@warn` 止まりで素通りする(実測で確認)
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