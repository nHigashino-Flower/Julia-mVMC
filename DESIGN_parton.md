# DESIGN: パートン平均場モード for Julia-mVMC

- ステータス: **M2 完了**(§8 の 0〜18 と P 層 P0〜P2 全緑・既存回帰全緑。
  SR の gap 崩壊漂流は **v3.12 で根治**(占有追跡 `PartonOccMode`。既定は従来と
  ビット一致の opt-in)。機構は REPORT §15、実装の記録は §11)
- 改訂: v3.15 草案 (2026-10-23) — Φ 直接変分モード(案 A・案 B)の構造設計。**実装前**(§12)
- 改訂: v3.13 (2026-08-14) — 占有の読み戻し InPmfOcc(§2.3.1 / §3.3.1 / §8-18)
- 改訂: v3.12 (2026-08-14) — SR 漂流の根治: 占有追跡 PartonOccMode(§1.1 / §2.1 / §2.5 / §3.3.1 / §8-17)
- 改訂: v3.11 (2026-08-13) — M2 完了: 物理密度 Jastrow(§1.1 / §2.3.2 / §8-16)+ SR 応急処置(§7 / §8-17)
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
Ψ(x) = P_J ⟨x|φ⟩,   ⟨x|φ⟩ = ip(x) = Σ_qp w_qp Π_f det A^(f)_qp(x)
A^(f)_qp[m, n] = sgn_qp(r_m) · Φ^(f)[map_qp(r_m), n]     (行=粒子, 列=軌道, Ne×Ne)

P_J(x) = exp( Σ_{i<j} v_{ij} · n^b_i · n^b_j ),   n^b_i ∈ {0, 1}   (v3.11 M2 後半)
```

- **P_J は物理ボソン密度 n^b で定義**する。パートン和 Σ_f n^(f) = F·n^b で定義すると
  同じ v が F² 倍の意味になり F を跨いで入力の意味が変わる。上流の変数 x = n − 1 とも
  異なる(差は一体項+定数。係数の辻褄合わせはしない — §2)
- **P_J は配置のみに依存し qp に依存しない**ので Σ_qp の外に出る。§1.4 の受理式・
  E_loc 式は既にこの形(exp(log_pr) フック)。O_p = ∂lnΨ/∂v_p = cnt_p は**実数**
  (§7 の共役シムの前提を保つ。虚スロットは厳密 0、Im は OptFlag で強制凍結)
- P_G(Gutzwiller)は固縛下で自明化する(ダブロン数 = NElec の定数)ため**存在しない**。
  指定されたら門番が理由つきで拒否する(§2)

- Φ^(f): H_MF^(f)(α) の固有ベクトルのうち**占有集合 `O^(f)` に属する Ne 本**(占有軌道)
- **占有集合 `O` は導出量ではなく状態の一部**(v3.12)。`PartonOccMode` で選び方を切り替える:
  - `0 = aufbau`(既定): 下から Ne 個。v3.11 までの唯一の規則で、既定挙動は不変
  - `1 = mom`: 前ステップの占有部分空間との**重なり**が大きい Ne 本
    (`score_m = ‖Φ_ref† φ_m‖²` の上位。フレーバーごとに独立)
  - 値 2 以降は予約(門番が明示的に拒否)
- **なぜ必要か**(REPORT §15): 変分多様体内の level crossing は回避型なので Φ(α) 自体は
  連続だが、混成が効く境界層の幅は `w ~ gap` しかない。実測では gap ~ 5e-6 に対し SR の
  ステップ長が 0.0017 = 層の幅の 300 倍以上あり、離散ステップから見ると Φ は不連続に
  飛ぶ(主角 89.99°、同じ step で E が +1.1)。エネルギー順(断熱追跡)ではなく重なり
  (ダイアバティック追跡)で選ぶと、この急回転そのものが消えて ∂Φ が有界に戻る。
  **共変トラスト領域 `δ†Sδ ≤ r²` は原理的に効かない** — S は現在地の局所計量なので
  300 倍先の層を予測できない
- **タイブレークは決定論的**(スコア降順 → 固有値昇順 → band index 昇順)。同一入力・
  同一シードでビット再現するために必要
- `mom` では α → Φ が**履歴依存**になるので、占有集合を `_pmfocc_{init,opt}.dat` に
  出力し `(α*, O*)` の組で状態が閉じるようにする(§3.3.1)。SR 終了後に
  「占有がアウフバウと一致し、かつ gap が健全か」を検査して記録する(§2.5)
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
  0=強制無効(完全に従来経路)。対称判定はテンプレート build が行う(§7)。
  高速路は「H^(f) が全フレーバーで同一」を前提に f=1 の結果を配るので、占有集合も
  同一になるはず。**そうでなければ前提が崩れているので検出してエラー**にする(v3.12)
- **`PartonOccMode`**(v3.12): 0=aufbau(既定、v3.11 と完全に同一)/ 1=mom(占有追跡)。
  値 2 以降は予約で門番が拒否。機構と根拠は §1.1 / REPORT §15
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

**確定順序(v3.12 で占有を追加、v3.13 で `InPmfOcc` を追加)**:

```
pmfpara.def → 乱数 → InPmfPara.def 上書き → OptFlag 実体化 → 門番
  → 初期占有の確定: InPmfOcc.def があればそれ / 無ければ aufbau
  → gauge_target_norm 確定 → SR ループ
```

- 占有の確定は **α の確定より後**でなければならない(検証に NSite / NElec / NFlavor と
  α から組んだ H が要る)
- `_pmfocc_init.dat` には**実際に採用された初期占有**を書く(`InPmfOcc` 由来でも書く)。
  これで「どの占有で始めたか」が常に残る
- **`InPmfOcc` は初期占有としてのみ効く**。以後は `PartonOccMode` に従う:
  `0 = aufbau` なら次の step から選び直され、`1 = mom` なら**これが最初の参照**に
  なって枝を追う(継続 run が前 run の枝を引き継ぐ経路。§2.5)
- 全 step 占有を固定するモード(`explicit` = `PartonOccMode = 2`)は**作っていない**。
  値 2 は予約のままで門番が拒否し続ける。ウォームスタート / 鎖方式には初期占有の
  読み戻しだけで足りる

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

### 2.3.2 jastrowidx.def / InJastrow.def(v3.11 M2 後半)

**既存パーサ・既存経路をそのまま使う**(新しい登録点なし): `jastrowidx.def` は
namelist キーワード `Jastrow`、初期値上書きは `InJastrow.def`(どちらも
`parse_expert_mode_files` 内で処理済み)。SR の書き戻し・MPI 同期・
`zqp_jastrow_opt.dat` 出力も既存の projection_layout 経由がそのまま効く。

パートン固有の門番(v3.11):
- **Jastrow のみ受け入れる**。Gutzwiller は「固縛下でダブロン数 = NElec の定数に
  なり因子が自明化する」旨の明示エラー(黙って無視しない)。DH2/DH4 も拒否
  (スピン分解ダブロンが固縛で凍る)
- **v は実数**(複素はエラー)。Im スロットは `parton_materialize_flags!` が強制凍結。
  理由は §7 の共役シム(「MF 以外の O は実数」が S 不変性の前提)
- **全ての非対角ペアに idx が張られていること**(未指定 = −1 の残留は添字事故に
  なるため拒否。v = 0 のペアは明示の idx で表現する)。対称性
  `idx[i,j] == idx[j,i]` も検査
- jastrowidx.def の opt フラグ(idx flag 行)は Re 側に引き継がれる

上流との定義差(**意図的**): 上流 `make_proj_cnt!` は変数 x_i = n_i − 1
(サイト総電子数 − 1)で数えるが、パートンは **n^b_i ∈ {0,1}** で数える。
パートン和 F·n^b − 1 を使うと v の意味が F 依存になるため。同じ jastrowidx.def の
**idx の張り方**(Σ_{i<j}・自己項なし・対称行列・exp の符号 +)は上流と同一で、
v の**値の意味だけ**が変数の取り方ぶん異なる。x = n−1 との差は一体項+定数
(並進不変クラスかつ粒子数固定なら定数)。係数の辻褄合わせは入れない。

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
- **終端の自己完結性検査(v3.12)**。SR ループ終了後、最終 α で組み直した状態について
  次の 2 条件を検査し、`zvo_parton_runinfo.dat` に記録する(§3.3.1):
  1. `occ_gap_ok` — 占有↔非占有のギャップが数値誤差より十分大きく、かつ正
  2. `occ_aufbau_ok` — 選ばれた占有が「下から Ne 個」と一致する

  両方成立(`occ_selfcontained = 1`)なら **α\* 単独で状態が決まる**ので従来と同じ意味を
  持つ。不成立なら**非アウフバウ最適解が本物**ということで、`_pmfocc_opt.dat` を必ず
  添えて使う必要がある旨を `@warn` で知らせる。**どちらに転んでも黙って壊れない**。
  - **この検査が、占有を入力で固定する経路(`InPmfOcc`)の要否を決めた**。3 シード検証で
    `PartonOccMode = 1` の run が `occ_selfcontained = 0` を返した(占有の 67〜78% の
    step が非アウフバウ、終端の `occ_min_gap` も負)ため、**v3.13 で `InPmfOcc` を実装**
    (§2.3.1 / §3.3.1 / §8-18)。読み戻しは**初期占有としてのみ**で、全 step 固定の
    `explicit` は作っていない
  - 注意: **現行(aufbau)が常に成立するわけではない**。gap が潰れた α\* では条件 1 が
    破れ、対角化しても占有が決まらない(バンドが分離しないので Chern 解析もできない)。
    その意味で MOM は逸脱ではなく救済
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
| `zqp_pmfocc_opt.dat` / `zqp_pmfocc_init.dat` | `NPmfOcc` | `flavor band_index`(**どちらも 0-based**) |
| `zqp_pmfham_opt.dat` | `NPmfHam` | `site1 flavor1 site2 flavor2 ReH ImH`(**サイト・フレーバーとも 0-based**) |

`zqp_pmfocc_*.dat` は**占有集合 `O`**(§1.1)。行順は `(flavor, band_index)` の辞書順に
固定。`_init` は初期占有の確定直後、`_opt` は SR ループ後に**最終 α で組み直してから**
書く(pmfham と同じ理由)。`PartonOccMode = 1` では α → Φ が履歴依存になるので、
このファイルが無いと α\* から状態を再現できない。`aufbau` の run でも書く
(ファイルの有無で読み手の経路が分岐すると壊れやすいため)。

**読み戻し(v3.13)**: `namelist.def` に `InPmfOcc` を書くと、そのファイルの占有集合を
**初期占有**として採用する(`InPmfPara` と同じくファイルの有無がスイッチで、modpara に
キーは足さない)。形式は `zqp_pmfocc_*.dat` と同一なので、**出力をそのまま渡せる**。
`(α*, O*)` の組で状態が閉じるので、ウォームスタート / 鎖方式・PhysCal で最適化済み
状態から始められる。検証(行数・範囲・フレーバーあたり本数・重複)は門番
`validate_parton_occupation` が計算開始前に行い、破れたらエラーで停止する
(部分適用しない)。

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
| `zvo_parton_diag.dat` | step, min_gap, 受理率, α ノルム(ゲージ射影の前/後), 試行数, 受理数, 再計算数, **n_occ_deviation**, **principal_angle_max**(v3.12)。既存カウンタと `PartonMFHamiltonian` の保持値を読むだけ。`min_gap` は**占有↔非占有の符号つき差**(§1.1)。`n_occ_deviation` は占有が「下から Ne 個」と一致しない本数で、常時非ゼロなら非アウフバウ最適解が本物ということ。`principal_angle_max` は前ステップの占有部分空間との主角の最大値(度)で、枝の乗り換えは 90° として現れる |
| `zvo_parton_time.dat` | step, step_sec, cumulative_sec, **n_out, n_in, n_sample_total, n_update_total**(v3.10)。列名ヘッダの前に `# key value` で n_mpi_rank / n_julia_thread / nvmc_sample_total を併記(単体でスケーリング解析可)。**役割分担**: CalcTimer(§3.3.2)= 区間別の累積時間を最後に 1 回 / time = ステップ毎の経過時間と**サンプリング量**を逐次追記。区間別内訳は複製しない。n_out は初回 WarmUp+Sample・以降 Sample+1(式の家は `parton_n_out`)。C 版の `<head>_time_<idx>.dat` は受理率+試行カウンタ+ctime で列構成がパートンの実態と合わず(hop/ex/lsf は存在しない)、その役割は diag が担う |
| `zqp_pmfband_opt.dat` | `flavor band_index eigenvalue occupied`(**flavor・band_index とも 0-based**)。`occupied` は**実際の占有集合**(§1.1)。aufbau では下から `NElec` 個が 1 になり従来と同じ。唯一の正は `zqp_pmfocc_opt.dat` 側。`# key value` 形式で NFlavor / NSite / NElec と各フレーバーの HOMO-LUMO ギャップを併記 |
| `zqp_pmfvec_opt.dat` | 固有ベクトル。既定 OFF(`with_vectors = true` のときだけ) |
| `zvo_parton_runinfo.dat` | `key value` 形式(`#` はコメント行)。base_seed / PartonMode / n_idx / githash / wall_sec、**PartonOccMode と終端の自己完結性検査**(`occ_selfcontained` / `occ_gap_ok` / `occ_aufbau_ok` / `occ_n_deviation` / `occ_min_gap`。v3.12、§2.5)に加え(v3.10)並列構成: n_mpi_rank / n_julia_thread / blas_num_threads / inner_threads_enabled / nsplit_size / nvmc_sample_per_rank / **nvmc_sample_total**(= NVMCSample × n_rank。comm0 allreduce の実効統計量。NSplitSize>1 解禁時は式の再確認が要る旨をコード内コメントに記載)。githash 取得失敗時は `"unknown"` で run は落ちない |
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

**環境変数の注意(v3.10)**: Julia のスレッド数は `JULIA_NUM_THREADS` /
`julia -t N` で決まり、**`OMP_NUM_THREADS` では決まらない**。C 版のジョブ
スクリプト流用で `OMP_NUM_THREADS=8` でも黙って 1 スレッドになる事故を、
ドライバ起動時の `parton_warn_threading_config()` が検出して `@warn` する
(検出条件: INNER_THREADS 有効 && nthreads==1、または OMP_NUM_THREADS>=2 &&
nthreads==1)。実際の構成は runinfo の `n_julia_thread` で事後確認できる。
Julia スレッド並列区間では BLAS を 1 スレッドに落とし、抜けるとき元へ戻す
(二重並列の回避。設定値は runinfo の `blas_num_threads`)。
実行手順の実例は `docs/parton_run.md`。

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
  - **前提(v3.11 再確認)**: 「MF 以外のスロットの O が実数」。射影(Jastrow)の
    O_p = cnt_p は実数・虚スロット厳密 0 なので前提を満たす(§8-16-4 が恒久検証。
    v の実数性と Im 凍結は門番と flags 実体化が配線で保証)
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
- **SR 安定化の応急処置(v3.11、参照 chi-VMC 互換の最小セット)**。機構は
  REPORT §14: フェルミ準位の level crossing 近傍で ∂Φ ∝ 1/gap(パラメータ空間の
  準ノード。配置空間の「ゼロ踏み」とは別物で、**サンプル数に依存しない決定論的
  増幅**)→ sDiagMax ∝ 1/gap²(実測 1e14 まで)→ RedCut の相対閾値が健全な方向を
  大量カット(実測 82/130)→ SR が漂流。対処:
  (1) 契約 0′ の **gap_tol clamp**(|ε_n−ε_u| < 1e-8 の摂動項を 0。参照
  `vmc_chi_grad.jl` と同式・同既定値。厳密縮退でも Inf/NaN を出さない)、
  (2) **NaN/Inf ゲート**(SR 更新後のパラメータに非有限が混入した iter は
  pack/unpack スナップショットで巻き戻し、MC だけ進める。判定は sync 後 =
  全 rank 同一決定)、(3) **RedCut は 1e-6 を推奨**(参照の「必須」値。
  fixture 既定も 1e-6 へ。ライブラリ既定はもともと 1e-6)。
  §8-17 が恒久検証。**根治は §10 の未決事項**。
  **注(v3.12 の診断で訂正)**: 上の「RedCut が健全方向を大量カット → SR が漂流」は
  **症状の記述であって主犯ではない**。真の主犯はアウフバウ占有規則による枝の
  不連続な乗り換えで、RedCut を撤廃しても軌跡は変わらない。「漂流」の実体も
  **凍結**(発症すると 1 step の α 変化が 3.5e-8 に落ちる)。証拠は REPORT §15
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
16. **物理密度 Jastrow**(v3.11 追加、恒久): (1) 無効時の恒等(n_proj = 0 でフックが
   厳密 0・カウンタ no-op。実 run のバイト一致は v3.10 出力との突き合わせで実測済み)、
   (2) 差分 vs 全数 — `parton_update_proj_cnt!` の多数回適用 = `parton_make_proj_cnt!`
   の全数再構築、`parton_log_proj_ratio` = 全数 lnP の差(自己項の取り違えはここで
   落ちる)、(3) 有限差分 — ∂E/∂v_p が力ベクトルと一致(全数展開、射影 + MF の
   全実自由度)、(4) 共役シムの前提 — 射影 O の虚スロットが厳密 0、S が共役の有無で
   不変、(5) 変分改善 — v = 0 からの厳密勾配ステップで E_var が厳密に下がる、
   (6) QP 併用 — n_qp = 2 で 2〜5 が成立(P_J が qp と直交)、(7) F = 2 / 3、
   (8) Gutzwiller / 複素 v / 不完全 idx の拒否、(9) 既存全緑。
   実装: `MVMCOptimizers.jl/test/test_parton_jastrow.jl`

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

17. **SR 安定化**(v3.11 で 17-1/17-2、v3.12 で 17-3 以降を追加。恒久)。
   v3.11 の応急処置(`test_parton_srguard.jl`): (17-1) 契約 0′ の gap_tol clamp —
   厳密な縮退でも ∂Φ が有限、(17-2) NaN/Inf ゲートの部品(pack が検出・unpack が復元)。

   v3.12 の**占有追跡**(`test_parton_occmode.jl`。機構は REPORT §15):
   (17-3) `PartonOccMode` の既定 0・パース・2 以上の拒否、
   (17-4) 選択則 — 部分空間の重なりで選ぶ / 参照なしは aufbau / 昇順で返す /
   線形結合でも部分空間を拾う / 位相不変 / **決定論的タイブレーク**(縮退でも同じ),
   (17-5) 契約 0 — 占有集合が状態に入る、厳密な交差で aufbau は乗り換え mom は追う、
   **健全域では両者がビット一致**、
   (17-6) `min_gap` の符号つき定義(aufbau で従来と一致、非アウフバウで負)、
   (17-7) 契約 0′ — 非アウフバウ占有でも `Φ† ∂Φ = 0` と「∂Φ が非占有部分空間に収まる」
   が成立(占有集合を取り違えるとここが破れる)、(17-7a) build が modpara を写す、
   (17-8) `_pmfocc` 出力と `pmfband` の occupied 列、(17-9) ドライバの出力と整合、
   (17-10) 終端の自己完結性検査、(17-10b) 検査結果が runinfo に残る、
   (17-11) 診断列 `n_occ_deviation` / `principal_angle_max`、
   (17-12) **決定性** — 同一入力・同一シードで占有履歴を含めバイト一致、
   (17-13) 高速路との整合(フレーバー対称なら占有も一致。崩れたら落とす)。

   **既定のビット一致**は変更前の worktree(HEAD)と同じ入力を回して実測で確認する
   (手順は REPORT §16)。数値・再現に関わる出力(`zvo_out` / `zvo_var` / `zvo_SRinfo` /
   `zqp_*`)がバイト一致し、差分は診断列と runinfo の追加行のみであること。

18. **`InPmfOcc`(占有の読み戻し)**(v3.13 追加、恒久。`test_parton_inpmfocc.jl`):
   (18-1) 形式 — ファイルが無ければ `nothing`(既定挙動不変)、`zqp_pmfocc_opt.dat` を
   そのまま渡せて 1-based・昇順に正規化される、
   (18-2) 検証の発火 — 行数不一致 / band 範囲外 / flavor 範囲外 / 重複 /
   フレーバーあたり本数不一致 / ヘッダ件数の食い違いの 6 ケースで**計算開始前に**
   エラー停止、
   (18-3) `forced_occ` — 選択則を上書きして初期占有を採用、以後は通常の選択則に戻る、
   mom はこれを最初の参照として枝を追う、
   (18-4) **往復(本命)** — `PartonOccMode = 1` の run の α と占有を入力にした継続 run で、
   初期占有が前 run の終端占有と一致し **Φ がビット一致**する
   (`(α*, O*)` で状態が閉じたことの直接検証)、
   (18-5) 非アウフバウ占有の往復 — aufbau 初期化では再現できない状態が再現できる、
   (18-6) aufbau との併用 — 初期占有は `InPmfOcc` だが以後は選び直される(仕様どおり)。

   **`InPmfOcc` を与えない run は v3.12 とビット一致**(commit `c2431f5` との出力比較で実測)。

## 9. マイルストーン

- **M1**(完了): 一般 F 構造での ParaOpt 初点火(射影なし・n_qp=1・複素 1 変種・直接 SR)。
  Done: §8 の 0〜9 全緑(F=2/F=3 の ED 一致含む)。
  テストは `MVMCOptimizers.jl/test/test_parton_*.jl` と
  `MVMCExpertModeParsers.jl/test/test_parton_*.jl`
- **M2**(完了): 運動量射影 ON + 相関因子。
  前半(v3.6〜v3.8): qp 写像の base 正規化・ParaQPTrans 虚部・gather の向きの
  決定論的検証(§8-11)・QP 門番(§8-12)・CB 模型の QP 構成と変分的性質(§8-13)。
  実測: `:orbit` では射影が全シードで改善し Kx=0 が最良。`:orbit_flavor` +
  ランダム初期値では射影 ON が劣化するが、4 系統の切り分けで実装バグではなく
  最適化ランドスケープの問題と確定(REPORT §12-4。ウォームスタートなら留まる)。
  後半(v3.11): **物理密度 n^b ベースの Jastrow**(§1.1 / §2.3.2 / §8-16)。
  既存 Gutzwiller は固縛で自明化するため門番が理由つきで拒否
- **M3**: PhysCal、log 空間 det、NSplitSize>1、
  eigen ワークスペース化、命名・負債整理(返済トリガー: M1 全緑)

## 10. リスクと未決事項

- [x] 物理密度 Jastrow の設計 → v3.11 で実装(§1.1 / §2.3.2 / §8-16)
- [x] **gap 崩壊による SR 漂流の根治** → **v3.12 で占有追跡(`PartonOccMode = 1`)を実装**
  (§1.1 / §2.1 / §2.5 / §8-17)。既定は `0 = aufbau` で v3.11 とビット一致。
  以下は診断で確定した「やらないことにした対処」と、その理由:
  - **RedCut の置換は入れていない**(実測に基づく判断)。MOM 有効時の RedCut 発火は
    0/400 step で、RedCut 撤廃単独では軌跡が変わらない(−15.3462 → −15.3411)。
    **現時点で測定可能な利得がゼロ**。必要になったときの実装は設計済みなので安い:
    - 案 1(RedCut 撤廃)と案 2(死床 + 均衡化 + Tikhonov)は**数値的に完全同一**
      (cos = 1.0000)。乗法 StaDel と「均衡化 + 一様 Tikhonov」の同値性の実証
    - **死床は発火しない**(`S_aa < 2·SE` でも median 相対でも切る本数 0)。ゲージ
      平坦方向は**群全体の実数倍**なので座標軸に乗らず、対角に現れない。
      `D^{-1/2}` が死方向を増幅する懸念も該当しない
    - `SE(S_aa)` は既定の store 経路(`NStoreO ≠ 0`)ならサンプル毎 O があるので
      追加累積器は不要。非 store 経路のみ `Σ_s w_s |O_a|⁴` の累積器が 1 本要る
    - 発火が実際に観測されたら実装する。**勾配式に触れないので §8 の等式テストは無傷**
  - **共変トラスト領域 `δ†Sδ ≤ r²` は不採用**。S は現在地の局所計量なので、幅
    `w ~ gap` の境界層をその 300 倍先から予測できない(実測でも破綻直前の FS ステップ長は
    直前 step と同サイズ)
  - **絶対閾値は不採用**(単位・テンプレートごとの自然スケール差で壊れる)
  - **分母正則化 `D → D/(D²+ε²)` は不採用**。目的関数が整数占有のままで勾配だけ
    差し替えるのは「別の汎関数の勾配で元の汎関数を下る」ことになり誤り。正統化には
    目的関数ごと平滑化(有限温度占有)が必要で重い。**M3 候補として記録のみ**
- [x] **`InPmfOcc`(占有の読み戻し)** → **v3.13 で実装**(§2.3.1 / §3.3.1 / §8-18)。
  §2.5 の終端検査が `occ_selfcontained = 0` を返した(占有の 67〜78% の step が
  非アウフバウ、終端の `occ_min_gap` も負)ため必要と実測された。読み戻しは
  **初期占有としてのみ**で、全 step 固定の `explicit`(= `PartonOccMode = 2`)は
  **作っていない** — ウォームスタート / 鎖方式には初期占有だけで足りる。
  値 2 は予約のままで門番が拒否し続ける。
  併せて、`aufbau` の run でも `occ_min_gap` が 3.7e-08 のような値だと形式的には
  `selfcontained = 1` でもバンドが分離せず占有を復元できない点に注意(この意味では
  MOM の終端の方が状態として健全)
- [ ] **盆地選択の手当て(MOM のスコープ外)**: seed 11272 はコールドスタートで
  E ≈ −14.1 の別盆地に入り、MOM はその中で安定化するだけ(REPORT §16-4)。
  aufbau の枝の乗り換えには「悪い盆地から偶然抜ける」働きもあったので、MOM は
  漂流を止める代わりに探索も止める。必要なのは初期値・アニーリング・
  ウォームスタート(§13 で実証済み)であってソルバの修理ではない
  - **機構の記録(REPORT §15 で確定。§14 の RedCut 主犯説は訂正)**: 真の主犯は
    **アウフバウ占有規則による枝の不連続な乗り換え**。level crossing は変分多様体内では
    回避型なので Φ(α) 自体は連続だが、混成が効く境界層の幅は `w ~ gap` しかなく、
    gap ~ 5e-6 に対し SR のステップ長 0.0017 = 層の 300 倍以上。離散ステップから見ると
    占有スライスが飛び、E が +1.1 跳ぶ(主角 89.99° と同期。実測)。RedCut の大量カットは
    **症状**で、撤廃しても軌跡は変わらない(発症後に α を凍結させて回復を妨げる共犯)。
    「漂流」の実体も**凍結**(発症期の 1 step の α 変化が 3.5e-8)
  - **条件数説は棄却**(Tikhonov 後 cond(C) = 4.6e3)。**セクター説は適用外**
    (H_MF が並進非対称・充填非整数で Chern 数が定義できない)
- [ ] **S 行列の定義差(上流規約、既存 mVMC 由来)**: `build_s_matrix_and_g_vector!` は
  `S_ab = Re⟨O_a O_b*⟩ − Re⟨O_a⟩Re⟨O_b⟩` と組むが、実パラメータの Fubini–Study 計量は
  `Im⟨O_a⟩Im⟨O_b⟩` の項も引く。射影 O が実数・f_ij が正則な既存モードでは
  `Im⟨O⟩ = 0` で差は出ないが、**MF ブロックは非正則**なので残る(実測で相対
  1.3e-4〜1.5e-1、REPORT §15-7)。既存 run の再現性に関わるので本作業では変更しない
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

- v3.14 (2026-08-18, fixture の向き正準化): **`parton_fixture` のボンド係数の向きを
  idx クラス代表に揃えた**(バグ修正)。`bond_class` は無向ボンド(`i < j` 正規化)の
  クラス代表を `min(ki, kj)` で選ぶが、pmftrans の係数 t はリスト向きのまま書かれて
  いた。並進コピーで端点の大小が入れ替わるボンド(8×8 ef4 で 1536 行中 624 行)は
  t が共役側で載り、組み立て規約「リスト向きで α·t + h.c.」の下で同一 idx に α·t と
  α·conj(t) が同居 — **α が複素だと拡大セル並進が破れる**(8×8 ef4 の実 run で
  y 方向残差 0.6〜0.8、スペクトル差 0.33 を実測)。α = 1(実数)では h.c. と合流して
  同一の H になるため初期ハミルトニアンの検査では発見できず、SR が α を複素平面へ
  動かして初めて発症する。**変分多様体そのものが意図した ef4/xexet2(並進共変)と
  違っていた**ことになる。修正は fixture 側のみ(本体 `parton_build_mf_templates!` の
  契約は無罪)。回帰テスト `test/physics/test_fixture_orientation.jl`(複素 α で
  拡大セル並進の厳密不変性 + α=1 で cb 模型の再現 + 破っている並進の検出力)。
  P2-5 は修正で H_MF が保存並進を厳密に保つようになり空 k セクター(重み 0)が
  現れるため、e の検査を w > 0 のセクターに限定するよう適応した。
  **同族バグの前例**: 参照VMC の Jastrow idx(2026-08-08 発見、
  `symmetrize_jastrow_idx` で修正済み)。「端点の大小で決まる量」は並進共変性を
  壊す、という教訓の 2 例目。参照VMC の orbital(f_ij)は全順序対リスト + 固定係数
  なしなので構造的に免疫、upstream にも波及しない(pmftrans/pmfpara は fork 追加)。
  発見の経緯と解析ツール(`tools/parton_bands.jl` / `tools/parton_band_chern.jl`)は
  `playground_nozomi/cb_nu12_boson/PROJECT.md` の 2026-08-18 メモを参照
- v3.13 (2026-08-14, 占有の読み戻し): **`InPmfOcc`** を実装(§2.3.1 / §3.3.1 / §8-18)。
  v3.12 の終端検査が `occ_selfcontained = 0` を返した(REPORT §16-5)ため、
  `PartonOccMode = 1` の run は α\* 単独では状態を再現できない。これを埋める。
  - 作法は `InPmfPara` と同じ: **ファイルの有無がスイッチ**、modpara にキーを足さない、
    適用はパートンドライバ。形式は `zqp_pmfocc_*.dat` と同一で**出力をそのまま渡せる**
  - 効くのは**初期占有としてのみ**。以後は `PartonOccMode` に従う(`mom` はこれを
    最初の参照として枝を追うので、継続 run が前 run の枝を引き継ぐ)
  - 全 step 固定の `explicit` は**作らない**。ウォームスタート / 鎖方式には初期占有の
    読み戻しで足り、必要になってから追加する(値 2 は予約のまま)
  - `parse_input_parameter_file` は `idx Re Im` の 3 列専用なので流用できない。
    既存の汎用ヘルパ(`read_def_file` / `clean_line` / `split_def_line`)を組み合わせた
    最小のリーダを fork 所有ファイルに置く。**新規パーサファイルは作らない**
  - 検証(行数・範囲・フレーバーあたり本数・重複・ヘッダ件数)は門番
    `validate_parton_occupation` が計算開始前に一手に行う。**部分適用しない**
  - `InPmfOcc` を与えない run は v3.12(`c2431f5`)と**ビット一致**を実測
- v3.12 (2026-08-14, SR 漂流の根治): **占有集合を状態の一級市民にした**(§1.1 / §2.5 /
  §3.3.1 / §8-17)。`PartonOccMode` = 0 aufbau(既定、v3.11 とビット一致)/ 1 mom
  (前ステップの占有部分空間との重なりで選ぶ)。値 2 以降は予約で門番が拒否。
  - **主犯はアウフバウ規則による枝の乗り換え、RedCut は症状**(REPORT §15 で確定。
    §14 の主犯説を訂正)。混成が効く境界層の幅 `w ~ gap` に対し SR のステップ長が
    300 倍以上あるため、離散ステップから見ると占有スライスが不連続に飛ぶ
  - 契約 0/0′ は「1:Ne / Ne+1:NSite」から**占有集合 O 経由**へ。`min_gap` は
    「占有↔非占有の差」で**符号つき**(aufbau では従来と一致)
  - **既定経路は添字を UnitRange のまま渡す**。`@view U[:, occ]` を Vector{Int} で
    作ると非連続 view になり `mul!` が BLAS gemm から generic matmul へ落ちて総和順が
    変わる(実測: E の最終桁が 1e-15 ずれた)。値が同じでも演算列が変わるので、
    aufbau と MOM で経路を分けている
  - 占有は `.def` 族 `zqp_pmfocc_{init,opt}.dat` に出力。`(α*, O*)` の組で状態が閉じる。
    読み戻し(`InPmfOcc`)は**作っていない** — 要否は §2.5 の終端検査が決める
  - **RedCut の置換は入れない**(MOM 有効時の発火が 0/400 step、撤廃単独では軌跡が
    変わらない = 測定可能な利得がゼロ)。設計は §10 に記録して発火観測時に実装
- v3.11 (2026-08-13, M2 完了 + SR 応急処置): **物理密度 Jastrow**(§1.1 / §2.3.2 /
  §8-16)。P_J = exp(Σ_{i<j} v n^b n^b)を n^b で定義(上流の x = n−1 やパートン和
  F·n^b は v の意味が F 依存になるため不採用。構造規約 — Σ_{i<j}・自己項なし・
  対称 idx・符号 + — は上流と同一。係数の辻褄合わせはしない)。
  P_J は qp と直交(配置のみ依存)。比は純粋関数 O(Ne)、カウンタは契約 5 と
  サンプル保存のみが読む。受理は上流と同じ「log をまとめて 1 回 exp +
  !isfinite → 棄却」で、**Jastrow なしは従来式そのまま = v3.10 出力と全ファイル
  バイト一致を実測**。Gutzwiller は固縛で自明化するため明示エラー。
  既存経路(パーサ / InJastrow / SR 書き戻し / MPI 同期 / 出力)を再利用し
  **新しい登録点はゼロ**。共役シムの前提(射影 O 実数)は門番 + Im 凍結 +
  虚スロット 0 の 3 点で配線保証(§8-16-4)。
  **最終検証**(CB transJastrow・拡大セル (2,0)・Kx=Ky=0・nqp=2・sample 2000):
  コールドスタートの同時最適化は SR 漂流(下記)に嵌るが、**ウォームスタート
  (nojast 最適 α + v=0 開始)で E_tail100 = −16.2636 < 対照 −16.2572 <
  ベースライン −16.2554** — Jastrow の変分改善を実 run で確認(REPORT §13)。
  **SR 漂流の機構特定**(REPORT §14): フェルミ準位 level crossing で
  ∂Φ ∝ 1/gap(パラメータ空間の準ノード。配置空間のゼロ踏みとは署名で区別、
  サンプル数に依存しない)→ sDiagMax ∝ 1/gap²(実測 1e14)→ RedCut が健全な
  方向を大量カット(82/130)→ 漂流。参照 chi-VMC も同現象に遭遇しており
  (diag_max 1e12、n_cut 101/184)、clamp + NaN ゲート + red_cut 1e-6 で
  乗り切っている。**応急処置として同じ最小セットを実装**(§7 / §8-17)。
  根治(gap トラスト領域等)は §10 の未決事項
- v3.10 (2026-08-13, 出力整合): **zvo_parton_time の形式確定 + 並列メタデータ +
  実行環境の警告**(§3.3.1 / §3.3.2)。
  (1) C 版の per-step 出力 `<head>_time_<idx>.dat`(`vmcclock.c` `OutputTime`)を
  調査 — 実体は**受理率 3 種(hop/ex/lsf)+ 試行カウンタ + ctime 文字列**で、
  経過秒もサンプリング量も持たない。hop/ex/lsf はパートンに存在せず
  (固縛ホップ 1 種)、受理率・試行数は `zvo_parton_diag.dat` が既に担うため
  **列構成は踏襲しない**(エスカレーションの上で決定)。time は「時間と仕事量」に
  専念: `step step_sec cumulative_sec n_out n_in n_sample_total n_update_total`。
  n_sample_total(統計量)と n_update_total(仕事量)は役割が違うので両方持つ。
  n_out の式の家は `parton_n_out`(サンプリング本体と writer が共用、burn_flag を
  立てる**前**に読む)。旧 DESIGN 注記「C 版にステップ毎出力は無い」は不正確
  だったので訂正。
  (2) runinfo に並列構成 7 キーを追加(既存 n_rank / n_thread は明示的な
  n_mpi_rank / n_julia_thread に**置き換え** — 二重管理しない)。データ行は
  既存どおり素の `key value`(指示の例示は `# key value` だったが、既存 runinfo の
  規約と §8-10-5 のパーサが `#` をコメントとして読み飛ばすため、既存規約を優先)。
  (3) 起動時警告: `OMP_NUM_THREADS` では Julia のスレッド数は決まらない —
  C 版ジョブスクリプト流用の無警告 1 スレッド事故を起動時に `@warn`。
  Julia スレッド並列区間で BLAS を 1 に落として復元。`docs/parton_run.md` に
  Slurm 例(README から 1 行参照)
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

## 12. Φ 直接変分モード(v3.15 草案。実装前)

> **第 1 段は §12.11 の上流型の簡素版で実装する**(2026-10-31 決定)。正準化と案 B は後回し。

### 12.0 目的と位置づけ

CFL(c = f₁f₂f₃。f₁, f₂ に Chern バンド、f₃ にゼロ磁場のフェルミ海)を開殻のまま扱いたい。
α モードは H_MF(α) の固有ベクトルを Φ とし、∂Φ/∂α を一次摂動 `U_unocc·(W ./ D)` で出す(§1.5)。
開殻では D → 0 となり、`gap_tol` によって勾配が黙って 0 になる(`parton_orbital.jl:603-606`)。

そこで Slater 行列式の軌道行列 Φ_f そのものを変分するモードを足す。数式と数値検算は
設計メモ「Φ 直接変分の二案」(claude.ai アーティファクト、2026-10-23)の §1〜§9 にある。

- **案 A(direct)**: Φ_f の全要素(F·Ns·Ne 個の複素数)を変分する
- **案 B(thouless)**: Φ = Φ_ref + U⊥Z とし、Z ∈ C^{(Ns−Ne)×Ne} を変分する。Z は毎ステップ 0 に戻す。
  α モードの式から W と 1/D を外した形そのもので、GL(Ne) の冗長性を持たない

制約(従来どおり):

- **`PartonOrbParam = 0` の既定挙動はビット一致**
- マルコフ連鎖の乱数消費順を変えない
- 勾配等式テストは式が変わるので、解析式で立て直す

### 12.1 決定事項(2026-10-23)

| 項目 | 決定 | 理由 |
|---|---|---|
| modpara キー | **`PartonOrbParam`**(0 = alpha〔既定〕/ 1 = direct〔案 A〕/ 2 = thouless〔案 B〕) | 切り替えるのは行列式の計算法ではなく、軌道のパラメータ化だから(仮名 `PartonDetCalcMode` から改名) |
| 案の位置づけ | **本番は案 B**。案 A は小さい系の検証と基準値用 | 案 A の S は GL(Ne) 方向に厳密な零空間を持ち、`DSROptStaDel = 0`(既定)では Cholesky が破綻する(§12.2) |
| S 行列の定義 | **上流の規約のまま**(Im⟨O⟩Im⟨O⟩ の項なし) | g は正しく、収束先は変わらない。計量に全体位相への rank-1 の正のペナルティが足されるだけ。上流の f_ij と同じ前例。共通の SR に手を入れない |
| α 解からの開始 | **tools の変換スクリプトだけ**。α モードの出力は一切変えない | pmftrans.def・zqp_pmfpara_opt.dat・**zqp_pmfocc_opt.dat**(MOM の run は aufbau で取り直すと Φ が変わる)から Φ を組み直して InPmfOrb.def を書く |
| 状態のフィールド名 | `PartonOptimizationState` を型パラメータ化し、**フィールド名は `mfham` のまま** | テストの約 300 行を無変更で済ませる。Φ モードでも名前が `mfham` になる点は docstring で補う。**後で改名する可能性あり** |
| Φ の大きさ | **D_AmpMax と同じ形**。毎ステップ、フレーバーごとに Φ_f ← (D / max_{s,n}\|Φ_f[s,n]\|)·Φ_f。D は modpara キー **`PartonOrbAmpMax`**(Φ の要素の最大振幅、既定 4.0 = 上流の D_AmpMax と同じ値) | §12.4。上流は Pfaffian が 1 つなので全体で 1 つのスケールだが、ここは det が F 個あり、それぞれに定数倍の自由度がある |
| 最終平均 | 作らない。α と同じく**最終値**を出す | このコードは平均を実装していない(§12.2) |

### 12.2 調査で確定した事実(設計の前提)

| 事実 | 根拠 | 設計への影響 |
|---|---|---|
| **最終平均は実装されていない**。`store_opt_data!` の書き先 `state.opt_data` を読む箇所がなく、永続化は最終値 | `data_io.jl:46-50`、テスト `test_parton_initial_params.jl:207-214` | 射影子平均の専用経路は作らない。`parton_output.jl:348-351` と `parton_run_para_opt_from_namelist.jl:56-61` の「平均」というコメントは実態と食い違う |
| **共役シムの理由は E_loc の規約差**。上流の e は `conj(z/ip)` で `HO += e·O` と蓄積するが、パートンの E_loc は比をそのまま足す。シム後のスロット (O*, (iO)*) で g_x, g_y が正しい勾配になることを導出して確認した | `vmc_main_cal.jl:422, 3410`、`parton_vmc_main_cal.jl:90, 98`、`stochastic_opt.jl:401-404` | **正則な Φ スロットにも共役シムが必要**。詰め方は (O, i·O)。§7 と `_parton_conjugate_mf_slots!` の docstring にある「非正則だから」という説明は、成り立つ範囲が限られる |
| S = Re⟨O_iO_j*⟩ − Re⟨O_i⟩Re⟨O_j⟩(Im⟨O_i⟩Im⟨O_j⟩ の項なし) | `stochastic_opt.jl:386` | §12.1 で上流の規約のままと決定 |
| 案 A の GL(Ne) 方向は O がサンプルごとに定数(恒等式 **Φᵀ·G = I**。転置で随伴ではない)なので S の厳密な零空間。RedCut は対角で判定するので検出できない。`DEFAULT_DSR_OPT_STA_DEL = 0.0` | パーサ `utils/constants.jl:132`、`stochastic_opt.jl:491-531` | 案 A は門番で `DSROptStaDel > 0` を必須にする |
| 正規直交な Φ では \|det X_f\| が Ne とともに指数的に小さくなる。F=3 で 12×12 から `abs(ip) < 1e-30` の除外に全サンプルが掛かる | §12.4 の実測、`parton_vmc_main_cal.jl:494, 591` | Φ の大きさの規約(§12.4) |
| modpara の未知キーは**何も言わずに無視される** | パーサ `parsers/modpara_parser.jl:228`(warnings は表示されない) | キー名を打ち間違えると mode 0 に落ちる。α の門番で打ち間違いを案内し、runinfo にモードを記録する |
| 範囲外の OptFlag は**黙って「凍結」**と読まれる | `stochastic_opt.jl:21-26` | Φ 用のフラグ実体化が必須。流用すると Φ ブロックが丸ごと最適化されない |
| `parse_input_parameter_file` は読めない値を 0.0 にし、NaN を通し、件数の不一致を警告で済ませる | パーサ `utils/read_input_parameters.jl:20-71` | InPmfOrb は厳格な専用リーダで読む |
| スレッド並列の測定経路は、6 引数コンストラクタで状態の外箱を作り直す | `parton_vmc_main_cal.jl:488-489` | 外箱に Union フィールドを足すと、そこで Φ が落ちる。型パラメータ化する(§12.5) |
| 錨(各ステップ冒頭の厳密再計算)の後に ip ≠ 0 の検査がない | `parton_vmc_sampling.jl:370` | Φ モードは更新幅が大きいので、Φ ループ側で検査を足す |

### 12.3 棚卸し表

「流用はできるが物理的な意味が違う」箇所の一覧。判定の記号:

- (a) そのまま流用
- (b) アクセサや型の拡大で流用
- (c) 新しい型・変数・ファイル
- (d) Φ モードでは使わない、または門番で拒否

#### 12.3.1 状態

| 名前 | α モードでの意味(書き手 ← 何から決まるか) | 案 A | 案 B | 判定 | 理由 |
|---|---|---|---|---|---|
| `PartonMFHamiltonian`(`parton_types.jl:74-134`) | H_MF・固有系・占有・∂Φ・ゲージ群の器 | 存在しない | 存在しない | (c) 別の型 | ゼロ埋めのダミーは使わない |
| `.orbitals[f]` | H_MF(α) の固有ベクトルのうち占有 Ne 本。**導出量**(契約 0 `parton_orbital.jl:451` ← α, occ) | 正準化した Φ_f。主状態から毎ステップ導出 | 正準化した c_f·Φ_ref,f | **(b)** 3 つの型が共通に持つフィールドとし、振幅系の契約にする | 振幅系(契約 1・2・3・E_loc)は `.orbitals[f]` しか読まない |
| — | — | raw Φ = SR 値そのもの(data 側) | raw Φ(状態側。δZ を足した結果) | (c) | **永続化するのは raw Φ だけ**。毎ステップ先頭で正準化する。ランク間の一致とリスタートのビット再現のため |
| `h_mf / eig_vals / eig_vecs` | H_MF の固有系 | なし | なし | (d) | |
| — | — | — | U⊥,f(Φ_ref の正規直交補空間) | (c) `PartonThoulessPhi.u_perp` | α の `eig_vecs[:, unocc]` は H の固有ベクトルで、U⊥ は任意の補空間。名前を共有しない。毎ステップ決定論的に作り直し、永続化しない |
| `dorbitals / dh_uo_scratch` | ∂Φ/∂θ(契約 0′) | 不要 | 不要 | (d) | O は直接計算する |
| `min_gap / occ / occ_mode` | gap と占有(aufbau/MOM) | 定義なし | 定義なし | (d) | |
| `flavor_symmetric` → `amp.n_stored` | det^F 高速路(`parton_vmc_sampling.jl:62`) | n_stored = F に固定 | 同左 | (c) Φ の構築関数が F を渡す | 高速路は Φ^(f) が全 f で同一と仮定する(`parton_vmc_main_cal.jl:184-201`) |
| `gauge_scale/shift_groups / gauge_target_norm` | α のゲージ平坦方向 | 該当なし | 該当なし | (d) | ゲージの種類が違う(Φ モードでは GL(Ne) とパートンゲージ) |
| `PartonAmplitudeData / SamplingWorkspace / Configuration` | det・逆行列・配置のキャッシュ | 同じ | 同じ | (a) | |
| `PartonMainCalThreadContext` | スレッド別の amp と O バッファ。再利用条件は len_o と n_proj | 同じ | 同じ | (a) | 1 run の中で len_o は固定。ただし案 A では `o_all`(len_o × NVMCSample)が大きくなる |
| `PartonOptimizationState.mfham`(`parton_types.jl:431`) | α の状態 | Φ 状態 | Φ 状態 | (c) 型パラメータ化。フィールド名は据え置き | §12.5 |

#### 12.3.2 振幅・E_loc・O

| 名前 | α モードでの意味 | Φ モード | 判定 | 理由 |
|---|---|---|---|---|
| 契約 1 `parton_recompute_amplitude_all!`(`parton_calculate_m_all.jl:9`)/ 契約 2 `parton_amplitude_ratio!`(`parton_vmc_sampling.jl:478`)/ 契約 3 `parton_update_amplitude!`(:528)/ 初期配置(:89) | `.orbitals[f]` から det・比・更新 | 同じ計算 | **(b)** シグネチャ 4 本の型注釈を `PartonOrbitalSource` に広げる。本体は無変更 | Julia は具体型で特殊化するので、mode 0 の演算列は同一 |
| 契約 4 `parton_local_energy`(`parton_vmc_main_cal.jl:69-103`) | E_loc = Σ t·比 | 同じ | (a) | |
| 契約 5 `parton_calculate_o!`(:135-212) | α の各方向について Σ dPhi·A⁻¹。Re と Im を別々に計算 | 案 A: O^A[s,n] に δ_{s,π(r_m)}·sgn·A⁻¹[n,m] を散らし書き / 案 B: O^B = **transpose(U⊥)**·G | (c) 新しいメソッド | 案 B を `U⊥'*G`(随伴)で書くのは誤り。§7 の「転置積、dot 禁止」と同じ罠 |
| Re/Im の詰め方 | 独立に計算(非正則) | (O, i·O)(正則) | (c) | |
| `_parton_conjugate_mf_slots!`(:351-362) | MF スロットを共役にする | **Φ スロットにも必要** | (b) 範囲を n_orb にして流用し、docstring の理由を直す | オフセットが n_proj 固定なので、門番で `parton_n_idx(data) == 0` を確認する |
| `_parton_project_gauge_from_o!`(:273-309) | α のゲージ方向を O から除く | 呼ばない | (d) | |
| `parton_fill_sr_opt_o!`(:227-254) | Jastrow → MF → 共役 → ゲージ射影 | Jastrow → Φ ブロック → 共役 | (c) 状態型でディスパッチする Φ メソッド | 呼び出し元(:502-505, :601-603)は `project_gauge`/`alpha` を渡すので、受け取って無視する。作業領域を共有の状態に置かず(スレッド競合)、`sr_opt_o` に直接書く |
| `abs(ip) < 1e-30` の除外(:494, :591) | 節上のサンプルを捨てる | 同じ | (a) | Φ の大きさの規約で桁落ちを防ぐ(§12.4) |
| 錨(`parton_vmc_sampling.jl:370`) | 各ステップ冒頭の厳密再計算 | 同じ | (a) | Φ ループ側で ip ≠ 0 の検査を足す |

#### 12.3.3 SR とパラメータの受け渡し

| 名前 | α モードでの意味 | 案 A | 案 B | 判定 | 理由 |
|---|---|---|---|---|---|
| 並び [射影 \| MF] | MF = α(idx 共有あり) | [射影 \| raw Φ] | [射影 \| Z] | (c) | |
| `_PARAM_PMF` ロケータ(`stochastic_opt.jl:69, 128, 195, 227, 251`) | `pmfpara_terms[].value` | — | — | (c) 新種別 `_PARAM_PMFORB` を**1 つ**足す | SR は数を読み書きするだけ。A と B の意味の差(絶対値か相対値か)は SR の後で扱う。種別を分けても振る舞いは変わらず、登録ハンクが倍になる |
| — | — | raw Φ の平坦ベクトル | Z の平坦ベクトル(毎ステップ 0) | (c) パーサ `ExpertModeData.pmforb_params::Vector{ComplexF64}` | SR から見えるのは data だけ。長さは状態を組む前に確定させ、以後は変えない |
| 開始位置 | `pmf_start = n_proj+n_rbm+n_orb+n_opttrans+1` | `pmforb_start = n_para − n_orb + 1` | 同左 | (c) | 末尾ブロックを前提にし、式の三重実装を避ける |
| `count_variational_parameters`(パーサ `utils/read_input_parameters.jl:163-171`) | + n_pmf | + length(pmforb_params) | 同左 | (b) | |
| `pack/unpack_parameters`(`parameter_sync.jl:229-254`) | 全ロケータ | 同じ | 同じ | (a) | |
| `parton_materialize_flags!`(`parton_vmc_para_opt.jl:121-150`) | 長さ 2(n_proj+n_idx)、オンサイトの Im を凍結 | Jastrow の Re は引き継ぎ・Im は凍結、Φ は全部可動 | 同左 | (c) Φ 版 | 範囲外のフラグは黙って凍結される |
| `validate_parton_opt_flags`(`parton_unsupported_inputs.jl:492-526`) | 長さと MF 全凍結の検査 | n_orb で検査 | 同左 | (c) Φ 版 | |
| `stochastic_opt!`(直接法) | S_ii *= 1+StaDel、RedCut、Phase 6 で `+= r_re + i·r_im` | 同じ | 同じ | (a) | |
| `parton_sync_parameters!`(`parton_vmc_para_opt.jl:24-40`) | bcast → α のゲージ射影 | bcast だけ使う | 同左 | (b) `mfham` を渡さない形で呼ぶ | |
| 正準化と引き戻し | — | 毎ステップ先頭で Φ = Löwdin(raw) → **data に書き戻す**(SR の δ が正準化した Φ に足されるように)→ 振幅スケール。SR 後は raw = data | 毎ステップ先頭で Φ_ref = Löwdin(raw)、U⊥ = 補空間、data の Z ← 0 → 振幅スケール。SR 後に raw ← Φ_ref + U⊥δZ | (c) `parton_orbparam_canonicalize!` | 案 B の閉じた式 (Φ_ref+U⊥δZ)(I+δZ†δZ)^{-1/2} と厳密演算では同じ。実際のグラム行列から作るので誤差が溜まらない |
| ランク間の一致 | α は毎ステップ同期するので一致する | 正準化は rank 0 で計算して bcast | 同左(Φ_ref と U⊥) | (c) | 「全ランクが同じ計算をすれば同じ結果」には頼らない(BLAS のカーネル差で ulp がずれうる) |
| NaN ゲート(`parton_vmc_para_opt.jl:257, 280-284`) | 同期とゲージ射影の後に判定して復元 | raw を更新する**前**に判定。Löwdin の λ_min/λ_max < tol でも棄却 | 同左 | (c) | 判断は bcast で揃える |
| `store_opt_data!` / `output_data!` / `output_opt_data!` の α ハンク(`data_io.jl:36, 138, 187, 195`) | α を zvo_var・zqp_opt・pmfpara_opt に出す | Φ は登録しない | 同左 | (d) | 案 A は列が巨大(CFL 8×8 で 1 行に 3072 複素)、案 B は常に 0。Φ は専用ファイルに出す |
| `output_opt_data!` 自体 | Jastrow を zqp_jastrow_opt.dat に永続化 | **Φ ループでも必ず呼ぶ** | 同左 | (a) | 呼ばないと Jastrow の最適値が残らない |
| `_duplicate_checked_sections`(`parameter_sync.jl:206`) | pmfpara の共有 idx の診断 | 該当なし | 同左 | (d) | |

#### 12.3.4 初期化と入力

| 名前 | α モードでの意味 | Φ モード | 判定 | 理由 |
|---|---|---|---|---|
| pmftrans.def / pmfpara.def | t と α | 使わない | (d) 門番 | Φ モードに α は存在しない |
| `parton_init_alpha!`(専用 RNG) | 値の無い α を乱数で埋める | なし | (d) | Φ の初期化は乱数を使わない(サンプリング RNG を消費しない規律を保つ)。**§12.12 で変更**: θ を初期化専用の乱数列で初期化する(サンプリング RNG は消費しない) |
| InPmfPara / InPmfOcc | α と占有の読み戻し | なし | (d) 門番 | |
| — | — | **InPmfOrb.def**(必須)。平坦添字 ((f·Ns)+s)·Ne+n、5 行ヘッダ、`%.18e` | (c) 厳格な専用リーダ | 件数・範囲・有限性・列フルランク(σ_min/σ_max > tol)を error で検査する。**§12.12 で変更**: InPmfOrb は任意で θ を持つ(idx は θ の番号)。列フルランクは組み立てた Φ について初期化後に検査 |
| `validate_parton_data`(`parton_unsupported_inputs.jl:288-454`) | pmfpara/pmftrans/physhop の必須検査と共通検査 | 必須検査だけを外す | (b) kwarg `require_mf_inputs` | physhop・Jastrow・DH・CoulombIntra・OptTrans の検査は再利用する |
| ドライバ(`parton_run_para_opt_from_namelist.jl:20-138`) | パース → α 初期化 → フラグ → 門番 → … | パース直後に Φ の入口へ分岐 | (b) 1 分岐 | α 初期化とフラグ実体化(:39-47)より前で分ける |
| α 解からの変換 | — | tools の変換スクリプト | (c) | §12.1 |

#### 12.3.5 出力・診断・タイマー・tools

| 名前 | α モード | Φ モード | 判定 |
|---|---|---|---|
| `zqp_pmfpara_*` / `zqp_pmfocc_*` / `zqp_pmfham_opt` / `zqp_pmfband_opt` / `zqp_pmfvec_opt` | α・占有・H_MF・固有系 | 書かない | (d) |
| — | — | `zqp_pmforb_{init,opt}.dat` = raw Φ(InPmfOrb としてそのまま読み戻せる) | (c) |
| `zvo_parton_diag.dat`(列 2 min_gap、列 10 occ dev) | 毎ステップ | 書かない | (d) |
| — | — | `zvo_parton_orbdiag.dat`: MC 健全性の列(受理率、試行数、need_recompute、厳密再計算)、ゲージ不変な Φ 指標(主角の最大値、‖P⊥δΦ‖、raw Φ の σ_min、‖raw†raw − I‖、パートンゲージの偏り)、log₁₀\|ip\| の最小と最大 | (c) 列の意味が違うのでファイルを分ける |
| `zvo_parton_runinfo.dat` | 開始時と終了時に "w" で書く(ドライバ :99, :132) | 既存 writer の後に Φ の項目(`PartonOrbParam`、`PartonOrbAmpMax`、n_orb、入力 Φ の正規直交からのずれ、PartonGaugeFix / FlavorSymFast は「該当なし」)を "a" で追記する。occ_* と n_idx は書かない | (c) |
| `zvo_parton_time` / `zvo_conv` / `zvo_SRinfo` / `zvo_out` | α に依存しない | 同じ | (a) |
| CalcTimer 801(契約 0)/ 802(契約 0′) | H 構築と ∂Φ | 存在しない | (d) |
| CalcTimer 803–813 | 共有関数の中で計測される | 同じ意味(809 = Φ 版の O) | (a) |
| — | — | **820 番台**: 820 正準化、821 状態の bcast、822 orbdiag。固定リスト(`parton_output.jl:402-417`)は 801/802 に 0 を出すので、Φ 用の writer を新規ファイルに置く | (c) |
| `tools/plot_parton_conv.jl`(diag の列 2・10 を読む。:105-109) | 動く | diag が無いと error になる | (b) orbdiag へのフォールバックを足す |
| tools の Chern・バンド道具 | pmfham・pmfband・pmfocc を読む | 入力が存在しない | 対象外。射影子 P = ΦΦ† から Chern を出す経路(`fhs_chern_projector`、`bott_index` は P 経由)は後日 |

### 12.4 Φ の大きさ(D_AmpMax と同じ形)

Φ_f → c_f Φ_f は GL(Ne) の一部で、det X_f が c_f^{Ne} 倍になるだけなので状態は変わらない。
上流の `sync_modified_parameter!`(`parameter_sync.jl:138-150`)は、SR 更新のたびに Slater の f_ij 全体を定数倍し、
max\|f\| = D_AMP_MAX = 4.0 にそろえる(Pf の全体スケールの固定)。Φ モードでも同じ位置(SR 更新のあと、次のサンプリングの前)で、
**フレーバーごとに** Φ_f ← (D / max_{s,n}\|Φ_f[s,n]\|)·Φ_f とする。D は modpara キー `PartonOrbAmpMax`(既定 4.0)。

なぜ要るか。正規直交な Φ では \|det X_f\| が Ne とともに指数的に小さくなる。フェルミ海(平面波)の軌道でランダムな 2 万配置を測った結果:

| 格子 | Ns | Ne | log₁₀\|det X\|、Φ†Φ=I(\|det\|² 重みの平均) | det が平均 1 前後になる D | D = 4 での 1 フレーバーの log₁₀\|det\| | D = 4 での F=3 の積 |
|---|---|---|---|---|---|---|
| 4×4 | 16 | 4 | −1.5 | 0.59 | +3.3 | 10¹⁰ |
| 8×8 | 64 | 16 | −6.7 | 0.33 | +17 | 10⁵² |
| 12×12 | 144 | 36 | −16.1 | 0.23 | +44 | 10¹³³ |
| 16×16 | 256 | 64 | −29.4 | 0.18 | +86 | 10²⁵⁹ |

- Φ†Φ = I のままだと、F=3 の積は 12×12 で約 10⁻⁴⁸ になり、`abs(ip) < 1e-30` の除外に全サンプルが掛かる(α モードも同じ規約なので、同じ潜在的な問題がある。ビット一致を守るため今回は触らない)
- D = 4(上流と同じ値)は 16×16 まで閾値にもあふれ(10³⁰⁸)にも当たらない。ただし 16×16 では余裕が小さく、それより大きい系では D を下げる
- 目安は **D ≈ 3/√Ns**。このとき \|det\| は系のサイズによらず 1 前後になる
- 上の表はフェルミ海の値。Chern バンドの軌道は局在しているので max\|Φ\| が大きく、同じ D でも det はもう少し小さくなる
- orbdiag に log₁₀\|ip\| の最小と最大を毎ステップ書き、閾値かあふれに 10 桁以内まで近づいたら警告する
- O と SR 更新は c_f に依存しない(案 B の O^B は定数倍で不変。案 A は毎ステップ正準化するので整合する)

### 12.5 型

```julia
abstract type PartonOrbitalSource end
# 契約: すべての具象型は orbitals::Vector{Matrix{ComplexF64}}(Ns×Ne、振幅系が読む Φ)を持つ

mutable struct PartonMFHamiltonian <: PartonOrbitalSource       # 既存。supertype を足すだけ

mutable struct PartonDirectPhi <: PartonOrbitalSource            # 案 A
    orbitals::Vector{Matrix{ComplexF64}}   # c_f·Löwdin(raw)。raw は data.pmforb_params にある
end

mutable struct PartonThoulessPhi <: PartonOrbitalSource          # 案 B
    orbitals::Vector{Matrix{ComplexF64}}   # c_f·Φ_ref
    phi_ref::Vector{Matrix{ComplexF64}}    # Löwdin(raw_phi)、正規直交
    u_perp::Vector{Matrix{ComplexF64}}     # Φ_ref の補空間。毎ステップ作り直し、永続化しない
    raw_phi::Vector{Matrix{ComplexF64}}    # 主状態(Z は data 側で毎ステップ 0)
end

mutable struct PartonOptimizationState{O<:PartonOrbitalSource}
    ...; mfham::O; ...   # フィールド名は据え置き。Φ モードでも mfham という名前になる(後で改名の可能性あり)
end
```

- **モードは状態型で表す**。整数キーは構築時に 1 回だけ状態型に変換し、以後は多重ディスパッチで分ける。
  シングルトン型は作らない(状態型と 1 対 1 で冗長)。if 分岐にすると分岐点が 6 箇所(準備・O・引き戻し・出力・終端検査・初期化)に散る。
  ディスパッチなら 1 モード 1 メソッドにまとまり、α の関数本体を触らずに済む
- **型パラメータ化を採る理由**(Union フィールド案との比較):
  - スレッド経路の `PartonOptimizationState(pstate.state, amp_t, …, pstate.mfham, …)` が O を推論するので、Φ が自動で伝わる
  - 「α と Φ のどちらか一方だけが存在する」ことが型で保証される
  - 型が安定する
  - 6 引数コンストラクタ(`parton_types.jl:446`)は注釈を広げるだけで済む
- α 専用のコードは `mfham.n_idx`・`.min_gap`・`.gauge_*` を読む。Φ 型にはこれらが無いので、Φ モードで誤って通ればエラーで止まる(安全側)
- 名前は上流の Slater(f_ij)と紛れないようにする(`slater_orbitals` は使わない)。接頭辞は `parton_` / `Parton`

### 12.6 ファイル構成

**新規ファイル**(`MVMCOptimizers.jl/src/`。抽象型だけは `parton_types.jl` の先頭に置く):

| ファイル | 責務 |
|---|---|
| `parton_orbparam_types.jl` | `PartonDirectPhi`・`PartonThoulessPhi`、モード定数、平坦添字 ⇄ (f, s, n) |
| `parton_orbparam_inputs.jl` | InPmfOrb の厳格な読み書き、Φ の門番、Φ 用のフラグ実体化と長さ検査、案 A の `DSROptStaDel > 0`、SROptData のメモリ見積もり |
| `parton_orbparam_o.jl` | O の共通カーネル((qp, f) ごとの w·Πdet/ip と A⁻¹)、案 A の散らし書き、案 B の transpose(U⊥)·G、`parton_fill_sr_opt_o!` の Φ メソッド、Φᵀ·G = I の検査 |
| `parton_orbparam_canonical.jl` | 正準化(Löwdin、U⊥、振幅スケール)、data と状態の同期、raw の更新、引き戻しの妥当性判定、rank 0 からの bcast、パートンゲージの監視量 |
| `parton_orbparam_output.jl` | `zqp_pmforb_*`、orbdiag、runinfo の追記、CalcTimer の Φ セクション |
| `parton_vmc_para_opt_orbparam.jl` | Φ 状態の構築(n_stored = F)、SR ループ、ドライバ本体 |

**Φ ループの 1 ステップ**:

1. 正準化(rank 0 で計算 → bcast)
2. サンプリング → main_cal → weight_average → `output_data!`
3. `stochastic_opt!` → info を bcast
4. `parton_sync_parameters!(data, ctx)`
5. NaN と妥当性のゲート(判断は bcast で揃える)
6. raw を更新
7. orbdiag を書く

ループの後に `output_opt_data!`(Jastrow)、`zqp_pmforb_opt.dat`、conv、CalcTimer を書く。

**既存ファイルへの変更**(すべてマーカー付き。転送・分岐・型注釈の範囲に収める):

| ファイル | 変更 |
|---|---|
| `parton_types.jl` | 抽象型、`PartonMFHamiltonian <: …`、状態の型パラメータ化、6 引数コンストラクタの注釈を広げる |
| `parton_calculate_m_all.jl`、`parton_vmc_sampling.jl` | シグネチャ 4 本の型注釈 |
| `parton_unsupported_inputs.jl` | α の門番で `PartonOrbParam ≠ 0` を拒否し、キー名の打ち間違いを案内する(pmfpara が空で namelist に InPmfOrb がある場合)。`require_mf_inputs` kwarg |
| `parton_run_para_opt_from_namelist.jl` | パース直後の 1 分岐 |
| `stochastic_opt.jl` | enum 1 行、foreach 2 本、get/set 2 本 |
| `MVMCOptimizers.jl` | include |
| パーサ `utils/constants.jl`、`types/expert_types.jl`(ModPara に 3 箇所)、`parsers/modpara_parser.jl` | 新キー `PartonOrbParam`・`PartonOrbAmpMax`(§3.1 の 3 登録点) |
| パーサ `utils/constants.jl` | `"InPmfOrb"` |
| パーサ `types/expert_types.jl`(ExpertModeData に 2 箇所) | `pmforb_params` |
| パーサ `utils/read_input_parameters.jl` | count に 1 行 |
| `tools/plot_parton_conv.jl` | orbdiag へのフォールバック |
| `tools/`(新規) | α 解 → InPmfOrb の変換スクリプト |

### 12.7 案 A と案 B で共有するもの / しないもの

- **共有する**: InPmfOrb の読み書き(どちらも raw Φ)、Löwdin、振幅スケール、振幅系、O の共通カーネル、共役シム、Jastrow の詰め方、正準化の bcast、NaN と妥当性のゲート、orbdiag の大半、ループとドライバの骨格
- **共有しない**:
  - SR 値の意味: 案 A は raw Φ そのもの、案 B は Z(毎ステップ 0)
  - raw の置き場: 案 A は data 側、案 B は状態側
  - U⊥ の有無
  - O の最終段: 散らし書きか transpose(U⊥)·G か
  - S の性質: 案 A は零空間を持つので StaDel が必須、案 B は持たない

### 12.8 門番(`PartonOrbParam` が 1 または 2 のとき)

| 対象 | 動作 | 理由 |
|---|---|---|
| `PartonOrbParam` の値が {0, 1, 2} の外 | エラー | 予約 |
| namelist に PartonMFTrans / PartonMFPara / InPmfPara / InPmfOcc がある | エラー | α を混ぜない |
| `parton_n_idx(data) ≠ 0` | エラー | 共役シムのオフセットの前提 |
| InPmfOrb が無い、または件数・範囲・有限性・列フルランクのいずれかを満たさない | エラー | 緩いリーダの穴を塞ぐ。**§12.12 で変更**: InPmfOrb が無いのはエラーでなく乱数初期値。列フルランクは初期化後の Φ で検査 |
| `PartonOccMode ≠ 0` | エラー | 占有の選択が存在しない |
| 案 A で `DSROptStaDel ≤ 0` | エラー | S の厳密な零空間 |
| `PartonOrbAmpMax ≤ 0` | エラー | |
| OptFlag の長さ ≠ 2(n_proj + n_orb) | エラー | 範囲外のフラグは黙って凍結される |
| SROptData のメモリ見積もり((2(1+n_para))² 複素)が閾値を超える | エラー | 案 A の 8×8 で約 270 MB。蓄積はサンプルごとに O(n²) の単スレッドループ |
| `PartonGaugeFix` / `PartonFlavorSymFast` | 拒否しない。runinfo に「該当なし」と記録する | どちらも既定値が 1 なので、明示的な 0 を要求すると全入力に 2 行を強制することになり脆い。Φ モードでは常に n_stored = F |
| 既存の共通検査(NSRCG = 0、NVMCCalMode = 0、ComplexType = 1 など) | 流用 | |
| α 側(mode 0)の門番 | `PartonOrbParam ≠ 0` で `parton_vmc_para_opt!` が直接呼ばれたらエラー | 誤解を招く「No mean-field parameter terms」を避ける |

### 12.9 テスト計画

設計メモ §8 の ①〜④ をもとにする。

| 番号 | 内容 | 方法 | 置き場 |
|---|---|---|---|
| Φ-0 | mode 0 のビット一致 | 既存スイートが全緑 + 変更前 worktree と E2E 出力(zvo_out・var・SRinfo・pmfpara_opt ほか全部)をバイト比較(REPORT_nu12_stuck.md:1064-1080 の手順) | 手動 + 既存 |
| Φ-0b | 型の拡大で振幅系が変わらない | α の orbitals から Φ 状態を作り、ip・比・更新・E_loc がビット一致 | 新 `test_parton_orbparam_types.jl` |
| Φ-1(①) | O と有限差分(案 A・B、QP と Jastrow 込み) | ln ip の中心差分。案 B は Φ_ref ± εU⊥[:,u]e_nᵀ。共役前に Im スロット = i·Re スロット。O^B = transpose(U⊥)·O^A、Φᵀ·G = I | 新 `test_parton_orbparam_o.jl` |
| Φ-2 | 力 = 勾配(解析式で立て直す) | 全配置を列挙し、上流の `calculate_oo!` と `build_s_matrix_and_g_vector!` を通した g/(−dt) を ∂E/∂ReΦ・∂E/∂ImΦ・∂E/∂Z の中心差分と比べる。共役を外すと崩れること、Jastrow 込みで S が不変なこと、store 経路と非 store 経路の両方 | 新 `test_parton_orbparam_force_gradient.jl` |
| Φ-3(②) | S のゼロ固有値の数 | 厳密な共分散で、案 A = F·Ne² + (F−1)(Ns−1)、案 B = (F−1)(Ns−1)(複素次元)、階数が一致すること。コードの S(Im·Im 項なし)で非零になる方向も数値で確認して記録する | 同上 |
| Φ-4 | 正準化と引き戻し | Löwdin と振幅スケールで E が不変、張る空間が不変。多数ステップ後も正規直交。案 B の補空間の直交性。続けて走った run と再起動した run がビット一致 | 新 `test_parton_orbparam_canonical.jl` |
| Φ-5 | 入出力 | `zqp_pmforb_opt.dat` → InPmfOrb の往復がビット一致。異常入力(件数、範囲、NaN、ランク落ち)を拒否(§12.12 以降、ランク落ちは初期化後の検査で拒否) | 新 `test_parton_orbparam_io.jl` |
| Φ-6 | 門番 | §12.8 の全行 | `test_parton_gatekeeper.jl` に追加 |
| Φ-7 | ロケータ | count、pack/unpack、get/set/delta、`pmforb_start` | `test_parton_locator.jl` に追加 |
| Φ-8 | 並列 | スレッド並列 = 逐次(ビット一致)。2 ランクでランクごとの Φ が一致 | `test_parton_threading.jl` に追加 + 手動 MPI |
| Φ-9(③) | 既存の厳密解を保つ | KM ν=1/2 4×4 を α 最適解から(変換スクリプト経由で)開始し、E = −4.0000・分散 ~0 のまま | `test/physics`(P 層) |
| Φ-10(④) | α より下がる | KM ν=1/3 9×4 を α の最良(−3.93989)から開始 | 物理検証 run(CI 外) |

実装の順番(TDD):

1. Φ-0b → Φ-7 → Φ-5
2. Φ-1 → Φ-2/3
3. Φ-4 → Φ-6 → Φ-8
4. Φ-0
5. Φ-9 → Φ-10

Φ-0 は、マージ前に毎回、変更前 worktree との全出力バイト比較で確認する。

### 12.10 未決・後回し

- **SR の蓄積コスト**: SROptData は (2(1+n_para))² の複素数を持つ。`calculate_oo!` はサンプルごとに O(n²) の単スレッドループ、`finalize_oo_store!` は素朴な三重ループ。8×8・F=2 で n_orb は案 A が 2048、案 B が 1536 で、大きい系では SR の蓄積が支配的になる見込み(推測)。必要になったら、Φ 専用の BLAS 版 finalize をディスパッチ経由で呼ぶ(mode 0 は従来のまま)
- **パートンゲージ**(Φ_f → D_fΦ_f、D_f は対角で Π_f D_f(i) = 1)は、案 A・B 共通で (F−1)(Ns−1) 個の冗長な方向になる。サンプルごとに O·G ≡ 0 が厳密に成り立つので、一次の SR 更新はこの方向に動かない。最初は orbdiag で監視するだけにし、偏りが育つのを観測したら SR 更新からゲージ方向を射影で除く
- **α モードの \|det\| の桁落ち**(§12.4)は既存の潜在的な問題。ビット一致を守るため今回は触らない
- **`mfham` フィールドの改名**(§12.5)
- **射影子 P = ΦΦ† による Chern・フラックス・バンドの道具**

### 12.11 第 1 段: 上流型の簡素版(2026-10-31 決定)

ユーザーの判断:「とりあえず Pfaffian と同じ設計でやってみて、致命的な問題が出たら変える」。
上流 mVMC が f_ij をそのまま変分し、D_AmpMax で全体スケールだけをそろえているのと同じ形で、
行列式の成分 Φ_f[s,n] をそのまま変分する。**§12.0〜12.10 と食い違う点はこの節が優先する**。

| 項目 | 決定 | §12 本体との差 |
|---|---|---|
| 実装するモード | `PartonOrbParam = 1`(案 A)だけ。値 2(案 B)は予約で、門番がエラーにする | 案 B は後回し |
| SR が動かす値 | raw Φ そのもの(`data.pmforb_params`)。振幅系は毎ステップ data から Φ を写すだけ | Löwdin による正準化をしない |
| 冗長性の処理 | 上流と同じく、SR のあとにフレーバーごとに Φ_f ← (D / max_{s,n}\|Φ_f[s,n]\|)·Φ_f とする。**上限で切るのではなく、毎回そろえる**(縮めるだけでなく拡大もする。`parameter_init.jl:274-276` の上流の規約)。D = `PartonOrbAmpMax`(既定 4.0)。倍率は正の実数なので全体位相は固定しない。GL(Ne) のうちスケール以外の方向(SL(Ne))は SR の正則化(`DSROptStaDel`)任せにする | §12.4 の正準化を外し、スケールだけ残す |
| 適用する時点 | 初期化(InPmfOrb を読んだ直後)と、毎ステップの SR 更新の後(bcast → NaN ゲート → 再スケール) | 上流の `sync_modified_parameter!` と同じ位置 |
| ランク間の一致 | bcast のあと全ランクが同じ再スケールを行う(上流と同じ。要素の絶対値と積だけなので決定的) | rank 0 で計算して bcast する正準化は不要 |
| 監視 | `zvo_parton_orbdiag.dat`: step、フレーバーごとの再スケール前の max\|Φ_f\|、log₁₀ cond(Φ_f)(σ_max/σ_min)、捨てたサンプル数 | 「致命的な問題」の判定材料。SL(Ne) 方向の漂流は cond に出る |
| 門番 | §12.8 のうち案 A に関わる行。`DSROptStaDel > 0` を必須とする(GL(Ne) の零空間) | |
| 後回し | Löwdin による正準化、案 B、パートンゲージの射影、BLAS 版 finalize、`tools/plot_parton_conv.jl` の orbdiag 対応 | |

**致命的な問題の目安**(観測したら §12 本体の正準化へ進む):

- cond(Φ_f) が step とともに単調に増える(SL(Ne) 方向の漂流)
- SR が NaN で止まる、または ip ≈ 0 で捨てるサンプルが増え続ける
- α の最良解から始めて、E が α より上がる。Φ モードの状態族は α モードの状態族を含む(同じ Jastrow・QP)ので、下がるか同じでなければならない
### 12.12 制限付き Φ(PartonOrbIdx:Pfaffian 互換 + Bloch 形)(2026-10-31 決定)

目的は**大規模系での変分パラメータの削減**。規約と機能は上流の Pfaffian(`orbitalidx.def` /
`InOrbital` / 乱数初期化 / D_AmpMax)にそろえる。§12.11 の上流型は「制限なし」の特別な場合として残る。

| 項目 | 決定 |
|---|---|
| 切り替え | `PartonOrbParam = 1` のまま、namelist に **`PartonOrbIdx`** があれば制限付き。無ければ恒等写像(各要素が独立)で、§12.11 の挙動とビット一致 |
| 対応 | Φ_f[s,n] = c·θ_idx。SR が動かすのは θ(`data.pmforb_params`、長さ NPmfOrbIdx)。Φ は毎ステップ θ から組み立てる導出量 |
| def の書式 | 5 行ヘッダ(2 行目 `NPmfOrbIdx N`、3 行目 `ComplexType`)のあと、全 F·Ns·Ne 要素について `site flavor orbital idx [c]`(0-based)。c は省略で 1、1 列なら符号 ±1(Pfaffian 互換)、2 列なら `Re Im`(\|c\| = 1 を要求)。続けて N 行の `idx OptFlag` |
| 網羅 | **全要素の記載が必須**。欠け・重複・範囲外・どの要素からも使われない idx は error。ゼロ固定は「OptFlag 0・初期値 0 の idx」で書く(Pfaffian と同じ) |
| 初期値 | OptFlag ≠ 0 の idx は乱数(Pfaffian と同じ (U[−1,1) + i·U[−1,1))/√2)、OptFlag = 0 は 0。乱数は**初期化専用の乱数列**(α の `parton_init_alpha!` と同じ base_seed の規律で、サンプリングの乱数列を消費しない)。`InPmfOrb`(`idx Re Im`、全 idx)があれば上書きする。**InPmfOrb は必須でなくなる** |
| 再スケール | idx を共有するフレーバーを 1 つの組にまとめ(連結成分)、組ごとに θ の最大絶対値を `PartonOrbAmpMax` にそろえる。共有が無ければフレーバーごと(§12.11 と同じ)。\|c\| = 1 なので max\|Φ\| = max\|θ\| |
| O | O_θ = Σ_{k ∈ idx} c_k·O_Φ,k。各寄与を写像経由で θ のスロットへ直接足す(作業領域を持たない)。恒等写像では従来の経路をそのまま通す |
| 門番 | 組み立てた Φ_f の列フルランクを、初期化(乱数または InPmfOrb)の後に検査する |
| Bloch def の生成 | `tools/parton_bloch_orbidx.jl`。`parton_lattice.dat`(`SITE i x y σ`・`MFSHIFT`・`grid`)と modpara(NElec・NFlavor)と占有指定から def を書く。Φ_f[s,(k,b)] = e^{ik·a(s)}·θ_{(f,σ,k,b)}(σ = 拡大セル内の位置、a(s) = 代表元から s へ移す MFSHIFT)。k は拡大セルの並進群の指標(N_cell 個)。既定の占有は全 k に Ne/N_cell 本(割り切れなければ error)。`flavor m1 m2 nbands` の占有ファイルで指定もできる(CFL のフェルミ海用)。physhop がその並進で不変でなければ警告 |
| α 解からの変換 | 変換道具に Bloch def を渡すと、α の射影子 P = ΦΦ† を k ごとのブロック P(k)(N_σ × N_σ)に分けて対角化し、固有値 ≈ 1 の固有ベクトルを u_{k,b} とする(最小二乗の近似を使わない)。本数が占有指定と合わなければ error |
| パラメータ数 | Bloch でフレーバーあたり Ns·Ne → Ns·Ne / N_cell。SR の ⟨O O†⟩ は n_para² で効く |
| 第 1 版の対象外 | 磁気並進(ゲージ変換を伴う並進)/ パートン境界条件のひねり θ_f / \|c\| ≠ 1 の係数 |
