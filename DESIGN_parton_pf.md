# DESIGN: 多フレーバー Pfaffian パートンモード(PartonMode = 2)

- ステータス: **P0(設計)草案 — 実装前**。2026-10-31 起草、ブランチ `parton-pf`(d7cc568 から分岐)
- 位置づけ: `DESIGN_parton.md` の別冊。本体の §1〜§7 の規約(登録点・マーカー・命名・数値ポリシー)は
  すべてここでも有効で、**食い違う点だけをこの文書に書く**。Φ 直接変分(本体 §12)が落ち着いたら
  本体の §13 へ統合するかを判断する(ユーザー判断 2026-10-31: まずは別ファイル)
- 行番号はすべて d7cc568 時点のもの

## 0. 目的と位置づけ

PartonMode = 1 の振幅 Π_a det Φ^a(r) を、1 個の Pfaffian Pf F(r) に置き換えるモードを足す。
F は全パートン(K·N 本)の対軌道 f_{αβ} から作る。f は反対称な (K·Ns)×(K·Ns) 行列で、一粒子状態は α = (サイト, フレーバー)。

- **det^K 積状態は変分空間に厳密に含まれる**(§1.3 の包含定理)。PartonMode = 1 の最適解から始めると、
  初期点で Ψ_Pf ∝ Ψ_det が成り立つ。したがってエネルギーは PartonMode = 1 の最終値から下がるしかない
- **K = 2 で上流 mVMC の一般スピン Pfaffian と同じ規約にする**(ユーザー判断)。上流が生成する形の
  `orbitalidxgen.def`(行が I < J の向き)は、そのまま PartonPairOrbital として読める(§2.1)
- 再利用するもの: PartonConfiguration、全フレーバー同時移動(NExUpdatePath = 6)、サンプリング骨格
  `parton_make_sample!`、E_loc `parton_local_energy`、Jastrow、量子数射影、SR(`stochastic_opt!`)、
  共役シム、MPI 同期、出力群と門番の大部分
- 差し替えるもの: 振幅部(契約 1〜3 と O)と、パラメータ部(α を f に)の 2 か所。α は初期化にしか使わない

制約(従来どおり):

- **PartonMode = 0 / 1 の既定挙動はビット一致**。既存の回帰テストとベースラインもビット一致を保つ
- **マルコフ連鎖の乱数消費順を変えない**。骨格を共有するので、同じ更新列なら同じ乱数列を消費する(§4.4)
- opt-in のみ。新しい modpara キーは足さない(`PartonMode = 2` だけで切り替える)

## 1. 理論と固定式

### 1.1 記号

| 記号 | 意味 | コード上の名前 |
|---|---|---|
| Ns | サイト数 | `mp.nsite` |
| K | フレーバー数 | `mp.nflavor` |
| N | 物理粒子数 = フレーバーあたりのパートン数 | `mp.nelec` |
| M = K·N | パートン総数 = F(r) の次元 | `n_parton_total` |
| α = (i, a) | 一粒子状態。合成添字は **α = i + a·Ns**(0-based、上流の site + spin·Ns の自然な拡張) | |
| m | 物理粒子のラベル(`cfg.ele_idx` の並び。**サイト順にはソートしない**) | |
| I = (m−1)K + a | F の行。粒子 m のフレーバー a のパートン。x_I = (r_m, a) | |

行の並びは「粒子が主・フレーバーが従」である。ラベルを固定した第一量子化なので、
粒子が動いても行の番号は変わらない(PartonMode = 1 の A^(f)[m, n] と同じ扱い)。

### 1.2 波動関数

```
Ψ(x) = P_J(x) · ip(x),   ip(x) = Σ_qp w_qp · Pf F_qp(x)

F_qp(x)[I, J] = s_qp(r_I) · s_qp(r_J) · f[(π_qp(r_I), a_I), (π_qp(r_J), a_J)]     (M×M、反対称)
f[α, β]       = 2 · sgn[α, β] · p[idx(α, β)]          (def に行がある対)
f[α, β]       = 0                                       (def に行がない対。パラメータではない)
```

- P_J・w_qp・π_qp(`qp_trans`)・s_qp(`qp_trans_sgn`)は PartonMode = 1 と同一(本体 §1.1)
- **係数 2 と符号の規約は上流と同じ**(ユーザー判断)。上流の一般スピン経路は idx を対称・sgn を反対称に格納し、
  `slt_ij − slt_ji` を取るので F_IJ = 2·sgn·p になる(`slater_update.jl:686-697`、`orbital_qptrans_utils.jl:150-157`)
- 符号は**書かれた行の向きと符号列のとおり**: 行 `i a j b idx sgn` から sgn[α, β] = sgn、sgn[β, α] = −sgn。
  符号列は省略時 +1。上流の APFlag = 1 の扱いと同じ。
  - 上流は APFlag = 0(NMPTrans ≥ 0)のとき符号列を捨て、添字の大小で ±1 を決める(`orbital_qptrans_utils.jl:162-170`)
  - パートンの門番は NMPTrans < 0 を禁じているので、上流にそのまま倣うと符号付きの idx 共有が書けなくなる。
    これは、周期境界をまたぐ同フレーバー対を並進で共有するのに要る
  - 上流が生成する def は行が I < J の向きで、APFlag = 0 では sgn = +1 なので、どちらの扱いでも同じ F になる
    (fsz の例題 66 行で確認。APBC の例題は sgn = −1 の行を 10 本含む)
- 係数 2 は Ψ を全配置で同じ定数 2^{M/2} 倍にするだけで、比・O(対数微分)・SR・D_AmpMax 後の p はどちらの規約でも変わらない。
  2 倍は浮動小数点で厳密な演算なので、違いが出るのは |ip| の絶対値(1e−30 のノード判定)だけである
- 物理粒子の交換で Pf は (−1)^K 倍になる。K が偶数ならボソン、奇数ならフェルミオンの統計が自動で出る(数値確認済み)

### 1.3 包含定理

`Pf(B A Bᵀ) = det(B)·Pf(A)` から次が出る。

- Ξ = diag(Φ^1, …, Φ^K)(K·Ns × M、列はフレーバー順)
- J は M×M の反対称行列で、Pf J ≠ 0 とする(ペアリング)
- f = Ξ J Ξᵀ、すなわち p = (Ξ J Ξᵀ)[α, β] / (2·sgn[α, β]) とおく

このとき、すべての配置とすべての qp で次が成り立つ。

```
Pf F_qp(x) = s_{N,K} · Pf(J) · Π_a det A^(a)_qp(x),     s_{N,K} = (−1)^{C(K,2)·C(N,2)}
⇒ ip_Pf(x) = s_{N,K} · Pf(J) · ip_det(x)               (比例定数は x にも qp にもよらない)
```

ここで A^(a)_qp は PartonMode = 1 の行列である(本体 §1.1。qp の符号 s_qp(r) は各フレーバーの行に 1 回ずつ掛かる)。

系:

- **J の選び方は自由**(冗長性)。初期点では、ΞδJΞᵀ の方向について O がサンプルごとに定数になる。
  したがって S に厳密な零空間がある(idx の縮約がなければ複素 M(M−1)/2 次元。加えて全体の複素スケール方向が常に 1 本ある)
- **この零空間は RedCut では落ちない**。RedCut は S の対角要素だけで判定するが(`stochastic_opt.jl:525-531`)、
  零空間は座標軸に沿わないからである。正則化は `DSROptStaDel > 0` が担う。**門番は付けず記録だけにする**
  (ユーザー判断: StaDel = 0 で走らせることはない)。T1-e で次元を数値で確認して記録する
- K ≥ 2 なら、同フレーバーどうしを組にしない J がとれる。つまり異フレーバー成分(a ≠ b)だけの def でも det^K 積を表現できる
- **KN が奇数なら Pf ≡ 0**。門番で拒否する。将来、不対軌道を足した拡張 Pf で対応する。
  注意: **K = 3 では N が偶数でなければならない**(KM 9×4 の N = 12 は可。K = 3 で 1 粒子だけドープすると拒否される)

### 1.4 一粒子状態の抽出(行ブロック SVD)

- f の行ブロックを f_{a,:} := f[(·, a), :](Ns × K·Ns)とする。f = ΞJΞᵀ なら f_{a,:} = Φ^a·R_a で、階数はちょうど N。
  上位 N 本の左特異ベクトルが V_a = span{φ^a_n} を張る。複素共役はどのフレーバーにも要らない
- 個々のブロック f_{ab} の SVD では足りない(rank J_{ab} < N がありうる)
- 特異値は J と D_AmpMax のスケールを映すだけで、物理的な意味はない。物理量は射影子 P_a = U^a U^{a†} で扱う
- 一般の f では、上位 N 本を近似的な V_a とし、**σ_N / σ_{N+1}** を診断量として出す。det^K 多様体からの離れ具合を表す

### 1.5 比・更新・O の固定式(小行列で数値検証済み)

粒子 m を r_old → r_new へ動かす。行・列の集合 S = {(m−1)K + a : a = 1..K}(|S| = K。実装は |S| 一般で書く)。
qp ごとに以下を行う(添字 qp は省略。G = F⁻¹)。

```
Δ  = F′ − F                                  (S の行と列だけが変わる)
D[J, s] = Δ[s, J]      (J ∉ S)
D[J, s] = Δ[s, J] / 2  (J ∈ S)               ⇒ Δ = E Dᵀ − D Eᵀ,  E = [e_s]_{s∈S}
B = [E, D]  (M×2K),   C = [[0, I], [−I, 0]],   C⁻¹ = −C

X     = C⁻¹ + Bᵀ G B                         (2K×2K、反対称)
GB    = [G[:, S], G·D]                       (M×2K。コストの主項 O(K·M²))
Bᵀ G B のブロック = [[G[S,S], (GD)[S,:]], [−(GD)[S,:]ᵀ, Dᵀ(GD)]]
R_qp  = Pf(X) / Pf(C⁻¹),    Pf(C⁻¹) = (−1)^{K(K+1)/2}
G′    = G + (GB) · X⁻¹ · (GB)ᵀ               (Bᵀ G = −(GB)ᵀ を使う。O(K·M²))
Pf F′ = R_qp · Pf F
ratio = Σ_qp w_qp Pf_qp R_qp / Σ_qp w_qp Pf_qp
```

O(f は p について正則。I < J の和):

```
O_k = (1/ip) Σ_qp w_qp Pf F_qp · Σ_{I<J} G_qp[J, I] · s_qp(r_I) s_qp(r_J) · 2·sgn[α_I, α_J] · [idx(α_I, α_J) = k]
      (α_I = (π_qp(r_I), a_I))
```

- ∂ln Pf F / ∂F_IJ = G_JI(F_JI = −F_IJ と連動させた変化)を使っている。上流の
  `buffer += −InvM·cs·sgn`(`vmc_main_cal.jl:3058-3069`、全順序対の和)と同じ量である
- O は正則なので **(O, i·O) と詰め、Φ モードと同じく共役シムを掛ける**。理由は本体 §12.2:
  パートンの E_loc は比をそのまま足す規約で、上流は `conj(z/ip)` を足すので、その差を受け渡し点で吸収する
- 数値検証(`check_theory.jl`。T1 として test へ移す):
  - 包含定理: K = 2, 3, 4 で相対誤差 ≤ 2.6e−14
  - 交換で (−1)^K 倍になること
  - 比: ≤ 1.1e−14、Woodbury による G 更新: ≤ 2.6e−14
  - Pf(C⁻¹) = −1, 1, 1(K = 2, 3, 4)
  - 対数微分と G_JI の差 1.5e−10(差分の刻み 1e−6)
  - SVD 射影子と Φ 射影子の差 ≤ 8.5e−16

### 1.6 コスト(qp 1 つあたり)

| 処理 | コスト | PartonMode = 1 との比 |
|---|---|---|
| 提案 1 回の比(契約 2) | O(K·M²)(G·D が主項) | det は O(K·N)。M²/N = K²N 倍重い |
| 受理時の G 更新(契約 3) | O(K·M²) | det は O(K·N²) |
| 厳密再計算(契約 1) | O(M³)(LTL 分解 1 回で Pf と逆行列) | det は O(K·N³) |
| サンプルごとの O | O(M²/2) | |
| E_loc | (動ける合成ホップの本数)× O(K·M²) | |

E_loc が支配的になりそうなら、サンプルごとに Q = 𝔽 G 𝔽ᵀ を前計算する案がある
(𝔽[β, J] = f[β, x_J]、(K·Ns)×(K·Ns)、O(K·Ns·M² + (K·Ns)²·M))。そうすれば 1 ホップあたり O(K²) で済む。
**第 1 段では入れない**(§10。実測してから判断する)。

## 2. 入力契約

### 2.1 namelist キーと def(ユーザー判断: 新キー + 専用の厳格パーサ)

| キー | ファイル | 必須 | 書式 |
|---|---|---|---|
| `PartonPairOrbital` | 対軌道 f の idx 構造 | **必須** | 上流の `orbitalidxgen.def` と同じ |
| `InPartonPairOrbital` | f の初期値 | 任意(無ければ seed に従った乱数。§2.3) | 上流の `InOrbital.def` と同じ |
| `Jastrow` / `InJastrow` / `PhysHop` / `CoulombInter` / `TransSym` | 既存 | 既存どおり | 既存どおり |

**PartonPairOrbital**(上流の OrbitalGeneral の列構成で、spin をフレーバーと読み替える):

```
=============================================
NOrbitalIdx     <n_idx>
ComplexType          1
=============================================
=============================================
<site1> <flavor1> <site2> <flavor2> <idx> [<sgn>]     ← データ行(0-based)
...
<idx> <optflag>                                        ← 末尾の OptFlag 行(上流と同じ 2 列)
```

- ヘッダは 5 行。キーワードも上流と同じ `NOrbitalIdx` / `ComplexType` にする(K = 2 なら上流の def がそのまま通る)
- 厳格に検査し、違反はエラーにする(上流のパーサは範囲外を黙って捨て、網羅性も重複も見ない。§3)。
  分担: パーサは書式(ヘッダ・列数・整数・符号 ±1)だけを見る(パースの時点では NSite / NFlavor が確定していない
  ことがあるため)。意味の検査(下の範囲・対角・重複・連番)はテンプレート build(`PartonPairOrbital` の構築)が行う:
  - site ∈ [0, Ns)、flavor ∈ [0, K)
  - α ≠ β(対角は禁止)
  - 同じ非順序対の重複は禁止(逆向きの重複も禁止)
  - idx は 0..n_idx−1 の連番で、全部使われていること
  - sgn ∈ {+1, −1}
  - 行は 5 列か 6 列
  - OptFlag 行の idx が範囲内であること、重複しないこと
- **def に無い対は構造的な 0**(パラメータではない)。上流の fsz 経路は未列挙の対を黙って idx 0 に結び付けるが
  (`orbital_qptrans_utils.jl:55` が `zeros(Int, …)` で初期化)、これは踏襲しない(§11 の upstream 報告候補)

**InPartonPairOrbital**: 上流 InOrbital と同じ 5 行ヘッダ + `idx Re Im`。
上流の汎用リーダ `parse_input_parameter_file` は読めない値を 0 にし、NaN を通し、件数の不一致を警告で済ませる(本体 §12.2)。
そこで Φ モードの InPmfOrb と同じく**厳格な専用リーダ**で読む。件数・範囲・重複・有限性を検査する。
**全 idx を含むこと**を要求する(部分的な上書きはしない。出力 `zqp_parton_pair_*.dat` は常に全 idx を書くので往復は成り立つ)。

### 2.2 modpara

新しいキーは足さない。

- `PartonMode = 2` で切り替える。**`NFlavor ≥ 2`**(ユーザー判断 2026-10-31。K = 1 は N 偶数のスピンレス
  Pfaffian として理屈では動くが、用途がはっきりするまで門番で拒否する。後で緩めるのは容易)
- **更新経路は `NExUpdatePath = 6`(全フレーバーが同じ座標でホップする固縛移動)のみ**。PartonMode = 1 と同じ値を
  共用する(ユーザー判断 2026-10-31。意味が同じ値を 2 つ作らない。T3 では modpara の違いが PartonMode だけになる)。
  6 はフォークで足したパートン専用の値で、上流が使うのは 0〜3(`vmc_sampling.jl:1187-1215`)。
  将来 Pf 専用の手(2 粒子の交換など。契約 2・3 は |S| 一般で書くので乗る)を足すときは新しい値を割り当てる
- そのほかの必須設定は PartonMode = 1 の共通検査をそのまま使う(`2Sz = 0`、`ComplexType = 1`、`NSRCG = 0` ほか。本体 §2.1)
- `PartonBlockUpdateSize`(既定 16)を Pf の厳密再計算の周期にもそのまま使う
- D_AmpMax は上流の定数 `D_AMP_MAX = 4.0`(`parameter_sync.jl:9`。modpara キーではない)を使う(§4.6)
- `PartonOrbParam ≠ 0`、`PartonOccMode ≠ 0` は門番で拒否する(Pf モードには Φ のパラメータ化も占有も無い)
- `PartonGaugeFix` / `PartonFlavorSymFast` は拒否せず、runinfo に「該当なし」と書く(本体 §12.8 と同じ理由)

### 2.3 初期値(ユーザー判断 2026-10-31: 指定が無ければ seed に従った乱数)

本番の主な使い方は**乱数初期化**である(PartonMode = 1 の結果を初期値に使う予定は少ない)。

```
PartonPairOrbital 読み込み(OptFlag 行を含む)
  → OptFlag の実体化
  → InPartonPairOrbital があればそれを読む / 無ければ乱数で生成
  → D_AmpMax で規格化
  → 門番 → サンプリング RNG → QP → 状態 → zqp_parton_pair_init.dat → SR ループ
```

- **分布は上流と同じ**(パーサ `parameter_init.jl:162, 205-220`): 可動成分は p = [2(r₁−½) + 2i(r₂−½)]/√2、
  OptFlag = 0 の成分は 0。そのため、乱数より前に OptFlag を実体化する
- **乱数列は初期値専用**(ユーザー判断。α モードの `parton_init_alpha!` と同じ規律。`parton_initial_params.jl:73-101`):
  - 解決済みの基準 seed(ランクのずらしを加える前の値)で専用の SFMT を作る
  - **サンプリング用の乱数列は消費しない**。初期値を乱数で作ったかファイルから読んだかで、サンプリングの乱数列が変わらない
  - 全ランクが構成的に同じ値を得るので、bcast は要らない
  - 上流は 1 本の列から初期値を引いてからサンプリングに使うので(`run_para_opt_from_namelist.jl:133-148`)、
    **乱数の値そのものは上流と一致しない**(分布は同じ)
- `zqp_parton_pair_init.dat` には**実際に採用された初期値**(D_AmpMax 後)を書く。乱数で始めた run も、これを
  InPartonPairOrbital に渡せばビット再現できる

## 3. 調査で確定した既存コードの事実(設計の前提)

| 事実 | 根拠 | 設計への影響 |
|---|---|---|
| Julia 移植の OrbitalGeneral パーサは、C 形式の 6 列 `i s_i j s_j idx sgn` を `site1 site2 idx sgn` と誤読する。テストでも例題でも未使用(すべてコメントアウト) | `orbital_parser.jl:205-226`、`examples/inputs/heisenberg_chain_fsz/namelist.def:13` | 専用パーサを新設。上流は触らない(§11) |
| 上流の一般スピン経路は F_IJ = 2·sgn_IJ·p | `slater_update.jl:686-697` | §1.2 の規約(ユーザー判断) |
| APFlag = 0 のとき、上流は fsz の符号列を捨てて添字の大小で ±1 にする | `orbital_qptrans_utils.jl:162-170` | §1.2(書かれた向きと符号を使う) |
| 上流の出力は `orbital_terms` を**行ごと**に書く。`zqp_orbital_opt.dat` は行番号を idx として書き、ヘッダも 4 行。読み手は idx ごと・5 行ヘッダ | `data_io.jl:135-137, 181-185, 235-245`、`initial_params.jl:64-67` | 上流の `orbital_terms` は流用しない(ユーザー判断: 専用の器) |
| RedCut は S の対角要素だけで判定する。`DSROptStaDel` の既定値は 0.0 | `stochastic_opt.jl:525-531`、パーサ `constants.jl:132-133` | §1.3。記録だけ |
| D_AmpMax は上流の定数 4.0。SR 更新のたびに max\|p\| をちょうど 4 にそろえる(拡大もする。OptFlag = 0 の成分も含む) | `parameter_sync.jl:9, 138-151` | §4.6 で同じ式を自前で掛ける |
| 上流の軌道の乱数初期化は 1 本の SFMT(種 = `resolve_rnd_seed`)から引き、同じ列をサンプリングに続けて使う。分布は複素なら [2(r₁−½) + 2i(r₂−½)]/√2、OptFlag = 0 は 0 | `run_para_opt_from_namelist.jl:133-148`、パーサ `parameter_init.jl:162-220` | §2.3: 分布は合わせ、列は専用にする |
| `parton_sync_parameters!` は bcast だけ(α のときだけゲージ射影)。上流の `sync_modified_parameter!` は相関因子のシフトもするので、パートンでは使えない | `parton_vmc_para_opt.jl:24-40` | 同上 |
| 骨格 `parton_make_sample!` は `pstate.amp`・`pstate.workspace` を経由し、契約 1〜3 を型でディスパッチして呼ぶ。1 提案の乱数は `rng_mod` 2 回 +(候補が有効なら)`rng_real2` 1 回 | `parton_vmc_sampling.jl:355-427, 435-441` | §4.4: 骨格は無変更で共有 |
| `PartonOptimizationState` の `amp`・`workspace` は det 用の具象型で固定されている。スレッド文脈も同じ(`amps::Vector{PartonAmplitudeData}`、`_parton_thread_ctx!` の固定コンストラクタ) | `parton_types.jl:423-455`、`parton_vmc_main_cal.jl:415-441` | §4.2: 型パラメータ化。スレッド経路は第 1 段で使わない |
| 契約 2・3 のシグネチャは配置(cfg)を受け取らない | `parton_vmc_sampling.jl:475-483, 526-534` | §4.3: Pf の振幅キャッシュが各行のサイトを自前で持つ |
| `parton_make_initial_sample!` は `amp::PartonAmplitudeData`・`ws::PartonSamplingWorkspace` と注釈されている | `parton_vmc_sampling.jl:86-94` | 注釈を抽象型へ広げる(本体は無変更。乱数消費が同一になる) |
| 測定の `parton_main_cal!` は振幅も O もディスパッチで呼び、`force_threaded = false` を渡せば逐次になる | `parton_vmc_main_cal.jl:531-648` | Pf ループは逐次で呼ぶ |
| `validate_parton_modpara` は PartonMode ≠ 1 を拒否する | `parton_unsupported_inputs.jl:103-109` | 許可するモードを引数で渡せるようにする |
| `validate_parton_data` は orbital_terms が空でないことと NOrbitalIdx ≠ 0 を拒否する | `parton_unsupported_inputs.jl:239-246, 331-338` | 「既存スピン Orbital との同時指定」の門番としてそのまま使える |
| PfaPack の LTL 分解(`julia_zsktf2!` / `julia_zsktf2_turbo!`)→ `utu2pfa`(Pf)→ `utu2inv!`(逆行列)が上流の全再計算の部品。上流は格納の都合で最後に符号を反転する | `calculate_m_all.jl:172-215` | 契約 1 で流用する。符号は T1 の G·F = I で機械的に確かめる |
| 奇数次元の Pf・不対軌道は上流にも無い | `types.jl:120, 193`、`PfaPack.jl/src/pfaffian.jl:24` | KN 奇数は門番で拒否 |
| PfaPack の部品の規約(P1 で実測): `julia_zsktf2!` は上三角だけを読む。特異なら info > 0。`utu2inv!` の結果は X⁻¹ そのもの(‖A·X − I‖ ~ 1e−15) | `check_pfapack.jl`(scratchpad) | 下の行の理由で、結局 Pf モードでは使わない |
| **PfaPack の `julia_zsktf2!` は、消去すべき副列が全部 0 になるステップのある正則行列で info = 2 を返す**(P1 第 4 段で発見)。最小例: 2×2 ブロックの直和 [[0,a],[−a,0]] ⊕ [[0,b],[−b,0]]、その行と列を交互に並べたもの、C⁻¹ = [[0, −I], [I, 0]](K = 2)。いずれも正則なのに info = 2(本来は消去を飛ばせばよいだけ)。Pf モードでは、同フレーバーだけを組んだ f(adjacent ペアリング)で全配置が Pf = 0 と判定された。契約 2 の X = C⁻¹ + BᵀGB も BᵀGB が小さいとこの形に近づく | `debug_adjacent.jl`(scratchpad)、T1 の「交互ブロック構造」テスト | **契約 1・2 の Pf は自前の Parlett–Reid(列ピボットつき、割り当てなし)、G は LU** に差し替えた(§8)。§12 の報告候補 |
| **PfaPack の `pfaffian_ltl!` は複素行列で n ≥ 4 のとき誤った値を返す**(実数行列と n = 2 では正しい)。本番経路は使っておらず、使うのは PfaPack 自身のテストだけ | `PfaPack.jl/src/pfaffian.jl:38`、P1 の実測(`utu2pfa` と独立な Parlett–Reid は一致し、`pfaffian_ltl!` だけがずれる) | Pf モードでは使わない。§12 の報告候補 |

## 4. アーキテクチャ

### 4.1 決定事項

| 項目 | 決定 | 理由 / 出どころ |
|---|---|---|
| ブランチと作業場所 | `parton-pf`(d7cc568 から)を worktree `../Julia-mVMC-pf` で | ユーザー判断。別セッションが本体ディレクトリで Φ を改装中のため |
| 入力 | 新キー `PartonPairOrbital` / `InPartonPairOrbital` + 専用の厳格パーサ | ユーザー判断(§2.1) |
| パラメータの器 | 専用の器 `data.parton_pair_params::Vector{ComplexF64}`(idx ごと)と `data.parton_pair_terms`。ロケータ種別を 1 つ追加 | ユーザー判断(§3 の往復不能を避ける) |
| F の規約 | ×2・書かれた向きと符号(省略時 +1) | ユーザー判断(§1.2) |
| S の零空間 | 門番なし・記録だけ | ユーザー判断(§1.3) |
| モードの表し方 | 状態型でディスパッチする(本体 §12.5 と同じ流儀)。`if` 分岐は入口の 1 か所だけ | |
| 振幅の型 | `PartonOptimizationState{O, A, W}` に型パラメータ化する(§4.2) | 骨格・E_loc・測定を無変更で共有するため |
| スレッド並列の測定 | **第 1 段では使わない**(Pf ループは `force_threaded = false`)。`JULIA_MVMC_INNER_THREADS` が立っていれば警告して逐次で走る | スレッド文脈の一般化は共有ファイルへの変更が大きい。逐次でビット一致を固めてから(§10) |
| D_AmpMax | 上流と同じ式・同じ定数を自前で掛ける(§4.6) | 上流の `sync_modified_parameter!` は相関因子もずらすので呼べない |
| 初期化 | 指定が無ければ初期値専用の乱数列で上流と同じ分布(§2.3)。InPartonPairOrbital は任意。PartonMode = 1 の結果からの変換は tools の最小限のスクリプト(§5、T3・T4 の継続用) | ユーザー判断。α モードの出力は変えない |
| zvo_var / zqp_opt への登録 | **しない**。専用ファイルに出す(§6) | Φ モードと同じ。f は列が巨大になりうる |
| CalcTimer | 骨格の 803〜813 はそのまま同じ意味で計測される。Pf 固有は **830 番台**(830 f の組立、831 D_AmpMax、832 SVD 診断) | 820 番台は Φ が使用中 |

### 4.2 型

```julia
# parton_types.jl(既存。マーカー付きで変更)
abstract type PartonAmplitudeStore end          # 新設: 配置ごとの振幅キャッシュ
abstract type PartonWorkspaceStore end          # 新設: ホットループの作業領域
mutable struct PartonAmplitudeData <: PartonAmplitudeStore        # 既存。supertype を足すだけ
mutable struct PartonSamplingWorkspace <: PartonWorkspaceStore    # 既存。supertype を足すだけ

mutable struct PartonOptimizationState{O<:PartonOrbitalSource, A<:PartonAmplitudeStore, W<:PartonWorkspaceStore}
    state; amp::A; config; workspace::W; mfham::O; physham; thread_ctx
end
# 6 引数コンストラクタは注釈を抽象型へ広げるだけ(A, W は推論される)
```

```julia
# parton_pair_types.jl(新規)
mutable struct PartonPairOrbital <: PartonOrbitalSource      # 「軌道の出どころ」= 対軌道 f
    n_site, n_flavor, n_elec, n_idx
    pair_idx::Matrix{Int32}       # (K·Ns)², 0 = def に無い対。1-based の idx
    pair_sgn::Matrix{Int8}        # 反対称(sgn[β,α] = −sgn[α,β])
    pair_matrix::Matrix{ComplexF64}   # f[α,β] = 2·sgn·p。毎ステップ data.parton_pair_params から組み直す
end

mutable struct PartonPairAmplitudeData <: PartonAmplitudeStore
    inv_f::Vector{ComplexF64}     # qp ごとの G = F⁻¹(M²)。ストライドは pair_inv_block だけに封じる
    pf_f::Vector{ComplexF64}      # qp ごとの Pf F(生の複素値)
    row_site::Vector{Int}         # 粒子 m の現在のサイト(契約 2・3 は cfg を受け取らないため自前で持つ)
    n_qp, n_flavor, n_elec
end

mutable struct PartonPairSamplingWorkspace <: PartonWorkspaceStore
    f_scratch::Matrix{ComplexF64}       # M×M。厳密再計算の組立と LTL 分解
    iwork / v_t / m_work                # PfaPack の作業領域
    d_buf::Matrix{ComplexF64}           # M×K
    gb_blocks::Vector{Matrix{ComplexF64}}   # qp ごとの GB(M×2K)。契約 2 が書き、契約 3 が読む
    x_inv_blocks::Vector{Matrix{ComplexF64}}  # qp ごとの X⁻¹(2K×2K)
    ratio_blocks::Vector{ComplexF64}    # qp ごとの R_qp
    x_scratch::Matrix{ComplexF64}       # 2K×2K の Pf 計算
end
```

- `pstate.mfham` は Pf モードでは `PartonPairOrbital` になる(フィールド名は本体 §12.5 の判断どおり据え置き)。
  抽象型 `PartonOrbitalSource` の docstring にある「全具象型が `orbitals` を持つ」は「det 系の具象型は」に直す。
  det の契約は `amp::PartonAmplitudeData` でディスパッチされるので、Pf の器が誤って det の契約に入ることはない(入ればメソッドが無いエラー)
- 名前は上流の Slater(f_ij、`slater_*`)と紛れないよう `parton_pair_` / `PartonPair` を接頭辞にする
  (Pf は Pfaffian の略として定着しているが、新しい名前では略さない — 利用者の命名規約)

### 4.3 契約(Pf 版のメソッドを足す。骨格からはディスパッチで呼ばれる)

| 契約 | 関数(既存名に Pf のメソッドを足す) | 要点 |
|---|---|---|
| 0 | `parton_pair_load!(src, data)` | `data.parton_pair_params` → `pair_matrix`。毎ステップ冒頭。0 → 1-based の変換はテンプレート build(`parton_build_pair_template!`、起動時 1 回)だけで行う(本体 §3.1) |
| 1 | `parton_recompute_amplitude_all!(amp::PartonPairAmplitudeData, src::PartonPairOrbital, cfg, data, ws)` | qp ごとに F_qp を組んで LTL 分解 → Pf と G。特異は Pf = 0(det モードと同じ)。`row_site` も cfg から確定する |
| 1 | `parton_calculate_ip(amp::PartonPairAmplitudeData, w)` | Σ_qp w_qp Pf_qp |
| 2 | `parton_amplitude_ratio!(ws::PartonPairSamplingWorkspace, amp, src, data, w, m, r_new)` | §1.5。**純粋**(ws だけに書く。E_loc の仮想移動で何度呼ばれても状態は動かない)。GB・X⁻¹・R_qp を ws に残す |
| 3 | `parton_update_amplitude!(amp::PartonPairAmplitudeData, src, data, ws, m, r_new)` | G′ = G + (GB)X⁻¹(GB)ᵀ、Pf *= R_qp、`row_site[m] = r_new`。\|R_qp\| < ratio_floor(1e−12)なら `:need_recompute`(det モードと同じプロトコル) |
| 4 | `parton_local_energy`(既存のまま) | 契約 2 を仮想移動として呼ぶだけ。統計の符号は Pf が運ぶ |
| 5 | `parton_fill_sr_opt_o!(sr_opt_o, amp::PartonPairAmplitudeData, src::PartonPairOrbital, …; conjugate, project_gauge, alpha)` | Jastrow → f ブロック(§1.5、(O, i·O))→ 共役シム `_parton_conjugate_mf_slots!(sr_opt_o, n_proj, n_idx)`。`project_gauge` / `alpha` は受け取って無視する(Φ 版と同じ) |
| OO | `parton_finalize_oo_store!(::PartonPairOrbital, …)` | Φ 版と同じ BLAS 版。Φ のファイルは別セッションが編集中なので本体を複製し、合流後に統合する(§10) |
| 初期配置 | `parton_make_initial_sample!`(既存) | 注釈を抽象型へ広げるだけ。Fisher–Yates の乱数消費は同一 |

### 4.4 乱数消費の同一性(T3 の前提)

骨格 `parton_make_sample!` と `parton_make_initial_sample!` を**無変更で共有**するので、1 提案あたりの乱数消費は
PartonMode = 1 と同じになる(`rng_mod` で粒子 → `rng_mod` で行き先 → 有効なら比を計算して `rng_real2` を 1 回)。

f = ΞJΞᵀ の初期点では比が丸め誤差の範囲で一致する(§1.3)。そのため受理判定も一致する。
食い違うのは、乱数が |ratio|² の 1e−14 以内に落ちたときだけである。

厳密再計算の時期は 2 つの経路で変わる。
- `:need_recompute` の判定が、det は (qp, f) ブロックごと、Pf は qp ごとに行われる
- ただし再計算は乱数を消費しないので、連鎖には影響しない

### 4.5 SR ループ(`parton_pair_vmc_para_opt!`、Φ ループと同じ骨組み)

1. `parton_pair_load!`(data → `pair_matrix`)
2. `parton_make_sample!` → `parton_main_cal!(…; force_threaded = false)`
3. weight_average → reduce_counter → `output_data!`
4. `stochastic_opt!`(info は bcast してから判定)
5. `parton_sync_parameters!(data, ctx)`(bcast のみ)→ NaN ゲート(非有限なら巻き戻す)→ **D_AmpMax 再スケール**
6. 診断(`zvo_parton_pairdiag.dat`)→ `store_opt_data!`

ループの後に `output_opt_data!`(Jastrow のため必ず呼ぶ)、`zqp_parton_pair_opt.dat`、SVD 出力、conv、CalcTimer を書く。

### 4.6 D_AmpMax(上流と同じ式)

`p ← (D_AMP_MAX / max_k |p_k|)·p`。OptFlag = 0 の成分も含めて全成分に掛け、拡大もする。適用するのは次の 2 か所:
- 初期化(InPartonPairOrbital を読んだ直後。上流の `run_para_opt_from_namelist.jl:192` に対応)
- 毎ステップの SR 更新後(bcast → NaN ゲートの後。上流の `sync_modified_parameter!` と同じ位置)

Pf(λF) = λ^{M/2} Pf F なので物理は変わらない。倍率は正の実数なので、全体位相は固定しない。

**|ip| の桁の見積もり**(本体 §12.4 の Pf 版)は P1 で実測する。α 解を変換した f で、KM 9×4(M = 36)の
log₁₀|ip| の分布を測り、1e−30 の閾値にも、あふれにも当たらないことを確かめる。当たるようなら、そのときにキーを足す。

### 4.7 ファイル構成

**新規**(`MVMCOptimizers.jl/src/`):

| ファイル | 責務 |
|---|---|
| `parton_pair_types.jl` | 3 つの型、合成添字、テンプレート build(0 → 1-based はここだけ)、`parton_pair_load!` |
| `parton_pair_amplitude.jl` | 契約 1・2・3 と ip。2K×2K の小さい Pf と逆行列 |
| `parton_pair_o.jl` | 契約 5 と OO の finalize |
| `parton_pair_inputs.jl` | InPartonPairOrbital の厳格リーダ、乱数初期化(専用の列)、フラグの実体化と長さ検査、門番、D_AmpMax |
| `parton_pair_output.jl` | `zqp_parton_pair_*`、SVD 出力、pairdiag、runinfo の追記、CalcTimer の Pf セクション |
| `parton_vmc_para_opt_pair.jl` | 状態の構築、SR ループ、ドライバ本体 |

パーサ側の新規: `MVMCExpertModeParsers.jl/src/parsers/parton_pair_orbital_parser.jl`

**既存ファイルへの変更**(すべて `# --- parton-mode (fork addition) ---` 付き。転送・分岐・型注釈・末尾追加の範囲に収める):

| ファイル | 変更 | 別セッション(Φ)との衝突の見込み |
|---|---|---|
| パーサ `utils/constants.jl` | `MVMC_KEYWORDS` に 2 行 | **あり**(Φ も同じ表を編集中。末尾追加どうしなので解消は容易) |
| パーサ入口 `MVMCExpertModeParsers.jl` | include 1 行 + `parse_file_by_type!` の elseif | 小 |
| パーサ `types/expert_types.jl` | `PartonPairOrbitalTerm` 型、ExpertModeData の末尾 2 フィールド + 既定値 | 小(末尾追加) |
| パーサ `utils/read_input_parameters.jl` | `count_variational_parameters` に 1 行 | 小 |
| `stochastic_opt.jl` | ロケータ種別 `_PARAM_PARTON_PAIR`(enum 1 行、foreach 2 本、get/set 2 本)。f ブロックは末尾 | **あり**(Φ の `_PARAM_PMFORB` の隣。末尾追加) |
| `parton_types.jl` | 抽象型 2 つ、supertype 2 か所、状態の型パラメータ化、6 引数コンストラクタの注釈 | 中 |
| `parton_vmc_sampling.jl` | `parton_make_initial_sample!` の注釈 2 か所 | 小 |
| `parton_unsupported_inputs.jl` | `validate_parton_modpara` に許可モードの引数 | 中 |
| `parton_run_para_opt_from_namelist.jl` | パース直後の 1 分岐(`parton_mode == 2` → Pf の入口)。PartonOrbParam の分岐より前 | 小 |
| `MVMCOptimizers.jl` | include と export | 小 |
| `tools/`(新規) | α 解 → InPartonPairOrbital の変換スクリプト(§5) | なし |

## 5. PartonMode = 1 の結果からの変換(T3・T4 の継続用。tools の最小限のスクリプト)

本番の主な初期化は §2.3 の乱数で、このスクリプトは T3(PartonMode = 1 との一致)と T4 の継続最適化(E_Pf ≤ E_det の裏取り)のためにある。

`tools/parton_alpha_to_pair.jl <alpha_stage_in> <alpha_stage_out> <new_dir> [interflavor|adjacent|auto] [<pair_def>]`

(`<pair_def>` を省略すると全対・共有なしの def を書く。ログは `<new_dir>/parton_alpha_to_pair.log`)

1. α\*(`zqp_pmfpara_opt.dat`)と占有(`zqp_pmfocc_opt.dat`)から H_MF を組み直し、各フレーバーの占有軌道 Φ^a から Ξ を作る
   (Φ 変換スクリプトと同じライブラリ関数を使う。MOM の run は占有ファイルが必須 — 本体 §12.1)
2. ペアリング J を選んで f = ΞJΞᵀ を作る
   - `interflavor`(def が a ≠ b の成分しか持たないときの既定): 異フレーバーの列どうしを完全マッチングする。
     K ≥ 2 なら常にとれる。K = 3 の例: フレーバー 1 の前半 N/2 を 2 と、後半を 3 と、2 の残りを 3 と組む
   - `adjacent`(それ以外の既定): 列の隣どうし (2j−1, 2j) を組にする
   - **k と Q−k を組む `bloch` は作らない**(ユーザー判断 2026-10-31。PartonMode = 1 の結果を初期値に使う予定が少ないため)。
     並進対称性は上流と同じく def の idx 共有で表すだけで、SR・D_AmpMax・振幅のどこにも別の処方は無い。
     影響は変換の厳密さだけにある。def が並進で共有していると、手順 3 の射影が厳密なのは ΞJΞᵀ 自体が並進不変なとき
     (J = U J Uᵀ、TΞ = ΞU)に限る。素朴な J ではこれが一般に成り立たないので、残差が残り、
     出発点が det^K 状態から少しずれる。**共有なしの def ならどの J でも厳密**(T3 と T4 の継続はこちらで行う)
   - 拡大ユニットセルのアンザッツでは、def の共有は**平均場で生き残った並進(G_orb)だけ**で張ること。
     物理格子の全並進で共有すると、det^K 状態がそもそも変分空間に入らない(def の生成は外部スクリプトの責務)
3. def の idx 構造へ射影する。p_k = 平均_{行 ∈ k} f[α, β] / (2·sgn) とし、**残差 ‖f − f_proj‖ / ‖f‖ をログに出す**。
   残差が 0 でなければ Ψ_Pf ∝ Ψ_det は近似にとどまる
4. D_AmpMax で規格化し、InPartonPairOrbital 形式で書き出す。namelist / modpara も新ディレクトリに書く

T3 用に、**全対(共有なし)の def を書く補助**もこのスクリプトに持たせる(`--write-full-def`)。共有なしなら残差は厳密に 0。
対称性で縮約した def は外部スクリプトで作る。

## 6. 出力

すべて rank 0 のみ。接頭辞は既存規約に合わせる(`zqp` = 読み戻せる .def 族、`zvo` = 診断)。

| ファイル | 内容 | 形式 |
|---|---|---|
| `zqp_parton_pair_init.dat` / `_opt.dat` | p(idx ごと) | InPartonPairOrbital そのもの(5 行ヘッダ `NOrbitalIdx`、`idx Re Im`、`%.18e`)。往復でビット一致 |
| `zqp_parton_pair_sv_init.dat` / `_opt.dat` | 行ブロック SVD の全特異値 | `# flavor index sigma`(0-based)+ `# key value` で各フレーバーの σ_N/σ_{N+1} |
| `zqp_parton_pair_orb_init.dat` / `_opt.dat` | 上位 N 本の左特異ベクトル U^a | **InPmfOrb と同じ形式**(`NPmfOrb`、平坦添字 ((f·Ns)+s)·Ne+n)。Φ モードの入力としてそのまま渡せ、射影子 P_a = U U† を使う下流(Chern など)の入力にもなる |
| `zvo_parton_pairdiag.dat` | step、n_skipped、受理率、need_recompute、厳密再計算、再スケール前の max\|p\|、各フレーバーの σ_N/σ_{N+1}(log₁₀\|ip\| の最小と最大は、共有の `parton_main_cal!` を触らずには取れないので後回し) | `#` 列名ヘッダ。step 0 で "w"、以降 "a" |
| `zvo_parton_runinfo.dat` | 既存の writer の後に `PartonMode 2`、n_pair_idx、M、D_AMP_MAX、PartonGaugeFix / FlavorSymFast は「該当なし」を "a" で追記 | `key value` |
| `zvo_out` / `zvo_SRinfo` / `zvo_parton_time` / `zvo_conv` / `zvo_CalcTimer` / `zqp_opt`(Jastrow) | 既存どおり | |

- 書かないもの: `zqp_pmfpara_*` / `pmfocc` / `pmfham` / `pmfband` / `pmforb_*`、`zvo_parton_diag.dat`
- SVD の毎ステップ計算は O(Ns·(K·Ns)²·K)(KM 9×4 で約 1.3e6)で、サンプリングに比べて小さい

## 7. 門番(PartonMode = 2 のとき)

| 対象 | 動作 | 理由 |
|---|---|---|
| K·N が奇数 | エラー | Pf ≡ 0。将来は不対軌道で拡張する旨を案内する |
| NFlavor < 2 | エラー | ユーザー判断(§2.2) |
| NExUpdatePath ≠ 6 | エラー | 固縛移動のみ(§2.2。PartonMode = 1 と同じ検査を流用) |
| `PartonOrbParam ≠ 0` / `PartonOccMode ≠ 0` | エラー | Pf モードに存在しない |
| Orbital / OrbitalAntiParallel / OrbitalParallel / OrbitalGeneral / InOrbital 系がある(orbital_terms が空でない、または NOrbitalIdx ≠ 0) | エラー | 既存スピン Orbital との同時指定。`validate_parton_data` の既存の検査で落ちる |
| PartonMFTrans / PartonMFPara / InPmfPara / InPmfOcc / InPmfOrb がある | エラー | α・Φ を混ぜない |
| PartonPairOrbital が無い、または §2.1 の検査に違反(InPartonPairOrbital は任意。あれば同じく厳格に検査し、全 idx を要求する) | エラー | 緩いリーダの穴を塞ぐ |
| OptFlag の長さ ≠ 2(n_proj + n_pair_idx)、または f が全凍結 | エラー | 範囲外のフラグは黙って凍結される(本体 §2.5) |
| `parton_n_idx(data) ≠ 0` または `length(pmforb_params) ≠ 0` | エラー | 共役シムのオフセットの前提(f ブロックは n_proj の直後) |
| physhop・Jastrow・CoulombIntra・DH・Gutzwiller・OptTrans・NSplitSize など | 既存の共通検査を流用(`validate_parton_data(…; require_mf_inputs = false)`) | |
| `JULIA_MVMC_INNER_THREADS` が立っている | 警告して逐次で走る | §4.1 |
| `DSROptStaDel = 0` | **止めない**(記録だけ) | ユーザー判断 |

## 8. 数値ポリシー

- Pf は生の複素値で持つ(乗法更新 + 定期的な厳密再計算)。特異 = Pf 0 = ノード
- **Pf は自前の Parlett–Reid(列ピボットつき、`_parton_pair_pfaffian!`)、G = F⁻¹ は LU(`lu!` + `inv!`)の後に
  反対称へ射影する**。PfaPack の LTL 部品は交互ブロック構造の正則行列でゼロピボットを返すので使わない(§3)。
  厳密再計算は O(M³) が 2 回(Pf と LU)になるが、錨と測定のサンプルごとにしか呼ばれない
- 振幅の縮約は**転置積**で書き、`dot()` は使わない(本体 §7 と同じ罠)。2K×2K の Pf は小さいので専用の割り当てなしルーチンで計算する
- ホットループでの確保はゼロにする。GB・X⁻¹ は ws に置く
- **受理のたびに G を厳密に反対称へ射影する**(G ← (G − Gᵀ)/2、O(M²))。P1 で見つけた不安定性への対処:
  - 更新式 G′ = G + GB·X⁻¹·GBᵀ は BᵀG = −(GB)ᵀ(G の反対称性)を使っている
  - 丸め誤差で G に乗った対称成分は、X を通って U·sym(X⁻¹)·Uᵀ として戻る。そのため受理のたびに等比的に増幅される
  - 実測(K=2・N=3・n_qp=3): 1 回ごとに約 10 倍、40 回で G の相対誤差 1e6、Pf は NaN
  - 射影すると 1e−14 前後で止まる。上流の 1 電子更新 `InvM[i][j] += v1[i]v2[j] − v1[j]v2[i]` が構造的に反対称な形をとるのと同じ理由
  - T1-d が「高速更新の後も G == −Gᵀ」を検査する
- それ以外の誤差は、`PartonBlockUpdateSize` 回の受理ごとの厳密再計算で戻す(det モードと同じ周期)
- 錨で特異だった qp(Pf = 0)があれば、受理時に `:need_recompute` を返す。乗法更新では 0 から戻れないので、
  移動後に正則になっていても取りこぼさないようにする。det モードは 0 のまま進むが、再計算は乱数を消費しないので
  連鎖には影響しない
- デバッグ用の恒等式:
  - G·F = I
  - G の反対称性
  - 錨の冪等性
  - 高速更新を重ねた結果 = 厳密再計算(T1-d)

## 9. テスト計画

| 番号 | 内容 | 方法 | 置き場 |
|---|---|---|---|
| T0 | 既存のビット一致 | 既存スイート全緑(着手時: Parton 2988 / Unit 793 / Slater 25 / 基本 15)+ α・Φ の E2E 出力を変更前とバイト比較 | 既存 + 手動 |
| T1-a | 包含定理 | ランダム複素軌道、K = 2, 3, 4、ランダム配置で Pf F / Π det が配置によらず一定(QP あり・なしの両方、比例定数が qp にもよらない) | 新 `test_parton_pair_linalg.jl` |
| T1-b | SVD 射影子 | 行ブロック SVD の射影子 = Φ^a の射影子 | 同上 |
| T1-c | 比と G 更新 | 契約 2・3 が直接計算と相対 1e−10 以内で一致。Pf(C⁻¹) の符号、交換で (−1)^K | 同上 |
| T1-d | 契約 1 vs 3 | 複素 f・QP あり・|S| = K で、多数回の高速更新の後に厳密再計算と一致。`:need_recompute` の経路 | 同上 |
| T1-e | S の零空間 | 初期点 ΞJΞᵀ で、厳密列挙の共分散の零固有値の数 = J の自由度(idx 空間に射影した次元)+ 全体スケール。RedCut で落ちないことを記録する | 同上 |
| T1-f | K = 2 で上流と一致 | 同じ def(行を I<J で書いたもの)から、上流の fsz の F(`orbital_idx_matrix` / `orbital_sgn` を直接組む)と Pf モードの F が二重占有配置で要素ごとに一致する(行の並べ替えを除く) | 同上 |
| T1-g | 入出力 | def と InPartonPairOrbital の厳格検査(異常系)、`zqp_parton_pair_opt.dat` → InPartonPairOrbital の往復がビット一致 | 新 `test_parton_pair_io.jl` |
| T1-h | 乱数初期化 | 同じ seed で同じ値、凍結成分は 0、分布の式が上流と同じ、**サンプリング用の乱数列を消費しない**(乱数初期化の run と、その `_init.dat` を読ませた run とで、サンプリングの系列と step 0 の E がビット一致) | 同上 |
| T2 | 勾配 | O と ln ip の中心差分(Re/Im を別々に。idx 共有・符号・QP・Jastrow 込み)。全配置を列挙し、上流の `calculate_oo!` / `build_s_matrix_and_g_vector!` を通した g/(−dt) が ∂E/∂Re p・∂E/∂Im p に一致する。共役を外すと崩れること、store 経路と非 store 経路の両方 | 新 `test_parton_pair_gradient.jl` |
| T3 | PartonMode = 1 との一致 | 同じ α\*・占有・シードで、PartonMode = 1 と(ΞJΞᵀ で初期化した)PartonMode = 2 を 1 SR ステップ走らせる。step 0 の E が丸め誤差で一致し、受理数が一致する。QP あり・なし。全対の def | 新 `test_parton_pair_run.jl`(トイ系)+ P2 で KM/CB の E2E |
| T4 | ED ベンチ(CI 外) | 推奨試走条件(dt 0.003・StaDel 0.1・RedCut 1e−8・NVMCSample 2000・1 プロセス)で次の 2 通りを回す(ユーザー判断)。(i) **乱数初期化**から(本番の使い方。主の報告)。(ii) **PartonMode = 1 の最良解からの継続**(共有なし def の系だけ。出発点が厳密なので、包含定理どおり E_Pf ≤ E_det になることの裏取り)。E・ED との差・E_det との比較・最適化後の σ_N/σ_{N+1} を報告する | 物理検証 run |

T4 の対象系(ユーザー判断 2026-10-31: **4 系すべて**):

| 系 | K | N | M | ED | PartonMode = 1 の最良 | f の自由度(共有なし) |
|---|---|---|---|---|---|---|
| CB 4×4 ν=1/2 | 2 | 8 | 16 | −16.304913 | −16.28315(c4n2) | 496 |
| KM 4×4 ν=1/2 | 2 | 4 | 8 | −4(α で厳密) | −4.0000 | 496 |
| CB 6×3 ν=1/3(U = 1) | 3 | 6 | 18 | −12.126195 | −12.0675(2+1) | 1431 |
| KM 9×4 ν=1/3 | 3 | 12 | 36 | −3.941102 | −3.93989(indep) | 5778(並進で共有して約 1/4 に) |

- KM ν=1/2 はすでに厳密なので、健全性の確認(−4 から動かないこと)にしかならない。意味のあるベンチは CB 4×4 と CB 6×3 / KM 9×4
- KM 9×4 は共有なしだと SR 行列が約 2 GB になるので、生き残った並進で共有した def(外部スクリプト)を使い、
  乱数初期化(i)だけを回す。継続(ii)は CB 4×4・CB 6×3・KM 4×4 で行う

速度: 1 sweep(`parton_make_sample!` 1 回)の時間を、高速更新版と「提案ごとに全再計算する版」とで比べ、比を報告する(ベンチスクリプトを tools に置く)。

## 10. マイルストーン・後回し

- **P0**(この文書): 調査と設計。コードは書かない
- **P1**: 次をマイルストーンごとに全緑でコミットする。T1・T2 が緑になること
  1. 型・テンプレート・契約 1〜3(T1-a〜d)
  2. 入出力・門番・ロケータ(T1-g)
  3. O と SR の配線・ループ・ドライバ(T2)
  4. 変換スクリプト・SVD 出力(T1-b、T1-e、T1-f)
- **P2**: T3・T4、既存の回帰のビット一致(T0)、速度の報告

後回し:

- スレッド並列の測定(`PartonMainCalThreadContext` の型パラメータ化。逐次でビット一致を固めてから)
- **高速化はペンディング**(ユーザー判断 2026-10-31)。当面は MPI(サンプル並列)で回す。16 ランクで CB 4×4 は
  1 step 0.079 s(1 プロセスの約 10 倍速)。再開するときの候補:
  1. **W = 𝔽G の前計算**(𝔽[β, J] = f[β, x_J]、K·Ns × M): 比に要る d_sᵀG が W の行の引き算になり、
     1 提案 O(K·M²) → O(K²·M)、受理時の W の更新は O(K²·Ns·M)。E_loc も同じ式で軽くなる。上流と無関係な式の改良。
     **実装するときは NSRCG と同じく modpara のキーで選べる形にし、今の直接計算を残す**(ユーザー判断 2026-10-31)。
     既定(0)は今の実装でビット一致を保ち、1 で前計算を使う。両者が丸め誤差で一致することをテストで確かめる。
     キー名は実装時に決める(略さない名前)
  2. **QP ごとのスレッド並列は上流の更新を待つ**。上流の Julia 移植では QP ループのスレッド化はデバッグ用の
     オプトイン(`JULIA_MVMC_PFAPACK_THREADS`、C 版との一致検査は未通過、PfaPack の呼び出しは既定でロック)で、
     方針が固まっていない(`docs/manual/05_compatibility.md:104-134`)。Pf モードの契約は自前のコードなので
     上流の並列化は自動では効かない。上流のゲートと作業領域の規約が固まったら、同じ方針で Pf の qp ループに当てる
- E_loc の前計算 Q = 𝔽G𝔽ᵀ(実測で E_loc が支配的なら。上の W の前計算に含まれる)
- `parton_finalize_oo_store!` の Φ 版との共通化(合流後)
- KN 奇数のための不対軌道(拡張 Pf)
- SROptData のメモリ見積もりの門番(本体 §12.8 の残件と共通)
- 本体 DESIGN への統合(§13 化)と `docs/parton_run.md` への手順の追記

## 11. 決定ログと残りの確認点

2026-10-31 のユーザー判断:

| 項目 | 決定 |
|---|---|
| 作業場所 | worktree `../Julia-mVMC-pf`、ブランチ `parton-pf` |
| DESIGN の置き場 | 別ファイル(この文書) |
| 進め方 | プロンプトの P0 手順(調査 → DESIGN → 確認 → 実装) |
| 入力 | 新キー + 専用の厳格パーサ |
| パラメータの器 | 専用の器 |
| F の規約 | ×2・書かれた向きと符号(省略時 +1)。K = 2 で上流と一致させる |
| S の零空間 | 門番なし・記録だけ(StaDel = 0 では走らせない) |
| 更新経路 | `NExUpdatePath = 6` を共用(固縛移動のみ) |
| K | K ≥ 2 |
| T4 | CB 4×4・CB 6×3・KM 9×4・KM 4×4 の 4 系。乱数初期化からと、PartonMode = 1 からの継続(共有なし def の系)の両方 |
| 初期値 | 指定が無ければ seed に従った乱数(分布は上流と同じ、列は初期値専用)。InPartonPairOrbital は任意 |
| `bloch` ペアリング | 作らない(PartonMode = 1 の結果を初期値に使う予定が少ない)。変換スクリプトは T3・T4 用の最小限 |
| T4 の def(2026-10-31 追加) | 同フレーバー対を行ごと外し、拡大セルの並進で共有する(`tools/parton_gen_pair_def.jl`) |
| T4 の継続 run | 回さない(共有 def では素朴な J の射影が壊れるため。§13.4)。T4 は乱数初期化だけ |

残りの確認点(P1 に入る前に):

1. **出力名**: §6(`zqp_parton_pair_*`、`zvo_parton_pairdiag.dat`)でよいか

## 12. upstream 報告候補(DESIGN_parton.md §11 に合流時に移す)

- OrbitalGeneral のパーサが C 形式の 6 列を誤読する(`orbital_parser.jl:205-226`)。経路は実質未検証
- fsz 経路で未列挙の対が黙って idx 0 に結び付く(`orbital_qptrans_utils.jl:55`)
- 軌道の出力が行ごと・行番号を idx として書き、ヘッダも 4 行なので、idx を共有すると InOrbital と往復できない(`data_io.jl:235-245`)
- 複素 FSZ のホッピングで TwoSz ≠ −1 のとき常に spin flip を選ぶ(`vmc_sampling.jl:4056`。実数版 `:6593-6635` は正しい)
- `sync_modified_parameter!` の docstring は「≤ D_AMP_MAX を保証」だが、実装は常に 4 にそろえる(`parameter_sync.jl:78, 88`)
- **PfaPack の `pfaffian_ltl!` が複素行列で誤った値を返す**(n ≥ 4。実数と n = 2 は正しい。§3)。PfaPack のテストは
  実数か n = 2 の複素しか見ていないので通っている
- **PfaPack の `julia_zsktf2!` が交互ブロック構造の正則行列でゼロピボット(info > 0)を返す**(§3)。上流の本番経路
  (sz は `julia_zsktf2_turbo!`、fsz は `julia_zsktf2!` で info を捨てている)にも同じ形の行列が来うるかは未調査

## 13. 実装の記録(P1、2026-10-31)

コミット(ブランチ `parton-pf`): `fa2132a`(型・契約 1〜3)→ `bb8f561`(入力・初期値・D_AmpMax・門番・ロケータ)→
`ad116e0`(O・SR ループ・ドライバ)→ `ea0a7f2`(SVD 出力・変換道具・Pf の部品の差し替え)。

### 13.1 テスト(すべて緑)

| 層 | 着手時 | P1 後 |
|---|---|---|
| MVMCOptimizers の Parton Mode Tests | 2988 | 4401(Pf 1413 件を追加。既存 2988 件は無変更で緑) |
| Unit / Slater / 基本 | 793 / 25 / 15 | 同じ |
| パーサ | 全緑 | 全緑(Pf のパーサ 29 件を追加) |
| P 層(test/physics) | 1292 | 1292 |

対応する受け入れテスト(§9):

- T1-a〜e・T1-g・T1-h: `test_parton_pair_linalg.jl` / `_io.jl` / `_gradient.jl` / `_tools.jl`
- T2: `test_parton_pair_gradient.jl`(O = 中心差分、力 = 勾配を store・非 store の両経路で、共役シムの回帰ガード)
- T3 のトイ版: `test_parton_pair_skeleton.jl`(同じ乱数列・同じサンプル列)と `test_parton_pair_run.jl`
  (E2E で step 0 の E・E²・受理率が α と一致。QP あり・なし)

### 13.2 1 sweep の時間(`tools/parton_pair_bench.jl`、1 スレッド)

1 sweep = NSite 回の提案。f と Φ は複素乱数、QP なし。

| 系 | M | Pf 高速更新 | 提案ごと全再計算 | 比 | det モード(参考) |
|---|---|---|---|---|---|
| CB 4×4 相当(K=2, N=8) | 16 | 0.013 ms | 0.080 ms | 6.4 倍 | 0.002 ms |
| KM 9×4 相当(K=3, N=12) | 36 | 0.112 ms | 0.927 ms | 8.2 倍 | 0.011 ms |

- 「提案ごと全再計算」は契約 1(Pf と LU、O(M³))の時間 × NSite で見積もった
- Pf の高速更新は det モードの約 6〜10 倍(1 提案 O(K·M²) 対 O(K·N))。E_loc は 1 ホップごとに同じ O(K·M²) が掛かる

### 13.3 実装中に決めたこと・見つけたこと

- **受理のたびに G を反対称へ射影する**(§8)。射影しないと、丸めで乗った対称成分が等比的に増幅されて 40 回で NaN に至る
- **契約 1・2 の Pf は自前の Parlett–Reid、G は LU**(§3・§8)。PfaPack の `julia_zsktf2!` は、2×2 ブロックの直和や
  C⁻¹ そのもののような正則行列でも info = 2 を返す
- ロケータの f のブロックは Φ の直前に置いた(Φ の「末尾」前提の式に触れないため)。Pf モードの並びは [射影 | p]
- 測定は逐次だけ(スレッド並列の経路は、共有のスレッド文脈を一般化するまで使わない。§10)
- pairdiag に log₁₀|ip| の最小・最大は入れていない(共有の `parton_main_cal!` を触らずには取れない)

### 13.4 P2 の経過(2026-10-31)

**T0(既存モードのビット一致)— 合格**。変更前(d7cc568 を `git archive` で展開)と変更後で、同じ入力を回して全出力を比べた。

| 入力 | 一致したファイル |
|---|---|
| α の最小入力(8 step) | 12 本すべてバイト一致 |
| Φ の最小入力(8 step) | 8 本すべてバイト一致 |
| KM 4×4 の α run(Jastrow・QP 込み、20 step) | 13 本すべてバイト一致 |

runinfo は時刻と githash の行を除いて一致。実行時間のファイル(time・CalcTimer)は比較から外した。

**T3 の実系版 — 合格**。α の最良解(stage1)から、α 側は同じ α*・占有・Jastrow・seed で 1 step、Pf 側は変換道具
(全対 def、adjacent ペアリング)で 1 step 回した。

| 系 | E(α) | E(Pf) | 相対差 | 受理率(α / Pf) | 射影の残差 | σ_N/σ_{N+1} |
|---|---|---|---|---|---|---|
| KM 4×4 ν=1/2(K=2) | −4.0000000000 | −4.0000000000 | 0 | 0.283194 / 0.283194 | 6.8e−17 | 3.4e15 |
| CB 4×4 ν=1/2(K=2) | −16.2711327608 | −16.2711327608 | 0 | 0.209479 / 0.209479 | 9.4e−17 | 3.1e15 |
| CB 6×3 ν=1/3(K=3) | −12.0695397441 | −12.0695397441 | 2.9e−16 | 0.106049 / 0.106049 | 6.9e−17 | 3.6e15 |

**T4 の準備で分かったこと**:

- **全対・共有なしの def は重すぎる**。CB 6×3(Ns = 36、K = 3)では f が 5778 個(n_para 5814)になり、1 step 22.8 s。
  内訳は ⟨O O†⟩ の組み立て 61%・SR の求解 33% で、Pf そのもの(サンプリング・E_loc・O)は 4%(実測、CalcTimer)
- そこで **def を制限した**(ユーザー提案): 同フレーバー対を行ごと外し(構造的 0)、平均場を保つ並進
  (`pl_mf_shifts`)で写り合う対を共有する。生成は `tools/parton_gen_pair_def.jl`。使った並進が α run の pmfpara の
  idx 写像を保つことも検査する(4 系とも成立)
  - 同フレーバー対の 0 固定は OptFlag 0 ではなく**行を外す**(ユーザー判断。OptFlag 0 だと ⟨O O†⟩ の蓄積が減らない)

| 系 | f の数(全対 → 共有 def) | 1 step(共有 def、試走) |
|---|---|---|
| CB 4×4 ν=1/2 | 2016 → 256 | 0.79 s |
| CB 6×3 ν=1/3 | 5778 → 648 | 0.94 s |
| KM 4×4 ν=1/2 | 496 → 64 | 0.16 s |
| KM 9×4 ν=1/3 | 5778 → 972 | 1.38 s |

- **共有 def への継続は素朴な J では成り立たない**。射影の残差は 0.71〜0.94 になり、KM 9×4 は全サンプルで
  |ip| < 1e−30(事実上 0 の状態)、CB 4×4 はほぼ特異な状態になった(KM 4×4 だけは E = −4 のまま)。
  並進不変な J は、各フレーバーの占有軌道への並進の作用 TΦ^a = Φ^a U_T で J を平均すれば作れる
  (J ← 平均_T U_T J U_Tᵀ。Bloch 基底の対角化は要らない)が、**継続は回さない**ことにした(ユーザー判断)
- 試走で、契約 3 の `lu!(X)` が SingularException を投げて止まる不具合を踏んだ(|R| が下限を超えていても X が
  数値的に特異になりうる)。LU が特異なら `:need_recompute` を返すよう直した(`2f0928e`)

**T4 本番**: 乱数初期化だけを 4 系、推奨試走条件(dt 0.003・StaDel 0.1・RedCut 1e−8・2000 step)で回した。
1 プロセスでは時間がかかるので、MPI 16 ランク(ランクあたり NVMCSample 125、合計 2000)に切り替えた(ユーザー了承)。
CB 4×4 で 1 step 0.079 s(1 プロセスの約 10 倍速)。その後、1 段目の出力を In で読み込む追い込みの段
(dt 0.001・StaDel 0.05・RedCut 1e−10、ユーザー指定)も回した。run は `playground_nozomi/parton_pf/T4/`(本体側、git 管理外)。

| 系 | 1 段目 | 追い込み後 | ED | PartonMode = 1 との差 |
|---|---|---|---|---|
| CB 4×4 ν=1/2 | −16.2704 | −16.2719 ± 0.0003 | −16.3049 | +0.0035 |
| CB 6×3 ν=1/3 | −11.5551 | −11.6229 ± 0.0007 | −12.1262 | +0.44(セクター不一致。下記) |
| KM 4×4 ν=1/2 | −3.8361 | −3.8770 ± 0.0006 | −4 | +0.12 |
| KM 9×4 ν=1/3 | −3.7443 | −3.7615 ± 0.0004 | −3.9411 | +0.18(セクター不一致。下記) |

### 13.5 T4 で分かったこと(2026-10-31)

**(1) 並進で共有した def は、Pf の状態の運動量を縛る**。f を並進 T で符号 χ_T(±1)付きで共有すると、F(Tr) = χ_T F(r)
なので Pf の状態の T の固有値は χ_T^{M/2} に限られる。一方 PartonMode = 1 の det^K は、占有軌道の運動量の和を持てる。
Ly = 4 のトーラスで 1 本のバンドを埋めると y 運動量の和は 0 + π/2 + π + 3π/2 ≡ π(偶数長の環の偶奇の効果)。

| 系 | M/2 | det^K の固有値(共有した並進) | 旧 def(符号 +1) |
|---|---|---|---|
| CB 4×4 / KM 4×4(K=2) | 8 / 4 | すべて +1 | 合う |
| CB 6×3(K=3) | 9 | x の 3 セル並進で −1 | **合わない** |
| KM 9×4(K=3) | 6 | y の 1 並進で −1 | **合わない**(M/2 偶数なので ±1 の符号ではどうやっても作れない) |

- 対処は mVMC の標準どおり、**その並進を f の共有から外し(f のセルを大きくする)、QP に入れて重みでセクターを選ぶ**
  (ユーザー判断。符号つきの共有はパートンに境界条件のひねりを課すのと同じで、標準ではない)。
  CB 6×3 は f のセル 6×1・QP 6 本(重み (−1)^x)、KM 9×4 は f のセル 9×2・QP 6 本(y の 1 並進で −1 = Ky = π)。
  QP の重みは、α run の重みを同じ運動量の指標として新しい代表に広げたもの(`T4/make_sector_inputs.jl`)
- **検証**: この def と QP で、PartonMode = 1 の解が厳密に表せる(異フレーバーのブロックが乱数の J を f の共有群で
  平均して正則な J を作り、射影の残差 ≤ 1e−14、ip_Pf / ip_det は 200 配置で 1.6e−12 まで一定)。
  素朴な(エネルギー順の)異フレーバー組の J を平均すると特異になりうるので、乱数の J から平均すること
- 同じセルでも、Pf の対関数は det^K より狭い運動量のセクターしか表せない。def で並進を共有するときは、
  目的の状態の運動量と合っているかを必ず確かめる(`check_momentum.jl` の方法: 占有軌道への並進の作用 U_T の det)

**(2) KM 9×4 の diagCut の 37 刻みの出入りは実装ミスではない**。最適化で Jastrow の最近接の斥力が強まり(v = −1.5 → −2.8)、
2 粒子が隣り合う配置がほとんどサンプルされなくなる(2000 中 10)。そのとき最近接の対(x・y 方向に 36 個ずつ)と、
その方向の Jastrow 1 個の S の対角が 0 になり、RedCut でまとめて落ちる(36 + 1 = 37)。S の零方向は
パートンゲージ(実 37 次元、理論どおり)・Jastrow の一様シフト(1)・ほとんどサンプルされない最近接の対、で全部説明できた。
StaDel を 0.1 → 0.01 にしても出入りは変わらない(E はむしろ −3.744 → −3.721)

**(3) KM 4×4 を厳密解(PartonMode = 1 の det² を射影した f)から始めると、2000 step の間 E = −4.000000・σ²/E² ~ 1e−16 のまま**。
SR は厳密解の上で安定。乱数初期化で −3.88 に止まったのは、実装やサンプリングではなく最適化の地形(初期値)の問題

**(4) フレーバー対称・空間反対称の def**(`tools/parton_gen_pair_def.jl --flavor-sym`)は Ψ ∝ Pf(g)^K の族になる。
並進で共有すると各因子 Pf(g) が運動量 0 に限られるので、運動量 π の det Φ を因子に持つ det^K は含まない(KM 2 系で確認。
並進平均した B が特異)。空間対称にすると K が奇数で Pf ≡ 0。試走は保留
