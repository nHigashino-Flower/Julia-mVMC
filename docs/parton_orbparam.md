# Φ 直接変分モード(PartonOrbParam = 1)の使い方

パートン波動関数の Slater 行列式の軌道行列 Φ_f[s,n](フレーバー f、サイト s、軌道 n)を、
SR の変分パラメータとして直接最適化するモード。上流 mVMC が Pfaffian の f_ij を直接変分するのと
同じ扱いで、Pfaffian の `orbitalidx.def` と同じ規約の制限(パラメータの共有・凍結・Bloch 形)を
def で掛けられる。

設計の根拠は `DESIGN_parton.md` §12.11(上流型)と §12.12(制限付き Φ・Bloch 形)、
試走の結果は `playground_nozomi/projects/kapit_mueller/km/REPORT_orbparam.md`(git 管理外)にある。

## 1. α モードとの違い

| | α モード(`PartonOrbParam 0`、既定) | Φ 直接変分モード(`PartonOrbParam 1`) |
|---|---|---|
| 変分するもの | 平均場ハミルトニアンのホッピング α(`pmfpara.def`) | 軌道行列 Φ の要素(制限があれば θ) |
| Φ の作り方 | H_MF(α) を対角化して下から Ne 本(占有の選び方あり) | θ から直接組み立てる |
| 開殻(縮退した占有) | 勾配が 0 にされて動かない | 問題なく動く |
| パラメータ数 | 少ない(ホッピングの種類の数) | 多い(制限なしで F·Ns·Ne)。制限で減らす |

開殻の状態(例: CFL の f₃ のフェルミ海)や、α の平均場では表せない軌道を扱いたいときに使う。

> 物理量の測定(NVMCCalMode = 1、⟨b†_i b_j⟩・⟨n_i n_j⟩)は `docs/parton_run.md` の「物理量測定」節。

## 2. 最小の入力

### modpara.def

α モードの modpara に次を足す(または書き換える)。

```
PartonMode      1
PartonOrbParam  1        # Φ 直接変分
PartonOrbAmpMax 4.0      # 省略可。再スケールの値(上流の D_AmpMax と同じ既定値)
PartonOccMode   0        # Φ モードには占有の選択が無いので 0(または行を消す)
DSROptStaDel    0.1      # 必ず正の値(下の注意)
```

- **`DSROptStaDel` は正でなければならない**。行列式は列の基底の取り替え(GL(Ne))で状態が
  変わらないので、S 行列が厳密な零方向を持つ。0 だと SR の連立方程式が解けない
- **`PartonOrbParam` のつづりに注意**。modpara の未知キーは黙って無視され、α モードとして
  走り出す(α 側の門番が「pmfpara が無い」と言って止まる)
- その他の条件は α モードと同じ(`NVMCCalMode 0`、`ComplexType 1`、`NSRCG 0` か `1`、
  `NExUpdatePath 6`、`2Sz 0`、`NSplitSize 1` など)

### namelist.def

```
ModPara        modpara.def
PhysHop        physhop.def
CoulombInter   coulombinter.def
Jastrow        jastrowidx.def      # 任意
TransSym       qptransidx.def      # 任意(量子数射影)
InPmfOrb       InPmfOrb.def        # 任意(初期値。無ければ乱数)
PartonOrbIdx   pmforbidx.def       # 任意(制限。無ければ全要素が独立)
InJastrow      InJastrow.def       # 任意
```

**α の入力(`PartonMFTrans` / `PartonMFPara` / `InPmfPara` / `InPmfOcc`)は書かない**(書くと error)。

### 実行

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=MVMCOptimizers.jl -e \
  'using MVMCOptimizers; exit(parton_run_para_opt_from_namelist(ARGS[1]; output_dir = ARGS[2]))' \
  <dir>/namelist.def <out_dir>
```

α モードと同じ入口で、modpara の `PartonOrbParam` を見て自動で分岐する。MPI での並列は
`docs/parton_run.md` を参照。

### 試走の目安

新しい系を試すときは次の条件から始める(n_para が大きいので、サンプル数を多めに取る)。

```
DSROptStepDt 0.003
DSROptStaDel 0.1
DSROptRedCut 1.0e-8
NSROptItrStep 2000
NVMCSample   2000      # 少ないと雑音で E がむしろ上がる。目安は NVMCSample ≫ 2·n_para
```

## 3. 変分パラメータと再スケール

- SR が動かすのは θ(制限が無ければ Φ の全要素そのもの)。Φ は毎ステップ θ から組み立てる
- **SR 更新のたびに、θ の最大絶対値を `PartonOrbAmpMax` にそろえる**(上流の D_AmpMax と同じ。
  縮めるだけでなく拡大もする)。行列式の大きさが極端になるのを防ぐためで、状態は変わらない
- そろえる単位は「θ を共有するフレーバーの組」。共有が無ければフレーバーごと
- 列の基底の取り替え(GL(Ne))のうちスケール以外は正準化しない(上流型)。漂流は
  `zvo_parton_orbdiag.dat` の `log10cond_*` で監視する(7 節)

## 4. 初期値

1. **乱数**(既定): OptFlag が可動の θ は (U[−1,1) + i·U[−1,1))/√2、凍結は 0
   (Pfaffian の初期化と同じ分布)。乱数は初期化専用の列で、`RndSeed` から決まる。
   サンプリングの乱数列は消費しない
2. **`InPmfOrb`**: あれば乱数の値を上書きする。書式は 5 行ヘッダ + `idx Re Im`(idx は θ の
   番号、0-based)。**全 θ を書く**(部分的な指定は error)

```
===============================
NPmfOrb 108
===============================
== idx RePhi ImPhi ==
===============================
0  1.234000000000000000e+00 -5.000000000000000000e-01
1  ...
```

前の run の `zqp_pmforb_opt.dat` をそのまま `InPmfOrb` に渡せば続きから再開できる
(制限の def は同じものを使う)。

## 5. 制限(PartonOrbIdx)

Pfaffian の `orbitalidx.def` と同じ規約で、Φ の各要素を θ に結び付ける:

  **Φ_f[s,n] = c·θ_idx**

```
===============================
NPmfOrbIdx 8                     # θ の個数 N
ComplexType 1                    # 読まれない(書式を orbitalidx.def にそろえるための行)
===============================
===============================
site flavor orbital idx [c]      # 全 F·Ns·Ne 要素を 1 行ずつ(0-based)
...
idx OptFlag                      # N 行。1 = 可動、0 = 凍結
...
```

- 係数 c は、省略で 1、1 列なら符号 ±1(Pfaffian 互換)、2 列なら `Re Im`(|c| = 1)
- **全要素の記載が必須**。欠け・重複・範囲外・どの要素にも使われない idx は error
- ゼロに固定したい要素は「OptFlag 0 で初期値 0 の idx」に結び付ける(Pfaffian と同じ)

### 書ける制限の例

| やりたいこと | 書き方 |
|---|---|
| フレーバー 1 をフレーバー 0 と同じにする | フレーバー 1 の各要素に、フレーバー 0 の同じ (site, orbital) と同じ idx を書く |
| 同じ軌道の中でサイトを同一視する | 複数のサイトの (site, flavor, orbital) に同じ idx を書く(符号・位相も付けられる) |
| あるフレーバーを固定して他だけ動かす | そのフレーバーの idx をすべて OptFlag 0 にする(例: CFL で f₁f₂ を固定し f₃ だけ動かす) |
| ゼロ固定 | OptFlag 0・初期値 0 の idx に結び付ける |

**注意: 列(軌道)の番号には空間的な意味が無い**(列の基底は自由に取り替えられる)。そのため、
Pfaffian の f_{Ti,Tj} = f_ij のような並進の同一視は、要素ごとの共有だけでは書けない
(全軌道が「運動量 0」に縛られてしまう)。並進対称性を入れたいときは、次の Bloch 形を使う。

共有しすぎて Φ の列が平行になると(例: 全サイトを 1 つの θ に寄せる)、初期化の直後に
「列フルランクでない」という error で止まる。

## 6. Bloch 形(拡大ユニットセルの並進対称性)

大規模系でパラメータを減らすための制限。拡大ユニットセル(平均場が持つ並進の周期)の並進で
状態が不変になるように、各軌道を Bloch 関数にする:

  **Φ_f[s,(k,b)] = e^{ik·a(s)}·θ_{(f, σ(s), k, b)}**

- σ(s): 拡大セル内での位置。a(s): セル内の代表サイトから s へ移す拡大セルの並進
- 変分パラメータはフレーバーあたり N_σ·Ne 個(全要素なら Ns·Ne)。**拡大セルの数 N_cell 分の 1**
- CDW のような周期を持つ平均場も、拡大セルをその周期に取れば同じ枠に入る

### 波数 k はどう決まるか

**k は自分で計算しなくてよい。拡大セルの並進群から自動で決まる。**

- 生成道具が `parton_lattice.dat` の `MFSHIFT`(拡大セルの並進の一覧)を読み、その並進群の
  既約表現(= 拡大セルのブリルアンゾーンの k 点)を数え上げる。個数はちょうど N_cell
- 各 k は格子の波数 2π(m1/Lx, m2/Ly) の代表 (m1, m2) で表す(同じ指標を与える (m1, m2) は同一視)
- 例: KM 9×4 で拡大セル 9×1 なら、並進は y 方向の 4 個だけなので k = (0, 2πm2/4)、m2 = 0〜3
- **各 k の Bloch 関数の中身(セル内の振幅 u_k)は変分パラメータ**で、すべての k を 1 回の run で
  同時に最適化する。k ごとに別の計算をする必要はない

**自分で決めるのは「どの k に何本の軌道を置くか」(占有)だけ:**

- 既定は「全 k に NElec / N_cell 本ずつ」(割り切れなければ error)。バンドをちょうど埋める状態
  (例: Chern バンドを満たす KM ν=1/3 の各フレーバー)はこれでよい
- フェルミ海などは占有ファイルで指定する。行は `flavor m1 m2 nbands`、フレーバーごとの本数の
  合計は NElec

```
# フレーバー 2(f₃)だけフェルミ海、ほかは全 k に 1 本(例)
0 0 0 1
0 0 1 1
...
2 0 0 1
2 0 1 2
...
```

- **占有は最適化の途中で変わらない**(状態の量子数に当たる)。占有を変えると、拡大セルの並進に
  ついての全運動量(Σ k)が変わる。量子数射影(`TransSym`)のセクターや ED と比べるときは、
  占有と射影のセクターが同じ運動量を指しているか確かめること。α 解から変換した場合は
  α の状態の占有がそのまま使われるので、この点は自動でそろう

### def の作り方

```bash
# stage_in には parton_lattice.dat と modpara.def が要る(def 生成の道具が書く)
julia --project=MVMCOptimizers.jl tools/parton_bloch_orbidx.jl <stage_in> <out_def> [occupation_file]
```

出力に N_cell と θ の個数が表示される。k 点の一覧は Julia から確認できる:

```julia
include("tools/parton_bloch_orbidx.jl")
bs = parton_bloch_structure("<stage_in>")      # 占有ファイルは occupation_file = "..."
bs.kpoints        # k の代表 (m1, m2)
bs.columns        # フレーバーごとの軌道列 (k の位置, band)
bs.map.n_idx      # θ の個数
```

`physhop.def` が拡大セルの並進で不変でなければ警告が出る(Bloch 形の制限自体は有効だが、
物理 H の対称性と合っていないので良い変分状態にならない可能性がある)。

### 対象外

- ゲージ変換を伴う並進(磁気並進)。`MFSHIFT` に載っている素の並進だけを使う
- パートン境界条件のひねり θ_f(k がずれる分は未対応)

## 7. α 解から始める

α モードの最良解から Φ モードを始めるのが確実(α と同じ状態から出発するので、E は α の値から始まる)。

```bash
# 制限なし(全要素)
julia --project=MVMCOptimizers.jl tools/parton_alpha_to_orbparam.jl \
    <alpha_stage_in> <alpha_stage_out> <new_dir>

# Bloch 形(<alpha_stage_in>/parton_lattice.dat が要る)
julia --project=MVMCOptimizers.jl tools/parton_alpha_to_orbparam.jl \
    <alpha_stage_in> <alpha_stage_out> <new_dir> --bloch [occupation_file]
```

- α の最終値(`zqp_pmfpara_opt.dat`)と占有(`zqp_pmfocc_opt.dat`)から軌道を組み、`InPmfOrb.def` に書く
- `--bloch` では、α の射影子 P = ΦΦ† を k ごとのブロックに分けて対角化し、固有値 ≈ 1 の
  固有ベクトルから θ を作る(近似なし)。α の状態が拡大セルの並進で不変でない、または占有が
  合わないと error になる
- Jastrow の最終値は `InJastrow.def` に直して引き継ぐ。modpara は `PartonOrbParam 1`・
  `PartonOccMode 0` に書き換わる(`DSROptStaDel` が 0 なら error)
- 変換後の `<new_dir>` の modpara で、step 数・サンプル数などを試走の条件に直してから回す

## 8. 出力

`zvo_out.dat`・`zvo_var.dat`(Jastrow のみ)・`zvo_SRinfo.dat`・`zvo_conv.dat`・`zqp_opt.dat` は
α モードと同じ。Φ モード固有のもの:

| ファイル | 内容 |
|---|---|
| `zqp_pmforb_init.dat` | 開始時の θ(再スケール後)。`InPmfOrb` の書式 |
| `zqp_pmforb_opt.dat` | 最終の θ。`InPmfOrb` としてそのまま読み戻せる |
| `zvo_parton_orbdiag.dat` | step ごとの診断(下表) |
| `zvo_parton_runinfo.dat` | 末尾に `PartonOrbParam`・`PartonOrbAmpMax`・`n_orb`(Φ の要素数)・`NPmfOrbIdx`(θ の個数)・`PartonOrbIdx`(def のパス、無ければ none) |

`zvo_parton_orbdiag.dat` の列:

| 列 | 意味 | 見どころ |
|---|---|---|
| `step` | SR のステップ | |
| `n_skipped` | 捨てたサンプル数(このランク。ip が 0 か非有限、または E_loc が非有限。2026-11-05 までは \|ip\| < 1e−30) | 増え続けたら行列式のあふれ・0 落ち |
| `accept_ratio` | Metropolis の受理率 | |
| `n_need_recompute` / `n_exact_recompute` | 高速更新の破綻回数 / 厳密再計算の回数 | 急増したら Φ の条件数の悪化 |
| `xmax_pre_<組>` | 再スケール前の max\|θ\|(組ごと) | D からのずれ = SR 更新の大きさ |
| `log10cond_<フレーバー>` | Φ_f の log10(σ_max/σ_min) | **単調に増え続けたら正準化しないことの弊害**(下の注意) |

## 9. 注意と目安

- **致命的な問題の目安**(DESIGN §12.11): `log10cond` が単調に増え続ける / SR が NaN で止まる /
  捨てるサンプルが増え続ける / α の最良解から始めて E が α より上がる。これらが出たら、
  §12 本体の正準化(Löwdin)を検討する。2026-10-31 の試走ではどれも出ていない
- **速さ**: SR 用の ⟨O O†⟩ は n_para の 2 乗、連立方程式は 3 乗で効く(Φ モードではこの部分を
  BLAS で組んでいる)。Bloch 形で n_para を減らすと、残りはパラメータ数に依らない E_loc と
  サンプリングが主になる(KM 9×4 で全体の 3/4)
- **よくある error**:

| メッセージの要点 | 原因 |
|---|---|
| `needs DSROptStaDel > 0` | `DSROptStaDel` が 0 |
| `does not use PartonMFTrans` など | namelist に α の入力が残っている |
| `PartonOrbIdx: ... line ...` | def の書式違反(行番号が出る) |
| `holds N entries but ... are required` | InPmfOrb の件数が θ の個数と違う(def を変えたのに前の InPmfOrb を使った等) |
| `not of full column rank` | 共有しすぎ、または InPmfOrb の値で列が平行 |
| `P(k) has ... eigenvalues ≈ 1` | `--bloch` で、α の状態が並進不変でない / 占有が合わない |
| `NElec ... is not divisible` | Bloch の既定の占有が作れない。占有ファイルを渡す |
