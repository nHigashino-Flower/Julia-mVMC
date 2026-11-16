# Pfaffian パートンモード(PartonMode = 2)の使い方

<!-- --- parton-mode (fork addition) --- -->

パートン波動関数を、K 個のフレーバーをまとめた 1 本の Pfaffian

    Ψ(x) = Σ_qp w_qp · Pf F_qp(x),   F_IJ = f[(r_I, a_I), (r_J, a_J)]

で置き換えるモード。M = K·N 個のパートン(粒子 m のフレーバー a、行番号 I = (m−1)K + a)を同じ対関数 f で組む。
f の要素は SR で直接変分する(上流 mVMC の Pfaffian の f_ij と同じ扱い)。

- **PartonMode = 1(det^K)を厳密に含む**: f = ΞJΞᵀ(Ξ = diag(Φ¹,…,Φᴷ)、J は反対称)なら
  Pf F = ±Pf(J)·∏_a det Φ^a(x_a)(包含定理、DESIGN §1.3)。だから Pf モードの最適値は Mode 1 以下になる
- K = 2 なら上流の一般スピン Pfaffian と同じ形(スピンをフレーバーと読み替える。f は上流と同じ ×2 規約)

設計の正は `DESIGN_parton_pf.md`(決定ログは §11、実装と試走の記録は §13)。ベンチマークの run は
`playground_nozomi/dev/parton_pf/T4/`(git 管理外)にある。

## 1. PartonMode = 1 との違い

| | PartonMode = 1(α / Φ) | PartonMode = 2(Pf) |
|---|---|---|
| 波動関数 | ∏_a det Φ^a(x_a) | Pf F(x)(det^K を含む) |
| 変分するもの | 平均場のホッピング α、または軌道 Φ | 対関数 f の要素(def の idx ごと) |
| 初期値 | α = 1(物理のホッピングを分けた平均場 = 正しい相から出発) | **既定は乱数**(構造を仮定しない) |
| 1 提案のコスト | O(K·N) | O(K·M²)(約 6〜10 倍、§9) |
| パラメータ数 | 少ない | 多い(def の共有で減らす) |

## 2. 最小の入力

### modpara.def(KM 4×4、ν = 1/2、K = 2 の例)

```
NVMCCalMode 0
Nsite 16
NElec 4
NFlavor 2
PartonMode 2
NExUpdatePath 6          # 全フレーバーが同じ座標でホップする固縛移動(これ以外は門番が拒否)
ComplexType 1
2Sz 0
NMPTrans 2               # QP の本数(qptransidx.def と合わせる)
RndSeed 1001             # f の乱数初期化にも使う(初期値専用の乱数列。サンプリングの列は消費しない)
NVMCWarmUp 500
NVMCInterval 1
NVMCSample 125           # ランクあたり。MPI 16 ランクで合計 2000
NSROptItrStep 2000
NSROptItrSmp 100
DSROptStepDt 0.01        # §5 の推奨
DSROptStaDel 0.01        # §5 の推奨(0.1 は遅すぎる)
DSROptRedCut 1e-8
PartonBlockUpdateSize 16 # 厳密再計算の周期
PartonGaugeFix 1         # Pf モードでは使わない(runinfo に「該当なし」)
PartonFlavorSymFast 0    # 同上
PartonPairAmpMax 4.0     # 省略可(2026-11-05 追加)。SR 後に max|p| をこの値へそろえる(既定 4 = 上流の D_AmpMax)
```

`PartonMode 2` で切り替える。Pf 専用の modpara キーは `PartonPairAmpMax` だけ(省略可、§9)。門番が拒否するもの:
`NFlavor < 2`、K·N が奇数、`NExUpdatePath ≠ 6`、`PartonOrbParam ≠ 0`、`PartonOccMode ≠ 0`、
`PartonPairAmpMax` が正の有限値でない、`PartonOrbAmpMax ≠ 4`(Φ 用のキー。Pf では `PartonPairAmpMax` を使う)、既存スピン Orbital 系、α・Φ の入力
(PartonMFTrans / PartonMFPara / InPmfPara / InPmfOcc / InPmfOrb / PartonOrbIdx)、`NVMCCalMode ≠ 0`(物理量測定は未対応)。
逆に PartonMode ≠ 2 の run に PartonPairOrbital / InPartonPairOrbital / `PartonPairAmpMax ≠ 4` を書くと、α・Φ 側の共通検査が止める。
`DSROptStaDel = 0` は止めないが、S にゲージの零方向があるので必ず正の値を入れること(§9)。

### namelist.def

```
ModPara             modpara.def
PhysHop             physhop.def
Jastrow             jastrowidx.def
TransSym            qptransidx.def
CoulombInter        coulombinter.def
PartonPairOrbital   partonpairorbital.def      # 必須: f の idx 構造
InPartonPairOrbital InPartonPairOrbital.def    # 任意: f の初期値(無ければ乱数)
InJastrow           InJastrow.def              # 任意
```

### PartonPairOrbital(f の idx 構造)

上流の `orbitalidxgen.def` と同じ列構成で、スピンをフレーバーと読み替える。K = 2 なら上流の def がそのまま通る。

```
=============================================
NOrbitalIdx     64
ComplexType          1
=============================================
=============================================
<site1> <flavor1> <site2> <flavor2> <idx> [<sgn>]     ← データ行(0-based、sgn は ±1、省略時 +1)
...
<idx> <optflag>                                        ← 末尾の OptFlag 行(1 = 動かす、0 = 凍結)
```

- f[(site1,flavor1),(site2,flavor2)] = 2·sgn·p_idx、逆向きは −2·sgn·p_idx(反対称)
- **def に書かない対は構造的な 0**(パラメータではない)。上流の fsz 経路のように黙って idx 0 に結び付けない
- 厳格に検査する: 範囲・対角の禁止・同じ非順序対の重複(逆向きを含む)・idx の連番・OptFlag の網羅

### InPartonPairOrbital(f の初期値)

上流の InOrbital と同じ 5 行ヘッダ(`NOrbitalIdx`)+ `idx Re Im`。**全 idx を含むこと**(部分的な上書きはしない)。
出力の `zqp_parton_pair_opt.dat` がそのままこの形式なので、**前の段の出力を次の段の初期値に渡すだけで継続できる**。

## 3. def の作り方

全対・共有なしの def はパラメータが多すぎて重い(CB 6×3 で f が 5778 個、1 step 22.8 s。DESIGN §13.4)。
次の 2 つの制限を掛けた def を道具で作る。

1. **同フレーバー対を外す**(行を書かない = 構造的 0)。OptFlag 0 の凍結ではパラメータ数が減らないので行ごと外す
2. **拡大ユニットセルの並進で共有する**: 平均場を保つ並進(`pl_mf_shifts(model, ex, ey)`)で写り合う対を同じ idx にする

```bash
cd /home/nozomihigashino/Julia-mVMC
# KM: <Lx> <Ly> <φ> <K> <ex> <ey>(拡大セル)。最後に α run の pmfpara.def を渡すと並進の整合を検査する
julia --project=MVMCOptimizers.jl tools/parton_gen_pair_def.jl out.def km 4 4 1/2 2 4 1
# checkerboard: <nx> <ny> <K> <ex> <ey>
julia --project=MVMCOptimizers.jl tools/parton_gen_pair_def.jl out.def cb 4 4 2 2 2
```

| 系 | f の数(全対 → 共有 def) | 1 step(MPI 16、合計 2000 サンプル) |
|---|---|---|
| KM 4×4 ν=1/2(セル 4×1) | 496 → 64 | 0.02 s |
| CB 4×4 ν=1/2 | 2016 → 256 | 0.08 s |
| KM 9×4 ν=1/3(セル 9×1) | 5778 → 972 | 0.52 s |
| KM 9×4 ν=1/3(セル 9×2、§4 のセクター合わせ) | 1944 | 2.08 s |

`--flavor-sym` はフレーバー対称・空間反対称の def(Ψ ∝ Pf(g)^K)を書く。並進で共有すると det^K を含まないことがある
(DESIGN §13.5 (4))ので、通常は使わない。

## 4. ★運動量のセクター(def で並進を共有するときの必須確認)

f を並進 T で(符号 χ_T = ±1 付きで)共有すると F(Tr) = χ_T F(r) なので、**Pf の状態の T の固有値は χ_T^{M/2} に限られる**。
一方 det^K は占有軌道の運動量の和を持てる(Ly = 4 でバンドを 1 本埋めると y 運動量の和は π)。

| 系 | M/2 | det^K の固有値(共有した並進) | 並進をそのまま共有した def |
|---|---|---|---|
| CB 4×4 / KM 4×4(K = 2) | 8 / 4 | すべて +1 | 合う |
| CB 6×3(K = 3) | 9 | x の 3 セル並進で −1 | 合わない |
| KM 9×4(K = 3) | 6 | y の 1 並進で −1 | 合わない(±1 の符号ではどうやっても作れない) |

**合わない並進は f の共有から外し(f のセルを大きくする)、QP に入れて重みでセクターを選ぶ**(mVMC の標準どおり)。
KM 9×4 なら f のセル 9×1 → 9×2、QP は x{0,3,6} × y{0,1} の 6 本で、y の 1 並進に重み −1(Ky = π)。
作り方は `playground_nozomi/dev/parton_pf/T4/make_sector_inputs.jl`。

**確かめ方**: 目的の状態(例えば Mode 1 の解)の占有軌道 Φ^a に並進を作用させ、U_T = Φ^a† P_T Φ^a の det の積を見る。
新しい def と QP で Mode 1 の解が厳密に表せること(異フレーバーのブロックに乱数を詰めた J を共有群で平均し、射影の残差
≤ 1e−14、ip_Pf / ip_det が配置によらず一定)を確かめたスクリプトは `playground_nozomi/dev/parton_pf/T4/checks/check_sector.jl`、
運動量の確かめ方は同じ場所の `check_momentum.jl`(DESIGN §13.5 (1))。

注意: 運動量を合わせても、**同じ運動量の位相縮退(トーラス上の多重項)のどれに乗るかは選べない**。
CB 6×3 ν=1/3 は ED の 3 重項が全部 K = (−0.5, 0) で C2 だけが違い、パートン解は C2 = −1 側に落ちる
(本体 `playground_nozomi/projects/checkerboard/nu12_boson/REPORT_sector.md`)。この系の ED との差は実装の物差しにならない。

## 5. 初期値と SR の設定(★ここが一番効く)

### 初期値

- **既定は乱数**(InPartonPairOrbital を書かない)。分布は上流と同じ(p = [2(r₁−½) + 2i(r₂−½)]/√2、OptFlag 0 は 0)、
  乱数列は `RndSeed` から作る初期値専用の列で、サンプリングの列を消費しない
- 読み込んだ値・乱数のどちらも、max|p| = `PartonPairAmpMax`(既定 4.0 = 上流の D_AmpMax)にそろえてから始める。SR の各ステップ後も同じ
- `zqp_parton_pair_init.dat` に実際に採用した初期値が出るので、これを In に渡せばビット再現できる
- PartonMode = 1 の解から始める道具は `tools/parton_alpha_to_pair.jl`(T3・継続の検証用)。並進で共有した def では
  素朴な J だと射影が厳密でない(残差 0.7〜0.9)。厳密にするには J を共有群で平均する(§4)

### SR の推奨設定(KM 4×4 ν=1/2、乱数スタート、seed 1001、MPI 16・合計 2000 サンプルで実測)

| dt | StaDel | 初めて −3.99 を下回った step | 5000 step の E | 要した虚時間 τ = 2·dt·step |
|---|---|---|---|---|
| 0.003 | 0.1 | 約 11300(追い込みを挟む) | (14000 step で −3.99928) | 約 68 |
| 0.01 | 0.1 | 2679 | −3.99996 | 54 |
| 0.003 | 0.01 | 3503 | −3.99969 | 21 |
| 0.01 | 0.01 | 1060 | −4.00000 | 21 |
| 0.003 | 0.001 | 1591 | −4.00000 | 9.5 |
| **0.03** | **0.01** | **156** | **−4.00000** | 9.4 |

(ED = −4。どの run も再スケール前の max|f| ≤ 4.13 で、暴れは無かった)

- **必要な虚時間を決めているのは StaDel**。対角を (1+δ) 倍する正則化なので、δ = 0.1 では S の固有値が対角の 1 割を
  下回る方向(多数のパラメータを協調して動かす方向。乱数の f からパートン構造へ移るのに要る)の更新がほぼ消える。
  0.1 → 0.01 → 0.001 で必要な τ が 54〜68 → 21 → 9.5 に減った
- **dt はその τ を step 数に換算するだけ**で、0.03 まで安定だった。乱数スタートの 1 段目は **dt 0.01〜0.03・StaDel 0.01** から始める
- 「dt を絞る追い込み」は、虚時間あたりで頭打ちになってから(§6)。まだ下がっているうちに dt を絞ると 1 step の進みが遅くなるだけ
  (KM 4×4 で dt 0.003 → 0.001 にしたら、虚時間あたりの降下は同じで、step あたり 3 倍遅くなった)
- 継続は前の段の `zqp_parton_pair_opt.dat` を InPartonPairOrbital に、`zqp_jastrow_opt.dat` を 5 行ヘッダに直して InJastrow に渡す
  (`playground_nozomi/dev/parton_pf/T4/prepare_extend.jl`)

**未検証**: 上の表は 1 seed・KM 4×4(実パラメータ 164、サンプル 2000 で S が十分に決まる)だけ。実パラメータがサンプル数を
超える系(KM 9×4 は 2052〜3996 > 2000。S の階数が足りない)で StaDel を下げて安定かはまだ確かめていない。短い試走で
max|f|(pairdiag の xmax_pre)と E の跳びを見てから長く回すこと。

## 6. 収束の判定(表現力の限界と最適化不足を取り違えないために)

```bash
julia --project=/home/nozomihigashino/Julia-mVMC/tools tools/plot_parton_conv.jl <out_dir> [--window 100] [--dt 0.01]
```

Pf の出力(`zvo_parton_pairdiag.dat` があり `zvo_parton_diag.dat` が無い)を自動で見分けて、8 枚の図を
`<out_dir>/parton_conv.pdf` に描く。PartonMode = 1 の出力に対しては従来と同じ図と出力(この経路には入らない)。

| パネル | 内容 |
|---|---|
| 1 | E(生 + 移動平均)と σ²/E²(右軸) |
| 2 | log\|E_avg − E_tail\|(直線 = 指数緩和、τ_E) |
| 3 | \|ΔE\| / 100 step と 10·SE(step 単位の旧条件。参考) |
| 4 | フレーバーごとの σ_N/σ_{N+1} と受理率 |
| **5** | **dE/dτ vs step(符号つき・線形軸)と ±2·SE の帯。帯が 0 をまたいだら頭打ち**(判定 1)。縦軸は後半 6 割の値に合わせる |
| 6 | 同じ量の対数版: \|dE/dτ\| と 2·SE(全体の桁の推移) |
| 7 | 判定 1 の t 値(±2 の線) |
| 8 | 判定 2: σ_N/σ_{N+1} の window 間の相対変化(5% の線、最大のフレーバー) |

dE/dτ の各点は、**それまでの 2·window step を 10 ブロックに分けた平均に直線を当てはめた傾き**(点は window/2 step ごと)。
1 step の差分や 100 step 平均どうしの差は使わない。KM 4×4 の末尾(dE/dτ ≈ −5e−5、dt 0.003)では、1 step の差分の
信号/雑音は 3e−4、100 step 平均どうしの差でも 0.3 以下で雑音に埋もれる。区間内の全 step を使う当てはめなら同じ区間長で
ずっと精度が出る。区間の長さは `--window` で変える(run の長さの 1/20〜1/40 が目安)。

標準出力の例(末尾の時点の判定):

```
E_tail = -3.99930 ± 4.7e-05   σ²/E²_tail = 1.08e-04   drift = -2.8e-05(条件 2: 満たす)
dE/dτ = -5.36e-05(dt = 0.003、τ = 2·dt·step)  t = -1.2 → 頭打ち
τ_E = 1101 step/e-fold
構造: σ_N/σ_{N+1} の window 間の変化 2.1%(最大のフレーバー)→ 落ち着いた
```

dt は `--dt` か、`<out_dir>/../modpara.def`・`<stage>_in/modpara.def` から読む。

**表現力の限界と言ってよいのは、次の 5 つをすべて満たしたときだけ**(どれも基底状態を知らなくても使える):

1. **虚時間で停留**: dE/dτ の |t| < 2。step 単位の条件(|drift| ≤ 10·SE)は dt を絞った段で通ってしまうので使わない
2. **構造が落ち着いている**: σ_N/σ_{N+1} がもう動いていない。E は O(q²) なので構造(O(q))より先に平らに見える
   (Mode 1 で min_gap を見ていたのと同じ理由)
3. **揺さぶっても下がらない**: dt を上げる・StaDel を下げるなどで続けても E が下がらない(鞍点の除外)
4. **seed によらない**: 乱数の seed を変えても同じ E に誤差の範囲で集まる
5. **包含関係と矛盾しない**: 変分空間に入っていると分かっている状態(Mode 1 の最適解、制限した def の最適解)より高くない。
   高ければ確実に最適化不足

実例(KM 4×4、dt 0.003・StaDel 0.1、`T4/km44_nu12_rand_s3`): step 4500〜6500 で E が −3.962 → −3.971 と平らに見えたが、
t = −8〜−14 で、σ_N/σ_{N+1} も 1000 step で 25〜40% 動いていた(パネル 5 の帯は 0 をまたがず、パネル 8 の判定 2 は
5% を大きく超えている)。特異値比が 1 まで下がって 2 本が入れ替わった step 6800 の直後に、
E は 500 step で −3.994 まで落ちた。**分散外挿(同じ軌跡で E = E₀ + c·σ²)は使えない**(KM 9×4 の旧 def で
E₀ = −3.836 と出て、Mode 1 の −3.940 より上 = 誤って「表現力の限界」と読める数字になった)。

## 7. 実行

```bash
cd /home/nozomihigashino/Julia-mVMC/MVMCOptimizers.jl
# 1 プロセス
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. -e \
  'using MVMCOptimizers; exit(parton_run_para_opt_from_namelist(ARGS[1]; output_dir = ARGS[2]))' \
  <run_dir>/namelist.def <out_dir>
# MPI(サンプル並列。NVMCSample はランクあたり)。システムの mpiexec は PMIx 不整合で落ちるので MPI.jl 付属のものを使う
MPX=$(julia --project=. -e 'using MPI; MPI.mpiexec(p -> print(p))')
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $MPX -n 16 julia --project=. run_case_mpi.jl <run_dir>/namelist.def <out_dir>
```

`run_case_mpi.jl` は `BLAS.set_num_threads(1)` してから `parton_run_para_opt_from_namelist` を呼ぶだけ
(`playground_nozomi/dev/parton_pf/T4/run_case_mpi.jl`)。BLAS を 1 に固定しないと 16 ランク × 16 スレッドがコアを奪い合う。
QP ごとのスレッド並列は上流の更新を待つ(DESIGN §10)。

> **パラメータが多い run(共有なしなど、数千〜)は `NSRCG 1`(SR-CG)が速い**。直接法は 1 ノード分のコアしか使えないので、
> 複数ノードの本番も SR-CG で。`docs/parton_run.md` の「SR の解法」節。

> 物理量の測定(NVMCCalMode = 1、⟨b†_i b_j⟩・⟨n_i n_j⟩)は `docs/parton_run.md` の「物理量測定」節。

## 8. 出力

| ファイル | 内容 |
|---|---|
| `zqp_parton_pair_init.dat` / `_opt.dat` | f の値(idx ごと)。InPartonPairOrbital の形式そのもの。往復でビット一致 |
| `zqp_parton_pair_sv_init.dat` / `_opt.dat` | フレーバーごとの行ブロック SVD の全特異値と σ_N/σ_{N+1} |
| `zqp_parton_pair_orb_init.dat` / `_opt.dat` | 上位 N 本の左特異ベクトル(InPmfOrb と同じ形式。Φ モードの入力や Chern の計算にそのまま渡せる) |
| `zvo_parton_pairdiag.dat` | step・捨てたサンプル数(ip が 0 か非有限、または E_loc が非有限。2026-11-05 以降)・受理率・need_recompute・厳密再計算・再スケール前の max\|p\|・各フレーバーの σ_N/σ_{N+1} |
| `zvo_parton_runinfo.dat` | 既存の項目 + `PartonMode 2`・n_pair_idx・M・`D_AmpMax`(= 使った `PartonPairAmpMax`) |
| `zvo_out` / `zvo_SRinfo` / `zvo_parton_time` / `zvo_CalcTimer` / `zqp_opt`(Jastrow) | 既存どおり |

σ_N/σ_{N+1}: 各フレーバーの行ブロック f[(·,a), :] の N 番目と N+1 番目の特異値の比。パートン形(f = ΞJΞᵀ)なら ∞
(厳密解スタートで 3e11)。乱数スタートで −4 に届いた KM 4×4 の解は 7.5 で、det² そのものではない Pf で厳密エネルギーに届いている。

## 9. 注意と目安

- **|ip| の大きさと `PartonPairAmpMax`(2026-11-05)**: Pf(λF) = λ^{M/2} Pf F なので、`PartonPairAmpMax` は物理(比・E_loc・O・SR)を
  変えず、|ip| の絶対値だけを決める。サンプルを捨てるのは上流と同じく ip が 0 か非有限、または E_loc が非有限のときだけ
  (以前は |ip| < 1e−30 で捨てていて、Li–Wu ν=2/3 6×6(M = 72)で 1 成分の跳び → 再スケールで |Pf| が 10^{−43} 倍 → 全サンプルを
  捨てて NaN、になった)。今は |ip| が浮動小数点の範囲(〜1e±308)を外れるときだけが問題で、目安は |Pf| ≲ (AmpMax·√M)^{M/2}
  (あふれ側)。大きい系であふれそうなら上流と同じく下げ、小さすぎて 0 に落ちるなら上げる。`n_skipped` が増えたら疑うこと

- **S の零方向**: パートンゲージ(f → λ_α λ_β f、∏_a λ_(i,a) 一定)・Jastrow の一様シフト・ほとんどサンプルされない対。
  RedCut は対角しか見ないので消せない。**StaDel > 0 が必須**
- **diagCut の段々の出入り**: 最近接の斥力が強い系(KM 9×4 は V = 7)では、隣り合う配置がほとんどサンプルされず、
  ある種類の最近接の対が 1 step の全サンプルに 1 つも現れないと、その種類の f(最近接の対にしか現れない idx)と
  Jastrow 1 個の O が厳密に 0 になって RedCut で落ちる。diagCut はその塊の単位で動く
  (KM 9×4 で 37 = 36 + 1(セル 9×1)、73 = 72 + 1(セル 9×2)、種類は Jastrow の最近接 idx の 6 個)。実装ミスではない。
  抑えるにはサンプル数を増やすしかない(StaDel では変わらない)。最近接の f を凍結するのは det^K を変分空間から外すので勧めない
- **1 提案のコスト**は det モードの約 6〜10 倍(KM 9×4 相当で Pf 高速更新 0.112 ms / sweep、DESIGN §13.2)
- **Pf は自前の Parlett–Reid と LU**。PfaPack の `pfaffian_ltl!` は複素 n ≥ 4 で値を誤り、`julia_zsktf2!` は正則行列でも
  info = 2 を返す(upstream 報告候補、DESIGN §12)
- 受理のたびに G = F⁻¹ を反対称へ射影している(しないと丸めの対称成分が等比的に増えて 40 回で NaN)
- 物理量の測定(`NVMCCalMode = 1`)は `docs/parton_run.md` の「物理量測定」節(2026-11-01 対応)
