# パートンモードの実行手順(並列・ジョブスクリプト)

> Φ 直接変分モード(`PartonOrbParam = 1`、制限 `PartonOrbIdx`・Bloch 形)の入力と使い方は `docs/parton_orbparam.md`。

<!-- --- parton-mode (fork addition) --- -->

設計の正は `DESIGN_parton.md`(§3.3.1 出力一覧 / §3.3.2 CalcTimer)。
ここは**実行時の落とし穴**だけをまとめる。

## スレッド数の決まり方(最重要)

**Julia のスレッド数は `JULIA_NUM_THREADS` または `julia -t N` で決まる。
`OMP_NUM_THREADS` では決まらない。** C 版 mVMC のジョブスクリプトを流用すると、
`OMP_NUM_THREADS=8` と書いてあっても警告なしに 1 スレッドで走る
(パートンドライバは起動時にこの構成を検出して `@warn` を出す。
実際に何スレッドで走ったかは `zvo_parton_runinfo.dat` の `n_julia_thread` で
事後確認できる)。

パートンの測定フェーズ(`parton_main_cal!`)のサンプル並列は
`JULIA_MVMC_INNER_THREADS=1` の opt-in(既定は逐次)。並列でも出力は逐次と
**バイト一致**する(DESIGN §7 性能ポリシー)。

## Slurm ジョブスクリプト例

```bash
#!/bin/bash
#SBATCH --ntasks=4               # MPI ランク数(サンプル並列)
#SBATCH --cpus-per-task=8        # ランクあたり Julia スレッド数

# Julia のスレッド数は OMP_NUM_THREADS では決まらない
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK   # または julia -t $SLURM_CPUS_PER_TASK
export JULIA_MVMC_INNER_THREADS=1               # 測定フェーズのサンプル並列(opt-in)

srun julia --project=. run_parton.jl namelist.def
```

## 並列の役割分担

| 層 | 何を並列化 | 効果 |
|---|---|---|
| MPI(ランク) | 独立なマルコフ連鎖(rank ごとに seed オフセット) | **統計が n_rank 倍** |
| Julia スレッド | 測定フェーズの保存済みサンプル評価 | 同じ統計を速く |

- **実効サンプル数 = `NVMCSample × ランク数`**。`weight_average_we!` が comm0 で
  allreduce して合算後の重みで正規化するため、統計誤差の母数はこの総数
  (`zvo_parton_runinfo.dat` の `nvmc_sample_total`、および
  `zvo_parton_time.dat` ヘッダにも併記)
- `NSplitSize > 1`(グループ内サンプル分割)はパートンでは未検証のため門番が拒否する
  (DESIGN §10)
- BLAS: スレッド並列区間ではドライバが自動で `BLAS.set_num_threads(1)` にし、
  抜けるときに元へ戻す。設定値は runinfo の `blas_num_threads` に記録される

## 時間・性能の出力

- `zvo_CalcTimer.dat`(ID 800–813): **区間別の累積時間**。パートンでは既定 ON、
  `MVMC_C_TIMER=0` で切れる
- `zvo_parton_time.dat`: **ステップ毎の経過時間とサンプリング量**
  (`n_out / n_in / n_sample_total / n_update_total`)。ヘッダに並列構成
  (`n_mpi_rank / n_julia_thread / nvmc_sample_total`)を併記しているので、
  このファイル単体で「1 更新あたりの時間 vs 並列数」のスケーリング解析ができる。
  初回ステップだけ `n_out = NVMCWarmUp + NVMCSample` で長く、
  2 回目以降は `NVMCSample + 1`(burn 再開)になるのは仕様

## 物理量測定(NVMCCalMode = 1)

仕様は `DESIGN_parton.md` §13。α・Φ・Pf の 3 モードで共通に使える。固定したパラメータ(最適化の出力)で
射影後の波動関数をサンプリングし、物理粒子 b(F が偶数ならボソン、奇数ならフェルミオン。硬芯・スピンなし)の
1 体 ⟨b†_i b_j⟩ と 2 体 ⟨b†_i b_j b†_k b_l⟩ を測る。ゲージ依存量 ⟨f†f⟩ は対象外(H_MF から直接得られる)。

### 手順

最適化の入力ディレクトリをコピーし、次の 3 つを変える。入口は最適化と同じ
`parton_run_para_opt_from_namelist`(MPI なら `run_case_mpi.jl` もそのまま使える)。

1. **modpara.def**: `NVMCCalMode 1`、`NDataQtySmp <ビン数>`(5 程度。ビンごとに 1 組のファイルが出て、
   ばらつきから統計誤差を見積もれる)、`NDataIdxStart 1`(ファイル番号の始まり)。`NVMCSample` はランクあたり・ビンあたり
2. **greenone.def / greentwo.def** を作る:
   `julia tools/parton_gen_green_def.jl <Nsite> <dir>`(全 (i, j) の 1 体と全 (i, i, j, j) の密度相関)
3. **namelist.def** に Green の def と、最適化の出力から作る In ファイルを足す

```
OneBodyG       greenone.def
TwoBodyG       greentwo.def
InPmfPara      zqp_pmfpara_opt.dat     # α
InPmfOcc       zqp_pmfocc_opt.dat      # α(必須)
InJastrow      InJastrow.def           # Jastrow の項があるとき(必須)
```

| モード | 必須の In | 最適化の出力のどれか |
|---|---|---|
| α | `InPmfPara`(pmfpara.def が全 idx の値を持つなら不要)と **`InPmfOcc`** | `zqp_pmfpara_opt.dat`・`zqp_pmfocc_opt.dat` |
| Φ | `InPmfOrb` | `zqp_pmforb_opt.dat` |
| Pf | `InPartonPairOrbital` | `zqp_parton_pair_opt.dat` |
| 共通 | Jastrow の項があれば `InJastrow` | 段の引き継ぎと同じ方法で作る |

欠けていれば門番が止める(測定では乱数で埋めた値を使わない)。α で `InPmfOcc` を必須にしているのは、
MOM の枝で終わった解を aufbau で組み直すと**別の状態を黙って測る**ことになるため。

### 測れる 2 体

1 粒子の移動で済む 4 類だけ。**別々の 2 粒子を動かす項(i ≠ j、k ≠ l、k ≠ j)は門番が止める**。

| 類 | 添字 | 例 |
|---|---|---|
| density | i = j、k = l | ⟨n_i n_k⟩ → S(q) |
| hop_then_density | i ≠ j、k = l | ⟨b†_i b_j n_k⟩ |
| density_then_hop | i = j、k ≠ l | ⟨n_i b†_k b_l⟩ |
| same_particle | i ≠ j、k ≠ l、k = j | ⟨b†_i b_j b†_j b_l⟩ = ⟨b†_i (1 − n_j) b_l⟩ |

スピン列はすべて 0。`TwoBodyGEx` は使えない。

### 出力(上流と同じ書式、番号 = ビン + NDataIdxStart)

| ファイル | 中身 |
|---|---|
| `zvo_out.dat` | ビンごとに 1 行(E の実部・虚部・⟨H²⟩・規格化分散 …) |
| `zvo_cisajs_001.dat` … | `i 0 j 0 Re Im`(⟨b†_i b_j⟩) |
| `zvo_cisajscktalt_001.dat` … | `i 0 j 0 k 0 l 0 Re Im`(⟨b†_i b_j b†_k b_l⟩) |

検算: 各ビンで Σ_i ⟨n_i⟩ = N、および E = Σ_physhop [t⟨b†_j b_i⟩ + t*⟨b†_i b_j⟩] + Σ_coulomb V⟨n_i n_j⟩ が
丸め誤差で成り立つ(同じサンプルで組んでいるため)。小さい系なら全配置列挙の厳密値と比べられる
(`MVMCOptimizers.parton_exact_expectations`)。

### 注意

- 測定ループは逐次(スレッド並列なし)。並列は MPI のサンプル並列で(BLAS = 1、MPI.jl 付属の mpiexec)
- NVMCCalMode = 0(最適化)で OneBodyG / TwoBodyG を書くと warn が出て、読まれない
