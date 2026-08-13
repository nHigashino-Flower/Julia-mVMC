# パートンモードの実行手順(並列・ジョブスクリプト)

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
