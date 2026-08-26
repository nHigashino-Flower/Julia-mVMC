# Julia-mVMC を fork して運用又は追加機能実装

## CLAUDE.mdの運用ルール
・定期的にCLAUDE.mdの体裁を整える
・恒常的に認知が必用なことは逐次 CLAUDE.mdに書き足す。
・セッションをリセットした時に、続きから再開しやすいように、TODOリスト等を活用する。

## 現在の状況(2026-08-26 更新)

### ★新規: Kapit-Mueller 模型を追加(2026-08-26)

「mVMC parton mode に Kapit-Mueller 模型を追加し、φ=1/2 の ν=1/2 FCI と
φ=1/3 の ν=1/3 FCI で ED にどれだけ迫れるか」というユーザー依頼。

- **設計**: `docs/superpowers/specs/2026-08-26-kapit-mueller-parton-design.md`(唯一の正)
- **判断ログ**: `playground_nozomi/km/PROJECT.md`
- **一行結論**: **ν=1/2 はパートンアンザッツが厳密解**
  (E = −4.000000000000 / −6.000000000000、分散 0、ED と完全一致、最適化不要)。
  ν=1/3 は 9×4 で不足 0.0028(step 601 時点、最適化継続中)
- **模型抽象層 `test/physics/lattice_model.jl` を新設**し `parton_fixture` を模型一般化。
  checkerboard は `CheckerboardLatticeModel` として既存関数へ委譲するので**ビット一致**。
  P 層は 1054 → **1292 全緑**
- **ED 側**(別リポジトリ `~/ED`)に `KapitMuellerLattice/Model/System` と
  **`AbstractStatistics` 軸(Fermion / HardcoreBoson)** を追加。既存 13 テストは
  ビット一致のまま **14/14 全緑**。出力は
  `Data/ExactDiagonalization/KapitMueller/<statistics>/Lx=..-Ly=..-N=../<model>/`
- **重要な設計事実**(すべてスパイクで実測):
  - **Landau ゲージ必須**。`G(z; φ/F)^F = G(z; φ)` が厳密成立するのは Landau だけ
  - **変位和方式**。元 ED の `select_minimum_bonds` は対蹠変位で並進対称性を壊す
  - **Lx は F/φ の倍数**(φ·Lx ∉ ℤ だと H が非エルミート)。6×6 は使えない → 9×4 / 9×5
  - **hopmax = 8.0 が既定**。第 3 近接(2.0)では **ν=1/2 が FCI にならない**
    (多重項幅 0.125 > 外ギャップ 0.047)
  - **QP 並進射影は自明**。パートンバンドが完全充填なので `T_x^q|MF⟩ ∝ |MF⟩`。
    F 重の位相縮退は**パートン境界条件のひねり**で作る(REPORT_sector.md と同じ構造)
- **踏んだ罠**(`playground_nozomi/km/PROJECT.md` に詳細):
  idx クラスの鍵は **(セルクラス, dx, dy) ではなく軌道の最小元**にすること /
  KM にはトーラス自己像のオンサイト項があり `coulombinter` の対角行で渡す /
  自己交換ペアは **Ly が偶数**のときに出る(クラスごと OptFlag=0 で凍結)
- **ローカルの `Manifest.toml` が消えていた**。julia 1.10 は `[sources]`(1.11 機能)を
  解釈しないので、`Pkg.develop` に 4 パッケージを**一括で**渡して復旧する
  (1 つずつだと相互依存で解決できない)

### checkerboard(従来の作業)


### ★最優先で読む: `playground_nozomi/cb_nu12_boson/REPORT_sector.md`

2026-08-26 の一連の検証(seed 不安定性の診断 → セクター不変量 → 厳密列挙 →
トーラス位相セクター → Step 5 再最適化)を**再開可能な形でまとめた唯一の入口**。
判断ログの本体は `PROJECT.md` の 2026-08-26 の 3 エントリ。

**一行結論**: seed 不安定性の主因は**パートン平均場がどの位相セクターに落ちるか**で、
ν=1/3 のクリーン解は**トーラス位相縮退の誤った状態(E3)に乗っていた**。
パートン境界条件の Z_3 ひねり(y 方向)で正しい状態に移し、厳密勾配で再最適化して
**⟨E1|Ψ⟩² = 0.0030 → 0.8676 / E = −12.0486**(ED −12.1262、残差 0.0776)。
残差はほぼ全部**多様体外への 11.2% の漏れ = 表現能力の天井**。

- **アンザッツ比較キャンペーンは 2026-08-26 18:39 に停止**(135/170 run 完了、残り 35)。
  解析にコアを回すためのユーザー判断。再開は
  `submit_ansatz.jl --nu 1/3 --wave 1 --k13 -0.5 0`
- **新しい道具**(すべて `playground_nozomi/cb_nu12_boson/scripts/`、git 管理外):
  `sector_scan.jl`(セクター不変量)/ `exact_enum.jl` + `step2_exact.jl`(全配置の厳密列挙)/
  `torus_sector.jl`(境界ひねり)/ `step5_*.jl`(2 セクターアンザッツの厳密勾配最適化)
- **ED 固有ベクトルを素のバイナリに抽出済み**(`ed_nu13_6x3_vecs.bin` /
  `ed_nu12_4x4_vecs.bin`)。**JLD2 も julia 1.8.5 も通常は不要になった**
- **σ²/E²(`zvo_out.dat` の 4 列目)は最終 E の順位を ρ = +0.72〜1.00 で予測する**。
  ED 不要の品質指標として使える。ただし**早期(step 150)では効かない**


・パートン平均場モードの**基本実装が一通り完了**(DESIGN v3.14 / fixture 向き正準化まで)。
　§8 のテスト 0〜18 と P 層 P0〜P2(+ fixture orientation)全 442 が全緑。
・**8×8 未ドープのベンチマーク達成**: 共変 ef4 で E = −65.0794、gap 開、
　**C = −1 を 3 経路(平滑 k / native 射影 / Bott)が機械精度で一致**して確認
　(`runs_orientfix/L08_ef4_s1009`)。chi-VMC(−65.05)より低く、基底状態 = FCI を再現。
・**ホールドープ探索の 4×4 は完了**(`runs_doped/` の 90 run が全完走、走行中のジョブなし)。
　フェーズ A(60run)= QP ありが 3 充填すべてで勝ち → フェーズ B は QP あり確定。
　フェーズ C(bond_flavor 30run、2026-08-20 完了)= **軌道縮約 96 idx の縛りは
　δ=0 の不足 0.031 のうち 0.005 しか説明しない**(→ 主因は Jastrow 側)。
　bf は**ドープ系で 0/20 全滅**(局所 U(1) ゲージで diagCut が Npara の半分に張り付く)。
　4×4 の ED 比較: 未ドープ不足 0.031 / h1 不足 0.039(N=7 の ED 基底は 8 重縮退)。
・**方針転換(2026-08-20 ユーザー決定)**: SR のみで 4×4 未ドープの ED
　(−16.304913)と小数第2位一致(E ≤ −16.295)を狙う。フェーズ B・キャンペーンより優先。
・**発見 1(セクター)**: ED のトポロジカル 2 重項(−16.3049 / −16.2994)は両方 K=Γ で、
　**C4(副格子込み)固有値 −1 / +1 だけで区別される**(n=2 が基底)。
　**C4 n=2 射影で −16.2737 → −16.28315(+0.0094、現在の最良)**。残り 0.012。
・**発見 2(PSG 本体、ユーザー指示で実装)**: U(1) PSG 対称クラスを一式実装
　(`scripts/psg_def.jl`、共変性 solver・セクター・Bott 検証込み)。理論条件を
　全て満たすクラス(ω=π、Bott=−1、K=Γ)でも **E は −9.8 で頭打ち**。
　診断の結果、**ef4 当たり解は 2×2 フラックス結晶で射影対称性を自発破壊
　(gap 0.0016 のほぼギャップレス)し、QP 射影が対称性を復元していた**。
　4×4 では対称多様体が破れ解に大差で負ける。詳細・開問題は
　`playground_nozomi/cb_nu12_boson/PSG_NOTES.md`(唯一の正)。
・**FCI アンザッツ比較キャンペーン フェーズ 1 を走行中**(2026-08-25 投入)。
　「色々なアンザッツを試して最もエネルギーが下がるものを調べる」というユーザー依頼。
　設計 = `docs/superpowers/specs/2026-08-25-fci-ansatz-survey-design.md`、
　計画 = `docs/superpowers/plans/2026-08-25-fci-ansatz-survey.md`。
　比較軸 = 拡大セル × フレーバー群(sym / 2+1 / indep)× グラフ(model / full)。
　- ν=1/2: 4×4・N=8・U=V=0(ED −16.304913)。{xexet2, ef4} × {sym, indep} × {model, full} = 8 構成
　- ν=1/3: 6×3・N=6・NN 斥力 U=1.0(**ED を新規計算: −12.126195092720709**、
　  3 重準縮退の幅 0.0254 / 多様体外へのギャップ 0.0515)。
　  {xexet3, ef9} × {sym, 2+1, indep} × {model, full} = 12 構成
　- **第 1 波(17 構成 × 10 seed = 170 run)を走行中**。ν=1/2 の 80 run は完走、
　  ν=1/3 は 90 中 30 完了(2026-08-26 08:20 時点)。残り約 7〜9 h
　  第 2 波 = ν=1/3 の `ef9 × full` 3 構成(163 h)は**後回し**(ユーザー指示)
　- 出力: `playground_nozomi/cb_nu12_boson/runs_ansatz/`(roster = `_roster.dat`)
　- **ν=1/2 は全構成完走**。E_best = −16.275461(ef4·indep·full)、不足 0.0295。
　  3 軸(ef4>xexet2 / indep>sym / full>model)とも理屈通りだが**全部足して 0.0069**
　  しか動かない(ED 不足の 1/4)。既存の c4n2 = −16.28315 を誰も超えていない
　- **ν=1/3 の中間**: `2+1`(U(2)×U(1) Higgs)が **−12.067462** で sym(−12.022887)
　  より **0.045 低い**。ν=1/2 の indep−sym = 0.0025 とは桁違い。
　  `full` は `sym` と 0.0005 差 → **第 2 波は投入不要の傍証**
・**2026-08-26 の検証(詳細は PROJECT.md の同日エントリ)**:
　- **QP 射影の実装は ν=1/2・ν=1/3 とも正しい**(Claude が 2 度誤指摘して撤回)。
　  ν=1/3 の精度不足は射影のバグではない
　- ED の ν=1/3 3 重項は**3 つとも K = (−0.5, 0)**、C2 固有値は基底 **+1** / 他 2 つ −1
　- **C2 量子数射影を実装**(`--c2 N`、射影先は n = 0)。実効ギャップが
　  0.0515 → 0.0769 に広がる。第 1 波完走後に検証 run を回す
　- 有限サイズ効果は**アス比ではなく短辺の長さ**が支配。Ny=3 では 3×3/4×3/6×3 とも
　  幅/gap ≈ 0.5 で飽和し、**ED が届く範囲では改善不能**(短辺 4 は 6×4 で数日規模)
　- `zvo_out.dat` の 4 列目 = **規格化分散 σ²/E²**。ED 不要の品質指標として使える
・フェーズ B(8×8 ドープ)・キャンペーン再投入・PSG の続きはこの後。
・詳細・判断ログはすべて `playground_nozomi/cb_nu12_boson/PROJECT.md`(唯一の正)。

## ブランチ運用(2026-08-14 確定)

**単一ディレクトリ + ブランチ切り替えで運用する。worktree は常設しない。**

| ブランチ | 役割 |
|---|---|
| `parton-mode` | **日常の作業ブランチ**(GitHub の default branch)。実装・文書・テストはすべてここ |
| `main` | **upstream(tmisawa/Julia-mVMC)の純粋な鏡**。マージしない・コミットしない |
| `wip/main-leftovers` | 2026-08-14 以前に main の作業ツリーに残っていた初期版の保全先(参照専用) |

`main` を汚さない理由は 2 つ:

1. **upstream 追従を常に fast-forward で済ませる**。upstream は v0.5.0 以降は静かだが
   v0.3 → 0.4 → 0.5 と刻む開発者なので v0.6 はありうる。マージしてあると登録点
   9〜11 箇所で衝突解決が発生する
2. **upstream への PR / issue の出口を確保する**。`main` が純粋なら
   `git switch -c fix/xxx main` で即座にクリーンな作業ブランチが作れる。
   報告候補は DESIGN §11 に 4 件(per-block writer のヘッダ 4 行 / 直接法 SRinfo の
   欠落 / ParaQPTrans の虚部未読み込み / S の定義差)

### upstream へ PR を出すときだけ worktree を生やす

```bash
git worktree add ../mvmc-upstream main     # 一時的に main を別ディレクトリへ
cd ../mvmc-upstream && git switch -c fix/xxx
# …作業・PR…
cd - && git worktree remove ../mvmc-upstream    # 使い終わったら畳む
```

- worktree は `.gitignore` の `/.claude/worktrees/` で履歴から締め出してある
- **サブモジュールを含む worktree は `git worktree remove` が効かない**。
  ディレクトリを消してから `git worktree prune` すること

### upstream の追従

```bash
git remote add upstream https://github.com/tmisawa/Julia-mVMC.git   # 未設定なら
git fetch upstream && git switch main && git merge --ff-only upstream/main
git push origin main
```

## 作業場所

| 対象 | パス | 備考 |
|---|---|---|
| リポジトリ | `/home/nozomihigashino/Julia-mVMC` | **単一**。ブランチを切り替えて使う |
| 設計文書 | `DESIGN_parton.md` | **唯一の正**。仕様の判断は必ずここを先に読む |
| 調査レポート | `REPORT_nu12_stuck.md` | ν=1/2 局所解問題の調査記録(§15 が機構の確定、§16/§17 が対処) |
| 実行手順 | `docs/parton_run.md` | |
| テスト | `MVMCOptimizers.jl/test/` `MVMCExpertModeParsers.jl/test/` `test/` | |
| 手元の計算 | `playground_nozomi/` | **git 管理外**(`.gitignore`)。入力・出力・診断コードが混在 |
| アンザッツ比較の出力 | `playground_nozomi/cb_nu12_boson/runs_ansatz/` | 第 1 波 170 run。`_roster.dat` が一次データ |

注意点:

- 設計・仕様の判断は `DESIGN_parton.md` を先に読む。modpara キー追加時は
  DESIGN §3.1 の「3 登録点 + マーカー」規律に従う
- `playground_nozomi/` は ignore してあるので、**診断コードもリポジトリには残らない**。
  REPORT §15-11 / §16 の再現手順が参照するので、消さないこと
  (`test/` か `tools/` へ移してコミットするかは未決)
- 実行には julia 1.10 を使う(`~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia`)。
  `/opt/julia-1.8.5` では `[sources]` が未対応でパッケージが解決できない

### 主要な出力ファイル(playground_nozomi/<run>/…_out/)

| ファイル | 内容 |
|---|---|
| `zvo_out.dat` | ステップ毎の E, E², variance |
| `zvo_var.dat` | ステップ毎の全変分パラメータ履歴(末尾ブロックが パートン α) |
| `zvo_SRinfo.dat` | `Npara Msize optCut diagCut sDiagMax sDiagMin absRmax imax` |
| `zvo_parton_diag.dat` | `step min_gap … n_occ_deviation principal_angle_max` |
| `zqp_pmfpara_{init,opt}.dat` | α の初期値 / 最適化後(`InPmfPara.def` として読み戻せる) |
| `zqp_pmfocc_{init,opt}.dat` | 占有集合(`InPmfOcc.def` として読み戻せる) |
| `zvo_parton_runinfo.dat` | run のメタデータ + 終端の自己完結性検査(`occ_selfcontained` ほか) |

## ohtaka(計算サーバ)側

**リポジトリの切り替えは ohtaka でも必要**(忘れると古いコードでジョブが走る)。
手順は下の「ohtaka 側の切り替え」を参照。ジョブスクリプトは**リポジトリに置いて
履歴に残す**方針(どのスクリプトでどの run を出したかを辿れるようにするため)。

- スレッド数は `JULIA_NUM_THREADS` で指定する。**`OMP_NUM_THREADS` では効かない**
  (DESIGN §3.3.2。C 版のジョブスクリプトを流用した事故を避けるため門番が警告する)
- **ローカル機(S1707-095)は物理 16 コア / 論理 32(Ryzen 9 7950X、SMT2)**。
  `nproc = 32` は論理コア数。2026-08-18 の実測で **`-n 16` + `JULIA_NUM_THREADS=1`
  + `BLAS=1` が最適**、SMT 併用も Julia スレッド増設も -0.4〜-45% と損。
  **同時に回す run は 1 本**(2 run 並走は直列より遅い)。表は PROJECT.md 参照

### ohtaka 側の切り替え

```bash
cd <ohtaka の clone>
git fetch origin
git status --porcelain            # ジョブスクリプト等の未コミット変更を先に確認
git switch parton-mode            # 無ければ git checkout -B parton-mode origin/parton-mode
git submodule update --init --recursive
```

## TODO リスト

### セクター解析の続き(2026-08-26、詳細は `REPORT_sector.md` §10)

- [ ] **段階 (d)**: sym → 2+1(84 idx)へ拡張して Step 5 を再実行し、
      多様体外 11.2% の漏れが減るか(= 表現能力の天井が上がるか)を見る
- [ ] **段階 (e)**: C2 拘束 |c₁| = |c₂| を課した版との比較(今回は拘束なしのみ)
- [ ] **SR 版との比較**(未実装)。厳密 S 行列 + ゲージ方向の射影除去 + 対角シフト 1e-4
- [ ] **Step C の仕切り直し**: 拡大胞 (1,3) で FCI 相に乗る解を作る
      (今回は 5 seed とも Ω ≈ 0 で判定不能)。x/y 非対称性の由来はここでしか決着しない
- [ ] **MC 版 VMC への移植**(境界ひねりを `pmftrans.def` へ焼き込む / セクター和を
      QP 群へ拡張 / SR のゲージ方向除去。占有はアウフバウで足りる)
- [ ] **キャンペーンの残り 35 run の再開判断**


- [x] **SR 漂流の機構確定**(2026-08-14、REPORT §15)。主犯は RedCut ではなく
  **アウフバウ占有規則による枝の不連続な乗り換え**
- [x] **占有追跡(MOM)**(DESIGN v3.12)。`PartonOccMode` = 0 aufbau(既定、ビット一致)/ 1 mom
- [x] **`InPmfOcc`(占有の読み戻し)**(DESIGN v3.13)。`(α*, O*)` の組で状態が閉じる
- [x] **fixture の向き正準化(DESIGN v3.14、2026-08-18)**。修正済み・P 層 442 全緑。
  比較 run(`runs_orientfix/L08_ef4_s1009`)は**合格**: E −65.0794、gap +0.64、
  C = −1 を 3 経路一致で確認。旧 12 run は `runs/` に保存(壊れたアンザッツの参照値)
- [x] **ホールドープ探索 4×4(フェーズ A 60 run + フェーズ C 30 run)完了**(2026-08-20)。
  集計と判定は PROJECT.md の 2026-08-19 / 2026-08-20 のメモ
- [ ] **ED 小数第2位一致の検証(中断中、ユーザー検討待ち)**: 実測済みは
  c4n2 = **−16.28315**(+0.0094、当たり 4/10)/ pso = 利得なし(コールドで
  自由度を足すと盆地悪化)/ PSG 対称クラス = −9.8 頭打ち(対称性破れの発見)。
  psf コールドとウォーム複合(`submit_psg_warm.sh`)は未走。
  詳細は PROJECT.md 2026-08-20 の一連のメモ + PSG_NOTES.md
- [ ] **道具は整備済み**: chain.jl に `--c4 n`(C4 量子数射影)/ `--psg`(クラス内
  拡張)/ `--psgclass`(PSG 対称アンザッツ)/ `--redcut` / `--warm`(α*/占有/
  Jastrow のウォームスタート)。ED セクター計算 = `scripts/ed_c4_eigenvalue.jl`
- [ ] test/physics の P 層回帰(fixture に psg kwargs 追加、既定 off)を
  単独実行で確認する(未実施。マシンは現在空いている)
- [ ] **フェーズ B(8×8 ドープ)の投入判断(保留)**: `submit_doped.jl --phase B` が待機中
  (δ{1,2,4}×seed{1001-1003} = 9 run ≈ 32 h、`PHASE_B_QP = true` のままでよい)。
  4×4 の当たり率 5〜6/10 から seed 3 本では当たり 1〜2 本 → `SEEDS_B` 増量を検討
- [ ] **キャンペーン再投入(保留中)**: 共変 ef4 / 段1 3000 step / L08 10 seed・
  L10 以降 5 seed / 16 ランク直列。条件は揃っている。フェーズ B と実行順を決める
- [ ] **ドープ計算の次の整備候補**(PROJECT.md 2026-08-18 の議論):
  `PartonOccMode = 2`(explicit、エニオン分散測定の中核)/ twist 境界条件
  (バンド端縮退割り)/ C4 射影 / Lanczos。sDiagMax ~ 1/gap² に注意
- [x] **tools/ の未コミット分をコミット**(2026-08-25)。parton_bands.jl /
  parton_band_chern.jl / parton_chern_consistency.jl / parton_chern_validate.jl /
  plot_parton_conv.jl、fixture の向き正準化テスト、DESIGN v3.14

### FCI アンザッツ比較キャンペーン(2026-08-25 開始)

- [x] ν=1/3 の ED を新規計算(6×3・N=6・U_NN=1.0)。E0 = −12.126195092720709、
  3 重準縮退を確認。`test/physics/test_p0_ed_reference.jl` の P0-d に登録
- [x] fixture に `flavor_groups`(sym / 2+1 / indep)と `graph = :full`(全サイト対)を追加
- [x] gen_def / chain.jl の充填一般化(ν=1/3・6×3・(3,1)/(3,3) アンザッツ)
- [x] 投入ドライバ `submit_ansatz.jl`(2 波)・集計 `analyze_ansatz.jl`・
  conv 図の一括生成 `plot_runs.jl`
- [ ] **第 1 波の完走待ち**(ν=1/3 の残り 60 run、2026-08-26 07:51 に再投入済み。
  ログ = `runs_ansatz/_logs/submit_resume.log`)。完走したら
  `analyze_ansatz.jl --chern` で集計 → PROJECT.md に結論を記録
- [ ] **完走後すぐ: C2 射影の検証 run**。`scripts/submit_c2.jl` が待機中
  (sym / 2+1 × `--c2 0` × 10 seed = 20 run、約 3.3 h)。C2 は平均場が
  flux 差 1.50 rad で破っており、ν=1/2 の C4(0.362 rad)より破れが 4 倍大きい
- [ ] **sym と 2+1 の差 0.045 の切り分け**(物理か最適化か)。2+1 の解を
  3 フレーバー平均して sym の初期値にし `--warm` で再最適化する。
  sym は最も seed 不安定な設定なので 10 seed で底を引けていない可能性がある
- [ ] **第 2 波の投入判断**: ν=1/3 の `ef9 × full` 3 構成(163 h)。
  **第 1 波で `full` が `model` に勝たなければ投入不要**
- [ ] フェーズ 2(大規模系)の設計。バンド図と k 分散はここで復活する

#### このキャンペーンで判明した既存コードの問題(重要)

- **def の中身を読まずに関数のドキュメントから結論を出さない**(2026-08-26 の教訓)。
  `cb_qp_translations` の docstring は「x 方向 kext 本」と書いてあるが、実際に
  `gen_def.jl` が使うのは `ef_qp_unit_cells(nx,ny,ex,ey)` で、ef4 では
  `{(0,0),(1,0),(0,1),(1,1)}` = 商群 Z₂×Z₂ の正しい代表。**`qptransidx.def` の
  置換を同定してから判断すること**
- **QP の重みは「剰余類代表への χ_K の制限」であって商群の指標ではない**。
  |MF⟩ が H_MF 保存部分群 S の固有状態なら P_K|MF⟩ ∝ (Σ_{s∈S} χ_K(s)*λ_s) ×
  Σ_{代表} χ_K(r)* T_r|MF⟩ と簡約される。χ_K が S 上で自明でなくても射影は機能する
  (ν=1/3 の重み (1,−1,+1) は正しい)
- **seed 依存性はアンザッツ構造に起因する**。`α → 対角化 → 占有選択 → Slater 行列式`
  の**離散的な占有選択**がパラメータ空間を分枝に分けるので、通常の mVMC(ペア軌道 =
  Pfaffian の滑らかな関数)並みの seed 安定性は原理的に望めない。**フレーバー縮退が
  大きいほど不安定**(5 組すべてで sym > indep)。収束自体はプラトーに達しており、
  step 数不足でも最適化失敗でもない。対処は `--warm` / `PartonOccMode = 2`(未実装)
- **分散外挿に異なるアンザッツの点を混ぜてはいけない**。族が違えば別の曲線に乗る。
  ν=1/2 の 8 構成で R² = 0.223、ν=1/3 は行き過ぎと、符号すら揃わない。
  正しくやるには同一アンザッツで σ² を下げた点列(= Lanczos)が要る
- **`PartonFlavorSymFast` は使わない**。全構成で `0` を書く(`gen_def.jl` の `write_modpara`)。
  対称の検出は idx 写像の構造判定(起動時 1 回)で、`:sym` は必ず発火する。ところが
  `PartonOccMode = 1`(MOM)は各フレーバーが独立に占有を追跡するので、フェルミ面近傍が
  準縮退すると占有が分岐して高速路の前提が壊れ、コアの門番が落とす。**MOM が標準である
  以上この高速路に使い道はない**(効かないか壊すかのどちらか)。既存 run は全部 `:indep`
  だったので一度も発火しておらず、`:sym` を入れて初めて表面化した
- **収束判定の gap 条件は絶対値で見る**(`stage_io.jl`)。`min_gap` は
  「非占有の最低準位 − 占有の最高準位」の**符号つき**量で、**MOM の非アウフバウ占有では
  負になるのが正常**(コア `parton_orbital.jl:452-467` のコメントが明言)。
  旧基準 `gp_tail > 0` は MOM 導入前のもので、健全な run を誤って弾いていた
  (`graph = :full` の 7 本が全滅して発覚。うち 2 本は model の最良より低い E だった)
- **判定は roster ではなく run ディレクトリから再計算する**(`analyze_ansatz.jl`)。
  roster の `converged` 列は run が走った時点のスナップショットで、判定基準を変えても
  遡らない。判定基準は今日だけで 2 回変わっている
- **`--psg full` にも同じ罠がある(未修正)**。`psg_shells_all(4,4)` の d²=16 は
  (±4,0) = 対蹠変位で、拡大セル並進が端点を入れ替えるため複素 α で並進が破れる。
  `psf` は 1 本も完走していないので実害の観測は無いが、**再開する前に
  `cb_translation_swapped_pairs` と同じ除外を入れること**
- **ED の jld2 を読むスクリプトは `/opt/julia-1.8.5` で走らせる**。JLD2 が
  julia 1.10 のどの環境にも入っていない(`ed_translation_sector.jl` が該当)
- **`occ_selfcontained = 0` の run は α だけでは再現できない**。占有集合 `O` も要る
  (`InPmfOcc`、DESIGN v3.13)。段間引き継ぎや後解析で落とさないこと
- **`--k` の符号規約は自己共役でない k 点では未検証**。今回の K = (−0.5, 0) は
  両成分とも自己共役点なので符号反転の曖昧さが無いが、一般の k に使い回すときは要検証
- [ ] **盆地選択の手当て**(REPORT §16-4)。seed 11272 はコールドスタートで別盆地に入り、
  MOM はその中で安定化するだけ。初期値・アニーリング・鎖方式(`InPmfOcc` が土台)の領域
- [ ] **RedCut の置換は保留**(測定可能な利得がゼロ)。設計は DESIGN §10 に記録済みで、
  発火を観測したら実装する
- [ ] **`explicit`(全 step 占有固定 = `PartonOccMode = 2`)**: 必要になってから。値 2 は予約
- [ ] S 行列の定義差(上流規約が `Im⟨O_a⟩Im⟨O_b⟩` を落としている。REPORT §15-7)。
  既存 run の再現性に関わるので保留。upstream 報告候補
- [ ] `NSplitSize > 1` の解禁検証(M3)
- [ ] 診断コード(`playground_nozomi/diag_sr/`)を `test/` か `tools/` へ移すか判断する
