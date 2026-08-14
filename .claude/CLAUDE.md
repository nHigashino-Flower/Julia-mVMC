# Julia-mVMC を fork して運用又は追加機能実装

## CLAUDE.mdの運用ルール
・定期的にCLAUDE.mdの体裁を整える
・恒常的に認知が必用なことは逐次 CLAUDE.mdに書き足す。
・セッションをリセットした時に、続きから再開しやすいように、TODOリスト等を活用する。

## 現在の状況

・パートン平均場モードの**基本実装が一通り完了**(DESIGN v3.13 / `InPmfOcc` まで)。
　§8 のテスト 0〜18 と P 層 P0〜P2 が全緑、既存回帰も全緑。

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

### ohtaka 側の切り替え

```bash
cd <ohtaka の clone>
git fetch origin
git status --porcelain            # ジョブスクリプト等の未コミット変更を先に確認
git switch parton-mode            # 無ければ git checkout -B parton-mode origin/parton-mode
git submodule update --init --recursive
```

## TODO リスト

- [x] **SR 漂流の機構確定**(2026-08-14、REPORT §15)。主犯は RedCut ではなく
  **アウフバウ占有規則による枝の不連続な乗り換え**
- [x] **占有追跡(MOM)**(DESIGN v3.12)。`PartonOccMode` = 0 aufbau(既定、ビット一致)/ 1 mom
- [x] **`InPmfOcc`(占有の読み戻し)**(DESIGN v3.13)。`(α*, O*)` の組で状態が閉じる
- [ ] **盆地選択の手当て**(REPORT §16-4)。seed 11272 はコールドスタートで別盆地に入り、
  MOM はその中で安定化するだけ。初期値・アニーリング・鎖方式(`InPmfOcc` が土台)の領域
- [ ] **RedCut の置換は保留**(測定可能な利得がゼロ)。設計は DESIGN §10 に記録済みで、
  発火を観測したら実装する
- [ ] **`explicit`(全 step 占有固定 = `PartonOccMode = 2`)**: 必要になってから。値 2 は予約
- [ ] S 行列の定義差(上流規約が `Im⟨O_a⟩Im⟨O_b⟩` を落としている。REPORT §15-7)。
  既存 run の再現性に関わるので保留。upstream 報告候補
- [ ] `NSplitSize > 1` の解禁検証(M3)
- [ ] 診断コード(`playground_nozomi/diag_sr/`)を `test/` か `tools/` へ移すか判断する
