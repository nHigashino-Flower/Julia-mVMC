# FCI アンザッツ比較キャンペーン(方針 1・フェーズ 1 = ED 系)設計

作成 2026-08-25。brainstorming で確定した内容。判断ログは
`playground_nozomi/cb_nu12_boson/PROJECT.md`(唯一の正)に日付付きで転記する。

## 0. 背景と目的

パートン平均場 VMC(`PartonMode = 1`)で CheckerBoard 模型の FCI 相を、
複数のアンザッツで計算し、**最もエネルギーが下がるアンザッツを特定し、
その平均場がチャーン数の意味で FCI になっているか**を確かめる。

2 つの方針のうち本文書は**方針 1**(相互作用固定・アンザッツ走査)のフェーズ 1
= **ED 参照値のある小系**に限る。フェーズ 2(大規模系)と方針 2
(ν=1/2、U=1.0、V=0〜8 の FCI–CDW(π,π) 転移、ef4 固定)は本文書の範囲外。

### 対象系(ユーザー決定)

| | ν=1/2 | ν=1/3 |
|---|---|---|
| 統計 | ハードコア・ボゾン(F=2) | スピンレス・フェルミオン(F=3) |
| 相互作用 | U=V=0 | **U_NN = 1.0、V = 0**(ED の "U" は最近接斥力) |
| 系 | 4×4 セル、32 サイト、N=8 | **6×3 セル、36 サイト、N=6**(新規 ED) |
| ED 参照 | −16.304913354429445(2 重準縮退、K=Γ、C4 n=2) | 新規計算(3 重準縮退の見込み) |

ν=1/3 で 6×3 を採る理由: 既存 ED(5×3)には拡大セル (3,1)/(3,3) が敷き詰め
られない。6×3 なら全部載り、次元 C(36,6) ≈ 1.9×10⁶ で ED は ~1.5 h。

## 1. 比較行列

| 軸 | ν=1/2 | ν=1/3 |
|---|---|---|
| 拡大セル (ex,ey) | `xexet2` (2,1) / `ef4` (2,2) | `xexet3` (3,1) / `ef9` (3,3) |
| フレーバー群 | `sym`(全共有)/ `indep`(全独立) | `sym` / `2+1`(2 共有 + 1 独立)/ `indep` |
| グラフ | `model`: t_ij(d²=2,4,8)/ `full`: 全距離 | `model`: t_ij + V の Wick 分解 / `full` |
| 射影 | 並進 QP、K=Γ | 並進 QP、**K は新規 ED の基底から決める** |
| 構成数 | 2×2×2 = 8 | 2×3×2 = 12 |

- seed 10 本(1001–1010)、**段1 のみ** 3000 step(段2/3 は E を動かさないと
  4×4 で実測済み、PROJECT.md 2026-08-20)。mpi 16 直列・同時 1 run。
- 合計 200 run。4×4 ef4 c4n2 の実測が 615 s/run なので、`model` は ~10 分、
  `full` はパラメータ数が数倍なので +α。**総計 40〜50 h** の見積り。

### 1.1 各軸の定義と根拠

**拡大セル**: idx クラスの鍵は `(拡大セル内位置 + 副格子, dx, dy)`
(`test/physics/parton_fixture.jl:45-51, 175-186`)。同じクラスのボンドは拡大セル
並進で移り合う。QP の並進は拡大セルの剰余類(`ef_qp_unit_cells`)。

**フレーバー群**: 現行 `idx_mode` の `:orbit`(全共有)と `:orbit_flavor`(全独立)
を、**フレーバー → 群番号の写像 `flavor_groups`** に一般化する
(`sym` = `[0,0,(0)]`、`indep` = `[0,1,(2)]`、`2+1` = `[0,0,1]`)。
ν=1/3 では `:orbit` が `:orbit_flavor` に勝つ既知の逆転(REPORT §5-2)があるので、
`2+1` はその中間点として意味がある。

**グラフ**:
- `model`: pmftrans の係数は模型のホッピング t_ij(d²=2,4,8)。
  ν=1/3 は既存の `u_mf = 1.0` 経路(NN 係数 t+V、対角に V。fixture `:196-221`)
  = 「t_ij + V の Wick 分解」。**Fock 項はクラス内定数なので α に吸収され、
  実効的な差分はオンサイト α(Hartree)の有無**であることを注記しておく。
- `full`: **全サイト対(オンサイト含む、トーラス上の全変位)に係数 1 を置き、
  全 idx を乱数初期化**する。通常 mVMC のペア軌道 f_ij(全距離)に近い設定。
  クラス分けは拡大セル並進 × フレーバー群で `model` と同じ。
  既存 `--psg full` との違いは (a) 元ボンドも係数 1、(b) 新 idx も乱数初期化
  (`psg full` は α=0 の 7 列)、(c) オンサイトを常に含む、の 3 点。
  変位の列挙は既存 `cb_psg_extra_bonds`(`parton_fixture.jl:102-127`)を流用。
- 係数だけ変えた変種(全 1 だが距離は模型のまま)は、クラス内で t_ij が定数
  なので α のスケール変換で `model` と同じ多様体になる → 軸にしない。

**射影**: 並進 QP のみ(C4 射影・PSG クラス・`--bf` は本キャンペーンでは使わない)。
ν=1/3 の ED 基底は 3 重準縮退で、3 状態の運動量が異なり得るので、
基底(最低)の K を ED 固有ベクトルの並進固有値から決めて `--k` で渡す。

## 2. 判定基準

1. **エネルギー**: 構成ごとに、収束判定(`stage_io.jl` の 4 条件)を通った run の
   末尾 100 step 平均 E の最小値 E_best と、ED からの不足 ΔE = E_best − E_ED。
   当たり率 = (E_best + 0.01 以内の run 数) / 10。
2. **FCI 判定**: 各構成の E_best run について `tools/parton_band_chern.jl` で
   フレーバーごとの占有バンド Chern 数 C_f と min_gap を出す。
   FCI = 全フレーバーで |C_f| = 1、符号が一致、平均場ギャップが開いている
   (ν=1/2: 2 パートン × C=1、ν=1/3: 3 パートン × C=1)。
3. 「最良アンザッツ」= ν ごとに ΔE 最小で、かつ 2 を満たすもの。
   ΔE 最小が 2 を満たさない場合はその事実を記録し、2 を満たす中での最小も併記。

## 3. 実装仕様

コア(`MVMCOptimizers.jl` 等の `MVMC*`)は**無改修**。F=3 はコアで一般対応済み
(`DESIGN_parton.md:17-19`、F=3 の ED 収束テストあり)。

### 3.1 `test/physics/parton_fixture.jl`

- `parton_fixture(...; flavor_groups::Union{Nothing,Vector{Int}} = nothing, graph::Symbol = :model)` を追加。
  - `flavor_groups` が与えられたら `_flavor_key` は `(key, groups[f+1])` を返す。
    `nothing` なら従来の `idx_mode` 規約(`:orbit` / `:orbit_flavor` / `:bond_flavor`)
    をそのまま使う(既存テストのビット一致を保つ)。長さ ≠ F はエラー。
  - `graph = :full`: 模型ボンド・`u_mf`・`psg_*` の経路を使わず、
    オンサイト + 全変位(`cb_psg_extra_bonds` を d²∈{2,4,8} も含めて列挙する
    ように拡張、または同型の列挙関数)を係数 1 で出し、`psg_idx` は空
    (= 全 idx が 5 列乱数初期化)。`graph = :full` と `u_mf != 0` /
    `psg_onsite` / `psg_shells` の併用はエラー。
  - 向きの正準化(v3.14)は既存ボンドループと同一規約を守る。
- `write_parton_def_files` に同じ kwargs を通す。
- **回帰テスト(P 層に追加)**:
  - `flavor_groups = [0,0,1]`(F=3)で n_idx = 2 × n_idx(`:orbit`)。
    `[0,1,2]` で `:orbit_flavor` とビット一致、`[0,0,0]` で `:orbit` とビット一致。
  - `graph = :full` で H_MF がエルミート、任意の複素 α で拡大セル並進に共変
    (既存の共変性テストと同じ検査)、n_idx = (変位クラス数 + オンサイト) × 群数。
  - 6×3(nx ≠ ny)で (3,1)/(3,3) の fixture が生成でき、クラス数が理論値に一致。

### 3.2 `playground_nozomi/cb_nu12_boson/scripts/gen_def.jl`

- `FILLINGS = Dict(:nu12 => (q=2, F=2, u_mf=0.0, u_phys=0.0, stat=:boson), :nu13 => (q=3, F=3, u_mf=1.0, u_phys=1.0, stat=:fermion))`。
- `SYSTEMS` を `(nx, ny, nu)` で引けるようにし、`(4,4,:nu12)`・`(6,3,:nu13)` を登録
  (Ne = nx·ny / q を計算)。既存の ν=1/2 正方系はそのまま。
- `ANSATZ` に `:xexet3 => (3,1)`、`:ef9 => (3,3)` を追加。
- 充填の門番を `q·Ne == nx·ny`(`allow_doped` はホール側のみ従来通り)に一般化。
- `gen_stage1_def(...; nu, flavor::Symbol, graph::Symbol, kx, ky)`:
  `nflavor`/`u_mf`/`u_phys` を `FILLINGS[nu]` から、`flavor_groups` を
  `flavor ∈ (:sym, :two_one, :indep)` から作って fixture に渡す。
- `run_id` に接尾辞を追加: 系は `L{nx}`(nx=ny)/ `L{nx}x{ny}`(nx≠ny)、
  ν=1/3 は `_nu13`、フレーバー群 `_fsym` / `_f21` / `_find`、グラフ `_full`。
  **後方互換**: ν=1/2・`indep`・`model` のときだけ従来通り無タグ
  (既存 run_id を変えない)。例: `L04_ef4_fsym_full_s1001`、
  `L6x3_ef9_nu13_f21_s1001`。

### 3.3 `scripts/chain.jl`

- CLI: `--nu 1/2|1/3`(既定 1/2)、`--flavor sym|2+1|indep`(既定 indep)、
  `--graph model|full`(既定 model)、`--k KX KY`(既定 0 0)。
  K の単位は `write_qptransidx` の規約(`gen_def.jl:229`、重み
  `exp(2πi(kx·ducx + ky·ducy))`、duc は単位胞単位)に従い、**単位胞の逆格子
  ベクトルを 1 とする分数**(例: 6×3 なら kx ∈ {0, 1/6, …, 5/6}、ky ∈ {0, 1/3, 2/3})。
  ED 側の `ed_translation_sector.jl` も同じ単位で出力する。
- `SYSTEMS` の引き方を `(nx, nu)` に変更。`prepare_stage` の段2/3 再生成で
  **`nflavor = 2` のハードコード(`chain.jl:295`)を除去**し、`FILLINGS` から取る。
- `run_stage` / `prepare_stage` に上記 kwargs を通す。

### 3.4 ED(`~/ED`、リポジトリ外)

- `ED/Code/run/Checkerboard/LocalState_nu13_6x3.jl` + `.sh`(PBS、`run.sh` と同じ投入先)。
  `LocalState_benchmark.jl` を雛形に、`system_list = [(6, 3, 6)]`、Fermion、q=3、
  既定ホッピング(t=1, t1=0.293, t2=−0.293, t3=0.207, ψ=π/4)、U=1.0、V=0.0、
  n=0、ξ=η=0、**r = 0.0**(乱雑ポテンシャル無し。4×4 ボゾンと同じ)、neig=10。
  出力は `calc_local_energy_and_state_v1_2` が既存階層
  `Fermion/t=…-ψ=0.785/Nx=6-Ny=3-N=6-q=3-r=0.0/n=0/U=1.0-V=0.0/Psite-Vp=0-0.0/`
  に `<header>_result_eigen_periodic.{txt,jld2}` を書く。header は
  `withoutRandomPotential`(r=0.0 なので)。リポジトリ側パーサ
  (`ed_reference.jl:103-110`)はディレクトリ内に `*result_eigen_periodic.txt` が
  **丁度 1 つ**あることを要求するので、同じディレクトリに別 header で二重に
  出さない。jld2 は ~300 MB の見込み。
- `scripts/ed_translation_sector.jl`(`ed_c4_eigenvalue.jl` と同型): jld2 の
  固有ベクトルに単位胞並進 T(1,0)・T(0,1) を作用させ、最低 4 準位の固有値
  (= e^{iK·a})を表にする。**フェルミオンなのでビットマスクの置換符号を入れる**
  (ボゾン版は符号なし)。完全縮退があれば多様体内で T を対角化する。
  出力 → 基底の (kx, ky) を PROJECT.md に記録し、`--k` に渡す。
- `test/physics/ed_reference.jl` に `ED_CASE_FERMION_NU13_6x3` を登録し、
  `test_p0_ed_reference.jl` に P0-d(6×3、3 重準縮退、U=1.0/V=0)を追加。

### 3.5 `scripts/submit_ansatz.jl`(`submit_doped.jl` を雛形に)

- CLI: `--nu 1/2|1/3|all`、`--k13 KX KY`(ν=1/3 の K。未指定で ν=1/3 を含む
  場合はエラー)、`--np`、`--dry-run`、`--seeds a:b`。
- run リスト = §1 の行列 × seed。順序: ν=1/2 の 8 構成 → ν=1/3 の 12 構成、
  各構成内は seed 昇順(構成ごとに早く判定を出せる順)。
- 完走判定(`chain.log` に stage1)でスキップ、`runs_ansatz/_roster.dat` に
  `run_id nu ansatz flavor graph seed started wall_sec status E_tail min_gap sDiagMax converged`
  を追記。`expect_gap = true`。失敗は記録して**続行**(1 構成の失敗で全体を止めない)。
- 出力先 `playground_nozomi/cb_nu12_boson/runs_ansatz/`、ログ `_logs/`。

### 3.6 `scripts/analyze_ansatz.jl`

- roster を読み、構成別に E_best / 収束 run の平均 / 当たり率 / ΔE を表にする。
- 各構成の E_best run に `tools/parton_band_chern.jl` を当て、C_f・min_gap を
  同じ表に足す(6×3・F=3 で動くことは事前に 1 run で確認する。
  native grid は `(nux/ex, nuy/ey)` で nx≠ny に対応済み)。
- `runs_ansatz/_summary.md` に書き出し、PROJECT.md に結論を転記。

## 4. 実行順

1. **ED 6×3 を投入**(依存なし、~1.5 h)。並行して実装 3.1〜3.3。
2. fixture の回帰テスト + `CBNU12_QUICK=1` のスモーク(ν=1/2 4 構成、ν=1/3 は
   K=Γ 仮置きで 2 構成)で配管を確認。
3. `submit_ansatz.jl --nu 1/2` を投入(8 構成 × 10 seed ≈ 15 h)。
4. ED 完了 → `ed_translation_sector.jl` で K 確定 → P0-d 登録 →
   `submit_ansatz.jl --nu 1/3 --k13 …` を投入(≈ 25〜35 h)。
5. `analyze_ansatz.jl` → 結論を PROJECT.md へ → フェーズ 2(大規模系)の設計。

## 5. リスクと注意

- **ν=1/3 の `indep` は悪化する既知の傾向**(REPORT §5-2、5000 step でも ED +9%)。
  結果がそうなっても計画通り記録する(比較行列の一部)。
- **`full` のパラメータ数**: 6×3 `ef9` `indep` で概算 ~10³ idx。SR(S 行列)は
  Npara² なので 1 run が `model` の数倍になり得る。スモークで sec/step を測り、
  見積りを更新する。
- **コールドで自由度を足すと盆地選択が悪化する**(pso の教訓)。`full` の当たり率
  が低いときはウォームスタート(`--warm`)での再試行を追加課題にする。
- ν=1/3 の K が Γ でなければ、既存の 5×3 run(K=Γ 前提)とは条件が違うことを明記。
- 判定は `stage_io.jl` の 4 条件を使う。sDiagMax の暴走(ハズレ盆地)は roster で
  見える。
- ED の jld2(~300 MB)は `ED/Data/` に置く(リポジトリには入れない)。

## 6. 範囲外(後続)

- フェーズ 2(大規模系: 8×8 以上 / 6×6, 9×6 など)
- 方針 2(ν=1/2、U=1.0、V スキャン、ef4 固定)
- C4 射影・PSG 対称クラス・ペアリング(Z2)・フレーバー混成
- 段2/段3、Lanczos
