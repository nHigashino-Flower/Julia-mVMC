# 制限付き Φ(PartonOrbIdx・Bloch 形)実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Φ 直接変分モードに、Pfaffian の `orbitalidx.def` と同じ規約の制限(Φ_f[s,n] = c·θ_idx)を def で掛けられるようにし、その特別な場合として拡大ユニットセルの Bloch 形を生成・変換する道具を足して、大規模系の変分パラメータを Ns·Ne → Ns·Ne / N_cell に減らす。

**Architecture:** 写像 `PartonOrbIndexMap`(要素 → θ の位置と係数・OptFlag・再スケールの組)を `PartonDirectPhi` に持たせ、SR が動かす `data.pmforb_params` を θ に読み替える。def が無いときは恒等写像で、恒等のときは既存の経路をそのまま通す(§12.11 とビット一致)。Bloch 形は def の中身の一種で、`tools/parton_bloch_orbidx.jl` が `parton_lattice.dat` から生成し、α 解からの θ は射影子 P を k ごとに対角化して作る。

**Tech Stack:** Julia 1.10、Test 標準ライブラリ、MVMCExpertModeParsers.jl / MVMCOptimizers.jl

**Spec:** `DESIGN_parton.md` §12.12(本計画の正)、§12.11(恒等写像のときの挙動)

## Global Constraints

- `PartonOrbParam = 0`(α)の全数値出力はビット一致(Task 7 で HEAD `d7cc568` と比較)
- `PartonOrbParam = 1` で `PartonOrbIdx` が無い run も、`d7cc568` と数値出力がビット一致(恒等写像は既存の経路を通す)
- 乱数初期化は初期化専用の乱数列(`SFMT19937RNG` を base_seed で初期化、`rng_real2` を使う)。サンプリングの乱数列を消費しない
- upstream ファイルの編集は登録点だけにし、`# --- parton-mode (fork addition) ---` を一字一句同じに付ける
- def の添字は 0-based。0-based → 1-based の変換は読み手(PartonOrbIdx / InPmfOrb のリーダ)の中だけで行う
- O の縮約は転置積(`dot` や `'` を使わない)
- コメントは日本語、門番・リーダの error 文は英語で、ファイルのキー名(`PartonOrbIdx` / `InPmfOrb`)を含める
- 名前は略さない。接頭辞は `parton_orbparam_` / `parton_bloch_`
- git commit はユーザーの指示があるまでしない
- `JL=~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia`。テストの単独実行は `cd MVMCOptimizers.jl && $JL --project=. -e 'using Test; include("test/parton_test_helpers.jl"); include("test/<file>.jl")'`
- 新しいテストファイルは `MVMCOptimizers.jl/test/runtests.jl` の「Φ 直接変分」の include 群の末尾に足す
- 試走はユーザー推奨の条件(dt 0.003・StaDel 0.1・RedCut 1e-8・2000 step・NVMCSample 2000・1 プロセス)

## Review Focus

1. **\|c\| ≠ 1 の係数や ±1 以外の符号**が def にある → 読み込みで error(Task 2 にテスト)
2. **同一視しすぎてフレーバーの軌道がランク落ちする**(例:全サイトを 1 つの idx に寄せる)→ 初期化の後、行列式が恒等的に 0 という分かりにくい失敗の前に、列フルランクの検査で error(Task 4 にテスト)
3. **占有ファイルの k が並進群の指標に無い・本数の合計が NElec と違う** → 生成時に error(Task 5 にテスト)
4. **α 解が拡大セルの並進で不変でない、または占有の本数が k ごとに合わない** → P(k) の固有値 ≈ 1 の本数が指定と違うので error(Task 6 にテスト)
5. **MPI の全ランクで θ の乱数初期値が同じ**で、サンプリングの乱数列を消費しない(Task 4 にテスト)

---

### Task 1: 写像の型・恒等写像・組み立て・組ごとの再スケール

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_orbparam_types.jl`(写像の型と恒等写像、`PartonDirectPhi` の末尾フィールド、`parton_orbparam_load_orbitals!`)
- Modify: `MVMCOptimizers.jl/src/parton_orbparam_inputs.jl`(`parton_orbparam_rescale!`)
- Test: Create `MVMCOptimizers.jl/test/test_parton_orbparam_map.jl`

**Interfaces:**
- Produces:
  - `struct PartonOrbIndexMap`:`n_idx::Int`、`idx::Vector{Int}`(長さ n_orb、平坦位置 k → θ の 1-based 位置)、`coeff::Vector{ComplexF64}`(長さ n_orb)、`opt_flags::Vector{Bool}`(長さ n_idx)、`scale_groups::Vector{Vector{Int}}`(組ごとの θ の 1-based 位置、昇順)、`is_identity::Bool`
  - `PartonOrbIndexMap(idx, coeff, opt_flags, n_site, n_elec, n_flavor)`:`scale_groups` は「同じ idx を使うフレーバーを結んだ連結成分」ごとに、その組のフレーバーが使う θ の位置を集めたもの(組の並びは最小フレーバー番号順)。`is_identity = idx == 1:n_orb && all(==(1), coeff)`
  - `parton_orbparam_identity_map(n_site::Int, n_elec::Int, n_flavor::Int) -> PartonOrbIndexMap`(OptFlag は全部 true。組はフレーバーごとの連続ブロック)
  - `PartonDirectPhi` に末尾フィールド `map::PartonOrbIndexMap`。`PartonDirectPhi(n_site, n_elec, n_flavor)` は恒等写像、`PartonDirectPhi(n_site, n_elec, n_flavor, map)` を追加
  - `parton_orbparam_load_orbitals!(phi, data)`:`phi.map.is_identity` なら既存のループをそのまま使い、そうでなければ Φ_k = coeff[k]·θ[idx[k]]。長さの検査は `phi.map.n_idx` に対して行う
  - `parton_orbparam_rescale!(data, map::PartonOrbIndexMap) -> Vector{Float64}`(組ごとの再スケール前の max\|θ\|)。`parton_orbparam_rescale!(data)` は恒等写像で呼ぶ(既存の呼び出しと同じ演算順)

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "写像: 恒等写像は既存と同じ組み立て・同じ再スケール" begin
    # orbparam_toy_data() の Φ を、恒等写像の PartonDirectPhi と既存の 3 引数コンストラクタの両方で load → orbitals が ==
    # 恒等写像の scale_groups == [collect(1:12), collect(13:24)](6 サイト・Ne 2・F 2)、is_identity == true
    # rescale!(data, identity) と rescale!(data) の結果(返り値と pmforb_params)が ==
end
@testset "写像: サイト・フレーバーの同一視と係数で Φ が組み立つ" begin
    # n_site 4・Ne 2・F 2。フレーバー 0 と 1 で同じ θ を使い、サイト 2,3 はサイト 0,1 の θ を符号 −1 と位相 im で共有
    # → load 後 phi.orbitals[2] == phi.orbitals[1]、Φ_1[3,n] == −Φ_1[1,n]、Φ_1[4,n] == im·Φ_1[2,n]
    # scale_groups は 1 組だけ(フレーバーが連結)
end
@testset "写像: 共有の無いフレーバーは別の組、共有があれば 1 組" begin
    # F 3 で、フレーバー 0 と 2 だけが idx を共有 → 組は 2 つ([f0, f2] と [f1])
end
@testset "写像: 組ごとの再スケールは共有を壊さない" begin
    # 上の F 3 の写像で rescale! → 各組の max|θ| ≈ PartonOrbAmpMax(rtol 1e-15)、load 後もフレーバー 0 と 2 の共有要素が ==
end
```

- [ ] **Step 2: 失敗を確認する**(`PartonOrbIndexMap not defined`)
- [ ] **Step 3: 実装する**
- [ ] **Step 4: 通過を確認する**。続けて `test_parton_orbparam_types.jl`・`test_parton_orbparam_io.jl` も単独実行で PASS(既存の呼び出しが変わっていないこと)

---

### Task 2: PartonOrbIdx の厳格なリーダとライタ

**Files:**
- Modify: `MVMCExpertModeParsers.jl/src/utils/constants.jl`(`MVMC_KEYWORDS` の `"InPmfOrb"` の後に `"PartonOrbIdx" => "pmforbidx.def"`、マーカー付き)
- Modify: `MVMCOptimizers.jl/src/parton_orbparam_inputs.jl`
- Test: `MVMCOptimizers.jl/test/test_parton_orbparam_map.jl`(追加)

**Interfaces:**
- Consumes: Task 1 の `PartonOrbIndexMap` と 6 引数コンストラクタ、`parton_orbparam_flat_index`
- Produces:
  - `parton_orbparam_read_orbidx(data, namelist_path) -> Union{Nothing,PartonOrbIndexMap}`(namelist に `PartonOrbIdx` が無ければ nothing)
  - `parton_orbparam_write_orbidx(path, map, n_site, n_elec, n_flavor) -> String`:5 行ヘッダ(`NPmfOrbIdx N` / `ComplexType 1`)、要素行は `site flavor orbital idx` に、c == 1 なら係数なし、c == −1 なら `-1`、それ以外は `% .18e % .18e`。続けて `idx OptFlag`

リーダの検査(すべて error、メッセージに `PartonOrbIdx` とパスと行番号):

| 検査 | 内容 |
|---|---|
| ヘッダ | 2 行目が `NPmfOrbIdx <N>`、N ≥ 1 |
| 要素行 | 4〜6 列。site ∈ [0,Ns)、flavor ∈ [0,F)、orbital ∈ [0,Ne)、idx ∈ [0,N)。5 列目だけなら ±1、5・6 列目なら \|Re + i·Im\| − 1 の絶対値 ≤ 1e-10 |
| 網羅 | 各 (site, flavor, orbital) がちょうど 1 回。要素行は F·Ns·Ne 行 |
| OptFlag 行 | 2 列。各 idx がちょうど 1 回、値は 0 か 1、N 行 |
| 未使用 | どの要素からも参照されない idx があれば error |

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "PartonOrbIdx: 書いて読むと同じ写像(係数 1・−1・複素)"   # write → read の idx・coeff・opt_flags・scale_groups が ==
@testset "PartonOrbIdx: namelist に無ければ nothing"
@testset "PartonOrbIdx: 壊れた入力は error($case)" for case in
    (:header, :site_range, :idx_range, :missing_element, :duplicate_element, :bad_sign,
     :non_unit_coeff, :missing_optflag, :bad_optflag, :unused_idx)
    # 例外は ErrorException で、メッセージに "PartonOrbIdx" を含む
end
```

- [ ] **Step 2: 失敗を確認する → Step 3: 実装する → Step 4: 通過を確認する**(パーサの `Pkg.test()` も PASS)

---

### Task 3: O・OptFlag・門番・状態の構築を写像に対応させる

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_orbparam_o.jl`
- Modify: `MVMCOptimizers.jl/src/parton_orbparam_inputs.jl`(`parton_orbparam_materialize_flags!`・`validate_parton_orbparam_inputs`)
- Modify: `MVMCOptimizers.jl/src/parton_vmc_para_opt_orbparam.jl`(`parton_orbparam_build_state`)
- Test: Create `MVMCOptimizers.jl/test/test_parton_orbparam_map_o.jl`

**Interfaces:**
- Consumes: Task 1 の `phi.map`
- Produces:
  - `parton_orbparam_calculate_o!`(シグネチャ不変):`phi.map.is_identity` なら既存のループ。そうでなければ各寄与を `sr_opt_o[o_slot_re(n_proj + map.idx[k])] += map.coeff[k] * ws * Ainv[n, m]` で θ のスロットへ直接足し、Im スロットは k = 1..map.n_idx について i·Re
  - `parton_fill_sr_opt_o!`(Φ メソッド):共役シムの範囲を `phi.map.n_idx` にする
  - `parton_orbparam_materialize_flags!(data, map)`:Φ ブロックの長さは `map.n_idx`、各 idx の Re・Im を `map.opt_flags[idx]` にする。1 引数版は恒等写像で呼ぶ
  - `validate_parton_orbparam_inputs(data, ctx, map)`:`length(pmforb_params) == map.n_idx`、OptFlag の長さ 2(n_proj + map.n_idx)。2 引数版は恒等写像で呼ぶ
  - `parton_orbparam_build_state(data, map)`:`PartonDirectPhi(Ns, Ne, F, map)` で組み、n_para は `count_variational_parameters`(= n_proj + map.n_idx)。1 引数版は恒等写像で呼ぶ

フィクスチャ: `orbparam_toy_data(; n_site = 6, n_elec = 2, n_flavor = 2)` に、Task 1 の「サイト・フレーバーの同一視と係数」型の写像(6 サイトに拡張:サイト 3〜5 がサイト 0〜2 を位相 `exp(im·2π/3)` で共有、フレーバー 1 はフレーバー 0 と符号 −1 で共有、全部可動)を掛け、`data.pmforb_params` を n_idx 個の複素乱数にする。

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "写像付き O = ∂ln ip/∂θ(有限差分、恒等 QP と並進 QP 2 本)"   # θ_j を Re・Im 方向に ±1e-6 動かした log(ip₊/ip₋)/(2δ) と共役前のスロットが一致(rtol 1e-6)
@testset "写像付き Euler の恒等式: Σ_j θ_j O_j = NFlavor·NElec"       # Φ は θ について線形なので成り立つ(atol 1e-10)
@testset "写像付き 力 = 勾配(store 経路、Jastrow 込み)"               # test_parton_orbparam_force_gradient.jl の列挙と S・g の組み立てを写像付きの状態で通し、∂E/∂Re θ・∂E/∂Im θ と一致(rtol 1e-4)
@testset "写像付き OptFlag の実体化と門番"                              # OptFlag 0 の idx は Re・Im とも false。長さは 2(n_proj + n_idx)。pmforb_params の長さ違いは error
```

- [ ] **Step 2: 失敗を確認する → Step 3: 実装する → Step 4: 通過を確認する**。`test_parton_orbparam_o.jl`・`test_parton_orbparam_force_gradient.jl` も PASS

---

### Task 4: 初期値(乱数・InPmfOrb の任意化)・ランク検査・ドライバ

**Files:**
- Modify: `MVMCOptimizers.jl/src/parton_orbparam_inputs.jl`
- Modify: `MVMCOptimizers.jl/src/parton_vmc_para_opt_orbparam.jl`(`parton_orbparam_run_from_namelist`・`parton_orbparam_vmc_para_opt!`・`parton_orbparam_append_runinfo`)
- Modify: `MVMCOptimizers.jl/test/test_parton_orbparam_io.jl`(`:rank_deficient` と `:missing_entry` の扱いを新しい仕様に合わせる)
- Test: `MVMCOptimizers.jl/test/test_parton_orbparam_run.jl`(追加)

**Interfaces:**
- Consumes: Task 1〜3 の全関数
- Produces:
  - `parton_orbparam_init_random!(data, map, base_seed::Integer) -> Nothing`:`data.pmforb_params` を長さ `map.n_idx` で作り直す。idx 昇順に、OptFlag true なら `ComplexF64(2·rng_real2(rng) − 1, 2·rng_real2(rng) − 1) / √2`(Re を先に引く)、false なら 0。乱数は `SFMT19937RNG()` を `base_seed` で初期化した専用の列
  - `parton_orbparam_read_in_pmforb!(data, namelist_path, map) -> Union{Nothing,String}`:`InPmfOrb` が無ければ nothing(error にしない)。件数・範囲・重複・有限性は `map.n_idx` に対して検査(列ランクの検査はリーダから外す)。2 引数版は恒等写像で呼ぶ
  - `parton_orbparam_check_rank(phi::PartonDirectPhi) -> Nothing`:フレーバーごとに σ_min/σ_max > 1e-12。違反は error で、メッセージに「`PartonOrbIdx` の共有のしすぎか `InPmfOrb` の値」を案内する
  - ドライバの順序:modpara 検査 → `read_orbidx`(無ければ恒等写像)→ `init_random!`(base_seed = `resolve_rnd_seed(ctx, mp.rnd_seed, seed) - ctx.group1`、α と同じ式)→ `read_in_pmforb!` で上書き → `rescale!(data, map)` → α 入力の拒否 → `materialize_flags!(data, map)` → 門番 → … → `build_state(data, map)` → `check_rank` → 以下は従来どおり
  - `parton_orbparam_vmc_para_opt!`:再スケールは `rescale!(data, pstate.mfham.map)`。orbdiag の `xmax_pre_*` は組ごと(恒等写像では従来と同じ列)
  - runinfo の追記に `n_idx <map.n_idx>` と `PartonOrbIdx <パス or none>` の 2 行を足す

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "乱数初期化: 同じ base_seed なら同じ θ、OptFlag 0 は 0、サンプリングの乱数列に触れない"
    # 2 回呼んで ==。OptFlag false の idx は 0。呼ぶ前後でサンプリング用 rng(別に作ったもの)の次の値が変わらないことは、関数が rng を受け取らないことで構造的に保証される旨をテスト名に残し、θ の再現性だけを検査する
@testset "InPmfOrb が無ければ乱数初期値で走る"                 # _write_min_orbparam_input の namelist から InPmfOrb を外して E2E が status 0、init の θ は各組で max = PartonOrbAmpMax
@testset "制限付き E2E: フレーバー共有の def で走り、θ の個数が減る"   # 4 サイト・F 2 でフレーバー 1 がフレーバー 0 を共有する def → n_idx = 8、runinfo に "n_idx 8"、opt の θ を InPmfOrb にして再開できる
@testset "共有しすぎてランク落ちすると check_rank で止まる"      # 全サイトを軌道ごとに 1 つの idx に寄せた def(Ne 2 → 2 列が平行)→ error、メッセージに "PartonOrbIdx"
```

`test_parton_orbparam_io.jl` の「壊れた入力」から `:rank_deficient` と `:missing_entry` を外し、`:missing_entry` は「InPmfOrb が無ければ nothing を返す」testset に置き換える。`:rank_deficient` は上の check_rank の testset が引き継ぐ。

- [ ] **Step 2: 失敗を確認する → Step 3: 実装する → Step 4: 通過を確認する**。MVMCOptimizers の `Pkg.test()` が全緑

---

### Task 5: Bloch def の生成道具

**Files:**
- Create: `tools/parton_bloch_orbidx.jl`
- Test: Create `MVMCOptimizers.jl/test/test_parton_bloch.jl`(道具のファイルを include する)

**Interfaces:**
- Consumes: Task 1 の 6 引数コンストラクタ、Task 2 の `parton_orbparam_write_orbidx`
- Produces:
  - `parton_bloch_structure(stage_in::AbstractString; occupation_file = nothing)`:`stage_in/parton_lattice.dat`(`grid Lx Ly`、`SITE i x y σ`、`MFSHIFT tx ty`)と `stage_in/modpara.def`(NElec・NFlavor・Nsite)を読み、次のフィールドを持つ NamedTuple を返す:`n_site`・`n_elec`・`n_flavor`・`grid`・`sigma::Vector{Int}`(サイトごとの σ、0-based)・`shifts::Vector{NTuple{2,Int}}`・`shift_of_site::Vector{Int}`(代表元から s へ移す shift の 1-based 位置)・`kpoints::Vector{NTuple{2,Int}}`(指標の代表 (m1, m2))・`columns::Vector{Vector{NTuple{2,Int}}}`(フレーバーごとの軌道列 = (k の位置, band)、k の位置の昇順・band の昇順)・`map::PartonOrbIndexMap`
  - `parton_bloch_write(stage_in, out_path; occupation_file = nothing) -> String`
  - CLI:`julia --project=MVMCOptimizers.jl tools/parton_bloch_orbidx.jl <stage_in> <out_def> [occupation_file]`

構成の決まりごと:

- 並進:座標 (x, y) → ((x + tx) mod Lx, (y + ty) mod Ly)。σ の代表元は「その σ を持つ最小のサイト番号」。各サイトについて、代表元から自分へ移す MFSHIFT がちょうど 1 つあること(無い・複数は error)
- k:(m1, m2) ∈ [0,Lx) × [0,Ly) を m1 外側・m2 内側の順に走査し、MFSHIFT 全体での指標 e^{2πi(m1·tx/Lx + m2·ty/Ly)} の組(位相を 1e-9 で比較)が新しいものだけを代表として残す。個数が MFSHIFT の本数と一致しなければ error
- 係数:Φ_f[s, (k, b)] = e^{2πi(m1·tx/Lx + m2·ty/Ly)}·θ_{(f, σ(s), k, b)}((tx, ty) = shift_of_site の並進)。θ の番号は (f, k, b, σ) の辞書順(f 外側、σ 最内)
- 占有:既定はすべての k に NElec / N_cell 本(割り切れなければ error)。占有ファイルは `flavor m1 m2 nbands` の行で、(m1, m2) はその指標の代表と同値なら可、どの指標とも合わなければ error、フレーバーごとの本数の合計が NElec と違えば error
- OptFlag はすべて 1
- `stage_in/physhop.def` があれば、MFSHIFT の各並進でホッピングの集合が不変か調べ、不変でなければ `@warn`(error にはしない)

- [ ] **Step 1: 失敗するテストを書く**(テストの中で小さな stage_in を書く:grid 4×2、SITE はサイト番号 i = x + 4y、σ = x mod 2、MFSHIFT は (0,0)(2,0)(0,1)(2,1)、modpara は Nsite 8・NElec 4・NFlavor 2)

```julia
@testset "Bloch: k の個数 = N_cell、θ の個数 = F·N_σ·Ne"                   # length(kpoints) == 4、map.n_idx == 2·2·4 == 16(全要素なら 64)
@testset "Bloch: 乱数 θ から組んだ Φ の射影子は拡大セルの並進で不変"     # 各 MFSHIFT の置換 T で P[T s, T s'] ≈ P[s, s'](atol 1e-12)。全要素の乱数 Φ では成り立たないことも対照で確認
@testset "Bloch: def を書いて読むと同じ写像"                                # parton_bloch_write → parton_orbparam_read_orbidx の idx・coeff が ==
@testset "Bloch: 占有ファイル(k ごとの本数指定)"                          # 1 つの k に 2 本・別の k に 2 本 → 列数 4、k の位置ごとの本数が指定どおり
@testset "Bloch: 壊れた占有は error($case)" for case in (:unknown_k, :wrong_total, :indivisible)
```

- [ ] **Step 2: 失敗を確認する → Step 3: 実装する → Step 4: 通過を確認する**

---

### Task 6: α 解 → Bloch の θ

**Files:**
- Modify: `tools/parton_bloch_orbidx.jl`(`parton_bloch_theta_from_orbitals` を追加)
- Modify: `tools/parton_alpha_to_orbparam.jl`(`bloch` と `occupation_file` のキーワード、CLI の `--bloch [occupation_file]`)
- Test: `MVMCOptimizers.jl/test/test_parton_bloch.jl`(追加)

**Interfaces:**
- Consumes: Task 5 の `parton_bloch_structure` の戻り値
- Produces:
  - `parton_bloch_theta_from_orbitals(bs, orbitals::Vector{Matrix{ComplexF64}}) -> Vector{ComplexF64}`:フレーバーごとに P = Φ(Φ†Φ)⁻¹Φ†。k ごとに基底 v_{σ,k}[s] = e^{ik·a(s)}/√N_cell(σ(s) = σ のサイトだけ)を並べた B_k で P(k) = B_k† P B_k を作り、エルミート固有値分解する。固有値 > 1 − 1e-8 の本数が `columns` のその k の本数と違えば error(メッセージに k と固有値を出す)。固有ベクトルを θ_{(f,σ,k,b)} に写す
  - `parton_alpha_to_orbparam(alpha_in, alpha_out, new_dir; bloch = false, occupation_file = nothing)`:`bloch = true` なら `alpha_in` の `parton_lattice.dat` から構造を作り、θ を上の関数で求めて `InPmfOrb.def` に書き、`pmforbidx.def` を書き、namelist に `PartonOrbIdx pmforbidx.def` を足す

- [ ] **Step 1: 失敗するテストを書く**

```julia
@testset "α → Bloch: 射影子が一致する"
    # dimerized_mf_data()(4 サイト環・F 2・Ne 2)の α 軌道。stage_in は grid 4×1、SITE i i 0 (i mod 2)、MFSHIFT (0,0)(2,0)、modpara Nsite 4・NElec 2・NFlavor 2
    # θ = parton_bloch_theta_from_orbitals → Bloch の写像で Φ を組む → P_bloch ≈ P_α(atol 1e-10)
@testset "α → Bloch: 並進で不変でない軌道は error"   # α 軌道の行を並べ替えて並進不変を壊した軌道 → error、メッセージに "eigen"
@testset "α → Bloch: 変換道具の E2E"                  # _write_min_parton_input の α run(stage_in に上の parton_lattice.dat を足す)→ bloch = true で変換 → namelist に PartonOrbIdx、Φ モードが 2 step 走る、全配置の E_loc が α の最終状態と一致(rtol 1e-10)
```

- [ ] **Step 2: 失敗を確認する → Step 3: 実装する → Step 4: 通過を確認する**

---

### Task 7: ビット一致の確認

**Files:** なし(検証のみ)

- [ ] **Step 1: 全スイート**:パーサと MVMCOptimizers の `Pkg.test()`、`$JL --project=@. test/physics/runtests.jl`(P 層 1292)。Expected: 全 PASS
- [ ] **Step 2: d7cc568 の worktree との比較**(手順は前回計画の Task 9 と同じ。サブモジュールはシンボリックリンク、`MVMCOptimizers.jl/Manifest.toml` を複写)
  - α:KM ν=1/2 4×4 の stage1_in を 20 step に縮めた入力。前回と同じ 13 ファイルがバイト一致
  - Φ(def なし):`playground_nozomi/projects/kapit_mueller/km/runs_orbparam/L4x4_nu12_randphi_s2026/in` を 20 step に縮めた入力。`zvo_out.dat`・`zvo_var.dat`・`zvo_SRinfo.dat`・`zvo_conv.dat`・`zqp_opt.dat`・`zqp_jastrow_opt.dat`・`zqp_pmforb_init.dat`・`zqp_pmforb_opt.dat`・`zvo_parton_orbdiag.dat` がバイト一致(runinfo は追記 2 行と時刻だけ違う)
- [ ] **Step 3: worktree を畳む**(ディレクトリを消してから `git worktree prune`)

---

### Task 8: 試走(KM ν=1/3 9×4 を Bloch 形で)

**Files:**
- Modify: `playground_nozomi/projects/kapit_mueller/km/REPORT_orbparam.md`(git 管理外)
- Modify: `.claude/CLAUDE.md`

- [ ] **Step 1**:`L9x4_nu13_indep_k0_hm8p0_s1001/stage1` を `--bloch` で変換し、n_idx を確認する(Φ ブロックが 432 → 108 になる見込み)
- [ ] **Step 2**:試走の推奨条件で 1 プロセス実行し、`zvo_parton_time.dat` で 1 step の秒数を (b′)(0.22 s)と比べる
- [ ] **Step 3**:E を α(−3.9399)・(b′)・ED(−3.941102)と比べ、§12.11 の目安(cond の漂流・NaN・捨てたサンプル)で判定してレポートに書く
- [ ] **Step 4**:CLAUDE.md の現在の状況と TODO を更新する
