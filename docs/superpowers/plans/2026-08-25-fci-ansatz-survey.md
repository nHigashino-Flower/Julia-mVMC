# FCI アンザッツ比較キャンペーン(フェーズ 1)実装計画

> **エージェント作業者へ:** 必須サブスキル: `superpowers:subagent-driven-development`(推奨)
> または `superpowers:executing-plans` でタスク単位に実装すること。手順は
> チェックボックス(`- [ ]`)で追跡する。

**Goal:** ν=1/2(4×4 ボゾン)と ν=1/3(6×3 フェルミオン)の FCI 相について、
拡大セル × フレーバー群 × グラフの 20 構成をパートン平均場 VMC で走らせ、
最もエネルギーが下がるアンザッツとその Chern 数を特定する。

**Architecture:** コア(`MVMCOptimizers.jl` などの `MVMC*` パッケージ)は無改修。
変更は (a) 共有 fixture `test/physics/parton_fixture.jl` のアンザッツ生成の一般化、
(b) `playground_nozomi/cb_nu12_boson/scripts/` の def 生成・ドライバ・投入・解析、
(c) `~/ED`(git 管理外)の 6×3 ED 計算と並進セクター測定、の 3 層。
fixture の変更はすべて **既定値で従来と完全にビット一致**させ、新しい挙動は
新しい kwarg でのみ有効化する。

**Tech Stack:** Julia 1.10(`~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia`)、
MPI.jl(mpiexec は MPI.jl 付属のもの)、`Test` 標準ライブラリ、
ED 側は `~/Code/Module*.jl` + `~/K-S-Model/ParentCode/ModuleParentHamCB.jl`、PBS(Torque)。

**Spec:** `docs/superpowers/specs/2026-08-25-fci-ansatz-survey-design.md`

## Global Constraints

- **Julia は 1.10 を使う**: `~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia`。
  `/opt/julia-1.8.5` は `[sources]` 未対応で解決できない。
- **BLAS スレッドは 1 に固定**(`LinearAlgebra.BLAS.set_num_threads(1)`)。
  MPI 16 ランクで固定を忘れると load が 168 に張り付く。
- **スレッド数は `JULIA_NUM_THREADS`**。`OMP_NUM_THREADS` は効かない。
- **ローカル機は物理 16 コア**。VMC は `mpiexec -n 16` + `JULIA_NUM_THREADS=1`、
  **同時 1 run**。2 run 並走は直列より遅い。
- **コミットは明示依頼があるまで行わない**(このリポジトリの運用ルール)。
  各タスクの「コミット」ステップは `git add` までを実行し、**コミットはしない**。
  代わりに `git status --short` で差分を確認して次へ進む。
- **`playground_nozomi/` は git 管理外**(`.gitignore`)。ここのスクリプトは
  履歴に残らない。`test/`・`tools/`・`docs/` はリポジトリ内。
- **既存 run_id を変えない**: ν=1/2・`indep`・`model` の組合せは従来通り無タグ。
- **P 層テストの走らせ方**: `julia --project=@. test/physics/runtests.jl`
  (ワークスペース root から)。単独ファイルは
  `julia --project=@. -e 'using Test; include("test/physics/ed_reference.jl"); include("test/physics/checkerboard_model.jl"); include("test/physics/parton_fixture.jl"); include("test/physics/test_ansatz_variants.jl")'`。
- **数値の許容**: 共変性・エルミート性の検査は `1e-12 * norm(H)`。
- **模型定数**: t=1.0, t1=0.2928932188134525, t2=−0.2928932188134525,
  t3=0.20710678118654754, ψ=π/4(`CheckerboardParams` の既定)。
- **ED 参照**: ν=1/2 4×4 = −16.304913354429445。ν=1/3 6×3 = 本計画で新規計算。

---

## File Structure

**リポジトリ内(git 管理)**

| ファイル | 責務 |
|---|---|
| `test/physics/parton_fixture.jl`(改修) | pmftrans/pmfpara/physhop/jastrowidx の生成。`flavor_groups`(フレーバー群)と `graph=:full`(全距離グラフ)を追加 |
| `test/physics/test_ansatz_variants.jl`(新規) | 上記 2 機能の回帰テスト(後方互換のビット一致・共変性・idx 数・6×3 対応) |
| `test/physics/runtests.jl`(改修) | 新テストの登録、6×3 ED ケースの存在判定 |
| `test/physics/ed_reference.jl`(改修) | `ED_CASE_FERMION_NU13_6X3` の登録 |
| `test/physics/test_p0_ed_reference.jl`(改修) | P0-d(6×3 ED 台帳)の追加 |
| `docs/superpowers/specs/2026-08-25-fci-ansatz-survey-design.md` | 設計(既存) |

**playground(git 管理外)** — すべて `playground_nozomi/cb_nu12_boson/scripts/`

| ファイル | 責務 |
|---|---|
| `gen_def.jl`(改修) | `FILLINGS` / `SYSTEMS` の (nx,ny,nu) 化 / `ANSATZ` 追加 / 充填門番の一般化 / `run_id` タグ / `gen_stage1_def` の新 kwargs |
| `chain.jl`(改修) | CLI `--nu --flavor --graph --k`、段2/3 の `nflavor` ハードコード除去 |
| `submit_ansatz.jl`(新規) | 20 構成 × 10 seed の直列投入・roster 追記・スキップ |
| `analyze_ansatz.jl`(新規) | roster 集計 + Chern 判定 → `_summary.md` |
| `ed_translation_sector.jl`(新規) | ED 固有ベクトルの並進固有値(フェルミオン符号込み) |

**`~/ED`(git 管理外・別リポジトリ)**

| ファイル | 責務 |
|---|---|
| `Code/run/Checkerboard/LocalState_nu13_6x3.jl`(新規) | 6×3・N=6・Fermion・U=1.0 の ED 固有状態計算 |
| `Code/run/Checkerboard/LocalState_nu13_6x3.sh`(新規) | PBS 投入スクリプト |

---

## Task 1: ED 6×3 の計算を投入する

最初に投入する。~1.5 h 走る間に Task 2 以降を進める。

**Files:**
- Create: `/home/nozomihigashino/ED/Code/run/Checkerboard/LocalState_nu13_6x3.jl`
- Create: `/home/nozomihigashino/ED/Code/run/Checkerboard/LocalState_nu13_6x3.sh`

**Interfaces:**
- Consumes: `~/Code/ModuleSpectrumFlow.jl` の `calc_local_energy_and_state_v1_2(SP1, SP2, IP, neig, folder_path, header)`、`~/K-S-Model/ParentCode/ModuleParentHamCB.jl` の `ED_Hamiltnian_CB`、`~/Code/ModuleBasic.jl`
- Produces: `/home/nozomihigashino/ED/Data/Checkerboard/Fermion/t=1.0-t1=0.293-t2=-0.293-t3=0.207-ψ=0.785/Nx=6-Ny=3-N=6-q=3-r=0.0/n=0/U=1.0-V=0.0/Psite-Vp=0-0.0/withoutRandomPotential_result_eigen_periodic.{txt,jld2}`

- [ ] **Step 1: ED スクリプトを書く**

`LocalState_benchmark.jl` を雛形にする。違いは (a) 系が 6×3・N=6 の 1 つ、
(b) 模型定数が既定値(t1/t2/t3 が非ゼロ)、(c) U は 1.0 の 1 点のみ、
(d) `r = 0.0`(乱雑ポテンシャル無し。4×4 ボゾン参照と揃える)。

`/home/nozomihigashino/ED/Code/run/Checkerboard/LocalState_nu13_6x3.jl`:

```julia
# =====================================================================
# LocalState_nu13_6x3.jl
#
# ν = 1/3 スピンレス・フェルミオン(6×3 セル・36 サイト・N=6)の ED 固有状態。
# パートン VMC のアンザッツ比較キャンペーン(方針 1 フェーズ 1)の参照値。
# 既存の 5×3 参照(r=1.0e8)と違い **r = 0.0**(乱雑ポテンシャル無し)。
# 拡大セル (3,1)/(3,3) が敷き詰められるのが 6×3 を選ぶ理由。
# 固定: n=0(flux なし)、ξ=η=0(境界ツイストなし)、V=0.0、neig=10。
# =====================================================================
include("/home/nozomihigashino/K-S-Model/ParentCode/ModuleParentHamCB.jl")
using .ED_Hamiltnian_CB
include("/home/nozomihigashino/Code/ModuleExternalFields.jl")
using .EFData_CB
using .ExternalFieldsBasic
include("/home/nozomihigashino/Code/ModuleBasic.jl")
using .makeFolders
include("/home/nozomihigashino/Code/ModuleSpectrumFlow.jl")
using .EDSpectrumFlow
using .ED_Polarization
EDSpectrumFlow.set_active_module(ED_Hamiltnian_CB)
ED_Polarization.set_active_module(ED_Hamiltnian_CB)

function ImplementLocalState()
    Nx, Ny, nelec = 6, 3, 6
    Ns = 2 * Nx * Ny                 # 36
    bc = "periodic"
    statistics = "Fermion"
    q = 3

    t  = 1.0
    t1 = 1 / (2 + sqrt(2))
    t2 = -1 / (2 + sqrt(2))
    t3 = 1 / (2 + 2 * sqrt(2))
    ψ  = pi / 4
    pseud_t  = round(t,  digits=3)
    pseud_t1 = round(t1, digits=3)
    pseud_t2 = round(t2, digits=3)
    pseud_t3 = round(t3, digits=3)
    pseud_ψ  = round(ψ,  digits=3)

    ξ = 0.0
    η = 0.0
    ϕ = 0.0
    U = 1.0
    V = 0.0
    neig = 10
    r = 0.0
    header = "withoutRandomPotential"

    R = randomPotentialField(Ns, r)
    psite = [0]
    Vp = [0.0]
    p_bond_1 = [0]
    p_bond_2 = [0]
    BondVp = [0.0]
    EFlist = zeros(Float64, Ns)

    println("=========================================================")
    println("System :: Nx=$Nx, Ny=$Ny, Ns=$Ns, nelec=$nelec, r=$r, U=$U")
    println("=========================================================")

    n = 0
    folder_path = joinpath(
        "/home", "nozomihigashino", "ED", "Data", "Checkerboard", statistics,
        "t=$pseud_t-t1=$pseud_t1-t2=$pseud_t2-t3=$pseud_t3-ψ=$pseud_ψ",
        "Nx=$Nx-Ny=$Ny-N=$nelec-q=$q-r=$r",
    )
    mkpath(folder_path)

    SP1 = SystemParams1(Nx, Ny, nelec, bc, statistics)
    SP2 = SystemParams2(t, t1, t2, t3, ψ, ϕ, ξ, η,
                        R, psite, Vp, p_bond_1, p_bond_2, BondVp, EFlist)
    IP = IntrctParams(U, V)
    calc_local_energy_and_state_v1_2(SP1, SP2, IP, neig, folder_path, header)
end

ImplementLocalState()
```

`/home/nozomihigashino/ED/Code/run/Checkerboard/LocalState_nu13_6x3.sh`:

```sh
#!/bin/sh
#PBS -q batch
#PBS -l nodes=1:ppn=1
#PBS -l walltime=1000:00:00
cd $PBS_O_WORKDIR
export JULIA_NUM_THREADS=1
julia /home/nozomihigashino/ED/Code/run/Checkerboard/LocalState_nu13_6x3.jl
```

- [ ] **Step 2: `randomPotentialField(Ns, 0.0)` が動くことを確認する**

r=0.0 で `1/r` を計算していると Inf になる。実装を先に読む:

Run:
```bash
grep -n 'function randomPotentialField' -A 15 /home/nozomihigashino/Code/ModuleBasic.jl /home/nozomihigashino/Code/ModuleExternalFields.jl 2>/dev/null
```
Expected: r で割る実装なら **r=0.0 は使えない**。その場合の対応は Step 3 で判断する。
既存の 4×4 ボゾン参照は `r=0.0` のディレクトリを持つので、r=0.0 の経路は
どこかに存在するはず。存在しなければ `r = 1.0e8`(実効振幅 1e-8)に変え、
ディレクトリ名も `r=1.0e8` になることを Task 8 の `ED_CASE_FERMION_NU13_6X3`
のパスに反映する。

- [ ] **Step 3: 小さい系で 1 分スモークする**

本番投入の前に、`Nx, Ny, nelec = 4, 3, 4`(既存 ED がある系)に一時的に
書き換えて走らせ、**出力ディレクトリが既存と同じ形で生えるか**だけ確認する。

Run:
```bash
cd /home/nozomihigashino/ED/Code/run/Checkerboard
sed 's/Nx, Ny, nelec = 6, 3, 6/Nx, Ny, nelec = 4, 3, 4/' LocalState_nu13_6x3.jl > /tmp/claude-1004/-home-nozomihigashino-Julia-mVMC/88b12273-f9d2-435c-8378-38b8185b93cc/scratchpad/ed_smoke.jl
JULIA_NUM_THREADS=1 julia /tmp/claude-1004/-home-nozomihigashino-Julia-mVMC/88b12273-f9d2-435c-8378-38b8185b93cc/scratchpad/ed_smoke.jl 2>&1 | tail -20
ls "/home/nozomihigashino/ED/Data/Checkerboard/Fermion/t=1.0-t1=0.293-t2=-0.293-t3=0.207-ψ=0.785/Nx=4-Ny=3-N=4-q=3-r=0.0/n=0/U=1.0-V=0.0/Psite-Vp=0-0.0/"
```
Expected: `withoutRandomPotential_result_eigen_periodic.txt` と `.jld2` が生える。
既に `Nx=4-Ny=3-N=4-q=3-r=0.0` に別 header のファイルがある場合は
**そのディレクトリに 2 つ目の `*result_eigen_periodic.txt` を作らないこと**
(`ed_reference.jl:106-108` が「丁度 1 つ」を要求する)。その場合はスモークを
`/tmp` 配下の folder_path に変えて実行する。

- [ ] **Step 4: 本番を投入する**

Run:
```bash
cd /home/nozomihigashino/ED/Code/run/Checkerboard && qsub LocalState_nu13_6x3.sh
qstat
```
Expected: ジョブ ID が返り、`qstat` に載る。**walltime は `qstat -f` の
`resources_used.walltime`(実時間)で確認する**。CPU 時間ではない。

- [ ] **Step 5: 記録**

```bash
git -C /home/nozomihigashino/Julia-mVMC status --short
```
`~/ED` はこのリポジトリの外なのでコミット対象外。投入したジョブ ID と時刻を
`playground_nozomi/cb_nu12_boson/PROJECT.md` の新セクションにメモする(Task 12 で本記録)。

---

## Task 2: fixture にフレーバー群(`flavor_groups`)を入れる

**Files:**
- Modify: `test/physics/parton_fixture.jl:77-91`(`_idx_key` / `_flavor_key`)、`:152-158`(シグネチャ)、`:305-327`(`write_parton_def_files`)
- Test: `test/physics/test_ansatz_variants.jl`(新規)

**Interfaces:**
- Consumes: 既存の `parton_fixture(nx, ny, nflavor, ex, ey; u_mf, u_bonds, idx_mode, psg_onsite, psg_shells, p)`
- Produces: `parton_fixture(...; flavor_groups::Union{Nothing,Vector{Int}} = nothing, ...)`。
  `flavor_groups[f+1]` がフレーバー f の群番号。`nothing` は従来通り `idx_mode` に従う。
  `write_parton_def_files(...; flavor_groups = nothing, ...)` も同じ kwarg を通す。

- [ ] **Step 1: 失敗するテストを書く**

`test/physics/test_ansatz_variants.jl` を新規作成:

```julia
"""
アンザッツ変種(フレーバー群 / 全距離グラフ)の回帰テスト
--- parton-mode (fork addition) ---

FCI アンザッツ比較キャンペーン(docs/superpowers/specs/2026-08-25-fci-ansatz-survey-design.md)
で追加した `flavor_groups` と `graph = :full` の検査。

要点:
1. **後方互換**: 既定(`flavor_groups = nothing`, `graph = :model`)は従来出力と
   完全一致。`[0,0,…]` は `:orbit` と、`[0,1,…,F-1]` は `:orbit_flavor` と一致
2. **idx 数**: 群数に比例する
3. **共変性**: 複素 α で拡大セル並進に不変(v3.14 と同じ検査)
4. **nx ≠ ny**(6×3)でも成立する
"""

using LinearAlgebra

# 複素 α で H_MF を組む(test_fixture_orientation.jl と同一規約)。
function _av_assemble(fx, α::Vector{ComplexF64}, nsite::Int, nflavor::Int)
    tval = Dict{NTuple{3,Int},ComplexF64}()
    for (s1, f1, s2, f2, v) in fx.pmftrans
        tval[(s1, f1, s2)] = v
    end
    H = [zeros(ComplexF64, nsite, nsite) for _ = 1:nflavor]
    for (s1, f1, s2, f2, idx, _) in fx.pmfpara
        v = α[idx + 1] * tval[(s1, f1, s2)]
        H[f1 + 1][s1 + 1, s2 + 1] += v
        s1 != s2 && (H[f1 + 1][s2 + 1, s1 + 1] += conj(v))
    end
    return H
end

function _av_cell_perm(nx::Int, ny::Int, sx::Int, sy::Int)
    nsite = 2 * nx * ny
    perm = Vector{Int}(undef, nsite)
    for s = 0:(nsite - 1)
        x, y = cb_site_to_xy(s, nx)
        perm[s + 1] = cb_xy_to_site(mod(x + 2sx, 2nx), mod(y + 2sy, 2ny), nx) + 1
    end
    return perm
end

"""
検査用の α。**オンサイト idx は実数**にする。

オンサイト行(site1 == site2)は h.c. を足さない規約なので、α が複素だと
H[s,s] が複素になりエルミート性が壊れる。本体は「オンサイト群の Im を強制凍結」
するので実際の run では起きない(DESIGN §2.3.1)。テストでも同じ制約を課す。
"""
function _av_alpha(fx)
    onsite = Set{Int}()
    for (s1, _, s2, _, idx, _) in fx.pmfpara
        s1 == s2 && push!(onsite, idx)
    end
    return ComplexF64[k in onsite ? ComplexF64(1.0 + 0.1k) :
                      (1.0 + 0.1k) * exp(im * (0.7k + 0.3))
                      for k = 0:(fx.n_idx - 1)]
end

"fixture の同一性(pmftrans / pmfpara / n_idx がビット一致)"
function _av_same(a, b)
    a.n_idx == b.n_idx && a.pmftrans == b.pmftrans && a.pmfpara == b.pmfpara
end

@testset "flavor_groups" begin
    # --- 後方互換: 既定は従来と完全一致 ---
    for (nx, ny, F, ex, ey) in ((4, 4, 2, 2, 2), (6, 3, 3, 3, 1))
        base = parton_fixture(nx, ny, F, ex, ey; idx_mode = :orbit_flavor)
        same = parton_fixture(nx, ny, F, ex, ey; idx_mode = :orbit_flavor,
                              flavor_groups = nothing)
        @test _av_same(base, same)
    end

    # --- [0,0,...] は :orbit と一致、[0,1,...] は :orbit_flavor と一致 ---
    for (nx, ny, F, ex, ey) in ((4, 4, 2, 2, 2), (6, 3, 3, 3, 3))
        orbit = parton_fixture(nx, ny, F, ex, ey; idx_mode = :orbit)
        allsym = parton_fixture(nx, ny, F, ex, ey; idx_mode = :orbit_flavor,
                                flavor_groups = fill(0, F))
        @test _av_same(orbit, allsym)

        of = parton_fixture(nx, ny, F, ex, ey; idx_mode = :orbit_flavor)
        allind = parton_fixture(nx, ny, F, ex, ey; idx_mode = :orbit_flavor,
                                flavor_groups = collect(0:(F - 1)))
        @test _av_same(of, allind)
    end

    # --- 2+1(F=3): idx 数は :orbit の 2 倍 ---
    orbit3 = parton_fixture(6, 3, 3, 3, 1; idx_mode = :orbit)
    two_one = parton_fixture(6, 3, 3, 3, 1; idx_mode = :orbit_flavor,
                             flavor_groups = [0, 0, 1])
    @test two_one.n_idx == 2 * orbit3.n_idx
    # フレーバー 0 と 1 が同じ idx を共有し、2 は別
    idx_of = Dict{Tuple{Int,Int,Int,Int},Int}()
    for (s1, f1, s2, f2, i, _) in two_one.pmfpara
        idx_of[(s1, f1, s2, f2)] = i
    end
    shared = [(s1, s2) for (s1, f1, s2, f2, _, _) in two_one.pmfpara if f1 == 0]
    for (s1, s2) in shared
        @test idx_of[(s1, 0, s2, 0)] == idx_of[(s1, 1, s2, 1)]
        @test idx_of[(s1, 0, s2, 0)] != idx_of[(s1, 2, s2, 2)]
    end

    # --- 群数が F と違う / 範囲外はエラー ---
    @test_throws ErrorException parton_fixture(4, 4, 2, 2, 2;
                                               flavor_groups = [0, 1, 2])

    # --- 共変性(複素 α、6×3 の (3,1) と (3,3))。ν=1/3 本番の u_mf = 1.0 も含める ---
    for (ex, ey) in ((3, 1), (3, 3)), grp in ([0, 0, 0], [0, 0, 1], [0, 1, 2]),
        umf in (0.0, 1.0)
        nx, ny, F = 6, 3, 3
        nsite = 2 * nx * ny
        fx = parton_fixture(nx, ny, F, ex, ey; idx_mode = :orbit_flavor,
                            flavor_groups = grp, u_mf = umf)
        H = _av_assemble(fx, _av_alpha(fx), nsite, F)
        for f = 1:F
            @test norm(H[f] - H[f]') < 1e-12 * norm(H[f])
            for (sx, sy) in ((ex, 0), (0, ey))
                perm = _av_cell_perm(nx, ny, sx, sy)
                @test norm(H[f][perm, perm] - H[f]) < 1e-12 * norm(H[f])
            end
        end
    end
end
```

- [ ] **Step 2: 落ちることを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. -e '
using Test
include("test/physics/checkerboard_model.jl")
include("test/physics/parton_fixture.jl")
include("test/physics/test_ansatz_variants.jl")' 2>&1 | tail -20
```
Expected: FAIL。`parton_fixture` が `flavor_groups` を知らないので
`MethodError: unsupported keyword argument`。

- [ ] **Step 3: 最小実装**

`test/physics/parton_fixture.jl` の `_flavor_key`(`:91`)を差し替える:

```julia
_idx_key(mode::Symbol, orbit_key, bond) =
    mode === :bond_flavor ? (:bond, bond) : orbit_key

"""
`groups` が与えられたらフレーバー f は `groups[f+1]` 番の群に属し、同じ群の
フレーバーが 1 つの idx を共有する(`fill(0,F)` = 全共有 = `:orbit` 相当、
`collect(0:F-1)` = 全独立 = `:orbit_flavor` 相当、`[0,0,1]` = 2+1)。
`nothing` なら従来の `idx_mode` 規約。
"""
_flavor_key(mode::Symbol, key, f::Int, groups::Union{Nothing,Vector{Int}}) =
    groups === nothing ? (mode === :orbit ? key : (key, f)) : (key, groups[f + 1])
```

`parton_fixture` のシグネチャ(`:152-158`)に kwarg を追加し、冒頭で検証する:

```julia
function parton_fixture(nx::Int, ny::Int, nflavor::Int, ex::Int, ey::Int;
                        u_mf::Float64 = 0.0,
                        u_bonds::Symbol = :nn,
                        idx_mode::Symbol = :orbit,
                        flavor_groups::Union{Nothing,Vector{Int}} = nothing,
                        psg_onsite::Bool = false,
                        psg_shells::Vector{Int} = Int[],
                        p::CheckerboardParams = CheckerboardParams())
    psg_onsite && u_mf != 0.0 &&
        error("psg_onsite と u_mf != 0 は同じ (i,i) 行を二重に作るので併用不可")
    if flavor_groups !== nothing
        length(flavor_groups) == nflavor || error(
            "flavor_groups の長さ $(length(flavor_groups)) が NFlavor = $(nflavor) と違います")
        all(0 .<= flavor_groups .< nflavor) || error(
            "flavor_groups の値は 0..$(nflavor-1): $(flavor_groups)")
        idx_mode === :bond_flavor && error(
            "flavor_groups と idx_mode = :bond_flavor は併用しません")
    end
```

本体の `_flavor_key(...)` 呼び出し **4 箇所すべて**(ボンドループ `:206`、
オンサイト `:216`、PSG オンサイト `:236`、PSG シェル `:254`)に
`flavor_groups` を第 4 引数として渡す:

```julia
            idx = get!(class_of_idx, _flavor_key(idx_mode, key, f, flavor_groups),
                       length(class_of_idx))
```

`write_parton_def_files`(`:305-327`)にも kwarg を足して素通しする:

```julia
                                flavor_groups::Union{Nothing,Vector{Int}} = nothing,
```
```julia
    fx = parton_fixture(nx, ny, F, ex, ey; u_mf = u_mf, idx_mode = idx_mode,
                        flavor_groups = flavor_groups,
                        psg_onsite = psg_onsite, psg_shells = psg_shells)
```

- [ ] **Step 4: 通ることを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. -e '
using Test
include("test/physics/checkerboard_model.jl")
include("test/physics/parton_fixture.jl")
include("test/physics/test_ansatz_variants.jl")' 2>&1 | tail -20
```
Expected: `Test Summary: | Pass  Total` で Fail 0。

- [ ] **Step 5: 既存 P 層が壊れていないことを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
OPENBLAS_NUM_THREADS=1 ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. test/physics/runtests.jl 2>&1 | tail -25
```
Expected: 既存の P0/P1/P2/orientation がすべて Pass(442 件が緑だった状態を維持)。

- [ ] **Step 6: ステージング**(コミットはしない)

```bash
cd /home/nozomihigashino/Julia-mVMC
git add test/physics/parton_fixture.jl test/physics/test_ansatz_variants.jl
git status --short
```

---

## Task 3: fixture に全距離グラフ(`graph = :full`)を入れる

**Files:**
- Modify: `test/physics/parton_fixture.jl:102-127`(`cb_psg_extra_bonds`)、`:152-280`(`parton_fixture` 本体)、`:305-327`(`write_parton_def_files`)
- Test: `test/physics/test_ansatz_variants.jl`(Task 2 で作成済み。testset を追加)

**Interfaces:**
- Consumes: Task 2 の `parton_fixture(...; flavor_groups)`
- Produces: `parton_fixture(...; graph::Symbol = :model)`。`:full` のとき
  pmftrans はオンサイト + 全変位(トーラス上の全サイト対)を**係数 1** で並べ、
  `psg_idx` は空(= 全 idx が 5 列 = 乱数初期化)。
  `cb_all_pairs(nx, ny) -> Vector{NTuple{5,Int}}`(`(i, j, dx, dy, d2)`、無向 1 本ずつ)を新設。

- [ ] **Step 1: 失敗するテストを書く**

`test/physics/test_ansatz_variants.jl` の末尾に追記:

```julia
@testset "graph = :full" begin
    # --- 4×4 ef4、F=2、全独立 ---
    nx, ny, F, ex, ey = 4, 4, 2, 2, 2
    nsite = 2 * nx * ny
    full = parton_fixture(nx, ny, F, ex, ey; graph = :full,
                          flavor_groups = collect(0:(F - 1)))

    # (a) 無向サイト対をすべて 1 回ずつ覆う + オンサイト nsite 個
    pairs = Set{Tuple{Int,Int}}()
    diag = Set{Int}()
    for (s1, f1, s2, f2, _) in full.pmftrans
        f1 == 0 || continue
        s1 == s2 ? push!(diag, s1) : push!(pairs, minmax(s1, s2))
    end
    @test length(diag) == nsite
    @test length(pairs) == div(nsite * (nsite - 1), 2)

    # (b) 係数はすべて 1(実)
    @test all(v == ComplexF64(1) for (_, _, _, _, v) in full.pmftrans)

    # (c) psg_idx は空 → 全 idx が乱数初期化
    @test isempty(full.psg_idx)

    # (d) idx 数 = 群数 × (拡大セル並進で割った軌道数)。
    #     独立(F 群)は共有(1 群)のちょうど F 倍
    sym = parton_fixture(nx, ny, F, ex, ey; graph = :full,
                         flavor_groups = fill(0, F))
    @test full.n_idx == F * sym.n_idx

    # (e) physhop は模型の t_ij のまま(平均場のグラフを広げても物理 H は不変)
    model = parton_fixture(nx, ny, F, ex, ey; flavor_groups = collect(0:(F - 1)))
    @test full.physhop == model.physhop

    # (f) 共変性: 複素 α で拡大セル並進に不変(4×4 と 6×3 の両方)
    for (nx2, ny2, F2, ex2, ey2) in ((4, 4, 2, 2, 1), (4, 4, 2, 2, 2),
                                     (6, 3, 3, 3, 1), (6, 3, 3, 3, 3))
        ns2 = 2 * nx2 * ny2
        fx = parton_fixture(nx2, ny2, F2, ex2, ey2; graph = :full,
                            flavor_groups = collect(0:(F2 - 1)))
        # graph = :full はオンサイトを必ず含むので α はオンサイトだけ実にする
        H = _av_assemble(fx, _av_alpha(fx), ns2, F2)
        for f = 1:F2
            @test norm(H[f] - H[f]') < 1e-12 * norm(H[f])
            for (sx, sy) in ((ex2, 0), (0, ey2))
                perm = _av_cell_perm(nx2, ny2, sx, sy)
                @test norm(H[f][perm, perm] - H[f]) < 1e-12 * norm(H[f])
            end
        end
    end

    # (g) 併用禁止
    @test_throws ErrorException parton_fixture(4, 4, 2, 2, 2; graph = :full,
                                               u_mf = 1.0)
    @test_throws ErrorException parton_fixture(4, 4, 2, 2, 2; graph = :full,
                                               psg_onsite = true)
    @test_throws ErrorException parton_fixture(4, 4, 2, 2, 2; graph = :full,
                                               psg_shells = [10])
    @test_throws ErrorException parton_fixture(4, 4, 2, 2, 2; graph = :bogus)
end
```

- [ ] **Step 2: 落ちることを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. -e '
using Test
include("test/physics/checkerboard_model.jl")
include("test/physics/parton_fixture.jl")
include("test/physics/test_ansatz_variants.jl")' 2>&1 | tail -20
```
Expected: FAIL(`graph` が未知の kwarg)。

- [ ] **Step 3: 全サイト対の列挙関数を書く**

`cb_psg_extra_bonds` の直後(`test/physics/parton_fixture.jl:127` の後)に追加:

```julia
"""
    cb_all_pairs(nx, ny) -> Vector{NTuple{5,Int}}

トーラス上の**全**無向サイト対を 1 本ずつ返す(`(i, j, dx, dy, d2)`)。
`graph = :full` 用。変位の列挙順を d² 昇順 → (dx, dy) 昇順 → サイト昇順に
固定してあるので、並進コピーは必ず同じ変位クラスへ落ちる(idx 共有が
射影的並進対称性を保つ条件)。巻き付きで同じサイト対に落ちる変位は先着 1 本。
"""
function cb_all_pairs(nx::Int, ny::Int)
    lx, ly = 2 * nx, 2 * ny
    nsite = 2 * nx * ny
    seen = Set{Tuple{Int,Int}}()
    disps = Tuple{Int,Int,Int}[]
    for dx = -(lx ÷ 2):(lx ÷ 2), dy = -(ly ÷ 2):(ly ÷ 2)
        (dx == 0 && dy == 0) && continue
        mod(dx, 2) == mod(dy, 2) || continue        # CB の副格子条件
        push!(disps, (dx^2 + dy^2, dx, dy))
    end
    sort!(disps)
    out = NTuple{5,Int}[]
    for (d2, dx, dy) in disps, i = 0:(nsite - 1)
        x0, y0 = cb_site_to_xy(i, nx)
        j = cb_xy_to_site(mod(x0 + dx, lx), mod(y0 + dy, ly), nx)
        i == j && continue
        key = minmax(i, j)
        key in seen && continue
        push!(seen, key)
        push!(out, (i, j, dx, dy, d2))
    end
    length(out) == div(nsite * (nsite - 1), 2) || error(
        "全サイト対の列挙が不足: $(length(out)) / $(div(nsite*(nsite-1), 2))")
    return out
end
```

- [ ] **Step 4: `parton_fixture` に `:full` 経路を足す**

シグネチャに `graph::Symbol = :model` を追加し、冒頭の検証を拡張:

```julia
    graph in (:model, :full) || error("graph は :model / :full。graph = $(graph)")
    if graph === :full
        (u_mf != 0.0 || psg_onsite || !isempty(psg_shells)) && error(
            "graph = :full は全サイト対を自前で並べるので u_mf / psg_onsite / " *
            "psg_shells とは併用しません")
    end
```

ボンドループの入力を切り替える(`bonds = cb_undirected_bonds(nx, ny)` の直後):

```julia
    nsite = 2 * nx * ny
    bonds = cb_undirected_bonds(nx, ny)
    # graph = :full は平均場のグラフだけを全サイト対へ広げる。物理 H(physhop)は
    # 模型の t_ij のまま(下の physhop ループが `bonds` を使う)。
    mf_bonds = graph === :full ? cb_all_pairs(nx, ny) : bonds
```

ボンドループの `for (i, j, dx, dy, d2) in bonds` を `in mf_bonds` に変え、
係数の決定だけ分岐する(向きの正準化・idx 割り当ては共通のまま):

```julia
        _, y0 = cb_site_to_xy(a, nx)
        if graph === :full
            coeff = ComplexF64(1)      # 全 1。t の距離制限を外し α に全部持たせる
        else
            t = cb_hopping(p, da, db, y0)
            # 内部検査: 反転はエルミート共役に一致するはず
            _, y0i = cb_site_to_xy(i, nx)
            abs(t - (ki <= kj ? cb_hopping(p, dx, dy, y0i) :
                                conj(cb_hopping(p, dx, dy, y0i)))) < 1e-13 ||
                error("cb_hopping is not Hermitian for bond ($i, $j, $dx, $dy)")
            add_u = (u_mf != 0.0) && (u_bonds === :all || d2 == 2)
            coeff = t + (add_u ? u_mf : 0.0)
        end
```

オンサイト項を `:full` でも出す(既存の `if u_mf != 0.0` ブロックの条件を広げ、
係数を分岐):

```julia
    # --- 平均場: オンサイト。u_mf != 0(ν=1/3 の Hartree)か graph = :full ---
    if u_mf != 0.0 || graph === :full
        onsite_coeff = graph === :full ? ComplexF64(1) : ComplexF64(u_mf)
        for i = 0:(nsite - 1)
            key = _idx_key(idx_mode, (enlarged_cell_class(i, nx, ny, ex, ey), 0, 0),
                           (i, i))
            for f = 0:(nflavor - 1)
                idx = get!(class_of_idx, _flavor_key(idx_mode, key, f, flavor_groups),
                           length(class_of_idx))
                push!(pmftrans, (ComplexF64(i), ComplexF64(f), ComplexF64(i),
                                 ComplexF64(f), onsite_coeff))
                push!(pmfpara, (i, f, i, f, idx, ComplexF64(1, 0)))
            end
        end
    end
```

`write_parton_def_files` に `graph::Symbol = :model` を足して素通しする。

- [ ] **Step 5: 通ることを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. -e '
using Test
include("test/physics/checkerboard_model.jl")
include("test/physics/parton_fixture.jl")
include("test/physics/test_ansatz_variants.jl")' 2>&1 | tail -20
```
Expected: Fail 0。

- [ ] **Step 6: 既存 P 層が壊れていないことを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
OPENBLAS_NUM_THREADS=1 ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. test/physics/runtests.jl 2>&1 | tail -25
```
Expected: すべて Pass。

- [ ] **Step 7: 新テストをランナーに登録してステージング**

`test/physics/runtests.jl` の「fixture orientation」testset の直後に追加:

```julia
    # アンザッツ変種(flavor_groups / graph = :full)。ED 非依存。
    @testset "ansatz variants" begin
        include(joinpath(PHYSICS_DIR, "test_ansatz_variants.jl"))
    end
```

```bash
cd /home/nozomihigashino/Julia-mVMC
OPENBLAS_NUM_THREADS=1 ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. test/physics/runtests.jl 2>&1 | tail -25
git add test/physics/parton_fixture.jl test/physics/test_ansatz_variants.jl test/physics/runtests.jl
git status --short
```

---

## Task 4: `gen_def.jl` を充填一般化する

**Files:**
- Modify: `playground_nozomi/cb_nu12_boson/scripts/gen_def.jl:52-58`(`ANSATZ`)、`:92-106`(`SYSTEMS` / `SEEDS`)、`:107-127`(`run_id`)、`:336-410`(`gen_stage1_def`)
- Test: `playground_nozomi/cb_nu12_boson/scripts/test_gen_def.jl`(新規、playground 内の軽量チェック)

**Interfaces:**
- Consumes: Task 2/3 の `write_parton_def_files(...; flavor_groups, graph)`
- Produces:
  - `const FILLINGS = Dict{Symbol,NamedTuple}` — `FILLINGS[:nu12] = (q=2, F=2, u_mf=0.0, u_phys=0.0, stat=:boson)`、`FILLINGS[:nu13] = (q=3, F=3, u_mf=1.0, u_phys=1.0, stat=:fermion)`
  - `system_for(nx, nu) -> (nx, ny, ne, nu)`
  - `flavor_groups_for(flavor::Symbol, F::Int) -> Vector{Int}`
  - `run_id(nx, ny, ansatz, seed; nu, flavor, graph, holes, use_qp, bond_flavor, psg, c4, psgclass)`
  - `gen_stage1_def(dir; nx, ny, ne, ansatz, seed, nu, flavor, graph, kx, ky, …)`

- [ ] **Step 1: 失敗するテストを書く**

`playground_nozomi/cb_nu12_boson/scripts/test_gen_def.jl`(新規):

```julia
# gen_def.jl の充填一般化の軽量チェック(playground 内。git 管理外)。
#   julia --project=<MVMCOptimizers.jl> scripts/test_gen_def.jl
using Test
include(joinpath(@__DIR__, "gen_def.jl"))

@testset "FILLINGS / systems / ansatz" begin
    @test FILLINGS[:nu12].q == 2 && FILLINGS[:nu12].F == 2
    @test FILLINGS[:nu13].q == 3 && FILLINGS[:nu13].F == 3
    @test FILLINGS[:nu13].u_mf == 1.0 && FILLINGS[:nu13].u_phys == 1.0

    @test system_for(4, :nu12) == (nx = 4, ny = 4, ne = 8, nu = :nu12)
    @test system_for(6, :nu13) == (nx = 6, ny = 3, ne = 6, nu = :nu13)
    @test_throws ErrorException system_for(5, :nu13)     # 未登録

    @test ANSATZ[:xexet3] == (ex = 3, ey = 1)
    @test ANSATZ[:ef9] == (ex = 3, ey = 3)
end

@testset "flavor_groups_for" begin
    @test flavor_groups_for(:sym, 2) == [0, 0]
    @test flavor_groups_for(:indep, 2) == [0, 1]
    @test flavor_groups_for(:sym, 3) == [0, 0, 0]
    @test flavor_groups_for(:two_one, 3) == [0, 0, 1]
    @test flavor_groups_for(:indep, 3) == [0, 1, 2]
    @test_throws ErrorException flavor_groups_for(:two_one, 2)   # F=2 に 2+1 は無い
end

@testset "run_id の後方互換とタグ" begin
    # ν=1/2・indep・model は従来通り無タグ
    @test run_id(4, 4, :ef4, 1001) == "L04_ef4_s1001"
    @test run_id(8, 8, :ef4, 1009; nu = :nu12, flavor = :indep, graph = :model) ==
          "L08_ef4_s1009"
    @test run_id(4, 4, :ef4, 1001; holes = 1) == "L04_ef4_h1_s1001"
    # 新しい軸はタグが付く
    @test run_id(4, 4, :ef4, 1001; flavor = :sym) == "L04_ef4_fsym_s1001"
    @test run_id(4, 4, :ef4, 1001; graph = :full) == "L04_ef4_full_s1001"
    @test run_id(6, 3, :ef9, 1001; nu = :nu13, flavor = :two_one) ==
          "L6x3_ef9_nu13_f21_s1001"
    @test run_id(6, 3, :xexet3, 1005; nu = :nu13, flavor = :sym, graph = :full) ==
          "L6x3_xexet3_nu13_fsym_full_s1005"
end

@testset "gen_stage1_def が 6×3 ν=1/3 で def を吐く" begin
    dir = mktempdir()
    r = gen_stage1_def(dir; nx = 6, ny = 3, ne = 6, ansatz = :ef9, seed = 1001,
                       nu = :nu13, flavor = :two_one, graph = :model,
                       stage = STAGES[1])
    @test isfile(joinpath(dir, "namelist.def"))
    @test r.nsite == 36
    # modpara に NFlavor 3 が書かれている
    mp = read(joinpath(dir, "modpara.def"), String)
    @test occursin(r"NFlavor\s+3", mp)
    @test occursin(r"NElec\s+6", mp)
    # QP は「全並進群 / H_MF の安定化群(拡大セル並進)」の剰余類なので
    #   n_qp = (nx·ny) / ((nx/ex)·(ny/ey)) = ex·ey
    # 既知値と整合: ef4 (2,2) -> 4、xexet2 (2,1) -> 2(PROJECT.md のアンザッツ対応表)
    @test r.n_qp == 3 * 3          # ef9 (ex,ey) = (3,3)
end

@testset "充填の門番" begin
    dir = mktempdir()
    @test_throws ErrorException gen_stage1_def(dir; nx = 6, ny = 3, ne = 9,
                                               ansatz = :ef9, seed = 1, nu = :nu13,
                                               stage = STAGES[1])
    @test_throws ErrorException gen_stage1_def(dir; nx = 4, ny = 4, ne = 8,
                                               ansatz = :ef9, seed = 1, nu = :nu12,
                                               stage = STAGES[1])   # 4 % 3 != 0
end
```

- [ ] **Step 2: 落ちることを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/test_gen_def.jl 2>&1 | tail -20
```
Expected: FAIL(`FILLINGS` が未定義)。

- [ ] **Step 3: 実装**

`gen_def.jl` の `ANSATZ`(`:52-58`)を差し替え:

```julia
"アンザッツ名 → 拡大ユニットセル (ex, ey)。ν=1/2 は参照VMC の EF ラベルと 1 対 1。"
const ANSATZ = Dict(
    :ef4     => (ex = 2, ey = 2),   # make_EFidx_type4  (8 クラス)
    :xexet2  => (ex = 2, ey = 1),   # make_EFidx_xext(K=2) (4 クラス)
    :ef9     => (ex = 3, ey = 3),   # ν=1/3 用(2026-08-25)
    :xexet3  => (ex = 3, ey = 1),   # ν=1/3 用(2026-08-25)
)

"""
充填ごとの定数。`q` = 充填の分母(ν = 1/q)、`F` = NFlavor、`u_mf` = 平均場に
載せる相互作用(ν=1/3 は V の Wick 分解 = NN 係数 t+V、対角 V)、
`u_phys` = 物理ハミルトニアンの NN 斥力。

ν=1/2 はハードコア・ボゾン(F=2、U=V=0)、ν=1/3 はスピンレス・フェルミオン
(F=3、NN 斥力 1.0)。設計は docs/superpowers/specs/2026-08-25-fci-ansatz-survey-design.md
"""
const FILLINGS = Dict(
    :nu12 => (q = 2, F = 2, u_mf = 0.0, u_phys = 0.0, stat = :boson),
    :nu13 => (q = 3, F = 3, u_mf = 1.0, u_phys = 1.0, stat = :fermion),
)
```

`SYSTEMS`(`:92-100`)を `(nx, ny, ne, nu)` に拡張:

```julia
"対象系 (Nux, Nuy, Ne, ν)。Ne = Nux·Nuy / q。"
const SYSTEMS = [
    (nx =  4, ny =  4, ne =   8, nu = :nu12),   # ドープ探索・アンザッツ比較の最小系
    (nx =  8, ny =  8, ne =  32, nu = :nu12),
    (nx = 10, ny = 10, ne =  50, nu = :nu12),
    (nx = 12, ny = 12, ne =  72, nu = :nu12),
    (nx = 14, ny = 14, ne =  98, nu = :nu12),
    (nx = 16, ny = 16, ne = 128, nu = :nu12),
    (nx =  6, ny =  3, ne =   6, nu = :nu13),   # ED 参照ありの ν=1/3(2026-08-25)
]

"(nx, ν) から系を引く。"
function system_for(nx::Int, nu::Symbol)
    k = findfirst(s -> s.nx == nx && s.nu == nu, SYSTEMS)
    k === nothing && error("未登録の系: Nux = $(nx), ν = $(nu)。SYSTEMS を確認")
    return SYSTEMS[k]
end

"""
    flavor_groups_for(flavor, F) -> Vector{Int}

フレーバー群の指定 → `parton_fixture` の `flavor_groups`。
`:sym` 全共有 / `:indep` 全独立 / `:two_one` 2 共有 + 1 独立(F=3 のみ)。
"""
function flavor_groups_for(flavor::Symbol, F::Int)
    flavor === :sym && return fill(0, F)
    flavor === :indep && return collect(0:(F - 1))
    if flavor === :two_one
        F == 3 || error(":two_one は F = 3 のみ(F = $(F))")
        return [0, 0, 1]
    end
    error("flavor は :sym / :two_one / :indep。flavor = $(flavor)")
end
```

`run_id`(`:107-127`)を差し替え。**後方互換**が要点:

```julia
"""
run_id。例: `L08_ef4_s1001`。系は正方なら `L{nx}`、そうでなければ `L{nx}x{ny}`。

**後方互換**: ν=1/2・flavor = :indep・graph = :model のときは従来通り
追加タグを付けない(既存 run のディレクトリ名を変えない)。
タグ順: 系 _ アンザッツ _ [nu13] _ [fsym|f21] _ [full] _ [bf] _ [pso|psf] _
[pc…] _ [c4n…] _ [h…] _ [noqp] _ s{seed}
"""
function run_id(nx::Int, ny::Int, ansatz::Symbol, seed::Int;
                nu::Symbol = :nu12, flavor::Symbol = :indep, graph::Symbol = :model,
                holes::Int = 0, use_qp::Bool = true, bond_flavor::Bool = false,
                psg::Symbol = :none, c4::Union{Nothing,Int} = nothing,
                psgclass::Union{Nothing,String} = nothing)
    sys = nx == ny ? @sprintf("L%02d", nx) : @sprintf("L%dx%d", nx, ny)
    tag = (nu === :nu12 ? "" : "_" * String(nu)) *
          (flavor === :indep ? "" : flavor === :sym ? "_fsym" : "_f21") *
          (graph === :model ? "" : "_full") *
          (bond_flavor ? "_bf" : "") *
          (psg === :onsite ? "_pso" : psg === :full ? "_psf" : "") *
          (psgclass === nothing ? "" : "_pc$(psgclass)") *
          (c4 === nothing ? "" : @sprintf("_c4n%d", c4)) *
          (holes > 0 ? @sprintf("_h%d", holes) : "") * (use_qp ? "" : "_noqp")
    return @sprintf("%s_%s%s_s%04d", sys, String(ansatz), tag, seed)
end
```

`gen_stage1_def`(`:341-410`)のシグネチャと門番・fixture 呼び出しを更新:

```julia
function gen_stage1_def(dir::AbstractString; nx::Int, ny::Int, ne::Int,
                        ansatz::Symbol, seed::Int,
                        nu::Symbol = :nu12,
                        flavor::Symbol = :indep, graph::Symbol = :model,
                        kx::Float64 = 0.0, ky::Float64 = 0.0,
                        stage = STAGES[1], occ_mode::Int = 1,
                        allow_doped::Bool = false, use_qp::Bool = true,
                        idx_mode::Symbol = :orbit_flavor,
                        psg::Symbol = :none, c4::Union{Nothing,Int} = nothing)
    haskey(FILLINGS, nu) || error("未知の充填 $(nu)。候補: $(keys(FILLINGS))")
    fil = FILLINGS[nu]
    psg in (:none, :onsite, :full) || error("psg は :none / :onsite / :full。psg = $(psg)")
    c4 !== nothing && !use_qp && error("--c4 は QP 射影の上に重ねる(use_qp = false と併用不可)")
    graph === :full && (psg !== :none || fil.u_mf != 0.0) && error(
        "graph = :full は psg / u_mf と併用しません(fixture が全サイト対を自前で並べる)")
    haskey(ANSATZ, ansatz) || error("未知のアンザッツ $(ansatz)。候補: $(keys(ANSATZ))")
    ex, ey = ANSATZ[ansatz].ex, ANSATZ[ansatz].ey
    nx % ex == 0 || error("Nux = $(nx) が拡大セル ex = $(ex) で割り切れません")
    ny % ey == 0 || error("Nuy = $(ny) が拡大セル ey = $(ey) で割り切れません")
    # ν = 1/q ちょうどが既定。ドープ(ホール側)は明示フラグでのみ許す
    if allow_doped
        0 < ne <= div(nx * ny, fil.q) || error(
            "ドープは 0 < Ne ≤ Nux·Nuy/$(fil.q)(ホール側のみ)。Ne = $(ne)")
    else
        fil.q * ne == nx * ny || error(
            "ν = 1/$(fil.q) では Ne = Nux·Nuy/$(fil.q)。Ne = $(ne), Nux·Nuy = $(nx*ny)。" *
            "ドープ計算は allow_doped = true を明示すること")
    end

    mkpath(dir)

    _, fx = write_parton_def_files(dir, nx, ny, fil.F, ex, ey;
                                   u_mf = graph === :full ? 0.0 : fil.u_mf,
                                   u_phys = fil.u_phys,
                                   n_elec = ne,
                                   nvmc_sample = stage.sample,
                                   nsr_step = stage.step,
                                   nsr_smp = stage.smp,
                                   dt = stage.dt,
                                   sta_del = stage.del,
                                   red_cut = stage.redcut,
                                   seed = seed,
                                   idx_mode = idx_mode,
                                   flavor_groups = flavor_groups_for(flavor, fil.F),
                                   graph = graph,
                                   psg_onsite = psg !== :none,
                                   psg_shells = psg === :full ? psg_shells_all(nx, ny) : Int[],
                                   jastrow_full_trans = true)
```

同関数内の `write_modpara` 呼び出しの `nflavor = nflavor` を `nflavor = fil.F` に変える。
`nflavor::Int = 2` の kwarg は**削除**する(呼び出し側は Task 5 で直す)。

- [ ] **Step 4: 通ることを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/test_gen_def.jl 2>&1 | tail -20
```
Expected: Fail 0。

- [ ] **Step 5: 既存 ν=1/2 の def がビット一致することを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl -e '
include("playground_nozomi/cb_nu12_boson/scripts/gen_def.jl")
d = mktempdir()
gen_stage1_def(d; nx = 4, ny = 4, ne = 8, ansatz = :ef4, seed = 1001, stage = STAGES[1])
for f in ("pmftrans.def", "pmfpara.def", "physhop.def", "qptransidx.def", "jastrowidx.def")
    println(f, "  ", bytes2hex(reinterpret(UInt8, [hash(read(joinpath(d, f), String))])))
end'
```
Expected: 既存 run(`runs_doped/L04_ef4_s1001/stage1_in/`)の同名ファイルと
一致する。比較:
```bash
for f in pmftrans.def pmfpara.def physhop.def qptransidx.def jastrowidx.def; do
  echo -n "$f "; md5sum < playground_nozomi/cb_nu12_boson/runs_doped/L04_ef4_s1001/stage1_in/$f
done
```
生成した一時ディレクトリの同名ファイルと md5 が一致すること
(`modpara.def` は seed / step が入るので比較対象外)。

- [ ] **Step 6: ステージング**

`gen_def.jl` は git 管理外なので `git add` の対象にならない。
`git status --short` で **playground が差分に出ないこと**を確認するだけ。

```bash
git status --short
```

---

## Task 5: `chain.jl` を充填一般化する

**Files:**
- Modify: `playground_nozomi/cb_nu12_boson/scripts/chain.jl:285-300`(`prepare_stage` の modpara 再生成)、`:306-372`(`run_stage`)、`:378-445`(CLI)

**Interfaces:**
- Consumes: Task 4 の `FILLINGS` / `system_for` / `run_id(nx, ny, …)` / `gen_stage1_def(…; nu, flavor, graph, kx, ky)`
- Produces: CLI `chain.jl <nx> <ansatz> <seed> [--nu 1/2|1/3] [--flavor sym|2+1|indep] [--graph model|full] [--k KX KY] …`。
  既存フラグ(`--label --spec --prev --runs --nsteps --holes --noqp --bf --psg --c4 --redcut --warm --psgclass`)は不変。

- [ ] **Step 1: CLI を拡張する**

`main(argv)` の系の引き方(`:386-389`)を差し替え:

```julia
    nx = parse(Int, argv[1])
    ansatz = Symbol(argv[2])
    seed = parse(Int, argv[3])
```
の後、**引数ループの後**に移動して次のようにする(`--nu` を読んでから系を引くため):

```julia
    # 既定値
    runs_dir = DEFAULT_RUNS_DIR
    label = "stage1"
    spec = 1
    prev = nothing
    nsteps = nothing
    holes = 0
    use_qp = true
    bond_flavor = false
    psg = :none
    c4 = nothing
    redcut = nothing
    warm = nothing
    psgclass = nothing
    nu = :nu12
    flavor = :indep
    graph = :model
    kx = 0.0
    ky = 0.0
    i = 4
    while i <= length(argv)
        a = argv[i]
        if a == "--runs"
            runs_dir = argv[i + 1]; i += 2
        elseif a == "--label"
            label = argv[i + 1]; i += 2
        elseif a == "--spec"
            spec = parse(Int, argv[i + 1]); i += 2
        elseif a == "--prev"
            prev = argv[i + 1]; i += 2
        elseif a == "--nsteps"
            nsteps = parse(Int, argv[i + 1]); i += 2
        elseif a == "--holes"
            holes = parse(Int, argv[i + 1]); i += 2
        elseif a == "--noqp"
            use_qp = false; i += 1
        elseif a == "--bf"
            bond_flavor = true; i += 1
        elseif a == "--psg"
            psg = Symbol(argv[i + 1]); i += 2
        elseif a == "--c4"
            c4 = parse(Int, argv[i + 1]); i += 2
        elseif a == "--redcut"
            redcut = parse(Float64, argv[i + 1]); i += 2
        elseif a == "--warm"
            warm = argv[i + 1]; i += 2
        elseif a == "--psgclass"
            psgclass = argv[i + 1]; i += 2
        elseif a == "--nu"
            s = argv[i + 1]
            nu = s in ("1/2", "nu12") ? :nu12 :
                 s in ("1/3", "nu13") ? :nu13 : error("--nu は 1/2 か 1/3(与: $s)")
            i += 2
        elseif a == "--flavor"
            s = argv[i + 1]
            flavor = s == "sym" ? :sym : s == "2+1" ? :two_one :
                     s == "indep" ? :indep : error("--flavor は sym / 2+1 / indep(与: $s)")
            i += 2
        elseif a == "--graph"
            s = argv[i + 1]
            graph = s == "model" ? :model : s == "full" ? :full :
                    error("--graph は model / full(与: $s)")
            i += 2
        elseif a == "--k"
            # K は単位胞の逆格子ベクトルを 1 とする分数(gen_def の write_qptransidx 規約)
            kx = parse(Float64, argv[i + 1]); ky = parse(Float64, argv[i + 2]); i += 3
        else
            error("未知の引数: $(a)")
        end
    end

    sys = system_for(nx, nu)
    ny, ne = sys.ny, sys.ne
```

使い方の文字列(`:379-385`)にも新フラグを足す:

```julia
    length(argv) >= 3 || error(
        "使い方: mpiexec -n 16 julia --project=<MVMCOptimizers.jl> chain.jl " *
        "<nx> <ansatz> <seed> --label <名前> --spec <1|2|3> [--prev <名前>] " *
        "[--runs DIR] [--nsteps N] [--holes N] [--noqp] [--bf] " *
        "[--psg onsite|full] [--c4 N] [--redcut X] [--warm STAGE_OUT_DIR] " *
        "[--nu 1/2|1/3] [--flavor sym|2+1|indep] [--graph model|full] [--k KX KY]")
```

`run_stage(...)` の呼び出しに `nu = nu, flavor = flavor, graph = graph, kx = kx, ky = ky` を追加。

- [ ] **Step 2: `run_stage` / `prepare_stage` に通す**

`run_stage`(`:311-321`)のシグネチャに追加:

```julia
                   nu::Symbol = :nu12, flavor::Symbol = :indep,
                   graph::Symbol = :model, kx::Float64 = 0.0, ky::Float64 = 0.0,
```

`run_id` 呼び出し(`:327-329`)を新シグネチャに合わせる:

```julia
    rid = run_id(nx, ny, ansatz, seed; nu = nu, flavor = flavor, graph = graph,
                 holes = holes, use_qp = use_qp,
                 bond_flavor = bond_flavor, psg = psg, c4 = c4,
                 psgclass = psgclass)
```

`prepare_stage` 呼び出し(`:341-345`)にも `nu = nu, flavor = flavor, graph = graph, kx = kx, ky = ky` を追加し、
`prepare_stage` 側のシグネチャ・`gen_stage1_def` 呼び出しにも同じものを通す。

- [ ] **Step 3: 段2/3 の `nflavor = 2` ハードコードを除去する**

`chain.jl:295` 付近の段2/3 modpara 再生成:

```julia
    n_qp = parse(Int, split(strip(readlines(joinpath(in_dir, "qptransidx.def"))[2]))[2])
    write_modpara(joinpath(in_dir, "modpara.def");
                  nsite = 2 * nx * ny, n_elec = ne, nflavor = FILLINGS[nu].F,
                  stage = stage, seed = stage_seed, n_mp_trans = n_qp)
```

`prepare_stage` が `nu` を受け取っていることを確認する(Step 2 で追加済み)。

- [ ] **Step 4: 既存 ν=1/2 の呼び出しが壊れていないか、他の呼び出し元を直す**

`run_id` のシグネチャが変わったので、呼んでいる箇所をすべて直す:

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC/playground_nozomi/cb_nu12_boson/scripts
grep -n 'run_id(' *.jl *.sh 2>/dev/null
```
Expected: `chain.jl` / `submit_doped.jl` / `submit_serial.jl` / `analyze.jl` /
`psg_def.jl` などがヒットする。**すべて第 2 引数に `ny` を入れる**
(既存はすべて正方系なので `run_id(nx, nx, …)`)。

- [ ] **Step 5: スモークで配管を確認する**

`CBNU12_QUICK=1` で ν=1/2 の従来経路と ν=1/3 の新経路を 1 本ずつ:

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
MPIEXEC=$(~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl -e 'using MPI; MPI.mpiexec(p -> print(p))')
SCR=/tmp/claude-1004/-home-nozomihigashino-Julia-mVMC/88b12273-f9d2-435c-8378-38b8185b93cc/scratchpad
CBNU12_QUICK=1 JULIA_NUM_THREADS=1 $MPIEXEC -n 4 \
  ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/chain.jl 4 ef4 9001 \
  --label stage1 --spec 1 --runs $SCR/smoke 2>&1 | tail -5
CBNU12_QUICK=1 JULIA_NUM_THREADS=1 $MPIEXEC -n 4 \
  ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/chain.jl 6 ef9 9001 \
  --nu 1/3 --flavor 2+1 --graph model --label stage1 --spec 1 --runs $SCR/smoke 2>&1 | tail -5
ls $SCR/smoke
```
Expected: `L04_ef4_s9001` と `L6x3_ef9_nu13_f21_s9001` が生え、両方とも
`stage1_out/zvo_out.dat` に 30 行(QUICK の step 数)ある。エラーなし。

- [ ] **Step 6: `graph = :full` もスモークする**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
CBNU12_QUICK=1 JULIA_NUM_THREADS=1 $MPIEXEC -n 4 \
  ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/chain.jl 4 ef4 9002 \
  --graph full --label stage1 --spec 1 --runs $SCR/smoke 2>&1 | tail -5
grep -c . $SCR/smoke/L04_ef4_full_s9002/stage1_in/pmftrans.def
head -2 $SCR/smoke/L04_ef4_full_s9002/stage1_in/pmfpara.def | tail -1
```
Expected: run が完走し、`pmftrans.def` の行数が
`5 + 2 × (32×31/2 + 32) = 5 + 2×528 = 1061`(F=2、オンサイト込み)。
`NPartonMFParaIdx` の値が `model` より大きい。

- [ ] **Step 7: sec/step を記録する**

`full` は SR が Npara² なので所要時間の見積り更新に使う。

Run:
```bash
cat $SCR/smoke/L04_ef4_s9001/chain.log $SCR/smoke/L04_ef4_full_s9002/chain.log
```
Expected: `sec_per_step` が両方出る。`full / model` の比を Task 12 の
PROJECT.md メモに書く。

---

## Task 6: `submit_ansatz.jl`(投入ドライバ)を書く

**Files:**
- Create: `playground_nozomi/cb_nu12_boson/scripts/submit_ansatz.jl`

**Interfaces:**
- Consumes: Task 4 の `run_id` / `SYSTEMS` / `system_for`、Task 5 の `chain.jl` CLI、`stage_io.jl` の `stage_summary(out_dir; expect_gap)`
- Produces: `playground_nozomi/cb_nu12_boson/runs_ansatz/_roster.dat`(1 run 1 行)、
  `runs_ansatz/_logs/<run_id>_stage1.log`

- [ ] **Step 1: ドライバを書く**

`playground_nozomi/cb_nu12_boson/scripts/submit_ansatz.jl`:

```julia
#=
FCI アンザッツ比較キャンペーン フェーズ 1 の直列投入ドライバ(2026-08-25)
--- cb_nu12_boson ---

    julia --project=MVMCOptimizers.jl scripts/submit_ansatz.jl \
        [--nu 1/2|1/3|all] [--wave 1|2] [--k13 KX KY] [--seeds 1001:1010] [--np 16] [--dry-run]

設計: docs/superpowers/specs/2026-08-25-fci-ansatz-survey-design.md

比較行列(段1 のみ 3000 step、16 ランク直列・同時 1 run):
  ν=1/2 (4×4, N=8, U=V=0):  {xexet2, ef4} × {sym, indep} × {model, full} = 8 構成
  ν=1/3 (6×3, N=6, U_NN=1):  {xexet3, ef9} × {sym, 2+1, indep} × {model, full} = 12 構成
各 10 seed。ν=1/3 の K は ED の基底から決めて --k13 で渡す(未指定はエラー)。

**波(--wave)**: 本番サンプルでの実測(t = 0.0865 + 1.458e-6·QP·Npara² [s/step])では
`ef9 × full` の 3 構成だけで 163 h / 10 seed を食う。ユーザー指示により ν=1/3 の
`full` は後回しでよいので、**重い `ef9 × full` だけを第 2 波へ切り出す**。
  第 1 波(既定) = 17 構成 = 48 h  (ν=1/2 全 8 + ν=1/3 model 全 6 + ν=1/3 xexet3×full 3)
  第 2 波        =  3 構成 = 163 h (ef9 sym full 12h / ef9 2+1 full 47h / ef9 indep full 104h)
第 1 波で「full が model に勝つか」が ν=1/2 の 4 構成と ν=1/3 xexet3 の 3 構成で分かるので、
負けていれば第 2 波は投入しなくてよい。
=#

using Printf
using Dates

const SCRIPT_DIR = @__DIR__
const PROJ_DIR = dirname(SCRIPT_DIR)
const JMVMC_ROOT = dirname(dirname(PROJ_DIR))
const PROJECT = joinpath(JMVMC_ROOT, "MVMCOptimizers.jl")
const JULIA_BIN = joinpath(homedir(),
    ".julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia")
const CHAIN = joinpath(SCRIPT_DIR, "chain.jl")

include(joinpath(SCRIPT_DIR, "gen_def.jl"))     # run_id / SYSTEMS / FILLINGS
include(joinpath(SCRIPT_DIR, "stage_io.jl"))    # stage_summary

function resolve_mpiexec()
    haskey(ENV, "CBNU12_MPIEXEC") && return ENV["CBNU12_MPIEXEC"]
    out = read(`$JULIA_BIN --project=$PROJECT -e 'using MPI; MPI.mpiexec(p -> print(p))'`,
               String)
    p = String(strip(out))
    isfile(p) || error("mpiexec が見つかりません: $(p)")
    return p
end

"1 構成 1 seed を mpiexec 配下で起動する。"
function launch(cfg, seed::Int; mpiexec::String, np::Int,
                runs_dir::String, log_dir::String)
    rid = run_id(cfg.nx, cfg.ny, cfg.ansatz, seed;
                 nu = cfg.nu, flavor = cfg.flavor, graph = cfg.graph)
    logf = joinpath(log_dir, "$(rid)_stage1.log")
    args = String[string(cfg.nx), String(cfg.ansatz), string(seed),
                  "--label", "stage1", "--spec", "1", "--runs", runs_dir,
                  "--nu", cfg.nu === :nu12 ? "1/2" : "1/3",
                  "--flavor", cfg.flavor === :sym ? "sym" :
                              cfg.flavor === :two_one ? "2+1" : "indep",
                  "--graph", String(cfg.graph),
                  "--k", string(cfg.kx), string(cfg.ky)]
    cmd = `$mpiexec -n $np $JULIA_BIN --project=$PROJECT $CHAIN $args`
    t0 = time()
    ok = try
        open(logf, "w") do io
            run(pipeline(cmd; stdout = io, stderr = io))
        end
        true
    catch
        false
    end
    return (rid = rid, ok = ok, wall = time() - t0, log = logf)
end

"比較行列の構成リスト。"
function configs(nu_sel::String, wave::Int, kx13::Float64, ky13::Float64)
    out = NamedTuple[]
    # 第 2 波に回すのは「ν=1/3 かつ ef9 かつ full」の 3 構成だけ(163 h / 10 seed)。
    # 第 1 波 = 残り 17 構成 = 48 h。full が model に勝たなければ第 2 波は不要になる。
    deferred(nu, ansatz, graph) = nu === :nu13 && ansatz === :ef9 && graph === :full
    want(nu, ansatz, graph) = wave == 2 ? deferred(nu, ansatz, graph) :
                                          !deferred(nu, ansatz, graph)
    if nu_sel in ("1/2", "all")
        s = system_for(4, :nu12)
        for ansatz in (:xexet2, :ef4), flavor in (:sym, :indep),
            graph in (:model, :full)
            want(:nu12, ansatz, graph) || continue
            push!(out, (nx = s.nx, ny = s.ny, ne = s.ne, nu = :nu12,
                        ansatz = ansatz, flavor = flavor, graph = graph,
                        kx = 0.0, ky = 0.0))
        end
    end
    if nu_sel in ("1/3", "all")
        s = system_for(6, :nu13)
        for ansatz in (:xexet3, :ef9), flavor in (:sym, :two_one, :indep),
            graph in (:model, :full)
            want(:nu13, ansatz, graph) || continue
            push!(out, (nx = s.nx, ny = s.ny, ne = s.ne, nu = :nu13,
                        ansatz = ansatz, flavor = flavor, graph = graph,
                        kx = kx13, ky = ky13))
        end
    end
    return out
end

function main(argv)
    nu_sel = "1/2"
    wave = 1
    kx13 = NaN
    ky13 = NaN
    seeds = collect(1001:1010)
    np = 16
    dry_run = false
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--nu"
            nu_sel = argv[i + 1]; i += 2
        elseif a == "--wave"
            wave = parse(Int, argv[i + 1])
            wave in (1, 2) || error("--wave は 1 か 2")
            i += 2
        elseif a == "--k13"
            kx13 = parse(Float64, argv[i + 1]); ky13 = parse(Float64, argv[i + 2]); i += 3
        elseif a == "--seeds"
            lo, hi = split(argv[i + 1], ":")
            seeds = collect(parse(Int, lo):parse(Int, hi)); i += 2
        elseif a == "--np"
            np = parse(Int, argv[i + 1]); i += 2
        elseif a == "--dry-run"
            dry_run = true; i += 1
        else
            error("未知の引数: $(a)")
        end
    end
    nu_sel in ("1/2", "1/3", "all") || error("--nu は 1/2 / 1/3 / all")
    if nu_sel in ("1/3", "all") && (isnan(kx13) || isnan(ky13))
        error("ν=1/3 を含むときは --k13 KX KY が必須(ED の基底の運動量。" *
              "scripts/ed_translation_sector.jl で測る)")
    end

    cfgs = configs(nu_sel, wave, kx13, ky13)
    runs_dir = joinpath(PROJ_DIR, "runs_ansatz")
    log_dir = joinpath(runs_dir, "_logs")
    roster = joinpath(runs_dir, "_roster.dat")

    total = length(cfgs) * length(seeds)
    println("=== アンザッツ比較 ν=$(nu_sel) 第 $(wave) 波: $(length(cfgs)) 構成 × $(length(seeds)) seed " *
            "= $(total) run(直列、mpi=$(np))===")
    for c in cfgs
        @printf("  %s  (%s, %s, %s, K=(%.4f, %.4f))\n",
                run_id(c.nx, c.ny, c.ansatz, seeds[1];
                       nu = c.nu, flavor = c.flavor, graph = c.graph),
                c.ansatz, c.flavor, c.graph, c.kx, c.ky)
    end
    dry_run && (println("--dry-run なので実行しません。"); return)

    mkpath(runs_dir); mkpath(log_dir)
    mpiexec = resolve_mpiexec()
    isfile(roster) || open(roster, "w") do io
        println(io, "# run_id  nu  ansatz  flavor  graph  seed  started  wall_sec  " *
                    "status  E_tail  min_gap_tail  sDiagMax_tail  converged")
    end

    k = 0
    for cfg in cfgs, seed in seeds
        k += 1
        rid = run_id(cfg.nx, cfg.ny, cfg.ansatz, seed;
                     nu = cfg.nu, flavor = cfg.flavor, graph = cfg.graph)
        out_dir = joinpath(runs_dir, rid, "stage1_out")
        chainlog = joinpath(runs_dir, rid, "chain.log")
        if isfile(chainlog) && occursin("stage1", read(chainlog, String))
            @printf("[%d/%d] %s 済み(スキップ)\n", k, total, rid)
            continue
        end
        t_start = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
        @printf("[%d/%d] %s 開始 %s\n", k, total, rid, t_start)
        flush(stdout)
        r = launch(cfg, seed; mpiexec = mpiexec, np = np,
                   runs_dir = runs_dir, log_dir = log_dir)
        s = stage_summary(out_dir; expect_gap = true)
        open(roster, "a") do io
            if s === nothing
                @printf(io, "%s %s %s %s %s %d %s %.1f %d - - - -\n",
                        rid, cfg.nu, cfg.ansatz, cfg.flavor, cfg.graph, seed,
                        t_start, r.wall, r.ok ? 0 : 1)
            else
                # stage_io.jl の stage_summary が返すのは e_tail / min_gap_tail /
                # sdiagmax_tail / converged(**小文字**。E_tail や sDiagMax_tail ではない)
                @printf(io, "%s %s %s %s %s %d %s %.1f %d %.6f %.6g %.6g %s\n",
                        rid, cfg.nu, cfg.ansatz, cfg.flavor, cfg.graph, seed,
                        t_start, r.wall, r.ok ? 0 : 1,
                        s.e_tail, s.min_gap_tail, s.sdiagmax_tail, s.converged)
            end
        end
        # 1 構成の失敗で全体を止めない(記録して続行)
        r.ok || @warn "run が失敗しました(続行)" rid = rid log = r.log
    end
    println("=== 全 $(total) run 終了 ===")
end

main(ARGS)
```

- [ ] **Step 2: `stage_summary` の返すフィールド名を確認する**

**確認済み(プレフライト、Ruling 5)**: 返るのは `e_tail` / `min_gap_tail` /
`sdiagmax_tail` / `converged`(すべて小文字。`E_tail` や `sDiagMax_tail` ではない)。
Step 1 のコードはこの名前で書いてある。念のため実物で確認する:

Run:
```bash
sed -n '154,172p' /home/nozomihigashino/Julia-mVMC/playground_nozomi/cb_nu12_boson/scripts/stage_io.jl
```
Expected: `e_tail = t.mean` / `min_gap_tail = gp_tail` / `sdiagmax_tail = sd_tail` /
`converged = …` が並ぶ。違っていたら Step 1 の書式指定をそちらへ合わせる。

- [ ] **Step 3: dry-run で構成リストを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/submit_ansatz.jl --nu all --k13 0 0 --dry-run
```
Expected: 20 構成 × 10 seed = 200 run と表示され、run_id が §1 の行列と一致する
(`L04_xexet2_fsym_s1001` … `L6x3_ef9_nu13_full_s1001` など)。

- [ ] **Step 4: ν=1/3 で `--k13` を忘れたらエラーになることを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/submit_ansatz.jl --nu 1/3 --dry-run 2>&1 | tail -3
```
Expected: 「--k13 KX KY が必須」のエラー。

- [ ] **Step 5: 2 run だけ実走して roster の書式を確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
CBNU12_QUICK=1 ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia \
  --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/submit_ansatz.jl \
  --nu 1/2 --seeds 9001:9001 --np 4 2>&1 | tail -20
head -5 playground_nozomi/cb_nu12_boson/runs_ansatz/_roster.dat
```
Expected: 8 run(QUICK なので各数秒)が走り、roster に 8 行 + ヘッダ。
**確認後、QUICK で作った run を消す**:
```bash
rm -rf playground_nozomi/cb_nu12_boson/runs_ansatz
```

---

## Task 7: ED の並進セクター測定スクリプトを書く

Task 1 の ED が完走してから実行する。

**プレフライト裁定(Ruling 2、2026-08-25)**: 本タスクは新規実装ではなく
**既存 `playground_nozomi/cb_nu12_boson/scripts/ed_c4_eigenvalue.jl` の派生**として書く。
当初の記述には実行不能な欠陥が 3 件あった:

1. マスクの全探索(`for m = 0:(1<<36 - 1)`)は 6.9e10 反復で不可能。
   既存実装は **Gosper のホップ**で C(36,6) = 1.95e6 のマスクだけを昇順に列挙する
2. 固有ベクトルは **複素**(ψ=π/4 でホッピングが複素 → H は複素エルミート)。
   `Vector{Float64}` を仮定してはいけない
3. jld2 のキーは推測不要。**`"totalenergy"`(固有値ベクトル)と `"eigenstates"`(列が固有ベクトル)**

既存実装の `rank_of` / Gosper 列挙 / 複素 `overlap` をそのまま流用し、
**フェルミオンの置換符号**と、NX/NY/NE/パスの CLI 化だけを足す。

**Files:**
- Create: `playground_nozomi/cb_nu12_boson/scripts/ed_translation_sector.jl`
- Reference: `playground_nozomi/cb_nu12_boson/scripts/ed_c4_eigenvalue.jl`(流用元。**変更しない**)

**Interfaces:**
- Consumes: Task 1 の jld2(キー `"totalenergy"` / `"eigenstates"`)
- Produces: 標準出力に「準位 / E / ⟨T(1,0)⟩ / ⟨T(0,1)⟩ / (kx, ky)」の表。
  kx, ky は単位胞逆格子を 1 とする分数(`write_qptransidx` の kx/ky 規約と同じ)。

- [ ] **Step 1: 流用元を読む**

Run:
```bash
cat /home/nozomihigashino/Julia-mVMC/playground_nozomi/cb_nu12_boson/scripts/ed_c4_eigenvalue.jl
```
Expected: `rank_of`(colex 順位)、`perm_mask`(符号なし置換)、`overlap`(Gosper
ホップで基底を走査しつつ ⟨ψ|P|φ⟩ を積む)、`site_to_xy` / `xy_to_site` / `t_of` が読める。
**この 5 つをそのまま持ってくる**。書き換えるのは `perm_mask` → 符号付き版と、
定数 `NX/NY/NSITE/NE/PATH` → CLI 引数化の 2 点だけ。

- [ ] **Step 2: スクリプトを書く**

`playground_nozomi/cb_nu12_boson/scripts/ed_translation_sector.jl`:

```julia
# ED 固有状態の単位胞並進固有値(運動量セクター)を測る。
# scripts/ed_c4_eigenvalue.jl の派生。違いは 2 点だけ:
#   (1) スピンレス・フェルミオンなので置換に**符号**が付く(ボゾン版は符号なし)
#   (2) NX/NY/NE と jld2 のパスを CLI 引数にした
#
#   julia --project=MVMCOptimizers.jl scripts/ed_translation_sector.jl \
#       <jld2> <NX> <NY> <NE> [NLEV] [--boson]
#
# 基底規約(~/Code/ModuleBasic.jl の basis_list): Int64 ビットマスク
# (bit = サイト 0..NSITE-1)を数値昇順 = 占有集合の colex 順。
# 出力の (kx, ky) は単位胞逆格子を 1 とする分数 = gen_def.jl の write_qptransidx の
# kx/ky 規約。chain.jl の --k にそのまま渡せる。
using JLD2, LinearAlgebra, Printf

length(ARGS) >= 4 || error(
    "使い方: ed_translation_sector.jl <jld2> <NX> <NY> <NE> [NLEV] [--boson]")
const PATH = ARGS[1]
const NX = parse(Int, ARGS[2])
const NY = parse(Int, ARGS[3])
const NE = parse(Int, ARGS[4])
const NLEV = length(ARGS) >= 5 && !startswith(ARGS[5], "--") ? parse(Int, ARGS[5]) : 4
const BOSON = "--boson" in ARGS      # 検算用(ハードコアボゾン = 符号なし)
const LX, LY = 2NX, 2NY
const NSITE = 2 * NX * NY

site_to_xy(s) = (y = div(s, NX); col = s - NX * y; (iseven(y) ? 2col : 2col + 1, y))
xy_to_site(x, y) = iseven(y) ? NX * y + div(x, 2) : NX * y + div(x - 1, 2)

"単位胞並進 (dux, duy)。倍密グリッドなので 2 倍する。"
t_of(s, dux, duy) = ((x, y) = site_to_xy(s);
                     xy_to_site(mod(x + 2dux, LX), mod(y + 2duy, LY)))

"占有マスクの colex 順位(1-based)"
function rank_of(mask::Int64)
    r, k, m = 0, 0, mask
    while m != 0
        p = trailing_zeros(m); k += 1
        r += binomial(p, k)
        m &= m - 1
    end
    return r + 1
end

"""
サイト置換 p(0-based、p[s+1] = 移動先)をマスクへ適用し `(新マスク, 符号)` を返す。

フェルミオン: 基底を c†_{p_1} c†_{p_2} … (p_1 < p_2 < …) と定義すると、置換後の
生成演算子列を昇順へ並べ替える互換の偶奇が符号 = 「移動先リストの転倒数」の偶奇。
ハードコアボゾン(`--boson`)では常に +1。
"""
function perm_mask_sign(mask::Int64, p::Vector{Int})
    dest = Int[]
    m = mask
    while m != 0
        s = trailing_zeros(m)
        push!(dest, p[s + 1])
        m &= m - 1
    end
    out = Int64(0)
    for d in dest
        out |= Int64(1) << d
    end
    BOSON && return out, 1.0
    inv = 0
    @inbounds for a = 1:length(dest), b = (a + 1):length(dest)
        dest[a] > dest[b] && (inv += 1)
    end
    return out, iseven(inv) ? 1.0 : -1.0
end

"""
⟨ψ|P_p|φ⟩ = Σ_x sgn(p, x) conj(ψ[rank(p·x)]) φ[rank(x)]

基底の走査は Gosper のホップ(固定 popcount のマスクを数値昇順に生成)。
NSITE=36 / NE=6 なら C(36,6) = 1,947,792 回で済む(全探索 2^36 は不可能)。
"""
function overlap(psi::Vector{ComplexF64}, phi::Vector{ComplexF64}, p::Vector{Int})
    acc = zero(ComplexF64)
    mask = Int64((1 << NE) - 1)
    last = Int64((1 << NE) - 1) << (NSITE - NE)
    idx = 0
    while mask <= last
        idx += 1
        nm, sg = perm_mask_sign(mask, p)
        acc += sg * conj(psi[rank_of(nm)]) * phi[idx]
        t = (mask | (mask - 1)) + 1
        mask = t | (div(t & -t, mask & -mask) >> 1) - 1
    end
    @assert idx == binomial(NSITE, NE)
    return acc
end

function main()
    p10 = [t_of(s, 1, 0) for s = 0:(NSITE - 1)]
    p01 = [t_of(s, 0, 1) for s = 0:(NSITE - 1)]
    @assert sort(p10) == collect(0:(NSITE - 1))
    @assert sort(p01) == collect(0:(NSITE - 1))

    println("loading eigenstates … ", PATH); flush(stdout)
    f = jldopen(PATH, "r")
    E = f["totalenergy"]
    S = f["eigenstates"]
    close(f)
    size(S, 1) == binomial(NSITE, NE) || error(
        "基底次元が合いません: $(size(S, 1)) vs C($(NSITE),$(NE)) = $(binomial(NSITE, NE))")

    nst = min(NLEV, length(E), size(S, 2))
    @printf("%-4s %-18s %-34s %-34s %s\n",
            "lev", "E", "<T(1,0)>", "<T(0,1)>", "(kx, ky)")
    for n = 1:nst
        v = Vector{ComplexF64}(S[:, n]); v ./= norm(v)
        o10 = overlap(v, v, p10)
        o01 = overlap(v, v, p01)
        kx = angle(o10) / (2π)
        ky = angle(o01) / (2π)
        @printf("%-4d %-18.12f %+.6f%+.6fim (|.|=%.6f) %+.6f%+.6fim (|.|=%.6f) (%+.6f, %+.6f)\n",
                n, E[n], real(o10), imag(o10), abs(o10),
                real(o01), imag(o01), abs(o01), kx, ky)
    end
    println()
    println("読み方: |⟨T⟩| ≈ 1 ならその準位は T の固有状態で、位相 arg⟨T⟩ = 2πk が運動量。")
    println("        |⟨T⟩| が 1 から離れていたら縮退多様体内で T が混ざっている ->")
    println("        多様体内で T を同時対角化して読み直すこと。")
    println("        k は単位胞逆格子を 1 とする分数(chain.jl の --k にそのまま渡せる)。")
end

main()
```

- [ ] **Step 3: 既知の答え(4×4 ボゾン)で検算する**

ν=1/2 の 4×4 ED は基底・第1励起とも ⟨T(1,0)⟩ = +1(PROJECT.md 2026-08-20)。
ハードコアボゾンなので `--boson` を付けて走らせる。

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/ed_translation_sector.jl \
  "/home/nozomihigashino/ED/Data/Checkerboard/Boson/t=1.0-t1=0.293-t2=-0.293-t3=0.207-ψ=0.785/Nx=4-Ny=4-N=8-q=2-r=0.0/n=0/U=0.0-V=0.0/Psite-Vp=0-0.0/result_eigen_periodic.jld2" \
  4 4 8 3 --boson 2>&1 | tail -10
```
Expected: lev 1 の E が **−16.304913…**、⟨T(1,0)⟩ と ⟨T(0,1)⟩ がともに
**+1.000000+0.000000im(|.| = 1.000000)**、(kx, ky) = (0, 0)。
一致しなければ基底規約か写像が違うので、`ed_c4_eigenvalue.jl` と 1 行ずつ突き合わせる。

- [ ] **Step 4: フェルミオン符号の自己検査**

符号の実装が正しければ、**符号を入れても入れなくても |⟨T⟩| は 1 のまま**
(位相だけが変わり得る)。6×3 の本番データで両方走らせて比べる。

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
D="/home/nozomihigashino/ED/Data/Checkerboard/Fermion/t=1.0-t1=0.293-t2=-0.293-t3=0.207-ψ=0.785/Nx=6-Ny=3-N=6-q=3-r=0.0/n=0/U=1.0-V=0.0/Psite-Vp=0-0.0"
ls "$D"
J="$D/withoutRandomPotential_result_eigen_periodic.jld2"
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/ed_translation_sector.jl "$J" 6 3 6 6 2>&1 | tail -14
echo "--- 符号なし(対照) ---"
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/ed_translation_sector.jl "$J" 6 3 6 6 --boson 2>&1 | tail -14
```
**注意**: Task 1 で `r = 1.0e8` になっていた場合はディレクトリ名の `r=0.0` を
`r=1.0e8` に読み替える。`ls "$D"` が失敗したら `find /home/nozomihigashino/ED/Data/Checkerboard/Fermion -path '*Nx=6-Ny=3-N=6*' -name '*.jld2'` で探す。

Expected: 6 準位の E と ⟨T⟩ が出る。**符号ありの方で |⟨T⟩| ≈ 1** になっていること
(符号なしで |⟨T⟩| が 1 から外れ、符号ありで 1 になるなら符号の実装が効いている証拠)。
**基底(lev 1)の (kx, ky) を記録する**。|⟨T⟩| が両方とも 1 から離れていたら
縮退多様体で混ざっているので、その事実を記録して Task 10 は Γ (0,0) で走らせる。

- [ ] **Step 5: 3 重準縮退を確認する**

Run:
```bash
grep -E '^[0-9]+-th energy' "$D/withoutRandomPotential_result_eigen_periodic.txt" | head -6
```
Expected: 最低 3 準位の分裂が、3-th と 4-th の間のギャップより小さい(= 3 重準縮退)。
そうでなければ ν=1/3 FCI の 3 重項が立っていないので、その事実を記録する
(計画は続行。判定の前提が変わるだけ)。

---

## Task 8: 6×3 ED をリポジトリの P0 テストに登録する

**Files:**
- Modify: `test/physics/ed_reference.jl:190-198`(ケース定数)
- Modify: `test/physics/test_p0_ed_reference.jl`(P0-d を追加)
- Modify: `test/physics/runtests.jl:33-46`(スキップ判定)

**Interfaces:**
- Consumes: Task 1 の ED 出力、Task 7 で確認した基底エネルギー
- Produces: `ED_CASE_FERMION_NU13_6X3::String`

- [ ] **Step 1: 実測値を読む**

Run:
```bash
D="/home/nozomihigashino/ED/Data/Checkerboard/Fermion/t=1.0-t1=0.293-t2=-0.293-t3=0.207-ψ=0.785/Nx=6-Ny=3-N=6-q=3-r=0.0/n=0/U=1.0-V=0.0/Psite-Vp=0-0.0"
grep -E '^[0-9]+-th energy' "$D/withoutRandomPotential_result_eigen_periodic.txt" | head -5
```
Expected: 最低 5 準位が出る。**0-th の値と、0..2 の分裂(spread)、
2-th と 3-th のギャップ**をメモする。以下 Step 2 の `E0` / `SPREAD` に埋める。

- [ ] **Step 2: 失敗するテストを書く**

`test/physics/test_p0_ed_reference.jl` の P0-b の直後に追加
(`E0` と `SPREAD` は Step 1 の実測値に置き換える):

```julia
@testset "P0-d ν=1/3 フェルミオン(6×3 ユニットセル・36 サイト・6 粒子)" begin
    ref = read_ed_reference(ED_CASE_FERMION_NU13_6X3)
    print(ed_ledger(ref))

    @test ref.nx == 6
    @test ref.ny == 3
    @test ref.nsite == 36
    @test ref.nsite == 2 * ref.nx * ref.ny
    @test ref.nelec == 6
    @test ref.statistics == "Fermion"
    @test ref.boundary == "periodic"
    @test ref.nelec // (ref.nx * ref.ny) == 1 // 3

    @test ref.t == 1.0
    @test isapprox(ref.t1, 0.2928932188134525; atol = 1e-15)
    @test isapprox(ref.t2, -0.2928932188134525; atol = 1e-15)
    @test isapprox(ref.t3, 0.20710678118654754; atol = 1e-15)
    @test isapprox(ref.psi, 0.7853981633974483; atol = 1e-15)
    @test ref.phi == 0.0 && ref.xi == 0.0 && ref.eta == 0.0
    @test ref.u == 1.0 && ref.v == 0.0                          # NN 斥力のみ

    # 乱雑ポテンシャルなし(4×4 ボゾン参照と同じ条件)
    @test ref.random_potential_max == 0.0

    e_min, spread, gap = ed_ground_manifold(ref, 3)             # 3 重準縮退
    @test isapprox(e_min, E0; atol = 1e-12)                     # ← Step 1 の実測値
    @test isapprox(spread, SPREAD; atol = 1e-12)                # ← 同上
    @test gap > spread                                          # 多様体の外が離れている
end
```

- [ ] **Step 3: 落ちることを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
OPENBLAS_NUM_THREADS=1 ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. test/physics/runtests.jl 2>&1 | tail -20
```
Expected: `ED_CASE_FERMION_NU13_6X3` が未定義で UndefVarError。

- [ ] **Step 4: ケース定数を登録する**

`test/physics/ed_reference.jl` の末尾(`ED_CASE_FERMION_NU13` の直後)に追加。
**ディレクトリ名の `r=` は Task 1 Step 2 の判断に合わせる**(r=0.0 が使えなければ `r=1.0e8`):

```julia
# ν=1/3 アンザッツ比較キャンペーン用(2026-08-25 に新規計算)。
# 5×3 と違い r = 0.0(乱雑ポテンシャル無し)。拡大セル (3,1)/(3,3) が敷き詰められる。
const ED_CASE_FERMION_NU13_6X3 = joinpath(
    ED_ROOT,
    "Fermion/t=1.0-t1=0.293-t2=-0.293-t3=0.207-ψ=0.785",
    "Nx=6-Ny=3-N=6-q=3-r=0.0/n=0/U=1.0-V=0.0/Psite-Vp=0-0.0",
)
```

`test/physics/runtests.jl` の P0/P1 ブロックに 6×3 のスキップ判定を足す
(既存の 2 ケース判定はそのまま、P0-d だけ別条件にする):

```julia
    if !isdir(ED_CASE_BOSON_NU12) || !isdir(ED_CASE_FERMION_NU13)
        @warn """外部 ED データが見つからないので P0/P1 をスキップします。
                 boson:   $ED_CASE_BOSON_NU12
                 fermion: $ED_CASE_FERMION_NU13"""
    else
        @testset "P0 ED reference" begin
            include(joinpath(PHYSICS_DIR, "test_p0_ed_reference.jl"))
        end
        @testset "P1 model conventions" begin
            include(joinpath(PHYSICS_DIR, "test_p1_onebody.jl"))
        end
    end
```

P0-d は `test_p0_ed_reference.jl` の中で自分でスキップする:

```julia
if !isdir(ED_CASE_FERMION_NU13_6X3)
    @warn "6×3 の ED データが無いので P0-d をスキップします: $ED_CASE_FERMION_NU13_6X3"
else
    @testset "P0-d ν=1/3 フェルミオン(6×3 ユニットセル・36 サイト・6 粒子)" begin
        # (Step 2 の内容)
    end
end
```

- [ ] **Step 5: 通ることを確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
OPENBLAS_NUM_THREADS=1 ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. test/physics/runtests.jl 2>&1 | tail -25
```
Expected: P0-d を含めて全部 Pass。

- [ ] **Step 6: ステージング**

```bash
cd /home/nozomihigashino/Julia-mVMC
git add test/physics/ed_reference.jl test/physics/test_p0_ed_reference.jl test/physics/runtests.jl
git status --short
```

---

## Task 9: ν=1/2 の 8 構成を投入する

**Files:**
- 変更なし(Task 6 のドライバを走らせるだけ)

**Interfaces:**
- Consumes: Task 6 の `submit_ansatz.jl`
- Produces: `playground_nozomi/cb_nu12_boson/runs_ansatz/` に 80 run + roster 80 行

- [ ] **Step 1: マシンが空いていることを確認する**

Run:
```bash
uptime; ps aux | grep -c '[j]ulia'
```
Expected: load average が 1 未満、julia プロセスが 0。
**走っていたら終わるまで待つ**(同時 2 run は直列より遅い)。

- [ ] **Step 2: dry-run で最終確認**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/submit_ansatz.jl --nu 1/2 --dry-run
```
Expected: 8 構成 × 10 seed = 80 run。

- [ ] **Step 3: バックグラウンドで投入する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
nohup ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia \
  --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/submit_ansatz.jl --nu 1/2 \
  > playground_nozomi/cb_nu12_boson/runs_ansatz_nu12_submit.log 2>&1 &
echo $!
```
Expected: PID が返る。ログに `[1/80] L04_xexet2_fsym_s1001 開始` が出る。

- [ ] **Step 4: 最初の 2 run が通ることを確認する**

数十分後(`model` は ~10 分、`full` は未知):

Run:
```bash
tail -5 /home/nozomihigashino/Julia-mVMC/playground_nozomi/cb_nu12_boson/runs_ansatz_nu12_submit.log
tail -3 /home/nozomihigashino/Julia-mVMC/playground_nozomi/cb_nu12_boson/runs_ansatz/_roster.dat
```
Expected: roster に E_tail が入った行が積まれている。E_tail が
**−16 付近**(ν=1/2 4×4 の ED は −16.3049)。桁が違ったら止めて原因を調べる。

- [ ] **Step 5: `full` の 1 run の所要時間を測って見積りを更新する**

`L04_xexet2_fsym_full_s1001` が終わったら:

Run:
```bash
grep full /home/nozomihigashino/Julia-mVMC/playground_nozomi/cb_nu12_boson/runs_ansatz/_roster.dat | head -3
```
Expected: `wall_sec` が読める。80 run の総見積りを更新し、Task 12 でメモする。
`full` が `model` の 10 倍を超えるようなら、seed を 10 → 5 に減らすかどうかを
**ユーザーに確認する**(勝手に減らさない)。

---

## Task 10: ν=1/3 の 12 構成を投入する

Task 7 で K が確定し、Task 9 が終わってから。

**Files:**
- 変更なし

**Interfaces:**
- Consumes: Task 7 の (kx, ky)、Task 6 のドライバ
- Produces: `runs_ansatz/` に 120 run 追加

- [ ] **Step 1: K を確認する**

Task 7 Step 3 で測った基底の (kx, ky) を使う。⟨T⟩ が ±1 から外れていた場合は
**Γ(0, 0)で走らせた上で、その旨を記録する**(セクター混合の可能性を残す)。

- [ ] **Step 2: 1 run スモークして E の桁を確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
KX=<Task 7 の値>; KY=<Task 7 の値>
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/submit_ansatz.jl \
  --nu 1/3 --k13 $KX $KY --seeds 1001:1001 --np 16 2>&1 | tail -20
grep nu13 playground_nozomi/cb_nu12_boson/runs_ansatz/_roster.dat
```
Expected: 12 run が順に走る(構成ごとに 1 seed)。E_tail が ED の
基底エネルギー(Task 8 Step 1 の `E0`)と**同じ桁**。極端に違ったら止める。

- [ ] **Step 3: 残りの seed を投入する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
nohup ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia \
  --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/submit_ansatz.jl \
  --nu 1/3 --k13 $KX $KY \
  > playground_nozomi/cb_nu12_boson/runs_ansatz_nu13_submit.log 2>&1 &
echo $!
```
Expected: 済みの 12 run はスキップされ、残り 108 run が走る。

- [ ] **Step 4: 進捗を確認する**

Run:
```bash
wc -l /home/nozomihigashino/Julia-mVMC/playground_nozomi/cb_nu12_boson/runs_ansatz/_roster.dat
tail -3 /home/nozomihigashino/Julia-mVMC/playground_nozomi/cb_nu12_boson/runs_ansatz_nu13_submit.log
```
Expected: roster が 200 行 + ヘッダに向かって伸びる。

---

## Task 11: 集計と Chern 判定(`analyze_ansatz.jl`)

**Files:**
- Create: `playground_nozomi/cb_nu12_boson/scripts/analyze_ansatz.jl`
- Modify: `tools/parton_band_chern.jl`(`EXEY` に `ef9` / `xexet3` を追加、usage 文字列 2 箇所)

**Interfaces:**
- Consumes: `runs_ansatz/_roster.dat`、`tools/parton_band_chern.jl`
  (CLI: `<out_dir> --nux N --nuy N --ansatz X [--grid 32]`、環境は `--project=tools`)
- Produces: `runs_ansatz/_summary.md`

- [ ] **Step 1: Chern ツールに ν=1/3 のアンザッツを登録する**

**プレフライトで判明(Ruling 6)**: `tools/parton_band_chern.jl` の CLI は
`<out_dir> --nux N [--nuy N] --ansatz ef4|xexet2 [--grid 32] [--out out.pdf]` で、
実行環境は **`--project=tools`**(Plots を隔離した環境。`MVMCOptimizers.jl` ではない)。
アンザッツ → (ex, ey) の辞書 `EXEY` に **`ef4` と `xexet2` しか無い**ので、
ν=1/3 の `ef9` / `xexet3` を足さないと `未対応のアンザッツ` で落ちる。

`tools/parton_band_chern.jl` の `EXEY` を探して 2 行足す:

Run:
```bash
grep -n 'EXEY' /home/nozomihigashino/Julia-mVMC/tools/parton_band_chern.jl
```
Expected: `const EXEY = Dict(:ef4 => (2, 2), :xexet2 => (2, 1))` 相当の定義が見つかる。
そこへ `:ef9 => (3, 3), :xexet3 => (3, 1)` を足し、使い方の文字列 2 箇所
(冒頭コメントと `usage:` のエラー文)の `ef4|xexet2` を
`ef4|xexet2|ef9|xexet3` に直す。

- [ ] **Step 2: 6×3・F=3 で `parton_band_chern.jl` が動くことを先に確かめる**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
RID=$(grep nu13 playground_nozomi/cb_nu12_boson/runs_ansatz/_roster.dat | head -1 | awk '{print $1}')
ANS=$(grep nu13 playground_nozomi/cb_nu12_boson/runs_ansatz/_roster.dat | head -1 | awk '{print $3}')
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=tools \
  tools/parton_band_chern.jl playground_nozomi/cb_nu12_boson/runs_ansatz/$RID/stage1_out \
  --nux 6 --nuy 3 --ansatz $ANS 2>&1 | tail -20
```
Expected: 3 フレーバー分の Chern 数(`C_occ`)が出る。落ちたら **nx ≠ ny か F=3 の
どちらが原因かを切り分けて** ツールを直す(`tools/parton_band_chern.jl:78` の
`n1, n2 = div(o.nux, o.ex), div(o.nuy, o.ey)` は nx≠ny に対応済みなので、
落ちるとすれば別の場所)。

- [ ] **Step 3: 集計スクリプトを書く**

`playground_nozomi/cb_nu12_boson/scripts/analyze_ansatz.jl`:

```julia
#=
アンザッツ比較キャンペーンの集計(2026-08-25)

    julia --project=MVMCOptimizers.jl scripts/analyze_ansatz.jl [--chern]

roster を読んで構成別に E_best / 収束数 / 当たり率 / ΔE(対 ED)を表にする。
--chern を付けると各構成の最良 run に tools/parton_band_chern.jl を当てる
(1 構成あたり数十秒かかるので既定は off)。
出力: runs_ansatz/_summary.md
=#
using Printf, Statistics

const SCRIPT_DIR = @__DIR__
const PROJ_DIR = dirname(SCRIPT_DIR)
const JMVMC_ROOT = dirname(dirname(PROJ_DIR))
const JULIA_BIN = joinpath(homedir(),
    ".julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia")
const CHERN_TOOL = joinpath(JMVMC_ROOT, "tools", "parton_band_chern.jl")
const TOOLS_ENV = joinpath(JMVMC_ROOT, "tools")   # Plots を隔離した環境
const RUNS = joinpath(PROJ_DIR, "runs_ansatz")

# ED 参照値。ν=1/3 は Task 8 で実測した値に置き換えること。
const E_ED = Dict("nu12" => -16.304913354429445,
                  "nu13" => NaN)      # ← 6×3 の実測値を入れる

"roster を読む"
function read_roster(path)
    rows = NamedTuple[]
    for line in eachline(path)
        startswith(line, "#") && continue
        f = split(line)
        length(f) >= 13 || continue
        push!(rows, (rid = f[1], nu = f[2], ansatz = f[3], flavor = f[4],
                     graph = f[5], seed = parse(Int, f[6]),
                     wall = parse(Float64, f[8]), status = parse(Int, f[9]),
                     E = f[10] == "-" ? NaN : parse(Float64, f[10]),
                     gap = f[11] == "-" ? NaN : parse(Float64, f[11]),
                     sdiag = f[12] == "-" ? NaN : parse(Float64, f[12]),
                     conv = f[13] == "true"))
    end
    return rows
end

function main(argv)
    do_chern = "--chern" in argv
    rows = read_roster(joinpath(RUNS, "_roster.dat"))
    isempty(rows) && error("roster が空です: $(joinpath(RUNS, "_roster.dat"))")

    keys_ = unique([(r.nu, r.ansatz, r.flavor, r.graph) for r in rows])
    io = IOBuffer()
    println(io, "# アンザッツ比較 集計(", length(rows), " run)\n")
    println(io, "| ν | アンザッツ | フレーバー | グラフ | run | 収束 | E_best | ΔE | 当たり率 | 中央 wall |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")

    best_run = Dict{NTuple{4,String},String}()
    for k in sort(keys_)
        sub = [r for r in rows
               if (r.nu, r.ansatz, r.flavor, r.graph) == k && r.status == 0]
        conv = [r for r in sub if r.conv && !isnan(r.E)]
        if isempty(conv)
            @printf(io, "| %s | %s | %s | %s | %d | 0 | - | - | - | - |\n",
                    k..., length(sub))
            continue
        end
        eb, ib = findmin([r.E for r in conv])
        hits = count(r -> r.E <= eb + 0.01, conv)
        ed = get(E_ED, k[1], NaN)
        best_run[k] = conv[ib].rid
        @printf(io, "| %s | %s | %s | %s | %d | %d | %.6f | %s | %d/%d | %.0f s |\n",
                k..., length(sub), length(conv), eb,
                isnan(ed) ? "-" : @sprintf("%+.6f", eb - ed),
                hits, length(conv), median([r.wall for r in conv]))
    end

    if do_chern
        println(io, "\n## 最良 run の Chern 数\n")
        println(io, "| 構成 | run_id | 出力 |")
        println(io, "|---|---|---|")
        for (k, rid) in sort(collect(best_run))
            out = joinpath(RUNS, rid, "stage1_out")
            nux, nuy = k[1] == "nu12" ? (4, 4) : (6, 3)
            txt = try
                # Plots は tools 環境にしか無い。--nux/--nuy/--ansatz は必須引数
                read(`$JULIA_BIN --project=$TOOLS_ENV $CHERN_TOOL $out
                      --nux $nux --nuy $nuy --ansatz $(k[2])`, String)
            catch e
                "(失敗: $e)"
            end
            # C_occ の行だけ抜く
            lines = [l for l in split(txt, '\n') if occursin("C_occ", l) || occursin("flavor", l)]
            @printf(io, "| %s | %s | %s |\n", join(k, "/"), rid,
                    replace(join(lines, " ; "), "|" => "\\|"))
        end
    end

    s = String(take!(io))
    print(s)
    write(joinpath(RUNS, "_summary.md"), s)
    println("\n-> ", joinpath(RUNS, "_summary.md"))
end

main(ARGS)
```

- [ ] **Step 4: ν=1/2 の分だけで動かす**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/analyze_ansatz.jl 2>&1 | tail -20
```
Expected: 8 構成の表が出る。ΔE が正(VMC は変分上界なので ED より高い)。
**ΔE が負なら重大なバグ**(ハミルトニアンか ED 参照が違う)なので止めて調べる。

- [ ] **Step 5: Chern 判定込みで動かす**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/analyze_ansatz.jl --chern 2>&1 | tail -30
```
Expected: 各構成の最良 run について、フレーバーごとの C_occ が出る。
FCI = 全フレーバーで |C| = 1 で符号が揃う。

- [ ] **Step 6: ν=1/3 の ED 値を埋めて全体を集計する**

Task 8 Step 1 の実測値を `analyze_ansatz.jl` の `E_ED["nu13"]` に入れてから:

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=MVMCOptimizers.jl \
  playground_nozomi/cb_nu12_boson/scripts/analyze_ansatz.jl --chern 2>&1 | tail -40
cat playground_nozomi/cb_nu12_boson/runs_ansatz/_summary.md
```
Expected: 20 構成の表が完成する。

---

## Task 12: 結果を PROJECT.md と CLAUDE.md に記録する

**Files:**
- Modify: `playground_nozomi/cb_nu12_boson/PROJECT.md`(メモ節に追記)
- Modify: `.claude/CLAUDE.md`(現在の状況と TODO を更新)

**Interfaces:**
- Consumes: Task 11 の `_summary.md`、Task 7 の K、Task 9/10 の所要時間実測

- [ ] **Step 1: PROJECT.md にメモを追記する**

`playground_nozomi/cb_nu12_boson/PROJECT.md` の「## セクション」の**直前**
(既存メモ群の末尾)に追加する。内容:

- 見出し `### 2026-08-25 FCI アンザッツ比較キャンペーン フェーズ 1(ED 系)`
- 設計文書へのリンク(`../../docs/superpowers/specs/2026-08-25-fci-ansatz-survey-design.md`)
- 6×3 ED の投入と結果(E0 / 3 重項の分裂 / 4 番目までのギャップ / 基底の K)
- `_summary.md` の表(そのまま貼る)
- **結論**: ν ごとの最良アンザッツ、ΔE、Chern 判定の可否
- 実測の所要時間(`model` / `full` の sec/step 比、総 wall)
- 開いた問題(K が ±1 から外れた場合の同時対角化、`full` の当たり率が低い場合の
  ウォームスタート、など実際に起きたものだけ)

- [ ] **Step 2: `.claude/CLAUDE.md` の「現在の状況」を更新する**

- 冒頭の日付を `(2026-08-25 更新)` に
- アンザッツ比較キャンペーンの結果を 3〜4 行で(最良アンザッツと ΔE)
- TODO リストに `- [x] FCI アンザッツ比較 フェーズ 1(ED 系)完了` を追加し、
  `- [ ] フェーズ 2(大規模系)の設計` と `- [ ] 方針 2(V スキャン)` を積む
- 「作業場所」の表に `runs_ansatz/` を追加

- [ ] **Step 3: P 層テストが全部緑であることを最終確認する**

Run:
```bash
cd /home/nozomihigashino/Julia-mVMC
OPENBLAS_NUM_THREADS=1 ~/.julia/juliaup/julia-1.10.2+0.x64.linux.gnu/bin/julia --project=@. test/physics/runtests.jl 2>&1 | tail -20
```
Expected: Fail 0 / Error 0。

- [ ] **Step 4: ステージングして差分を報告する**

```bash
cd /home/nozomihigashino/Julia-mVMC
git add .claude/CLAUDE.md docs/superpowers/
git status --short
git diff --cached --stat
```
Expected: `test/physics/*`(Task 2/3/8)、`.claude/CLAUDE.md`、`docs/superpowers/*`
がステージされている。`playground_nozomi/` は git 管理外なので出ない。
**コミットはユーザーの明示依頼を待つ。**

---

## 実行順のまとめ

```
Task 1(ED 投入、~1.5 h 待ち)
   ├─ 並行 ─> Task 2(flavor_groups)-> Task 3(graph=:full)
   │           -> Task 4(gen_def)-> Task 5(chain)-> Task 6(submit)
   │           -> Task 9(ν=1/2 投入、~15 h)
   └─ ED 完了 ─> Task 7(K 測定)-> Task 8(P0-d 登録)
                    -> Task 10(ν=1/3 投入、~25〜35 h)
                       -> Task 11(集計 + Chern)-> Task 12(記録)
```
