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

    # --- 群数が F と違う / 範囲外 / :bond_flavor 併用 はそれぞれ単独でエラー ---
    @test_throws ErrorException parton_fixture(4, 4, 2, 2, 2;
                                               flavor_groups = [0, 1, 2])   # 長さ違い
    @test_throws ErrorException parton_fixture(4, 4, 2, 2, 2;
                                               flavor_groups = [0, 2])      # 長さは合うが範囲外
    @test_throws ErrorException parton_fixture(4, 4, 2, 2, 2;
                                               idx_mode = :bond_flavor,
                                               flavor_groups = [0, 1])      # 併用禁止

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

@testset "graph = :full" begin
    # --- 4×4 ef4、F=2、全独立 ---
    nx, ny, F, ex, ey = 4, 4, 2, 2, 2
    nsite = 2 * nx * ny
    full = parton_fixture(nx, ny, F, ex, ey; graph = :full,
                          flavor_groups = collect(0:(F - 1)))

    # (a) 無向サイト対を、並進で端点が入れ替わるものを除いてすべて 1 回ずつ覆う
    #     + オンサイト nsite 個
    pairs = Set{Tuple{Int,Int}}()
    diag = Set{Int}()
    for (s1, f1, s2, f2, _) in full.pmftrans
        f1 == 0 || continue
        s1 == s2 ? push!(diag, s1) : push!(pairs, minmax(s1, s2))
    end
    swapped = cb_translation_swapped_pairs(nx, ny, ex, ey)
    @test length(diag) == nsite
    @test length(pairs) == div(nsite * (nsite - 1), 2) - length(swapped)
    @test full.n_swapped_dropped == length(swapped)
    @test !isempty(swapped)                     # このサイズでは実際に落ちる
    @test isempty(intersect(pairs, swapped))    # 落としたものが残っていない

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
