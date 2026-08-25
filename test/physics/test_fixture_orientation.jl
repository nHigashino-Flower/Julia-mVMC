"""
fixture の向き正準化の回帰テスト(2026-08-18 のバグ修正)
--- parton-mode (fork addition) ---

`parton_fixture` は無向ボンドを `i < j` で正規化し、idx クラスの代表を
`min(ki, kj)` で選ぶ。係数 t をリスト向きのまま書くと、並進で端点の大小が
入れ替わったコピーが conj(t) 側で載り、同一 idx に α·t と α·conj(t) が同居して
**複素 α で拡大セル並進が破れる**(8×8 ef4 の実 run で y 方向残差 0.6〜0.8 を実測)。
α = 1(実数)では h.c. と合流して同一の H になるため、初期ハミルトニアンだけを
見る検査では発見できない — このテストは複素 α で H を組んで不変性を直接見る。

組み立ては本体 `parton_build_mf_templates!` / `parton_update_orbitals!` と同じ
「リストされた向きで H[site1, site2] += α·value、ホッピングは h.c. を追加」。
"""

using LinearAlgebra

"fixture 出力から複素 α で H_MF を組む(本体の組み立て規約と同一)。"
function _assemble_mf(fx, α::Vector{ComplexF64}, nsite::Int, nflavor::Int)
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

"基本セル (sx, sy) 並進のサイト置換(1-based)。"
function _cell_perm(nx::Int, ny::Int, sx::Int, sy::Int)
    nsite = 2 * nx * ny
    perm = Vector{Int}(undef, nsite)
    for s = 0:(nsite - 1)
        x, y = cb_site_to_xy(s, nx)
        perm[s + 1] = cb_xy_to_site(mod(x + 2sx, 2nx), mod(y + 2sy, 2ny), nx) + 1
    end
    return perm
end

@testset "orientation canonicalisation (complex α invariance)" begin
    nx = ny = 4
    nflavor = 2
    nsite = 2 * nx * ny
    for (ex, ey) in ((2, 2), (2, 1))
        fx = parton_fixture(nx, ny, nflavor, ex, ey;
                            u_mf = 0.0, idx_mode = :orbit_flavor)
        # 決定的な複素 α(位相も絶対値も idx ごとに変える)
        α = ComplexF64[(1.0 + 0.1k) * exp(im * (0.7k + 0.3)) for k = 0:(fx.n_idx - 1)]
        H = _assemble_mf(fx, α, nsite, nflavor)

        for f = 1:nflavor
            @test norm(H[f] - H[f]') < 1e-12 * norm(H[f])   # エルミート性
            # 拡大セル並進で厳密に不変(修正前は y 方向が O(1) で破れる)
            for (sx, sy) in ((ex, 0), (0, ey))
                perm = _cell_perm(nx, ny, sx, sy)
                @test norm(H[f][perm, perm] - H[f]) < 1e-12 * norm(H[f])
            end
            # 検出力の確認: 破っているはずの基本セル並進 (1,0) は不変にならない
            # (ex = 2 のクラスは ucx の偶奇を区別する)
            perm1 = _cell_perm(nx, ny, 1, 0)
            @test norm(H[f][perm1, perm1] - H[f]) > 0.1 * norm(H[f])
        end
    end
end

@testset "α = 1 reproduces the checkerboard model" begin
    nx = ny = 4
    nflavor = 2
    nsite = 2 * nx * ny
    Hcb = cb_onebody(nx, ny)
    for (ex, ey) in ((2, 2), (2, 1))
        fx = parton_fixture(nx, ny, nflavor, ex, ey;
                            u_mf = 0.0, idx_mode = :orbit_flavor)
        H = _assemble_mf(fx, ones(ComplexF64, fx.n_idx), nsite, nflavor)
        for f = 1:nflavor
            @test norm(H[f] - Hcb) < 1e-12 * norm(Hcb)
        end
    end
end
