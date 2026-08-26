"""
K 層: Kapit-Mueller 模型のパートン平均場アンザッツ
--- parton-mode (fork addition) ---

設計書 `docs/superpowers/specs/2026-08-26-kapit-mueller-parton-design.md` §4 の
K0〜K5 と P1 / P2 をここで固定する。

| 検査 | 内容 |
|---|---|
| K0 | 1 体 H のエルミート性 |
| K1 | プラケットフラックスが一様 φ |
| K2 | 厳密並進群 = `⟨T_x^q, T_y⟩`(物理)/ `⟨T_y⟩`(平均場) |
| K3 | 平均場の最低バンドが N でちょうど充填・ギャップあり・平坦 |
| K5 | フレーバー積 `G(φ/F)^F == G(φ)`(Landau ゲージ固有) |
| P1 | ED 実装(別リポジトリ・独立実装)の 1 体 H と 1e-10 一致 |
| P2 | QP 並進が物理 H を保ち、位数が F |
| F  | fixture が組む pmftrans が平均場 H を再現し、自己交換クラスを凍結する |

Chern 数(K4)は ED 側 `test_13_kapitmueller.jl` で検査済み(全系 C = −1)。
ここではツイスト格子を張らない(重いので二重に持たない)。
"""

using LinearAlgebra

const KM_TEST_CASES = [
    # (tag, Lx, Ly, φ, F, N, 全行列ダンプあり)
    ("km_nu12_4x4", 4, 4, 1 // 2, 2, 4, true),
    ("km_nu12_4x6", 4, 6, 1 // 2, 2, 6, false),
    ("km_nu13_9x4", 9, 4, 1 // 3, 3, 4, true),
    ("km_nu13_9x5", 9, 5, 1 // 3, 3, 5, false),
]

"`KMLatticeModel` から 1 体ハミルトニアン H[j+1, i+1] = t(i→j) を組む。"
function km_onebody_matrix(m::KMLatticeModel, φ::Float64)
    n = pl_nsite(m)
    return [km_pair_hopping(m, j, i, φ) for i = 0:(n - 1), j = 0:(n - 1)]
end

"単位正方 (x,y) のリンク位相の巡回和 / 2π(mod 1)。"
function km_plaquette_flux(H, m::KMLatticeModel, x::Int, y::Int)
    s1 = pl_xy_to_site(m, x, y);         s2 = pl_xy_to_site(m, x + 1, y)
    s3 = pl_xy_to_site(m, x + 1, y + 1); s4 = pl_xy_to_site(m, x, y + 1)
    z = H[s2 + 1, s1 + 1] * H[s3 + 1, s2 + 1] * H[s4 + 1, s3 + 1] * H[s1 + 1, s4 + 1]
    return mod(angle(z) / (2π), 1.0)
end

"H を厳密に保つ並進 (tx, ty) を全部列挙する。"
function km_exact_translations(H, m::KMLatticeModel; tol = 1e-11)
    lx, ly = pl_grid(m)
    n = pl_nsite(m)
    ok = Tuple{Int,Int}[]
    for tx = 0:(lx - 1), ty = 0:(ly - 1)
        p = pl_shift_perm(m, tx, ty)
        maximum(abs.([H[p[i] + 1, p[j] + 1] for i = 1:n, j = 1:n] - H)) < tol &&
            push!(ok, (tx, ty))
    end
    return ok
end

"ED ダンプ(`ed_dump/ed_onebody_<tag>.dat`)を読む。"
function read_km_dump(path::AbstractString)
    blocks = Dict{Tuple{String,Float64},NamedTuple}()
    meta = Dict{String,String}()
    open(path) do io
        cur = nothing
        H = Dict{Tuple{Int,Int},ComplexF64}()
        ev = Float64[]
        for line in eachline(io)
            startswith(line, "#") && continue
            f = split(line)
            isempty(f) && continue
            if f[1] == "BLOCK"
                cur = (String(f[2]), parse(Float64, f[4]), parse(Float64, f[6]))
                H = Dict{Tuple{Int,Int},ComplexF64}()
                ev = Float64[]
            elseif f[1] == "ONEBODY"
                H[(parse(Int, f[2]), parse(Int, f[3]))] =
                    complex(parse(Float64, f[4]), parse(Float64, f[5]))
            elseif f[1] == "EIGVAL"
                push!(ev, parse(Float64, f[2]))
            elseif f[1] == "ENDBLOCK"
                blocks[(cur[1], cur[2])] = (flux = cur[3], H = H, ev = ev)
            elseif length(f) == 2
                meta[f[1]] = f[2]
            end
        end
    end
    return meta, blocks
end

@testset "K Kapit-Mueller 模型" begin
    dumpdir = joinpath(@__DIR__, "ed_dump")

    @testset "K0/K1/K2/K3/K5 $(tag)" for (tag, Lx, Ly, φr, F, N, _) in KM_TEST_CASES
        φ = float(φr)
        q = round(Int, 1 / φ)
        m = KMLatticeModel(Lx, Ly, φ, F; hopmax = 8.0)
        ex, ey = km_default_cell(m)

        for (kind, ff, nfill) in (("物理", φ, round(Int, φ * Lx * Ly)),
                                  ("平均場", φ / F, N))
            H = km_onebody_matrix(m, ff)
            # K0
            @test maximum(abs.(H - H')) < 1e-12
            # K1
            fl = [km_plaquette_flux(H, m, x, y) for x = 0:(Lx - 1), y = 0:(Ly - 1)]
            @test maximum(abs.(fl .- ff)) < 1e-3
            # K3(平均場のみ): バンドがちょうど N 状態・平坦・ギャップあり
            ev = eigvals(Hermitian((H + H') / 2))
            if kind == "平均場"
                @test round(Int, ff * Lx * Ly) == N
                @test ev[N] - ev[1] < 1e-8              # 厳密平坦
                @test ev[N + 1] - ev[N] > 0.1
            end
        end

        # K2: 物理は ⟨T_x^q, T_y⟩、平均場は ⟨T_y⟩
        Hp = km_onebody_matrix(m, φ)
        Hm = km_onebody_matrix(m, φ / F)
        @test sort(km_exact_translations(Hp, m)) ==
              sort([(a, b) for a = 0:(Lx - 1) for b = 0:(Ly - 1) if a % q == 0])
        @test sort(km_exact_translations(Hm, m)) == sort([(0, b) for b = 0:(Ly - 1)])
        # 契約が実測と合っていること
        @test sort(pl_physical_shifts(m)) == sort(km_exact_translations(Hp, m))
        @test sort(pl_mf_shifts(m, ex, ey)) == sort(km_exact_translations(Hm, m))

        # K5: フレーバー積
        @test maximum(abs(km_gauge(x, dx, dy, φ / F)^F - km_gauge(x, dx, dy, φ))
                      for x = 0:(Lx - 1), dx = -2:2, dy = -2:2) < 1e-12
    end

    @testset "P1 ED 実装との 1 体 H 一致 $(tag)" for (tag, Lx, Ly, φr, F, N, full) in
                                                     KM_TEST_CASES
        meta, blocks = read_km_dump(joinpath(dumpdir, "ed_onebody_$(tag).dat"))
        @test parse(Int, meta["Lx"]) == Lx
        @test parse(Int, meta["Ly"]) == Ly
        @test parse(Int, meta["F"]) == F
        for hm in (2.0, 8.0)
            m = KMLatticeModel(Lx, Ly, float(φr), F; hopmax = hm)
            for (kind, ff) in (("phys", float(φr)), ("mf", float(φr) / F))
                blk = blocks[(kind, hm)]
                H = km_onebody_matrix(m, ff)
                # スペクトル一致(全系)
                ev = eigvals(Hermitian((H + H') / 2))
                @test maximum(abs.(ev - blk.ev)) < 1e-10
                # 全行列一致(ダンプがある系のみ)
                if full
                    @test length(blk.H) == count(x -> abs(x) > 1e-14, H)
                    @test maximum(abs(H[r + 1, c + 1] - v) for ((r, c), v) in blk.H) < 1e-10
                end
            end
        end
    end

    @testset "P2 QP 並進 $(tag)" for (tag, Lx, Ly, φr, F, N, _) in KM_TEST_CASES
        m = KMLatticeModel(Lx, Ly, float(φr), F; hopmax = 8.0)
        ex, ey = km_default_cell(m)
        qp = pl_qp_shifts(m, ex, ey)
        # 位数 = F(= 期待されるトーラス位相縮退)
        @test length(qp) == F
        # 代表の y 成分は 0(平均場が T_y を保つので剰余に出ない)
        @test all(t[2] == 0 for t in qp)
        # QP 並進はどれも物理 H を厳密に保つ
        Hp = km_onebody_matrix(m, float(φr))
        n = pl_nsite(m)
        for (tx, ty) in qp
            p = pl_shift_perm(m, tx, ty)
            @test maximum(abs.([Hp[p[i] + 1, p[j] + 1] for i = 1:n, j = 1:n] - Hp)) < 1e-11
        end
    end

    @testset "F fixture の健全性 $(tag)" for (tag, Lx, Ly, φr, F, N, _) in KM_TEST_CASES
        m = KMLatticeModel(Lx, Ly, float(φr), F; hopmax = 8.0)
        ex, ey = km_default_cell(m)
        fx = parton_fixture(m, F, ex, ey; idx_mode = :orbit)
        nsite = pl_nsite(m)

        # pmftrans を組み立て直したものが φ/F の平均場 H と一致する(α = 1)
        H = zeros(ComplexF64, nsite, nsite)
        for (s1, f1, s2, f2, v) in fx.pmftrans
            f1 == 0 || continue
            if s1 == s2
                H[s1 + 1, s2 + 1] += v
            else
                H[s1 + 1, s2 + 1] += v
                H[s2 + 1, s1 + 1] += conj(v)
            end
        end
        Href = km_onebody_matrix(m, float(φr) / F)
        @test maximum(abs.(H - Href)) < 1e-10
        # オンサイト項(トーラス自己像)が入っていること
        @test abs(H[1, 1] - Href[1, 1]) < 1e-14

        # physhop は物理 t_ij(片方向)。組み直して物理 H に一致すること
        Hp = zeros(ComplexF64, nsite, nsite)
        for (i, j, v) in fx.physhop
            Hp[j + 1, i + 1] += v
            Hp[i + 1, j + 1] += conj(v)
        end
        for (i, e0) in fx.phys_onsite
            Hp[i + 1, i + 1] += e0
        end
        @test maximum(abs.(Hp - km_onebody_matrix(m, float(φr)))) < 1e-10
        # オンサイト項は **物理の並進群で不変**(KM の自己像は Landau 位相が x に
        # 依存するので、x 方向には周期 q でしか一定にならない)。
        e0s = last.(fx.phys_onsite)
        for (tx, ty) in pl_physical_shifts(m)
            pm = pl_shift_perm(m, tx, ty)
            @test maximum(abs(e0s[pm[k] + 1] - e0s[k]) for k = 1:nsite) < 1e-13
        end

        # **任意の α で H_MF が T_y 共変であること**。これが idx クラス分けの本質で、
        # 向きの正準化・自己交換クラスの凍結が効いているかを直接確かめる唯一の検査。
        # (α = 1 では h.c. と合流して破れが見えないので、必ず複素で試す)
        alphas = ComplexF64[cis(0.7 * k) * (1 + 0.1 * k) for k = 1:fx.n_idx]
        for k in fx.frozen_idx
            alphas[k + 1] = ComplexF64(1)     # 凍結クラスは α = 1 のまま動かない
        end
        Ha = zeros(ComplexF64, nsite, nsite)
        for ((s1, f1, s2, _, v), (_, _, _, _, idx, _)) in zip(fx.pmftrans, fx.pmfpara)
            f1 == 0 || continue
            a = alphas[idx + 1]
            if s1 == s2
                Ha[s1 + 1, s2 + 1] += real(a) * v
            else
                Ha[s1 + 1, s2 + 1] += a * v
                Ha[s2 + 1, s1 + 1] += conj(a * v)
            end
        end
        @test maximum(abs.(Ha - Ha')) < 1e-12
        for (tx, ty) in pl_mf_shifts(m, ex, ey)
            pm = pl_shift_perm(m, tx, ty)
            @test maximum(abs.([Ha[pm[i] + 1, pm[j] + 1] for i = 1:nsite, j = 1:nsite] - Ha)) < 1e-10
        end

        # 自己交換ペアは「変位 (0, Ly/2)」で発生する → Ly が偶数のときだけ
        swapped = pl_swapped_pairs(m, ex, ey)
        @test iseven(Ly) == !isempty(swapped)
        @test isempty(swapped) == isempty(fx.frozen_idx)

        # α を実数に縛る必要があるボンドは、係数が実数になっているはず
        for (i, j) in swapped
            @test abs(imag(km_pair_hopping(m, i, j, float(φr) / F))) < 1e-12
        end
    end
end
