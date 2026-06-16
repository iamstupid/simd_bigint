// Scalar model: TRUNCATED mixed-radix 3*2^k NTT (radix-3 on top,
// binary van-der-Hoeven TFT below), for transform sizes N = 3L
// (L = 2^k vectors) truncated to z in (2L, 3L].
//
// THE POINT (ladder dispatch): sizes {2^j} U {3*2^j} cover every
// demand within a factor 1.5, and the coverage ranges are chosen so
// the radix-3 level is ALWAYS FULL (z > 2L => all three children
// demanded): no truncated radix-3 variants exist anywhere; truncation
// lives entirely inside binary child 2, where the standard vdH
// machinery applies unchanged.
//
// NEW MATH (validated here) = the inverse-side radix-3 seam. With
// children moduli (y^L - zeta^t), zeta^3 = 1:
//     b^t_i = a_i + zeta^t a_{i+L} + zeta^{2t} a_{i+2L}
// After FULL inverses of children 0,1 and with parent tails zero
// (a_{i+2L} = 0 for i >= z2 = z - 2L), a 2x2 Vandermonde solve gives
// the demanded outputs AND child 2's coefficient tails:
//     a_{i+L} = (b^0_i - b^1_i) / (1 - zeta)
//     a_i     =  b^0_i - a_{i+L}
//     b^2_i   =  a_i + zeta^2 a_{i+L}            (i in [z2, L))
// then child 2 runs the standard truncated binary inverse with those
// tails, and a plain 3-point inverse combine finishes i < z2.
// (Scaling: subtree inverses return L*b^t; the solve and the /3
// combine keep everything at L*a, one global /L at the end.)
//
// Roots are passed as EXPONENTS of Omega (order F = 3*2^J), so no
// table-layout questions arise: node modulus y^m - Omega^er, twiddle
// Omega^{er/2}, children er/2 and er/2 + F/2. Requires 3 | p-1: of
// the production set only 918552577 (219 = 3*73) and 943718401
// (225 = 3^2 5^2) qualify -- a production 3*2^k engine needs a new
// prime set (noted, out of scope here).
//
// Gates:
//   1. eval: full mixed forward == direct evaluation at the recorded
//      leaf moduli roots
//   2. round trip: mixed fwd(trunc) -> inv(trunc) -> /L == identity,
//      every z2 exhaustively for small k
//   3. product: full pipeline (fwd a, fwd b, conv4 twists, mixed
//      inverse) == schoolbook polynomial product mod p
//   4. ladder: sweep every tv, dispatch {2^j, 3*2^j}, verify product,
//      and compare mults/coeff smoothness vs binary-only dispatch
// Build: clang -O2 -std=c11 p30_tft3_model_test.c -lm -o /tmp/tft3
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef uint32_t u32;

#define W 4                       /* lanes per vector (y = x^W) */

static u64 rngs = 42;
static u64 xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }

static u64 MCNT;                  /* modular mult counter (gate 4) */

static u64 addm(u64 a, u64 b, u64 p){ u64 s = a + b; return s >= p ? s - p : s; }
static u64 subm(u64 a, u64 b, u64 p){ return a >= b ? a - b : a + p - b; }
static u64 mulm(u64 a, u64 b, u64 p){ ++MCNT; return (u64)((u128)a * b % p); }
static u64 powm(u64 a, u64 e, u64 p){
    u64 r = 1;
    for(; e; e >>= 1, a = mulm(a, a, p))
        if(e & 1) r = mulm(r, a, p);
    return r;
}
static u64 invm(u64 a, u64 p){ return powm(a, p - 2, p); }

static u64 primitive_root(u64 p){
    u64 f[16]; int nf = 0;
    u64 n = p - 1;
    for(u64 d = 2; d * d <= n; d += (d == 2 ? 1 : 2))
        if(n % d == 0){ f[nf++] = d; while(n % d == 0) n /= d; }
    if(n > 1) f[nf++] = n;
    for(u64 g = 2; ; ++g){
        int ok = 1;
        for(int i = 0; i < nf && ok; ++i)
            if(powm(g, (p - 1) / f[i], p) == 1) ok = 0;
        if(ok) return g;
    }
}

/* ---- transform context: Omega of order F = 3*2^J ------------------ */
typedef struct {
    u64 p, g;
    u64 F, H, EZ;                 /* F = 3*2^J; H = F/2 (-1); EZ = F/3 */
    u64 Om;
    u64 inv2, inv3;
} mctx;

static void mctx_build(mctx *c, u64 p, int J){
    c->p = p;
    c->g = primitive_root(p);
    c->F = (u64)3 << J;
    if((p - 1) % c->F){ fprintf(stderr, "prime %llu lacks 3*2^%d\n",
                                (unsigned long long)p, J); exit(1); }
    c->H = c->F >> 1;
    c->EZ = (u64)1 << J;
    c->Om = powm(c->g, (p - 1) / c->F, p);
    c->inv2 = (p + 1) >> 1;
    c->inv3 = invm(3, p);
}
static u64 ompow(const mctx *c, u64 e){ return powm(c->Om, e % c->F, c->p); }

typedef struct { u64 l[W]; } vec_t;

/* ---- binary truncated forward, node (B[0..m), modulus y^m - Om^er),
 * output trunc zo; leaf records its modulus exponent into tw[] ------ */
static void tfwd(const mctx *c, vec_t *B, u64 m, u64 er, u64 zo,
                 u64 *tw, u64 pos){
    if(m == 1){ tw[pos] = er % c->F; return; }
    u64 h = m >> 1, p = c->p;
    u64 w = ompow(c, er >> 1);
    if(zo <= h){
        for(u64 i = 0; i < h; ++i)
            for(int l = 0; l < W; ++l)
                B[i].l[l] = addm(B[i].l[l], mulm(w, B[i + h].l[l], p), p);
        tfwd(c, B, h, er >> 1, zo, tw, pos);
        return;
    }
    for(u64 i = 0; i < h; ++i)
        for(int l = 0; l < W; ++l){
            u64 a = B[i].l[l], wb = mulm(w, B[i + h].l[l], p);
            B[i].l[l]     = addm(a, wb, p);
            B[i + h].l[l] = subm(a, wb, p);
        }
    tfwd(c, B, h, er >> 1, h, tw, pos);
    tfwd(c, B + h, h, (er >> 1) + c->H, zo - h, tw, pos + h);
}

/* ---- binary truncated inverse (subtree-scaled: exit B[0..z) = m*u);
 * entry B[0..z) spectral, B[z..m) = m*u tails ----------------------- */
static void tinv(const mctx *c, vec_t *B, u64 m, u64 er, u64 z){
    if(m == 1 || z == 0) return;
    u64 h = m >> 1, p = c->p;
    u64 w = ompow(c, er >> 1);
    if(z < h){
        for(u64 i = z; i < h; ++i)
            for(int l = 0; l < W; ++l)
                B[i].l[l] = mulm(addm(B[i].l[l],
                                      mulm(w, B[i + h].l[l], p), p),
                                 c->inv2, p);
        tinv(c, B, h, er >> 1, z);
        for(u64 i = 0; i < z; ++i)
            for(int l = 0; l < W; ++l)
                B[i].l[l] = subm(addm(B[i].l[l], B[i].l[l], p),
                                 mulm(w, B[i + h].l[l], p), p);
        return;
    }
    u64 zr = z - h;
    tinv(c, B, h, er >> 1, h);
    for(u64 i = zr; i < h; ++i)
        for(int l = 0; l < W; ++l){
            u64 t = subm(B[i].l[l], mulm(w, B[i + h].l[l], p), p);
            B[i].l[l] = addm(B[i].l[l], t, p);
            B[i + h].l[l] = t;
        }
    tinv(c, B + h, h, (er >> 1) + c->H, zr);
    u64 wi = invm(w, p);
    for(u64 i = 0; i < zr; ++i)
        for(int l = 0; l < W; ++l){
            u64 s = addm(B[i].l[l], B[i + h].l[l], p);
            u64 d = subm(B[i].l[l], B[i + h].l[l], p);
            B[i].l[l] = s;
            B[i + h].l[l] = mulm(d, wi, p);
        }
}

/* ---- mixed 3L forward, z in (2L, 3L]: full radix-3 pass + children */
static void mfwd(const mctx *c, vec_t *B, u64 L, u64 z, u64 *tw){
    u64 p = c->p;
    u64 zt  = ompow(c, c->EZ);            /* zeta   */
    u64 zt2 = ompow(c, 2 * c->EZ);        /* zeta^2 */
    for(u64 i = 0; i < L; ++i)
        for(int l = 0; l < W; ++l){
            u64 a = B[i].l[l], b = B[i + L].l[l], d = B[i + 2*L].l[l];
            B[i].l[l]       = addm(addm(a, b, p), d, p);
            B[i + L].l[l]   = addm(a, addm(mulm(zt,  b, p),
                                           mulm(zt2, d, p), p), p);
            B[i + 2*L].l[l] = addm(a, addm(mulm(zt2, b, p),
                                           mulm(zt,  d, p), p), p);
        }
    tfwd(c, B,         L, 0,         L,       tw, 0);
    tfwd(c, B + L,     L, c->EZ,     L,       tw, L);
    tfwd(c, B + 2*L,   L, 2 * c->EZ, z - 2*L, tw, 2*L);
}

/* ---- mixed 3L inverse, z in (2L, 3L], parent tails ZERO;
 * exit B[0..z) = L * a_i (caller divides by L) ---------------------- */
static void minv(const mctx *c, vec_t *B, u64 L, u64 z){
    u64 p = c->p;
    u64 z2 = z - 2*L;
    u64 zt  = ompow(c, c->EZ);
    u64 zt2 = ompow(c, 2 * c->EZ);
    u64 i1z = invm(subm(1, zt, p), p);    /* 1/(1 - zeta) */
    tinv(c, B,     L, 0,     L);          /* full: B[0..L)  = L*b0 */
    tinv(c, B + L, L, c->EZ, L);          /* full: B[L..2L) = L*b1 */
    for(u64 i = z2; i < L; ++i)           /* 2x2 seam solve + tails */
        for(int l = 0; l < W; ++l){
            u64 c0 = B[i].l[l], c1 = B[i + L].l[l];
            u64 u1 = mulm(subm(c0, c1, p), i1z, p);     /* L*a_{i+L} */
            u64 u0 = subm(c0, u1, p);                   /* L*a_i     */
            B[i + 2*L].l[l] = addm(u0, mulm(zt2, u1, p), p); /* L*b2 */
            B[i].l[l]     = u0;
            B[i + L].l[l] = u1;
        }
    tinv(c, B + 2*L, L, 2 * c->EZ, z2);   /* trunc: B[2L..2L+z2) = L*b2 */
    for(u64 i = 0; i < z2; ++i)           /* 3-point inverse combine */
        for(int l = 0; l < W; ++l){
            u64 c0 = B[i].l[l], c1 = B[i + L].l[l], c2 = B[i + 2*L].l[l];
            u64 s0 = addm(addm(c0, c1, p), c2, p);
            u64 s1 = addm(c0, addm(mulm(zt2, c1, p), mulm(zt, c2, p), p), p);
            u64 s2 = addm(c0, addm(mulm(zt, c1, p), mulm(zt2, c2, p), p), p);
            B[i].l[l]       = mulm(s0, c->inv3, p);
            B[i + L].l[l]   = mulm(s1, c->inv3, p);
            B[i + 2*L].l[l] = mulm(s2, c->inv3, p);
        }
}

/* ---- pointwise: W-point twisted cyclic conv mod (x^W - ww) -------- */
static void convW(const mctx *c, vec_t *fa, const vec_t *fb, u64 ww){
    u64 p = c->p, out[W];
    for(int l = 0; l < W; ++l){
        u128 s1 = 0, s2 = 0;
        for(int i = 0; i < W; ++i){
            int j = l - i;
            if(j >= 0) s1 += (u128)fa->l[i] * fb->l[j];
            else       s2 += (u128)fa->l[i] * fb->l[j + W];
        }
        out[l] = addm((u64)(s1 % p), mulm(ww, (u64)(s2 % p), p), p);
    }
    for(int l = 0; l < W; ++l) fa->l[l] = out[l];
}

/* ==== full multiply pipeline, transform size NV vectors (mixed if
 * NV = 3*2^k, binary if NV = 2^k), trunc tv; coefficients mod p ===== */
static void pipe_mul(const mctx *c, u64 *prod, const u64 *a, u64 an,
                     const u64 *b, u64 bn, u64 NV, int mixed, u64 tv){
    u64 L = mixed ? NV / 3 : NV;
    vec_t *fa = calloc(NV, sizeof *fa), *fb = calloc(NV, sizeof *fb);
    u64 *tw = calloc(NV, sizeof *tw);
    for(u64 i = 0; i < an; ++i) fa[i / W].l[i % W] = a[i];
    for(u64 i = 0; i < bn; ++i) fb[i / W].l[i % W] = b[i];
    if(mixed){
        mfwd(c, fa, L, tv, tw);
        mfwd(c, fb, L, tv, tw);
    }else{
        tfwd(c, fa, NV, 0, tv, tw, 0);
        tfwd(c, fb, NV, 0, tv, tw, 0);
    }
    for(u64 k = 0; k < tv; ++k)
        convW(c, &fa[k], &fb[k], ompow(c, tw[k]));
    /* inverse contract: B[tv..NV) = scaled coefficient tails = ZERO
     * for a product (the truncated fwd left intermediates there) */
    memset(fa + tv, 0, (NV - tv) * sizeof *fa);
    u64 sc;
    if(mixed){ minv(c, fa, L, tv);        sc = invm(L % c->p,  c->p); }
    else     { tinv(c, fa, NV, 0, tv);    sc = invm(NV % c->p, c->p); }
    u64 zlen = an + bn - 1;
    for(u64 i = 0; i < zlen; ++i)
        prod[i] = mulm(fa[i / W].l[i % W], sc, c->p);
    free(fa); free(fb); free(tw);
}

static void school(const mctx *c, u64 *prod, const u64 *a, u64 an,
                   const u64 *b, u64 bn){
    u64 zlen = an + bn - 1;
    for(u64 i = 0; i < zlen; ++i) prod[i] = 0;
    for(u64 i = 0; i < an; ++i)
        for(u64 j = 0; j < bn; ++j)
            prod[i + j] = addm(prod[i + j],
                               mulm(a[i], b[j], c->p), c->p);
}

/* ---- gate 1: full mixed forward == direct evaluation -------------- */
static int gate_eval(const mctx *c, int k){
    u64 L = (u64)1 << k, NV = 3 * L;
    vec_t *B = calloc(NV, sizeof *B);
    u64 *tw = calloc(NV, sizeof *tw);
    u64 *poly = malloc(NV * 8);
    for(u64 i = 0; i < NV; ++i){
        poly[i] = xr() % c->p;
        B[i].l[0] = poly[i];
    }
    mfwd(c, B, L, NV, tw);
    int ok = 1;
    for(u64 pos = 0; pos < NV && ok; ++pos){
        u64 y = ompow(c, tw[pos]), e = 0, yp = 1;
        for(u64 i = 0; i < NV; ++i){
            e = addm(e, mulm(poly[i], yp, c->p), c->p);
            yp = mulm(yp, y, c->p);
        }
        if(B[pos].l[0] != e){
            printf("  eval FAIL k=%d pos=%llu\n", k, (unsigned long long)pos);
            ok = 0;
        }
    }
    free(B); free(tw); free(poly);
    return ok;
}

/* ---- gate 2: mixed round trip, all z2 ------------------------------ */
static int gate_rt(const mctx *c, int k){
    u64 L = (u64)1 << k, NV = 3 * L;
    vec_t *B = calloc(NV, sizeof *B), *orig = calloc(NV, sizeof *B);
    u64 *tw = calloc(NV, sizeof *tw);
    u64 sc = invm(L % c->p, c->p);
    int ok = 1;
    for(u64 z = 2*L + 1; z <= NV && ok; ++z){
        memset(B, 0, NV * sizeof *B);
        for(u64 i = 0; i < z; ++i)
            for(int l = 0; l < W; ++l)
                B[i].l[l] = xr() % c->p;
        memcpy(orig, B, NV * sizeof *B);
        mfwd(c, B, L, z, tw);
        minv(c, B, L, z);
        for(u64 i = 0; i < z && ok; ++i)
            for(int l = 0; l < W; ++l)
                if(mulm(B[i].l[l], sc, c->p) != orig[i].l[l]){
                    printf("  rt FAIL k=%d z=%llu i=%llu l=%d\n",
                           k, (unsigned long long)z, (unsigned long long)i, l);
                    ok = 0; break;
                }
    }
    free(B); free(orig); free(tw);
    return ok;
}

/* ---- ladder: smallest s in {2^j} U {3*2^j} with s >= tv ----------- */
static u64 ladder(u64 tv, int *mixed){
    u64 s2 = 1; while(s2 < tv) s2 <<= 1;
    u64 s3 = 3; while(s3 < tv) s3 <<= 1;          /* 3*2^j ladder */
    if(tv <= 2){ *mixed = 0; return tv == 1 ? 1 : 2; }
    if(s3 < s2 && s3 >= tv && s3 > 3){ *mixed = 1; return s3; }
    *mixed = 0; return s2;
}

int main(void){
    const u64 PRS[2] = { 918552577, 943718401 };   /* 3 | p-1 */
    int allok = 1;

    for(int pi = 0; pi < 2; ++pi){
        mctx c;
        mctx_build(&c, PRS[pi], 16);
        printf("p = %llu (Omega order 3*2^16)\n", (unsigned long long)c.p);

        /* gate 1: eval */
        int ok = 1;
        for(int k = 0; k <= 4; ++k) ok &= gate_eval(&c, k);
        printf("  gate 1 (mixed fwd == direct eval, k<=4): %s\n",
               ok ? "OK" : "FAIL");
        allok &= ok;

        /* gate 2: round trips, exhaustive z2 */
        ok = 1;
        for(int k = 0; k <= 6; ++k) ok &= gate_rt(&c, k);
        printf("  gate 2 (round trip, k<=6, all z2): OK=%d\n", ok);
        allok &= ok;

        /* gate 3: products vs schoolbook */
        ok = 1;
        long cases = 0;
        for(int k = 1; k <= 5 && ok; ++k){
            u64 L = (u64)1 << k, NV = 3 * L;
            for(u64 tv = 2*L + 1; tv <= NV && ok; ++tv){
                /* zlen range hitting this tv: pick extremes + middle */
                u64 zmax = tv * W, zmin = (tv - 1) * W + 1;
                u64 zl[3] = { zmin, (zmin + zmax) / 2, zmax };
                for(int t = 0; t < 3 && ok; ++t){
                    u64 zlen = zl[t];
                    u64 an = zlen / 2 + 1, bn = zlen + 1 - an;
                    static u64 A[4096], Bb[4096], P1[8192], P2[8192];
                    for(u64 i = 0; i < an; ++i) A[i] = xr() % c.p;
                    for(u64 i = 0; i < bn; ++i) Bb[i] = xr() % c.p;
                    pipe_mul(&c, P1, A, an, Bb, bn, NV, 1, tv);
                    school(&c, P2, A, an, Bb, bn);
                    for(u64 i = 0; i < zlen; ++i)
                        if(P1[i] != P2[i]){
                            printf("  prod FAIL k=%d tv=%llu zlen=%llu i=%llu\n",
                                   k, (unsigned long long)tv,
                                   (unsigned long long)zlen,
                                   (unsigned long long)i);
                            ok = 0; break;
                        }
                    ++cases;
                }
            }
        }
        printf("  gate 3 (mixed product vs schoolbook): %ld cases, %s\n",
               cases, ok ? "OK" : "FAIL");
        allok &= ok;

        /* gate 4: ladder dispatch sweep + smoothness metric */
        ok = 1;
        double worst_l = 0, worst_b = 0, best_l = 1e18, best_b = 1e18;
        for(u64 tv = 5; tv <= 192; ++tv){
            u64 zlen = tv * W - (xr() % W);
            u64 an = zlen / 2 + 1, bn = zlen + 1 - an;
            static u64 A[4096], Bb[4096], P1[8192], P2[8192];
            for(u64 i = 0; i < an; ++i) A[i] = xr() % c.p;
            for(u64 i = 0; i < bn; ++i) Bb[i] = xr() % c.p;
            int mixed; u64 NV = ladder(tv, &mixed);
            MCNT = 0;
            pipe_mul(&c, P1, A, an, Bb, bn, NV, mixed, tv);
            double ml = (double)MCNT / (double)zlen;
            u64 NB = 1; while(NB < tv) NB <<= 1;   /* binary-only */
            MCNT = 0;
            pipe_mul(&c, P2, A, an, Bb, bn, NB, 0, tv);
            double mb = (double)MCNT / (double)zlen;
            for(u64 i = 0; i < zlen; ++i)
                if(P1[i] != P2[i]){
                    printf("  ladder vs binary FAIL tv=%llu NV=%llu mixed=%d "
                           "zlen=%llu i=%llu\n", (unsigned long long)tv,
                           (unsigned long long)NV, mixed,
                           (unsigned long long)zlen, (unsigned long long)i);
                    ok = 0; break;
                }
            school(&c, P2, A, an, Bb, bn);
            for(u64 i = 0; i < zlen; ++i)
                if(P1[i] != P2[i]){
                    printf("  ladder vs school FAIL tv=%llu NV=%llu mixed=%d "
                           "zlen=%llu i=%llu\n", (unsigned long long)tv,
                           (unsigned long long)NV, mixed,
                           (unsigned long long)zlen, (unsigned long long)i);
                    ok = 0; break;
                }
            if(tv >= 48){                          /* steady window */
                if(ml > worst_l) worst_l = ml;
                if(mb > worst_b) worst_b = mb;
                if(ml < best_l) best_l = ml;
                if(mb < best_b) best_b = mb;
            }
        }
        printf("  gate 4 (ladder sweep tv=5..192): %s\n", ok ? "OK" : "FAIL");
        printf("    mults/coeff max/min, tv>=48:  ladder %.2f/%.2f = %.3f"
               "   binary-only %.2f/%.2f = %.3f\n",
               worst_l, best_l, worst_l / best_l,
               worst_b, best_b, worst_b / best_b);
        allok &= ok;
    }
    printf(allok ? "ALL OK\n" : "FAILURES\n");
    return allok ? 0 : 1;
}
