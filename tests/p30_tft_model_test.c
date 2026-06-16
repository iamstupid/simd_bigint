// Scalar model verifying the p30 design's transform math (docs §3d):
//   - radix-2 van-der-Hoeven TRUNCATED NTT (forward + inverse with
//     mixed value/coefficient state), FLINT twiddle convention
//     (per-node constant twiddle w2[j] = w[2j], bit-reversed tables)
//   - W = 4 lane decomposition y = x^4: the transform runs on whole
//     4-lane "vectors" (16 independent->4 independent scalar TFTs in
//     lockstep); NO lane mixing anywhere in the transform
//   - pointwise = 4-point twisted cyclic convolution mod (x^4 - w[k])
//     at bit-reversed spectral position k (prefix-stable twist table)
//   - 5-prime CRT (ascending-order Garner), 64-bit chunking
//     (1 limb = 1 coefficient), 150-bit coefficients -> 3-limb adds
// Gates:
//   0. w-table sanity: full forward == direct DFT at w[k] (small M)
//   1. TFT round-trip: fwd(trunc) -> inv(trunc) -> scale == identity
//   2. full multiply vs GMP mpn_mul: exhaustive small shapes, trunc-
//      boundary shapes, adversarial patterns, random large shapes
// Build: clang -O2 -std=c11 p30_tft_model_test.c -I/tmp/gmp-build \
//          /tmp/gmp-build/.libs/libgmp.a -o p30_tft_model_test
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef uint32_t u32;

#define W 4                      /* inner (x-) dimension: lanes per vector */
#define NP 5                     /* CRT primes */
#define MAXLG 16                 /* model cap: M <= 2^16 vectors */

/* ascending order is load-bearing for Garner (x0 < p_i for all i) */
static const u64 PR[NP] = { 918552577, 935329793, 943718401,
                            985661441, 998244353 };

static u64 rngs = 42;
static u64 xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }

/* ---- scalar mod-p helpers ---------------------------------------- */
static u64 addm(u64 a, u64 b, u64 p){ u64 s = a + b; return s >= p ? s - p : s; }
static u64 subm(u64 a, u64 b, u64 p){ return a >= b ? a - b : a + p - b; }
static u64 mulm(u64 a, u64 b, u64 p){ return (u64)((u128)a * b % p); }
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

/* ---- per-prime transform context ---------------------------------- */
/* w = bit-reversed root sequence (FLINT fft_small convention):
 *   w[0] = 1;  w[2^{t-1} + s] = zeta_t * w[s]   (zeta_t of order 2^t,
 *   zeta_t^2 = zeta_{t-1}, zeta_1 = -1)
 * Node (size m, index j) butterfly twiddle: w2[j] = w[2j].
 * Forward output: out[k] = eval at w[k]  =>  y-value at spectral
 * position k is w[k]: the conv twist table IS the w table (prefix-
 * stable in k). */
typedef struct {
    u64 p, inv2;
    u64 w[1u << MAXLG];
    u64 winv[1u << MAXLG];
    int lg;                       /* tables valid for M = 2^lg */
} tctx;

static void ctx_build(tctx* c, u64 p, int lg){
    c->p = p;
    c->lg = lg;
    c->inv2 = (p + 1) >> 1;
    u64 g = primitive_root(p);
    c->w[0] = c->winv[0] = 1;
    for(int t = 1; t <= lg; ++t){
        u64 z  = powm(g, (p - 1) >> t, p);     /* order 2^t */
        u64 zi = invm(z, p);
        u64 blk = (u64)1 << (t - 1);
        for(u64 s = 0; s < blk; ++s){
            c->w[blk + s]    = mulm(z,  c->w[s],    p);
            c->winv[blk + s] = mulm(zi, c->winv[s], p);
        }
    }
}

/* one y-coefficient across the W lanes */
typedef struct { u64 l[W]; } vec_t;

/* ---- forward truncated NTT (DIF), node (B[0..m), j), output trunc zo:
 * entry: B = coefficients (tail zeros fine); exit: B[0..zo) spectral.
 * Butterfly: c_i = a_i + w*b_i ; d_i = a_i - w*b_i ; w = w2[j] const. */
static void tft_fwd(const tctx* c, vec_t* B, u64 m, u64 j, u64 zo){
    if(m == 1) return;
    u64 h = m >> 1;
    u64 w = c->w[2 * j];
    if(zo <= h){
        for(u64 i = 0; i < h; ++i)
            for(int l = 0; l < W; ++l)
                B[i].l[l] = addm(B[i].l[l], mulm(w, B[i + h].l[l], c->p), c->p);
        tft_fwd(c, B, h, 2 * j, zo);
        return;
    }
    for(u64 i = 0; i < h; ++i)
        for(int l = 0; l < W; ++l){
            u64 a = B[i].l[l], wb = mulm(w, B[i + h].l[l], c->p);
            B[i].l[l]     = addm(a, wb, c->p);
            B[i + h].l[l] = subm(a, wb, c->p);
        }
    tft_fwd(c, B, h, 2 * j, h);
    tft_fwd(c, B + h, h, 2 * j + 1, zo - h);
}

/* ---- inverse truncated NTT, node (B[0..m), j), trunc z:
 * entry: B[0..z) spectral; B[z..m) = m * u_i (scaled known coeffs;
 *        zeros at the top call).
 * exit:  B[0..z) = m * u_i; B[z..m) unspecified.
 * Derivation (forward c = a + w b, d = a - w b => a = c - w b,
 * d = c - 2 w b, a = (c+d)/2, b = (c-d)/(2w); child scale h = m/2):
 *  case z < h:  (tail-fill) for i in [z,h): B[i] = (B[i] + w B[i+h])/2
 *                 ( = h*c_i, since m*a_i + w*m*b_i = 2h*c_i )
 *               recurse left (h, 2j, z)
 *               for i in [0,z): B[i] = 2*B[i] - w*B[i+h]
 *                 ( m*a_i = 2(h c_i) - w(m b_i) )
 *  case z >= h: recurse left FULL (h, 2j, h)  -> B[i] = h*c_i
 *               for i in [z-h, h):                       (BEFORE the
 *                 t = B[i] - w*B[i+h]   ( = h*d_i )       right child
 *                 B[i] = B[i] + t       ( = m*u_i )       trashes its
 *                 B[i+h] = t                              tail!)
 *               recurse right (h, 2j+1, z-h)
 *               for i in [0, z-h):
 *                 (B[i], B[i+h]) = (B[i] + B[i+h],
 *                                   (B[i] - B[i+h]) * winv[2j])
 *                 ( m*a_i = h c_i + h d_i ;
 *                   m*b_i = (h c_i - h d_i) / w )                     */
static void tft_inv(const tctx* c, vec_t* B, u64 m, u64 j, u64 z){
    if(m == 1 || z == 0) return;
    u64 h = m >> 1;
    u64 w = c->w[2 * j];
    if(z < h){
        for(u64 i = z; i < h; ++i)
            for(int l = 0; l < W; ++l)
                B[i].l[l] = mulm(addm(B[i].l[l],
                                      mulm(w, B[i + h].l[l], c->p), c->p),
                                 c->inv2, c->p);
        tft_inv(c, B, h, 2 * j, z);
        for(u64 i = 0; i < z; ++i)
            for(int l = 0; l < W; ++l)
                B[i].l[l] = subm(addm(B[i].l[l], B[i].l[l], c->p),
                                 mulm(w, B[i + h].l[l], c->p), c->p);
        return;
    }
    u64 zr = z - h;
    tft_inv(c, B, h, 2 * j, h);
    for(u64 i = zr; i < h; ++i)
        for(int l = 0; l < W; ++l){
            u64 t = subm(B[i].l[l], mulm(w, B[i + h].l[l], c->p), c->p);
            B[i].l[l] = addm(B[i].l[l], t, c->p);
            B[i + h].l[l] = t;
        }
    tft_inv(c, B + h, h, 2 * j + 1, zr);
    u64 wi = c->winv[2 * j];
    for(u64 i = 0; i < zr; ++i)
        for(int l = 0; l < W; ++l){
            u64 s = addm(B[i].l[l], B[i + h].l[l], c->p);
            u64 d = subm(B[i].l[l], B[i + h].l[l], c->p);
            B[i].l[l] = s;
            B[i + h].l[l] = mulm(d, wi, c->p);
        }
}

/* ---- pointwise: 4-point twisted cyclic conv mod (x^4 - ww), ww = w[k]:
 * c_l = sum_{i+j=l} a_i b_j + ww * sum_{i+j=4+l} a_i b_j               */
static void conv4(const tctx* c, vec_t* fa, const vec_t* fb, u64 ww){
    u64 out[W];
    for(int l = 0; l < W; ++l){
        u128 s1 = 0, s2 = 0;
        for(int i = 0; i < W; ++i){
            int j = l - i;
            if(j >= 0) s1 += (u128)fa->l[i] * fb->l[j];
            else       s2 += (u128)fa->l[i] * fb->l[j + W];
        }
        out[l] = addm((u64)(s1 % c->p),
                      mulm(ww, (u64)(s2 % c->p), c->p), c->p);
    }
    for(int l = 0; l < W; ++l) fa->l[l] = out[l];
}

/* ---- gate 0: full forward == direct DFT at w[k] (x-transform via
 * lane decomposition + conv-twist relation checked separately below) */
static int gate0(const tctx* c, int lg){
    u64 M = (u64)1 << lg;
    vec_t* B = calloc(M, sizeof *B);
    u64* ref = malloc(M * 8);
    u64* poly = malloc(M * 8);
    for(u64 i = 0; i < M; ++i){
        poly[i] = xr() % c->p;
        B[i].l[0] = poly[i];                       /* test lane 0 alone */
    }
    tft_fwd(c, B, M, 0, M);
    int bad = 0;
    for(u64 k = 0; k < M && !bad; ++k){
        u64 e = 0, x = c->w[k], xp = 1;
        for(u64 i = 0; i < M; ++i){
            e = addm(e, mulm(poly[i], xp, c->p), c->p);
            xp = mulm(xp, x, c->p);
        }
        ref[k] = e;
        if(B[k].l[0] != e) bad = 1;
    }
    free(B); free(ref); free(poly);
    return !bad;
}

/* ---- gate 1: fwd(trunc) -> zero tail -> inv(trunc) -> /M == id ---- */
static int gate1(const tctx* c, int lg, u64 trunc){
    u64 M = (u64)1 << lg;
    vec_t* B = calloc(M, sizeof *B);
    vec_t* orig = calloc(M, sizeof *B);
    for(u64 i = 0; i < trunc; ++i)
        for(int l = 0; l < W; ++l)
            orig[i].l[l] = B[i].l[l] = xr() % c->p;
    tft_fwd(c, B, M, 0, trunc);
    for(u64 i = trunc; i < M; ++i) memset(&B[i], 0, sizeof B[i]);
    tft_inv(c, B, M, 0, trunc);
    u64 mi = invm(M % c->p, c->p);
    int bad = 0;
    for(u64 i = 0; i < trunc && !bad; ++i)
        for(int l = 0; l < W; ++l)
            if(mulm(B[i].l[l], mi, c->p) != orig[i].l[l]) bad = 1;
    free(B); free(orig);
    return !bad;
}

/* ---- the model multiplier ----------------------------------------- */
static tctx CTX[NP];             /* built to MAXLG once */

static void model_mul(u64* out, const u64* a, u64 an, const u64* b, u64 bn){
    u64 zlen = an + bn - 1;
    u64 tv = (zlen + W - 1) / W;             /* trunc in vectors */
    int lg = 0;
    while(((u64)1 << lg) < tv) lg++;
    u64 M = (u64)1 << lg;

    static u64 res[NP][(1u << MAXLG) * W];
    for(int q = 0; q < NP; ++q){
        const tctx* c = &CTX[q];
        u64 p = c->p;
        vec_t* va = calloc(M, sizeof *va);
        vec_t* vb = calloc(M, sizeof *vb);
        for(u64 i = 0; i < an; ++i) va[i / W].l[i % W] = a[i] % p;
        for(u64 i = 0; i < bn; ++i) vb[i / W].l[i % W] = b[i] % p;
        tft_fwd(c, va, M, 0, tv);
        tft_fwd(c, vb, M, 0, tv);
        for(u64 k = 0; k < tv; ++k)
            conv4(c, &va[k], &vb[k], c->w[k]);
        for(u64 i = tv; i < M; ++i) memset(&va[i], 0, sizeof va[i]);
        tft_inv(c, va, M, 0, tv);
        u64 mi = invm(M % p, p);
        for(u64 i = 0; i < zlen; ++i)
            res[q][i] = mulm(va[i / W].l[i % W], mi, p);
        free(va); free(vb);
    }

    /* Garner (ascending primes): digits v0..v4, exact, no mod-P value */
    u64 PPM[NP][NP];             /* PPM[i][k] = prod_{j<k} p_j mod p_i  */
    u64 INVP[NP];                /* (prod_{j<i} p_j)^-1 mod p_i         */
    for(int i = 1; i < NP; ++i){
        u64 pi = PR[i], pp = 1;
        for(int k = 0; k < i; ++k){
            PPM[i][k] = pp;
            pp = mulm(pp, PR[k] % pi, pi);
        }
        INVP[i] = invm(pp, pi);
    }
    memset(out, 0, (an + bn) * 8);
    u64 tail[2] = { 0, 0 };                     /* spill guard */
    for(u64 i = 0; i < zlen; ++i){
        u64 v[NP];
        v[0] = res[0][i];
        for(int q = 1; q < NP; ++q){
            u64 pi = PR[q], t = v[0] % pi;
            for(int k = 1; k < q; ++k)
                t = addm(t, mulm(v[k], PPM[q][k], pi), pi);
            v[q] = mulm(subm(res[q][i] % pi, t, pi), INVP[q], pi);
        }
        /* Horner compose: c = (((v4 p3 + v3) p2 + v2) p1 + v1) p0 + v0 */
        u128 x = v[4];
        x = x * PR[3] + v[3];
        x = x * PR[2] + v[2];
        x = x * PR[1] + v[1];                   /* < 2^120 */
        u128 m0 = (u128)(u64)x * PR[0] + v[0];
        u128 m1 = (u128)(u64)(x >> 64) * PR[0] + (u64)(m0 >> 64);
        u64 c0 = (u64)m0, c1 = (u64)m1, c2 = (u64)(m1 >> 64);
        /* accumulate at limb offset i (64-bit chunking: exact) */
        u128 s = (u128)out[i] + c0;
        out[i] = (u64)s;
        u64* o1 = i + 1 < an + bn ? &out[i + 1] : &tail[0];
        u64* o2 = i + 2 < an + bn ? &out[i + 2] : &tail[i + 2 - (an + bn)];
        s = (u128)*o1 + c1 + (u64)(s >> 64);
        *o1 = (u64)s;
        s = (u128)*o2 + c2 + (u64)(s >> 64);
        *o2 = (u64)s;
        for(u64 j = i + 3; (s >> 64) && j < an + bn; ++j){
            s = (u128)out[j] + (u64)(s >> 64);
            out[j] = (u64)s;
        }
    }
    if(tail[0] || tail[1]){ fprintf(stderr, "tail spill!\n"); exit(1); }
}

/* ---- gate 2 harness ------------------------------------------------ */
static u64 A[(1u << MAXLG) * W], B2[(1u << MAXLG) * W];
static u64 R[(1u << MAXLG) * W * 2 + 4], REF[(1u << MAXLG) * W * 2 + 4];
static int fails, runs;

static void fill(u64* p, u64 n, int pat){
    for(u64 i = 0; i < n; ++i)
        p[i] = pat == 1 ? ~(u64)0 : pat == 2 ? ((i & 31) == 0 ? ~(u64)0 : 0) : xr();
}
static void check(u64 an, u64 bn, int pat){
    fill(A, an, pat);
    fill(B2, bn, pat);
    if(an >= bn) mpn_mul((mp_ptr)REF, (mp_srcptr)A, an, (mp_srcptr)B2, bn);
    else         mpn_mul((mp_ptr)REF, (mp_srcptr)B2, bn, (mp_srcptr)A, an);
    model_mul(R, A, an, B2, bn);
    ++runs;
    if(memcmp(R, REF, (an + bn) * 8)){
        u64 i = 0;
        while(R[i] == REF[i]) ++i;
        printf("FAIL an=%llu bn=%llu pat=%d limb %llu got %016llx want %016llx\n",
               (unsigned long long)an, (unsigned long long)bn, pat,
               (unsigned long long)i, (unsigned long long)R[i],
               (unsigned long long)REF[i]);
        ++fails;
    }
}

int main(void){
    for(int q = 0; q < NP; ++q) ctx_build(&CTX[q], PR[q], MAXLG);

    /* gate 0 + twist-relation spot check */
    for(int q = 0; q < NP; ++q)
        for(int lg = 0; lg <= 6; ++lg)
            if(!gate0(&CTX[q], lg)){
                printf("gate0 FAIL prime %d lg %d\n", q, lg);
                return 1;
            }
    printf("gate 0 (forward == DFT at w[k], 5 primes x lg 0..6): OK\n");

    /* gate 1: every trunc value at small lg, sampled at larger */
    for(int q = 0; q < NP; ++q){
        for(int lg = 0; lg <= 6; ++lg){
            u64 M = (u64)1 << lg;
            for(u64 tr = 1; tr <= M; ++tr)
                if(!gate1(&CTX[q], lg, tr)){
                    printf("gate1 FAIL prime %d lg %d trunc %llu\n",
                           q, lg, (unsigned long long)tr);
                    return 1;
                }
        }
        for(int lg = 7; lg <= 10; ++lg){
            u64 M = (u64)1 << lg;
            u64 trs[7] = { 1, M / 2 - 1, M / 2, M / 2 + 1, M - 1, M,
                           M / 2 + (xr() % (M / 2)) };
            for(int t = 0; t < 7; ++t)
                if(!gate1(&CTX[q], lg, trs[t])){
                    printf("gate1 FAIL prime %d lg %d trunc %llu\n",
                           q, lg, (unsigned long long)trs[t]);
                    return 1;
                }
        }
    }
    printf("gate 1 (TFT round-trip, exhaustive trunc lg<=6 + edges lg<=10): OK\n");

    /* gate 2: vs GMP */
    for(u64 an = 1; an <= 48; ++an)              /* exhaustive small */
        for(u64 bn = 1; bn <= 48; ++bn)
            check(an, bn, 0);
    int g2a = runs;
    printf("gate 2a (exhaustive 1..48 x 1..48): %d cases OK so far, %d fails\n",
           g2a, fails);

    /* trunc-boundary shapes: totals around every pow2-vector edge */
    for(int lg = 4; lg <= 12; ++lg){
        u64 edge = ((u64)1 << lg) * W;           /* total limbs at boundary */
        for(int d = -5; d <= 5; ++d){
            u64 tot = edge + d;
            u64 an = tot / 2, bn = tot - an;
            check(an, bn, 0);
            check(an + (tot / 3), bn - (tot / 3), 0);   /* unbalanced */
        }
    }
    printf("gate 2b (pow2-boundary totals, lg 4..12): %d cases, %d fails\n",
           runs - g2a, fails);

    /* adversarial + larger random */
    int g2b = runs;
    check(2048, 2048, 1);
    check(2048, 2048, 2);
    check(4096, 4096, 1);
    check(1, 1, 1);
    check(3000, 17, 1);
    for(int t = 0; t < 60; ++t){
        u64 an = 1 + xr() % 4096, bn = 1 + xr() % 4096;
        check(an, bn, (int)(xr() % 3));
    }
    check(30000, 30000, 0);
    check(30000, 30000, 1);
    check(65536 * 2 - 7, 13, 0);
    printf("gate 2c (adversarial + random): %d cases, %d fails\n",
           runs - g2b, fails);

    if(fails) printf("FAILED: %d of %d\n", fails, runs);
    else      printf("OK: all %d multiply cases\n", runs);
    return fails != 0;
}
