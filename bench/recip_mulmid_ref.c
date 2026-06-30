/* recip_mulmid_ref.c -- Newton reciprocal in the NEGATE / 2W form, written to
 * line up with the asm kernel (recip_step_8_16 / reciprocal_u512_mp step3),
 * with the residual computed by Harvey's mpn_mulmid on a NARROW window.
 *
 * This is the negate-form rewrite of the Bodrato "wide window" reference.  The
 * leading 1 is NEVER formed: the residual's sign is read from the top bit of
 * 2W and handled by an abs + add/sub branch, so the middle product only needs
 * coeffs [h-1, n] of D*x_hi -- i.e. pad D by ONE limb, not h.
 *
 * --------------------------------------------------------------------------
 * mpn_mulmid(rp, ap, an, bp, bn)  [an>=bn>=1] writes an-bn+3 limbs =
 *     EXACT rectangle  Sum_{bn-1 <= i+j <= an-1} ap[i]*bp[j] * B^{(i+j)-(bn-1)}
 *
 * NARROW residual product (this file):
 *     mpn_mulmid(rp, Dpad[n+1], n+1, x_hi[h], h)   ->  coeffs [h-1, n]
 *   vs WIDE Bodrato (recovered, now discarded): pad D to n+h -> coeffs [h-1,n+h-1].
 *   Narrow suffices here because the negate form never builds B^h*Dbar - D*iph;
 *   it reads the residue-class sign from 2W's top bit instead.
 * --------------------------------------------------------------------------
 *
 * Data flow (== asm), for one h -> n=2h step, inputs D (n limbs), x_hi (h limbs):
 *   (1) W   = D*x_hi window [h-1, n]            (Wl = n-h+2 limbs)   <-- mpn_mulmid
 *   (2) V   = 2*W                               (shld doubling; Xh = 2*x_hi)
 *   (3) neg = (top bit of V[Wl-1] CLEAR);  m = neg?0:~0;  |M| = V^m  (drop +1)
 *   (4) Delta = high product x_hi*|M|, keep positions [h, n+1]; drop |M|[0]
 *   (5) out = x_hi*B^h  +Delta (V top set) | -Delta (V top clear)
 *
 * Result is faithful to ~2*n*64 - 9 bits; both residue classes handled.
 *
 * Build: clang -std=gnu17 -O2 -march=native recip_mulmid_ref.c \
 *              /home/dev/bigint/GIMP/.libs/libgmp.a -o /tmp/.../ref
 */
#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
typedef uint64_t u64; typedef unsigned __int128 u128;
extern void __gmpn_mulmid(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
static const unsigned long B = 64;

/* ---- negate-form Newton seed ladder (verbatim from mp_check_faithful.c) ---- */
static void nstep(mpz_t x, const mpz_t D, int Dl, int pin, int pout) {
    mpz_t Dt, P1, H1, t, m, P2; mpz_inits(Dt, P1, H1, t, m, P2, NULL);
    mpz_fdiv_q_2exp(Dt, D, (long)(Dl - pout) * B);
    mpz_mul(P1, x, Dt); mpz_fdiv_q_2exp(H1, P1, (long)pin * B); mpz_fdiv_r_2exp(H1, H1, (long)pout * B);
    mpz_ui_pow_ui(m, 2, (unsigned long)pout * B); mpz_sub_ui(m, m, 1); mpz_sub(t, m, H1);
    mpz_mul(P2, x, t); mpz_fdiv_q_2exp(x, P2, (long)pin * B - 1); mpz_fdiv_r_2exp(x, x, (long)pout * B);
    mpz_clears(Dt, P1, H1, t, m, P2, NULL);
}
static void seed_xhi(mpz_t Xh, const mpz_t D, int n, int h) {
    mpz_t num, dt; mpz_inits(num, dt, NULL);
    mpz_ui_pow_ui(num, 2, 127); mpz_sub_ui(num, num, 1);
    mpz_fdiv_q_2exp(dt, D, (long)(n - 1) * B); mpz_fdiv_q(Xh, num, dt);
    int p = 1, sched[] = {3, 4, 8, 16, 32};
    for (int s = 0; sched[s] <= h; s++) { nstep(Xh, D, n, p, sched[s]); p = sched[s]; if (p == h) break; }
    if (p < h) nstep(Xh, D, n, p, h);
    mpz_clears(num, dt, NULL);
}

/* One negate-form step.  Writes the n-limb reciprocal I (faithful, pre-adjust).
 * Returns 1 if the SUB branch (negative residue class) was taken, else 0.      */
static int recip_step_negate(u64 *I, const u64 *Dl, int n, const u64 *xh, int h) {
    int Wl = n - h + 2;                          /* window limbs, coeffs [h-1, n] */

    /* (1) W = D*x_hi coeffs [h-1, n]  via mpn_mulmid, NARROW (pad D by 1 limb). */
    u64 Dpad[80], rp[80];
    for (int i = 0; i < n; i++) Dpad[i] = Dl[i];
    Dpad[n] = 0;                                 /* the single pad limb -> reach coeff n */
    __gmpn_mulmid(rp, Dpad, n + 1, xh, h);       /* <<<< THE MIDDLE PRODUCT >>>> n-h+4 limbs */
    u64 *Wd = rp;                                /* Wd[0..Wl-1] = coeffs [h-1, n]; rest guard */

    /* cross-check: mpn_mulmid must equal the hand-rolled parallelogram */
    {
        u128 acc[80] = {0};
        for (int i = 0; i < h; i++) for (int k = 0; k < Wl; k++) {
            int j = (h - 1) + k - i; if (j < 0 || j >= n) continue;
            u128 p = (u128)xh[i] * Dl[j]; acc[k] += (u64)p; acc[k + 1] += (u64)(p >> 64);
        }
        u128 c = 0;
        for (int k = 0; k < Wl; k++) { u128 t = acc[k] + c; if ((u64)t != Wd[k]) { fprintf(stderr, "W mismatch k=%d\n", k); exit(2); } c = t >> 64; }
    }

    /* (2) V = 2*W */
    u64 V[80]; { u64 cy = 0; for (int k = 0; k < Wl; k++) { u64 nv = (Wd[k] << 1) | cy; cy = Wd[k] >> 63; V[k] = nv; } }

    /* (3) neg = V top bit CLEAR ; |M| = V ^ m,  m = neg?0:~0  (+1 dropped) */
    int neg = (V[Wl - 1] >> 63) == 0;
    u64 m = neg ? 0 : ~0ULL;
    u64 Mab[80]; for (int k = 0; k < Wl; k++) Mab[k] = V[k] ^ m;

    /* (4) Delta = high product x_hi*|M|, keep positions [h, n+1]; drop |M|[0] */
    u128 acc2[80] = {0};
    for (int i = 0; i < h; i++) for (int k = 1 /*DROP |M|[0]*/; k < Wl; k++) {
        int pos = i + k, a = pos - (h + 1);
        if (pos < h || pos > n + 1) continue;
        u128 p = (u128)xh[i] * Mab[k];
        if (a >= 0) { acc2[a] += (u64)p; acc2[a + 1] += (u64)(p >> 64); }
        else        { acc2[0]  += (u64)(p >> 64); }          /* a==-1 (pos==h): only hi */
    }
    u64 Da[80]; { u128 c = 0; for (int a = 0; a <= n - h; a++) { u128 t = acc2[a] + c; Da[a] = (u64)t; c = t >> 64; } }

    /* (5) out = x_hi*B^h  +/- Delta */
    if (!neg) {                                  /* ADD (V top set) */
        for (int k = 0; k < h; k++) I[k] = Da[k];
        u128 s = (u128)xh[0] + Da[h]; I[h] = (u64)s; u64 cy = (u64)(s >> 64);
        for (int k = 1; k < h; k++) { u128 t = (u128)xh[k] + cy; I[h + k] = (u64)t; cy = (u64)(t >> 64); }
    } else {                                     /* SUB (V top clear) */
        u64 bw = 0;
        for (int k = 0; k < h; k++) { u128 s = (u128)0 - Da[k] - bw; I[k] = (u64)s; bw = (s >> 64) ? 1 : 0; }
        u128 s = (u128)xh[0] - Da[h] - bw; I[h] = (u64)s; bw = (s >> 64) ? 1 : 0;
        for (int k = 1; k < h; k++) { u128 t = (u128)xh[k] - bw; I[h + k] = (u64)t; bw = (t >> 64) ? 1 : 0; }
    }
    return neg;
}

int main(int argc, char **argv) {
    int h = argc > 1 ? atoi(argv[1]) : 8; int n = 2 * h;
    long TR = argc > 2 ? atol(argv[2]) : 100000;
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs, argc > 3 ? atoi(argv[3]) : 1);
    mpz_t D, Xh, P, p2, e, d2, Iz; mpz_inits(D, Xh, P, p2, e, d2, Iz, NULL);
    u64 Dl[80], xh[80], I[80];
    double minp = 1e9, sump = 0; long cnt = 0, negc = 0;
    for (long tr = 0; tr < TR; tr++) {
        mpz_urandomb(D, rs, (unsigned long)n * B); mpz_setbit(D, (unsigned long)n * B - 1);
        for (int i = 0; i < n; i++) Dl[i] = 0; mpz_export(Dl, NULL, -1, 8, 0, 0, D);
        seed_xhi(Xh, D, n, h);
        for (int i = 0; i < h; i++) xh[i] = 0; mpz_export(xh, NULL, -1, 8, 0, 0, Xh);

        negc += recip_step_negate(I, Dl, n, xh, h);

        /* precision: how close is I*D to a power of two */
        mpz_import(Iz, n, -1, 8, 0, 0, I);
        mpz_mul(P, Iz, D); long Lb = mpz_sizeinbase(P, 2);
        mpz_ui_pow_ui(p2, 2, Lb); mpz_sub(e, p2, P); mpz_ui_pow_ui(d2, 2, Lb - 1); mpz_sub(d2, P, d2);
        long eh = mpz_sgn(e) ? (long)mpz_sizeinbase(e, 2) : 0, el = mpz_sgn(d2) ? (long)mpz_sizeinbase(d2, 2) : 0;
        double prec = (mpz_cmp(e, d2) < 0) ? (double)(Lb - eh) : (double)(Lb - 1 - el);
        if (prec < minp) minp = prec; sump += prec; cnt++;
    }
    printf("NEGATE/2W via mpn_mulmid(narrow [h-1,n]) h=%d->%d: min=%.0f avg=%.1f neg=%ld/%ld (target ~%d)\n",
           h, n, minp, sump / cnt, negc, cnt, n * 64 - 18);
    printf("  (min correct bits; both residue classes exercised; W==hand-rolled verified each trial)\n");
    mpz_clears(D, Xh, P, p2, e, d2, Iz, NULL);
    return 0;
}
