// p30 bottom-stage microbench (stage-2 idea 2): batch-transpose NTT
// bottom vs the twisted-conv pointwise.
//
// Identity: the pointwise step at spectral vector k is a 16-point
// twisted cyclic conv mod (x^16 - w[k]) -- which is exactly 4 more
// (in-lane) NTT levels of the full 16M-point transform, a plain
// pointwise product (one wide REDC per point), and 4 inverse levels.
// The in-lane level twiddles are the SAME prefix-stable BR table read
// 4 levels deeper: for the group of 16 positions k0..k0+15 (lockstep
// after a 16x16 transpose) they are table VECTORS
//   L1: w[2k]      L2: w[4k], w[4k+2]      L3: w[8k+2c] (c<4)
//   L4: w[16k+2c] (c<8)
// pre-swizzled at plan time into per-block tables (15 slots of
// [c[16] | rec[16]] = 1920B/block fwd, same inv) so consumption is
// pure contiguous loads, no shuffles.
//
// Replacement seam accounting (per 16-vector block, 256 points):
//   A (current): 8x p30_conv16_kara2                  [b-side only]
//   B (this):    b-side: transpose + 4 lv + pw + 4 inv lv + transpose
//                a-side: + transpose + 4 lv (during fwd(a), measured
//                separately as "a-ext")
// B output == 16 * A output (mod p): unscaled inverse contributes 16,
// both variants carry the same single R^-1 Montgomery stray (absorbed
// by the engine's final scale constant).
//
// Build: clang -O3 -march=native -std=c11 -I../include \
//          p30_bottom_bench.c -lm -o /tmp/p30_bottom_bench
#define _POSIX_C_SOURCE 199309L
#define SCRATCH_IMPLEMENTATION
#include "p30_ref.h"
#include <stdio.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rngs = 42;
static uint64_t xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }
static volatile uint32_t SINK;

/* ------------------------------------------------------------------ */
/* 16x16 u32 transpose, 64 ops. Self-tested in main.                   */
/* ------------------------------------------------------------------ */
static inline void tr16(__m512i v[16]){
    __m512i s[16];
    for(int i = 0; i < 16; i += 2){
        s[i]   = _mm512_unpacklo_epi32(v[i], v[i + 1]);
        s[i+1] = _mm512_unpackhi_epi32(v[i], v[i + 1]);
    }
    for(int i = 0; i < 16; i += 4){
        v[i]   = _mm512_unpacklo_epi64(s[i],   s[i + 2]);
        v[i+1] = _mm512_unpackhi_epi64(s[i],   s[i + 2]);
        v[i+2] = _mm512_unpacklo_epi64(s[i+1], s[i + 3]);
        v[i+3] = _mm512_unpackhi_epi64(s[i+1], s[i + 3]);
    }
    for(int i = 0; i < 4; ++i){
        s[i]    = _mm512_shuffle_i32x4(v[i], v[i + 4], 0x88);
        s[i+4]  = _mm512_shuffle_i32x4(v[i], v[i + 4], 0xdd);
        s[i+8]  = _mm512_shuffle_i32x4(v[i + 8], v[i + 12], 0x88);
        s[i+12] = _mm512_shuffle_i32x4(v[i + 8], v[i + 12], 0xdd);
    }
    for(int i = 0; i < 4; ++i){
        v[i]    = _mm512_shuffle_i32x4(s[i], s[i + 8], 0x88);
        v[i+8]  = _mm512_shuffle_i32x4(s[i], s[i + 8], 0xdd);
        v[i+4]  = _mm512_shuffle_i32x4(s[i + 4], s[i + 12], 0x88);
        v[i+12] = _mm512_shuffle_i32x4(s[i + 4], s[i + 12], 0xdd);
    }
}
/* row mapping produced by the network above: out[ROWMAP[r]] = column r.
 * Filled by the self-test; identity expected for the canonical net. */
static int ROWMAP[16];

/* ------------------------------------------------------------------ */
/* per-block bottom twiddle tables: 15 slots x [c[16] | rec[16]].      */
/* fwd consumption order: L1, L2a, L2b, L3c0..3, L4c0..7.              */
/* inv order: L4c0..7, L3c0..3, L2a, L2b, L1 (from winv).              */
/* ------------------------------------------------------------------ */
#define BSLOT 32                       /* u32 per slot */
#define BTAB  (15 * BSLOT)             /* u32 per block table */
static void build_btab(uint32_t *tab, size_t k0, const p30_cc *w, int inv){
    size_t idx[15][16];
    size_t s = 0;
    if(!inv){
        for(int t = 0; t < 16; ++t) idx[0][t] = 2 * (k0 + t);
        s = 1;
        for(size_t c = 0; c < 2; ++c, ++s)
            for(int t = 0; t < 16; ++t) idx[s][t] = 4 * (k0 + t) + 2 * c;
        for(size_t c = 0; c < 4; ++c, ++s)
            for(int t = 0; t < 16; ++t) idx[s][t] = 8 * (k0 + t) + 2 * c;
        for(size_t c = 0; c < 8; ++c, ++s)
            for(int t = 0; t < 16; ++t) idx[s][t] = 16 * (k0 + t) + 2 * c;
    }else{
        for(size_t c = 0; c < 8; ++c, ++s)
            for(int t = 0; t < 16; ++t) idx[s][t] = 16 * (k0 + t) + 2 * c;
        for(size_t c = 0; c < 4; ++c, ++s)
            for(int t = 0; t < 16; ++t) idx[s][t] = 8 * (k0 + t) + 2 * c;
        for(size_t c = 0; c < 2; ++c, ++s)
            for(int t = 0; t < 16; ++t) idx[s][t] = 4 * (k0 + t) + 2 * c;
        for(int t = 0; t < 16; ++t) idx[14][t] = 2 * (k0 + t);
    }
    for(int sl = 0; sl < 15; ++sl)
        for(int t = 0; t < 16; ++t){
            tab[sl * BSLOT + t]      = w[idx[sl][t]].c;
            tab[sl * BSLOT + 16 + t] = w[idx[sl][t]].rec;
        }
}

/* Barrett with PER-LANE (c, rec): p30_bar2p assumes broadcast rec (its
 * odd-lane q uses the even lane's rec); here rec differs per lane, so
 * the odd half needs the shifted rec plane (one extra vpsrlq). */
static inline __m512i bar2p_v(__m512i a, __m512i vc, __m512i vrec, __m512i vp){
    __m512i pe = _mm512_mul_epu32(a, vrec);
    __m512i po = _mm512_mul_epu32(_mm512_srli_epi64(a, 32),
                                  _mm512_srli_epi64(vrec, 32));
    __m512i q  = _mm512_mask_mov_epi32(_mm512_srli_epi64(pe, 32),
                                       (__mmask16)0xAAAA, po);
    return _mm512_sub_epi32(_mm512_mullo_epi32(a, vc),
                            _mm512_mullo_epi32(q, vp));
}

/* vector-twiddle butterflies (twiddles differ per lane = per position) */
#define BFV(lo_, hi_, sl_) do{                                           \
        __m512i wc_ = _mm512_load_si512(tab + (sl_) * BSLOT);            \
        __m512i wr_ = _mm512_load_si512(tab + (sl_) * BSLOT + 16);       \
        __m512i a_  = p30_sh2(x[lo_], vp2);                              \
        __m512i wb_ = bar2p_v(x[hi_], wc_, wr_, vp);                   \
        x[lo_] = _mm512_add_epi32(a_, wb_);                              \
        x[hi_] = _mm512_add_epi32(a_, _mm512_sub_epi32(vp2, wb_));       \
    }while(0)
#define IBFV(lo_, hi_, sl_) do{                                          \
        __m512i wc_ = _mm512_load_si512(tab + (sl_) * BSLOT);            \
        __m512i wr_ = _mm512_load_si512(tab + (sl_) * BSLOT + 16);       \
        __m512i a_ = p30_sh2(x[lo_], vp2);                               \
        __m512i b_ = p30_sh2(x[hi_], vp2);                               \
        x[lo_] = _mm512_add_epi32(a_, b_);                               \
        x[hi_] = bar2p_v(_mm512_add_epi32(_mm512_sub_epi32(a_, b_),      \
                                            vp2),                       \
                           wc_, wr_, vp);                                \
    }while(0)

static inline void lanes_fwd4(__m512i x[16], const uint32_t *tab,
                              __m512i vp, __m512i vp2){
    BFV(0,8,0); BFV(1,9,0); BFV(2,10,0); BFV(3,11,0);
    BFV(4,12,0); BFV(5,13,0); BFV(6,14,0); BFV(7,15,0);
    BFV(0,4,1); BFV(1,5,1); BFV(2,6,1); BFV(3,7,1);
    BFV(8,12,2); BFV(9,13,2); BFV(10,14,2); BFV(11,15,2);
    for(int c = 0; c < 4; ++c){
        BFV(4*c, 4*c+2, 3 + c); BFV(4*c+1, 4*c+3, 3 + c);
    }
    for(int c = 0; c < 8; ++c)
        BFV(2*c, 2*c+1, 7 + c);
}
static inline void lanes_inv4(__m512i x[16], const uint32_t *tab,
                              __m512i vp, __m512i vp2){
    for(int c = 0; c < 8; ++c)
        IBFV(2*c, 2*c+1, c);
    for(int c = 0; c < 4; ++c){
        IBFV(4*c, 4*c+2, 8 + c); IBFV(4*c+1, 4*c+3, 8 + c);
    }
    IBFV(0,4,12); IBFV(1,5,12); IBFV(2,6,12); IBFV(3,7,12);
    IBFV(8,12,13); IBFV(9,13,13); IBFV(10,14,13); IBFV(11,15,13);
    IBFV(0,8,14); IBFV(1,9,14); IBFV(2,10,14); IBFV(3,11,14);
    IBFV(4,12,14); IBFV(5,13,14); IBFV(6,14,14); IBFV(7,15,14);
}

/* a-side extension: canonical block -> transposed + 4 levels, [0,2p) */
static void aext_block(uint32_t *dst, const uint32_t *src,
                       const uint32_t *tab, const p30_prime *pp,
                       __m512i vp, __m512i vp2){
    (void)pp;
    __m512i x[16];
    for(int r = 0; r < 16; ++r)
        x[ROWMAP[r]] = _mm512_load_si512(src + 16 * r);
    tr16(x);
    lanes_fwd4(x, tab, vp, vp2);
    for(int r = 0; r < 16; ++r)
        _mm512_store_si512(dst + 16 * r, p30_sh2(x[r], vp2));
}

/* b-side fused bottom: lazy block -> transpose, 4 lv, pw vs extended
 * a, 4 inv lv, transpose back. Output = 16 * conv * R^-1, lazy. */
static void tb_block(uint32_t *B, const uint32_t *Aext,
                     const uint32_t *ftab, const uint32_t *itab,
                     const p30_prime *pp, __m512i vp, __m512i vp2){
    const __m512i vJ  = _mm512_set1_epi64(pp->J);
    const __m512i vP  = _mm512_set1_epi64(pp->p);
    const __m512i vP2w = _mm512_set1_epi64(pp->p2);
    __m512i x[16];
    for(int r = 0; r < 16; ++r)
        x[ROWMAP[r]] = _mm512_load_si512(B + 16 * r);
    tr16(x);
    {   const uint32_t *tab = ftab;
        lanes_fwd4(x, tab, vp, vp2); }
    for(int r = 0; r < 16; ++r){
        __m512i b = p30_sh2(x[r], vp2);                 /* [0,2p) */
        __m512i a = _mm512_load_si512(Aext + 16 * r);   /* [0,2p) */
        __m512i re = _mm512_mul_epu32(a, b);
        __m512i ro = _mm512_mul_epu32(_mm512_srli_epi64(a, 32),
                                      _mm512_srli_epi64(b, 32));
        x[r] = p30_kredc(re, ro, vJ, vP, vP2w);         /* < 2.65p */
    }
    {   const uint32_t *tab = itab;
        lanes_inv4(x, tab, vp, vp2); }
    tr16(x);
    for(int r = 0; r < 16; ++r)
        _mm512_store_si512(B + 16 * r, x[ROWMAP[r]]);
}

int main(void){
    p30_plan *pl = &p30_tls_plan;
    p30_plan_ensure(pl, 16);
    const p30_prime *pp = &pl->b.pr[4];
    __m512i vp = _mm512_set1_epi32((int)pp->p);
    __m512i vp2 = _mm512_add_epi32(vp, vp);

    /* ---- transpose self-test: learn/verify the row mapping ---- */
    {
        static uint32_t m[256];
        __m512i v[16];
        for(int r = 0; r < 16; ++r){
            for(int l = 0; l < 16; ++l) m[16*r+l] = (uint32_t)(16*r+l);
            v[r] = _mm512_loadu_si512(m + 16*r);
        }
        tr16(v);
        for(int r = 0; r < 16; ++r){
            uint32_t row[16];
            _mm512_storeu_si512(row, v[r]);
            /* row r should hold column c: entries c, 16+c, 32+c, ... */
            uint32_t c = row[0];
            if(c > 15){ printf("transpose broken (row %d starts %u)\n", r, c); return 1; }
            for(int l = 0; l < 16; ++l)
                if(row[l] != (uint32_t)(16*l + c)){
                    printf("transpose broken at out-row %d lane %d: %u\n",
                           r, l, row[l]);
                    return 1;
                }
            ROWMAP[c] = r;       /* column c landed in register r */
        }
        int id = 1;
        for(int r = 0; r < 16; ++r) id &= (ROWMAP[r] == r);
        printf("transpose OK (%s row order)\n", id ? "identity" : "permuted");
    }

    /* ---- data + tables ---- */
    enum { NBLK = 2048 };                       /* 2048 blocks = 2MB/side */
    static uint32_t Bb[NBLK*256] __attribute__((aligned(64)));
    static uint32_t Ac[NBLK*256] __attribute__((aligned(64)));  /* canonical */
    static uint32_t Ax[NBLK*256] __attribute__((aligned(64)));  /* extended  */
    static uint32_t FT[NBLK*BTAB] __attribute__((aligned(64)));
    static uint32_t IT[NBLK*BTAB] __attribute__((aligned(64)));
    for(size_t i = 0; i < (size_t)NBLK*256; ++i){
        Bb[i] = (uint32_t)(xr() % (2u * pp->p));
        Ac[i] = (uint32_t)(xr() % pp->p);
    }
    for(size_t b = 0; b < NBLK; ++b){
        size_t k0 = 16 * (b & 255);             /* k0 in [0, 4080] */
        build_btab(FT + b*BTAB, k0, pp->w,    0);
        build_btab(IT + b*BTAB, k0, pp->winv, 1);
        aext_block(Ax + b*256, Ac + b*256, FT + b*BTAB, pp, vp, vp2);
    }

    /* ---- validate: tb == 16 * kara (mod p) ---- */
    for(size_t b = 0; b < 512; ++b){
        size_t k0 = 16 * (b & 255);
        uint32_t kb[256] __attribute__((aligned(64)));
        uint32_t tb[256] __attribute__((aligned(64)));
        memcpy(kb, Bb + b*256, sizeof kb);
        memcpy(tb, Bb + b*256, sizeof tb);
        for(size_t t = 0; t < 16; t += 2)
            p30_conv16_kara2(kb + 16*t, Ac + b*256 + 16*t,
                             pp->w[k0 + t], pp->w[k0 + t + 1], pp, vp);
        tb_block(tb, Ax + b*256, FT + b*BTAB, IT + b*BTAB, pp, vp, vp2);
        for(size_t i = 0; i < 256; ++i){
            uint32_t want = (uint32_t)((uint64_t)(kb[i] % pp->p) * 16u % pp->p);
            if(tb[i] % pp->p != want){
                printf("MISMATCH blk=%zu i=%zu got %u want %u\n",
                       b, i, tb[i] % pp->p, want);
                return 1;
            }
        }
    }
    printf("validation OK (tb == 16*kara mod p, 512 blocks)\n");

    /* ---- bench ---- */
    for(int big = 0; big < 2; ++big){
        size_t nb = big ? NBLK : 8;             /* 8 blocks: L1-hot */
        long reps = big ? 60 : 16000;
        double tk = 1e30, tt = 1e30, ta = 1e30;
        for(int pass = 0; pass < 9; ++pass){
            double t0 = now();
            for(long r = 0; r < reps; ++r)
                for(size_t b = 0; b < nb; ++b){
                    size_t k0 = 16 * (b & 255);
                    for(size_t t = 0; t < 16; t += 2)
                        p30_conv16_kara2(Bb + b*256 + 16*t, Ac + b*256 + 16*t,
                                         pp->w[k0 + t], pp->w[k0 + t + 1],
                                         pp, vp);
                }
            double t = (now() - t0) / (double)(reps * nb) * 1e9;
            SINK = Bb[xr() % (nb*256)];
            if(t < tk) tk = t;
        }
        for(int pass = 0; pass < 9; ++pass){
            double t0 = now();
            for(long r = 0; r < reps; ++r)
                for(size_t b = 0; b < nb; ++b)
                    tb_block(Bb + b*256, Ax + b*256,
                             FT + b*BTAB, IT + b*BTAB, pp, vp, vp2);
            double t = (now() - t0) / (double)(reps * nb) * 1e9;
            SINK = Bb[xr() % (nb*256)];
            if(t < tt) tt = t;
        }
        for(int pass = 0; pass < 9; ++pass){
            double t0 = now();
            for(long r = 0; r < reps; ++r)
                for(size_t b = 0; b < nb; ++b)
                    aext_block(Ax + b*256, Ac + b*256, FT + b*BTAB,
                               pp, vp, vp2);
            double t = (now() - t0) / (double)(reps * nb) * 1e9;
            SINK = Ax[xr() % (nb*256)];
            if(t < ta) ta = t;
        }
        printf("---- %s ----\n", big ? "streaming (2MB/side)" : "L1-hot (8 blocks)");
        printf("  kara mid    %8.2f ns/blk  %6.3f ns/pos\n", tk, tk / 16);
        printf("  tb mid      %8.2f ns/blk  %6.3f ns/pos\n", tt, tt / 16);
        printf("  a-ext       %8.2f ns/blk  %6.3f ns/pos\n", ta, ta / 16);
        printf("  tb+aext vs kara: %.3f x\n", tk / (tt + ta));
    }
    return 0;
}
