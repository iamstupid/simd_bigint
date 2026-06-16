// p30 leaf-layer microbench (stage-2 idea 1): fully in-register,
// unrolled leaf subtrees vs the current recursive bottoms.
//
// The engine's full-tree bottoms depend on lg mod 3 (radix-8 moths from
// the top leave leaf blocks of m = 8 / 16 / 32 vectors):
//   m =  8 (lg==0 mod 3): fwd8(e=1) + 4x kara2 + inv8(e=1)   [3 L1 trips]
//   m = 16 (lg==1):       moth(e=2) + 8x fused_rec(m=2)      [many calls,
//                         single-position schoolbook convs]
//   m = 32 (lg==2):       moth(e=4) + 8x fused_rec(m=4)
// Candidates keep the whole leaf block in registers: load once, run the
// remaining forward levels, kara-conv the register pairs against the
// a-spectrum, run the inverse levels, store once.
//
// Also: plain forward bottoms (fwd(a) side): p30_fwd_rec vs unrolled
// in-register m=16 / m=32 kernels.
//
// Build: clang -O3 -march=native -std=c11 -I../include p30_leaf_bench.c \
//          -lm -o /tmp/p30_leaf_bench
#define _POSIX_C_SOURCE 199309L
#define SCRATCH_IMPLEMENTATION
#include "p30_ref.h"
#include <stdio.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rngs = 42;
static uint64_t xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }

/* ------------------------------------------------------------------ */
/* register-input kara2: body of p30_conv16_kara2 minus the fa memory  */
/* round trip. xa0/xa1 arrive LAZY [0,4p); outputs are lazy (< 3p).    */
/* ------------------------------------------------------------------ */
static inline void kara2_reg(__m512i *pa0, __m512i *pa1, const uint32_t *fb,
                             p30_cc wwk, p30_cc wwk1,
                             const p30_prime *pp, __m512i vp){
    const __m512i vP2w = _mm512_set1_epi64(pp->p2);
    const __m512i vJ = _mm512_set1_epi64(pp->J);
    const __m512i vP = _mm512_set1_epi64(pp->p);
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    __m512i wc = _mm512_mask_mov_epi32(_mm512_set1_epi32((int)wwk.c),
                     (__mmask16)0xFF00, _mm512_set1_epi32((int)wwk1.c));
    __m512i wr = _mm512_mask_mov_epi32(_mm512_set1_epi32((int)wwk.rec),
                     (__mmask16)0xFF00, _mm512_set1_epi32((int)wwk1.rec));
    __m512i xa0 = p30_kshrp(p30_sh2(*pa0, vp2), vp);
    __m512i xa1 = p30_kshrp(p30_sh2(*pa1, vp2), vp);
    __m512i xb0 = _mm512_load_si512(fb);
    __m512i xb1 = _mm512_load_si512(fb + 16);
    __m512i AE = _mm512_permutex2var_epi32(xa0, P30_DEIL_E, xa1);
    __m512i AO = _mm512_permutex2var_epi32(xa0, P30_DEIL_O, xa1);
    __m512i BE = _mm512_permutex2var_epi32(xb0, P30_DEIL_E, xb1);
    __m512i BO = _mm512_permutex2var_epi32(xb0, P30_DEIL_O, xb1);
    __m512i AS = p30_kshrp(_mm512_add_epi32(AE, AO), vp);
    __m512i BS = p30_kshrp(_mm512_add_epi32(BE, BO), vp);
    __m512i AWE = p30_kshrp(p30_bar2p(AE, wc, wr, vp), vp);
    __m512i AWO = p30_kshrp(p30_bar2p(AO, wc, wr, vp), vp);
    __m512i AWS = p30_kshrp(p30_bar2p(AS, wc, wr, vp), vp);
    __m512i kb_tab[8];
    __m512i re0 = _mm512_setzero_si512(), ro0 = _mm512_setzero_si512();
    __m512i re2 = _mm512_setzero_si512(), ro2 = _mm512_setzero_si512();
    __m512i res = _mm512_setzero_si512(), ros = _mm512_setzero_si512();
#define KFILL(B) do{ \
        kb_tab[0] = P30_KBCAST(B, 0); kb_tab[1] = P30_KBCAST(B, 1);      \
        kb_tab[2] = P30_KBCAST(B, 2); kb_tab[3] = P30_KBCAST(B, 3);      \
        kb_tab[4] = P30_KBCAST(B, 4); kb_tab[5] = P30_KBCAST(B, 5);      \
        kb_tab[6] = P30_KBCAST(B, 6); kb_tab[7] = P30_KBCAST(B, 7);      \
    }while(0)
    KFILL(BE); P30_KCONV8(AE, AWE, BE, re0, ro0);
    KFILL(BO); P30_KCONV8(AO, AWO, BO, re2, ro2);
    KFILL(BS); P30_KCONV8(AS, AWS, BS, res, ros);
#undef KFILL
    __m512i m0 = p30_kshrp3(p30_kredc(re0, ro0, vJ, vP, vP2w), vp, vp2);
    __m512i m2 = p30_kshrp3(p30_kredc(re2, ro2, vJ, vP, vP2w), vp, vp2);
    __m512i ms = p30_kshrp3(p30_kredc(res, ros, vJ, vP, vP2w), vp, vp2);
    __m512i m1 = _mm512_sub_epi32(_mm512_add_epi32(ms, vp2),
                                  _mm512_add_epi32(m0, m2));
    __m512i awm2 = p30_bar2p(m2, wc, wr, vp);
    __m512i ym2 = _mm512_permutex2var_epi32(m2, P30_ROT1, awm2);
    __m512i CE = _mm512_add_epi32(m0, ym2);
    __m512i CO = m1;
    *pa0 = _mm512_permutex2var_epi32(CE, P30_ITLV_E, CO);
    *pa1 = _mm512_permutex2var_epi32(CE, P30_ITLV_O, CO);
}

/* in-register butterflies (range discipline identical to the moths) */
#define BF(lo_, hi_, T) do{                                              \
        __m512i a_  = p30_sh2(x[lo_], vp2);                              \
        __m512i wb_ = p30_bar2p(x[hi_], (T).c, (T).rec, vp);             \
        x[lo_] = _mm512_add_epi32(a_, wb_);                              \
        x[hi_] = _mm512_add_epi32(a_, _mm512_sub_epi32(vp2, wb_));       \
    }while(0)
#define IBF(lo_, hi_, T) do{                                             \
        __m512i a_ = p30_sh2(x[lo_], vp2);                               \
        __m512i b_ = p30_sh2(x[hi_], vp2);                               \
        x[lo_] = _mm512_add_epi32(a_, b_);                               \
        x[hi_] = p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a_, b_),    \
                                            vp2),                       \
                           (T).c, (T).rec, vp);                          \
    }while(0)
#define VCC(t) p30_vcc_of(t)

/* ==== fused leaf m=8: load once, 3 fwd lv, 4x kara, 3 inv lv ======== */
static void fused8_reg(uint32_t *B, const uint32_t *A, size_t j,
                       const p30_prime *pp, __m512i vp, __m512i vp2){
    __m512i x[8];
    for(int c = 0; c < 8; ++c) x[c] = _mm512_load_si512(B + 16 * c);
    {   p30_vcc t1  = VCC(pp->w[2*j]);
        BF(0,4,t1); BF(1,5,t1); BF(2,6,t1); BF(3,7,t1); }
    {   p30_vcc t2a = VCC(pp->w[4*j]), t2b = VCC(pp->w[4*j+2]);
        BF(0,2,t2a); BF(1,3,t2a); BF(4,6,t2b); BF(5,7,t2b); }
    {   p30_vcc t30 = VCC(pp->w[8*j]),   t31 = VCC(pp->w[8*j+2]);
        p30_vcc t32 = VCC(pp->w[8*j+4]), t33 = VCC(pp->w[8*j+6]);
        BF(0,1,t30); BF(2,3,t31); BF(4,5,t32); BF(6,7,t33); }
    for(int t = 0; t < 4; ++t)
        kara2_reg(&x[2*t], &x[2*t+1], A + 32*t,
                  pp->w[8*j + 2*t], pp->w[8*j + 2*t + 1], pp, vp);
    {   p30_vcc t30 = VCC(pp->winv[8*j]),   t31 = VCC(pp->winv[8*j+2]);
        p30_vcc t32 = VCC(pp->winv[8*j+4]), t33 = VCC(pp->winv[8*j+6]);
        IBF(0,1,t30); IBF(2,3,t31); IBF(4,5,t32); IBF(6,7,t33); }
    {   p30_vcc t2a = VCC(pp->winv[4*j]), t2b = VCC(pp->winv[4*j+2]);
        IBF(0,2,t2a); IBF(1,3,t2a); IBF(4,6,t2b); IBF(5,7,t2b); }
    {   p30_vcc t1  = VCC(pp->winv[2*j]);
        IBF(0,4,t1); IBF(1,5,t1); IBF(2,6,t1); IBF(3,7,t1); }
    for(int c = 0; c < 8; ++c) _mm512_store_si512(B + 16 * c, x[c]);
}

/* ==== fused leaf m=2 (child of an m=16 moth): 1 BF + kara2 + 1 IBF == */
static inline void fused2_reg(uint32_t *B, const uint32_t *A, size_t j,
                              const p30_prime *pp, __m512i vp, __m512i vp2){
    __m512i x[2];
    x[0] = _mm512_load_si512(B);
    x[1] = _mm512_load_si512(B + 16);
    {   p30_vcc t = VCC(pp->w[2*j]); BF(0,1,t); }
    kara2_reg(&x[0], &x[1], A, pp->w[2*j], pp->w[2*j+1], pp, vp);
    {   p30_vcc t = VCC(pp->winv[2*j]); IBF(0,1,t); }
    _mm512_store_si512(B,      x[0]);
    _mm512_store_si512(B + 16, x[1]);
}

/* ==== fused leaf m=4 (child of an m=32 moth): 2 lv + 2x kara + 2 lv = */
static inline void fused4_reg(uint32_t *B, const uint32_t *A, size_t j,
                              const p30_prime *pp, __m512i vp, __m512i vp2){
    __m512i x[4];
    for(int c = 0; c < 4; ++c) x[c] = _mm512_load_si512(B + 16 * c);
    {   p30_vcc t1 = VCC(pp->w[2*j]); BF(0,2,t1); BF(1,3,t1); }
    {   p30_vcc t2a = VCC(pp->w[4*j]), t2b = VCC(pp->w[4*j+2]);
        BF(0,1,t2a); BF(2,3,t2b); }
    kara2_reg(&x[0], &x[1], A,      pp->w[4*j],   pp->w[4*j+1], pp, vp);
    kara2_reg(&x[2], &x[3], A + 32, pp->w[4*j+2], pp->w[4*j+3], pp, vp);
    {   p30_vcc t2a = VCC(pp->winv[4*j]), t2b = VCC(pp->winv[4*j+2]);
        IBF(0,1,t2a); IBF(2,3,t2b); }
    {   p30_vcc t1 = VCC(pp->winv[2*j]); IBF(0,2,t1); IBF(1,3,t1); }
    for(int c = 0; c < 4; ++c) _mm512_store_si512(B + 16 * c, x[c]);
}

/* ==== fused m=16 variant A: moth passes + 8x fused2_reg ============= */
static void fused16_moth(uint32_t *B, const uint32_t *A, size_t j,
                         const p30_prime *pp, __m512i vp, __m512i vp2){
    p30_fwd8(B, 16, j, pp, vp, vp2);
    for(size_t c = 0; c < 8; ++c)
        fused2_reg(B + 32 * c, A + 32 * c, 8 * j + c, pp, vp, vp2);
    p30_inv8(B, 16, j, pp, vp, vp2);
}

/* ==== fused m=16 variant B: everything in (spilled) registers ======= */
static void fused16_reg(uint32_t *B, const uint32_t *A, size_t j,
                        const p30_prime *pp, __m512i vp, __m512i vp2){
    __m512i x[16];
    for(int c = 0; c < 16; ++c) x[c] = _mm512_load_si512(B + 16 * c);
    {   p30_vcc t = VCC(pp->w[2*j]);
        BF(0,8,t); BF(1,9,t); BF(2,10,t); BF(3,11,t);
        BF(4,12,t); BF(5,13,t); BF(6,14,t); BF(7,15,t); }
    {   p30_vcc a = VCC(pp->w[4*j]), b = VCC(pp->w[4*j+2]);
        BF(0,4,a); BF(1,5,a); BF(2,6,a); BF(3,7,a);
        BF(8,12,b); BF(9,13,b); BF(10,14,b); BF(11,15,b); }
    for(int c = 0; c < 4; ++c){
        p30_vcc t = VCC(pp->w[8*j + 2*(size_t)c]);
        BF(4*c, 4*c+2, t); BF(4*c+1, 4*c+3, t);
    }
    for(int c = 0; c < 8; ++c){
        p30_vcc t = VCC(pp->w[16*j + 2*(size_t)c]);
        BF(2*c, 2*c+1, t);
    }
    for(int t = 0; t < 8; ++t)
        kara2_reg(&x[2*t], &x[2*t+1], A + 32*t,
                  pp->w[16*j + 2*t], pp->w[16*j + 2*t + 1], pp, vp);
    for(int c = 0; c < 8; ++c){
        p30_vcc t = VCC(pp->winv[16*j + 2*(size_t)c]);
        IBF(2*c, 2*c+1, t);
    }
    for(int c = 0; c < 4; ++c){
        p30_vcc t = VCC(pp->winv[8*j + 2*(size_t)c]);
        IBF(4*c, 4*c+2, t); IBF(4*c+1, 4*c+3, t);
    }
    {   p30_vcc a = VCC(pp->winv[4*j]), b = VCC(pp->winv[4*j+2]);
        IBF(0,4,a); IBF(1,5,a); IBF(2,6,a); IBF(3,7,a);
        IBF(8,12,b); IBF(9,13,b); IBF(10,14,b); IBF(11,15,b); }
    {   p30_vcc t = VCC(pp->winv[2*j]);
        IBF(0,8,t); IBF(1,9,t); IBF(2,10,t); IBF(3,11,t);
        IBF(4,12,t); IBF(5,13,t); IBF(6,14,t); IBF(7,15,t); }
    for(int c = 0; c < 16; ++c) _mm512_store_si512(B + 16 * c, x[c]);
}

/* ==== fused m=32: moth passes + 8x fused4_reg ======================= */
static void fused32_moth(uint32_t *B, const uint32_t *A, size_t j,
                         const p30_prime *pp, __m512i vp, __m512i vp2){
    p30_fwd8(B, 32, j, pp, vp, vp2);
    for(size_t c = 0; c < 8; ++c)
        fused4_reg(B + 64 * c, A + 64 * c, 8 * j + c, pp, vp, vp2);
    p30_inv8(B, 32, j, pp, vp, vp2);
}

/* ==== fused m=32 variant B: radix-2 seam + two in-register 16s ====== */
static void fused32_r2(uint32_t *B, const uint32_t *A, size_t j,
                       const p30_prime *pp, __m512i vp, __m512i vp2){
    p30_vcc w = VCC(pp->w[2*j]);
    for(size_t v = 0; v < 16; ++v){
        __m512i a  = p30_sh2(_mm512_load_si512(B + 16 * v), vp2);
        __m512i wb = p30_bar2p(_mm512_load_si512(B + 16 * (v + 16)),
                               w.c, w.rec, vp);
        _mm512_store_si512(B + 16 * v, _mm512_add_epi32(a, wb));
        _mm512_store_si512(B + 16 * (v + 16),
            _mm512_add_epi32(a, _mm512_sub_epi32(vp2, wb)));
    }
    fused16_reg(B,       A,       2*j,     pp, vp, vp2);
    fused16_reg(B + 256, A + 256, 2*j + 1, pp, vp, vp2);
    p30_vcc wi = VCC(pp->winv[2*j]);
    for(size_t v = 0; v < 16; ++v){
        __m512i a = p30_sh2(_mm512_load_si512(B + 16 * v), vp2);
        __m512i b = p30_sh2(_mm512_load_si512(B + 16 * (v + 16)), vp2);
        _mm512_store_si512(B + 16 * v, _mm512_add_epi32(a, b));
        _mm512_store_si512(B + 16 * (v + 16),
            p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a, b), vp2),
                      wi.c, wi.rec, vp));
    }
}

/* ==== split leaves: new fast bottoms, conv from memory ============== */
static void fwd16_reg(uint32_t*, size_t, const p30_prime*, __m512i, __m512i);
static void fwd32_r2(uint32_t*, size_t, const p30_prime*, __m512i, __m512i);
static void inv16_reg(uint32_t*, size_t, const p30_prime*, __m512i, __m512i);
static void fused16_split(uint32_t *B, const uint32_t *A, size_t j,
                          const p30_prime *pp, __m512i vp, __m512i vp2){
    fwd16_reg(B, j, pp, vp, vp2);
    for(size_t t = 0; t < 8; ++t)
        p30_conv16_kara2(B + 32*t, A + 32*t,
                         pp->w[16*j + 2*t], pp->w[16*j + 2*t + 1], pp, vp);
    inv16_reg(B, j, pp, vp, vp2);
}
static void fused32_split(uint32_t *B, const uint32_t *A, size_t j,
                          const p30_prime *pp, __m512i vp, __m512i vp2){
    fwd32_r2(B, j, pp, vp, vp2);
    for(size_t t = 0; t < 16; ++t)
        p30_conv16_kara2(B + 32*t, A + 32*t,
                         pp->w[32*j + 2*t], pp->w[32*j + 2*t + 1], pp, vp);
    /* inv32 = 2x inv16_reg + inverse seam */
    inv16_reg(B,       2*j,     pp, vp, vp2);
    inv16_reg(B + 256, 2*j + 1, pp, vp, vp2);
    p30_vcc wi = VCC(pp->winv[2*j]);
    for(size_t v = 0; v < 16; ++v){
        __m512i a = p30_sh2(_mm512_load_si512(B + 16 * v), vp2);
        __m512i b = p30_sh2(_mm512_load_si512(B + 16 * (v + 16)), vp2);
        _mm512_store_si512(B + 16 * v, _mm512_add_epi32(a, b));
        _mm512_store_si512(B + 16 * (v + 16),
            p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a, b), vp2),
                      wi.c, wi.rec, vp));
    }
}

/* ==== plain forward bottoms ========================================= */
static void fwd16_reg(uint32_t *B, size_t j,
                      const p30_prime *pp, __m512i vp, __m512i vp2){
    __m512i x[16];
    for(int c = 0; c < 16; ++c) x[c] = _mm512_load_si512(B + 16 * c);
    {   p30_vcc t = VCC(pp->w[2*j]);
        BF(0,8,t); BF(1,9,t); BF(2,10,t); BF(3,11,t);
        BF(4,12,t); BF(5,13,t); BF(6,14,t); BF(7,15,t); }
    {   p30_vcc a = VCC(pp->w[4*j]), b = VCC(pp->w[4*j+2]);
        BF(0,4,a); BF(1,5,a); BF(2,6,a); BF(3,7,a);
        BF(8,12,b); BF(9,13,b); BF(10,14,b); BF(11,15,b); }
    for(int c = 0; c < 4; ++c){
        p30_vcc t = VCC(pp->w[8*j + 2*(size_t)c]);
        BF(4*c, 4*c+2, t); BF(4*c+1, 4*c+3, t);
    }
    for(int c = 0; c < 8; ++c){
        p30_vcc t = VCC(pp->w[16*j + 2*(size_t)c]);
        BF(2*c, 2*c+1, t);
    }
    for(int c = 0; c < 16; ++c) _mm512_store_si512(B + 16 * c, x[c]);
}
static inline void fwd4_reg(uint32_t *B, size_t j,
                            const p30_prime *pp, __m512i vp, __m512i vp2){
    __m512i x[4];
    for(int c = 0; c < 4; ++c) x[c] = _mm512_load_si512(B + 16 * c);
    {   p30_vcc t1 = VCC(pp->w[2*j]); BF(0,2,t1); BF(1,3,t1); }
    {   p30_vcc t2a = VCC(pp->w[4*j]), t2b = VCC(pp->w[4*j+2]);
        BF(0,1,t2a); BF(2,3,t2b); }
    for(int c = 0; c < 4; ++c) _mm512_store_si512(B + 16 * c, x[c]);
}
/* m=32: one radix-2 memory level + two in-register 16s */
static void fwd32_r2(uint32_t *B, size_t j,
                     const p30_prime *pp, __m512i vp, __m512i vp2){
    p30_vcc w = VCC(pp->w[2*j]);
    for(size_t v = 0; v < 16; ++v){
        __m512i a  = p30_sh2(_mm512_load_si512(B + 16 * v), vp2);
        __m512i wb = p30_bar2p(_mm512_load_si512(B + 16 * (v + 16)),
                               w.c, w.rec, vp);
        _mm512_store_si512(B + 16 * v, _mm512_add_epi32(a, wb));
        _mm512_store_si512(B + 16 * (v + 16),
            _mm512_add_epi32(a, _mm512_sub_epi32(vp2, wb)));
    }
    fwd16_reg(B, 2*j, pp, vp, vp2);
    fwd16_reg(B + 256, 2*j + 1, pp, vp, vp2);
}
/* m=32: moth + 8x in-register 4s */
static void fwd32_moth(uint32_t *B, size_t j,
                       const p30_prime *pp, __m512i vp, __m512i vp2){
    p30_fwd8(B, 32, j, pp, vp, vp2);
    for(size_t c = 0; c < 8; ++c)
        fwd4_reg(B + 64 * c, 8 * j + c, pp, vp, vp2);
}
/* inverse m=16 in-register */
static void inv16_reg(uint32_t *B, size_t j,
                      const p30_prime *pp, __m512i vp, __m512i vp2){
    __m512i x[16];
    for(int c = 0; c < 16; ++c) x[c] = _mm512_load_si512(B + 16 * c);
    for(int c = 0; c < 8; ++c){
        p30_vcc t = VCC(pp->winv[16*j + 2*(size_t)c]);
        IBF(2*c, 2*c+1, t);
    }
    for(int c = 0; c < 4; ++c){
        p30_vcc t = VCC(pp->winv[8*j + 2*(size_t)c]);
        IBF(4*c, 4*c+2, t); IBF(4*c+1, 4*c+3, t);
    }
    {   p30_vcc a = VCC(pp->winv[4*j]), b = VCC(pp->winv[4*j+2]);
        IBF(0,4,a); IBF(1,5,a); IBF(2,6,a); IBF(3,7,a);
        IBF(8,12,b); IBF(9,13,b); IBF(10,14,b); IBF(11,15,b); }
    {   p30_vcc t = VCC(pp->winv[2*j]);
        IBF(0,8,t); IBF(1,9,t); IBF(2,10,t); IBF(3,11,t);
        IBF(4,12,t); IBF(5,13,t); IBF(6,14,t); IBF(7,15,t); }
    for(int c = 0; c < 16; ++c) _mm512_store_si512(B + 16 * c, x[c]);
}

/* ------------------------------------------------------------------ */
/* harness                                                             */
/* ------------------------------------------------------------------ */
typedef void (*fused_fn)(uint32_t*, const uint32_t*, size_t,
                         const p30_prime*, __m512i, __m512i);
static const p30_prime *PP;
static __m512i VP, VP2;
static volatile uint32_t SINK;

static int validate_fused(size_t m, fused_fn cand, const char *name){
    static uint32_t b0[32 * 16], b1[32 * 16], aa[32 * 16];
    for(size_t jt = 0; jt < 512; ++jt){
        size_t j = (jt * 37 + jt) & 1023;
        for(size_t i = 0; i < m * 16; ++i){
            b0[i] = (uint32_t)(xr() % (2u * PP->p));     /* lazy entry */
            aa[i] = (uint32_t)(xr() % PP->p);            /* canonical  */
        }
        memcpy(b1, b0, m * 16 * 4);
        p30_fused_rec(b0, aa, m, j, m, PP, VP);
        cand(b1, aa, j, PP, VP, VP2);
        for(size_t i = 0; i < m * 16; ++i)
            if(b0[i] % PP->p != b1[i] % PP->p){
                printf("%s FAIL m=%zu j=%zu i=%zu got %u want %u\n",
                       name, m, j, i, b1[i] % PP->p, b0[i] % PP->p);
                return 0;
            }
    }
    return 1;
}

static void eng_fused_m(uint32_t *B, const uint32_t *A, size_t m, size_t j){
    p30_fused_rec(B, A, m, j, m, PP, VP);
}

/* timing: NB blocks of m vectors, repeatedly processed in place
 * (outputs stay in [0,4p), valid re-entry). kind: 0 = engine fused_rec,
 * 1 = candidate. */
static double bench_fused(size_t m, size_t nb, long reps, int kind,
                          fused_fn cand, uint32_t *B, const uint32_t *A){
    double best = 1e30;
    for(int pass = 0; pass < 9; ++pass){
        double t0 = now();
        for(long r = 0; r < reps; ++r)
            for(size_t k = 0; k < nb; ++k){
                size_t j = k & 1023;
                if(kind == 0) eng_fused_m(B + m * 16 * k, A + m * 16 * k, m, j);
                else cand(B + m * 16 * k, A + m * 16 * k, j, PP, VP, VP2);
            }
        double t = (now() - t0) / (double)(reps * nb) * 1e9;
        SINK = B[xr() % (m * 16 * nb)];
        if(t < best) best = t;
    }
    return best;
}

typedef void (*fwd_fn)(uint32_t*, size_t, const p30_prime*, __m512i, __m512i);
static int validate_fwd(size_t m, fwd_fn cand, const char *name){
    static uint32_t b0[32 * 16], b1[32 * 16];
    for(size_t jt = 0; jt < 512; ++jt){
        size_t j = (jt * 37 + jt) & 1023;
        for(size_t i = 0; i < m * 16; ++i)
            b0[i] = (uint32_t)(xr() % (2u * PP->p));
        memcpy(b1, b0, m * 16 * 4);
        p30_fwd_rec(b0, m, j, m, PP, VP);
        cand(b1, j, PP, VP, VP2);
        for(size_t i = 0; i < m * 16; ++i)
            if(b0[i] % PP->p != b1[i] % PP->p){
                printf("%s FAIL m=%zu j=%zu i=%zu\n", name, m, j, i);
                return 0;
            }
    }
    return 1;
}
static int validate_inv(size_t m, fwd_fn cand, const char *name){
    static uint32_t b0[32 * 16], b1[32 * 16];
    for(size_t jt = 0; jt < 512; ++jt){
        size_t j = (jt * 37 + jt) & 1023;
        for(size_t i = 0; i < m * 16; ++i)
            b0[i] = (uint32_t)(xr() % (2u * PP->p));
        memcpy(b1, b0, m * 16 * 4);
        p30_inv_rec(b0, m, j, m, PP, VP);
        cand(b1, j, PP, VP, VP2);
        for(size_t i = 0; i < m * 16; ++i)
            if(b0[i] % PP->p != b1[i] % PP->p){
                printf("%s FAIL m=%zu j=%zu i=%zu\n", name, m, j, i);
                return 0;
            }
    }
    return 1;
}
static void eng_fwd_m(uint32_t *B, size_t m, size_t j){
    p30_fwd_rec(B, m, j, m, PP, VP);
}
static void eng_inv_m(uint32_t *B, size_t m, size_t j){
    p30_inv_rec(B, m, j, m, PP, VP);
}
static double bench_fwd(size_t m, size_t nb, long reps, int kind, int inv,
                        fwd_fn cand, uint32_t *B){
    double best = 1e30;
    for(int pass = 0; pass < 9; ++pass){
        double t0 = now();
        for(long r = 0; r < reps; ++r)
            for(size_t k = 0; k < nb; ++k){
                size_t j = k & 1023;
                if(kind == 0){
                    if(inv) eng_inv_m(B + m * 16 * k, m, j);
                    else    eng_fwd_m(B + m * 16 * k, m, j);
                }
                else cand(B + m * 16 * k, j, PP, VP, VP2);
            }
        double t = (now() - t0) / (double)(reps * nb) * 1e9;
        SINK = B[xr() % (m * 16 * nb)];
        if(t < best) best = t;
    }
    return best;
}

int main(void){
    p30_plan *pl = &p30_tls_plan;
    p30_plan_ensure(pl, 16);
    PP = &pl->b.pr[4];
    VP = _mm512_set1_epi32((int)PP->p);
    VP2 = _mm512_add_epi32(VP, VP);

    int ok = 1;
    ok &= validate_fused(8,  fused8_reg,   "fused8_reg");
    ok &= validate_fused(16, fused16_moth, "fused16_moth");
    ok &= validate_fused(16, fused16_reg,  "fused16_reg");
    ok &= validate_fused(32, fused32_moth, "fused32_moth");
    ok &= validate_fused(32, fused32_r2,   "fused32_r2");
    ok &= validate_fused(16, fused16_split, "fused16_split");
    ok &= validate_fused(32, fused32_split, "fused32_split");
    ok &= validate_fwd(16, fwd16_reg,  "fwd16_reg");
    ok &= validate_fwd(32, fwd32_r2,   "fwd32_r2");
    ok &= validate_fwd(32, fwd32_moth, "fwd32_moth");
    ok &= validate_inv(16, inv16_reg,  "inv16_reg");
    if(!ok) return 1;
    printf("validation OK\n");

    /* working sets: L1-hot (B+A ~ 32KB) and L2 (B+A ~ 4MB) */
    enum { BIGV = 1 << 19 };                 /* u32s: 2MB per array */
    static uint32_t Bb[BIGV] __attribute__((aligned(64)));
    static uint32_t Aa[BIGV] __attribute__((aligned(64)));
    for(size_t i = 0; i < BIGV; ++i){
        Bb[i] = (uint32_t)(xr() % (2u * PP->p));
        Aa[i] = (uint32_t)(xr() % PP->p);
    }

    for(int big = 0; big < 2; ++big){
        size_t setv = big ? BIGV : (1 << 12);          /* u32s in play */
        long reps;
        printf("---- %s ----\n", big ? "L2-resident (2MB+2MB)"
                                     : "L1-hot (16KB+16KB)");
        struct { const char *name; size_t m; fused_fn fn; } FU[] = {
            { "fused8  engine", 8,  NULL },
            { "fused8  reg   ", 8,  fused8_reg },
            { "fused16 engine", 16, NULL },
            { "fused16 moth  ", 16, fused16_moth },
            { "fused16 reg   ", 16, fused16_reg },
            { "fused32 engine", 32, NULL },
            { "fused32 moth  ", 32, fused32_moth },
            { "fused32 r2+16r", 32, fused32_r2 },
            { "fused16 split ", 16, fused16_split },
            { "fused32 split ", 32, fused32_split },
        };
        for(size_t i = 0; i < sizeof FU / sizeof *FU; ++i){
            size_t nb = setv / (FU[i].m * 16);
            reps = (big ? 40 : 4000);
            double t = bench_fused(FU[i].m, nb, reps,
                                   FU[i].fn ? 1 : 0, FU[i].fn, Bb, Aa);
            printf("  %s  %8.2f ns/blk  %6.3f ns/pt\n",
                   FU[i].name, t, t / (double)(FU[i].m * 16));
        }
        struct { const char *name; size_t m; int inv; fwd_fn fn; } FW[] = {
            { "fwd16   engine", 16, 0, NULL },
            { "fwd16   reg   ", 16, 0, fwd16_reg },
            { "fwd32   engine", 32, 0, NULL },
            { "fwd32   r2+16r", 32, 0, fwd32_r2 },
            { "fwd32   moth+4", 32, 0, fwd32_moth },
            { "inv16   engine", 16, 1, NULL },
            { "inv16   reg   ", 16, 1, inv16_reg },
        };
        for(size_t i = 0; i < sizeof FW / sizeof *FW; ++i){
            size_t nb = setv / (FW[i].m * 16);
            reps = (big ? 80 : 8000);
            double t = bench_fwd(FW[i].m, nb, reps,
                                 FW[i].fn ? 1 : 0, FW[i].inv, FW[i].fn, Bb);
            printf("  %s  %8.2f ns/blk  %6.3f ns/pt\n",
                   FW[i].name, t, t / (double)(FW[i].m * 16));
        }
    }
    return 0;
}
