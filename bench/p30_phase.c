// phase profiler for the CURRENT engine (mirrors p30_mul_r's body)
#define _POSIX_C_SOURCE 199309L
#define SCRATCH_IMPLEMENTATION
#include "p30.h"
#include <stdio.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }
static uint64_t rngs = 42;
static uint64_t xr(void){ rngs ^= rngs<<13; rngs ^= rngs>>7; rngs ^= rngs<<17; return rngs; }

int main(void){
    size_t ns[3] = { 32768, 262144, 2097152 };
    enum { MAXN = 1<<21 };
    static uint64_t A[MAXN], B[MAXN];
    for(size_t i = 0; i < MAXN; ++i){ A[i] = xr(); B[i] = xr(); }
    p30_plan *pl = &p30_tls_plan;
    p30_plan_init(pl);
    for(int k = 0; k < 3; ++k){
        size_t n = ns[k];
        uint64_t zlen = 2*n - 1;
        uint64_t tv = (zlen + 15)/16;
        int lg = 0; while(((uint64_t)1<<lg) < tv) lg++;
        size_t M = (size_t)1 << lg;
        p30_pset_ensure(&pl->b, lg);
        const p30_pset *ps = &pl->b;
        scratch *sc = scratch_thread();
        SCRATCH(sc);
        size_t bv = M*16 + 64;
        uint32_t *blk = SALLOC(sc, uint32_t, 10*bv);
        uint32_t *ra[5], *rb[5];
        for(int q = 0; q < 5; ++q){ ra[q] = blk + (size_t)(2*q)*bv; rb[q] = blk + (size_t)(2*q+1)*bv; }
        uint64_t *st = SALLOC(sc, uint64_t, zlen + 2 + 8);
        int fin = (M >= P30_FIN_MIN_M);
        int nt = fin && (M >= P30_FIN_NT_MIN_M);
        double t_in=1e30, t_fa=1e30, t_fb=1e30, t_sc=1e30, t_out=1e30;
        int reps = n > (1<<19) ? 2 : 6;
        for(int pass = 0; pass < 5; ++pass){
            double t0, t1;
            t0 = now();
            for(int r = 0; r < reps; ++r){
                if(fin){
                    size_t zrow = (tv + (M>>3) - 1)/(M>>3);
                    p30_input_f3(ra, A, n, M, zrow, ps, nt);
                    p30_input_f3(rb, B, n, M, zrow, ps, nt);
                }else{ p30_input(ra, A, n, M, ps); p30_input(rb, B, n, M, ps); }
            }
            t1 = now(); if((t1-t0)/reps < t_in) t_in = (t1-t0)/reps;
            t0 = now();
            for(int r = 0; r < reps; ++r)
                for(int q = 0; q < 5; ++q){
                    const p30_prime *pp = &ps->pr[q];
                    __m512i vp = _mm512_set1_epi32((int)pp->p);
                    if(fin) p30_fwd_pre(ra[q], M, 0, tv, 3, pp, vp);
                    else    p30_fwd_rec(ra[q], M, 0, tv, pp, vp);
                }
            t1 = now(); if((t1-t0)/reps < t_fa) t_fa = (t1-t0)/reps;
            t0 = now();
            for(int r = 0; r < reps; ++r)
                for(int q = 0; q < 5; ++q){
                    const p30_prime *pp = &ps->pr[q];
                    __m512i vp = _mm512_set1_epi32((int)pp->p);
                    if(fin) p30_fused_pre(rb[q], ra[q], M, 0, tv, 3, pp, vp);
                    else    p30_fused_rec(rb[q], ra[q], M, 0, tv, pp, vp);
                }
            t1 = now(); if((t1-t0)/reps < t_fb) t_fb = (t1-t0)/reps;
            t0 = now();
            for(int r = 0; r < reps; ++r)
                for(int q = 0; q < 5; ++q){
                    const p30_prime *pp = &ps->pr[q];
                    __m512i vp = _mm512_set1_epi32((int)pp->p);
                    uint32_t sc32 = p30_mulm(p30_invm((uint32_t)(M % pp->p), pp->p), pp->R1, pp->p);
                    p30_vcc vs = p30_vcc_of(p30_cc_of(sc32, pp->p));
                    for(size_t kk = 0; kk < tv; ++kk){
                        uint32_t *v = rb[q] + 16*kk;
                        _mm512_store_si512(v, p30_barm(_mm512_load_si512(v), vs.c, vs.rec, vp));
                    }
                }
            t1 = now(); if((t1-t0)/reps < t_sc) t_sc = (t1-t0)/reps;
            t0 = now();
            for(int r = 0; r < reps; ++r) p30_output(st, rb, zlen, ps);
            t1 = now(); if((t1-t0)/reps < t_out) t_out = (t1-t0)/reps;
        }
        double s = 1e9/(double)n;
        printf("n=%7zu (M=2^%d%s%s): in %5.2f | fwd(a) %5.2f | fused(b) %5.2f | scale %5.2f | out %5.2f | sum %5.2f ns/limb\n",
               n, lg, fin?" f3":"", nt?"+NT":"",
               t_in*s, t_fa*s, t_fb*s, t_sc*s, t_out*s,
               (t_in+t_fa+t_fb+t_sc+t_out)*s);
    }
    return 0;
}
