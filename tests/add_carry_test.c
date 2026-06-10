/* Fuzz u52_add_carry against a scalar base-2^52 reference.
 * Build: clang -O2 -march=native -std=gnu11 -I../include add_carry_test.c -o /tmp/act && /tmp/act
 */
#include "types.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* borrow/carry machinery copied from canon.h (avoids the canon_pos defs). */
#define MASK52 set1_64((1ull<<52)-1)
#define carry_prop_t(add, sub, prp, r, a, b, carry, ...) { \
    _vec res = add(a, b); __mmask16 _prop = eq(res, prp); __mmask16 _carr = gtu(res, MASK52); \
    carry |= _carr << 1; carry += _prop; _prop ^= carry; carry >>=8; \
    res = and_v(sub(res, MASK52, _prop, res), MASK52); store_vec(r++, res, ##__VA_ARGS__); }
#define carry_prop(r, a, b, carry, ...) carry_prop_t(add, sub, MASK52, r, a, b, carry, ##__VA_ARGS__)

#define ULL unsigned long long
#include "add_carry_under_test.inc"   /* the function being tested */

#define LIMB_MASK ((1ull<<52)-1)
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return rng; }
static uint64_t rlimb(void){
    uint64_t v = xr() & LIMB_MASK;
    switch(xr()&7){ case 0: return 0; case 1: return LIMB_MASK; case 2: return LIMB_MASK-1; default: return v; }
}
static uint64_t* mk(const uint64_t*s,uint64_t n){
    uint64_t vecs=(n+7)/8; uint64_t*p=(uint64_t*)calloc(vecs*8+16,8); memcpy(p,s,n*8); return p;
}
/* out = a + b, base 2^52, length max(na,nb); returns carry-out (0/1). */
static uint64_t ref_add(uint64_t*out,const uint64_t*A,uint64_t na,const uint64_t*B,uint64_t nb){
    uint64_t n = na>nb?na:nb, c=0;
    for(uint64_t i=0;i<n;i++){
        uint64_t x=i<na?A[i]:0, y=i<nb?B[i]:0, s=x+y+c;
        out[i]=s & LIMB_MASK; c = s>>52;
    }
    return c;
}

int main(void){
    const int ITERS=2000000;
    uint64_t A[64],B[64],ref[80],got[88]; int fails=0;
    for(int it=0; it<ITERS; ++it){
        uint64_t na=1+(xr()%40), nb=1+(xr()%40);
        for(uint64_t i=0;i<na;i++) A[i]=rlimb();
        for(uint64_t i=0;i<nb;i++) B[i]=rlimb();
        bool extra = xr()&1;
        uint64_t mx = na>nb?na:nb;
        uint64_t rc = ref_add(ref,A,na,B,nb);

        uint64_t *ap=mk(A,na), *bp=mk(B,nb);
        memset(got,0,sizeof(got));
        uint64_t L = u52_add_carry((pvec)got,(cpvec)ap,(cpvec)bp,na,nb,extra);

        int ok=1;
        uint64_t expL = extra ? mx+1 : mx;
        if(L != expL) ok=0;
        for(uint64_t i=0;i<mx;i++) if(got[i]!=ref[i]) ok=0;       // low limbs
        if(extra){ if(got[mx]!=rc) ok=0; }                        // carry limb stored
        /* when !extra, carry rc may be 1 and is correctly dropped (length mx) */
        if(!ok){
            if(fails<8) printf("FAIL it=%d na=%llu nb=%llu extra=%d L=%llu expL=%llu rc=%llu\n",
                it,(ULL)na,(ULL)nb,extra,(ULL)L,(ULL)expL,(ULL)rc);
            fails++;
        }
        free(ap); free(bp);
    }
    if(fails){ printf("FAILED: %d\n",fails); return 1; }
    printf("OK: %d iterations match scalar reference\n", ITERS);
    return 0;
}
