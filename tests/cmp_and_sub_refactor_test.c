/* Standalone equivalence + correctness fuzz for u52_cmp_and_sub.
 *
 * Compares:
 *   - cmp_and_sub_orig : verbatim copy of the current mainstream function
 *   - cmp_and_sub_new  : proposed cleaned-up compare/setup logic
 * against a plain scalar base-2^52 reference, and against each other.
 *
 * Build: gcc -O2 -march=native -std=gnu11 -I../include \
 *            cmp_and_sub_refactor_test.c -o /tmp/cmptest && /tmp/cmptest
 */
#include "types.h"
#include <assert.h>

/* borrow_prop machinery, copied verbatim from canon.h (avoids pulling in the
 * canon_pos function definitions, which need C++ for their static-const init). */
#define MASK52 set1_64((1ull<<52)-1)
#define carry_prop_t(add, sub, prp, r, a, b, carry, ...) { \
    _vec res = add(a, b); \
    __mmask16 _prop = eq(res, prp); \
    __mmask16 _carr = gtu(res, MASK52); \
    carry |= _carr << 1; \
    carry += _prop; \
    _prop ^= carry; \
    carry >>=8; \
    res = and_v(sub(res, MASK52, _prop, res), MASK52); \
    store_vec(r++, res, ##__VA_ARGS__); \
}
#define borrow_prop(r, a, b, carry, ...) carry_prop_t(sub, add, zero(), r, a, b, carry, ##__VA_ARGS__)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- original (verbatim, renamed) ---------------- */
static inline int64_t cmp_and_sub_orig(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb) {
    assert(na > 0);
    assert(nb > 0);

    uint8_t sig = 0;

    if(na < nb){
        cpvec t = a; a = b; b = t;
        uint64_t nt = na; na = nb; nb = nt;
        sig = 1;
    }
    if(!na){
        return 0;
    }
    uint8_t at = --na & 7, bt = --nb & 7;
    __mmask8 am = (1<<(at+1)) - 1, bm = (1<<(bt+1)) - 1;
    _vec ax, bx;
    na >>= 3, nb >>= 3;
    cpvec ap = a + na, bp = b + nb;

    for(ax = load_vec(ap, am); na > nb; --na, ax = load_vec(--ap)){
        if(neq(ax, zero())) goto compared;
        at = 7;
    }

    for(bx = load_vec(bp, bm); ~na; --na, --nb){
        const uint8_t neq_r = (uint8_t)neq(ax, bx);
        if(neq_r){
            const uint8_t lt_r = (uint8_t)ltu(ax, bx);
            const uint8_t gt_r = neq_r ^ lt_r;
            if(lt_r > gt_r){
                cpvec t = a; a = b; b = t;
                uint64_t nt = na; na = nb; nb = nt;
                uint8_t mt = at; at = bt; bt = mt;
                sig ^= 1;
            }
            goto compared;
        }
        if(!na) break;
        ax = load_vec(--ap);
        bx = load_vec(--bp);
        at = bt = 7;
    }
    return 0;
compared:
    {
        am = (1<<(at+1)) - 1, bm = (1<<(bt+1)) - 1;
        uint64_t i;
        uint16_t lbr = 0;
        for(i = 0; i < nb; ++i){
            ax = load_vec(a++), bx = load_vec(b++);
            borrow_prop(r, ax, bx, lbr);
        }
        if(na == nb){
            ax = load_vec(a, am), bx = load_vec(b, bm);
            borrow_prop(r, ax, bx, lbr, am);
        }else{
            ax = load_vec(a++), bx = load_vec(b, bm);
            borrow_prop(r, ax, bx, lbr);
            for(++i;i<na;++i){
                ax = load_vec(a++);
                borrow_prop(r, ax, zero(), lbr);
            }
            ax = load_vec(a, am);
            borrow_prop(r, ax, zero(), lbr, am);
        }
        return sig ? -(int64_t)(na*8+at+1) : (int64_t)(na*8+at+1);
    }
}

/* ---------------- proposed refactor ---------------- */
#define SWAP(T,x,y) do{ T _s=(x); (x)=(y); (y)=_s; }while(0)

static inline int64_t cmp_and_sub_new(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb) {
    assert(na > 0 && nb > 0);

    uint8_t sig = 0;
    if(na < nb){ SWAP(cpvec,a,b); SWAP(uint64_t,na,nb); sig = 1; }

    --na; --nb;
    uint8_t at = na & 7, bt = nb & 7;
    __mmask8 am = (__mmask8)((1u<<(at+1)) - 1);
    __mmask8 bm = (__mmask8)((1u<<(bt+1)) - 1);
    na >>= 3; nb >>= 3;
    cpvec ap = a + na, bp = b + nb;
    _vec ax, bx;

    /* skip a's vectors that sit above b's range (compared against zero) */
    for(ax = load_vec(ap, am); na > nb; --na, ax = load_vec(--ap)){
        if(neq(ax, zero())) goto compared;   /* a is strictly longer -> a > b */
        at = 7;
    }

    /* same vector index now: walk down comparing a vs b */
    for(bx = load_vec(bp, bm); ; --na, --nb){
        const uint8_t ne = (uint8_t)neq(ax, bx);
        if(ne){
            const uint8_t lt = (uint8_t)ltu(ax, bx);
            /* lt and gt bitmasks are disjoint, so lt>gt iff the most
               significant differing lane is one where a < b */
            if(lt > (uint8_t)(ne ^ lt)){
                SWAP(cpvec,a,b); SWAP(uint64_t,na,nb); SWAP(uint8_t,at,bt);
                sig ^= 1;
            }
            goto compared;
        }
        if(!na) return 0;                     /* every lane equal -> a == b */
        ax = load_vec(--ap);
        bx = load_vec(--bp);
        at = bt = 7;
    }
compared:
    {
        am = (__mmask8)((1u<<(at+1)) - 1);
        bm = (__mmask8)((1u<<(bt+1)) - 1);
        uint64_t i;
        uint16_t lbr = 0;
        for(i = 0; i < nb; ++i){
            ax = load_vec(a++), bx = load_vec(b++);
            borrow_prop(r, ax, bx, lbr);
        }
        if(na == nb){
            ax = load_vec(a, am), bx = load_vec(b, bm);
            borrow_prop(r, ax, bx, lbr, am);
        }else{
            ax = load_vec(a++), bx = load_vec(b, bm);
            borrow_prop(r, ax, bx, lbr);
            for(++i;i<na;++i){
                ax = load_vec(a++);
                borrow_prop(r, ax, zero(), lbr);
            }
            ax = load_vec(a, am);
            borrow_prop(r, ax, zero(), lbr, am);
        }
        return sig ? -(int64_t)(na*8+at+1) : (int64_t)(na*8+at+1);
    }
}
#undef SWAP

/* ---------------- scalar reference ---------------- */
#define LIMB_MASK ((1ull<<52)-1)

static int ref_cmp(const uint64_t*A,uint64_t na,const uint64_t*B,uint64_t nb){
    uint64_t n = na>nb?na:nb;
    for(uint64_t i=n; i-->0;){
        uint64_t x = i<na?A[i]:0, y = i<nb?B[i]:0;
        if(x!=y) return x<y? -1 : 1;
    }
    return 0;
}
/* out = |A-B|, A assumed >= B; returns true #limbs (0 if zero). */
static uint64_t ref_sub(uint64_t*out,const uint64_t*A,uint64_t na,const uint64_t*B,uint64_t nb){
    int64_t borrow=0; uint64_t len=0;
    for(uint64_t i=0;i<na;i++){
        int64_t x=(int64_t)A[i], y=(int64_t)(i<nb?B[i]:0);
        int64_t d = x - y - borrow;
        if(d<0){ d += (1ll<<52); borrow=1; } else borrow=0;
        out[i]=(uint64_t)d;
        if(out[i]) len=i+1;
    }
    return len;
}

static uint64_t rng_state = 0x123456789abcdef0ull;
static uint64_t xrand(void){
    rng_state ^= rng_state<<13; rng_state ^= rng_state>>7; rng_state ^= rng_state<<17;
    return rng_state;
}
static uint64_t rand_limb(void){
    uint64_t v = xrand() & LIMB_MASK;
    /* bias toward extremes to exercise borrows */
    switch(xrand()&7){ case 0: return 0; case 1: return LIMB_MASK; case 2: return 1; default: return v; }
}

/* pad n limbs into a fresh vector-sized zeroed buffer */
static uint64_t* mk(const uint64_t*src,uint64_t n){
    uint64_t vecs = (n+7)/8;
    uint64_t *p = (uint64_t*)calloc(vecs*8 + 16, sizeof(uint64_t));
    memcpy(p, src, n*sizeof(uint64_t));
    return p;
}

int main(void){
    const int ITERS = 2000000;
    uint64_t A[64], B[64], refmag[64], rorig[80], rnew[80];
    int fails=0;
    for(int it=0; it<ITERS; ++it){
        uint64_t na = 1 + (xrand()% 40), nb = 1 + (xrand()% 40);
        for(uint64_t i=0;i<na;i++) A[i]=rand_limb();
        for(uint64_t i=0;i<nb;i++) B[i]=rand_limb();
        /* occasionally force equality */
        if((xrand()&15)==0){ nb=na; memcpy(B,A,na*sizeof(uint64_t)); }

        int s = ref_cmp(A,na,B,nb);
        /* reference magnitude = |A-B| */
        uint64_t reflen;
        if(s>=0) reflen = ref_sub(refmag,A,na,B,nb);
        else     reflen = ref_sub(refmag,B,nb,A,na);

        uint64_t *ap = mk(A,na), *bp = mk(B,nb);
        memset(rorig,0,sizeof(rorig)); memset(rnew,0,sizeof(rnew));

        int64_t Lo = cmp_and_sub_orig((pvec)rorig,(cpvec)ap,(cpvec)bp,na,nb);
        int64_t Ln = cmp_and_sub_new ((pvec)rnew ,(cpvec)ap,(cpvec)bp,na,nb);

        int ok = 1;
        /* orig vs new: identical return */
        if(Lo != Ln) ok=0;
        /* sign correctness */
        int sign = (Lo>0)-(Lo<0);
        if(sign != s) ok=0;
        if(s==0 && Lo!=0) ok=0;
        if(s!=0){
            uint64_t L = (uint64_t)(Lo<0?-Lo:Lo);
            if(L < reflen) ok=0;
            for(uint64_t i=0;i<L;i++){
                uint64_t want = i<reflen? refmag[i] : 0;
                if(rorig[i]!=want) ok=0;
                if(rnew[i] !=want) ok=0;
            }
        }
        if(!ok && fails<10){
            printf("FAIL it=%d na=%llu nb=%llu s=%d Lo=%lld Ln=%lld reflen=%llu\n",
                   it,(unsigned long long)na,(unsigned long long)nb,s,
                   (long long)Lo,(long long)Ln,(unsigned long long)reflen);
            fails++;
        }
        if(!ok) fails++;
        free(ap); free(bp);
    }
    if(fails){ printf("FAILED: %d mismatches\n", fails); return 1; }
    printf("OK: %d iterations, orig==new and both match scalar reference\n", ITERS);
    return 0;
}
