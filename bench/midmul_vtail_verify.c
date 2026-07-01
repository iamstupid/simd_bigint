// Fuzz midmul_basecase_vtail vs the exact middle band (GMP), over many shapes.
// Build: clang -O3 -march=native -x c++ -I../include midmul_vtail_verify.c -o midmul_vtail_verify -lgmp
#include "mulmid.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#define M52 ((1ull << 52) - 1)
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static void limb_to_mpz(mpz_t o, uint64_t v){ mpz_set_ui(o,(unsigned long)(v>>32)); mpz_mul_2exp(o,o,32); mpz_add_ui(o,o,(unsigned long)(v&0xffffffffull)); }
static void value_of(mpz_t out, const uint64_t *r, long n){ mpz_set_ui(out,0); mpz_t t; mpz_init(t); for(long p=n-1;p>=0;--p){ mpz_mul_2exp(out,out,52); limb_to_mpz(t,r[p]); mpz_add(out,out,t);} mpz_clear(t); }
static void ref_band(mpz_t out,const uint64_t*a,const uint64_t*b,long an,long bn){ mpz_set_ui(out,0); mpz_t term,bb; mpz_inits(term,bb,NULL);
    for(long i=0;i<an;++i)for(long j=0;j<bn;++j){ long k=i+j; if(k<an-1||k>bn-1)continue; limb_to_mpz(term,a[i]); limb_to_mpz(bb,b[j]); mpz_mul(term,term,bb); mpz_mul_2exp(term,term,52*(k-(an-1))); mpz_add(out,out,term);} mpz_clears(term,bb,NULL); }

int main(int argc, char** argv){
    long trials = argc>1?atol(argv[1]):200000;
    mpz_t vs, vv, vref; mpz_inits(vs,vv,vref,NULL);
    long fails=0, mism_simple=0;
    for(long t=0;t<trials;++t){
        long an = 1 + (xr()% 130);
        long bn = an + 1 + (xr()% (2*an));            // an+1 .. 3an
        long rn = bn-an+1, nout = rn+1;
        long apad=((an+7)/8)*8+8, bpad=((bn+7)/8)*8+40, rpad=((nout+7)/8)*8+16;
        uint64_t *A=(uint64_t*)calloc(apad,8),*B=(uint64_t*)calloc(bpad,8),*Rv=(uint64_t*)calloc(rpad,8),*Rs=(uint64_t*)calloc(rpad,8);
        for(long i=0;i<an;i++)A[i]=xr()&M52; for(long j=0;j<bn;j++)B[j]=xr()&M52;
        midmul_basecase_vtail((plimb)Rv,(const _limb*)A,(const _limb*)B,an,bn);
        midmul_basecase_simple((plimb)Rs,(const _limb*)A,(const _limb*)B,an,bn);
        ref_band(vref,A,B,an,bn);
        value_of(vv,Rv,nout); value_of(vs,Rs,nout);
        if(mpz_cmp(vv,vref)!=0){ if(++fails<=8) fprintf(stderr,"VTAIL wrong an=%ld bn=%ld (rn%%8=%ld an%%8=%ld)\n",an,bn,rn%8,an%8); }
        if(mpz_cmp(vs,vref)!=0){ mism_simple++; }
        free(A);free(B);free(Rv);free(Rs);
    }
    printf("vtail: %ld/%ld mismatches vs exact band;  simple: %ld mismatches\n", fails, trials, mism_simple);
    mpz_clears(vs,vv,vref,NULL);
    return fails?1:0;
}
