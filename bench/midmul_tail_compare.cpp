// 3-way, shape bn = 2*an (n = an), matched ~n^2 work:
//   midmul_basecase_simple  (masked horizontal tail)
//   midmul_basecase_vtail   (vertical dot-product tail)
//   mul_u52_basecase        (square an=bn, reference throughput)
// Verifies both midmul variants vs GMP, then emits CSV: n,simple_ns,vtail_ns,mul_ns
// Build: clang++ -O3 -march=native -I../include midmul_tail_compare.cpp -o midmul_tail_compare -lgmp
#include "mulmid.h"
#include "mul.h"
#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#define M52 ((1ull << 52) - 1)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static uint64_t rng = 0x243f6a8885a308d3ull;
static uint64_t xr(){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return rng; }
static int cmpd(const void*x,const void*y){ double a=*(const double*)x,b=*(const double*)y; return (a>b)-(a<b); }
#define PASSES 15
static void limb_to_mpz(mpz_t o,uint64_t v){ mpz_set_ui(o,(unsigned long)(v>>32)); mpz_mul_2exp(o,o,32); mpz_add_ui(o,o,(unsigned long)(v&0xffffffffull)); }
static void value_of(mpz_t out,const uint64_t*r,long n){ mpz_set_ui(out,0); mpz_t t; mpz_init(t); for(long p=n-1;p>=0;--p){mpz_mul_2exp(out,out,52); limb_to_mpz(t,r[p]); mpz_add(out,out,t);} mpz_clear(t);}
static void ref_band(mpz_t out,const uint64_t*a,const uint64_t*b,long an,long bn){ mpz_set_ui(out,0); mpz_t term,bb; mpz_inits(term,bb,NULL);
    for(long i=0;i<an;++i)for(long j=0;j<bn;++j){long k=i+j; if(k<an-1||k>bn-1)continue; limb_to_mpz(term,a[i]); limb_to_mpz(bb,b[j]); mpz_mul(term,term,bb); mpz_mul_2exp(term,term,52*(k-(an-1))); mpz_add(out,out,term);} mpz_clears(term,bb,NULL);}

int main(int argc,char**argv){
    long N_MAX=argc>1?atol(argv[1]):128, N_MIN=argc>2?atol(argv[2]):8, STEP=argc>3?atol(argv[3]):1;
    mpz_t vout,vref; mpz_inits(vout,vref,NULL);
    long fs=0,fv=0;
    printf("n,simple_ns,vtail_ns,mul_ns\n");
    for(long n=N_MIN;n<=N_MAX;n+=STEP){
        long bn=2*n, rn=bn-n+1, nout=rn+1;
        std::vector<uint64_t> A(((n+7)/8)*8+8,0), B(((bn+7)/8)*8+40,0);
        std::vector<uint64_t> Rs(((nout+7)/8)*8+16,0), Rv(((nout+7)/8)*8+16,0), Rm(((2*n+7)/8)*8+16,0);
        for(long i=0;i<n;i++)A[i]=xr()&M52; for(long j=0;j<bn;j++)B[j]=xr()&M52;
        plimb rs=(plimb)Rs.data(), rv=(plimb)Rv.data(); pvec rm=(pvec)Rm.data();
        const _limb*a=(const _limb*)A.data(),*b=(const _limb*)B.data();
        cpvec av=(cpvec)A.data(), bv=(cpvec)B.data();
        // correctness
        midmul_basecase_simple(rs,a,b,n,bn); value_of(vout,Rs.data(),nout); ref_band(vref,A.data(),B.data(),n,bn); if(mpz_cmp(vout,vref)){if(++fs<=3)fprintf(stderr,"simple wrong n=%ld\n",n);}
        midmul_basecase_vtail (rv,a,b,n,bn); value_of(vout,Rv.data(),nout);                                     if(mpz_cmp(vout,vref)){if(++fv<=3)fprintf(stderr,"vtail wrong n=%ld\n",n);}
        // timing
        long reps=(long)(20000000.0/((double)n*n))+8;
        double ts[PASSES],tv[PASSES],tm[PASSES];
        for(int p=0;p<PASSES;++p){
            double t0;
            t0=now(); for(long q=0;q<reps;q++) midmul_basecase_simple(rs,a,b,n,bn); ts[p]=(now()-t0)/reps;
            t0=now(); for(long q=0;q<reps;q++) midmul_basecase_vtail (rv,a,b,n,bn); tv[p]=(now()-t0)/reps;
            t0=now(); for(long q=0;q<reps;q++) mul_u52_basecase(rm,av,bv,n,n);      tm[p]=(now()-t0)/reps;
        }
        qsort(ts,PASSES,sizeof(double),cmpd); qsort(tv,PASSES,sizeof(double),cmpd); qsort(tm,PASSES,sizeof(double),cmpd);
        printf("%ld,%.3f,%.3f,%.3f\n",n,ts[PASSES/2]*1e9,tv[PASSES/2]*1e9,tm[PASSES/2]*1e9); fflush(stdout);
    }
    fprintf(stderr,"correctness: simple %ld, vtail %ld mismatches\n",fs,fv);
    mpz_clears(vout,vref,NULL);
    return (fs||fv)?1:0;
}
