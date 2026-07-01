// simple vs wtail (fused horizontal-bottom + vertical-top tail), shape bn=2*an.
// Verifies both vs GMP, then emits CSV: n,simple_ns,wtail_ns
// Build: clang++ -O3 -march=native -I../include midmul_wtail_compare.cpp -o midmul_wtail_compare -lgmp
#include "mulmid.h"
#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#define M52 ((1ull<<52)-1)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static uint64_t rng=0x243f6a8885a308d3ull;
static uint64_t xr(){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return rng; }
static int cmpd(const void*x,const void*y){ double a=*(const double*)x,b=*(const double*)y; return (a>b)-(a<b); }
#define PASSES 15
static void l2m(mpz_t o,uint64_t v){ mpz_set_ui(o,(unsigned long)(v>>32)); mpz_mul_2exp(o,o,32); mpz_add_ui(o,o,(unsigned long)(v&0xffffffffu)); }
static void value_of(mpz_t o,const uint64_t*r,long n){ mpz_set_ui(o,0); mpz_t t; mpz_init(t); for(long p=n-1;p>=0;--p){mpz_mul_2exp(o,o,52);l2m(t,r[p]);mpz_add(o,o,t);} mpz_clear(t); }
static void ref_band(mpz_t o,const uint64_t*a,const uint64_t*b,long an,long bn){ mpz_set_ui(o,0); mpz_t T,B; mpz_inits(T,B,NULL);
    for(long i=0;i<an;++i)for(long j=0;j<bn;++j){long k=i+j;if(k<an-1||k>bn-1)continue;l2m(T,a[i]);l2m(B,b[j]);mpz_mul(T,T,B);mpz_mul_2exp(T,T,52*(k-(an-1)));mpz_add(o,o,T);} mpz_clears(T,B,NULL); }

int main(int argc,char**argv){
    long N_MAX=argc>1?atol(argv[1]):128, N_MIN=argc>2?atol(argv[2]):8, STEP=argc>3?atol(argv[3]):1;
    mpz_t vout,vref; mpz_inits(vout,vref,NULL);
    long fs=0,fw=0;
    printf("n,simple_ns,wtail_ns\n");
    for(long n=N_MIN;n<=N_MAX;n+=STEP){
        long bn=2*n, rn=bn-n+1, nout=rn+1;
        std::vector<uint64_t> A(((n+7)/8)*8+8,0), B(((bn+7)/8)*8+64,0);
        std::vector<uint64_t> Rs(((nout+7)/8)*8+16,0), Rw(((nout+7)/8)*8+16,0);
        for(long i=0;i<n;i++)A[i]=xr()&M52; for(long j=0;j<bn;j++)B[j]=xr()&M52;
        plimb rs=(plimb)Rs.data(), rw=(plimb)Rw.data();
        const _limb*a=(const _limb*)A.data(),*b=(const _limb*)B.data();
        midmul_basecase_simple      (rs,a,b,n,bn); value_of(vout,Rs.data(),nout); ref_band(vref,A.data(),B.data(),n,bn); if(mpz_cmp(vout,vref)){if(++fs<=3)fprintf(stderr,"simple wrong n=%ld\n",n);}
        midmul_basecase_simple_wtail(rw,a,b,n,bn); value_of(vout,Rw.data(),nout);                                       if(mpz_cmp(vout,vref)){if(++fw<=3)fprintf(stderr,"wtail wrong n=%ld\n",n);}
        long reps=(long)(20000000.0/((double)n*n))+8;
        double ts[PASSES],tw[PASSES];
        for(int p=0;p<PASSES;++p){
            double t0;
            t0=now(); for(long q=0;q<reps;q++) midmul_basecase_simple      (rs,a,b,n,bn); ts[p]=(now()-t0)/reps;
            t0=now(); for(long q=0;q<reps;q++) midmul_basecase_simple_wtail(rw,a,b,n,bn); tw[p]=(now()-t0)/reps;
        }
        qsort(ts,PASSES,sizeof(double),cmpd); qsort(tw,PASSES,sizeof(double),cmpd);
        printf("%ld,%.3f,%.3f\n",n,ts[PASSES/2]*1e9,tw[PASSES/2]*1e9); fflush(stdout);
    }
    fprintf(stderr,"correctness: simple %ld, wtail %ld mismatches\n",fs,fw);
    mpz_clears(vout,vref,NULL);
    return (fs||fw)?1:0;
}
