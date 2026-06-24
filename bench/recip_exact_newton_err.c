// Clean EXACT Newton (full precision, no mulhi, no D-trunc) from the same 63-bit
// seed, 4 doublings -> isolates the seed-limited bound (vs the mulhi scheme's 2^22).
#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc,char**argv){
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs,argc>1?atoi(argv[1]):1);
    mpz_t D,x,seed,num,dt,Dx,corr,R,Ep,err;
    mpz_inits(D,x,seed,num,dt,Dx,corr,R,Ep,err,NULL);
    const long E=2047;
    mpz_ui_pow_ui(Ep,2,E);
    long maxerr=0; double sumb=0; long n=0, negs=0;
    for(int t=0;t<200000;t++){
        mpz_urandomb(D,rs,16*64); mpz_setbit(D,16*64-1);
        // seed: floor((2^127-1)/D_toplimb), placed in the 2^E frame
        mpz_ui_pow_ui(num,2,127); mpz_sub_ui(num,num,1);
        mpz_fdiv_q_2exp(dt,D,15*64); mpz_fdiv_q(seed,num,dt);   // ~63-bit recip of top limb
        // x ~ 2^E/D ; seed ~ 2^127/Dtop = 2^127/(D/2^960) -> x0 = seed<<(E-127-960)? align by value:
        // set x0 so x0 ~ 2^E/D: x0 = seed << (E - 127 - 960)  (E=2047 -> shift 960)
        mpz_mul_2exp(x, seed, E-127-(15*64));                  // = seed << 960
        // 4 exact Newton steps: x <- x + floor(x*(2^E - D*x)/2^E)
        for(int it=0; it<4; it++){
            mpz_mul(Dx, D, x); mpz_sub(corr, Ep, Dx);          // 2^E - D*x  (residual, signed)
            mpz_mul(corr, corr, x); mpz_fdiv_q_2exp(corr, corr, E);  // x*(2^E-Dx)/2^E
            mpz_add(x, x, corr);
        }
        mpz_fdiv_q(R, Ep, D);                                  // exact floor(2^E/D)
        mpz_sub(err, R, x);
        if(mpz_sgn(err)<0) negs++;
        long e = mpz_cmpabs_ui(err,4000000000)>0?4000000000:labs(mpz_get_si(err));
        if(e>maxerr)maxerr=e; if(e>0)sumb+=log2((double)e); n++;
    }
    printf("EXACT Newton, 63-bit seed, 4 doublings (no mulhi, full D):\n");
    printf("  max|R-x| = %ld ULP (2^%.1f)   avg_errbits=%.1f   neg(overshoot)=%ld\n",
           maxerr, maxerr>0?log2((double)maxerr):0, sumb/n, negs);
    printf("  (ideal from 63*16=1008-bit relative accuracy: ~2^%ld ULP)\n", (long)(E-1008));
    return 0;
}
