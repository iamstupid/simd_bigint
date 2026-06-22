// FORCE underestimate via ceil sub-divisor + give mulmid ONE guard limb (an=n+1
// -> coeffs [h-1,n]) so ceil's slight overshoot (M just past B^n) is representable;
// reduce mod B^{n+1} (HIGH'=k*B^{n+1} still cancels). No sign branch.
#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
extern void __gmpn_mulmid(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
static const unsigned long B=64;
static long g_max=0;
static void exact_recip(mpz_t I,const mpz_t D,long n){ mpz_t num;mpz_init(num);
    mpz_ui_pow_ui(num,2,B*2*n);mpz_fdiv_q(I,num,D);mpz_clear(num);}
// V = mulmid(D padded to n+1, iph, h) = coeffs [h-1, n] shifted by h-1
static void mulmid_V1(mpz_t V,const mpz_t D,long n,const mpz_t iph,long h){
    long an=n+1;                                       // ONE guard limb
    mp_limb_t *dp=calloc(an,8),*ip=calloc(h,8),*rp=calloc(an-h+3,8);
    size_t c=0; mpz_export(dp,&c,-1,8,0,0,D);          // low n limbs; dp[n]=0
    if(mpz_sgn(iph)) mpz_export(ip,&c,-1,8,0,0,iph);
    __gmpn_mulmid(rp,dp,an,ip,h); mpz_import(V,an-h+3,-1,8,0,0,rp);
    free(dp);free(ip);free(rp);
}
static void recip(mpz_t I,const mpz_t D,long n){
    if(n<=2){ exact_recip(I,D,n); return; }
    long h=(n+1)/2,l=n-h;
    mpz_t Dh,Xh,iph,X0,Dbar,V,M,Vsh,Delta,Bh,Bl,r;
    mpz_inits(Dh,Xh,iph,X0,Dbar,V,M,Vsh,Delta,Bh,Bl,r,NULL);
    mpz_ui_pow_ui(Bl,2,B*l); mpz_ui_pow_ui(Bh,2,B*h);
    mpz_fdiv_qr(Dh,r,D,Bl); if(mpz_sgn(r)) mpz_add_ui(Dh,Dh,1);   // ceil -> underestimate
    if(mpz_cmp(Dh,Bh)>=0) mpz_sub_ui(Dh,Dh,1);
    recip(Xh,Dh,h);
    mpz_sub(iph,Xh,Bh);
    if(mpz_sgn(iph)<0)mpz_set_ui(iph,0); if(mpz_cmp(iph,Bh)>=0)mpz_sub_ui(iph,Bh,1);
    mpz_mul_2exp(X0,Xh,B*l);
    mpz_ui_pow_ui(Dbar,2,B*n); mpz_sub(Dbar,Dbar,D);
    mulmid_V1(V,D,n,iph,h);                            // narrow + 1 guard limb
    mpz_mul_2exp(M,Dbar,B*h); mpz_mul_2exp(Vsh,V,B*(h-1)); mpz_sub(M,M,Vsh);
    mpz_fdiv_r_2exp(M,M,B*(n+1));                      // mod B^{n+1} (HIGH' cancels)
    mpz_mul(Delta,X0,M); mpz_fdiv_q_2exp(Delta,Delta,B*(n+h)); mpz_add(I,X0,Delta);
    mpz_t W,Bn2; mpz_inits(W,Bn2,NULL); mpz_ui_pow_ui(Bn2,2,B*2*n);
    mpz_mul(W,D,I); mpz_sub(W,Bn2,W); long it=0;
    while(mpz_sgn(W)<0){mpz_sub_ui(I,I,1);mpz_add(W,W,D);if(++it>1000)break;}
    while(mpz_cmp(W,D)>=0){mpz_add_ui(I,I,1);mpz_sub(W,W,D);if(++it>1000)break;}
    if(it>g_max)g_max=it; mpz_clears(W,Bn2,NULL);
    mpz_clears(Dh,Xh,iph,X0,Dbar,V,M,Vsh,Delta,Bh,Bl,r,NULL);
}
int main(int argc,char**argv){
    gmp_randstate_t rs;gmp_randinit_default(rs);gmp_randseed_ui(rs,argc>1?atoi(argv[1]):1);
    int NMAX=argc>2?atoi(argv[2]):250, TR=argc>3?atoi(argv[3]):80;
    mpz_t D,I,Iex;mpz_inits(D,I,Iex,NULL); long fails=0,tot=0;
    for(long n=3;n<=NMAX;n++)for(int t=0;t<TR;t++){
        mpz_urandomb(D,rs,B*n);mpz_setbit(D,B*n-1);
        recip(I,D,n);exact_recip(Iex,D,n);
        tot++; if(mpz_cmp(I,Iex)!=0)fails++;
    }
    printf("ceil + 1-guard-limb mulmid + mod B^{n+1}: FAILS=%ld/%ld  worstcorr=%ld\n",fails,tot,g_max);
    return 0;
}
