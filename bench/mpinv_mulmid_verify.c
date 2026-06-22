// Confirm WHY mpn_mulmid's natural window is insufficient for the MSB reciprocal:
// the Newton step needs D*iph coeffs up to n+h-1 (the cancelling high part).
// MODE 0 (narrow): mulmid(D,n,iph,h) -> coeffs [h-1,n-1]            (insufficient)
// MODE 1 (wide):   mulmid(D_pad,n+h,iph,h) -> coeffs [h-1,n+h-1]    (sufficient)
// Report fails + how much of the product the "MP" actually computes.
#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
extern void __gmpn_mulmid(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
static const unsigned long B=64;
static int MODE=1;
static void exact_recip(mpz_t I,const mpz_t D,long n){ mpz_t num;mpz_init(num);
    mpz_ui_pow_ui(num,2,B*2*n);mpz_fdiv_q(I,num,D);mpz_clear(num);}
// returns D*iph restricted to coeffs [h-1, top], placed at original positions (low h-1 dropped)
static void diph(mpz_t out,const mpz_t D,long n,const mpz_t iph,long h){
    long an = MODE? n+h : n;                         // pad D with h zero limbs in wide mode
    mp_limb_t *dp=calloc(an,8),*ip=calloc(h,8),*rp=calloc(an-h+3,8);
    size_t c=0; mpz_export(dp,&c,-1,8,0,0,D);        // low n limbs filled; high (an-n) stay 0
    if(mpz_sgn(iph)) mpz_export(ip,&c,-1,8,0,0,iph);
    __gmpn_mulmid(rp,dp,an,ip,h);                    // coeffs [h-1, an-1] shifted by h-1
    mpz_import(out,an-h+3,-1,8,0,0,rp);
    mpz_mul_2exp(out,out,B*(h-1));                   // back to original positions
    free(dp);free(ip);free(rp);
}
static long g_max=0;
static void recip(mpz_t I,const mpz_t D,long n){
    if(n<=2){ exact_recip(I,D,n); return; }
    long h=(n+1)/2,l=n-h;
    mpz_t Dh,Xh,iph,X0,Dbar,M,Delta,Bh; mpz_inits(Dh,Xh,iph,X0,Dbar,M,Delta,Bh,NULL);
    mpz_fdiv_q_2exp(Dh,D,B*l); recip(Xh,Dh,h);
    mpz_ui_pow_ui(Bh,2,B*h); mpz_sub(iph,Xh,Bh);
    if(mpz_sgn(iph)<0)mpz_set_ui(iph,0); if(mpz_cmp(iph,Bh)>=0)mpz_sub_ui(iph,Bh,1);
    mpz_mul_2exp(X0,Xh,B*l);
    mpz_ui_pow_ui(Dbar,2,B*n); mpz_sub(Dbar,Dbar,D);
    mpz_mul_2exp(M,Dbar,B*h);                        // B^h*Dbar
    diph(Delta,D,n,iph,h); mpz_sub(M,M,Delta);       // M = B^h*Dbar - D*iph(window)  <<< mulmid
    mpz_mul(Delta,X0,M); mpz_fdiv_q_2exp(Delta,Delta,B*(n+h));
    mpz_add(I,X0,Delta);
    mpz_t W,Bn2; mpz_inits(W,Bn2,NULL); mpz_ui_pow_ui(Bn2,2,B*2*n);
    mpz_mul(W,D,I); mpz_sub(W,Bn2,W); long it=0;
    while(mpz_sgn(W)<0){mpz_sub_ui(I,I,1);mpz_add(W,W,D);if(++it>100000)break;}
    while(mpz_cmp(W,D)>=0){mpz_add_ui(I,I,1);mpz_sub(W,W,D);if(++it>100000)break;}
    if(it>g_max)g_max=it; mpz_clears(W,Bn2,NULL);
    mpz_clears(Dh,Xh,iph,X0,Dbar,M,Delta,Bh,NULL);
}
int main(int argc,char**argv){
    gmp_randstate_t rs;gmp_randinit_default(rs);gmp_randseed_ui(rs,argc>1?atoi(argv[1]):1);
    MODE=argc>2?atoi(argv[2]):1; int NMAX=argc>3?atoi(argv[3]):120, TR=argc>4?atoi(argv[4]):40;
    mpz_t D,I,Iex;mpz_inits(D,I,Iex,NULL); long fails=0,tot=0;
    for(long n=3;n<=NMAX;n++)for(int t=0;t<TR;t++){
        mpz_urandomb(D,rs,B*n);mpz_setbit(D,B*n-1);
        recip(I,D,n);exact_recip(Iex,D,n);
        tot++; if(mpz_cmp(I,Iex)!=0)fails++;
    }
    printf("MODE=%s : FAILS=%ld/%ld  worstcorr=%ld\n",
           MODE?"WIDE[h-1,n+h-1]":"NARROW[h-1,n-1]", fails,tot,g_max);
    return 0;
}
