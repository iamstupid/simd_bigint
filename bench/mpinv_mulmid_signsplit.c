// Test hypothesis: narrow-window mulmid fails ONLY when X0 OVERestimates
// (true M = B^h*Dbar - D*iph < 0), i.e. the under/over-estimate case is unhandled.
#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
extern void __gmpn_mulmid(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);
static const unsigned long B=64;
static long fM_neg=0, fM_pos=0, okM_neg=0, okM_pos=0;   // fail/ok split by sign(M)
static void exact_recip(mpz_t I,const mpz_t D,long n){ mpz_t num;mpz_init(num);
    mpz_ui_pow_ui(num,2,B*2*n);mpz_fdiv_q(I,num,D);mpz_clear(num);}
static void mulmid_V(mpz_t V,const mpz_t D,long n,const mpz_t iph,long h){
    mp_limb_t *dp=calloc(n,8),*ip=calloc(h,8),*rp=calloc(n-h+3,8);
    size_t c=0; mpz_export(dp,&c,-1,8,0,0,D);
    if(mpz_sgn(iph)) mpz_export(ip,&c,-1,8,0,0,iph);
    __gmpn_mulmid(rp,dp,n,ip,h); mpz_import(V,n-h+3,-1,8,0,0,rp);
    free(dp);free(ip);free(rp);
}
static void recip(mpz_t I,const mpz_t D,long n){
    if(n<=2){ exact_recip(I,D,n); return; }
    long h=(n+1)/2,l=n-h;
    mpz_t Dh,Xh,iph,X0,Dbar,DbarLo,V,Vw,M,Mtrue,Delta,Bh,t;
    mpz_inits(Dh,Xh,iph,X0,Dbar,DbarLo,V,Vw,M,Mtrue,Delta,Bh,t,NULL);
    mpz_fdiv_q_2exp(Dh,D,B*l); recip(Xh,Dh,h);
    mpz_ui_pow_ui(Bh,2,B*h); mpz_sub(iph,Xh,Bh);
    if(mpz_sgn(iph)<0)mpz_set_ui(iph,0); if(mpz_cmp(iph,Bh)>=0)mpz_sub_ui(iph,Bh,1);
    mpz_mul_2exp(X0,Xh,B*l);
    mpz_ui_pow_ui(Dbar,2,B*n); mpz_sub(Dbar,Dbar,D);
    mpz_fdiv_r_2exp(DbarLo,Dbar,B*l);
    // TRUE M (for diagnostic only)
    mpz_mul_2exp(Mtrue,Dbar,B*h); mpz_mul(t,D,iph); mpz_sub(Mtrue,Mtrue,t);
    // narrow-window computed M (assumes M>=0)
    mulmid_V(V,D,n,iph,h); mpz_fdiv_r_2exp(Vw,V,B*(l+1));
    mpz_mul_2exp(M,DbarLo,B*h); mpz_mul_2exp(Vw,Vw,B*(h-1)); mpz_sub(M,M,Vw);
    mpz_mul(Delta,X0,M); mpz_fdiv_q_2exp(Delta,Delta,B*(n+h)); mpz_add(I,X0,Delta);
    // bounded correction
    mpz_t W,Bn2; mpz_inits(W,Bn2,NULL); mpz_ui_pow_ui(Bn2,2,B*2*n);
    mpz_mul(W,D,I); mpz_sub(W,Bn2,W); long it=0;
    while(mpz_sgn(W)<0){mpz_sub_ui(I,I,1);mpz_add(W,W,D);if(++it>100000)break;}
    while(mpz_cmp(W,D)>=0){mpz_add_ui(I,I,1);mpz_sub(W,W,D);if(++it>100000)break;}
    // tally vs exact at THIS level
    mpz_t Iex; mpz_init(Iex); exact_recip(Iex,D,n);
    int bad = (mpz_cmp(I,Iex)!=0);
    int mneg = (mpz_sgn(Mtrue)<0);
    if(bad){ if(mneg)fM_neg++; else fM_pos++; } else { if(mneg)okM_neg++; else okM_pos++; }
    mpz_clear(Iex); mpz_clears(W,Bn2,NULL);
    mpz_clears(Dh,Xh,iph,X0,Dbar,DbarLo,V,Vw,M,Mtrue,Delta,Bh,t,NULL);
}
int main(int argc,char**argv){
    gmp_randstate_t rs;gmp_randinit_default(rs);gmp_randseed_ui(rs,argc>1?atoi(argv[1]):1);
    int NMAX=argc>2?atoi(argv[2]):120, TR=argc>3?atoi(argv[3]):40;
    mpz_t D,I,Iex;mpz_inits(D,I,Iex,NULL);
    for(long n=3;n<=NMAX;n++)for(int t=0;t<TR;t++){
        mpz_urandomb(D,rs,B*n);mpz_setbit(D,B*n-1); recip(I,D,n);
    }
    printf("split by sign(true M) at each recursion level:\n");
    printf("  M>=0 (underest): ok=%ld fail=%ld\n", okM_pos, fM_pos);
    printf("  M<0  (overest) : ok=%ld fail=%ld\n", okM_neg, fM_neg);
    return 0;
}
