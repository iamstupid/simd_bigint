#define _GNU_SOURCE
#include "../include/div.h"
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define M52 ((1ull<<52)-1)
static void mpz_to_u52(uint64_t* L,int n,const mpz_t X){ mpz_t t; mpz_init_set(t,X);
  for(int i=0;i<n;i++){ L[i]=mpz_get_ui(t)&M52; mpz_fdiv_q_2exp(t,t,52);} mpz_clear(t);}
static void u52_to_mpz(mpz_t X,const uint64_t* L,int n){ mpz_set_ui(X,0);
  for(int i=n-1;i>=0;i--){ mpz_mul_2exp(X,X,52); mpz_add_ui(X,X,L[i]); } }
static long bad_exact=0,bad_seed=0,ovf=0;
static mpz_t beta,beta3m1,qT,rT,mq,mr;
// check one (D,U) with both exact and seed reciprocal
static void check(const mpz_t D, const mpz_t U){
  mpz_t V; mpz_init(V); mpz_fdiv_q(V,beta3m1,D); mpz_sub(V,V,beta);  // exact v
  mpz_fdiv_qr(qT,rT,U,D);
  if(mpz_cmp_ui(qT,0)>=0){ mpz_t bm1; mpz_init(bm1); mpz_sub_ui(bm1,beta,1); if(mpz_cmp(qT,bm1)==0)ovf++; mpz_clear(bm1);}
  alignas(64) uint64_t d[16],u[24],vv[8]; mpz_to_u52(d,16,D); mpz_to_u52(u,24,U); mpz_to_u52(vv,8,V);
  uint64_t D64[13]={0}; size_t cnt; mpz_export(D64,&cnt,-1,8,0,0,D);
  // exact
  for(int which=0;which<2;which++){
    _vec vrec = which? recip_3by2_seed(D64) : load_vec((cpvec)vv);
    _div32 R = div3by2(u,d,vrec);
    alignas(64) uint64_t q1[8],r0[8],r1[8]; store_vec((pvec)q1,R.q1); store_vec((pvec)r0,R.r.lo); store_vec((pvec)r1,R.r.hi);
    u52_to_mpz(mq,q1,8); uint64_t rl[16]; for(int i=0;i<8;i++){rl[i]=r0[i];rl[i+8]=r1[i];} u52_to_mpz(mr,rl,16);
    int ok = (mpz_cmp(mq,qT)==0)&&(mpz_cmp(mr,rT)==0);
    if(!ok){ if(which)bad_seed++; else bad_exact++; if((which?bad_seed:bad_exact)<=3) gmp_printf("  %s MISMATCH got q=%Zx r=%Zx  exp q=%Zx r=%Zx\n", which?"SEED":"EXACT", mq,mr,qT,rT); }
  }
  mpz_clear(V);
}
int main(int c,char**v){
  gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs,c>1?strtoul(v[1],0,0):1);
  mpz_t D,U,UH,U0,b2; mpz_inits(beta,beta3m1,qT,rT,mq,mr,D,U,UH,U0,b2,NULL);
  mpz_ui_pow_ui(beta,2,416); mpz_mul(b2,beta,beta); mpz_mul(beta3m1,b2,beta); mpz_sub_ui(beta3m1,beta3m1,1);
  const long N=300000;
  for(long it=0;it<N;it++){ mpz_urandomb(D,rs,832); mpz_setbit(D,831);
    mpz_urandomm(UH,rs,D); mpz_urandomb(U0,rs,416); mpz_mul(U,UH,beta); mpz_add(U,U,U0); check(D,U); }
  // adversarial: max quotient (UH=D-1, U0=beta-1) -> q=beta-1 (overflow branch)
  for(long it=0;it<50000;it++){ mpz_urandomb(D,rs,832); mpz_setbit(D,831);
    mpz_sub_ui(UH,D,1); mpz_sub_ui(U0,beta,1); mpz_mul(U,UH,beta); mpz_add(U,U,U0); check(D,U); }
  // adversarial: tiny remainder (U = k*D), various k
  for(long it=0;it<50000;it++){ mpz_urandomb(D,rs,832); mpz_setbit(D,831);
    mpz_urandomb(UH,rs,415); mpz_mod(UH,UH,beta); mpz_mul(U,UH,D); check(D,U); }  // U = q*D, r should be 0
  printf("div3by2: exact bad=%ld, seed bad=%ld  (overflow-branch cases hit: %ld)\n", bad_exact,bad_seed,ovf);
  return (bad_exact||bad_seed)?1:0;
}
