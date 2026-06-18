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
int main(int c,char**v){
  gmp_randstate_t rs; gmp_randinit_default(rs);
  unsigned long seed=c>1?strtoul(v[1],0,0):12345; gmp_randseed_ui(rs,seed);
  mpz_t D,U,UH,U0,beta,beta2,beta3m1,V,qT,rT,mq,mr,t; mpz_inits(D,U,UH,U0,beta,beta2,beta3m1,V,qT,rT,mq,mr,t,NULL);
  mpz_ui_pow_ui(beta,2,416); mpz_mul(beta2,beta,beta); mpz_mul(beta3m1,beta2,beta); mpz_sub_ui(beta3m1,beta3m1,1);
  long bad=0,nq=0,nr=0; const long N=300000;
  for(long it=0;it<N;it++){
    // D: 832-bit, top bit set (normalized)
    mpz_urandomb(D,rs,832); mpz_setbit(D,831);
    // exact v = floor((beta^3-1)/D) - beta
    mpz_fdiv_q(V,beta3m1,D); mpz_sub(V,V,beta);
    // <u2,u1> in [0,D); u0 in [0,beta); U = <u2,u1>*beta + u0
    mpz_urandomm(UH,rs,D); mpz_urandomb(U0,rs,416);
    mpz_mul(U,UH,beta); mpz_add(U,U,U0);
    // true q,r
    mpz_fdiv_qr(qT,rT,U,D);
    // to limbs
    alignas(64) uint64_t d[16],u[24],vv[8];
    mpz_to_u52(d,16,D); mpz_to_u52(u,24,U); mpz_to_u52(vv,8,V);
    _div32 R = div3by2(u,d,load_vec((cpvec)vv));
    alignas(64) uint64_t q1[8],r0[8],r1[8]; store_vec((pvec)q1,R.q1); store_vec((pvec)r0,R.r.lo); store_vec((pvec)r1,R.r.hi);
    u52_to_mpz(mq,q1,8);
    uint64_t rl[16]; for(int i=0;i<8;i++){rl[i]=r0[i];rl[i+8]=r1[i];} u52_to_mpz(mr,rl,16);
    if(mpz_cmp(mq,qT)!=0){ if(nq<3) gmp_printf("Q MISMATCH it=%ld got=%Zx exp=%Zx\n",it,mq,qT); nq++; bad++; }
    else if(mpz_cmp(mr,rT)!=0){ if(nr<3) gmp_printf("R MISMATCH it=%ld got=%Zx exp=%Zx\n",it,mr,rT); nr++; bad++; }
  }
  printf("div3by2 (exact v): %s  bad=%ld (q=%ld r=%ld) / %ld\n", bad?"FAIL":"OK", bad,nq,nr,N);
  return bad?1:0;
}
