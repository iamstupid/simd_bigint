#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
static void nstep(mpz_t x,const mpz_t D,int Dl,int pi,int po){
    mpz_t Dt,P1,H1,t,m,P2; mpz_inits(Dt,P1,H1,t,m,P2,NULL);
    mpz_fdiv_q_2exp(Dt,D,(long)(Dl-po)*64); mpz_mul(P1,x,Dt);
    mpz_fdiv_q_2exp(H1,P1,(long)pi*64); mpz_fdiv_r_2exp(H1,H1,(long)po*64);
    mpz_ui_pow_ui(m,2,(unsigned long)po*64); mpz_sub_ui(m,m,1); mpz_sub(t,m,H1);
    mpz_mul(P2,x,t); mpz_fdiv_q_2exp(x,P2,(long)pi*64-1); mpz_fdiv_r_2exp(x,x,(long)po*64);
    mpz_clears(Dt,P1,H1,t,m,P2,NULL);
}
static unsigned long limb(const mpz_t a,int k){ // limb k
    mpz_t t; mpz_init(t); mpz_fdiv_q_2exp(t,a,(long)k*64); unsigned long v=mpz_get_ui(t); mpz_clear(t); return v;}
int main(int ac,char**av){
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs,ac>1?atoi(av[1]):1);
    mpz_t D,r,R8,P1535,rd,Delta,rddig,MP,ri,dwin,num,dt;
    mpz_inits(D,r,R8,P1535,rd,Delta,rddig,MP,ri,dwin,num,dt,NULL);
    mpz_ui_pow_ui(P1535,2,1535);
    mpz_urandomb(D,rs,16*64); mpz_setbit(D,16*64-1);
    mpz_ui_pow_ui(num,2,127); mpz_sub_ui(num,num,1);
    mpz_fdiv_q_2exp(dt,D,15*64); mpz_fdiv_q(r,num,dt);
    nstep(r,D,16,1,2); nstep(r,D,16,2,4); nstep(r,D,16,4,8);
    mpz_fdiv_q(R8,P1535,D);                 // true 512-bit reciprocal
    mpz_sub(num,R8,r); printf("r deviation e8 = R8 - r = %ld ULP\n", mpz_get_si(num));
    mpz_mul(rd,r,D); mpz_sub(Delta,P1535,rd);
    printf("rd bits=%lu (top limb idx=%lu)\n", mpz_sizeinbase(rd,2),(mpz_sizeinbase(rd,2)-1)/64);
    printf("Delta=2^1535-rd bits=%lu\n", mpz_sizeinbase(Delta,2));
    // band MP
    mpz_set_ui(MP,0);
    for(int i=0;i<8;i++){ mpz_fdiv_q_2exp(ri,r,(long)i*64); mpz_fdiv_r_2exp(ri,ri,64);
        mpz_fdiv_q_2exp(dwin,D,(long)(7-i)*64); mpz_fdiv_r_2exp(dwin,dwin,9*64); mpz_addmul(MP,ri,dwin); }
    printf("band MP bits=%lu\n", mpz_sizeinbase(MP,2));
    // rd's actual digits, shifted down 7 limbs (so band-frame: MP limb k <-> rd limb 7+k)
    mpz_fdiv_q_2exp(rddig,rd,7*64);
    printf("\n  k | MP[k]              rd[7+k]            match?\n");
    for(int k=0;k<11;k++){ unsigned long a=limb(MP,k),b=limb(rddig,k);
        printf(" %2d | %016lx   %016lx   %s\n",k,a,b,a==b?"YES":"no"); }
    // Delta digits (shifted down 7 limbs) -- what neg(MP) should give
    printf("\n  k | ~MP[k] (band frame)  Delta[7+k]\n");
    mpz_t Ddig; mpz_init(Ddig); mpz_fdiv_q_2exp(Ddig,Delta,7*64);
    for(int k=0;k<11;k++){ unsigned long a=~limb(MP,k),b=limb(Ddig,k);
        printf(" %2d | %016lx   %016lx   %s\n",k,a,b,a==b?"YES":"no"); }
    return 0;
}
