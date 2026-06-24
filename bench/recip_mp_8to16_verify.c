#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static void nstep(mpz_t x,const mpz_t D,int Dl,int pi,int po){
    mpz_t Dt,P1,H1,t,m,P2; mpz_inits(Dt,P1,H1,t,m,P2,NULL);
    mpz_fdiv_q_2exp(Dt,D,(long)(Dl-po)*64); mpz_mul(P1,x,Dt);
    mpz_fdiv_q_2exp(H1,P1,(long)pi*64); mpz_fdiv_r_2exp(H1,H1,(long)po*64);
    mpz_ui_pow_ui(m,2,(unsigned long)po*64); mpz_sub_ui(m,m,1); mpz_sub(t,m,H1);
    mpz_mul(P2,x,t); mpz_fdiv_q_2exp(x,P2,(long)pi*64-1); mpz_fdiv_r_2exp(x,x,(long)po*64);
    mpz_clears(Dt,P1,H1,t,m,P2,NULL);
}
static void band(mpz_t MP,const mpz_t r,const mpz_t D,int lo,int hi){
    mpz_t ri,dwin,term; mpz_inits(ri,dwin,term,NULL); mpz_set_ui(MP,0);
    for(int i=0;i<8;i++){
        mpz_fdiv_q_2exp(ri,r,(long)i*64); mpz_fdiv_r_2exp(ri,ri,64);
        int dl=lo-i,dh=hi-i; if(dl<0)dl=0; if(dh>15)dh=15; int w=dh-dl+1; if(w<=0)continue;
        mpz_fdiv_q_2exp(dwin,D,(long)dl*64); mpz_fdiv_r_2exp(dwin,dwin,(long)w*64);
        mpz_mul(term,ri,dwin); mpz_mul_2exp(term,term,(long)(i+dl-lo)*64); mpz_add(MP,MP,term);
    }
    mpz_clears(ri,dwin,term,NULL);
}
int main(int ac,char**av){
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs,ac>1?atoi(av[1]):1);
    mpz_t D,r,Ep,P1535,R8,MP,MPw,t,maskw,corr,R0,rp,R,err,num,dt;
    mpz_inits(D,r,Ep,P1535,R8,MP,MPw,t,maskw,corr,R0,rp,R,err,num,dt,NULL);
    mpz_ui_pow_ui(Ep,2,2047); mpz_ui_pow_ui(P1535,2,1535);
    struct{int lo,hi,W;long Sc;const char*nm;} cfg[]={
        {7,15,9,511,"user [7,15] w9 (orig)"},
        {7,16,9,511,"[7,16] w9"},
        {6,17,10,511,"[6,17] w10 (guard both ends)"},
    };
    for(int c=0;c<3;c++){
        int lo=cfg[c].lo,hi=cfg[c].hi,W=cfg[c].W; long Sc=cfg[c].Sc;
        long maxerr=0,negs=0,n=0; double sb=0;
        for(int tt=0;tt<200000;tt++){
            mpz_urandomb(D,rs,16*64); mpz_setbit(D,16*64-1);
            mpz_ui_pow_ui(num,2,127); mpz_sub_ui(num,num,1);
            mpz_fdiv_q_2exp(dt,D,15*64); mpz_fdiv_q(r,num,dt);
            nstep(r,D,16,1,2); nstep(r,D,16,2,4); nstep(r,D,16,4,8);
            mpz_fdiv_q(R8,P1535,D); if(mpz_cmp(r,R8)>0) mpz_set(r,R8);   // FORCE underestimate (clamp)
            band(MP,r,D,lo,hi);
            mpz_fdiv_q_2exp(MPw,MP,(long)(8-lo)*64); mpz_fdiv_r_2exp(MPw,MPw,(long)W*64);
            mpz_ui_pow_ui(maskw,2,(unsigned long)W*64); mpz_sub_ui(maskw,maskw,1); mpz_sub(t,maskw,MPw);
            mpz_mul(corr,r,t); mpz_fdiv_q_2exp(corr,corr,Sc);
            mpz_mul_2exp(R0,r,512); mpz_add(rp,R0,corr); mpz_fdiv_r_2exp(rp,rp,16*64);
            mpz_fdiv_q(R,Ep,D); mpz_sub(err,R,rp);
            if(mpz_sgn(err)<0)negs++;
            long e=mpz_cmpabs_ui(err,4000000000)>0?4000000000:labs(mpz_get_si(err));
            if(e>maxerr)maxerr=e; if(e>0)sb+=log2((double)e); n++;
        }
        printf("%-32s : max=%ld (2^%.1f) avg=%.1f neg=%ld\n",
               cfg[c].nm,maxerr,maxerr>0?log2((double)maxerr):0,sb/n,negs);
    }
    return 0;
}
