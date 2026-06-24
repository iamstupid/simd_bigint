// Parametrized FAITHFUL replica of the 4->8 asm structure, for any h. nstep-ladder seed.
//   W = D*x_hi window [h-1,n] (n-h+2 limbs) ; V=2W ; M=-V nearest-0 (sign=V top bit) ;
//   |M|=V^m ; drop |M|[0] ; Delta = HP[h+1,n+1] of x_hi*|M|[1..n-h+1] ; out=x_hi*B^h +/- Delta.
#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
typedef uint64_t u64; typedef unsigned __int128 u128;
static const unsigned long B=64;
static void nstep(mpz_t x,const mpz_t D,int Dl,int pin,int pout){
    mpz_t Dt,P1,H1,t,m,P2; mpz_inits(Dt,P1,H1,t,m,P2,NULL);
    mpz_fdiv_q_2exp(Dt,D,(long)(Dl-pout)*B);
    mpz_mul(P1,x,Dt); mpz_fdiv_q_2exp(H1,P1,(long)pin*B); mpz_fdiv_r_2exp(H1,H1,(long)pout*B);
    mpz_ui_pow_ui(m,2,(unsigned long)pout*B); mpz_sub_ui(m,m,1); mpz_sub(t,m,H1);
    mpz_mul(P2,x,t); mpz_fdiv_q_2exp(x,P2,(long)pin*B-1); mpz_fdiv_r_2exp(x,x,(long)pout*B);
    mpz_clears(Dt,P1,H1,t,m,P2,NULL);
}
static void seed_xhi(mpz_t Xh,const mpz_t D,int n,int h){
    mpz_t num,dt; mpz_inits(num,dt,NULL);
    mpz_ui_pow_ui(num,2,127); mpz_sub_ui(num,num,1); mpz_fdiv_q_2exp(dt,D,(long)(n-1)*B); mpz_fdiv_q(Xh,num,dt);
    int p=1, sched[]={3,4,8,16,32};
    for(int s=0; sched[s]<=h; s++){ nstep(Xh,D,n,p,sched[s]); p=sched[s]; if(p==h)break; }
    if(p<h) nstep(Xh,D,n,p,h);
    mpz_clears(num,dt,NULL);
}
int main(int argc,char**argv){
    int h=argc>1?atoi(argv[1]):4; int n=2*h; int DROP=argc>2?atoi(argv[2]):1; // DROP=1: drop |M|[0]
    long TR=argc>3?atol(argv[3]):100000;
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs,1);
    mpz_t D,Xh,P,p2,e,d2,Iz; mpz_inits(D,Xh,P,p2,e,d2,Iz,NULL);
    u64 Dl[64],xh[64]; double minp=1e9,sump=0; long cnt=0,negc=0;
    int Wl=n-h+2;          // window limbs (positions [h-1,n])
    for(long tr=0;tr<TR;tr++){
        mpz_urandomb(D,rs,(unsigned long)n*B); mpz_setbit(D,(unsigned long)n*B-1);
        for(int i=0;i<n;i++)Dl[i]=0; mpz_export(Dl,NULL,-1,8,0,0,D);
        seed_xhi(Xh,D,n,h);
        for(int i=0;i<h;i++)xh[i]=0; mpz_export(xh,NULL,-1,8,0,0,Xh);
        // parallelogram W = D*x_hi window [h-1,n] -> acc[0..Wl-1] (+guard acc[Wl])
        u128 acc[80]={0};
        for(int i=0;i<h;i++) for(int k=0;k<Wl;k++){ int j=(h-1)+k-i; if(j<0||j>=n)continue;
            u128 p=(u128)xh[i]*Dl[j]; acc[k]+=(u64)p; acc[k+1]+=(u64)(p>>64); }
        u64 Wd[80]; { u128 c=0; for(int k=0;k<=Wl;k++){ u128 t=acc[k]+c; Wd[k]=(u64)t; c=t>>64; } }
        // V = 2*W
        u64 V[80]; { u64 cy=0; for(int k=0;k<Wl;k++){ u64 nv=(Wd[k]<<1)|cy; cy=Wd[k]>>63; V[k]=nv; } }
        int neg=(V[Wl-1]>>63)==0; negc+=neg;
        u64 m=neg?0:~0ULL; u64 Mab[80]; for(int k=0;k<Wl;k++) Mab[k]=V[k]^m;  // |M|, no +1
        // Delta HP: x_hi * |M|[lo..Wl-1], lo=DROP?1:0.  keep positions [h+1, n+1], a=pos-(h+1)
        // |M|[k] at position k.  product x_hi[i]*|M|[k] at position i+k.
        u128 acc2[80]={0}; int lo=DROP?1:0;
        for(int i=0;i<h;i++) for(int k=lo;k<Wl;k++){ int pos=i+k; int a=pos-(h+1);
            if(pos< h || pos> n+1) continue;            // keep positions [h, n+1] (a=-1..); compute a>=-1
            u128 p=(u128)xh[i]*Mab[k];
            if(a>=0){ acc2[a]+=(u64)p; acc2[a+1]+=(u64)(p>>64); }
            else acc2[0]+=(u64)(p>>64);                  // a=-1 (position h): only hi into kept a[0]
        }
        u64 Da[80]; { u128 c=0; for(int a=0;a<=n+1-(h+1);a++){ u128 t=acc2[a]+c; Da[a]=(u64)t; c=t>>64; } }
        int DL=n+1-(h+1)+1;   // Delta limbs = n-h+1
        u64 I[64];
        if(!neg){
            for(int k=0;k<h;k++) I[k]=Da[k];                 // Delta at [0, n-h]; positions [0,?]
            // x_hi at [h, 2h-1]; Delta has DL=n-h+1 limbs at [0, n-h]=[0,h]; overlap at pos h
            u128 s=(u128)xh[0]+Da[h]; I[h]=(u64)s; u64 cy=(u64)(s>>64);
            for(int k=1;k<h;k++){ u128 t=(u128)xh[k]+cy; I[h+k]=(u64)t; cy=(u64)(t>>64); }
        } else {
            u64 bw=0;
            for(int k=0;k<h;k++){ u128 s=(u128)0-Da[k]-bw; I[k]=(u64)s; bw=(s>>64)?1:0; }
            u128 s=(u128)xh[0]-Da[h]-bw; I[h]=(u64)s; bw=(s>>64)?1:0;
            for(int k=1;k<h;k++){ u128 t=(u128)xh[k]-bw; I[h+k]=(u64)t; bw=(t>>64)?1:0; }
        }
        (void)DL;
        mpz_import(Iz,n,-1,8,0,0,I);
        mpz_mul(P,Iz,D); long Lb=mpz_sizeinbase(P,2);
        mpz_ui_pow_ui(p2,2,Lb); mpz_sub(e,p2,P); mpz_ui_pow_ui(d2,2,Lb-1); mpz_sub(d2,P,d2);
        long eh=mpz_sgn(e)?(long)mpz_sizeinbase(e,2):0, el=mpz_sgn(d2)?(long)mpz_sizeinbase(d2,2):0;
        double prec=(mpz_cmp(e,d2)<0)?(double)(Lb-eh):(double)(Lb-1-el);
        if(prec<minp)minp=prec; sump+=prec; cnt++;
    }
    printf("FAITHFUL h=%d->%d DROP|M|[0]=%d: min=%.0f avg=%.1f neg=%ld/%ld (target %d)\n",
           h,n,DROP,minp,sump/cnt,negc,cnt,(2*h*64 - 20));
    return 0;
}
