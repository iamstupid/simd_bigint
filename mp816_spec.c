// CORRECT full-limb 8->16 (asm spec). W=parallelogram window [h-1,n] (10 limbs);
// s=W top bit; |R| = s ? negate(W) : W ; corr = (2*x_hi*|R|)>>(h+1) limbs ; I=X0 ± corr.
#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
typedef uint64_t u64; typedef unsigned __int128 u128;
static const unsigned long B=64;
extern void recip_u512_mp(const u64* D, u64* out);
static double precof(mpz_t I, mpz_t D){
    mpz_t P,p2,e,d2; mpz_inits(P,p2,e,d2,NULL); double prec;
    mpz_mul(P,I,D); long L=mpz_sizeinbase(P,2);
    mpz_ui_pow_ui(p2,2,L); mpz_sub(e,p2,P); mpz_ui_pow_ui(d2,2,L-1); mpz_sub(d2,P,d2);
    long eh=mpz_sgn(e)?(long)mpz_sizeinbase(e,2):0, el=mpz_sgn(d2)?(long)mpz_sizeinbase(d2,2):0;
    prec=(mpz_cmp(e,d2)<0)?(double)(L-eh):(double)(L-1-el);
    mpz_clears(P,p2,e,d2,NULL); return prec;
}
int main(int argc,char**argv){
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs,argc>1?atoi(argv[1]):1);
    long TR=argc>2?atol(argv[2]):200000;
    const int h=8,n=16, W=n-h+2;   // W=10
    mpz_t D,Iz; mpz_inits(D,Iz,NULL);
    double minp=1e9,sump=0; long cnt=0,negc=0;
    for(long tr=0;tr<TR;tr++){
        mpz_urandomb(D,rs,(unsigned long)n*B); mpz_setbit(D,(unsigned long)n*B-1);
        u64 Dl[16]={0}; mpz_export(Dl,NULL,-1,8,0,0,D);
        u64 Dhi[8]; for(int i=0;i<8;i++)Dhi[i]=Dl[8+i];
        u64 xh[8]; recip_u512_mp(Dhi,xh);
        // parallelogram window [h-1,n] -> Wd[0..W-1]
        u128 acc[20]={0};
        for(int i=0;i<h;i++)
            for(int k=0;k<W;k++){ int j=(h-1)+k-i; if(j<0||j>=n) continue;
                u128 p=(u128)xh[i]*Dl[j]; acc[k]+=(u64)p; acc[k+1]+=(u64)(p>>64); }
        u64 Wd[20]; { u128 c=0; for(int k=0;k<=W;k++){ u128 t=acc[k]+c; Wd[k]=(u64)t; c=t>>64; } }
        int neg = (Wd[W-1]>>63)==0;       // W top bit CLEAR -> R negative (sign -)
        negc+=neg;
        u64 Rab[20];                       // |R| (W limbs): neg? W : negate(W)
        if(!neg){ u64 b=1; for(int k=0;k<W;k++){ u64 v=(~Wd[k])+b; b=(b&&v==0)?1:0; Rab[k]=v; } } // negate
        else    { for(int k=0;k<W;k++) Rab[k]=Wd[k]; }
        // corr = (2 * x_hi * |R|) >> (h+1) limbs.  product x_hi(8)*|R|(10) -> up to 18 limbs.
        u128 pa[24]={0};
        for(int i=0;i<h;i++) for(int k=0;k<W;k++){ u128 p=(u128)xh[i]*Rab[k]; pa[i+k]+=(u64)p; pa[i+k+1]+=(u64)(p>>64); }
        u64 Pr[24]; { u128 c=0; for(int t=0;t<h+W+1;t++){ u128 tt=pa[t]+c; Pr[t]=(u64)tt; c=tt>>64; } }
        // 2*P then >>(h+1) limbs: take limbs [h+1, ...], with a <<1 (bit from limb h)
        u64 corr[20]; int CL=8;            // keep 9 corr limbs [0..8] -> from Pr[h+1-? ]
        // 2*Pr >> (h+1)*64 bits = Pr >> ((h+1)*64 - 1) bits = (Pr>>(h+1)limbs)<<1 | bit (Pr[h] top)
        for(int t=0;t<=CL;t++){ u64 lo=Pr[h+1+t], hi=Pr[h+2+t];
            corr[t]=(lo<<1)|((t==0?Pr[h]:Pr[h+t])>>63); }
        // simpler: corr[t] = (Pr[h+1+t]<<1)|(Pr[h+t]>>63)
        for(int t=0;t<=CL;t++) corr[t]=(Pr[h+1+t]<<1)|(Pr[h+t]>>63);
        u64 I[16];
        if(neg){
            u64 bw=0;
            for(int k=0;k<h;k++){ u128 s=(u128)0-corr[k]-bw; I[k]=(u64)s; bw=(s>>64)?1:0; }
            u128 s=(u128)xh[0]-corr[h]-bw; I[h]=(u64)s; bw=(s>>64)?1:0;
            for(int k=1;k<h;k++){ u128 t=(u128)xh[k]-bw; I[h+k]=(u64)t; bw=(t>>64)?1:0; }
        } else {
            for(int k=0;k<h;k++) I[k]=corr[k];
            u128 s=(u128)xh[0]+corr[h]; I[h]=(u64)s; u64 cy=(u64)(s>>64);
            for(int k=1;k<h;k++){ u128 t=(u128)xh[k]+cy; I[h+k]=(u64)t; cy=(u64)(t>>64); }
        }
        mpz_import(Iz,16,-1,8,0,0,I);
        double prec=precof(Iz,D);
        if(prec<minp)minp=prec; sump+=prec; cnt++;
    }
    printf("CLEAN4 full-limb 8->16: min=%.0f avg=%.1f neg=%ld/%ld %s\n",minp,sump/cnt,negc,cnt,minp>=1003?"OK":"SHORT");
    return 0;
}
