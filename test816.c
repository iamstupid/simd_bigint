#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
typedef uint64_t u64;
extern void recip_u512_mp(const u64* D, u64* out);
extern void recip_step_8_16(const u64* x_hi, const u64* D, u64* out);
int main(int argc,char**argv){
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs,argc>1?atoi(argv[1]):1);
    long TR=argc>2?atol(argv[2]):200000;
    mpz_t D,P,p2,e,d2,Iz; mpz_inits(D,P,p2,e,d2,Iz,NULL);
    double minp=1e9,sump=0; long cnt=0;
    for(long tr=0;tr<TR;tr++){
        mpz_urandomb(D,rs,16*64); mpz_setbit(D,16*64-1);
        u64 Dl[16]={0}; mpz_export(Dl,NULL,-1,8,0,0,D);
        u64 Dhi[8]; for(int i=0;i<8;i++)Dhi[i]=Dl[8+i];
        u64 xh[8]; recip_u512_mp(Dhi,xh);
        u64 I[16]; recip_step_8_16(xh, Dl, I);
        mpz_import(Iz,16,-1,8,0,0,I);
        mpz_mul(P,Iz,D); long Lb=mpz_sizeinbase(P,2);
        mpz_ui_pow_ui(p2,2,Lb); mpz_sub(e,p2,P); mpz_ui_pow_ui(d2,2,Lb-1); mpz_sub(d2,P,d2);
        long eh=mpz_sgn(e)?(long)mpz_sizeinbase(e,2):0, el=mpz_sgn(d2)?(long)mpz_sizeinbase(d2,2):0;
        double prec=(mpz_cmp(e,d2)<0)?(double)(Lb-eh):(double)(Lb-1-el);
        if(prec<minp)minp=prec; sump+=prec; cnt++;
    }
    printf("recip_step_8_16: min prec=%.0f avg=%.1f (need>=1003) %s\n",minp,sump/cnt,minp>=1003?"OK":"FAIL");
    return 0;
}
