#define SCRATCH_IMPLEMENTATION
#define _GNU_SOURCE
#include "../include/divrem.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t to_u64(uint64_t* L, uint64_t cap, const mpz_t X){
    size_t cnt=0; for(uint64_t i=0;i<cap;i++)L[i]=0; mpz_export(L,&cnt,-1,8,0,0,X); return cnt?cnt:1;
}
static long BAD=0, TOT=0;
static void check(const char* tag, const mpz_t N, const mpz_t D){
    if(mpz_sgn(D)==0 || mpz_sgn(N)<0) return;    // kernel is unsigned
    static uint64_t Nl[300],Dl[80],Ql[300],Rl[80];
    uint64_t nn=to_u64(Nl,300,N), dn=to_u64(Dl,80,D);
    if(u64_bit_length(Dl,dn)<=416) return;       // need dn>=2 blocks
    uint64_t qo,ro; divrem_u64(Ql,Rl,Nl,nn,Dl,dn,&qo,&ro);
    mpz_t Q,R,Qc,Rc; mpz_inits(Q,R,Qc,Rc,NULL);
    mpz_import(Q,qo,-1,8,0,0,Ql); mpz_import(R,ro,-1,8,0,0,Rl);
    mpz_tdiv_qr(Qc,Rc,N,D);
    TOT++;
    if(mpz_cmp(Q,Qc)||mpz_cmp(R,Rc)){ BAD++; if(BAD<=8) gmp_printf("BAD[%s] Nb=%lu Db=%lu\n  Q=%Zx\n Qc=%Zx\n  R=%Zx\n Rc=%Zx\n",
        tag, mpz_sizeinbase(N,2), mpz_sizeinbase(D,2), Q,Qc,R,Rc); }
    mpz_clears(Q,R,Qc,Rc,NULL);
}

int main(int c,char**v){
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs, c>1?strtoul(v[1],0,0):1);
    mpz_t N,D,k,r; mpz_inits(N,D,k,r,NULL);
    for(int t=0;t<60000;t++){
        unsigned db = 417 + gmp_urandomm_ui(rs, 2000);
        mpz_urandomb(D,rs,db); mpz_setbit(D,db-1);
        // 1) exact multiple: R must be 0
        mpz_urandomb(k,rs, 1+gmp_urandomm_ui(rs,1700)); mpz_mul(N,D,k); check("mult",N,D);
        // 2) multiple minus 1
        mpz_sub_ui(N,N,1); check("mult-1",N,D);
        // 3) N == D  (Q=1,R=0)
        check("eqD",D,D);
        // 4) N == D-1 (Q=0,R=D-1)
        mpz_sub_ui(N,D,1); check("D-1",N,D);
        // 5) k*D + (D-1)  (max remainder)
        mpz_urandomb(k,rs,1+gmp_urandomm_ui(rs,1700)); mpz_mul(N,D,k); mpz_add(N,N,D); mpz_sub_ui(N,N,1); check("maxrem",N,D);
        // 6) k*D + small
        mpz_urandomb(k,rs,1+gmp_urandomm_ui(rs,1700)); mpz_mul(N,D,k);
        mpz_urandomb(r,rs, db>64?64:db-1); mpz_add(N,N,r); check("smallrem",N,D);
        // 7) divisor exactly 2 blocks (832 bits) -> no-low-submul path
        mpz_urandomb(D,rs,832); mpz_setbit(D,831);
        mpz_urandomb(N,rs, 833 + gmp_urandomm_ui(rs,2000)); check("dn2",N,D);
        // 8) block-boundary bit lengths
        mpz_set_ui(D,0); mpz_setbit(D, 416*2 ); mpz_setbit(D, 416*2-1); // ~ top of block 2
        mpz_urandomb(N,rs, 416*4); check("boundary",N,D);
    }
    printf("divrem edge: BAD=%ld / %ld\n", BAD, TOT);
    return BAD?1:0;
}
