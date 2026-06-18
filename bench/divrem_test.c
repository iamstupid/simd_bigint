#define SCRATCH_IMPLEMENTATION
#define _GNU_SOURCE
#include "../include/divrem.h"
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

// import/export helpers (little-endian u64 limbs)
static uint64_t mpz_to_u64(uint64_t* L, uint64_t cap, const mpz_t X){
    size_t cnt = 0;
    for(uint64_t i=0;i<cap;i++) L[i]=0;
    mpz_export(L, &cnt, -1, 8, 0, 0, X);
    return cnt ? cnt : 1;
}
static void u64_to_mpz(mpz_t X, const uint64_t* L, uint64_t n){
    mpz_import(X, n, -1, 8, 0, 0, L);
}

int main(int argc, char** argv){
    gmp_randstate_t rs; gmp_randinit_default(rs);
    gmp_randseed_ui(rs, argc>1 ? strtoul(argv[1],0,0) : 1);
    mpz_t N,D,Q,R, Qc,Rc, t; mpz_inits(N,D,Q,R,Qc,Rc,t,NULL);

    long bad=0, total=0, rare=0;
    // sweep divisor bit lengths (force > 416 so dn>=2) and dividend lengths
    for(int trial=0; trial<200000; trial++){
        // divisor: 417 .. 2200 bits
        unsigned dbits = 417 + gmp_urandomm_ui(rs, 1784);
        // dividend: dbits-50 .. dbits+1500 bits
        long extra = (long)gmp_urandomm_ui(rs, 1600) - 50;
        long nbits = (long)dbits + extra; if(nbits < 1) nbits = 1;
        mpz_urandomb(D, rs, dbits); mpz_setbit(D, dbits-1);   // ensure top bit
        mpz_urandomb(N, rs, (mp_bitcnt_t)nbits);
        if(mpz_sgn(D)==0) continue;

        // occasionally force the rare n2==d1 path: make N a near-multiple of D
        if((trial & 31)==0){
            mpz_urandomb(t, rs, dbits>200?200:dbits);     // small remainder
            mpz_mul_ui(Q, D, 1 + gmp_urandomm_ui(rs, 1u<<20));
            mpz_mul_2exp(Q, Q, gmp_urandomm_ui(rs, 800)); // shift up
            mpz_add(N, Q, t);
        }

        static uint64_t Nl[256], Dl[64], Ql[256], Rl[64];
        uint64_t nn = mpz_to_u64(Nl, 256, N);
        uint64_t dn = mpz_to_u64(Dl, 64, D);
        if(u64_bit_length(Dl,dn) <= 416) continue;   // need dn>=2 blocks

        uint64_t qn_out, rn_out;
        divrem_u64(Ql, Rl, Nl, nn, Dl, dn, &qn_out, &rn_out);

        u64_to_mpz(Q, Ql, qn_out);
        u64_to_mpz(R, Rl, rn_out);
        mpz_tdiv_qr(Qc, Rc, N, D);

        total++;
        if(mpz_cmp(Q,Qc)!=0 || mpz_cmp(R,Rc)!=0){
            if(bad<5){
                gmp_printf("MISMATCH trial=%d  Nbits=%ld Dbits=%u\n", trial, nbits, dbits);
                gmp_printf("  Q  =%Zx\n  Qc =%Zx\n  R  =%Zx\n  Rc =%Zx\n", Q,Qc,R,Rc);
            }
            bad++;
        }
    }
    printf("divrem_u64: bad=%ld / %ld  (rare-path-ish forced ~%ld)\n", bad, total, total/32);
    mpz_clears(N,D,Q,R,Qc,Rc,t,NULL);
    return bad?1:0;
}
