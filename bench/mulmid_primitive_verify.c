// Verify mpn_mulmid against the EXACT middle-product definition:
// MP = sum_{i+j in [bn-1, an-1]} a_i b_j B^{i+j-(bn-1)}, output an-bn+3 limbs.
#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

extern void __gmpn_mulmid(mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t);

int main(int argc,char**argv){
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs, argc>1?atoi(argv[1]):1);
    mpz_t ref,got,term; mpz_inits(ref,got,term,NULL);
    long fails=0, tot=0;
    for(int t=0;t<20000;t++){
        mp_size_t an = 2 + gmp_urandomm_ui(rs, 80);
        mp_size_t bn = 1 + gmp_urandomm_ui(rs, an);     // 1<=bn<=an
        mp_size_t rn = an-bn+3;
        mp_limb_t *ap=calloc(an,8), *bp=calloc(bn,8), *rp=calloc(rn,8);
        for(mp_size_t i=0;i<an;i++) ap[i]= (mp_limb_t)(0x9e3779b97f4a7c15ULL*(i*7+t+1));
        for(mp_size_t j=0;j<bn;j++) bp[j]= (mp_limb_t)(0xc2b2ae3d27d4eb4fULL*(j*5+t+3));
        __gmpn_mulmid(rp, ap, an, bp, bn);
        mpz_import(got, rn, -1,8,0,0, rp);

        // exact MP definition
        mpz_set_ui(ref,0);
        for(mp_size_t i=0;i<an;i++) for(mp_size_t j=0;j<bn;j++){
            mp_size_t k=i+j;
            if(k<bn-1 || k>an-1) continue;
            mpz_set_ui(term, ap[i]);
            mpz_mul_ui(term, term, bp[j]);          // a_i*b_j (128-bit)
            mpz_mul_2exp(term, term, 64*(k-(bn-1)));
            mpz_add(ref, ref, term);
        }
        tot++;
        if(mpz_cmp(got,ref)!=0){ fails++;
            if(fails<=8){ mpz_t d; mpz_init(d); mpz_sub(d,got,ref);
                gmp_printf("MISMATCH an=%ld bn=%ld  diffbits=%ld sgn=%d\n",
                           (long)an,(long)bn,(long)(mpz_sgn(d)?mpz_sizeinbase(d,2):0),mpz_sgn(d));
                mpz_clear(d); }
        }
        free(ap);free(bp);free(rp);
    }
    printf("mpn_mulmid vs exact MP definition: MISMATCHES=%ld / %ld\n", fails, tot);
    return 0;
}
