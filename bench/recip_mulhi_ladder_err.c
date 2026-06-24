// Model the negate-form mulhi-Newton reciprocal (matches reciprocal_u416.s),
// validate against the real asm, then measure error of the 1->2->4->8->16 ladder.
#define _GNU_SOURCE
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
extern void recip_mono(const unsigned long* D, unsigned long* xout);  // 7-limb asm

// one negate-form mulhi Newton step: x (p_in limbs) -> x (p_out limbs).
// Dtop = top p_out limbs of D (Dlimbs total).  Clean mulhi: discard low p_in limbs.
static void step(mpz_t x, const mpz_t D, int Dlimbs, int p_in, int p_out){
    mpz_t Dtop,P1,H1,t,mask,P2;
    mpz_inits(Dtop,P1,H1,t,mask,P2,NULL);
    mpz_fdiv_q_2exp(Dtop, D, (long)(Dlimbs-p_out)*64);     // top p_out limbs
    mpz_mul(P1, x, Dtop);
    mpz_fdiv_q_2exp(H1, P1, (long)p_in*64);                // discard low p_in
    mpz_fdiv_r_2exp(H1, H1, (long)p_out*64);               // keep p_out (mulhi)
    mpz_ui_pow_ui(mask,2,(unsigned long)p_out*64); mpz_sub_ui(mask,mask,1);
    mpz_sub(t, mask, H1);                                  // t = ~H1  (negate)
    mpz_mul(P2, x, t);
    mpz_fdiv_q_2exp(x, P2, (long)p_in*64 - 1);             // <<1 then high p_out
    mpz_fdiv_r_2exp(x, x, (long)p_out*64);
    mpz_clears(Dtop,P1,H1,t,mask,P2,NULL);
}
// seed + schedule.  schedule: array of (p_in,p_out) pairs, nsteps.
static void model(mpz_t x, const mpz_t D, int Dlimbs, const int sched[][2], int nsteps){
    // seed: x = floor((2^127-1)/D_top_limb)
    mpz_t num,dt; mpz_inits(num,dt,NULL);
    mpz_ui_pow_ui(num,2,127); mpz_sub_ui(num,num,1);
    mpz_fdiv_q_2exp(dt, D, (long)(Dlimbs-1)*64);           // top limb
    mpz_fdiv_q(x, num, dt);
    for(int s=0;s<nsteps;s++) step(x, D, Dlimbs, sched[s][0], sched[s][1]);
    mpz_clears(num,dt,NULL);
}
int main(int argc,char**argv){
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs, argc>1?atoi(argv[1]):1);
    mpz_t D,x,xa,xe,Epow,prod; mpz_inits(D,x,xa,xe,Epow,prod,NULL);

    // ---- (1) validate model vs real asm: schedule 1->2->4->7, Dlimbs=7 ----
    {
        int sched[3][2]={{1,2},{2,4},{4,7}};
        long mism=0, tot=0; long maxdiff=0;
        for(int t=0;t<20000;t++){
            mpz_urandomb(D,rs,7*64); mpz_setbit(D,7*64-1);     // normalized 7-limb
            unsigned long Dl[7],xo[7];
            for(int i=0;i<7;i++) Dl[i]=0;
            mpz_export(Dl,NULL,-1,8,0,0,D);
            recip_mono(Dl,xo);
            mpz_import(xa,7,-1,8,0,0,xo);                      // asm output
            model(x, D, 7, sched, 3);                          // model output
            tot++;
            if(mpz_cmp(x,xa)!=0){ mism++; mpz_sub(prod,x,xa); long d=labs(mpz_get_si(prod));
                if(mpz_cmpabs_ui(prod,1000000)>0) d=1000000; if(d>maxdiff)maxdiff=d; }
        }
        printf("[validate] model vs recip_mono asm (1->2->4->7): mism=%ld/%ld  max|model-asm|=%ld ULP\n",
               mism,tot,maxdiff);
    }

    // ---- (2) measure error of the user's ladder 1->2->4->8->16, Dlimbs=16 ----
    {
        int sched[4][2]={{1,2},{2,4},{4,8},{8,16}};
        // determine E once: x*D ~ 2^E.  Expect E = 2*1024-1 = 2047.
        long Emin=1<<30, Emax=0;
        long maxerr=0, sumerr=0; long histbits[80]={0}; long n=0;
        for(int t=0;t<200000;t++){
            mpz_urandomb(D,rs,16*64); mpz_setbit(D,16*64-1);
            model(x, D, 16, sched, 4);
            // E from x*D bit length
            mpz_mul(prod,x,D); long E=(long)mpz_sizeinbase(prod,2);
            if(E<Emin)Emin=E; if(E>Emax)Emax=E;
            mpz_ui_pow_ui(Epow,2,(unsigned long)E);
            mpz_fdiv_q(xe,Epow,D);                              // exact floor(2^E/D)
            mpz_sub(prod, xe, x);                               // error (>=0 expected: model underestimates)
            long ebits = mpz_sgn(prod)? (long)mpz_sizeinbase(prod,2) : 0;
            int neg = mpz_sgn(prod)<0;
            long e = mpz_cmpabs_ui(prod,2000000000)>0 ? 2000000000 : labs(mpz_get_si(prod));
            if(e>maxerr)maxerr=e; sumerr+=e>0?1:0; n++;
            if(neg) ebits=70+ (ebits>9?9:ebits);                // flag negatives in high bins
            if(ebits<80) histbits[ebits]++;
        }
        printf("\n[measure] ladder 1->2->4->8->16 (16-limb, %ld trials):\n", n);
        printf("  E (bits of x*D): min=%ld max=%ld  (exact = floor(2^E/D))\n", Emin, Emax);
        printf("  max error = %ld ULP  (~2^%.1f)\n", maxerr, maxerr>0?__builtin_log2((double)maxerr):0.0);
        printf("  error magnitude histogram (bits of (exact - model)):\n");
        for(int b=0;b<70;b++) if(histbits[b]) printf("    %2d-bit error: %ld\n", b, histbits[b]);
        long negs=0; for(int b=70;b<80;b++) negs+=histbits[b];
        if(negs) printf("    NEGATIVE errors (model OVERESTIMATES): %ld  <-- unexpected\n", negs);
    }
    return 0;
}
