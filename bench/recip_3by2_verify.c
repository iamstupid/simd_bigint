// Verifies the 3/2 block reciprocal seed bound for 3by2_recip.md.
//
//   B = 2^416 (8 lanes x 52b).  I normalized: 2^831 <= I < 2^832 (u64[13]).
//   D    = top 7 u64 of I = floor(I/2^384)            (448b, normalized)
//   R    = recip_mono(D) ~ floor(2^895/D)             (448b)
//   seed = floor(R/2^31) - 2^416                      (416b block) == from_u64_for_reciprocal(R)
//   V    = floor((2^1248-1)/I) - 2^416                (exact 3/2 reciprocal)
//
// Claim (verified here): seed - V in {-1, 0}  (one-sided, |err| <= 1).
//
//   modes:  random | adversarial | boundary <offset>
//   build:  gcc -O3 -march=native recip_3by2_verify.c ../reciprocal_u416.s -o /tmp/v -lgmp
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void recip_mono(const uint64_t *D, uint64_t *xout);
static uint64_t rng(uint64_t *s){ uint64_t x=*s; x^=x<<7; x^=x>>9; x*=0x9e3779b97f4a7c15ull; *s=x; return x; }

static mpz_t P2_416, P2_1248, P2_831, P2_832, mbad;
static long g_lo=1<<30, g_hi=-(1<<30), hist[9]={0}, tested=0; static int have_bad=0;

static void test_I(const mpz_t mI){
  if(mpz_cmp(mI,P2_831)<0 || mpz_cmp(mI,P2_832)>=0) return;     // require normalized
  uint64_t I[13]={0}, D[7], R[7]; size_t cnt=0;
  mpz_export(I,&cnt,-1,8,0,0,mI);
  memcpy(D,I+6,sizeof(D));
  recip_mono(D,R);
  mpz_t mR,V,num,seed,tmp; mpz_inits(mR,V,num,seed,tmp,NULL);
  mpz_import(mR,7,-1,8,0,0,R);
  mpz_fdiv_q_2exp(seed,mR,31); mpz_sub(seed,seed,P2_416);       // seed = floor(R/2^31)-2^416
  mpz_sub_ui(num,P2_1248,1); mpz_fdiv_q(V,num,mI); mpz_sub(V,V,P2_416);
  mpz_sub(tmp,seed,V); long e=mpz_get_si(tmp);
  if(e<g_lo)g_lo=e; if(e>g_hi)g_hi=e;
  long c=e<-4?-4:(e>4?4:e); hist[c+4]++; tested++;
  if((e>1||e<-1)&&!have_bad){ mpz_set(mbad,mI); have_bad=1; }
  mpz_clears(mR,V,num,seed,tmp,NULL);
}
static void test_u64(const uint64_t I[13]){ mpz_t mI; mpz_init(mI); mpz_import(mI,13,-1,8,0,0,I); test_I(mI); mpz_clear(mI); }

int main(int argc,char**argv){
  mpz_inits(P2_416,P2_1248,P2_831,P2_832,mbad,NULL);
  mpz_ui_pow_ui(P2_416,2,416); mpz_ui_pow_ui(P2_1248,2,1248);
  mpz_ui_pow_ui(P2_831,2,831); mpz_ui_pow_ui(P2_832,2,832);
  const char *mode = argc>1?argv[1]:"random";
  uint64_t s = argc>2?strtoull(argv[2],0,0):0xC0FFEEULL;

  if(!strcmp(mode,"random")){
    for(long t=0;t<5000000;t++){ uint64_t I[13]; for(int i=0;i<13;i++)I[i]=rng(&s);
      I[12]|=0x8000000000000000ull; test_u64(I); }
  } else if(!strcmp(mode,"adversarial")){
    for(long t=0;t<3000000;t++){ uint64_t I[13]; for(int i=0;i<6;i++)I[i]=~0ull;     // low 384b all 1
      for(int i=6;i<13;i++)I[i]=rng(&s); I[12]|=0x8000000000000000ull; test_u64(I); }
    for(long t=0;t<2000000;t++){ uint64_t I[13]; for(int i=0;i<6;i++)I[i]=~0ull;     // minimal high (D~2^447)
      for(int i=6;i<12;i++)I[i]=rng(&s)&((t&1)?0:0xFFFF); I[12]=0x8000000000000000ull|(rng(&s)&((t&3)?0:7)); test_u64(I); }
    for(long k=0;k<2000000;k++){ uint64_t I[13]={0}; I[12]=0x8000000000000000ull; I[0]=(uint64_t)k; test_u64(I); } // 2^831+k
  } else { // boundary: force 2^1248/I onto integer m
    unsigned long off = argc>3?strtoul(argv[3],0,0):0;
    mpz_t m,I; mpz_inits(m,I,NULL);
    for(int region=0;region<2;region++){
      if(region==0){ mpz_set(m,P2_416); mpz_add_ui(m,m,1+off); }            // m~2^416 -> I~2^832
      else { mpz_mul_ui(m,P2_416,2); mpz_sub_ui(m,m,1+off); }              // m~2^417 -> I~2^831 (small D)
      for(long step=0;step<6000000;step++){
        if(region==0) mpz_add_ui(m,m,1); else mpz_sub_ui(m,m,1);
        mpz_fdiv_q(I,P2_1248,m);
        for(int d=-2;d<=2;d++){ mpz_t Id; mpz_init(Id); mpz_set(Id,I);
          if(d>=0)mpz_add_ui(Id,Id,d); else mpz_sub_ui(Id,Id,-d); test_I(Id); mpz_clear(Id); }
      }
    }
  }
  printf("[%s] tested=%ld  err=seed-V in [%ld,%ld]\n",mode,tested,g_lo,g_hi);
  for(int i=0;i<9;i++) if(hist[i]) printf("  err=%+d : %ld\n",i-4,hist[i]);
  if(have_bad) gmp_printf("  !!! |err|>1 at I=%#Zx\n",mbad);
  return 0;
}
