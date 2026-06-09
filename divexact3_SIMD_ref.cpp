// divexact3 for u52 limbs (little-endian), in place. Precondition: value is a multiple of 3.
// Build (Zen 4/5): gcc -O3 -march=znver5  divexact3.c -o d3   (or -march=znver4)
//        generic : gcc -O3 -mavx512f -mavx512vl -mavx512dq -mavx512ifma -mavx512vpopcntdq divexact3.c -o d3
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define M52 ((1ULL<<52)-1)
static const uint64_t DINV3 = 3002399751580331ULL; // 3^{-1} mod 2^52

/* ---------------- scalar reference (ground truth) ---------------- */
static uint64_t divexact3_scalar(uint64_t *a, size_t n){
    uint64_t c = 0;
    for(size_t i=0;i<n;i++){
        uint64_t s = (a[i]-c) & M52;
        uint64_t q = (s*DINV3) & M52;
        int64_t num = (int64_t)(3*q) - (int64_t)(a[i]-c);
        c = (uint64_t)(num >> 52);
        a[i] = q;
    }
    return c; // 0 iff 3 | value
}

/* ---------------- shared vector helpers ---------------- */
// mod 3, exact on [0, 8191]  (m=2731, s=13 ; 3m = 2^13+1)
static inline __m512i mod3(__m512i x){
    __m512i q = _mm512_srli_epi64(_mm512_mullo_epi64(x, _mm512_set1_epi64(2731)), 13);
    return _mm512_sub_epi64(x, _mm512_add_epi64(_mm512_slli_epi64(q,1), q));
}
// x << k limbs toward high lane, low lanes zero-filled
static inline __m512i shl1(__m512i x){ return _mm512_alignr_epi64(x,_mm512_setzero_si512(),7); }
static inline __m512i shl2(__m512i x){ return _mm512_alignr_epi64(x,_mm512_setzero_si512(),6); }
static inline __m512i shl4(__m512i x){ return _mm512_alignr_epi64(x,_mm512_setzero_si512(),4); }

// residue ax_i = (a_i mod 3) lifted: pc(even bits) + 2*pc(odd bits), since 2^i = (-1)^i mod 3
static inline __m512i ax_of(__m512i a){
    __m512i pe = _mm512_popcnt_epi64(_mm512_and_si512(a,_mm512_set1_epi64(0x5555555555555ULL))); // even bit positions
    __m512i po = _mm512_popcnt_epi64(_mm512_and_si512(a,_mm512_set1_epi64(0xAAAAAAAAAAAAAULL))); // odd bit positions
    return _mm512_add_epi64(pe, _mm512_add_epi64(po,po));  // per lane <= 26 + 2*26 = 78
}
// raw inclusive prefix sum of residues across the 8 lanes (NO mod3 mid-chain)
static inline __m512i prefix(__m512i ax){
    ax = _mm512_add_epi64(ax, shl1(ax));
    ax = _mm512_add_epi64(ax, shl2(ax));
    return _mm512_add_epi64(ax, shl4(ax)); // per lane <= 78*8 = 624
}
// given block residues+inclusive prefix P (incl. carry-in) and own residues ax -> store quotient limbs
static inline __m512i quotient(__m512i v, __m512i P, __m512i ax, __m512i vd, __m512i z){
    __m512i E = _mm512_sub_epi64(P, ax);            // exclusive raw prefix (borrow position, pre-negate)
    __m512i b = mod3(_mm512_add_epi64(E,E));         // borrow = (-E) mod 3 = 2E mod 3, in {0,1,2}
    __m512i s = _mm512_and_si512(_mm512_sub_epi64(v,b), _mm512_set1_epi64(M52));
    return _mm512_madd52lo_epu64(z, s, vd);          // lo52(s * dinv3)
}
// extract lane 7 to scalar (top 128b lane -> qword index 1)
static inline uint64_t lane7(__m512i v){
    return (uint64_t)_mm_extract_epi64(_mm512_extracti64x2_epi64(v,3), 1);
}

/* =====================================================================
   A) AUTO: single-vector loop, let the compiler unroll.    <-- the one you asked for
   ===================================================================== */
void divexact3_auto(uint64_t *a, size_t n){
    uint64_t carry = 0;                 // inclusive residue total entering current block, in {0,1,2}
    size_t i = 0;
    const __m512i vd = _mm512_set1_epi64(DINV3), z = _mm512_setzero_si512();
    #pragma GCC unroll 4
    for(; i+8<=n; i+=8){
        __m512i v  = _mm512_loadu_si512(a+i);
        __m512i ax = ax_of(v);
        __m512i P  = _mm512_add_epi64(prefix(ax), _mm512_set1_epi64(carry)); // <= 624 + 2
        carry = lane7(P) % 3;           // carry-out -> next block carry-in
        _mm512_storeu_si512(a+i, quotient(v, P, ax, vd, z));
    }
    // scalar tail, seeded with borrow = (-carry) mod 3
    uint64_t c = (3 - carry) % 3;
    for(; i<n; i++){
        uint64_t s=(a[i]-c)&M52; uint64_t q=(s*DINV3)&M52;
        int64_t num=(int64_t)(3*q)-(int64_t)(a[i]-c); c=(uint64_t)(num>>52); a[i]=q;
    }
}

/* =====================================================================
   B) HAND: explicit 4-way software pipeline (carry stays in-vector between blocks).
   ===================================================================== */
void divexact3_hand(uint64_t *a, size_t n){
    uint64_t carry = 0;
    size_t i = 0;
    const __m512i vd=_mm512_set1_epi64(DINV3), z=_mm512_setzero_si512(), L7=_mm512_set1_epi64(7);
    for(; i+32<=n; i+=32){
        __m512i a0=_mm512_loadu_si512(a+i),    b0=_mm512_loadu_si512(a+i+8);
        __m512i c0=_mm512_loadu_si512(a+i+16), d0=_mm512_loadu_si512(a+i+24);
        __m512i xa=ax_of(a0), xb=ax_of(b0), xc=ax_of(c0), xd=ax_of(d0); // independent
        __m512i pa=prefix(xa), pb=prefix(xb), pc=prefix(xc), pd=prefix(xd); // independent
        pa=_mm512_add_epi64(pa,_mm512_set1_epi64(carry));
        pb=_mm512_add_epi64(pb,_mm512_permutexvar_epi64(L7,pa));   // splat lane7, stay in-vector
        pc=_mm512_add_epi64(pc,_mm512_permutexvar_epi64(L7,pb));
        pd=_mm512_add_epi64(pd,_mm512_permutexvar_epi64(L7,pc));
        carry = lane7(pd) % 3;                                     // one scalar extract per 32 limbs
        _mm512_storeu_si512(a+i,    quotient(a0,pa,xa,vd,z));
        _mm512_storeu_si512(a+i+8,  quotient(b0,pb,xb,vd,z));
        _mm512_storeu_si512(a+i+16, quotient(c0,pc,xc,vd,z));
        _mm512_storeu_si512(a+i+24, quotient(d0,pd,xd,vd,z));
    }
    uint64_t c = (3 - carry) % 3;
    for(; i<n; i++){
        uint64_t s=(a[i]-c)&M52; uint64_t q=(s*DINV3)&M52;
        int64_t num=(int64_t)(3*q)-(int64_t)(a[i]-c); c=(uint64_t)(num>>52); a[i]=q;
    }
}

/* ---------------- self-test + benchmark ---------------- */
static inline uint64_t rd(){ unsigned a,d; __asm__ volatile("rdtscp":"=a"(a),"=d"(d)::"rcx"); return ((uint64_t)d<<32)|a; }
static uint64_t rng=0x243F6A8885A308D3ULL;
static uint64_t rnd(){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return rng; }

static int check(void(*fn)(uint64_t*,size_t), uint64_t*src, size_t n, const char*name){
    uint64_t *x=malloc(n*8), *y=malloc(n*8);
    memcpy(x,src,n*8); memcpy(y,src,n*8);
    divexact3_scalar(x,n); fn(y,n);
    int ok = memcmp(x,y,n*8)==0;
    if(!ok) for(size_t i=0;i<n;i++) if(x[i]!=y[i]){ printf("  %s mismatch @%zu scalar=%013llx vec=%013llx\n",name,i,(unsigned long long)x[i],(unsigned long long)y[i]); break; }
    free(x); free(y); return ok;
}

int main(){
    // correctness: build random arrays, force divisibility (sum of limbs mod 3 == value mod 3)
    int all=1;
    size_t ns[]={32,33,40,97,1000,1003};
    for(int k=0;k<6;k++){
        size_t n=ns[k]; uint64_t*a=malloc(n*8);
        for(size_t i=0;i<n;i++)a[i]=rnd()&M52;
        uint64_t r=0; for(size_t i=0;i<n;i++)r=(r+a[i]%3)%3; a[0]=(a[0]-r)&M52;
        all &= check(divexact3_auto,a,n,"auto");
        all &= check(divexact3_hand,a,n,"hand");
        free(a);
    }
    { size_t n=128; uint64_t*a=calloc(n,8); a[0]=3; all&=check(divexact3_auto,a,n,"auto_zerotail"); all&=check(divexact3_hand,a,n,"hand_zerotail"); free(a); }
    { size_t n=64; uint64_t*a=calloc(n,8); for(size_t i=0;i<n;i++)a[i]=M52; uint64_t r=0;for(size_t i=0;i<n;i++)r=(r+a[i]%3)%3;a[0]=(a[0]-r)&M52; all&=check(divexact3_auto,a,n,"auto_ones"); all&=check(divexact3_hand,a,n,"hand_ones"); free(a); }
    printf("correctness: %s\n\n", all?"ALL PASS":"FAILED");
    if(!all) return 1;

    // benchmark
    size_t sizes[]={1024,8192,65536,1048576};
    const char*lbl[]={"8KB","64KB","512KB","8MB"};
    printf("%-8s %12s %12s %8s\n","wset","auto c/limb","hand c/limb","ratio");
    for(int si=0;si<4;si++){
        size_t n=sizes[si]; uint64_t*a=aligned_alloc(64,n*8),*b=aligned_alloc(64,n*8);
        for(size_t i=0;i<n;i++)a[i]=rnd()&M52; uint64_t r=0;for(size_t i=0;i<n;i++)r=(r+a[i]%3)%3;a[0]=(a[0]-r)&M52;
        int reps=(n<=65536)?20000:200;
        double ba=1e30,bh=1e30;
        for(int t=0;t<7;t++){memcpy(b,a,n*8);uint64_t t0=rd();for(int q=0;q<reps;q++)divexact3_auto(b,n);uint64_t t1=rd();double c=(double)(t1-t0)/((double)reps*n);if(c<ba)ba=c;}
        for(int t=0;t<7;t++){memcpy(b,a,n*8);uint64_t t0=rd();for(int q=0;q<reps;q++)divexact3_hand(b,n);uint64_t t1=rd();double c=(double)(t1-t0)/((double)reps*n);if(c<bh)bh=c;}
        printf("%-8s %12.3f %12.3f %7.2fx\n",lbl[si],ba,bh,ba/bh);
        free(a);free(b);
    }
    return 0;
}