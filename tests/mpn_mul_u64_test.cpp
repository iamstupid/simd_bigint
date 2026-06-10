// Differential test for the public u64 entry simd_mpn_mul: vs a u128
// schoolbook on u64 limbs. Covers the le6 path, the conversion boundary,
// top-limb bit-length edges (clz path), zero operands/high limbs, and all
// dispatch shape branches behind the conversion.
// Build: clang++ -O2 -march=native -I../include mpn_mul_u64_test.cpp \
//          ../src/x86_64/mul_basecase_le6.S -o /tmp/mut && /tmp/mut
#define SCRATCH_IMPLEMENTATION
#include "u64cvt.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef unsigned __int128 u128;
static uint64_t rng = 0xfeedfacefeedfaceull;
struct seed_init { seed_init(){ const char* e = getenv("SEED"); if(e) rng ^= strtoull(e, 0, 0); } } seed_init_;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static uint64_t rlimb64(void){
    switch(xr() & 7){
    case 0: return 0;
    case 1: return ~0ull;
    case 2: return 1;
    case 3: return 1ull << 63;
    default: return xr();
    }
}

static void ref_mul64(std::vector<uint64_t>& r, const uint64_t* a, size_t an,
                      const uint64_t* b, size_t bn){
    r.assign(an + bn, 0);
    for(size_t i = 0; i < an; i++){
        uint64_t c = 0;
        for(size_t j = 0; j < bn; j++){
            u128 t = (u128)a[i] * b[j] + r[i + j] + c;
            r[i + j] = (uint64_t)t;
            c = (uint64_t)(t >> 64);
        }
        r[i + bn] += c;
    }
}

static int run_case(const std::vector<uint64_t>& A, size_t an,
                    const std::vector<uint64_t>& B, size_t bn, const char* tag){
    std::vector<uint64_t> got(an + bn + 2, 0xabababababababab), ref;
    simd_mpn_mul(got.data(), A.data(), an, B.data(), bn);
    ref_mul64(ref, A.data(), an, B.data(), bn);
    for(size_t i = 0; i < an + bn; i++)
        if(got[i] != ref[i]){
            printf("FAIL [%s] an=%zu bn=%zu limb=%zu got=%016llx ref=%016llx\n",
                   tag, an, bn, i, (unsigned long long)got[i], (unsigned long long)ref[i]);
            return 1;
        }
    // entry must not write past an+bn
    if(got[an + bn] != 0xabababababababab || got[an + bn + 1] != 0xabababababababab){
        printf("FAIL [%s] an=%zu bn=%zu wrote past the result\n", tag, an, bn);
        return 1;
    }
    return 0;
}

int main(void){
    int fails = 0, cases = 0;

    // le6 region and the boundary into the converted path
    for(size_t an = 1; an <= 10; an++)
        for(size_t bn = 1; bn <= an; bn++)
            for(int t = 0; t < 24; t++){
                std::vector<uint64_t> A(an), B(bn);
                for(auto &x : A) x = rlimb64();
                for(auto &x : B) x = rlimb64();
                fails += run_case(A, an, B, bn, "small"); cases++;
            }

    // general grid: sizes across all dispatch branches, random + biased limbs
    const size_t ans[] = {7, 8, 12, 13, 16, 26, 40, 65, 80, 104, 130, 160, 208, 260, 400,
                          532, 650, 800, 1024, 1300};  // >= 3 composite levels
    for(size_t ai = 0; ai < sizeof(ans)/sizeof(*ans); ai++)
        for(size_t bi = 0; bi <= ai; bi++)
            for(int t = 0; t < 4; t++){
                size_t an = ans[ai], bn = ans[bi];
                std::vector<uint64_t> A(an), B(bn);
                for(auto &x : A) x = rlimb64();
                for(auto &x : B) x = rlimb64();
                fails += run_case(A, an, B, bn, "grid"); cases++;
            }

    // clz / bit-length edges on the top limb
    const uint64_t tops[] = {1, 2, 3, (1ull<<51), (1ull<<52)-1, (1ull<<52),
                             (1ull<<63), ~0ull};
    for(size_t ti = 0; ti < 8; ti++)
        for(size_t tj = 0; tj < 8; tj++){
            size_t an = 26, bn = 13;
            std::vector<uint64_t> A(an), B(bn);
            for(auto &x : A) x = xr();
            for(auto &x : B) x = xr();
            A[an-1] = tops[ti]; B[bn-1] = tops[tj];
            fails += run_case(A, an, B, bn, "tops"); cases++;
        }

    // zero high limbs and all-zero operands
    {
        std::vector<uint64_t> A(40), B(20, 0);
        for(auto &x : A) x = xr();
        for(size_t i = 10; i < 40; i++) A[i] = 0;          // a has zero top 3/4
        fails += run_case(A, 40, B, 20, "zeros"); cases++; // b entirely zero
        for(auto &x : B) x = xr();
        fails += run_case(A, 40, B, 20, "zerotop"); cases++;
    }

    // strip-mining behind the conversion
    {
        std::vector<uint64_t> A(620), B(90);
        for(auto &x : A) x = rlimb64();
        for(auto &x : B) x = rlimb64();
        fails += run_case(A, 620, B, 90, "strip"); cases++;
    }

    if(fails){ printf("FAILED: %d of %d cases\n", fails, cases); return 1; }
    printf("OK: %d u64 entry cases match scalar reference\n", cases);
    return 0;
}
