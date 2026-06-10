// Differential test: mul_u52_toom32 vs scalar __int128 schoolbook, sweeping
// the toom32 shape window, plus degenerate vm1 == 0 cases.
// Build: clang++ -O2 -march=native -I../include toom32_test.cpp -o /tmp/t32 && /tmp/t32
#include "mul.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define M52 ((1ull << 52) - 1)
typedef unsigned __int128 u128;

static uint64_t rng = 0x5eed5eed5eed5eedull;
struct seed_init { seed_init(){ const char* e = getenv("SEED"); if(e) rng ^= strtoull(e, 0, 0); } } seed_init_;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static uint64_t rdig(void){
    switch(xr() & 7){
    case 0: return 0;
    case 1: return M52;
    case 2: return M52 - 1;
    default: return xr() & M52;
    }
}

static void ref_mul(std::vector<uint64_t>& r, const std::vector<uint64_t>& A, size_t an,
                    const std::vector<uint64_t>& B, size_t bn){
    std::vector<u128> acc(an + bn, 0);
    for(size_t i = 0; i < an; i++)
        for(size_t j = 0; j < bn; j++)
            acc[i + j] += (u128)A[i] * B[j];
    r.assign(an + bn, 0);
    u128 c = 0;
    for(size_t i = 0; i < an + bn; i++){
        u128 v = acc[i] + c;
        r[i] = (uint64_t)(v & M52);
        c = v >> 52;
    }
    assert(c == 0);
}

static int run_case(scratch* sc, const std::vector<uint64_t>& A, size_t an,
                    const std::vector<uint64_t>& B, size_t bn, const char* tag){
    size_t rl = an + bn, rv = (rl + 7) / 8;
    std::vector<uint64_t> got(rv * 8 + 8, 0), ref;
    mul_u52_toom32((pvec)got.data(), (cpvec)A.data(), (cpvec)B.data(), an, bn, sc);
    mpn_canon_neg((pvec)got.data(), (cpvec)got.data(), rl);
    ref_mul(ref, A, an, B, bn);
    for(size_t i = 0; i < rl; i++){
        if(got[i] != ref[i]){
            printf("FAIL [%s] an=%zu bn=%zu limb=%zu got=%016llx ref=%016llx\n",
                   tag, an, bn, i, (unsigned long long)got[i], (unsigned long long)ref[i]);
            return 1;
        }
    }
    return 0;
}

int main(void){
    scratch sc = scratch_create_ex(4096, 1u << 22, 1u << 22);
    int fails = 0, cases = 0;

    // full shape window sweep for a set of bn values
    const size_t bns[] = {10, 12, 16, 24, 33, 40, 56, 64, 70};
    for(size_t bi = 0; bi < sizeof(bns)/sizeof(*bns); bi++){
        size_t bn = bns[bi];
        for(size_t an = bn + 2; an + 6 <= 3 * bn; an++){
            uint64_t n = (2*an >= 3*bn) ? (an + 2) / 3 : ((bn + 1) >> 1);
            uint64_t s = an - 2*n, t = bn - n;
            if(!(0 < s && s <= n && 0 < t && t <= n && s + t > n)) continue;
            std::vector<uint64_t> A((an + 7) / 8 * 8, 0), B((bn + 7) / 8 * 8, 0);
            for(size_t i = 0; i < an; i++) A[i] = rdig();
            for(size_t i = 0; i < bn; i++) B[i] = rdig();
            fails += run_case(&sc, A, an, B, bn, "sweep"); cases++;

            // degenerate A(-1) == 0: a2 = 0, a1 = a0
            for(size_t i = 0; i < n; i++) A[n + i] = A[i];
            for(size_t i = 0; i < s; i++) A[2*n + i] = 0;
            fails += run_case(&sc, A, an, B, bn, "am1=0"); cases++;

            // degenerate B(-1) == 0: b0 high zero, b1 = b0 low
            for(size_t i = t; i < n; i++) B[i] = 0;
            for(size_t i = 0; i < t; i++) B[n + i] = B[i];
            fails += run_case(&sc, A, an, B, bn, "bm1=0"); cases++;
        }
    }

    // a few larger shapes that recurse through karatsuba
    const size_t big[][2] = {{150, 100}, {192, 128}, {210, 144}, {260, 170}};
    for(int k = 0; k < 4; k++){
        size_t an = big[k][0], bn = big[k][1];
        uint64_t n = (2*an >= 3*bn) ? (an + 2) / 3 : ((bn + 1) >> 1);
        uint64_t s = an - 2*n, t = bn - n;
        if(!(0 < s && s <= n && 0 < t && t <= n && s + t > n)) continue;
        for(int trial = 0; trial < 8; trial++){
            std::vector<uint64_t> A((an + 7) / 8 * 8, 0), B((bn + 7) / 8 * 8, 0);
            for(size_t i = 0; i < an; i++) A[i] = rdig();
            for(size_t i = 0; i < bn; i++) B[i] = rdig();
            fails += run_case(&sc, A, an, B, bn, "big"); cases++;
        }
    }

    scratch_destroy(&sc);
    if(fails){ printf("FAILED: %d of %d cases\n", fails, cases); return 1; }
    printf("OK: %d toom32 cases match scalar reference\n", cases);
    return 0;
}
