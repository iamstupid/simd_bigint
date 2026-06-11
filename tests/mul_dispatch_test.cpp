// Differential test for the shape-aware mul_u52_dispatch: full (an, bn) grid
// vs scalar __int128 schoolbook, plus targeted strip-mining shapes.
// Build: clang++ -O2 -march=native -I../include mul_dispatch_test.cpp -o /tmp/mdt && /tmp/mdt
#include "mul.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define M52 ((1ull << 52) - 1)
typedef unsigned __int128 u128;

static uint64_t rng = 0xd15ea5ed15ea5ed1ull;
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

static int run_case(scratch* sc, size_t an, size_t bn){
    std::vector<uint64_t> A((an + 7) / 8 * 8, 0), B((bn + 7) / 8 * 8, 0);
    for(size_t i = 0; i < an; i++) A[i] = rdig();
    for(size_t i = 0; i < bn; i++) B[i] = rdig();
    size_t rl = an + bn, rv = (rl + 7) / 8;
    std::vector<uint64_t> got(rv * 8 + 8, 0), ref;
    mul_u52_dispatch((pvec)got.data(), (cpvec)A.data(), (cpvec)B.data(), an, bn, sc);
    mpn_canon_neg((pvec)got.data(), (cpvec)got.data(), rl);
    ref_mul(ref, A, an, B, bn);
    for(size_t i = 0; i < rl; i++){
        if(got[i] != ref[i]){
            printf("FAIL an=%zu bn=%zu limb=%zu got=%016llx ref=%016llx\n",
                   an, bn, i, (unsigned long long)got[i], (unsigned long long)ref[i]);
            return 1;
        }
    }
    return 0;
}

int main(void){
    scratch sc = scratch_create_ex(4096, 1u << 24, 1u << 24);
    int fails = 0, cases = 0;

    // dense small grid: every branch boundary at low sizes (incl. bn = 1)
    for(size_t an = 1; an <= 64; an += 1 + (an > 24))
        for(size_t bn = 1; bn <= an; bn += 1 + (bn > 16)){
            fails += run_case(&sc, an, bn); cases++;
        }

    // band/ratio coverage: bn around T22, T32OK, and into the X2 band; an
    // sweeps all ratio cutpoints (1, 5/4, 3/2, 7/4, 2, 5/2, 3, and beyond)
    const double ratios[] = {1.0, 1.13, 1.25, 1.4, 1.5, 1.6, 1.75, 1.9, 2.0,
                             2.3, 2.5, 2.8, 3.0, 3.4, 4.2, 5.5, 7.3};
    const size_t bns[] = {80, 85, 86, 90, 96, 100, 110, 128, 150, 200, 260, 321, 360};
    for(size_t bi = 0; bi < sizeof(bns)/sizeof(*bns); bi++)
        for(size_t ri = 0; ri < sizeof(ratios)/sizeof(*ratios); ri++){
            size_t bn = bns[bi];
            size_t an = (size_t)(bn * ratios[ri]) + (xr() % 3);
            if(an < bn) an = bn;
            fails += run_case(&sc, an, bn); cases++;
        }

    // deep strip-mining: several full strips plus assorted remainders
    const size_t strips[][2] = {{700, 90}, {730, 100}, {1100, 96}, {640, 86},
                                {903, 129}, {1000, 88}, {520, 100}, {560, 86},
                                // sliver-guard shapes: an mod 2bn in [0, 8]
                                {640, 80}, {645, 80}, {1040, 130}, {482, 96},
                                {966, 96}, {1287, 160}, {833, 104}};
    for(int k = 0; k < 15; k++){
        fails += run_case(&sc, strips[k][0], strips[k][1]); cases++;
    }

    scratch_destroy(&sc);
    if(fails){ printf("FAILED: %d of %d cases\n", fails, cases); return 1; }
    printf("OK: %d dispatch grid cases match scalar reference\n", cases);
    return 0;
}
