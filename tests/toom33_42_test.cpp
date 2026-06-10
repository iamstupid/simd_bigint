// Differential test: mul_u52_toom33 / mul_u52_toom42 vs scalar __int128
// schoolbook, sweeping each shape window, plus degenerate vm1 == 0 cases and
// a direct divexact3 spot-check.
// Build: clang++ -O2 -march=native -I../include toom33_42_test.cpp -o /tmp/t3342 && /tmp/t3342
#include "mul.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define M52 ((1ull << 52) - 1)
typedef unsigned __int128 u128;

static uint64_t rng = 0x7007007007007007ull;
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

typedef int (*toomfn)(pvec, cpvec, cpvec, uint64_t, uint64_t, scratch*);

static int run_case(scratch* sc, toomfn fn, const std::vector<uint64_t>& A, size_t an,
                    const std::vector<uint64_t>& B, size_t bn, const char* tag){
    size_t rl = an + bn, rv = (rl + 7) / 8;
    std::vector<uint64_t> got(rv * 8 + 8, 0), ref;
    fn((pvec)got.data(), (cpvec)A.data(), (cpvec)B.data(), an, bn, sc);
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

static int divexact3_spot(void){
    int fails = 0;
    for(int it = 0; it < 20000; ++it){
        size_t n = 1 + (xr() % 64), vp = (n + 7) / 8 * 8;
        std::vector<uint64_t> a(vp, 0), ref(vp, 0);
        for(size_t i = 0; i < n; i++) a[i] = rdig();
        uint64_t r3 = 0;
        for(size_t i = 0; i < n; i++) r3 = (r3 + a[i] % 3) % 3;
        if(r3){                                    // force divisibility, no wrap
            if(a[0] <= M52 - (3 - r3)) a[0] += 3 - r3;
            else a[0] -= r3;
        }
        // scalar reference
        uint64_t c = 0;
        const uint64_t DINV3 = 3002399751580331ull;
        for(size_t i = 0; i < vp; i++){
            uint64_t s = (a[i] - c) & M52, q = (s * DINV3) & M52;
            long long num = (long long)(3 * q) - (long long)(a[i] - c);
            c = (uint64_t)(num >> 52);
            ref[i] = q;
        }
        std::vector<uint64_t> got(a);
        for(size_t i = n; i < vp; i++) got[i] = 0xabababababababaull;  // tail sentinel
        mpn_u52_divexact3((pvec)got.data(), (cpvec)got.data(), n);
        for(size_t i = 0; i < n; i++) if(got[i] != ref[i]){ fails++; break; }
        for(size_t i = n; i < vp; i++) if(got[i] != 0xabababababababaull){ fails++; break; }
    }
    return fails;
}

int main(void){
    scratch sc = scratch_create_ex(4096, 1u << 22, 1u << 22);
    int fails = 0, cases = 0;

    int d3 = divexact3_spot();
    if(d3) printf("divexact3 spot FAILS: %d\n", d3);
    fails += d3;

    // toom33: n = ceil(an/3), bn in (2n, an]
    for(size_t an = 8; an <= 180; an += (an < 60 ? 1 : 3)){
        uint64_t n = (an + 2) / 3;
        if(n < 2) continue;
        for(size_t bn = 2*n + 1; bn <= an; bn += 3){
            uint64_t s = an - 2*n, t = bn - 2*n;
            if(!(0 < s && s <= n && 0 < t && t <= n)) continue;
            std::vector<uint64_t> A((an + 7) / 8 * 8, 0), B((bn + 7) / 8 * 8, 0);
            for(size_t i = 0; i < an; i++) A[i] = rdig();
            for(size_t i = 0; i < bn; i++) B[i] = rdig();
            fails += run_case(&sc, mul_u52_toom33, A, an, B, bn, "33"); cases++;
            // degenerate A(-1) == 0: a2 = 0, a1 = a0
            for(size_t i = 0; i < n; i++) A[n + i] = A[i];
            for(size_t i = 0; i < s; i++) A[2*n + i] = 0;
            fails += run_case(&sc, mul_u52_toom33, A, an, B, bn, "33am1=0"); cases++;
            // degenerate B(-1) == 0: b2 = 0, b1 = b0
            for(size_t i = 0; i < n; i++) B[n + i] = B[i];
            for(size_t i = 0; i < t; i++) B[2*n + i] = 0;
            fails += run_case(&sc, mul_u52_toom33, A, an, B, bn, "33bm1=0"); cases++;
        }
    }

    // toom42: bn in (n, 2n] with n from the formula; sweep an, derive windows
    for(size_t an = 20; an <= 240; an += (an < 60 ? 1 : 3)){
        for(size_t bn = an / 4 + 2; bn * 3 <= an * 2; bn += 2){
            uint64_t n = (an >= 2*bn) ? (an + 3) >> 2 : ((bn + 1) >> 1);
            if(n < 2) continue;
            uint64_t s = an - 3*n, t = bn - n;
            if(!(0 < s && s <= n && 0 < t && t <= n)) continue;
            if((int64_t)(an - 3*n) <= 0) continue;
            std::vector<uint64_t> A((an + 7) / 8 * 8, 0), B((bn + 7) / 8 * 8, 0);
            for(size_t i = 0; i < an; i++) A[i] = rdig();
            for(size_t i = 0; i < bn; i++) B[i] = rdig();
            fails += run_case(&sc, mul_u52_toom42, A, an, B, bn, "42"); cases++;
            // degenerate A(-1) == 0: a1 = a0, a3 = a2 low s, a2 high zero
            for(size_t i = 0; i < n; i++) A[n + i] = A[i];
            for(size_t i = s; i < n; i++) A[2*n + i] = 0;
            for(size_t i = 0; i < s; i++) A[3*n + i] = A[2*n + i];
            fails += run_case(&sc, mul_u52_toom42, A, an, B, bn, "42am1=0"); cases++;
            // degenerate B(-1) == 0: b0 high zero, b1 = b0 low
            for(size_t i = t; i < n; i++) B[i] = 0;
            for(size_t i = 0; i < t; i++) B[n + i] = B[i];
            fails += run_case(&sc, mul_u52_toom42, A, an, B, bn, "42bm1=0"); cases++;
        }
    }

    // larger shapes recursing through karatsuba
    const size_t big33[][2] = {{192, 160}, {240, 200}, {288, 288}};
    for(int k = 0; k < 3; k++)
        for(int trial = 0; trial < 6; trial++){
            size_t an = big33[k][0], bn = big33[k][1];
            std::vector<uint64_t> A((an + 7) / 8 * 8, 0), B((bn + 7) / 8 * 8, 0);
            for(size_t i = 0; i < an; i++) A[i] = rdig();
            for(size_t i = 0; i < bn; i++) B[i] = rdig();
            fails += run_case(&sc, mul_u52_toom33, A, an, B, bn, "33big"); cases++;
        }
    const size_t big42[][2] = {{300, 160}, {380, 200}, {400, 210}};
    for(int k = 0; k < 3; k++)
        for(int trial = 0; trial < 6; trial++){
            size_t an = big42[k][0], bn = big42[k][1];
            std::vector<uint64_t> A((an + 7) / 8 * 8, 0), B((bn + 7) / 8 * 8, 0);
            for(size_t i = 0; i < an; i++) A[i] = rdig();
            for(size_t i = 0; i < bn; i++) B[i] = rdig();
            fails += run_case(&sc, mul_u52_toom42, A, an, B, bn, "42big"); cases++;
        }

    scratch_destroy(&sc);
    if(fails){ printf("FAILED: %d of %d cases\n", fails, cases); return 1; }
    printf("OK: %d toom33/42 cases match scalar reference\n", cases);
    return 0;
}
