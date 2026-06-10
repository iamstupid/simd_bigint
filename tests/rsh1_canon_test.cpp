// Fuzz mpn_u52_add_rsh1_canon / mpn_u52_sub_rsh1_canon against a scalar
// base-2^52 reference, with REDUNDANT inputs (arbitrary 64-bit lanes).
// Build: clang++ -O2 -march=native -I../include rsh1_canon_test.cpp -o /tmp/rsh1t && /tmp/rsh1t
#include "canon.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define M52 ((1ull << 52) - 1)
typedef unsigned __int128 u128;

static uint64_t rng = 0xc0ffee123456789ull;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

// redundant lane: up to ~2^58, biased toward edge values
static uint64_t rlane(void){
    switch(xr() & 7){
    case 0: return 0;
    case 1: return M52;
    case 2: return xr() & ((1ull << 58) - 1);          // heavy redundancy
    case 3: return (1ull << 52);                        // exactly one overflow bit
    default: return xr() & M52;                         // canonical-ish
    }
}

// canonical digits of a redundant lane array (value = sum lanes[i]*2^52i)
static void canon_ref(std::vector<uint64_t>& d, const uint64_t* l, size_t n, size_t outn){
    d.assign(outn, 0);
    u128 c = 0;
    for(size_t i = 0; i < outn; i++){
        u128 t = c + (i < n ? l[i] : 0);
        d[i] = (uint64_t)(t & M52);
        c = t >> 52;
    }
    assert(c == 0);
}
static int dig_cmp(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b){
    for(size_t i = a.size(); i-- > 0;){
        uint64_t x = a[i], y = b[i];
        if(x != y) return x < y ? -1 : 1;
    }
    return 0;
}
// (A ± B)/2 on canonical digit vectors, A >= B for sub, even
static void ref_halved(std::vector<uint64_t>& r, const std::vector<uint64_t>& A,
                       const std::vector<uint64_t>& B, bool sub){
    size_t n = A.size();
    std::vector<uint64_t> t(n);
    long long c = 0;
    for(size_t i = 0; i < n; i++){
        long long v = (long long)A[i] + (sub ? -(long long)B[i] : (long long)B[i]) + c;
        c = v >> 63 ? -1 : (v > (long long)M52 ? 1 : 0);
        t[i] = (uint64_t)(v - (c << 52)) & M52;
    }
    assert(c == 0);
    r.assign(n, 0);
    for(size_t i = 0; i < n; i++)
        r[i] = (t[i] >> 1) | ((i + 1 < n ? t[i + 1] & 1 : 0) << 51);
}

int main(void){
    const int ITERS = 400000;
    int fails = 0;
    for(int it = 0; it < ITERS; ++it){
        size_t n = 1 + (xr() % 64);
        size_t n2 = n + 2;                       // pad so carry-out is zero
        size_t vecs = (n2 + 7) / 8;
        std::vector<uint64_t> X(vecs * 8, 0), Y(vecs * 8, 0), got(vecs * 8 + 8, 0);
        for(size_t i = 0; i < n; i++){ X[i] = rlane(); Y[i] = rlane(); }
        // live foreign data past the region: the kernel must tail-mask loads
        for(size_t i = n2; i < vecs * 8; i++){ X[i] = xr(); Y[i] = xr(); }
        bool sub = xr() & 1;

        std::vector<uint64_t> A, B, ref;
        canon_ref(A, X.data(), n, n2);
        canon_ref(B, Y.data(), n, n2);
        if(sub && dig_cmp(A, B) < 0){ std::swap(X, Y); std::swap(A, B); }
        // force even parity of X ± Y: value mod 2 == lane0 mod 2
        if(((X[0] ^ Y[0]) & 1)){
            Y[0] += 1;                           // sub: X-Y odd => >=1, stays >=0
            canon_ref(B, Y.data(), n, n2);
            if(sub && dig_cmp(A, B) < 0){        // X == Y+? edge: swap back
                std::swap(X, Y); std::swap(A, B);
            }
        }
        ref_halved(ref, A, B, sub);

        _vec cy = sub
            ? mpn_u52_sub_rsh1_canon((pvec)got.data(), (cpvec)X.data(), (cpvec)Y.data(), n2, zero())
            : mpn_u52_add_rsh1_canon((pvec)got.data(), (cpvec)X.data(), (cpvec)Y.data(), n2, zero());

        uint64_t cyl[8]; memcpy(cyl, &cy, 64);
        int ok = 1;
        for(size_t i = 0; i < n2; i++) if(got[i] != ref[i]) ok = 0;
        for(size_t i = 0; i < n2; i++) if(got[i] > M52) ok = 0;           // canonical out
        // carry-out: only lane 7 of the returned vec is the unconsumed top
        // overflow (lanes 0-6 hold already-consumed his); fits => zero
        if(cyl[7]) ok = 0;
        if(!ok){
            if(fails < 6){
                printf("FAIL it=%d n=%zu sub=%d\n", it, n, (int)sub);
                for(size_t i = 0; i < n2; i++)
                    if(got[i] != ref[i])
                        printf("  limb %2zu got=%016llx ref=%016llx (X=%016llx Y=%016llx)\n",
                            i, (unsigned long long)got[i], (unsigned long long)ref[i],
                            (unsigned long long)(i<X.size()?X[i]:0), (unsigned long long)(i<Y.size()?Y[i]:0));
            }
            fails++;
        }
    }
    if(fails){ printf("FAILED: %d\n", fails); return 1; }
    printf("OK: %d iterations (add+sub rsh1 canon, redundant inputs)\n", ITERS);
    return 0;
}
