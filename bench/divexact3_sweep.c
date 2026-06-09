#define main divexact3_ref_main
#include "../divexact3_SIMD_ref.cpp"
#undef main

#include <inttypes.h>

typedef void (*div3_fn)(uint64_t *, size_t);

static void divexact3_scalar_void(uint64_t *a, size_t n) {
    (void)divexact3_scalar(a, n);
}

static void make_divisible_by3(uint64_t *a, size_t n) {
    uint64_t rem = 0;
    for (size_t i = 0; i < n; ++i) rem = (rem + a[i] % 3) % 3;
    if (rem) {
        const uint64_t addend = 3 - rem;
        if (a[0] <= M52 - addend) a[0] += addend;
        else a[0] -= rem;
    }
}

static uint64_t fill_cases(uint64_t *base, size_t stride, size_t n, size_t cases) {
    uint64_t h = 0;
    for (size_t c = 0; c < cases; ++c) {
        uint64_t *a = base + c * stride;
        for (size_t i = 0; i < n; ++i) {
            a[i] = rnd() & M52;
            h ^= a[i] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        make_divisible_by3(a, n);
    }
    return h;
}

static int check_len(size_t n) {
    enum { CASES = 16 };
    const size_t stride = n + 8;
    uint64_t *base = (uint64_t *)aligned_alloc(64, CASES * stride * sizeof(uint64_t));
    uint64_t *x = (uint64_t *)aligned_alloc(64, stride * sizeof(uint64_t));
    uint64_t *y = (uint64_t *)aligned_alloc(64, stride * sizeof(uint64_t));
    int ok = 1;
    fill_cases(base, stride, n, CASES);
    for (size_t c = 0; c < CASES; ++c) {
        memcpy(x, base + c * stride, n * sizeof(uint64_t));
        memcpy(y, base + c * stride, n * sizeof(uint64_t));
        divexact3_scalar(x, n);
        divexact3_auto(y, n);
        if (memcmp(x, y, n * sizeof(uint64_t)) != 0) {
            printf("auto mismatch n=%zu case=%zu\n", n, c);
            ok = 0;
            break;
        }
        memcpy(y, base + c * stride, n * sizeof(uint64_t));
        divexact3_hand(y, n);
        if (memcmp(x, y, n * sizeof(uint64_t)) != 0) {
            printf("hand mismatch n=%zu case=%zu\n", n, c);
            ok = 0;
            break;
        }
    }
    free(base);
    free(x);
    free(y);
    return ok;
}

static double bench_one(div3_fn fn, const uint64_t *base, uint64_t *work,
                        size_t stride, size_t n, size_t cases,
                        int reps, int rounds, uint64_t *hash) {
    double best = 1e100;
    const size_t bytes = cases * stride * sizeof(uint64_t);
    for (int r = 0; r < rounds; ++r) {
        memcpy(work, base, bytes);
        const uint64_t t0 = rd();
        for (int rep = 0; rep < reps; ++rep) {
            for (size_t c = 0; c < cases; ++c) {
                fn(work + c * stride, n);
            }
        }
        const uint64_t t1 = rd();
        for (size_t c = 0; c < cases; ++c) {
            *hash ^= work[c * stride + ((size_t)r + c) % n]
                   + 0x9e3779b97f4a7c15ULL + (*hash << 6) + (*hash >> 2);
        }
        const double cpl = (double)(t1 - t0) / ((double)reps * (double)cases * (double)n);
        if (cpl < best) best = cpl;
    }
    return best;
}

int main(int argc, char **argv) {
    const char *csv_path = argc > 1 ? argv[1] :
        "experimental/simd_bigint/bench/divexact3_sweep_8_512_step1.csv";
    enum { CASES = 4, ROUNDS = 5 };
    const size_t min_n = 8, max_n = 512, max_stride = max_n + 8;
    uint64_t *base = (uint64_t *)aligned_alloc(64, CASES * max_stride * sizeof(uint64_t));
    uint64_t *work = (uint64_t *)aligned_alloc(64, CASES * max_stride * sizeof(uint64_t));
    uint64_t hash = 0;

    for (size_t n = 1; n <= max_n; ++n) {
        if (!check_len(n)) return 1;
    }

    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        perror(csv_path);
        return 1;
    }
    fprintf(csv, "n,scalar_cycles_per_limb,auto_cycles_per_limb,hand_cycles_per_limb,"
                 "auto_speedup,hand_speedup,hand_vs_auto\n");

    for (size_t n = min_n; n <= max_n; ++n) {
        const size_t stride = n + 8;
        hash ^= fill_cases(base, stride, n, CASES);
        int reps = (int)(32768 / (CASES * n));
        if (reps < 16) reps = 16;
        if (reps > 2048) reps = 2048;

        const double scalar = bench_one(divexact3_scalar_void, base, work, stride, n,
                                        CASES, reps, ROUNDS, &hash);
        const double auto_v = bench_one(divexact3_auto, base, work, stride, n,
                                        CASES, reps, ROUNDS, &hash);
        const double hand = bench_one(divexact3_hand, base, work, stride, n,
                                      CASES, reps, ROUNDS, &hash);
        fprintf(csv, "%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                n, scalar, auto_v, hand, scalar / auto_v, scalar / hand,
                auto_v / hand);
    }

    fclose(csv);
    printf("csv=%s hash=%016" PRIx64 "\n", csv_path, hash);
    free(base);
    free(work);
    return 0;
}
