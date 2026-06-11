// Cost-curve fitter for the measurement-based dispatch. Sweeps every
// algorithm at its canonical shape (bn maximized), fits weighted cubics
// (relative least squares, so small sizes order correctly too), and emits
// include/u52-costmodel.h. Built with -DU52_COST_TUNE_BUILD so the model
// coefficients are the mutable arrays below: algorithm internals recurse
// through the live model, so fitting iterates to a fixpoint (3 passes).
//
// Build: clang++ -O3 -march=native -DU52_COST_TUNE_BUILD -I../include \
//          fit_u52_cost.cpp -o fit_u52_cost
// Run:   ./fit_u52_cost > ../include/u52-costmodel.h
#include "mul.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

// live model (seeds from the v1 fits; overwritten per pass)
double u52_cost_base_c[4]  = {3.86, 0.0155, -0.0053, 0.0269};
double u52_cost_t22_c[12]  = {-23, 1.46, 0, 0.0151, 0, 0, -23, 1.46, 0, 0.0151, 0, 0};
double u52_cost_t32_c[12]  = {29, 1.25, 0, 0.0170, 0, 0, 29, 1.25, 0, 0.0170, 0, 0};
double u52_cost_t42_c[12]  = {100, 0.63, 0, 0.0201, 0, 0, 100, 0.63, 0, 0.0201, 0, 0};
double u52_cost_t33_c[12]  = {61, 1.81, 0, 0.0147, 0, 0, 61, 1.81, 0, 0.0147, 0, 0};

#define M52 ((1ull << 52) - 1)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 7

static scratch g_sc;
static std::vector<uint64_t> A, B, R;

typedef int (*mulfn)(pvec, cpvec, cpvec, uint64_t, uint64_t, scratch*);
static int f_base(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch*){
    mul_u52_basecase(r, a, b, an, bn); return 0;
}

static double measure(mulfn fn, uint64_t an, uint64_t bn){
    A.assign((an + 7) / 8 * 8, 0); B.assign((bn + 7) / 8 * 8, 0);
    R.assign(((an + bn + 7) / 8 * 8) + 16, 0);
    for(uint64_t i = 0; i < an; i++) A[i] = xr() & M52;
    for(uint64_t i = 0; i < bn; i++) B[i] = xr() & M52;
    long reps = (long)(300000 / (an + bn)) + 1;
    double tp[PASSES];
    for(int p = 0; p < PASSES; p++){
        double t0 = now();
        for(long q = 0; q < reps; q++)
            fn((pvec)R.data(), (cpvec)A.data(), (cpvec)B.data(), an, bn, &g_sc);
        tp[p] = (now() - t0) / reps;
    }
    qsort(tp, PASSES, 8, cmpd);
    return tp[PASSES / 2] * 1e9;
}

// weighted least squares for t ~ sum c_k * basis_k, weights 1/t^2
static void wls(const std::vector<std::vector<double>>& X, const std::vector<double>& t,
                double* out, int np){
    double M[8][9] = {};
    for(size_t i = 0; i < t.size(); i++){
        double w = 1.0 / (t[i] * t[i]);
        for(int r = 0; r < np; r++){
            for(int c = 0; c < np; c++) M[r][c] += w * X[i][r] * X[i][c];
            M[r][np] += w * X[i][r] * t[i];
        }
    }
    for(int p = 0; p < np; p++){                // gaussian elimination
        int piv = p;
        for(int r = p + 1; r < np; r++) if(fabs(M[r][p]) > fabs(M[piv][p])) piv = r;
        for(int c = 0; c <= np; c++){ double tt = M[p][c]; M[p][c] = M[piv][c]; M[piv][c] = tt; }
        for(int r = 0; r < np; r++){
            if(r == p || M[p][p] == 0) continue;
            double f = M[r][p] / M[p][p];
            for(int c = 0; c <= np; c++) M[r][c] -= f * M[p][c];
        }
    }
    for(int p = 0; p < np; p++) out[p] = M[p][p] != 0 ? M[p][np] / M[p][p] : 0;
}

// piecewise (an < 400 | an >= 400) 2-D fit over the algorithm's shape window
static void fit_2d(const char* name, mulfn fn, const uint64_t* ans, int na,
                   const double* rhos, int nr, double out[12]){
    std::vector<std::vector<double>> X0, X1;
    std::vector<double> T0, T1;
    for(int i = 0; i < na; i++) for(int j = 0; j < nr; j++){
        uint64_t an = ans[i];
        uint64_t bn = (uint64_t)((double)an * rhos[j] + 0.5);
        if(bn < 8 || bn > an) continue;
        if(fn == mul_u52_karatsuba && bn <= an / 2) continue;
        if(fn == mul_u52_toom32 && !u52_toom32_shape_ok(an, bn)) continue;
        if(fn == mul_u52_toom42 && !u52_toom42_shape_ok(an, bn)) continue;
        if(fn == mul_u52_toom33 && !u52_toom33_shape_ok(an, bn)) continue;
        double t = measure(fn, an, bn);
        double a = (double)an, b = (double)bn;
        std::vector<double> row = {1.0, a, b, a * b, a * a, a * a * b};
        if(an < 400){ X0.push_back(row); T0.push_back(t); }
        else        { X1.push_back(row); T1.push_back(t); }
        fprintf(stderr, "  %s an=%llu bn=%llu t=%.0fns\n", name,
                (unsigned long long)an, (unsigned long long)bn, t);
    }
    wls(X0, T0, out, 6);
    wls(X1, T1, out + 6, 6);
    fprintf(stderr, "  -> %s fitted (%zu + %zu pts)\n", name, T0.size(), T1.size());
}

int main(){
    g_sc = scratch_create_ex(4096, 1u << 26, 1u << 26);
    const uint64_t ans_d[] = {48, 64, 90, 128, 180, 256, 300, 360, 440, 512, 720, 1024,
                              1448, 2048, 2896, 3400};
    const int nd = sizeof(ans_d) / sizeof(*ans_d);

    for(int pass = 0; pass < 3; pass++){
        fprintf(stderr, "== pass %d ==\n", pass);
        // basecase: bilinear over rectangles (covers leftovers too)
        {
            std::vector<std::vector<double>> X;
            std::vector<double> T;
            const uint64_t as[] = {8, 16, 32, 64, 96, 128, 256, 512};
            const uint64_t bs[] = {8, 16, 32, 48, 64, 96, 128};
            for(uint64_t a : as) for(uint64_t b : bs){
                if(b > a) continue;
                double t = measure(f_base, a, b);
                X.push_back({1.0, (double)a, (double)b, (double)a * (double)b});
                T.push_back(t);
            }
            wls(X, T, u52_cost_base_c, 4);
            fprintf(stderr, "  -> base: {%.4g, %.4g, %.4g, %.4g}\n",
                    u52_cost_base_c[0], u52_cost_base_c[1], u52_cost_base_c[2], u52_cost_base_c[3]);
        }
        // FULL validity windows, including the degenerate edges: the model
        // must price an algorithm's bad shapes correctly so selection avoids
        // them smoothly -- hard trust-region gates just recreate cutpoints.
        static const double r22[] = {0.52, 0.6, 0.7, 0.85, 1.0};
        static const double r32[] = {0.36, 0.42, 0.5, 0.6, 0.67, 0.78, 0.9};
        static const double r42[] = {0.27, 0.32, 0.38, 0.44, 0.5, 0.57, 0.64};
        static const double r33[] = {0.68, 0.75, 0.85, 1.0};
        fit_2d("t22", mul_u52_karatsuba, ans_d, nd, r22, 5, u52_cost_t22_c);
        fit_2d("t32", mul_u52_toom32,    ans_d, nd, r32, 7, u52_cost_t32_c);
        fit_2d("t42", mul_u52_toom42,    ans_d, nd, r42, 7, u52_cost_t42_c);
        fit_2d("t33", mul_u52_toom33,    ans_d, nd, r33, 4, u52_cost_t33_c);
    }

    printf("/* u52-costmodel.h - GENERATED by bench/fit_u52_cost; ns units.\n"
           " * Direct algorithms: piecewise 2-D, split at an = 400 u52 limbs;\n"
           " * per-piece basis {1, an, bn, an*bn, an^2, an^2*bn}.\n"
           " * Basecase: bilinear {1, an, bn, an*bn}. */\n");
    printf("#ifndef U52_COST_SEEDED\n#define U52_COST_SEEDED\n");
    printf("static const double u52_cost_base_c[4]  = {%.8g, %.8g, %.8g, %.8g};\n",
           u52_cost_base_c[0], u52_cost_base_c[1], u52_cost_base_c[2], u52_cost_base_c[3]);
    const char* names[] = {"t22", "t32", "t42", "t33"};
    double* cs[] = {u52_cost_t22_c, u52_cost_t32_c, u52_cost_t42_c, u52_cost_t33_c};
    for(int i = 0; i < 4; i++){
        printf("static const double u52_cost_%s_c[12]  = {", names[i]);
        for(int k = 0; k < 12; k++) printf("%.8g%s", cs[i][k], k < 11 ? (k == 5 ? ",\n        " : ", ") : "");
        printf("};\n");
    }
    printf("#endif\n");
    scratch_destroy(&g_sc);
    return 0;
}
