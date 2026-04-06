/*
 * garch_mle_papi.c
 *
 * PAPI-instrumented GARCH(1,1) log-likelihood evaluation.
 *
 *   sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
 *   L(theta) = -0.5 * sum_{t=1}^{T} [ log(sigma^2_t) + r^2_t / sigma^2_t ]
 *
 * Evaluates log-likelihood on a parameter grid (simulating MLE optimiser).
 *
 * Usage:  ./garch_mle  <T>  <N_EVAL>
 *
 * Output (stdout, one CSV line):
 *   kernel,T,n_eval,gflops,l1_miss,seconds
 *
 * Uses PAPI low-level eventset API (PAPI_create_eventset/PAPI_start/PAPI_stop)
 * for direct in-memory L1 miss reads -- no JSON file parsing needed.
 */

#include <assert.h>
#include <math.h>
#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void __attribute__((noinline)) do_not_optimize_d(double *_) { (void)_; }
static volatile double sink;

/* ---- xorshift PRNG ---------------------------------------------- */
static unsigned long long xor_state = 0xABCDEF0123456789ULL;

static inline unsigned long long xorshift64(void)
{
    unsigned long long x = xor_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xor_state = x;
    return x;
}

static inline double uniform01_d(void)
{
    return (double)(xorshift64() & 0xFFFFFFFFFFFFULL) / (double)0x1000000000000ULL;
}

static inline double randn_d(void)
{
    double u1 = uniform01_d();
    double u2 = uniform01_d();
    if (u1 < 1e-30) u1 = 1e-30;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* ---- generate synthetic GARCH(1,1) return series ---------------- */
static void generate_returns(double *r, int T,
                             double omega, double alpha, double beta)
{
    double sigma2 = omega / (1.0 - alpha - beta);
    for (int t = 0; t < T; t++) {
        double z = randn_d();
        r[t] = sqrt(sigma2) * z;
        sigma2 = omega + alpha * r[t] * r[t] + beta * sigma2;
    }
}

/* ---- single log-likelihood evaluation --------------------------- */
static double garch_loglik(const double *r, int T,
                           double omega, double alpha, double beta)
{
    double sigma2 = omega / (1.0 - alpha - beta);
    double ll = 0.0;
    for (int t = 0; t < T; t++) {
        sigma2 = omega + alpha * r[t] * r[t] + beta * sigma2;
        ll -= 0.5 * (log(sigma2) + r[t] * r[t] / sigma2);
    }
    return ll;
}

/* ----------------------------------------------------------------- */
int main(int argc, char **argv)
{
    assert(argc == 3);
    int T      = atoi(argv[1]);
    int n_eval = atoi(argv[2]);
    assert(T > 0 && n_eval > 0);

    double *r = (double *)malloc((size_t)T * sizeof(double));
    assert(r);

    double omega_true = 0.00001;
    double alpha_true = 0.05;
    double beta_true  = 0.90;

    generate_returns(r, T, omega_true, alpha_true, beta_true);

    /* pre-build parameter grid */
    int grid_side = (int)cbrt((double)n_eval);
    if (grid_side < 1) grid_side = 1;

    double *omegas = (double *)malloc(grid_side * sizeof(double));
    double *alphas = (double *)malloc(grid_side * sizeof(double));
    double *betas  = (double *)malloc(grid_side * sizeof(double));
    assert(omegas && alphas && betas);

    for (int i = 0; i < grid_side; i++) {
        double frac = (double)i / (double)(grid_side > 1 ? grid_side - 1 : 1);
        omegas[i] = 0.000005 + frac * 0.00002;
        alphas[i] = 0.02     + frac * 0.08;
        betas[i]  = 0.85     + frac * 0.10;
    }

    /* ---- PAPI setup: native event for L1 data cache load misses --- */
    int ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT)
        fprintf(stderr, "PAPI_library_init failed: %d\n", ret);

    int EventSet = PAPI_NULL;
    long long values[1] = { 0 };
    int event_code = 0;

    ret = PAPI_create_eventset(&EventSet);
    if (ret != PAPI_OK)
        fprintf(stderr, "PAPI_create_eventset failed: %s\n", PAPI_strerror(ret));

    /* Try preset first, fall back to native perf event */
    ret = PAPI_add_event(EventSet, PAPI_L1_DCM);
    if (ret != PAPI_OK) {
        ret = PAPI_event_name_to_code("perf::L1-DCACHE-LOAD-MISSES", &event_code);
        if (ret != PAPI_OK)
            fprintf(stderr, "PAPI_event_name_to_code failed: %s\n", PAPI_strerror(ret));
        else {
            ret = PAPI_add_event(EventSet, event_code);
            if (ret != PAPI_OK)
                fprintf(stderr, "PAPI_add_event(native) failed: %s\n", PAPI_strerror(ret));
        }
    }

    double best_ll = -1e30;
    int actual_evals = 0;
    struct timespec t0, t1;

    ret = PAPI_start(EventSet);
    if (ret != PAPI_OK)
        fprintf(stderr, "PAPI_start failed: %s\n", PAPI_strerror(ret));

    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* ---- timed region ---- */
    for (int io = 0; io < grid_side; io++)
      for (int ia = 0; ia < grid_side; ia++)
        for (int ib = 0; ib < grid_side; ib++) {
            double om = omegas[io];
            double al = alphas[ia];
            double be = betas[ib];
            if (al + be >= 1.0) continue;
            double ll = garch_loglik(r, T, om, al, be);
            actual_evals++;
            if (ll > best_ll)
                best_ll = ll;
        }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    ret = PAPI_stop(EventSet, values);
    if (ret != PAPI_OK)
        fprintf(stderr, "PAPI_stop failed: %s\n", PAPI_strerror(ret));

    do_not_optimize_d(&best_ll);
    sink = best_ll;

    double seconds = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double flops   = 7.0 * (double)T * (double)actual_evals;
    double gflops  = (flops / seconds) * 1e-9;

    long long l1_miss = values[0];

    printf("garch_mle,%d,%d,%.15g,%lld,%.15g\n",
           T, actual_evals, gflops, l1_miss, seconds);

    free(betas);
    free(alphas);
    free(omegas);
    free(r);
    return 0;
}
